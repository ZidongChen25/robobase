"""Ground-truth RGB-latent successor diagnostic for BiGym CQN-AS.

This module is intentionally an evaluation-only simulator oracle.  It keeps a
loaded RGB CQN-AS checkpoint fixed, proposes a small set of full K-step C2F
action chunks, rolls each chunk forward in an exactly restored BiGym simulator,
encodes the resulting *observations* with the checkpoint's own RGB encoder, and
uses the unchanged target critic to compute

    sum_{j=0}^{h-1} gamma**j r_j + gamma**h (1-done_h) V_target(z_h).

No MuJoCo state is passed to the critic.  The simulator is used only to replace
the learned map ``(z_t, a_t:t+h) -> z_t+h`` with a ground-truth latent target.
It is therefore a causal headroom test for a later deployable latent predictor,
not a method that can run on a real robot.
"""

from __future__ import annotations

import copy
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from robobase.envs.bigym_branch_state import (
    capture_bigym_branch_state,
    restore_bigym_branch_state,
)


_ROLLOUT_STATE_ATTRIBUTES = (
    "rng_key",
    "_eval_action_history",
    "_eval_action_history_valid",
    "_eval_open_loop_plan",
    "_eval_open_loop_position",
    "_eval_open_loop_valid",
)


def _capture_agent_rollout_state(agent) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for name in _ROLLOUT_STATE_ATTRIBUTES:
        if not hasattr(agent, name):
            continue
        value = getattr(agent, name)
        if value is None:
            state[name] = None
        else:
            try:
                state[name] = np.asarray(value).copy()
            except Exception:
                state[name] = copy.deepcopy(value)
    return state


def _restore_agent_rollout_state(agent, state: dict[str, Any]) -> None:
    for name, value in state.items():
        if value is None:
            setattr(agent, name, None)
        elif name == "rng_key":
            setattr(agent, name, jnp.asarray(value))
        else:
            setattr(agent, name, copy.deepcopy(value))


def _unbatch_observation(observations: dict[str, Any]) -> dict[str, np.ndarray]:
    result = {}
    for name, value in observations.items():
        array = np.asarray(value)
        if array.shape[0] != 1:
            raise ValueError("BiGym latent-successor gate requires eval batch size 1.")
        result[name] = array[0].copy()
    return result


def _direct_plus_other_bins(
    direct_plan: np.ndarray,
    forced_bin_plans: np.ndarray,
    *,
    action_dimension: int,
) -> tuple[np.ndarray, int]:
    """Put the true direct plan first and retain the other four forced bins.

    ``_forced_bin_plans`` produces one plan for every bin, but its greedy path
    does not consume the rollout tie-break RNG.  The actual direct plan returned
    by ``CQNAS.act`` is consequently the only valid candidate-zero baseline.
    We remove the forced plan nearest to its intervened coordinate and retain
    the other bins as siblings.
    """

    direct = np.asarray(direct_plan, dtype=np.float32)
    siblings = np.asarray(forced_bin_plans, dtype=np.float32)
    if siblings.ndim != 3 or direct.shape != siblings.shape[1:]:
        raise ValueError("direct and forced-bin plans have incompatible shapes.")
    if not 0 <= int(action_dimension) < direct.shape[-1]:
        raise ValueError("action_dimension lies outside the plan action space.")
    coordinate = int(action_dimension)
    nearest = int(
        np.argmin(np.abs(siblings[:, 0, coordinate] - direct[0, coordinate]))
    )
    keep = np.arange(siblings.shape[0]) != nearest
    candidates = np.concatenate([direct[None], siblings[keep]], axis=0)
    return candidates.astype(np.float32, copy=False), nearest


def _safe_candidate_choice(
    scores: np.ndarray,
    returns: np.ndarray,
    done: np.ndarray,
    *,
    switch_margin: float,
) -> dict[str, Any]:
    """Reject simulator failures/timeouts before comparing successor values."""

    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    realized = np.asarray(returns, dtype=np.float32).reshape(-1)
    terminal = np.asarray(done, dtype=np.float32).reshape(-1)
    if not (values.shape == realized.shape == terminal.shape) or not values.size:
        raise ValueError("candidate score arrays must be matching non-empty vectors")
    invalid = (terminal > 0.5) & (realized <= 0.0)
    unmasked_choice = int(np.argmax(values))
    all_invalid = bool(np.all(invalid))
    if all_invalid:
        raw_choice = 0
    else:
        safe_scores = np.where(invalid, -np.inf, values)
        raw_choice = int(np.argmax(safe_scores))
    margin = float(values[raw_choice] - values[0])
    switch = raw_choice != 0 and margin >= float(switch_margin)
    return {
        "choice": raw_choice if switch else 0,
        "raw_choice": raw_choice,
        "unmasked_choice": unmasked_choice,
        "margin": margin,
        "switch": bool(switch),
        "invalid": invalid,
        "all_invalid": all_invalid,
        "unmasked_invalid_win": bool(invalid[unmasked_choice]),
    }


def _register_candidate(agent, candidate_plan: np.ndarray) -> np.ndarray:
    """Apply exactly the plan-history transform used by ``CQNAS.act``."""

    chunk = np.asarray(candidate_plan, dtype=np.float32)[None].copy()
    if not agent.temporal_ensemble:
        raise ValueError("The registered BiGym K=16 gate requires temporal ensemble.")
    register = agent._temporal_replan_mask(eval_mode=True, batch_size=1)
    if not bool(register[0]):
        raise ValueError(
            "The registered gate requires temporal_ensemble_replan_interval=1."
        )
    # Normal CQN-AS inference consumes one key before the plan is registered.
    agent._next_action_key()
    executed = agent._ensemble_current_action(
        chunk,
        eval_mode=True,
        register_mask=register,
    )
    chunk[:, 0] = executed
    return chunk[0]


def _advance_committed_plan(agent) -> np.ndarray:
    """Advance one step without replanning and return the effective env chunk."""

    zeros = np.zeros(
        (1, agent.action_sequence, agent.action_dim), dtype=np.float32
    )
    executed = agent._ensemble_current_action(
        zeros,
        eval_mode=True,
        register_mask=np.zeros((1,), dtype=np.bool_),
    )
    zeros[:, 0] = executed
    return zeros[0]


class CQNASBigymGroundTruthLatentSuccessor:
    """Evaluation wrapper for exact K-step successor-latent reranking."""

    def __init__(
        self,
        base_agent,
        *,
        discount: float = 0.99,
        horizon: int = 16,
        proposal_level: int = 1,
        switch_margin: float = 1e-5,
        dimension_selection: str = "q_span",
        rerank_interval: int = 16,
    ):
        if not base_agent.use_pixels:
            raise ValueError("Ground-truth latent successor requires an RGB agent.")
        if int(base_agent.action_sequence) <= 1:
            raise ValueError("Ground-truth latent successor requires K > 1.")
        if not bool(base_agent.temporal_ensemble):
            raise ValueError("Ground-truth latent successor requires temporal ensemble.")
        if int(base_agent.temporal_ensemble_replan_interval) != 1:
            raise ValueError("Ground-truth latent successor requires replan interval 1.")
        if not 1 <= int(horizon) <= int(base_agent.action_sequence):
            raise ValueError("horizon must lie in [1, action_sequence].")
        if not 0 <= int(proposal_level) < int(base_agent.levels):
            raise ValueError("proposal_level must lie in [0, levels).")
        if dimension_selection not in {"q_span", "round_robin"}:
            raise ValueError("unknown dimension-selection mode.")
        if int(rerank_interval) < 1:
            raise ValueError("rerank_interval must be positive.")

        self.base = base_agent
        self.discount = float(discount)
        self.horizon = int(horizon)
        self.proposal_level = int(proposal_level)
        self.switch_margin = float(switch_margin)
        self.dimension_selection = str(dimension_selection)
        self.rerank_interval = int(rerank_interval)
        self._rollout_env = None
        self._episode_decisions = 0
        self._policy_calls = 0

        self._calls = 0
        self._switches = 0
        self._raw_sibling_wins = 0
        self._positive_margin_calls = 0
        self._score_span_sum = 0.0
        self._score_span_max = 0.0
        self._latent_span_sum = 0.0
        self._latent_span_max = 0.0
        self._margin_sum = 0.0
        self._margin_max = 0.0
        self._branch_seconds = 0.0
        self._branch_steps = 0
        self._candidate_count = 0
        self._rewarding_candidates = 0
        self._terminal_candidates = 0
        self._invalid_candidates = 0
        self._unmasked_invalid_wins = 0
        self._all_invalid_calls = 0
        self._selected_invalid_siblings = 0

        # These functions run at every control step.  Build/JIT them once here;
        # the sparse-anchor counterfactual script intentionally constructs its
        # JIT locally, which is unsuitable for a closed-loop evaluator.
        self._candidate_impl = jax.jit(self._build_candidate_fn())
        self._score_impl = jax.jit(self._build_score_fn())

    @property
    def logging(self):
        return self.base.logging

    @logging.setter
    def logging(self, value):
        self.base.logging = value

    def __getattr__(self, name: str):
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)

    def set_rollout_env(self, env) -> None:
        if getattr(env, "is_vector_env", False):
            raise ValueError("Ground-truth BiGym branches require one unvectorized env.")
        self._rollout_env = env

    def reset(self, step: int, agents_to_reset: list[int]):
        self.base.reset(step, agents_to_reset)
        self._episode_decisions = 0

    def _raw_direct_plan(self) -> np.ndarray:
        history = self.base._eval_action_history
        valid = self.base._eval_action_history_valid
        if history is None or valid is None or not bool(valid[0, 0]):
            raise RuntimeError("CQN-AS did not register its direct K-step plan.")
        return np.asarray(history[0, 0], dtype=np.float32).copy()

    def _build_candidate_fn(self):
        base = self.base
        proposal_level = self.proposal_level
        dimension_selection = self.dimension_selection

        if getattr(base, "direct_scalar_q", False):
            raise ValueError("The registered gate requires the canonical C51 critic.")
        if getattr(base, "pessimistic_twin_critic", False):
            raise ValueError("The registered gate requires one canonical critic.")

        from robobase.method.cqn_research import zoom_in

        def level_q(critic_params, features, low, high, level):
            batch_size = features.shape[0]
            one_hot = jnp.broadcast_to(
                jax.nn.one_hot(level, base.levels, dtype=jnp.float32),
                (batch_size, base.levels),
            )
            logits = base.critic_model.apply(
                critic_params,
                features,
                one_hot,
                (0.5 * (low + high)).reshape(
                    (batch_size, base.action_sequence, base.action_dim)
                ),
            )
            return jnp.sum(jax.nn.softmax(logits, axis=-1) * base.support, axis=-1)

        def candidate_fn(params, target_critic_params, obs_inputs, call_index):
            features = base._rl_features(
                params.get("encoder", None), obs_inputs, stop_gradient=True
            )
            critic_params = (
                target_critic_params
                if base.use_target_network_for_rollout
                else params["critic"]
            )
            low = jnp.broadcast_to(base.action_low, (1, base._flat_action_dim))
            high = jnp.broadcast_to(base.action_high, (1, base._flat_action_dim))
            for level in range(proposal_level):
                q_values = level_q(critic_params, features, low, high, level)
                index = jnp.argmax(q_values, axis=-1)
                low, high = zoom_in(
                    low,
                    high,
                    index.reshape((1, base._flat_action_dim)),
                    base.bins,
                    base.action_low,
                    base.action_high,
                )

            proposal_q = level_q(
                critic_params, features, low, high, proposal_level
            )
            current_q = proposal_q[0, 0]
            if dimension_selection == "q_span":
                action_dimension = jnp.argmax(jnp.ptp(current_q, axis=-1))
            else:
                action_dimension = jnp.mod(call_index, base.action_dim)

            count = base.bins
            repeated_features = jnp.repeat(features, count, axis=0)
            candidate_low = jnp.repeat(low, count, axis=0)
            candidate_high = jnp.repeat(high, count, axis=0)
            for level in range(proposal_level, base.levels):
                q_values = level_q(
                    critic_params,
                    repeated_features,
                    candidate_low,
                    candidate_high,
                    level,
                )
                index = jnp.argmax(q_values, axis=-1)
                if level == proposal_level:
                    index = index.at[:, 0, action_dimension].set(
                        jnp.arange(count, dtype=index.dtype)
                    )
                candidate_low, candidate_high = zoom_in(
                    candidate_low,
                    candidate_high,
                    index.reshape((count, base._flat_action_dim)),
                    base.bins,
                    base.action_low,
                    base.action_high,
                )
            plans = (0.5 * (candidate_low + candidate_high)).reshape(
                (count, base.action_sequence, base.action_dim)
            )
            return plans, action_dimension, current_q[action_dimension]

        return candidate_fn

    def _build_score_fn(self):
        base = self.base

        def score_fn(
            params,
            target_critic_params,
            obs_inputs,
            returns,
            bootstrap_discounts,
            done,
        ):
            features = base._rl_features(
                params.get("encoder", None),
                obs_inputs,
                stop_gradient=True,
            )
            next_action, _ = base._greedy_action(
                target_critic_params,
                features,
                key=None,
            )
            next_logits, _ = base._critic_logits_per_level(
                target_critic_params,
                features,
                next_action,
            )
            next_q = jnp.sum(
                jax.nn.softmax(next_logits, axis=-1) * base.support,
                axis=-1,
            ).mean(axis=(1, 2))
            scores = returns + bootstrap_discounts * (1.0 - done) * next_q
            return scores, features

        return score_fn

    def _candidate_plans(
        self,
        observation: dict[str, np.ndarray],
        direct_plan: np.ndarray,
    ) -> tuple[np.ndarray, int, int, np.ndarray]:
        obs_inputs = self.base._prepare_rl_obs_inputs(
            {name: value[None] for name, value in observation.items()}
        )
        siblings, action_dimension, predicted_q = self._candidate_impl(
            self.base.params,
            self.base.target_critic_params,
            obs_inputs,
            jnp.asarray(self._calls, dtype=jnp.int32),
        )
        self.base._block(siblings, action_dimension, predicted_q)
        siblings = np.asarray(jax.device_get(siblings), dtype=np.float32)
        action_dimension = int(np.asarray(jax.device_get(action_dimension)))
        candidates, dropped_bin = _direct_plus_other_bins(
            direct_plan,
            siblings,
            action_dimension=action_dimension,
        )
        return (
            candidates,
            action_dimension,
            dropped_bin,
            np.asarray(jax.device_get(predicted_q)),
        )

    def _branch_candidates(
        self,
        candidates: np.ndarray,
        env_state,
        agent_state: dict[str, Any],
    ) -> tuple[list[dict[str, np.ndarray]], np.ndarray, np.ndarray, np.ndarray]:
        if self._rollout_env is None:
            raise RuntimeError("Ground-truth latent successor has no rollout env.")
        final_observations: list[dict[str, np.ndarray]] = []
        returns = []
        bootstrap_discounts = []
        done_values = []
        for candidate in candidates:
            restore_bigym_branch_state(self._rollout_env, env_state)
            _restore_agent_rollout_state(self.base, agent_state)
            action_chunk = _register_candidate(self.base, candidate)
            total_return = 0.0
            discount = 1.0
            done = False
            observation = None
            steps = 0
            for horizon_index in range(self.horizon):
                if horizon_index > 0:
                    action_chunk = _advance_committed_plan(self.base)
                observation, reward, terminated, truncated, _ = self._rollout_env.step(
                    action_chunk
                )
                total_return += discount * float(reward)
                discount *= self.discount
                steps += 1
                done = bool(terminated or truncated)
                if done:
                    break
            if observation is None:
                raise RuntimeError("Ground-truth branch executed zero steps.")
            final_observations.append(
                {name: np.asarray(value).copy() for name, value in observation.items()}
            )
            returns.append(total_return)
            bootstrap_discounts.append(discount)
            done_values.append(float(done))
            self._branch_steps += steps
        return (
            final_observations,
            np.asarray(returns, dtype=np.float32),
            np.asarray(bootstrap_discounts, dtype=np.float32),
            np.asarray(done_values, dtype=np.float32),
        )

    def _score_ground_truth_latents(
        self,
        final_observations: list[dict[str, np.ndarray]],
        returns: np.ndarray,
        bootstrap_discounts: np.ndarray,
        done: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        invalid = (np.asarray(done) > 0.5) & (np.asarray(returns) <= 0.0)
        valid_indices = np.flatnonzero(~invalid)
        if not valid_indices.size:
            # Near a time limit every branch may truncate.  The selector falls
            # back to the direct plan, so neither an unstable observation nor a
            # meaningless timeout latent needs to reach the RGB encoder.
            return np.asarray(returns, dtype=np.float32), np.zeros(
                (len(final_observations), 1), dtype=np.float32
            )
        replacement = int(valid_indices[0])
        batched = {
            name: np.stack([observation[name] for observation in final_observations])
            for name in final_observations[0]
        }
        for values in batched.values():
            values[invalid] = values[replacement]
        obs_inputs = self.base._prepare_rl_obs_inputs(batched)
        scores, features = self._score_impl(
            self.base.params,
            self.base.target_critic_params,
            obs_inputs,
            jnp.asarray(returns),
            jnp.asarray(bootstrap_discounts),
            jnp.asarray(done),
        )
        self.base._block(scores, features)
        return (
            np.asarray(jax.device_get(scores), dtype=np.float32),
            np.asarray(jax.device_get(features), dtype=np.float32),
        )

    def act(self, observations: dict, step: int, eval_mode: bool):
        if not eval_mode:
            raise ValueError("Ground-truth latent successor is evaluation-only.")
        if self._rollout_env is None:
            raise RuntimeError("Workspace did not provide the live BiGym env.")
        decision_index = self._episode_decisions
        self._episode_decisions += 1
        self._policy_calls += 1
        if decision_index % self.rerank_interval != 0:
            return self.base.act(observations, step, eval_mode=True)
        unbatched = _unbatch_observation(observations)
        env_state = capture_bigym_branch_state(self._rollout_env)
        agent_state = _capture_agent_rollout_state(self.base)

        # Obtain the checkpoint's actual tie-broken direct plan, then discard
        # its state mutation.  Every candidate is subsequently registered from
        # the identical pre-decision history and RNG key.
        self.base.act(observations, step, eval_mode=True)
        direct_plan = self._raw_direct_plan()
        candidates, _, _, _ = self._candidate_plans(unbatched, direct_plan)

        started = time.perf_counter()
        try:
            final_observations, returns, bootstrap_discounts, done = (
                self._branch_candidates(candidates, env_state, agent_state)
            )
            scores, features = self._score_ground_truth_latents(
                final_observations,
                returns,
                bootstrap_discounts,
                done,
            )
        finally:
            restore_bigym_branch_state(self._rollout_env, env_state)
            _restore_agent_rollout_state(self.base, agent_state)
        self._branch_seconds += time.perf_counter() - started

        if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(features)):
            raise FloatingPointError("Non-finite ground-truth latent successor score.")
        selection = _safe_candidate_choice(
            scores,
            returns,
            done,
            switch_margin=self.switch_margin,
        )
        choice = int(selection["choice"])
        raw_choice = int(selection["raw_choice"])
        margin = float(selection["margin"])
        switch = bool(selection["switch"])
        invalid = np.asarray(selection["invalid"], dtype=bool)
        valid_scores = scores[~invalid]
        score_span = float(np.ptp(valid_scores)) if valid_scores.size > 1 else 0.0
        valid_features = features[~invalid]
        latent_span = (
            float(
                np.max(
                    np.linalg.norm(
                        valid_features - valid_features[0:1],
                        axis=-1,
                    )
                )
            )
            if len(valid_features) > 1
            else 0.0
        )

        self._calls += 1
        self._switches += int(switch)
        self._raw_sibling_wins += int(raw_choice != 0)
        self._positive_margin_calls += int(margin > 0.0)
        self._score_span_sum += score_span
        self._score_span_max = max(self._score_span_max, score_span)
        self._latent_span_sum += latent_span
        self._latent_span_max = max(self._latent_span_max, latent_span)
        self._margin_sum += max(margin, 0.0)
        self._margin_max = max(self._margin_max, max(margin, 0.0))
        self._candidate_count += len(candidates)
        self._rewarding_candidates += int(np.sum(returns > 0.0))
        self._terminal_candidates += int(np.sum(done > 0.5))
        self._invalid_candidates += int(np.sum(invalid))
        self._unmasked_invalid_wins += int(selection["unmasked_invalid_win"])
        self._all_invalid_calls += int(selection["all_invalid"])
        self._selected_invalid_siblings += int(choice != 0 and invalid[choice])

        selected = _register_candidate(self.base, candidates[choice])
        return selected[None]

    def rollout_diagnostics(self) -> dict[str, float]:
        diagnostics = dict(self.base.rollout_diagnostics())
        calls = max(self._calls, 1)
        candidates = max(self._candidate_count, 1)
        diagnostics.update(
            {
                "gt_latent_calls": float(self._calls),
                "gt_latent_policy_calls": float(self._policy_calls),
                "gt_latent_gate_rate": self._calls / max(self._policy_calls, 1),
                "gt_latent_switch_rate": self._switches / calls,
                "gt_latent_raw_sibling_win_rate": self._raw_sibling_wins / calls,
                "gt_latent_positive_margin_rate": self._positive_margin_calls / calls,
                "gt_latent_mean_positive_margin": self._margin_sum / calls,
                "gt_latent_max_positive_margin": self._margin_max,
                "gt_latent_mean_score_span": self._score_span_sum / calls,
                "gt_latent_max_score_span": self._score_span_max,
                "gt_latent_mean_successor_feature_span": self._latent_span_sum / calls,
                "gt_latent_max_successor_feature_span": self._latent_span_max,
                "gt_latent_branch_steps_per_second": (
                    self._branch_steps / self._branch_seconds
                    if self._branch_seconds > 0.0
                    else 0.0
                ),
                "gt_latent_rewarding_candidate_fraction": (
                    self._rewarding_candidates / candidates
                ),
                "gt_latent_terminal_candidate_fraction": (
                    self._terminal_candidates / candidates
                ),
                "gt_latent_invalid_candidate_fraction": (
                    self._invalid_candidates / candidates
                ),
                "gt_latent_unmasked_invalid_win_rate": (
                    self._unmasked_invalid_wins / calls
                ),
                "gt_latent_all_invalid_call_rate": self._all_invalid_calls / calls,
                "gt_latent_selected_invalid_sibling_rate": (
                    self._selected_invalid_siblings / calls
                ),
            }
        )
        return diagnostics


__all__ = [
    "CQNASBigymGroundTruthLatentSuccessor",
    "_capture_agent_rollout_state",
    "_direct_plus_other_bins",
    "_restore_agent_rollout_state",
    "_safe_candidate_choice",
]
