"""Deployable one-step RGB-latent consequence reranking for CQN-AS.

The wrapped CQN-AS update is unchanged.  A separately optimized ensemble learns
the supervised transition

    (z_t, u_t) -> (z_{t+1} - z_t, reward_t, done_t)

where ``z`` is the wrapped critic's own RGB feature and ``u_t`` is the action
actually executed by the environment.  CQN-AS still proposes K=16 chunks.  At
reranking time every candidate chunk is passed through the real temporal-
ensemble history to obtain its effective first action, the learned model
predicts one successor latent, and the unchanged target CQN supplies the
continuation value.  No predicted latent is recursively fed back into the
model.

The first registered experiment reranks evaluation only and every 16th policy
decision.  Thus online collection, TD/BC, self imitation, and exploration are
the canonical baseline; only checkpoint evaluation consumes model predictions.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from robobase.method.cqn_as_bigym_latent_successor import (
    _capture_agent_rollout_state,
    _direct_plus_other_bins,
    _register_candidate,
    _restore_agent_rollout_state,
)


class LatentConsequenceModel(nn.Module):
    """Residual latent dynamics with sparse reward and episode-end heads."""

    latent_dim: int
    hidden_dims: tuple[int, ...]

    @nn.compact
    def __call__(
        self,
        latent: jax.Array,
        executed_action: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        normalized = nn.LayerNorm(name="latent_norm")(latent)
        x = jnp.concatenate([normalized, executed_action], axis=-1)
        for index, width in enumerate(self.hidden_dims):
            x = nn.Dense(
                width,
                use_bias=False,
                kernel_init=nn.initializers.orthogonal(),
                name=f"dense_{index}",
            )(x)
            x = nn.LayerNorm(name=f"norm_{index}")(x)
            x = nn.silu(x)
        delta = nn.Dense(
            self.latent_dim,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="latent_delta",
        )(x)
        reward_logit = nn.Dense(
            1,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.constant(-6.0),
            name="reward_logit",
        )(x)[..., 0]
        done_logit = nn.Dense(
            1,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.constant(-6.0),
            name="done_logit",
        )(x)[..., 0]
        return delta, reward_logit, done_logit


class CQNASLatentConsequence:
    """Composition wrapper that leaves canonical CQN-AS learning untouched."""

    def __init__(
        self,
        base_agent,
        *,
        seed: int,
        discount: float = 0.99,
        hidden_dims: tuple[int, ...] = (512, 512),
        ensemble_size: int = 5,
        model_lr: float = 3e-4,
        model_weight_decay: float = 1e-4,
        bootstrap_probability: float = 0.8,
        latent_loss_weight: float = 1.0,
        reward_loss_weight: float = 1.0,
        done_loss_weight: float = 0.25,
        positive_weight: float = 256.0,
        huber_delta: float = 1.0,
        minimum_model_updates: int = 500,
        proposal_level: int = 1,
        dimension_selection: str = "q_span",
        uncertainty_beta: float = 0.0,
        switch_margin: float = 1e-5,
        maximum_score_std: float = 1.0,
        maximum_failure_probability: float = 0.5,
        rerank_interval: int = 16,
        rerank_train: bool = False,
        rerank_eval: bool = True,
    ):
        if not bool(base_agent.use_pixels):
            raise ValueError("Latent consequence requires an RGB CQN-AS agent.")
        if int(base_agent.action_sequence) != 16:
            raise ValueError("Registered latent consequence requires K=16.")
        if not bool(base_agent.temporal_ensemble):
            raise ValueError("Latent consequence requires temporal ensemble.")
        if int(base_agent.temporal_ensemble_replan_interval) != 1:
            raise ValueError("Latent consequence requires one-step replanning.")
        if getattr(base_agent, "pessimistic_twin_critic", False):
            raise ValueError("Registered latent consequence uses one canonical critic.")
        if ensemble_size < 2:
            raise ValueError("ensemble_size must be at least two.")
        if not 0.0 < bootstrap_probability <= 1.0:
            raise ValueError("bootstrap_probability must lie in (0, 1].")
        if not 0 <= int(proposal_level) < int(base_agent.levels):
            raise ValueError("proposal_level must lie in [0, levels).")
        if dimension_selection not in {"q_span", "round_robin"}:
            raise ValueError("unknown dimension-selection mode.")
        if int(rerank_interval) < 1:
            raise ValueError("rerank_interval must be positive.")
        if int(minimum_model_updates) < 0:
            raise ValueError("minimum_model_updates must be non-negative.")
        if float(huber_delta) <= 0.0:
            raise ValueError("huber_delta must be positive.")
        if bool(rerank_train):
            raise ValueError(
                "Stage 7 is an eval-only reranker; training-time intervention "
                "requires a separate collection experiment."
            )

        self.base = base_agent
        self.discount = float(discount)
        self.ensemble_size = int(ensemble_size)
        self.bootstrap_probability = float(bootstrap_probability)
        self.latent_loss_weight = float(latent_loss_weight)
        self.reward_loss_weight = float(reward_loss_weight)
        self.done_loss_weight = float(done_loss_weight)
        self.positive_weight = float(positive_weight)
        self.huber_delta = float(huber_delta)
        self.minimum_model_updates = int(minimum_model_updates)
        self.proposal_level = int(proposal_level)
        self.dimension_selection = str(dimension_selection)
        self.uncertainty_beta = float(uncertainty_beta)
        self.switch_margin = float(switch_margin)
        self.maximum_score_std = float(maximum_score_std)
        self.maximum_failure_probability = float(maximum_failure_probability)
        self.rerank_interval = int(rerank_interval)
        self.rerank_train = bool(rerank_train)
        self.rerank_eval = bool(rerank_eval)

        self.latent_dim = int(self.base._rl_feature_dim)
        self.action_dim = int(self.base.action_dim)
        self.model = LatentConsequenceModel(
            latent_dim=self.latent_dim,
            hidden_dims=tuple(int(value) for value in hidden_dims),
        )
        keys = jax.random.split(
            jax.random.PRNGKey(int(seed) + 0x1A7E), self.ensemble_size + 1
        )
        dummy_latent = jnp.zeros((1, self.latent_dim), dtype=jnp.float32)
        dummy_action = jnp.zeros((1, self.action_dim), dtype=jnp.float32)
        members = [
            self.model.init(key, dummy_latent, dummy_action) for key in keys[:-1]
        ]
        self.model_params = jax.tree.map(
            lambda *leaves: jnp.stack(leaves, axis=0), *members
        )
        self.model_optimizer = optax.adamw(
            float(model_lr), weight_decay=float(model_weight_decay)
        )
        self.model_opt_state = self.model_optimizer.init(self.model_params)
        self.model_rng = keys[-1]
        self.model_updates = 0

        self._episode_decisions_train = 0
        self._episode_decisions_eval = 0
        self._train_calls = 0
        self._train_switches = 0
        self._eval_calls = 0
        self._eval_switches = 0
        self._score_span_sum = 0.0
        self._score_std_sum = 0.0
        self._positive_margin_sum = 0.0
        self._failure_rejections = 0
        self._uncertainty_rejections = 0
        self._last_action_signal_ratio = 1.0

        self._model_update_impl = self._build_model_update()
        self._score_impl = self._build_score_candidates()
        self._candidate_impl = self._build_candidate_fn()
        if self.base._jit_enabled:
            self._model_update_impl = jax.jit(self._model_update_impl)
            self._score_impl = jax.jit(self._score_impl)
            self._candidate_impl = jax.jit(self._candidate_impl)

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

    def _build_model_update(self):
        model = self.model
        optimizer = self.model_optimizer
        ensemble_size = self.ensemble_size
        bootstrap_probability = self.bootstrap_probability
        latent_weight = self.latent_loss_weight
        reward_weight = self.reward_loss_weight
        done_weight = self.done_loss_weight
        positive_weight = self.positive_weight
        huber_delta = self.huber_delta

        def update_fn(
            params,
            opt_state,
            rng,
            latent,
            executed_action,
            next_latent,
            reward,
            done,
        ):
            rng, mask_key = jax.random.split(rng)
            masks = jax.random.bernoulli(
                mask_key,
                bootstrap_probability,
                (ensemble_size, latent.shape[0]),
            ).astype(jnp.float32)
            target_delta = jax.lax.stop_gradient(next_latent - latent)
            # Per-batch, per-coordinate scaling prevents a few high-variance
            # encoder channels from owning the loss.  Prediction remains in raw
            # critic-feature units so it can be consumed directly by target CQN.
            delta_scale = jax.lax.stop_gradient(
                jnp.maximum(jnp.sqrt(jnp.mean(jnp.square(target_delta), axis=0)), 1e-3)
            )

            def loss_fn(current_params):
                def member_loss(member_params, mask):
                    prediction, reward_logit, done_logit = model.apply(
                        member_params, latent, executed_action
                    )
                    scaled_error = (prediction - target_delta) / delta_scale
                    latent_loss = jnp.mean(
                        optax.huber_loss(scaled_error, delta=huber_delta), axis=-1
                    )
                    reward_loss = optax.sigmoid_binary_cross_entropy(
                        reward_logit, reward
                    ) * (1.0 + reward * (positive_weight - 1.0))
                    done_loss = optax.sigmoid_binary_cross_entropy(
                        done_logit, done
                    ) * (1.0 + done * (positive_weight - 1.0))
                    row_loss = (
                        latent_weight * latent_loss
                        + reward_weight * reward_loss
                        + done_weight * done_loss
                    )
                    denominator = jnp.maximum(jnp.sum(mask), 1.0)
                    return jnp.sum(row_loss * mask) / denominator

                losses = jax.vmap(member_loss)(current_params, masks)
                return jnp.mean(losses), losses

            (loss, member_losses), grads = jax.value_and_grad(
                loss_fn, has_aux=True
            )(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            def predict(member_params, action):
                return model.apply(member_params, latent, action)[0]

            aligned = jax.vmap(predict, in_axes=(0, None))(
                new_params, executed_action
            )
            shuffled_action = jnp.roll(executed_action, shift=1, axis=0)
            shuffled = jax.vmap(predict, in_axes=(0, None))(
                new_params, shuffled_action
            )
            aligned_mse = jnp.mean(
                jnp.square((aligned - target_delta[None]) / delta_scale[None, None])
            )
            shuffled_mse = jnp.mean(
                jnp.square((shuffled - target_delta[None]) / delta_scale[None, None])
            )
            metrics = {
                "latent_consequence_loss": loss,
                "latent_consequence_member_loss_std": jnp.std(member_losses),
                "latent_consequence_aligned_mse": aligned_mse,
                "latent_consequence_shuffled_action_mse": shuffled_mse,
                "latent_consequence_action_signal_ratio": shuffled_mse
                / jnp.maximum(aligned_mse, 1e-8),
                "latent_consequence_delta_rms": jnp.sqrt(
                    jnp.mean(jnp.square(target_delta))
                ),
            }
            return new_params, new_opt_state, rng, metrics

        return update_fn

    def _build_candidate_fn(self):
        base = self.base
        proposal_level = self.proposal_level
        dimension_selection = self.dimension_selection
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
            return plans, action_dimension, features

        return candidate_fn

    def _build_score_candidates(self):
        base = self.base
        model = self.model
        discount = self.discount
        uncertainty_beta = self.uncertainty_beta

        def score_fn(
            target_critic_params,
            model_params,
            latent,
            executed_actions,
        ):
            batch, proposals, action_dim = executed_actions.shape
            tiled_latent = jnp.repeat(latent[:, None, :], proposals, axis=1)
            flat_latent = tiled_latent.reshape((batch * proposals, latent.shape[-1]))
            flat_action = executed_actions.reshape((batch * proposals, action_dim))

            def member_scores(member_params):
                delta, reward_logit, done_logit = model.apply(
                    member_params, flat_latent, flat_action
                )
                predicted_latent = flat_latent + delta
                next_action, _ = base._greedy_action(
                    target_critic_params,
                    predicted_latent,
                    key=None,
                )
                next_logits, _ = base._critic_logits_per_level(
                    target_critic_params,
                    predicted_latent,
                    next_action,
                )
                next_value = jnp.sum(
                    jax.nn.softmax(next_logits, axis=-1) * base.support,
                    axis=-1,
                ).mean(axis=(1, 2))
                reward_probability = jax.nn.sigmoid(reward_logit)
                done_probability = jax.nn.sigmoid(done_logit)
                score = reward_probability + discount * (
                    1.0 - done_probability
                ) * next_value
                failure_probability = done_probability * (
                    1.0 - reward_probability
                )
                return (
                    score.reshape((batch, proposals)),
                    failure_probability.reshape((batch, proposals)),
                )

            member_score, member_failure = jax.vmap(member_scores)(model_params)
            mean = jnp.mean(member_score, axis=0)
            std = jnp.std(member_score, axis=0)
            lower = mean - uncertainty_beta * std
            return lower, mean, std, jnp.mean(member_failure, axis=0)

        return score_fn

    def _direct_plan(self) -> np.ndarray:
        history = self.base._eval_action_history
        if history is None:
            history = self.base._train_action_history
        valid = self.base._eval_action_history_valid
        if valid is None:
            valid = self.base._train_action_history_valid
        if history is None or valid is None or not bool(valid[0, 0]):
            raise RuntimeError("CQN-AS did not register its direct K=16 plan.")
        return np.asarray(history[0, 0], dtype=np.float32).copy()

    def _model_batch_inputs(self, batch: dict[str, np.ndarray]):
        action = np.asarray(batch["action"], dtype=np.float32)
        expected = (int(self.base.action_sequence), int(self.base.action_dim))
        if action.ndim != 3 or tuple(action.shape[1:]) != expected:
            raise ValueError(
                "Latent consequence requires replay action [B,K,A] with "
                f"K=16; got {action.shape}."
            )
        obs_inputs = self.base._prepare_rl_obs_inputs(batch)
        next_obs_inputs = self.base._next_rl_obs_inputs(batch)
        encoder_params = self.base.params.get("encoder", None)
        latent = self.base._rl_features(
            encoder_params, obs_inputs, stop_gradient=True
        )
        next_latent = self.base._rl_features(
            encoder_params, next_obs_inputs, stop_gradient=True
        )
        reward = np.asarray(batch["reward"], dtype=np.float32).reshape(-1)
        terminal = np.asarray(batch["terminal"], dtype=np.float32).reshape(-1)
        truncated = np.asarray(
            batch.get("truncated", np.zeros_like(terminal)), dtype=np.float32
        ).reshape(-1)
        done = np.maximum(terminal, truncated)
        return (
            latent,
            jnp.asarray(action[:, 0], dtype=jnp.float32),
            next_latent,
            jnp.asarray(reward, dtype=jnp.float32),
            jnp.asarray(done, dtype=jnp.float32),
        )

    def _update_model(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        inputs = self._model_batch_inputs(batch)
        (
            self.model_params,
            self.model_opt_state,
            self.model_rng,
            metrics,
        ) = self._model_update_impl(
            self.model_params,
            self.model_opt_state,
            self.model_rng,
            *inputs,
        )
        self.model_updates += 1
        self.base._block(metrics["latent_consequence_loss"])
        self._last_action_signal_ratio = float(
            np.asarray(
                jax.device_get(metrics["latent_consequence_action_signal_ratio"])
            )
        )
        if not self.logging:
            return {}
        result = {
            key: float(np.asarray(jax.device_get(value)))
            for key, value in metrics.items()
        }
        result["latent_consequence_model_updates"] = float(self.model_updates)
        return result

    def update(
        self,
        replay_iter: Iterator[dict[str, np.ndarray]],
        step: int,
        replay_buffer=None,
    ) -> dict[str, Any]:
        captured: list[dict[str, np.ndarray]] = []

        def capture_iterator():
            while True:
                batch = next(replay_iter)
                captured.append(batch)
                yield batch

        metrics = self.base.update(capture_iterator(), step, replay_buffer)
        for batch in captured:
            metrics.update(self._update_model(batch))
        return metrics

    def _candidate_plans(
        self,
        observations: dict[str, Any],
        direct_plan: np.ndarray,
        call_index: int,
    ) -> tuple[np.ndarray, jax.Array]:
        obs_inputs = self.base._prepare_rl_obs_inputs(observations)
        forced, action_dimension, latent = self._candidate_impl(
            self.base.params,
            self.base.target_critic_params,
            obs_inputs,
            jnp.asarray(call_index, dtype=jnp.int32),
        )
        self.base._block(forced, action_dimension, latent)
        forced_np = np.asarray(jax.device_get(forced), dtype=np.float32)
        action_dimension_int = int(np.asarray(jax.device_get(action_dimension)))
        candidates, _ = _direct_plus_other_bins(
            direct_plan,
            forced_np,
            action_dimension=action_dimension_int,
        )
        return candidates, latent

    def _effective_candidate_chunks(
        self,
        candidates: np.ndarray,
        agent_state: dict[str, Any],
    ) -> np.ndarray:
        chunks = []
        for candidate in candidates:
            _restore_agent_rollout_state(self.base, agent_state)
            chunks.append(_register_candidate(self.base, candidate))
        _restore_agent_rollout_state(self.base, agent_state)
        return np.stack(chunks).astype(np.float32, copy=False)

    def act(self, observations: dict, step: int, eval_mode: bool):
        enabled = self.rerank_eval if eval_mode else self.rerank_train
        counter_name = (
            "_episode_decisions_eval" if eval_mode else "_episode_decisions_train"
        )
        decision_index = int(getattr(self, counter_name))
        setattr(self, counter_name, decision_index + 1)
        if (
            not enabled
            or self.model_updates < self.minimum_model_updates
            or decision_index % self.rerank_interval != 0
        ):
            return self.base.act(observations, step, eval_mode)
        if int(next(iter(observations.values())).shape[0]) != 1:
            raise ValueError("Registered latent reranker requires one eval environment.")

        agent_state = _capture_agent_rollout_state(self.base)
        self.base.act(observations, step, eval_mode)
        direct_plan = self._direct_plan()
        candidates, latent = self._candidate_plans(
            observations,
            direct_plan,
            self._eval_calls if eval_mode else self._train_calls,
        )
        effective_chunks = self._effective_candidate_chunks(candidates, agent_state)
        executed_actions = effective_chunks[:, 0][None]
        lower, _mean, std, failure_probability = self._score_impl(
            self.base.target_critic_params,
            self.model_params,
            latent,
            jnp.asarray(executed_actions),
        )
        self.base._block(lower, std, failure_probability)
        lower_np = np.asarray(jax.device_get(lower), dtype=np.float32)[0]
        std_np = np.asarray(jax.device_get(std), dtype=np.float32)[0]
        failure_np = np.asarray(
            jax.device_get(failure_probability), dtype=np.float32
        )[0]
        if not (
            np.all(np.isfinite(lower_np))
            and np.all(np.isfinite(std_np))
            and np.all(np.isfinite(failure_np))
        ):
            raise FloatingPointError("Non-finite learned latent consequence score.")

        raw_choice = int(np.argmax(lower_np))
        margin = float(lower_np[raw_choice] - lower_np[0])
        uncertainty_reject = bool(std_np[raw_choice] > self.maximum_score_std)
        failure_reject = bool(
            failure_np[raw_choice] > self.maximum_failure_probability
        )
        switch = bool(
            raw_choice != 0
            and margin >= self.switch_margin
            and not uncertainty_reject
            and not failure_reject
        )
        choice = raw_choice if switch else 0

        if eval_mode:
            self._eval_calls += 1
            self._eval_switches += int(switch)
        else:
            self._train_calls += 1
            self._train_switches += int(switch)
        self._score_span_sum += float(np.ptp(lower_np))
        self._score_std_sum += float(std_np[raw_choice])
        self._positive_margin_sum += max(margin, 0.0)
        self._failure_rejections += int(failure_reject)
        self._uncertainty_rejections += int(uncertainty_reject)

        _restore_agent_rollout_state(self.base, agent_state)
        selected = _register_candidate(self.base, candidates[choice])
        return selected[None]

    def reset(self, step: int, agents_to_reset: list[int]):
        self.base.reset(step, agents_to_reset)
        if any(index < self.base.num_train_envs for index in agents_to_reset):
            self._episode_decisions_train = 0
        if any(index >= self.base.num_train_envs for index in agents_to_reset):
            self._episode_decisions_eval = 0

    def rollout_diagnostics(self) -> dict[str, float]:
        diagnostics = dict(self.base.rollout_diagnostics())
        total_calls = self._train_calls + self._eval_calls
        denominator = max(total_calls, 1)
        diagnostics.update(
            {
                "latent_consequence_model_updates": float(self.model_updates),
                "latent_consequence_action_signal_ratio": float(
                    self._last_action_signal_ratio
                ),
                "latent_consequence_train_calls": float(self._train_calls),
                "latent_consequence_eval_calls": float(self._eval_calls),
                "latent_consequence_train_switch_rate": (
                    self._train_switches / max(self._train_calls, 1)
                ),
                "latent_consequence_eval_switch_rate": (
                    self._eval_switches / max(self._eval_calls, 1)
                ),
                "latent_consequence_mean_score_span": (
                    self._score_span_sum / denominator
                ),
                "latent_consequence_mean_selected_score_std": (
                    self._score_std_sum / denominator
                ),
                "latent_consequence_mean_positive_margin": (
                    self._positive_margin_sum / denominator
                ),
                "latent_consequence_failure_rejection_rate": (
                    self._failure_rejections / denominator
                ),
                "latent_consequence_uncertainty_rejection_rate": (
                    self._uncertainty_rejections / denominator
                ),
            }
        )
        return diagnostics

    def state_dict(self) -> dict:
        return {
            "base": self.base.state_dict(),
            "latent_consequence_model_params": self.base._tree_to_numpy(
                self.model_params
            ),
            "latent_consequence_model_updates": self.model_updates,
            "latent_consequence_action_signal_ratio": self._last_action_signal_ratio,
        }

    def load_state_dict(self, state_dict: dict):
        if "base" not in state_dict:
            self.base.load_state_dict(state_dict)
            return
        self.base.load_state_dict(state_dict["base"])
        self.model_params = self.base._tree_from_numpy(
            state_dict["latent_consequence_model_params"]
        )
        self.model_updates = int(
            state_dict.get("latent_consequence_model_updates", 0)
        )
        self._last_action_signal_ratio = float(
            state_dict.get("latent_consequence_action_signal_ratio", 1.0)
        )

    def checkpoint_state_dict(self) -> dict:
        state = self.base.checkpoint_state_dict()
        state["latent_consequence"] = {
            "model_opt_state": self.base._tree_to_numpy(self.model_opt_state),
            "model_rng": np.asarray(self.model_rng),
            "episode_decisions_train": self._episode_decisions_train,
            "episode_decisions_eval": self._episode_decisions_eval,
            "train_calls": self._train_calls,
            "train_switches": self._train_switches,
            "eval_calls": self._eval_calls,
            "eval_switches": self._eval_switches,
            "score_span_sum": self._score_span_sum,
            "score_std_sum": self._score_std_sum,
            "positive_margin_sum": self._positive_margin_sum,
            "failure_rejections": self._failure_rejections,
            "uncertainty_rejections": self._uncertainty_rejections,
        }
        return state

    def load_checkpoint_state_dict(self, state_dict: dict):
        self.base.load_checkpoint_state_dict(state_dict)
        stored = state_dict.get("latent_consequence", {})
        if "model_opt_state" in stored:
            self.model_opt_state = self.base._tree_from_numpy(
                stored["model_opt_state"]
            )
        if "model_rng" in stored:
            self.model_rng = jnp.asarray(stored["model_rng"])
        for key, attribute in (
            ("episode_decisions_train", "_episode_decisions_train"),
            ("episode_decisions_eval", "_episode_decisions_eval"),
            ("train_calls", "_train_calls"),
            ("train_switches", "_train_switches"),
            ("eval_calls", "_eval_calls"),
            ("eval_switches", "_eval_switches"),
            ("failure_rejections", "_failure_rejections"),
            ("uncertainty_rejections", "_uncertainty_rejections"),
        ):
            if key in stored:
                setattr(self, attribute, int(stored[key]))
        for key, attribute in (
            ("score_span_sum", "_score_span_sum"),
            ("score_std_sum", "_score_std_sum"),
            ("positive_margin_sum", "_positive_margin_sum"),
        ):
            if key in stored:
                setattr(self, attribute, float(stored[key]))


__all__ = ["CQNASLatentConsequence", "LatentConsequenceModel"]
