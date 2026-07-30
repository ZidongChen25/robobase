#!/usr/bin/env python3
"""Causal same-state action-ranking audit for direct CQN-AS checkpoints.

At each requested MovePlate state, the probe forces every bin of one CQN
coarse-to-fine decision, refines the remaining levels greedily, then restores
the exact simulator/controller/agent state and rolls out a common continuation
policy.  This directly tests whether critic bin ordering predicts realized
return, rather than merely agreeing with demonstrations.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


POLICY_RNG_PROTOCOL = "common_prngkey_probe_seed_plus_eval_seed"


_AGENT_ROLLOUT_ATTRIBUTES = (
    "rng_key",
    "_eval_action_history",
    "_eval_action_history_valid",
    "_eval_open_loop_plan",
    "_eval_open_loop_position",
    "_eval_open_loop_valid",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--eval-seeds", default="20000,20001,20002,20003")
    parser.add_argument("--anchor-steps", default="50")
    parser.add_argument("--force-level", type=int, default=0)
    parser.add_argument(
        "--dimension-selection",
        choices=("q_span", "round_robin"),
        default="q_span",
        help=(
            "How to choose the intervened action dimension. q_span keeps the "
            "legacy discovery probe that favors the critic; round_robin uses "
            "only eval-seed/anchor position and is independent of Q, BC, and "
            "realized return."
        ),
    )
    parser.add_argument(
        "--score-level",
        type=int,
        help=(
            "C2F level whose chosen-action Q scores structured candidates. "
            "By default this equals the intervention level. Use the deepest "
            "level to distinguish clipped actions that share a coarser token."
        ),
    )
    parser.add_argument(
        "--flow-readout",
        choices=("auto", "distill", "integrated"),
        default="auto",
        help=(
            "For CQN-Flow, score sibling actions with the online distilled "
            "scalar head or with integrated flow endpoints. auto selects "
            "distill whenever the checkpoint trained that head."
        ),
    )
    parser.add_argument(
        "--num-flow-steps",
        type=int,
        help=(
            "Eval-time Euler steps for an integrated CQN-Flow readout. "
            "Omit to preserve the checkpoint configuration."
        ),
    )
    parser.add_argument(
        "--return-sample-aggregation",
        choices=("config", "mean", "entropic", "truncated_mean"),
        default="config",
    )
    parser.add_argument("--num-action-flow-samples", type=int)
    parser.add_argument("--return-sample-truncate-top", type=int)
    parser.add_argument(
        "--policy-value-beta",
        type=_policy_value_beta,
        default="config",
        help=(
            "Policy/value mixing at branch-state collection: config preserves "
            "the saved run setting, bc disables value action selection, and "
            "a non-negative number applies Q_norm + beta * log pi_BC."
        ),
    )
    parser.add_argument(
        "--intervention-mode",
        choices=(
            "effective_policy",
            "raw_plan",
            "structured_k0",
            "structured_horizon",
            "sibling_horizon",
        ),
        default="effective_policy",
        help=(
            "effective_policy registers the forced plan in the agent's "
            "temporal/open-loop state and executes the resulting effective "
            "action; raw_plan bypasses that policy-side transformation; "
            "structured_k0 keeps the normal policy history fixed and directly "
            "executes a-cell, a, and a+cell for one current-action coordinate; "
            "structured_horizon repeats that executed-action intervention for "
            "the configured exploration horizon; sibling_horizon keeps one "
            "C2F prefix fixed, forces every bin at --force-level, and repeats "
            "the resulting executed-action delta for that horizon."
        ),
    )
    parser.add_argument(
        "--intervention-horizon",
        type=int,
        help=(
            "Override the configured repeated-intervention horizon for "
            "structured_horizon or sibling_horizon. Use 1 for a strict "
            "one-step do-action test of an effective_k0 critic."
        ),
    )
    parser.add_argument("--max-continuation-steps", type=int, default=300)
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=20_000,
        help=(
            "Number of state-level bootstrap replicates used for confidence "
            "intervals. Zero disables bootstrap."
        ),
    )
    return parser.parse_args()


def configure_process(gpu_id: int | None) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")
    if gpu_id is not None and gpu_id >= 0:
        gpu = str(gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        os.environ["JAX_CUDA_VISIBLE_DEVICES"] = gpu
        os.environ["MUJOCO_EGL_DEVICE_ID"] = gpu


def _policy_value_beta(value: str) -> str | float | None:
    normalized = value.strip().lower()
    if normalized == "config":
        return "config"
    if normalized == "bc":
        return None
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "policy-value-beta must be config, bc, or a non-negative number"
        ) from exc
    if not np.isfinite(number) or number < 0.0:
        raise argparse.ArgumentTypeError(
            "policy-value-beta must be config, bc, or a non-negative number"
        )
    return number


def _integer_list(value: str, name: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated integers") from exc
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _resolve_value_readout(
    method_name: str,
    requested: str,
    *,
    has_flow_distill: bool,
    direct_scalar_q: bool = False,
) -> str:
    """Resolve the value head whose ordering the causal probe audits."""

    normalized_method = str(method_name).lower()
    if normalized_method == "cqn_as":
        if requested != "auto":
            raise ValueError("--flow-readout applies only to CQN-Flow")
        return (
            "direct_scalar_q"
            if direct_scalar_q
            else "categorical_c51"
        )
    if normalized_method != "cqn_flow":
        raise ValueError("causal probe requires CQN-AS or CQN-Flow")
    if requested == "distill" and not has_flow_distill:
        raise ValueError(
            "--flow-readout=distill requires flow_distill_lambda > 0"
        )
    if requested == "distill" or (
        requested == "auto" and has_flow_distill
    ):
        return "distill"
    return "integrated"


def _policy_readout_label(
    value_readout: str,
    policy_value_beta: float | None,
    *,
    separate_bc_policy: bool,
) -> str:
    if not separate_bc_policy:
        return value_readout
    if policy_value_beta is None:
        return "bc"
    return f"{value_readout}_plus_bc"


def _copy_observation(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value).copy() for key, value in observation.items()}


def _batched(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[None] for key, value in observation.items()}


def _capture_agent_state(agent) -> dict[str, Any]:
    state = {}
    for name in _AGENT_ROLLOUT_ATTRIBUTES:
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


def _restore_agent_state(agent, state: dict[str, Any]) -> None:
    import jax.numpy as jnp

    for name, value in state.items():
        if value is None:
            setattr(agent, name, None)
        elif name == "rng_key":
            setattr(agent, name, jnp.asarray(value))
        else:
            setattr(agent, name, copy.deepcopy(value))


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def _pairwise_sign_stats(
    predicted: np.ndarray,
    realized: np.ndarray,
    *,
    atol: float = 1e-12,
) -> tuple[float, int]:
    """Return accuracy/count over pairs with distinguishable outcomes."""

    correct = 0
    count = 0
    for left in range(realized.size):
        for right in range(left + 1, realized.size):
            realized_delta = float(realized[left] - realized[right])
            if abs(realized_delta) <= atol:
                continue
            predicted_delta = float(predicted[left] - predicted[right])
            correct += int(np.sign(predicted_delta) == np.sign(realized_delta))
            count += 1
    return (float(correct / count) if count else float("nan"), count)


def _ranking_metrics(
    scores: np.ndarray,
    realized: np.ndarray,
) -> dict[str, Any]:
    """Summarize one same-state ranking, including tied realized maxima."""

    values = np.asarray(scores, dtype=np.float64)
    returns = np.asarray(realized, dtype=np.float64)
    if values.shape != returns.shape or values.ndim != 1 or not values.size:
        raise ValueError("ranking scores and returns must be matching vectors")
    predicted_best = int(np.argmax(values))
    realized_max = float(np.max(returns))
    pairwise, count = _pairwise_sign_stats(values, returns)
    return {
        "pairwise_sign_accuracy": pairwise,
        "num_informative_pairs": count,
        "spearman": _spearman(values, returns),
        "top1_match": bool(
            returns[predicted_best] >= realized_max - 1e-12
        ),
        "realized_regret": float(
            realized_max - returns[predicted_best]
        ),
    }


def _action_nearness_scores(policy_scores: np.ndarray) -> np.ndarray:
    """Rank sibling bins only by distance to the independent BC preference."""

    values = np.asarray(policy_scores, dtype=np.float64)
    if values.ndim != 1 or not values.size:
        raise ValueError("policy scores must be a non-empty vector")
    preferred = int(np.argmax(values))
    return -np.abs(
        np.arange(values.size, dtype=np.float64) - preferred
    )


def _select_action_dimension(
    q_values: np.ndarray,
    *,
    selection: str,
    state_index: int,
) -> int:
    """Select a dimension either favorably to Q or independently of Q."""

    values = np.asarray(q_values)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("Q values must have shape [action_dimension, bins]")
    if selection == "q_span":
        return int(np.argmax(np.ptp(values, axis=-1)))
    if selection == "round_robin":
        if state_index < 0:
            raise ValueError("state index must be non-negative")
        return int(state_index % values.shape[0])
    raise ValueError(f"unknown dimension selection: {selection}")


def _selected_path_log_probability(
    policy_scores,
    selected_bins,
    *,
    sequence_index: int | None = None,
    action_dimension: int | None = None,
):
    """Score selected BC bins, optionally only one executed coordinate."""

    import jax
    import jax.numpy as jnp

    scores = jnp.asarray(policy_scores)
    indices = jnp.asarray(selected_bins)
    if scores.ndim < 2 or scores.shape[:-1] != indices.shape:
        raise ValueError(
            "policy scores must equal selected-bin shape plus a bin axis"
        )
    if (sequence_index is None) != (action_dimension is None):
        raise ValueError(
            "sequence index and action dimension must be set together"
        )
    selected = jnp.take_along_axis(
        jax.nn.log_softmax(scores, axis=-1),
        indices[..., None],
        axis=-1,
    )[..., 0]
    if sequence_index is not None and action_dimension is not None:
        if (
            selected.ndim != 3
            or not 0 <= sequence_index < selected.shape[1]
            or not 0 <= action_dimension < selected.shape[2]
        ):
            raise ValueError("selected policy coordinate is out of range")
        return selected[:, sequence_index, action_dimension]
    return selected.reshape((selected.shape[0], -1)).sum(axis=-1)


def _percentile_interval(values: np.ndarray) -> list[float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return [float("nan"), float("nan")]
    return [
        float(np.percentile(finite, 2.5)),
        float(np.percentile(finite, 97.5)),
    ]


def _state_bootstrap(
    records: list[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap aggregate metrics with states, not action pairs, as IID units."""

    informative = [
        record for record in records if record["realized_return_span"] > 0
    ]
    payload = {
        "unit": "informative_state",
        "confidence": 0.95,
        "num_replicates": int(replicates),
        "num_states": len(informative),
        "pairwise_sign_accuracy_ci": [float("nan"), float("nan")],
        "mean_spearman_ci": [float("nan"), float("nan")],
        "top1_match_rate_ci": [float("nan"), float("nan")],
        "mean_realized_regret_ci": [float("nan"), float("nan")],
    }
    if not informative or replicates <= 0:
        return payload

    pair_counts = np.asarray(
        [record["num_informative_pairs"] for record in informative],
        dtype=np.float64,
    )
    pair_correct = np.asarray(
        [
            record["pairwise_sign_accuracy"]
            * record["num_informative_pairs"]
            for record in informative
        ],
        dtype=np.float64,
    )
    spearman = np.asarray(
        [record["spearman"] for record in informative], dtype=np.float64
    )
    top1 = np.asarray(
        [record["top1_match"] for record in informative], dtype=np.float64
    )
    regret = np.asarray(
        [record["realized_regret"] for record in informative],
        dtype=np.float64,
    )
    rng = np.random.default_rng(int(seed))
    pairwise_samples = np.full(replicates, np.nan, dtype=np.float64)
    spearman_samples = np.full(replicates, np.nan, dtype=np.float64)
    top1_samples = np.empty(replicates, dtype=np.float64)
    regret_samples = np.empty(replicates, dtype=np.float64)
    for bootstrap_index in range(replicates):
        indices = rng.integers(0, len(informative), size=len(informative))
        denominator = float(pair_counts[indices].sum())
        if denominator > 0:
            pairwise_samples[bootstrap_index] = float(
                pair_correct[indices].sum() / denominator
            )
        finite_spearman = spearman[indices]
        finite_spearman = finite_spearman[np.isfinite(finite_spearman)]
        if finite_spearman.size:
            spearman_samples[bootstrap_index] = float(
                finite_spearman.mean()
            )
        top1_samples[bootstrap_index] = float(top1[indices].mean())
        regret_samples[bootstrap_index] = float(regret[indices].mean())

    payload.update(
        {
            "pairwise_sign_accuracy_ci": _percentile_interval(
                pairwise_samples
            ),
            "mean_spearman_ci": _percentile_interval(spearman_samples),
            "top1_match_rate_ci": _percentile_interval(top1_samples),
            "mean_realized_regret_ci": _percentile_interval(regret_samples),
        }
    )
    return payload


def _forced_bin_plans(
    agent,
    observation,
    *,
    force_level: int,
    dimension_selection: str = "q_span",
    dimension_state_index: int = 0,
):
    """Return Q, BC proxies, and greedily refined plans for forced bins."""

    import jax
    import jax.numpy as jnp

    obs_inputs = agent._prepare_rl_obs_inputs(_batched(observation))
    features = agent._rl_features(
        agent.params.get("encoder", None), obs_inputs, stop_gradient=True
    )
    policy_features = features
    if getattr(agent, "distinct_policy_encoder", False):
        policy_features = agent._rl_features(
            agent.params.get("policy_encoder", None),
            obs_inputs,
            stop_gradient=True,
        )
    critic_params = (
        agent.target_critic_params
        if agent.use_target_network_for_rollout
        else (
            {
                "critic": agent.params["critic"],
                "advantage": agent.params["advantage"],
            }
            if getattr(agent, "hybrid_flow_v_direct_a", False)
            else agent.params["critic"]
        )
    )
    if force_level < 0 or force_level >= agent.levels:
        raise ValueError(
            f"force level {force_level} outside [0, {agent.levels - 1}]"
        )
    is_flow = "flow" in agent.__class__.__name__.lower()

    def level_q(params, encoded_features, low, high, level, batch_size):
        if is_flow:
            if getattr(agent, "_branch_value_readout", None) == "distill":
                return agent._flow_distill_level(
                    agent.params["flow_distill_readout"],
                    encoded_features,
                    level,
                    low,
                    high,
                )
            flow_params = (
                params["critic"]
                if getattr(agent, "hybrid_flow_v_direct_a", False)
                else params
            )
            flow_key = jax.random.fold_in(jax.random.PRNGKey(1701), level)
            if agent.fixed_action_flow_sources:
                flow_key = jax.random.fold_in(
                    jax.random.PRNGKey(agent._seed + 1729), level
                )
            source = agent._flow_source(
                flow_key,
                1,
                agent.bins,
                num_samples=agent.num_action_flow_samples,
            )
            source = jnp.broadcast_to(
                source,
                (batch_size, *source.shape[1:]),
            )
            endpoints = agent._integrate_level(
                flow_params,
                encoded_features,
                agent._level_condition(low, high, level),
                flow_key,
                source=source,
            )
            q_values = agent._endpoint_q(endpoints)
            if getattr(agent, "hybrid_flow_v_direct_a", False):
                _, centered_advantage = agent._advantage_level(
                    params["advantage"],
                    encoded_features,
                    level,
                    low,
                    high,
                )
                q_values = q_values + centered_advantage
            return q_values

        one_hot = jnp.broadcast_to(
            jax.nn.one_hot(level, agent.levels, dtype=jnp.float32),
            (batch_size, agent.levels),
        )
        critic_output = agent.critic_model.apply(
            params,
            encoded_features,
            one_hot,
            (0.5 * (low + high)).reshape(
                (batch_size, agent.action_sequence, agent.action_dim)
            ),
        )
        if getattr(agent, "direct_scalar_q", False):
            return critic_output[..., 0]
        return jnp.sum(
            jax.nn.softmax(critic_output, axis=-1) * agent.support,
            axis=-1,
        )

    def greedy_path(params, encoded_features):
        low = jnp.broadcast_to(agent.action_low, (1, agent._flat_action_dim))
        high = jnp.broadcast_to(agent.action_high, (1, agent._flat_action_dim))
        q_levels = []
        selected_levels = []
        for level in range(agent.levels):
            q_values = level_q(
                params, encoded_features, low, high, level, 1
            )
            index = jnp.argmax(q_values, axis=-1)
            q_levels.append(q_values)
            selected_levels.append(index)
            low, high = _zoom(
                low,
                high,
                index.reshape((1, agent._flat_action_dim)),
                agent,
            )
        return jnp.stack(q_levels), jnp.stack(selected_levels)

    def _zoom(low, high, index, target_agent):
        from robobase.method.cqn import zoom_in

        return zoom_in(
            low,
            high,
            index,
            target_agent.bins,
            target_agent.action_low,
            target_agent.action_high,
        )

    if getattr(agent, "separate_bc_policy", False):
        # Keep the intervention on the behavior manifold: the independent BC
        # head supplies the plan and all unforced bins, while Q only scores the
        # current effective-action coordinate selected for intervention.
        _, policy_bins = jax.block_until_ready(
            agent._policy_action(
                agent.params["policy"],
                policy_features,
                key=None,
            )
        )
        low = jnp.broadcast_to(agent.action_low, (1, agent._flat_action_dim))
        high = jnp.broadcast_to(agent.action_high, (1, agent._flat_action_dim))
        for level in range(force_level):
            policy_index = policy_bins[:, level].reshape(
                (1, agent._flat_action_dim)
            )
            low, high = _zoom(low, high, policy_index, agent)
        current_q = np.asarray(
            jax.block_until_ready(
                level_q(
                    critic_params,
                    features,
                    low,
                    high,
                    force_level,
                    1,
                )
            )
        )[0, 0]
        action_dimension = _select_action_dimension(
            current_q,
            selection=dimension_selection,
            state_index=dimension_state_index,
        )
        predicted_q = current_q[action_dimension].copy()
        one_hot = jnp.broadcast_to(
            jax.nn.one_hot(
                force_level,
                agent.levels,
                dtype=jnp.float32,
            ),
            (1, agent.levels),
        )
        midpoint = (0.5 * (low + high)).reshape(
            (1, agent.action_sequence, agent.action_dim)
        )
        policy_logits = (
            agent._policy_bin_scores(
                agent.params["policy"],
                policy_features,
                one_hot,
                midpoint,
            )
            if hasattr(agent, "_policy_bin_scores")
            else agent.policy_model.apply(
                agent.params["policy"],
                policy_features,
                one_hot,
                midpoint,
            )[..., 0]
        )
        policy_prior_scores = np.asarray(
            jax.block_until_ready(policy_logits)
        )[0, 0, action_dimension].copy()

        def constrained_policy_paths(policy_params, encoded_features):
            count = agent.bins
            repeated_features = jnp.repeat(encoded_features, count, axis=0)
            path_log_probability = jnp.zeros(
                (count,), dtype=repeated_features.dtype
            )
            candidate_low = jnp.broadcast_to(
                agent.action_low, (count, agent._flat_action_dim)
            )
            candidate_high = jnp.broadcast_to(
                agent.action_high, (count, agent._flat_action_dim)
            )
            for level in range(agent.levels):
                one_hot = jnp.broadcast_to(
                    jax.nn.one_hot(
                        level, agent.levels, dtype=jnp.float32
                    ),
                    (count, agent.levels),
                )
                midpoint = (
                    0.5 * (candidate_low + candidate_high)
                ).reshape(
                    (count, agent.action_sequence, agent.action_dim)
                )
                policy_logits = (
                    agent._policy_bin_scores(
                        policy_params,
                        repeated_features,
                        one_hot,
                        midpoint,
                    )
                    if hasattr(agent, "_policy_bin_scores")
                    else agent.policy_model.apply(
                        policy_params,
                        repeated_features,
                        one_hot,
                        midpoint,
                    )[..., 0]
                )
                policy_index = jnp.argmax(policy_logits, axis=-1)
                if level == force_level:
                    policy_index = policy_index.at[:, 0, action_dimension].set(
                        jnp.arange(count, dtype=policy_index.dtype)
                    )
                path_log_probability = (
                    path_log_probability
                    + _selected_path_log_probability(
                        policy_logits,
                        policy_index,
                        sequence_index=0,
                        action_dimension=action_dimension,
                    )
                )
                candidate_low, candidate_high = _zoom(
                    candidate_low,
                    candidate_high,
                    policy_index.reshape((count, agent._flat_action_dim)),
                    agent,
                )
            return (
                (0.5 * (candidate_low + candidate_high)).reshape(
                    (count, agent.action_sequence, agent.action_dim)
                ),
                path_log_probability,
            )

        constrained_policy_fn = (
            jax.jit(constrained_policy_paths)
            if agent._jit_enabled
            else constrained_policy_paths
        )
        plans, policy_path_scores = jax.block_until_ready(
            constrained_policy_fn(agent.params["policy"], policy_features)
        )
        return (
            predicted_q,
            np.asarray(plans, dtype=np.float32),
            action_dimension,
            policy_prior_scores,
            np.asarray(policy_path_scores, dtype=np.float64),
        )

    greedy_fn = jax.jit(greedy_path) if agent._jit_enabled else greedy_path
    q_levels, _ = jax.block_until_ready(
        greedy_fn(critic_params, features)
    )
    q_levels_np = np.asarray(q_levels)
    # Select the current-action dimension where this level makes its strongest
    # value distinction. This is favorable to the critic and therefore a
    # conservative test: failure cannot be blamed on choosing a flat head.
    current_q = q_levels_np[force_level, 0, 0]
    action_dimension = _select_action_dimension(
        current_q,
        selection=dimension_selection,
        state_index=dimension_state_index,
    )
    predicted_q = current_q[action_dimension].copy()

    def constrained_paths(params, encoded_features):
        count = agent.bins
        repeated_features = jnp.repeat(encoded_features, count, axis=0)
        low = jnp.broadcast_to(
            agent.action_low, (count, agent._flat_action_dim)
        )
        high = jnp.broadcast_to(
            agent.action_high, (count, agent._flat_action_dim)
        )
        for level in range(agent.levels):
            q_values = level_q(
                params,
                repeated_features,
                low,
                high,
                level,
                count,
            )
            index = jnp.argmax(q_values, axis=-1)
            if level == force_level:
                index = index.at[:, 0, action_dimension].set(
                    jnp.arange(count, dtype=index.dtype)
                )
            low, high = _zoom(
                low,
                high,
                index.reshape((count, agent._flat_action_dim)),
                agent,
            )
        return (0.5 * (low + high)).reshape(
            (count, agent.action_sequence, agent.action_dim)
        )

    constrained_fn = (
        jax.jit(constrained_paths) if agent._jit_enabled else constrained_paths
    )
    plans = jax.block_until_ready(constrained_fn(critic_params, features))
    return (
        predicted_q,
        np.asarray(plans, dtype=np.float32),
        action_dimension,
        None,
        None,
    )


def _structured_k0_plans(
    agent,
    observation,
    baseline_plan: np.ndarray,
    *,
    intervention_horizon: int = 1,
    score_level: int | None = None,
):
    """Score and construct the exact local interventions used during rollout.

    The normal policy has already updated its temporal/open-loop history before
    this function is called.  Every candidate therefore shares that same
    policy state and differs only in one actually executed ``k=0`` coordinate.
    We choose the coordinate with the largest predicted local Q span, which is
    deliberately favorable to the critic.
    """

    import jax
    import jax.numpy as jnp

    intervention_level = int(
        getattr(agent, "structured_exploration_level", 1)
    )
    if intervention_level < 0 or intervention_level >= agent.levels:
        raise ValueError(
            "structured exploration level outside the critic hierarchy"
        )
    score_level = (
        intervention_level if score_level is None else int(score_level)
    )
    if score_level < 0 or score_level >= agent.levels:
        raise ValueError("score level outside the critic hierarchy")
    base = np.asarray(baseline_plan, dtype=np.float32)
    if base.shape != (agent.action_sequence, agent.action_dim):
        raise ValueError(
            "baseline plan has shape "
            f"{base.shape}, expected "
            f"{(agent.action_sequence, agent.action_dim)}"
        )
    low = np.asarray(agent._step_action_low, dtype=np.float32)
    high = np.asarray(agent._step_action_high, dtype=np.float32)
    cell_width = (high - low) / float(
        agent.bins ** (intervention_level + 1)
    )
    offsets = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)
    candidates = np.repeat(
        base[None, None, :, :],
        agent.action_dim,
        axis=0,
    )
    candidates = np.repeat(candidates, offsets.size, axis=1)
    horizon = min(int(intervention_horizon), agent.action_sequence)
    if horizon < 1:
        raise ValueError("intervention horizon must be at least one")
    for dimension in range(agent.action_dim):
        for sequence_step in range(horizon):
            candidates[dimension, :, sequence_step, dimension] = np.clip(
                base[sequence_step, dimension]
                + offsets * cell_width[dimension],
                low[dimension],
                high[dimension],
            )
    flat_candidates = candidates.reshape(
        (agent.action_dim * offsets.size, agent.action_sequence, agent.action_dim)
    )

    obs_inputs = agent._prepare_rl_obs_inputs(_batched(observation))
    features = agent._rl_features(
        agent.params.get("encoder", None), obs_inputs, stop_gradient=True
    )
    features = jnp.repeat(features, flat_candidates.shape[0], axis=0)
    critic_params = (
        agent.target_critic_params
        if agent.use_target_network_for_rollout
        else (
            {
                "critic": agent.params["critic"],
                "advantage": agent.params["advantage"],
            }
            if getattr(agent, "hybrid_flow_v_direct_a", False)
            else agent.params["critic"]
        )
    )
    if "flow" in agent.__class__.__name__.lower():
        if getattr(agent, "_branch_value_readout", None) == "distill":
            chosen_q, _ = agent._flow_distill_outputs_per_level(
                agent.params["flow_distill_readout"],
                features,
                jnp.asarray(flat_candidates),
            )
        else:
            key = jax.random.PRNGKey(agent._seed + 31_337)
            chosen_q, _ = agent._q_values_per_level(
                critic_params,
                features,
                jnp.asarray(flat_candidates),
                key,
            )
    elif getattr(agent, "direct_scalar_q", False):
        chosen_q, _ = agent._direct_q_per_level(
            critic_params,
            features,
            jnp.asarray(flat_candidates),
        )
    else:
        chosen_logits, _ = agent._critic_logits_per_level(
            critic_params,
            features,
            jnp.asarray(flat_candidates),
        )
        chosen_q = jnp.sum(
            jax.nn.softmax(chosen_logits, axis=-1) * agent.support,
            axis=-1,
        )
    chosen_q = np.asarray(jax.block_until_ready(chosen_q))
    chosen_q = chosen_q.reshape(
        (agent.action_dim, offsets.size, agent.levels, agent._flat_action_dim)
    )
    local_scores = np.stack(
        [
            chosen_q[dimension, :, score_level, dimension]
            for dimension in range(agent.action_dim)
        ]
    )
    action_spans = np.ptp(candidates[:, :, 0, :], axis=1)
    valid = np.asarray(
        [
            action_spans[dimension, dimension] > 0
            for dimension in range(agent.action_dim)
        ]
    )
    score_spans = np.ptp(local_scores, axis=1)
    score_spans = np.where(valid, score_spans, -np.inf)
    action_dimension = int(np.argmax(score_spans))
    return (
        local_scores[action_dimension].copy(),
        candidates[action_dimension].copy(),
        action_dimension,
    )


def _coherent_sibling_plans(
    agent,
    baseline_plan: np.ndarray,
    sibling_plans: np.ndarray,
    *,
    action_dimension: int,
    intervention_horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Repeat fixed-prefix sibling-bin deltas over the intervention horizon."""

    base = np.asarray(baseline_plan, dtype=np.float32)
    siblings = np.asarray(sibling_plans, dtype=np.float32)
    expected_shape = (agent.action_sequence, agent.action_dim)
    if base.shape != expected_shape:
        raise ValueError(
            f"baseline plan has shape {base.shape}, expected {expected_shape}"
        )
    if siblings.ndim != 3 or siblings.shape[1:] != expected_shape:
        raise ValueError(
            "sibling plans have shape "
            f"{siblings.shape}, expected (N, {expected_shape[0]}, "
            f"{expected_shape[1]})"
        )
    if action_dimension < 0 or action_dimension >= agent.action_dim:
        raise ValueError("action dimension outside the agent action space")
    horizon = min(int(intervention_horizon), agent.action_sequence)
    if horizon < 1:
        raise ValueError("intervention horizon must be at least one")

    deltas = (
        siblings[:, 0, action_dimension] - base[0, action_dimension]
    ).astype(np.float32)
    candidates = np.repeat(base[None], siblings.shape[0], axis=0)
    low = float(agent._step_action_low[action_dimension])
    high = float(agent._step_action_high[action_dimension])
    for sequence_step in range(horizon):
        candidates[:, sequence_step, action_dimension] = np.clip(
            base[sequence_step, action_dimension] + deltas,
            low,
            high,
        )
    return candidates, deltas


def _rollout_branch(
    env,
    agent,
    first_plan: np.ndarray,
    *,
    start_step: int,
    gamma: float,
    max_continuation_steps: int,
    intervention_dimension: int | None = None,
    intervention_delta: float = 0.0,
    intervention_horizon: int = 1,
) -> dict[str, Any]:
    observation, reward, terminated, truncated, _ = env.step(first_plan)
    total_return = float(reward)
    discount = gamma
    length = 1
    success = bool(float(reward) > 0.25)
    while not (terminated or truncated) and length < max_continuation_steps:
        plan = agent.act(_batched(observation), start_step + length, True)[0]
        if (
            intervention_dimension is not None
            and length < intervention_horizon
        ):
            dimension = int(intervention_dimension)
            plan = np.asarray(plan, dtype=np.float32).copy()
            plan[0, dimension] = np.clip(
                plan[0, dimension] + float(intervention_delta),
                float(agent._step_action_low[dimension]),
                float(agent._step_action_high[dimension]),
            )
        observation, reward, terminated, truncated, _ = env.step(plan)
        total_return += discount * float(reward)
        success = success or bool(float(reward) > 0.25)
        discount *= gamma
        length += 1
    return {
        "discounted_return": total_return,
        "success": success,
        "rollout_length": length,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def _effective_policy_plan(agent, candidate_plan: np.ndarray):
    """Apply the same plan-history transform used by ``CQNAS.act``.

    Besides changing the immediately executed action, this deliberately
    mutates the restored agent history so the continuation policy sees the
    candidate plan exactly as it would after a normal policy decision.
    """

    action_chunk = np.asarray(candidate_plan, dtype=np.float32)[None].copy()
    if agent.temporal_ensemble:
        register_mask = agent._temporal_replan_mask(
            eval_mode=True,
            batch_size=1,
        )
        if not bool(register_mask[0]):
            # This matches ``act`` on a non-replanning step: no new plan is
            # registered and the effective action comes from prior history.
            action_chunk[...] = 0.0
        else:
            # Normal eval inference consumes one action key before registering
            # the plan. Keep continuation RNG aligned across branches.
            agent._next_action_key()
        executed_action = agent._ensemble_current_action(
            action_chunk,
            eval_mode=True,
            register_mask=register_mask,
        )
        action_chunk[:, 0] = executed_action
        return action_chunk[0], bool(register_mask[0])

    needs_refresh = agent._open_loop_needs_refresh(
        eval_mode=True,
        batch_size=1,
    )
    if not needs_refresh:
        action_chunk = agent._eval_open_loop_plan.copy()
    else:
        agent._next_action_key()
    action_chunk = agent._open_loop_action_chunk(
        action_chunk,
        eval_mode=True,
    )
    return action_chunk[0], bool(needs_refresh)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import jax
    from omegaconf import OmegaConf

    from robobase.envs.bigym_branch_state import (
        capture_bigym_branch_state,
        restore_bigym_branch_state,
    )
    from robobase.workspace import Workspace

    run_dir = args.run_dir.expanduser().resolve()
    snapshot = (
        args.snapshot
        if args.snapshot is not None
        else run_dir / "snapshots" / "latest_snapshot.pkl"
    ).expanduser().resolve()
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    if not snapshot.is_file():
        raise FileNotFoundError(snapshot)
    cfg = OmegaConf.load(cfg_path)
    method_name = str(cfg.method.name).lower()
    value_readout = _resolve_value_readout(
        method_name,
        args.flow_readout,
        has_flow_distill=(
            float(cfg.method.get("flow_distill_lambda", 0.0)) > 0.0
        ),
        direct_scalar_q=bool(
            cfg.method.get("direct_scalar_q", False)
        ),
    )
    if args.num_flow_steps is not None and (
        args.num_flow_steps < 1
        or method_name != "cqn_flow"
        or value_readout != "integrated"
    ):
        raise ValueError(
            "--num-flow-steps requires a positive integrated CQN-Flow readout"
        )
    if (
        args.num_action_flow_samples is not None
        and args.num_action_flow_samples < 1
    ):
        raise ValueError("--num-action-flow-samples must be positive")
    if (
        args.return_sample_truncate_top is not None
        and args.return_sample_truncate_top < 0
    ):
        raise ValueError(
            "--return-sample-truncate-top must be non-negative"
        )
    if args.return_sample_aggregation == "truncated_mean" and (
        args.num_action_flow_samples is None
        or args.return_sample_truncate_top is None
        or args.return_sample_truncate_top < 1
        or args.return_sample_truncate_top
        >= args.num_action_flow_samples
    ):
        raise ValueError(
            "truncated_mean requires an explicit action sample count and "
            "truncation in [1, samples)."
        )

    seeds = _integer_list(args.eval_seeds, "--eval-seeds")
    anchor_steps = sorted(set(_integer_list(args.anchor_steps, "--anchor-steps")))
    if anchor_steps[0] < 0:
        raise ValueError("anchor steps must be non-negative")
    if args.max_continuation_steps < 1:
        raise ValueError("--max-continuation-steps must be at least 1")
    if args.bootstrap_replicates < 0:
        raise ValueError("--bootstrap-replicates must be non-negative")

    OmegaConf.set_struct(cfg, False)
    if args.policy_value_beta != "config":
        cfg.method.policy_value_beta = args.policy_value_beta
    active_policy_value_beta = cfg.method.get("policy_value_beta", None)
    if method_name == "cqn_flow":
        if args.num_flow_steps is not None:
            cfg.method.num_flow_steps = args.num_flow_steps
        if args.num_action_flow_samples is not None:
            cfg.method.num_action_flow_samples = (
                args.num_action_flow_samples
            )
        if args.return_sample_aggregation != "config":
            cfg.method.return_sample_aggregation = (
                args.return_sample_aggregation
            )
        if args.return_sample_truncate_top is not None:
            cfg.method.return_sample_truncate_top = (
                args.return_sample_truncate_top
            )
        use_distill = value_readout == "distill"
        use_value_for_actions = active_policy_value_beta is not None
        cfg.method.flow_distill_action_readout = (
            use_value_for_actions and use_distill
        )
        cfg.method.flow_q_action_readout = (
            use_value_for_actions
            and not use_distill
            and str(
                cfg.method.get("critic_architecture", "flow_q")
            ).lower()
            == "flow_q"
        )
    cfg.create_train_env = False
    cfg.num_train_envs = 0
    cfg.num_train_frames = 0
    cfg.num_eval_envs = 1
    cfg.num_eval_episodes = 1
    cfg.demo_batch_size = None
    cfg.use_self_imitation = False
    cfg.log_train_video = False
    cfg.log_eval_video = False
    cfg.save_snapshot = False
    cfg.save_csv = False
    cfg.gpu_id = None
    cfg.wandb.use = False
    cfg.tb.use = False
    cfg.replay.num_workers = 0
    cfg.replay.persist = False
    cfg.replay.reuse_saved = False
    cfg.backend.replay_prefetch_size = 0
    cfg.backend.replay_device_prefetch = False
    cfg.backend.fused_update_steps = 1
    cfg.backend.update_block_every_steps = 1
    OmegaConf.resolve(cfg)

    gamma = float(cfg.replay.gamma)
    structured_intervention = args.intervention_mode in {
        "structured_k0",
        "structured_horizon",
    }
    sibling_intervention = args.intervention_mode == "sibling_horizon"
    repeated_intervention = (
        args.intervention_mode == "structured_horizon"
        or sibling_intervention
    )
    if args.intervention_horizon is not None and (
        args.intervention_horizon < 1 or not repeated_intervention
    ):
        raise ValueError(
            "--intervention-horizon requires a positive "
            "structured_horizon or sibling_horizon probe"
        )
    intervention_horizon = (
        int(args.intervention_horizon)
        if args.intervention_horizon is not None
        else int(cfg.method.get("structured_exploration_horizon", 1))
        if repeated_intervention
        else 1
    )
    analysis_force_level = (
        int(cfg.method.get("structured_exploration_level", 1))
        if structured_intervention
        else int(args.force_level)
    )
    if args.score_level is not None and not structured_intervention:
        raise ValueError(
            "--score-level is only valid for structured_k0/"
            "structured_horizon probes; sibling_horizon scores its forced "
            "level by construction"
        )
    if (
        args.dimension_selection != "q_span"
        and not sibling_intervention
    ):
        raise ValueError(
            "--dimension-selection=round_robin currently requires "
            "--intervention-mode=sibling_horizon"
        )
    analysis_score_level = (
        int(args.score_level)
        if args.score_level is not None
        else analysis_force_level
    )
    records = []
    with tempfile.TemporaryDirectory(prefix="cqn-branch-counterfactual-") as work_dir:
        workspace = Workspace(cfg, work_dir=work_dir)
        try:
            workspace.load_snapshot(snapshot, load_replay_buffer=False)
            workspace._ensure_eval_envs_created()
            env = workspace.eval_env
            if env is None:
                raise RuntimeError("counterfactual probe requires a live eval env")
            agent = workspace.agent
            agent._branch_value_readout = value_readout
            np.random.seed(int(args.probe_seed))
            num_action_dimensions = int(agent.action_dim)

            for eval_seed_index, eval_seed in enumerate(seeds):
                # Checkpoint training can leave different saved RNG states.
                # Reset to a common per-environment key so matched policies
                # collect the same branch states and tie-breaking randomness.
                agent.rng_key = jax.random.PRNGKey(
                    int(args.probe_seed) + int(eval_seed)
                )
                observation, _ = env.reset(seed=eval_seed)
                agent.reset(0, [0])
                terminated = truncated = False
                step = 0
                for anchor_index, anchor_step in enumerate(anchor_steps):
                    while step < anchor_step and not (terminated or truncated):
                        plan = agent.act(_batched(observation), step, True)[0]
                        observation, _, terminated, truncated, _ = env.step(plan)
                        step += 1
                    if terminated or truncated:
                        break

                    branch_observation = _copy_observation(observation)
                    env_state = capture_bigym_branch_state(env)
                    agent_state = _capture_agent_state(agent)
                    rollout_agent_state = agent_state
                    candidate_deltas = None
                    policy_prior_scores = None
                    policy_path_scores = None
                    nearness_reference = None
                    if structured_intervention:
                        baseline_plan = agent.act(
                            _batched(branch_observation), step, True
                        )[0]
                        rollout_agent_state = _capture_agent_state(agent)
                        predicted_q, plans, action_dimension = (
                            _structured_k0_plans(
                                agent,
                                branch_observation,
                                baseline_plan,
                                intervention_horizon=intervention_horizon,
                                score_level=analysis_score_level,
                            )
                        )
                        nearness_reference = float(
                            baseline_plan[0, action_dimension]
                        )
                    elif sibling_intervention:
                        baseline_plan = agent.act(
                            _batched(branch_observation), step, True
                        )[0]
                        rollout_agent_state = _capture_agent_state(agent)
                        (
                            predicted_q,
                            sibling_plans,
                            action_dimension,
                            policy_prior_scores,
                            policy_path_scores,
                        ) = (
                            _forced_bin_plans(
                                agent,
                                branch_observation,
                                force_level=analysis_force_level,
                                dimension_selection=(
                                    args.dimension_selection
                                ),
                                dimension_state_index=(
                                    eval_seed_index * len(anchor_steps)
                                    + anchor_index
                                ),
                            )
                        )
                        nearness_reference = float(
                            baseline_plan[0, action_dimension]
                        )
                        plans, candidate_deltas = _coherent_sibling_plans(
                            agent,
                            baseline_plan,
                            sibling_plans,
                            action_dimension=action_dimension,
                            intervention_horizon=intervention_horizon,
                        )
                    else:
                        (
                            predicted_q,
                            plans,
                            action_dimension,
                            policy_prior_scores,
                            policy_path_scores,
                        ) = (
                            _forced_bin_plans(
                                agent,
                                branch_observation,
                                force_level=analysis_force_level,
                                dimension_selection=(
                                    args.dimension_selection
                                ),
                                dimension_state_index=(
                                    eval_seed_index * len(anchor_steps)
                                    + anchor_index
                                ),
                            )
                        )
                    policy_nearness_scores = (
                        None
                        if policy_prior_scores is None
                        else _action_nearness_scores(policy_prior_scores)
                    )
                    outcomes = []
                    for bin_index, candidate_plan in enumerate(plans):
                        restore_bigym_branch_state(env, env_state)
                        _restore_agent_state(agent, rollout_agent_state)
                        raw_forced_action = float(
                            candidate_plan[0, action_dimension]
                        )
                        decision_registered = True
                        rollout_plan = candidate_plan.copy()
                        if args.intervention_mode == "effective_policy":
                            rollout_plan, decision_registered = (
                                _effective_policy_plan(agent, rollout_plan)
                            )
                        structured_delta = 0.0
                        if structured_intervention:
                            cell_width = (
                                float(agent._step_action_high[action_dimension])
                                - float(agent._step_action_low[action_dimension])
                            ) / float(
                                agent.bins ** (analysis_force_level + 1)
                            )
                            structured_delta = (
                                bin_index - len(plans) // 2
                            ) * cell_width
                        elif sibling_intervention:
                            structured_delta = float(
                                candidate_deltas[bin_index]
                            )
                        outcome = _rollout_branch(
                            env,
                            agent,
                            rollout_plan,
                            start_step=step,
                            gamma=gamma,
                            max_continuation_steps=int(
                                args.max_continuation_steps
                            ),
                            intervention_dimension=(
                                action_dimension
                                if (
                                    structured_intervention
                                    or sibling_intervention
                                )
                                else None
                            ),
                            intervention_delta=(
                                structured_delta
                                if (
                                    structured_intervention
                                    or sibling_intervention
                                )
                                else 0.0
                            ),
                            intervention_horizon=intervention_horizon,
                        )
                        outcome["bin"] = bin_index
                        outcome["predicted_q"] = float(predicted_q[bin_index])
                        outcome["raw_forced_action"] = raw_forced_action
                        outcome["effective_forced_action"] = float(
                            rollout_plan[0, action_dimension]
                        )
                        outcome["forced_action"] = outcome[
                            "effective_forced_action"
                        ]
                        outcome["decision_registered"] = decision_registered
                        outcome["intervention_delta"] = structured_delta
                        outcome["intervention_horizon"] = intervention_horizon
                        if policy_prior_scores is not None:
                            outcome["policy_prior_score"] = float(
                                policy_prior_scores[bin_index]
                            )
                            outcome["policy_path_score"] = float(
                                policy_path_scores[bin_index]
                            )
                            outcome["action_nearness_score"] = float(
                                policy_nearness_scores[bin_index]
                            )
                        elif nearness_reference is not None:
                            outcome["action_nearness_score"] = -abs(
                                outcome["effective_forced_action"]
                                - nearness_reference
                            )
                        outcomes.append(outcome)

                    realized = np.asarray(
                        [item["discounted_return"] for item in outcomes]
                    )
                    q_values = np.asarray(
                        [item["predicted_q"] for item in outcomes]
                    )
                    raw_actions = np.asarray(
                        [item["raw_forced_action"] for item in outcomes]
                    )
                    effective_actions = np.asarray(
                        [item["effective_forced_action"] for item in outcomes]
                    )
                    predicted_best = int(np.argmax(q_values))
                    realized_best = int(np.argmax(realized))
                    realized_max = float(np.max(realized))
                    top1_match = bool(
                        realized[predicted_best] >= realized_max - 1e-12
                    )
                    pairwise_accuracy, informative_pairs = (
                        _pairwise_sign_stats(q_values, realized)
                    )
                    policy_proxy = None
                    if policy_prior_scores is not None:
                        policy_proxy = _ranking_metrics(
                            np.asarray(policy_prior_scores),
                            realized,
                        )
                    policy_path_proxy = None
                    if policy_path_scores is not None:
                        policy_path_proxy = _ranking_metrics(
                            np.asarray(policy_path_scores),
                            realized,
                        )
                    nearness_proxy = None
                    if nearness_reference is not None:
                        nearness_proxy = _ranking_metrics(
                            np.asarray(
                                [
                                    item["action_nearness_score"]
                                    for item in outcomes
                                ]
                            ),
                            realized,
                        )
                    records.append(
                        {
                            "eval_seed": eval_seed,
                            "anchor_step": anchor_step,
                            "force_level": analysis_force_level,
                            "action_dimension": action_dimension,
                            "predicted_best_bin": predicted_best,
                            "realized_best_bin": realized_best,
                            # A predicted bin that attains a tied maximum is a
                            # hit even if np.argmax returns another tied index.
                            "top1_match": top1_match,
                            "spearman": _spearman(q_values, realized),
                            "realized_regret": float(
                                np.max(realized) - realized[predicted_best]
                            ),
                            "predicted_q_span": float(np.ptp(q_values)),
                            "raw_action_span": float(np.ptp(raw_actions)),
                            "effective_action_span": float(
                                np.ptp(effective_actions)
                            ),
                            "realized_return_span": float(np.ptp(realized)),
                            "pairwise_sign_accuracy": pairwise_accuracy,
                            "num_informative_pairs": informative_pairs,
                            "policy_prior_proxy": policy_proxy,
                            "policy_path_proxy": policy_path_proxy,
                            "action_nearness_proxy": nearness_proxy,
                            "outcomes": outcomes,
                        }
                    )

                    # Abandon the last branch and continue the common baseline
                    # trajectory to any later anchor.
                    restore_bigym_branch_state(env, env_state)
                    _restore_agent_state(agent, agent_state)
                    observation = branch_observation
        finally:
            workspace.shutdown()

    informative = [record for record in records if record["realized_return_span"] > 0]
    total_pairs = sum(record["num_informative_pairs"] for record in records)
    pairwise_correct = sum(
        record["pairwise_sign_accuracy"] * record["num_informative_pairs"]
        for record in records
        if record["num_informative_pairs"]
    )
    payload = {
        "status": "ok",
        "run_dir": str(run_dir),
        "snapshot": str(snapshot),
        "method": str(cfg.method.name),
        "value_readout": value_readout,
        "num_flow_steps": (
            int(cfg.method.get("num_flow_steps"))
            if method_name == "cqn_flow"
            and value_readout == "integrated"
            else None
        ),
        "num_action_flow_samples": (
            int(cfg.method.get("num_action_flow_samples"))
            if method_name == "cqn_flow"
            and value_readout == "integrated"
            else None
        ),
        "return_sample_aggregation": (
            str(cfg.method.get("return_sample_aggregation", "mean"))
            if method_name == "cqn_flow"
            and value_readout == "integrated"
            else None
        ),
        "return_sample_truncate_top": (
            int(cfg.method.get("return_sample_truncate_top", 0))
            if method_name == "cqn_flow"
            and value_readout == "integrated"
            else None
        ),
        "policy_value_beta": (
            None
            if active_policy_value_beta is None
            else float(active_policy_value_beta)
        ),
        "policy_readout": (
            _policy_readout_label(
                value_readout,
                active_policy_value_beta,
                separate_bc_policy=bool(
                    cfg.method.get("separate_bc_policy", False)
                ),
            )
        ),
        "eval_seeds": seeds,
        "probe_seed": int(args.probe_seed),
        "policy_rng_protocol": POLICY_RNG_PROTOCOL,
        "anchor_steps": anchor_steps,
        "force_level": analysis_force_level,
        "score_level": analysis_score_level,
        "dimension_selection": args.dimension_selection,
        "num_action_dimensions": num_action_dimensions,
        "informative_states_per_dimension": {
            str(dimension): int(
                sum(
                    record["action_dimension"] == dimension
                    for record in informative
                )
            )
            for dimension in range(num_action_dimensions)
        },
        "intervention_mode": args.intervention_mode,
        "intervention_horizon": intervention_horizon,
        "proposal_source": (
            "normal_policy_plus_executed_action_intervention"
            if structured_intervention
            else (
                "bc_policy_fixed_prefix_sibling_bins_plus_"
                "executed_action_intervention"
                if sibling_intervention
                else (
                    "bc_policy"
                    if bool(cfg.method.get("separate_bc_policy", False))
                    else "critic"
                )
            )
        ),
        "num_candidates_per_state": (
            3
            if structured_intervention
            else int(cfg.method.bins)
        ),
        "num_states": len(records),
        "num_informative_states": len(informative),
        "top1_match_rate": (
            float(np.mean([record["top1_match"] for record in informative]))
            if informative
            else float("nan")
        ),
        "mean_spearman": (
            float(np.nanmean([record["spearman"] for record in informative]))
            if informative
            else float("nan")
        ),
        "mean_realized_regret": (
            float(np.mean([record["realized_regret"] for record in informative]))
            if informative
            else float("nan")
        ),
        "num_informative_pairs": total_pairs,
        "pairwise_sign_accuracy": (
            float(pairwise_correct / total_pairs)
            if total_pairs
            else float("nan")
        ),
        "records": records,
    }
    payload["state_bootstrap"] = _state_bootstrap(
        records,
        replicates=int(args.bootstrap_replicates),
        seed=int(args.probe_seed) + 10_007,
    )
    return payload


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        configure_process(args.gpu_id)
        payload = run(args)
        payload["elapsed_seconds"] = time.time() - started
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except BaseException as exc:
        payload = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "run_dir": str(args.run_dir),
            "elapsed_seconds": time.time() - started,
        }
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(payload["traceback"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
