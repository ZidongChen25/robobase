#!/usr/bin/env python3
"""Post-hoc simulator-branch Monte-Carlo oracle for a CQN-family critic.

This script deliberately is not a deployable RL algorithm.  It restores exact
MovePlate simulator states, evaluates matched local action interventions, and
uses the realized return ordering from *training* anchors as an explicit
pairwise ranking target.  A disjoint seed split is retained for evaluation.

The experiment answers two narrower questions before comparing C51 with Flow
Matching:

1. Can the existing frozen visual/state representation plus C51 critic express
   the local action effect when counterfactual supervision is supplied?
2. Does that ordering generalize to unseen simulator seeds, or only memorize
   the oracle anchors?

This is deliberately not conventional replay-only fitted Q evaluation (FQE):
the targets require restoring simulator states and rolling out every do-action
branch.  The independent BC policy is never updated and must be the exact
continuation policy (``policy_value_beta=null``).  The encoder and Flow-V field
are frozen.  Direct CQN updates only its critic; the Flow-V/direct-A hybrid
updates only the direct advantage head.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import random
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

try:
    from analyze_cqn_branch_counterfactual import (
        _batched,
        _capture_agent_state,
        _copy_observation,
        _integer_list,
        _pairwise_sign_stats,
        _percentile_interval,
        _restore_agent_state,
        _rollout_branch,
        _spearman,
        _state_bootstrap,
        configure_process,
    )
except ModuleNotFoundError:  # Imported as ``scripts.*`` by unit tests.
    from scripts.analyze_cqn_branch_counterfactual import (
        _batched,
        _capture_agent_state,
        _copy_observation,
        _integer_list,
        _pairwise_sign_stats,
        _percentile_interval,
        _restore_agent_state,
        _rollout_branch,
        _spearman,
        _state_bootstrap,
        configure_process,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--finetuned-snapshot", type=Path)
    parser.add_argument(
        "--comparison-snapshot",
        type=Path,
        help=(
            "Coverage-only mode: score a critic-only branch-oracle snapshot "
            "on the exact same cache after verifying that encoder and BC "
            "policy parameters are bitwise unchanged."
        ),
    )
    parser.add_argument(
        "--dataset-cache",
        type=Path,
        help=(
            "Optional compressed branch-feature cache. Existing caches are "
            "validated and reused so C51/FM can consume identical samples."
        ),
    )
    parser.add_argument(
        "--coverage-only",
        action="store_true",
        help=(
            "Collect/load the branch cache and report informative-state "
            "coverage without fitting any critic parameters."
        ),
    )
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument(
        "--train-seeds",
        default="22000,22001,22002,22003,22004,22005,22006,22007",
    )
    parser.add_argument(
        "--heldout-seeds",
        default="23000,23001,23002,23003,23004,23005,23006,23007",
    )
    parser.add_argument("--anchor-steps", default="30,75,120")
    parser.add_argument(
        "--action-dimensions",
        default="13,14",
        help=(
            "Current-action coordinates to branch. The default is the two "
            "coordinates repeatedly selected by the preceding Stage-VII "
            "critic-favorable probe."
        ),
    )
    parser.add_argument(
        "--baseline-outcome",
        choices=("all", "success", "failure"),
        default="all",
        help=(
            "Optionally screen each seed with one unbranched policy rollout "
            "and collect branches only from successful or failed baselines. "
            "This is useful for sparse-reward coverage audits."
        ),
    )
    parser.add_argument(
        "--candidate-mode",
        choices=("local_triplet", "sibling_bins"),
        default="local_triplet",
        help=(
            "local_triplet uses a-cell/a/a+cell. sibling_bins fixes one C2F "
            "prefix and branches over every bin at --force-level."
        ),
    )
    parser.add_argument(
        "--force-level",
        type=int,
        default=1,
        help="C2F sibling level used when --candidate-mode=sibling_bins.",
    )
    parser.add_argument("--intervention-horizon", type=int, default=4)
    parser.add_argument(
        "--score-level",
        type=int,
        default=-1,
        help="C2F Q level used by ranking loss; -1 selects the deepest level.",
    )
    parser.add_argument("--max-continuation-steps", type=int, default=250)
    parser.add_argument(
        "--continuation-repeats",
        type=int,
        default=1,
        help=(
            "Repeat each restored (state, candidate action) continuation. "
            "The scalar training target remains the sample mean."
        ),
    )
    parser.add_argument(
        "--continuation-rng-mode",
        choices=("restored", "independent"),
        default="restored",
        help=(
            "restored replays the exact captured RNG state; independent "
            "re-seeds environment/global/agent RNGs for every repeat."
        ),
    )
    parser.add_argument(
        "--continuation-seed-offset",
        type=int,
        default=700_000,
        help="Base seed used by independent continuation repeats.",
    )
    parser.add_argument("--updates", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--sampling-mode",
        choices=("random_balanced", "full_batch"),
        default="random_balanced",
        help=(
            "random_balanced reproduces the original stochastic minibatches; "
            "full_batch uses every cached train state on every update."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument(
        "--delta-regression-weight",
        type=float,
        default=0.0,
        help=(
            "Weight on smooth-L1 matching of pairwise Q/A deltas to realized "
            "return deltas. Unlike sign loss, this also suppresses spurious "
            "differences for tied-return sibling bins."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--train-return-shuffle",
        choices=("none", "within_state", "global"),
        default="none",
        help=(
            "Negative control applied only to oracle-training labels. "
            "Evaluation and coverage always use the original realized returns."
        ),
    )
    parser.add_argument(
        "--oracle-validation-seed",
        type=int,
        help=(
            "Reserve one train-cache simulator seed from oracle updates and "
            "report it separately for checkpoint/update-count selection."
        ),
    )
    parser.add_argument("--return-atol", type=float, default=1e-12)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def _candidate_plans(
    agent,
    baseline_plan: np.ndarray,
    *,
    action_dimension: int,
    intervention_horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the exact local ``-cell / nominal / +cell`` plan triplet."""

    base = np.asarray(baseline_plan, dtype=np.float32)
    expected_shape = (agent.action_sequence, agent.action_dim)
    if base.shape != expected_shape:
        raise ValueError(
            f"baseline plan has shape {base.shape}, expected {expected_shape}"
        )
    if action_dimension < 0 or action_dimension >= agent.action_dim:
        raise ValueError(
            f"action dimension {action_dimension} outside "
            f"[0, {agent.action_dim - 1}]"
        )
    horizon = min(int(intervention_horizon), agent.action_sequence)
    if horizon < 1:
        raise ValueError("intervention horizon must be at least one")

    level = int(getattr(agent, "structured_exploration_level", 1))
    low = np.asarray(agent._step_action_low, dtype=np.float32)
    high = np.asarray(agent._step_action_high, dtype=np.float32)
    cell_width = (high - low) / float(agent.bins ** (level + 1))
    requested_delta = np.asarray([-1.0, 0.0, 1.0], np.float32) * cell_width[
        action_dimension
    ]
    candidates = np.repeat(base[None], 3, axis=0)
    for sequence_step in range(horizon):
        candidates[:, sequence_step, action_dimension] = np.clip(
            base[sequence_step, action_dimension] + requested_delta,
            low[action_dimension],
            high[action_dimension],
        )
    return candidates, requested_delta.astype(np.float32)


def _sibling_candidate_plans(
    agent,
    baseline_plan: np.ndarray,
    *,
    action_dimension: int,
    intervention_horizon: int,
    force_level: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct every fixed-prefix sibling bin and repeat its action delta."""

    base = np.asarray(baseline_plan, dtype=np.float32)
    expected_shape = (agent.action_sequence, agent.action_dim)
    if base.shape != expected_shape:
        raise ValueError(
            f"baseline plan has shape {base.shape}, expected {expected_shape}"
        )
    if action_dimension < 0 or action_dimension >= agent.action_dim:
        raise ValueError(
            f"action dimension {action_dimension} outside "
            f"[0, {agent.action_dim - 1}]"
        )
    if force_level < 0 or force_level >= agent.levels:
        raise ValueError(
            f"force level {force_level} outside [0, {agent.levels - 1}]"
        )
    horizon = min(int(intervention_horizon), agent.action_sequence)
    if horizon < 1:
        raise ValueError("intervention horizon must be at least one")

    action_low = float(agent._step_action_low[action_dimension])
    action_high = float(agent._step_action_high[action_dimension])
    prefix_count = agent.bins**force_level
    prefix_width = (action_high - action_low) / float(prefix_count)
    baseline_action = float(base[0, action_dimension])
    prefix_index = int(
        np.clip(
            np.floor((baseline_action - action_low) / prefix_width),
            0,
            prefix_count - 1,
        )
    )
    prefix_low = action_low + prefix_index * prefix_width
    child_width = prefix_width / float(agent.bins)
    centers = prefix_low + (
        np.arange(agent.bins, dtype=np.float32) + 0.5
    ) * child_width
    centers = np.clip(centers, action_low, action_high).astype(np.float32)
    deltas = (centers - baseline_action).astype(np.float32)
    candidates = np.repeat(base[None], agent.bins, axis=0)
    for sequence_step in range(horizon):
        candidates[:, sequence_step, action_dimension] = np.clip(
            base[sequence_step, action_dimension] + deltas,
            action_low,
            action_high,
        )
    return candidates, deltas


def _frozen_feature(agent, observation: dict[str, Any]) -> np.ndarray:
    import jax

    obs_inputs = agent._prepare_rl_obs_inputs(_batched(observation))
    features = agent._rl_features(
        agent.params.get("encoder", None),
        obs_inputs,
        stop_gradient=True,
    )
    return np.asarray(jax.device_get(features[0]))


def _frozen_policy_feature(
    agent,
    observation: dict[str, Any],
) -> np.ndarray:
    import jax

    obs_inputs = agent._prepare_rl_obs_inputs(_batched(observation))
    encoder_params = agent.params.get("encoder", None)
    if getattr(agent, "distinct_policy_encoder", False):
        encoder_params = agent.params.get("policy_encoder", None)
    features = agent._rl_features(
        encoder_params,
        obs_inputs,
        stop_gradient=True,
    )
    return np.asarray(jax.device_get(features[0]))


def _policy_candidate_log_probabilities(
    agent,
    policy_feature: np.ndarray,
    candidate_actions: np.ndarray,
    *,
    action_dimension: int,
    score_level: int,
) -> np.ndarray:
    """Evaluate the actual independent-BC prior on fixed-prefix candidates."""

    import jax
    import jax.numpy as jnp

    actions = jnp.asarray(candidate_actions, dtype=jnp.float32)
    features = jnp.broadcast_to(
        jnp.asarray(policy_feature, dtype=jnp.float32)[None],
        (actions.shape[0], policy_feature.shape[-1]),
    )
    logits, encoded_bins = agent._policy_logits_per_level(
        agent.params["policy"],
        features,
        actions,
    )
    selected_log_probability = jnp.take_along_axis(
        jax.nn.log_softmax(logits, axis=-1),
        encoded_bins[..., None],
        axis=-1,
    )[..., 0]
    # The intervention always targets the actually executed k=0 token.
    scores = selected_log_probability[
        :,
        int(score_level),
        int(action_dimension),
    ]
    return np.asarray(jax.device_get(scores), np.float32)


def _score_candidates(
    agent,
    critic_params,
    features,
    actions,
    action_dimensions,
    score_level: int,
):
    """Score the trained component at one level/current coordinate."""

    import jax
    import jax.numpy as jnp

    features = jnp.asarray(features, dtype=jnp.float32)
    actions = jnp.asarray(actions, dtype=jnp.float32)
    action_dimensions = jnp.asarray(action_dimensions, dtype=jnp.int32)
    batch_size, candidates = actions.shape[:2]
    repeated_features = jnp.repeat(features, candidates, axis=0)
    flat_actions = actions.reshape(
        (batch_size * candidates, agent.action_sequence, agent.action_dim)
    )
    if getattr(agent, "hybrid_flow_v_direct_a", False):
        _, _, chosen_q, _ = agent._advantage_outputs_per_level(
            critic_params,
            repeated_features,
            flat_actions,
        )
    elif getattr(agent, "direct_scalar_q", False):
        chosen_q, _ = agent._direct_q_per_level(
            critic_params,
            repeated_features,
            flat_actions,
        )
    else:
        chosen_logits, _ = agent._critic_logits_per_level(
            critic_params,
            repeated_features,
            flat_actions,
        )
        probabilities = jax.nn.softmax(chosen_logits, axis=-1)
        chosen_q = jnp.sum(probabilities * agent.support, axis=-1)
    chosen_q = chosen_q.reshape(
        (batch_size, candidates, agent.levels, agent._flat_action_dim)
    )
    level_q = chosen_q[:, :, int(score_level), :]
    gather_index = jnp.broadcast_to(
        action_dimensions[:, None, None],
        (batch_size, candidates, 1),
    )
    return jnp.take_along_axis(level_q, gather_index, axis=-1)[..., 0]


def _baseline_rollout_success(
    env,
    agent,
    *,
    eval_seed: int,
    max_steps: int,
) -> bool:
    """Screen a seed cheaply before running every counterfactual branch."""

    observation, _ = env.reset(seed=int(eval_seed))
    agent.reset(0, [0])
    terminated = truncated = False
    cumulative_reward = 0.0
    step = 0
    while step < max_steps and not (terminated or truncated):
        plan = agent.act(_batched(observation), step, True)[0]
        observation, reward, terminated, truncated, _ = env.step(plan)
        cumulative_reward += float(np.asarray(reward).sum())
        step += 1
    return cumulative_reward > 0.25


def _continuation_repeat_seed(
    base_seed: int,
    *,
    eval_seed: int,
    anchor_step: int,
    action_dimension: int,
    candidate_index: int,
    repeat_index: int,
) -> int:
    """Derive a stable uint32 seed for one restored continuation."""

    components = (
        base_seed,
        eval_seed,
        anchor_step,
        action_dimension,
        candidate_index,
        repeat_index,
    )
    sequence = np.random.SeedSequence(
        [int(component) & 0xFFFF_FFFF for component in components]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _set_continuation_rng(env, agent, seed: int) -> None:
    """Re-seed stochastic continuation sources without changing branch state."""

    import jax

    seed = int(seed) & 0xFFFF_FFFF
    random.seed(seed)
    np.random.seed(seed)
    agent.rng_key = jax.random.PRNGKey(seed)

    current = env
    seen: set[int] = set()
    wrapper_index = 0
    while id(current) not in seen:
        seen.add(id(current))
        if "_np_random" in vars(current):
            local_seed = _continuation_repeat_seed(
                seed,
                eval_seed=wrapper_index,
                anchor_step=0,
                action_dimension=0,
                candidate_index=0,
                repeat_index=0,
            )
            current._np_random = np.random.default_rng(local_seed)
            if "_np_random_seed" in vars(current):
                current._np_random_seed = local_seed
        if "env" not in vars(current):
            break
        current = vars(current)["env"]
        wrapper_index += 1


def _collect_branch_split(
    workspace,
    *,
    seeds: list[int],
    anchor_steps: list[int],
    action_dimensions: list[int],
    intervention_horizon: int,
    max_continuation_steps: int,
    gamma: float,
    candidate_mode: str,
    force_level: int,
    baseline_outcome: str,
    continuation_repeats: int = 1,
    continuation_rng_mode: str = "restored",
    continuation_seed_offset: int = 700_000,
) -> dict[str, Any]:
    from robobase.envs.bigym_branch_state import (
        capture_bigym_branch_state,
        restore_bigym_branch_state,
    )

    env = workspace.eval_env
    agent = workspace.agent
    if env is None:
        raise RuntimeError("branch oracle requires a live eval environment")

    features: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    returns: list[np.ndarray] = []
    return_samples: list[np.ndarray] = []
    dimensions: list[int] = []
    metadata: list[dict[str, Any]] = []
    for eval_seed in seeds:
        baseline_success = None
        if baseline_outcome != "all":
            baseline_success = _baseline_rollout_success(
                env,
                agent,
                eval_seed=int(eval_seed),
                max_steps=max(anchor_steps) + max_continuation_steps,
            )
            if (baseline_outcome == "success") != baseline_success:
                continue
        observation, _ = env.reset(seed=int(eval_seed))
        agent.reset(0, [0])
        terminated = truncated = False
        step = 0
        for anchor_step in anchor_steps:
            while step < anchor_step and not (terminated or truncated):
                plan = agent.act(_batched(observation), step, True)[0]
                observation, _, terminated, truncated, _ = env.step(plan)
                step += 1
            if terminated or truncated:
                break

            branch_observation = _copy_observation(observation)
            env_state = capture_bigym_branch_state(env)
            agent_state = _capture_agent_state(agent)
            baseline_plan = agent.act(
                _batched(branch_observation), step, True
            )[0]
            rollout_agent_state = _capture_agent_state(agent)
            feature = _frozen_feature(agent, branch_observation)
            policy_feature = _frozen_policy_feature(
                agent,
                branch_observation,
            )

            for action_dimension in action_dimensions:
                if candidate_mode == "sibling_bins":
                    candidate_actions, requested_delta = (
                        _sibling_candidate_plans(
                            agent,
                            baseline_plan,
                            action_dimension=action_dimension,
                            intervention_horizon=intervention_horizon,
                            force_level=force_level,
                        )
                    )
                else:
                    candidate_actions, requested_delta = _candidate_plans(
                        agent,
                        baseline_plan,
                        action_dimension=action_dimension,
                        intervention_horizon=intervention_horizon,
                    )
                policy_log_probability = (
                    _policy_candidate_log_probabilities(
                        agent,
                        policy_feature,
                        candidate_actions,
                        action_dimension=action_dimension,
                        score_level=force_level,
                    )
                )
                branch_returns = []
                branch_return_samples = []
                branch_outcomes = []
                branch_repeat_outcomes = []
                for candidate_index, candidate_plan in enumerate(
                    candidate_actions
                ):
                    candidate_outcomes = []
                    candidate_samples = []
                    for repeat_index in range(continuation_repeats):
                        restore_bigym_branch_state(env, env_state)
                        _restore_agent_state(agent, rollout_agent_state)
                        if continuation_rng_mode == "independent":
                            repeat_seed = _continuation_repeat_seed(
                                continuation_seed_offset,
                                eval_seed=int(eval_seed),
                                anchor_step=int(anchor_step),
                                action_dimension=int(action_dimension),
                                candidate_index=int(candidate_index),
                                repeat_index=int(repeat_index),
                            )
                            _set_continuation_rng(env, agent, repeat_seed)
                        outcome = _rollout_branch(
                            env,
                            agent,
                            candidate_plan,
                            start_step=step,
                            gamma=gamma,
                            max_continuation_steps=max_continuation_steps,
                            intervention_dimension=action_dimension,
                            intervention_delta=float(
                                requested_delta[candidate_index]
                            ),
                            intervention_horizon=intervention_horizon,
                        )
                        candidate_samples.append(outcome["discounted_return"])
                        candidate_outcomes.append(outcome)
                    branch_return_samples.append(candidate_samples)
                    branch_returns.append(float(np.mean(candidate_samples)))
                    branch_outcomes.append(candidate_outcomes[0])
                    branch_repeat_outcomes.append(candidate_outcomes)

                features.append(feature.copy())
                actions.append(candidate_actions.copy())
                returns.append(np.asarray(branch_returns, np.float32))
                return_samples.append(
                    np.asarray(branch_return_samples, np.float32)
                )
                dimensions.append(int(action_dimension))
                metadata.append(
                    {
                        "eval_seed": int(eval_seed),
                        "anchor_step": int(anchor_step),
                        "action_dimension": int(action_dimension),
                        "candidate_mode": candidate_mode,
                        "force_level": int(force_level),
                        "baseline_rollout_success": baseline_success,
                        "requested_delta": requested_delta.tolist(),
                        "actual_first_delta": (
                            candidate_actions[:, 0, action_dimension]
                            - baseline_plan[0, action_dimension]
                        ).tolist(),
                        "policy_log_probability": (
                            policy_log_probability.tolist()
                        ),
                        "outcomes": branch_outcomes,
                        "repeat_outcomes": branch_repeat_outcomes,
                    }
                )

            restore_bigym_branch_state(env, env_state)
            _restore_agent_state(agent, agent_state)
            observation = branch_observation

    if not features:
        raise RuntimeError("branch collection produced no anchor states")
    return {
        "features": np.stack(features),
        "actions": np.stack(actions),
        "returns": np.stack(returns),
        "return_samples": np.stack(return_samples),
        "action_dimensions": np.asarray(dimensions, np.int32),
        "metadata": metadata,
    }


def _return_stochasticity_summary(
    dataset: dict[str, Any],
    *,
    return_atol: float,
) -> dict[str, Any]:
    samples = np.asarray(
        dataset.get(
            "return_samples",
            np.asarray(dataset["returns"])[..., None],
        ),
        np.float64,
    )
    if samples.ndim != 3:
        raise ValueError(
            "return_samples must have shape [states, candidates, repeats], "
            f"got {samples.shape}"
        )
    spans = np.ptp(samples, axis=-1)
    standard_deviations = np.std(samples, axis=-1)
    variable = spans > float(return_atol)

    success_probabilities = []
    for state_metadata in dataset["metadata"]:
        repeated = state_metadata.get("repeat_outcomes")
        if repeated is None:
            repeated = [
                [outcome] for outcome in state_metadata.get("outcomes", [])
            ]
        for candidate_outcomes in repeated:
            if candidate_outcomes:
                success_probabilities.append(
                    np.mean(
                        [
                            bool(outcome.get("success", False))
                            for outcome in candidate_outcomes
                        ]
                    )
                )
    success_probabilities = np.asarray(success_probabilities, np.float64)
    stochastic_success = (
        (success_probabilities > 0.0) & (success_probabilities < 1.0)
        if success_probabilities.size
        else np.zeros((0,), dtype=bool)
    )
    return {
        "num_states": int(samples.shape[0]),
        "num_candidates": int(samples.shape[0] * samples.shape[1]),
        "repeats": int(samples.shape[2]),
        "num_variable_return_candidates": int(variable.sum()),
        "variable_return_fraction": float(variable.mean()),
        "mean_return_std": float(standard_deviations.mean()),
        "max_return_std": float(standard_deviations.max()),
        "mean_return_span": float(spans.mean()),
        "max_return_span": float(spans.max()),
        "num_stochastic_success_candidates": int(stochastic_success.sum()),
        "stochastic_success_fraction": (
            float(stochastic_success.mean())
            if stochastic_success.size
            else 0.0
        ),
    }


def _oracle_training_data(
    dataset: dict[str, Any],
    *,
    shuffle_mode: str,
    seed: int,
) -> dict[str, Any]:
    """Copy a branch dataset and optionally break action/return association."""

    if shuffle_mode == "none":
        return dataset
    shuffled = dict(dataset)
    returns = np.asarray(dataset["returns"]).copy()
    rng = np.random.default_rng(400_000 + int(seed))
    if shuffle_mode == "within_state":
        for state_index in range(returns.shape[0]):
            returns[state_index] = returns[
                state_index,
                rng.permutation(returns.shape[1]),
            ]
    elif shuffle_mode == "global":
        returns = returns.reshape(-1)[
            rng.permutation(returns.size)
        ].reshape(returns.shape)
    else:
        raise ValueError(f"unsupported train return shuffle {shuffle_mode!r}")
    shuffled["returns"] = returns
    return shuffled


def _subset_branch_dataset(
    dataset: dict[str, Any],
    mask: np.ndarray,
) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (len(dataset["metadata"]),):
        raise ValueError(
            "branch subset mask must have one entry per metadata record"
        )
    if not np.any(mask):
        raise ValueError("branch subset must not be empty")
    result = dict(dataset)
    for key in (
        "features",
        "actions",
        "returns",
        "return_samples",
        "action_dimensions",
    ):
        if key in dataset:
            result[key] = np.asarray(dataset[key])[mask]
    result["metadata"] = [
        record
        for record, selected in zip(dataset["metadata"], mask)
        if selected
    ]
    return result


def _records_from_scores(
    scores: np.ndarray,
    returns: np.ndarray,
    metadata: list[dict[str, Any]],
    *,
    return_atol: float,
) -> list[dict[str, Any]]:
    records = []
    for state_index, state_metadata in enumerate(metadata):
        predicted = np.asarray(scores[state_index], np.float64)
        realized = np.asarray(returns[state_index], np.float64)
        pairwise_accuracy, informative_pairs = _pairwise_sign_stats(
            predicted,
            realized,
            atol=return_atol,
        )
        predicted_best = int(np.argmax(predicted))
        realized_max = float(np.max(realized))
        return_span = float(np.ptp(realized))
        behavior_proxy_top1 = float("nan")
        behavior_proxy_regret = float("nan")
        behavior_proxy_pairwise = float("nan")
        behavior_proxy_pairs = 0
        if "actual_first_delta" in state_metadata:
            action_delta = np.asarray(
                state_metadata["actual_first_delta"],
                np.float64,
            )
            behavior_proxy_score = -np.abs(action_delta)
            behavior_proxy_bin = int(np.argmax(behavior_proxy_score))
            behavior_proxy_top1 = bool(
                realized[behavior_proxy_bin] >= realized_max - return_atol
            )
            behavior_proxy_regret = float(
                realized_max - realized[behavior_proxy_bin]
            )
            behavior_proxy_pairwise, behavior_proxy_pairs = (
                _pairwise_sign_stats(
                    behavior_proxy_score,
                    realized,
                    atol=return_atol,
                )
            )
        policy_prior_top1 = float("nan")
        policy_prior_regret = float("nan")
        policy_prior_pairwise = float("nan")
        policy_prior_pairs = 0
        if "policy_log_probability" in state_metadata:
            policy_score = np.asarray(
                state_metadata["policy_log_probability"],
                np.float64,
            )
            policy_bin = int(np.argmax(policy_score))
            policy_prior_top1 = bool(
                realized[policy_bin] >= realized_max - return_atol
            )
            policy_prior_regret = float(
                realized_max - realized[policy_bin]
            )
            policy_prior_pairwise, policy_prior_pairs = (
                _pairwise_sign_stats(
                    policy_score,
                    realized,
                    atol=return_atol,
                )
            )
        records.append(
            {
                "eval_seed": state_metadata["eval_seed"],
                "anchor_step": state_metadata["anchor_step"],
                "action_dimension": state_metadata["action_dimension"],
                "predicted_q": predicted.tolist(),
                "realized_return": realized.tolist(),
                "predicted_q_span": float(np.ptp(predicted)),
                "realized_return_span": return_span,
                "random_top1_probability": float(
                    np.mean(realized >= realized_max - return_atol)
                ),
                "behavior_proxy_top1": behavior_proxy_top1,
                "behavior_proxy_regret": behavior_proxy_regret,
                "behavior_proxy_pairwise_sign_accuracy": (
                    behavior_proxy_pairwise
                ),
                "behavior_proxy_num_informative_pairs": (
                    behavior_proxy_pairs
                ),
                "policy_prior_top1": policy_prior_top1,
                "policy_prior_regret": policy_prior_regret,
                "policy_prior_pairwise_sign_accuracy": (
                    policy_prior_pairwise
                ),
                "policy_prior_num_informative_pairs": policy_prior_pairs,
                "num_informative_pairs": informative_pairs,
                "pairwise_sign_accuracy": pairwise_accuracy,
                "spearman": _spearman(predicted, realized),
                "top1_match": bool(
                    realized[predicted_best] >= realized_max - return_atol
                ),
                "realized_regret": float(
                    realized_max - realized[predicted_best]
                ),
            }
        )
    return records


def _summarize_records(
    records: list[dict[str, Any]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    informative = [
        record for record in records if record["num_informative_pairs"] > 0
    ]
    total_pairs = sum(
        record["num_informative_pairs"] for record in informative
    )
    correct_pairs = sum(
        record["pairwise_sign_accuracy"]
        * record["num_informative_pairs"]
        for record in informative
    )
    behavior_pairs = sum(
        record["behavior_proxy_num_informative_pairs"]
        for record in informative
    )
    behavior_correct_pairs = sum(
        record["behavior_proxy_pairwise_sign_accuracy"]
        * record["behavior_proxy_num_informative_pairs"]
        for record in informative
        if record["behavior_proxy_num_informative_pairs"]
    )
    behavior_top1_values = [
        float(record["behavior_proxy_top1"])
        for record in informative
        if np.isfinite(record["behavior_proxy_top1"])
    ]
    behavior_regret_values = [
        float(record["behavior_proxy_regret"])
        for record in informative
        if np.isfinite(record["behavior_proxy_regret"])
    ]
    policy_pairs = sum(
        record["policy_prior_num_informative_pairs"]
        for record in informative
    )
    policy_correct_pairs = sum(
        record["policy_prior_pairwise_sign_accuracy"]
        * record["policy_prior_num_informative_pairs"]
        for record in informative
        if record["policy_prior_num_informative_pairs"]
    )
    policy_top1_values = [
        float(record["policy_prior_top1"])
        for record in informative
        if np.isfinite(record["policy_prior_top1"])
    ]
    policy_regret_values = [
        float(record["policy_prior_regret"])
        for record in informative
        if np.isfinite(record["policy_prior_regret"])
    ]
    state_q_span = np.asarray(
        [record["predicted_q_span"] for record in records],
        np.float64,
    )
    state_return_span = np.asarray(
        [record["realized_return_span"] for record in records],
        np.float64,
    )
    dimensions = sorted(
        {int(record["action_dimension"]) for record in records}
    )
    dimension_q_span = np.asarray(
        [
            np.mean(
                [
                    record["predicted_q_span"]
                    for record in records
                    if int(record["action_dimension"]) == dimension
                ]
            )
            for dimension in dimensions
        ],
        np.float64,
    )
    dimension_return_span = np.asarray(
        [
            np.mean(
                [
                    record["realized_return_span"]
                    for record in records
                    if int(record["action_dimension"]) == dimension
                ]
            )
            for dimension in dimensions
        ],
        np.float64,
    )
    return {
        "num_states": len(records),
        "num_informative_states": len(informative),
        "num_informative_pairs": total_pairs,
        "pairwise_sign_accuracy": (
            float(correct_pairs / total_pairs)
            if total_pairs
            else float("nan")
        ),
        "mean_spearman": (
            float(np.nanmean([record["spearman"] for record in informative]))
            if informative
            else float("nan")
        ),
        "top1_match_rate": (
            float(np.mean([record["top1_match"] for record in informative]))
            if informative
            else float("nan")
        ),
        "random_top1_probability": (
            float(
                np.mean(
                    [
                        record["random_top1_probability"]
                        for record in informative
                    ]
                )
            )
            if informative
            else float("nan")
        ),
        "behavior_proxy_pairwise_sign_accuracy": (
            float(behavior_correct_pairs / behavior_pairs)
            if behavior_pairs
            else float("nan")
        ),
        "behavior_proxy_top1_match_rate": (
            float(np.mean(behavior_top1_values))
            if behavior_top1_values
            else float("nan")
        ),
        "behavior_proxy_mean_realized_regret": (
            float(np.mean(behavior_regret_values))
            if behavior_regret_values
            else float("nan")
        ),
        "policy_prior_pairwise_sign_accuracy": (
            float(policy_correct_pairs / policy_pairs)
            if policy_pairs
            else float("nan")
        ),
        "policy_prior_top1_match_rate": (
            float(np.mean(policy_top1_values))
            if policy_top1_values
            else float("nan")
        ),
        "policy_prior_mean_realized_regret": (
            float(np.mean(policy_regret_values))
            if policy_regret_values
            else float("nan")
        ),
        "mean_realized_regret": (
            float(
                np.mean(
                    [record["realized_regret"] for record in informative]
                )
            )
            if informative
            else float("nan")
        ),
        "state_q_return_span_spearman": _spearman(
            state_q_span,
            state_return_span,
        ),
        "dimension_q_return_span_spearman": _spearman(
            dimension_q_span,
            dimension_return_span,
        ),
        "state_bootstrap": _state_bootstrap(
            records,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "seed_bootstrap": _seed_cluster_bootstrap(
            records,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + 1,
        ),
        "records": records,
    }


def _seed_cluster_bootstrap(
    records: list[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap whole simulator seeds to preserve within-trajectory dependence."""

    informative = [
        record for record in records if record["num_informative_pairs"] > 0
    ]
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in informative:
        grouped.setdefault(int(record["eval_seed"]), []).append(record)
    seed_ids = sorted(grouped)
    payload = {
        "unit": "informative_eval_seed",
        "confidence": 0.95,
        "num_replicates": int(replicates),
        "num_seeds": len(seed_ids),
        "num_states": len(informative),
        "pairwise_sign_accuracy_ci": [float("nan"), float("nan")],
        "mean_spearman_ci": [float("nan"), float("nan")],
        "top1_match_rate_ci": [float("nan"), float("nan")],
        "mean_realized_regret_ci": [float("nan"), float("nan")],
    }
    if not seed_ids or replicates <= 0:
        return payload

    rng = np.random.default_rng(int(seed))
    samples = {
        "pairwise": np.full(replicates, np.nan, np.float64),
        "spearman": np.full(replicates, np.nan, np.float64),
        "top1": np.full(replicates, np.nan, np.float64),
        "regret": np.full(replicates, np.nan, np.float64),
    }
    for bootstrap_index in range(replicates):
        selected = rng.integers(0, len(seed_ids), size=len(seed_ids))
        sampled_records = [
            record
            for selected_index in selected
            for record in grouped[seed_ids[int(selected_index)]]
        ]
        pair_counts = np.asarray(
            [record["num_informative_pairs"] for record in sampled_records],
            np.float64,
        )
        pair_correct = np.asarray(
            [
                record["pairwise_sign_accuracy"]
                * record["num_informative_pairs"]
                for record in sampled_records
            ],
            np.float64,
        )
        samples["pairwise"][bootstrap_index] = float(
            pair_correct.sum() / pair_counts.sum()
        )
        spearman = np.asarray(
            [record["spearman"] for record in sampled_records],
            np.float64,
        )
        finite_spearman = spearman[np.isfinite(spearman)]
        if finite_spearman.size:
            samples["spearman"][bootstrap_index] = float(
                finite_spearman.mean()
            )
        samples["top1"][bootstrap_index] = float(
            np.mean([record["top1_match"] for record in sampled_records])
        )
        samples["regret"][bootstrap_index] = float(
            np.mean(
                [
                    record["realized_regret"]
                    for record in sampled_records
                ]
            )
        )

    payload.update(
        {
            "pairwise_sign_accuracy_ci": _percentile_interval(
                samples["pairwise"]
            ),
            "mean_spearman_ci": _percentile_interval(
                samples["spearman"]
            ),
            "top1_match_rate_ci": _percentile_interval(samples["top1"]),
            "mean_realized_regret_ci": _percentile_interval(
                samples["regret"]
            ),
        }
    )
    return payload


def _branch_coverage_summary(
    dataset: dict[str, Any],
    *,
    return_atol: float,
) -> dict[str, Any]:
    """Summarize where a branch dataset contains an identifiable action effect."""

    returns = np.asarray(dataset["returns"], np.float64)
    dimensions = np.asarray(dataset["action_dimensions"], np.int32)
    metadata = list(dataset["metadata"])
    if returns.ndim != 2:
        raise ValueError(
            f"branch returns must be rank two, got shape {returns.shape}"
        )
    if dimensions.shape != (returns.shape[0],):
        raise ValueError(
            "action_dimensions must have one entry per branch state"
        )
    if len(metadata) != returns.shape[0]:
        raise ValueError("metadata must have one entry per branch state")

    return_spans = np.ptp(returns, axis=1)
    informative = return_spans > float(return_atol)
    has_outcomes = all("outcomes" in item for item in metadata)
    any_success = all_success = None
    if has_outcomes:
        any_success = np.asarray(
            [
                any(
                    bool(outcome.get("success", False))
                    for outcome in item["outcomes"]
                )
                for item in metadata
            ],
            dtype=np.bool_,
        )
        all_success = np.asarray(
            [
                bool(item["outcomes"])
                and all(
                    bool(outcome.get("success", False))
                    for outcome in item["outcomes"]
                )
                for item in metadata
            ],
            dtype=np.bool_,
        )

    def group(indices: np.ndarray) -> dict[str, Any]:
        count = int(indices.size)
        informative_count = int(np.sum(informative[indices]))
        payload = {
            "num_states": count,
            "num_informative_states": informative_count,
            "informative_fraction": (
                float(informative_count / count) if count else float("nan")
            ),
            "mean_return_span": (
                float(np.mean(return_spans[indices]))
                if count
                else float("nan")
            ),
            "max_return_span": (
                float(np.max(return_spans[indices]))
                if count
                else float("nan")
            ),
        }
        if has_outcomes:
            any_success_count = int(np.sum(any_success[indices]))
            all_success_count = int(np.sum(all_success[indices]))
            payload.update(
                {
                    "num_any_success_states": any_success_count,
                    "any_success_fraction": (
                        float(any_success_count / count)
                        if count
                        else float("nan")
                    ),
                    "num_all_success_states": all_success_count,
                    "all_success_fraction": (
                        float(all_success_count / count)
                        if count
                        else float("nan")
                    ),
                }
            )
        return payload

    all_indices = np.arange(returns.shape[0], dtype=np.int64)
    anchor_steps = np.asarray(
        [int(item["anchor_step"]) for item in metadata],
        np.int32,
    )
    return {
        "overall": group(all_indices),
        "by_dimension": {
            str(int(dimension)): group(
                np.flatnonzero(dimensions == dimension)
            )
            for dimension in np.unique(dimensions)
        },
        "by_anchor_step": {
            str(int(anchor_step)): group(
                np.flatnonzero(anchor_steps == anchor_step)
            )
            for anchor_step in np.unique(anchor_steps)
        },
    }


def _train_oracle_critic(
    agent,
    initial_critic_params,
    dataset: dict[str, Any],
    *,
    updates: int,
    batch_size: int,
    learning_rate: float,
    temperature: float,
    weight_decay: float,
    return_atol: float,
    seed: int,
    score_level: int,
    delta_regression_weight: float,
    sampling_mode: str = "random_balanced",
):
    import jax
    import jax.numpy as jnp

    if updates < 1:
        raise ValueError("--updates must be at least one")
    if batch_size < 1:
        raise ValueError("--batch-size must be at least one")
    if learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if temperature <= 0:
        raise ValueError("--temperature must be positive")
    if delta_regression_weight < 0:
        raise ValueError("--delta-regression-weight must be non-negative")
    if sampling_mode not in {"random_balanced", "full_batch"}:
        raise ValueError(f"unsupported sampling mode: {sampling_mode}")

    returns = np.asarray(dataset["returns"], np.float32)
    informative_indices = np.flatnonzero(np.ptp(returns, axis=1) > return_atol)
    if not informative_indices.size:
        raise RuntimeError("training split has no informative branch states")
    all_indices = np.arange(returns.shape[0], dtype=np.int64)

    optimizer = agent.optax.adamw(
        learning_rate,
        weight_decay=weight_decay,
    )
    opt_state = optimizer.init(initial_critic_params)
    candidate_count = int(returns.shape[1])
    pair_left, pair_right = np.triu_indices(candidate_count, k=1)
    left = jnp.asarray(pair_left, dtype=jnp.int32)
    right = jnp.asarray(pair_right, dtype=jnp.int32)

    def loss_fn(critic_params, batch):
        q_values = _score_candidates(
            agent,
            critic_params,
            batch["features"],
            batch["actions"],
            batch["action_dimensions"],
            score_level,
        )
        return_delta = batch["returns"][:, left] - batch["returns"][:, right]
        labels = jnp.sign(return_delta)
        mask = (jnp.abs(return_delta) > return_atol).astype(jnp.float32)
        q_delta = q_values[:, left] - q_values[:, right]
        pair_loss = jax.nn.softplus(-labels * q_delta / temperature)
        denominator = jnp.maximum(jnp.sum(mask), 1.0)
        ranking_loss = jnp.sum(pair_loss * mask) / denominator
        delta_error = q_delta - return_delta
        abs_delta_error = jnp.abs(delta_error)
        delta_regression_loss = jnp.mean(
            jnp.where(
                abs_delta_error <= 1.0,
                0.5 * jnp.square(delta_error),
                abs_delta_error - 0.5,
            )
        )
        loss = ranking_loss + (
            float(delta_regression_weight) * delta_regression_loss
        )
        accuracy = jnp.sum(
            (jnp.sign(q_delta) == labels).astype(jnp.float32) * mask
        ) / denominator
        return loss, (
            accuracy,
            jnp.mean(jnp.ptp(q_values, axis=1)),
            ranking_loss,
            delta_regression_loss,
        )

    def update_step(critic_params, current_opt_state, batch):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            critic_params,
            batch,
        )
        parameter_updates, current_opt_state = optimizer.update(
            grads,
            current_opt_state,
            critic_params,
        )
        critic_params = agent.optax.apply_updates(
            critic_params,
            parameter_updates,
        )
        return critic_params, current_opt_state, loss, aux

    update_step = jax.jit(update_step)
    rng = np.random.default_rng(seed)
    critic_params = initial_critic_params
    history = []
    log_steps = set(
        np.linspace(0, updates - 1, num=min(11, updates), dtype=int).tolist()
    )
    for update_index in range(updates):
        if sampling_mode == "full_batch":
            # Cached states are already grouped uniformly by collection seed,
            # anchor, and action dimension.  Visiting each exactly once removes
            # the high-variance 10-minibatch lottery seen in the branch-CV gate.
            indices = all_indices
        elif delta_regression_weight > 0:
            informative_batch_size = max(1, batch_size // 2)
            indices = np.concatenate(
                [
                    rng.choice(
                        informative_indices,
                        size=informative_batch_size,
                        replace=True,
                    ),
                    rng.choice(
                        all_indices,
                        size=batch_size - informative_batch_size,
                        replace=True,
                    ),
                ]
            )
            rng.shuffle(indices)
        else:
            indices = rng.choice(
                informative_indices,
                size=batch_size,
                replace=True,
            )
        batch = {
            "features": jnp.asarray(dataset["features"][indices]),
            "actions": jnp.asarray(dataset["actions"][indices]),
            "returns": jnp.asarray(dataset["returns"][indices]),
            "action_dimensions": jnp.asarray(
                dataset["action_dimensions"][indices]
            ),
        }
        critic_params, opt_state, loss, aux = update_step(
            critic_params,
            opt_state,
            batch,
        )
        if update_index in log_steps:
            loss, aux = jax.device_get((loss, aux))
            history.append(
                {
                    "update": update_index + 1,
                    "loss": float(loss),
                    "batch_pairwise_accuracy": float(aux[0]),
                    "batch_q_span": float(aux[1]),
                    "ranking_loss": float(aux[2]),
                    "delta_regression_loss": float(aux[3]),
                }
            )
    return critic_params, history, int(informative_indices.size)


def _all_scores(
    agent,
    critic_params,
    dataset: dict[str, Any],
    *,
    score_level: int,
    score_batch_size: int = 64,
) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    if score_batch_size < 1:
        raise ValueError("score_batch_size must be positive")
    count = int(dataset["features"].shape[0])
    score_chunks = []
    for start in range(0, count, score_batch_size):
        stop = min(start + score_batch_size, count)
        scores = _score_candidates(
            agent,
            critic_params,
            jnp.asarray(dataset["features"][start:stop]),
            jnp.asarray(dataset["actions"][start:stop]),
            jnp.asarray(dataset["action_dimensions"][start:stop]),
            score_level,
        )
        score_chunks.append(np.asarray(jax.device_get(scores)))
    return np.concatenate(score_chunks, axis=0)


def _replace_agent_critic(agent, critic_params) -> None:
    if getattr(agent, "hybrid_flow_v_direct_a", False):
        if hasattr(agent.params, "copy"):
            try:
                agent.params = agent.params.copy(
                    {"advantage": critic_params}
                )
            except TypeError:
                params = dict(agent.params)
                params["advantage"] = critic_params
                agent.params = params
        else:
            params = dict(agent.params)
            params["advantage"] = critic_params
            agent.params = params
        target_params = dict(agent.target_critic_params)
        target_params["advantage"] = copy.deepcopy(critic_params)
        agent.target_critic_params = target_params
        return

    if hasattr(agent.params, "copy"):
        try:
            agent.params = agent.params.copy({"critic": critic_params})
        except TypeError:
            params = dict(agent.params)
            params["critic"] = critic_params
            agent.params = params
    else:
        params = dict(agent.params)
        params["critic"] = critic_params
        agent.params = params
    agent.target_critic_params = copy.deepcopy(critic_params)


def _write_finetuned_snapshot(
    source_snapshot: Path,
    output_snapshot: Path,
    agent,
    metadata: dict[str, Any],
) -> None:
    with source_snapshot.open("rb") as file:
        payload = pickle.load(file)
    payload["agent"] = agent.state_dict()
    # The original full-training optimizer is intentionally retained.  This
    # checkpoint is an analysis artifact; branch probes only read parameters.
    payload["branch_oracle_metadata"] = metadata
    output_snapshot.parent.mkdir(parents=True, exist_ok=True)
    with output_snapshot.open("wb") as file:
        pickle.dump(payload, file)


def _write_dataset_cache(
    path: Path,
    train_data: dict[str, Any],
    heldout_data: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "train_features": train_data["features"],
        "train_actions": train_data["actions"],
        "train_returns": train_data["returns"],
        "train_action_dimensions": train_data["action_dimensions"],
        "train_metadata": np.asarray(json.dumps(train_data["metadata"])),
        "heldout_features": heldout_data["features"],
        "heldout_actions": heldout_data["actions"],
        "heldout_returns": heldout_data["returns"],
        "heldout_action_dimensions": heldout_data["action_dimensions"],
        "heldout_metadata": np.asarray(json.dumps(heldout_data["metadata"])),
        "cache_metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    for prefix, data in (("train", train_data), ("heldout", heldout_data)):
        if "return_samples" in data:
            arrays[f"{prefix}_return_samples"] = data["return_samples"]
    np.savez_compressed(path, **arrays)


def _trees_bitwise_equal(left: Any, right: Any) -> bool:
    """Return whether two nested array trees have identical leaves."""

    import jax

    if jax.tree_util.tree_structure(left) != jax.tree_util.tree_structure(
        right
    ):
        return False
    return all(
        np.array_equal(np.asarray(left_leaf), np.asarray(right_leaf))
        for left_leaf, right_leaf in zip(
            jax.tree_util.tree_leaves(left),
            jax.tree_util.tree_leaves(right),
            strict=True,
        )
    )


def _frozen_policy_state(agent) -> dict[str, Any]:
    """Extract every state component that defines cached features/BC actions."""

    state = agent.state_dict()
    params = state["params"]
    required = ("encoder", "policy", "policy_encoder")
    missing = [name for name in required if name not in params]
    if missing:
        raise ValueError(
            "matched comparison requires distinct frozen policy/value towers; "
            f"missing params {missing}"
        )
    frozen = {name: copy.deepcopy(params[name]) for name in required}
    if "encoder_state" in state:
        frozen["encoder_state"] = copy.deepcopy(state["encoder_state"])
    return frozen


def _load_dataset_cache(
    path: Path,
    expected_metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["cache_metadata"].item()))
        if metadata != expected_metadata:
            raise ValueError(
                "dataset cache metadata mismatch: "
                f"expected {expected_metadata}, found {metadata}"
            )

        def split(prefix: str) -> dict[str, Any]:
            result = {
                "features": np.asarray(payload[f"{prefix}_features"]),
                "actions": np.asarray(payload[f"{prefix}_actions"]),
                "returns": np.asarray(payload[f"{prefix}_returns"]),
                "action_dimensions": np.asarray(
                    payload[f"{prefix}_action_dimensions"], np.int32
                ),
                "metadata": json.loads(
                    str(payload[f"{prefix}_metadata"].item())
                ),
            }
            samples_key = f"{prefix}_return_samples"
            result["return_samples"] = (
                np.asarray(payload[samples_key])
                if samples_key in payload
                else result["returns"][..., None]
            )
            return result

        return split("train"), split("heldout")


def run(args: argparse.Namespace) -> dict[str, Any]:
    from omegaconf import OmegaConf

    from robobase.workspace import Workspace

    run_dir = args.run_dir.expanduser().resolve()
    snapshot = (
        args.snapshot
        if args.snapshot is not None
        else run_dir / "snapshots" / "latest_snapshot.pkl"
    ).expanduser().resolve()
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not snapshot.is_file():
        raise FileNotFoundError(snapshot)
    cfg = OmegaConf.load(config_path)
    method_name = str(cfg.method.name).lower()
    if method_name not in {"cqn_as", "cqn_flow"}:
        raise ValueError(
            "branch oracle fine-tuning requires CQN-AS or CQN-Flow"
        )
    if not bool(cfg.method.get("separate_bc_policy", False)):
        raise ValueError("branch oracle requires separate_bc_policy=true")
    if cfg.method.get("policy_value_beta", None) is not None:
        raise ValueError(
            "branch oracle requires policy_value_beta=null so every branch "
            "uses the same independent-BC continuation"
        )
    if args.comparison_snapshot is not None and not args.coverage_only:
        raise ValueError(
            "comparison_snapshot is valid only with --coverage-only"
        )
    if (
        args.comparison_snapshot is not None
        and args.finetuned_snapshot is not None
    ):
        raise ValueError(
            "comparison_snapshot and finetuned_snapshot are mutually exclusive"
        )

    train_seeds = _integer_list(args.train_seeds, "--train-seeds")
    heldout_seeds = _integer_list(args.heldout_seeds, "--heldout-seeds")
    if set(train_seeds) & set(heldout_seeds):
        raise ValueError("train and held-out seeds must be disjoint")
    oracle_validation_seed = (
        None
        if args.oracle_validation_seed is None
        else int(args.oracle_validation_seed)
    )
    if (
        oracle_validation_seed is not None
        and oracle_validation_seed not in train_seeds
    ):
        raise ValueError(
            "oracle validation seed must be one of --train-seeds"
        )
    anchor_steps = sorted(
        set(_integer_list(args.anchor_steps, "--anchor-steps"))
    )
    action_dimensions = sorted(
        set(_integer_list(args.action_dimensions, "--action-dimensions"))
    )
    if anchor_steps[0] < 0:
        raise ValueError("anchor steps must be non-negative")
    if args.intervention_horizon < 1:
        raise ValueError("intervention horizon must be at least one")
    if args.max_continuation_steps < 1:
        raise ValueError("max continuation steps must be at least one")
    if args.continuation_repeats < 1:
        raise ValueError("continuation repeats must be at least one")
    if args.return_atol < 0:
        raise ValueError("return atol must be non-negative")
    baseline_outcome = str(args.baseline_outcome)

    OmegaConf.set_struct(cfg, False)
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
    with tempfile.TemporaryDirectory(prefix="cqn-branch-oracle-") as work_dir:
        workspace = Workspace(cfg, work_dir=work_dir)
        try:
            workspace.load_snapshot(snapshot, load_replay_buffer=False)
            workspace._ensure_eval_envs_created()
            agent = workspace.agent
            hybrid_advantage = bool(
                getattr(agent, "hybrid_flow_v_direct_a", False)
            )
            direct_scalar_q = bool(
                getattr(agent, "direct_scalar_q", False)
            )
            frozen_policy_before_fit = (
                _frozen_policy_state(agent)
                if direct_scalar_q and not args.coverage_only
                else None
            )
            if method_name == "cqn_flow" and not hybrid_advantage:
                raise ValueError(
                    "CQN-Flow branch oracle currently requires "
                    "critic_architecture=flow_v_direct_a"
                )
            invalid_dimensions = [
                dimension
                for dimension in action_dimensions
                if dimension < 0 or dimension >= agent.action_dim
            ]
            if invalid_dimensions:
                raise ValueError(
                    "action dimensions outside valid range: "
                    f"{invalid_dimensions}; action_dim={agent.action_dim}"
                )
            if args.candidate_mode == "sibling_bins":
                score_level = (
                    int(args.force_level)
                    if int(args.score_level) == -1
                    else int(args.score_level)
                )
                if score_level != int(args.force_level):
                    raise ValueError(
                        "sibling_bins must be scored at --force-level so all "
                        "candidates share one prefix"
                    )
            else:
                score_level = (
                    agent.levels - 1
                    if int(args.score_level) == -1
                    else int(args.score_level)
                )
            if score_level < 0 or score_level >= agent.levels:
                raise ValueError(
                    f"score level {score_level} outside "
                    f"[0, {agent.levels - 1}]"
                )
            if hybrid_advantage:
                base_critic_params = (
                    agent.target_critic_params["advantage"]
                    if agent.use_target_network_for_rollout
                    else agent.params["advantage"]
                )
            else:
                base_critic_params = (
                    agent.target_critic_params
                    if agent.use_target_network_for_rollout
                    else agent.params["critic"]
                )

            cache_metadata = {
                "source_snapshot": str(snapshot),
                "train_seeds": train_seeds,
                "heldout_seeds": heldout_seeds,
                "anchor_steps": anchor_steps,
                "action_dimensions": action_dimensions,
                "candidate_mode": str(args.candidate_mode),
                "force_level": int(args.force_level),
                "intervention_horizon": int(args.intervention_horizon),
                "max_continuation_steps": int(args.max_continuation_steps),
                "gamma": gamma,
            }
            if baseline_outcome != "all":
                cache_metadata["baseline_outcome"] = baseline_outcome
            if (
                int(args.continuation_repeats) != 1
                or str(args.continuation_rng_mode) != "restored"
            ):
                cache_metadata.update(
                    {
                        "continuation_repeats": int(
                            args.continuation_repeats
                        ),
                        "continuation_rng_mode": str(
                            args.continuation_rng_mode
                        ),
                        "continuation_seed_offset": int(
                            args.continuation_seed_offset
                        ),
                    }
                )
            dataset_cache = (
                args.dataset_cache.expanduser().resolve()
                if args.dataset_cache is not None
                else None
            )
            if dataset_cache is not None and dataset_cache.is_file():
                train_data, heldout_data = _load_dataset_cache(
                    dataset_cache,
                    cache_metadata,
                )
            else:
                train_data = _collect_branch_split(
                    workspace,
                    seeds=train_seeds,
                    anchor_steps=anchor_steps,
                    action_dimensions=action_dimensions,
                    intervention_horizon=int(args.intervention_horizon),
                    max_continuation_steps=int(args.max_continuation_steps),
                    gamma=gamma,
                    candidate_mode=str(args.candidate_mode),
                    force_level=int(args.force_level),
                    baseline_outcome=baseline_outcome,
                    continuation_repeats=int(args.continuation_repeats),
                    continuation_rng_mode=str(args.continuation_rng_mode),
                    continuation_seed_offset=int(
                        args.continuation_seed_offset
                    ),
                )
                heldout_data = _collect_branch_split(
                    workspace,
                    seeds=heldout_seeds,
                    anchor_steps=anchor_steps,
                    action_dimensions=action_dimensions,
                    intervention_horizon=int(args.intervention_horizon),
                    max_continuation_steps=int(args.max_continuation_steps),
                    gamma=gamma,
                    candidate_mode=str(args.candidate_mode),
                    force_level=int(args.force_level),
                    baseline_outcome=baseline_outcome,
                    continuation_repeats=int(args.continuation_repeats),
                    continuation_rng_mode=str(args.continuation_rng_mode),
                    continuation_seed_offset=int(
                        args.continuation_seed_offset
                    ),
                )
                if dataset_cache is not None:
                    _write_dataset_cache(
                        dataset_cache,
                        train_data,
                        heldout_data,
                        cache_metadata,
                    )

            coverage = {
                "train": _branch_coverage_summary(
                    train_data,
                    return_atol=float(args.return_atol),
                ),
                "heldout": _branch_coverage_summary(
                    heldout_data,
                    return_atol=float(args.return_atol),
                ),
            }
            return_stochasticity = {
                "train": _return_stochasticity_summary(
                    train_data,
                    return_atol=float(args.return_atol),
                ),
                "heldout": _return_stochasticity_summary(
                    heldout_data,
                    return_atol=float(args.return_atol),
                ),
            }
            if args.coverage_only:
                train_scores = _all_scores(
                    agent,
                    base_critic_params,
                    train_data,
                    score_level=score_level,
                )
                heldout_scores = _all_scores(
                    agent,
                    base_critic_params,
                    heldout_data,
                    score_level=score_level,
                )
                comparison_snapshot = None
                frozen_component_bitwise_equal = None
                comparison_train_scores = None
                comparison_heldout_scores = None
                if args.comparison_snapshot is not None:
                    if dataset_cache is None or not dataset_cache.is_file():
                        raise ValueError(
                            "comparison_snapshot requires an existing "
                            "--dataset-cache"
                        )
                    comparison_snapshot = (
                        args.comparison_snapshot.expanduser().resolve()
                    )
                    if not comparison_snapshot.is_file():
                        raise FileNotFoundError(comparison_snapshot)
                    frozen_before = _frozen_policy_state(agent)
                    workspace.load_snapshot(
                        comparison_snapshot,
                        load_replay_buffer=False,
                    )
                    agent = workspace.agent
                    frozen_after = _frozen_policy_state(agent)
                    frozen_component_bitwise_equal = (
                        _trees_bitwise_equal(frozen_before, frozen_after)
                    )
                    if not frozen_component_bitwise_equal:
                        raise ValueError(
                            "comparison snapshot changed encoder or BC policy; "
                            "cached conditions are not matched"
                        )
                    comparison_metadata = getattr(
                        workspace,
                        "branch_oracle_metadata",
                        {},
                    )
                    recorded_source = comparison_metadata.get(
                        "source_snapshot"
                    )
                    if (
                        recorded_source is None
                        or Path(recorded_source).expanduser().resolve()
                        != snapshot
                    ):
                        raise ValueError(
                            "comparison snapshot is not a branch-oracle "
                            f"descendant of {snapshot}"
                        )
                    comparison_hybrid = bool(
                        getattr(agent, "hybrid_flow_v_direct_a", False)
                    )
                    if comparison_hybrid != hybrid_advantage:
                        raise ValueError(
                            "comparison snapshot changed critic architecture"
                        )
                    if comparison_hybrid:
                        comparison_params = (
                            agent.target_critic_params["advantage"]
                            if agent.use_target_network_for_rollout
                            else agent.params["advantage"]
                        )
                    else:
                        comparison_params = (
                            agent.target_critic_params
                            if agent.use_target_network_for_rollout
                            else agent.params["critic"]
                        )
                    comparison_train_scores = _all_scores(
                        agent,
                        comparison_params,
                        train_data,
                        score_level=score_level,
                    )
                    comparison_heldout_scores = _all_scores(
                        agent,
                        comparison_params,
                        heldout_data,
                        score_level=score_level,
                    )

                def score_summary(data, scores, bootstrap_seed):
                    records = _records_from_scores(
                        scores,
                        data["returns"],
                        data["metadata"],
                        return_atol=float(args.return_atol),
                    )
                    return _summarize_records(
                        records,
                        bootstrap_replicates=int(
                            args.bootstrap_replicates
                        ),
                        bootstrap_seed=bootstrap_seed,
                    )

                results = {
                    "train_before": score_summary(
                        train_data,
                        train_scores,
                        int(args.seed) + 100,
                    ),
                    "heldout_before": score_summary(
                        heldout_data,
                        heldout_scores,
                        int(args.seed) + 101,
                    ),
                }
                if comparison_snapshot is not None:
                    results.update(
                        {
                            "train_after": score_summary(
                                train_data,
                                comparison_train_scores,
                                int(args.seed) + 102,
                            ),
                            "heldout_after": score_summary(
                                heldout_data,
                                comparison_heldout_scores,
                                int(args.seed) + 103,
                            ),
                        }
                    )
                return {
                    "status": "ok",
                    "target_estimator": "simulator_branch_monte_carlo",
                    "continuation_policy": "frozen_independent_bc",
                    "continuation_policy_value_beta": None,
                    "run_dir": str(run_dir),
                    "source_snapshot": str(snapshot),
                    "comparison_snapshot": (
                        str(comparison_snapshot)
                        if comparison_snapshot is not None
                        else None
                    ),
                    "frozen_component_bitwise_equal": (
                        frozen_component_bitwise_equal
                    ),
                    "finetuned_snapshot": None,
                    "dataset_cache": (
                        str(dataset_cache)
                        if dataset_cache is not None
                        else None
                    ),
                    "train_seeds": train_seeds,
                    "heldout_seeds": heldout_seeds,
                    "anchor_steps": anchor_steps,
                    "action_dimensions": action_dimensions,
                    "candidate_mode": str(args.candidate_mode),
                    "force_level": int(args.force_level),
                    "trained_component": "none",
                    "critic_parameterization": (
                        "direct_scalar_q"
                        if direct_scalar_q
                        else (
                            "flow_v_direct_a"
                            if hybrid_advantage
                            else "categorical_c51"
                        )
                    ),
                    "coverage_only": True,
                    "baseline_outcome": baseline_outcome,
                    "intervention_horizon": int(
                        args.intervention_horizon
                    ),
                    "score_level": score_level,
                    "max_continuation_steps": int(
                        args.max_continuation_steps
                    ),
                    "continuation_repeats": int(
                        args.continuation_repeats
                    ),
                    "continuation_rng_mode": str(
                        args.continuation_rng_mode
                    ),
                    "num_train_states": int(
                        train_data["features"].shape[0]
                    ),
                    "num_heldout_states": int(
                        heldout_data["features"].shape[0]
                    ),
                    "num_informative_train_states": int(
                        coverage["train"]["overall"][
                            "num_informative_states"
                        ]
                    ),
                    "updates": 0,
                    "batch_size": int(args.batch_size),
                    "learning_rate": float(args.learning_rate),
                    "temperature": float(args.temperature),
                    "delta_regression_weight": float(
                        args.delta_regression_weight
                    ),
                    "train_return_shuffle": str(
                        args.train_return_shuffle
                    ),
                    "oracle_validation_seed": oracle_validation_seed,
                    "weight_decay": float(args.weight_decay),
                    "coverage": coverage,
                    "return_stochasticity": return_stochasticity,
                    "training_history": [],
                    "results": results,
                }

            train_scores_before = _all_scores(
                agent,
                base_critic_params,
                train_data,
                score_level=score_level,
            )
            oracle_fit_data = train_data
            oracle_validation_data = None
            if oracle_validation_seed is not None:
                train_record_seeds = np.asarray(
                    [
                        int(record["eval_seed"])
                        for record in train_data["metadata"]
                    ],
                    np.int32,
                )
                oracle_validation_mask = (
                    train_record_seeds == oracle_validation_seed
                )
                oracle_validation_data = _subset_branch_dataset(
                    train_data,
                    oracle_validation_mask,
                )
                oracle_fit_data = _subset_branch_dataset(
                    train_data,
                    ~oracle_validation_mask,
                )
            heldout_scores_before = _all_scores(
                agent,
                base_critic_params,
                heldout_data,
                score_level=score_level,
            )
            oracle_train_data = _oracle_training_data(
                oracle_fit_data,
                shuffle_mode=str(args.train_return_shuffle),
                seed=int(args.seed),
            )
            critic_params, training_history, informative_train_states = (
                _train_oracle_critic(
                    agent,
                    base_critic_params,
                    oracle_train_data,
                    updates=int(args.updates),
                    batch_size=int(args.batch_size),
                    learning_rate=float(args.learning_rate),
                    temperature=float(args.temperature),
                    weight_decay=float(args.weight_decay),
                    return_atol=float(args.return_atol),
                    seed=int(args.seed),
                    score_level=score_level,
                    delta_regression_weight=float(
                        args.delta_regression_weight
                    ),
                    sampling_mode=str(args.sampling_mode),
                )
            )
            train_scores_after = _all_scores(
                agent,
                critic_params,
                train_data,
                score_level=score_level,
            )
            heldout_scores_after = _all_scores(
                agent,
                critic_params,
                heldout_data,
                score_level=score_level,
            )
            validation_scores_before = None
            validation_scores_after = None
            if oracle_validation_data is not None:
                validation_scores_before = _all_scores(
                    agent,
                    base_critic_params,
                    oracle_validation_data,
                    score_level=score_level,
                )
                validation_scores_after = _all_scores(
                    agent,
                    critic_params,
                    oracle_validation_data,
                    score_level=score_level,
                )

            def summary(data, scores, bootstrap_seed):
                records = _records_from_scores(
                    scores,
                    data["returns"],
                    data["metadata"],
                    return_atol=float(args.return_atol),
                )
                return _summarize_records(
                    records,
                    bootstrap_replicates=int(args.bootstrap_replicates),
                    bootstrap_seed=bootstrap_seed,
                )

            results = {
                "train_before": summary(
                    train_data, train_scores_before, int(args.seed) + 100
                ),
                "train_after": summary(
                    train_data, train_scores_after, int(args.seed) + 101
                ),
                "heldout_before": summary(
                    heldout_data,
                    heldout_scores_before,
                    int(args.seed) + 102,
                ),
                "heldout_after": summary(
                    heldout_data,
                    heldout_scores_after,
                    int(args.seed) + 103,
                ),
            }
            if oracle_validation_data is not None:
                results.update(
                    {
                        "validation_before": summary(
                            oracle_validation_data,
                            validation_scores_before,
                            int(args.seed) + 104,
                        ),
                        "validation_after": summary(
                            oracle_validation_data,
                            validation_scores_after,
                            int(args.seed) + 105,
                        ),
                    }
                )
            _replace_agent_critic(agent, critic_params)
            frozen_policy_bitwise_equal_after_fit = (
                _trees_bitwise_equal(
                    frozen_policy_before_fit,
                    _frozen_policy_state(agent),
                )
                if frozen_policy_before_fit is not None
                else None
            )
            if (
                direct_scalar_q
                and not frozen_policy_bitwise_equal_after_fit
            ):
                raise RuntimeError(
                    "counterfactual critic fine-tuning changed the frozen "
                    "encoder or independent BC policy"
                )

            finetuned_snapshot = (
                args.finetuned_snapshot.expanduser().resolve()
                if args.finetuned_snapshot is not None
                else None
            )
            artifact_metadata = {
                "source_snapshot": str(snapshot),
                "train_seeds": train_seeds,
                "heldout_seeds": heldout_seeds,
                "anchor_steps": anchor_steps,
                "action_dimensions": action_dimensions,
                "candidate_mode": str(args.candidate_mode),
                "force_level": int(args.force_level),
                "intervention_horizon": int(args.intervention_horizon),
                "continuation_repeats": int(args.continuation_repeats),
                "continuation_rng_mode": str(args.continuation_rng_mode),
                "score_level": score_level,
                "initialization_seed": int(args.seed),
                "updates": int(args.updates),
                "learning_rate": float(args.learning_rate),
                "temperature": float(args.temperature),
                "delta_regression_weight": float(
                    args.delta_regression_weight
                ),
                "train_return_shuffle": str(args.train_return_shuffle),
                "sampling_mode": str(args.sampling_mode),
                "oracle_validation_seed": oracle_validation_seed,
            }
            if finetuned_snapshot is not None:
                _write_finetuned_snapshot(
                    snapshot,
                    finetuned_snapshot,
                    agent,
                    artifact_metadata,
                )
        finally:
            workspace.shutdown()

    return {
        "status": "ok",
        "target_estimator": "simulator_branch_monte_carlo",
        "continuation_policy": "frozen_independent_bc",
        "continuation_policy_value_beta": None,
        "run_dir": str(run_dir),
        "source_snapshot": str(snapshot),
        "finetuned_snapshot": (
            str(finetuned_snapshot) if finetuned_snapshot is not None else None
        ),
        "dataset_cache": (
            str(dataset_cache) if dataset_cache is not None else None
        ),
        "train_seeds": train_seeds,
        "heldout_seeds": heldout_seeds,
        "anchor_steps": anchor_steps,
        "action_dimensions": action_dimensions,
        "candidate_mode": str(args.candidate_mode),
        "force_level": int(args.force_level),
        "trained_component": (
            "advantage" if hybrid_advantage else "critic"
        ),
        "critic_parameterization": (
            "direct_scalar_q"
            if direct_scalar_q
            else (
                "flow_v_direct_a"
                if hybrid_advantage
                else "categorical_c51"
            )
        ),
        "frozen_policy_bitwise_equal_after_fit": (
            frozen_policy_bitwise_equal_after_fit
        ),
        "coverage_only": False,
        "baseline_outcome": baseline_outcome,
        "intervention_horizon": int(args.intervention_horizon),
        "score_level": score_level,
        "max_continuation_steps": int(args.max_continuation_steps),
        "continuation_repeats": int(args.continuation_repeats),
        "continuation_rng_mode": str(args.continuation_rng_mode),
        "num_train_states": int(train_data["features"].shape[0]),
        "num_heldout_states": int(heldout_data["features"].shape[0]),
        "num_informative_train_states": informative_train_states,
        "initialization_seed": int(args.seed),
        "updates": int(args.updates),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "temperature": float(args.temperature),
        "delta_regression_weight": float(args.delta_regression_weight),
        "train_return_shuffle": str(args.train_return_shuffle),
        "sampling_mode": str(args.sampling_mode),
        "oracle_validation_seed": oracle_validation_seed,
        "weight_decay": float(args.weight_decay),
        "coverage": coverage,
        "return_stochasticity": return_stochasticity,
        "training_history": training_history,
        "results": results,
    }


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
        compact = {
            "status": payload["status"],
            "elapsed_seconds": payload["elapsed_seconds"],
            "num_train_states": payload["num_train_states"],
            "num_informative_train_states": payload[
                "num_informative_train_states"
            ],
            "coverage_only": payload["coverage_only"],
            "coverage": payload["coverage"],
            "return_stochasticity": payload["return_stochasticity"],
            "results": {
                name: {
                    key: value
                    for key, value in result.items()
                    if key != "records"
                }
                for name, result in payload["results"].items()
            },
            "training_history": payload["training_history"],
            "finetuned_snapshot": payload["finetuned_snapshot"],
        }
        print(json.dumps(compact, indent=2, sort_keys=True))
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
