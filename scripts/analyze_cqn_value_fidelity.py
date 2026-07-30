#!/usr/bin/env python3
"""Audit whether a CQN-AS-family checkpoint ranks value or demo actions.

This is a read-only checkpoint diagnostic.  It evaluates replay action chunks
along their coarse-to-fine paths and reports two deliberately separate views:

* imitation: how often the replay/demo action bin is the critic argmax;
* value: correlation between the replay-action Q estimate and future return.

The latter is observational rather than causal.  It is a cheap first gate
before the stronger simulator branch-counterfactual audit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


GROUPS = (
    "demo_success",
    "demo_failure",
    "online_success",
    "online_failure",
)
EXPLORATION_GROUPS = ("online_explored", "online_unexplored")


@dataclass(frozen=True)
class Sample:
    episode_path: Path
    episode_index: int
    transition_index: int
    episode_length: int
    group: str
    discounted_return: float
    first_success_return: float
    future_success: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument(
        "--data-run-dir",
        type=Path,
        help="Run whose replay/*.npz supplies common observations/actions.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--samples-per-group", type=int, default=8)
    parser.add_argument(
        "--samples-per-exploration-group",
        type=int,
        default=0,
        help=(
            "Additional online transition samples stratified by the replay "
            "structured_explore flag. Zero disables this audit."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--offline-episode-count",
        type=int,
        help=(
            "Episodes inserted before online collection. This is required to "
            "distinguish failed demonstrations (whose demo label is zero) from "
            "failed online episodes."
        ),
    )
    parser.add_argument(
        "--critic",
        choices=("config", "online", "target"),
        default="config",
    )
    return parser.parse_args()


def configure_process(gpu_id: int | None) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.30")
    if gpu_id is not None and gpu_id >= 0:
        gpu = str(gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        os.environ["JAX_CUDA_VISIBLE_DEVICES"] = gpu
        os.environ["MUJOCO_EGL_DEVICE_ID"] = gpu


def _parse_episode_metadata(path: Path) -> tuple[int, int, int]:
    episode_index, episode_length, global_index = [
        int(value) for value in path.stem.split("_")[-3:]
    ]
    return episode_index, episode_length, global_index


def _discounted_sum(values: np.ndarray, gamma: float) -> np.ndarray:
    result = np.zeros(values.shape[0], dtype=np.float64)
    running = 0.0
    for index in range(values.shape[0] - 1, -1, -1):
        running = float(values[index]) + gamma * running
        result[index] = running
    return result


def _scan_episode(
    path: Path,
    gamma: float,
    offline_episode_count: int | None,
) -> tuple[str, int, np.ndarray, np.ndarray, np.ndarray | None]:
    episode_index, episode_length, _ = _parse_episode_metadata(path)
    with np.load(path) as episode:
        rewards = np.asarray(episode["reward"][:episode_length], np.float64)
        is_demo = bool(np.asarray(episode["demo"])[0])
        structured_explore = (
            np.asarray(
                episode["structured_explore"][:episode_length],
                dtype=np.bool_,
            )
            if "structured_explore" in episode.files
            else None
        )
    success_indices = np.flatnonzero(rewards > 0.25)
    successful = bool(success_indices.size)
    source = (
        "demo"
        if (
            episode_index < offline_episode_count
            if offline_episode_count is not None
            else is_demo
        )
        else "online"
    )
    group = f"{source}_{'success' if successful else 'failure'}"
    discounted_returns = _discounted_sum(rewards, gamma)
    first_success_returns = np.zeros(episode_length, dtype=np.float64)
    if successful:
        first_success = int(success_indices[0])
        earlier = np.arange(first_success + 1)
        first_success_returns[earlier] = gamma ** (first_success - earlier)
    return (
        group,
        episode_index,
        discounted_returns,
        first_success_returns,
        structured_explore,
    )


def select_samples(
    replay_dir: Path,
    *,
    gamma: float,
    samples_per_group: int,
    samples_per_exploration_group: int,
    seed: int,
    offline_episode_count: int | None,
) -> tuple[list[Sample], dict[str, int], dict[str, int]]:
    if samples_per_group < 1:
        raise ValueError("--samples-per-group must be at least 1")
    episodes: dict[str, list[tuple[Path, int, np.ndarray, np.ndarray]]] = {
        group: [] for group in GROUPS
    }
    episode_exploration: dict[Path, np.ndarray | None] = {}
    for path in sorted(replay_dir.glob("*.npz")):
        (
            group,
            episode_index,
            returns,
            first_success_returns,
            structured_explore,
        ) = _scan_episode(path, gamma, offline_episode_count)
        if group in episodes:
            episodes[group].append(
                (path, episode_index, returns, first_success_returns)
            )
            episode_exploration[path] = structured_explore

    counts = {group: len(items) for group, items in episodes.items()}
    missing = [group for group, items in episodes.items() if not items]
    if missing:
        raise ValueError(
            f"replay is missing required episode groups: {', '.join(missing)}"
        )

    rng = np.random.default_rng(seed)
    samples: list[Sample] = []
    for group in GROUPS:
        items = list(episodes[group])
        rng.shuffle(items)
        # Round-robin across episodes, then spread anchors through time.  This
        # avoids letting a single long trajectory dominate the diagnostic.
        rounds = int(math.ceil(samples_per_group / len(items)))
        made = 0
        for round_index in range(rounds):
            fraction = (round_index + 0.5) / rounds
            for path, episode_index, returns, first_success_returns in items:
                if made >= samples_per_group:
                    break
                episode_length = int(returns.shape[0])
                transition_index = min(
                    episode_length - 1,
                    max(0, int(round(fraction * (episode_length - 1)))),
                )
                samples.append(
                    Sample(
                        episode_path=path,
                        episode_index=episode_index,
                        transition_index=transition_index,
                        episode_length=episode_length,
                        group=group,
                        discounted_return=float(returns[transition_index]),
                        first_success_return=float(
                            first_success_returns[transition_index]
                        ),
                        future_success=bool(
                            first_success_returns[transition_index] > 0.0
                        ),
                    )
                )
                made += 1
    if samples_per_exploration_group < 0:
        raise ValueError(
            "--samples-per-exploration-group must be non-negative"
        )
    exploration_counts: dict[str, int] = {}
    if samples_per_exploration_group > 0:
        strata: dict[
            str,
            list[tuple[Path, int, np.ndarray, np.ndarray, np.ndarray]],
        ] = {group: [] for group in EXPLORATION_GROUPS}
        for source_group in ("online_success", "online_failure"):
            for path, episode_index, returns, first_success_returns in episodes[
                source_group
            ]:
                flags = episode_exploration[path]
                if flags is None:
                    raise ValueError(
                        "replay lacks structured_explore required by "
                        "--samples-per-exploration-group"
                    )
                for target_group, target_flag in (
                    ("online_explored", True),
                    ("online_unexplored", False),
                ):
                    indices = np.flatnonzero(flags == target_flag)
                    if indices.size:
                        strata[target_group].append(
                            (
                                path,
                                episode_index,
                                returns,
                                first_success_returns,
                                indices,
                            )
                        )
        exploration_counts = {
            group: int(sum(item[-1].size for item in items))
            for group, items in strata.items()
        }
        missing = [group for group, count in exploration_counts.items() if not count]
        if missing:
            raise ValueError(
                "replay is missing exploration transition strata: "
                + ", ".join(missing)
            )
        for group in EXPLORATION_GROUPS:
            items = list(strata[group])
            rng.shuffle(items)
            rounds = int(
                math.ceil(samples_per_exploration_group / len(items))
            )
            made = 0
            for round_index in range(rounds):
                fraction = (round_index + 0.5) / rounds
                for (
                    path,
                    episode_index,
                    returns,
                    first_success_returns,
                    indices,
                ) in items:
                    if made >= samples_per_exploration_group:
                        break
                    index = int(
                        indices[
                            min(
                                indices.size - 1,
                                int(round(fraction * (indices.size - 1))),
                            )
                        ]
                    )
                    samples.append(
                        Sample(
                            episode_path=path,
                            episode_index=episode_index,
                            transition_index=index,
                            episode_length=int(returns.shape[0]),
                            group=group,
                            discounted_return=float(returns[index]),
                            first_success_return=float(
                                first_success_returns[index]
                            ),
                            future_success=bool(first_success_returns[index] > 0.0),
                        )
                    )
                    made += 1
    return samples, counts, exploration_counts


def _load_batch(agent, samples: list[Sample]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    frame_stack = int(next(iter(agent.observation_space.values())).shape[0])
    action_sequence = int(agent.action_sequence)
    observation_keys = tuple(agent.observation_space.keys())
    observations: dict[str, list[np.ndarray]] = {
        key: [] for key in observation_keys
    }
    actions = []
    cache: dict[Path, dict[str, np.ndarray]] = {}
    try:
        for sample in samples:
            if sample.episode_path not in cache:
                with np.load(sample.episode_path) as episode:
                    cache[sample.episode_path] = {
                        key: np.asarray(episode[key])
                        for key in (*observation_keys, "action")
                        if key in episode.files
                    }
            episode = cache[sample.episode_path]
            missing = [key for key in observation_keys if key not in episode]
            if missing:
                raise KeyError(
                    f"{sample.episode_path} lacks observations: {missing}"
                )
            index = sample.transition_index
            obs_indices = np.clip(
                np.arange(index - frame_stack + 1, index + 1),
                0,
                sample.episode_length,
            )
            for key in observation_keys:
                observations[key].append(episode[key][obs_indices])
            action_indices = np.clip(
                np.arange(index, index + action_sequence),
                0,
                sample.episode_length - 1,
            )
            actions.append(episode["action"][action_indices])
    finally:
        cache.clear()
    return (
        {key: np.stack(values) for key, values in observations.items()},
        np.stack(actions).astype(np.float32),
    )


def _checkpoint_q_batch(agent, observations, actions, *, use_target, seed):
    import jax
    import jax.numpy as jnp

    from robobase.method.cqn import encode_action

    obs_inputs = agent._prepare_rl_obs_inputs(observations)
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
        if use_target
        else (
            {
                "critic": agent.params["critic"],
                "advantage": agent.params["advantage"],
            }
            if getattr(agent, "hybrid_flow_v_direct_a", False)
            else agent.params["critic"]
        )
    )
    key = jax.random.PRNGKey(int(seed))
    method_name = agent.__class__.__name__.lower()
    if "flow" in method_name:
        chosen_q, all_q = agent._q_values_per_level(
            critic_params, features, actions, key
        )
        _, critic_bins = agent._greedy_action(
            critic_params, features, key=jax.random.fold_in(key, 991)
        )
    else:
        chosen_logits, all_logits = agent._critic_logits_per_level(
            critic_params, features, actions
        )
        chosen_q = jnp.sum(
            jax.nn.softmax(chosen_logits, axis=-1) * agent.support, axis=-1
        )
        all_q = jnp.sum(
            jax.nn.softmax(all_logits, axis=-1) * agent.support, axis=-1
        )
        try:
            _, critic_bins = agent._greedy_action(
                critic_params, features, key=None
            )
        except TypeError:
            _, critic_bins = agent._greedy_action(critic_params, features)

    if getattr(agent, "separate_bc_policy", False):
        _, behavior_bins = agent._policy_action(
            agent.params["policy"], policy_features, key=None
        )
    else:
        behavior_bins = critic_bins

    flat_actions = jnp.asarray(actions, dtype=jnp.float32).reshape(
        (actions.shape[0], -1)
    )
    expert_bins = encode_action(
        flat_actions,
        agent.action_low,
        agent.action_high,
        agent.levels,
        agent.bins,
    ).reshape(
        (
            actions.shape[0],
            agent.levels,
            agent.action_sequence,
            agent.action_dim,
        )
    )
    ready = jax.block_until_ready(
        (chosen_q, all_q, expert_bins, critic_bins, behavior_bins)
    )
    return tuple(np.asarray(value) for value in ready)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < order.size:
        end = start + 1
        while end < order.size and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return result


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 3:
        return float("nan")
    return _pearson(_ranks(x), _ranks(y))


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def summarize(records: list[dict[str, Any]], bins: int) -> dict[str, Any]:
    def one_group(items: list[dict[str, Any]]) -> dict[str, Any]:
        predicted = np.asarray([item["predicted_q"] for item in items])
        raw_return = np.asarray([item["discounted_return"] for item in items])
        success_return = np.asarray(
            [item["first_success_return"] for item in items]
        )
        return {
            "num_samples": len(items),
            "imitation": {
                "replay_bin_top1_rate": _safe_mean(
                    np.asarray([item["replay_bin_top1_rate"] for item in items])
                ),
                "replay_bin_top1_rate_current_action": _safe_mean(
                    np.asarray(
                        [
                            item["replay_bin_top1_rate_current_action"]
                            for item in items
                        ]
                    )
                ),
                "greedy_bin_agreement": _safe_mean(
                    np.asarray([item["greedy_bin_agreement"] for item in items])
                ),
                "greedy_bin_agreement_current_action": _safe_mean(
                    np.asarray(
                        [
                            item["greedy_bin_agreement_current_action"]
                            for item in items
                        ]
                    )
                ),
                "behavior_bin_agreement": _safe_mean(
                    np.asarray(
                        [item["behavior_bin_agreement"] for item in items]
                    )
                ),
                "behavior_bin_agreement_current_action": _safe_mean(
                    np.asarray(
                        [
                            item["behavior_bin_agreement_current_action"]
                            for item in items
                        ]
                    )
                ),
                "critic_behavior_disagreement": _safe_mean(
                    np.asarray(
                        [item["critic_behavior_disagreement"] for item in items]
                    )
                ),
                "critic_behavior_disagreement_current_action": _safe_mean(
                    np.asarray(
                        [
                            item["critic_behavior_disagreement_current_action"]
                            for item in items
                        ]
                    )
                ),
                "normalized_replay_bin_rank": _safe_mean(
                    np.asarray(
                        [item["normalized_replay_bin_rank"] for item in items]
                    )
                ),
            },
            "value": {
                "predicted_q_mean": _safe_mean(predicted),
                "discounted_return_mean": _safe_mean(raw_return),
                "first_success_return_mean": _safe_mean(success_return),
                "q_raw_return_pearson": _pearson(predicted, raw_return),
                "q_raw_return_spearman": _spearman(predicted, raw_return),
                "q_first_success_return_pearson": _pearson(
                    predicted, success_return
                ),
                "q_first_success_return_spearman": _spearman(
                    predicted, success_return
                ),
            },
            "collapse": {
                "candidate_q_span": _safe_mean(
                    np.asarray([item["candidate_q_span"] for item in items])
                ),
                "candidate_top2_gap": _safe_mean(
                    np.asarray([item["candidate_top2_gap"] for item in items])
                ),
                "max_minus_replay_q": _safe_mean(
                    np.asarray([item["max_minus_replay_q"] for item in items])
                ),
            },
        }

    result = {
        group: one_group([r for r in records if r["group"] == group])
        for group in GROUPS
    }
    for group in EXPLORATION_GROUPS:
        items = [record for record in records if record["group"] == group]
        if items:
            result[group] = one_group(items)
    result["all"] = one_group(
        [record for record in records if record["group"] in GROUPS]
    )
    result["interpretation"] = {
        "high_imitation_low_value": (
            "High replay-bin agreement with weak return correlation is evidence "
            "for an imitation shortcut, not calibrated value learning."
        ),
        "observational_limit": (
            "Return correlation is observational and can be confounded by state/time; "
            "the simulator branch-counterfactual test is the causal follow-up."
        ),
        "normalized_rank_range": [0.0, 1.0],
        "random_rank_reference": 0.5,
        "random_top1_reference": 1.0 / bins,
    }
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    from omegaconf import OmegaConf

    from robobase.workspace import Workspace

    run_dir = args.run_dir.expanduser().resolve()
    snapshot = (
        args.snapshot
        if args.snapshot is not None
        else run_dir / "snapshots" / "latest_snapshot.pkl"
    ).expanduser().resolve()
    data_run_dir = (
        args.data_run_dir if args.data_run_dir is not None else run_dir
    ).expanduser().resolve()
    replay_dir = data_run_dir / "replay"
    cfg_path = run_dir / ".hydra" / "config.yaml"
    for path in (cfg_path, snapshot, replay_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    cfg = OmegaConf.load(cfg_path)
    gamma = float(cfg.replay.gamma)
    samples, episode_group_counts, exploration_transition_counts = select_samples(
        replay_dir,
        gamma=gamma,
        samples_per_group=int(args.samples_per_group),
        samples_per_exploration_group=int(
            args.samples_per_exploration_group
        ),
        seed=int(args.seed),
        offline_episode_count=args.offline_episode_count,
    )

    OmegaConf.set_struct(cfg, False)
    cfg.create_train_env = False
    cfg.num_train_envs = 0
    cfg.num_train_frames = 0
    cfg.num_eval_envs = 0
    cfg.num_eval_episodes = 0
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

    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="cqn-value-fidelity-") as work_dir:
        workspace = Workspace(cfg, work_dir=work_dir)
        try:
            workspace.load_snapshot(snapshot, load_replay_buffer=False)
            if args.critic == "config":
                use_target = bool(
                    cfg.method.get("use_target_network_for_rollout", True)
                )
            else:
                use_target = args.critic == "target"

            batch_size = int(args.batch_size)
            if batch_size < 1:
                raise ValueError("--batch-size must be at least 1")
            for start in range(0, len(samples), batch_size):
                batch_samples = samples[start : start + batch_size]
                valid_count = len(batch_samples)
                if valid_count < batch_size:
                    batch_samples = batch_samples + [batch_samples[-1]] * (
                        batch_size - valid_count
                    )
                observations, actions = _load_batch(workspace.agent, batch_samples)
                (
                    chosen_q,
                    all_q,
                    replay_bins,
                    critic_bins,
                    behavior_bins,
                ) = _checkpoint_q_batch(
                    workspace.agent,
                    observations,
                    actions,
                    use_target=use_target,
                    seed=int(args.seed) + start,
                )
                chosen_q = chosen_q[:valid_count]
                all_q = all_q[:valid_count]
                replay_bins = replay_bins[:valid_count]
                critic_bins = critic_bins[:valid_count]
                behavior_bins = behavior_bins[:valid_count]
                effective_k0 = (
                    str(
                        cfg.method.get("critic_sequence_mode", "full")
                    ).lower()
                    == "effective_k0"
                )
                if effective_k0:
                    chosen_q = chosen_q[:, :, : workspace.agent.action_dim]
                    all_q = all_q[:, :, : workspace.agent.action_dim]
                maxima = np.max(all_q, axis=-1)
                sorted_q = np.sort(all_q, axis=-1)
                ranks = np.sum(
                    all_q > chosen_q[..., None] + 1e-6, axis=-1
                )
                top1 = chosen_q >= maxima - 1e-6
                if effective_k0:
                    top1_current_action = top1
                else:
                    top1_current_action = top1.reshape(
                        (
                            valid_count,
                            workspace.agent.levels,
                            workspace.agent.action_sequence,
                            workspace.agent.action_dim,
                        )
                    )[:, :, 0]
                critic_agreement = replay_bins == critic_bins
                behavior_agreement = replay_bins == behavior_bins
                critic_behavior_disagreement = critic_bins != behavior_bins
                for offset, sample in enumerate(batch_samples[:valid_count]):
                    # The final zoom level is the highest-resolution estimate.
                    predicted_q = float(np.mean(chosen_q[offset, -1]))
                    records.append(
                        {
                            "episode": sample.episode_path.name,
                            "episode_index": sample.episode_index,
                            "transition_index": sample.transition_index,
                            "group": sample.group,
                            "future_success": sample.future_success,
                            "discounted_return": sample.discounted_return,
                            "first_success_return": sample.first_success_return,
                            "predicted_q": predicted_q,
                            "replay_bin_top1_rate": float(np.mean(top1[offset])),
                            "replay_bin_top1_rate_current_action": float(
                                np.mean(top1_current_action[offset])
                            ),
                            "greedy_bin_agreement": float(
                                np.mean(critic_agreement[offset])
                            ),
                            "greedy_bin_agreement_current_action": float(
                                np.mean(critic_agreement[offset, :, 0])
                            ),
                            "behavior_bin_agreement": float(
                                np.mean(behavior_agreement[offset])
                            ),
                            "behavior_bin_agreement_current_action": float(
                                np.mean(behavior_agreement[offset, :, 0])
                            ),
                            "critic_behavior_disagreement": float(
                                np.mean(critic_behavior_disagreement[offset])
                            ),
                            "critic_behavior_disagreement_current_action": float(
                                np.mean(
                                    critic_behavior_disagreement[offset, :, 0]
                                )
                            ),
                            "normalized_replay_bin_rank": float(
                                np.mean(ranks[offset]) / max(1, workspace.agent.bins - 1)
                            ),
                            "candidate_q_span": float(
                                np.mean(np.ptp(all_q[offset], axis=-1))
                            ),
                            "candidate_top2_gap": float(
                                np.mean(sorted_q[offset, ..., -1] - sorted_q[offset, ..., -2])
                            ),
                            "max_minus_replay_q": float(
                                np.mean(maxima[offset] - chosen_q[offset])
                            ),
                        }
                    )
        finally:
            workspace.shutdown()

    return {
        "status": "ok",
        "run_dir": str(run_dir),
        "snapshot": str(snapshot),
        "data_run_dir": str(data_run_dir),
        "method": str(cfg.method.name),
        "critic": "target" if use_target else "online",
        "gamma": gamma,
        "samples_per_group": int(args.samples_per_group),
        "offline_episode_count": args.offline_episode_count,
        "episode_group_counts": episode_group_counts,
        "exploration_transition_counts": exploration_transition_counts,
        "summary": summarize(records, int(cfg.method.bins)),
        "samples": records,
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
