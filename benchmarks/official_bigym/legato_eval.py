"""Closed-loop BiGym evaluator for the pinned official Legato policy core."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.official_bigym.legato_adapter import OfficialBigymPolicy
from benchmarks.official_bigym.legato_adapter import OfficialPolicyConfig
from benchmarks.official_bigym.legato_checkpoint import (
    load_checkpoint,
    read_checkpoint_metadata,
)
from benchmarks.official_bigym.legato_features import FrozenFMVisualFeatures
from benchmarks.official_bigym.legato_upstream import UPSTREAM_COMMIT


FeatureFn = Callable[[Any], np.ndarray | jax.Array]
ActionFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class EpisodeResult:
    episode_return: float
    episode_length: int
    success: bool
    boundary_jump: float
    first_difference: float
    second_difference: float
    jerk: float


class EpisodeAudit:
    """Accumulate auditable per-episode outcomes from chunked env steps."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self._episode_return = 0.0
        self._executed_action_steps = 0

    def record_step(
        self,
        *,
        seed: int,
        reward: float,
        executed_action_steps: int,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
    ) -> None:
        if executed_action_steps <= 0:
            raise ValueError("Each environment call must execute at least one action.")
        self._episode_return += float(reward)
        self._executed_action_steps += int(executed_action_steps)
        if not (terminated or truncated):
            return
        success_value = info.get("task_success")
        success = (
            None
            if success_value is None
            else bool(np.asarray(success_value).astype(int).item())
        )
        self.records.append(
            {
                "episode_index": len(self.records),
                "seed": int(seed),
                "success": success,
                "episode_return": float(self._episode_return),
                "executed_action_steps": int(self._executed_action_steps),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        self._episode_return = 0.0
        self._executed_action_steps = 0


def _validate_episode_audit(
    records: list[dict[str, Any]],
    metrics: Mapping[str, Any],
    *,
    num_episodes: int,
    seed_start: int,
    action_repeat: int,
) -> None:
    if len(records) != num_episodes:
        raise RuntimeError(
            f"Expected {num_episodes} episode records, found {len(records)}."
        )
    expected_seeds = list(range(seed_start, seed_start + num_episodes))
    actual_seeds = [int(record["seed"]) for record in records]
    if actual_seeds != expected_seeds:
        raise RuntimeError(
            f"Episode seeds are not consecutive and unique: {actual_seeds}."
        )

    expected_reward = float(np.mean([record["episode_return"] for record in records]))
    expected_length = float(
        np.mean([record["executed_action_steps"] for record in records])
        * action_repeat
    )
    if not np.isclose(expected_reward, float(metrics["episode_reward"]), atol=1e-6):
        raise RuntimeError("Per-episode rewards do not match Workspace aggregation.")
    if not np.isclose(expected_length, float(metrics["episode_length"]), atol=1e-6):
        raise RuntimeError("Per-episode lengths do not match Workspace aggregation.")
    successes = [record["success"] for record in records]
    if all(success is not None for success in successes):
        expected_success = float(np.mean(successes))
        if not np.isclose(
            expected_success, float(metrics["episode_success"]), atol=1e-12
        ):
            raise RuntimeError("Per-episode successes do not match Workspace aggregation.")


class WorkspaceOfficialPolicyAgent:
    """Eval-only agent boundary for ``Workspace.eval`` on real BiGym envs.

    ``feature_boundary.agent`` is the restored FM baseline and supplies its
    frozen visual encoder. The official policy supplies only the action core.
    Workspace keeps the restored FM action horizon while its execution length
    matches the adapter's execute horizon.
    """

    logging = False

    def __init__(
        self,
        adapter: OfficialBigymPolicy,
        feature_boundary: FrozenFMVisualFeatures,
        *,
        output_horizon: int | None = None,
    ):
        self.adapter = adapter
        self.feature_boundary = feature_boundary
        self.output_horizon = (
            adapter.config.action_horizon
            if output_horizon is None
            else int(output_horizon)
        )
        if self.output_horizon < adapter.config.execute_horizon:
            raise ValueError("output_horizon must cover every executed action.")
        self._state = None
        self._needs_bootstrap: set[int] = set()
        self._rng = jax.random.key(0)
        self._eval_rng_base_key = jax.random.key(0)
        self._active_eval_seeds: list[int] | None = None
        self._eval_call_by_stage: dict[tuple[int, str], int] = {}
        self._action_segments: dict[int, list[np.ndarray]] = {}
        self._completed_segments: list[list[np.ndarray]] = []
        self._pending_actions: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._clipped_action_values = 0
        self._total_action_values = 0

    def act(self, observations: dict, step: int, eval_mode: bool):
        del step
        if not eval_mode:
            raise RuntimeError("Official benchmark boundary is eval-only.")
        features = self.feature_boundary.encode(observations)
        batch_size = features.shape[0]
        if self._state is None or self._state.previous_chunk.shape[0] != batch_size:
            self._needs_bootstrap = set(range(batch_size))
        if self._needs_bootstrap:
            key = self._next_policy_key("bootstrap")
            bootstrap = self.adapter.bootstrap(key, features)
            if self._state is None or self._state.previous_chunk.shape[0] != batch_size:
                self._state = bootstrap
            else:
                mask = jnp.asarray(
                    [index in self._needs_bootstrap for index in range(batch_size)]
                )
                self._state = type(self._state)(
                    jnp.where(
                        mask[:, None, None],
                        bootstrap.previous_chunk,
                        self._state.previous_chunk,
                    ),
                    jnp.where(mask, True, self._state.valid),
                )
            self._needs_bootstrap.clear()
        key = self._next_policy_key("predict")
        prediction = self.adapter.predict(key, features, self._state)
        self._state = prediction.next_state
        raw_execute = prediction.execute_actions
        execute = jnp.clip(raw_execute, -1.0, 1.0)
        raw_host_execute = np.asarray(
            jax.device_get(raw_execute), dtype=np.float32
        )
        host_execute = np.asarray(jax.device_get(execute), dtype=np.float32)
        if self._pending_actions:
            raise RuntimeError(
                "Environment execution feedback must be recorded before the next act()."
            )
        self._pending_actions = {
            index: (chunk, raw_host_execute[index])
            for index, chunk in enumerate(host_execute)
        }
        pad_length = self.output_horizon - execute.shape[1]
        padding = jnp.zeros(
            (batch_size, pad_length, self.adapter.action_dim), dtype=execute.dtype
        )
        return np.asarray(
            jax.device_get(jnp.concatenate([execute, padding], axis=1)),
            dtype=np.float32,
        )

    def reset(self, step: int, agents_to_reset: list[int]):
        del step
        for index in (int(value) for value in agents_to_reset):
            self._pending_actions.pop(index, None)
            previous = self._action_segments.pop(index, None)
            if previous:
                self._completed_segments.append(previous)
            self._needs_bootstrap.add(index)

    def set_eval_env_running(self, value: bool):
        if value:
            self._state = None
            self._needs_bootstrap.clear()
            self._action_segments.clear()
            self._completed_segments.clear()
            self._pending_actions.clear()
            self._clipped_action_values = 0
            self._total_action_values = 0

    def record_action_execution(self, info: Mapping[str, Any]) -> None:
        """Commit only the action prefix that the ActionSequence wrapper executed."""
        if len(self._pending_actions) != 1 or 0 not in self._pending_actions:
            raise RuntimeError(
                "Execution feedback currently requires one pending eval environment."
            )
        if "action_sequence_mask" not in info:
            raise KeyError("Evaluation info is missing action_sequence_mask.")
        mask = np.asarray(info["action_sequence_mask"], dtype=np.bool_).reshape(-1)
        executed_count = int(mask.sum())
        if not np.array_equal(mask, np.arange(mask.size) < executed_count):
            raise ValueError("action_sequence_mask must describe a contiguous prefix.")
        chunk, raw_chunk = self._pending_actions.pop(0)
        if not 0 < executed_count <= len(chunk):
            raise ValueError(
                "Executed action count must lie within the returned policy chunk."
            )
        executed = chunk[:executed_count]
        raw_executed = raw_chunk[:executed_count]
        self._action_segments.setdefault(0, []).append(executed)
        self._clipped_action_values += int(
            np.logical_or(raw_executed < -1.0, raw_executed > 1.0).sum()
        )
        self._total_action_values += int(raw_executed.size)

    def set_active_eval_seeds(self, seeds_in_batch_order) -> None:
        """Align policy noise to the current single-environment episode seed."""
        if seeds_in_batch_order is None:
            self._active_eval_seeds = None
            return
        seeds = [int(seed) for seed in seeds_in_batch_order]
        if len(seeds) != 1:
            raise ValueError(
                "Official Legato evaluation currently requires num_eval_envs=1 "
                "for per-episode policy-noise alignment."
            )
        self._active_eval_seeds = seeds
        seed = seeds[0]
        self._eval_call_by_stage[(seed, "bootstrap")] = 0
        self._eval_call_by_stage[(seed, "predict")] = 0

    def reset_aligned_eval_noise(self) -> None:
        """Clear episode-key counters before a complete evaluation run."""
        self._eval_call_by_stage.clear()

    def _next_policy_key(self, stage: str) -> jax.Array:
        if self._active_eval_seeds is None:
            self._rng, key = jax.random.split(self._rng)
            return key
        if stage not in {"bootstrap", "predict"}:
            raise ValueError(f"Unknown policy RNG stage: {stage!r}.")
        seed = self._active_eval_seeds[0]
        counter_key = (seed, stage)
        call_index = self._eval_call_by_stage.get(counter_key, 0)
        self._eval_call_by_stage[counter_key] = call_index + 1
        key = jax.random.fold_in(self._eval_rng_base_key, seed & 0xFFFFFFFF)
        key = jax.random.fold_in(key, 0 if stage == "bootstrap" else 1)
        return jax.random.fold_in(key, call_index)

    def rollout_diagnostics(self) -> dict[str, float]:
        segments = list(self._completed_segments) + [
            chunks for chunks in self._action_segments.values() if chunks
        ]
        boundary_jumps = []
        continuation_jumps = []
        delay = self.adapter.config.inference_delay
        differences: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: []}
        for chunks in segments:
            for left, right in zip(chunks[:-1], chunks[1:]):
                boundary_jumps.append(np.linalg.norm(right[0] - left[-1]))
            if 0 < delay < self.adapter.config.execute_horizon:
                continuation_jumps.extend(
                    np.linalg.norm(chunk[delay] - chunk[delay - 1])
                    for chunk in chunks
                    if len(chunk) > delay
                )
            actions = np.concatenate(chunks, axis=0)
            for order in differences:
                if len(actions) > order:
                    differences[order].append(
                        np.linalg.norm(np.diff(actions, n=order, axis=0), axis=-1)
                    )

        def mean(values: list[np.ndarray | np.floating]) -> float:
            if not values:
                return float("nan")
            flattened = [np.asarray(value).reshape(-1) for value in values]
            return float(np.concatenate(flattened).mean())

        return {
            "normalized_action_boundary_jump": mean(boundary_jumps),
            "normalized_action_continuation_jump": mean(continuation_jumps),
            "normalized_action_first_difference": mean(differences[1]),
            "normalized_action_second_difference": mean(differences[2]),
            "normalized_action_jerk": mean(differences[3]),
            "policy_action_clip_fraction": (
                self._clipped_action_values / self._total_action_values
                if self._total_action_values
                else 0.0
            ),
        }


def _reset_env(env, seed: int):
    result = env.reset(seed=seed)
    return result[0] if isinstance(result, tuple) else result


def _step_env(env, action: np.ndarray):
    result = env.step(action)
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return observation, float(reward), bool(terminated or truncated), info
    if len(result) == 4:
        observation, reward, done, info = result
        return observation, float(reward), bool(done), info
    raise ValueError("Environment step must return a Gym four- or five-tuple.")


def _features(feature_fn: FeatureFn, observation: Any, obs_dim: int) -> jax.Array:
    features = jnp.asarray(feature_fn(observation), dtype=jnp.float32)
    if features.ndim == 1:
        features = features[None]
    if features.shape != (1, obs_dim):
        raise ValueError(f"feature_fn must return [{obs_dim}] or [1,{obs_dim}].")
    return features


def _success_from_info(info: Mapping[str, Any], episode_return: float) -> bool:
    for key in ("success", "is_success", "task_success"):
        if key in info:
            return bool(np.asarray(info[key]).any())
    return episode_return > 0.25


def _smoothness(actions: np.ndarray, boundaries: list[int]) -> tuple[float, ...]:
    if not len(actions):
        return (float("nan"),) * 4
    boundary_values = []
    for index in boundaries:
        if 0 < index < len(actions):
            boundary_values.append(np.linalg.norm(actions[index] - actions[index - 1]))

    def difference(order: int) -> float:
        if len(actions) <= order:
            return float("nan")
        return float(np.linalg.norm(np.diff(actions, n=order, axis=0), axis=-1).mean())

    boundary = float(np.mean(boundary_values)) if boundary_values else float("nan")
    return boundary, difference(1), difference(2), difference(3)


def evaluate_episode(
    env,
    adapter: OfficialBigymPolicy,
    feature_fn: FeatureFn,
    *,
    seed: int,
    max_steps: int,
    action_fn: ActionFn | None = None,
    clip_normalized_actions: bool = True,
) -> EpisodeResult:
    """Evaluate official asynchronous chunk semantics in a synchronous Gym env.

    The delay is simulated exactly as in the official Kinetix evaluator: each
    control cycle executes ``d`` actions from the previous chunk and the rest
    from the new chunk. This function does not claim wall-clock overlap between
    inference and environment stepping.
    """
    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    observation = _reset_env(env, seed)
    rng = jax.random.key(seed)
    rng, key = jax.random.split(rng)
    state = adapter.bootstrap(key, _features(feature_fn, observation, adapter.obs_dim))

    episode_return = 0.0
    episode_length = 0
    final_info: Mapping[str, Any] = {}
    executed_actions = []
    boundaries = []
    done = False
    while not done and episode_length < max_steps:
        rng, key = jax.random.split(rng)
        prediction = adapter.predict(
            key,
            _features(feature_fn, observation, adapter.obs_dim),
            state,
        )
        state = prediction.next_state
        action_chunk = np.asarray(
            jax.device_get(prediction.execute_actions[0]), dtype=np.float32
        )
        if clip_normalized_actions:
            action_chunk = np.clip(action_chunk, -1.0, 1.0)
        boundaries.append(len(executed_actions))
        for action in action_chunk:
            env_action = action if action_fn is None else action_fn(action)
            observation, reward, done, final_info = _step_env(env, env_action)
            executed_actions.append(action)
            episode_return += reward
            episode_length += 1
            if done or episode_length >= max_steps:
                break

    actions = np.asarray(executed_actions, dtype=np.float32)
    boundary, first, second, jerk = _smoothness(actions, boundaries)
    return EpisodeResult(
        episode_return=episode_return,
        episode_length=episode_length,
        success=_success_from_info(final_info, episode_return),
        boundary_jump=boundary,
        first_difference=first,
        second_difference=second,
        jerk=jerk,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an official vanilla/RTC/Legato core on FlipCutlery."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--fm-snapshot", required=True, type=Path)
    parser.add_argument("--policy-checkpoint", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("vanilla", "rtc", "legato"))
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-eval-episodes", type=int, default=50)
    parser.add_argument("--num-eval-envs", type=int, default=1)
    parser.add_argument("--execute-horizon", type=int, default=4)
    parser.add_argument("--inference-delay", type=int, default=0)
    parser.add_argument("--num-flow-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--episode-limit-steps",
        type=int,
        help=(
            "Optional environment-step cap. Omit it to inherit the saved FM "
            "run's evaluation protocol."
        ),
    )
    parser.add_argument("--pixel-dataset-root", type=Path)
    parser.add_argument("--state-dataset-root", type=Path)
    parser.add_argument("--lang-feature-path", type=Path)
    return parser.parse_args()


def _set_if_present(cfg, dotted_key: str, value) -> None:
    node = cfg
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in node or node[part] is None:
            return
        node = node[part]
    node[parts[-1]] = value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _probe_features(workspace, boundary: FrozenFMVisualFeatures) -> jax.Array:
    replay = workspace.replay_buffer
    if hasattr(replay, "sample_batch_indices") and len(replay) > 0:
        batch = replay.sample_batch_indices([0])
        keys = getattr(replay, "observation_elements", {})
        observations = {key: batch[key] for key in keys}
        return boundary.encode(observations)

    workspace._ensure_eval_envs_created()
    env = workspace.eval_env
    if env is None:
        raise ValueError("Feature probing currently requires one evaluation environment.")
    observation, _ = env.reset(seed=int(workspace.cfg.seed))
    batched = {key: np.expand_dims(value, 0) for key, value in observation.items()}
    return boundary.encode(batched)


def _run_cli(args: argparse.Namespace) -> dict[str, Any]:
    from omegaconf import OmegaConf

    from robobase.workspace import Workspace

    class ExecutionFeedbackWorkspace(Workspace):
        """Benchmark-only Workspace that reports the exact executed chunk prefix."""

        def __init__(self, *workspace_args, **workspace_kwargs):
            super().__init__(*workspace_args, **workspace_kwargs)
            self.episode_audit = EpisodeAudit()

        def _perform_env_steps(self, observations, env, eval_mode):
            result = super()._perform_env_steps(observations, env, eval_mode)
            if eval_mode:
                _, reward, termination, truncation, info = result[1]
                self.agent.record_action_execution(info)
                is_vector = bool(getattr(env, "is_vector_env", False))
                if is_vector:
                    terminated = bool(np.asarray(termination).reshape(-1)[0])
                    truncated = bool(np.asarray(truncation).reshape(-1)[0])
                    executed_steps = int(self._executed_vector_action_steps(info)[0])
                    episode_info = self._extract_vector_env_info(
                        info,
                        0,
                        prefer_final=terminated or truncated,
                    )
                    step_reward = float(np.asarray(reward).reshape(-1)[0])
                else:
                    terminated = bool(termination)
                    truncated = bool(truncation)
                    mask = np.asarray(info["action_sequence_mask"], dtype=np.bool_)
                    executed_steps = int(mask.sum())
                    episode_info = info
                    step_reward = float(reward)
                episode_index = len(self.episode_audit.records)
                episode_seed = self._eval_seed_for_episode(episode_index)
                if episode_seed is None:
                    raise RuntimeError("Formal evaluation requires explicit episode seeds.")
                self.episode_audit.record_step(
                    seed=episode_seed,
                    reward=step_reward,
                    executed_action_steps=executed_steps,
                    terminated=terminated,
                    truncated=truncated,
                    info=episode_info,
                )
            return result

    cfg_path = args.run_dir / ".hydra" / "config.yaml"
    for path in (cfg_path, args.fm_snapshot, args.policy_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_struct(cfg, False)
    if str(cfg.env.task_name) != "flip_cutlery":
        raise ValueError(
            f"This benchmark entry point is locked to flip_cutlery, got {cfg.env.task_name!r}."
        )
    if args.num_eval_episodes <= 0 or args.num_eval_envs <= 0:
        raise ValueError("Evaluation episode and environment counts must be positive.")
    if args.episode_limit_steps is not None and args.episode_limit_steps <= 0:
        raise ValueError("--episode-limit-steps must be positive when provided.")
    if args.num_eval_envs != 1:
        raise ValueError(
            "Official Legato evaluation requires --num-eval-envs 1 so policy "
            "noise can be paired exactly by episode seed."
        )

    cfg.create_train_env = False
    cfg.num_train_envs = 0
    cfg.num_train_frames = 0
    cfg.num_eval_episodes = int(args.num_eval_episodes)
    cfg.num_eval_envs = int(args.num_eval_envs)
    cfg.execution_length = int(args.execute_horizon)
    cfg.seed = int(args.seed)
    cfg.env.eval_seed_start = int(args.seed)
    if "eval_seeds" in cfg.env:
        cfg.env.eval_seeds = None
    if args.episode_limit_steps is not None:
        cfg.env.episode_length = int(args.episode_limit_steps)
        cfg.env.episode_length_is_env_steps = True
    cfg.log_train_video = False
    cfg.log_eval_video = False
    cfg.save_snapshot = False
    cfg.save_csv = False
    cfg.gpu_id = None
    if args.pixel_dataset_root is not None:
        cfg.env.pixel_dataset_root = str(args.pixel_dataset_root.expanduser().resolve())
    if args.state_dataset_root is not None:
        cfg.env.state_dataset_root = str(args.state_dataset_root.expanduser().resolve())
    if args.lang_feature_path is not None:
        cfg.method.lang_feature_source = "precomputed"
        cfg.method.lang_feature_path = str(args.lang_feature_path.expanduser().resolve())
    elif str(cfg.method.get("lang_feature_source", "tokens")) not in {
        "tokens",
        "jax",
        "hash",
        "precomputed",
    }:
        cfg.method.lang_feature_source = "tokens"
    _set_if_present(cfg, "wandb.use", False)
    _set_if_present(cfg, "tb.use", False)
    _set_if_present(cfg, "replay.num_workers", 0)
    _set_if_present(cfg, "lazy_replay.num_workers", 0)
    _set_if_present(cfg, "lazy_replay.persistent_workers", False)
    _set_if_present(cfg, "backend.replay_prefetch_size", 0)
    _set_if_present(cfg, "backend.replay_device_prefetch", False)
    OmegaConf.resolve(cfg)

    metadata = read_checkpoint_metadata(args.policy_checkpoint)
    policy_cfg = OfficialPolicyConfig(**metadata["config"])
    policy_cfg = replace(
        policy_cfg,
        execute_horizon=int(args.execute_horizon),
        inference_delay=int(args.inference_delay),
        num_flow_steps=int(args.num_flow_steps),
    )

    args.work_dir.mkdir(parents=True, exist_ok=True)
    workspace = ExecutionFeedbackWorkspace(cfg, work_dir=str(args.work_dir))
    try:
        workspace.load_snapshot(args.fm_snapshot, load_replay_buffer=False)
        fm_agent = workspace.agent
        boundary = FrozenFMVisualFeatures(fm_agent)
        probe = _probe_features(workspace, boundary)
        action_dim = int(fm_agent.action_dim)
        adapter = OfficialBigymPolicy(
            mode=args.mode,
            obs_dim=int(probe.shape[-1]),
            action_dim=action_dim,
            config=policy_cfg,
            seed=int(args.seed),
        )
        checkpoint = load_checkpoint(
            args.policy_checkpoint,
            adapter,
            strict=False,
        )
        workspace.agent = WorkspaceOfficialPolicyAgent(
            adapter,
            boundary,
            output_horizon=int(cfg.action_sequence),
        )
        metrics = workspace.eval()
        episode_records = workspace.episode_audit.records
        _validate_episode_audit(
            episode_records,
            metrics,
            num_episodes=int(cfg.num_eval_episodes),
            seed_start=int(cfg.env.eval_seed_start),
            action_repeat=int(cfg.action_repeat),
        )
    finally:
        workspace.shutdown()

    return {
        "status": "ok",
        "task": "flip_cutlery",
        "mode": args.mode,
        "run_dir": str(args.run_dir.resolve()),
        "fm_snapshot": str(args.fm_snapshot.resolve()),
        "fm_snapshot_sha256": _sha256(args.fm_snapshot),
        "policy_checkpoint": str(args.policy_checkpoint.resolve()),
        "policy_checkpoint_sha256": _sha256(args.policy_checkpoint),
        "policy_checkpoint_step": int(checkpoint["step"]),
        "policy_checkpoint_training": checkpoint.get("extra", {}),
        "official_upstream_commit": UPSTREAM_COMMIT,
        "feature_dim": int(probe.shape[-1]),
        "action_dim": action_dim,
        "workspace_action_horizon": int(cfg.action_sequence),
        "action_execution_start": int(cfg.get("action_execution_start", 0)),
        "policy_config": asdict(policy_cfg),
        "lang_feature_source": str(cfg.method.get("lang_feature_source", "tokens")),
        "lang_feature_path": cfg.method.get("lang_feature_path", None),
        "num_eval_episodes": int(cfg.num_eval_episodes),
        "num_eval_envs": int(cfg.num_eval_envs),
        "policy_rng_mode": "episode_seed_stage_call_index",
        "eval_seed_start": cfg.env.get("eval_seed_start", None),
        "episode_limit_steps": int(cfg.env.episode_length),
        "episode_limit_source": (
            "cli_override"
            if args.episode_limit_steps is not None
            else "saved_config"
        ),
        "episode_length_is_env_steps": bool(
            cfg.env.get("episode_length_is_env_steps", False)
        ),
        "control_frequency_hz": 20,
        "observation_cameras": list(cfg.env.cameras),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "metrics": {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float, np.integer, np.floating))
        },
        "episodes": episode_records,
    }


def main() -> int:
    args = _parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        payload = _run_cli(args)
        payload["elapsed_seconds"] = time.time() - started
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return 0
    except BaseException as exc:
        payload = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        raise


__all__ = [
    "EpisodeAudit",
    "EpisodeResult",
    "WorkspaceOfficialPolicyAgent",
    "_validate_episode_audit",
    "evaluate_episode",
]


if __name__ == "__main__":
    raise SystemExit(main())
