#!/usr/bin/env python3
"""Diagnose whether a CQN-Flow checkpoint uses iterative value transport.

The script compares several Euler depths under common random numbers and
measures curvature/source contraction along the configured trajectory.  It is
strictly a read-only mechanistic diagnostic: its output must not participate
in checkpoint, hyperparameter, or policy-value-mixing selection.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


def _parse_step_counts(value: str) -> tuple[int, ...]:
    try:
        counts = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--step-counts must be comma-separated integers"
        ) from exc
    if not counts or any(count < 1 for count in counts):
        raise argparse.ArgumentTypeError(
            "--step-counts must contain positive integers"
        )
    if len(set(counts)) != len(counts):
        raise argparse.ArgumentTypeError(
            "--step-counts must not contain duplicates"
        )
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--num-observations", type=int, default=16)
    parser.add_argument("--num-source-samples", type=int, default=8)
    parser.add_argument(
        "--step-counts",
        type=_parse_step_counts,
        default=(1, 2, 4, 8),
        help="Comma-separated Euler depths compared with common sources.",
    )
    parser.add_argument("--eval-seed-start", type=int)
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument(
        "--include-condition-actions",
        action="store_true",
        help="Persist the BC actions used only to define C2F conditions.",
    )
    parser.add_argument(
        "--critic",
        choices=("config", "online", "target"),
        default="config",
        help="Select online/EMA critic; 'config' matches rollout settings.",
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


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    run_dir = args.run_dir.expanduser().resolve()
    snapshot = args.snapshot or run_dir / "snapshots" / "latest_snapshot.pkl"
    return (
        run_dir,
        snapshot.expanduser().resolve(),
        args.output.expanduser().resolve(),
    )


def _eval_seeds(cfg, *, count: int, seed_start: int | None) -> list[int]:
    if count < 1:
        raise ValueError("--num-observations must be at least 1")
    if seed_start is not None:
        return list(range(int(seed_start), int(seed_start) + count))

    configured = cfg.env.get("eval_seeds", None)
    if configured is not None:
        configured = [int(seed) for seed in configured]
        if not configured:
            raise ValueError("env.eval_seeds must contain at least one seed")
        return [configured[index % len(configured)] for index in range(count)]

    configured_start = cfg.env.get("eval_seed_start", None)
    start = 20000 if configured_start is None else int(configured_start)
    return list(range(start, start + count))


def _stack_observations(
    observations: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    if not observations:
        raise ValueError("at least one observation is required")
    expected_keys = tuple(observations[0].keys())
    for index, observation in enumerate(observations[1:], start=1):
        if tuple(observation.keys()) != expected_keys:
            raise ValueError(
                f"observation {index} keys do not match the first reset"
            )
    return {
        key: np.stack(
            [np.asarray(observation[key]) for observation in observations]
        )
        for key in expected_keys
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return array.tolist()


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    from omegaconf import OmegaConf

    from robobase.workspace import Workspace

    run_dir, snapshot, _ = resolve_paths(args)
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing config: {cfg_path}")
    if not snapshot.is_file():
        raise FileNotFoundError(f"missing snapshot: {snapshot}")
    if args.num_source_samples < 2:
        raise ValueError("--num-source-samples must be at least 2")

    cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_struct(cfg, False)
    if str(cfg.method.get("name", "")).lower() != "cqn_flow":
        raise ValueError(
            f"checkpoint method is not CQN-Flow: {cfg.method.get('name')}"
        )
    if str(cfg.method.get("value_mode", "")).lower() not in {
        "scalar",
        "return_sample",
    }:
        raise ValueError(
            "flow-utilization probe requires scalar or return-sample value mode"
        )
    if bool(cfg.method.get("hybrid_flow_v_direct_a", False)):
        raise ValueError("flow-V/direct-A hybrids confound this diagnostic")
    if int(cfg.method.get("num_flow_steps", 0)) < 2:
        raise ValueError("configured num_flow_steps must be at least 2")
    if not bool(cfg.method.get("separate_bc_policy", False)):
        raise ValueError(
            "reset-state probing requires an independent BC policy to define "
            "the action conditions"
        )

    seeds = _eval_seeds(
        cfg,
        count=int(args.num_observations),
        seed_start=args.eval_seed_start,
    )

    # Preserve run-time observation normalization but disable training, replay,
    # logging, and persistent workspace side effects.
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

    with tempfile.TemporaryDirectory(
        prefix="cqn-flow-utilization-"
    ) as work_dir:
        workspace = Workspace(cfg, work_dir=work_dir)
        try:
            workspace.load_snapshot(snapshot, load_replay_buffer=False)
            workspace._ensure_eval_envs_created()
            if workspace.eval_env is None:
                raise RuntimeError(
                    "flow-utilization probe requires one live evaluation env"
                )
            reset_observations = [
                workspace.eval_env.reset(seed=seed)[0] for seed in seeds
            ]
            observations = _stack_observations(reset_observations)
            if args.critic == "config":
                use_target = bool(
                    cfg.method.get("use_target_network_for_rollout", True)
                )
            else:
                use_target = args.critic == "target"
            metrics = workspace.agent.flow_utilization_probe(
                observations,
                actions=None,
                num_source_samples=int(args.num_source_samples),
                step_counts=tuple(args.step_counts),
                seed=int(args.probe_seed),
                use_target_network=use_target,
            )
        finally:
            workspace.shutdown()

    json_metrics = _jsonable(metrics)
    if not args.include_condition_actions:
        json_metrics.pop("condition_action", None)

    return {
        "status": "ok",
        "diagnostic_only": True,
        "selection_use_forbidden": True,
        "run_dir": str(run_dir),
        "snapshot": str(snapshot),
        "task": str(cfg.env.get("task_name", "")),
        "value_mode": str(cfg.method.value_mode),
        "configured_num_flow_steps": int(cfg.method.num_flow_steps),
        "step_counts": list(args.step_counts),
        "condition_action_source": "independent_bc_policy",
        "critic": "target" if use_target else "online",
        "num_observations": len(seeds),
        "num_source_samples": int(args.num_source_samples),
        "probe_seed": int(args.probe_seed),
        "include_condition_actions": bool(args.include_condition_actions),
        "eval_seeds": seeds,
        "metrics": json_metrics,
    }


def main() -> int:
    args = parse_args()
    _, _, output = resolve_paths(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        configure_process(args.gpu_id)
        payload = run_probe(args)
        payload["elapsed_seconds"] = time.time() - started
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
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
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        print(payload["traceback"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
