#!/usr/bin/env python3
"""Evaluate a saved CQN DMC checkpoint and compare it with Figure 9."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


PAPER_TASK = "cartpole_swingup_sparse"
# Approximate endpoint read from Figure 9.  The paper curve is a four-seed mean,
# so this is deliberately reported as a reference rather than an exact oracle.
PAPER_REFERENCE_RETURN = 780.0
PAPER_REFERENCE_TOLERANCE = 100.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--num-eval-episodes", type=int, default=50)
    parser.add_argument("--eval-seed-start", type=int, default=20000)
    parser.add_argument(
        "--paper-reference-return", type=float, default=PAPER_REFERENCE_RETURN
    )
    parser.add_argument(
        "--paper-tolerance", type=float, default=PAPER_REFERENCE_TOLERANCE
    )
    return parser.parse_args()


def _configure_process(gpu_id: int | None) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.25")
    if gpu_id is not None and gpu_id >= 0:
        gpu = str(gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        os.environ["JAX_CUDA_VISIBLE_DEVICES"] = gpu
        os.environ["MUJOCO_EGL_DEVICE_ID"] = gpu


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    run_dir = args.run_dir.expanduser().resolve()
    snapshot = args.snapshot
    if snapshot is None:
        snapshot = run_dir / "snapshots" / "latest_snapshot.pkl"
    snapshot = snapshot.expanduser().resolve()
    output = args.output or (run_dir / "cqn_paper_eval.json")
    work_dir = args.work_dir or (run_dir / "eval_only")
    return snapshot, output.expanduser().resolve(), work_dir.expanduser().resolve()


def _run_eval(args: argparse.Namespace) -> dict:
    from omegaconf import OmegaConf

    from robobase.workspace import Workspace

    run_dir = args.run_dir.expanduser().resolve()
    snapshot, _, work_dir = _resolve_paths(args)
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing config: {cfg_path}")
    if not snapshot.is_file():
        raise FileNotFoundError(f"missing snapshot: {snapshot}")
    if args.num_eval_episodes < 1:
        raise ValueError("--num-eval-episodes must be at least 1")
    if args.paper_tolerance < 0:
        raise ValueError("--paper-tolerance must be non-negative")

    cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_struct(cfg, False)
    if str(cfg.method.get("name", "")).lower() != "cqn":
        raise ValueError(f"checkpoint method is not CQN: {cfg.method.get('name')}")
    if str(cfg.env.task_name) != PAPER_TASK:
        raise ValueError(
            f"expected {PAPER_TASK}, checkpoint config has {cfg.env.task_name}"
        )

    cfg.create_train_env = False
    cfg.num_train_envs = 0
    cfg.num_train_frames = 0
    cfg.num_eval_envs = 1
    cfg.num_eval_episodes = int(args.num_eval_episodes)
    cfg.env.eval_seed_start = int(args.eval_seed_start)
    cfg.log_train_video = False
    cfg.log_eval_video = False
    cfg.save_snapshot = False
    cfg.save_csv = False
    cfg.gpu_id = None
    cfg.wandb.use = False
    cfg.tb.use = False
    cfg.replay.num_workers = 0
    if cfg.get("lazy_replay", None) is not None:
        cfg.lazy_replay.num_workers = 0
        cfg.lazy_replay.persistent_workers = False
    if cfg.get("backend", None) is not None:
        cfg.backend.replay_prefetch_size = 0
        cfg.backend.replay_device_prefetch = False
        cfg.backend.fused_update_steps = 1
        cfg.backend.update_block_every_steps = 1
    OmegaConf.resolve(cfg)

    work_dir.mkdir(parents=True, exist_ok=True)
    workspace = Workspace(cfg, work_dir=str(work_dir))
    try:
        workspace.load_snapshot(snapshot, load_replay_buffer=False)
        metrics = workspace.eval()
    finally:
        workspace.shutdown()

    numeric_metrics = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }
    episode_return = numeric_metrics.get("episode_reward")
    if episode_return is None:
        raise RuntimeError("evaluation did not return episode_reward")
    reference = float(args.paper_reference_return)
    tolerance = float(args.paper_tolerance)
    delta = episode_return - reference
    if delta > tolerance:
        alignment = "above_reference_band"
    elif delta < -tolerance:
        alignment = "below_reference_band"
    else:
        alignment = "within_reference_band"

    return {
        "status": "ok",
        "task": str(cfg.env.task_name),
        "run_dir": str(run_dir),
        "snapshot": str(snapshot),
        "num_eval_episodes": int(cfg.num_eval_episodes),
        "eval_seed_start": int(cfg.env.eval_seed_start),
        "metrics": numeric_metrics,
        "paper_comparison": {
            "source": "CQN paper Figure 9 endpoint, approximate visual reading",
            "paper_statistic": "mean across 4 training seeds",
            "reference_return": reference,
            "tolerance": tolerance,
            "return_delta": delta,
            "alignment": alignment,
            "meets_reference_lower_band": episode_return >= reference - tolerance,
        },
    }


def main() -> int:
    args = _parse_args()
    _, output, _ = _resolve_paths(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        _configure_process(args.gpu_id)
        payload = _run_eval(args)
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
