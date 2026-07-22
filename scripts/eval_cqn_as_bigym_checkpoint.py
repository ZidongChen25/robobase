#!/usr/bin/env python3
"""Evaluate a JAX CQN-AS BiGym move_plate checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


PAPER_TASK = "move_plate"
PAPER_ENDPOINT_SUCCESS_PERCENT = 64.0
PAPER_ENDPOINT_STD_PERCENT = 7.48331477


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--num-eval-episodes", type=int, default=25)
    parser.add_argument("--eval-seed-start", type=int, default=20000)
    parser.add_argument(
        "--method-temporal-ensemble",
        choices=("config", "true", "false"),
        default="config",
        help=(
            "Override CQN-AS agent-side temporal ensembling. 'false' evaluates "
            "one cached K-step plan open-loop while preserving primitive-step replay "
            "and environment stepping semantics."
        ),
    )
    parser.add_argument(
        "--temporal-ensemble-replan-interval",
        type=int,
        help=(
            "Infer/register one new chunk every N primitive steps while agent-side "
            "temporal ensembling remains enabled. The config default is 1."
        ),
    )
    parser.add_argument(
        "--single-run-tolerance-percent",
        type=float,
        default=20.0,
        help="Tolerance around the paper's eight-run mean for one checkpoint.",
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


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    run_dir = args.run_dir.expanduser().resolve()
    snapshot = args.snapshot or run_dir / "snapshots" / "latest_snapshot.pkl"
    output = args.output or run_dir / "cqn_as_move_plate_eval.json"
    work_dir = args.work_dir or run_dir / "eval_only"
    return (
        run_dir,
        snapshot.expanduser().resolve(),
        output.expanduser().resolve(),
        work_dir.expanduser().resolve(),
    )


def run_eval(args: argparse.Namespace) -> dict:
    from omegaconf import OmegaConf

    from robobase.workspace import Workspace

    run_dir, snapshot, _, work_dir = resolve_paths(args)
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing config: {cfg_path}")
    if not snapshot.is_file():
        raise FileNotFoundError(f"missing snapshot: {snapshot}")
    if args.num_eval_episodes < 1:
        raise ValueError("--num-eval-episodes must be at least 1")

    cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_struct(cfg, False)
    if str(cfg.method.get("name", "")).lower() != "cqn_as":
        raise ValueError(f"checkpoint method is not CQN-AS: {cfg.method.get('name')}")
    if str(cfg.env.task_name) != PAPER_TASK:
        raise ValueError(f"expected {PAPER_TASK}, got {cfg.env.task_name}")

    method_temporal_ensemble = getattr(
        args,
        "method_temporal_ensemble",
        "config",
    )
    replan_interval = getattr(
        args,
        "temporal_ensemble_replan_interval",
        None,
    )
    if method_temporal_ensemble != "config":
        cfg.method.temporal_ensemble = method_temporal_ensemble == "true"
    if replan_interval is not None:
        cfg.method.temporal_ensemble_replan_interval = int(replan_interval)

    cfg.create_train_env = False
    cfg.num_train_envs = 0
    cfg.num_train_frames = 0
    cfg.num_eval_envs = 1
    cfg.num_eval_episodes = int(args.num_eval_episodes)
    cfg.env.eval_seed_start = int(args.eval_seed_start)
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
    if "episode_success" not in numeric_metrics:
        raise RuntimeError("BiGym evaluation did not report episode_success")
    success_percent = 100.0 * numeric_metrics["episode_success"]
    delta = success_percent - PAPER_ENDPOINT_SUCCESS_PERCENT
    tolerance = float(args.single_run_tolerance_percent)
    alignment = (
        "within_single_run_band"
        if abs(delta) <= tolerance
        else ("above_single_run_band" if delta > 0 else "below_single_run_band")
    )
    return {
        "status": "ok",
        "task": PAPER_TASK,
        "run_dir": str(run_dir),
        "snapshot": str(snapshot),
        "num_eval_episodes": int(cfg.num_eval_episodes),
        "eval_seed_start": int(cfg.env.eval_seed_start),
        "method_temporal_ensemble": bool(cfg.method.temporal_ensemble),
        "temporal_ensemble_replan_interval": int(
            cfg.method.get("temporal_ensemble_replan_interval", 1)
        ),
        "metrics": numeric_metrics,
        "success_percent": success_percent,
        "paper_comparison": {
            "source": "official CQN-AS bigym_results.pkl at 100000 environment steps",
            "paper_statistic": "mean and standard deviation across 8 runs",
            "reference_success_percent": PAPER_ENDPOINT_SUCCESS_PERCENT,
            "reference_std_percent": PAPER_ENDPOINT_STD_PERCENT,
            "single_run_tolerance_percent": tolerance,
            "success_delta_percent": delta,
            "alignment": alignment,
        },
    }


def main() -> int:
    args = parse_args()
    _, _, output, _ = resolve_paths(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        configure_process(args.gpu_id)
        payload = run_eval(args)
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
