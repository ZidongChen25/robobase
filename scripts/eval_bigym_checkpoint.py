#!/usr/bin/env python3
"""Evaluate any BiGym checkpoint written by this repo.

Method- and task-agnostic counterpart to eval_cqn_as_bigym_checkpoint.py, which
asserts CQN-AS on move_plate. Implements the CLI contract async_eval_watcher.py
expects (--run-dir/--snapshot/--gpu-id/--num-eval-episodes/--eval-seed-start/
--output, JSON with success_percent), so it can be passed as --eval-script.

Compute and render devices are separate arguments on purpose: EGL enumeration
does not follow CUDA or nvidia-smi order, and deriving one id from the other has
repeatedly put renders on a different card than the compute. --gpu-id therefore
also accepts a GPU-<uuid> pin, and MUJOCO_EGL_DEVICE_ID inherited from the
environment always wins over anything derived here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument(
        "--gpu-id",
        type=str,
        default=None,
        help="Compute device: numeric CUDA id or GPU-<uuid>.",
    )
    parser.add_argument(
        "--egl-device-id",
        type=str,
        default=None,
        help="Render device for MUJOCO_EGL_DEVICE_ID. EGL ids do not follow "
        "CUDA order; measure before assuming. An inherited environment value "
        "takes precedence over this flag.",
    )
    parser.add_argument("--num-eval-episodes", type=int, default=50)
    parser.add_argument("--eval-seed-start", type=int, default=400)
    parser.add_argument("--num-eval-envs", type=int, default=1)
    parser.add_argument(
        "--obs-delay",
        type=int,
        default=None,
        help="Override delayed-policy conditioning h at evaluation time. "
        "Defaults to the value the checkpoint trained with.",
    )
    return parser.parse_args()


def configure_process(gpu_id: str | None, egl_device_id: str | None) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")
    if gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        os.environ["JAX_CUDA_VISIBLE_DEVICES"] = gpu_id
    if egl_device_id:
        os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", egl_device_id)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    run_dir = args.run_dir.expanduser().resolve()
    snapshot = args.snapshot or run_dir / "snapshots" / "latest_snapshot.pkl"
    output = args.output or run_dir / "bigym_eval.json"
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

    cfg.create_train_env = False
    cfg.num_train_envs = 0
    cfg.num_train_frames = 0
    cfg.num_eval_envs = int(args.num_eval_envs)
    cfg.num_eval_episodes = int(args.num_eval_episodes)
    cfg.env.eval_seed_start = int(args.eval_seed_start)
    if args.obs_delay is not None:
        cfg.obs_delay = int(args.obs_delay)
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
    return {
        "status": "ok",
        "task": str(cfg.env.task_name),
        "method": str(cfg.method.get("name", "")),
        "run_dir": str(run_dir),
        "snapshot": str(snapshot),
        "num_eval_episodes": int(cfg.num_eval_episodes),
        "eval_seed_start": int(cfg.env.eval_seed_start),
        "obs_delay": int(cfg.get("obs_delay", 0) or 0),
        "action_sequence": int(cfg.action_sequence),
        "execution_length": int(cfg.execution_length),
        "metrics": numeric_metrics,
        "success_percent": 100.0 * numeric_metrics["episode_success"],
    }


def main() -> int:
    args = parse_args()
    _, _, output, _ = resolve_paths(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        configure_process(args.gpu_id, args.egl_device_id)
        payload = run_eval(args)
        payload["elapsed_seconds"] = time.time() - started
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except Exception as error:  # noqa: BLE001 - reported through the JSON payload
        payload = {
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
        }
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
