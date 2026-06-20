#!/usr/bin/env python3
"""Evaluate one Flow Matching checkpoint with a chosen sampling step count."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--flow-steps", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--num-eval-episodes", type=int, default=None)
    parser.add_argument("--num-eval-envs", type=int, default=1)
    parser.add_argument("--execution-length", type=int, default=None)
    parser.add_argument("--flow-schedule", type=str, default=None)
    return parser.parse_args()


def _configure_process(gpu_id: int | None) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.25")
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
    if gpu_id is not None and gpu_id >= 0:
        gpu = str(gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        os.environ["JAX_CUDA_VISIBLE_DEVICES"] = gpu
        os.environ["MUJOCO_EGL_DEVICE_ID"] = gpu


def _set_if_present(cfg, dotted_key: str, value) -> None:
    node = cfg
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in node or node[part] is None:
            return
        node = node[part]
    node[parts[-1]] = value


def _run_eval(args: argparse.Namespace) -> dict:
    from omegaconf import OmegaConf

    from robobase.workspace import Workspace

    cfg_path = args.run_dir / ".hydra" / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing config: {cfg_path}")
    if not args.snapshot.is_file():
        raise FileNotFoundError(f"missing snapshot: {args.snapshot}")

    cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_struct(cfg, False)

    cfg.create_train_env = False
    cfg.num_train_envs = 0
    cfg.num_train_frames = 0
    cfg.num_eval_envs = int(args.num_eval_envs)
    if args.num_eval_episodes is not None:
        cfg.num_eval_episodes = int(args.num_eval_episodes)
    cfg.log_train_video = False
    cfg.log_eval_video = False
    cfg.save_snapshot = False
    cfg.save_csv = False
    cfg.gpu_id = None
    if args.execution_length is not None:
        cfg.execution_length = int(args.execution_length)

    _set_if_present(cfg, "wandb.use", False)
    _set_if_present(cfg, "tb.use", False)
    _set_if_present(cfg, "replay.num_workers", 0)
    _set_if_present(cfg, "lazy_replay.num_workers", 0)
    _set_if_present(cfg, "lazy_replay.persistent_workers", False)
    _set_if_present(cfg, "backend.replay_prefetch_size", 0)
    _set_if_present(cfg, "backend.replay_device_prefetch", False)
    _set_if_present(cfg, "backend.fused_update_steps", 1)
    _set_if_present(cfg, "backend.update_block_every_steps", 1)

    cfg.method.num_flow_steps = int(args.flow_steps)
    if "objective" in cfg.method and cfg.method.objective is not None:
        cfg.method.objective.num_flow_steps = int(args.flow_steps)
    if args.flow_schedule is not None:
        cfg.method.sample_schedule = str(args.flow_schedule)
        if "objective" in cfg.method and cfg.method.objective is not None:
            cfg.method.objective.sample_schedule = str(args.flow_schedule)

    OmegaConf.resolve(cfg)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    workspace = Workspace(cfg, work_dir=str(args.work_dir))
    try:
        workspace.load_snapshot(args.snapshot, load_replay_buffer=False)
        metrics = workspace.eval()
    finally:
        workspace.shutdown()

    task_name = cfg.env.task_name if "env" in cfg and "task_name" in cfg.env else None
    return {
        "status": "ok",
        "task": task_name,
        "run_dir": str(args.run_dir),
        "snapshot": str(args.snapshot),
        "flow_steps": int(args.flow_steps),
        "num_eval_episodes": int(cfg.num_eval_episodes),
        "num_eval_envs": int(cfg.num_eval_envs),
        "execution_length": int(cfg.execution_length),
        "flow_schedule": str(cfg.method.get("sample_schedule", "uniform")),
        "gpu_id": args.gpu_id,
        "metrics": {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        },
    }


def main() -> int:
    args = _parse_args()
    _configure_process(args.gpu_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        payload = _run_eval(args)
        payload["elapsed_seconds"] = time.time() - started
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return 0
    except BaseException as exc:
        payload = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "run_dir": str(args.run_dir),
            "snapshot": str(args.snapshot),
            "flow_steps": int(args.flow_steps),
            "flow_schedule": args.flow_schedule,
            "gpu_id": args.gpu_id,
            "elapsed_seconds": time.time() - started,
        }
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(payload["traceback"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
