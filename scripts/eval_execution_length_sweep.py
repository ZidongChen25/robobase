#!/usr/bin/env python3
"""Evaluate one checkpoint across execution lengths with one loaded model."""

from __future__ import annotations

import argparse
import csv
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
    parser.add_argument("--execution-lengths", nargs="+", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--num-eval-episodes", type=int, default=None)
    parser.add_argument("--num-eval-envs", type=int, default=1)
    parser.add_argument("--eval-seed-start", type=int, default=None)
    parser.add_argument("--action-sequence", type=int, default=None)
    parser.add_argument("--backbone-sequence-length", type=int, default=None)
    parser.add_argument("--flow-steps", type=int, default=None)
    parser.add_argument("--flow-schedule", type=str, default=None)
    parser.add_argument(
        "--temporal-ensemble",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the checkpoint run's temporal-ensemble setting.",
    )
    parser.add_argument("--lang-feature-source", type=str, default=None)
    parser.add_argument("--lang-feature-device", type=str, default=None)
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


def _metrics_to_float(metrics: dict) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }


def _write_result(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _restore_eval_state(workspace, snapshot: Path) -> None:
    workspace.load_snapshot(snapshot, load_replay_buffer=False)
    if hasattr(workspace.agent, "reset_aligned_eval_noise"):
        workspace.agent.reset_aligned_eval_noise()


def _load_cfg(args: argparse.Namespace):
    from omegaconf import OmegaConf

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
    if "env" in cfg and cfg.env is not None:
        if args.eval_seed_start is not None:
            cfg.env.eval_seed_start = int(args.eval_seed_start)
        elif "eval_seed_start" not in cfg.env:
            cfg.env.eval_seed_start = 0

    if args.action_sequence is not None:
        cfg.action_sequence = int(args.action_sequence)
        _set_if_present(cfg, "method.backbone.sequence_length", int(args.action_sequence))
    if args.temporal_ensemble is not None:
        cfg.temporal_ensemble = bool(args.temporal_ensemble)
    if args.backbone_sequence_length is not None:
        _set_if_present(
            cfg,
            "method.backbone.sequence_length",
            int(args.backbone_sequence_length),
        )

    _set_if_present(cfg, "wandb.use", False)
    _set_if_present(cfg, "tb.use", False)
    _set_if_present(cfg, "replay.num_workers", 0)
    _set_if_present(cfg, "lazy_replay.num_workers", 0)
    _set_if_present(cfg, "lazy_replay.persistent_workers", False)
    _set_if_present(cfg, "backend.replay_prefetch_size", 0)
    _set_if_present(cfg, "backend.replay_device_prefetch", False)
    _set_if_present(cfg, "backend.fused_update_steps", 1)
    _set_if_present(cfg, "backend.update_block_every_steps", 1)

    if args.flow_steps is not None and "method" in cfg:
        cfg.method.num_flow_steps = int(args.flow_steps)
        if "objective" in cfg.method and cfg.method.objective is not None:
            cfg.method.objective.num_flow_steps = int(args.flow_steps)
    if args.flow_schedule is not None and "method" in cfg:
        cfg.method.sample_schedule = str(args.flow_schedule)
        if "objective" in cfg.method and cfg.method.objective is not None:
            cfg.method.objective.sample_schedule = str(args.flow_schedule)
    if args.lang_feature_source is not None and "method" in cfg:
        cfg.method.lang_feature_source = str(args.lang_feature_source)
    if args.lang_feature_device is not None and "method" in cfg:
        cfg.method.lang_feature_device = str(args.lang_feature_device)

    OmegaConf.resolve(cfg)
    return cfg


def _run_sweep(args: argparse.Namespace) -> list[dict]:
    from robobase.workspace import Workspace

    cfg = _load_cfg(args)
    action_sequence = int(cfg.action_sequence)
    bad_lengths = [v for v in args.execution_lengths if v < 1 or v > action_sequence]
    if bad_lengths:
        raise ValueError(
            "execution lengths must be in [1, action_sequence]; "
            f"got {bad_lengths} with action_sequence={action_sequence}."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    workspace = Workspace(cfg, work_dir=str(args.work_dir))
    rows: list[dict] = []
    try:
        for execution_length in args.execution_lengths:
            started = time.time()
            print(f"eval execution_length={execution_length} started", flush=True)
            workspace._close_eval_envs()
            workspace.cfg.execution_length = int(execution_length)
            _restore_eval_state(workspace, args.snapshot)
            try:
                metrics = workspace.eval()
                payload = {
                    "status": "ok",
                    "task": (
                        str(cfg.env.task_name)
                        if "env" in cfg and "task_name" in cfg.env
                        else None
                    ),
                    "run_dir": str(args.run_dir),
                    "snapshot": str(args.snapshot),
                    "num_eval_episodes": int(cfg.num_eval_episodes),
                    "num_eval_envs": int(cfg.num_eval_envs),
                    "action_sequence": int(cfg.action_sequence),
                    "execution_length": int(execution_length),
                    "temporal_ensemble": bool(cfg.temporal_ensemble),
                    "flow_steps": (
                        int(cfg.method.num_flow_steps)
                        if "method" in cfg and "num_flow_steps" in cfg.method
                        else None
                    ),
                    "flow_schedule": (
                        str(cfg.method.get("sample_schedule", "uniform"))
                        if "method" in cfg
                        else None
                    ),
                    "gpu_id": args.gpu_id,
                    "metrics": _metrics_to_float(metrics),
                    "elapsed_seconds": time.time() - started,
                }
            except BaseException as exc:
                payload = {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "run_dir": str(args.run_dir),
                    "snapshot": str(args.snapshot),
                    "execution_length": int(execution_length),
                    "gpu_id": args.gpu_id,
                    "elapsed_seconds": time.time() - started,
                }
                _write_result(
                    args.output_dir / "results_json" / f"exec{execution_length}.json",
                    payload,
                )
                raise

            _write_result(
                args.output_dir / "results_json" / f"exec{execution_length}.json",
                payload,
            )
            row = {
                "execution_length": int(execution_length),
                "status": payload["status"],
                "episode_success": payload["metrics"].get("episode_success"),
                "episode_reward": payload["metrics"].get("episode_reward"),
                "episode_length": payload["metrics"].get("episode_length"),
                "elapsed_seconds": payload["elapsed_seconds"],
                "run_dir": str(args.run_dir),
                "snapshot": str(args.snapshot),
            }
            rows.append(row)
            print(
                "eval execution_length="
                f"{execution_length} done: success={row['episode_success']} "
                f"reward={row['episode_reward']} "
                f"elapsed={row['elapsed_seconds']:.1f}s",
                flush=True,
            )
    finally:
        workspace._close_eval_envs()
        workspace.shutdown()
    return rows


def _write_summary(rows: list[dict], output_dir: Path) -> None:
    summary_json = output_dir / "execution_length_sweep_results.json"
    _write_result(summary_json, {"rows": rows})

    csv_path = output_dir / "execution_length_sweep_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "execution_length",
                "status",
                "episode_success",
                "episode_reward",
                "episode_length",
                "elapsed_seconds",
                "run_dir",
                "snapshot",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = _parse_args()
    _configure_process(args.gpu_id)
    try:
        rows = _run_sweep(args)
        _write_summary(rows, args.output_dir)
        return 0
    except BaseException:
        print(traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
