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
    parser.add_argument("--action-sequence", type=int, default=None)
    parser.add_argument("--backbone-sequence-length", type=int, default=None)
    parser.add_argument("--execution-length", type=int, default=None)
    parser.add_argument("--flow-schedule", type=str, default=None)
    parser.add_argument("--lang-feature-path", type=Path, default=None)
    parser.add_argument("--encoder-weights-path", type=Path, default=None)
    parser.add_argument(
        "--xla-fusion-cache-dir",
        type=Path,
        default=None,
        help=(
            "Persist XLA GPU fusion autotuning decisions in this directory and "
            "enable deterministic GPU operations. Reuse the populated directory "
            "for repeatable checkpoint evaluations on identical hardware."
        ),
    )
    return parser.parse_args()


def _append_xla_flag(flag: str) -> None:
    """Append one XLA flag while rejecting a conflicting caller override."""
    flag_name = flag.split("=", 1)[0]
    current = os.environ.get("XLA_FLAGS", "").split()
    existing = [
        value
        for value in current
        if value == flag_name or value.startswith(f"{flag_name}=")
    ]
    if existing:
        if existing != [flag]:
            raise ValueError(
                f"Conflicting {flag_name} values in XLA_FLAGS: "
                f"{existing!r} versus requested {flag!r}."
            )
        return
    current.append(flag)
    os.environ["XLA_FLAGS"] = " ".join(current)


def _configure_process(
    gpu_id: int | None,
    xla_fusion_cache_dir: Path | None = None,
) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.25")
    _append_xla_flag("--xla_gpu_enable_command_buffer=")
    if xla_fusion_cache_dir is not None:
        cache_dir = xla_fusion_cache_dir.expanduser().resolve()
        if any(character.isspace() for character in str(cache_dir)):
            raise ValueError(
                "--xla-fusion-cache-dir cannot contain whitespace because XLA_FLAGS "
                f"is whitespace-delimited; got {cache_dir}."
            )
        cache_dir.mkdir(parents=True, exist_ok=True)
        _append_xla_flag(
            f"--xla_gpu_per_fusion_autotune_cache_dir={cache_dir}"
        )
        _append_xla_flag("--xla_gpu_deterministic_ops=true")
        _append_xla_flag("--xla_gpu_exclude_nondeterministic_ops=true")
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
    if args.action_sequence is not None:
        cfg.action_sequence = int(args.action_sequence)
        _set_if_present(cfg, "method.backbone.sequence_length", int(args.action_sequence))
        cfg.method.backbone.sequence_length = int(args.action_sequence)
    if args.backbone_sequence_length is not None:
        _set_if_present(
            cfg, "method.backbone.sequence_length", int(args.backbone_sequence_length)
        )
        cfg.method.backbone.sequence_length = int(args.backbone_sequence_length)

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
    if args.lang_feature_path is not None:
        lang_feature_path = args.lang_feature_path.expanduser().resolve()
        cfg.method.lang_feature_source = "precomputed"
        cfg.method.lang_feature_path = str(lang_feature_path)
    if args.encoder_weights_path is not None:
        encoder_weights_path = args.encoder_weights_path.expanduser().resolve()
        if not encoder_weights_path.is_file():
            raise FileNotFoundError(
                f"missing encoder weights: {encoder_weights_path}"
            )
        if cfg.method.get("encoder_model", None) is None:
            raise ValueError(
                "--encoder-weights-path requires method.encoder_model in the run config."
            )
        cfg.method.encoder_model.pretrained = True
        cfg.method.encoder_model.pretrained_weights_path = str(
            encoder_weights_path
        )

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
        "action_sequence": int(cfg.action_sequence),
        "execution_length": int(cfg.execution_length),
        "flow_schedule": str(cfg.method.get("sample_schedule", "uniform")),
        "train_time_schedule": str(
            cfg.method.get("objective", {}).get(
                "train_time_schedule",
                cfg.method.get("train_time_schedule", "uniform"),
            )
        ),
        "lang_feature_source": str(cfg.method.get("lang_feature_source", "tokens")),
        "lang_feature_path": cfg.method.get("lang_feature_path", None),
        "encoder_weights_path": (
            cfg.method.get("encoder_model", {}).get(
                "pretrained_weights_path",
                None,
            )
        ),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "xla_fusion_cache_dir": (
            str(args.xla_fusion_cache_dir.expanduser().resolve())
            if getattr(args, "xla_fusion_cache_dir", None) is not None
            else None
        ),
        "gpu_id": args.gpu_id,
        "metrics": {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        },
    }


def main() -> int:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        _configure_process(args.gpu_id, args.xla_fusion_cache_dir)
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
            "lang_feature_path": (
                str(args.lang_feature_path)
                if args.lang_feature_path is not None
                else None
            ),
            "encoder_weights_path": (
                str(args.encoder_weights_path)
                if args.encoder_weights_path is not None
                else None
            ),
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "xla_fusion_cache_dir": (
                str(args.xla_fusion_cache_dir.expanduser().resolve())
                if args.xla_fusion_cache_dir is not None
                else None
            ),
            "gpu_id": args.gpu_id,
            "elapsed_seconds": time.time() - started,
        }
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(payload["traceback"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
