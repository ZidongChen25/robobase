#!/usr/bin/env python3
"""Evaluate a JAX Q-chunking robomimic checkpoint (async-eval protocol).

Loads a training run's Hydra config, rebuilds an eval-only workspace on the
requested GPU, restores the snapshot, and reports success over N episodes as
JSON (``success_percent``), matching the interface expected by
``scripts/async_eval_watcher.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


# Official QC (ACFQL best-of-N) square-mh reference from the released plot data
# (plot_data/robomimic-individual.pkl, mean success over seeds). The step axis
# counts 1M offline gradient steps followed by 1M online environment steps,
# matching this repo's global_env_steps convention.
PAPER_TASK = "square_mh"
PAPER_OFFLINE_END_SUCCESS_PERCENT = 36.8
PAPER_ENDPOINT_SUCCESS_PERCENT = 92.8
PAPER_REFERENCE_CURVE = {
    0: 0.0, 100_000: 4.8, 200_000: 7.6, 300_000: 22.0, 400_000: 37.6,
    500_000: 36.0, 600_000: 43.1, 700_000: 42.8, 800_000: 36.8, 900_000: 38.1,
    1_000_000: 36.8, 1_100_000: 59.6, 1_200_000: 66.0, 1_300_000: 74.0,
    1_400_000: 77.2, 1_500_000: 76.8, 1_600_000: 87.2, 1_700_000: 89.2,
    1_800_000: 91.2, 1_900_000: 91.2, 2_000_000: 92.8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument(
        "--gpu-id",
        type=str,
        help="CUDA device for evaluation: a numeric id or an unambiguous "
        "GPU-<uuid> string (preferred here because a degraded card makes "
        "numeric CUDA indices differ from nvidia-smi indices).",
    )
    parser.add_argument("--num-eval-episodes", type=int, default=50)
    parser.add_argument("--eval-seed-start", type=int, default=400)
    parser.add_argument(
        "--actor-num-samples",
        type=int,
        default=None,
        help="Diagnostic override for best-of-N width (1 = pure BC flow).",
    )
    return parser.parse_args()


def configure_process(gpu_id: str | None) -> None:
    # Low-dim robomimic evaluation never renders, so leave MUJOCO_GL and
    # MUJOCO_EGL_DEVICE_ID unset: under a CUDA_VISIBLE_DEVICES pin NVIDIA
    # filters EGL device enumeration, and an absolute EGL device id would
    # crash the import-time EGL context creation.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")
    if gpu_id:
        gpu = str(gpu_id).strip()
        if not (gpu.lstrip("-").isdigit() and int(gpu) < 0):
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu
            os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    run_dir = args.run_dir.expanduser().resolve()
    snapshot = args.snapshot or run_dir / "snapshots" / "latest_snapshot.pkl"
    output = args.output or run_dir / "q_chunking_robomimic_eval.json"
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
    if str(cfg.method.get("name", "")).lower() != "q_chunking":
        raise ValueError(
            f"checkpoint method is not q_chunking: {cfg.method.get('name')}"
        )
    if str(cfg.env.env_name) != "robomimic":
        raise ValueError(f"checkpoint env is not robomimic: {cfg.env.env_name}")

    cfg.create_train_env = False
    cfg.num_train_envs = 0
    cfg.num_train_frames = 0
    cfg.num_pretrain_steps = 0
    cfg.num_eval_envs = 1
    cfg.num_eval_episodes = int(args.num_eval_episodes)
    cfg.env.eval_seed_start = int(args.eval_seed_start)
    if args.actor_num_samples is not None:
        cfg.method.actor_num_samples = int(args.actor_num_samples)
    cfg.demos = 0
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
        raise RuntimeError("robomimic evaluation did not report episode_success")
    success_percent = 100.0 * numeric_metrics["episode_success"]
    return {
        "status": "ok",
        "task": PAPER_TASK,
        "run_dir": str(run_dir),
        "snapshot": str(snapshot),
        "num_eval_episodes": int(cfg.num_eval_episodes),
        "eval_seed_start": int(cfg.env.eval_seed_start),
        "metrics": numeric_metrics,
        "success_percent": success_percent,
        "mean_reward": numeric_metrics.get("episode_reward"),
        "paper_comparison": {
            "source": (
                "official qc plot_data/robomimic-individual.pkl, (square, QC), "
                "mean success over seeds; steps = 1M offline + 1M online"
            ),
            "offline_end_success_percent": PAPER_OFFLINE_END_SUCCESS_PERCENT,
            "final_success_percent": PAPER_ENDPOINT_SUCCESS_PERCENT,
            "reference_curve_percent": {
                str(step): value for step, value in PAPER_REFERENCE_CURVE.items()
            },
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
