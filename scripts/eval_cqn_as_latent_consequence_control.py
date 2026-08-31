#!/usr/bin/env python3
"""Matched selector controls for a trained RGB latent-consequence checkpoint.

The checkpoint is loaded once through its frozen training configuration.  The
evaluation policy is then selected without changing any learned parameters:

``direct``
    Discard the auxiliary model at action selection and run the wrapped
    canonical CQN-AS policy.
``learned``
    Use the learned one-step latent/reward/done predictor for sparse reranking.
``ground_truth``
    Discard the predictor at action selection and use an exactly restored
    BiGym simulator for the same H=1 candidate scores.

This three-way comparison separates auxiliary-training interference from
successor-model approximation error on the very same checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--gpu-id", type=str)
    parser.add_argument("--egl-device-id", type=str)
    parser.add_argument(
        "--selector",
        required=True,
        choices=("direct", "learned", "ground_truth"),
    )
    parser.add_argument("--num-eval-episodes", type=int, default=50)
    parser.add_argument("--eval-seed-start", type=int, default=400)
    parser.add_argument("--proposal-level", type=int, default=1)
    parser.add_argument("--switch-margin", type=float, default=1e-5)
    parser.add_argument("--rerank-interval", type=int, default=16)
    parser.add_argument(
        "--dimension-selection",
        choices=("q_span", "round_robin"),
        default="q_span",
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
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_deterministic_ops=true")
    if gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        os.environ["JAX_CUDA_VISIBLE_DEVICES"] = gpu_id
    if egl_device_id:
        os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", egl_device_id)


def _configure_eval(cfg, args: argparse.Namespace) -> None:
    from omegaconf import OmegaConf

    OmegaConf.set_struct(cfg, False)
    if str(cfg.env.get("env_name", "")).lower() != "bigym":
        raise ValueError("latent-consequence controls require a BiGym run")
    if not bool(cfg.pixels):
        raise ValueError("latent-consequence controls require pixels=true")
    if int(cfg.action_sequence) != 16 or int(cfg.execution_length) != 1:
        raise ValueError(
            "registered controls require action_sequence=16 and execution_length=1"
        )

    cfg.create_train_env = False
    cfg.num_train_envs = 0
    cfg.num_train_frames = 0
    cfg.num_eval_envs = 1
    cfg.num_eval_episodes = int(args.num_eval_episodes)
    cfg.env.eval_seed_start = int(args.eval_seed_start)
    cfg.num_pretrain_steps = 0
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


def run(args: argparse.Namespace) -> dict:
    from omegaconf import OmegaConf

    from robobase.method.cqn_as_bigym_latent_successor import (
        CQNASBigymGroundTruthLatentSuccessor,
    )
    from robobase.method.cqn_as_latent_consequence import CQNASLatentConsequence
    from robobase.workspace import Workspace

    run_dir = args.run_dir.expanduser().resolve()
    snapshot = args.snapshot.expanduser().resolve()
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing config: {cfg_path}")
    if not snapshot.is_file():
        raise FileNotFoundError(f"missing snapshot: {snapshot}")
    if args.num_eval_episodes < 1:
        raise ValueError("num-eval-episodes must be positive")

    cfg = OmegaConf.load(cfg_path)
    _configure_eval(cfg, args)
    work_dir = (
        args.work_dir.expanduser().resolve()
        if args.work_dir is not None
        else args.output.expanduser().resolve().parent / "eval_workspace"
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    workspace = Workspace(cfg, work_dir=str(work_dir))
    selector_agent = None
    trained_agent = None
    try:
        workspace.load_snapshot(snapshot, load_replay_buffer=False)
        trained_agent = workspace.agent
        if not isinstance(trained_agent, CQNASLatentConsequence):
            raise TypeError(
                "expected a CQNASLatentConsequence checkpoint, got "
                f"{type(trained_agent).__qualname__}"
            )

        # Loading happens before unwrapping so both the canonical base and the
        # auxiliary ensemble always come from exactly the same checkpoint.
        if args.selector == "direct":
            selector_agent = trained_agent.base
        elif args.selector == "learned":
            selector_agent = trained_agent
        else:
            selector_agent = CQNASBigymGroundTruthLatentSuccessor(
                trained_agent.base,
                discount=float(cfg.get("discount", 0.99)),
                horizon=1,
                proposal_level=int(args.proposal_level),
                switch_margin=float(args.switch_margin),
                dimension_selection=str(args.dimension_selection),
                rerank_interval=int(args.rerank_interval),
            )
        workspace.agent = selector_agent
        metrics = workspace.eval()
        selector_diagnostics = selector_agent.rollout_diagnostics()
        model_diagnostics = trained_agent.rollout_diagnostics()
    finally:
        workspace.shutdown()

    numeric_metrics = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }
    numeric_selector_diagnostics = {
        key: float(value)
        for key, value in selector_diagnostics.items()
        if isinstance(value, (int, float))
    }
    numeric_model_diagnostics = {
        key: float(value)
        for key, value in model_diagnostics.items()
        if isinstance(value, (int, float))
    }
    return {
        "status": "ok",
        "diagnostic": "matched_latent_consequence_selector_control",
        "selector": str(args.selector),
        "task": str(cfg.env.task_name),
        "run_dir": str(run_dir),
        "snapshot": str(snapshot),
        "action_sequence": int(cfg.action_sequence),
        "execution_length": int(cfg.execution_length),
        "horizon": 1 if args.selector == "ground_truth" else None,
        "proposal_level": int(args.proposal_level),
        "dimension_selection": str(args.dimension_selection),
        "switch_margin": float(args.switch_margin),
        "rerank_interval": int(args.rerank_interval),
        "num_eval_episodes": int(args.num_eval_episodes),
        "eval_seed_start": int(args.eval_seed_start),
        "model_updates": int(trained_agent.model_updates),
        "metrics": numeric_metrics,
        "selector_diagnostics": numeric_selector_diagnostics,
        "model_diagnostics": numeric_model_diagnostics,
        "success_percent": 100.0 * numeric_metrics["episode_success"],
    }


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        configure_process(args.gpu_id, args.egl_device_id)
        payload = run(args)
        payload["elapsed_seconds"] = time.time() - started
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except Exception as error:  # noqa: BLE001 - persist complete eval failure
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
