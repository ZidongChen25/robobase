#!/usr/bin/env python3
"""Evaluate one CQN-Flow BC/value blend with per-seed outcomes.

The evaluator constructs the agent with an eval-time ``policy_value_beta``
override before loading checkpoint parameters.  ``bc`` selects the independent
BC policy when one exists; single-tower CQN-AS/CQN-Flow checkpoints retain
their native value-selected action path.  For Flow-V/direct-A checkpoints a
non-negative number selects

    normalized_A + beta * log pi_BC.

For FLOQ-distill checkpoints it instead enables the online scalar readout and
selects

    normalized_Q_distill + beta * log pi_BC.

Each requested seed is evaluated separately so multiple variants can be
compared with paired statistics after the runs finish.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path


def _policy_value_beta(value: str) -> float | None:
    normalized = value.strip().lower()
    if normalized in {"bc", "none", "null"}:
        return None
    try:
        beta = float(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "policy value beta must be 'bc' or a non-negative number"
        ) from exc
    if not math.isfinite(beta) or beta < 0.0:
        raise argparse.ArgumentTypeError(
            "policy value beta must be finite and non-negative"
        )
    return beta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--gpu-id", required=True, type=int)
    parser.add_argument("--num-eval-episodes", type=int, default=50)
    parser.add_argument("--eval-seed-start", type=int, default=32000)
    parser.add_argument(
        "--policy-value-beta",
        required=True,
        type=_policy_value_beta,
        help="'bc' for BC-only, 0 for A-only, or beta >= 0 for A + BC.",
    )
    parser.add_argument(
        "--return-sample-aggregation",
        choices=("config", "mean", "entropic", "truncated_mean"),
        default="config",
        help=(
            "Eval-time return-flow readout. 'entropic' uses "
            "eta*log(mean(exp(return/eta))); 'truncated_mean' sorts return "
            "samples and drops the requested largest values."
        ),
    )
    parser.add_argument("--return-sample-temperature", type=float)
    parser.add_argument(
        "--return-sample-truncate-top",
        type=int,
        help=(
            "Eval-time number of largest return samples discarded by "
            "truncated_mean."
        ),
    )
    parser.add_argument(
        "--flow-readout",
        choices=("auto", "distill", "integrated"),
        default="auto",
        help=(
            "Use the cheap FLOQ-distilled head or the integrated flow field "
            "for numeric policy/value modes."
        ),
    )
    parser.add_argument("--num-flow-steps", type=int)
    parser.add_argument(
        "--num-action-flow-samples",
        type=int,
        help=(
            "Eval-time number of flow sources aggregated for each candidate "
            "action bin. This does not change checkpoint parameters."
        ),
    )
    return parser.parse_args()


def configure_process(gpu_id: int) -> None:
    gpu = str(gpu_id)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["JAX_CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["MUJOCO_EGL_DEVICE_ID"] = gpu


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    run_dir = args.run_dir.expanduser().resolve()
    snapshot = args.snapshot or run_dir / "snapshots" / "latest_snapshot.pkl"
    return (
        run_dir,
        snapshot.expanduser().resolve(),
        args.output.expanduser().resolve(),
        args.work_dir.expanduser().resolve(),
    )


def _prepare_cfg(args: argparse.Namespace, run_dir: Path, work_dir: Path):
    from omegaconf import OmegaConf

    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing config: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_struct(cfg, False)
    method_name = str(cfg.method.get("name", "")).lower()
    if method_name not in {"cqn_as", "cqn_flow"}:
        raise ValueError(
            "checkpoint method is not cqn_as/cqn_flow: "
            f"{cfg.method.get('name')}"
        )

    cfg.method.policy_value_beta = args.policy_value_beta
    if args.num_flow_steps is not None and args.num_flow_steps < 1:
        raise ValueError("--num-flow-steps must be positive")
    if (
        args.num_action_flow_samples is not None
        and args.num_action_flow_samples < 1
    ):
        raise ValueError("--num-action-flow-samples must be positive")
    if args.return_sample_temperature is not None and (
        not math.isfinite(args.return_sample_temperature)
        or args.return_sample_temperature <= 0.0
    ):
        raise ValueError("--return-sample-temperature must be positive")
    if (
        args.return_sample_truncate_top is not None
        and args.return_sample_truncate_top < 0
    ):
        raise ValueError(
            "--return-sample-truncate-top must be non-negative"
        )
    if method_name == "cqn_flow":
        if args.num_flow_steps is not None:
            cfg.method.num_flow_steps = args.num_flow_steps
        if args.num_action_flow_samples is not None:
            cfg.method.num_action_flow_samples = (
                args.num_action_flow_samples
            )
        if args.return_sample_aggregation != "config":
            cfg.method.return_sample_aggregation = (
                args.return_sample_aggregation
            )
        if args.return_sample_temperature is not None:
            cfg.method.return_sample_temperature = (
                args.return_sample_temperature
            )
        if args.return_sample_truncate_top is not None:
            cfg.method.return_sample_truncate_top = (
                args.return_sample_truncate_top
            )
    has_flow_distill = (
        float(cfg.method.get("flow_distill_lambda", 0.0)) > 0.0
    )
    if method_name != "cqn_flow" and args.flow_readout != "auto":
        raise ValueError("--flow-readout applies only to CQN-Flow")
    if args.flow_readout == "distill" and not has_flow_distill:
        raise ValueError(
            "--flow-readout=distill requires a checkpoint trained with "
            "flow_distill_lambda > 0"
        )
    flow_backend = (
        "distill"
        if method_name == "cqn_flow"
        and (
            args.flow_readout == "distill"
            or (args.flow_readout == "auto" and has_flow_distill)
        )
        else "integrated"
    )
    if method_name == "cqn_flow" and flow_backend == "distill":
        cfg.method.flow_distill_action_readout = (
            args.policy_value_beta is not None
        )
        cfg.method.flow_q_action_readout = False
    elif (
        method_name == "cqn_flow"
        and str(cfg.method.get("critic_architecture", "flow_q")).lower()
        == "flow_q"
    ):
        cfg.method.flow_distill_action_readout = False
        cfg.method.flow_q_action_readout = (
            args.policy_value_beta is not None
        )
    cfg.create_train_env = False
    cfg.num_train_envs = 0
    cfg.num_train_frames = 0
    cfg.num_eval_envs = 1
    # Run one seed at a time to retain paired episode outcomes.
    cfg.num_eval_episodes = 1
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
    cfg.hydra = {"run": {"dir": str(work_dir)}}
    OmegaConf.resolve(cfg)
    return cfg


def _numeric_metrics(metrics: dict) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }


def run_eval(args: argparse.Namespace) -> dict:
    import jax

    from robobase.workspace import Workspace

    run_dir, snapshot, _, work_dir = _resolve_paths(args)
    if not snapshot.is_file():
        raise FileNotFoundError(f"missing snapshot: {snapshot}")
    if args.num_eval_episodes < 1:
        raise ValueError("--num-eval-episodes must be at least 1")

    work_dir.mkdir(parents=True, exist_ok=True)
    cfg = _prepare_cfg(args, run_dir, work_dir)
    workspace = Workspace(cfg, work_dir=str(work_dir))
    episode_results = []
    try:
        workspace.load_snapshot(snapshot, load_replay_buffer=False)
        active_beta = getattr(workspace.agent, "policy_value_beta", object())
        if active_beta != args.policy_value_beta:
            raise RuntimeError(
                "agent policy_value_beta does not match the eval override: "
                f"{active_beta!r} != {args.policy_value_beta!r}"
            )
        has_flow_distill = (
            float(getattr(workspace.agent, "flow_distill_lambda", 0.0)) > 0.0
        )
        method_name = str(cfg.method.get("name", "")).lower()
        direct_scalar_q = bool(cfg.method.get("direct_scalar_q", False))
        flow_backend = (
            "distill"
            if method_name == "cqn_flow"
            and (
                args.flow_readout == "distill"
                or (args.flow_readout == "auto" and has_flow_distill)
            )
            else "integrated"
        )
        if method_name == "cqn_flow" and flow_backend == "distill":
            expected_action_readout = args.policy_value_beta is not None
            if (
                workspace.agent.flow_distill_action_readout
                != expected_action_readout
            ):
                raise RuntimeError(
                    "flow_distill_action_readout does not match the requested "
                    f"policy mode: expected {expected_action_readout}"
                )
            if "flow_distill_readout" not in workspace.agent.params:
                raise RuntimeError(
                    "FLOQ-distill checkpoint has no scalar readout parameters"
                )
            if bool(getattr(workspace.agent, "flow_q_action_readout", False)):
                raise RuntimeError(
                    "integrated flow readout is unexpectedly enabled"
                )
        elif method_name == "cqn_flow":
            expected_flow_q_readout = (
                args.policy_value_beta is not None
                and str(
                    cfg.method.get("critic_architecture", "flow_q")
                ).lower()
                == "flow_q"
            )
            if (
                bool(getattr(workspace.agent, "flow_q_action_readout", False))
                != expected_flow_q_readout
            ):
                raise RuntimeError(
                    "flow_q_action_readout does not match the requested "
                    f"policy mode: expected {expected_flow_q_readout}"
                )
            if bool(
                getattr(
                    workspace.agent,
                    "flow_distill_action_readout",
                    False,
                )
            ):
                raise RuntimeError(
                    "distilled flow readout is unexpectedly enabled"
                )

        for episode in range(args.num_eval_episodes):
            seed = int(args.eval_seed_start) + episode
            # Use common random numbers across independently launched variants.
            # CQN-AS/CQN-Flow only consumes this key for action inference and
            # tie-breaking during evaluation.
            workspace.agent.rng_key = jax.random.PRNGKey(seed + 910_000)
            workspace.cfg.env.eval_seed_start = seed
            metrics = _numeric_metrics(workspace.eval())
            if "episode_success" not in metrics:
                raise RuntimeError(
                    f"evaluation seed {seed} did not report episode_success"
                )
            episode_results.append({"seed": seed, **metrics})
            if (episode + 1) % 10 == 0 or episode + 1 == args.num_eval_episodes:
                successes = sum(
                    result["episode_success"] for result in episode_results
                )
                print(
                    f"completed {episode + 1}/{args.num_eval_episodes}: "
                    f"success={successes / len(episode_results):.3f}",
                    flush=True,
                )
    finally:
        workspace.shutdown()

    successes = [result["episode_success"] for result in episode_results]
    return {
        "status": "ok",
        "task": str(cfg.env.task_name),
        "run_dir": str(run_dir),
        "snapshot": str(snapshot),
        "policy_value_beta": args.policy_value_beta,
        "value_readout": (
            (
                "direct_scalar_q"
                if direct_scalar_q
                else "direct_c51"
            )
            if str(cfg.method.get("name", "")).lower() == "cqn_as"
            else (
                "flow_distill"
                if flow_backend == "distill"
                else (
                    "flow_q"
                    if str(
                        cfg.method.get("critic_architecture", "flow_q")
                    ).lower()
                    == "flow_q"
                    else "direct_advantage"
                )
            )
        ),
        "flow_readout": (
            None
            if str(cfg.method.get("name", "")).lower() != "cqn_flow"
            else flow_backend
        ),
        "num_flow_steps": (
            None
            if str(cfg.method.get("name", "")).lower() != "cqn_flow"
            else int(cfg.method.num_flow_steps)
        ),
        "num_action_flow_samples": (
            None
            if str(cfg.method.get("name", "")).lower() != "cqn_flow"
            else int(cfg.method.num_action_flow_samples)
        ),
        "return_sample_aggregation": (
            None
            if str(cfg.method.get("name", "")).lower() != "cqn_flow"
            else str(cfg.method.get("return_sample_aggregation", "mean"))
        ),
        "return_sample_temperature": (
            None
            if str(cfg.method.get("name", "")).lower() != "cqn_flow"
            else float(cfg.method.get("return_sample_temperature", 1.0))
        ),
        "return_sample_truncate_top": (
            None
            if str(cfg.method.get("name", "")).lower() != "cqn_flow"
            else int(cfg.method.get("return_sample_truncate_top", 0))
        ),
        "policy": (
            "native_cqn_as"
            if (
                str(cfg.method.get("name", "")).lower() == "cqn_as"
                and not bool(cfg.method.get("separate_bc_policy", False))
            )
            else (
                "native_flow_value"
                if (
                    str(cfg.method.get("name", "")).lower() == "cqn_flow"
                    and not bool(
                        cfg.method.get("separate_bc_policy", False)
                    )
                )
                else (
                    "bc_only"
                    if args.policy_value_beta is None
                    else (
                        "value_only"
                        if args.policy_value_beta == 0.0
                        else "value_plus_bc"
                    )
                )
            )
        ),
        "num_eval_episodes": int(args.num_eval_episodes),
        "eval_seed_start": int(args.eval_seed_start),
        "eval_seed_end": int(args.eval_seed_start + args.num_eval_episodes - 1),
        "episode_success": float(sum(successes) / len(successes)),
        "episode_results": episode_results,
    }


def main() -> int:
    args = parse_args()
    configure_process(args.gpu_id)
    _, _, output, _ = _resolve_paths(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
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
