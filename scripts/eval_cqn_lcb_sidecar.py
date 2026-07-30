#!/usr/bin/env python3
"""Evaluate frozen BC with conservative counterfactual-value sidecars.

The behavior policy and both image encoders come from ``--snapshot`` and are
never modified.  Each sidecar must be a branch-oracle descendant of that exact
snapshot.  At every evaluation inference, the script:

1. obtains the ordinary temporally-ensembled BC plan;
2. constructs the same level-1, H-step sibling interventions used to collect
   the simulator branch cache;
3. scores all ``action_dim * bins`` candidates in parallel with independent
   advantage heads;
4. applies at most one intervention when behavior support and the ensemble
   lower confidence bound both pass.

If no candidate passes, the returned plan is exactly the BC plan.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


def _non_negative_finite(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--sidecar-snapshot", action="append", type=Path)
    parser.add_argument(
        "--sidecar-seed",
        action="append",
        type=int,
        help=(
            "Initialization seed corresponding to each sidecar. Required for "
            "older artifacts that predate embedded initialization metadata."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--gpu-id", required=True, type=int)
    parser.add_argument("--num-eval-episodes", type=int, default=50)
    parser.add_argument("--eval-seed-start", type=int, default=33_000)
    parser.add_argument("--force-level", type=int, default=1)
    parser.add_argument("--intervention-horizon", type=int, default=4)
    parser.add_argument(
        "--lcb-scale",
        type=_non_negative_finite,
        default=1.0,
    )
    parser.add_argument(
        "--min-lcb-margin",
        type=_non_negative_finite,
        default=0.0,
    )
    parser.add_argument(
        "--max-bc-logprob-drop",
        type=_non_negative_finite,
        default=0.5,
    )
    parser.add_argument(
        "--bc-only",
        action="store_true",
        help="Evaluate the identical frozen behavior policy without sidecars.",
    )
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help=(
            "Score and log sidecar decisions but always execute the exact BC "
            "plan. This is for threshold-rate calibration, not policy quality."
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
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["JAX_CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["MUJOCO_EGL_DEVICE_ID"] = gpu


def _resolve_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, list[Path], Path, Path]:
    run_dir = args.run_dir.expanduser().resolve()
    snapshot = args.snapshot or run_dir / "snapshots" / "latest_snapshot.pkl"
    sidecars = [
        path.expanduser().resolve()
        for path in (args.sidecar_snapshot or [])
    ]
    return (
        run_dir,
        snapshot.expanduser().resolve(),
        sidecars,
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
    if str(cfg.method.get("name", "")).lower() != "cqn_flow":
        raise ValueError(
            f"checkpoint method is not cqn_flow: {cfg.method.get('name')}"
        )
    if str(cfg.method.get("critic_architecture", "")).lower() != (
        "flow_v_direct_a"
    ):
        raise ValueError("LCB sidecars require flow_v_direct_a architecture")
    if not bool(cfg.method.get("separate_bc_policy", False)):
        raise ValueError("LCB sidecars require separate_bc_policy=true")
    if not bool(cfg.method.get("distinct_policy_encoder", False)):
        raise ValueError("LCB sidecars require distinct_policy_encoder=true")
    if not bool(cfg.method.get("freeze_bc_policy", False)):
        raise ValueError("LCB sidecars require freeze_bc_policy=true")

    # The ordinary rollout path must remain BC-only.  The post-act wrapper is
    # solely responsible for any conservative intervention.
    cfg.method.policy_value_beta = None
    cfg.create_train_env = False
    cfg.num_train_envs = 0
    cfg.num_train_frames = 0
    cfg.num_eval_envs = 1
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


def _trees_bitwise_equal(left: Any, right: Any) -> bool:
    import jax

    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    if left_tree != right_tree or len(left_leaves) != len(right_leaves):
        return False
    return all(
        np.array_equal(np.asarray(left_leaf), np.asarray(right_leaf))
        for left_leaf, right_leaf in zip(
            left_leaves,
            right_leaves,
            strict=True,
        )
    )


def _load_sidecars(
    snapshot: Path,
    sidecar_paths: list[Path],
    base_params: Any,
    provided_seeds: list[int] | None = None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    if len(sidecar_paths) < 2:
        raise ValueError("LCB evaluation requires at least two sidecar snapshots")
    if provided_seeds is not None:
        if len(provided_seeds) != len(sidecar_paths):
            raise ValueError(
                "provide exactly one --sidecar-seed per sidecar snapshot"
            )
        if len(set(provided_seeds)) != len(provided_seeds):
            raise ValueError("sidecar initialization seeds must be unique")
    frozen_keys = ("encoder", "policy_encoder", "policy")
    advantage_params = []
    metadata = []
    for sidecar_index, path in enumerate(sidecar_paths):
        if not path.is_file():
            raise FileNotFoundError(f"missing sidecar snapshot: {path}")
        with path.open("rb") as file:
            payload = pickle.load(file)
        branch_metadata = payload.get("branch_oracle_metadata")
        if not isinstance(branch_metadata, dict):
            raise ValueError(f"{path} has no branch_oracle_metadata")
        source = Path(
            str(branch_metadata.get("source_snapshot", ""))
        ).expanduser().resolve()
        if source != snapshot:
            raise ValueError(
                f"{path} descends from {source}, expected {snapshot}"
            )
        state = payload.get("agent", {})
        params = state.get("params", {})
        if "advantage" not in params:
            raise ValueError(f"{path} has no advantage parameters")
        for key in frozen_keys:
            if key not in params or key not in base_params:
                raise ValueError(f"{path} is missing frozen component {key}")
            if not _trees_bitwise_equal(params[key], base_params[key]):
                raise ValueError(f"{path} changed frozen component {key}")
        advantage_params.append(params["advantage"])
        embedded_seed = branch_metadata.get("initialization_seed")
        provided_seed = (
            None
            if provided_seeds is None
            else int(provided_seeds[sidecar_index])
        )
        if (
            embedded_seed is not None
            and provided_seed is not None
            and int(embedded_seed) != provided_seed
        ):
            raise ValueError(
                f"{path} embedded seed {embedded_seed} does not match "
                f"provided seed {provided_seed}"
            )
        metadata.append(
            {
                "path": str(path),
                "initialization_seed": (
                    int(embedded_seed)
                    if embedded_seed is not None
                    else provided_seed
                ),
                "updates": branch_metadata.get("updates"),
                "force_level": branch_metadata.get("force_level"),
                "intervention_horizon": branch_metadata.get(
                    "intervention_horizon"
                ),
                "train_seeds": branch_metadata.get("train_seeds"),
                "heldout_seeds": branch_metadata.get("heldout_seeds"),
            }
        )

    for left_index in range(len(advantage_params)):
        for right_index in range(left_index + 1, len(advantage_params)):
            if _trees_bitwise_equal(
                advantage_params[left_index],
                advantage_params[right_index],
            ):
                raise ValueError(
                    "sidecar advantage heads are bitwise identical; "
                    "independent initialization is required"
                )
    return advantage_params, metadata


def _numeric_metrics(metrics: dict) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }


def _install_lcb_wrapper(
    agent,
    advantage_params: list[Any],
    *,
    force_level: int,
    intervention_horizon: int,
    lcb_scale: float,
    min_lcb_margin: float,
    max_bc_logprob_drop: float,
    diagnostic_only: bool = False,
):
    import jax
    import jax.numpy as jnp

    ensemble_params = jax.tree.map(
        lambda *leaves: jnp.stack(
            [jnp.asarray(leaf) for leaf in leaves],
            axis=0,
        ),
        *advantage_params,
    )
    value_encoder_params = agent.params.get("encoder", None)
    policy_encoder_params = agent.params.get("policy_encoder", None)
    policy_params = agent.params["policy"]

    def select_plan(obs_inputs, baseline_plan):
        value_features = agent._rl_features(
            value_encoder_params,
            obs_inputs,
            stop_gradient=True,
        )
        policy_features = agent._rl_features(
            policy_encoder_params,
            obs_inputs,
            stop_gradient=True,
        )
        return agent._supported_lcb_policy_plan(
            ensemble_params,
            value_features,
            policy_params,
            policy_features,
            baseline_plan,
            force_level=force_level,
            intervention_horizon=intervention_horizon,
            lcb_scale=lcb_scale,
            min_lcb_margin=min_lcb_margin,
            max_bc_logprob_drop=max_bc_logprob_drop,
        )

    select_plan = jax.jit(select_plan)
    original_act = agent.act
    counters = {
        "inference_count": 0,
        "applied_override_count": 0,
        "eligible_dimension_count": 0,
        "selected_lcb_sum": 0.0,
        "selected_lcb_values": [],
        "selected_dimension_histogram": np.zeros(
            agent.action_dim,
            dtype=np.int64,
        ),
    }

    def reset_counters() -> None:
        counters["inference_count"] = 0
        counters["applied_override_count"] = 0
        counters["eligible_dimension_count"] = 0
        counters["selected_lcb_sum"] = 0.0
        counters["selected_lcb_values"].clear()
        counters["selected_dimension_histogram"].fill(0)

    def snapshot_counters() -> dict[str, Any]:
        applied = int(counters["applied_override_count"])
        inference = int(counters["inference_count"])
        selected_lcb_values = np.asarray(
            counters["selected_lcb_values"],
            dtype=np.float64,
        )
        return {
            "inference_count": inference,
            "applied_override_count": applied,
            "override_rate": (
                float(applied / inference) if inference else 0.0
            ),
            "eligible_dimension_count": int(
                counters["eligible_dimension_count"]
            ),
            "mean_selected_lcb_delta": (
                float(counters["selected_lcb_sum"] / applied)
                if applied
                else 0.0
            ),
            "selected_lcb_delta_quantiles": (
                {
                    str(quantile): float(
                        np.quantile(selected_lcb_values, quantile)
                    )
                    for quantile in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
                }
                if selected_lcb_values.size
                else None
            ),
            "selected_dimension_histogram": counters[
                "selected_dimension_histogram"
            ].tolist(),
        }

    def gated_act(observations: dict, step: int, eval_mode: bool):
        baseline_plan = original_act(observations, step, eval_mode)
        if not eval_mode:
            return baseline_plan
        obs_inputs = agent._prepare_rl_obs_inputs(observations)
        result = jax.device_get(select_plan(obs_inputs, baseline_plan))
        applied = np.asarray(result.applied_override, dtype=bool)
        selected_dimension = np.asarray(
            result.selected_dimension,
            dtype=np.int32,
        )
        selected_lcb = np.asarray(
            result.selected_lcb_delta,
            dtype=np.float64,
        )
        counters["inference_count"] += int(applied.size)
        counters["applied_override_count"] += int(applied.sum())
        counters["eligible_dimension_count"] += int(
            np.asarray(result.eligible_override_mask, dtype=bool).sum()
        )
        counters["selected_lcb_sum"] += float(selected_lcb[applied].sum())
        counters["selected_lcb_values"].extend(
            selected_lcb[applied].tolist()
        )
        for dimension in selected_dimension[applied]:
            counters["selected_dimension_histogram"][int(dimension)] += 1
        if diagnostic_only:
            return baseline_plan
        return np.asarray(result.action, dtype=np.float32)

    agent.act = gated_act
    return reset_counters, snapshot_counters


def run_eval(args: argparse.Namespace) -> dict:
    import jax

    from robobase.workspace import Workspace

    run_dir, snapshot, sidecars, _, work_dir = _resolve_paths(args)
    if not snapshot.is_file():
        raise FileNotFoundError(f"missing snapshot: {snapshot}")
    if args.num_eval_episodes < 1:
        raise ValueError("--num-eval-episodes must be at least one")
    if args.bc_only and sidecars:
        raise ValueError("--bc-only cannot be combined with sidecar snapshots")
    if args.bc_only and args.diagnostic_only:
        raise ValueError("--bc-only cannot be combined with --diagnostic-only")
    if not args.bc_only and len(sidecars) < 2:
        raise ValueError("provide at least two --sidecar-snapshot paths")

    work_dir.mkdir(parents=True, exist_ok=True)
    cfg = _prepare_cfg(args, run_dir, work_dir)
    workspace = Workspace(cfg, work_dir=str(work_dir))
    sidecar_metadata: list[dict[str, Any]] = []
    reset_counters = lambda: None
    snapshot_counters = lambda: {
        "inference_count": 0,
        "applied_override_count": 0,
        "override_rate": 0.0,
        "eligible_dimension_count": 0,
        "mean_selected_lcb_delta": 0.0,
        "selected_dimension_histogram": [0] * workspace.agent.action_dim,
    }
    episode_results = []
    try:
        workspace.load_snapshot(snapshot, load_replay_buffer=False)
        if not args.bc_only:
            advantage_params, sidecar_metadata = _load_sidecars(
                snapshot,
                sidecars,
                workspace.agent.params,
                provided_seeds=args.sidecar_seed,
            )
            reset_counters, snapshot_counters = _install_lcb_wrapper(
                workspace.agent,
                advantage_params,
                force_level=int(args.force_level),
                intervention_horizon=int(args.intervention_horizon),
                lcb_scale=float(args.lcb_scale),
                min_lcb_margin=float(args.min_lcb_margin),
                max_bc_logprob_drop=float(args.max_bc_logprob_drop),
                diagnostic_only=bool(args.diagnostic_only),
            )

        for episode in range(args.num_eval_episodes):
            eval_seed = int(args.eval_seed_start) + episode
            workspace.agent.rng_key = jax.random.PRNGKey(
                eval_seed + 910_000
            )
            workspace.cfg.env.eval_seed_start = eval_seed
            reset_counters()
            metrics = _numeric_metrics(workspace.eval())
            if "episode_success" not in metrics:
                raise RuntimeError(
                    f"evaluation seed {eval_seed} has no episode_success"
                )
            override = snapshot_counters()
            episode_results.append(
                {"seed": eval_seed, **metrics, **override}
            )
            if (episode + 1) % 10 == 0 or (
                episode + 1 == args.num_eval_episodes
            ):
                success = np.mean(
                    [
                        result["episode_success"]
                        for result in episode_results
                    ]
                )
                override_rate = np.mean(
                    [result["override_rate"] for result in episode_results]
                )
                print(
                    f"completed {episode + 1}/{args.num_eval_episodes}: "
                    f"success={success:.3f}, "
                    f"mean_episode_override_rate={override_rate:.3f}",
                    flush=True,
                )
    finally:
        workspace.shutdown()

    total_inferences = sum(
        int(result["inference_count"]) for result in episode_results
    )
    total_overrides = sum(
        int(result["applied_override_count"]) for result in episode_results
    )
    success = np.mean(
        [result["episode_success"] for result in episode_results]
    )
    override_episode_results = [
        result
        for result in episode_results
        if int(result["applied_override_count"]) > 0
    ]
    return {
        "status": "ok",
        "task": str(cfg.env.task_name),
        "run_dir": str(run_dir),
        "snapshot": str(snapshot),
        "policy": (
            "bc_only"
            if args.bc_only
            else (
                "supported_lcb_diagnostic_only"
                if args.diagnostic_only
                else "supported_lcb_sidecar"
            )
        ),
        "sidecars": sidecar_metadata,
        "thresholds": (
            None
            if args.bc_only
            else {
                "force_level": int(args.force_level),
                "intervention_horizon": int(args.intervention_horizon),
                "lcb_scale": float(args.lcb_scale),
                "min_lcb_margin": float(args.min_lcb_margin),
                "max_bc_logprob_drop": float(
                    args.max_bc_logprob_drop
                ),
                "max_plan_overrides_per_inference": 1,
                "diagnostic_only": bool(args.diagnostic_only),
            }
        ),
        "num_eval_episodes": int(args.num_eval_episodes),
        "eval_seed_start": int(args.eval_seed_start),
        "eval_seed_end": int(
            args.eval_seed_start + args.num_eval_episodes - 1
        ),
        "episode_success": float(success),
        "total_inferences": total_inferences,
        "total_applied_overrides": total_overrides,
        "override_rate": (
            float(total_overrides / total_inferences)
            if total_inferences
            else 0.0
        ),
        "num_episodes_with_override": len(override_episode_results),
        "success_given_episode_with_override": (
            float(
                np.mean(
                    [
                        result["episode_success"]
                        for result in override_episode_results
                    ]
                )
            )
            if override_episode_results
            else None
        ),
        "episode_results": episode_results,
    }


def main() -> int:
    args = parse_args()
    configure_process(args.gpu_id)
    _, _, _, output, _ = _resolve_paths(args)
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
