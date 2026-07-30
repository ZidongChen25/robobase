#!/usr/bin/env python3
"""Evaluate clean CQN-AS with conservative FLOQ-value interventions.

The clean checkpoint remains the action-producing baseline.  A separately
trained expected-value FLOQ checkpoint may replace at most one action
dimension of that plan.  An intervention is allowed only when its distilled
Q margin, behavior-policy support, and integrated-flow source agreement all
pass fixed thresholds.  Otherwise the executed action is exactly the clean
CQN-AS action.
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
from typing import Any, NamedTuple

import numpy as np


class FlowFallbackPlanResult(NamedTuple):
    action: Any
    candidate_indices: Any
    baseline_indices: Any
    eligible_override_mask: Any
    applied_override: Any
    selected_dimension: Any
    selected_value_margin: Any
    selected_source_mean_delta: Any
    selected_source_win_fraction: Any


def _finite_nonnegative(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise argparse.ArgumentTypeError("expected a finite non-negative value")
    return number


def _unit_interval(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("expected a number in [0, 1]")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-run-dir", required=True, type=Path)
    parser.add_argument("--clean-snapshot", required=True, type=Path)
    parser.add_argument("--flow-run-dir", type=Path)
    parser.add_argument("--flow-snapshot", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--gpu-id", required=True, type=int)
    parser.add_argument("--num-eval-episodes", type=int, default=50)
    parser.add_argument("--eval-seed-start", type=int, default=60_000)
    parser.add_argument("--force-level", type=int, default=1)
    parser.add_argument("--intervention-horizon", type=int, default=4)
    parser.add_argument("--policy-value-beta", type=_finite_nonnegative, default=1.0)
    parser.add_argument("--min-value-margin", type=_finite_nonnegative, default=0.75)
    parser.add_argument(
        "--max-bc-logprob-drop",
        type=_finite_nonnegative,
        default=0.5,
    )
    parser.add_argument(
        "--max-best-bc-logprob-drop",
        type=_finite_nonnegative,
        default=0.5,
    )
    parser.add_argument(
        "--min-source-win-fraction",
        type=_unit_interval,
        default=0.625,
    )
    parser.add_argument(
        "--min-source-mean-delta",
        type=float,
        default=0.0,
    )
    parser.add_argument("--clean-only", action="store_true")
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="Log proposed interventions but execute the clean action.",
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


def select_flow_fallback_plan(
    baseline_plan,
    candidate_plans,
    baseline_indices,
    distilled_q,
    policy_logits,
    flow_q_samples,
    *,
    policy_value_beta: float,
    min_value_margin: float,
    max_bc_logprob_drop: float,
    max_best_bc_logprob_drop: float,
    min_source_win_fraction: float,
    min_source_mean_delta: float,
) -> FlowFallbackPlanResult:
    """Apply at most one supported, source-consistent FLOQ intervention."""

    import jax
    import jax.numpy as jnp

    baseline_plan = jnp.asarray(baseline_plan, dtype=jnp.float32)
    candidate_plans = jnp.asarray(candidate_plans, dtype=jnp.float32)
    baseline_indices = jnp.asarray(baseline_indices, dtype=jnp.int32)
    distilled_q = jnp.asarray(distilled_q, dtype=jnp.float32)
    policy_logits = jnp.asarray(policy_logits, dtype=jnp.float32)
    flow_q_samples = jnp.asarray(flow_q_samples, dtype=jnp.float32)
    if baseline_plan.ndim != 3:
        raise ValueError("baseline_plan must have shape [B, K, D]")
    batch_size, action_sequence, action_dim = baseline_plan.shape
    if distilled_q.ndim != 3 or distilled_q.shape[:2] != (
        batch_size,
        action_dim,
    ):
        raise ValueError("distilled_q must have shape [B, D, bins]")
    bins = distilled_q.shape[-1]
    if candidate_plans.shape != (
        batch_size,
        action_dim,
        bins,
        action_sequence,
        action_dim,
    ):
        raise ValueError(
            "candidate_plans must have shape [B, D, bins, K, D]"
        )
    if baseline_indices.shape != (batch_size, action_dim):
        raise ValueError("baseline_indices must have shape [B, D]")
    if policy_logits.shape != distilled_q.shape:
        raise ValueError("policy_logits shape must match distilled_q")
    if (
        flow_q_samples.ndim != 4
        or flow_q_samples.shape[0] != batch_size
        or flow_q_samples.shape[2:] != (action_dim, bins)
    ):
        raise ValueError("flow_q_samples must have shape [B, R, D, bins]")

    centered_q = distilled_q - distilled_q.mean(axis=-1, keepdims=True)
    q_scale = jnp.sqrt(
        jnp.mean(jnp.square(centered_q), axis=-1, keepdims=True) + 1e-6
    )
    normalized_q = centered_q / q_scale
    log_policy = jax.nn.log_softmax(policy_logits, axis=-1)
    combined_score = normalized_q + float(policy_value_beta) * log_policy
    candidate_indices = jnp.argmax(combined_score, axis=-1)

    baseline_q = jnp.take_along_axis(
        normalized_q,
        baseline_indices[..., None],
        axis=-1,
    )[..., 0]
    candidate_q = jnp.take_along_axis(
        normalized_q,
        candidate_indices[..., None],
        axis=-1,
    )[..., 0]
    value_margin = candidate_q - baseline_q

    baseline_logp = jnp.take_along_axis(
        log_policy,
        baseline_indices[..., None],
        axis=-1,
    )[..., 0]
    candidate_logp = jnp.take_along_axis(
        log_policy,
        candidate_indices[..., None],
        axis=-1,
    )[..., 0]
    best_logp = log_policy.max(axis=-1)

    sample_candidate_index = jnp.broadcast_to(
        candidate_indices[:, None, :, None],
        (
            batch_size,
            flow_q_samples.shape[1],
            action_dim,
            1,
        ),
    )
    sample_baseline_index = jnp.broadcast_to(
        baseline_indices[:, None, :, None],
        sample_candidate_index.shape,
    )
    candidate_samples = jnp.take_along_axis(
        flow_q_samples,
        sample_candidate_index,
        axis=-1,
    )[..., 0]
    baseline_samples = jnp.take_along_axis(
        flow_q_samples,
        sample_baseline_index,
        axis=-1,
    )[..., 0]
    source_delta = candidate_samples - baseline_samples
    source_mean_delta = source_delta.mean(axis=1)
    source_win_fraction = (
        (source_delta > 0.0).astype(jnp.float32)
        + 0.5 * (source_delta == 0.0).astype(jnp.float32)
    ).mean(axis=1)

    eligible = (
        (candidate_indices != baseline_indices)
        & (value_margin >= float(min_value_margin))
        & (
            candidate_logp
            >= baseline_logp - float(max_bc_logprob_drop)
        )
        & (
            candidate_logp
            >= best_logp - float(max_best_bc_logprob_drop)
        )
        & (source_mean_delta >= float(min_source_mean_delta))
        & (
            source_win_fraction >= float(min_source_win_fraction)
        )
    )
    eligible_margin = jnp.where(eligible, value_margin, -jnp.inf)
    selected_dimension = jnp.argmax(eligible_margin, axis=-1)
    applied = jnp.any(eligible, axis=-1)
    batch_index = jnp.arange(batch_size)
    selected_bin = candidate_indices[batch_index, selected_dimension]
    selected_plan = candidate_plans[
        batch_index,
        selected_dimension,
        selected_bin,
    ]
    action = jnp.where(
        applied[:, None, None],
        selected_plan,
        baseline_plan,
    )

    def selected_or_zero(values):
        selected = values[batch_index, selected_dimension]
        return jnp.where(applied, selected, 0.0)

    return FlowFallbackPlanResult(
        action=action,
        candidate_indices=candidate_indices,
        baseline_indices=baseline_indices,
        eligible_override_mask=eligible,
        applied_override=applied,
        selected_dimension=jnp.where(applied, selected_dimension, -1),
        selected_value_margin=selected_or_zero(value_margin),
        selected_source_mean_delta=selected_or_zero(source_mean_delta),
        selected_source_win_fraction=selected_or_zero(
            source_win_fraction
        ),
    )


def _prepare_cfg(
    run_dir: Path,
    work_dir: Path,
    *,
    eval_env: bool,
    eval_seed_start: int,
):
    from omegaconf import OmegaConf

    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_struct(cfg, False)
    cfg.create_train_env = False
    cfg.num_train_envs = 0
    cfg.num_train_frames = 0
    cfg.num_eval_envs = 1 if eval_env else 0
    cfg.num_eval_episodes = 1 if eval_env else 0
    cfg.env.eval_seed_start = int(eval_seed_start)
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
    cfg.replay.persist = False
    cfg.replay.reuse_saved = False
    cfg.backend.replay_prefetch_size = 0
    cfg.backend.replay_device_prefetch = False
    cfg.backend.fused_update_steps = 1
    cfg.backend.update_block_every_steps = 1
    cfg.hydra = {"run": {"dir": str(work_dir)}}
    if str(cfg.method.get("name", "")).lower() == "cqn_flow":
        cfg.method.policy_value_beta = None
        cfg.method.flow_distill_action_readout = False
        cfg.method.flow_q_action_readout = False
    OmegaConf.resolve(cfg)
    return cfg


def _numeric_metrics(metrics: dict) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }


def _validate_pair(clean_agent, flow_agent) -> None:
    if flow_agent.value_mode != "scalar":
        raise ValueError("fallback requires expected-value scalar FLOQ")
    if float(flow_agent.flow_distill_lambda) <= 0.0:
        raise ValueError("flow checkpoint has no distilled scalar readout")
    if not bool(flow_agent.separate_bc_policy):
        raise ValueError("flow checkpoint has no independent BC policy")
    for name in ("levels", "bins", "action_sequence", "action_dim"):
        if int(getattr(clean_agent, name)) != int(getattr(flow_agent, name)):
            raise ValueError(f"clean/flow {name} mismatch")
    np.testing.assert_allclose(
        np.asarray(clean_agent._step_action_low),
        np.asarray(flow_agent._step_action_low),
    )
    np.testing.assert_allclose(
        np.asarray(clean_agent._step_action_high),
        np.asarray(flow_agent._step_action_high),
    )


def _install_flow_wrapper(
    clean_agent,
    flow_agent,
    *,
    force_level: int,
    intervention_horizon: int,
    policy_value_beta: float,
    min_value_margin: float,
    max_bc_logprob_drop: float,
    max_best_bc_logprob_drop: float,
    min_source_win_fraction: float,
    min_source_mean_delta: float,
    diagnostic_only: bool,
):
    import jax
    import jax.numpy as jnp

    from robobase.method.cqn_flow import sibling_bin_candidate_plans

    _validate_pair(clean_agent, flow_agent)
    value_encoder_params = flow_agent.params.get("encoder", None)
    policy_encoder_params = flow_agent.params.get("encoder", None)
    if flow_agent.distinct_policy_encoder:
        policy_encoder_params = flow_agent.params.get(
            "policy_encoder",
            None,
        )
    critic_params = flow_agent.params["critic"]
    readout_params = flow_agent.params["flow_distill_readout"]
    policy_params = flow_agent.params["policy"]

    def select_plan(obs_inputs, baseline_plan, environment_step):
        value_features = flow_agent._rl_features(
            value_encoder_params,
            obs_inputs,
            stop_gradient=True,
        )
        policy_features = flow_agent._rl_features(
            policy_encoder_params,
            obs_inputs,
            stop_gradient=True,
        )
        candidate_plans, _ = sibling_bin_candidate_plans(
            baseline_plan,
            jnp.asarray(flow_agent._step_action_low),
            jnp.asarray(flow_agent._step_action_high),
            bins=flow_agent.bins,
            force_level=force_level,
            intervention_horizon=intervention_horizon,
        )
        _, all_distilled_q = flow_agent._flow_distill_outputs_per_level(
            readout_params,
            value_features,
            baseline_plan,
        )
        distilled_q = all_distilled_q[:, force_level, 0, :, :]

        policy_scores, encoded_bins = (
            flow_agent._policy_logits_per_level(
                policy_params,
                policy_features,
                baseline_plan,
            )
        )
        policy_scores = policy_scores.reshape(
            (
                baseline_plan.shape[0],
                flow_agent.levels,
                flow_agent.action_sequence,
                flow_agent.action_dim,
                flow_agent.bins,
            )
        )
        policy_logits = policy_scores[:, force_level, 0, :, :]
        baseline_indices = encoded_bins.reshape(
            (
                baseline_plan.shape[0],
                flow_agent.levels,
                flow_agent.action_sequence,
                flow_agent.action_dim,
            )
        )[:, force_level, 0, :]

        source_key = jax.random.fold_in(
            jax.random.PRNGKey(flow_agent._seed + 81_337),
            environment_step,
        )
        _, all_endpoints = flow_agent._endpoints_per_level(
            critic_params,
            value_features,
            baseline_plan,
            source_key,
        )
        flow_q_samples = flow_agent._endpoint_q_samples(
            all_endpoints
        )[:, :, force_level, 0, :, :]
        return select_flow_fallback_plan(
            baseline_plan,
            candidate_plans,
            baseline_indices,
            distilled_q,
            policy_logits,
            flow_q_samples,
            policy_value_beta=policy_value_beta,
            min_value_margin=min_value_margin,
            max_bc_logprob_drop=max_bc_logprob_drop,
            max_best_bc_logprob_drop=max_best_bc_logprob_drop,
            min_source_win_fraction=min_source_win_fraction,
            min_source_mean_delta=min_source_mean_delta,
        )

    select_plan = jax.jit(select_plan)
    original_act = clean_agent.act
    counters: dict[str, Any] = {
        "inference_count": 0,
        "applied_override_count": 0,
        "eligible_dimension_count": 0,
        "selected_margins": [],
        "selected_source_mean_deltas": [],
        "selected_source_win_fractions": [],
        "selected_dimension_histogram": np.zeros(
            clean_agent.action_dim,
            dtype=np.int64,
        ),
    }

    def reset_counters() -> None:
        counters["inference_count"] = 0
        counters["applied_override_count"] = 0
        counters["eligible_dimension_count"] = 0
        counters["selected_margins"].clear()
        counters["selected_source_mean_deltas"].clear()
        counters["selected_source_win_fractions"].clear()
        counters["selected_dimension_histogram"].fill(0)

    def snapshot_counters() -> dict[str, Any]:
        inference = int(counters["inference_count"])
        applied = int(counters["applied_override_count"])

        def mean_or_zero(key: str) -> float:
            values = counters[key]
            return float(np.mean(values)) if values else 0.0

        return {
            "inference_count": inference,
            "applied_override_count": applied,
            "override_rate": float(applied / inference) if inference else 0.0,
            "eligible_dimension_count": int(
                counters["eligible_dimension_count"]
            ),
            "mean_selected_value_margin": mean_or_zero(
                "selected_margins"
            ),
            "mean_selected_source_mean_delta": mean_or_zero(
                "selected_source_mean_deltas"
            ),
            "mean_selected_source_win_fraction": mean_or_zero(
                "selected_source_win_fractions"
            ),
            # Keep the proposal-level values so a diagnostic-only split can
            # freeze sparse thresholds by coverage without consulting task
            # outcomes.  At most one value is appended per inference.
            "selected_value_margins": list(counters["selected_margins"]),
            "selected_source_mean_deltas": list(
                counters["selected_source_mean_deltas"]
            ),
            "selected_source_win_fractions": list(
                counters["selected_source_win_fractions"]
            ),
            "selected_dimension_histogram": counters[
                "selected_dimension_histogram"
            ].tolist(),
        }

    def gated_act(observations: dict, step: int, eval_mode: bool):
        baseline_plan = original_act(observations, step, eval_mode)
        if not eval_mode:
            return baseline_plan
        flow_obs_inputs = flow_agent._prepare_rl_obs_inputs(observations)
        result = jax.device_get(
            select_plan(
                flow_obs_inputs,
                baseline_plan,
                jnp.asarray(step, dtype=jnp.int32),
            )
        )
        applied = np.asarray(result.applied_override, dtype=bool)
        selected_dimension = np.asarray(
            result.selected_dimension,
            dtype=np.int32,
        )
        selected_margin = np.asarray(
            result.selected_value_margin,
            dtype=np.float64,
        )
        selected_source_delta = np.asarray(
            result.selected_source_mean_delta,
            dtype=np.float64,
        )
        selected_source_win = np.asarray(
            result.selected_source_win_fraction,
            dtype=np.float64,
        )
        counters["inference_count"] += int(applied.size)
        counters["applied_override_count"] += int(applied.sum())
        counters["eligible_dimension_count"] += int(
            np.asarray(result.eligible_override_mask, dtype=bool).sum()
        )
        counters["selected_margins"].extend(
            selected_margin[applied].tolist()
        )
        counters["selected_source_mean_deltas"].extend(
            selected_source_delta[applied].tolist()
        )
        counters["selected_source_win_fractions"].extend(
            selected_source_win[applied].tolist()
        )
        for dimension in selected_dimension[applied]:
            counters["selected_dimension_histogram"][int(dimension)] += 1
        if diagnostic_only:
            return baseline_plan
        return np.asarray(result.action, dtype=np.float32)

    clean_agent.act = gated_act
    return reset_counters, snapshot_counters


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    import jax

    from robobase.workspace import Workspace

    clean_run = args.clean_run_dir.expanduser().resolve()
    clean_snapshot = args.clean_snapshot.expanduser().resolve()
    output = args.output.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    if not clean_snapshot.is_file():
        raise FileNotFoundError(clean_snapshot)
    if args.num_eval_episodes < 1:
        raise ValueError("num-eval-episodes must be positive")
    if args.force_level < 0 or args.intervention_horizon < 1:
        raise ValueError("invalid force-level/intervention-horizon")
    if not math.isfinite(args.min_source_mean_delta):
        raise ValueError("min-source-mean-delta must be finite")
    if args.clean_only and args.diagnostic_only:
        raise ValueError("clean-only cannot be diagnostic-only")
    if not args.clean_only and (
        args.flow_run_dir is None or args.flow_snapshot is None
    ):
        raise ValueError(
            "flow-run-dir and flow-snapshot are required for interventions"
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    clean_cfg = _prepare_cfg(
        clean_run,
        work_dir / "clean",
        eval_env=True,
        eval_seed_start=int(args.eval_seed_start),
    )
    if str(clean_cfg.method.get("name", "")).lower() != "cqn_as":
        raise ValueError("clean checkpoint must use method=cqn_as")
    clean_workspace = Workspace(
        clean_cfg,
        work_dir=str(work_dir / "clean"),
    )
    flow_workspace = None
    reset_counters = lambda: None
    snapshot_counters = lambda: {
        "inference_count": 0,
        "applied_override_count": 0,
        "override_rate": 0.0,
        "eligible_dimension_count": 0,
        "mean_selected_value_margin": 0.0,
        "mean_selected_source_mean_delta": 0.0,
        "mean_selected_source_win_fraction": 0.0,
        "selected_value_margins": [],
        "selected_source_mean_deltas": [],
        "selected_source_win_fractions": [],
        "selected_dimension_histogram": [0]
        * clean_workspace.agent.action_dim,
    }
    episode_results = []
    flow_run = None
    flow_snapshot = None
    try:
        clean_workspace.load_snapshot(
            clean_snapshot,
            load_replay_buffer=False,
        )
        if not args.clean_only:
            assert args.flow_run_dir is not None
            assert args.flow_snapshot is not None
            flow_run = args.flow_run_dir.expanduser().resolve()
            flow_snapshot = args.flow_snapshot.expanduser().resolve()
            if not flow_snapshot.is_file():
                raise FileNotFoundError(flow_snapshot)
            flow_cfg = _prepare_cfg(
                flow_run,
                work_dir / "flow",
                eval_env=False,
                eval_seed_start=int(args.eval_seed_start),
            )
            if str(flow_cfg.method.get("name", "")).lower() != "cqn_flow":
                raise ValueError("flow checkpoint must use method=cqn_flow")
            if str(clean_cfg.env.task_name) != str(flow_cfg.env.task_name):
                raise ValueError("clean/flow task mismatch")
            flow_workspace = Workspace(
                flow_cfg,
                work_dir=str(work_dir / "flow"),
            )
            flow_workspace.load_snapshot(
                flow_snapshot,
                load_replay_buffer=False,
            )
            reset_counters, snapshot_counters = _install_flow_wrapper(
                clean_workspace.agent,
                flow_workspace.agent,
                force_level=int(args.force_level),
                intervention_horizon=int(args.intervention_horizon),
                policy_value_beta=float(args.policy_value_beta),
                min_value_margin=float(args.min_value_margin),
                max_bc_logprob_drop=float(args.max_bc_logprob_drop),
                max_best_bc_logprob_drop=float(
                    args.max_best_bc_logprob_drop
                ),
                min_source_win_fraction=float(
                    args.min_source_win_fraction
                ),
                min_source_mean_delta=float(args.min_source_mean_delta),
                diagnostic_only=bool(args.diagnostic_only),
            )

        for episode in range(int(args.num_eval_episodes)):
            eval_seed = int(args.eval_seed_start) + episode
            clean_workspace.agent.rng_key = jax.random.PRNGKey(
                eval_seed + 940_000
            )
            if flow_workspace is not None:
                flow_workspace.agent.rng_key = jax.random.PRNGKey(
                    eval_seed + 950_000
                )
            clean_workspace.cfg.env.eval_seed_start = eval_seed
            reset_counters()
            metrics = _numeric_metrics(clean_workspace.eval())
            if "episode_success" not in metrics:
                raise RuntimeError("evaluation emitted no episode_success")
            episode_results.append(
                {
                    "seed": eval_seed,
                    **metrics,
                    **snapshot_counters(),
                }
            )
            if (episode + 1) % 10 == 0 or (
                episode + 1 == int(args.num_eval_episodes)
            ):
                print(
                    f"completed {episode + 1}/{args.num_eval_episodes}: "
                    f"success={np.mean([row['episode_success'] for row in episode_results]):.3f}",
                    flush=True,
                )
    finally:
        if flow_workspace is not None:
            flow_workspace.shutdown()
        clean_workspace.shutdown()

    total_inferences = sum(
        int(row["inference_count"]) for row in episode_results
    )
    total_overrides = sum(
        int(row["applied_override_count"]) for row in episode_results
    )
    return {
        "status": "ok",
        "task": str(clean_cfg.env.task_name),
        "clean_run_dir": str(clean_run),
        "clean_snapshot": str(clean_snapshot),
        "flow_run_dir": None if flow_run is None else str(flow_run),
        "flow_snapshot": (
            None if flow_snapshot is None else str(flow_snapshot)
        ),
        "policy": (
            "clean_cqn_as"
            if args.clean_only
            else (
                "floq_clean_fallback_diagnostic"
                if args.diagnostic_only
                else "floq_clean_fallback"
            )
        ),
        "thresholds": (
            None
            if args.clean_only
            else {
                "force_level": int(args.force_level),
                "intervention_horizon": int(args.intervention_horizon),
                "policy_value_beta": float(args.policy_value_beta),
                "min_value_margin": float(args.min_value_margin),
                "max_bc_logprob_drop": float(args.max_bc_logprob_drop),
                "max_best_bc_logprob_drop": float(
                    args.max_best_bc_logprob_drop
                ),
                "min_source_win_fraction": float(
                    args.min_source_win_fraction
                ),
                "min_source_mean_delta": float(
                    args.min_source_mean_delta
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
        "episode_success": float(
            np.mean([row["episode_success"] for row in episode_results])
        ),
        "total_inferences": total_inferences,
        "total_applied_overrides": total_overrides,
        "override_rate": (
            float(total_overrides / total_inferences)
            if total_inferences
            else 0.0
        ),
        "num_episodes_with_override": int(
            sum(
                int(row["applied_override_count"]) > 0
                for row in episode_results
            )
        ),
        "episode_results": episode_results,
    }


def main() -> int:
    args = parse_args()
    configure_process(args.gpu_id)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        payload = run_eval(args)
        payload["elapsed_seconds"] = time.time() - started
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        payload = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
        }
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(payload["traceback"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
