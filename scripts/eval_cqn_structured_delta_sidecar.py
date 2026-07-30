#!/usr/bin/env python3
"""Evaluate frozen CQN behavior with a structured local-delta value sidecar.

The behavior policy and all checkpoint parameters remain frozen.  The sidecar
predicts an optimal local delta for every action dimension from the frozen
state feature.  It may replace at most one dimension of the ordinary BC plan,
and only when the candidate remains inside a configured BC-log-probability
support set, improves structured value by a minimum margin, and the PCA state
embedding is inside a configured radius.  Otherwise the executed plan is
bitwise the original BC plan.
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

try:
    from eval_cqn_lcb_sidecar import (
        _numeric_metrics,
        _prepare_cfg,
        configure_process,
    )
except ImportError:
    from scripts.eval_cqn_lcb_sidecar import (
        _numeric_metrics,
        _prepare_cfg,
        configure_process,
    )


class StructuredDeltaPlanResult(NamedTuple):
    action: Any
    candidate_indices: Any
    bc_indices: Any
    eligible_override_mask: Any
    applied_override: Any
    selected_dimension: Any
    selected_value_margin: Any
    state_rms: Any


def _finite_nonnegative(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise argparse.ArgumentTypeError("expected a finite non-negative value")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-summary", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--gpu-id", required=True, type=int)
    parser.add_argument("--num-eval-episodes", type=int, default=50)
    parser.add_argument("--eval-seed-start", type=int, default=58_000)
    parser.add_argument("--force-level", type=int, default=1)
    parser.add_argument("--intervention-horizon", type=int, default=4)
    parser.add_argument(
        "--min-value-margin",
        type=_finite_nonnegative,
        default=0.01,
    )
    parser.add_argument(
        "--max-bc-logprob-drop",
        type=_finite_nonnegative,
        default=0.5,
    )
    parser.add_argument(
        "--max-state-rms",
        type=_finite_nonnegative,
        default=3.0,
    )
    parser.add_argument("--bc-only", action="store_true")
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="Log decisions but always execute the exact BC plan.",
    )
    return parser.parse_args()


def select_structured_delta_plan(
    baseline_plan,
    candidate_plans,
    candidate_deltas,
    policy_candidate_scores,
    predicted_optimal_delta,
    state_rms,
    *,
    reliability_mask=None,
    min_value_margin: float,
    max_bc_logprob_drop: float,
    max_state_rms: float,
) -> StructuredDeltaPlanResult:
    """Select at most one supported structured-value intervention."""

    import jax.numpy as jnp

    baseline_plan = jnp.asarray(baseline_plan, dtype=jnp.float32)
    candidate_plans = jnp.asarray(candidate_plans, dtype=jnp.float32)
    candidate_deltas = jnp.asarray(candidate_deltas, dtype=jnp.float32)
    policy_candidate_scores = jnp.asarray(
        policy_candidate_scores,
        dtype=jnp.float32,
    )
    predicted_optimal_delta = jnp.asarray(
        predicted_optimal_delta,
        dtype=jnp.float32,
    )
    state_rms = jnp.asarray(state_rms, dtype=jnp.float32)
    expected_scores = (
        baseline_plan.shape[0],
        baseline_plan.shape[-1],
        candidate_plans.shape[2],
    )
    if candidate_deltas.shape != expected_scores:
        raise ValueError("candidate_deltas must have shape [B, D, bins]")
    if policy_candidate_scores.shape != expected_scores:
        raise ValueError(
            "policy_candidate_scores must have shape [B, D, bins]"
        )
    if predicted_optimal_delta.shape != expected_scores[:2]:
        raise ValueError("predicted_optimal_delta must have shape [B, D]")
    if state_rms.shape != (baseline_plan.shape[0],):
        raise ValueError("state_rms must have shape [B]")

    value_scores = -jnp.abs(
        candidate_deltas - predicted_optimal_delta[..., None]
    )
    bc_indices = jnp.argmin(jnp.abs(candidate_deltas), axis=-1)
    best_policy_score = jnp.max(
        policy_candidate_scores,
        axis=-1,
        keepdims=True,
    )
    support_mask = (
        policy_candidate_scores
        >= best_policy_score - float(max_bc_logprob_drop)
    )
    supported_value = jnp.where(support_mask, value_scores, -jnp.inf)
    candidate_indices = jnp.argmax(supported_value, axis=-1)
    candidate_value = jnp.take_along_axis(
        value_scores,
        candidate_indices[..., None],
        axis=-1,
    )[..., 0]
    bc_value = jnp.take_along_axis(
        value_scores,
        bc_indices[..., None],
        axis=-1,
    )[..., 0]
    value_margin = candidate_value - bc_value
    eligible = (
        (candidate_indices != bc_indices)
        & (value_margin >= float(min_value_margin))
        & (state_rms[:, None] <= float(max_state_rms))
    )
    if reliability_mask is not None:
        reliability_mask = jnp.asarray(reliability_mask, dtype=bool)
        if reliability_mask.shape != expected_scores[:2]:
            raise ValueError("reliability_mask must have shape [B, D]")
        eligible = eligible & reliability_mask
    eligible_margin = jnp.where(eligible, value_margin, -jnp.inf)
    selected_dimension = jnp.argmax(eligible_margin, axis=-1)
    applied = jnp.any(eligible, axis=-1)
    batch_index = jnp.arange(baseline_plan.shape[0])
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
    selected_margin = jnp.where(
        applied,
        eligible_margin[batch_index, selected_dimension],
        0.0,
    )
    selected_dimension = jnp.where(applied, selected_dimension, -1)
    return StructuredDeltaPlanResult(
        action=action,
        candidate_indices=candidate_indices,
        bc_indices=bc_indices,
        eligible_override_mask=eligible,
        applied_override=applied,
        selected_dimension=selected_dimension,
        selected_value_margin=selected_margin,
        state_rms=state_rms,
    )


def _resolve_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path | None, Path | None, Path, Path]:
    run_dir = args.run_dir.expanduser().resolve()
    snapshot = (
        args.snapshot.expanduser().resolve()
        if args.snapshot is not None
        else (run_dir / "snapshots" / "latest_snapshot.pkl").resolve()
    )
    model = (
        None if args.model is None else args.model.expanduser().resolve()
    )
    model_summary = (
        None
        if args.model_summary is None
        else args.model_summary.expanduser().resolve()
    )
    return (
        run_dir,
        snapshot,
        model,
        model_summary,
        args.output.expanduser().resolve(),
        args.work_dir.expanduser().resolve(),
    )


def _load_verified_model(
    model_path: Path,
    summary_path: Path,
    snapshot: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "ok":
        raise ValueError("structured-delta model summary is not successful")
    embedded_model = Path(summary["model"]["path"]).expanduser().resolve()
    if embedded_model != model_path:
        raise ValueError(
            f"summary model path {embedded_model} differs from {model_path}"
        )
    fit_cache = Path(summary["fit_cache"]).expanduser().resolve()
    with np.load(fit_cache, allow_pickle=False) as cache:
        if "cache_metadata" not in cache:
            raise ValueError("fit cache has no source-snapshot metadata")
        cache_metadata = json.loads(
            str(np.asarray(cache["cache_metadata"]).item())
        )
    source_snapshot = Path(
        cache_metadata["source_snapshot"]
    ).expanduser().resolve()
    if source_snapshot != snapshot:
        raise ValueError(
            f"model features descend from {source_snapshot}, expected {snapshot}"
        )
    with np.load(model_path, allow_pickle=False) as model_file:
        required = (
            "state_mean",
            "state_components",
            "state_scale",
            "ridge_weights",
            "anchor_steps",
            "action_dim_count",
        )
        missing = [key for key in required if key not in model_file]
        if missing:
            raise KeyError(f"structured model is missing {missing}")
        model = {key: np.asarray(model_file[key]) for key in required}
        for optional in (
            "supported_anchor_steps",
            "supported_action_dimensions",
        ):
            if optional in model_file:
                model[optional] = np.asarray(model_file[optional])
    supported_anchor_steps = np.asarray(
        model.get("supported_anchor_steps", model["anchor_steps"]),
        np.int32,
    )
    supported_action_dimensions = np.asarray(
        model.get(
            "supported_action_dimensions",
            np.arange(int(np.asarray(model["action_dim_count"]).item())),
        ),
        np.int32,
    )
    return model, {
        "path": str(model_path),
        "summary": str(summary_path),
        "fit_cache": str(fit_cache),
        "source_snapshot": str(source_snapshot),
        "pca_components": int(model["state_components"].shape[0]),
        "parameter_count": int(model["ridge_weights"].size),
        "supported_anchor_steps": supported_anchor_steps.tolist(),
        "supported_action_dimensions": (
            supported_action_dimensions.tolist()
        ),
    }


def _install_structured_wrapper(
    agent,
    model: dict[str, np.ndarray],
    *,
    force_level: int,
    intervention_horizon: int,
    min_value_margin: float,
    max_bc_logprob_drop: float,
    max_state_rms: float,
    diagnostic_only: bool,
):
    import jax
    import jax.numpy as jnp

    from robobase.method.cqn_flow import sibling_bin_candidate_plans

    state_mean = jnp.asarray(model["state_mean"], dtype=jnp.float32)
    components = jnp.asarray(
        model["state_components"],
        dtype=jnp.float32,
    )
    state_scale = jnp.asarray(model["state_scale"], dtype=jnp.float32)
    weights = jnp.asarray(model["ridge_weights"], dtype=jnp.float32)
    anchor_steps = jnp.asarray(model["anchor_steps"], dtype=jnp.float32)
    action_dim_count = int(np.asarray(model["action_dim_count"]).item())
    if action_dim_count != agent.action_dim:
        raise ValueError(
            f"model action_dim={action_dim_count}, agent={agent.action_dim}"
        )
    rank = int(components.shape[0])
    anchor_count = int(anchor_steps.shape[0])
    expected_weights = 1 + action_dim_count + anchor_count + (
        rank * action_dim_count
    )
    if weights.size != expected_weights:
        raise ValueError(
            f"ridge weight count {weights.size}, expected {expected_weights}"
        )
    supported_anchor_steps = np.asarray(
        model.get("supported_anchor_steps", model["anchor_steps"]),
        np.int32,
    )
    supported_action_dimensions = np.asarray(
        model.get(
            "supported_action_dimensions",
            np.arange(action_dim_count),
        ),
        np.int32,
    )
    if (
        supported_anchor_steps.ndim != 1
        or supported_action_dimensions.ndim != 1
        or not supported_anchor_steps.size
        or not supported_action_dimensions.size
    ):
        raise ValueError("structured reliability support must be non-empty")
    if not set(supported_anchor_steps.tolist()).issubset(
        set(np.asarray(model["anchor_steps"], np.int32).tolist())
    ):
        raise ValueError("supported anchors are not present in the model")
    if np.any(supported_action_dimensions < 0) or np.any(
        supported_action_dimensions >= action_dim_count
    ):
        raise ValueError("supported action dimension is out of range")
    supported_anchor_mask = jnp.asarray(
        np.isin(
            np.asarray(model["anchor_steps"], np.int32),
            supported_anchor_steps,
        ),
        dtype=bool,
    )
    supported_dimension_mask = jnp.asarray(
        np.isin(
            np.arange(action_dim_count, dtype=np.int32),
            supported_action_dimensions,
        ),
        dtype=bool,
    )
    intercept = weights[0]
    dimension_bias = weights[1 : 1 + action_dim_count]
    anchor_bias = weights[
        1 + action_dim_count : 1 + action_dim_count + anchor_count
    ]
    interaction = weights[
        1 + action_dim_count + anchor_count :
    ].reshape((rank, action_dim_count))
    value_encoder_params = agent.params.get("encoder", None)
    policy_encoder_params = agent.params.get("policy_encoder", None)
    policy_params = agent.params["policy"]

    def select_plan(obs_inputs, baseline_plan, environment_step):
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
        state = (value_features - state_mean) @ components.T
        state = state / state_scale
        state_rms = jnp.sqrt(jnp.mean(jnp.square(state), axis=-1))
        anchor_index = jnp.argmin(
            jnp.abs(anchor_steps - environment_step.astype(jnp.float32))
        )
        reliability_mask = jnp.broadcast_to(
            supported_dimension_mask[None],
            (baseline_plan.shape[0], action_dim_count),
        )
        reliability_mask = (
            reliability_mask & supported_anchor_mask[anchor_index]
        )
        predicted_delta = (
            intercept
            + dimension_bias[None]
            + anchor_bias[anchor_index]
            + state @ interaction
        )
        candidate_plans, candidate_deltas = sibling_bin_candidate_plans(
            baseline_plan,
            jnp.asarray(agent._step_action_low),
            jnp.asarray(agent._step_action_high),
            bins=agent.bins,
            force_level=force_level,
            intervention_horizon=intervention_horizon,
        )
        batch_size = baseline_plan.shape[0]
        candidate_count = agent.action_dim * agent.bins
        flat_candidates = candidate_plans.reshape(
            (
                batch_size * candidate_count,
                agent.action_sequence,
                agent.action_dim,
            )
        )
        repeated_policy_features = jnp.repeat(
            policy_features,
            candidate_count,
            axis=0,
        )
        policy_logits, encoded_bins = agent._policy_logits_per_level(
            policy_params,
            repeated_policy_features,
            flat_candidates,
        )
        selected_log_probability = jnp.take_along_axis(
            jax.nn.log_softmax(policy_logits, axis=-1),
            encoded_bins[..., None],
            axis=-1,
        )[..., 0]
        selected_log_probability = selected_log_probability.reshape(
            (
                batch_size,
                agent.action_dim,
                agent.bins,
                agent.levels,
                agent._flat_action_dim,
            )
        )[..., force_level, :]
        dimension_index = jnp.broadcast_to(
            jnp.arange(agent.action_dim)[None, :, None, None],
            (batch_size, agent.action_dim, agent.bins, 1),
        )
        policy_candidate_scores = jnp.take_along_axis(
            selected_log_probability,
            dimension_index,
            axis=-1,
        )[..., 0]
        return select_structured_delta_plan(
            baseline_plan,
            candidate_plans,
            candidate_deltas,
            policy_candidate_scores,
            predicted_delta,
            state_rms,
            reliability_mask=reliability_mask,
            min_value_margin=min_value_margin,
            max_bc_logprob_drop=max_bc_logprob_drop,
            max_state_rms=max_state_rms,
        )

    select_plan = jax.jit(select_plan)
    original_act = agent.act
    counters = {
        "inference_count": 0,
        "applied_override_count": 0,
        "eligible_dimension_count": 0,
        "selected_margin_sum": 0.0,
        "selected_margins": [],
        "state_rms": [],
        "selected_dimension_histogram": np.zeros(
            agent.action_dim,
            dtype=np.int64,
        ),
    }

    def reset_counters() -> None:
        counters["inference_count"] = 0
        counters["applied_override_count"] = 0
        counters["eligible_dimension_count"] = 0
        counters["selected_margin_sum"] = 0.0
        counters["selected_margins"].clear()
        counters["state_rms"].clear()
        counters["selected_dimension_histogram"].fill(0)

    def snapshot_counters() -> dict[str, Any]:
        applied = int(counters["applied_override_count"])
        inference = int(counters["inference_count"])
        margins = np.asarray(counters["selected_margins"], np.float64)
        state_rms_values = np.asarray(counters["state_rms"], np.float64)
        return {
            "inference_count": inference,
            "applied_override_count": applied,
            "override_rate": float(applied / inference) if inference else 0.0,
            "eligible_dimension_count": int(
                counters["eligible_dimension_count"]
            ),
            "mean_selected_value_margin": (
                float(counters["selected_margin_sum"] / applied)
                if applied
                else 0.0
            ),
            "selected_value_margin_quantiles": (
                {
                    str(q): float(np.quantile(margins, q))
                    for q in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
                }
                if margins.size
                else None
            ),
            "state_rms_quantiles": (
                {
                    str(q): float(np.quantile(state_rms_values, q))
                    for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
                }
                if state_rms_values.size
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
        result = jax.device_get(
            select_plan(
                obs_inputs,
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
        state_rms_value = np.asarray(result.state_rms, dtype=np.float64)
        counters["inference_count"] += int(applied.size)
        counters["applied_override_count"] += int(applied.sum())
        counters["eligible_dimension_count"] += int(
            np.asarray(result.eligible_override_mask, dtype=bool).sum()
        )
        counters["selected_margin_sum"] += float(
            selected_margin[applied].sum()
        )
        counters["selected_margins"].extend(
            selected_margin[applied].tolist()
        )
        counters["state_rms"].extend(state_rms_value.tolist())
        for dimension in selected_dimension[applied]:
            counters["selected_dimension_histogram"][int(dimension)] += 1
        if diagnostic_only:
            return baseline_plan
        return np.asarray(result.action, dtype=np.float32)

    agent.act = gated_act
    return reset_counters, snapshot_counters


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    import jax

    from robobase.workspace import Workspace

    (
        run_dir,
        snapshot,
        model_path,
        model_summary_path,
        _,
        work_dir,
    ) = _resolve_paths(args)
    if not snapshot.is_file():
        raise FileNotFoundError(snapshot)
    if args.num_eval_episodes < 1:
        raise ValueError("num_eval_episodes must be positive")
    if args.bc_only and args.diagnostic_only:
        raise ValueError("bc_only cannot be diagnostic_only")
    if not args.bc_only and (
        model_path is None or model_summary_path is None
    ):
        raise ValueError("sidecar evaluation requires model and model-summary")
    if args.force_level < 0 or args.intervention_horizon < 1:
        raise ValueError("invalid force-level/intervention-horizon")

    work_dir.mkdir(parents=True, exist_ok=True)
    cfg = _prepare_cfg(args, run_dir, work_dir)
    workspace = Workspace(cfg, work_dir=str(work_dir))
    model_metadata = None
    reset_counters = lambda: None
    snapshot_counters = lambda: {
        "inference_count": 0,
        "applied_override_count": 0,
        "override_rate": 0.0,
        "eligible_dimension_count": 0,
        "mean_selected_value_margin": 0.0,
        "selected_dimension_histogram": [0] * workspace.agent.action_dim,
    }
    episode_results = []
    try:
        workspace.load_snapshot(snapshot, load_replay_buffer=False)
        if not args.bc_only:
            assert model_path is not None and model_summary_path is not None
            model, model_metadata = _load_verified_model(
                model_path,
                model_summary_path,
                snapshot,
            )
            reset_counters, snapshot_counters = _install_structured_wrapper(
                workspace.agent,
                model,
                force_level=int(args.force_level),
                intervention_horizon=int(args.intervention_horizon),
                min_value_margin=float(args.min_value_margin),
                max_bc_logprob_drop=float(args.max_bc_logprob_drop),
                max_state_rms=float(args.max_state_rms),
                diagnostic_only=bool(args.diagnostic_only),
            )
        for episode in range(args.num_eval_episodes):
            eval_seed = int(args.eval_seed_start) + episode
            workspace.agent.rng_key = jax.random.PRNGKey(
                eval_seed + 930_000
            )
            workspace.cfg.env.eval_seed_start = eval_seed
            reset_counters()
            metrics = _numeric_metrics(workspace.eval())
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
                episode + 1 == args.num_eval_episodes
            ):
                print(
                    f"completed {episode + 1}/{args.num_eval_episodes}: "
                    f"success={np.mean([row['episode_success'] for row in episode_results]):.3f}",
                    flush=True,
                )
    finally:
        workspace.shutdown()

    total_inferences = sum(
        int(row["inference_count"]) for row in episode_results
    )
    total_overrides = sum(
        int(row["applied_override_count"]) for row in episode_results
    )
    return {
        "status": "ok",
        "task": str(cfg.env.task_name),
        "run_dir": str(run_dir),
        "snapshot": str(snapshot),
        "policy": (
            "bc_only"
            if args.bc_only
            else (
                "structured_delta_diagnostic_only"
                if args.diagnostic_only
                else "structured_delta_sidecar"
            )
        ),
        "model": model_metadata,
        "thresholds": (
            None
            if args.bc_only
            else {
                "force_level": int(args.force_level),
                "intervention_horizon": int(args.intervention_horizon),
                "min_value_margin": float(args.min_value_margin),
                "max_bc_logprob_drop": float(args.max_bc_logprob_drop),
                "max_state_rms": float(args.max_state_rms),
                "anchor_mapping": "nearest_training_anchor",
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
    _, _, _, _, output, _ = _resolve_paths(args)
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
