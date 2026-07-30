#!/usr/bin/env python3
"""Apply a preregistered mechanistic gate to flow-utilization diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-q-span", type=float, default=1e-3)
    parser.add_argument("--max-source-contraction", type=float, default=0.95)
    parser.add_argument("--min-normalized-curvature", type=float, default=0.01)
    parser.add_argument("--max-one-step-agreement", type=float, default=0.98)
    parser.add_argument("--min-one-step-normalized-rmse", type=float, default=0.02)
    return parser.parse_args()


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (str, bool)) or value is None:
        return True
    try:
        return bool(np.all(np.isfinite(np.asarray(value, dtype=np.float64))))
    except (TypeError, ValueError):
        return True


def summarize(
    payload: dict[str, Any],
    *,
    min_q_span: float,
    max_source_contraction: float,
    min_normalized_curvature: float,
    max_one_step_agreement: float,
    min_one_step_normalized_rmse: float,
) -> dict[str, Any]:
    thresholds = {
        "min_q_span": float(min_q_span),
        "max_source_contraction": float(max_source_contraction),
        "min_normalized_curvature": float(min_normalized_curvature),
        "max_one_step_agreement": float(max_one_step_agreement),
        "min_one_step_normalized_rmse": float(
            min_one_step_normalized_rmse
        ),
    }
    if not all(math.isfinite(value) for value in thresholds.values()):
        raise ValueError("all thresholds must be finite")
    if payload.get("status") != "ok":
        raise ValueError("probe payload is not complete")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("probe payload is missing metrics")

    step_counts = [
        int(value)
        for value in metrics.get(
            "step_counts", payload.get("step_counts", [])
        )
    ]
    configured_steps = int(
        metrics.get(
            "configured_num_flow_steps",
            payload.get("configured_num_flow_steps", -1),
        )
    )
    if 1 not in step_counts:
        raise ValueError("probe must include a one-step readout")
    if configured_steps not in step_counts:
        raise ValueError("probe must include its configured flow depth")
    one_index = step_counts.index(1)
    configured_index = step_counts.index(configured_steps)

    ranking = np.asarray(
        metrics["mean_step_ranking_agreement"], dtype=np.float64
    )
    normalized_rmse = np.asarray(
        metrics["mean_step_normalized_q_rmse"], dtype=np.float64
    )
    if ranking.shape != (len(step_counts),) or normalized_rmse.shape != (
        len(step_counts),
    ):
        raise ValueError("step comparison metrics have the wrong shape")

    q_span = float(
        np.mean(
            np.asarray(
                metrics["per_level_configured_q_span"],
                dtype=np.float64,
            )
        )
    )
    contraction = float(metrics["mean_source_contraction_ratio"])
    curvature = float(metrics["mean_normalized_curvature_rms"])
    one_step_agreement = float(ranking[one_index])
    one_step_normalized_rmse = float(normalized_rmse[one_index])
    configured_agreement = float(ranking[configured_index])
    configured_normalized_rmse = float(
        normalized_rmse[configured_index]
    )

    depth_sensitive = (
        one_step_agreement <= thresholds["max_one_step_agreement"]
        and one_step_normalized_rmse
        >= thresholds["min_one_step_normalized_rmse"]
    )
    nonlinear_or_depth_sensitive = (
        curvature >= thresholds["min_normalized_curvature"]
        or depth_sensitive
    )
    checks = {
        "all_metrics_finite": _all_finite(metrics),
        "configured_readout_self_consistent": (
            configured_agreement >= 1.0 - 1e-6
            and configured_normalized_rmse <= 1e-6
        ),
        "nondegenerate_action_q_span": q_span >= thresholds["min_q_span"],
        "source_noise_contracted": (
            contraction <= thresholds["max_source_contraction"]
        ),
        "nonlinear_or_depth_sensitive": nonlinear_or_depth_sensitive,
    }
    gate = "pass" if all(checks.values()) else "fail"
    if gate == "pass":
        interpretation = (
            "The trained scalar field is nondegenerate, suppresses source "
            "noise, and performs computation not reducible to its one-step "
            "readout under the preregistered tolerances."
        )
    else:
        interpretation = (
            "At least one mechanistic criterion failed; this checkpoint does "
            "not establish nontrivial iterative value transport."
        )

    return {
        "status": "ok",
        "gate": gate,
        "diagnostic_only": True,
        "selection_use_forbidden": True,
        "checks": checks,
        "thresholds": thresholds,
        "measurements": {
            "configured_num_flow_steps": configured_steps,
            "step_counts": step_counts,
            "mean_configured_q_span": q_span,
            "mean_source_contraction_ratio": contraction,
            "mean_normalized_curvature_rms": curvature,
            "one_step_ranking_agreement": one_step_agreement,
            "one_step_normalized_q_rmse": one_step_normalized_rmse,
            "configured_ranking_agreement": configured_agreement,
            "configured_normalized_q_rmse": configured_normalized_rmse,
            "depth_sensitive": depth_sensitive,
        },
        "interpretation": interpretation,
        "probe": {
            "run_dir": payload.get("run_dir"),
            "snapshot": payload.get("snapshot"),
            "critic": payload.get("critic"),
            "eval_seeds": payload.get("eval_seeds"),
        },
    }


def main() -> int:
    args = parse_args()
    payload = json.loads(args.input.read_text())
    summary = summarize(
        payload,
        min_q_span=args.min_q_span,
        max_source_contraction=args.max_source_contraction,
        min_normalized_curvature=args.min_normalized_curvature,
        max_one_step_agreement=args.max_one_step_agreement,
        min_one_step_normalized_rmse=args.min_one_step_normalized_rmse,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
