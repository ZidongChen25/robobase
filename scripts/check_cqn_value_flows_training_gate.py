#!/usr/bin/env python3
"""Validate matched CQN-conditioned Value Flows smoke/training runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from omegaconf import OmegaConf


REQUIRED_FINITE_METRICS = (
    "critic_loss",
    "td_critic_loss",
    "bcfm_loss",
    "dcfm_loss",
    "flow_critic_grad_norm",
    "velocity_head_grad_norm",
    "flow_critic_grad_nonfinite_fraction",
    "encoder_grad_nonfinite_fraction",
    "target_q_mean",
    "endpoint_q_mean",
    "confidence_weight_mean",
    "confidence_weight_min",
    "confidence_weight_max",
    "confidence_weight_std",
    "confidence_return_std_mean",
    "confidence_return_std_min",
    "confidence_return_std_max",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--required-snapshot-step", type=int, default=1_000)
    parser.add_argument("--min-log-rows", type=int, default=2)
    parser.add_argument("--expected-flow-steps", type=int, default=10)
    parser.add_argument("--expected-flow-samples", type=int, default=4)
    parser.add_argument("--expected-confidence-temp", type=float)
    parser.add_argument("--min-weight-range", type=float, default=1e-6)
    return parser.parse_args()


def _numeric_rows(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                parsed = {
                    key: float(row[key])
                    for key in REQUIRED_FINITE_METRICS
                }
                parsed["iteration"] = float(row["iteration"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(parsed)
    return rows


def check_run(
    run_dir: Path,
    *,
    required_snapshot_step: int,
    min_log_rows: int,
    expected_flow_steps: int,
    expected_flow_samples: int,
    expected_confidence_temp: float | None,
    min_weight_range: float,
) -> dict:
    run_dir = run_dir.expanduser().resolve()
    cfg_path = run_dir / ".hydra" / "config.yaml"
    csv_path = run_dir / "train.csv"
    snapshot = (
        run_dir
        / "snapshots"
        / f"{int(required_snapshot_step)}_snapshot.pkl"
    )
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    method = OmegaConf.load(cfg_path).method
    rows = _numeric_rows(csv_path)
    all_finite = bool(rows) and all(
        math.isfinite(value)
        for row in rows
        for value in row.values()
    )
    actual_temp = method.get("confidence_weight_temp", None)
    declared_temp = (
        actual_temp is None
        if expected_confidence_temp is None
        else actual_temp is not None
        and math.isclose(
            float(actual_temp),
            float(expected_confidence_temp),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    max_flow_grad = (
        max(row["flow_critic_grad_norm"] for row in rows)
        if rows
        else float("nan")
    )
    max_dcfm = (
        max(row["dcfm_loss"] for row in rows)
        if rows
        else float("nan")
    )
    max_nonfinite = (
        max(
            max(
                row["flow_critic_grad_nonfinite_fraction"],
                row["encoder_grad_nonfinite_fraction"],
            )
            for row in rows
        )
        if rows
        else float("nan")
    )
    if expected_confidence_temp is None:
        confidence_checks = {
            "confidence_is_exact_noop": (
                bool(rows)
                and all(
                    row["confidence_weight_mean"] == 1.0
                    and row["confidence_weight_min"] == 1.0
                    and row["confidence_weight_max"] == 1.0
                    and row["confidence_weight_std"] == 0.0
                    and row["confidence_return_std_mean"] == 0.0
                    for row in rows
                )
            )
        }
    else:
        weight_range = (
            max(row["confidence_weight_max"] for row in rows)
            - min(row["confidence_weight_min"] for row in rows)
            if rows
            else float("nan")
        )
        confidence_checks = {
            "confidence_weights_in_official_range": (
                bool(rows)
                and min(row["confidence_weight_min"] for row in rows) > 0.5
                and max(row["confidence_weight_max"] for row in rows) <= 1.0
            ),
            "confidence_return_std_positive": (
                bool(rows)
                and min(
                    row["confidence_return_std_min"] for row in rows
                )
                > 0.0
            ),
            "confidence_weights_nonconstant": (
                bool(rows) and weight_range > float(min_weight_range)
            ),
        }
    checks = {
        "declared_value_flows_core": all(
            (
                str(method.get("value_mode", "")).lower() == "return_sample",
                float(method.get("bcfm_lambda", 0.0)) == 1.0,
                float(method.get("dcfm_lambda", 0.0)) == 1.0,
                int(method.get("num_flow_steps", 0))
                == int(expected_flow_steps),
                int(method.get("num_flow_samples", 0))
                == int(expected_flow_samples),
                int(method.get("num_target_flow_samples", 0))
                == int(expected_flow_samples),
                str(method.get("flow_source_type", "")).lower()
                == "gaussian",
                str(method.get("time_embedding_type", "")).lower() == "raw",
                bool(method.get("clip_flow_trajectory", False)),
                int(method.get("num_update_steps", 0)) == 4,
                bool(method.get("separate_bc_policy", False)),
                bool(method.get("distinct_policy_encoder", False)),
                str(method.get("td_target_action_source", "")).lower()
                == "bc_policy",
                method.get("policy_value_beta", None) is None,
            )
        ),
        "declared_confidence_temperature": declared_temp,
        "snapshot_exists": snapshot.is_file(),
        "enough_log_rows": len(rows) >= int(min_log_rows),
        "all_required_metrics_finite": all_finite,
        "zero_nonfinite_gradient_fraction": (
            bool(rows) and max_nonfinite == 0.0
        ),
        "flow_critic_receives_gradient": (
            bool(rows) and max_flow_grad > 1e-8
        ),
        "dcfm_is_active_after_target_lag": (
            bool(rows) and max_dcfm > 1e-10
        ),
        **confidence_checks,
    }
    return {
        "status": "ok",
        "run_dir": str(run_dir),
        "required_snapshot": str(snapshot),
        "num_log_rows": len(rows),
        "expected_confidence_temp": expected_confidence_temp,
        "max_flow_critic_grad_norm": max_flow_grad,
        "max_dcfm_loss": max_dcfm,
        "checks": checks,
        "gate": "pass" if all(checks.values()) else "fail",
    }


def main() -> int:
    args = parse_args()
    payload = check_run(
        args.run_dir,
        required_snapshot_step=args.required_snapshot_step,
        min_log_rows=args.min_log_rows,
        expected_flow_steps=args.expected_flow_steps,
        expected_flow_samples=args.expected_flow_samples,
        expected_confidence_temp=args.expected_confidence_temp,
        min_weight_range=args.min_weight_range,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["gate"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
