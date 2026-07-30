#!/usr/bin/env python3
"""Validate matched FlowIQN / quantile-regularized flow training arms."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from omegaconf import OmegaConf


ARM_COEFFICIENTS = {
    "anchor_only": (1.0, 0.0),
    "joint_equal": (1.0, 1.0),
    "dbc_ratio": (0.01, 1.0),
}

REQUIRED_FINITE_METRICS = (
    "critic_loss",
    "td_critic_loss",
    "bcfm_loss",
    "quantile_endpoint_loss",
    "dcfm_loss",
    "evor_td_loss",
    "pcbf_loss",
    "mc_return_loss",
    "flow_critic_grad_norm",
    "velocity_head_grad_norm",
    "critic_update_norm",
    "flow_critic_grad_nonfinite_fraction",
    "encoder_grad_nonfinite_fraction",
    "target_q_mean",
    "endpoint_q_mean",
    "policy_bc_loss",
    "policy_demo_top1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--expected-arm",
        required=True,
        choices=tuple(ARM_COEFFICIENTS),
    )
    parser.add_argument("--required-snapshot-step", type=int, default=1_000)
    parser.add_argument("--min-log-rows", type=int, default=2)
    parser.add_argument("--min-loss-range", type=float, default=1e-10)
    return parser.parse_args()


def _numeric_rows(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                parsed = {
                    key: float(row[key]) for key in REQUIRED_FINITE_METRICS
                }
                parsed["iteration"] = float(row["iteration"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(parsed)
    return rows


def _close(actual: object, expected: float) -> bool:
    try:
        return math.isclose(
            float(actual),
            float(expected),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    except (TypeError, ValueError):
        return False


def check_run(
    run_dir: Path,
    *,
    expected_arm: str,
    required_snapshot_step: int,
    min_log_rows: int,
    min_loss_range: float,
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
    if expected_arm not in ARM_COEFFICIENTS:
        raise ValueError(f"unknown arm: {expected_arm}")

    method = OmegaConf.load(cfg_path).method
    rows = _numeric_rows(csv_path)
    expected_bcfm, expected_quantile = ARM_COEFFICIENTS[expected_arm]
    all_finite = bool(rows) and all(
        math.isfinite(value) for row in rows for value in row.values()
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
    max_bcfm = (
        max(row["bcfm_loss"] for row in rows) if rows else float("nan")
    )
    max_quantile = (
        max(row["quantile_endpoint_loss"] for row in rows)
        if rows
        else float("nan")
    )
    quantile_range = (
        max(row["quantile_endpoint_loss"] for row in rows)
        - min(row["quantile_endpoint_loss"] for row in rows)
        if rows
        else float("nan")
    )
    max_flow_grad = (
        max(row["flow_critic_grad_norm"] for row in rows)
        if rows
        else float("nan")
    )
    max_velocity_grad = (
        max(row["velocity_head_grad_norm"] for row in rows)
        if rows
        else float("nan")
    )
    max_update_norm = (
        max(row["critic_update_norm"] for row in rows)
        if rows
        else float("nan")
    )
    unused_objectives_zero = bool(rows) and all(
        row["dcfm_loss"] == 0.0
        and row["evor_td_loss"] == 0.0
        and row["pcbf_loss"] == 0.0
        and row["mc_return_loss"] == 0.0
        for row in rows
    )

    checks = {
        "declared_registered_arm": all(
            (
                _close(method.get("bcfm_lambda"), expected_bcfm),
                _close(
                    method.get("quantile_endpoint_lambda"),
                    expected_quantile,
                ),
                _close(method.get("quantile_huber_kappa"), 1.0),
            )
        ),
        "declared_flowiqn_distribution": all(
            (
                str(method.get("value_mode", "")).lower() == "return_sample",
                bool(method.get("flow_iqn_quantile_coupling", False)),
                str(method.get("flow_source_type", "")).lower() == "uniform",
                _close(method.get("flow_source_min"), 0.9),
                _close(method.get("flow_source_max"), 1.0),
                not bool(method.get("antithetic_flow_sources", True)),
                bool(method.get("fixed_action_flow_sources", False)),
                bool(method.get("action_flow_quantile_grid", False)),
                int(method.get("num_flow_steps", 0)) == 8,
                int(method.get("num_flow_samples", 0)) == 8,
                int(method.get("num_target_flow_samples", 0)) == 8,
                int(method.get("num_action_flow_samples", 0)) == 4,
            )
        ),
        "declared_fixed_bc_target_and_two_towers": all(
            (
                int(method.get("num_update_steps", 0)) == 4,
                bool(method.get("separate_bc_policy", False)),
                bool(method.get("distinct_policy_encoder", False)),
                str(method.get("td_target_action_source", "")).lower()
                == "bc_policy",
                method.get("td_target_policy_value_beta", None) is None,
                method.get("policy_value_beta", None) is None,
                _close(method.get("mc_return_weight"), 0.0),
            )
        ),
        "unused_objectives_exactly_zero": unused_objectives_zero,
        "snapshot_exists": snapshot.is_file(),
        "enough_log_rows": len(rows) >= int(min_log_rows),
        "all_required_metrics_finite": all_finite,
        "zero_nonfinite_gradient_fraction": bool(rows)
        and max_nonfinite == 0.0,
        "sorted_cfm_loss_active": bool(rows) and max_bcfm > 1e-10,
        "quantile_loss_matches_arm": (
            bool(rows)
            and (
                (expected_quantile == 0.0 and max_quantile == 0.0)
                or (
                    expected_quantile > 0.0
                    and max_quantile > 1e-10
                    and quantile_range > float(min_loss_range)
                )
            )
        ),
        "flow_critic_receives_gradient": bool(rows)
        and max_flow_grad > 1e-8,
        "velocity_head_receives_gradient": bool(rows)
        and max_velocity_grad > 1e-8,
        "critic_update_is_nonzero": bool(rows)
        and max_update_norm > 1e-10,
        "bc_policy_metric_valid": bool(rows)
        and all(
            0.0 <= row["policy_demo_top1"] <= 1.0 for row in rows
        ),
    }
    return {
        "status": "ok",
        "arm": expected_arm,
        "run_dir": str(run_dir),
        "required_snapshot": str(snapshot),
        "num_log_rows": len(rows),
        "max_bcfm_loss": max_bcfm,
        "max_quantile_endpoint_loss": max_quantile,
        "quantile_endpoint_loss_range": quantile_range,
        "max_flow_critic_grad_norm": max_flow_grad,
        "max_velocity_head_grad_norm": max_velocity_grad,
        "max_critic_update_norm": max_update_norm,
        "checks": checks,
        "gate": "pass" if all(checks.values()) else "fail",
    }


def main() -> int:
    args = parse_args()
    payload = check_run(
        args.run_dir,
        expected_arm=args.expected_arm,
        required_snapshot_step=args.required_snapshot_step,
        min_log_rows=args.min_log_rows,
        min_loss_range=args.min_loss_range,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["gate"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
