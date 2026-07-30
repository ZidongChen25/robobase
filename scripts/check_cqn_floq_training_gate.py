#!/usr/bin/env python3
"""Validate a short expected-FLOQ training run before scaling it."""

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
    "flow_distill_loss",
    "mc_return_loss",
    "flow_critic_grad_norm",
    "flow_distill_readout_grad_norm",
    "velocity_head_grad_norm",
    "flow_critic_grad_nonfinite_fraction",
    "encoder_grad_nonfinite_fraction",
    "target_q_mean",
    "endpoint_q_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--required-snapshot-step", type=int, default=1_000)
    parser.add_argument("--min-log-rows", type=int, default=2)
    parser.add_argument("--min-max-flow-grad", type=float, default=1e-8)
    parser.add_argument("--min-max-distill-grad", type=float, default=1e-8)
    parser.add_argument("--expected-bcfm-lambda", type=float, required=True)
    parser.add_argument("--expected-source-min", type=float)
    parser.add_argument("--expected-source-max", type=float)
    parser.add_argument(
        "--expected-td-target-action-source",
        default="replay_next",
        choices=("critic", "replay_next", "bc_policy", "policy_value"),
    )
    parser.add_argument("--expected-td-target-policy-value-beta", type=float)
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
                # A resumed CSV may contain a repeated header.
                continue
            rows.append(parsed)
    return rows


def _optional_float_matches(actual, expected: float | None) -> bool:
    if expected is None:
        return actual is None
    return actual is not None and math.isclose(
        float(actual),
        float(expected),
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def check_run(
    run_dir: Path,
    *,
    required_snapshot_step: int,
    min_log_rows: int,
    min_max_flow_grad: float,
    min_max_distill_grad: float,
    expected_bcfm_lambda: float,
    expected_source_min: float | None,
    expected_source_max: float | None,
    expected_td_target_action_source: str = "replay_next",
    expected_td_target_policy_value_beta: float | None = None,
) -> dict:
    if (expected_source_min is None) != (expected_source_max is None):
        raise ValueError(
            "expected source bounds must both be set or both be omitted"
        )
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

    cfg = OmegaConf.load(cfg_path)
    method = cfg.method
    rows = _numeric_rows(csv_path)
    all_finite = bool(rows) and all(
        math.isfinite(value)
        for row in rows
        for value in row.values()
    )
    max_flow_nonfinite = (
        max(row["flow_critic_grad_nonfinite_fraction"] for row in rows)
        if rows
        else float("nan")
    )
    max_encoder_nonfinite = (
        max(row["encoder_grad_nonfinite_fraction"] for row in rows)
        if rows
        else float("nan")
    )
    max_flow_grad = (
        max(row["flow_critic_grad_norm"] for row in rows)
        if rows
        else float("nan")
    )
    max_distill_grad = (
        max(row["flow_distill_readout_grad_norm"] for row in rows)
        if rows
        else float("nan")
    )
    common_fidelity = all(
        (
            str(method.get("value_mode", "")).lower() == "scalar",
            int(method.get("num_flow_steps", 0)) == 8,
            int(method.get("num_flow_samples", 0)) == 8,
            int(method.get("num_target_flow_samples", 0)) == 8,
            int(method.get("num_action_flow_samples", 0)) == 8,
            str(method.get("flow_source_type", "")).lower() == "uniform",
            math.isclose(
                float(method.get("flow_distill_lambda", -1.0)),
                1.0,
            ),
            int(method.get("num_update_steps", 0)) == 4,
            bool(method.get("separate_bc_policy", False)),
            bool(method.get("distinct_policy_encoder", False)),
            str(method.get("critic_sequence_mode", "")).lower()
            == "effective_k0",
        )
    )
    checks = {
        "common_expected_floq_fidelity": common_fidelity,
        "declared_bcfm_lambda": math.isclose(
            float(method.get("bcfm_lambda", float("nan"))),
            float(expected_bcfm_lambda),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "declared_source_min": _optional_float_matches(
            method.get("flow_source_min", None),
            expected_source_min,
        ),
        "declared_source_max": _optional_float_matches(
            method.get("flow_source_max", None),
            expected_source_max,
        ),
        "declared_td_target_action_source": (
            str(method.get("td_target_action_source", "")).lower()
            == str(expected_td_target_action_source).lower()
        ),
        "declared_td_target_policy_value_beta": _optional_float_matches(
            method.get("td_target_policy_value_beta", None),
            expected_td_target_policy_value_beta,
        ),
        "rollout_remains_exact_bc": (
            method.get("policy_value_beta", None) is None
        ),
        "snapshot_exists": snapshot.is_file(),
        "enough_log_rows": len(rows) >= int(min_log_rows),
        "all_required_metrics_finite": all_finite,
        "zero_nonfinite_gradient_fraction": (
            bool(rows)
            and max_flow_nonfinite == 0.0
            and max_encoder_nonfinite == 0.0
        ),
        "flow_critic_receives_gradient": (
            bool(rows) and max_flow_grad > float(min_max_flow_grad)
        ),
        "distill_readout_receives_gradient": (
            bool(rows)
            and max_distill_grad > float(min_max_distill_grad)
        ),
    }
    return {
        "status": "ok",
        "run_dir": str(run_dir),
        "train_csv": str(csv_path),
        "required_snapshot": str(snapshot),
        "num_log_rows": len(rows),
        "required_finite_metrics": list(REQUIRED_FINITE_METRICS),
        "max_flow_critic_grad_nonfinite_fraction": max_flow_nonfinite,
        "max_encoder_grad_nonfinite_fraction": max_encoder_nonfinite,
        "max_flow_critic_grad_norm": max_flow_grad,
        "max_flow_distill_readout_grad_norm": max_distill_grad,
        "last_iteration": rows[-1]["iteration"] if rows else None,
        "checks": checks,
        "gate": "pass" if all(checks.values()) else "fail",
    }


def main() -> int:
    args = parse_args()
    payload = check_run(
        args.run_dir,
        required_snapshot_step=args.required_snapshot_step,
        min_log_rows=args.min_log_rows,
        min_max_flow_grad=args.min_max_flow_grad,
        min_max_distill_grad=args.min_max_distill_grad,
        expected_bcfm_lambda=args.expected_bcfm_lambda,
        expected_source_min=args.expected_source_min,
        expected_source_max=args.expected_source_max,
        expected_td_target_action_source=(
            args.expected_td_target_action_source
        ),
        expected_td_target_policy_value_beta=(
            args.expected_td_target_policy_value_beta
        ),
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["gate"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
