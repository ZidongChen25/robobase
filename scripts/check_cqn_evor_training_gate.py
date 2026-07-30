#!/usr/bin/env python3
"""Validate an isolated CQN-AS-adapted EVOR FlowTD training run."""

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
    "evor_td_loss",
    "bcfm_loss",
    "dcfm_loss",
    "pcbf_loss",
    "flow_critic_grad_norm",
    "velocity_head_grad_norm",
    "critic_update_norm",
    "flow_critic_grad_nonfinite_fraction",
    "encoder_grad_nonfinite_fraction",
    "target_q_mean",
    "endpoint_q_mean",
    "mc_return_mean",
    "policy_bc_loss",
    "policy_demo_top1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--required-snapshot-step", type=int, default=1_000)
    parser.add_argument("--min-log-rows", type=int, default=2)
    parser.add_argument("--expected-flow-steps", type=int, default=10)
    parser.add_argument("--expected-action-flow-samples", type=int, default=16)
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


def check_run(
    run_dir: Path,
    *,
    required_snapshot_step: int,
    min_log_rows: int,
    expected_flow_steps: int,
    expected_action_flow_samples: int,
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

    method = OmegaConf.load(cfg_path).method
    rows = _numeric_rows(csv_path)
    all_finite = bool(rows) and all(
        math.isfinite(value) for row in rows for value in row.values()
    )
    max_evor_loss = (
        max(row["evor_td_loss"] for row in rows)
        if rows
        else float("nan")
    )
    evor_loss_range = (
        max(row["evor_td_loss"] for row in rows)
        - min(row["evor_td_loss"] for row in rows)
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
    auxiliary_objectives_zero = bool(rows) and all(
        row["bcfm_loss"] == 0.0
        and row["dcfm_loss"] == 0.0
        and row["pcbf_loss"] == 0.0
        for row in rows
    )
    policy_metric_valid = bool(rows) and all(
        0.0 <= row["policy_demo_top1"] <= 1.0 for row in rows
    )
    mc_return_data_present = bool(rows) and any(
        abs(row["mc_return_mean"]) > 1e-8 for row in rows
    )

    checks = {
        "declared_isolated_evor_flowtd": all(
            (
                str(method.get("value_mode", "")).lower() == "return_sample",
                float(method.get("evor_td_lambda", 0.0)) == 1.0,
                float(method.get("bcfm_lambda", 0.0)) == 0.0,
                float(method.get("dcfm_lambda", 0.0)) == 0.0,
                float(method.get("pcbf_loss_coeff", 0.0)) == 0.0,
                float(method.get("pcbf_lambda", 0.0)) == 0.0,
                float(method.get("flow_distill_lambda", 0.0)) == 0.0,
                float(method.get("endpoint_q_lambda", 0.0)) == 0.0,
                float(method.get("source_consistency_lambda", 0.0)) == 0.0,
                not bool(method.get("flow_iqn_quantile_coupling", False)),
                method.get("confidence_weight_temp", None) is None,
                float(method.get("mc_return_weight", 0.0)) == 0.0,
            )
        ),
        "declared_evor_compute_and_readout": all(
            (
                int(method.get("num_flow_steps", 0))
                == int(expected_flow_steps),
                int(method.get("num_flow_samples", 0)) == 1,
                int(method.get("num_target_flow_samples", 0)) == 1,
                int(method.get("num_action_flow_samples", 0))
                == int(expected_action_flow_samples),
                str(method.get("flow_source_type", "")).lower()
                == "gaussian",
                not bool(method.get("antithetic_flow_sources", True)),
                bool(method.get("fixed_action_flow_sources", False)),
                str(method.get("return_sample_aggregation", "")).lower()
                == "entropic",
                float(method.get("return_sample_temperature", 0.0)) == 1.0,
                str(method.get("time_embedding_type", "")).lower() == "raw",
                not bool(method.get("clip_flow_trajectory", True)),
                not bool(method.get("flow_q_action_readout", True)),
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
            )
        ),
        "snapshot_exists": snapshot.is_file(),
        "enough_log_rows": len(rows) >= int(min_log_rows),
        "all_required_metrics_finite": all_finite,
        "zero_nonfinite_gradient_fraction": bool(rows)
        and max_nonfinite == 0.0,
        "evor_loss_active": bool(rows) and max_evor_loss > 1e-10,
        "evor_loss_nonconstant": bool(rows)
        and evor_loss_range > float(min_loss_range),
        "auxiliary_flow_objectives_exactly_zero": auxiliary_objectives_zero,
        "flow_critic_receives_gradient": bool(rows)
        and max_flow_grad > 1e-8,
        "velocity_head_receives_gradient": bool(rows)
        and max_velocity_grad > 1e-8,
        "critic_update_is_nonzero": bool(rows) and max_update_norm > 1e-10,
        "offline_return_endpoint_present": mc_return_data_present,
        "bc_policy_metric_valid": policy_metric_valid,
    }
    return {
        "status": "ok",
        "run_dir": str(run_dir),
        "required_snapshot": str(snapshot),
        "num_log_rows": len(rows),
        "max_evor_td_loss": max_evor_loss,
        "evor_td_loss_range": evor_loss_range,
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
        required_snapshot_step=args.required_snapshot_step,
        min_log_rows=args.min_log_rows,
        expected_flow_steps=args.expected_flow_steps,
        expected_action_flow_samples=args.expected_action_flow_samples,
        min_loss_range=args.min_loss_range,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["gate"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
