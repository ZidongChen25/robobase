#!/usr/bin/env python3
"""Validate a short direct scalar-Q training run before scaling seeds."""

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
    "direct_q_loss",
    "mc_return_loss",
    "direct_q_grad_norm",
    "direct_q_grad_nonfinite_fraction",
    "target_q_mean",
    "direct_q_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--required-snapshot-step", type=int, default=1_000)
    parser.add_argument("--min-log-rows", type=int, default=2)
    parser.add_argument("--min-max-direct-q-grad", type=float, default=1e-8)
    parser.add_argument(
        "--expected-td-target-action-source",
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
                # CSV logging can append a repeated header after resume.
                continue
            rows.append(parsed)
    return rows


def check_run(
    run_dir: Path,
    *,
    required_snapshot_step: int,
    min_log_rows: int,
    min_max_direct_q_grad: float,
    expected_td_target_action_source: str | None = None,
    expected_td_target_policy_value_beta: float | None = None,
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

    cfg = OmegaConf.load(cfg_path)
    rows = _numeric_rows(csv_path)
    all_finite = bool(rows) and all(
        math.isfinite(value)
        for row in rows
        for value in row.values()
    )
    max_nonfinite = (
        max(row["direct_q_grad_nonfinite_fraction"] for row in rows)
        if rows
        else float("nan")
    )
    max_direct_q_grad = (
        max(row["direct_q_grad_norm"] for row in rows)
        if rows
        else float("nan")
    )
    checks = {
        "direct_scalar_q_config": bool(
            cfg.method.get("direct_scalar_q", False)
        ),
        "matched_mse_loss": (
            str(cfg.method.get("direct_q_loss", "")).lower() == "mse"
        ),
        "snapshot_exists": snapshot.is_file(),
        "enough_log_rows": len(rows) >= int(min_log_rows),
        "all_required_metrics_finite": all_finite,
        "zero_nonfinite_gradient_fraction": (
            bool(rows) and max_nonfinite == 0.0
        ),
        "critic_receives_gradient": (
            bool(rows)
            and max_direct_q_grad > float(min_max_direct_q_grad)
        ),
    }
    if expected_td_target_action_source is not None:
        checks["declared_td_target_action_source"] = (
            str(cfg.method.get("td_target_action_source", "")).lower()
            == str(expected_td_target_action_source).lower()
        )
        actual_target_beta = cfg.method.get(
            "td_target_policy_value_beta",
            None,
        )
        checks["declared_td_target_policy_value_beta"] = (
            (
                actual_target_beta is None
                and expected_td_target_policy_value_beta is None
            )
            or (
                actual_target_beta is not None
                and expected_td_target_policy_value_beta is not None
                and math.isclose(
                    float(actual_target_beta),
                    float(expected_td_target_policy_value_beta),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
        )
        checks["rollout_remains_exact_bc"] = (
            cfg.method.get("policy_value_beta", None) is None
        )
    return {
        "status": "ok",
        "run_dir": str(run_dir),
        "train_csv": str(csv_path),
        "required_snapshot": str(snapshot),
        "num_log_rows": len(rows),
        "required_finite_metrics": list(REQUIRED_FINITE_METRICS),
        "max_direct_q_grad_nonfinite_fraction": max_nonfinite,
        "max_direct_q_grad_norm": max_direct_q_grad,
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
        min_max_direct_q_grad=args.min_max_direct_q_grad,
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
