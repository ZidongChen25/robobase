#!/usr/bin/env python3
"""Aggregate Stage-8 dense-return seeds against locked original CQNAS."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED_STEPS = (2500, 5000, 7500, 10000)


def _arm(run_dir: Path) -> dict:
    path = run_dir / "val50_seeds400.csv"
    curve = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            step = int(float(row["env_steps"]))
            if step in EXPECTED_STEPS:
                curve[step] = float(row["episode_success"])
    missing = sorted(set(EXPECTED_STEPS) - set(curve))
    if missing:
        raise ValueError(f"{path} is missing validation steps {missing}")
    best_step, best_success = max(
        curve.items(),
        key=lambda item: (item[1], -item[0]),
    )
    return {
        "run_dir": str(run_dir.resolve()),
        "curve": {str(step): value for step, value in curve.items()},
        "best_step": best_step,
        "best_success": best_success,
        "selected_snapshot": str(
            (run_dir / "snapshots" / f"{best_step}_snapshot.pkl").resolve()
        ),
    }


def summarize(
    run_dirs: list[Path],
    baseline_summary_path: Path,
) -> dict:
    dense = {
        f"seed{index}": _arm(run_dir)
        for index, run_dir in enumerate(run_dirs, start=1)
    }
    dense_mean = sum(
        arm["best_success"] for arm in dense.values()
    ) / len(dense)

    with baseline_summary_path.open() as handle:
        baseline_raw = json.load(handle)
    baseline = {
        name: {
            "best_step": run["best_step"],
            "best_success": run["best_success"],
            "run_dir": run["run_dir"],
        }
        for name, run in baseline_raw["runs"].items()
    }
    baseline_mean = sum(
        arm["best_success"] for arm in baseline.values()
    ) / len(baseline)
    gate_pass = dense_mean >= baseline_mean
    return {
        "protocol": {
            "selection_seeds": [400, 449],
            "episodes_per_checkpoint": 50,
            "expected_steps": list(EXPECTED_STEPS),
            "checkpoint_tie_break": "earliest checkpoint",
            "heldout_seeds": [800, 999],
            "heldout_status": (
                "unlocked" if gate_pass else "sealed_validation_fail"
            ),
            "validation_gate": (
                "dense three-seed mean >= locked original three-seed mean"
            ),
        },
        "dense_return": {
            "runs": dense,
            "mean_validation_best": dense_mean,
        },
        "original_cqn_as": {
            "source": str(baseline_summary_path.resolve()),
            "runs": baseline,
            "mean_validation_best": baseline_mean,
        },
        "dense_minus_original": dense_mean - baseline_mean,
        "validation_gate": "pass" if gate_pass else "fail",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed1-run", type=Path, required=True)
    parser.add_argument("--seed2-run", type=Path, required=True)
    parser.add_argument("--seed3-run", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        [args.seed1_run, args.seed2_run, args.seed3_run],
        args.baseline_summary,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
