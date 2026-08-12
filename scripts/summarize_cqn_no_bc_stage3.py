#!/usr/bin/env python3
"""Select the Stage-3 worst-case conservative-Q treatment."""

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
    }


def summarize(mean_floor_dir: Path, max_floor_dir: Path) -> dict:
    arms = {
        "mc_mean_floor": _arm(mean_floor_dir),
        "mc_max_floor": _arm(max_floor_dir),
    }
    delta = (
        arms["mc_max_floor"]["best_success"]
        - arms["mc_mean_floor"]["best_success"]
    )
    treatment = arms["mc_max_floor"]
    gate_pass = treatment["best_success"] >= 0.20 and delta >= 0.10
    return {
        "protocol": {
            "selection_seeds": [400, 449],
            "episodes_per_checkpoint": 50,
            "expected_steps": list(EXPECTED_STEPS),
            "checkpoint_tie_break": "earliest checkpoint",
            "gate": "max-floor best >= 20% and improvement >= 10pp",
        },
        "arms": arms,
        "contrasts": {"max_minus_mean_floor": delta},
        "selected_variant": (
            "mc_max_floor"
            if treatment["best_success"]
            > arms["mc_mean_floor"]["best_success"]
            else "mc_mean_floor"
        ),
        "selected_step": (
            treatment["best_step"]
            if treatment["best_success"]
            > arms["mc_mean_floor"]["best_success"]
            else arms["mc_mean_floor"]["best_step"]
        ),
        "selected_success": max(
            treatment["best_success"],
            arms["mc_mean_floor"]["best_success"],
        ),
        "worst_case_floor_gate": "pass" if gate_pass else "fail",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mean-floor-run", type=Path, required=True)
    parser.add_argument("--max-floor-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.mean_floor_run, args.max_floor_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
