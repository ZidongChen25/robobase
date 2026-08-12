#!/usr/bin/env python3
"""Select the Stage-5 demo-only offline-RL treatment."""

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


def summarize(mixed_dir: Path, demo_only_dir: Path) -> dict:
    arms = {
        "mc_mixed_replay": _arm(mixed_dir),
        "mc_demo_only": _arm(demo_only_dir),
    }
    delta = (
        arms["mc_demo_only"]["best_success"]
        - arms["mc_mixed_replay"]["best_success"]
    )
    treatment = arms["mc_demo_only"]
    gate_pass = treatment["best_success"] >= 0.40 and delta >= 0.15
    treatment_wins = (
        treatment["best_success"]
        > arms["mc_mixed_replay"]["best_success"]
    )
    selected_name = "mc_demo_only" if treatment_wins else "mc_mixed_replay"
    return {
        "protocol": {
            "selection_seeds": [400, 449],
            "episodes_per_checkpoint": 50,
            "expected_steps": list(EXPECTED_STEPS),
            "checkpoint_tie_break": "earliest checkpoint",
            "gate": "demo-only best >= 40% and improvement >= 15pp",
        },
        "arms": arms,
        "contrasts": {"demo_only_minus_mixed_replay": delta},
        "selected_variant": selected_name,
        "selected_step": arms[selected_name]["best_step"],
        "selected_success": arms[selected_name]["best_success"],
        "demo_only_gate": "pass" if gate_pass else "fail",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixed-run", type=Path, required=True)
    parser.add_argument("--demo-only-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.mixed_run, args.demo_only_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
