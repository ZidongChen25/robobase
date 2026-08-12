#!/usr/bin/env python3
"""Select the Stage-12 chunk-aligned eight-step Q treatment."""

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


def summarize(control_dir: Path, treatment_dir: Path) -> dict:
    arms = {
        "k8_nstep1_control": _arm(control_dir),
        "k8_nstep8_treatment": _arm(treatment_dir),
    }
    control = arms["k8_nstep1_control"]
    treatment = arms["k8_nstep8_treatment"]
    delta = treatment["best_success"] - control["best_success"]
    gate_pass = treatment["best_success"] >= 0.64 and delta >= 0.15
    treatment_wins = treatment["best_success"] > control["best_success"]
    selected_name = (
        "k8_nstep8_treatment"
        if treatment_wins
        else "k8_nstep1_control"
    )
    return {
        "protocol": {
            "selection_seeds": [400, 449],
            "episodes_per_checkpoint": 50,
            "expected_steps": list(EXPECTED_STEPS),
            "checkpoint_tie_break": "earliest checkpoint",
            "action_sequence": 8,
            "replan_interval": 8,
            "heldout_status": "sealed",
            "gate": "nstep8 best >=64% and improvement >=15pp",
        },
        "arms": arms,
        "contrasts": {"nstep8_minus_nstep1": delta},
        "selected_variant": selected_name,
        "selected_step": arms[selected_name]["best_step"],
        "selected_success": arms[selected_name]["best_success"],
        "chunk_horizon_gate": "pass" if gate_pass else "fail",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--treatment-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.control_run, args.treatment_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
