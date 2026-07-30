"""Select Stage-1 no-BC checkpoints on the preregistered validation split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED_STEPS = (2500, 5000, 7500, 10000)


def _curve(run_dir: Path) -> dict[int, float]:
    path = run_dir / "val50_seeds400.csv"
    rows = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            step = int(float(row["env_steps"]))
            if step in EXPECTED_STEPS:
                rows[step] = float(row["episode_success"])
    missing = sorted(set(EXPECTED_STEPS) - set(rows))
    if missing:
        raise ValueError(f"{path} is missing validation steps {missing}")
    return rows


def _selected(curve: dict[int, float]) -> tuple[int, float]:
    # Earliest checkpoint breaks exact success-rate ties.
    return max(curve.items(), key=lambda item: (item[1], -item[0]))


def summarize(control_dir: Path, treatment_dir: Path) -> dict:
    control_curve = _curve(control_dir)
    treatment_curve = _curve(treatment_dir)
    control_step, control_success = _selected(control_curve)
    treatment_step, treatment_success = _selected(treatment_curve)
    delta = treatment_success - control_success
    mechanism_pass = treatment_success >= 0.20 and delta >= 0.10
    return {
        "protocol": {
            "selection_seeds": [400, 449],
            "episodes_per_checkpoint": 50,
            "expected_steps": list(EXPECTED_STEPS),
            "tie_break": "earliest checkpoint",
            "mechanism_pass": (
                "treatment best >= 20% and >= control best + 10pp"
            ),
        },
        "control": {
            "run_dir": str(control_dir.resolve()),
            "curve": {str(k): v for k, v in control_curve.items()},
            "best_step": control_step,
            "best_success": control_success,
        },
        "treatment": {
            "run_dir": str(treatment_dir.resolve()),
            "curve": {str(k): v for k, v in treatment_curve.items()},
            "best_step": treatment_step,
            "best_success": treatment_success,
        },
        "treatment_minus_control": delta,
        "mechanism_gate": "pass" if mechanism_pass else "fail",
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
