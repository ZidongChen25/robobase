#!/usr/bin/env python3
"""Evaluate Stage-16 sequence-aligned return Q on seeds 1 and 2."""

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
    baseline_seed1: Path,
    baseline_seed2: Path,
    treatment_seed1: Path,
    treatment_seed2: Path,
) -> dict:
    baselines = {
        "seed1": _arm(baseline_seed1),
        "seed2": _arm(baseline_seed2),
    }
    treatments = {
        "seed1": _arm(treatment_seed1),
        "seed2": _arm(treatment_seed2),
    }
    baseline_mean = sum(
        item["best_success"] for item in baselines.values()
    ) / 2.0
    treatment_mean = sum(
        item["best_success"] for item in treatments.values()
    ) / 2.0
    improvements = {
        seed: treatments[seed]["best_success"]
        - baselines[seed]["best_success"]
        for seed in treatments
    }
    both_at_least_60 = all(
        item["best_success"] >= 0.60 for item in treatments.values()
    )
    mean_improvement = treatment_mean - baseline_mean
    gate_pass = (
        both_at_least_60
        and treatment_mean >= 0.64
        and mean_improvement >= 0.15
    )
    return {
        "protocol": {
            "selection_seeds": [400, 449],
            "episodes_per_checkpoint": 50,
            "expected_steps": list(EXPECTED_STEPS),
            "checkpoint_tie_break": "earliest checkpoint",
            "sequence_aligned_mc_discount": 0.99,
            "heldout_status": "sealed",
            "gate": (
                "both treatment seeds >=60%, mean >=64%, "
                "mean improvement >=15pp"
            ),
        },
        "locked_baselines": baselines,
        "sequence_return_treatments": treatments,
        "baseline_mean": baseline_mean,
        "treatment_mean": treatment_mean,
        "per_seed_improvements": improvements,
        "mean_improvement": mean_improvement,
        "both_at_least_60": both_at_least_60,
        "sequence_return_gate": "pass" if gate_pass else "fail",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-seed1", type=Path, required=True)
    parser.add_argument("--baseline-seed2", type=Path, required=True)
    parser.add_argument("--treatment-seed1", type=Path, required=True)
    parser.add_argument("--treatment-seed2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.baseline_seed1,
        args.baseline_seed2,
        args.treatment_seed1,
        args.treatment_seed2,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
