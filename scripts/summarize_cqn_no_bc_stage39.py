#!/usr/bin/env python3
"""Summarize the Stage-39 reward-gated dense-Q validation experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


STEPS = (2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000)


def curve(path: Path) -> dict[int, float]:
    values = {
        int(float(row["env_steps"])): float(row["episode_success"])
        for row in csv.DictReader(path.open(newline=""))
    }
    missing = sorted(set(STEPS) - set(values))
    if missing:
        raise ValueError(f"{path} is missing steps {missing}")
    return {step: values[step] for step in STEPS}


def selected(values: dict[int, float]) -> tuple[int, float]:
    return max(values.items(), key=lambda item: (item[1], -item[0]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--baseline-run-base", type=Path, required=True)
    args = parser.parse_args()

    treatments = {}
    baselines = {}
    deltas = {}
    for seed in (1, 2):
        treatment_run = args.stage_dir / f"seed{seed}"
        baseline_run = args.baseline_run_base / f"dense_b256_seed{seed}"
        treatment_curve = curve(treatment_run / "val50_seeds400_steps.csv")
        baseline_curve = curve(baseline_run / "val50_seeds400_steps.csv")
        treatment_step, treatment_best = selected(treatment_curve)
        baseline_step, baseline_best = selected(baseline_curve)
        label = f"seed{seed}"
        treatments[label] = {
            "curve": {str(k): v for k, v in treatment_curve.items()},
            "best_step": treatment_step,
            "best_success": treatment_best,
        }
        baselines[label] = {
            "curve": {str(k): v for k, v in baseline_curve.items()},
            "best_step": baseline_step,
            "best_success": baseline_best,
        }
        deltas[label] = treatment_best - baseline_best

    treatment_mean = sum(v["best_success"] for v in treatments.values()) / 2
    baseline_mean = sum(v["best_success"] for v in baselines.values()) / 2
    mechanism_pass = (
        all(delta >= 0.0 for delta in deltas.values())
        and treatment_mean - baseline_mean >= 0.05
    )
    result = {
        "protocol": {
            "training_seeds": [1, 2],
            "selection_seeds": [400, 449],
            "episodes_per_checkpoint": 50,
            "selection_steps": list(STEPS),
            "checkpoint_tie_break": "earliest checkpoint",
            "heldout_seeds_800_999": "sealed",
            "pass": "both seed deltas nonnegative and mean gain >=5pp",
        },
        "locked_stage37_baselines": baselines,
        "positive_return_dense_treatments": treatments,
        "per_seed_deltas": deltas,
        "baseline_mean": baseline_mean,
        "treatment_mean": treatment_mean,
        "mean_gain": treatment_mean - baseline_mean,
        "mechanism_pass": mechanism_pass,
        "next_decision": (
            "run_seed3_confirmation" if mechanism_pass else "reject_mechanism"
        ),
    }
    output = args.stage_dir / "stage39_summary.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
