#!/usr/bin/env python3
"""Summarize batch-256 replication and the optional 101k dense No-BC run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SHORT_STEPS = (2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000)
FULL_STEPS = (20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 101000)
REFERENCE_BEST = {"seed1": 0.60, "seed2": 0.46, "seed3": 0.56}


def _curve(path: Path, steps: tuple[int, ...]) -> dict[int, float]:
    values: dict[int, float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            step = int(float(row["env_steps"]))
            if step in steps:
                values[step] = float(row["episode_success"])
    missing = sorted(set(steps) - set(values))
    if missing:
        raise ValueError(f"{path} is missing steps {missing}")
    return values


def short_summary(run_base: Path) -> dict:
    arms = {}
    for seed in (1, 2, 3):
        run_dir = run_base / f"dense_b256_seed{seed}"
        curve = _curve(run_dir / "val50_seeds400_steps.csv", SHORT_STEPS)
        best_step, best_success = max(curve.items(), key=lambda item: (item[1], -item[0]))
        arms[f"seed{seed}"] = {
            "run_dir": str(run_dir.resolve()),
            "curve": {str(step): curve[step] for step in SHORT_STEPS},
            "best_step": best_step,
            "best_success": best_success,
            "late_best_17p5k_20k": max(curve[17500], curve[20000]),
            "selected_snapshot": str(
                (run_dir / "snapshots" / f"{best_step}_snapshot.pkl").resolve()
            ),
        }
    mean_best = sum(arm["best_success"] for arm in arms.values()) / 3
    seeds_at_least_40 = sum(arm["best_success"] >= 0.40 for arm in arms.values())
    late_seeds_at_least_40 = sum(
        arm["late_best_17p5k_20k"] >= 0.40 for arm in arms.values()
    )
    pass_gate = (
        mean_best >= 0.49
        and seeds_at_least_40 >= 2
        and late_seeds_at_least_40 >= 2
    )
    return {
        "protocol": {
            "research_question": "Does the retained online-only dense No-BC baseline reproduce at batch 256+256?",
            "training_seeds": [1, 2, 3],
            "online_environment_steps": 20000,
            "batch_size": 256,
            "demo_batch_size": 256,
            "selection_seeds": [400, 449],
            "episodes_per_checkpoint": 50,
            "selection_steps": list(SHORT_STEPS),
            "checkpoint_tie_break": "earliest checkpoint",
            "heldout_seeds_800_999": "sealed_until_gate_passes",
            "promotion_gate": (
                "mean validation-best >=49%, at least 2/3 best >=40%, and "
                "at least 2/3 have >=40% at 17.5k or 20k"
            ),
        },
        "historical_b16_reference_best": REFERENCE_BEST,
        "dense_b256": arms,
        "mean_validation_best": mean_best,
        "seeds_at_least_40pct": seeds_at_least_40,
        "late_seeds_at_least_40pct": late_seeds_at_least_40,
        "promotion_pass": pass_gate,
        "next_decision": (
            "continue_preregistered_seed1_to_fixed_101k"
            if pass_gate
            else "stop_batch256_dense_without_full_run"
        ),
    }


def full_summary(run_base: Path) -> dict:
    run_dir = run_base / "dense_b256_seed1"
    validation = _curve(run_dir / "val50_seeds400_full.csv", FULL_STEPS)
    heldout = _curve(run_dir / "heldout200_seeds800_endpoint.csv", (101000,))
    best_step, best_success = max(
        validation.items(), key=lambda item: (item[1], -item[0])
    )
    endpoint = heldout[101000]
    return {
        "protocol": {
            "training_seed": 1,
            "online_environment_steps": 101000,
            "checkpoint_selection": "descriptive validation curve only",
            "final_metric": "fixed raw-101k checkpoint on 200 episodes, seeds 800--999",
            "official_seed1_reference": 0.62,
        },
        "validation_curve": {str(step): validation[step] for step in FULL_STEPS},
        "validation_best_step": best_step,
        "validation_best_success": best_success,
        "heldout_fixed_101k_success": endpoint,
        "delta_vs_official_seed1": endpoint - 0.62,
        "next_decision": (
            "run_remaining_training_seeds_to_101k"
            if endpoint >= 0.62
            else "do_not_claim_full_scale_parity_from_seed1"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-base", type=Path, required=True)
    parser.add_argument("--mode", choices=("short", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = short_summary(args.run_base) if args.mode == "short" else full_summary(args.run_base)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
