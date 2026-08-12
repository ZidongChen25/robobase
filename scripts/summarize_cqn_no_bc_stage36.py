#!/usr/bin/env python3
"""Summarize the Stage-36 batch-256 dense No-BC replication."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED_STEPS = (2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000)
REFERENCE_BEST = {"seed1": 0.60, "seed2": 0.46, "seed3": 0.56}


def _arm(run_dir: Path) -> dict:
    curve: dict[int, float] = {}
    csv_path = run_dir / "val50_seeds400_steps.csv"
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            step = int(float(row["env_steps"]))
            if step in EXPECTED_STEPS:
                curve[step] = float(row["episode_success"])
    missing = sorted(set(EXPECTED_STEPS) - set(curve))
    if missing:
        raise ValueError(f"{csv_path} is missing validation steps {missing}")
    best_step, best_success = max(curve.items(), key=lambda item: (item[1], -item[0]))
    return {
        "run_dir": str(run_dir.resolve()),
        "curve": {str(step): curve[step] for step in EXPECTED_STEPS},
        "best_step": best_step,
        "best_success": best_success,
        "selected_snapshot": str(
            (run_dir / "snapshots" / f"{best_step}_snapshot.pkl").resolve()
        ),
    }


def summarize(base: Path) -> dict:
    arms = {f"seed{s}": _arm(base / f"dense_b256_seed{s}") for s in (1, 2, 3)}
    mean_best = sum(arm["best_success"] for arm in arms.values()) / len(arms)
    deltas = {
        seed: arms[seed]["best_success"] - reference
        for seed, reference in REFERENCE_BEST.items()
    }
    seeds_at_least_40 = sum(arm["best_success"] >= 0.40 for arm in arms.values())
    replication_pass = mean_best >= 0.49 and seeds_at_least_40 >= 2
    return {
        "protocol": {
            "research_question": (
                "Does the retained dense No-BC baseline reproduce at the original "
                "CQN-AS batch scale of 256 ordinary plus 256 demo samples?"
            ),
            "training_seeds": [1, 2, 3],
            "online_environment_steps": 20000,
            "batch_size": 256,
            "demo_batch_size": 256,
            "selection_seeds": [400, 449],
            "episodes_per_checkpoint": 50,
            "selection_steps": list(EXPECTED_STEPS),
            "checkpoint_tie_break": "earliest checkpoint",
            "heldout_seeds_800_999": "sealed",
            "replication_gate": (
                "three-seed validation-best mean >=49% and at least two seeds >=40%"
            ),
        },
        "historical_b16_reference_best": REFERENCE_BEST,
        "dense_b256": arms,
        "per_seed_delta_vs_historical_b16": deltas,
        "mean_validation_best": mean_best,
        "seeds_at_least_40pct": seeds_at_least_40,
        "replication_pass": replication_pass,
        "next_decision": (
            "promote_dense_b256_to_matched_101k_no_bc_vs_bc"
            if replication_pass
            else "do_not_claim_batch_scaled_dense_no_bc_replication"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.base)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
