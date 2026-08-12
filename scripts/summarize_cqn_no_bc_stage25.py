#!/usr/bin/env python3
"""Apply the matched three-seed 20k candidate-backup development gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.summarize_cqn_no_bc_stage23 import _extended_arm


_TOL = 1e-12


def _development_decision(
    improvements: dict[str, float],
) -> tuple[str, dict[str, float | int | bool]]:
    mean_improvement = sum(improvements.values()) / len(improvements)
    nonnegative_seed_count = sum(
        delta >= -_TOL for delta in improvements.values()
    )
    gate_pass = (
        mean_improvement >= 0.05 - _TOL
        and nonnegative_seed_count >= 2
    )
    decision = (
        "run_independent100_confirmation"
        if gate_pass
        else "stop_exact_candidate_only_variant_without_full_budget_claim"
    )
    return decision, {
        "mean_improvement": mean_improvement,
        "nonnegative_seed_count": nonnegative_seed_count,
        "development_gate_pass": gate_pass,
    }


def summarize(
    baseline_seed1: Path,
    baseline_seed2: Path,
    baseline_seed3: Path,
    treatment_seed1: Path,
    treatment_seed2: Path,
    treatment_seed3: Path,
) -> dict:
    baselines = {
        "seed1": _extended_arm(baseline_seed1, candidate=False),
        "seed2": _extended_arm(baseline_seed2, candidate=False),
        "seed3": _extended_arm(baseline_seed3, candidate=False),
    }
    treatments = {
        "seed1": _extended_arm(treatment_seed1, candidate=True),
        "seed2": _extended_arm(treatment_seed2, candidate=True),
        "seed3": _extended_arm(treatment_seed3, candidate=True),
    }
    improvements = {
        seed: treatments[seed]["best_success"]
        - baselines[seed]["best_success"]
        for seed in treatments
    }
    baseline_mean = sum(
        arm["best_success"] for arm in baselines.values()
    ) / len(baselines)
    treatment_mean = sum(
        arm["best_success"] for arm in treatments.values()
    ) / len(treatments)
    decision, flags = _development_decision(improvements)
    return {
        "protocol": {
            "development_only": True,
            "official_parity_claim": False,
            "selection_seeds": [400, 449],
            "episodes_per_checkpoint": 50,
            "selection_range": "2.5k--20k union",
            "checkpoint_tie_break": "earliest checkpoint",
            "gate": (
                "mean treatment-control delta >=5pp and >=2/3 "
                "deltas nonnegative"
            ),
            "independent_confirmation_if_pass": {
                "episodes_per_selected_checkpoint": 100,
                "eval_seeds": [43000, 43099],
            },
            "heldout_seeds_800_999": "sealed",
        },
        "locked_no_bc_controls": baselines,
        "candidate_backup_treatments": treatments,
        "baseline_mean": baseline_mean,
        "treatment_mean": treatment_mean,
        "per_seed_improvements": improvements,
        "mean_improvement": treatment_mean - baseline_mean,
        "decision_flags": flags,
        "next_decision": decision,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-seed1", type=Path, required=True)
    parser.add_argument("--baseline-seed2", type=Path, required=True)
    parser.add_argument("--baseline-seed3", type=Path, required=True)
    parser.add_argument("--treatment-seed1", type=Path, required=True)
    parser.add_argument("--treatment-seed2", type=Path, required=True)
    parser.add_argument("--treatment-seed3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.baseline_seed1,
        args.baseline_seed2,
        args.baseline_seed3,
        args.treatment_seed1,
        args.treatment_seed2,
        args.treatment_seed3,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
