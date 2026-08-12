#!/usr/bin/env python3
"""Summarize Stage-24 candidate seed-1/3 extensions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.summarize_cqn_no_bc_stage23 import (
    _extended_arm,
    _short_arm,
)


_TOL = 1e-12


def _scale_replication(delta: float, best_step: int) -> str:
    if delta >= 0.05 - _TOL and best_step > 10000:
        return "strong_replication"
    if delta >= -_TOL:
        return "partial_nonnegative_replication"
    return "not_replicated_on_seed3"


def summarize(
    baseline_seed1: Path,
    baseline_seed3: Path,
    treatment_seed1: Path,
    treatment_seed3: Path,
) -> dict:
    seed1_control = _short_arm(baseline_seed1, candidate=False)
    seed1_treatment = _extended_arm(treatment_seed1, candidate=True)
    seed3_control = _extended_arm(baseline_seed3, candidate=False)
    seed3_treatment = _extended_arm(treatment_seed3, candidate=True)

    seed3_delta = (
        seed3_treatment["best_success"] - seed3_control["best_success"]
    )
    seed3_replication = _scale_replication(
        seed3_delta,
        seed3_treatment["best_step"],
    )
    seed1_old_best = max(seed1_treatment["old_curve"].values())
    return {
        "protocol": {
            "development_only": True,
            "official_parity_claim": False,
            "selection_seeds": [400, 449],
            "episodes_per_checkpoint": 50,
            "selection_range": "2.5k--20k union",
            "checkpoint_tie_break": "earliest checkpoint",
            "seed3_strong_replication_gate": (
                "candidate-control delta >=5pp and selected after 10k"
            ),
            "seed1_comparison_status": (
                "candidate curve locked; matched 20k control pending"
            ),
            "heldout_seeds_800_999": "sealed",
        },
        "seed3_matched_scale_replication": {
            "locked_no_bc_control": seed3_control,
            "candidate_backup_treatment": seed3_treatment,
            "improvement": seed3_delta,
            "replication": seed3_replication,
        },
        "seed1_candidate_extension": {
            "locked_short_no_bc_control": seed1_control,
            "candidate_backup_treatment": seed1_treatment,
            "candidate_gain_over_own_10k_best": (
                seed1_treatment["best_success"] - seed1_old_best
            ),
            "post10k_method_delta": "pending matched control extension",
        },
        "next_decision": (
            "extend_seed1_no_bc_control_then_apply_three_seed_gate"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-seed1", type=Path, required=True)
    parser.add_argument("--baseline-seed3", type=Path, required=True)
    parser.add_argument("--treatment-seed1", type=Path, required=True)
    parser.add_argument("--treatment-seed3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.baseline_seed1,
        args.baseline_seed3,
        args.treatment_seed1,
        args.treatment_seed3,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
