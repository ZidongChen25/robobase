#!/usr/bin/env python3
"""Close the interrupted Stage-29 ordered-success 20k continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.summarize_cqn_no_bc_stage23 import _extended_arm


_TOL = 1e-12


def summarize(
    ordinary_seed1: Path,
    ordinary_seed2: Path,
    ordered_seed1: Path,
    ordered_seed2: Path,
) -> dict:
    ordinary = {
        "seed1": _extended_arm(ordinary_seed1, candidate=False),
        "seed2": _extended_arm(ordinary_seed2, candidate=False),
    }
    ordered = {
        "seed1": _extended_arm(ordered_seed1, candidate=False),
        "seed2": _extended_arm(ordered_seed2, candidate=False),
    }
    improvements = {
        seed: ordered[seed]["best_success"] - ordinary[seed]["best_success"]
        for seed in ordered
    }
    mean_improvement = sum(improvements.values()) / len(improvements)
    mechanism_pass = (
        mean_improvement >= 0.05 - _TOL
        and all(delta >= -_TOL for delta in improvements.values())
    )
    good_20k_boundary = {
        seed: (
            arm["new_curve"]["20000"] >= 0.50 - _TOL
            and arm["new_curve"]["20000"]
            >= arm["new_curve"]["17500"] - _TOL
        )
        for seed, arm in ordered.items()
    }
    scale_continuation = (
        any(good_20k_boundary.values())
        and mean_improvement >= -0.05 - _TOL
    )
    return {
        "protocol": {
            "development_only": True,
            "official_parity_claim": False,
            "selection_seeds": [400, 449],
            "episodes_per_checkpoint": 50,
            "selection_range": "2.5k--20k union",
            "checkpoint_tie_break": "earliest checkpoint",
            "mechanism_gate": (
                "mean ordered-success minus ordinary >=5pp and both "
                "deltas nonnegative"
            ),
            "scale_gate": (
                "at least one 20k endpoint >=50% and >=17.5k, while "
                "selected mean trails ordinary by <=5pp"
            ),
            "heldout_seeds_800_999": "sealed",
        },
        "matched_ordinary_no_bc_controls": ordinary,
        "ordered_success_treatments": ordered,
        "per_seed_improvement": improvements,
        "mean_improvement": mean_improvement,
        "decision_flags": {
            "mechanism_pass": mechanism_pass,
            "good_20k_boundary": good_20k_boundary,
            "scale_continuation": scale_continuation,
        },
        "next_decision": (
            "continue_ordered_success"
            if mechanism_pass or scale_continuation
            else "stop_ordered_success_without_full_budget_claim"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ordinary-seed1", type=Path, required=True)
    parser.add_argument("--ordinary-seed2", type=Path, required=True)
    parser.add_argument("--ordered-seed1", type=Path, required=True)
    parser.add_argument("--ordered-seed2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.ordinary_seed1,
        args.ordinary_seed2,
        args.ordered_seed1,
        args.ordered_seed2,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
