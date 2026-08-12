#!/usr/bin/env python3
"""Summarize the matched Stage-27 reward-scale 20k continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.summarize_cqn_no_bc_stage23 import _extended_arm


_TOL = 1e-12


def _decision(
    improvements: dict[str, float],
    treatment_arms: dict[str, dict],
) -> tuple[str, dict]:
    mean_improvement = sum(improvements.values()) / len(improvements)
    both_nonnegative = all(delta >= -_TOL for delta in improvements.values())
    mechanism_pass = mean_improvement >= 0.05 - _TOL and both_nonnegative
    good_20k_boundary = {
        seed: (
            float(arm["new_curve"]["20000"]) >= 0.50 - _TOL
            and float(arm["new_curve"]["20000"])
            >= float(arm["new_curve"]["17500"]) - _TOL
        )
        for seed, arm in treatment_arms.items()
    }
    scale_continuation = (
        any(good_20k_boundary.values())
        and mean_improvement >= -0.05 - _TOL
    )
    if mechanism_pass and scale_continuation:
        decision = "extend_reward_scale_seeds1_2_to50k"
    elif mechanism_pass:
        decision = "run_reward_scale_seed3"
    elif scale_continuation:
        decision = "extend_reward_scale_seeds1_2_to50k_for_scale"
    else:
        decision = "stop_reward_scale_variant_without_full_budget_claim"
    return decision, {
        "mean_improvement_vs_ordinary": mean_improvement,
        "both_nonnegative": both_nonnegative,
        "mechanism_pass": mechanism_pass,
        "good_20k_boundary": good_20k_boundary,
        "scale_continuation": scale_continuation,
    }


def summarize(
    baseline_seed1: Path,
    baseline_seed2: Path,
    treatment_seed1: Path,
    treatment_seed2: Path,
) -> dict:
    baselines = {
        "seed1": _extended_arm(baseline_seed1, candidate=False),
        "seed2": _extended_arm(baseline_seed2, candidate=False),
    }
    treatments = {
        "seed1": _extended_arm(treatment_seed1, candidate=False),
        "seed2": _extended_arm(treatment_seed2, candidate=False),
    }
    improvements = {
        seed: treatments[seed]["best_success"]
        - baselines[seed]["best_success"]
        for seed in treatments
    }
    decision, flags = _decision(improvements, treatments)
    return {
        "protocol": {
            "development_only": True,
            "official_parity_claim": False,
            "selection_seeds": [400, 449],
            "episodes_per_checkpoint": 50,
            "selection_range": "2.5k--20k union",
            "checkpoint_tie_break": "earliest checkpoint",
            "mechanism_gate": (
                "mean treatment-control delta >=5pp and both nonnegative"
            ),
            "scale_gate": (
                "at least one 20k endpoint >=50% and >=17.5k, while "
                "two-seed selected mean trails ordinary by <=5pp"
            ),
            "optimized_objective": "single reward-scaled dense C51/MC Q loss",
            "heldout_seeds_800_999": "sealed",
        },
        "matched_ordinary_no_bc_controls": baselines,
        "reward_scale_treatments": treatments,
        "per_seed_improvement_vs_ordinary": improvements,
        "mean_improvement_vs_ordinary": flags[
            "mean_improvement_vs_ordinary"
        ],
        "decision_flags": flags,
        "next_decision": decision,
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
