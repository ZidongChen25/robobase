#!/usr/bin/env python3
"""Summarize the Stage-31 direct per-bin C51 head experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from scripts.summarize_cqn_no_bc_stage30 import (
    ONLINE_STEPS,
    _phase_contract,
    _treatment_arm,
)


_TOL = 1e-12


def _use_dueling(run_dir: Path, seed: int, expected: bool) -> dict[str, object]:
    phase_dir = run_dir.parent / "phase_configs"
    values = {}
    for phase in ("offline", "online"):
        path = phase_dir / f"{phase}_seed{seed}.yaml"
        cfg = OmegaConf.load(path)
        actual = bool(cfg.method.use_dueling)
        if actual is not expected:
            raise ValueError(
                f"{path}: method.use_dueling={actual}, expected {expected}"
            )
        if bool(cfg.method.mc_return_value_only):
            raise ValueError(f"{path}: mc_return_value_only must remain false")
        values[phase] = {
            "config": str(path.resolve()),
            "use_dueling": actual,
            "mc_return_value_only": False,
        }
    return values


def _decision(
    improvements: dict[str, float],
    direct: dict[str, dict],
) -> tuple[str, dict[str, object]]:
    mean_improvement = sum(improvements.values()) / len(improvements)
    both_nonnegative = all(delta >= -_TOL for delta in improvements.values())
    any_seed_at_least_20 = any(
        arm["best_success"] >= 0.20 - _TOL for arm in direct.values()
    )
    mechanism_pass = (
        mean_improvement >= 0.10 - _TOL
        and both_nonnegative
        and any_seed_at_least_20
    )
    good_boundary = {
        seed: (
            arm["online_curve"]["20000"] >= 0.20 - _TOL
            and arm["online_curve"]["20000"]
            >= arm["online_curve"]["17500"] - _TOL
        )
        for seed, arm in direct.items()
    }
    scale_continuation = (
        mean_improvement >= 0.05 - _TOL and any(good_boundary.values())
    )
    if mechanism_pass:
        decision = "run_direct_head_seed3_then_update_matched_confirmation"
    elif scale_continuation:
        decision = "extend_direct_head_to50k_before_rejection"
    else:
        decision = "stop_direct_head_and_test_pessimistic_double_q"
    return decision, {
        "mean_improvement": mean_improvement,
        "both_nonnegative": both_nonnegative,
        "any_seed_at_least_20pct": any_seed_at_least_20,
        "mechanism_pass": mechanism_pass,
        "good_20k_boundary": good_boundary,
        "scale_continuation": scale_continuation,
    }


def summarize(
    stage30_summary: Path,
    direct_seed1: Path,
    direct_seed2: Path,
) -> dict:
    baseline_document = json.loads(stage30_summary.read_text())
    baseline = baseline_document["offline_then_online_treatments"]
    direct = {
        "seed1": _treatment_arm(direct_seed1),
        "seed2": _treatment_arm(direct_seed2),
    }
    contracts = {
        "direct_seed1": _phase_contract(direct_seed1, 1, treatment=True),
        "direct_seed2": _phase_contract(direct_seed2, 2, treatment=True),
    }
    direct_head_contract = {
        "seed1": _use_dueling(direct_seed1, 1, False),
        "seed2": _use_dueling(direct_seed2, 2, False),
    }
    # Stage 30 is immutable and its summary already verifies the reward-only
    # phase contract. Check the one isolated architecture difference from the
    # archived phase configs as well.
    baseline_dirs = {
        seed: Path(arm["run_dir"]) for seed, arm in baseline.items()
    }
    baseline_head_contract = {
        "seed1": _use_dueling(baseline_dirs["seed1"], 1, True),
        "seed2": _use_dueling(baseline_dirs["seed2"], 2, True),
    }
    improvements = {
        seed: direct[seed]["best_success"] - baseline[seed]["best_success"]
        for seed in direct
    }
    baseline_mean = sum(
        arm["best_success"] for arm in baseline.values()
    ) / len(baseline)
    direct_mean = sum(arm["best_success"] for arm in direct.values()) / len(direct)
    decision, flags = _decision(improvements, direct)
    return {
        "protocol": {
            "development_only": True,
            "research_question": (
                "Does removing the shared dueling value stream fix expert "
                "action ranking after reward-only offline Q learning?"
            ),
            "only_method_difference": "method.use_dueling: true -> false",
            "optimized_objective": (
                "single canonical reward-based C51 TD/MC cross-entropy"
            ),
            "selection_seeds": [400, 449],
            "episodes_per_checkpoint": 50,
            "online_selection_steps": list(ONLINE_STEPS),
            "checkpoint_tie_break": "earliest checkpoint",
            "heldout_seeds_800_999": "sealed",
            "mechanism_gate": (
                "mean selected gain >=10pp, both seed deltas nonnegative, "
                "and at least one seed >=20%"
            ),
            "scale_gate": (
                "mean selected gain >=5pp and any 20k endpoint >=20% and "
                "nondecreasing from 17.5k"
            ),
        },
        "stage30_summary": str(stage30_summary.resolve()),
        "phase_contracts": contracts,
        "baseline_dueling_contract": baseline_head_contract,
        "direct_head_contract": direct_head_contract,
        "dueling_offline_then_online_baseline": baseline,
        "direct_head_offline_then_online": direct,
        "baseline_selected_mean": baseline_mean,
        "direct_selected_mean": direct_mean,
        "per_seed_improvement": improvements,
        "mean_improvement": direct_mean - baseline_mean,
        "decision_flags": flags,
        "next_decision": decision,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage30-summary", type=Path, required=True)
    parser.add_argument("--direct-seed1", type=Path, required=True)
    parser.add_argument("--direct-seed2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.stage30_summary,
        args.direct_seed1,
        args.direct_seed2,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
