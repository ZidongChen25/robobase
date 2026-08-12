#!/usr/bin/env python3
"""Summarize the Stage-22 development-only candidate-backup screen."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED_STEPS = (2500, 5000, 7500, 10000)
CANDIDATE_FIELDS = (
    "behavior_candidate_fraction",
    "behavior_candidate_score",
    "greedy_candidate_score",
    "behavior_minus_greedy_q",
    "chosen_q_mean",
    "unseen_q_mean",
    "chosen_unseen_q_gap",
)


def _selected_metrics(run_dir: Path, step: int) -> dict[str, float]:
    path = run_dir / "train.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no training metrics")
    row = min(
        rows,
        key=lambda item: (
            abs(int(float(item["env_steps"])) - step),
            int(float(item["env_steps"])) > step,
        ),
    )
    missing = [field for field in CANDIDATE_FIELDS if not row.get(field)]
    if missing:
        raise ValueError(f"{path} is missing candidate metrics {missing}")
    return {
        "metric_env_steps": float(row["env_steps"]),
        **{field: float(row[field]) for field in CANDIDATE_FIELDS},
    }


def _arm(run_dir: Path, *, candidate: bool) -> dict:
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
    result = {
        "run_dir": str(run_dir.resolve()),
        "curve": {str(step): value for step, value in curve.items()},
        "best_step": best_step,
        "best_success": best_success,
        "selected_snapshot": str(
            (run_dir / "snapshots" / f"{best_step}_snapshot.pkl").resolve()
        ),
    }
    if candidate:
        result["selected_candidate_metrics"] = _selected_metrics(
            run_dir,
            best_step,
        )
    return result


def _development_decision(
    treatments: dict[str, dict],
    improvements: dict[str, float],
) -> tuple[str, dict[str, bool]]:
    mean_improvement = sum(improvements.values()) / len(improvements)
    immediate_replication = (
        mean_improvement >= 0.05
        and all(delta >= 0.0 for delta in improvements.values())
    )
    signs_disagree = (
        any(delta > 0.0 for delta in improvements.values())
        and any(delta < 0.0 for delta in improvements.values())
    )
    add_seed3 = mean_improvement > 0.0 and signs_disagree
    rising_at_boundary = any(
        arm["best_step"] == 10000
        and arm["curve"]["10000"] >= arm["curve"]["7500"]
        for arm in treatments.values()
    )
    continue_20k = mean_improvement >= -0.05 and rising_at_boundary

    if immediate_replication:
        decision = "run_seed3_and_independent_dev_confirmation"
    elif add_seed3:
        decision = "run_seed3_to_resolve_mixed_signs"
    elif continue_20k:
        decision = "continue_treatments_to_20k"
    else:
        decision = "stop_development_candidate_without_full_budget_claim"
    return decision, {
        "immediate_replication": immediate_replication,
        "mixed_sign_seed3": add_seed3,
        "rising_at_10k": rising_at_boundary,
        "scale_continuation": continue_20k,
    }


def summarize(
    baseline_seed1: Path,
    baseline_seed2: Path,
    treatment_seed1: Path,
    treatment_seed2: Path,
) -> dict:
    baselines = {
        "seed1": _arm(baseline_seed1, candidate=False),
        "seed2": _arm(baseline_seed2, candidate=False),
    }
    treatments = {
        "seed1": _arm(treatment_seed1, candidate=True),
        "seed2": _arm(treatment_seed2, candidate=True),
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
    decision, decision_flags = _development_decision(
        treatments,
        improvements,
    )
    return {
        "protocol": {
            "development_only": True,
            "official_parity_claim": False,
            "selection_seeds": [400, 449],
            "episodes_per_checkpoint": 50,
            "expected_steps": list(EXPECTED_STEPS),
            "checkpoint_tie_break": "earliest checkpoint",
            "q_objective": "dense_return_c51_with_replay_candidate_bellman_max",
            "candidate_score": "deepest C2F level mean over K*action_dim",
            "candidate_set": ["critic_greedy", "replay_action_tp1"],
            "demo_flag_used_by_update": False,
            "heldout_seeds_800_999": "sealed",
            "full_run_reference": {
                "budget": 101000,
                "fixed_endpoint": True,
                "episodes_per_training_seed": 200,
                "official_four_seed_mean": 0.646,
            },
        },
        "locked_no_bc_controls": baselines,
        "candidate_backup_treatments": treatments,
        "baseline_mean": baseline_mean,
        "treatment_mean": treatment_mean,
        "per_seed_improvements": improvements,
        "mean_improvement": treatment_mean - baseline_mean,
        "decision_flags": decision_flags,
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
