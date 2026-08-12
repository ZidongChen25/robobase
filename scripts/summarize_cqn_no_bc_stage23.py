#!/usr/bin/env python3
"""Summarize Stage-23 seed replication and matched 20k scale checks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SHORT_STEPS = (2500, 5000, 7500, 10000)
EXTENDED_STEPS = (12500, 15000, 17500, 20000)
CANDIDATE_FIELDS = (
    "behavior_candidate_fraction",
    "behavior_candidate_score",
    "greedy_candidate_score",
    "behavior_minus_greedy_q",
    "chosen_q_mean",
    "unseen_q_mean",
    "chosen_unseen_q_gap",
)
_TOL = 1e-12


def _read_curve(
    path: Path,
    expected_steps: tuple[int, ...],
) -> dict[int, float]:
    curve = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                step = int(float(row["env_steps"]))
            except (TypeError, ValueError):
                continue
            if step in expected_steps:
                curve[step] = float(row["episode_success"])
    missing = sorted(set(expected_steps) - set(curve))
    if missing:
        raise ValueError(f"{path} is missing validation steps {missing}")
    return curve


def _best(curve: dict[int, float]) -> tuple[int, float]:
    return max(curve.items(), key=lambda item: (item[1], -item[0]))


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


def _short_arm(run_dir: Path, *, candidate: bool) -> dict:
    curve = _read_curve(run_dir / "val50_seeds400.csv", SHORT_STEPS)
    best_step, best_success = _best(curve)
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


def _extended_arm(run_dir: Path, *, candidate: bool) -> dict:
    old_curve = _read_curve(run_dir / "val50_seeds400.csv", SHORT_STEPS)
    new_curve = _read_curve(
        run_dir / "val50_ext20k_seeds400.csv",
        EXTENDED_STEPS,
    )
    combined = old_curve | new_curve
    best_step, best_success = _best(combined)
    result = {
        "run_dir": str(run_dir.resolve()),
        "old_curve": {
            str(step): value for step, value in old_curve.items()
        },
        "new_curve": {
            str(step): value for step, value in new_curve.items()
        },
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


def _decisions(
    short_improvements: dict[str, float],
    *,
    scale_delta: float,
    scale_best_step: int,
) -> tuple[str, dict[str, bool | int | float]]:
    short_mean_delta = sum(short_improvements.values()) / len(
        short_improvements
    )
    nonnegative_seeds = sum(
        delta >= -_TOL for delta in short_improvements.values()
    )
    replication_pass = (
        short_mean_delta >= 0.05 - _TOL and nonnegative_seeds >= 2
    )
    scale_pass = (
        scale_delta >= 0.05 - _TOL and scale_best_step > 10000
    )
    if replication_pass and scale_pass:
        decision = "run_independent100_and_extend_candidate_seeds1_3"
    elif replication_pass:
        decision = "run_independent100_confirmation"
    elif scale_pass:
        decision = "extend_candidate_seeds1_3_before_confirmation"
    else:
        decision = (
            "stop_exact_candidate_only_variant_without_full_budget_claim"
        )
    return decision, {
        "three_seed_mean_delta": short_mean_delta,
        "nonnegative_seed_count": nonnegative_seeds,
        "replication_pass": replication_pass,
        "scale_delta": scale_delta,
        "scale_best_after_10k": scale_best_step > 10000,
        "scale_pass": scale_pass,
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
        "seed1": _short_arm(baseline_seed1, candidate=False),
        "seed2": _short_arm(baseline_seed2, candidate=False),
        "seed3": _short_arm(baseline_seed3, candidate=False),
    }
    treatments = {
        "seed1": _short_arm(treatment_seed1, candidate=True),
        "seed2": _short_arm(treatment_seed2, candidate=True),
        "seed3": _short_arm(treatment_seed3, candidate=True),
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

    scale_baseline = _extended_arm(baseline_seed2, candidate=False)
    scale_treatment = _extended_arm(treatment_seed2, candidate=True)
    scale_delta = (
        scale_treatment["best_success"] - scale_baseline["best_success"]
    )
    next_decision, decision_flags = _decisions(
        improvements,
        scale_delta=scale_delta,
        scale_best_step=scale_treatment["best_step"],
    )

    return {
        "protocol": {
            "development_only": True,
            "official_parity_claim": False,
            "selection_seeds": [400, 449],
            "episodes_per_checkpoint": 50,
            "short_steps": list(SHORT_STEPS),
            "extended_steps": list(EXTENDED_STEPS),
            "checkpoint_tie_break": "earliest checkpoint",
            "replication_gate": (
                "three-seed mean delta >=5pp and >=2/3 deltas nonnegative"
            ),
            "scale_gate": (
                "seed2 extended-best delta >=5pp and selected after 10k"
            ),
            "heldout_seeds_800_999": "sealed",
            "full_run_reference": {
                "budget": 101000,
                "fixed_endpoint": True,
                "episodes_per_training_seed": 200,
                "official_four_seed_mean": 0.646,
            },
        },
        "short_replication": {
            "locked_no_bc_controls": baselines,
            "candidate_backup_treatments": treatments,
            "baseline_mean": baseline_mean,
            "treatment_mean": treatment_mean,
            "per_seed_improvements": improvements,
            "mean_improvement": treatment_mean - baseline_mean,
        },
        "seed2_scale_check": {
            "locked_no_bc_control": scale_baseline,
            "candidate_backup_treatment": scale_treatment,
            "improvement": scale_delta,
        },
        "decision_flags": decision_flags,
        "next_decision": next_decision,
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
