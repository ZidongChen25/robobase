#!/usr/bin/env python3
"""Summarize Stage-26 exact demo-trajectory then candidate-max runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.summarize_cqn_no_bc_stage23 import (
    CANDIDATE_FIELDS,
    EXTENDED_STEPS,
    SHORT_STEPS,
    _extended_arm,
)


ALL_STEPS = SHORT_STEPS + EXTENDED_STEPS
_TOL = 1e-12


def _read_full_curve(path: Path) -> dict[int, float]:
    curve = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                step = int(float(row["env_steps"]))
            except (TypeError, ValueError):
                continue
            if step in ALL_STEPS:
                curve[step] = float(row["episode_success"])
    missing = sorted(set(ALL_STEPS) - set(curve))
    if missing:
        raise ValueError(f"{path} is missing validation steps {missing}")
    return curve


def _selected_metrics(run_dir: Path, step: int) -> dict[str, float]:
    fields = CANDIDATE_FIELDS + (
        "demo_behavior_force_fraction",
        "demo_behavior_force_probability",
    )
    with (run_dir / "train.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = min(
        rows,
        key=lambda item: (
            abs(int(float(item["env_steps"])) - step),
            int(float(item["env_steps"])) > step,
        ),
    )
    missing = [field for field in fields if not row.get(field)]
    if missing:
        raise ValueError(
            f"{run_dir / 'train.csv'} is missing trajectory metrics {missing}"
        )
    return {
        "metric_env_steps": float(row["env_steps"]),
        **{field: float(row[field]) for field in fields},
    }


def _trajectory_arm(run_dir: Path) -> dict:
    curve = _read_full_curve(run_dir / "val50_seeds400.csv")
    best_step, best_success = max(
        curve.items(),
        key=lambda item: (item[1], -item[0]),
    )
    return {
        "run_dir": str(run_dir.resolve()),
        "curve": {str(step): value for step, value in curve.items()},
        "best_step": best_step,
        "best_success": best_success,
        "selected_snapshot": str(
            (run_dir / "snapshots" / f"{best_step}_snapshot.pkl").resolve()
        ),
        "selected_candidate_metrics": _selected_metrics(
            run_dir,
            best_step,
        ),
    }


def _decision(
    ordinary_improvements: dict[str, float],
    *,
    scale_continuation: bool = False,
) -> tuple[str, dict[str, float | bool]]:
    mean_improvement = sum(ordinary_improvements.values()) / len(
        ordinary_improvements
    )
    both_nonnegative = all(
        delta >= -_TOL for delta in ordinary_improvements.values()
    )
    strong_pass = mean_improvement >= 0.05 - _TOL and both_nonnegative
    mixed_positive = (
        mean_improvement > _TOL
        and any(delta > _TOL for delta in ordinary_improvements.values())
        and any(delta < -_TOL for delta in ordinary_improvements.values())
    )
    if strong_pass and scale_continuation:
        decision = "run_seed3_and_extend_seeds1_2_to50k"
    elif strong_pass:
        decision = "run_seed3_then_independent100_if_replicated"
    elif mixed_positive and scale_continuation:
        decision = "run_seed3_and_extend_seeds1_2_to50k_for_scale"
    elif mixed_positive:
        decision = "run_seed3_to_resolve_mixed_signs"
    elif scale_continuation:
        decision = "extend_seeds1_2_to50k_before_rejection"
    else:
        decision = (
            "stop_exact_force_then_candidate_variant_without_full_budget_claim"
        )
    return decision, {
        "mean_improvement_vs_ordinary": mean_improvement,
        "both_nonnegative": both_nonnegative,
        "strong_pass": strong_pass,
        "mixed_positive": mixed_positive,
        "scale_continuation": scale_continuation,
    }


def summarize(
    ordinary_seed1: Path,
    ordinary_seed2: Path,
    candidate_seed1: Path,
    candidate_seed2: Path,
    trajectory_seed1: Path,
    trajectory_seed2: Path,
) -> dict:
    ordinary = {
        "seed1": _extended_arm(ordinary_seed1, candidate=False),
        "seed2": _extended_arm(ordinary_seed2, candidate=False),
    }
    candidate = {
        "seed1": _extended_arm(candidate_seed1, candidate=True),
        "seed2": _extended_arm(candidate_seed2, candidate=True),
    }
    trajectory = {
        "seed1": _trajectory_arm(trajectory_seed1),
        "seed2": _trajectory_arm(trajectory_seed2),
    }
    versus_ordinary = {
        seed: trajectory[seed]["best_success"]
        - ordinary[seed]["best_success"]
        for seed in trajectory
    }
    versus_candidate = {
        seed: trajectory[seed]["best_success"]
        - candidate[seed]["best_success"]
        for seed in trajectory
    }
    mean_improvement = sum(versus_ordinary.values()) / len(versus_ordinary)
    good_20k_boundary = {
        seed: (
            arm["curve"]["20000"] >= 0.50 - _TOL
            and arm["curve"]["20000"] >= arm["curve"]["17500"] - _TOL
        )
        for seed, arm in trajectory.items()
    }
    scale_continuation = (
        any(good_20k_boundary.values())
        and mean_improvement >= -0.05 - _TOL
    )
    decision, flags = _decision(
        versus_ordinary,
        scale_continuation=scale_continuation,
    )
    flags["good_20k_boundary"] = good_20k_boundary
    return {
        "protocol": {
            "development_only": True,
            "official_parity_claim": False,
            "selection_seeds": [400, 449],
            "episodes_per_checkpoint": 50,
            "steps": list(ALL_STEPS),
            "checkpoint_tie_break": "earliest checkpoint",
            "phase_a": (
                "0--10.5k exact action_tp1 forced on demo Bellman backups"
            ),
            "phase_b": "10.5k--20k candidate-max with force probability zero",
            "optimized_objective": "single reward-based dense C51/MC Q loss",
            "gate": (
                "two-seed mean delta vs ordinary >=5pp and both nonnegative"
            ),
            "scale_gate": (
                "at least one 20k endpoint >=50% and >=17.5k, while "
                "two-seed selected mean trails ordinary by <=5pp"
            ),
            "heldout_seeds_800_999": "sealed",
        },
        "locked_ordinary_no_bc_controls": ordinary,
        "locked_candidate_only_controls": candidate,
        "trajectory_then_candidate_treatments": trajectory,
        "per_seed_improvement_vs_ordinary": versus_ordinary,
        "mean_improvement_vs_ordinary": mean_improvement,
        "per_seed_improvement_vs_candidate_only": versus_candidate,
        "mean_improvement_vs_candidate_only": (
            sum(versus_candidate.values()) / len(versus_candidate)
        ),
        "decision_flags": flags,
        "next_decision": decision,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ordinary-seed1", type=Path, required=True)
    parser.add_argument("--ordinary-seed2", type=Path, required=True)
    parser.add_argument("--candidate-seed1", type=Path, required=True)
    parser.add_argument("--candidate-seed2", type=Path, required=True)
    parser.add_argument("--trajectory-seed1", type=Path, required=True)
    parser.add_argument("--trajectory-seed2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.ordinary_seed1,
        args.ordinary_seed2,
        args.candidate_seed1,
        args.candidate_seed2,
        args.trajectory_seed1,
        args.trajectory_seed2,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
