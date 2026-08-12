#!/usr/bin/env python3
"""Add a fresh reward-scale seed-3 replication to the Stage-27 result."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.summarize_cqn_no_bc_stage23 import (
    EXTENDED_STEPS,
    SHORT_STEPS,
    _extended_arm,
)


ALL_STEPS = SHORT_STEPS + EXTENDED_STEPS
_TOL = 1e-12


def _full_arm(run_dir: Path) -> dict:
    curve = {}
    paths = [run_dir / "val50_seeds400.csv"]
    for shard_name in (
        "val50_upper_seeds400.csv",
        "val50_tail_seeds400.csv",
    ):
        shard = run_dir / shard_name
        if shard.exists():
            paths.append(shard)
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                step = int(float(row["env_steps"]))
                if step in ALL_STEPS:
                    curve[step] = float(row["episode_success"])
    missing = sorted(set(ALL_STEPS) - set(curve))
    if missing:
        raise ValueError(f"{run_dir} is missing validation steps {missing}")
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
    }


def _decision(
    improvements: dict[str, float],
    *,
    stage27_scale_continuation: bool,
) -> tuple[str, dict]:
    mean_improvement = sum(improvements.values()) / len(improvements)
    nonnegative_seed_count = sum(
        delta >= -_TOL for delta in improvements.values()
    )
    replication_pass = (
        mean_improvement >= 0.05 - _TOL
        and nonnegative_seed_count >= 2
    )
    if replication_pass:
        decision = "run_independent100_and_extend_seeds1_2_3_to50k"
    elif stage27_scale_continuation:
        decision = "extend_seeds1_2_to50k_without_replication_claim"
    else:
        decision = "stop_reward_scale_variant_without_full_budget_claim"
    return decision, {
        "three_seed_mean_improvement": mean_improvement,
        "nonnegative_seed_count": nonnegative_seed_count,
        "replication_pass": replication_pass,
        "stage27_scale_continuation": stage27_scale_continuation,
    }


def summarize(
    stage27_summary: Path,
    baseline_seed3: Path,
    treatment_seed3: Path,
) -> dict:
    stage27 = json.loads(stage27_summary.read_text())
    baselines = dict(stage27["matched_ordinary_no_bc_controls"])
    treatments = dict(stage27["reward_scale_treatments"])
    baselines["seed3"] = _extended_arm(baseline_seed3, candidate=False)
    treatments["seed3"] = _full_arm(treatment_seed3)
    improvements = {
        seed: treatments[seed]["best_success"]
        - baselines[seed]["best_success"]
        for seed in treatments
    }
    decision, flags = _decision(
        improvements,
        stage27_scale_continuation=bool(
            stage27["decision_flags"]["scale_continuation"]
        ),
    )
    return {
        "protocol": {
            "development_only": True,
            "official_parity_claim": False,
            "selection_seeds": [400, 449],
            "episodes_per_checkpoint": 50,
            "selection_range": "2.5k--20k union",
            "checkpoint_tie_break": "earliest checkpoint",
            "replication_gate": (
                "three-seed mean delta >=5pp and >=2/3 nonnegative"
            ),
            "replication_pass_action": (
                "independent100 confirmation and matched 50k extension "
                "of reward-scale seeds1/2/3"
            ),
            "independent_confirmation_if_pass": {
                "episodes_per_selected_checkpoint": 100,
                "eval_seeds": [43000, 43099],
            },
            "heldout_seeds_800_999": "sealed",
        },
        "matched_ordinary_no_bc_controls": baselines,
        "reward_scale_treatments": treatments,
        "per_seed_improvement_vs_ordinary": improvements,
        "decision_flags": flags,
        "next_decision": decision,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage27-summary", type=Path, required=True)
    parser.add_argument("--baseline-seed3", type=Path, required=True)
    parser.add_argument("--treatment-seed3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.stage27_summary,
        args.baseline_seed3,
        args.treatment_seed3,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
