#!/usr/bin/env python3
"""Summarize the progressive Stage-36 offline-to-online quality gate.

The 20-episode treatment-only curve is a futility screen.  It never selects a
checkpoint for a paper comparison.  A non-zero coarse curve unlocks a matched
50-episode control/treatment sweep on the same selection seeds; held-out seeds
are never read here.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED_STEPS = (10000, 12500, 15000, 17500, 20000)
LATE_STEPS = (17500, 20000)


def _read_curve(path: Path, *, episodes: int) -> dict[int, float]:
    curve: dict[int, float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            step = int(float(row["env_steps"]))
            if step not in EXPECTED_STEPS:
                continue
            if step in curve:
                raise ValueError(f"{path} contains duplicate step {step}")
            if int(float(row["eval_episodes"])) != episodes:
                raise ValueError(
                    f"{path} step {step} does not use {episodes} episodes"
                )
            if int(float(row["eval_seed_start"])) != 400:
                raise ValueError(f"{path} step {step} does not start at seed 400")
            success = float(row["episode_success"])
            if not 0.0 <= success <= 1.0:
                raise ValueError(f"{path} step {step} has invalid success {success}")
            curve[step] = success
    missing = sorted(set(EXPECTED_STEPS) - set(curve))
    if missing:
        raise ValueError(f"{path} is missing validation steps {missing}")
    return curve


def _selected(curve: dict[int, float]) -> dict[str, object]:
    step, success = max(curve.items(), key=lambda item: (item[1], -item[0]))
    return {
        "curve": {str(s): curve[s] for s in EXPECTED_STEPS},
        "best_step": step,
        "best_success": success,
    }


def summarize(base: Path) -> dict[str, object]:
    control_dir = base / "control" / "offline_then_online_seed1"
    treatment_dir = base / "treatment" / "offline_then_online_seed1"
    coarse_path = treatment_dir / "val20_seeds400_coarse.csv"
    coarse = _read_curve(coarse_path, episodes=20)
    coarse_nonzero = any(value > 0.0 for value in coarse.values())
    late_success = max(coarse[step] for step in LATE_STEPS)

    result: dict[str, object] = {
        "protocol": {
            "training_seed": 1,
            "offline_updates": 10000,
            "initial_online_environment_steps": 10000,
            "batch_size": 256,
            "demo_batch_size": 256,
            "effective_online_batch_size": 512,
            "selection_seed_range": [400, 449],
            "coarse_seed_range": [400, 419],
            "selection_steps": list(EXPECTED_STEPS),
            "checkpoint_tie_break": "earliest checkpoint",
            "heldout_seeds_800_999": "sealed",
            "full_run_policy": (
                "forbidden until separately earned by nonzero, non-collapsing "
                "fixed-evaluation scaling curves"
            ),
        },
        "coarse_treatment": _selected(coarse),
        "coarse_nonzero": coarse_nonzero,
        "coarse_late_success": late_success,
        "coarse_extension_threshold": 0.20,
        "eligible_for_20k_online_extension": False,
        "matched_validation_complete": False,
    }

    if not coarse_nonzero:
        result["next_decision"] = "stop_all_zero_initial_scaling_curve"
        return result

    control_path = control_dir / "val50_seeds400_selection.csv"
    treatment_path = treatment_dir / "val50_seeds400_selection.csv"
    if not control_path.exists() or not treatment_path.exists():
        result["next_decision"] = "run_matched_50_episode_selection_curve"
        return result

    control = _read_curve(control_path, episodes=50)
    treatment = _read_curve(treatment_path, episodes=50)
    control_selected = _selected(control)
    treatment_selected = _selected(treatment)
    delta = float(treatment_selected["best_success"]) - float(
        control_selected["best_success"]
    )
    eligible = late_success >= 0.20
    result.update(
        {
            "matched_validation_complete": True,
            "matched_control": control_selected,
            "matched_treatment": treatment_selected,
            "treatment_minus_control_best": delta,
            "eligible_for_20k_online_extension": eligible,
            "next_decision": (
                "eligible_for_separately_launched_20k_online_extension"
                if eligible
                else "stop_after_initial_gate_late_curve_below_20pct"
            ),
        }
    )
    return result


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
