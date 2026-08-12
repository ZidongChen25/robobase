#!/usr/bin/env python3
"""Summarize the bounded Stage-38 raw-20k to raw-30k extension.

This stage can authorize only a separately designed 50k scaling sentinel.  It
cannot authorize a 101k run or open the held-out seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


INITIAL_STEPS = (10000, 12500, 15000, 17500, 20000)
EXTENSION_STEPS = (22500, 25000, 27500, 30000)
ALL_STEPS = INITIAL_STEPS + EXTENSION_STEPS


def _read_curve(
    path: Path, expected_steps: tuple[int, ...], *, episodes: int = 50
) -> dict[int, float]:
    curve: dict[int, float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            step = int(float(row["env_steps"]))
            if step not in expected_steps:
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
    missing = sorted(set(expected_steps) - set(curve))
    if missing:
        raise ValueError(f"{path} is missing validation steps {missing}")
    return curve


def _selected(curve: dict[int, float], steps: tuple[int, ...]) -> dict[str, object]:
    step, success = max(
        ((step, curve[step]) for step in steps),
        key=lambda item: (item[1], -item[0]),
    )
    return {
        "curve": {str(step): curve[step] for step in steps},
        "best_step": step,
        "best_success": success,
    }


def summarize(base: Path) -> dict[str, object]:
    initial_summary_path = base / "stage38_summary.json"
    initial_summary = json.loads(initial_summary_path.read_text())
    if not initial_summary.get("matched_validation_complete"):
        raise ValueError("Stage-38 matched validation is not complete")
    if not initial_summary.get("eligible_for_20k_online_extension"):
        raise ValueError("Stage-38 initial gate did not authorize this extension")

    per_seed: dict[str, dict[str, object]] = {}
    initial_bests: list[float] = []
    extension_bests: list[float] = []
    combined_bests: list[float] = []
    for seed in (1, 2):
        run_dir = base / f"dense_seed{seed}" / "offline_then_online"
        initial = _read_curve(
            run_dir / "val50_seeds400_selection.csv", INITIAL_STEPS
        )
        extension = _read_curve(
            run_dir / "val50_seeds400_extension.csv", EXTENSION_STEPS
        )
        combined = {**initial, **extension}
        initial_selected = _selected(combined, INITIAL_STEPS)
        extension_selected = _selected(combined, EXTENSION_STEPS)
        combined_selected = _selected(combined, ALL_STEPS)
        initial_bests.append(float(initial_selected["best_success"]))
        extension_bests.append(float(extension_selected["best_success"]))
        combined_bests.append(float(combined_selected["best_success"]))
        per_seed[f"seed{seed}"] = {
            "initial": initial_selected,
            "extension": extension_selected,
            "combined": combined_selected,
            "extension_endpoint": extension[30000],
            "extension_last_two_mean": (extension[27500] + extension[30000]) / 2.0,
        }

    initial_mean_best = sum(initial_bests) / len(initial_bests)
    extension_mean_best = sum(extension_bests) / len(extension_bests)
    combined_mean_best = sum(combined_bests) / len(combined_bests)
    both_extension_best_at_least_40 = all(value >= 0.40 for value in extension_bests)
    scaling_support = bool(
        both_extension_best_at_least_40
        and extension_mean_best >= 0.50
        and combined_mean_best >= initial_mean_best
    )

    bc_best = float(
        initial_summary["matched"]["stage36_bc_seed1"]["best_success"]
    )
    return {
        "protocol": {
            "research_question": (
                "Does the exact Stage-38 offline-to-online reward-Q policy "
                "remain useful or improve over a second 10k online block?"
            ),
            "training_seeds": [1, 2],
            "offline_updates": 10000,
            "total_online_environment_steps": 20000,
            "batch_size": 256,
            "demo_batch_size": 256,
            "effective_online_batch_size": 512,
            "selection_seed_range": [400, 449],
            "initial_steps": list(INITIAL_STEPS),
            "extension_steps": list(EXTENSION_STEPS),
            "checkpoint_tie_break": "earliest checkpoint",
            "heldout_seeds_800_999": "sealed",
            "automatic_full_run": False,
            "eligible_for_full_run": False,
        },
        "per_seed": per_seed,
        "initial_two_seed_mean_best": initial_mean_best,
        "extension_two_seed_mean_best": extension_mean_best,
        "combined_two_seed_mean_best": combined_mean_best,
        "both_extension_best_at_least_40pct": both_extension_best_at_least_40,
        "stage36_bc_seed1_best": bc_best,
        "combined_mean_best_minus_stage36_bc_seed1_best": combined_mean_best
        - bc_best,
        "eligible_for_separately_designed_50k_scaling_sentinel": scaling_support,
        "eligible_for_full_run": False,
        "next_decision": (
            "design_50k_scaling_sentinel_without_opening_heldout"
            if scaling_support
            else "stop_stage38_scaling_after_raw30k"
        ),
    }


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
