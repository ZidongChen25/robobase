#!/usr/bin/env python3
"""Summarize the Stage-38 dense reward-Q offline-to-online gate."""

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
            value = float(row["episode_success"])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{path} step {step} has invalid success {value}")
            curve[step] = value
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
        "late_best": max(curve[s] for s in LATE_STEPS),
    }


def _candidate_dir(base: Path, seed: int) -> Path:
    return base / f"dense_seed{seed}" / "offline_then_online"


def summarize(base: Path, baseline_base: Path) -> dict[str, object]:
    coarse = {
        f"seed{seed}": _read_curve(
            _candidate_dir(base, seed) / "val20_seeds400_coarse.csv",
            episodes=20,
        )
        for seed in (1, 2)
    }
    coarse_selected = {seed: _selected(curve) for seed, curve in coarse.items()}
    both_nonzero = all(item["best_success"] > 0.0 for item in coarse_selected.values())
    any_late_20 = any(item["late_best"] >= 0.20 for item in coarse_selected.values())
    coarse_pass = bool(both_nonzero and any_late_20)

    result: dict[str, object] = {
        "protocol": {
            "research_question": (
                "Does adding only the dense reward-Q target turn the exact "
                "Stage-36 baseline-matched No-BC offline-to-online recipe into "
                "a usable policy?"
            ),
            "training_seeds": [1, 2],
            "offline_updates": 10000,
            "initial_online_environment_steps": 10000,
            "batch_size": 256,
            "demo_batch_size": 256,
            "effective_online_batch_size": 512,
            "coarse_seed_range": [400, 419],
            "selection_seed_range": [400, 449],
            "selection_steps": list(EXPECTED_STEPS),
            "checkpoint_tie_break": "earliest checkpoint",
            "heldout_seeds_800_999": "sealed",
            "automatic_full_run": False,
        },
        "baseline_stage36": str(baseline_base.resolve()),
        "coarse_dense": coarse_selected,
        "coarse_both_seeds_nonzero": both_nonzero,
        "coarse_any_late_at_least_20pct": any_late_20,
        "coarse_qualification_pass": coarse_pass,
        "matched_validation_complete": False,
        "eligible_for_20k_online_extension": False,
    }
    if not coarse_pass:
        result["next_decision"] = "stop_dense_offline_gate_without_full_sweep"
        return result

    paths = {
        "dense_seed1": _candidate_dir(base, 1)
        / "val50_seeds400_selection.csv",
        "dense_seed2": _candidate_dir(base, 2)
        / "val50_seeds400_selection.csv",
        "stage36_nobc_seed1": baseline_base
        / "treatment"
        / "offline_then_online_seed1"
        / "val50_seeds400_stage38.csv",
        "stage36_bc_seed1": baseline_base
        / "control"
        / "offline_then_online_seed1"
        / "val50_seeds400_stage38.csv",
    }
    if not all(path.exists() for path in paths.values()):
        result["next_decision"] = "run_matched_50_episode_selection_curves"
        return result

    matched = {
        name: _selected(_read_curve(path, episodes=50))
        for name, path in paths.items()
    }
    dense_mean_best = (
        float(matched["dense_seed1"]["best_success"])
        + float(matched["dense_seed2"]["best_success"])
    ) / 2.0
    both_late_20 = all(
        float(matched[f"dense_seed{seed}"]["late_best"]) >= 0.20
        for seed in (1, 2)
    )
    mechanism_gain = float(matched["dense_seed1"]["best_success"]) - float(
        matched["stage36_nobc_seed1"]["best_success"]
    )
    extension_pass = bool(
        dense_mean_best >= 0.40 and both_late_20 and mechanism_gain > 0.0
    )
    result.update(
        {
            "matched_validation_complete": True,
            "matched": matched,
            "dense_two_seed_mean_best": dense_mean_best,
            "dense_both_seeds_late_at_least_20pct": both_late_20,
            "dense_seed1_minus_stage36_nobc_best": mechanism_gain,
            "dense_seed1_minus_stage36_bc_best": (
                float(matched["dense_seed1"]["best_success"])
                - float(matched["stage36_bc_seed1"]["best_success"])
            ),
            "eligible_for_20k_online_extension": extension_pass,
            "next_decision": (
                "eligible_for_separately_launched_20k_online_extension"
                if extension_pass
                else "stop_after_matched_initial_gate"
            ),
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--baseline-base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.base, args.baseline_base)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
