#!/usr/bin/env python3
"""Summarize the first two Stage-43 101k-online full-scale seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


INITIAL_STEPS = (12500, 15000, 17500, 20000, 22500, 25000, 27500, 30000)
REPLICATION_STEPS = (32500, 35000, 37500, 40000, 42500, 45000, 47500, 50000)
LATER_STEPS = (60000, 70000, 80000, 90000, 100000, 110000, 111000)
ALL_STEPS = INITIAL_STEPS + REPLICATION_STEPS + LATER_STEPS
TOLERANCE = 1e-12


def read_curve(path: Path, steps: tuple[int, ...]) -> dict[int, float]:
    values: dict[int, float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            step = int(float(row["env_steps"]))
            if step not in steps:
                continue
            if step in values:
                raise ValueError(f"{path} duplicate step {step}")
            if int(float(row["eval_episodes"])) != 50:
                raise ValueError(f"{path} step {step} is not 50 episodes")
            if int(float(row["eval_seed_start"])) != 400:
                raise ValueError(f"{path} step {step} does not start at 400")
            values[step] = float(row["episode_success"])
    missing = sorted(set(steps) - set(values))
    if missing:
        raise ValueError(f"{path} missing steps {missing}")
    return values


def selected(curve: dict[int, float]) -> dict[str, object]:
    step, value = max(curve.items(), key=lambda item: (item[1], -item[0]))
    return {"best_step": step, "best_success": value}


def fixed_demo_buffer(stage43_dir: Path, seed: int) -> dict[str, object]:
    run = stage43_dir / f"seed{seed}" / "fixed_expert_101k_online"
    manifest = json.loads((run / "stage43_branch_manifest.json").read_text())
    expected = int(manifest["replay"]["demo_replay"]["num_transitions"])
    rows = list(csv.DictReader((run / "train.csv").open(newline="")))
    observed = sorted({int(float(row["demo_buffer_size"])) for row in rows})
    return {
        "expected_transitions": expected,
        "observed_sizes": observed,
        "fixed": observed == [expected] == [9253],
    }


def summarize(stage43_dir: Path, stage42_dir: Path) -> dict[str, object]:
    stage42 = json.loads((stage42_dir / "stage42_raw50_summary.json").read_text())
    if not stage42["eligible_for_matched_raw101k_full_protocol"]:
        raise ValueError("Stage-42 raw50 gate did not authorize Stage 43")

    per_seed: dict[str, object] = {}
    overall_bests: list[float] = []
    later_bests: list[float] = []
    later_values: list[float] = []
    endpoints: list[float] = []
    buffers = {seed: fixed_demo_buffer(stage43_dir, seed) for seed in (1, 2)}
    for seed in (1, 2):
        stage42_run = (
            stage42_dir
            / f"seed{seed}"
            / "offline_dense_online_positive_fixed_expert"
        )
        stage43_run = stage43_dir / f"seed{seed}" / "fixed_expert_101k_online"
        initial = read_curve(
            stage42_run / "val50_seeds400_stage42.csv", INITIAL_STEPS
        )
        replication = read_curve(
            stage42_run / "val50_seeds400_stage42_raw50.csv",
            REPLICATION_STEPS,
        )
        later = read_curve(
            stage43_run / "val50_seeds400_stage43_seed12.csv", LATER_STEPS
        )
        full_curve = initial | replication | later
        overall_best = selected(full_curve)
        later_best = selected(later)
        endpoint = later[111000]
        overall_bests.append(float(overall_best["best_success"]))
        later_bests.append(float(later_best["best_success"]))
        later_values.extend(later.values())
        endpoints.append(endpoint)
        per_seed[f"seed{seed}"] = {
            "curve": {str(step): full_curve[step] for step in ALL_STEPS},
            "validation_selected_best": overall_best,
            "online_50k_to_101k_best": later_best,
            "raw111k_fixed_endpoint": endpoint,
            "demo_buffer": buffers[seed],
        }

    overall_mean_best = sum(overall_bests) / 2.0
    later_mean_best = sum(later_bests) / 2.0
    later_checkpoint_mean = sum(later_values) / len(later_values)
    endpoint_mean = sum(endpoints) / 2.0
    expansion_pass = bool(
        all(check["fixed"] for check in buffers.values())
        and all(value >= 0.60 - TOLERANCE for value in overall_bests)
        and overall_mean_best >= 0.70 - TOLERANCE
        and all(value >= 0.55 - TOLERANCE for value in later_bests)
        and later_mean_best >= 0.65 - TOLERANCE
        and later_checkpoint_mean >= 0.55 - TOLERANCE
        and all(value >= 0.45 - TOLERANCE for value in endpoints)
        and endpoint_mean >= 0.55 - TOLERANCE
    )
    return {
        "protocol": {
            "research_question": (
                "Does Stage-42 fixed-expert reward-only CQN-AS remain robust "
                "through 101k online interactions on both qualified seeds?"
            ),
            "training_seeds": [1, 2],
            "offline_reward_q_updates": 10000,
            "online_environment_interactions": 101000,
            "fixed_endpoint_raw_step": 111000,
            "batch_size": 256,
            "demo_batch_size": 256,
            "selection_seeds": [400, 449],
            "episodes_per_checkpoint": 50,
            "all_selection_steps": list(ALL_STEPS),
            "full_scale_steps": list(LATER_STEPS),
            "checkpoint_tie_break": "earliest checkpoint",
            "official_validation_endpoint_reference": {
                "per_seed": [0.76, 0.62, 0.68, 0.66],
                "mean": 0.68,
            },
            "seed34_expansion_pass": (
                "fixed expert replay; per-seed overall best >=60% and mean "
                ">=70%; per-seed online-50k+ best >=55% and mean >=65%; "
                "online-50k+ all-checkpoint mean >=55%; per-seed raw111k "
                "endpoint >=45% and mean >=55%"
            ),
            "heldout_seeds_800_999": "sealed",
            "automatic_seed34_launch": False,
        },
        "stage42_dir": str(stage42_dir.resolve()),
        "per_seed": per_seed,
        "validation_selected_mean_best": overall_mean_best,
        "online_50k_to_101k_mean_best": later_mean_best,
        "online_50k_to_101k_all_checkpoint_mean": later_checkpoint_mean,
        "raw111k_fixed_endpoint_mean": endpoint_mean,
        "all_demo_buffers_fixed": all(
            check["fixed"] for check in buffers.values()
        ),
        "eligible_for_fresh_seed34_full_runs": expansion_pass,
        "heldout_opened": False,
        "next_decision": (
            "launch_fresh_seed34_101k_online"
            if expansion_pass
            else "stop_stage43_seed12_and_design_next_reward_q_mechanism"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage43-dir", type=Path, required=True)
    parser.add_argument("--stage42-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.stage43_dir, args.stage42_dir)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
