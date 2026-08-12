#!/usr/bin/env python3
"""Summarize four-seed Stage-43 validation and decide held-out eligibility."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

if __package__:
    from scripts.summarize_cqn_no_bc_stage43_seed12 import ALL_STEPS, LATER_STEPS
else:
    from summarize_cqn_no_bc_stage43_seed12 import ALL_STEPS, LATER_STEPS


TOLERANCE = 1e-12


def read_curve(path: Path) -> dict[int, float]:
    values: dict[int, float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            step = int(float(row["env_steps"]))
            if step not in ALL_STEPS:
                continue
            if step in values:
                raise ValueError(f"{path} duplicate step {step}")
            if int(float(row["eval_episodes"])) != 50:
                raise ValueError(f"{path} step {step} is not 50 episodes")
            if int(float(row["eval_seed_start"])) != 400:
                raise ValueError(f"{path} step {step} does not start at 400")
            values[step] = float(row["episode_success"])
    missing = sorted(set(ALL_STEPS) - set(values))
    if missing:
        raise ValueError(f"{path} missing steps {missing}")
    return values


def selected(curve: dict[int, float]) -> dict[str, object]:
    step, value = max(curve.items(), key=lambda item: (item[1], -item[0]))
    return {"best_step": step, "best_success": value}


def observed_demo_sizes(path: Path) -> dict[str, object]:
    rows = list(csv.DictReader(path.open(newline="")))
    observed = sorted({int(float(row["demo_buffer_size"])) for row in rows})
    return {
        "expected_transitions": 9253,
        "observed_sizes": observed,
        "fixed": observed == [9253],
    }


def summarize(stage43_dir: Path) -> dict[str, object]:
    seed12 = json.loads((stage43_dir / "stage43_seed12_summary.json").read_text())
    if not seed12["eligible_for_fresh_seed34_full_runs"]:
        raise ValueError("Stage-43A did not authorize seeds 3/4")

    curves: dict[int, dict[int, float]] = {}
    buffers: dict[int, dict[str, object]] = {}
    for seed in (1, 2):
        record = seed12["per_seed"][f"seed{seed}"]
        curves[seed] = {
            int(step): float(value) for step, value in record["curve"].items()
        }
        if set(curves[seed]) != set(ALL_STEPS):
            raise ValueError(f"seed {seed} Stage-43A curve has wrong steps")
        buffers[seed] = record["demo_buffer"]
    for seed in (3, 4):
        run = stage43_dir / f"seed{seed}" / "fixed_expert_101k_online"
        curves[seed] = read_curve(run / "val50_seeds400_stage43_seed34.csv")
        buffers[seed] = observed_demo_sizes(run / "train.csv")

    per_seed: dict[str, object] = {}
    overall_bests: list[float] = []
    later_bests: list[float] = []
    endpoints: list[float] = []
    for seed in (1, 2, 3, 4):
        curve = curves[seed]
        overall_best = selected(curve)
        later = {step: curve[step] for step in LATER_STEPS}
        later_best = selected(later)
        endpoint = curve[111000]
        overall_bests.append(float(overall_best["best_success"]))
        later_bests.append(float(later_best["best_success"]))
        endpoints.append(endpoint)
        per_seed[f"seed{seed}"] = {
            "curve": {str(step): curve[step] for step in ALL_STEPS},
            "validation_selected_best": overall_best,
            "online_50k_to_101k_best": later_best,
            "raw111k_fixed_endpoint": endpoint,
            "demo_buffer": buffers[seed],
        }

    mean_best = sum(overall_bests) / 4.0
    later_mean_best = sum(later_bests) / 4.0
    endpoint_mean = sum(endpoints) / 4.0
    heldout_eligible = bool(
        all(check["fixed"] for check in buffers.values())
        and all(value >= 0.55 - TOLERANCE for value in overall_bests)
        and mean_best >= 0.68 - TOLERANCE
        and all(value >= 0.55 - TOLERANCE for value in later_bests)
        and later_mean_best >= 0.65 - TOLERANCE
        and all(value >= 0.45 - TOLERANCE for value in endpoints)
        and endpoint_mean >= 0.60 - TOLERANCE
    )
    return {
        "protocol": {
            "training_seeds": [1, 2, 3, 4],
            "offline_reward_q_updates": 10000,
            "online_environment_interactions": 101000,
            "fixed_endpoint_raw_step": 111000,
            "batch_size": 256,
            "demo_batch_size": 256,
            "selection_seeds": [400, 449],
            "episodes_per_checkpoint": 50,
            "selection_steps": list(ALL_STEPS),
            "checkpoint_tie_break": "earliest checkpoint",
            "heldout_gate": (
                "fixed expert replay; every selected best >=55% and mean "
                ">=68%; every online-50k+ best >=55% and mean >=65%; every "
                "raw111k endpoint >=45% and mean >=60%"
            ),
            "heldout_fixed_endpoints_only": True,
            "heldout_episodes_per_seed": 200,
            "heldout_seeds": [800, 999],
            "official_fixed_endpoint_reference": {
                "per_seed": [0.62, 0.605, 0.62, 0.74],
                "mean": 0.64625,
            },
        },
        "per_seed": per_seed,
        "validation_selected_mean_best": mean_best,
        "online_50k_to_101k_mean_best": later_mean_best,
        "raw111k_fixed_endpoint_mean": endpoint_mean,
        "all_demo_buffers_fixed": all(
            check["fixed"] for check in buffers.values()
        ),
        "eligible_for_sealed_heldout": heldout_eligible,
        "heldout_opened": False,
        "next_decision": (
            "run_four_fixed_raw111k_heldout_endpoints"
            if heldout_eligible
            else "stop_stage43_before_heldout_and_design_next_reward_q_mechanism"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage43-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.stage43_dir)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
