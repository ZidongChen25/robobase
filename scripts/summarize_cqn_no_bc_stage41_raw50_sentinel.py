#!/usr/bin/env python3
"""Summarize Stage-41's separately registered raw-50k scaling sentinel."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SENTINEL_STEPS = (32500, 35000, 37500, 40000, 42500, 45000, 47500, 50000)
LATE_STEPS = (45000, 47500, 50000)
TOLERANCE = 1e-12


def read_curve(path: Path) -> dict[int, float]:
    values: dict[int, float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            step = int(float(row["env_steps"]))
            if step not in SENTINEL_STEPS:
                continue
            if step in values:
                raise ValueError(f"{path} contains duplicate step {step}")
            if int(float(row["eval_episodes"])) != 50:
                raise ValueError(f"{path} step {step} is not a 50-episode eval")
            if int(float(row["eval_seed_start"])) != 400:
                raise ValueError(f"{path} step {step} does not start at seed 400")
            values[step] = float(row["episode_success"])
    missing = sorted(set(SENTINEL_STEPS) - set(values))
    if missing:
        raise ValueError(f"{path} is missing steps {missing}")
    return values


def selected(curve: dict[int, float]) -> dict[str, object]:
    step, value = max(curve.items(), key=lambda item: (item[1], -item[0]))
    return {"best_step": step, "best_success": value}


def summarize(stage_dir: Path) -> dict[str, object]:
    extension = json.loads(
        (stage_dir / "stage41_extension_summary.json").read_text()
    )
    if not extension["eligible_for_separately_designed_50k_sentinel"]:
        raise ValueError("Stage-41 raw30 gate did not authorize this sentinel")

    curves = {
        seed: read_curve(
            stage_dir
            / f"seed{seed}"
            / "offline_dense_online_positive_dense"
            / "val50_seeds400_stage41_raw50_sentinel.csv"
        )
        for seed in (1, 2)
    }
    prior_best = {
        seed: float(
            extension["per_seed"][f"seed{seed}"]["extension_best"][
                "best_success"
            ]
        )
        for seed in (1, 2)
    }

    per_seed: dict[str, object] = {}
    sentinel_bests: list[float] = []
    late_bests: list[float] = []
    endpoints: list[float] = []
    all_values: list[float] = []
    preserves_prior: list[bool] = []
    for seed in (1, 2):
        curve = curves[seed]
        best = selected(curve)
        late_curve = {step: curve[step] for step in LATE_STEPS}
        late_best = selected(late_curve)
        best_value = float(best["best_success"])
        late_best_value = float(late_best["best_success"])
        endpoint = curve[50000]
        sentinel_bests.append(best_value)
        late_bests.append(late_best_value)
        endpoints.append(endpoint)
        all_values.extend(curve.values())
        preserves_prior.append(
            best_value + TOLERANCE >= prior_best[seed] - 0.04
        )
        per_seed[f"seed{seed}"] = {
            "sentinel_curve": {
                str(step): curve[step] for step in SENTINEL_STEPS
            },
            "sentinel_best": best,
            "late_window_best": late_best,
            "prior_raw30_block_best": prior_best[seed],
            "sentinel_minus_prior": best_value - prior_best[seed],
            "within_4pp_of_prior_best": preserves_prior[-1],
            "raw50_endpoint": endpoint,
        }

    sentinel_mean_best = sum(sentinel_bests) / 2.0
    late_window_mean_best = sum(late_bests) / 2.0
    endpoint_mean = sum(endpoints) / 2.0
    all_checkpoint_mean = sum(all_values) / len(all_values)
    full_pass = bool(
        all(preserves_prior)
        and sentinel_mean_best >= 0.58 - TOLERANCE
        and all_checkpoint_mean >= 0.52 - TOLERANCE
        and all(value >= 0.48 - TOLERANCE for value in late_bests)
        and late_window_mean_best >= 0.56 - TOLERANCE
        and all(endpoint >= 0.45 - TOLERANCE for endpoint in endpoints)
        and endpoint_mean >= 0.52 - TOLERANCE
    )
    return {
        "protocol": {
            "research_question": (
                "Does Stage-41 preserve nonzero, task-level scaling from raw "
                "30k through raw 50k rather than expressing a transient peak?"
            ),
            "training_seeds": [1, 2],
            "offline_updates": 10000,
            "total_online_environment_steps": 40000,
            "batch_size": 256,
            "demo_batch_size": 256,
            "selection_seeds": [400, 449],
            "episodes_per_checkpoint": 50,
            "sentinel_steps": list(SENTINEL_STEPS),
            "late_steps": list(LATE_STEPS),
            "checkpoint_tie_break": "earliest checkpoint",
            "pass": (
                "each sentinel best within 4pp of its raw30-block best; mean "
                "sentinel best >=58%; mean over all 16 fixed evaluations "
                ">=52%; each late-window best >=48% and their mean >=56%; "
                "both raw50 endpoints >=45% and endpoint mean >=52%"
            ),
            "matched_reference": (
                "Stage-38 exact matched no-BC control through raw30 plus the "
                "frozen Stage-41 raw30 block; this stage tests persistence only"
            ),
            "heldout_seeds_800_999": "sealed",
            "automatic_full_run": False,
        },
        "per_seed": per_seed,
        "sentinel_mean_best": sentinel_mean_best,
        "late_window_mean_best": late_window_mean_best,
        "all_checkpoint_mean": all_checkpoint_mean,
        "raw50_endpoint_mean": endpoint_mean,
        "both_sentinel_bests_preserve_prior_within_4pp": all(preserves_prior),
        "eligible_for_full_run_protocol": full_pass,
        "heldout_opened": False,
        "next_decision": (
            "design_matched_raw101k_full_run_without_opening_heldout"
            if full_pass
            else "stop_stage41_scaling_after_raw50k"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.stage_dir)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
