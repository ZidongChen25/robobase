#!/usr/bin/env python3
"""Summarize Stage-41's separately authorized raw-30k scaling extension."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXTENSION_STEPS = (22500, 25000, 27500, 30000)
TOLERANCE = 1e-12


def read_curve(path: Path) -> dict[int, float]:
    values: dict[int, float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            step = int(float(row["env_steps"]))
            if step not in EXTENSION_STEPS:
                continue
            if step in values:
                raise ValueError(f"{path} contains duplicate step {step}")
            if int(float(row["eval_episodes"])) != 50:
                raise ValueError(f"{path} step {step} is not a 50-episode eval")
            if int(float(row["eval_seed_start"])) != 400:
                raise ValueError(f"{path} step {step} does not start at seed 400")
            values[step] = float(row["episode_success"])
    missing = sorted(set(EXTENSION_STEPS) - set(values))
    if missing:
        raise ValueError(f"{path} is missing steps {missing}")
    return values


def selected(curve: dict[int, float]) -> dict[str, object]:
    step, value = max(curve.items(), key=lambda item: (item[1], -item[0]))
    return {"best_step": step, "best_success": value}


def summarize(stage_dir: Path, stage38_dir: Path) -> dict[str, object]:
    initial = json.loads((stage_dir / "stage41_summary.json").read_text())
    stage38_extension = json.loads(
        (stage38_dir / "stage38_extension_summary.json").read_text()
    )
    if not initial["eligible_for_bounded_raw30_extension"]:
        raise ValueError("Stage-41 initial gate did not authorize this extension")

    treatment_curves = {
        seed: read_curve(
            stage_dir
            / f"seed{seed}"
            / "offline_dense_online_positive_dense"
            / "val50_seeds400_stage41_extension.csv"
        )
        for seed in (1, 2)
    }
    control_extension_best = {
        seed: float(
            stage38_extension["per_seed"][f"seed{seed}"]["extension"][
                "best_success"
            ]
        )
        for seed in (1, 2)
    }
    initial_best = {
        seed: float(
            initial["per_seed"][f"seed{seed}"]["treatment_post_best"][
                "best_success"
            ]
        )
        for seed in (1, 2)
    }

    per_seed: dict[str, object] = {}
    extension_bests: list[float] = []
    endpoints: list[float] = []
    beats_control: list[bool] = []
    reaches_initial: list[bool] = []
    for seed in (1, 2):
        curve = treatment_curves[seed]
        best = selected(curve)
        best_value = float(best["best_success"])
        endpoint = curve[30000]
        extension_bests.append(best_value)
        endpoints.append(endpoint)
        beats_control.append(
            best_value + TOLERANCE >= control_extension_best[seed]
        )
        reaches_initial.append(best_value + TOLERANCE >= initial_best[seed])
        per_seed[f"seed{seed}"] = {
            "extension_curve": {
                str(step): curve[step] for step in EXTENSION_STEPS
            },
            "extension_best": best,
            "initial_post_best": initial_best[seed],
            "stage38_extension_best": control_extension_best[seed],
            "extension_minus_stage38": best_value - control_extension_best[seed],
            "extension_reaches_initial_best": reaches_initial[-1],
            "raw30_endpoint": endpoint,
        }

    extension_mean_best = sum(extension_bests) / 2.0
    endpoint_mean = sum(endpoints) / 2.0
    sentinel_pass = bool(
        all(beats_control)
        and extension_mean_best >= 0.58 - TOLERANCE
        and any(reaches_initial)
        and all(endpoint >= 0.45 - TOLERANCE for endpoint in endpoints)
        and endpoint_mean >= 0.52 - TOLERANCE
    )
    return {
        "protocol": {
            "research_question": (
                "Does Stage-41's positive-return-dense online policy remain "
                "strong through a second 10k online block?"
            ),
            "training_seeds": [1, 2],
            "offline_updates": 10000,
            "total_online_environment_steps": 20000,
            "batch_size": 256,
            "demo_batch_size": 256,
            "selection_seeds": [400, 449],
            "episodes_per_checkpoint": 50,
            "extension_steps": list(EXTENSION_STEPS),
            "checkpoint_tie_break": "earliest checkpoint",
            "pass": (
                "both extension bests >= Stage-38 extension controls; mean "
                "extension best >=58%; at least one seed reaches its Stage-41 "
                "initial best; both raw30 endpoints >=45%; endpoint mean >=52%"
            ),
            "heldout_seeds_800_999": "sealed",
            "automatic_full_run": False,
            "eligible_for_full_run": False,
        },
        "initial_stage41_mean_best": float(initial["treatment_post_mean_best"]),
        "per_seed": per_seed,
        "extension_mean_best": extension_mean_best,
        "raw30_endpoint_mean": endpoint_mean,
        "both_extension_bests_beat_stage38": all(beats_control),
        "any_extension_best_reaches_initial": any(reaches_initial),
        "eligible_for_separately_designed_50k_sentinel": sentinel_pass,
        "eligible_for_full_run": False,
        "next_decision": (
            "design_raw50k_scaling_sentinel"
            if sentinel_pass
            else "stop_stage41_scaling_after_raw30k"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--stage38-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.stage_dir, args.stage38_dir)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
