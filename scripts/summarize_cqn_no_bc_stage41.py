#!/usr/bin/env python3
"""Summarize the paired Stage-41 positive-return online handoff gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


POST_STEPS = (12500, 15000, 17500, 20000)
TOLERANCE = 1e-12


def read_curve(path: Path, steps: tuple[int, ...] = POST_STEPS) -> dict[int, float]:
    values: dict[int, float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            step = int(float(row["env_steps"]))
            if step not in steps:
                continue
            if step in values:
                raise ValueError(f"{path} contains duplicate step {step}")
            if int(float(row["eval_episodes"])) != 50:
                raise ValueError(f"{path} step {step} is not a 50-episode eval")
            if int(float(row["eval_seed_start"])) != 400:
                raise ValueError(f"{path} step {step} does not start at seed 400")
            value = float(row["episode_success"])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{path} step {step} has invalid success {value}")
            values[step] = value
    missing = sorted(set(steps) - set(values))
    if missing:
        raise ValueError(f"{path} is missing steps {missing}")
    return values


def selected(curve: dict[int, float]) -> dict[str, object]:
    step, value = max(curve.items(), key=lambda item: (item[1], -item[0]))
    return {"best_step": step, "best_success": value}


def summarize(stage_dir: Path, stage38_dir: Path, stage40_dir: Path) -> dict[str, object]:
    treatments = {
        seed: read_curve(
            stage_dir
            / f"seed{seed}"
            / "offline_dense_online_positive_dense"
            / "val50_seeds400_stage41.csv"
        )
        for seed in (1, 2)
    }
    controls = {
        seed: read_curve(
            stage38_dir
            / f"dense_seed{seed}"
            / "offline_then_online"
            / "val50_seeds400_selection.csv"
        )
        for seed in (1, 2)
    }
    offline_boundary = {
        seed: float(
            next(
                row["episode_success"]
                for row in csv.DictReader(
                    (
                        stage38_dir
                        / f"dense_seed{seed}"
                        / "offline_then_online"
                        / "val50_seeds400_selection.csv"
                    ).open(newline="")
                )
                if int(float(row["env_steps"])) == 10000
            )
        )
        for seed in (1, 2)
    }

    per_seed: dict[str, object] = {}
    treatment_bests: list[float] = []
    control_bests: list[float] = []
    paired_noninferior: list[bool] = []
    endpoints: list[float] = []
    for seed in (1, 2):
        treatment_selected = selected(treatments[seed])
        control_selected = selected(controls[seed])
        treatment_best = float(treatment_selected["best_success"])
        control_best = float(control_selected["best_success"])
        endpoint = treatments[seed][20000]
        treatment_bests.append(treatment_best)
        control_bests.append(control_best)
        endpoints.append(endpoint)
        paired_noninferior.append(treatment_best + 0.02 >= control_best)
        per_seed[f"seed{seed}"] = {
            "shared_offline_raw10": offline_boundary[seed],
            "treatment_post_curve": {
                str(step): treatments[seed][step] for step in POST_STEPS
            },
            "stage38_full_dense_post_curve": {
                str(step): controls[seed][step] for step in POST_STEPS
            },
            "treatment_post_best": treatment_selected,
            "stage38_full_dense_post_best": control_selected,
            "paired_best_delta": treatment_best - control_best,
            "paired_noninferior_with_2pp_tolerance": paired_noninferior[-1],
            "treatment_raw20_endpoint": endpoint,
        }

    treatment_mean_best = sum(treatment_bests) / 2.0
    control_mean_best = sum(control_bests) / 2.0
    endpoint_mean = sum(endpoints) / 2.0
    mechanism_pass = bool(
        all(paired_noninferior)
        and treatment_mean_best >= control_mean_best - TOLERANCE
        and all(endpoint >= 0.40 - TOLERANCE for endpoint in endpoints)
        and endpoint_mean >= 0.50 - TOLERANCE
    )
    return {
        "protocol": {
            "research_question": (
                "After the shared full-dense 10k offline phase, does retaining "
                "dense reward-Q only for positive-return trajectories preserve "
                "the policy while avoiding full-dense online erosion?"
            ),
            "training_seeds": [1, 2],
            "shared_offline_updates": 10000,
            "initial_online_environment_steps": 10000,
            "batch_size": 256,
            "demo_batch_size": 256,
            "effective_online_batch_size": 512,
            "selection_seeds": [400, 449],
            "episodes_per_checkpoint": 50,
            "shared_offline_step": 10000,
            "post_handoff_selection_steps": list(POST_STEPS),
            "checkpoint_tie_break": "earliest checkpoint",
            "matched_control": "Stage-38 same-seed full-dense online",
            "descriptive_failed_control": str(stage40_dir.resolve()),
            "pass": (
                "both paired post-best values within 2pp of Stage-38; mean "
                "post-best noninferior; both raw20 endpoints >=40%; endpoint "
                "mean >=50%"
            ),
            "heldout_seeds_800_999": "sealed",
            "automatic_full_run": False,
            "eligible_for_full_run": False,
        },
        "stage38_control_dir": str(stage38_dir.resolve()),
        "stage40_failed_control_dir": str(stage40_dir.resolve()),
        "branch_integrity": "exact snapshot/replay manifest, not noisy re-evaluation",
        "per_seed": per_seed,
        "treatment_post_mean_best": treatment_mean_best,
        "stage38_full_dense_post_mean_best": control_mean_best,
        "mean_best_delta": treatment_mean_best - control_mean_best,
        "treatment_raw20_endpoint_mean": endpoint_mean,
        "mechanism_pass": mechanism_pass,
        "eligible_for_bounded_raw30_extension": mechanism_pass,
        "eligible_for_full_run": False,
        "next_decision": (
            "design_bounded_raw30_scaling_extension"
            if mechanism_pass
            else "stop_stage41_after_raw20_gate"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--stage38-dir", type=Path, required=True)
    parser.add_argument("--stage40-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.stage_dir, args.stage38_dir, args.stage40_dir)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
