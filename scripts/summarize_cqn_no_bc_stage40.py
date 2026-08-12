#!/usr/bin/env python3
"""Summarize the paired Stage-40 offline-to-online handoff gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


STEPS = (10000, 12500, 15000, 17500, 20000)
POST_HANDOFF_STEPS = (12500, 15000, 17500, 20000)
GATE_TOLERANCE = 1e-12


def read_curve(path: Path) -> dict[int, float]:
    values: dict[int, float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            step = int(float(row["env_steps"]))
            if step not in STEPS:
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
    missing = sorted(set(STEPS) - set(values))
    if missing:
        raise ValueError(f"{path} is missing steps {missing}")
    return values


def selected(curve: dict[int, float], steps: tuple[int, ...]) -> dict[str, object]:
    step, success = max(
        ((step, curve[step]) for step in steps),
        key=lambda item: (item[1], -item[0]),
    )
    return {"best_step": step, "best_success": success}


def summarize(stage_dir: Path, stage38_dir: Path) -> dict[str, object]:
    treatment_curves = {
        seed: read_curve(
            stage_dir
            / f"seed{seed}"
            / "offline_dense_online_canonical"
            / "val50_seeds400_stage40.csv"
        )
        for seed in (1, 2)
    }
    control_curves = {
        seed: read_curve(
            stage38_dir
            / f"dense_seed{seed}"
            / "offline_then_online"
            / "val50_seeds400_selection.csv"
        )
        for seed in (1, 2)
    }

    per_seed: dict[str, object] = {}
    deltas: list[float] = []
    branch_integrity = True
    for seed in (1, 2):
        treatment = treatment_curves[seed]
        control = control_curves[seed]
        treatment_best = selected(treatment, POST_HANDOFF_STEPS)
        control_best = selected(control, POST_HANDOFF_STEPS)
        delta = float(treatment_best["best_success"]) - float(
            control_best["best_success"]
        )
        deltas.append(delta)
        raw10_equal = treatment[10000] == control[10000]
        branch_integrity = branch_integrity and raw10_equal
        per_seed[f"seed{seed}"] = {
            "treatment_curve": {str(step): treatment[step] for step in STEPS},
            "control_curve": {str(step): control[step] for step in STEPS},
            "shared_offline_raw10_equal": raw10_equal,
            "treatment_post_handoff_best": treatment_best,
            "control_post_handoff_best": control_best,
            "paired_best_delta": delta,
            "treatment_raw20_endpoint": treatment[20000],
            "control_raw20_endpoint": control[20000],
        }

    treatment_mean_best = sum(
        float(per_seed[f"seed{seed}"]["treatment_post_handoff_best"]["best_success"])
        for seed in (1, 2)
    ) / 2.0
    control_mean_best = sum(
        float(per_seed[f"seed{seed}"]["control_post_handoff_best"]["best_success"])
        for seed in (1, 2)
    ) / 2.0
    treatment_endpoint_mean = sum(
        treatment_curves[seed][20000] for seed in (1, 2)
    ) / 2.0
    mechanism_pass = bool(
        branch_integrity
        and all(delta >= 0.0 for delta in deltas)
        and treatment_mean_best - control_mean_best >= 0.05 - GATE_TOLERANCE
        and treatment_endpoint_mean >= 0.50 - GATE_TOLERANCE
    )
    return {
        "protocol": {
            "research_question": (
                "Does using dense reward-Q only for the 10k offline phase, then "
                "handing off to canonical C51 online, prevent the paired "
                "Stage-38 online erosion and improve policy scaling?"
            ),
            "training_seeds": [1, 2],
            "shared_offline_updates": 10000,
            "initial_online_environment_steps": 10000,
            "batch_size": 256,
            "demo_batch_size": 256,
            "effective_online_batch_size": 512,
            "selection_seeds": [400, 449],
            "episodes_per_checkpoint": 50,
            "selection_steps": list(STEPS),
            "post_handoff_selection_steps": list(POST_HANDOFF_STEPS),
            "checkpoint_tie_break": "earliest checkpoint",
            "matched_control": "Stage-38 same-seed dense-offline+dense-online",
            "pass": (
                "raw10 branch integrity; both paired post-handoff best deltas "
                "nonnegative; mean gain >=5pp; raw20 treatment endpoint mean >=50%"
            ),
            "heldout_seeds_800_999": "sealed",
            "automatic_full_run": False,
            "eligible_for_full_run": False,
        },
        "stage38_control_dir": str(stage38_dir.resolve()),
        "per_seed": per_seed,
        "shared_raw10_branch_integrity": branch_integrity,
        "treatment_post_handoff_mean_best": treatment_mean_best,
        "control_post_handoff_mean_best": control_mean_best,
        "mean_best_gain": treatment_mean_best - control_mean_best,
        "treatment_raw20_endpoint_mean": treatment_endpoint_mean,
        "mechanism_pass": mechanism_pass,
        "eligible_for_bounded_raw30_extension": mechanism_pass,
        "eligible_for_full_run": False,
        "next_decision": (
            "design_bounded_raw30_scaling_extension"
            if mechanism_pass
            else "stop_stage40_after_raw20_gate"
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
