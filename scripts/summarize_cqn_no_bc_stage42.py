#!/usr/bin/env python3
"""Summarize Stage-42 fixed-expert-replay scaling through raw 30k."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


STEPS = (12500, 15000, 17500, 20000, 22500, 25000, 27500, 30000)
TOLERANCE = 1e-12


def read_curve(paths: tuple[Path, ...]) -> dict[int, float]:
    values: dict[int, float] = {}
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                step = int(float(row["env_steps"]))
                if step not in STEPS:
                    continue
                if step in values:
                    raise ValueError(f"duplicate step {step} across {paths}")
                if int(float(row["eval_episodes"])) != 50:
                    raise ValueError(f"{path} step {step} is not 50 episodes")
                if int(float(row["eval_seed_start"])) != 400:
                    raise ValueError(f"{path} step {step} does not start at 400")
                value = float(row["episode_success"])
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{path} step {step} invalid success {value}")
                values[step] = value
    missing = sorted(set(STEPS) - set(values))
    if missing:
        raise ValueError(f"{paths} missing steps {missing}")
    return values


def selected(curve: dict[int, float]) -> dict[str, object]:
    step, value = max(curve.items(), key=lambda item: (item[1], -item[0]))
    return {"best_step": step, "best_success": value}


def _fixed_demo_buffer(stage_dir: Path, seed: int) -> dict[str, object]:
    run = stage_dir / f"seed{seed}" / "offline_dense_online_positive_fixed_expert"
    manifest = json.loads((run / "stage42_branch_manifest.json").read_text())
    expected = int(manifest["replay"]["demo_replay"]["num_transitions"])
    rows = list(csv.DictReader((run / "train.csv").open(newline="")))
    observed = sorted(
        {
            int(float(row["demo_buffer_size"]))
            for row in rows
            if int(float(row["env_steps"])) >= 10000
        }
    )
    return {
        "expected_transitions": expected,
        "observed_sizes": observed,
        "fixed": observed == [expected],
    }


def summarize(stage_dir: Path, stage41_dir: Path, stage38_dir: Path) -> dict[str, object]:
    treatment = {
        seed: read_curve(
            (
                stage_dir
                / f"seed{seed}"
                / "offline_dense_online_positive_fixed_expert"
                / "val50_seeds400_stage42.csv",
            )
        )
        for seed in (1, 2)
    }
    growing_replay = {
        seed: read_curve(
            (
                stage41_dir
                / f"seed{seed}"
                / "offline_dense_online_positive_dense"
                / "val50_seeds400_stage41.csv",
                stage41_dir
                / f"seed{seed}"
                / "offline_dense_online_positive_dense"
                / "val50_seeds400_stage41_extension.csv",
            )
        )
        for seed in (1, 2)
    }
    full_dense = {
        seed: read_curve(
            (
                stage38_dir
                / f"dense_seed{seed}"
                / "offline_then_online"
                / "val50_seeds400_selection.csv",
                stage38_dir
                / f"dense_seed{seed}"
                / "offline_then_online"
                / "val50_seeds400_extension.csv",
            )
        )
        for seed in (1, 2)
    }
    buffer_checks = {
        seed: _fixed_demo_buffer(stage_dir, seed) for seed in (1, 2)
    }

    per_seed: dict[str, object] = {}
    bests: list[float] = []
    endpoints: list[float] = []
    all_values: list[float] = []
    for seed in (1, 2):
        curve = treatment[seed]
        best = selected(curve)
        prior_best = selected(growing_replay[seed])
        dense_best = selected(full_dense[seed])
        best_value = float(best["best_success"])
        endpoint = curve[30000]
        bests.append(best_value)
        endpoints.append(endpoint)
        all_values.extend(curve.values())
        per_seed[f"seed{seed}"] = {
            "fixed_expert_curve": {str(step): curve[step] for step in STEPS},
            "fixed_expert_best": best,
            "stage41_growing_replay_best": prior_best,
            "stage38_full_dense_best": dense_best,
            "best_delta_vs_stage41": (
                best_value - float(prior_best["best_success"])
            ),
            "best_delta_vs_stage38": (
                best_value - float(dense_best["best_success"])
            ),
            "raw30_endpoint": endpoint,
            "demo_buffer": buffer_checks[seed],
        }

    mean_best = sum(bests) / 2.0
    endpoint_mean = sum(endpoints) / 2.0
    all_checkpoint_mean = sum(all_values) / len(all_values)
    pass_gate = bool(
        all(check["fixed"] for check in buffer_checks.values())
        and all(value >= 0.55 - TOLERANCE for value in bests)
        and mean_best >= 0.60 - TOLERANCE
        and all(value >= 0.50 - TOLERANCE for value in endpoints)
        and endpoint_mean >= 0.55 - TOLERANCE
        and all_checkpoint_mean >= 0.54 - TOLERANCE
    )
    return {
        "protocol": {
            "research_question": (
                "Does keeping the prior replay fixed to expert trajectories "
                "remove Stage-41's seed-specific success-replay feedback and "
                "produce robust reward-Q scaling through raw 30k?"
            ),
            "training_seeds": [1, 2],
            "shared_offline_updates": 10000,
            "total_online_environment_steps": 20000,
            "batch_size": 256,
            "demo_batch_size": 256,
            "selection_seeds": [400, 449],
            "episodes_per_checkpoint": 50,
            "selection_steps": list(STEPS),
            "checkpoint_tie_break": "earliest checkpoint",
            "matched_control": "Stage-41 growing reward-only success replay",
            "secondary_control": "Stage-38 full-dense online",
            "pass": (
                "both bests >=55%; mean best >=60%; both raw30 endpoints "
                ">=50% and endpoint mean >=55%; all-checkpoint mean >=54%; "
                "protected expert replay remains exactly fixed"
            ),
            "heldout_seeds_800_999": "sealed",
            "automatic_full_run": False,
            "eligible_for_full_run": False,
        },
        "stage41_control_dir": str(stage41_dir.resolve()),
        "stage38_control_dir": str(stage38_dir.resolve()),
        "per_seed": per_seed,
        "fixed_expert_mean_best": mean_best,
        "raw30_endpoint_mean": endpoint_mean,
        "all_checkpoint_mean": all_checkpoint_mean,
        "all_demo_buffers_fixed": all(
            check["fixed"] for check in buffer_checks.values()
        ),
        "eligible_for_separately_designed_raw50_replication": pass_gate,
        "eligible_for_full_run": False,
        "heldout_opened": False,
        "next_decision": (
            "design_raw50_fixed_expert_replication"
            if pass_gate
            else "stop_stage42_after_raw30_gate"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--stage41-dir", type=Path, required=True)
    parser.add_argument("--stage38-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.stage_dir, args.stage41_dir, args.stage38_dir)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
