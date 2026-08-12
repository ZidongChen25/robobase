#!/usr/bin/env python3
"""Summarize Stage-42's raw-30k to raw-50k fixed-expert replication."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


STEPS = (32500, 35000, 37500, 40000, 42500, 45000, 47500, 50000)
LATE_STEPS = (45000, 47500, 50000)
TOLERANCE = 1e-12


def read_curve(path: Path) -> dict[int, float]:
    values: dict[int, float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            step = int(float(row["env_steps"]))
            if step not in STEPS:
                continue
            if step in values:
                raise ValueError(f"{path} duplicate step {step}")
            if int(float(row["eval_episodes"])) != 50:
                raise ValueError(f"{path} step {step} is not 50 episodes")
            if int(float(row["eval_seed_start"])) != 400:
                raise ValueError(f"{path} step {step} does not start at 400")
            values[step] = float(row["episode_success"])
    missing = sorted(set(STEPS) - set(values))
    if missing:
        raise ValueError(f"{path} missing steps {missing}")
    return values


def selected(curve: dict[int, float]) -> dict[str, object]:
    step, value = max(curve.items(), key=lambda item: (item[1], -item[0]))
    return {"best_step": step, "best_success": value}


def fixed_demo_buffer(stage_dir: Path, seed: int) -> dict[str, object]:
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


def summarize(stage_dir: Path, stage41_dir: Path) -> dict[str, object]:
    initial = json.loads((stage_dir / "stage42_summary.json").read_text())
    if not initial["eligible_for_separately_designed_raw50_replication"]:
        raise ValueError("Stage-42 raw30 gate did not authorize replication")
    stage41 = json.loads(
        (stage41_dir / "stage41_raw50_sentinel_summary.json").read_text()
    )
    curves = {
        seed: read_curve(
            stage_dir
            / f"seed{seed}"
            / "offline_dense_online_positive_fixed_expert"
            / "val50_seeds400_stage42_raw50.csv"
        )
        for seed in (1, 2)
    }
    initial_best = {
        seed: float(
            initial["per_seed"][f"seed{seed}"]["fixed_expert_best"][
                "best_success"
            ]
        )
        for seed in (1, 2)
    }
    stage41_best = {
        seed: float(
            stage41["per_seed"][f"seed{seed}"]["sentinel_best"][
                "best_success"
            ]
        )
        for seed in (1, 2)
    }
    buffers = {seed: fixed_demo_buffer(stage_dir, seed) for seed in (1, 2)}

    per_seed: dict[str, object] = {}
    bests: list[float] = []
    late_bests: list[float] = []
    endpoints: list[float] = []
    all_values: list[float] = []
    preserves_initial: list[bool] = []
    for seed in (1, 2):
        curve = curves[seed]
        best = selected(curve)
        late_best = selected({step: curve[step] for step in LATE_STEPS})
        best_value = float(best["best_success"])
        late_value = float(late_best["best_success"])
        endpoint = curve[50000]
        bests.append(best_value)
        late_bests.append(late_value)
        endpoints.append(endpoint)
        all_values.extend(curve.values())
        preserves_initial.append(
            best_value + TOLERANCE >= initial_best[seed] - 0.06
        )
        per_seed[f"seed{seed}"] = {
            "replication_curve": {str(step): curve[step] for step in STEPS},
            "replication_best": best,
            "late_window_best": late_best,
            "initial_raw30_block_best": initial_best[seed],
            "within_6pp_of_initial_best": preserves_initial[-1],
            "stage41_raw50_best": stage41_best[seed],
            "best_delta_vs_stage41": best_value - stage41_best[seed],
            "raw50_endpoint": endpoint,
            "demo_buffer": buffers[seed],
        }

    mean_best = sum(bests) / 2.0
    late_mean_best = sum(late_bests) / 2.0
    endpoint_mean = sum(endpoints) / 2.0
    all_checkpoint_mean = sum(all_values) / len(all_values)
    full_protocol_pass = bool(
        all(check["fixed"] for check in buffers.values())
        and all(preserves_initial)
        and mean_best >= 0.65 - TOLERANCE
        and all_checkpoint_mean >= 0.55 - TOLERANCE
        and all(value >= 0.55 - TOLERANCE for value in late_bests)
        and late_mean_best >= 0.60 - TOLERANCE
        and all(value >= 0.50 - TOLERANCE for value in endpoints)
        and endpoint_mean >= 0.58 - TOLERANCE
    )
    return {
        "protocol": {
            "research_question": (
                "Does fixed expert replay replicate Stage-42's two-seed "
                "robustness through raw 50k without growing the prior buffer?"
            ),
            "training_seeds": [1, 2],
            "shared_offline_updates": 10000,
            "total_online_environment_steps": 40000,
            "batch_size": 256,
            "demo_batch_size": 256,
            "selection_seeds": [400, 449],
            "episodes_per_checkpoint": 50,
            "replication_steps": list(STEPS),
            "late_steps": list(LATE_STEPS),
            "checkpoint_tie_break": "earliest checkpoint",
            "matched_control": "Stage-41 growing-success-replay raw50 block",
            "pass": (
                "each replication best within 6pp of its Stage-42 raw30-block "
                "best; mean best >=65%; all-checkpoint mean >=55%; both late "
                "bests >=55% and mean >=60%; both raw50 endpoints >=50% and "
                "mean >=58%; expert replay remains fixed"
            ),
            "heldout_seeds_800_999": "sealed",
            "automatic_full_run": False,
        },
        "stage41_control_dir": str(stage41_dir.resolve()),
        "per_seed": per_seed,
        "replication_mean_best": mean_best,
        "late_window_mean_best": late_mean_best,
        "all_checkpoint_mean": all_checkpoint_mean,
        "raw50_endpoint_mean": endpoint_mean,
        "all_demo_buffers_fixed": all(
            check["fixed"] for check in buffers.values()
        ),
        "eligible_for_matched_raw101k_full_protocol": full_protocol_pass,
        "heldout_opened": False,
        "next_decision": (
            "design_matched_raw101k_full_protocol"
            if full_protocol_pass
            else "stop_stage42_after_raw50_replication"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--stage41-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.stage_dir, args.stage41_dir)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
