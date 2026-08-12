#!/usr/bin/env python3
"""Compare four fixed Stage-43 endpoints with official fixed endpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


OFFICIAL_RUN_NAMES = (
    "move_plate_paper_seed1_100k_nw0_20260721",
    "move_plate_paper_seed2_100k_nw0_20260721",
    "move_plate_paper_seed3_100k_nw0_20260721",
    "move_plate_paper_seed4_100k_nw0_20260721",
)
TOLERANCE = 1e-12


def read_endpoint(
    path: Path, *, expected_step: int, episodes: int = 200, seed_start: int = 800
) -> float:
    rows = list(csv.DictReader(path.open(newline="")))
    matches = [
        row for row in rows if int(float(row["env_steps"])) == expected_step
    ]
    if len(matches) != 1:
        raise ValueError(f"{path} expected one step {expected_step}, got {len(matches)}")
    row = matches[0]
    if int(float(row["eval_episodes"])) != episodes:
        raise ValueError(f"{path} is not {episodes} episodes")
    if int(float(row["eval_seed_start"])) != seed_start:
        raise ValueError(f"{path} does not start at seed {seed_start}")
    return float(row["episode_success"])


def summarize(stage43_dir: Path, official_root: Path) -> dict[str, object]:
    full = json.loads((stage43_dir / "stage43_full_summary.json").read_text())
    if not full["eligible_for_sealed_heldout"]:
        raise ValueError("Stage-43 four-seed validation did not authorize held-out")

    nobc = []
    official = []
    per_seed: dict[str, object] = {}
    for seed, official_name in enumerate(OFFICIAL_RUN_NAMES, start=1):
        no_bc_path = (
            stage43_dir
            / f"seed{seed}"
            / "fixed_expert_101k_online"
            / "heldout200_seeds800_stage43.csv"
        )
        official_path = official_root / official_name / "ep200_seeds800.csv"
        no_bc_value = read_endpoint(no_bc_path, expected_step=111000)
        official_value = read_endpoint(official_path, expected_step=101000)
        nobc.append(no_bc_value)
        official.append(official_value)
        per_seed[f"seed{seed}"] = {
            "no_bc_raw111k": no_bc_value,
            "official_bc_raw101k": official_value,
            "paired_delta": no_bc_value - official_value,
            "no_bc_artifact": str(no_bc_path.resolve()),
            "official_artifact": str(official_path.resolve()),
        }

    no_bc_mean = sum(nobc) / 4.0
    official_mean = sum(official) / 4.0
    parity = no_bc_mean + TOLERANCE >= official_mean
    return {
        "protocol": {
            "comparison": "four fixed endpoints; no held-out checkpoint selection",
            "no_bc_budget": {
                "offline_reward_q_updates": 10000,
                "online_environment_interactions": 101000,
                "raw_endpoint": 111000,
            },
            "official_budget": {
                "offline_updates": 0,
                "online_environment_interactions": 101000,
                "raw_endpoint": 101000,
            },
            "training_seeds": [1, 2, 3, 4],
            "heldout_episodes_per_seed": 200,
            "heldout_seed_range": [800, 999],
            "parity_rule": "No-BC four-seed mean >= official four-seed mean",
        },
        "per_seed": per_seed,
        "no_bc_fixed_endpoint_mean": no_bc_mean,
        "official_fixed_endpoint_mean": official_mean,
        "mean_delta": no_bc_mean - official_mean,
        "official_reference_matches_locked_64_625pct": abs(
            official_mean - 0.64625
        ) <= TOLERANCE,
        "empirical_parity_or_better": parity,
        "goal_criterion_met": parity,
        "heldout_opened": True,
        "next_decision": (
            "complete_no_bc_cqnas_goal" if parity else "continue_reward_q_research"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage43-dir", type=Path, required=True)
    parser.add_argument(
        "--official-root", type=Path, default=Path("exp_local/pixel_cqn_as")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.stage43_dir, args.official_root)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
