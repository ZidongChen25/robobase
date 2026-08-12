#!/usr/bin/env python3
"""Run fixed-state diagnostics for every selected Stage-30 checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


DEFAULT_DATA_RUN = Path(
    "exp_local/cqn_value_fidelity_stage2/"
    "move_plate_full_first_success_seed1_gpu3_20260722_165946"
)


def _selected_jobs(stage_dir: Path, summary: dict) -> list[dict[str, str]]:
    jobs = []
    for family, label in (
        ("matched_online_only_controls", "online_only"),
        ("offline_then_online_treatments", "offline_then_online"),
    ):
        for seed, arm in summary[family].items():
            jobs.append(
                {
                    "name": f"{label}_{seed}_selected",
                    "run_dir": arm["run_dir"],
                    "snapshot": arm["selected_snapshot"],
                }
            )
    for seed, arm in summary["offline_then_online_treatments"].items():
        jobs.append(
            {
                "name": f"offline_endpoint_{seed}",
                "run_dir": arm["run_dir"],
                "snapshot": arm["offline_endpoint_snapshot"],
            }
        )
    return jobs


def _diagnostic_metrics(payload: dict) -> dict:
    demo = payload["summary"]["demo_success"]
    return {
        "snapshot": payload["snapshot"],
        "demo_success_samples": demo["num_samples"],
        "expert_bin_top1": demo["imitation"]["replay_bin_top1_rate"],
        "expert_bin_top1_current_action": demo["imitation"][
            "replay_bin_top1_rate_current_action"
        ],
        "expert_bin_top2": demo["imitation"]["replay_bin_top2_rate"],
        "expert_bin_top2_current_action": demo["imitation"][
            "replay_bin_top2_rate_current_action"
        ],
        "expert_q": demo["value"]["predicted_q_mean"],
        "greedy_q": demo["value"]["greedy_predicted_q_mean"],
        "expert_minus_greedy_q": demo["value"][
            "replay_minus_greedy_q_mean"
        ],
        "rtg_mae": demo["value"]["q_raw_return_mae"],
        "rtg_mae_by_terminal_distance": demo["value"][
            "q_raw_return_mae_by_terminal_distance"
        ],
        "q_rtg_pearson": demo["value"]["q_raw_return_pearson"],
        "max_minus_expert_q": demo["collapse"]["max_minus_replay_q"],
        "candidate_q_span": demo["collapse"]["candidate_q_span"],
    }


def run(args: argparse.Namespace) -> dict:
    repo = Path(__file__).resolve().parents[1]
    stage_dir = args.stage_dir.expanduser().resolve()
    data_run = args.data_run.expanduser().resolve()
    summary_path = stage_dir / "stage30_summary.json"
    summary = json.loads(summary_path.read_text())
    output_dir = stage_dir / "diagnostics"
    output_dir.mkdir(exist_ok=True)

    env = os.environ.copy()
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("MUJOCO_GL", "egl")
    results = {}
    for job in _selected_jobs(stage_dir, summary):
        output = output_dir / f"{job['name']}.json"
        log = output_dir / f"{job['name']}.log"
        command = [
            sys.executable,
            str(repo / "scripts/analyze_cqn_value_fidelity.py"),
            "--run-dir",
            job["run_dir"],
            "--snapshot",
            job["snapshot"],
            "--data-run-dir",
            str(data_run),
            "--output",
            str(output),
            "--gpu-id",
            str(args.gpu_id),
            "--samples-per-group",
            str(args.samples_per_group),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(args.sample_seed),
            "--offline-episode-count",
            str(args.offline_episode_count),
        ]
        with log.open("w") as handle:
            subprocess.run(
                command,
                cwd=repo,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        payload = json.loads(output.read_text())
        if payload.get("status") != "ok":
            raise RuntimeError(f"diagnostic failed: {output}")
        results[job["name"]] = _diagnostic_metrics(payload)

    result = {
        "protocol": {
            "read_only": True,
            "checkpoint_source": "Stage-30 validation-selected checkpoints",
            "common_data_run": str(data_run),
            "samples_per_group": args.samples_per_group,
            "sample_seed": args.sample_seed,
            "offline_episode_count": args.offline_episode_count,
            "heldout_environment_seeds_consumed": False,
        },
        "diagnostics": results,
    }
    output = stage_dir / "stage30_diagnostics_summary.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--data-run", type=Path, default=DEFAULT_DATA_RUN)
    parser.add_argument("--samples-per-group", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--offline-episode-count", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
