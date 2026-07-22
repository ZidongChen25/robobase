#!/usr/bin/env python3
"""Evaluate and aggregate multiple CQN-AS move_plate training runs."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

from eval_cqn_as_bigym_checkpoint import (
    PAPER_ENDPOINT_STD_PERCENT,
    PAPER_ENDPOINT_SUCCESS_PERCENT,
    configure_process,
    run_eval,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path("cqn_as_move_plate_eval.json"))
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--num-eval-episodes", type=int, default=25)
    parser.add_argument("--eval-seed-start", type=int, default=20000)
    parser.add_argument(
        "--method-temporal-ensemble",
        choices=("config", "true", "false"),
        default="config",
    )
    parser.add_argument("--temporal-ensemble-replan-interval", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_process(args.gpu_id)
    started = time.time()
    results = []
    for index, run_dir in enumerate(args.run_dir):
        result = run_eval(
            SimpleNamespace(
                run_dir=run_dir,
                snapshot=None,
                output=None,
                work_dir=run_dir / f"eval_paper_seed_{args.eval_seed_start}",
                gpu_id=args.gpu_id,
                num_eval_episodes=args.num_eval_episodes,
                eval_seed_start=args.eval_seed_start,
                method_temporal_ensemble=args.method_temporal_ensemble,
                temporal_ensemble_replan_interval=(
                    args.temporal_ensemble_replan_interval
                ),
                single_run_tolerance_percent=20.0,
            )
        )
        result["aggregate_index"] = index
        results.append(result)

    successes = [result["success_percent"] for result in results]
    mean = statistics.fmean(successes)
    std = statistics.pstdev(successes) if len(successes) > 1 else 0.0
    payload = {
        "status": "ok",
        "task": "move_plate",
        "num_runs": len(results),
        "num_eval_episodes_per_run": args.num_eval_episodes,
        "eval_seed_start": args.eval_seed_start,
        "method_temporal_ensemble": args.method_temporal_ensemble,
        "temporal_ensemble_replan_interval": (
            args.temporal_ensemble_replan_interval
        ),
        "success_percent_by_run": successes,
        "mean_success_percent": mean,
        "std_success_percent": std,
        "paper_comparison": {
            "source": "official CQN-AS bigym_results.pkl at 100000 environment steps",
            "paper_num_runs": 8,
            "reference_mean_success_percent": PAPER_ENDPOINT_SUCCESS_PERCENT,
            "reference_std_percent": PAPER_ENDPOINT_STD_PERCENT,
            "mean_delta_percent": mean - PAPER_ENDPOINT_SUCCESS_PERCENT,
        },
        "runs": results,
        "elapsed_seconds": time.time() - started,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
