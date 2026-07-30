#!/usr/bin/env python3
"""Evaluate and aggregate the four CQN DMC paper seeds."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from argparse import Namespace
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import eval_cqn_dmc_checkpoint as checkpoint_eval


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--num-eval-episodes", type=int, default=50)
    parser.add_argument("--eval-seed-start", type=int, default=20000)
    parser.add_argument(
        "--paper-reference-return",
        type=float,
        default=checkpoint_eval.PAPER_REFERENCE_RETURN,
    )
    parser.add_argument(
        "--paper-tolerance",
        type=float,
        default=checkpoint_eval.PAPER_REFERENCE_TOLERANCE,
    )
    return parser.parse_args()


def _run(args: argparse.Namespace) -> dict:
    if len(args.run_dirs) != 4:
        raise ValueError(
            "paper aggregation requires 4 run directories, "
            f"got {len(args.run_dirs)}"
        )
    checkpoint_eval._configure_process(args.gpu_id)
    results = []
    for run_dir in args.run_dirs:
        run_dir = run_dir.expanduser().resolve()
        results.append(
            checkpoint_eval._run_eval(
                Namespace(
                    run_dir=run_dir,
                    snapshot=None,
                    output=None,
                    work_dir=run_dir / "paper_eval_only",
                    gpu_id=args.gpu_id,
                    num_eval_episodes=args.num_eval_episodes,
                    eval_seed_start=args.eval_seed_start,
                    paper_reference_return=args.paper_reference_return,
                    paper_tolerance=args.paper_tolerance,
                )
            )
        )
    returns = [result["metrics"]["episode_reward"] for result in results]
    mean_return = statistics.fmean(returns)
    std_return = statistics.stdev(returns)
    reference = float(args.paper_reference_return)
    tolerance = float(args.paper_tolerance)
    delta = mean_return - reference
    if delta > tolerance:
        alignment = "above_reference_band"
    elif delta < -tolerance:
        alignment = "below_reference_band"
    else:
        alignment = "within_reference_band"
    return {
        "status": "ok",
        "task": checkpoint_eval.PAPER_TASK,
        "num_training_seeds": len(results),
        "num_eval_episodes_per_seed": int(args.num_eval_episodes),
        "seed_returns": returns,
        "mean_return": mean_return,
        "sample_std_return": std_return,
        "paper_comparison": {
            "source": "CQN paper Figure 9 endpoint, approximate visual reading",
            "reference_return": reference,
            "tolerance": tolerance,
            "return_delta": delta,
            "alignment": alignment,
            "meets_reference_lower_band": mean_return >= reference - tolerance,
        },
        "runs": results,
    }


def main() -> int:
    args = _parse_args()
    started = time.time()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = _run(args)
        payload["elapsed_seconds"] = time.time() - started
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except BaseException as exc:
        payload = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": time.time() - started,
        }
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
