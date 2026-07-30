#!/usr/bin/env python3
"""Gate a frozen prediction ensemble on an external CQN branch cache."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from benchmark_cqn_branch_value_models import _ranking_metrics
    from run_cqn_structured_delta_gate import (
        _atomic_json,
        _load_branch_data,
        _proxy_scores,
        _seed_bootstrap,
    )
except ImportError:
    from scripts.benchmark_cqn_branch_value_models import _ranking_metrics
    from scripts.run_cqn_structured_delta_gate import (
        _atomic_json,
        _load_branch_data,
        _proxy_scores,
        _seed_bootstrap,
    )


def _integer_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(
            "expected a non-empty list of unique integers"
        )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-summary", required=True, type=Path)
    parser.add_argument("--evaluation-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", default="direct")
    parser.add_argument("--model-seeds", type=_integer_list, default=[1, 2, 3])
    parser.add_argument("--min-evaluation-seeds", type=int, default=64)
    parser.add_argument("--min-pairwise", type=float, default=0.55)
    parser.add_argument("--min-spearman", type=float, default=0.10)
    parser.add_argument("--return-atol", type=float, default=1e-12)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=61)
    return parser.parse_args()


def ensemble_predictions(
    summary: dict[str, Any],
    *,
    method: str,
    model_seeds: list[int],
) -> np.ndarray:
    if summary.get("status") != "complete":
        raise ValueError("benchmark summary is not complete")
    by_seed: dict[int, np.ndarray] = {}
    for result in summary.get("results", []):
        if str(result.get("method")) != method:
            continue
        seed = int(result["seed"])
        if seed in by_seed:
            raise ValueError(f"duplicate prediction for model seed {seed}")
        by_seed[seed] = np.asarray(
            result["selected_heldout_predictions"],
            np.float64,
        )
    missing = [seed for seed in model_seeds if seed not in by_seed]
    extra = sorted(set(by_seed).difference(model_seeds))
    if missing or extra:
        raise ValueError(
            f"model seed mismatch: missing={missing}, extra={extra}"
        )
    shapes = {by_seed[seed].shape for seed in model_seeds}
    if len(shapes) != 1:
        raise ValueError(f"prediction shapes differ: {sorted(shapes)}")
    values = np.stack([by_seed[seed] for seed in model_seeds], axis=0)
    if not np.all(np.isfinite(values)):
        raise ValueError("prediction ensemble contains non-finite values")
    return np.mean(values, axis=0)


def _random_top1_probability(
    returns: np.ndarray,
    *,
    return_atol: float,
) -> float:
    informative = returns[
        np.ptp(returns, axis=1) > float(return_atol)
    ]
    if not informative.size:
        raise ValueError("evaluation cache has no informative states")
    return float(
        np.mean(
            [
                np.count_nonzero(
                    np.isclose(
                        row,
                        np.max(row),
                        atol=return_atol,
                        rtol=0.0,
                    )
                )
                / row.size
                for row in informative
            ]
        )
    )


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    if args.min_evaluation_seeds < 1:
        raise ValueError("min-evaluation-seeds must be positive")
    if args.bootstrap_replicates < 0:
        raise ValueError("bootstrap-replicates must be non-negative")
    if args.return_atol < 0.0:
        raise ValueError("return-atol must be non-negative")
    summary_path = args.benchmark_summary.expanduser().resolve()
    cache_path = args.evaluation_cache.expanduser().resolve()
    summary = json.loads(summary_path.read_text())
    predictions = ensemble_predictions(
        summary,
        method=str(args.method),
        model_seeds=list(args.model_seeds),
    )
    data = _load_branch_data(cache_path, ("heldout",))
    if predictions.shape != data.returns.shape:
        raise ValueError(
            f"prediction/evaluation shape mismatch: "
            f"{predictions.shape} != {data.returns.shape}"
        )
    behavior, policy = _proxy_scores(data.metadata)
    score_sets = {
        "model": predictions,
        "behavior": behavior,
        "policy": policy,
    }
    metrics = {
        name: _ranking_metrics(
            scores,
            data.returns,
            return_atol=args.return_atol,
        )
        for name, scores in score_sets.items()
    }
    bootstrap = _seed_bootstrap(
        data,
        score_sets,
        return_atol=args.return_atol,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    random_top1 = _random_top1_probability(
        data.returns,
        return_atol=args.return_atol,
    )
    model = metrics["model"]
    model_ci = bootstrap["metrics"]["model"]
    checks = {
        "enough_evaluation_seeds": (
            bootstrap["num_seeds"] >= args.min_evaluation_seeds
        ),
        "pairwise_above_absolute_threshold": (
            model["pairwise_accuracy"] > args.min_pairwise
        ),
        "pairwise_above_behavior": (
            model["pairwise_accuracy"]
            > metrics["behavior"]["pairwise_accuracy"]
        ),
        "pairwise_above_policy": (
            model["pairwise_accuracy"]
            > metrics["policy"]["pairwise_accuracy"]
        ),
        "spearman_above_absolute_threshold": (
            model["mean_spearman"] > args.min_spearman
        ),
        "top1_above_random": model["top1_accuracy"] > random_top1,
        "regret_below_behavior": (
            model["regret"] < metrics["behavior"]["regret"]
        ),
        "regret_below_policy": (
            model["regret"] < metrics["policy"]["regret"]
        ),
        "spearman_ci_lower_above_zero": (
            model_ci["mean_spearman"][0] > 0.0
        ),
        "pairwise_delta_behavior_ci_lower_nonnegative": (
            bootstrap["paired_deltas"]["model_minus_behavior"][
                "pairwise_accuracy"
            ][0]
            >= 0.0
        ),
        "pairwise_delta_policy_ci_lower_nonnegative": (
            bootstrap["paired_deltas"]["model_minus_policy"][
                "pairwise_accuracy"
            ][0]
            >= 0.0
        ),
        "regret_improvement_behavior_ci_lower_nonnegative": (
            bootstrap["paired_deltas"]["model_minus_behavior"][
                "regret_improvement"
            ][0]
            >= 0.0
        ),
        "regret_improvement_policy_ci_lower_nonnegative": (
            bootstrap["paired_deltas"]["model_minus_policy"][
                "regret_improvement"
            ][0]
            >= 0.0
        ),
    }
    return {
        "status": "ok",
        "gate": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "benchmark_summary": str(summary_path),
        "evaluation_cache": str(cache_path),
        "method": str(args.method),
        "model_seeds": list(args.model_seeds),
        "ensemble_reducer": "arithmetic_mean",
        "evaluation_seed_ids": sorted(
            {int(record["eval_seed"]) for record in data.metadata}
        ),
        "metrics": metrics,
        "random_top1_probability": random_top1,
        "seed_bootstrap": bootstrap,
        "thresholds": {
            "min_evaluation_seeds": int(args.min_evaluation_seeds),
            "min_pairwise": float(args.min_pairwise),
            "min_spearman": float(args.min_spearman),
            "return_atol": float(args.return_atol),
        },
    }


def main() -> None:
    args = parse_args()
    payload = run_gate(args)
    output = args.output.expanduser().resolve()
    _atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
