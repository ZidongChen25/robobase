#!/usr/bin/env python3
"""Validation-select a causal-value/behavior-proxy score blend.

This is a discovery gate after a frozen direct-value ensemble has shown low
choice regret but weaker all-pairs ordering than the behavior proxies.  Blend
family and weight are selected only on the benchmark's internal validation
simulator seed.  The external branch seeds are then evaluated once with the
same authenticity checks as the unblended ensemble.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from benchmark_cqn_branch_value_models import _ranking_metrics
    from run_cqn_prediction_ensemble_gate import _random_top1_probability
    from run_cqn_structured_delta_gate import (
        BranchData,
        _atomic_json,
        _load_branch_data,
        _proxy_scores,
        _seed_bootstrap,
    )
except ImportError:
    from scripts.benchmark_cqn_branch_value_models import _ranking_metrics
    from scripts.run_cqn_prediction_ensemble_gate import (
        _random_top1_probability,
    )
    from scripts.run_cqn_structured_delta_gate import (
        BranchData,
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


def _float_list(value: str) -> list[float]:
    values = [
        float(item.strip()) for item in value.split(",") if item.strip()
    ]
    if (
        not values
        or len(values) != len(set(values))
        or any(
            not math.isfinite(item) or not 0.0 < item <= 1.0
            for item in values
        )
    ):
        raise argparse.ArgumentTypeError(
            "weights must be unique finite values in (0, 1]"
        )
    return values


def _proxy_list(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    allowed = {"behavior", "policy", "mean"}
    if (
        not values
        or len(values) != len(set(values))
        or not set(values).issubset(allowed)
    ):
        raise argparse.ArgumentTypeError(
            "proxies must be a unique subset of behavior,policy,mean"
        )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-summary", required=True, type=Path)
    parser.add_argument("--dataset-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", default="direct")
    parser.add_argument("--model-seeds", type=_integer_list, default=[1, 2, 3])
    parser.add_argument(
        "--weights",
        type=_float_list,
        default=[0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument(
        "--proxies",
        type=_proxy_list,
        default=["behavior", "policy", "mean"],
    )
    parser.add_argument("--return-atol", type=float, default=1e-12)
    parser.add_argument("--min-evaluation-seeds", type=int, default=64)
    parser.add_argument("--min-pairwise", type=float, default=0.55)
    parser.add_argument("--min-spearman", type=float, default=0.10)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=63)
    return parser.parse_args()


def row_standardize(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, np.float64)
    if scores.ndim != 2:
        raise ValueError("scores must have shape [states, candidates]")
    centered = scores - scores.mean(axis=1, keepdims=True)
    scale = np.sqrt(
        np.mean(np.square(centered), axis=1, keepdims=True) + 1e-6
    )
    return centered / scale


def blend_scores(
    model: np.ndarray,
    behavior: np.ndarray,
    policy: np.ndarray,
    *,
    proxy: str,
    model_weight: float,
) -> np.ndarray:
    if not 0.0 < model_weight <= 1.0:
        raise ValueError("model_weight must be in (0, 1]")
    model_score = row_standardize(model)
    behavior_score = row_standardize(behavior)
    policy_score = row_standardize(policy)
    if proxy == "behavior":
        proxy_score = behavior_score
    elif proxy == "policy":
        proxy_score = policy_score
    elif proxy == "mean":
        proxy_score = row_standardize(
            0.5 * (behavior_score + policy_score)
        )
    else:
        raise ValueError(f"unsupported proxy {proxy!r}")
    return (
        float(model_weight) * model_score
        + (1.0 - float(model_weight)) * proxy_score
    )


def _ensemble_predictions(
    summary: dict[str, Any],
    *,
    method: str,
    model_seeds: list[int],
    field: str,
) -> np.ndarray:
    by_seed: dict[int, np.ndarray] = {}
    for result in summary.get("results", []):
        if str(result.get("method")) != method:
            continue
        seed = int(result["seed"])
        if seed in by_seed:
            raise ValueError(f"duplicate prediction for model seed {seed}")
        if field not in result:
            raise KeyError(f"benchmark result is missing {field}")
        by_seed[seed] = np.asarray(result[field], np.float64)
    missing = [seed for seed in model_seeds if seed not in by_seed]
    if missing:
        raise ValueError(f"missing model seeds {missing}")
    arrays = [by_seed[seed] for seed in model_seeds]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("model prediction shapes differ")
    return np.mean(np.stack(arrays), axis=0)


def _seed_subset(data: BranchData, eval_seed: int) -> BranchData:
    mask = np.asarray(
        [
            int(record["eval_seed"]) == int(eval_seed)
            for record in data.metadata
        ],
        dtype=bool,
    )
    if not np.any(mask):
        raise ValueError(f"validation seed {eval_seed} is absent from cache")
    indices = np.flatnonzero(mask)
    return BranchData(
        features=data.features[mask],
        returns=data.returns[mask],
        action_dimensions=data.action_dimensions[mask],
        metadata=tuple(data.metadata[index] for index in indices),
    )


def _selection_key(metrics: dict[str, float], weight: float) -> tuple:
    return (
        metrics["pairwise_accuracy"],
        -metrics["regret"],
        metrics["mean_spearman"],
        metrics["top1_accuracy"],
        weight,
    )


def select_validation_blend(
    model: np.ndarray,
    behavior: np.ndarray,
    policy: np.ndarray,
    targets: np.ndarray,
    *,
    proxies: list[str],
    weights: list[float],
    return_atol: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for proxy in proxies:
        for weight in weights:
            scores = blend_scores(
                model,
                behavior,
                policy,
                proxy=proxy,
                model_weight=weight,
            )
            metrics = _ranking_metrics(
                scores,
                targets,
                return_atol=return_atol,
            )
            rows.append(
                {
                    "proxy": proxy,
                    "model_weight": float(weight),
                    "metrics": metrics,
                }
            )
    if not rows:
        raise ValueError("blend family is empty")
    selected = max(
        rows,
        key=lambda row: _selection_key(
            row["metrics"],
            row["model_weight"],
        ),
    )
    return selected, rows


def run_discovery(args: argparse.Namespace) -> dict[str, Any]:
    if args.min_evaluation_seeds < 1:
        raise ValueError("min-evaluation-seeds must be positive")
    if args.bootstrap_replicates < 1:
        raise ValueError("bootstrap-replicates must be positive")
    if args.return_atol < 0.0:
        raise ValueError("return-atol must be non-negative")

    summary_path = args.benchmark_summary.expanduser().resolve()
    cache_path = args.dataset_cache.expanduser().resolve()
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "complete":
        raise ValueError("benchmark summary is not complete")
    validation_seed = int(summary["validation_seed"])
    validation_data = _seed_subset(
        _load_branch_data(cache_path, ("train",)),
        validation_seed,
    )
    evaluation_data = _load_branch_data(cache_path, ("heldout",))
    validation_model = _ensemble_predictions(
        summary,
        method=args.method,
        model_seeds=args.model_seeds,
        field="selected_validation_predictions",
    )
    evaluation_model = _ensemble_predictions(
        summary,
        method=args.method,
        model_seeds=args.model_seeds,
        field="selected_heldout_predictions",
    )
    if validation_model.shape != validation_data.returns.shape:
        raise ValueError("validation prediction/data shapes differ")
    if evaluation_model.shape != evaluation_data.returns.shape:
        raise ValueError("evaluation prediction/data shapes differ")

    validation_behavior, validation_policy = _proxy_scores(
        validation_data.metadata
    )
    behavior, policy = _proxy_scores(evaluation_data.metadata)
    selected, validation_rows = select_validation_blend(
        validation_model,
        validation_behavior,
        validation_policy,
        validation_data.returns,
        proxies=args.proxies,
        weights=args.weights,
        return_atol=args.return_atol,
    )
    scores = blend_scores(
        evaluation_model,
        behavior,
        policy,
        proxy=selected["proxy"],
        model_weight=selected["model_weight"],
    )
    score_sets = {
        "model": scores,
        "behavior": behavior,
        "policy": policy,
    }
    metrics = {
        name: _ranking_metrics(
            values,
            evaluation_data.returns,
            return_atol=args.return_atol,
        )
        for name, values in score_sets.items()
    }
    bootstrap = _seed_bootstrap(
        evaluation_data,
        score_sets,
        return_atol=args.return_atol,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    random_top1 = _random_top1_probability(
        evaluation_data.returns,
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
        "stage": "discovery",
        "gate": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "benchmark_summary": str(summary_path),
        "dataset_cache": str(cache_path),
        "method": args.method,
        "model_seeds": args.model_seeds,
        "ensemble_reducer": "arithmetic_mean",
        "normalization": "per-state RMS z-score over sibling bins",
        "validation_seed": validation_seed,
        "validation_selection_protocol": (
            "maximize pairwise, minimize regret, then maximize Spearman, "
            "top1, and model weight"
        ),
        "selected_blend": selected,
        "validation_rows": validation_rows,
        "evaluation_seed_ids": sorted(
            {
                int(record["eval_seed"])
                for record in evaluation_data.metadata
            }
        ),
        "metrics": metrics,
        "random_top1_probability": random_top1,
        "seed_bootstrap": bootstrap,
        "thresholds": {
            "min_evaluation_seeds": args.min_evaluation_seeds,
            "min_pairwise": args.min_pairwise,
            "min_spearman": args.min_spearman,
            "return_atol": args.return_atol,
        },
        "next_gate": (
            "collect a new external confirmation cache before deployment"
            if all(checks.values())
            else "close direct/proxy blend; retain structured audit only"
        ),
    }


def main() -> None:
    args = parse_args()
    payload = run_discovery(args)
    _atomic_json(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
