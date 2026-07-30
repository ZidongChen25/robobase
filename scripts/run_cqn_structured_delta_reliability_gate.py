#!/usr/bin/env python3
"""Cross-fit and gate a reliability-constrained structured CQN value model.

The value predictor is fitted only to realized same-state branch returns.
Behavior and policy proxies are used only on the historical fit cache to
identify action dimensions and trajectory anchors where leave-one-seed-out
value predictions beat both proxies.  That support set is then frozen before
the model is evaluated on a disjoint cache.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from benchmark_cqn_branch_value_models import _ranking_metrics
    from run_cqn_structured_delta_gate import (
        BranchData,
        _atomic_json,
        _candidate_scores,
        _load_branch_data,
        _proxy_scores,
        _seed_bootstrap,
        _split_names,
        _write_model,
        fit_structured_delta_model,
        predict_optimal_delta,
    )
except ImportError:
    from scripts.benchmark_cqn_branch_value_models import _ranking_metrics
    from scripts.run_cqn_structured_delta_gate import (
        BranchData,
        _atomic_json,
        _candidate_scores,
        _load_branch_data,
        _proxy_scores,
        _seed_bootstrap,
        _split_names,
        _write_model,
        fit_structured_delta_model,
        predict_optimal_delta,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-cache", required=True, type=Path)
    parser.add_argument("--evaluation-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fit-splits", default="train")
    parser.add_argument("--evaluation-splits", default="train,heldout")
    parser.add_argument("--pca-components", type=int, default=4)
    parser.add_argument("--ridge-alpha", type=float, default=0.01)
    parser.add_argument("--return-atol", type=float, default=1e-12)
    parser.add_argument("--min-support-informative", type=int, default=20)
    parser.add_argument("--min-evaluation-informative", type=int, default=40)
    parser.add_argument("--min-evaluation-seeds", type=int, default=16)
    parser.add_argument("--min-pairwise", type=float, default=0.55)
    parser.add_argument("--min-spearman", type=float, default=0.10)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=47)
    return parser.parse_args()


def _subset(data: BranchData, mask: np.ndarray) -> BranchData:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (data.features.shape[0],):
        raise ValueError("branch subset mask has the wrong shape")
    return BranchData(
        features=data.features[mask],
        returns=data.returns[mask],
        action_dimensions=data.action_dimensions[mask],
        metadata=tuple(
            record
            for record, keep in zip(data.metadata, mask, strict=True)
            if keep
        ),
    )


def _seed_ids(data: BranchData) -> np.ndarray:
    return np.asarray(
        sorted({int(record["eval_seed"]) for record in data.metadata}),
        np.int64,
    )


def _crossfit_predictions(
    data: BranchData,
    *,
    pca_components: int,
    ridge_alpha: float,
    return_atol: float,
) -> tuple[np.ndarray, list[int]]:
    seed_ids = _seed_ids(data)
    if seed_ids.size < 3:
        raise ValueError("cross-fitting requires at least three simulator seeds")
    record_seeds = np.asarray(
        [int(record["eval_seed"]) for record in data.metadata],
        np.int64,
    )
    prediction = np.full(data.features.shape[0], np.nan, np.float64)
    for heldout_seed in seed_ids:
        evaluation_mask = record_seeds == heldout_seed
        fit_mask = ~evaluation_mask
        model = fit_structured_delta_model(
            _subset(data, fit_mask),
            pca_components=pca_components,
            ridge_alpha=ridge_alpha,
            return_atol=return_atol,
        )
        prediction[evaluation_mask] = predict_optimal_delta(
            model,
            _subset(data, evaluation_mask),
        )
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("cross-fitted prediction contains non-finite values")
    return prediction, seed_ids.tolist()


def _metrics(
    data: BranchData,
    score_sets: dict[str, np.ndarray],
    *,
    return_atol: float,
) -> dict[str, dict[str, float]]:
    return {
        name: _ranking_metrics(
            scores,
            data.returns,
            return_atol=return_atol,
        )
        for name, scores in score_sets.items()
    }


def _reliable_group(
    metrics: dict[str, dict[str, float]],
    *,
    min_informative: int,
) -> tuple[bool, dict[str, bool]]:
    model = metrics["model"]
    behavior = metrics["behavior"]
    policy = metrics["policy"]
    checks = {
        "enough_informative_states": (
            model["num_informative_states"] >= min_informative
        ),
        "pairwise_above_behavior": (
            model["pairwise_accuracy"] > behavior["pairwise_accuracy"]
        ),
        "pairwise_above_policy": (
            model["pairwise_accuracy"] > policy["pairwise_accuracy"]
        ),
        "regret_below_behavior": model["regret"] < behavior["regret"],
        "regret_below_policy": model["regret"] < policy["regret"],
        "positive_spearman": model["mean_spearman"] > 0.0,
    }
    return all(checks.values()), checks


def _factor_report(
    data: BranchData,
    score_sets: dict[str, np.ndarray],
    *,
    field: str,
    values: Iterable[int],
    min_informative: int,
    return_atol: float,
) -> tuple[dict[str, Any], list[int]]:
    report: dict[str, Any] = {}
    supported = []
    metadata_values = np.asarray(
        [int(record[field]) for record in data.metadata],
        np.int32,
    )
    for value in values:
        mask = metadata_values == int(value)
        group_data = _subset(data, mask)
        group_scores = {
            name: scores[mask] for name, scores in score_sets.items()
        }
        metrics = _metrics(
            group_data,
            group_scores,
            return_atol=return_atol,
        )
        reliable, checks = _reliable_group(
            metrics,
            min_informative=min_informative,
        )
        if reliable:
            supported.append(int(value))
        report[str(int(value))] = {
            "supported": reliable,
            "checks": checks,
            "metrics": metrics,
        }
    return report, supported


def derive_crossfit_support(
    data: BranchData,
    *,
    pca_components: int,
    ridge_alpha: float,
    return_atol: float,
    min_informative: int,
) -> dict[str, Any]:
    prediction, seed_ids = _crossfit_predictions(
        data,
        pca_components=pca_components,
        ridge_alpha=ridge_alpha,
        return_atol=return_atol,
    )
    behavior, policy = _proxy_scores(data.metadata)
    score_sets = {
        "model": _candidate_scores(prediction, data.metadata),
        "behavior": behavior,
        "policy": policy,
    }
    anchors = sorted(
        {int(record["anchor_step"]) for record in data.metadata}
    )
    dimensions = list(range(int(np.max(data.action_dimensions)) + 1))
    anchor_report, supported_anchors = _factor_report(
        data,
        score_sets,
        field="anchor_step",
        values=anchors,
        min_informative=min_informative,
        return_atol=return_atol,
    )
    dimension_report, supported_dimensions = _factor_report(
        data,
        score_sets,
        field="action_dimension",
        values=dimensions,
        min_informative=min_informative,
        return_atol=return_atol,
    )
    support_mask = np.asarray(
        [
            int(record["anchor_step"]) in supported_anchors
            and int(record["action_dimension"]) in supported_dimensions
            for record in data.metadata
        ],
        dtype=bool,
    )
    supported_data = _subset(data, support_mask)
    supported_scores = {
        name: values[support_mask] for name, values in score_sets.items()
    }
    return {
        "crossfit_seed_ids": seed_ids,
        "selection_rule": {
            "min_informative_states_per_factor": int(min_informative),
            "requirements": [
                "model pairwise > behavior and policy",
                "model regret < behavior and policy",
                "model mean Spearman > 0",
            ],
            "factor_intersection": "anchor_step AND action_dimension",
        },
        "by_anchor_step": anchor_report,
        "by_action_dimension": dimension_report,
        "supported_anchor_steps": supported_anchors,
        "supported_action_dimensions": supported_dimensions,
        "supported_intersection_num_states": int(support_mask.sum()),
        "supported_intersection_metrics": (
            _metrics(
                supported_data,
                supported_scores,
                return_atol=return_atol,
            )
            if support_mask.any()
            else {}
        ),
    }


def _random_top1_probability(
    returns: np.ndarray,
    *,
    return_atol: float,
) -> float:
    informative = [
        row for row in returns if np.ptp(row) > return_atol
    ]
    if not informative:
        return math.nan
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
    if args.return_atol < 0.0:
        raise ValueError("return_atol must be non-negative")
    if args.min_support_informative < 1:
        raise ValueError("min-support-informative must be positive")
    if args.min_evaluation_informative < 1:
        raise ValueError("min-evaluation-informative must be positive")
    if args.min_evaluation_seeds < 2:
        raise ValueError("min-evaluation-seeds must be at least two")
    if args.bootstrap_replicates < 1:
        raise ValueError("bootstrap-replicates must be positive")

    fit_splits = _split_names(args.fit_splits)
    evaluation_splits = _split_names(args.evaluation_splits)
    fit_data = _load_branch_data(args.fit_cache, fit_splits)
    evaluation_all = _load_branch_data(
        args.evaluation_cache,
        evaluation_splits,
    )
    fit_seed_ids = set(_seed_ids(fit_data).tolist())
    evaluation_seed_ids = set(_seed_ids(evaluation_all).tolist())
    overlap = sorted(fit_seed_ids & evaluation_seed_ids)
    if overlap:
        raise ValueError(
            f"fit and evaluation simulator seeds overlap: {overlap}"
        )

    support = derive_crossfit_support(
        fit_data,
        pca_components=args.pca_components,
        ridge_alpha=args.ridge_alpha,
        return_atol=args.return_atol,
        min_informative=args.min_support_informative,
    )
    supported_anchors = support["supported_anchor_steps"]
    supported_dimensions = support["supported_action_dimensions"]
    evaluation_mask = np.asarray(
        [
            int(record["anchor_step"]) in supported_anchors
            and int(record["action_dimension"]) in supported_dimensions
            for record in evaluation_all.metadata
        ],
        dtype=bool,
    )
    if not evaluation_mask.any():
        raise ValueError("cross-fitted reliability support is empty")
    evaluation = _subset(evaluation_all, evaluation_mask)

    model = fit_structured_delta_model(
        fit_data,
        pca_components=args.pca_components,
        ridge_alpha=args.ridge_alpha,
        return_atol=args.return_atol,
    )
    prediction = predict_optimal_delta(model, evaluation)
    behavior, policy = _proxy_scores(evaluation.metadata)
    score_sets = {
        "model": _candidate_scores(prediction, evaluation.metadata),
        "behavior": behavior,
        "policy": policy,
    }
    metrics = _metrics(
        evaluation,
        score_sets,
        return_atol=args.return_atol,
    )
    bootstrap = _seed_bootstrap(
        evaluation,
        score_sets,
        return_atol=args.return_atol,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    model_metrics = metrics["model"]
    paired = bootstrap["paired_deltas"]
    random_top1 = _random_top1_probability(
        evaluation.returns,
        return_atol=args.return_atol,
    )
    checks = {
        "nonempty_supported_anchor_set": bool(supported_anchors),
        "nonempty_supported_dimension_set": bool(supported_dimensions),
        "enough_evaluation_seeds": (
            len(evaluation_seed_ids) >= args.min_evaluation_seeds
        ),
        "enough_evaluation_informative_states": (
            model_metrics["num_informative_states"]
            >= args.min_evaluation_informative
        ),
        "pairwise_above_absolute_threshold": (
            model_metrics["pairwise_accuracy"] > args.min_pairwise
        ),
        "spearman_above_absolute_threshold": (
            model_metrics["mean_spearman"] > args.min_spearman
        ),
        "top1_above_random": model_metrics["top1_accuracy"] > random_top1,
        "pairwise_above_behavior": (
            model_metrics["pairwise_accuracy"]
            > metrics["behavior"]["pairwise_accuracy"]
        ),
        "pairwise_above_policy": (
            model_metrics["pairwise_accuracy"]
            > metrics["policy"]["pairwise_accuracy"]
        ),
        "regret_below_behavior": (
            model_metrics["regret"] < metrics["behavior"]["regret"]
        ),
        "regret_below_policy": (
            model_metrics["regret"] < metrics["policy"]["regret"]
        ),
        "spearman_ci_lower_above_zero": (
            bootstrap["metrics"]["model"]["mean_spearman"][0] > 0.0
        ),
        "pairwise_delta_behavior_ci_lower_nonnegative": (
            paired["model_minus_behavior"]["pairwise_accuracy"][0] >= 0.0
        ),
        "pairwise_delta_policy_ci_lower_nonnegative": (
            paired["model_minus_policy"]["pairwise_accuracy"][0] >= 0.0
        ),
        "regret_improvement_behavior_ci_lower_nonnegative": (
            paired["model_minus_behavior"]["regret_improvement"][0] >= 0.0
        ),
        "regret_improvement_policy_ci_lower_nonnegative": (
            paired["model_minus_policy"]["regret_improvement"][0] >= 0.0
        ),
    }
    gate = "pass" if all(checks.values()) else "fail"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "structured_delta_reliable_model.npz"
    _write_model(
        model_path,
        model,
        supported_anchor_steps=np.asarray(supported_anchors, np.int32),
        supported_action_dimensions=np.asarray(
            supported_dimensions,
            np.int32,
        ),
    )
    return {
        "status": "ok",
        "gate": gate,
        "checks": checks,
        "fit_cache": str(args.fit_cache.resolve()),
        "fit_splits": list(fit_splits),
        "evaluation_cache": str(args.evaluation_cache.resolve()),
        "evaluation_splits": list(evaluation_splits),
        "fit_seed_ids": sorted(fit_seed_ids),
        "evaluation_seed_ids": sorted(evaluation_seed_ids),
        "fit_num_states": int(fit_data.features.shape[0]),
        "evaluation_num_states_total": int(
            evaluation_all.features.shape[0]
        ),
        "evaluation_num_states_supported": int(
            evaluation.features.shape[0]
        ),
        "support": support,
        "model": {
            "path": str(model_path.resolve()),
            "semantics": "-abs(candidate_delta - predicted_optimal_delta)",
            "pca_components": int(model.state_components.shape[0]),
            "ridge_alpha": float(model.ridge_alpha),
            "parameter_count": int(model.ridge_weights.size),
            "action_dim_count": int(model.action_dim_count),
            "anchor_steps": model.anchor_steps.tolist(),
            "supported_anchor_steps": supported_anchors,
            "supported_action_dimensions": supported_dimensions,
        },
        "thresholds": {
            "min_support_informative": int(
                args.min_support_informative
            ),
            "min_evaluation_informative": int(
                args.min_evaluation_informative
            ),
            "min_evaluation_seeds": int(args.min_evaluation_seeds),
            "min_pairwise": float(args.min_pairwise),
            "min_spearman": float(args.min_spearman),
            "bootstrap_replicates": int(args.bootstrap_replicates),
        },
        "metrics": metrics,
        "random_top1_probability": random_top1,
        "seed_bootstrap": bootstrap,
        "predicted_optimal_delta": {
            "min": float(np.min(prediction)),
            "max": float(np.max(prediction)),
            "mean": float(np.mean(prediction)),
            "std": float(np.std(prediction)),
        },
    }


def main() -> int:
    args = parse_args()
    output = args.output_dir.expanduser().resolve() / "gate_summary.json"
    started = time.monotonic()
    try:
        payload = run_gate(args)
        payload["elapsed_seconds"] = float(time.monotonic() - started)
        _atomic_json(output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        _atomic_json(
            output,
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": float(time.monotonic() - started),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
