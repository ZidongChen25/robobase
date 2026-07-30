#!/usr/bin/env python3
"""Fit and gate a low-capacity causal local-delta value sidecar.

The branch oracle supplies five sibling-bin continuations for the same
simulator state and action coordinate.  Instead of allowing a large C51 head
to memorize every branch, this model predicts one interpretable quantity:
the locally optimal action delta.  Candidate value is then

    score(s, d, delta) = -abs(delta - predicted_optimal_delta(s, d)).

The state dependence is deliberately low-rank.  A PCA projection of the
frozen CQN feature interacts with an action-dimension one-hot vector, while
anchor and dimension biases remain additive.  PCA and ridge fitting only read
the requested fit split.  Evaluation caches can therefore be collected after
the model and hyperparameters have been frozen.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from benchmark_cqn_branch_value_models import (
        _ranking_metrics,
        _rankdata,
    )
except ImportError:
    from scripts.benchmark_cqn_branch_value_models import (
        _ranking_metrics,
        _rankdata,
    )


@dataclass(frozen=True)
class BranchData:
    features: np.ndarray
    returns: np.ndarray
    action_dimensions: np.ndarray
    metadata: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class StructuredDeltaModel:
    state_mean: np.ndarray
    state_components: np.ndarray
    state_scale: np.ndarray
    ridge_weights: np.ndarray
    anchor_steps: np.ndarray
    action_dim_count: int
    ridge_alpha: float


@dataclass(frozen=True)
class SeedStats:
    pair_correct: float
    pair_total: float
    spearman_sum: float
    spearman_count: float
    top1_sum: float
    state_count: float
    regret_sum: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-cache", required=True, type=Path)
    parser.add_argument("--evaluation-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fit-splits", default="train")
    parser.add_argument("--evaluation-splits", default="heldout")
    parser.add_argument("--pca-components", type=int, default=4)
    parser.add_argument("--ridge-alpha", type=float, default=0.01)
    parser.add_argument("--return-atol", type=float, default=1e-12)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=37)
    parser.add_argument("--min-pairwise", type=float, default=0.55)
    parser.add_argument("--min-spearman", type=float, default=0.10)
    parser.add_argument(
        "--top1-reference",
        choices=("proxies", "random"),
        default="proxies",
        help=(
            "proxies requires top-1 to beat behavior and policy as well as "
            "random. random keeps pairwise and regret as the anti-cheat proxy "
            "checks and treats discontinuous top-1 only as a sanity check."
        ),
    )
    parser.add_argument(
        "--require-bootstrap",
        action="store_true",
        help=(
            "Require seed-cluster bootstrap lower bounds to establish positive "
            "pairwise/regret effects versus both non-return proxies."
        ),
    )
    return parser.parse_args()


def _split_names(value: str) -> tuple[str, ...]:
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if not names or any(name not in {"train", "heldout"} for name in names):
        raise ValueError("splits must be a comma-separated subset of train,heldout")
    if len(names) != len(set(names)):
        raise ValueError("split names must be unique")
    return names


def _load_branch_data(path: Path, splits: Iterable[str]) -> BranchData:
    features = []
    returns = []
    action_dimensions = []
    metadata: list[dict[str, Any]] = []
    with np.load(path, allow_pickle=False) as data:
        for split in splits:
            required = (
                f"{split}_features",
                f"{split}_returns",
                f"{split}_action_dimensions",
                f"{split}_metadata",
            )
            missing = [name for name in required if name not in data]
            if missing:
                raise KeyError(f"{path} is missing {missing}")
            split_features = np.asarray(data[required[0]], np.float32)
            split_returns = np.asarray(data[required[1]], np.float32)
            split_dimensions = np.asarray(data[required[2]], np.int32)
            split_metadata = json.loads(
                str(np.asarray(data[required[3]]).item())
            )
            if not (
                split_features.shape[0]
                == split_returns.shape[0]
                == split_dimensions.shape[0]
            ):
                raise ValueError(f"{path}:{split} has inconsistent row counts")
            if split_features.shape[0] != len(split_metadata):
                raise ValueError(f"{path}:{split} metadata count differs")
            if split_returns.ndim != 2 or split_returns.shape[1] < 2:
                raise ValueError("branch returns must have shape [states, bins]")
            features.append(split_features)
            returns.append(split_returns)
            action_dimensions.append(split_dimensions)
            metadata.extend(split_metadata)
    return BranchData(
        features=np.concatenate(features, axis=0),
        returns=np.concatenate(returns, axis=0),
        action_dimensions=np.concatenate(action_dimensions, axis=0),
        metadata=tuple(metadata),
    )


def _anchor_indices(
    metadata: tuple[dict[str, Any], ...],
    anchor_steps: np.ndarray,
) -> np.ndarray:
    mapping = {int(step): index for index, step in enumerate(anchor_steps)}
    try:
        return np.asarray(
            [mapping[int(record["anchor_step"])] for record in metadata],
            np.int32,
        )
    except KeyError as error:
        raise ValueError(
            f"evaluation contains unseen anchor step {error.args[0]}"
        ) from error


def _optimal_deltas(
    data: BranchData,
    *,
    return_atol: float,
) -> np.ndarray:
    targets = []
    for returns, record in zip(
        data.returns,
        data.metadata,
        strict=True,
    ):
        deltas = np.asarray(record["actual_first_delta"], np.float64)
        if deltas.shape != returns.shape:
            raise ValueError("actual_first_delta must match candidate returns")
        optimum = np.flatnonzero(
            np.isclose(
                returns,
                np.max(returns),
                atol=return_atol,
                rtol=0.0,
            )
        )
        targets.append(float(np.mean(deltas[optimum])))
    return np.asarray(targets, np.float64)


def _state_projection(
    features: np.ndarray,
    *,
    state_mean: np.ndarray,
    components: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return ((features - state_mean) @ components.T) / scale


def _design_matrix(
    data: BranchData,
    *,
    state_mean: np.ndarray,
    components: np.ndarray,
    scale: np.ndarray,
    anchor_steps: np.ndarray,
    action_dim_count: int,
) -> np.ndarray:
    dimensions = np.asarray(data.action_dimensions, np.int32)
    if np.any(dimensions < 0) or np.any(dimensions >= action_dim_count):
        raise ValueError("evaluation contains an unseen action dimension")
    state = _state_projection(
        np.asarray(data.features, np.float64),
        state_mean=state_mean,
        components=components,
        scale=scale,
    )
    dimension_one_hot = np.eye(action_dim_count, dtype=np.float64)[dimensions]
    anchor_one_hot = np.eye(len(anchor_steps), dtype=np.float64)[
        _anchor_indices(data.metadata, anchor_steps)
    ]
    state_dimension = (
        state[:, :, None] * dimension_one_hot[:, None, :]
    ).reshape((state.shape[0], -1))
    return np.concatenate(
        [
            np.ones((state.shape[0], 1), np.float64),
            dimension_one_hot,
            anchor_one_hot,
            state_dimension,
        ],
        axis=-1,
    )


def fit_structured_delta_model(
    data: BranchData,
    *,
    pca_components: int,
    ridge_alpha: float,
    return_atol: float,
) -> StructuredDeltaModel:
    if pca_components < 1:
        raise ValueError("pca_components must be positive")
    if ridge_alpha < 0.0:
        raise ValueError("ridge_alpha must be non-negative")
    state_mean = np.mean(data.features, axis=0, keepdims=True).astype(
        np.float64
    )
    centered = np.asarray(data.features, np.float64) - state_mean
    _, singular_values, right_vectors = np.linalg.svd(
        centered,
        full_matrices=False,
    )
    if not singular_values.size or singular_values[0] <= 0.0:
        raise ValueError("fit features have zero numerical rank")
    numerical_rank = int(
        np.sum(singular_values > singular_values[0] * 1e-5)
    )
    component_count = min(
        int(pca_components),
        numerical_rank,
        right_vectors.shape[0],
        right_vectors.shape[1],
    )
    components = right_vectors[:component_count]
    projected = centered @ components.T
    scale = np.maximum(np.std(projected, axis=0), 1e-5)
    anchor_steps = np.asarray(
        sorted({int(record["anchor_step"]) for record in data.metadata}),
        np.int32,
    )
    action_dim_count = int(np.max(data.action_dimensions)) + 1
    design = _design_matrix(
        data,
        state_mean=state_mean,
        components=components,
        scale=scale,
        anchor_steps=anchor_steps,
        action_dim_count=action_dim_count,
    )
    targets = _optimal_deltas(data, return_atol=return_atol)
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge_alpha
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ targets,
    )
    return StructuredDeltaModel(
        state_mean=state_mean,
        state_components=components,
        state_scale=scale,
        ridge_weights=weights,
        anchor_steps=anchor_steps,
        action_dim_count=action_dim_count,
        ridge_alpha=float(ridge_alpha),
    )


def predict_optimal_delta(
    model: StructuredDeltaModel,
    data: BranchData,
) -> np.ndarray:
    design = _design_matrix(
        data,
        state_mean=model.state_mean,
        components=model.state_components,
        scale=model.state_scale,
        anchor_steps=model.anchor_steps,
        action_dim_count=model.action_dim_count,
    )
    return design @ model.ridge_weights


def _candidate_scores(
    predicted_delta: np.ndarray,
    metadata: tuple[dict[str, Any], ...],
) -> np.ndarray:
    scores = []
    for prediction, record in zip(predicted_delta, metadata, strict=True):
        deltas = np.asarray(record["actual_first_delta"], np.float64)
        scores.append(-np.abs(deltas - prediction))
    return np.asarray(scores, np.float64)


def _proxy_scores(
    metadata: tuple[dict[str, Any], ...],
) -> tuple[np.ndarray, np.ndarray]:
    behavior = []
    policy = []
    for record in metadata:
        behavior.append(
            -np.abs(np.asarray(record["actual_first_delta"], np.float64))
        )
        policy.append(
            np.asarray(record["policy_log_probability"], np.float64)
        )
    return np.asarray(behavior), np.asarray(policy)


def _one_seed_stats(
    scores: np.ndarray,
    targets: np.ndarray,
    *,
    return_atol: float,
) -> SeedStats:
    pair_correct = 0.0
    pair_total = 0.0
    spearman_sum = 0.0
    spearman_count = 0.0
    top1_sum = 0.0
    state_count = 0.0
    regret_sum = 0.0
    for predicted, target in zip(scores, targets, strict=True):
        if np.ptp(target) <= return_atol:
            continue
        for left in range(target.size):
            for right in range(left + 1, target.size):
                target_delta = target[left] - target[right]
                if abs(target_delta) <= return_atol:
                    continue
                pair_correct += float(
                    (predicted[left] - predicted[right]) * target_delta > 0.0
                )
                pair_total += 1.0
        predicted_rank = _rankdata(predicted)
        target_rank = _rankdata(target)
        if np.std(predicted_rank) > 0.0 and np.std(target_rank) > 0.0:
            spearman_sum += float(
                np.corrcoef(predicted_rank, target_rank)[0, 1]
            )
            spearman_count += 1.0
        chosen = int(np.argmax(predicted))
        maximum = float(np.max(target))
        top1_sum += float(target[chosen] >= maximum - return_atol)
        regret_sum += maximum - float(target[chosen])
        state_count += 1.0
    return SeedStats(
        pair_correct=pair_correct,
        pair_total=pair_total,
        spearman_sum=spearman_sum,
        spearman_count=spearman_count,
        top1_sum=top1_sum,
        state_count=state_count,
        regret_sum=regret_sum,
    )


def _aggregate_stats(items: Iterable[SeedStats]) -> dict[str, float]:
    values = tuple(items)
    pair_total = sum(item.pair_total for item in values)
    spearman_count = sum(item.spearman_count for item in values)
    state_count = sum(item.state_count for item in values)
    return {
        "pairwise_accuracy": (
            sum(item.pair_correct for item in values) / pair_total
            if pair_total
            else math.nan
        ),
        "mean_spearman": (
            sum(item.spearman_sum for item in values) / spearman_count
            if spearman_count
            else math.nan
        ),
        "top1_accuracy": (
            sum(item.top1_sum for item in values) / state_count
            if state_count
            else math.nan
        ),
        "regret": (
            sum(item.regret_sum for item in values) / state_count
            if state_count
            else math.nan
        ),
    }


def _percentile_interval(values: np.ndarray) -> list[float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return [math.nan, math.nan]
    return [
        float(np.percentile(finite, 2.5)),
        float(np.percentile(finite, 97.5)),
    ]


def _seed_bootstrap(
    data: BranchData,
    score_sets: dict[str, np.ndarray],
    *,
    return_atol: float,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    seed_ids = np.asarray(
        sorted({int(record["eval_seed"]) for record in data.metadata}),
        np.int64,
    )
    record_seeds = np.asarray(
        [int(record["eval_seed"]) for record in data.metadata],
        np.int64,
    )
    per_method: dict[str, dict[int, SeedStats]] = {}
    for name, scores in score_sets.items():
        per_method[name] = {}
        for seed_id in seed_ids:
            mask = record_seeds == seed_id
            per_method[name][int(seed_id)] = _one_seed_stats(
                scores[mask],
                data.returns[mask],
                return_atol=return_atol,
            )
    payload: dict[str, Any] = {
        "unit": "simulator_seed",
        "num_seeds": int(seed_ids.size),
        "num_replicates": int(replicates),
        "metrics": {},
        "paired_deltas": {},
    }
    if replicates <= 0 or not seed_ids.size:
        return payload
    rng = np.random.default_rng(seed)
    samples = {
        name: {
            metric: np.full(replicates, np.nan, np.float64)
            for metric in (
                "pairwise_accuracy",
                "mean_spearman",
                "top1_accuracy",
                "regret",
            )
        }
        for name in score_sets
    }
    for replicate in range(replicates):
        selected = rng.integers(0, seed_ids.size, size=seed_ids.size)
        for name in score_sets:
            metrics = _aggregate_stats(
                per_method[name][int(seed_ids[index])]
                for index in selected
            )
            for metric, value in metrics.items():
                samples[name][metric][replicate] = value
    for name, metrics in samples.items():
        payload["metrics"][name] = {
            metric: _percentile_interval(values)
            for metric, values in metrics.items()
        }
    for proxy in ("behavior", "policy"):
        payload["paired_deltas"][f"model_minus_{proxy}"] = {
            "pairwise_accuracy": _percentile_interval(
                samples["model"]["pairwise_accuracy"]
                - samples[proxy]["pairwise_accuracy"]
            ),
            "top1_accuracy": _percentile_interval(
                samples["model"]["top1_accuracy"]
                - samples[proxy]["top1_accuracy"]
            ),
            "regret_improvement": _percentile_interval(
                samples[proxy]["regret"] - samples["model"]["regret"]
            ),
        }
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
        temporary = Path(file.name)
    os.replace(temporary, path)


def _write_model(
    path: Path,
    model: StructuredDeltaModel,
    *,
    supported_anchor_steps: np.ndarray | None = None,
    supported_action_dimensions: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_mean": model.state_mean.astype(np.float32),
        "state_components": model.state_components.astype(np.float32),
        "state_scale": model.state_scale.astype(np.float32),
        "ridge_weights": model.ridge_weights.astype(np.float64),
        "anchor_steps": model.anchor_steps,
        "action_dim_count": np.asarray(model.action_dim_count, np.int32),
        "ridge_alpha": np.asarray(model.ridge_alpha, np.float64),
    }
    if supported_anchor_steps is not None:
        payload["supported_anchor_steps"] = np.asarray(
            supported_anchor_steps,
            np.int32,
        )
    if supported_action_dimensions is not None:
        payload["supported_action_dimensions"] = np.asarray(
            supported_action_dimensions,
            np.int32,
        )
    np.savez_compressed(path, **payload)


def main() -> None:
    args = parse_args()
    if args.return_atol < 0.0:
        raise ValueError("return_atol must be non-negative")
    if args.bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates must be non-negative")
    fit_splits = _split_names(args.fit_splits)
    evaluation_splits = _split_names(args.evaluation_splits)
    started = time.monotonic()
    fit_data = _load_branch_data(args.fit_cache, fit_splits)
    evaluation_data = _load_branch_data(
        args.evaluation_cache,
        evaluation_splits,
    )
    model = fit_structured_delta_model(
        fit_data,
        pca_components=args.pca_components,
        ridge_alpha=args.ridge_alpha,
        return_atol=args.return_atol,
    )
    predicted_delta = predict_optimal_delta(model, evaluation_data)
    model_scores = _candidate_scores(
        predicted_delta,
        evaluation_data.metadata,
    )
    behavior_scores, policy_scores = _proxy_scores(evaluation_data.metadata)
    metrics = {
        name: _ranking_metrics(
            scores,
            evaluation_data.returns,
            return_atol=args.return_atol,
        )
        for name, scores in (
            ("model", model_scores),
            ("behavior", behavior_scores),
            ("policy", policy_scores),
        )
    }
    informative_returns = [
        returns
        for returns in evaluation_data.returns
        if np.ptp(returns) > args.return_atol
    ]
    if not informative_returns:
        raise ValueError("evaluation data has no informative branch states")
    random_top1_probability = float(
        np.mean(
            [
                np.count_nonzero(
                    np.isclose(
                        returns,
                        np.max(returns),
                        atol=args.return_atol,
                        rtol=0.0,
                    )
                )
                / returns.size
                for returns in informative_returns
            ]
        )
    )
    bootstrap = _seed_bootstrap(
        evaluation_data,
        {
            "model": model_scores,
            "behavior": behavior_scores,
            "policy": policy_scores,
        },
        return_atol=args.return_atol,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    model_metrics = metrics["model"]
    checks = {
        "pairwise_above_absolute_threshold": (
            model_metrics["pairwise_accuracy"] > args.min_pairwise
        ),
        "pairwise_above_behavior": (
            model_metrics["pairwise_accuracy"]
            > metrics["behavior"]["pairwise_accuracy"]
        ),
        "pairwise_above_policy": (
            model_metrics["pairwise_accuracy"]
            > metrics["policy"]["pairwise_accuracy"]
        ),
        "spearman_above_absolute_threshold": (
            model_metrics["mean_spearman"] > args.min_spearman
        ),
        "top1_above_random": (
            model_metrics["top1_accuracy"]
            > random_top1_probability
        ),
        "regret_below_behavior": (
            model_metrics["regret"] < metrics["behavior"]["regret"]
        ),
        "regret_below_policy": (
            model_metrics["regret"] < metrics["policy"]["regret"]
        ),
    }
    if args.top1_reference == "proxies":
        checks.update(
            {
                "top1_above_behavior": (
                    model_metrics["top1_accuracy"]
                    > metrics["behavior"]["top1_accuracy"]
                ),
                "top1_above_policy": (
                    model_metrics["top1_accuracy"]
                    > metrics["policy"]["top1_accuracy"]
                ),
            }
        )
    if args.require_bootstrap:
        model_ci = bootstrap["metrics"]["model"]
        checks.update(
            {
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
        )
    gate = "pass" if all(checks.values()) else "fail"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "structured_delta_model.npz"
    _write_model(model_path, model)
    summary = {
        "status": "ok",
        "gate": gate,
        "checks": checks,
        "fit_cache": str(args.fit_cache.resolve()),
        "fit_splits": list(fit_splits),
        "evaluation_cache": str(args.evaluation_cache.resolve()),
        "evaluation_splits": list(evaluation_splits),
        "fit_num_states": int(fit_data.features.shape[0]),
        "evaluation_num_states": int(evaluation_data.features.shape[0]),
        "evaluation_seed_ids": sorted(
            {int(record["eval_seed"]) for record in evaluation_data.metadata}
        ),
        "model": {
            "path": str(model_path.resolve()),
            "semantics": "-abs(candidate_delta - predicted_optimal_delta)",
            "pca_components": int(model.state_components.shape[0]),
            "ridge_alpha": float(model.ridge_alpha),
            "parameter_count": int(model.ridge_weights.size),
            "action_dim_count": int(model.action_dim_count),
            "anchor_steps": model.anchor_steps.tolist(),
        },
        "thresholds": {
            "min_pairwise": float(args.min_pairwise),
            "min_spearman": float(args.min_spearman),
            "top1_reference": str(args.top1_reference),
            "require_bootstrap": bool(args.require_bootstrap),
        },
        "metrics": metrics,
        "random_top1_probability": random_top1_probability,
        "seed_bootstrap": bootstrap,
        "predicted_optimal_delta": {
            "min": float(np.min(predicted_delta)),
            "max": float(np.max(predicted_delta)),
            "mean": float(np.mean(predicted_delta)),
            "std": float(np.std(predicted_delta)),
        },
        "elapsed_seconds": float(time.monotonic() - started),
    }
    _atomic_json(args.output_dir / "gate_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
