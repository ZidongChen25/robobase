#!/usr/bin/env python3
"""Matched direct-scalar, C51, and Flow-Matching value benchmark.

This is an intentionally policy-free experiment.  Every method sees the same
frozen CQN image/state feature, candidate action plan, action dimension, and
C2F sibling-bin identity.  A validation simulator seed selects the checkpoint;
the disjoint held-out simulator seeds are never used for selection.

The cache contains one realized continuation return for each (state, action)
condition.  Consequently this experiment measures conditional value
regression, action ranking, and source sensitivity.  It does not by itself
establish that Flow Matching learned an aleatoric return distribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--methods",
        default="direct,c51,flow",
        help=(
            "Comma-separated subset of direct,direct_rank,direct_top1,"
            "direct_softmax,"
            "direct_shuffle,c51,flow,flow_endpoint,flow_distill."
        ),
    )
    parser.add_argument(
        "--seeds",
        default="1,2,3",
        help="Comma-separated model initialization seeds.",
    )
    parser.add_argument(
        "--validation-seed",
        type=int,
        help="One train-cache simulator seed reserved for model selection.",
    )
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--hidden-dims", default="256,256")
    parser.add_argument("--updates", type=int, default=5_000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--atoms", type=int, default=51)
    parser.add_argument("--v-min", type=float, default=0.0)
    parser.add_argument("--v-max", type=float, default=1.0)
    parser.add_argument("--flow-steps", type=int, default=5)
    parser.add_argument("--flow-samples", type=int, default=16)
    parser.add_argument(
        "--flow-train-samples",
        type=int,
        default=1,
        help=(
            "Source samples per condition during flow training. Official "
            "FLOQ uses 8; the default preserves earlier benchmark runs."
        ),
    )
    parser.add_argument(
        "--flow-source-type",
        choices=("normal", "uniform"),
        default="normal",
    )
    parser.add_argument("--flow-source-min", type=float, default=0.0)
    parser.add_argument("--flow-source-max", type=float, default=0.1)
    parser.add_argument(
        "--flow-endpoint-lambda",
        type=float,
        default=1.0,
        help=(
            "Endpoint-MSE weight for flow_endpoint. Pure flow ignores this "
            "argument."
        ),
    )
    parser.add_argument(
        "--flow-distill-lambda",
        type=float,
        default=1.0,
        help=(
            "Weight for the official-style scalar readout MSE to the stopped "
            "mean integrated flow endpoint."
        ),
    )
    parser.add_argument(
        "--flow-distill-blend",
        type=float,
        default=0.0,
        help=(
            "At evaluation, blend the distilled scalar readout with the "
            "integrated endpoint mean: 0=readout, 1=endpoint. Training is "
            "unchanged, so this must be selected on validation only."
        ),
    )
    parser.add_argument(
        "--warmup-target",
        choices=("none", "policy_prior"),
        default="none",
        help=(
            "Optional non-stationary-target probe. policy_prior first fits "
            "the frozen BC policy's sibling-bin ranking, then switches to "
            "the simulator return after --warmup-updates."
        ),
    )
    parser.add_argument(
        "--warmup-updates",
        type=int,
        default=0,
        help="Number of initial updates spent fitting --warmup-target.",
    )
    parser.add_argument(
        "--selection-min-adaptation-updates",
        type=int,
        default=0,
        help=(
            "When a warmup target is used, validation checkpoint selection "
            "starts only after this many true-return updates."
        ),
    )
    parser.add_argument(
        "--target-noise-std",
        type=float,
        default=0.0,
        help=(
            "I.i.d. zero-mean noise added to each sampled training target. "
            "Validation and heldout targets remain clean."
        ),
    )
    parser.add_argument("--return-atol", type=float, default=1e-12)
    parser.add_argument(
        "--softmax-temperature",
        type=float,
        default=0.05,
        help=(
            "Per-state return softmax temperature for direct_softmax. "
            "Lower values approach the hard direct_top1 target."
        ),
    )
    return parser.parse_args()


def _integer_list(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("expected at least one integer")
    return result


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks with exact tie handling, without a scipy dependency."""

    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _ranking_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    return_atol: float,
) -> dict[str, float]:
    """Metrics over grouped candidate values shaped [states, bins]."""

    predictions = np.asarray(predictions, np.float64)
    targets = np.asarray(targets, np.float64)
    if predictions.shape != targets.shape or predictions.ndim != 2:
        raise ValueError(
            "predictions and targets must have the same [states, bins] shape"
        )
    informative = np.ptp(targets, axis=1) > float(return_atol)
    pair_correct = 0
    pair_total = 0
    correlations: list[float] = []
    top1: list[float] = []
    regrets: list[float] = []
    for predicted, target, keep in zip(
        predictions,
        targets,
        informative,
        strict=True,
    ):
        if not keep:
            continue
        for left in range(target.size):
            for right in range(left + 1, target.size):
                target_delta = target[left] - target[right]
                if abs(target_delta) <= return_atol:
                    continue
                predicted_delta = predicted[left] - predicted[right]
                pair_correct += int(predicted_delta * target_delta > 0.0)
                pair_total += 1
        target_rank = _rankdata(target)
        predicted_rank = _rankdata(predicted)
        if np.std(target_rank) > 0.0 and np.std(predicted_rank) > 0.0:
            correlations.append(
                float(np.corrcoef(target_rank, predicted_rank)[0, 1])
            )
        chosen = int(np.argmax(predicted))
        target_max = float(np.max(target))
        top1.append(float(target[chosen] >= target_max - return_atol))
        regrets.append(target_max - float(target[chosen]))

    return {
        "num_states": float(targets.shape[0]),
        "num_informative_states": float(np.sum(informative)),
        "mae": float(np.mean(np.abs(predictions - targets))),
        "mse": float(np.mean(np.square(predictions - targets))),
        "pairwise_accuracy": (
            float(pair_correct / pair_total) if pair_total else math.nan
        ),
        "mean_spearman": (
            float(np.mean(correlations)) if correlations else math.nan
        ),
        "top1_accuracy": float(np.mean(top1)) if top1 else math.nan,
        "regret": float(np.mean(regrets)) if regrets else math.nan,
        "predicted_span": float(
            np.mean(np.ptp(predictions[informative], axis=1))
        )
        if np.any(informative)
        else math.nan,
        "target_span": float(np.mean(np.ptp(targets[informative], axis=1)))
        if np.any(informative)
        else math.nan,
    }


def _categorical_targets(
    values: np.ndarray,
    *,
    atoms: int,
    v_min: float,
    v_max: float,
) -> np.ndarray:
    """Linearly project scalar values onto an equally spaced C51 support."""

    if atoms < 2 or not v_max > v_min:
        raise ValueError("C51 requires atoms >= 2 and v_max > v_min")
    values = np.clip(np.asarray(values, np.float32), v_min, v_max)
    positions = (values - v_min) / (v_max - v_min) * (atoms - 1)
    lower = np.floor(positions).astype(np.int32)
    upper = np.ceil(positions).astype(np.int32)
    probabilities = np.zeros((values.size, atoms), dtype=np.float32)
    rows = np.arange(values.size)
    upper_weight = positions - lower
    lower_weight = 1.0 - upper_weight
    probabilities[rows, lower] += lower_weight
    probabilities[rows, upper] += upper_weight
    return probabilities


@dataclass(frozen=True)
class ConditionerData:
    fit_x: np.ndarray
    fit_y: np.ndarray
    fit_policy_prior_y: np.ndarray | None
    fit_targets: np.ndarray
    validation_x: np.ndarray
    validation_targets: np.ndarray
    heldout_x: np.ndarray
    heldout_targets: np.ndarray
    validation_seed: int
    train_seeds: tuple[int, ...]
    heldout_seeds: tuple[int, ...]
    heldout_record_seeds: np.ndarray
    candidates: int
    condition_dim: int
    state_components: int
    state_mean: np.ndarray
    state_basis: np.ndarray
    state_scale: np.ndarray
    action_mean: np.ndarray
    action_scale: np.ndarray
    action_dim_count: int


def _policy_prior_rank_targets(
    records: list[dict[str, Any]],
    *,
    candidates: int,
) -> np.ndarray:
    """Map frozen-BC sibling preferences to rank targets in ``[0, 1]``."""

    if candidates < 2:
        raise ValueError("policy-prior ranks require at least two candidates")
    ranked = []
    for record_index, record in enumerate(records):
        if "policy_log_probability" not in record:
            raise ValueError(
                "policy_prior warmup requires policy_log_probability in every "
                f"cache record; missing record {record_index}"
            )
        scores = np.asarray(record["policy_log_probability"], np.float64)
        if scores.shape != (candidates,):
            raise ValueError(
                "policy_log_probability must have one value per candidate; "
                f"record {record_index} has shape {scores.shape}"
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError(
                f"non-finite policy_log_probability in record {record_index}"
            )
        ranked.append(_rankdata(scores) / float(candidates - 1))
    return np.asarray(ranked, np.float32)


def _within_state_return_rank_targets(
    returns: np.ndarray,
) -> np.ndarray:
    """Map counterfactual returns to tied-aware per-state ranks in ``[0, 1]``."""

    returns = np.asarray(returns, np.float32)
    if returns.ndim != 2 or returns.shape[1] < 2:
        raise ValueError("return ranks require [states, candidates>=2]")
    denominator = float(returns.shape[1] - 1)
    return np.asarray(
        [_rankdata(row) / denominator for row in returns],
        np.float32,
    )


def _within_state_best_targets(
    returns: np.ndarray,
    *,
    return_atol: float,
) -> np.ndarray:
    """Return tied-aware indicators for every counterfactual best action."""

    returns = np.asarray(returns, np.float32)
    if returns.ndim != 2 or returns.shape[1] < 2:
        raise ValueError("best targets require [states, candidates>=2]")
    if not math.isfinite(return_atol) or return_atol < 0.0:
        raise ValueError("return_atol must be finite and non-negative")
    return (
        returns >= np.max(returns, axis=1, keepdims=True) - return_atol
    ).astype(np.float32)


def _within_state_softmax_targets(
    returns: np.ndarray,
    *,
    temperature: float,
) -> np.ndarray:
    """Convert counterfactual returns to gap-sensitive action probabilities."""

    returns = np.asarray(returns, np.float32)
    if returns.ndim != 2 or returns.shape[1] < 2:
        raise ValueError("softmax targets require [states, candidates>=2]")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("softmax temperature must be finite and positive")
    logits = (
        returns - np.max(returns, axis=1, keepdims=True)
    ) / float(temperature)
    weights = np.exp(logits, dtype=np.float32)
    return weights / np.sum(weights, axis=1, keepdims=True)


def _training_phase(
    step: int,
    *,
    warmup_target: str,
    warmup_updates: int,
) -> str:
    """Return the target phase used for a zero-indexed gradient update."""

    if warmup_target == "none":
        return "return"
    if warmup_target != "policy_prior":
        raise ValueError(f"unsupported warmup target {warmup_target!r}")
    return "policy_prior" if step < warmup_updates else "return"


def _records_from_cache(data: Any, split: str) -> list[dict[str, Any]]:
    key = f"{split}_metadata"
    if key not in data:
        raise KeyError(f"cache is missing {key}")
    return json.loads(str(np.asarray(data[key]).item()))


def _build_conditioners(
    cache_path: Path,
    *,
    pca_components: int,
    validation_seed: int | None,
) -> ConditionerData:
    """Build identical state/action/dimension/bin conditions for all methods."""

    with np.load(cache_path, allow_pickle=False) as data:
        train_features = np.asarray(data["train_features"], np.float32)
        train_actions = np.asarray(data["train_actions"], np.float32)
        train_returns = np.asarray(data["train_returns"], np.float32)
        train_dimensions = np.asarray(
            data["train_action_dimensions"],
            np.int32,
        )
        heldout_features = np.asarray(data["heldout_features"], np.float32)
        heldout_actions = np.asarray(data["heldout_actions"], np.float32)
        heldout_returns = np.asarray(data["heldout_returns"], np.float32)
        heldout_dimensions = np.asarray(
            data["heldout_action_dimensions"],
            np.int32,
        )
        train_records = _records_from_cache(data, "train")
        heldout_records = _records_from_cache(data, "heldout")

    train_record_seeds = np.asarray(
        [int(record["eval_seed"]) for record in train_records],
        np.int32,
    )
    heldout_record_seeds = np.asarray(
        [int(record["eval_seed"]) for record in heldout_records],
        np.int32,
    )
    train_seeds = tuple(int(value) for value in np.unique(train_record_seeds))
    heldout_seeds = tuple(
        int(value) for value in np.unique(heldout_record_seeds)
    )
    if len(train_seeds) < 2:
        raise ValueError("cache needs at least two train simulator seeds")
    selected_validation_seed = (
        max(train_seeds) if validation_seed is None else int(validation_seed)
    )
    if selected_validation_seed not in train_seeds:
        raise ValueError(
            f"validation seed {selected_validation_seed} is not in "
            f"train seeds {train_seeds}"
        )
    fit_mask = train_record_seeds != selected_validation_seed
    validation_mask = ~fit_mask
    if not np.any(fit_mask) or not np.any(validation_mask):
        raise ValueError("fit and validation state splits must both be non-empty")

    # Fit all transforms without touching the validation or held-out seeds.
    state_mean = train_features[fit_mask].mean(axis=0, keepdims=True)
    centered_fit = train_features[fit_mask] - state_mean
    _, singular_values, right_vectors = np.linalg.svd(
        centered_fit,
        full_matrices=False,
    )
    # The cache repeats the same state feature once per action dimension.
    # Limiting PCA by row count alone therefore retains numerically null
    # directions and can amplify an unseen seed by 1e5 after standardization.
    numerical_rank = int(
        np.sum(singular_values > singular_values[0] * 1e-5)
    )
    component_count = min(
        int(pca_components),
        numerical_rank,
        right_vectors.shape[0],
        right_vectors.shape[1],
    )
    if component_count < 1:
        raise ValueError("pca_components must be positive")
    components = right_vectors[:component_count]

    def project(features: np.ndarray) -> np.ndarray:
        return (features - state_mean) @ components.T

    fit_state = project(train_features[fit_mask])
    validation_state = project(train_features[validation_mask])
    heldout_state = project(heldout_features)
    state_scale = np.maximum(fit_state.std(axis=0, keepdims=True), 1e-5)
    fit_state = fit_state / state_scale
    validation_state = validation_state / state_scale
    heldout_state = heldout_state / state_scale

    candidates = int(train_actions.shape[1])
    action_dim_count = int(train_actions.shape[-1])
    if train_returns.shape[1] != candidates:
        raise ValueError("action and return candidate axes disagree")
    fit_action_flat = train_actions[fit_mask].reshape(
        (-1, int(np.prod(train_actions.shape[2:]))),
    )
    action_mean = fit_action_flat.mean(axis=0, keepdims=True)
    action_scale = np.maximum(
        fit_action_flat.std(axis=0, keepdims=True),
        1e-5,
    )

    def make_conditions(
        state: np.ndarray,
        actions: np.ndarray,
        dimensions: np.ndarray,
    ) -> np.ndarray:
        state_repeated = np.repeat(state, candidates, axis=0)
        action_flat = actions.reshape(
            (-1, int(np.prod(actions.shape[2:]))),
        )
        action_flat = (action_flat - action_mean) / action_scale
        dimension_one_hot = np.eye(
            action_dim_count,
            dtype=np.float32,
        )[np.repeat(dimensions, candidates)]
        bin_one_hot = np.tile(
            np.eye(candidates, dtype=np.float32),
            (state.shape[0], 1),
        )
        return np.concatenate(
            [
                state_repeated,
                action_flat,
                dimension_one_hot,
                bin_one_hot,
            ],
            axis=-1,
        ).astype(np.float32)

    fit_x = make_conditions(
        fit_state,
        train_actions[fit_mask],
        train_dimensions[fit_mask],
    )
    validation_x = make_conditions(
        validation_state,
        train_actions[validation_mask],
        train_dimensions[validation_mask],
    )
    heldout_x = make_conditions(
        heldout_state,
        heldout_actions,
        heldout_dimensions,
    )
    fit_targets = train_returns[fit_mask]
    validation_targets = train_returns[validation_mask]
    fit_policy_prior_y = None
    if all("policy_log_probability" in record for record in train_records):
        policy_prior_targets = _policy_prior_rank_targets(
            train_records,
            candidates=candidates,
        )
        fit_policy_prior_y = policy_prior_targets[fit_mask].reshape(-1)
    if not np.any(np.ptp(validation_targets, axis=1) > 0.0):
        raise ValueError(
            f"validation seed {selected_validation_seed} contains no "
            "informative action-return contrast"
        )
    return ConditionerData(
        fit_x=fit_x,
        fit_y=fit_targets.reshape(-1).astype(np.float32),
        fit_policy_prior_y=fit_policy_prior_y,
        fit_targets=fit_targets,
        validation_x=validation_x,
        validation_targets=validation_targets,
        heldout_x=heldout_x,
        heldout_targets=heldout_returns,
        validation_seed=selected_validation_seed,
        train_seeds=train_seeds,
        heldout_seeds=heldout_seeds,
        heldout_record_seeds=heldout_record_seeds,
        candidates=candidates,
        condition_dim=int(fit_x.shape[-1]),
        state_components=component_count,
        state_mean=state_mean.astype(np.float32),
        state_basis=components.astype(np.float32),
        state_scale=state_scale.astype(np.float32),
        action_mean=action_mean.astype(np.float32),
        action_scale=action_scale.astype(np.float32),
        action_dim_count=action_dim_count,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, path)


def _make_antithetic_sources(
    *,
    seed: int,
    samples: int,
    count: int,
) -> np.ndarray:
    if samples < 2 or samples % 2:
        raise ValueError("flow_samples must be a positive even integer")
    rng = np.random.default_rng(seed)
    half = rng.normal(size=(samples // 2, count, 1)).astype(np.float32)
    return np.concatenate([half, -half], axis=0)


def _make_flow_sources(
    *,
    seed: int,
    samples: int,
    count: int,
    source_type: str,
    source_min: float,
    source_max: float,
) -> np.ndarray:
    """Create a deterministic, variance-reduced source bank for evaluation."""

    if samples < 2 or samples % 2:
        raise ValueError("flow_samples must be a positive even integer")
    if source_type == "normal":
        return _make_antithetic_sources(
            seed=seed,
            samples=samples,
            count=count,
        )
    if source_type != "uniform":
        raise ValueError(f"unsupported flow source type {source_type!r}")
    if not source_max > source_min:
        raise ValueError("flow_source_max must be greater than flow_source_min")
    rng = np.random.default_rng(seed)
    half = rng.uniform(
        0.0,
        1.0,
        size=(samples // 2, count, 1),
    ).astype(np.float32)
    unit_sources = np.concatenate([half, 1.0 - half], axis=0)
    return source_min + (source_max - source_min) * unit_sources


def _selection_key(metrics: dict[str, float]) -> tuple[float, float, float]:
    pairwise = metrics["pairwise_accuracy"]
    if not math.isfinite(pairwise):
        pairwise = -math.inf
    return pairwise, -metrics["regret"], -metrics["mae"]


def _train_one(
    *,
    method: str,
    seed: int,
    data: ConditionerData,
    hidden_dims: tuple[int, ...],
    args: argparse.Namespace,
    row_callback: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import optax
    from flax import linen as nn
    from flax import serialization

    class MLP(nn.Module):
        widths: tuple[int, ...]
        output_dim: int

        @nn.compact
        def __call__(self, inputs):
            hidden = inputs
            for width in self.widths:
                hidden = nn.Dense(
                    width,
                    kernel_init=nn.initializers.orthogonal(math.sqrt(2.0)),
                )(hidden)
                hidden = nn.silu(hidden)
            return nn.Dense(
                self.output_dim,
                kernel_init=nn.initializers.orthogonal(0.01),
            )(hidden)

    flow_method = method in {"flow", "flow_endpoint", "flow_distill"}
    if method not in {
        "direct",
        "direct_rank",
        "direct_top1",
        "direct_softmax",
        "direct_shuffle",
        "c51",
        "flow",
        "flow_endpoint",
        "flow_distill",
    }:
        raise ValueError(f"unsupported method {method!r}")
    if method == "flow_endpoint" and args.flow_endpoint_lambda <= 0.0:
        raise ValueError("flow_endpoint requires --flow-endpoint-lambda > 0")
    if method == "flow_distill" and args.flow_distill_lambda <= 0.0:
        raise ValueError("flow_distill requires --flow-distill-lambda > 0")
    if not 0.0 <= args.flow_distill_blend <= 1.0:
        raise ValueError("flow_distill_blend must be in [0, 1]")
    if args.flow_train_samples < 1:
        raise ValueError("flow_train_samples must be positive")
    if (
        args.flow_source_type == "uniform"
        and not args.flow_source_max > args.flow_source_min
    ):
        raise ValueError(
            "flow_source_max must be greater than flow_source_min"
        )
    output_dim = args.atoms if method == "c51" else 1
    model = MLP(hidden_dims, output_dim)
    model_input_dim = data.condition_dim + (5 if flow_method else 0)
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    velocity_or_value_params = model.init(
        init_key,
        jnp.zeros((1, model_input_dim), dtype=jnp.float32),
    )
    readout_model = None
    if method == "flow_distill":
        readout_model = MLP(hidden_dims, 1)
        key, readout_key = jax.random.split(key)
        params = {
            "velocity": velocity_or_value_params,
            "readout": readout_model.init(
                readout_key,
                jnp.zeros((1, data.condition_dim), dtype=jnp.float32),
            ),
        }
    else:
        params = velocity_or_value_params
    optimizer = optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.adamw(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        ),
    )
    opt_state = optimizer.init(params)
    support = jnp.linspace(
        args.v_min,
        args.v_max,
        args.atoms,
        dtype=jnp.float32,
    )

    if method in {"direct", "direct_rank", "direct_shuffle"}:

        @jax.jit
        def update(params, opt_state, condition, target, random_key):
            del random_key

            def loss_fn(current_params):
                prediction = model.apply(current_params, condition)[:, 0]
                return jnp.mean(jnp.square(prediction - target))

            loss, gradients = jax.value_and_grad(loss_fn)(params)
            updates, next_opt_state = optimizer.update(
                gradients,
                opt_state,
                params,
            )
            return (
                optax.apply_updates(params, updates),
                next_opt_state,
                loss,
            )

        @jax.jit
        def predict(params, condition):
            values = model.apply(params, condition)[:, 0]
            return values, jnp.asarray(0.0, jnp.float32)

    elif method in {"direct_top1", "direct_softmax"}:

        @jax.jit
        def update(params, opt_state, condition, target, random_key):
            del random_key

            def loss_fn(current_params):
                logits = model.apply(current_params, condition)[:, 0]
                return jnp.mean(
                    jax.nn.softplus(logits) - target * logits
                )

            loss, gradients = jax.value_and_grad(loss_fn)(params)
            updates, next_opt_state = optimizer.update(
                gradients,
                opt_state,
                params,
            )
            return (
                optax.apply_updates(params, updates),
                next_opt_state,
                loss,
            )

        @jax.jit
        def predict(params, condition):
            logits = model.apply(params, condition)[:, 0]
            return logits, jnp.asarray(0.0, jnp.float32)

    elif method == "c51":

        @jax.jit
        def update(params, opt_state, condition, target, random_key):
            del random_key
            positions = (
                (jnp.clip(target, args.v_min, args.v_max) - args.v_min)
                / (args.v_max - args.v_min)
                * (args.atoms - 1)
            )
            lower = jnp.floor(positions).astype(jnp.int32)
            upper = jnp.ceil(positions).astype(jnp.int32)
            target_probabilities = jax.nn.one_hot(
                lower,
                args.atoms,
            ) * (1.0 - (positions - lower))[:, None]
            target_probabilities += jax.nn.one_hot(
                upper,
                args.atoms,
            ) * (positions - lower)[:, None]

            def loss_fn(current_params):
                logits = model.apply(current_params, condition)
                return -jnp.mean(
                    jnp.sum(
                        target_probabilities
                        * jax.nn.log_softmax(logits, axis=-1),
                        axis=-1,
                    )
                )

            loss, gradients = jax.value_and_grad(loss_fn)(params)
            updates, next_opt_state = optimizer.update(
                gradients,
                opt_state,
                params,
            )
            return (
                optax.apply_updates(params, updates),
                next_opt_state,
                loss,
            )

        @jax.jit
        def predict(params, condition):
            probabilities = jax.nn.softmax(
                model.apply(params, condition),
                axis=-1,
            )
            values = jnp.sum(probabilities * support, axis=-1)
            return values, jnp.asarray(0.0, jnp.float32)

    else:

        def flow_input(condition, value, time_value):
            return jnp.concatenate(
                [
                    condition,
                    value,
                    time_value,
                    jnp.sin(math.pi * time_value),
                    jnp.cos(math.pi * time_value),
                    jnp.sin(2.0 * math.pi * time_value),
                ],
                axis=-1,
            )

        def source_samples(random_key, shape):
            if args.flow_source_type == "normal":
                return jax.random.normal(random_key, shape)
            return jax.random.uniform(
                random_key,
                shape,
                minval=args.flow_source_min,
                maxval=args.flow_source_max,
            )

        def velocity_params(current_params):
            if method == "flow_distill":
                return current_params["velocity"]
            return current_params

        def integrate(current_params, condition, sources):
            source_count, item_count = sources.shape[:2]
            repeated_condition = jnp.broadcast_to(
                condition[None],
                (source_count, item_count, condition.shape[-1]),
            ).reshape((-1, condition.shape[-1]))
            value = sources.reshape((-1, 1))
            step_size = 1.0 / float(args.flow_steps)
            for flow_step in range(args.flow_steps):
                time_value = jnp.full(
                    value.shape,
                    flow_step * step_size,
                    dtype=value.dtype,
                )
                velocity = model.apply(
                    velocity_params(current_params),
                    flow_input(repeated_condition, value, time_value),
                )
                value = value + step_size * velocity
            return value.reshape((source_count, item_count))

        @jax.jit
        def update(params, opt_state, condition, target, random_key):
            time_key, source_key = jax.random.split(random_key)
            source_count = int(args.flow_train_samples)
            batch_count = target.shape[0]
            time_value = jax.random.uniform(
                time_key,
                (source_count, batch_count, 1),
                minval=0.0,
                maxval=1.0,
            )
            source = source_samples(
                source_key,
                (source_count, batch_count, 1),
            )
            endpoint = jnp.broadcast_to(
                target[None, :, None],
                source.shape,
            )
            interpolated = (
                (1.0 - time_value) * source + time_value * endpoint
            )
            target_velocity = endpoint - source
            repeated_condition = jnp.broadcast_to(
                condition[None],
                (
                    source_count,
                    batch_count,
                    condition.shape[-1],
                ),
            ).reshape((-1, condition.shape[-1]))
            flat_time = time_value.reshape((-1, 1))
            flat_interpolated = interpolated.reshape((-1, 1))
            flat_target_velocity = target_velocity.reshape((-1, 1))

            def loss_fn(current_params):
                velocity = model.apply(
                    velocity_params(current_params),
                    flow_input(
                        repeated_condition,
                        flat_interpolated,
                        flat_time,
                    ),
                )
                velocity_loss = jnp.mean(
                    jnp.square(velocity - flat_target_velocity)
                )
                if method == "flow":
                    return velocity_loss

                endpoint_prediction = integrate(
                    current_params,
                    condition,
                    source,
                )
                if method == "flow_distill":
                    assert readout_model is not None
                    distilled_target = jax.lax.stop_gradient(
                        endpoint_prediction.mean(axis=0)
                    )
                    readout = readout_model.apply(
                        current_params["readout"],
                        condition,
                    )[:, 0]
                    distill_loss = jnp.mean(
                        jnp.square(readout - distilled_target)
                    )
                    return (
                        velocity_loss
                        + args.flow_distill_lambda * distill_loss
                    )

                endpoint_loss = jnp.mean(
                    jnp.square(
                        endpoint_prediction - target[None, :]
                    )
                )
                return (
                    velocity_loss
                    + args.flow_endpoint_lambda * endpoint_loss
                )

            loss, gradients = jax.value_and_grad(loss_fn)(params)
            updates, next_opt_state = optimizer.update(
                gradients,
                opt_state,
                params,
            )
            return (
                optax.apply_updates(params, updates),
                next_opt_state,
                loss,
            )

        @jax.jit
        def predict(params, condition, sources):
            samples = integrate(params, condition, sources)
            if method == "flow_distill":
                assert readout_model is not None
                readout_values = readout_model.apply(
                    params["readout"],
                    condition,
                )[:, 0]
                endpoint_values = samples.mean(axis=0)
                values = (
                    (1.0 - args.flow_distill_blend) * readout_values
                    + args.flow_distill_blend * endpoint_values
                )
            else:
                values = samples.mean(axis=0)
            return values, samples.std(axis=0).mean()

    fit_x = jnp.asarray(data.fit_x)
    fit_targets = np.asarray(data.fit_y)
    if method == "direct_rank":
        fit_targets = _within_state_return_rank_targets(
            data.fit_targets
        ).reshape(-1)
    if method == "direct_top1":
        fit_targets = _within_state_best_targets(
            data.fit_targets,
            return_atol=args.return_atol,
        ).reshape(-1)
    if method == "direct_softmax":
        fit_targets = _within_state_softmax_targets(
            data.fit_targets,
            temperature=args.softmax_temperature,
        ).reshape(-1)
    if method == "direct_shuffle":
        shuffle_rng = np.random.default_rng(300_000 + int(seed))
        fit_targets = fit_targets[
            shuffle_rng.permutation(fit_targets.shape[0])
        ]
    fit_y = jnp.asarray(fit_targets)
    fit_policy_prior_y = (
        None
        if data.fit_policy_prior_y is None
        else jnp.asarray(data.fit_policy_prior_y)
    )
    if args.warmup_target == "policy_prior" and fit_policy_prior_y is None:
        raise ValueError(
            "policy_prior warmup requested but cache lacks policy priors"
        )
    validation_x = jnp.asarray(data.validation_x)
    heldout_x = jnp.asarray(data.heldout_x)
    if flow_method:
        validation_sources = jnp.asarray(
            _make_flow_sources(
                seed=100_000 + seed,
                samples=args.flow_samples,
                count=data.validation_x.shape[0],
                source_type=args.flow_source_type,
                source_min=args.flow_source_min,
                source_max=args.flow_source_max,
            )
        )
        heldout_sources = jnp.asarray(
            _make_flow_sources(
                seed=200_000 + seed,
                samples=args.flow_samples,
                count=data.heldout_x.shape[0],
                source_type=args.flow_source_type,
                source_min=args.flow_source_min,
                source_max=args.flow_source_max,
            )
        )

    def evaluate(current_params, condition, targets, sources=None):
        if flow_method:
            values, source_std = predict(current_params, condition, sources)
        else:
            values, source_std = predict(current_params, condition)
        values_np = np.asarray(jax.device_get(values)).reshape(targets.shape)
        metrics = _ranking_metrics(
            values_np,
            targets,
            return_atol=args.return_atol,
        )
        metrics["flow_source_std"] = float(
            np.asarray(jax.device_get(source_std))
        )
        return metrics, values_np

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    best_validation_key = (-math.inf, -math.inf, -math.inf)
    best_step = 0
    best_heldout: dict[str, float] | None = None
    best_heldout_predictions: np.ndarray | None = None
    best_validation: dict[str, float] | None = None
    best_validation_predictions: np.ndarray | None = None
    best_params_bytes: bytes | None = None
    started = time.monotonic()
    selection_start_step = (
        int(args.warmup_updates)
        + int(args.selection_min_adaptation_updates)
        if args.warmup_target != "none"
        else 0
    )

    for step in range(args.updates + 1):
        if step % args.eval_every == 0 or step == args.updates:
            if flow_method:
                validation_metrics, validation_predictions = evaluate(
                    params,
                    validation_x,
                    data.validation_targets,
                    validation_sources,
                )
                heldout_metrics, heldout_predictions = evaluate(
                    params,
                    heldout_x,
                    data.heldout_targets,
                    heldout_sources,
                )
            else:
                validation_metrics, validation_predictions = evaluate(
                    params,
                    validation_x,
                    data.validation_targets,
                )
                heldout_metrics, heldout_predictions = evaluate(
                    params,
                    heldout_x,
                    data.heldout_targets,
                )
            row = {
                "method": method,
                "seed": seed,
                "step": step,
                "target_phase": (
                    "policy_prior"
                    if (
                        args.warmup_target != "none"
                        and step <= args.warmup_updates
                    )
                    else "return"
                ),
                "adaptation_updates": max(
                    0, step - int(args.warmup_updates)
                ),
                "elapsed_seconds": time.monotonic() - started,
                **{
                    f"validation_{name}": value
                    for name, value in validation_metrics.items()
                },
                **{
                    f"heldout_{name}": value
                    for name, value in heldout_metrics.items()
                },
            }
            rows.append(row)
            row_callback(row)
            selection_key = _selection_key(validation_metrics)
            if (
                step >= selection_start_step
                and selection_key > best_validation_key
            ):
                best_validation_key = selection_key
                best_step = step
                best_validation = validation_metrics
                best_validation_predictions = (
                    validation_predictions.copy()
                )
                best_heldout = heldout_metrics
                best_heldout_predictions = heldout_predictions.copy()
                best_params_bytes = serialization.to_bytes(params)
        if step == args.updates:
            break
        indices = rng.integers(
            0,
            data.fit_x.shape[0],
            size=args.batch_size,
        )
        target_phase = _training_phase(
            step,
            warmup_target=str(args.warmup_target),
            warmup_updates=int(args.warmup_updates),
        )
        current_fit_y = (
            fit_policy_prior_y
            if target_phase == "policy_prior"
            else fit_y
        )
        assert current_fit_y is not None
        batch_target = current_fit_y[indices]
        if args.target_noise_std > 0.0:
            batch_target = batch_target + jnp.asarray(
                rng.normal(
                    0.0,
                    float(args.target_noise_std),
                    size=args.batch_size,
                ),
                dtype=batch_target.dtype,
            )
        key, update_key = jax.random.split(key)
        params, opt_state, _ = update(
            params,
            opt_state,
            fit_x[indices],
            batch_target,
            update_key,
        )

    assert (
        best_validation is not None
        and best_validation_predictions is not None
        and best_heldout is not None
        and best_heldout_predictions is not None
        and best_params_bytes is not None
    )
    model_dir = args.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{method}_seed{seed}_selected.msgpack"
    model_path.write_bytes(best_params_bytes)
    return {
        "method": method,
        "seed": seed,
        "parameter_count": int(
            sum(np.asarray(leaf).size for leaf in jax.tree.leaves(params))
        ),
        "best_validation_step": best_step,
        "selected_model": {
            "path": str(model_path.resolve()),
            "format": "flax_msgpack",
            "hidden_dims": list(hidden_dims),
            "input_dim": model_input_dim,
            "output_dim": output_dim,
            "conditioner": str(
                (args.output_dir / "conditioner.npz").resolve()
            ),
        },
        "best_validation_metrics": best_validation,
        "selected_validation_predictions": (
            best_validation_predictions.astype(np.float32).tolist()
        ),
        "selected_heldout_metrics": best_heldout,
        "selected_heldout_predictions": (
            best_heldout_predictions.astype(np.float32).tolist()
        ),
        "final_metrics": {
            name.removeprefix("heldout_"): value
            for name, value in rows[-1].items()
            if name.startswith("heldout_")
        },
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> None:
    args = parse_args()
    methods = tuple(
        item.strip().lower()
        for item in args.methods.split(",")
        if item.strip()
    )
    unknown = set(methods) - {
        "direct",
        "direct_rank",
        "direct_top1",
        "direct_softmax",
        "direct_shuffle",
        "c51",
        "flow",
        "flow_endpoint",
        "flow_distill",
    }
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    seeds = _integer_list(args.seeds)
    hidden_dims = tuple(_integer_list(args.hidden_dims))
    if args.eval_every < 1 or args.updates < 1:
        raise ValueError("updates and eval_every must be positive")
    if args.warmup_updates < 0:
        raise ValueError("warmup_updates must be non-negative")
    if args.selection_min_adaptation_updates < 0:
        raise ValueError(
            "selection_min_adaptation_updates must be non-negative"
        )
    if args.target_noise_std < 0.0:
        raise ValueError("target_noise_std must be non-negative")
    if (
        not math.isfinite(args.softmax_temperature)
        or args.softmax_temperature <= 0.0
    ):
        raise ValueError("softmax_temperature must be finite and positive")
    if args.warmup_target == "none" and args.warmup_updates != 0:
        raise ValueError(
            "warmup_updates must be zero when warmup_target is none"
        )
    if args.warmup_target != "none" and args.warmup_updates < 1:
        raise ValueError(
            "warmup_updates must be positive when a warmup target is used"
        )
    if (
        args.warmup_updates + args.selection_min_adaptation_updates
        > args.updates
    ):
        raise ValueError(
            "warmup plus minimum adaptation updates exceeds total updates"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = _build_conditioners(
        args.dataset_cache,
        pca_components=args.pca_components,
        validation_seed=args.validation_seed,
    )
    np.savez_compressed(
        args.output_dir / "conditioner.npz",
        state_mean=data.state_mean,
        state_basis=data.state_basis,
        state_scale=data.state_scale,
        action_mean=data.action_mean,
        action_scale=data.action_scale,
        action_dim_count=np.asarray(data.action_dim_count, np.int32),
        candidates=np.asarray(data.candidates, np.int32),
        condition_dim=np.asarray(data.condition_dim, np.int32),
        state_components=np.asarray(data.state_components, np.int32),
    )
    manifest = {
        "status": "running",
        "dataset_cache": str(args.dataset_cache.resolve()),
        "methods": methods,
        "seeds": seeds,
        "validation_seed": data.validation_seed,
        "train_seeds": data.train_seeds,
        "heldout_seeds": data.heldout_seeds,
        "condition": {
            "state": (
                f"{data.state_components}-D PCA of frozen image/state feature"
            ),
            "action": "full candidate action plan",
            "action_dimension": "one-hot",
            "action_bin": "one-hot C2F sibling index",
            "condition_dim": data.condition_dim,
        },
        "selection": (
            "maximize validation pairwise accuracy, then minimize validation "
            "regret and MAE after the configured adaptation delay; heldout "
            "seeds never select checkpoints"
        ),
        "limitation": (
            "one return per condition: tests conditional value/ranking, not "
            "aleatoric return-distribution recovery"
        ),
        "arguments": dict(vars(args)),
        "completed": [],
    }
    manifest["arguments"]["dataset_cache"] = str(args.dataset_cache)
    manifest["arguments"]["output_dir"] = str(args.output_dir)
    _atomic_json(args.output_dir / "status.json", manifest)

    curve_path = args.output_dir / "curves.csv"
    curve_file = curve_path.open("w", newline="")
    curve_writer: csv.DictWriter | None = None

    def write_row(row: dict[str, Any]) -> None:
        nonlocal curve_writer
        if curve_writer is None:
            curve_writer = csv.DictWriter(curve_file, fieldnames=list(row))
            curve_writer.writeheader()
        curve_writer.writerow(row)
        curve_file.flush()

    results: list[dict[str, Any]] = []
    try:
        for method in methods:
            for seed in seeds:
                result = _train_one(
                    method=method,
                    seed=seed,
                    data=data,
                    hidden_dims=hidden_dims,
                    args=args,
                    row_callback=write_row,
                )
                results.append(result)
                manifest["completed"].append(
                    {
                        "method": method,
                        "seed": seed,
                        "best_validation_step": result[
                            "best_validation_step"
                        ],
                    }
                )
                _atomic_json(args.output_dir / "status.json", manifest)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        _atomic_json(args.output_dir / "status.json", manifest)
        raise
    finally:
        curve_file.close()

    aggregate: dict[str, Any] = {}
    for method in methods:
        selected = [
            result["selected_heldout_metrics"]
            for result in results
            if result["method"] == method
        ]
        metric_names = selected[0].keys()
        aggregate[method] = {
            name: {
                "mean": float(np.nanmean([entry[name] for entry in selected])),
                "std": float(np.nanstd([entry[name] for entry in selected])),
            }
            for name in metric_names
        }

    summary = {
        **manifest,
        "status": "complete",
        "results": results,
        "selected_heldout_aggregate": aggregate,
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    manifest["status"] = "complete"
    manifest["summary"] = str((args.output_dir / "summary.json").resolve())
    _atomic_json(args.output_dir / "status.json", manifest)
    print(json.dumps(_json_safe(aggregate), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
