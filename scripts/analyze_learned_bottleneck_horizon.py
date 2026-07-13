#!/usr/bin/env python3
"""Controlled learned-bottleneck experiment for action-chunk horizons.

The experiment fits a linear observation encoder, a fixed-width shared
bottleneck, and K future-action heads for K in {1, 2, 5, 10, 20}.  The joint
linear model is solved exactly with reduced-rank regression, so comparisons do
not depend on optimizer convergence.

Two causal axes are kept separate by construction:

* ``observability_rank`` and ``observability_scale`` alter which latent-state
  modes can be recovered from a deterministic future-action sequence.
* ``predictive_noise_std`` alters only process noise.  The closed-loop matrix
  is a scaled orthogonal matrix and the action readout has unit norm, so the
  predictive noise variance is identical across observability conditions at a
  fixed noise level and spectral radius.

The finite training set and observation nuisance features let unpredictable
future targets compete for the shared bottleneck.  Thus the same sweep can test
whether additional observable future actions improve the learned state probe
while additional predictive uncertainty hurts it and the first-action head.

Example:

    python scripts/analyze_learned_bottleneck_horizon.py \
      --output-csv /tmp/learned_bottleneck.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SharedDataset:
    """Inputs and standard-normal process innovations reused by every cell."""

    train_states: np.ndarray
    test_states: np.ndarray
    train_observations: np.ndarray
    test_observations: np.ndarray
    train_process_innovations: np.ndarray
    test_process_innovations: np.ndarray


@dataclass(frozen=True)
class ReducedRankBottleneck:
    """Linear encoder and heads learned by reduced-rank regression."""

    observation_mean: np.ndarray
    observation_scale: np.ndarray
    target_mean: np.ndarray
    encoder: np.ndarray
    heads: np.ndarray
    fitted_rank: int

    def encode(self, observations: np.ndarray) -> np.ndarray:
        normalized = (
            np.asarray(observations, dtype=np.float64) - self.observation_mean
        ) / self.observation_scale
        return normalized @ self.encoder

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return self.target_mean + self.encode(observations) @ self.heads


def make_shared_dataset(
    *,
    seed: int,
    n_train: int,
    n_test: int,
    state_dim: int,
    nuisance_dim: int,
    max_horizon: int,
    observation_noise_std: float,
) -> SharedDataset:
    """Generate one train/test input design shared across factorial cells."""

    if n_train < 2 or n_test < 2:
        raise ValueError("n_train and n_test must both be at least 2")
    if state_dim < 2 or state_dim % 2:
        raise ValueError("state_dim must be a positive even integer >= 2")
    if nuisance_dim < 0:
        raise ValueError("nuisance_dim must be non-negative")
    if max_horizon < 1:
        raise ValueError("max_horizon must be positive")
    if not np.isfinite(observation_noise_std) or observation_noise_std < 0.0:
        raise ValueError("observation_noise_std must be finite and non-negative")

    rng = np.random.default_rng(seed)
    observation_dim = state_dim + nuisance_dim
    mixing, _ = np.linalg.qr(rng.normal(size=(observation_dim, observation_dim)))

    def split(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        states = rng.normal(size=(size, state_dim))
        noisy_state = states + observation_noise_std * rng.normal(
            size=states.shape
        )
        nuisance = rng.normal(size=(size, nuisance_dim))
        raw_observation = np.concatenate([noisy_state, nuisance], axis=1)
        observations = raw_observation @ mixing
        innovations = rng.normal(
            size=(size, max(0, max_horizon - 1), state_dim)
        )
        return states, observations, innovations

    train_states, train_observations, train_innovations = split(n_train)
    test_states, test_observations, test_innovations = split(n_test)
    return SharedDataset(
        train_states=train_states,
        test_states=test_states,
        train_observations=train_observations,
        test_observations=test_observations,
        train_process_innovations=train_innovations,
        test_process_innovations=test_innovations,
    )


def rotational_dynamics(state_dim: int, spectral_radius: float) -> np.ndarray:
    """Build distinct 2-D rotations with a common spectral radius."""

    if state_dim < 2 or state_dim % 2:
        raise ValueError("state_dim must be a positive even integer >= 2")
    if not np.isfinite(spectral_radius) or spectral_radius < 0.0:
        raise ValueError("spectral_radius must be finite and non-negative")

    num_blocks = state_dim // 2
    angles = np.linspace(0.23, 1.07, num_blocks)
    dynamics = np.zeros((state_dim, state_dim), dtype=np.float64)
    for block, angle in enumerate(angles):
        cosine = np.cos(angle)
        sine = np.sin(angle)
        rotation = np.asarray([[cosine, -sine], [sine, cosine]])
        start = 2 * block
        dynamics[start : start + 2, start : start + 2] = (
            spectral_radius * rotation
        )
    return dynamics


def action_readout(
    state_dim: int,
    observability_rank: int,
    observability_scale: float,
) -> np.ndarray:
    """Create a unit-norm readout with controlled active modes/conditioning."""

    if state_dim < 2 or state_dim % 2:
        raise ValueError("state_dim must be a positive even integer >= 2")
    if (
        observability_rank < 2
        or observability_rank > state_dim
        or observability_rank % 2
    ):
        raise ValueError(
            "observability_rank must be even and in [2, state_dim]"
        )
    if not np.isfinite(observability_scale) or observability_scale < 0.0:
        raise ValueError("observability_scale must be finite and non-negative")

    readout = np.zeros(state_dim, dtype=np.float64)
    readout[0] = 1.0
    readout[1:observability_rank] = observability_scale
    return readout / np.linalg.norm(readout)


def rollout_actions(
    initial_states: np.ndarray,
    process_innovations: np.ndarray,
    dynamics: np.ndarray,
    readout: np.ndarray,
    predictive_noise_std: float,
    horizon: int,
) -> np.ndarray:
    """Roll out scalar expert actions from a partially stochastic system."""

    initial_states = np.asarray(initial_states, dtype=np.float64)
    process_innovations = np.asarray(process_innovations, dtype=np.float64)
    dynamics = np.asarray(dynamics, dtype=np.float64)
    readout = np.asarray(readout, dtype=np.float64)
    if initial_states.ndim != 2:
        raise ValueError("initial_states must have shape [batch, state_dim]")
    if dynamics.shape != (initial_states.shape[1], initial_states.shape[1]):
        raise ValueError("dynamics shape does not match initial_states")
    if readout.shape != (initial_states.shape[1],):
        raise ValueError("readout shape does not match initial_states")
    expected_innovation_shape = (
        initial_states.shape[0],
        max(0, horizon - 1),
        initial_states.shape[1],
    )
    if process_innovations.shape[:1] != expected_innovation_shape[:1] or (
        process_innovations.shape[1] < expected_innovation_shape[1]
        or process_innovations.shape[2:] != expected_innovation_shape[2:]
    ):
        raise ValueError(
            "process_innovations must contain at least horizon - 1 innovations"
        )
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not np.isfinite(predictive_noise_std) or predictive_noise_std < 0.0:
        raise ValueError("predictive_noise_std must be finite and non-negative")

    state = initial_states.copy()
    actions = []
    for offset in range(horizon):
        actions.append(state @ readout)
        if offset + 1 < horizon:
            state = state @ dynamics.T + (
                predictive_noise_std * process_innovations[:, offset]
            )
    return np.stack(actions, axis=1)


def observability_matrix(
    dynamics: np.ndarray,
    readout: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Return [c; cA; ...; cA^(H-1)]."""

    if horizon < 1:
        raise ValueError("horizon must be positive")
    dynamics = np.asarray(dynamics, dtype=np.float64)
    readout = np.asarray(readout, dtype=np.float64)
    rows = []
    power = np.eye(dynamics.shape[0], dtype=np.float64)
    for _ in range(horizon):
        rows.append(readout @ power)
        power = power @ dynamics
    return np.stack(rows)


def observability_metrics(matrix: np.ndarray) -> dict[str, float | int]:
    """Return rank and conditioning of the nonzero observable subspace."""

    matrix = np.asarray(matrix, dtype=np.float64)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    sigma_max = float(singular_values[0]) if singular_values.size else 0.0
    tolerance = max(matrix.shape) * np.finfo(np.float64).eps * sigma_max
    nonzero = singular_values[singular_values > tolerance]
    rank = int(nonzero.size)
    sigma_min = float(nonzero[-1]) if rank else 0.0
    condition = sigma_max / sigma_min if sigma_min > 0.0 else np.inf
    return {
        "rank": rank,
        "sigma_max": sigma_max,
        "sigma_min_nonzero": sigma_min,
        "condition_nonzero": condition,
        "gramian_trace": float(np.square(singular_values).sum()),
    }


def predictive_noise_variances(
    dynamics: np.ndarray,
    readout: np.ndarray,
    predictive_noise_std: float,
    horizon: int,
) -> np.ndarray:
    """Analytic Var[a_k | x_0] caused only by process innovations."""

    dynamics = np.asarray(dynamics, dtype=np.float64)
    readout = np.asarray(readout, dtype=np.float64)
    state_covariance = np.zeros_like(dynamics)
    process_covariance = (
        predictive_noise_std**2 * np.eye(dynamics.shape[0])
    )
    variances = []
    for _ in range(horizon):
        variances.append(float(readout @ state_covariance @ readout))
        state_covariance = (
            dynamics @ state_covariance @ dynamics.T + process_covariance
        )
    return np.asarray(variances)


def fit_reduced_rank_bottleneck(
    observations: np.ndarray,
    targets: np.ndarray,
    bottleneck_width: int,
) -> ReducedRankBottleneck:
    """Fit the exact rank-constrained empirical squared-error solution."""

    observations = np.asarray(observations, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if observations.ndim != 2 or targets.ndim != 2:
        raise ValueError("observations and targets must both be rank-2")
    if observations.shape[0] != targets.shape[0]:
        raise ValueError("observations and targets must have equal row counts")
    if bottleneck_width < 1:
        raise ValueError("bottleneck_width must be positive")

    observation_mean = observations.mean(axis=0)
    observation_scale = observations.std(axis=0)
    observation_scale = np.where(observation_scale > 1e-12, observation_scale, 1.0)
    normalized = (observations - observation_mean) / observation_scale
    target_mean = targets.mean(axis=0)
    centered_targets = targets - target_mean

    ordinary_coefficients = np.linalg.lstsq(
        normalized,
        centered_targets,
        rcond=None,
    )[0]
    fitted_targets = normalized @ ordinary_coefficients
    _, singular_values, right_vectors_t = np.linalg.svd(
        fitted_targets,
        full_matrices=False,
    )
    if singular_values.size:
        tolerance = (
            max(fitted_targets.shape)
            * np.finfo(np.float64).eps
            * singular_values[0]
        )
        empirical_rank = int(np.sum(singular_values > tolerance))
    else:
        empirical_rank = 0
    fitted_rank = min(bottleneck_width, empirical_rank)

    encoder = np.zeros(
        (observations.shape[1], bottleneck_width),
        dtype=np.float64,
    )
    heads = np.zeros(
        (bottleneck_width, targets.shape[1]),
        dtype=np.float64,
    )
    if fitted_rank:
        output_subspace = right_vectors_t[:fitted_rank].T
        encoder[:, :fitted_rank] = ordinary_coefficients @ output_subspace
        heads[:fitted_rank] = output_subspace.T
    return ReducedRankBottleneck(
        observation_mean=observation_mean,
        observation_scale=observation_scale,
        target_mean=target_mean,
        encoder=encoder,
        heads=heads,
        fitted_rank=fitted_rank,
    )


def linear_probe_r2(
    train_features: np.ndarray,
    train_states: np.ndarray,
    test_features: np.ndarray,
    test_states: np.ndarray,
) -> float:
    """Fit a train-only linear state probe and return held-out multivariate R2."""

    train_features = np.asarray(train_features, dtype=np.float64)
    train_states = np.asarray(train_states, dtype=np.float64)
    test_features = np.asarray(test_features, dtype=np.float64)
    test_states = np.asarray(test_states, dtype=np.float64)
    feature_mean = train_features.mean(axis=0)
    state_mean = train_states.mean(axis=0)
    coefficients = np.linalg.lstsq(
        train_features - feature_mean,
        train_states - state_mean,
        rcond=None,
    )[0]
    prediction = state_mean + (test_features - feature_mean) @ coefficients
    residual = np.square(prediction - test_states).sum()
    total = np.square(test_states - test_states.mean(axis=0)).sum()
    return float(1.0 - residual / total) if total > 0.0 else np.nan


def _validated_sweep_values(
    name: str,
    values: Sequence[int | float],
    *,
    integer: bool = False,
) -> tuple[int | float, ...]:
    if not values:
        raise ValueError(f"{name} must be non-empty")
    converted = tuple(int(value) if integer else float(value) for value in values)
    return converted


def run_sweep(
    *,
    horizons: Sequence[int] = (1, 2, 5, 10, 20),
    spectral_radii: Sequence[float] = (0.7, 0.95),
    observability_ranks: Sequence[int] = (2, 6),
    observability_scales: Sequence[float] = (0.1, 1.0),
    predictive_noise_stds: Sequence[float] = (0.25, 1.0),
    seeds: Sequence[int] = (0, 1, 2),
    n_train: int = 256,
    n_test: int = 2048,
    state_dim: int = 6,
    nuisance_dim: int = 18,
    bottleneck_width: int = 3,
    observation_noise_std: float = 0.1,
) -> list[dict[str, int | float]]:
    """Run the full factorial sweep and return tidy per-offset records."""

    horizons = _validated_sweep_values("horizons", horizons, integer=True)
    spectral_radii = _validated_sweep_values(
        "spectral_radii", spectral_radii
    )
    observability_ranks = _validated_sweep_values(
        "observability_ranks", observability_ranks, integer=True
    )
    observability_scales = _validated_sweep_values(
        "observability_scales", observability_scales
    )
    predictive_noise_stds = _validated_sweep_values(
        "predictive_noise_stds", predictive_noise_stds
    )
    seeds = _validated_sweep_values("seeds", seeds, integer=True)
    if any(horizon < 1 for horizon in horizons):
        raise ValueError("all horizons must be positive")
    if bottleneck_width < 1:
        raise ValueError("bottleneck_width must be positive")

    max_horizon = max(horizons)
    rows: list[dict[str, int | float]] = []
    for seed in seeds:
        dataset = make_shared_dataset(
            seed=int(seed),
            n_train=n_train,
            n_test=n_test,
            state_dim=state_dim,
            nuisance_dim=nuisance_dim,
            max_horizon=max_horizon,
            observation_noise_std=observation_noise_std,
        )
        for spectral_radius in spectral_radii:
            dynamics = rotational_dynamics(state_dim, float(spectral_radius))
            for requested_rank in observability_ranks:
                for observability_scale in observability_scales:
                    readout = action_readout(
                        state_dim,
                        int(requested_rank),
                        float(observability_scale),
                    )
                    for predictive_noise_std in predictive_noise_stds:
                        train_actions = rollout_actions(
                            dataset.train_states,
                            dataset.train_process_innovations,
                            dynamics,
                            readout,
                            float(predictive_noise_std),
                            max_horizon,
                        )
                        test_actions = rollout_actions(
                            dataset.test_states,
                            dataset.test_process_innovations,
                            dynamics,
                            readout,
                            float(predictive_noise_std),
                            max_horizon,
                        )
                        noise_variances = predictive_noise_variances(
                            dynamics,
                            readout,
                            float(predictive_noise_std),
                            max_horizon,
                        )
                        for horizon in horizons:
                            horizon = int(horizon)
                            model = fit_reduced_rank_bottleneck(
                                dataset.train_observations,
                                train_actions[:, :horizon],
                                bottleneck_width,
                            )
                            train_latent = model.encode(
                                dataset.train_observations
                            )
                            test_latent = model.encode(dataset.test_observations)
                            state_probe_r2 = linear_probe_r2(
                                train_latent,
                                dataset.train_states,
                                test_latent,
                                dataset.test_states,
                            )
                            active_state_probe_r2 = linear_probe_r2(
                                train_latent,
                                dataset.train_states[:, : int(requested_rank)],
                                test_latent,
                                dataset.test_states[:, : int(requested_rank)],
                            )
                            prediction = model.predict(
                                dataset.test_observations
                            )
                            per_offset_mse = np.square(
                                prediction - test_actions[:, :horizon]
                            ).mean(axis=0)
                            target_variance = test_actions[:, :horizon].var(
                                axis=0
                            )
                            per_offset_nmse = per_offset_mse / np.maximum(
                                target_variance,
                                1e-12,
                            )
                            observability = observability_metrics(
                                observability_matrix(
                                    dynamics,
                                    readout,
                                    horizon,
                                )
                            )
                            common = {
                                "seed": int(seed),
                                "train_horizon": horizon,
                                "spectral_radius": float(spectral_radius),
                                "observability_rank_requested": int(
                                    requested_rank
                                ),
                                "observability_scale": float(
                                    observability_scale
                                ),
                                "predictive_noise_std": float(
                                    predictive_noise_std
                                ),
                                "bottleneck_width": bottleneck_width,
                                "bottleneck_fitted_rank": model.fitted_rank,
                                "state_probe_r2": state_probe_r2,
                                "active_state_probe_r2": active_state_probe_r2,
                                "first_action_mse": float(per_offset_mse[0]),
                                "first_action_nmse": float(per_offset_nmse[0]),
                                "mean_chunk_mse": float(per_offset_mse.mean()),
                                "mean_chunk_nmse": float(per_offset_nmse.mean()),
                                "observability_rank_realized": int(
                                    observability["rank"]
                                ),
                                "observability_sigma_max": float(
                                    observability["sigma_max"]
                                ),
                                "observability_sigma_min_nonzero": float(
                                    observability["sigma_min_nonzero"]
                                ),
                                "observability_condition_nonzero": float(
                                    observability["condition_nonzero"]
                                ),
                                "observability_gramian_trace": float(
                                    observability["gramian_trace"]
                                ),
                                "mean_predictive_noise_variance": float(
                                    noise_variances[:horizon].mean()
                                ),
                                "final_predictive_noise_variance": float(
                                    noise_variances[horizon - 1]
                                ),
                            }
                            for offset in range(horizon):
                                rows.append(
                                    {
                                        **common,
                                        "eval_offset": offset,
                                        "offset_mse": float(
                                            per_offset_mse[offset]
                                        ),
                                        "offset_nmse": float(
                                            per_offset_nmse[offset]
                                        ),
                                        "offset_target_variance": float(
                                            target_variance[offset]
                                        ),
                                        "offset_predictive_noise_variance": float(
                                            noise_variances[offset]
                                        ),
                                    }
                                )
    return rows


def write_csv(
    rows: Sequence[dict[str, int | float]],
    output_path: Path,
) -> None:
    """Write tidy sweep rows with a stable header."""

    if not rows:
        raise ValueError("cannot write an empty sweep")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the controlled learned-bottleneck horizon sweep."
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("linear_bottleneck_chunk_sweep.csv"),
    )
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 5, 10, 20])
    parser.add_argument(
        "--spectral-radii", nargs="+", type=float, default=[0.7, 0.95]
    )
    parser.add_argument(
        "--observability-ranks", nargs="+", type=int, default=[2, 6]
    )
    parser.add_argument(
        "--observability-scales", nargs="+", type=float, default=[0.1, 1.0]
    )
    parser.add_argument(
        "--predictive-noise-stds", nargs="+", type=float, default=[0.25, 1.0]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-train", type=int, default=256)
    parser.add_argument("--n-test", type=int, default=2048)
    parser.add_argument("--state-dim", type=int, default=6)
    parser.add_argument("--nuisance-dim", type=int, default=18)
    parser.add_argument("--bottleneck-width", type=int, default=3)
    parser.add_argument("--observation-noise-std", type=float, default=0.1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        rows = run_sweep(
            horizons=args.horizons,
            spectral_radii=args.spectral_radii,
            observability_ranks=args.observability_ranks,
            observability_scales=args.observability_scales,
            predictive_noise_stds=args.predictive_noise_stds,
            seeds=args.seeds,
            n_train=args.n_train,
            n_test=args.n_test,
            state_dim=args.state_dim,
            nuisance_dim=args.nuisance_dim,
            bottleneck_width=args.bottleneck_width,
            observation_noise_std=args.observation_noise_std,
        )
    except ValueError as exc:
        parser.error(str(exc))
    write_csv(rows, args.output_csv)
    model_count = len(
        {
            (
                row["seed"],
                row["train_horizon"],
                row["spectral_radius"],
                row["observability_rank_requested"],
                row["observability_scale"],
                row["predictive_noise_std"],
            )
            for row in rows
        }
    )
    print(
        f"wrote {len(rows)} per-offset rows from {model_count} models "
        f"to {args.output_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
