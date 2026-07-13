#!/usr/bin/env python3
"""Cross-fit future-action predictability from replay observations.

This script is a lightweight, leakage-resistant diagnostic for action-chunk
datasets.  For every offset ``h`` it fits a multi-output ridge model

    current low_dim_state[t] -> action[t + h]

and reports held-out normalized MSE.  Entire replay episodes, rather than
individual transitions, are assigned to folds.  Ridge regularization is
selected with a second episode-group CV nested inside each outer training
split.  Confidence intervals resample replay episodes and reuse the same
bootstrap draw across all offsets.

The result is only a proxy for future expert-action predictability.  It is not
evidence by itself that a policy learned latent dynamics or an observable
state representation: policy multimodality, cross-episode shift, and model
misspecification can all contribute to the error.

Example:

    python scripts/analyze_replay_action_predictability.py \
      --replay-dir exp_local/my_run/replay \
      --max-horizon 20 \
      --output-csv /tmp/pen_action_predictability.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


PROXY_WARNING = (
    "future expert-action predictability proxy; not latent-dynamics proof"
)


@dataclass(frozen=True)
class ReplayEpisode:
    """One valid replay episode with no terminal storage sentinel."""

    episode_id: str
    states: np.ndarray
    actions: np.ndarray

    def __post_init__(self) -> None:
        states = np.asarray(self.states, dtype=np.float64)
        actions = np.asarray(self.actions, dtype=np.float64)
        if states.ndim < 2 or actions.ndim < 2:
            raise ValueError("states and actions must have a leading time axis")
        states = states.reshape(states.shape[0], -1)
        actions = actions.reshape(actions.shape[0], -1)
        if len(states) != len(actions):
            raise ValueError("states and actions must have equal valid lengths")
        if len(states) < 1:
            raise ValueError("an episode must contain at least one transition")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError("states and actions must be finite")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "actions", actions)


@dataclass(frozen=True)
class RidgeModel:
    """Standardized multi-output ridge model."""

    input_mean: np.ndarray
    input_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    coefficients: np.ndarray

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        normalized = (
            np.asarray(inputs, dtype=np.float64) - self.input_mean
        ) / self.input_scale
        return (
            normalized @ self.coefficients * self.target_scale
            + self.target_mean
        )


@dataclass(frozen=True)
class CrossFitResult:
    """Out-of-episode predictions for one future-action offset."""

    offset: int
    targets: np.ndarray
    predictions: np.ndarray
    groups: np.ndarray
    outer_fold_by_row: np.ndarray
    selected_alphas: tuple[float, ...]


def _encoded_episode_length(path: Path) -> int | None:
    """Read UniformReplayBuffer's ``..._episode_length_global.npz`` field."""

    parts = path.stem.rsplit("_", 3)
    if len(parts) != 4:
        return None
    try:
        return int(parts[-2])
    except ValueError:
        return None


def _replay_sort_key(path: Path) -> tuple[int, int, int, str]:
    """Sort UniformReplayBuffer files by global index, then episode index."""

    parts = path.stem.rsplit("_", 3)
    if len(parts) == 4:
        try:
            episode_index = int(parts[-3])
            global_index = int(parts[-1])
            return (0, global_index, episode_index, path.name)
        except ValueError:
            pass
    return (1, 0, 0, path.name)


def _valid_episode_length(
    path: Path,
    arrays: dict[str, np.ndarray],
    raw_length: int,
    sentinel_policy: str,
) -> int:
    if sentinel_policy not in {"auto", "always", "never"}:
        raise ValueError(
            "sentinel_policy must be one of: auto, always, never"
        )
    if sentinel_policy == "always":
        if raw_length < 2:
            raise ValueError(f"Cannot remove a sentinel from {path}")
        return raw_length - 1
    if sentinel_policy == "never":
        return raw_length

    encoded_length = _encoded_episode_length(path)
    if encoded_length is not None and encoded_length + 1 == raw_length:
        return encoded_length

    # UniformReplayBuffer writes -1 placeholders into its final terminal and
    # truncated slots.  This fallback handles renamed replay files while
    # avoiding unconditional removal from ordinary transition arrays.
    for key in ("terminal", "truncated"):
        if key in arrays:
            marker_array = np.asarray(arrays[key])
            if len(marker_array) < raw_length:
                continue
            marker = marker_array[:raw_length].reshape(raw_length, -1)[-1]
            if marker.size and np.all(marker < 0):
                return raw_length - 1
    return raw_length


def load_replay_episodes(
    replay_dir: Path | str,
    *,
    state_key: str = "low_dim_state",
    action_key: str = "action",
    sentinel_policy: str = "auto",
    max_episodes: int | None = None,
) -> list[ReplayEpisode]:
    """Load NPZ replay episodes without mutating or caching replay files."""

    replay_dir = Path(replay_dir)
    if not replay_dir.is_dir():
        raise FileNotFoundError(f"Replay directory does not exist: {replay_dir}")
    if max_episodes is not None and max_episodes < 1:
        raise ValueError("max_episodes must be positive when provided")

    paths = sorted(replay_dir.glob("*.npz"), key=_replay_sort_key)
    if max_episodes is not None:
        paths = paths[:max_episodes]
    episodes = []
    for path in paths:
        with np.load(path, allow_pickle=False) as handle:
            if state_key not in handle or action_key not in handle:
                continue
            keys = {state_key, action_key, "terminal", "truncated"}
            arrays = {
                key: np.asarray(handle[key])
                for key in keys
                if key in handle
            }
        states = arrays[state_key]
        actions = arrays[action_key]
        raw_length = min(len(states), len(actions))
        valid_length = _valid_episode_length(
            path,
            arrays,
            raw_length,
            sentinel_policy,
        )
        if valid_length < 1:
            continue
        episodes.append(
            ReplayEpisode(
                episode_id=path.name,
                states=states[:valid_length],
                actions=actions[:valid_length],
            )
        )

    if not episodes:
        raise RuntimeError(
            f"No NPZ episodes containing {state_key!r} and {action_key!r} "
            f"were found in {replay_dir}"
        )
    state_dims = {episode.states.shape[1] for episode in episodes}
    action_dims = {episode.actions.shape[1] for episode in episodes}
    if len(state_dims) != 1 or len(action_dims) != 1:
        raise ValueError(
            "All episodes must have common flattened state and action dimensions"
        )
    return episodes


def make_group_folds(
    group_ids: Sequence[int] | np.ndarray,
    n_splits: int,
    seed: int,
) -> tuple[np.ndarray, ...]:
    """Create deterministic, shuffled, mutually disjoint group folds."""

    groups = np.unique(np.asarray(group_ids, dtype=np.int64))
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if len(groups) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} groups, but received {len(groups)}"
        )
    rng = np.random.default_rng(seed)
    shuffled = groups.copy()
    rng.shuffle(shuffled)
    return tuple(part.copy() for part in np.array_split(shuffled, n_splits))


def make_offset_dataset(
    episodes: Sequence[ReplayEpisode],
    offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pair state[t] with action[t + offset] within each episode."""

    if offset < 0:
        raise ValueError("offset must be non-negative")
    inputs = []
    targets = []
    groups = []
    for group, episode in enumerate(episodes):
        count = len(episode.actions) - offset
        if count <= 0:
            continue
        inputs.append(episode.states[:count])
        targets.append(episode.actions[offset : offset + count])
        groups.append(np.full(count, group, dtype=np.int64))
    if not inputs:
        raise ValueError(f"No valid state/action pairs at offset {offset}")
    return np.concatenate(inputs), np.concatenate(targets), np.concatenate(groups)


def fit_ridge(
    inputs: np.ndarray,
    targets: np.ndarray,
    alpha: float,
) -> RidgeModel:
    """Fit standardized multi-output ridge using a stable eigensystem solve."""

    inputs = np.asarray(inputs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if inputs.ndim != 2 or targets.ndim != 2:
        raise ValueError("inputs and targets must be rank-2")
    if len(inputs) != len(targets) or len(inputs) < 1:
        raise ValueError("inputs and targets must have a non-empty common length")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive")

    input_mean = inputs.mean(axis=0)
    input_scale = inputs.std(axis=0)
    input_scale = np.where(input_scale > 1e-12, input_scale, 1.0)
    target_mean = targets.mean(axis=0)
    target_scale = targets.std(axis=0)
    target_scale = np.where(target_scale > 1e-12, target_scale, 1.0)
    normalized_inputs = (inputs - input_mean) / input_scale
    normalized_targets = (targets - target_mean) / target_scale

    gram = normalized_inputs.T @ normalized_inputs
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    rotated_rhs = eigenvectors.T @ (
        normalized_inputs.T @ normalized_targets
    )
    coefficients = eigenvectors @ (
        rotated_rhs / (eigenvalues[:, None] + alpha)
    )
    return RidgeModel(
        input_mean=input_mean,
        input_scale=input_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        coefficients=coefficients,
    )


def _validation_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    residual_sum = float(np.square(targets - predictions).sum())
    centered_sum = float(np.square(targets - targets.mean(axis=0)).sum())
    if centered_sum <= 1e-12:
        return residual_sum / targets.size
    return residual_sum / centered_sum


def _select_alpha_nested(
    inputs: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    *,
    alphas: np.ndarray,
    inner_splits: int,
    seed: int,
) -> float:
    inner_folds = make_group_folds(groups, inner_splits, seed)
    scores = np.zeros(len(alphas), dtype=np.float64)
    for validation_groups in inner_folds:
        validation = np.isin(groups, validation_groups)
        training = ~validation
        if not validation.any() or not training.any():
            raise ValueError("An inner group fold has no training or validation rows")
        for index, alpha in enumerate(alphas):
            model = fit_ridge(inputs[training], targets[training], float(alpha))
            scores[index] += _validation_score(
                targets[validation],
                model.predict(inputs[validation]),
            )
    return float(alphas[int(np.argmin(scores))])


def cross_fit_offset(
    episodes: Sequence[ReplayEpisode],
    offset: int,
    *,
    outer_group_folds: Sequence[np.ndarray],
    inner_splits: int,
    alphas: Sequence[float],
    cv_seed: int,
) -> CrossFitResult:
    """Generate predictions whose target episode was absent from model fitting."""

    inputs, targets, groups = make_offset_dataset(episodes, offset)
    alphas_array = np.unique(np.asarray(alphas, dtype=np.float64))
    if not len(alphas_array) or not np.isfinite(alphas_array).all():
        raise ValueError("alphas must contain at least one finite value")
    if np.any(alphas_array <= 0.0):
        raise ValueError("all alphas must be positive")

    predictions = np.empty_like(targets)
    outer_fold_by_row = np.full(len(targets), -1, dtype=np.int64)
    selected_alphas = []
    for outer_index, test_groups in enumerate(outer_group_folds):
        test = np.isin(groups, test_groups)
        train = ~test
        if not test.any() or not train.any():
            raise ValueError(
                f"Outer fold {outer_index} has no usable train or test rows "
                f"at offset {offset}"
            )
        usable_train_groups = np.unique(groups[train])
        if len(usable_train_groups) < inner_splits:
            raise ValueError(
                f"Outer fold {outer_index} has only {len(usable_train_groups)} "
                f"training groups, fewer than inner_splits={inner_splits}"
            )
        alpha = _select_alpha_nested(
            inputs[train],
            targets[train],
            groups[train],
            alphas=alphas_array,
            inner_splits=inner_splits,
            seed=cv_seed + 100 * offset + outer_index,
        )
        model = fit_ridge(inputs[train], targets[train], alpha)
        predictions[test] = model.predict(inputs[test])
        outer_fold_by_row[test] = outer_index
        selected_alphas.append(alpha)

    if np.any(outer_fold_by_row < 0):
        missing_groups = np.unique(groups[outer_fold_by_row < 0]).tolist()
        raise ValueError(
            f"Outer folds did not assign all rows; missing groups={missing_groups}"
        )
    return CrossFitResult(
        offset=offset,
        targets=targets,
        predictions=predictions,
        groups=groups,
        outer_fold_by_row=outer_fold_by_row,
        selected_alphas=tuple(selected_alphas),
    )


def predictability_metrics(result: CrossFitResult) -> dict[str, float]:
    """Return aggregate and equal-action-dimension normalized errors."""

    squared_error = np.square(result.targets - result.predictions)
    mse_by_dimension = squared_error.mean(axis=0)
    variance_by_dimension = result.targets.var(axis=0)
    valid = variance_by_dimension > 1e-12
    if not valid.any():
        raise ValueError(f"All action dimensions are constant at {result.offset=}")
    weighted_nmse = float(
        mse_by_dimension[valid].sum() / variance_by_dimension[valid].sum()
    )
    macro_nmse = float(
        np.mean(mse_by_dimension[valid] / variance_by_dimension[valid])
    )
    return {
        "mse": float(mse_by_dimension.mean()),
        "target_variance": float(variance_by_dimension.mean()),
        "nmse_variance_weighted": weighted_nmse,
        "nmse_macro_action_dim": macro_nmse,
        "crossfit_r2": 1.0 - weighted_nmse,
    }


def _group_sufficient_statistics(
    result: CrossFitResult,
    num_groups: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    action_dim = result.targets.shape[1]
    counts = np.zeros(num_groups, dtype=np.int64)
    target_sums = np.zeros((num_groups, action_dim), dtype=np.float64)
    target_square_sums = np.zeros_like(target_sums)
    residual_square_sums = np.zeros_like(target_sums)
    for group in range(num_groups):
        mask = result.groups == group
        if not mask.any():
            continue
        group_targets = result.targets[mask]
        group_residuals = group_targets - result.predictions[mask]
        counts[group] = len(group_targets)
        target_sums[group] = group_targets.sum(axis=0)
        target_square_sums[group] = np.square(group_targets).sum(axis=0)
        residual_square_sums[group] = np.square(group_residuals).sum(axis=0)
    return counts, target_sums, target_square_sums, residual_square_sums


def bootstrap_nmse_profiles(
    results: Sequence[CrossFitResult],
    *,
    num_groups: int,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Episode bootstrap with common resamples for every prediction offset."""

    if replicates < 1:
        raise ValueError("replicates must be positive")
    if num_groups < 2:
        raise ValueError("num_groups must be at least 2")
    statistics = [
        _group_sufficient_statistics(result, num_groups) for result in results
    ]
    rng = np.random.default_rng(seed)
    profiles = np.full((replicates, len(results)), np.nan, dtype=np.float64)
    for replicate in range(replicates):
        sampled_groups = rng.integers(0, num_groups, size=num_groups)
        for offset_index, (counts, sums, squares, residuals) in enumerate(
            statistics
        ):
            sample_count = int(counts[sampled_groups].sum())
            if sample_count < 2:
                continue
            target_sum = sums[sampled_groups].sum(axis=0)
            target_square_sum = squares[sampled_groups].sum(axis=0)
            residual_square_sum = residuals[sampled_groups].sum(axis=0)
            variance = (
                target_square_sum / sample_count
                - np.square(target_sum / sample_count)
            )
            variance = np.maximum(variance, 0.0)
            mse = residual_square_sum / sample_count
            valid = variance > 1e-12
            if valid.any():
                profiles[replicate, offset_index] = (
                    mse[valid].sum() / variance[valid].sum()
                )
    return profiles


def analyze_replay(
    episodes: Sequence[ReplayEpisode],
    *,
    max_horizon: int,
    outer_splits: int,
    inner_splits: int,
    alphas: Sequence[float],
    cv_seed: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence: float,
) -> tuple[list[dict[str, object]], np.ndarray]:
    """Run the complete cross-fit profile and episode bootstrap."""

    if max_horizon < 1:
        raise ValueError("max_horizon must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    groups = np.arange(len(episodes), dtype=np.int64)
    outer_folds = make_group_folds(groups, outer_splits, cv_seed)
    results = [
        cross_fit_offset(
            episodes,
            offset,
            outer_group_folds=outer_folds,
            inner_splits=inner_splits,
            alphas=alphas,
            cv_seed=cv_seed,
        )
        for offset in range(max_horizon)
    ]
    bootstrap = bootstrap_nmse_profiles(
        results,
        num_groups=len(episodes),
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    tail_probability = (1.0 - confidence) / 2.0
    rows = []
    for result_index, result in enumerate(results):
        samples = bootstrap[:, result_index]
        samples = samples[np.isfinite(samples)]
        if not len(samples):
            raise ValueError(
                f"No finite bootstrap samples at offset {result.offset}"
            )
        metrics = predictability_metrics(result)
        rows.append(
            {
                "offset": result.offset,
                "n_pairs": len(result.targets),
                "n_episodes_with_pairs": len(np.unique(result.groups)),
                **metrics,
                "nmse_ci_low": float(np.quantile(samples, tail_probability)),
                "nmse_ci_high": float(
                    np.quantile(samples, 1.0 - tail_probability)
                ),
                "selected_alpha_by_outer_fold": json.dumps(
                    result.selected_alphas,
                    separators=(",", ":"),
                ),
            }
        )
    return rows, bootstrap


CSV_FIELDS = (
    "proxy_type",
    "replay_dir",
    "state_key",
    "action_key",
    "sentinel_policy",
    "max_episodes",
    "offset",
    "n_pairs",
    "n_episodes_with_pairs",
    "state_dim",
    "action_dim",
    "outer_folds",
    "inner_folds",
    "cv_seed",
    "bootstrap_seed",
    "bootstrap_replicates",
    "bootstrap_confidence",
    "ridge_alpha_candidates",
    "selected_alpha_by_outer_fold",
    "mse",
    "target_variance",
    "nmse_variance_weighted",
    "nmse_macro_action_dim",
    "crossfit_r2",
    "nmse_ci_low",
    "nmse_ci_high",
)


def write_csv(path: Path | str, rows: Sequence[dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--max-horizon", type=int, default=20)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument(
        "--ridge-alphas",
        type=float,
        nargs="+",
        default=(0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0),
    )
    parser.add_argument("--cv-seed", type=int, default=0)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--state-key", default="low_dim_state")
    parser.add_argument("--action-key", default="action")
    parser.add_argument(
        "--sentinel-policy",
        choices=("auto", "always", "never"),
        default="auto",
    )
    parser.add_argument("--max-episodes", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    episodes = load_replay_episodes(
        args.replay_dir,
        state_key=args.state_key,
        action_key=args.action_key,
        sentinel_policy=args.sentinel_policy,
        max_episodes=args.max_episodes,
    )
    rows, bootstrap = analyze_replay(
        episodes,
        max_horizon=args.max_horizon,
        outer_splits=args.outer_folds,
        inner_splits=args.inner_folds,
        alphas=args.ridge_alphas,
        cv_seed=args.cv_seed,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        confidence=args.bootstrap_confidence,
    )

    alpha_json = json.dumps(
        sorted(set(float(alpha) for alpha in args.ridge_alphas)),
        separators=(",", ":"),
    )
    common = {
        "proxy_type": "future_expert_action_predictability",
        "replay_dir": str(args.replay_dir.resolve()),
        "state_key": args.state_key,
        "action_key": args.action_key,
        "sentinel_policy": args.sentinel_policy,
        "max_episodes": "" if args.max_episodes is None else args.max_episodes,
        "state_dim": episodes[0].states.shape[1],
        "action_dim": episodes[0].actions.shape[1],
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "cv_seed": args.cv_seed,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_confidence": args.bootstrap_confidence,
        "ridge_alpha_candidates": alpha_json,
    }
    output_rows = [{**common, **row} for row in rows]
    write_csv(args.output_csv, output_rows)

    print(PROXY_WARNING)
    print(
        f"episodes={len(episodes)} transitions={sum(len(e.actions) for e in episodes)} "
        f"state_dim={common['state_dim']} action_dim={common['action_dim']}"
    )
    print("offset,n_pairs,nmse,crossfit_r2,nmse_ci")
    for row in rows:
        print(
            f"{row['offset']},{row['n_pairs']},"
            f"{row['nmse_variance_weighted']:.6f},"
            f"{row['crossfit_r2']:.6f},"
            f"[{row['nmse_ci_low']:.6f},{row['nmse_ci_high']:.6f}]"
        )
    if len(rows) > 1:
        contrast = bootstrap[:, -1] - bootstrap[:, 0]
        contrast = contrast[np.isfinite(contrast)]
        tail_probability = (1.0 - args.bootstrap_confidence) / 2.0
        point_difference = (
            rows[-1]["nmse_variance_weighted"]
            - rows[0]["nmse_variance_weighted"]
        )
        low, high = np.quantile(
            contrast,
            [tail_probability, 1.0 - tail_probability],
        )
        print(
            f"last_minus_first_nmse={point_difference:.6f} "
            f"paired_ci=[{low:.6f},{high:.6f}]"
        )
    print(f"wrote {len(output_rows)} rows to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
