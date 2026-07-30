"""BiGym feature/action windows for the pinned official Legato policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np


Array = np.ndarray
EpisodeFeatureFn = Callable[[Any], Array]


@dataclass(frozen=True)
class MinMaxActionTransform:
    """Match RoboBase's BiGym min/max action transform without Gym imports."""

    minimum: Array
    maximum: Array
    margin: float = 0.0

    def __post_init__(self) -> None:
        minimum = np.asarray(self.minimum, dtype=np.float32)
        maximum = np.asarray(self.maximum, dtype=np.float32)
        if minimum.ndim != 1 or maximum.shape != minimum.shape:
            raise ValueError("minimum and maximum must be same-shaped vectors.")
        if np.any(maximum < minimum):
            raise ValueError("maximum must be greater than or equal to minimum.")
        if self.margin < 0:
            raise ValueError("margin must be non-negative.")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    @property
    def expanded_minimum(self) -> Array:
        return self.minimum - np.abs(self.minimum) * self.margin

    @property
    def expanded_maximum(self) -> Array:
        return self.maximum + np.abs(self.maximum) * self.margin

    def normalize(self, action: Array) -> Array:
        action = np.asarray(action, dtype=np.float32)
        unit = (action - self.expanded_minimum) / (
            self.expanded_maximum - self.expanded_minimum + 1e-8
        )
        return (unit * 2.0 - 1.0).astype(np.float32, copy=False)

    def denormalize(self, action: Array) -> Array:
        action = np.asarray(action, dtype=np.float32)
        unit = (np.clip(action, -1.0, 1.0) + 1.0) / 2.0
        result = unit * (self.expanded_maximum - self.expanded_minimum)
        return (result + self.expanded_minimum).astype(np.float32, copy=False)


@dataclass(frozen=True)
class FeatureEpisode:
    """One boundary-preserving BiGym episode after observation encoding."""

    features: Array
    actions: Array
    episode_id: str = ""

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float32)
        actions = np.asarray(self.actions, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError(f"features must have shape [T, F], got {features.shape}.")
        if actions.ndim != 2:
            raise ValueError(f"actions must have shape [T, A], got {actions.shape}.")
        if features.shape[0] != actions.shape[0]:
            raise ValueError("features and actions must have the same time dimension.")
        if not np.isfinite(features).all() or not np.isfinite(actions).all():
            raise ValueError("features and actions must be finite.")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "actions", actions)


@dataclass(frozen=True)
class WindowDataset:
    """Official Kinetix-style observation/action-chunk training examples."""

    features: Array
    action_chunks: Array
    episode_index: Array
    start_index: Array

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float32)
        action_chunks = np.asarray(self.action_chunks, dtype=np.float32)
        episode_index = np.asarray(self.episode_index, dtype=np.int32)
        start_index = np.asarray(self.start_index, dtype=np.int32)
        size = features.shape[0]
        if features.ndim != 2 or action_chunks.ndim != 3:
            raise ValueError("features/action_chunks must have shapes [N,F]/[N,H,A].")
        if any(x.shape != (size,) for x in (episode_index, start_index)):
            raise ValueError("window metadata must have shape [N].")
        if action_chunks.shape[0] != size:
            raise ValueError("features and action_chunks must have the same N.")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "action_chunks", action_chunks)
        object.__setattr__(self, "episode_index", episode_index)
        object.__setattr__(self, "start_index", start_index)

    def __len__(self) -> int:
        return self.features.shape[0]

    def batches(
        self,
        batch_size: int,
        *,
        seed: int,
        shuffle: bool = True,
        drop_remainder: bool = True,
    ) -> Iterator[tuple[Array, Array]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        indices = np.arange(len(self))
        if shuffle:
            np.random.default_rng(seed).shuffle(indices)
        stop = len(indices) if not drop_remainder else len(indices) // batch_size * batch_size
        for offset in range(0, stop, batch_size):
            batch_indices = indices[offset : offset + batch_size]
            if len(batch_indices) < batch_size and drop_remainder:
                break
            yield self.features[batch_indices], self.action_chunks[batch_indices]


def featurize_bigym_episodes(
    episodes: Iterable[Any],
    feature_fn: EpisodeFeatureFn,
    *,
    action_transform: MinMaxActionTransform | None = None,
) -> list[FeatureEpisode]:
    """Bridge ``bigym_data.BigymEpisode`` objects to encoded feature episodes.

    ``feature_fn`` owns the visual encoder and may combine ``episode.rgb`` and
    ``episode.state``. This keeps the official policy independent of a specific
    image backbone while preserving the exact official flat-observation API.
    """
    result = []
    for index, episode in enumerate(episodes):
        features = feature_fn(episode)
        actions = np.asarray(episode.action, dtype=np.float32)
        if action_transform is not None:
            actions = action_transform.normalize(actions)
        episode_id = str(getattr(episode, "seed", index))
        result.append(FeatureEpisode(features, actions, episode_id))
    return result


def build_window_dataset(
    episodes: Sequence[FeatureEpisode],
    horizon: int,
    *,
    stride: int = 1,
) -> WindowDataset:
    """Build full action windows without padding or crossing episode boundaries."""
    if horizon <= 0 or stride <= 0:
        raise ValueError("horizon and stride must be positive.")

    features: list[Array] = []
    chunks: list[Array] = []
    episode_indices: list[int] = []
    start_indices: list[int] = []
    for episode_index, episode in enumerate(episodes):
        stop = episode.actions.shape[0] - horizon + 1
        for start in range(0, max(stop, 0), stride):
            features.append(episode.features[start])
            chunks.append(episode.actions[start : start + horizon])
            episode_indices.append(episode_index)
            start_indices.append(start)

    if not features:
        raise ValueError("No full action windows are available for the requested horizon.")
    return WindowDataset(
        np.stack(features),
        np.stack(chunks),
        np.asarray(episode_indices),
        np.asarray(start_indices),
    )


def save_window_dataset(path: str | Path, dataset: WindowDataset) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=dataset.features,
        action_chunks=dataset.action_chunks,
        episode_index=dataset.episode_index,
        start_index=dataset.start_index,
    )


def load_window_dataset(path: str | Path) -> WindowDataset:
    with np.load(Path(path), allow_pickle=False) as data:
        return WindowDataset(
            data["features"],
            data["action_chunks"],
            data["episode_index"],
            data["start_index"],
        )


__all__ = [
    "FeatureEpisode",
    "MinMaxActionTransform",
    "WindowDataset",
    "build_window_dataset",
    "featurize_bigym_episodes",
    "load_window_dataset",
    "save_window_dataset",
]
