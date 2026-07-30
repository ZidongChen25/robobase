"""Visual feature boundary between RoboBase BiGym and official Legato core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np


ObservationBatch = Mapping[str, Any]
EpisodeBatchBuilder = Callable[[Any, slice], dict[str, np.ndarray]]


@dataclass(frozen=True)
class FrozenFMVisualFeatures:
    """Reuse a loaded FM agent's JAX observation encoder as a frozen boundary.

    The caller constructs and restores the FM agent through the existing
    RoboBase workspace. Both the official vanilla/RTC model and the official
    Legato model must receive outputs from this same object for a fair test.
    """

    agent: Any
    use_ema: bool = True
    require_pixels: bool = True

    def __post_init__(self) -> None:
        if self.require_pixels and not bool(getattr(self.agent, "use_pixels", False)):
            raise ValueError(
                "Official BiGym evaluation requires visual features; a proprio-only "
                "Kinetix observation is not a valid FlipCutlery adapter."
            )
        if getattr(self.agent, "_condition_as_local", False):
            raise ValueError(
                "The official flat-observation core cannot consume local UNet "
                "conditioning. Use a global FM encoder/checkpoint."
            )

    def encode(self, observations: ObservationBatch) -> jax.Array:
        """Encode a normal RoboBase observation batch to ``[B, F]``."""
        agent = self.agent
        if getattr(agent, "_trainable_encoder", False):
            inputs = agent._prepare_trainable_obs_inputs(dict(observations))
        else:
            inputs, _ = agent._prepare_obs_features(dict(observations))
        params = agent.ema_params if self.use_ema else agent.params
        features = agent._features_from_inputs(params, inputs)
        if isinstance(features, tuple):
            raise ValueError("Official Legato requires one global feature tensor.")
        features = jax.lax.stop_gradient(jnp.asarray(features, dtype=jnp.float32))
        if features.ndim < 2:
            raise ValueError(f"Encoded features must retain a batch axis: {features.shape}.")
        return features.reshape(features.shape[0], -1)

    def encode_numpy(self, observations: ObservationBatch) -> np.ndarray:
        return np.asarray(jax.device_get(self.encode(observations)), dtype=np.float32)


def encode_bigym_episode(
    episode: Any,
    feature_boundary: FrozenFMVisualFeatures,
    batch_builder: EpisodeBatchBuilder,
    *,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode one cache episode without retaining all raw images on device."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    length = int(episode.action.shape[0])
    chunks = []
    for start in range(0, length, batch_size):
        selection = slice(start, min(start + batch_size, length))
        observations = batch_builder(episode, selection)
        chunk = feature_boundary.encode_numpy(observations)
        if chunk.shape[0] != selection.stop - selection.start:
            raise ValueError("Feature batch length does not match the episode slice.")
        chunks.append(chunk)
    return np.concatenate(chunks, axis=0)


__all__ = ["FrozenFMVisualFeatures", "encode_bigym_episode"]
