"""Multi-camera feature fusion (Flax)."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp
import numpy as np


class JaxFusionMultiCamFeature(nn.Module):
    """Fuse multi-view features via flatten, average, or sum."""

    num_views: int
    feature_dim: int
    mode: str = "flatten"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Args: x — shape (batch, num_views, feature_dim)."""
        if self.mode == "flatten":
            return x.reshape((x.shape[0], -1))
        if self.mode == "average":
            return x.mean(axis=1)
        if self.mode == "sum":
            return x.sum(axis=1)
        raise ValueError(f"Mode {self.mode} is not supported.")

    @property
    def output_shape(self) -> tuple[int, ...]:
        if self.mode == "flatten":
            return (self.num_views * self.feature_dim,)
        return (self.feature_dim,)

    @staticmethod
    def from_input_shape(input_shape: tuple[int, int], mode: str = "flatten"):
        """Create from legacy (num_views, feature_dim) input_shape tuple."""
        return JaxFusionMultiCamFeature(
            num_views=int(input_shape[0]),
            feature_dim=int(input_shape[1]),
            mode=mode,
        )
