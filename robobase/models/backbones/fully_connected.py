"""Fully connected action-sequence backbone."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from robobase.models.backbones.common import (
    SinusoidalPosEmb,
    condition_to_vector,
    ensure_time_batch,
)


class JaxFullyConnectedBackbone(nn.Module):
    """MLP baseline matching the CleanDiffuser MLP backbone contract."""

    action_dim: int
    sequence_length: int
    condition_dim: int
    time_embed_dim: int = 256
    hidden_dims: tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(
        self,
        actions: jnp.ndarray,
        timesteps: jnp.ndarray,
        condition: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        batch_size = actions.shape[0]
        timesteps = ensure_time_batch(timesteps, batch_size)

        t_emb = SinusoidalPosEmb(self.time_embed_dim, name="time_embedding")(
            timesteps
        )
        if self.condition_dim > 0:
            condition_vec = condition_to_vector(condition, batch_size=batch_size)
            if condition_vec is None:
                condition_vec = jnp.zeros(
                    (batch_size, self.condition_dim), dtype=actions.dtype
                )
            t_emb = t_emb + nn.Dense(
                self.time_embed_dim,
                kernel_init=nn.initializers.normal(stddev=0.02),
                bias_init=nn.initializers.zeros,
                name="condition_projection",
            )(condition_vec)

        x = jnp.concatenate([actions.reshape((batch_size, -1)), t_emb], axis=-1)
        for index, hidden_dim in enumerate(self.hidden_dims):
            x = nn.Dense(int(hidden_dim), name=f"mlp_{index}")(x)
            x = nn.relu(x)
        x = nn.Dense(
            self.sequence_length * self.action_dim,
            name="output",
        )(x)
        return x.reshape((batch_size, self.sequence_length, self.action_dim))

    @property
    def output_shape(self) -> tuple[int, int]:
        return (self.sequence_length, self.action_dim)
