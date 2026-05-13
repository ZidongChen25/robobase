"""Shared Flax building blocks for action-sequence backbones."""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


def mish(x: jnp.ndarray) -> jnp.ndarray:
    return x * jnp.tanh(jax.nn.softplus(x))


def valid_group_count(num_channels: int, requested_groups: int) -> int:
    groups = max(1, min(int(requested_groups), int(num_channels)))
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class SinusoidalPosEmb(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        half_dim = self.dim // 2
        if half_dim == 0:
            return jnp.zeros((x.shape[0], 0), dtype=jnp.float32)
        emb = jnp.log(10000.0) / max(half_dim - 1, 1)
        emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
        emb = x.astype(jnp.float32)[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        if emb.shape[-1] < self.dim:
            emb = jnp.pad(emb, ((0, 0), (0, self.dim - emb.shape[-1])))
        return emb


class DenseMish(nn.Module):
    features: int
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(
            self.features,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        return mish(x)


def ensure_time_batch(time: jnp.ndarray, batch_size: int) -> jnp.ndarray:
    time = time.astype(jnp.float32)
    if time.ndim == 0:
        return jnp.broadcast_to(time[None], (batch_size,))
    if time.shape[0] == 1 and batch_size != 1:
        return jnp.broadcast_to(time, (batch_size,))
    return time


def condition_to_vector(
    condition: jnp.ndarray | None,
    *,
    batch_size: int,
) -> jnp.ndarray | None:
    if condition is None:
        return None
    condition = condition.astype(jnp.float32)
    if condition.ndim == 2:
        return condition
    return condition.reshape((batch_size, -1))
