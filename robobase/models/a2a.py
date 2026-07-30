"""Action-latent components used by Action-to-Action Flow Matching."""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


def _structured_orthogonal(key, shape, dtype=jnp.float32):
    """QR-free semi-orthogonal initialization.

    A2A initializes several wide Conv1D kernels orthogonally. JAX's stock
    initializer performs device QR, which can fail while creating a cuSolver
    handle on recent GPUs. A signed, shifted DCT basis has the same unit singular
    values without a solver call and remains valid inside Flax's jitted init.
    """
    if len(shape) < 2:
        raise ValueError("Orthogonal initialization requires at least two axes.")
    rows = 1
    for dimension in shape[:-1]:
        rows *= int(dimension)
    columns = int(shape[-1])
    size = max(rows, columns)
    rank = min(rows, columns)
    frequency_key, position_key, sign_key = jax.random.split(key, 3)
    frequency_offset = jax.random.randint(frequency_key, (), minval=0, maxval=size)
    position_offset = jax.random.randint(position_key, (), minval=0, maxval=size)
    frequencies = (jnp.arange(rank) + frequency_offset) % size
    positions = (jnp.arange(size) + position_offset) % size
    normalization = jnp.where(
        frequencies == 0,
        jnp.sqrt(1.0 / size),
        jnp.sqrt(2.0 / size),
    )
    basis = jnp.cos(
        (jnp.pi / size)
        * (positions[:, None].astype(jnp.float32) + 0.5)
        * frequencies[None, :].astype(jnp.float32)
    )
    signs = jax.random.rademacher(sign_key, (rank,), dtype=jnp.float32)
    q = basis * normalization[None, :] * signs[None, :]
    if rows < columns:
        q = q.T
    return jnp.asarray(q.reshape(shape), dtype=dtype)


_ORTHOGONAL = _structured_orthogonal
_XAVIER = nn.initializers.xavier_uniform()
_ZERO = nn.initializers.zeros


class TemporalActionEncoder(nn.Module):
    """Three-stage temporal Conv1D encoder from the public A2A policy."""

    latent_dim: int = 512
    hidden_dim: int = 512
    num_layers: int = 3
    kernel_size: int = 5

    @nn.compact
    def __call__(
        self,
        actions: jnp.ndarray,
        padding_mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        if actions.ndim != 3:
            raise ValueError(
                "TemporalActionEncoder expects (batch, horizon, action_dim), "
                f"got {actions.shape}."
            )
        x = actions.astype(jnp.float32)
        if padding_mask is None:
            valid = jnp.ones(actions.shape[:2], dtype=jnp.bool_)
        else:
            if padding_mask.shape != actions.shape[:2]:
                raise ValueError(
                    "TemporalActionEncoder padding_mask must match the batch and "
                    f"horizon axes; got {padding_mask.shape} for {actions.shape}."
                )
            valid = jnp.logical_not(padding_mask)
        # Zeroing makes the latent invariant to padded values. The explicit
        # validity channel distinguishes padding from a legitimate zero action.
        x = jnp.where(valid[..., None], x, 0.0)
        x = jnp.concatenate([x, valid[..., None].astype(x.dtype)], axis=-1)
        for index in range(self.num_layers):
            x = nn.Conv(
                features=self.hidden_dim,
                kernel_size=(self.kernel_size,),
                strides=(2,),
                padding="SAME",
                kernel_init=_ORTHOGONAL,
                bias_init=_ZERO,
                name=f"conv_{index}",
            )(x)
            x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        return nn.Dense(
            self.latent_dim,
            kernel_init=_ORTHOGONAL,
            bias_init=_ZERO,
            name="latent_projection",
        )(x)


class _DecoderMLP(nn.Module):
    hidden_dim: int
    dropout: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=_XAVIER,
            bias_init=_ZERO,
            name="dense_in",
        )(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dropout(rate=self.dropout, name="dropout_in")(x, deterministic=not train)
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=_XAVIER,
            bias_init=_ZERO,
            name="dense_out",
        )(x)
        return nn.Dropout(rate=self.dropout, name="dropout_out")(
            x, deterministic=not train
        )


class ActionChunkDecoder(nn.Module):
    """Decode one action latent into a fixed-horizon normalized action chunk."""

    horizon: int
    action_dim: int
    latent_dim: int = 512
    hidden_dim: int = 512
    num_layers: int = 4
    dropout: float = 0.0

    @nn.compact
    def __call__(self, latent: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        if latent.ndim == 3:
            if latent.shape[1] != 1:
                raise ValueError(
                    "ActionChunkDecoder accepts a single latent token, got "
                    f"{latent.shape}."
                )
            latent = latent[:, 0]
        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError(
                "ActionChunkDecoder latent shape must be (batch, latent_dim), "
                f"got {latent.shape}."
            )
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=_XAVIER,
            bias_init=_ZERO,
            name="input_projection",
        )(latent.astype(jnp.float32))
        for index in range(self.num_layers):
            x = _DecoderMLP(
                self.hidden_dim,
                self.dropout,
                name=f"block_{index}",
            )(x, train=train)
        x = nn.Dense(
            self.horizon * self.action_dim,
            kernel_init=_XAVIER,
            bias_init=_ZERO,
            name="output_projection",
        )(x)
        return x.reshape((x.shape[0], self.horizon, self.action_dim))


__all__ = ["ActionChunkDecoder", "TemporalActionEncoder"]
