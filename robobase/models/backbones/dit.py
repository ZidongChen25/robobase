"""DiT-style 1-D action-sequence backbone."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from robobase.models.backbones.common import (
    SinusoidalPosEmb,
    condition_to_vector,
    ensure_time_batch,
    mish,
)


def _modulate(x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class _DiTMLP(nn.Module):
    hidden_size: int
    dropout: float

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_size * 4)(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=True)
        x = nn.Dense(self.hidden_size)(x)
        return nn.Dropout(rate=self.dropout)(x, deterministic=True)


class _DiTBlock(nn.Module):
    hidden_size: int
    n_heads: int
    dropout: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, emb: jnp.ndarray) -> jnp.ndarray:
        modulation = nn.Dense(
            self.hidden_size * 6,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="adaLN_modulation",
        )(nn.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            modulation,
            6,
            axis=-1,
        )

        attn_in = _modulate(
            nn.LayerNorm(use_scale=False, use_bias=False, epsilon=1e-6)(x),
            shift_msa,
            scale_msa,
        )
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            dropout_rate=self.dropout,
            deterministic=True,
        )(attn_in, attn_in)
        x = x + gate_msa[:, None, :] * attn

        mlp_in = _modulate(
            nn.LayerNorm(use_scale=False, use_bias=False, epsilon=1e-6)(x),
            shift_mlp,
            scale_mlp,
        )
        x = x + gate_mlp[:, None, :] * _DiTMLP(
            self.hidden_size,
            self.dropout,
        )(mlp_in)
        return x


class _FinalLayer1D(nn.Module):
    hidden_size: int
    out_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, emb: jnp.ndarray) -> jnp.ndarray:
        shift, scale = jnp.split(
            nn.Dense(
                2 * self.hidden_size,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
                name="adaLN_modulation",
            )(nn.silu(emb)),
            2,
            axis=-1,
        )
        x = _modulate(
            nn.LayerNorm(use_scale=False, use_bias=False, epsilon=1e-6)(x),
            shift,
            scale,
        )
        return nn.Dense(
            self.out_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="linear",
        )(x)


class JaxDiT1DBackbone(nn.Module):
    """Flax adaptation of CleanDiffuser's DiT1d with adaLN-Zero."""

    action_dim: int
    sequence_length: int
    condition_dim: int
    time_embed_dim: int = 256
    d_model: int = 384
    n_heads: int = 6
    depth: int = 12
    dropout: float = 0.0

    @nn.compact
    def __call__(
        self,
        actions: jnp.ndarray,
        timesteps: jnp.ndarray,
        condition: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        batch_size = actions.shape[0]
        timesteps = ensure_time_batch(timesteps, batch_size)

        x = nn.Dense(self.d_model, name="x_proj")(actions.astype(jnp.float32))
        pos = SinusoidalPosEmb(self.d_model, name="pos_emb")(
            jnp.arange(actions.shape[1], dtype=jnp.float32)
        )
        x = x + pos[None, :, :]

        emb = SinusoidalPosEmb(self.time_embed_dim, name="time_embedding")(
            timesteps
        )
        if self.condition_dim > 0:
            condition_vec = condition_to_vector(condition, batch_size=batch_size)
            if condition_vec is None:
                condition_vec = jnp.zeros(
                    (batch_size, self.condition_dim), dtype=actions.dtype
                )
            emb = emb + nn.Dense(
                self.time_embed_dim,
                name="condition_projection",
            )(condition_vec)
        emb = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="map_emb_dense1",
        )(emb)
        emb = mish(emb)
        emb = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="map_emb_dense2",
        )(emb)
        emb = mish(emb)

        for layer in range(self.depth):
            x = _DiTBlock(
                self.d_model,
                self.n_heads,
                self.dropout,
                name=f"block_{layer}",
            )(x, emb)
        return _FinalLayer1D(
            self.d_model,
            self.action_dim,
            name="final_layer",
        )(x, emb)

    @property
    def output_shape(self) -> tuple[int, int]:
        return (self.sequence_length, self.action_dim)
