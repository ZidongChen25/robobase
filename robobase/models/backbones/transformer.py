"""ChiTransformer-style action-sequence backbone."""

from __future__ import annotations

import jax.numpy as jnp
import flax.linen as nn

from robobase.models.backbones.common import ensure_time_batch, mish


def _normal_init():
    return nn.initializers.normal(stddev=0.02)


def _causal_mask(length: int) -> jnp.ndarray:
    return jnp.tril(jnp.ones((1, 1, length, length), dtype=bool))


def _memory_mask(action_length: int, memory_length: int) -> jnp.ndarray:
    target_index = jnp.arange(action_length)[:, None]
    memory_index = jnp.arange(memory_length)[None, :]
    return (target_index >= (memory_index - 1))[None, None, :, :]


class _CleanDiffuserPositionalEmbedding(nn.Module):
    dim: int
    max_positions: int = 10000

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        half_dim = self.dim // 2
        if half_dim == 0:
            return jnp.zeros((x.shape[0], 0), dtype=jnp.float32)
        freqs = jnp.arange(half_dim, dtype=jnp.float32) / half_dim
        freqs = (1.0 / float(self.max_positions)) ** freqs
        emb = x.astype(jnp.float32)[:, None] * freqs[None, :]
        emb = jnp.concatenate([jnp.cos(emb), jnp.sin(emb)], axis=-1)
        if emb.shape[-1] < self.dim:
            emb = jnp.pad(emb, ((0, 0), (0, self.dim - emb.shape[-1])))
        return emb


class _TransformerMLP(nn.Module):
    d_model: int
    dropout: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        x = nn.Dense(
            4 * self.d_model,
            kernel_init=_normal_init(),
            bias_init=nn.initializers.zeros,
        )(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)
        x = nn.Dense(
            self.d_model,
            kernel_init=_normal_init(),
            bias_init=nn.initializers.zeros,
        )(x)
        return nn.Dropout(rate=self.dropout)(x, deterministic=not train)


class _EncoderBlock(nn.Module):
    d_model: int
    n_heads: int
    dropout: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        attn_in = nn.LayerNorm()(x)
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            dropout_rate=self.dropout,
            kernel_init=_normal_init(),
            bias_init=nn.initializers.zeros,
            deterministic=not train,
        )(attn_in, attn_in)
        x = x + attn
        x = x + _TransformerMLP(self.d_model, self.dropout)(
            nn.LayerNorm()(x),
            train=train,
        )
        return x


class _DecoderBlock(nn.Module):
    d_model: int
    n_heads: int
    dropout: float

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        memory: jnp.ndarray,
        *,
        self_mask: jnp.ndarray,
        memory_mask: jnp.ndarray,
        train: bool,
    ) -> jnp.ndarray:
        self_attn_in = nn.LayerNorm()(x)
        self_attn = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            dropout_rate=self.dropout,
            kernel_init=_normal_init(),
            bias_init=nn.initializers.zeros,
            deterministic=not train,
        )(self_attn_in, self_attn_in, mask=self_mask)
        x = x + self_attn

        cross_q = nn.LayerNorm()(x)
        cross_attn = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            dropout_rate=self.dropout,
            kernel_init=_normal_init(),
            bias_init=nn.initializers.zeros,
            deterministic=not train,
        )(cross_q, memory, mask=memory_mask)
        x = x + cross_attn
        x = x + _TransformerMLP(self.d_model, self.dropout)(
            nn.LayerNorm()(x),
            train=train,
        )
        return x


class JaxChiTransformerBackbone(nn.Module):
    """Flax adaptation of CleanDiffuser's ChiTransformer.

    This matches CleanDiffuser's condition layout: the diffusion timestep token
    followed by ``obs_steps`` observation tokens.
    """

    action_dim: int
    sequence_length: int
    condition_dim: int
    d_model: int = 256
    n_heads: int = 4
    num_layers: int = 8
    n_cond_layers: int = 0
    dropout: float = 0.0

    @nn.compact
    def __call__(
        self,
        actions: jnp.ndarray,
        timesteps: jnp.ndarray,
        condition: jnp.ndarray | None = None,
        *,
        train: bool = False,
    ) -> jnp.ndarray:
        batch_size = actions.shape[0]
        timesteps = ensure_time_batch(timesteps, batch_size)

        time_token = _CleanDiffuserPositionalEmbedding(
            self.d_model,
            name="time_embedding",
        )(
            timesteps
        )[:, None, :]
        if self.condition_dim > 0:
            if condition is None:
                condition_tokens = jnp.zeros(
                    (batch_size, 1, self.condition_dim), dtype=actions.dtype
                )
            elif condition.ndim == 2:
                condition_tokens = condition[:, None, :]
            else:
                condition_tokens = condition
            obs_tokens = nn.Dense(
                self.d_model,
                kernel_init=_normal_init(),
                bias_init=nn.initializers.zeros,
                name="obs_emb",
            )(condition_tokens.astype(jnp.float32))
            memory = jnp.concatenate([time_token, obs_tokens], axis=1)
        else:
            memory = time_token

        cond_pos = self.param(
            "cond_pos_emb",
            nn.initializers.normal(stddev=0.02),
            (1, memory.shape[1], self.d_model),
        )
        memory = memory + cond_pos[:, : memory.shape[1], :]
        memory = nn.Dropout(rate=0.0)(memory, deterministic=not train)
        memory = nn.Dense(
            4 * self.d_model,
            kernel_init=_normal_init(),
            bias_init=nn.initializers.zeros,
            name="cond_encoder_dense1",
        )(memory)
        memory = mish(memory)
        memory = nn.Dense(
            self.d_model,
            kernel_init=_normal_init(),
            bias_init=nn.initializers.zeros,
            name="cond_encoder_dense2",
        )(memory)
        for layer in range(self.n_cond_layers):
            memory = _EncoderBlock(
                self.d_model,
                self.n_heads,
                self.dropout,
                name=f"encoder_{layer}",
            )(memory, train=train)

        x = nn.Dense(
            self.d_model,
            kernel_init=_normal_init(),
            bias_init=nn.initializers.zeros,
            name="act_emb",
        )(actions.astype(jnp.float32))
        pos = self.param(
            "pos_emb",
            nn.initializers.normal(stddev=0.02),
            (1, self.sequence_length, self.d_model),
        )
        x = nn.Dropout(rate=0.0)(
            x + pos[:, : x.shape[1], :],
            deterministic=not train,
        )
        self_mask = _causal_mask(x.shape[1])
        memory_attn_mask = _memory_mask(x.shape[1], memory.shape[1])
        for layer in range(self.num_layers):
            x = _DecoderBlock(
                self.d_model,
                self.n_heads,
                self.dropout,
                name=f"decoder_{layer}",
            )(
                x,
                memory,
                self_mask=self_mask,
                memory_mask=memory_attn_mask,
                train=train,
            )

        x = nn.LayerNorm(name="ln_f")(x)
        return nn.Dense(
            self.action_dim,
            kernel_init=_normal_init(),
            bias_init=nn.initializers.zeros,
            name="head",
        )(x)

    @property
    def output_shape(self) -> tuple[int, int]:
        return (self.sequence_length, self.action_dim)
