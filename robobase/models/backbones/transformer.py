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
    operator_variant: str = "torch"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        half_dim = self.dim // 2
        if half_dim == 0:
            return jnp.zeros((x.shape[0], 0), dtype=jnp.float32)
        if self.operator_variant == "legacy":
            exponent_denominator = half_dim
        else:
            exponent_denominator = max(half_dim - 1, 1)
        freqs = jnp.arange(half_dim, dtype=jnp.float32) / exponent_denominator
        freqs = (1.0 / float(self.max_positions)) ** freqs
        emb = x.astype(jnp.float32)[:, None] * freqs[None, :]
        if self.operator_variant == "legacy":
            emb = jnp.concatenate([jnp.cos(emb), jnp.sin(emb)], axis=-1)
        else:
            emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        if emb.shape[-1] < self.dim:
            emb = jnp.pad(emb, ((0, 0), (0, self.dim - emb.shape[-1])))
        return emb


class _TransformerMLP(nn.Module):
    d_model: int
    dropout: float
    operator_variant: str

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        x = nn.Dense(
            4 * self.d_model,
            kernel_init=_normal_init(),
            bias_init=nn.initializers.zeros,
        )(x)
        x = nn.gelu(x, approximate=self.operator_variant == "legacy")
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
    operator_variant: str

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        epsilon = 1e-6 if self.operator_variant == "legacy" else 1e-5
        attn_in = nn.LayerNorm(epsilon=epsilon)(x)
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            dropout_rate=self.dropout,
            kernel_init=_normal_init(),
            bias_init=nn.initializers.zeros,
            deterministic=not train,
        )(attn_in, attn_in)
        if self.operator_variant == "torch":
            attn = nn.Dropout(
                rate=self.dropout,
                name="self_attn_output_dropout",
            )(attn, deterministic=not train)
        x = x + attn
        x = x + _TransformerMLP(
            self.d_model,
            self.dropout,
            self.operator_variant,
        )(
            nn.LayerNorm(epsilon=epsilon)(x),
            train=train,
        )
        return x


class _DecoderBlock(nn.Module):
    d_model: int
    n_heads: int
    dropout: float
    operator_variant: str

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
        epsilon = 1e-6 if self.operator_variant == "legacy" else 1e-5
        self_attn_in = nn.LayerNorm(epsilon=epsilon)(x)
        self_attn = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            dropout_rate=self.dropout,
            kernel_init=_normal_init(),
            bias_init=nn.initializers.zeros,
            deterministic=not train,
        )(self_attn_in, self_attn_in, mask=self_mask)
        if self.operator_variant == "torch":
            self_attn = nn.Dropout(
                rate=self.dropout,
                name="self_attn_output_dropout",
            )(self_attn, deterministic=not train)
        x = x + self_attn

        cross_q = nn.LayerNorm(epsilon=epsilon)(x)
        cross_attn = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            dropout_rate=self.dropout,
            kernel_init=_normal_init(),
            bias_init=nn.initializers.zeros,
            deterministic=not train,
        )(cross_q, memory, mask=memory_mask)
        if self.operator_variant == "torch":
            cross_attn = nn.Dropout(
                rate=self.dropout,
                name="cross_attn_output_dropout",
            )(cross_attn, deterministic=not train)
        x = x + cross_attn
        x = x + _TransformerMLP(
            self.d_model,
            self.dropout,
            self.operator_variant,
        )(
            nn.LayerNorm(epsilon=epsilon)(x),
            train=train,
        )
        return x


class JaxChiTransformerBackbone(nn.Module):
    """Flax ChiTransformer with explicit legacy and Torch RoboBase semantics."""

    action_dim: int
    sequence_length: int
    condition_dim: int
    d_model: int = 256
    n_heads: int = 4
    num_layers: int = 8
    n_cond_layers: int = 0
    full_memory_attention: bool = False
    dropout: float = 0.0
    embedding_dropout: float | None = None
    operator_variant: str = "legacy"

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
        if self.operator_variant not in {"legacy", "torch"}:
            raise ValueError(
                "Transformer operator_variant must be 'legacy' or 'torch', got "
                f"'{self.operator_variant}'."
            )
        norm_epsilon = 1e-6 if self.operator_variant == "legacy" else 1e-5
        embedding_dropout = (
            self.dropout if self.embedding_dropout is None else self.embedding_dropout
        )
        timesteps = ensure_time_batch(timesteps, batch_size)

        time_token = _CleanDiffuserPositionalEmbedding(
            self.d_model,
            operator_variant=self.operator_variant,
            name="time_embedding",
        )(timesteps)[:, None, :]
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
        memory = nn.Dropout(
            rate=0.0 if self.operator_variant == "legacy" else embedding_dropout
        )(memory, deterministic=not train)
        if self.operator_variant == "legacy" or self.n_cond_layers == 0:
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
        if self.n_cond_layers > 0:
            for layer in range(self.n_cond_layers):
                memory = _EncoderBlock(
                    self.d_model,
                    self.n_heads,
                    self.dropout,
                    self.operator_variant,
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
        x = nn.Dropout(
            rate=0.0 if self.operator_variant == "legacy" else embedding_dropout
        )(
            x + pos[:, : x.shape[1], :],
            deterministic=not train,
        )
        self_mask = _causal_mask(x.shape[1])
        if self.full_memory_attention:
            memory_attn_mask = jnp.ones((1, 1, x.shape[1], memory.shape[1]), dtype=bool)
        else:
            memory_attn_mask = _memory_mask(x.shape[1], memory.shape[1])
        for layer in range(self.num_layers):
            x = _DecoderBlock(
                self.d_model,
                self.n_heads,
                self.dropout,
                self.operator_variant,
                name=f"decoder_{layer}",
            )(
                x,
                memory,
                self_mask=self_mask,
                memory_mask=memory_attn_mask,
                train=train,
            )

        x = nn.LayerNorm(epsilon=norm_epsilon, name="ln_f")(x)
        return nn.Dense(
            self.action_dim,
            kernel_init=_normal_init(),
            bias_init=nn.initializers.zeros,
            name="head",
        )(x)

    @property
    def output_shape(self) -> tuple[int, int]:
        return (self.sequence_length, self.action_dim)
