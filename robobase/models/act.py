"""Flax ACT-style transformer encoder/decoder."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class _ACTMLP(nn.Module):
    hidden_dim: int
    dim_feedforward: int
    dropout: float

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.dim_feedforward)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=True)
        x = nn.Dense(self.hidden_dim)(x)
        return nn.Dropout(rate=self.dropout)(x, deterministic=True)


class _ACTEncoderLayer(nn.Module):
    hidden_dim: int
    nheads: int
    dim_feedforward: int
    dropout: float
    pre_norm: bool

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.pre_norm:
            attn_in = nn.LayerNorm(name="norm1")(x)
            attn = nn.MultiHeadDotProductAttention(
                num_heads=self.nheads,
                dropout_rate=self.dropout,
                deterministic=True,
                name="self_attn",
            )(attn_in, attn_in)
            x = x + attn
            x = x + _ACTMLP(
                self.hidden_dim,
                self.dim_feedforward,
                self.dropout,
                name="mlp",
            )(nn.LayerNorm(name="norm2")(x))
            return x

        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.nheads,
            dropout_rate=self.dropout,
            deterministic=True,
            name="self_attn",
        )(x, x)
        x = nn.LayerNorm(name="norm1")(x + attn)
        x = nn.LayerNorm(
            name="norm2",
        )(x + _ACTMLP(self.hidden_dim, self.dim_feedforward, self.dropout)(x))
        return x


class _ACTDecoderLayer(nn.Module):
    hidden_dim: int
    nheads: int
    dim_feedforward: int
    dropout: float
    pre_norm: bool

    @nn.compact
    def __call__(self, queries: jnp.ndarray, memory: jnp.ndarray) -> jnp.ndarray:
        if self.pre_norm:
            self_attn_in = nn.LayerNorm(name="norm1")(queries)
            self_attn = nn.MultiHeadDotProductAttention(
                num_heads=self.nheads,
                dropout_rate=self.dropout,
                deterministic=True,
                name="self_attn",
            )(self_attn_in, self_attn_in)
            queries = queries + self_attn

            cross_q = nn.LayerNorm(name="norm2")(queries)
            cross_attn = nn.MultiHeadDotProductAttention(
                num_heads=self.nheads,
                dropout_rate=self.dropout,
                deterministic=True,
                name="cross_attn",
            )(cross_q, memory)
            queries = queries + cross_attn
            queries = queries + _ACTMLP(
                self.hidden_dim,
                self.dim_feedforward,
                self.dropout,
                name="mlp",
            )(nn.LayerNorm(name="norm3")(queries))
            return queries

        self_attn = nn.MultiHeadDotProductAttention(
            num_heads=self.nheads,
            dropout_rate=self.dropout,
            deterministic=True,
            name="self_attn",
        )(queries, queries)
        queries = nn.LayerNorm(name="norm1")(queries + self_attn)

        cross_attn = nn.MultiHeadDotProductAttention(
            num_heads=self.nheads,
            dropout_rate=self.dropout,
            deterministic=True,
            name="cross_attn",
        )(queries, memory)
        queries = nn.LayerNorm(name="norm2")(queries + cross_attn)
        queries = nn.LayerNorm(
            name="norm3",
        )(
            queries
            + _ACTMLP(
                self.hidden_dim,
                self.dim_feedforward,
                self.dropout,
                name="mlp",
            )(queries)
        )
        return queries


class JaxACTTransformer(nn.Module):
    """ACT-style chunked-action transformer.

    This branch already centralizes low-dimensional and pixel observations into
    feature vectors. The model treats that feature vector as one memory token and
    decodes one learned query per action in the chunk.
    """

    action_sequence: int
    action_dim: int
    hidden_dim: int = 512
    enc_layers: int = 4
    dec_layers: int = 1
    dim_feedforward: int = 3200
    dropout: float = 0.1
    nheads: int = 8
    pre_norm: bool = False

    @nn.compact
    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        x = features.astype(jnp.float32)
        if x.ndim > 2:
            x = x.reshape((x.shape[0], -1))

        memory = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            name="input_projection",
        )(x)[:, None, :]
        memory_pos = self.param(
            "memory_pos",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.hidden_dim),
        )
        memory = memory + memory_pos

        for layer in range(self.enc_layers):
            memory = _ACTEncoderLayer(
                self.hidden_dim,
                self.nheads,
                self.dim_feedforward,
                self.dropout,
                self.pre_norm,
                name=f"encoder_{layer}",
            )(memory)

        query_embed = self.param(
            "query_embed",
            nn.initializers.normal(stddev=0.02),
            (1, self.action_sequence, self.hidden_dim),
        )
        queries = jnp.broadcast_to(
            query_embed,
            (x.shape[0], self.action_sequence, self.hidden_dim),
        )
        for layer in range(self.dec_layers):
            queries = _ACTDecoderLayer(
                self.hidden_dim,
                self.nheads,
                self.dim_feedforward,
                self.dropout,
                self.pre_norm,
                name=f"decoder_{layer}",
            )(queries, memory)

        queries = nn.LayerNorm(name="final_norm")(queries)
        return nn.Dense(self.action_dim, name="action_head")(queries)

    @property
    def output_shape(self) -> tuple[int, int]:
        return (self.action_sequence, self.action_dim)
