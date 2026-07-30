"""Reference-style ACT modules implemented with Flax/JAX."""

from __future__ import annotations

import math

import flax.linen as nn
import jax
import jax.numpy as jnp


_DETR_KERNEL_INIT = nn.initializers.xavier_uniform()
_PYTORCH_DEFAULT_KERNEL_INIT = nn.initializers.variance_scaling(
    1.0 / 3.0,
    "fan_in",
    "uniform",
)
_PYTORCH_MHA_QKV_KERNEL_INIT = nn.initializers.variance_scaling(
    0.5,
    "fan_avg",
    "uniform",
)
_ZERO_BIAS_INIT = nn.initializers.zeros


def _pytorch_default_bias_init(fan_in: int):
    return nn.initializers.uniform(1.0 / math.sqrt(max(1, int(fan_in))))


def _dense_with_pytorch_bias(
    x: jnp.ndarray,
    features: int,
    *,
    kernel_init,
    name: str | None = None,
) -> jnp.ndarray:
    return nn.Dense(
        features,
        kernel_init=kernel_init,
        bias_init=_pytorch_default_bias_init(int(x.shape[-1])),
        name=name,
    )(x)


def _sinusoid_encoding_table(n_position: int, d_hid: int) -> jnp.ndarray:
    positions = jnp.arange(n_position, dtype=jnp.float32)[:, None]
    dims = jnp.arange(d_hid, dtype=jnp.float32)[None, :]
    angles = positions / (10000 ** (2 * jnp.floor(dims / 2) / d_hid))
    return jnp.where((dims.astype(jnp.int32) % 2) == 0, jnp.sin(angles), jnp.cos(angles))


def _build_2d_sincos_pos_embed(hidden_dim: int, height: int, width: int) -> jnp.ndarray:
    if hidden_dim % 2 != 0:
        return jnp.zeros((height, width, hidden_dim), dtype=jnp.float32)
    y = jnp.arange(1, height + 1, dtype=jnp.float32)[:, None]
    x = jnp.arange(1, width + 1, dtype=jnp.float32)[None, :]
    y = y / (float(height) + 1e-6) * (2.0 * math.pi)
    x = x / (float(width) + 1e-6) * (2.0 * math.pi)

    num_pos_feats = hidden_dim // 2
    dim_t = 10000 ** (2 * jnp.floor(jnp.arange(num_pos_feats) / 2) / num_pos_feats)
    pos_x = x[:, :, None] / dim_t
    pos_y = y[:, :, None] / dim_t
    pos_x = jnp.stack((jnp.sin(pos_x[:, :, 0::2]), jnp.cos(pos_x[:, :, 1::2])), axis=3)
    pos_y = jnp.stack((jnp.sin(pos_y[:, :, 0::2]), jnp.cos(pos_y[:, :, 1::2])), axis=3)
    pos_x = pos_x.reshape((1, width, num_pos_feats)).repeat(height, axis=0)
    pos_y = pos_y.reshape((height, 1, num_pos_feats)).repeat(width, axis=1)
    return jnp.concatenate([pos_y, pos_x], axis=-1).astype(jnp.float32)


def _key_padding_mask(mask: jnp.ndarray | None, query_len: int):
    if mask is None:
        return None
    return jnp.logical_not(mask)[:, None, None, :].repeat(query_len, axis=2)


class ACTImageProjection(nn.Module):
    """Project ResNet spatial maps to hidden_dim and concatenate camera views."""

    hidden_dim: int

    @nn.compact
    def __call__(self, features: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # features: (B, V, H, W, C)
        batch_size, num_views, height, width, channels = features.shape
        x = features.reshape((batch_size * num_views, height, width, channels))
        x = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            kernel_init=_PYTORCH_DEFAULT_KERNEL_INIT,
            bias_init=_pytorch_default_bias_init(int(channels)),
            name="input_proj",
        )(x)
        x = x.reshape((batch_size, num_views, height, width, self.hidden_dim))
        x = jnp.concatenate([x[:, view_idx] for view_idx in range(num_views)], axis=2)

        pos_single = _build_2d_sincos_pos_embed(self.hidden_dim, height, width)
        pos = jnp.concatenate([pos_single for _ in range(num_views)], axis=1)
        pos = jnp.broadcast_to(pos[None], x.shape)
        return x, pos


class _ACTMLP(nn.Module):
    hidden_dim: int
    dim_feedforward: int
    dropout: float
    kernel_init: object = _DETR_KERNEL_INIT

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, deterministic: bool) -> jnp.ndarray:
        x = _dense_with_pytorch_bias(
            x,
            self.dim_feedforward,
            kernel_init=self.kernel_init,
        )
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        x = _dense_with_pytorch_bias(
            x,
            self.hidden_dim,
            kernel_init=self.kernel_init,
        )
        return nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)


class ACTTransformerEncoderLayer(nn.Module):
    hidden_dim: int
    nheads: int
    dim_feedforward: int
    dropout: float
    pre_norm: bool
    mlp_kernel_init: object = _DETR_KERNEL_INIT
    attention_out_kernel_init: object = _DETR_KERNEL_INIT

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        key_padding_mask: jnp.ndarray | None = None,
        pos: jnp.ndarray | None = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        pos_x = x if pos is None else x + pos
        mask = _key_padding_mask(key_padding_mask, x.shape[1])
        if self.pre_norm:
            normed = nn.LayerNorm(name="norm1")(x)
            normed_pos = normed if pos is None else normed + pos
            attn = nn.MultiHeadDotProductAttention(
                num_heads=self.nheads,
                dropout_rate=self.dropout,
                kernel_init=_PYTORCH_MHA_QKV_KERNEL_INIT,
                out_kernel_init=self.attention_out_kernel_init,
                bias_init=_ZERO_BIAS_INIT,
                out_bias_init=_ZERO_BIAS_INIT,
                name="self_attn",
            )(normed_pos, normed_pos, normed, mask=mask, deterministic=deterministic)
            x = x + nn.Dropout(rate=self.dropout)(attn, deterministic=deterministic)
            ff = _ACTMLP(
                self.hidden_dim,
                self.dim_feedforward,
                self.dropout,
                kernel_init=self.mlp_kernel_init,
                name="mlp",
            )(nn.LayerNorm(name="norm2")(x), deterministic=deterministic)
            return x + ff

        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.nheads,
            dropout_rate=self.dropout,
            kernel_init=_PYTORCH_MHA_QKV_KERNEL_INIT,
            out_kernel_init=self.attention_out_kernel_init,
            bias_init=_ZERO_BIAS_INIT,
            out_bias_init=_ZERO_BIAS_INIT,
            name="self_attn",
        )(pos_x, pos_x, x, mask=mask, deterministic=deterministic)
        x = nn.LayerNorm(name="norm1")(
            x + nn.Dropout(rate=self.dropout)(attn, deterministic=deterministic)
        )
        ff = _ACTMLP(
            self.hidden_dim,
            self.dim_feedforward,
            self.dropout,
            kernel_init=self.mlp_kernel_init,
            name="mlp",
        )(x, deterministic=deterministic)
        return nn.LayerNorm(name="norm2")(x + ff)


class ACTTransformerDecoderLayer(nn.Module):
    hidden_dim: int
    nheads: int
    dim_feedforward: int
    dropout: float
    pre_norm: bool

    @nn.compact
    def __call__(
        self,
        queries: jnp.ndarray,
        memory: jnp.ndarray,
        memory_key_padding_mask: jnp.ndarray | None = None,
        pos: jnp.ndarray | None = None,
        query_pos: jnp.ndarray | None = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        memory_mask = _key_padding_mask(memory_key_padding_mask, queries.shape[1])

        def with_query_pos(value):
            return value if query_pos is None else value + query_pos

        def with_pos(value):
            return value if pos is None else value + pos

        if self.pre_norm:
            normed = nn.LayerNorm(name="norm1")(queries)
            self_attn = nn.MultiHeadDotProductAttention(
                num_heads=self.nheads,
                dropout_rate=self.dropout,
                kernel_init=_PYTORCH_MHA_QKV_KERNEL_INIT,
                out_kernel_init=_DETR_KERNEL_INIT,
                bias_init=_ZERO_BIAS_INIT,
                out_bias_init=_ZERO_BIAS_INIT,
                name="self_attn",
            )(with_query_pos(normed), with_query_pos(normed), normed, deterministic=deterministic)
            queries = queries + nn.Dropout(rate=self.dropout)(
                self_attn,
                deterministic=deterministic,
            )

            normed = nn.LayerNorm(name="norm2")(queries)
            cross_attn = nn.MultiHeadDotProductAttention(
                num_heads=self.nheads,
                dropout_rate=self.dropout,
                kernel_init=_PYTORCH_MHA_QKV_KERNEL_INIT,
                out_kernel_init=_DETR_KERNEL_INIT,
                bias_init=_ZERO_BIAS_INIT,
                out_bias_init=_ZERO_BIAS_INIT,
                name="cross_attn",
            )(
                with_query_pos(normed),
                with_pos(memory),
                memory,
                mask=memory_mask,
                deterministic=deterministic,
            )
            queries = queries + nn.Dropout(rate=self.dropout)(
                cross_attn,
                deterministic=deterministic,
            )
            ff = _ACTMLP(
                self.hidden_dim,
                self.dim_feedforward,
                self.dropout,
                kernel_init=_DETR_KERNEL_INIT,
                name="mlp",
            )(nn.LayerNorm(name="norm3")(queries), deterministic=deterministic)
            return queries + ff

        self_attn = nn.MultiHeadDotProductAttention(
            num_heads=self.nheads,
            dropout_rate=self.dropout,
            kernel_init=_PYTORCH_MHA_QKV_KERNEL_INIT,
            out_kernel_init=_DETR_KERNEL_INIT,
            bias_init=_ZERO_BIAS_INIT,
            out_bias_init=_ZERO_BIAS_INIT,
            name="self_attn",
        )(
            with_query_pos(queries),
            with_query_pos(queries),
            queries,
            deterministic=deterministic,
        )
        queries = nn.LayerNorm(name="norm1")(
            queries + nn.Dropout(rate=self.dropout)(self_attn, deterministic=deterministic)
        )
        cross_attn = nn.MultiHeadDotProductAttention(
            num_heads=self.nheads,
            dropout_rate=self.dropout,
            kernel_init=_PYTORCH_MHA_QKV_KERNEL_INIT,
            out_kernel_init=_DETR_KERNEL_INIT,
            bias_init=_ZERO_BIAS_INIT,
            out_bias_init=_ZERO_BIAS_INIT,
            name="cross_attn",
        )(
            with_query_pos(queries),
            with_pos(memory),
            memory,
            mask=memory_mask,
            deterministic=deterministic,
        )
        queries = nn.LayerNorm(name="norm2")(
            queries + nn.Dropout(rate=self.dropout)(cross_attn, deterministic=deterministic)
        )
        ff = _ACTMLP(
            self.hidden_dim,
            self.dim_feedforward,
            self.dropout,
            kernel_init=_DETR_KERNEL_INIT,
            name="mlp",
        )(queries, deterministic=deterministic)
        return nn.LayerNorm(name="norm3")(queries + ff)


class ACTDetrTransformer(nn.Module):
    hidden_dim: int
    nheads: int
    enc_layers: int
    dec_layers: int
    dim_feedforward: int
    dropout: float
    pre_norm: bool

    @nn.compact
    def __call__(
        self,
        src: jnp.ndarray,
        query_embed: jnp.ndarray,
        pos_embed: jnp.ndarray,
        *,
        latent_input: jnp.ndarray | None = None,
        proprio_input: jnp.ndarray | None = None,
        additional_pos_embed: jnp.ndarray | None = None,
        task_emb: jnp.ndarray | None = None,
        deterministic: bool,
    ) -> jnp.ndarray:
        batch_size = src.shape[0]
        if src.ndim == 4:
            memory = src.reshape((batch_size, -1, src.shape[-1]))
            pos = pos_embed.reshape((batch_size, -1, pos_embed.shape[-1]))
            if latent_input is None or proprio_input is None or additional_pos_embed is None:
                raise ValueError("Image ACT transformer requires latent/proprio inputs.")
            additional = [latent_input, proprio_input]
            if task_emb is not None:
                additional.append(task_emb)
            additional_tokens = jnp.stack(additional, axis=1)
            additional_pos = additional_pos_embed[None, : len(additional), :]
            additional_pos = jnp.broadcast_to(additional_pos, additional_tokens.shape)
            memory = jnp.concatenate([additional_tokens, memory], axis=1)
            pos = jnp.concatenate([additional_pos, pos], axis=1)
        elif src.ndim == 3:
            memory = src
            pos = pos_embed
        else:
            raise ValueError(f"Unexpected transformer source shape {src.shape}.")

        encoder_layer = nn.remat(
            ACTTransformerEncoderLayer,
            prevent_cse=False,
            static_argnums=(4,),
        )
        for layer in range(self.enc_layers):
            memory = encoder_layer(
                self.hidden_dim,
                self.nheads,
                self.dim_feedforward,
                self.dropout,
                self.pre_norm,
                name=f"encoder_{layer}",
            )(memory, None, pos, deterministic)

        query_pos = jnp.broadcast_to(query_embed[None], (batch_size, *query_embed.shape))
        queries = jnp.zeros_like(query_pos)
        decoder_layer = nn.remat(
            ACTTransformerDecoderLayer,
            prevent_cse=False,
            static_argnums=(6,),
        )
        for layer in range(self.dec_layers):
            queries = decoder_layer(
                self.hidden_dim,
                self.nheads,
                self.dim_feedforward,
                self.dropout,
                self.pre_norm,
                name=f"decoder_{layer}",
            )(queries, memory, None, pos, query_pos, deterministic)
        return nn.LayerNorm(name="decoder_norm")(queries)


class JaxACTPolicy(nn.Module):
    """Mobile-GENIMA style ACT CVAE policy."""

    hidden_dim: int
    dropout: float
    nheads: int
    dim_feedforward: int
    enc_layers: int
    dec_layers: int
    pre_norm: bool
    state_dim: int
    action_dim: int
    num_queries: int
    latent_dim: int = 32
    use_lang_cond: bool = False

    @nn.compact
    def __call__(
        self,
        image_features: jnp.ndarray | None,
        image_pos: jnp.ndarray | None,
        qpos: jnp.ndarray,
        *,
        actions: jnp.ndarray | None = None,
        is_pad: jnp.ndarray | None = None,
        task_emb: jnp.ndarray | None = None,
        deterministic: bool = True,
        latent_key=None,
    ) -> tuple[jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None]:
        batch_size = qpos.shape[0]
        query_embed = self.param(
            "query_embed",
            nn.initializers.normal(stddev=1.0),
            (self.num_queries, self.hidden_dim),
        )
        cls_embed = self.param(
            "cls_embed",
            nn.initializers.normal(stddev=1.0),
            (1, self.hidden_dim),
        )
        additional_pos_embed = self.param(
            "additional_pos_embed",
            nn.initializers.normal(stddev=1.0),
            (3 if self.use_lang_cond else 2, self.hidden_dim),
        )

        if actions is not None:
            actions = actions[:, : self.num_queries]
            if is_pad is None:
                is_pad = jnp.zeros(actions.shape[:2], dtype=jnp.bool_)
            is_pad = is_pad[:, : self.num_queries].astype(jnp.bool_)
            action_embed = _dense_with_pytorch_bias(
                actions,
                self.hidden_dim,
                kernel_init=_PYTORCH_DEFAULT_KERNEL_INIT,
                name="encoder_action_proj",
            )
            qpos_embed = _dense_with_pytorch_bias(
                qpos,
                self.hidden_dim,
                kernel_init=_PYTORCH_DEFAULT_KERNEL_INIT,
                name="encoder_joint_proj",
            )[:, None]
            cls = jnp.broadcast_to(cls_embed[None], (batch_size, 1, self.hidden_dim))
            style_tokens = jnp.concatenate([cls, qpos_embed, action_embed], axis=1)
            style_is_pad = jnp.concatenate(
                [jnp.zeros((batch_size, 2), dtype=jnp.bool_), is_pad],
                axis=1,
            )
            pos_table = _sinusoid_encoding_table(1 + 1 + self.num_queries, self.hidden_dim)
            style_pos = jnp.broadcast_to(
                pos_table[None, : style_tokens.shape[1]],
                style_tokens.shape,
            )
            style_encoder_layer = nn.remat(
                ACTTransformerEncoderLayer,
                prevent_cse=False,
                static_argnums=(4,),
            )
            for layer in range(self.enc_layers):
                style_tokens = style_encoder_layer(
                    self.hidden_dim,
                    self.nheads,
                    self.dim_feedforward,
                    self.dropout,
                    self.pre_norm,
                    mlp_kernel_init=_PYTORCH_DEFAULT_KERNEL_INIT,
                    attention_out_kernel_init=_PYTORCH_DEFAULT_KERNEL_INIT,
                    name=f"style_encoder_{layer}",
                )(style_tokens, style_is_pad, style_pos, deterministic)
            latent_info = _dense_with_pytorch_bias(
                style_tokens[:, 0],
                self.latent_dim * 2,
                kernel_init=_PYTORCH_DEFAULT_KERNEL_INIT,
                name="latent_proj",
            )
            mu, logvar = jnp.split(latent_info, 2, axis=-1)
            std = jnp.exp(logvar / 2.0)
            if latent_key is None:
                latent_sample = mu
            else:
                latent_sample = mu + std * jax.random.normal(latent_key, std.shape)
            latent_input = _dense_with_pytorch_bias(
                latent_sample,
                self.hidden_dim,
                kernel_init=_PYTORCH_DEFAULT_KERNEL_INIT,
                name="latent_out_proj",
            )
        else:
            mu = logvar = None
            latent_sample = jnp.zeros((batch_size, self.latent_dim), dtype=jnp.float32)
            latent_input = _dense_with_pytorch_bias(
                latent_sample,
                self.hidden_dim,
                kernel_init=_PYTORCH_DEFAULT_KERNEL_INIT,
                name="latent_out_proj",
            )

        proprio_input = _dense_with_pytorch_bias(
            qpos,
            self.hidden_dim,
            kernel_init=_PYTORCH_DEFAULT_KERNEL_INIT,
            name="input_proj_robot_state_0",
        )
        proprio_input = nn.Dropout(rate=0.3, name="input_proj_robot_state_dropout")(
            proprio_input,
            deterministic=deterministic,
        )
        proprio_input = _dense_with_pytorch_bias(
            proprio_input,
            self.hidden_dim,
            kernel_init=_PYTORCH_DEFAULT_KERNEL_INIT,
            name="input_proj_robot_state_1",
        )

        task_input = None
        if task_emb is not None:
            if int(task_emb.shape[-1]) == int(self.hidden_dim):
                task_input = task_emb
            else:
                task_input = _dense_with_pytorch_bias(
                    task_emb,
                    self.hidden_dim,
                    kernel_init=_PYTORCH_DEFAULT_KERNEL_INIT,
                    name="task_proj",
                )

        if image_features is None:
            memory_tokens = [latent_input, proprio_input]
            if task_input is not None:
                memory_tokens.append(task_input)
            memory = jnp.stack(memory_tokens, axis=1)
            memory_pos = jnp.broadcast_to(
                additional_pos_embed[None, : len(memory_tokens)],
                memory.shape,
            )
            hs = ACTDetrTransformer(
                hidden_dim=self.hidden_dim,
                nheads=self.nheads,
                enc_layers=self.enc_layers,
                dec_layers=self.dec_layers,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                pre_norm=self.pre_norm,
                name="transformer",
            )(
                memory,
                query_embed,
                memory_pos,
                deterministic=deterministic,
            )
        else:
            hs = ACTDetrTransformer(
                hidden_dim=self.hidden_dim,
                nheads=self.nheads,
                enc_layers=self.enc_layers,
                dec_layers=self.dec_layers,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                pre_norm=self.pre_norm,
                name="transformer",
            )(
                image_features,
                query_embed,
                image_pos,
                latent_input=latent_input,
                proprio_input=proprio_input,
                additional_pos_embed=additional_pos_embed,
                task_emb=task_input,
                deterministic=deterministic,
            )

        action = _dense_with_pytorch_bias(
            hs,
            self.action_dim,
            kernel_init=_PYTORCH_DEFAULT_KERNEL_INIT,
            name="action_head",
        )
        return action, mu, logvar


class JaxACTTransformer(JaxACTPolicy):
    """Compatibility alias for older config/tests."""
