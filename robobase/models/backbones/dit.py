"""Pure-JAX DiT backbones for 1-D action sequences."""

from __future__ import annotations

import math

import flax.linen as nn
import jax
import jax.numpy as jnp

from robobase.models.backbones.common import (
    CleanDiffuserPosEmb,
    SinusoidalPosEmb,
    condition_to_vector,
    ensure_time_batch,
    mish,
)


_XAVIER_UNIFORM = nn.initializers.xavier_uniform()
_ZERO = nn.initializers.zeros
_TORCH_LINEAR_KERNEL = nn.initializers.variance_scaling(
    scale=1.0 / 3.0,
    mode="fan_in",
    distribution="uniform",
)


def _torch_linear_bias(fan_in: int) -> nn.initializers.Initializer:
    bound = 1.0 / math.sqrt(fan_in)

    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(
            key,
            shape,
            dtype,
            minval=-bound,
            maxval=bound,
        )

    return init


def _modulate(x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class _DiTMLP(nn.Module):
    """Legacy MLP kept parameter-compatible with the original JAX DiT."""

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
    """Legacy JAX block kept intact for existing checkpoints."""

    hidden_size: int
    n_heads: int
    dropout: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, emb: jnp.ndarray) -> jnp.ndarray:
        modulation = nn.Dense(
            self.hidden_size * 6,
            kernel_init=_ZERO,
            bias_init=_ZERO,
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


class _CleanMultiheadSelfAttention(nn.Module):
    """Torch MultiheadAttention layout with one fused QKV projection."""

    hidden_size: int
    n_heads: int
    dropout: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        head_dim = self.hidden_size // self.n_heads
        qkv = nn.Dense(
            3 * self.hidden_size,
            kernel_init=_XAVIER_UNIFORM,
            bias_init=_ZERO,
            name="in_proj",
        )(x)
        query, key, value = jnp.split(qkv, 3, axis=-1)
        batch_size, horizon, _ = query.shape
        query = query.reshape(batch_size, horizon, self.n_heads, head_dim)
        key = key.reshape(batch_size, horizon, self.n_heads, head_dim)
        value = value.reshape(batch_size, horizon, self.n_heads, head_dim)

        logits = jnp.einsum("bthd,bshd->bhts", query, key)
        logits = logits / math.sqrt(head_dim)
        weights = jax.nn.softmax(logits, axis=-1)
        weights = nn.Dropout(rate=self.dropout, name="attention_dropout")(
            weights,
            deterministic=not train,
        )
        attended = jnp.einsum("bhts,bshd->bthd", weights, value)
        attended = attended.reshape(batch_size, horizon, self.hidden_size)
        return nn.Dense(
            self.hidden_size,
            kernel_init=_XAVIER_UNIFORM,
            bias_init=_ZERO,
            name="out_proj",
        )(attended)


class _CleanDiTMLP(nn.Module):
    hidden_size: int
    dropout: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        x = nn.Dense(
            4 * self.hidden_size,
            kernel_init=_XAVIER_UNIFORM,
            bias_init=_ZERO,
            name="fc1",
        )(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dropout(rate=self.dropout, name="dropout")(
            x,
            deterministic=not train,
        )
        return nn.Dense(
            self.hidden_size,
            kernel_init=_XAVIER_UNIFORM,
            bias_init=_ZERO,
            name="fc2",
        )(x)


class _CleanDiTBlock(nn.Module):
    """CleanDiffuser DiTBlock, including its exact residual ordering."""

    hidden_size: int
    n_heads: int
    dropout: float

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        emb: jnp.ndarray,
        *,
        train: bool,
    ) -> jnp.ndarray:
        modulation = nn.Dense(
            6 * self.hidden_size,
            kernel_init=_ZERO,
            bias_init=_ZERO,
            name="adaLN_modulation",
        )(nn.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            modulation,
            6,
            axis=-1,
        )

        # CleanDiffuser assigns the modulated tensor back to x before adding
        # attention; this differs from the canonical DiT residual formulation.
        x = _modulate(
            nn.LayerNorm(
                use_scale=False,
                use_bias=False,
                epsilon=1e-6,
                name="norm1",
            )(x),
            shift_msa,
            scale_msa,
        )
        attention = _CleanMultiheadSelfAttention(
            self.hidden_size,
            self.n_heads,
            self.dropout,
            name="attention",
        )(x, train=train)
        x = x + gate_msa[:, None, :] * attention

        mlp_input = _modulate(
            nn.LayerNorm(
                use_scale=False,
                use_bias=False,
                epsilon=1e-6,
                name="norm2",
            )(x),
            shift_mlp,
            scale_mlp,
        )
        mlp_output = _CleanDiTMLP(
            self.hidden_size,
            self.dropout,
            name="mlp",
        )(mlp_input, train=train)
        return x + gate_mlp[:, None, :] * mlp_output


class _FinalLayer1D(nn.Module):
    hidden_size: int
    out_dim: int
    clean_diffuser: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, emb: jnp.ndarray) -> jnp.ndarray:
        shift, scale = jnp.split(
            nn.Dense(
                2 * self.hidden_size,
                kernel_init=_ZERO,
                bias_init=_ZERO,
                name="adaLN_modulation",
            )(nn.silu(emb)),
            2,
            axis=-1,
        )
        layer_norm_name = "norm_final" if self.clean_diffuser else None
        x = _modulate(
            nn.LayerNorm(
                use_scale=False,
                use_bias=False,
                epsilon=1e-6,
                name=layer_norm_name,
            )(x),
            shift,
            scale,
        )
        return nn.Dense(
            self.out_dim,
            kernel_init=_ZERO,
            bias_init=_ZERO,
            name="linear",
        )(x)


class _CleanDiffuserFourierEmbedding(nn.Module):
    """CleanDiffuser FourierEmbedding with frozen random frequencies."""

    dim: int
    scale: float = 16.0

    @nn.compact
    def __call__(self, timesteps: jnp.ndarray) -> jnp.ndarray:
        frequencies = self.param(
            "fourier_frequencies",
            nn.initializers.normal(stddev=self.scale),
            (self.dim // 8,),
        )
        frequencies = jax.lax.stop_gradient(frequencies)
        phase = timesteps.astype(jnp.float32)[:, None] * (
            2.0 * jnp.pi * frequencies[None, :]
        )
        embedding = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=-1)
        embedding = nn.Dense(
            self.dim,
            kernel_init=_XAVIER_UNIFORM,
            bias_init=_ZERO,
            name="dense1",
        )(embedding)
        embedding = mish(embedding)
        return nn.Dense(
            self.dim,
            kernel_init=_XAVIER_UNIFORM,
            bias_init=_ZERO,
            name="dense2",
        )(embedding)


class _CleanMLPCondition(nn.Module):
    """CleanDiffuser MLPCondition with sample-level classifier-free dropout."""

    in_dim: int
    out_dim: int
    hidden_dims: tuple[int, ...]
    dropout: float

    @nn.compact
    def __call__(self, condition: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        x = condition
        fan_in = self.in_dim
        for index, hidden_dim in enumerate(self.hidden_dims):
            x = nn.Dense(
                hidden_dim,
                kernel_init=_TORCH_LINEAR_KERNEL,
                bias_init=_torch_linear_bias(fan_in),
                name=f"hidden_{index}",
            )(x)
            x = nn.relu(x)
            fan_in = hidden_dim
        x = nn.Dense(
            self.out_dim,
            kernel_init=_TORCH_LINEAR_KERNEL,
            bias_init=_torch_linear_bias(fan_in),
            name="output",
        )(x)
        if train and self.dropout > 0.0:
            keep = jax.random.bernoulli(
                self.make_rng("dropout"),
                p=1.0 - self.dropout,
                shape=(x.shape[0], 1),
            )
            x = x * keep.astype(x.dtype)
        return x


class JaxDiT1DBackbone(nn.Module):
    """DiT1d with a legacy mode and an exact CleanDiffuser operator mode."""

    action_dim: int
    sequence_length: int
    condition_dim: int
    time_embed_dim: int = 256
    d_model: int = 384
    n_heads: int = 6
    depth: int = 12
    dropout: float = 0.0
    timestep_embedding_type: str = "campose"
    operator_variant: str = "legacy"
    condition_adapter: str = "linear"
    condition_hidden_dims: tuple[int, ...] = ()
    condition_dropout: float = 0.0
    fourier_scale: float = 16.0

    @nn.compact
    def __call__(
        self,
        actions: jnp.ndarray,
        timesteps: jnp.ndarray,
        condition: jnp.ndarray | None = None,
        *,
        train: bool = False,
    ) -> jnp.ndarray:
        if actions.ndim != 3:
            raise ValueError(
                "DiT actions must have shape (batch, horizon, action_dim), got "
                f"{actions.shape}."
            )
        if actions.shape[1] != self.sequence_length:
            raise ValueError(
                "DiT input horizon does not match sequence_length: "
                f"{actions.shape[1]} != {self.sequence_length}."
            )
        batch_size = actions.shape[0]
        timesteps = ensure_time_batch(timesteps, batch_size)
        clean_diffuser = self.operator_variant == "torch"

        dense_kwargs = (
            {"kernel_init": _XAVIER_UNIFORM, "bias_init": _ZERO}
            if clean_diffuser
            else {}
        )
        x = nn.Dense(self.d_model, name="x_proj", **dense_kwargs)(
            actions.astype(jnp.float32)
        )
        if clean_diffuser:
            # CleanDiffuser calls SinusoidalEmbedding with torch.arange's
            # integer dtype. Its implementation casts frequencies to that
            # dtype, so exact parity intentionally preserves this behavior.
            half_dim = self.d_model // 2
            frequency_scale = math.log(10000.0) / (half_dim - 1)
            frequencies = jnp.exp(
                jnp.arange(half_dim, dtype=jnp.float32) * -frequency_scale
            ).astype(jnp.int32)
            phase = (
                jnp.arange(actions.shape[1], dtype=jnp.int32)[:, None]
                * frequencies[None, :]
            )
            pos = jnp.concatenate(
                [jnp.sin(phase.astype(jnp.float32)), jnp.cos(phase)],
                axis=-1,
            )
        else:
            pos = SinusoidalPosEmb(self.d_model, name="pos_emb")(
                jnp.arange(actions.shape[1], dtype=jnp.float32)
            )
        x = x + pos[None, :, :]

        if self.timestep_embedding_type == "campose":
            emb = SinusoidalPosEmb(self.time_embed_dim, name="time_embedding")(
                timesteps
            )
        elif self.timestep_embedding_type == "clean_diffuser":
            emb = CleanDiffuserPosEmb(
                self.time_embed_dim,
                name="time_embedding",
            )(timesteps)
        elif self.timestep_embedding_type == "fourier":
            emb = _CleanDiffuserFourierEmbedding(
                self.time_embed_dim,
                scale=self.fourier_scale,
                name="time_embedding",
            )(timesteps)
        else:
            raise ValueError(
                "DiT timestep_embedding_type must be 'campose', "
                f"'clean_diffuser', or 'fourier', got {self.timestep_embedding_type!r}."
            )

        if self.condition_dim > 0 and not (
            clean_diffuser and condition is None
        ):
            condition_vec = condition_to_vector(condition, batch_size=batch_size)
            if condition_vec is None:
                condition_vec = jnp.zeros(
                    (batch_size, self.condition_dim), dtype=jnp.float32
                )
            if self.condition_adapter == "direct":
                condition_emb = condition_vec
            elif self.condition_adapter == "linear":
                condition_kwargs = {}
                if clean_diffuser:
                    condition_kwargs = {
                        "kernel_init": _TORCH_LINEAR_KERNEL,
                        "bias_init": _torch_linear_bias(self.condition_dim),
                    }
                condition_emb = nn.Dense(
                    self.time_embed_dim,
                    name="condition_projection",
                    **condition_kwargs,
                )(condition_vec)
            elif self.condition_adapter == "clean_mlp":
                condition_emb = _CleanMLPCondition(
                    self.condition_dim,
                    self.time_embed_dim,
                    self.condition_hidden_dims,
                    self.condition_dropout,
                    name="condition_projection",
                )(condition_vec, train=train)
            else:
                raise ValueError(
                    "DiT condition_adapter must be 'direct', 'linear', or "
                    f"'clean_mlp', got {self.condition_adapter!r}."
                )
            if condition_emb.shape[-1] != self.time_embed_dim:
                raise ValueError(
                    "DiT direct condition dimension must equal time_embed_dim: "
                    f"{condition_emb.shape[-1]} != {self.time_embed_dim}."
                )
            emb = emb + condition_emb

        map_dense_kwargs = {
            "kernel_init": nn.initializers.normal(stddev=0.02),
            "bias_init": _ZERO,
        }
        emb = nn.Dense(
            self.d_model,
            name="map_emb_dense1",
            **map_dense_kwargs,
        )(emb)
        emb = mish(emb)
        emb = nn.Dense(
            self.d_model,
            name="map_emb_dense2",
            **map_dense_kwargs,
        )(emb)
        emb = mish(emb)

        for layer in range(self.depth):
            if clean_diffuser:
                x = _CleanDiTBlock(
                    self.d_model,
                    self.n_heads,
                    self.dropout,
                    name=f"block_{layer}",
                )(x, emb, train=train)
            else:
                x = _DiTBlock(
                    self.d_model,
                    self.n_heads,
                    self.dropout,
                    name=f"block_{layer}",
                )(x, emb)
        return _FinalLayer1D(
            self.d_model,
            self.action_dim,
            clean_diffuser=clean_diffuser,
            name="final_layer",
        )(x, emb)

    @property
    def output_shape(self) -> tuple[int, int]:
        return (self.sequence_length, self.action_dim)
