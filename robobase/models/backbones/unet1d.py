"""Conditional 1-D U-Net backbone for action-sequence prediction."""

from __future__ import annotations

import math

import flax.linen as nn
import jax.numpy as jnp

from robobase.models.backbones.common import (
    CleanDiffuserPosEmb,
    SinusoidalPosEmb,
    ensure_time_batch,
    mish,
    valid_group_count,
)


_PYTORCH_DEFAULT_KERNEL_INIT = nn.initializers.variance_scaling(
    1.0 / 3.0,
    "fan_in",
    "uniform",
)


def _pytorch_default_bias_init(fan_in: int):
    return nn.initializers.uniform(1.0 / math.sqrt(max(int(fan_in), 1)))


class Conv1dBlock(nn.Module):
    """Conv1d + GroupNorm + Mish activation."""

    out_channels: int
    kernel_size: int
    n_groups: int
    operator_variant: str = "legacy"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        n_groups = valid_group_count(self.out_channels, self.n_groups)
        conv_kwargs = {"name": "conv"}
        norm_kwargs = {"name": "norm"}
        if self.operator_variant == "torch":
            conv_kwargs.update(
                kernel_init=_PYTORCH_DEFAULT_KERNEL_INIT,
                bias_init=_pytorch_default_bias_init(
                    self.kernel_size * int(x.shape[-1])
                ),
            )
            norm_kwargs["epsilon"] = 1e-5
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(self.kernel_size,),
            padding="SAME",
            **conv_kwargs,
        )(x)
        x = nn.GroupNorm(num_groups=n_groups, **norm_kwargs)(x)
        return mish(x)


class ResidualBlock(nn.Module):
    """Two Conv1dBlocks with conditioning injection and residual connection."""

    out_channels: int
    cond_dim: int
    kernel_size: int
    n_groups: int
    cond_predict_scale: bool = False
    operator_variant: str = "legacy"

    @nn.compact
    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
        in_channels = x.shape[-1]

        out = Conv1dBlock(
            self.out_channels,
            self.kernel_size,
            self.n_groups,
            self.operator_variant,
            name="block1",
        )(x)

        cond_channels = self.out_channels * (2 if self.cond_predict_scale else 1)
        dense_kwargs = {"name": "cond_dense"}
        if self.operator_variant == "torch":
            dense_kwargs.update(
                kernel_init=_PYTORCH_DEFAULT_KERNEL_INIT,
                bias_init=_pytorch_default_bias_init(int(cond.shape[-1])),
            )
        embed = nn.Dense(cond_channels, **dense_kwargs)(mish(cond))
        if self.cond_predict_scale:
            embed = embed.reshape((embed.shape[0], 2, self.out_channels))
            scale = embed[:, 0, :][:, None, :]
            bias = embed[:, 1, :][:, None, :]
            out = scale * out + bias
        else:
            out = out + embed[:, None, :]

        out = Conv1dBlock(
            self.out_channels,
            self.kernel_size,
            self.n_groups,
            self.operator_variant,
            name="block2",
        )(out)

        if in_channels != self.out_channels:
            residual_kwargs = {"name": "residual_dense"}
            if self.operator_variant == "torch":
                residual_kwargs.update(
                    kernel_init=_PYTORCH_DEFAULT_KERNEL_INIT,
                    bias_init=_pytorch_default_bias_init(int(in_channels)),
                )
            residual = nn.Dense(self.out_channels, **residual_kwargs)(x)
        else:
            residual = x
        return out + residual


class Downsample1d(nn.Module):
    channels: int
    operator_variant: str = "legacy"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        kwargs = {"name": "conv"}
        if self.operator_variant == "torch":
            kwargs.update(
                kernel_init=_PYTORCH_DEFAULT_KERNEL_INIT,
                bias_init=_pytorch_default_bias_init(3 * int(x.shape[-1])),
            )
        return nn.Conv(
            features=self.channels,
            kernel_size=(3,),
            strides=(2,),
            padding=((1, 1),) if self.operator_variant == "torch" else "SAME",
            **kwargs,
        )(x)


class Upsample1d(nn.Module):
    channels: int
    operator_variant: str = "legacy"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        kwargs = {"name": "conv_transpose"}
        if self.operator_variant == "torch":
            kwargs.update(
                transpose_kernel=True,
                kernel_init=_PYTORCH_DEFAULT_KERNEL_INIT,
                bias_init=_pytorch_default_bias_init(4 * self.channels),
            )
        return nn.ConvTranspose(
            features=self.channels,
            kernel_size=(4,),
            strides=(2,),
            padding="SAME",
            **kwargs,
        )(x)


class JaxConditionalUnet1D(nn.Module):
    """Conditional U-Net preserving the existing JAX diffusion-policy path."""

    action_dim: int
    sequence_length: int
    feature_dim: int
    diffusion_step_embed_dim: int
    down_dims: tuple[int, ...]
    input_action_dim: int | None = None
    kernel_size: int = 5
    n_groups: int = 8
    local_feature_dim: int = 0
    cond_predict_scale: bool = False
    global_condition_embed_dim: int = 0
    timestep_embedding_type: str = "campose"
    operator_variant: str = "legacy"

    @nn.compact
    def __call__(
        self,
        actions: jnp.ndarray,
        timesteps: jnp.ndarray,
        features: jnp.ndarray | None = None,
        local_features: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        sample = actions.astype(jnp.float32)
        if sample.ndim != 3:
            raise ValueError("actions must have shape (batch, sequence, action_dim).")
        input_action_dim = (
            self.action_dim
            if self.input_action_dim is None
            else int(self.input_action_dim)
        )
        if sample.shape[1:] != (self.sequence_length, input_action_dim):
            raise ValueError(
                "UNet action shape does not match its configured input shape: "
                f"{sample.shape[1:]} != {(self.sequence_length, input_action_dim)}."
            )
        timesteps = ensure_time_batch(timesteps, sample.shape[0])
        operator_variant = str(self.operator_variant).lower()
        if operator_variant not in {"legacy", "torch"}:
            raise ValueError(
                "UNet operator_variant must be 'legacy' or 'torch', got "
                f"{self.operator_variant!r}."
            )

        timestep_embedding_type = str(self.timestep_embedding_type).lower()
        if timestep_embedding_type == "campose":
            timestep_embedding = SinusoidalPosEmb
        elif timestep_embedding_type == "clean_diffuser":
            timestep_embedding = CleanDiffuserPosEmb
        else:
            raise ValueError(
                "UNet timestep_embedding_type must be 'campose' or "
                f"'clean_diffuser', got '{self.timestep_embedding_type}'."
            )
        t_emb = timestep_embedding(
            self.diffusion_step_embed_dim,
            name="pos_emb",
        )(timesteps)
        time_dense1_kwargs = {"name": "time_dense1"}
        if operator_variant == "torch":
            time_dense1_kwargs.update(
                kernel_init=_PYTORCH_DEFAULT_KERNEL_INIT,
                bias_init=_pytorch_default_bias_init(self.diffusion_step_embed_dim),
            )
        t_emb = nn.Dense(
            self.diffusion_step_embed_dim * 4,
            **time_dense1_kwargs,
        )(t_emb)
        t_emb = mish(t_emb)
        time_dense2_kwargs = {"name": "time_dense2"}
        if operator_variant == "torch":
            time_dense2_kwargs.update(
                kernel_init=_PYTORCH_DEFAULT_KERNEL_INIT,
                bias_init=_pytorch_default_bias_init(self.diffusion_step_embed_dim * 4),
            )
        global_feature = nn.Dense(
            self.diffusion_step_embed_dim,
            **time_dense2_kwargs,
        )(t_emb)

        if self.feature_dim > 0:
            if features is None:
                raise ValueError("Expected diffusion conditioning features.")
            if features.ndim != 2 or features.shape != (
                sample.shape[0],
                self.feature_dim,
            ):
                raise ValueError(
                    "Global conditioning features must have shape "
                    f"(batch, {self.feature_dim}), got {features.shape}."
                )
            global_feature = jnp.concatenate(
                [
                    global_feature,
                    nn.Dense(
                        self.global_condition_embed_dim,
                        **(
                            {
                                "kernel_init": _PYTORCH_DEFAULT_KERNEL_INIT,
                                "bias_init": _pytorch_default_bias_init(
                                    self.feature_dim
                                ),
                                "name": "global_cond_dense",
                            }
                            if operator_variant == "torch"
                            else {"name": "global_cond_dense"}
                        ),
                    )(features.astype(jnp.float32))
                    if self.global_condition_embed_dim > 0
                    else features.astype(jnp.float32),
                ],
                axis=-1,
            )

        encoded_feature_dim = (
            self.global_condition_embed_dim
            if self.feature_dim > 0 and self.global_condition_embed_dim > 0
            else max(self.feature_dim, 0)
        )
        cond_dim = self.diffusion_step_embed_dim + encoded_feature_dim
        all_dims = [input_action_dim] + [int(d) for d in self.down_dims]

        h_local = None
        if self.local_feature_dim > 0:
            if local_features is None:
                raise ValueError("Expected local diffusion conditioning features.")
            expected_local_shape = (
                sample.shape[0],
                self.sequence_length,
                self.local_feature_dim,
            )
            if local_features.shape != expected_local_shape:
                raise ValueError(
                    "Local conditioning features must have shape "
                    f"{expected_local_shape}, got {local_features.shape}."
                )
            local_features = local_features.astype(jnp.float32)
            local_dim = int(self.down_dims[0])
            h_local = (
                ResidualBlock(
                    local_dim,
                    cond_dim,
                    self.kernel_size,
                    self.n_groups,
                    self.cond_predict_scale,
                    operator_variant,
                    name="local_res1",
                )(local_features, global_feature),
                ResidualBlock(
                    local_dim,
                    cond_dim,
                    self.kernel_size,
                    self.n_groups,
                    self.cond_predict_scale,
                    operator_variant,
                    name="local_res2",
                )(local_features, global_feature),
            )

        x = sample
        hidden = []
        for i, (_, dim_out) in enumerate(zip(all_dims[:-1], all_dims[1:])):
            is_last = i >= len(all_dims) - 2
            x = ResidualBlock(
                dim_out,
                cond_dim,
                self.kernel_size,
                self.n_groups,
                self.cond_predict_scale,
                operator_variant,
                name=f"down_{i}_res1",
            )(x, global_feature)
            if i == 0 and h_local is not None:
                x = x + h_local[0]
            x = ResidualBlock(
                dim_out,
                cond_dim,
                self.kernel_size,
                self.n_groups,
                self.cond_predict_scale,
                operator_variant,
                name=f"down_{i}_res2",
            )(x, global_feature)
            hidden.append(x)
            if not is_last:
                x = Downsample1d(
                    dim_out,
                    operator_variant,
                    name=f"down_{i}_ds",
                )(x)

        mid_dim = all_dims[-1]
        x = ResidualBlock(
            mid_dim,
            cond_dim,
            self.kernel_size,
            self.n_groups,
            self.cond_predict_scale,
            operator_variant,
            name="mid_res1",
        )(x, global_feature)
        x = ResidualBlock(
            mid_dim,
            cond_dim,
            self.kernel_size,
            self.n_groups,
            self.cond_predict_scale,
            operator_variant,
            name="mid_res2",
        )(x, global_feature)

        reversed_pairs = list(reversed(list(zip(all_dims[:-1], all_dims[1:]))[1:]))
        for i, (dim_in, _) in enumerate(reversed_pairs):
            is_last = i >= len(all_dims) - 2
            x = jnp.concatenate([x, hidden.pop()], axis=-1)
            x = ResidualBlock(
                dim_in,
                cond_dim,
                self.kernel_size,
                self.n_groups,
                self.cond_predict_scale,
                operator_variant,
                name=f"up_{i}_res1",
            )(x, global_feature)
            x = ResidualBlock(
                dim_in,
                cond_dim,
                self.kernel_size,
                self.n_groups,
                self.cond_predict_scale,
                operator_variant,
                name=f"up_{i}_res2",
            )(x, global_feature)
            if not is_last:
                x = Upsample1d(
                    dim_in,
                    operator_variant,
                    name=f"up_{i}_us",
                )(x)
            if i == len(reversed_pairs) - 1 and h_local is not None:
                x = x + h_local[1]

        start_dim = int(self.down_dims[0])
        n_groups_final = valid_group_count(start_dim, 8)
        final_conv_kwargs = {"name": "final_conv"}
        final_norm_kwargs = {"name": "final_norm"}
        final_out_kwargs = {"name": "final_out"}
        if operator_variant == "torch":
            final_conv_kwargs.update(
                kernel_init=_PYTORCH_DEFAULT_KERNEL_INIT,
                bias_init=_pytorch_default_bias_init(
                    self.kernel_size * int(x.shape[-1])
                ),
            )
            final_norm_kwargs["epsilon"] = 1e-5
            final_out_kwargs.update(
                kernel_init=_PYTORCH_DEFAULT_KERNEL_INIT,
                bias_init=_pytorch_default_bias_init(start_dim),
            )
        x = nn.Conv(
            features=start_dim,
            kernel_size=(self.kernel_size,),
            padding="SAME",
            **final_conv_kwargs,
        )(x)
        x = nn.GroupNorm(
            num_groups=n_groups_final,
            **final_norm_kwargs,
        )(x)
        x = mish(x)
        x = nn.Conv(
            features=self.action_dim,
            kernel_size=(1,),
            padding="SAME",
            **final_out_kwargs,
        )(x)
        return x

    @property
    def output_shape(self) -> tuple[int, int]:
        return (self.sequence_length, self.action_dim)
