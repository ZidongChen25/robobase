"""Conditional 1-D U-Net backbone for action-sequence prediction."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from robobase.models.backbones.common import (
    SinusoidalPosEmb,
    ensure_time_batch,
    mish,
    valid_group_count,
)


class Conv1dBlock(nn.Module):
    """Conv1d + GroupNorm + Mish activation."""

    out_channels: int
    kernel_size: int
    n_groups: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        n_groups = valid_group_count(self.out_channels, self.n_groups)
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(self.kernel_size,),
            padding="SAME",
            name="conv",
        )(x)
        x = nn.GroupNorm(num_groups=n_groups, name="norm")(x)
        return mish(x)


class ResidualBlock(nn.Module):
    """Two Conv1dBlocks with conditioning injection and residual connection."""

    out_channels: int
    cond_dim: int
    kernel_size: int
    n_groups: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
        in_channels = x.shape[-1]

        out = Conv1dBlock(
            self.out_channels,
            self.kernel_size,
            self.n_groups,
            name="block1",
        )(x)

        embed = nn.Dense(self.out_channels, name="cond_dense")(mish(cond))[
            :, None, :
        ]
        out = out + embed

        out = Conv1dBlock(
            self.out_channels,
            self.kernel_size,
            self.n_groups,
            name="block2",
        )(out)

        if in_channels != self.out_channels:
            residual = nn.Dense(self.out_channels, name="residual_dense")(x)
        else:
            residual = x
        return out + residual


class Downsample1d(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.Conv(
            features=self.channels,
            kernel_size=(3,),
            strides=(2,),
            padding="SAME",
            name="conv",
        )(x)


class Upsample1d(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.ConvTranspose(
            features=self.channels,
            kernel_size=(4,),
            strides=(2,),
            padding="SAME",
            name="conv_transpose",
        )(x)


class JaxConditionalUnet1D(nn.Module):
    """Conditional U-Net preserving the existing JAX diffusion-policy path."""

    action_dim: int
    sequence_length: int
    feature_dim: int
    diffusion_step_embed_dim: int
    down_dims: tuple[int, ...]
    kernel_size: int = 5
    n_groups: int = 8

    @nn.compact
    def __call__(
        self,
        actions: jnp.ndarray,
        timesteps: jnp.ndarray,
        features: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        sample = actions.astype(jnp.float32)
        timesteps = ensure_time_batch(timesteps, sample.shape[0])

        t_emb = SinusoidalPosEmb(self.diffusion_step_embed_dim, name="pos_emb")(
            timesteps
        )
        t_emb = nn.Dense(
            self.diffusion_step_embed_dim * 4,
            name="time_dense1",
        )(t_emb)
        t_emb = mish(t_emb)
        global_feature = nn.Dense(
            self.diffusion_step_embed_dim,
            name="time_dense2",
        )(t_emb)

        if self.feature_dim > 0:
            if features is None:
                raise ValueError("Expected diffusion conditioning features.")
            global_feature = jnp.concatenate(
                [global_feature, features.astype(jnp.float32)],
                axis=-1,
            )

        cond_dim = self.diffusion_step_embed_dim + max(self.feature_dim, 0)
        all_dims = [self.action_dim] + [int(d) for d in self.down_dims]

        x = sample
        hidden = []
        for i, (_, dim_out) in enumerate(zip(all_dims[:-1], all_dims[1:])):
            is_last = i >= len(all_dims) - 2
            x = ResidualBlock(
                dim_out,
                cond_dim,
                self.kernel_size,
                self.n_groups,
                name=f"down_{i}_res1",
            )(x, global_feature)
            x = ResidualBlock(
                dim_out,
                cond_dim,
                self.kernel_size,
                self.n_groups,
                name=f"down_{i}_res2",
            )(x, global_feature)
            hidden.append(x)
            if not is_last:
                x = Downsample1d(dim_out, name=f"down_{i}_ds")(x)

        mid_dim = all_dims[-1]
        x = ResidualBlock(
            mid_dim,
            cond_dim,
            self.kernel_size,
            self.n_groups,
            name="mid_res1",
        )(x, global_feature)
        x = ResidualBlock(
            mid_dim,
            cond_dim,
            self.kernel_size,
            self.n_groups,
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
                name=f"up_{i}_res1",
            )(x, global_feature)
            x = ResidualBlock(
                dim_in,
                cond_dim,
                self.kernel_size,
                self.n_groups,
                name=f"up_{i}_res2",
            )(x, global_feature)
            if not is_last:
                x = Upsample1d(dim_in, name=f"up_{i}_us")(x)

        start_dim = int(self.down_dims[0])
        n_groups_final = valid_group_count(start_dim, 8)
        x = nn.Conv(
            features=start_dim,
            kernel_size=(self.kernel_size,),
            padding="SAME",
            name="final_conv",
        )(x)
        x = nn.GroupNorm(num_groups=n_groups_final, name="final_norm")(x)
        x = mish(x)
        x = nn.Conv(
            features=self.action_dim,
            kernel_size=(1,),
            padding="SAME",
            name="final_out",
        )(x)
        return x

    @property
    def output_shape(self) -> tuple[int, int]:
        return (self.sequence_length, self.action_dim)
