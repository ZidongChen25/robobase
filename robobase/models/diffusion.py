"""Conditional 1-D U-Net for diffusion policies (Flax)."""

from __future__ import annotations

import math

import flax.linen as nn
import jax
import jax.numpy as jnp


def _valid_group_count(num_channels: int, requested_groups: int) -> int:
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


class Conv1dBlock(nn.Module):
    """Conv1d + GroupNorm + Mish activation."""

    out_channels: int
    kernel_size: int
    n_groups: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        n_groups = _valid_group_count(self.out_channels, self.n_groups)
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(self.kernel_size,),
            padding="SAME",
            name="conv",
        )(x)
        x = nn.GroupNorm(num_groups=n_groups, name="norm")(x)
        return x * jnp.tanh(jax.nn.softplus(x))  # mish


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
            self.out_channels, self.kernel_size, self.n_groups, name="block1",
        )(x)

        # Condition injection
        cond_act = cond * jnp.tanh(jax.nn.softplus(cond))  # mish
        embed = nn.Dense(self.out_channels, name="cond_dense")(cond_act)[:, None, :]
        out = out + embed

        out = Conv1dBlock(
            self.out_channels, self.kernel_size, self.n_groups, name="block2",
        )(out)

        # Residual
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
    """Conditional 1-D U-Net for diffusion policy noise prediction."""

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
        timesteps = timesteps.astype(jnp.float32)
        if timesteps.ndim == 0:
            timesteps = jnp.broadcast_to(timesteps[None], (sample.shape[0],))
        elif timesteps.shape[0] == 1 and sample.shape[0] != 1:
            timesteps = jnp.broadcast_to(timesteps, (sample.shape[0],))

        # Timestep embedding
        t_emb = SinusoidalPosEmb(self.diffusion_step_embed_dim, name="pos_emb")(timesteps)
        t_emb = nn.Dense(self.diffusion_step_embed_dim * 4, name="time_dense1")(t_emb)
        t_emb = t_emb * jnp.tanh(jax.nn.softplus(t_emb))  # mish
        global_feature = nn.Dense(self.diffusion_step_embed_dim, name="time_dense2")(t_emb)

        if self.feature_dim > 0:
            if features is None:
                raise ValueError("Expected diffusion conditioning features.")
            global_feature = jnp.concatenate(
                [global_feature, features.astype(jnp.float32)], axis=-1,
            )

        cond_dim = self.diffusion_step_embed_dim + max(self.feature_dim, 0)
        all_dims = [self.action_dim] + [int(d) for d in self.down_dims]

        # Encoder (down path)
        x = sample
        hidden = []
        for i, (dim_in, dim_out) in enumerate(zip(all_dims[:-1], all_dims[1:])):
            is_last = i >= len(all_dims) - 2
            x = ResidualBlock(
                dim_out, cond_dim, self.kernel_size, self.n_groups,
                name=f"down_{i}_res1",
            )(x, global_feature)
            x = ResidualBlock(
                dim_out, cond_dim, self.kernel_size, self.n_groups,
                name=f"down_{i}_res2",
            )(x, global_feature)
            hidden.append(x)
            if not is_last:
                x = Downsample1d(dim_out, name=f"down_{i}_ds")(x)

        # Bottleneck
        mid_dim = all_dims[-1]
        x = ResidualBlock(
            mid_dim, cond_dim, self.kernel_size, self.n_groups, name="mid_res1",
        )(x, global_feature)
        x = ResidualBlock(
            mid_dim, cond_dim, self.kernel_size, self.n_groups, name="mid_res2",
        )(x, global_feature)

        # Decoder (up path)
        reversed_pairs = list(reversed(list(zip(all_dims[:-1], all_dims[1:]))[1:]))
        for i, (dim_in, dim_out) in enumerate(reversed_pairs):
            is_last = i >= len(all_dims) - 2
            x = jnp.concatenate([x, hidden.pop()], axis=-1)
            x = ResidualBlock(
                dim_in, cond_dim, self.kernel_size, self.n_groups,
                name=f"up_{i}_res1",
            )(x, global_feature)
            x = ResidualBlock(
                dim_in, cond_dim, self.kernel_size, self.n_groups,
                name=f"up_{i}_res2",
            )(x, global_feature)
            if not is_last:
                x = Upsample1d(dim_in, name=f"up_{i}_us")(x)

        # Final conv
        start_dim = int(self.down_dims[0])
        n_groups_final = _valid_group_count(start_dim, 8)
        x = nn.Conv(
            features=start_dim, kernel_size=(self.kernel_size,),
            padding="SAME", name="final_conv",
        )(x)
        x = nn.GroupNorm(num_groups=n_groups_final, name="final_norm")(x)
        x = x * jnp.tanh(jax.nn.softplus(x))  # mish
        x = nn.Conv(
            features=self.action_dim, kernel_size=(1,),
            padding="SAME", name="final_out",
        )(x)
        return x

    @property
    def output_shape(self) -> tuple[int, int]:
        return (self.sequence_length, self.action_dim)
