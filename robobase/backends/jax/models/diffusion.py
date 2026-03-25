from __future__ import annotations

from dataclasses import dataclass
import math


def _valid_group_count(num_channels: int, requested_groups: int) -> int:
    groups = max(1, min(int(requested_groups), int(num_channels)))
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class _SinusoidalPosEmb:
    def __init__(self, dim: int):
        self.dim = int(dim)

    def __call__(self, jnp, x):
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


@dataclass(frozen=True)
class JaxConditionalUnet1D:
    action_shape: tuple[int, int]
    feature_dim: int
    diffusion_step_embed_dim: int
    down_dims: tuple[int, ...]
    kernel_size: int = 5
    n_groups: int = 8

    def __post_init__(self):
        try:
            import flax.linen as nn
            import jax
            import jax.numpy as jnp
        except ImportError as exc:
            raise ImportError(
                "JAX diffusion support requires `flax` and `jax`."
            ) from exc

        object.__setattr__(self, "jax", jax)
        object.__setattr__(self, "jnp", jnp)
        object.__setattr__(self, "nn", nn)
        if not self.down_dims:
            raise ValueError("JAX diffusion requires at least one down_dim.")
        if int(self.action_shape[0]) < 4:
            raise ValueError("sequence_length must be >= 4.")
        if int(self.action_shape[0]) % 4 != 0:
            raise ValueError(
                "Action sequence length has to be a multiple of 4 for diffusion model."
            )
        object.__setattr__(self, "_start_dim", int(self.down_dims[0]))

    @property
    def output_shape(self):
        return self.action_shape

    def init_params(self, key):
        dummy_actions = self.jnp.zeros((1, *self.action_shape), dtype=self.jnp.float32)
        dummy_timesteps = self.jnp.zeros((1,), dtype=self.jnp.int32)
        dummy_features = None
        if self.feature_dim > 0:
            dummy_features = self.jnp.zeros((1, self.feature_dim), dtype=self.jnp.float32)
        return self._init_weights(key, dummy_actions, dummy_timesteps, dummy_features)

    def apply(self, params, actions, timesteps, features):
        return self._forward(params, actions, timesteps, features)

    def _mish(self, x):
        return x * self.jnp.tanh(self.jax.nn.softplus(x))

    def _dense(self, x, layer_params):
        return x @ layer_params["w"] + layer_params["b"]

    def _conv1d(self, x, layer_params, *, stride: int = 1):
        kernel_size = int(layer_params["w"].shape[0])
        padding = ((kernel_size // 2, kernel_size // 2),)
        y = self.jax.lax.conv_general_dilated(
            lhs=x,
            rhs=layer_params["w"],
            window_strides=(stride,),
            padding=padding,
            dimension_numbers=("NWC", "WIO", "NWC"),
        )
        return y + layer_params["b"]

    def _conv_transpose1d(self, x, layer_params, *, stride: int = 2):
        y = self.jax.lax.conv_transpose(
            lhs=x,
            rhs=layer_params["w"],
            strides=(stride,),
            padding="SAME",
            dimension_numbers=("NWC", "WIO", "NWC"),
        )
        return y + layer_params["b"]

    def _group_norm(self, x, norm_params, *, n_groups: int | None = None):
        requested_groups = self.n_groups if n_groups is None else n_groups
        num_groups = _valid_group_count(x.shape[-1], requested_groups)
        scale_value = norm_params["scale"]
        bias_value = norm_params["bias"]
        batch, horizon, channels = x.shape
        x = x.reshape((batch, horizon, num_groups, channels // num_groups))
        mean = x.mean(axis=(1, 3), keepdims=True)
        var = x.var(axis=(1, 3), keepdims=True)
        x = (x - mean) / self.jnp.sqrt(var + 1e-5)
        x = x.reshape((batch, horizon, channels))
        return x * scale_value.reshape((1, 1, channels)) + bias_value.reshape(
            (1, 1, channels)
        )

    def _conv1d_block(self, x, params, *, n_groups: int | None = None):
        x = self._conv1d(x, params["conv"])
        x = self._group_norm(x, params["norm"], n_groups=n_groups)
        return self._mish(x)

    def _residual_block(self, x, cond, params):
        out = self._conv1d_block(x, params["block1"], n_groups=self.n_groups)
        embed = self._dense(self._mish(cond), params["cond"])[:, None, :]
        out = out + embed
        out = self._conv1d_block(out, params["block2"], n_groups=self.n_groups)
        residual = x
        if params["residual"] is not None:
            residual = self._dense(x, params["residual"])
        return out + residual

    def _downsample(self, x, params):
        if params is None:
            return x
        return self._conv1d(x, params, stride=2)

    def _upsample(self, x, params):
        if params is None:
            return x
        return self._conv_transpose1d(x, params, stride=2)

    def _forward(self, params, actions, timesteps, features):
        sample = actions.astype(self.jnp.float32)
        timesteps = timesteps.astype(self.jnp.float32)
        if timesteps.ndim == 0:
            timesteps = self.jnp.broadcast_to(timesteps[None], (sample.shape[0],))
        elif timesteps.shape[0] == 1 and sample.shape[0] != 1:
            timesteps = self.jnp.broadcast_to(timesteps, (sample.shape[0],))

        timestep_embedding = _SinusoidalPosEmb(self.diffusion_step_embed_dim)(
            self.jnp,
            timesteps,
        )
        timestep_embedding = self._dense(
            timestep_embedding,
            params["diffusion_step_encoder"]["dense1"],
        )
        timestep_embedding = self._mish(timestep_embedding)
        global_feature = self._dense(
            timestep_embedding,
            params["diffusion_step_encoder"]["dense2"],
        )

        if self.feature_dim > 0:
            if features is None:
                raise ValueError("Expected diffusion conditioning features.")
            global_feature = self.jnp.concatenate(
                [global_feature, features.astype(self.jnp.float32)],
                axis=-1,
            )

        x = sample
        hidden = []
        for down_params in params["down_modules"]:
            x = self._residual_block(x, global_feature, down_params["res1"])
            x = self._residual_block(x, global_feature, down_params["res2"])
            hidden.append(x)
            x = self._downsample(x, down_params["downsample"])

        for mid_params in params["mid_modules"]:
            x = self._residual_block(x, global_feature, mid_params)

        for up_params in params["up_modules"]:
            x = self.jnp.concatenate([x, hidden.pop()], axis=-1)
            x = self._residual_block(x, global_feature, up_params["res1"])
            x = self._residual_block(x, global_feature, up_params["res2"])
            x = self._upsample(x, up_params["upsample"])

        x = self._conv1d_block(x, params["final_conv"]["block"], n_groups=8)
        return self._conv1d(x, params["final_conv"]["out"])

    def _init_dense(self, key, in_dim: int, out_dim: int):
        limit = 1.0 / math.sqrt(float(in_dim))
        weight_key, bias_key = self.jax.random.split(key)
        return {
            "w": self.jax.random.uniform(
                weight_key,
                shape=(in_dim, out_dim),
                minval=-limit,
                maxval=limit,
                dtype=self.jnp.float32,
            ),
            "b": self.jax.random.uniform(
                bias_key,
                shape=(out_dim,),
                minval=-limit,
                maxval=limit,
                dtype=self.jnp.float32,
            ),
        }

    def _init_conv(self, key, in_channels: int, out_channels: int, kernel_size: int):
        fan_in = float(kernel_size * in_channels)
        limit = 1.0 / math.sqrt(fan_in)
        weight_key, bias_key = self.jax.random.split(key)
        return {
            "w": self.jax.random.uniform(
                weight_key,
                shape=(kernel_size, in_channels, out_channels),
                minval=-limit,
                maxval=limit,
                dtype=self.jnp.float32,
            ),
            "b": self.jax.random.uniform(
                bias_key,
                shape=(out_channels,),
                minval=-limit,
                maxval=limit,
                dtype=self.jnp.float32,
            ),
        }

    def _init_conv_transpose(
        self,
        key,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
    ):
        fan_in = float(kernel_size * out_channels)
        limit = 1.0 / math.sqrt(fan_in)
        weight_key, bias_key = self.jax.random.split(key)
        return {
            "w": self.jax.random.uniform(
                weight_key,
                shape=(kernel_size, in_channels, out_channels),
                minval=-limit,
                maxval=limit,
                dtype=self.jnp.float32,
            ),
            "b": self.jax.random.uniform(
                bias_key,
                shape=(out_channels,),
                minval=-limit,
                maxval=limit,
                dtype=self.jnp.float32,
            ),
        }

    def _init_group_norm(self, out_channels: int):
        return {
            "scale": self.jnp.ones((out_channels,), dtype=self.jnp.float32),
            "bias": self.jnp.zeros((out_channels,), dtype=self.jnp.float32),
        }

    def _init_conv_block(self, key, in_channels: int, out_channels: int):
        conv_key = key
        return {
            "conv": self._init_conv(conv_key, in_channels, out_channels, self.kernel_size),
            "norm": self._init_group_norm(out_channels),
        }

    def _init_residual_block(self, key, in_channels: int, out_channels: int, cond_dim: int):
        key1, key2, key3, key4 = self.jax.random.split(key, 4)
        residual = None
        if in_channels != out_channels:
            residual = self._init_dense(key4, in_channels, out_channels)
        return {
            "block1": self._init_conv_block(key1, in_channels, out_channels),
            "block2": self._init_conv_block(key2, out_channels, out_channels),
            "cond": self._init_dense(key3, cond_dim, out_channels),
            "residual": residual,
        }

    def _init_weights(self, key, _dummy_actions, _dummy_timesteps, _dummy_features):
        num_down = len(self.down_dims)
        num_up = max(0, num_down - 1)
        split_count = 2 + (num_down * 3) + (2 * 2) + (num_up * 3) + 2
        keys = iter(self.jax.random.split(key, split_count))
        input_dim = int(self.action_shape[1])
        cond_dim = int(self.diffusion_step_embed_dim + max(self.feature_dim, 0))
        all_dims = [input_dim] + [int(v) for v in self.down_dims]

        params = {
            "diffusion_step_encoder": {
                "dense1": self._init_dense(
                    next(keys),
                    self.diffusion_step_embed_dim,
                    self.diffusion_step_embed_dim * 4,
                ),
                "dense2": self._init_dense(
                    next(keys),
                    self.diffusion_step_embed_dim * 4,
                    self.diffusion_step_embed_dim,
                ),
            },
            "down_modules": [],
            "mid_modules": [],
            "up_modules": [],
            "final_conv": {},
        }

        for index, (dim_in, dim_out) in enumerate(zip(all_dims[:-1], all_dims[1:])):
            is_last = index >= (len(all_dims) - 2)
            params["down_modules"].append(
                {
                    "res1": self._init_residual_block(
                        next(keys), dim_in, dim_out, cond_dim
                    ),
                    "res2": self._init_residual_block(
                        next(keys), dim_out, dim_out, cond_dim
                    ),
                    "downsample": None
                    if is_last
                    else self._init_conv(next(keys), dim_out, dim_out, 3),
                }
            )

        mid_dim = int(all_dims[-1])
        params["mid_modules"] = [
            self._init_residual_block(next(keys), mid_dim, mid_dim, cond_dim),
            self._init_residual_block(next(keys), mid_dim, mid_dim, cond_dim),
        ]

        reversed_pairs = list(reversed(list(zip(all_dims[:-1], all_dims[1:]))[1:]))
        for index, (dim_in, dim_out) in enumerate(reversed_pairs):
            is_last = index >= (len(all_dims) - 2)
            params["up_modules"].append(
                {
                    "res1": self._init_residual_block(
                        next(keys), dim_out * 2, dim_in, cond_dim
                    ),
                    "res2": self._init_residual_block(
                        next(keys), dim_in, dim_in, cond_dim
                    ),
                    "upsample": None
                    if is_last
                    else self._init_conv_transpose(next(keys), dim_in, dim_in, 4),
                }
            )

        params["final_conv"] = {
            "block": self._init_conv_block(
                next(keys),
                self._start_dim,
                self._start_dim,
            ),
            "out": self._init_conv(
                next(keys),
                self._start_dim,
                input_dim,
                1,
            ),
        }
        return params
