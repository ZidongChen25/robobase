"""Strict pure-JAX port of the public A2A image policy.

This module intentionally mirrors the initial-release RoboVerse policy instead
of sharing the more general RoboBase A2A components.  In particular, it keeps
the two independent action encoders, the GroupNorm ResNet18 observation path,
the 512-wide SimpleFlowNet and the six-step differentiable Euler rollout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from robobase.models.resnet import resnet_feature_model


Array = jax.Array
PyTree = Mapping[str, object]

_IMAGENET_MEAN = jnp.asarray((0.485, 0.456, 0.406), dtype=jnp.float32)
_IMAGENET_STD = jnp.asarray((0.229, 0.224, 0.225), dtype=jnp.float32)
_RESNET_CONV_INIT = nn.initializers.variance_scaling(2.0, "fan_out", "normal")
_PYTORCH_LINEAR_INIT = nn.initializers.variance_scaling(
    1.0 / 3.0, "fan_in", "uniform"
)
_ORTHOGONAL = nn.initializers.orthogonal()
_XAVIER = nn.initializers.xavier_uniform()
_ZERO = nn.initializers.zeros


@dataclass(frozen=True)
class OfficialA2AConfig:
    observation_steps: int = 8
    history_steps: int = 8
    action_steps: int = 8
    action_dim: int = 9
    latent_dim: int = 512
    hidden_dim: int = 512
    flow_layers: int = 4
    decoder_layers: int = 4
    flow_steps: int = 6
    flow_matcher: str = "conditional"
    image_range_normalization: bool = True
    num_cameras: int = 1
    vision_encoder: str = "official_gn"
    resize_to_224: bool = False
    consistency_weight: float = 1.0
    encoder_reconstruction_weight: float = 0.5
    flow_reconstruction_weight: float = 0.5

    def validate(self) -> None:
        if min(self.observation_steps, self.history_steps, self.action_steps) < 1:
            raise ValueError(
                "observation_steps, history_steps and action_steps must be positive."
            )
        if self.observation_steps > self.history_steps:
            raise ValueError("observation_steps cannot exceed history_steps.")
        if self.latent_dim != 512 or self.hidden_dim != 512:
            raise ValueError("Strict official mode requires 512-dimensional latents.")
        if self.flow_steps < 1:
            raise ValueError("flow_steps must be positive.")
        if self.flow_matcher not in {"conditional", "exact_ot"}:
            raise ValueError(
                "flow_matcher must be 'conditional' or 'exact_ot'; got "
                f"{self.flow_matcher!r}."
            )
        if self.num_cameras < 1:
            raise ValueError("num_cameras must be positive.")
        if self.vision_encoder not in {"official_gn", "fm_resnet"}:
            raise ValueError(
                "vision_encoder must be 'official_gn' or 'fm_resnet'."
            )


class _BasicBlock(nn.Module):
    features: int
    stride: int = 1

    @nn.compact
    def __call__(self, x: Array) -> Array:
        residual = x
        y = nn.Conv(
            self.features,
            (3, 3),
            strides=(self.stride, self.stride),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            kernel_init=_RESNET_CONV_INIT,
            name="conv1",
        )(x)
        y = nn.GroupNorm(
            num_groups=self.features // 16, epsilon=1e-5, name="gn1"
        )(y)
        y = nn.relu(y)
        y = nn.Conv(
            self.features,
            (3, 3),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            kernel_init=_RESNET_CONV_INIT,
            name="conv2",
        )(y)
        y = nn.GroupNorm(
            num_groups=self.features // 16, epsilon=1e-5, name="gn2"
        )(y)
        if self.stride != 1 or x.shape[-1] != self.features:
            residual = nn.Conv(
                self.features,
                (1, 1),
                strides=(self.stride, self.stride),
                use_bias=False,
                kernel_init=_RESNET_CONV_INIT,
                name="downsample_conv",
            )(residual)
            residual = nn.GroupNorm(
                num_groups=self.features // 16,
                epsilon=1e-5,
                name="downsample_gn",
            )(residual)
        return nn.relu(y + residual)


class OfficialResNet18(nn.Module):
    """Torchvision ResNet18 with every BatchNorm replaced by GroupNorm."""

    @nn.compact
    def __call__(self, image: Array) -> Array:
        x = nn.Conv(
            64,
            (7, 7),
            strides=(2, 2),
            padding=((3, 3), (3, 3)),
            use_bias=False,
            kernel_init=_RESNET_CONV_INIT,
            name="conv1",
        )(image)
        x = nn.GroupNorm(num_groups=4, epsilon=1e-5, name="gn1")(x)
        x = nn.relu(x)
        x = nn.max_pool(
            x, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1))
        )
        for stage, features in enumerate((64, 128, 256, 512)):
            for block in range(2):
                stride = 2 if stage > 0 and block == 0 else 1
                x = _BasicBlock(
                    features=features,
                    stride=stride,
                    name=f"layer{stage + 1}_{block}",
                )(x)
        return jnp.mean(x, axis=(1, 2))


class FMResNet18(nn.Module):
    """RoboBase FM ResNet18 trunk with its stable checkpoint tree."""

    def setup(self) -> None:
        self.trunk = resnet_feature_model(18)

    def __call__(self, image: Array) -> Array:
        return self.trunk(image)


class OfficialObservationEncoder(nn.Module):
    config: OfficialA2AConfig

    @nn.compact
    def __call__(self, images: Array, states: Array) -> Array:
        # Canonical layout is [B,O,V,C,H,W]. Keep old single-view calls valid.
        if images.ndim == 5:
            images = images[:, :, None]
        if images.ndim != 6 or images.shape[1:3] != (
            self.config.observation_steps,
            self.config.num_cameras,
        ):
            raise ValueError(
                "images must be [B,observation_steps,num_cameras,3,H,W], got "
                f"{images.shape}"
            )
        if states.shape[1:] != (
            self.config.observation_steps,
            self.config.action_dim,
        ):
            raise ValueError(
                "states must be "
                f"[B,{self.config.observation_steps},{self.config.action_dim}]"
            )
        batch = images.shape[0]
        x = images.reshape((-1, *images.shape[3:]))
        x = jnp.transpose(x, (0, 2, 3, 1))
        if images.dtype == jnp.uint8:
            x = x.astype(jnp.float32) / 255.0
        else:
            x = x.astype(jnp.float32)
        # RobotImageDataset maps [0, 1] RGB to [-1, 1] through its
        # LinearNormalizer before MultiImageObsEncoder applies ImageNet stats.
        if self.config.image_range_normalization:
            x = 2.0 * x - 1.0
        if self.config.resize_to_224:
            x = jax.image.resize(
                x, (x.shape[0], 224, 224, 3), method="bilinear"
            )
        x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
        if self.config.vision_encoder == "fm_resnet":
            visual = FMResNet18(name="rgb_model")(x)
            visual = jnp.mean(visual, axis=(1, 2))
        else:
            visual = OfficialResNet18(name="rgb_model")(x)
        visual = visual.reshape(
            (batch, self.config.observation_steps, self.config.num_cameras, -1)
        )
        state = states.astype(jnp.float32)
        features = jnp.concatenate(
            (visual.reshape((batch, -1)), state.reshape((batch, -1))), axis=-1
        )
        fan_in = int(features.shape[-1])
        return nn.Dense(
            self.config.latent_dim,
            kernel_init=_PYTORCH_LINEAR_INIT,
            bias_init=nn.initializers.uniform(1.0 / np.sqrt(fan_in)),
            name="obs_projector",
        )(features)


class OfficialActionEncoder(nn.Module):
    config: OfficialA2AConfig
    steps: int | None = None

    @nn.compact
    def __call__(self, actions: Array) -> Array:
        steps = self.config.action_steps if self.steps is None else self.steps
        if actions.shape[1:] != (steps, self.config.action_dim):
            raise ValueError(
                "actions must be "
                f"[B,{steps},{self.config.action_dim}], "
                f"got {actions.shape}"
            )
        x = actions.astype(jnp.float32)
        for layer in range(3):
            x = nn.Conv(
                self.config.hidden_dim,
                kernel_size=(5,),
                strides=(2,),
                padding=((2, 2),),
                kernel_init=_ORTHOGONAL,
                bias_init=_ZERO,
                name=f"conv{layer + 1}",
            )(x)
            x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        return nn.Dense(
            self.config.latent_dim,
            kernel_init=_ORTHOGONAL,
            bias_init=_ZERO,
            name="latent_proj",
        )(x)


class _Mlp(nn.Module):
    """Action-decoder MLP matching torch.nn.GELU's exact default."""

    features: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = nn.Dense(self.features, kernel_init=_XAVIER, bias_init=_ZERO, name="fc1")(x)
        x = nn.gelu(x, approximate=False)
        return nn.Dense(
            self.features, kernel_init=_XAVIER, bias_init=_ZERO, name="fc2"
        )(x)


class OfficialActionDecoder(nn.Module):
    config: OfficialA2AConfig

    @nn.compact
    def __call__(self, latent: Array) -> Array:
        x = nn.Dense(
            self.config.hidden_dim,
            kernel_init=_XAVIER,
            bias_init=_ZERO,
            name="input_proj",
        )(latent)
        for layer in range(self.config.decoder_layers):
            x = _Mlp(self.config.hidden_dim, name=f"layer{layer}")(x)
        x = nn.Dense(
            self.config.action_steps * self.config.action_dim,
            kernel_init=_XAVIER,
            bias_init=_ZERO,
            name="output_proj",
        )(x)
        return x.reshape((-1, self.config.action_steps, self.config.action_dim))


def _sinusoidal_embedding(time: Array, dim: int = 256) -> Array:
    half = dim // 2
    scale = np.log(10000.0) / (half - 1)
    frequencies = jnp.exp(jnp.arange(half, dtype=jnp.float32) * -scale)
    phase = time.astype(jnp.float32)[:, None] * frequencies[None]
    return jnp.concatenate((jnp.sin(phase), jnp.cos(phase)), axis=-1)


class _OfficialFlowMlp(nn.Module):
    """timm MLP with expansion followed by projection back to hidden width."""

    config: OfficialA2AConfig

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = nn.Dense(
            4 * self.config.hidden_dim,
            kernel_init=_XAVIER,
            bias_init=_ZERO,
            name="fc1",
        )(x)
        x = nn.gelu(x, approximate=True)
        return nn.Dense(
            self.config.hidden_dim,
            kernel_init=_XAVIER,
            bias_init=_ZERO,
            name="fc2",
        )(x)


class OfficialSimpleFlowNet(nn.Module):
    config: OfficialA2AConfig

    @nn.compact
    def __call__(self, latent: Array, time: Array, condition: Array) -> Array:
        x = nn.Dense(
            self.config.hidden_dim,
            kernel_init=_XAVIER,
            bias_init=_ZERO,
            name="input_proj",
        )(latent)
        t = nn.Dense(
            1024,
            kernel_init=nn.initializers.normal(0.02),
            bias_init=_ZERO,
            name="time_fc1",
        )(_sinusoidal_embedding(time))
        t = jax.nn.mish(t)
        t = nn.Dense(
            self.config.hidden_dim,
            kernel_init=nn.initializers.normal(0.02),
            bias_init=_ZERO,
            name="time_fc2",
        )(t)
        t += nn.Dense(
            self.config.hidden_dim,
            kernel_init=_XAVIER,
            bias_init=_ZERO,
            name="cond_embed",
        )(condition)
        for layer in range(self.config.flow_layers):
            modulation = nn.Dense(
                3 * self.config.hidden_dim,
                # FlowNetLayer zeroes this layer, then the enclosing
                # SimpleFlowNet.apply(_basic_init) overwrites it with Xavier.
                kernel_init=_XAVIER,
                bias_init=_ZERO,
                name=f"layer{layer}_time_modulator",
            )(nn.silu(t))
            gamma, scale, shift = jnp.split(modulation, 3, axis=-1)
            y = nn.LayerNorm(
                use_scale=False,
                use_bias=False,
                epsilon=1e-6,
                name=f"layer{layer}_norm",
            )(x)
            y = y * (scale + 1.0) + shift
            y = _OfficialFlowMlp(self.config, name=f"layer{layer}_mlp")(y)
            x = x + y * gamma
        x = nn.LayerNorm(name="norm")(x)
        return nn.Dense(
            self.config.latent_dim,
            kernel_init=_XAVIER,
            bias_init=_ZERO,
            name="out_proj",
        )(x)


class OfficialA2A(nn.Module):
    """Plug-and-play official A2A policy network without training state."""

    config: OfficialA2AConfig = OfficialA2AConfig()

    def setup(self) -> None:
        self.config.validate()
        self.observation_encoder = OfficialObservationEncoder(
            self.config, name="observation_encoder"
        )
        self.history_encoder = OfficialActionEncoder(
            self.config,
            steps=self.config.history_steps,
            name="history_action_encoder",
        )
        self.action_encoder = OfficialActionEncoder(self.config, name="action_encoder")
        self.action_decoder = OfficialActionDecoder(self.config, name="action_decoder")
        self.flow_net = OfficialSimpleFlowNet(self.config, name="flow_net")

    def encode(self, images: Array, states: Array, future_actions: Array):
        observation_states = states[:, -self.config.observation_steps :]
        condition = self.observation_encoder(images, observation_states)
        source = self.history_encoder(states)
        target = self.action_encoder(future_actions)
        return source, target, condition

    def integrate(self, source: Array, condition: Array) -> Array:
        dt = jnp.asarray(1.0 / self.config.flow_steps, dtype=source.dtype)

        first_time = jnp.zeros((source.shape[0],), dtype=source.dtype)
        first_velocity = self.flow_net(source, first_time, condition)
        source = source + dt * first_velocity

        def step(latent, index):
            time = jnp.full((latent.shape[0],), index / self.config.flow_steps)
            velocity = self.flow_net(latent, time, condition)
            return latent + dt * velocity, None

        return jax.lax.scan(
            step, source, jnp.arange(1, self.config.flow_steps)
        )[0]

    def predict_normalized(self, images: Array, states: Array) -> Array:
        observation_states = states[:, -self.config.observation_steps :]
        condition = self.observation_encoder(images, observation_states)
        source = self.history_encoder(states)
        return self.action_decoder(self.integrate(source, condition))

    def initialize_all(
        self, images: Array, states: Array, actions: Array
    ) -> tuple[Array, Array]:
        """Materialize every parameter collection without stochastic OT work."""

        future_start = self.config.history_steps - 1
        future = actions[:, future_start : future_start + self.config.action_steps]
        source, target, condition = self.encode(
            images,
            states[:, : self.config.history_steps],
            future,
        )
        return (
            self.action_decoder(self.integrate(source, condition)),
            self.action_decoder(target),
        )

    def compute_loss(
        self,
        images: Array,
        states: Array,
        actions: Array,
        ot_key: Array,
    ) -> tuple[Array, dict[str, Array]]:
        return official_a2a_loss(
            self, images, states, actions, ot_key=ot_key
        )


def _linear_sum_assignment(cost: Array) -> Array:
    """Return the optimal target column for each source row on device."""

    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError(f"OT cost must be square, got {cost.shape}.")
    size = int(cost.shape[0])
    potential_rows = jnp.zeros((size + 1,), dtype=cost.dtype)
    potential_cols = jnp.zeros((size + 1,), dtype=cost.dtype)
    matched_row = jnp.zeros((size + 1,), dtype=jnp.int32)
    predecessor = jnp.zeros((size + 1,), dtype=jnp.int32)

    def add_row(row_index, carry):
        u, v, matching, unused_way = carry
        del unused_way
        matching = matching.at[0].set(row_index + 1)
        minimum = jnp.full((size + 1,), jnp.inf, dtype=cost.dtype)
        used = jnp.zeros((size + 1,), dtype=jnp.bool_)
        way = jnp.zeros((size + 1,), dtype=jnp.int32)

        def search_condition(search):
            _u, _v, current_matching, _minimum, _used, _way, column = search
            return current_matching[column] != 0

        def search_body(search):
            u0, v0, current_matching, min0, used0, way0, column0 = search
            used0 = used0.at[column0].set(True)
            current_row = current_matching[column0]
            reduced = (
                cost[current_row - 1] - u0[current_row] - v0[1:]
            )
            reduced = jnp.concatenate((jnp.asarray([jnp.inf]), reduced))
            improved = jnp.logical_and(jnp.logical_not(used0), reduced < min0)
            min0 = jnp.where(improved, reduced, min0)
            way0 = jnp.where(improved, column0, way0)
            candidates = jnp.where(jnp.logical_not(used0), min0, jnp.inf)
            delta = jnp.min(candidates)
            next_column = jnp.argmin(candidates).astype(jnp.int32)
            increments = jnp.where(used0, delta, 0.0)
            u0 = u0.at[current_matching].add(increments)
            v0 = v0 - increments
            min0 = min0 - jnp.where(jnp.logical_not(used0), delta, 0.0)
            return u0, v0, current_matching, min0, used0, way0, next_column

        u, v, matching, minimum, used, way, free_column = jax.lax.while_loop(
            search_condition,
            search_body,
            (u, v, matching, minimum, used, way, jnp.asarray(0, jnp.int32)),
        )

        def augment_condition(augment):
            _matching, column = augment
            return column != 0

        def augment_body(augment):
            current_matching, column = augment
            previous = way[column]
            current_matching = current_matching.at[column].set(
                current_matching[previous]
            )
            return current_matching, previous

        matching, _ = jax.lax.while_loop(
            augment_condition, augment_body, (matching, free_column)
        )
        return u, v, matching, way

    _, _, matched_row, _ = jax.lax.fori_loop(
        0,
        size,
        add_row,
        (potential_rows, potential_cols, matched_row, predecessor),
    )
    columns = jnp.arange(size, dtype=jnp.int32)
    rows = matched_row[1:] - 1
    return jnp.zeros((size,), dtype=jnp.int32).at[rows].set(columns)


def sample_exact_ot_pairs(source: Array, target: Array, key: Array) -> tuple[Array, Array]:
    """Match torchcfm's exact uniform minibatch OT sampling semantics.

    The public implementation computes an EMD plan and samples ``B`` pairs from
    it with replacement. For equal uniform minibatches the plan is an
    assignment, so device-side Hungarian matching followed by uniform row
    resampling is exactly equivalent and avoids host synchronization.
    """

    batch = source.shape[0]
    cost = jax.lax.stop_gradient(
        jnp.sum((source[:, None, :] - target[None, :, :]) ** 2, axis=-1)
    )

    permutation = _linear_sum_assignment(cost)
    rows = jax.random.randint(key, (batch,), 0, batch)
    return source[rows], target[permutation[rows]]


def official_a2a_loss(
    model: OfficialA2A,
    images: Array,
    states: Array,
    actions: Array,
    *,
    ot_key: Array,
) -> tuple[Array, dict[str, Array]]:
    """Official four-term objective; inputs must already be normalized."""

    config = model.config
    future_start = config.history_steps - 1
    future = actions[:, future_start : future_start + config.action_steps]
    source, target, condition = model.encode(
        images[:, : config.observation_steps],
        states[:, : config.history_steps],
        future,
    )
    if config.flow_matcher == "exact_ot":
        paired_source, paired_target = sample_exact_ot_pairs(
            source, target, ot_key
        )
    else:
        paired_source, paired_target = source, target
    time_key = jax.random.fold_in(ot_key, 1)
    time = jax.random.uniform(time_key, (source.shape[0],))
    interpolated = (1.0 - time[:, None]) * paired_source + time[:, None] * paired_target
    velocity_target = paired_target - paired_source
    velocity = model.flow_net(interpolated, time, condition)
    flow_loss = jnp.mean(jnp.square(velocity - velocity_target))

    predicted_latent = model.integrate(source, condition)
    consistency_loss = jnp.mean(jnp.square(predicted_latent - target))
    flow_reconstruction = jnp.mean(
        jnp.abs(model.action_decoder(predicted_latent) - future)
    )
    encoder_reconstruction = jnp.mean(
        jnp.abs(model.action_decoder(target) - future)
    )
    total = (
        flow_loss
        + config.consistency_weight * consistency_loss
        + config.flow_reconstruction_weight * flow_reconstruction
        + config.encoder_reconstruction_weight * encoder_reconstruction
    )
    return total, {
        "loss": total,
        "flow_loss": flow_loss,
        "consistency_loss": consistency_loss,
        "flow_action_reconstruction_loss": flow_reconstruction,
        "encoder_action_reconstruction_loss": encoder_reconstruction,
    }


__all__ = [
    "OfficialA2A",
    "OfficialA2AConfig",
    "OfficialActionDecoder",
    "OfficialActionEncoder",
    "OfficialObservationEncoder",
    "OfficialResNet18",
    "OfficialSimpleFlowNet",
    "_linear_sum_assignment",
    "official_a2a_loss",
    "sample_exact_ot_pairs",
]
