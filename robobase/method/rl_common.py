"""Shared pure-JAX building blocks for continuous-control RL methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig

from robobase.method.bc import (
    BCActorModelSpec,
    BCEncoderModelSpec,
    BCModelSpec,
    BCViewFusionModelSpec,
    _build_model,
)
from robobase.method.jax_base import JaxMethodBase


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
LOG_TWO_PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class RLModelSpec:
    hidden_dims: tuple[int, ...] = (256, 256)
    activation: str = "relu"
    norm: str = "none"
    linear_bias: bool = True
    encoder_model: BCEncoderModelSpec | None = None
    view_fusion_model: BCViewFusionModelSpec | None = None


def _config_type(
    cfg: DictConfig | None,
    *,
    default: str,
    target_to_type: dict[str, str] | None = None,
) -> str:
    if cfg is None:
        return default
    config_type = cfg.get("type", None)
    if config_type is not None:
        return str(config_type).lower()
    target = str(cfg.get("_target_", "")).strip()
    if target_to_type and target in target_to_type:
        return target_to_type[target]
    return default


def rl_model_spec_from_cfg(cfg: DictConfig) -> RLModelSpec:
    """Parse the backend-neutral RL model section from a Hydra method config."""

    method_cfg = cfg.method
    model_cfg = method_cfg.get("model", None)
    if model_cfg is None:
        model_cfg = method_cfg
    hidden_dims = model_cfg.get(
        "hidden_dims",
        model_cfg.get("mlp_nodes", (256, 256)),
    )

    encoder_cfg = method_cfg.get("encoder_model", None)
    encoder_spec = None
    if encoder_cfg is not None:
        encoder_spec = BCEncoderModelSpec(
            type=_config_type(
                encoder_cfg,
                default="resnet",
                target_to_type={
                    "robobase.models.encoder.JaxResNetEncoder": "resnet",
                },
            ),
            model=str(encoder_cfg.get("model", "resnet18")),
            trainable=bool(encoder_cfg.get("trainable", True)),
            pretrained=bool(encoder_cfg.get("pretrained", False)),
            use_plucker=bool(encoder_cfg.get("use_plucker", False)),
            plucker_hidden_channels=int(
                encoder_cfg.get("plucker_hidden_channels", 64)
            ),
            plucker_identity_init=bool(
                encoder_cfg.get("plucker_identity_init", False)
            ),
        )

    fusion_cfg = method_cfg.get("view_fusion_model", None)
    fusion_spec = None
    if fusion_cfg is not None:
        fusion_spec = BCViewFusionModelSpec(
            type=_config_type(
                fusion_cfg,
                default="multicam_feature",
                target_to_type={
                    "robobase.models.fusion.JaxFusionMultiCamFeature": (
                        "multicam_feature"
                    ),
                },
            ),
            mode=str(fusion_cfg.get("mode", "flatten")).lower(),
        )

    return RLModelSpec(
        hidden_dims=tuple(int(value) for value in hidden_dims),
        activation=str(model_cfg.get("activation", "relu")).lower(),
        norm=str(model_cfg.get("norm", "none")).lower(),
        linear_bias=bool(model_cfg.get("linear_bias", True)),
        encoder_model=encoder_spec,
        view_fusion_model=fusion_spec,
    )


def activation(x: jax.Array, name: str) -> jax.Array:
    if name == "relu":
        return nn.relu(x)
    if name in {"silu", "swish"}:
        return nn.silu(x)
    if name == "tanh":
        return jnp.tanh(x)
    raise ValueError(f"Unsupported activation '{name}'.")


class MLP(nn.Module):
    hidden_dims: tuple[int, ...]
    output_dim: int
    activation_name: str = "relu"
    output_scale: float = 1.0

    @nn.compact
    def __call__(self, features: jax.Array) -> jax.Array:
        x = features.astype(jnp.float32)
        if x.ndim > 2:
            x = x.reshape((x.shape[0], -1))
        for index, width in enumerate(self.hidden_dims):
            x = nn.Dense(width, name=f"dense_{index}")(x)
            x = activation(x, self.activation_name)
        kernel_init = nn.initializers.variance_scaling(
            self.output_scale,
            "fan_in",
            "uniform",
        )
        return nn.Dense(
            self.output_dim,
            kernel_init=kernel_init,
            name="out",
        )(x)


class GaussianActor(nn.Module):
    hidden_dims: tuple[int, ...]
    action_dim: int
    activation_name: str = "relu"
    state_dependent_std: bool = True

    @nn.compact
    def __call__(self, features: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = features.astype(jnp.float32)
        for index, width in enumerate(self.hidden_dims):
            x = nn.Dense(width, name=f"dense_{index}")(x)
            x = activation(x, self.activation_name)
        mean = nn.Dense(self.action_dim, name="mean")(x)
        if self.state_dependent_std:
            raw_log_std = nn.Dense(self.action_dim, name="log_std")(x)
            raw_log_std = jnp.tanh(raw_log_std)
            log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
                raw_log_std + 1.0
            )
        else:
            log_std = self.param(
                "log_std",
                nn.initializers.zeros_init(),
                (self.action_dim,),
            )
            log_std = jnp.broadcast_to(log_std, mean.shape)
        return mean, log_std


class TwinQCritic(nn.Module):
    hidden_dims: tuple[int, ...]
    activation_name: str = "relu"

    @nn.compact
    def __call__(self, features: jax.Array, actions: jax.Array) -> jax.Array:
        x = jnp.concatenate([features, actions], axis=-1)
        q_values = []
        for critic_index in range(2):
            y = x
            for layer_index, width in enumerate(self.hidden_dims):
                y = nn.Dense(
                    width,
                    name=f"q{critic_index + 1}_dense_{layer_index}",
                )(y)
                y = activation(y, self.activation_name)
            q_values.append(
                nn.Dense(1, name=f"q{critic_index + 1}_out")(y)[..., 0]
            )
        return jnp.stack(q_values, axis=-1)


class ActorCritic(nn.Module):
    hidden_dims: tuple[int, ...]
    action_dim: int
    activation_name: str = "tanh"

    @nn.compact
    def __call__(
        self,
        features: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        policy = features.astype(jnp.float32)
        value = features.astype(jnp.float32)
        for index, width in enumerate(self.hidden_dims):
            policy = nn.Dense(width, name=f"policy_dense_{index}")(policy)
            policy = activation(policy, self.activation_name)
            value = nn.Dense(width, name=f"value_dense_{index}")(value)
            value = activation(value, self.activation_name)
        mean = nn.Dense(self.action_dim, name="policy_mean")(policy)
        log_std = self.param(
            "policy_log_std",
            nn.initializers.zeros_init(),
            (self.action_dim,),
        )
        log_std = jnp.broadcast_to(log_std, mean.shape)
        values = nn.Dense(1, name="value_out")(value)[..., 0]
        return mean, log_std, values


def normal_log_prob(
    value: jax.Array,
    mean: jax.Array,
    log_std: jax.Array,
) -> jax.Array:
    inv_std = jnp.exp(-log_std)
    elementwise = -0.5 * (
        jnp.square((value - mean) * inv_std) + 2.0 * log_std + LOG_TWO_PI
    )
    return elementwise.sum(axis=-1)


def normal_entropy(log_std: jax.Array) -> jax.Array:
    return (log_std + 0.5 * (1.0 + LOG_TWO_PI)).sum(axis=-1)


def squashed_normal_sample_and_log_prob(
    key: jax.Array,
    mean: jax.Array,
    log_std: jax.Array,
    *,
    deterministic: bool = False,
) -> tuple[jax.Array, jax.Array]:
    noise = jnp.zeros_like(mean) if deterministic else jax.random.normal(key, mean.shape)
    pre_tanh = mean + jnp.exp(log_std) * noise
    action = jnp.tanh(pre_tanh)
    correction = jnp.log(jnp.clip(1.0 - jnp.square(action), min=1e-6))
    log_prob = normal_log_prob(pre_tanh, mean, log_std) - correction.sum(axis=-1)
    return action, log_prob


def scale_unit_action(
    unit_action: jax.Array,
    action_low: jax.Array,
    action_high: jax.Array,
) -> jax.Array:
    return action_low + 0.5 * (unit_action + 1.0) * (action_high - action_low)


def unscale_action(
    action: jax.Array,
    action_low: jax.Array,
    action_high: jax.Array,
) -> jax.Array:
    scale = jnp.maximum(action_high - action_low, 1e-6)
    return jnp.clip(2.0 * (action - action_low) / scale - 1.0, -1.0, 1.0)


def next_observation_batch(batch: dict[str, Any], observation_keys) -> dict[str, Any]:
    next_batch = {}
    missing = []
    for key in observation_keys:
        next_key = f"{key}_tp1"
        if next_key not in batch:
            missing.append(next_key)
        else:
            next_batch[key] = batch[next_key]
    if missing:
        raise ValueError(
            "RL replay batches require next observations; missing keys: "
            + ", ".join(sorted(missing))
        )
    return next_batch


class JaxRLMethodBase(JaxMethodBase):
    """JAX method base with a shared trainable/frozen visual feature adapter."""

    def _setup_rl_features(self, model: RLModelSpec, *, seed: int) -> int:
        actor_spec = BCActorModelSpec(
            type="mlp_bottleneck_sequence",
            hidden_dims=model.hidden_dims,
            num_rnn_layers=1,
            rnn_hidden_size=128,
            keys_to_bottleneck=(),
            bottleneck_size=50,
            norm_after_bottleneck=True,
            tanh_after_bottleneck=True,
            output_sequence_network_type="mlp",
            output_sequence_length=self.action_sequence,
        )
        built, input_dim = _build_model(
            BCModelSpec(
                actor_model=actor_spec,
                encoder_model=model.encoder_model,
                view_fusion_model=model.view_fusion_model,
            ),
            observation_space=self.observation_space,
            action_space=self.action_space,
            encoder_jit=self._jit_enabled,
            encoder_seed=seed,
        )
        self.encoder = built.encoder_model
        self.view_fusion = built.view_fusion_model
        self.model_spec = model
        self._trainable_encoder = bool(
            self.encoder is not None
            and model.encoder_model is not None
            and model.encoder_model.trainable
        )
        self._time_feature_dim = (
            int(self.observation_space["time"].shape[-1])
            if "time" in self.observation_space
            else 0
        )
        self._init_cached_pixel_feature_key(self.__class__.__name__.lower())
        return int(input_dim) + self._time_feature_dim

    @property
    def _encoder_params(self):
        return None if not self._trainable_encoder else self.encoder.trainable_params

    def _prepare_rl_obs_inputs(self, batch_or_obs: dict[str, Any]):
        time_features = None
        if self._time_feature_dim:
            time_features = self._as_jax_array(
                batch_or_obs["time"],
                self.jnp.float32,
            )[:, -1].reshape((batch_or_obs["time"].shape[0], -1))
        if not self._trainable_encoder:
            features, _ = self._prepare_obs_features(batch_or_obs)
            if time_features is not None:
                features = self.jnp.concatenate([features, time_features], axis=-1)
            return features

        if self._has_cached_pixel_features(batch_or_obs):
            raise ValueError(
                "A trainable RL encoder requires raw RGB observations; disable "
                "replay.cache_frozen_image_features."
            )
        inputs = {}
        low_dim = self._extract_low_dim_batch(batch_or_obs)
        if low_dim is not None:
            inputs["low_dim"] = low_dim
        rgb, _ = self._extract_rgb_obs(batch_or_obs)
        if rgb is not None:
            inputs["rgb"] = rgb
        raymap = self._extract_raymap_obs(batch_or_obs)
        if raymap is not None:
            inputs["raymap"] = raymap
        camera_intrinsic, camera_c2w = self._extract_camera_param_obs(batch_or_obs)
        if camera_intrinsic is not None:
            inputs["camera_intrinsic"] = camera_intrinsic
            inputs["camera_c2w"] = camera_c2w
        if time_features is not None:
            inputs["time"] = time_features
        return inputs

    def _rl_features(self, encoder_params, obs_inputs, *, stop_gradient=False):
        if not self._trainable_encoder:
            features = obs_inputs
        else:
            values = []
            if "low_dim" in obs_inputs:
                values.append(obs_inputs["low_dim"])
            if "rgb" in obs_inputs:
                rgb_features = self.encoder.apply_trainable(
                    encoder_params,
                    obs_inputs["rgb"],
                    raymap_obs=obs_inputs.get("raymap", None),
                    camera_intrinsic_obs=obs_inputs.get("camera_intrinsic", None),
                    camera_c2w_obs=obs_inputs.get("camera_c2w", None),
                )
                values.append(self._fuse_multi_view(rgb_features))
            if "time" in obs_inputs:
                values.append(obs_inputs["time"])
            if not values:
                raise ValueError("RL method requires at least one observation feature.")
            features = values[0] if len(values) == 1 else jnp.concatenate(values, -1)
        return jax.lax.stop_gradient(features) if stop_gradient else features

    def _fuse_multi_view(self, rgb_feats):
        if rgb_feats is None:
            return None
        if self.view_fusion is not None:
            return self.view_fusion.apply({}, rgb_feats)
        return rgb_feats[:, 0]

    def _next_rl_obs_inputs(self, batch: dict[str, Any]):
        observation_keys = self.observation_space.keys()
        if self._has_cached_pixel_features(batch):
            observation_keys = tuple(
                key for key in observation_keys if key not in self._rgb_batch_keys
            )
        next_batch = next_observation_batch(batch, observation_keys)
        if self._has_cached_pixel_features(batch):
            cached_next_key = f"{self._cached_pixel_feature_key}_tp1"
            if cached_next_key not in batch:
                raise ValueError(
                    "RL replay batches with cached vision features require "
                    f"'{cached_next_key}'."
                )
            next_batch[self._cached_pixel_feature_key] = batch[cached_next_key]
        return self._prepare_rl_obs_inputs(next_batch)

    def _action_bounds(self) -> tuple[jax.Array, jax.Array]:
        low = np.asarray(self.action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(self.action_space.high, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
            raise ValueError("Continuous-control RL requires finite action bounds.")
        if np.any(high <= low):
            raise ValueError("Every action upper bound must exceed its lower bound.")
        return jnp.asarray(low), jnp.asarray(high)


__all__ = [
    "ActorCritic",
    "GaussianActor",
    "JaxRLMethodBase",
    "MLP",
    "RLModelSpec",
    "TwinQCritic",
    "normal_entropy",
    "normal_log_prob",
    "rl_model_spec_from_cfg",
    "scale_unit_action",
    "squashed_normal_sample_and_log_prob",
    "unscale_action",
]
