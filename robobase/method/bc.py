"""Behavioural Cloning: spec dataclasses, config parsing, and JAX implementation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Optional

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig

from robobase.method.bc_runtime import bc_actor_input_shapes, bc_observation_layout
from robobase.method.jax_base import JaxMethodBase
from robobase.models.encoder import JaxResNetEncoder
from robobase.models.fully_connected import JaxMLPWithSequenceOutput
from robobase.models.fusion import JaxFusionMultiCamFeature
from robobase.replay_buffer.replay_buffer import ReplayBuffer

# ---------------------------------------------------------------------------
# Spec dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BCActorModelSpec:
    type: str
    hidden_dims: tuple[int, ...]
    num_rnn_layers: int
    rnn_hidden_size: int
    keys_to_bottleneck: tuple[str, ...]
    bottleneck_size: int
    norm_after_bottleneck: bool
    tanh_after_bottleneck: bool
    output_sequence_network_type: str
    output_sequence_length: int


@dataclass(frozen=True)
class BCEncoderModelSpec:
    type: str
    model: str
    trainable: bool = False
    pretrained: bool = True
    use_plucker: bool = False
    plucker_hidden_channels: int = 64
    plucker_identity_init: bool = False


@dataclass(frozen=True)
class BCViewFusionModelSpec:
    type: str
    mode: str


@dataclass(frozen=True)
class BCModelSpec:
    actor_model: BCActorModelSpec
    encoder_model: BCEncoderModelSpec | None
    view_fusion_model: BCViewFusionModelSpec | None


@dataclass(frozen=True)
class BCSpec:
    lr: float
    adaptive_lr: bool
    num_train_steps: int
    actor_grad_clip: float | None
    model: BCModelSpec


# ---------------------------------------------------------------------------
# Config parsing helpers
# ---------------------------------------------------------------------------


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
    if target_to_type is not None:
        target = str(cfg.get("_target_", "")).strip()
        if target in target_to_type:
            return target_to_type[target]
    return default


def bc_model_spec_from_cfg(cfg: DictConfig) -> BCModelSpec:
    method_cfg = cfg.method
    actor_model_cfg = method_cfg.get("actor_model", None)
    if actor_model_cfg is None:
        raise ValueError("BC requires an actor_model config.")

    actor_model_type = _config_type(
        actor_model_cfg,
        default="mlp_bottleneck_sequence",
        target_to_type={
            "robobase.models.MLPWithBottleneckFeaturesAndSequenceOutput": "mlp_bottleneck_sequence",
            "robobase.models.fully_connected.JaxMLPWithSequenceOutput": "mlp_bottleneck_sequence",
        },
    )
    actor_model_spec = BCActorModelSpec(
        type=actor_model_type,
        hidden_dims=tuple(
            int(v)
            for v in actor_model_cfg.get(
                "hidden_dims",
                actor_model_cfg.get("mlp_nodes", (256, 256)),
            )
        ),
        num_rnn_layers=int(actor_model_cfg.get("num_rnn_layers", 1)),
        rnn_hidden_size=int(actor_model_cfg.get("rnn_hidden_size", 128)),
        keys_to_bottleneck=tuple(str(v) for v in actor_model_cfg.get("keys_to_bottleneck", ())),
        bottleneck_size=int(actor_model_cfg.get("bottleneck_size", 50)),
        norm_after_bottleneck=bool(actor_model_cfg.get("norm_after_bottleneck", True)),
        tanh_after_bottleneck=bool(actor_model_cfg.get("tanh_after_bottleneck", True)),
        output_sequence_network_type=str(
            actor_model_cfg.get("output_sequence_network_type", "rnn")
        ).lower(),
        output_sequence_length=int(
            actor_model_cfg.get("output_sequence_length", cfg.action_sequence)
        ),
    )

    encoder_model_cfg = method_cfg.get("encoder_model", None)
    encoder_model_spec = None
    if encoder_model_cfg is not None:
        encoder_model_type = _config_type(
            encoder_model_cfg,
            default="resnet",
            target_to_type={
                "robobase.models.encoder.JaxResNetEncoder": "resnet",
            },
        )
        encoder_model_spec = BCEncoderModelSpec(
            type=encoder_model_type,
            model=str(encoder_model_cfg.get("model", "resnet18")),
            trainable=bool(encoder_model_cfg.get("trainable", False)),
            # The original JAX ResNet path was always pretrained. Treat a
            # missing key as the legacy behavior so old run configs and their
            # snapshots retain the frozen batch statistics they were trained
            # with. Random initialization remains available explicitly.
            pretrained=bool(encoder_model_cfg.get("pretrained", True)),
            use_plucker=bool(encoder_model_cfg.get("use_plucker", False)),
            plucker_hidden_channels=int(
                encoder_model_cfg.get("plucker_hidden_channels", 64)
            ),
            plucker_identity_init=bool(
                encoder_model_cfg.get("plucker_identity_init", False)
            ),
        )

    view_fusion_model_cfg = method_cfg.get("view_fusion_model", None)
    view_fusion_model_spec = None
    if view_fusion_model_cfg is not None:
        view_fusion_model_type = _config_type(
            view_fusion_model_cfg,
            default="multicam_feature",
            target_to_type={
                "robobase.models.fusion.JaxFusionMultiCamFeature": "multicam_feature",
            },
        )
        view_fusion_model_spec = BCViewFusionModelSpec(
            type=view_fusion_model_type,
            mode=str(view_fusion_model_cfg.get("mode", "flatten")).lower(),
        )

    return BCModelSpec(
        actor_model=actor_model_spec,
        encoder_model=encoder_model_spec,
        view_fusion_model=view_fusion_model_spec,
    )


def bc_spec_from_cfg(cfg: DictConfig) -> BCSpec:
    return BCSpec(
        lr=float(cfg.method.lr),
        adaptive_lr=bool(cfg.method.adaptive_lr),
        num_train_steps=int(cfg.method.num_train_steps),
        actor_grad_clip=(
            None
            if cfg.method.actor_grad_clip is None
            else float(cfg.method.actor_grad_clip)
        ),
        model=bc_model_spec_from_cfg(cfg),
    )


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _BuiltBCModel:
    actor_model: JaxMLPWithSequenceOutput
    encoder_model: JaxResNetEncoder | None
    view_fusion_model: JaxFusionMultiCamFeature | None


def _build_model(
    model_spec: BCModelSpec,
    *,
    observation_space: spaces.Dict,
    action_space: spaces.Box,
    encoder_jit: bool,
) -> tuple[_BuiltBCModel, int]:
    obs_layout = bc_observation_layout(observation_space)
    actor_spec = model_spec.actor_model
    if actor_spec.type != "mlp_bottleneck_sequence":
        raise NotImplementedError(
            f"Unsupported BC actor model type '{actor_spec.type}'."
        )
    if actor_spec.output_sequence_length != int(action_space.shape[0]):
        raise ValueError(
            "BC actor model output_sequence_length does not match the action space."
        )
    if actor_spec.output_sequence_network_type not in {"rnn", "mlp"}:
        raise ValueError(
            f"Unsupported BC output_sequence_network_type "
            f"'{actor_spec.output_sequence_network_type}'."
        )

    encoder_model = None
    view_fusion_model = None

    if obs_layout.use_pixels:
        if model_spec.encoder_model is None:
            raise ValueError("Pixel BC requires encoder_model in the model spec.")
        if model_spec.encoder_model.type != "resnet":
            raise NotImplementedError(
                f"Unsupported BC encoder model type '{model_spec.encoder_model.type}'."
            )
        if obs_layout.rgb_input_shape is None:
            raise ValueError("Pixel BC expected a valid RGB input shape.")
        if model_spec.encoder_model.use_plucker:
            if not model_spec.encoder_model.trainable:
                raise ValueError(
                    "BC encoder_model.use_plucker=true requires "
                    "encoder_model.trainable=true so the Plucker adapter is trained."
                )
            if not obs_layout.has_camera_conditioning:
                raise ValueError(
                    "BC encoder_model.use_plucker=true requires raymap or camera "
                    "parameter observations paired with every RGB observation."
                )
        encoder_model = JaxResNetEncoder(
            input_shape=obs_layout.rgb_input_shape,
            model=model_spec.encoder_model.model,
            jit=encoder_jit,
            pretrained=model_spec.encoder_model.pretrained,
            use_plucker=model_spec.encoder_model.use_plucker,
            plucker_hidden_channels=model_spec.encoder_model.plucker_hidden_channels,
            plucker_identity_init=model_spec.encoder_model.plucker_identity_init,
        )
        if obs_layout.use_multicam_fusion:
            if model_spec.view_fusion_model is None:
                raise ValueError(
                    "Multi-camera pixel BC requires view_fusion_model in the model spec."
                )
            if model_spec.view_fusion_model.type != "multicam_feature":
                raise NotImplementedError(
                    f"Unsupported BC view fusion model type "
                    f"'{model_spec.view_fusion_model.type}'."
                )
            view_fusion_model = JaxFusionMultiCamFeature.from_input_shape(
                input_shape=encoder_model.output_shape,
                mode=model_spec.view_fusion_model.mode,
            )
    elif model_spec.encoder_model is not None and model_spec.encoder_model.type != "resnet":
        raise NotImplementedError(
            f"Unsupported BC encoder model type '{model_spec.encoder_model.type}'."
        )

    rgb_latent_size = 0
    if encoder_model is not None:
        rgb_latent_size = int(
            view_fusion_model.output_shape[-1]
            if view_fusion_model is not None
            else encoder_model.output_shape[-1]
        )

    actor_input_shapes = bc_actor_input_shapes(
        low_dim_size=obs_layout.low_dim_size,
        rgb_latent_size=rgb_latent_size,
        frame_stack_on_channel=True,
        time_dim=obs_layout.time_dim,
    )
    input_dim = int(np.prod(actor_input_shapes["features"]))

    return _BuiltBCModel(
        actor_model=JaxMLPWithSequenceOutput(
            hidden_dims=actor_spec.hidden_dims,
            action_sequence=int(action_space.shape[0]),
            action_dim=int(action_space.shape[1]),
            output_sequence_network_type=actor_spec.output_sequence_network_type,
        ),
        encoder_model=encoder_model,
        view_fusion_model=view_fusion_model,
    ), input_dim


# ---------------------------------------------------------------------------
# BC method implementation
# ---------------------------------------------------------------------------


class BC(JaxMethodBase):
    """Behavioural Cloning backed by JAX with Flax models."""

    def __init__(
        self,
        lr: float,
        adaptive_lr: bool,
        num_train_steps: int,
        model: BCModelSpec,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        num_train_envs: int,
        num_eval_envs: int,
        replay_alpha: float,
        replay_beta: float,
        frame_stack_on_channel: bool,
        intrinsic_reward_module=None,
        actor_grad_clip: Optional[float] = None,
        jit: bool = True,
        platform: str | None = None,
        seed: int = 0,
        is_rl: bool = False,
        use_ema: bool = False,
        update_block_every_steps: int = 1,
    ):
        super().__init__(
            lr=lr,
            adaptive_lr=adaptive_lr,
            num_train_steps=num_train_steps,
            observation_space=observation_space,
            action_space=action_space,
            num_train_envs=num_train_envs,
            num_eval_envs=num_eval_envs,
            replay_alpha=replay_alpha,
            replay_beta=replay_beta,
            frame_stack_on_channel=frame_stack_on_channel,
            intrinsic_reward_module=intrinsic_reward_module,
            actor_grad_clip=actor_grad_clip,
            jit=jit,
            platform=platform,
            seed=seed,
            is_rl=is_rl,
            use_ema=use_ema,
            update_block_every_steps=update_block_every_steps,
        )

        self.model_spec = model
        self._init_cached_pixel_feature_key("bc")

        built_model, input_dim = _build_model(
            self.model_spec,
            observation_space=observation_space,
            action_space=action_space,
            encoder_jit=jit,
        )
        self.actor_model = built_model.actor_model
        self.encoder = built_model.encoder_model
        self.view_fusion = built_model.view_fusion_model
        self._trainable_encoder = (
            self.encoder is not None
            and self.model_spec.encoder_model is not None
            and bool(self.model_spec.encoder_model.trainable)
        )
        self.rgb_latent_size = 0
        if self.encoder is not None:
            self.rgb_latent_size = int(
                self.view_fusion.output_shape[-1]
                if self.view_fusion is not None
                else self.encoder.output_shape[-1]
            )
        self.actor_input_shapes = bc_actor_input_shapes(
            low_dim_size=self.low_dim_size,
            rgb_latent_size=self.rgb_latent_size,
            frame_stack_on_channel=self.frame_stack_on_channel,
            time_dim=self.time_dim,
        )

        dummy_input = jnp.zeros((1, input_dim), dtype=jnp.float32)
        actor_params = self.actor_model.init(self.rng_key, dummy_input)
        self.params = (
            {"actor": actor_params, "encoder": self.encoder.trainable_params}
            if self._trainable_encoder
            else actor_params
        )

        learning_rate = lr
        if adaptive_lr:
            learning_rate = self.optax.cosine_decay_schedule(
                init_value=lr, decay_steps=self.num_train_steps,
            )
        transforms = []
        if actor_grad_clip is not None:
            transforms.append(self.optax.clip_by_global_norm(float(actor_grad_clip)))
        transforms.append(self.optax.adam(learning_rate))
        self.optimizer = self.optax.chain(*transforms)
        self.opt_state = self.optimizer.init(self.params)

        update_fn = self._build_update_fn()
        predict_fn = self._predict_impl
        if self._jit_enabled:
            update_fn = jax.jit(update_fn)
            predict_fn = jax.jit(predict_fn)
        self._update_impl = update_fn
        self._predict = predict_fn

    def _actor_params(self, params):
        if self._trainable_encoder:
            return params["actor"]
        return params

    def _prepare_trainable_obs_inputs(self, batch_or_obs: dict):
        inputs = {}
        low_dim_obs = self._extract_low_dim_batch(batch_or_obs)
        if low_dim_obs is not None:
            inputs["low_dim"] = low_dim_obs
        if self.use_pixels:
            if self._has_cached_pixel_features(batch_or_obs):
                raise ValueError(
                    "Trainable JAX BC encoder requires raw RGB observations; "
                    "disable replay.cache_frozen_image_features."
                )
            rgb_obs, _ = self._extract_rgb_obs(batch_or_obs)
            inputs["rgb"] = rgb_obs
            raymap_obs = self._extract_raymap_obs(batch_or_obs)
            if raymap_obs is not None:
                inputs["raymap"] = raymap_obs
            camera_intrinsic_obs, camera_c2w_obs = self._extract_camera_param_obs(
                batch_or_obs
            )
            if camera_intrinsic_obs is not None:
                inputs["camera_intrinsic"] = camera_intrinsic_obs
                inputs["camera_c2w"] = camera_c2w_obs
        return inputs

    def _features_from_inputs(self, params, obs_inputs):
        if not self._trainable_encoder:
            return obs_inputs
        features = []
        if "low_dim" in obs_inputs:
            features.append(obs_inputs["low_dim"])
        if "rgb" in obs_inputs:
            rgb_feats = self.encoder.apply_trainable(
                params["encoder"],
                obs_inputs["rgb"],
                raymap_obs=obs_inputs.get("raymap", None),
                camera_intrinsic_obs=obs_inputs.get("camera_intrinsic", None),
                camera_c2w_obs=obs_inputs.get("camera_c2w", None),
            )
            features.append(self._fuse_multi_view(rgb_feats))
        if not features:
            raise ValueError("BC requires at least one observation feature.")
        if len(features) == 1:
            return features[0]
        return self.jnp.concatenate(features, axis=-1)

    def _predict_impl(self, params, obs_inputs):
        obs_features = self._features_from_inputs(params, obs_inputs)
        return self.actor_model.apply(self._actor_params(params), obs_features)

    def _build_update_fn(self):
        optimizer = self.optimizer
        optax = self.optax

        def update_fn(params, opt_state, obs_inputs, actions, loss_coeff, action_pad_mask):
            def loss_fn(current_params):
                obs_features = self._features_from_inputs(current_params, obs_inputs)
                action_pred = self.actor_model.apply(
                    self._actor_params(current_params),
                    obs_features,
                )
                per_token_mse = jnp.square(action_pred - actions)
                reduce_dims = tuple(range(1, per_token_mse.ndim))
                if action_pad_mask is None:
                    mse_loss = per_token_mse.mean(axis=reduce_dims)
                else:
                    valid_mask = jnp.logical_not(action_pad_mask).astype(per_token_mse.dtype)
                    while valid_mask.ndim < per_token_mse.ndim:
                        valid_mask = valid_mask[..., None]
                    valid_mask = jnp.broadcast_to(valid_mask, per_token_mse.shape)
                    masked_loss = per_token_mse * valid_mask
                    denom = jnp.clip(valid_mask.sum(axis=reduce_dims), min=1.0)
                    mse_loss = masked_loss.sum(axis=reduce_dims) / denom
                return (mse_loss * loss_coeff).mean(), mse_loss

            (loss, mse_loss), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            new_pri = jnp.sqrt(mse_loss + 1e-10)
            max_pri = jnp.max(new_pri)
            return new_params, new_opt_state, loss, new_pri / jnp.where(max_pri > 0, max_pri, 1.0)

        return update_fn

    def _fuse_multi_view(self, rgb_feats):
        if rgb_feats is None:
            return None
        rgb_feats = self.jnp.asarray(rgb_feats, dtype=self.jnp.float32)
        if self.view_fusion is not None:
            return self.view_fusion.apply({}, rgb_feats)
        return rgb_feats[:, 0]

    def act(self, observations: dict, step: int, eval_mode: bool):
        del step, eval_mode
        if self._trainable_encoder:
            obs_inputs = self._prepare_trainable_obs_inputs(observations)
        else:
            obs_inputs, _ = self._prepare_obs_features(observations)
        actions = self._predict(self.params, obs_inputs)
        self._block(actions)
        return np.asarray(jax.device_get(actions), dtype=np.float32)

    def update(
        self,
        replay_iter: Iterator[dict],
        step: int,
        replay_buffer: ReplayBuffer = None,
    ) -> dict[str, np.ndarray]:
        del step
        batch = next(replay_iter)
        actions = self._as_jax_array(batch["action"], self.jnp.float32)
        action_pad_mask = self._extract_action_pad_mask(batch)
        loss_coeff = self._loss_weights(batch)
        if self._trainable_encoder:
            obs_inputs = self._prepare_trainable_obs_inputs(batch)
            metrics = {}
        else:
            obs_inputs, metrics = self._prepare_obs_features(batch)

        start_time = time.perf_counter()
        self.params, self.opt_state, actor_loss, new_priority = self._update_impl(
            self.params,
            self.opt_state,
            obs_inputs,
            actions,
            loss_coeff,
            action_pad_mask,
        )
        uses_priorities = self._uses_replay_priorities(replay_buffer)
        if self._should_block_update(uses_priorities):
            if uses_priorities:
                self._block(actor_loss, new_priority)
            else:
                self._block(actor_loss)
        elapsed = time.perf_counter() - start_time
        self._update_step_count += 1

        if uses_priorities:
            new_priority_np = np.asarray(
                jax.device_get(new_priority),
                dtype=np.float32,
            )
            self._maybe_update_priorities(replay_buffer, batch, new_priority_np)
        self._maybe_log_update_metrics(metrics, actor_loss, obs_inputs, elapsed)
        self._first_update_completed = True
        return metrics


__all__ = [
    "BCActorModelSpec",
    "BCEncoderModelSpec",
    "BCModelSpec",
    "BCSpec",
    "BCViewFusionModelSpec",
    "BC",
    "bc_model_spec_from_cfg",
    "bc_spec_from_cfg",
]
