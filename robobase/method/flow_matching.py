"""Rectified Flow / Flow Matching imitation-learning method in JAX."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Optional

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig

from robobase.language import lang_feature_rows, lang_token_rows, tokens_to_feature_jax
from robobase.method.bc import BCEncoderModelSpec, BCViewFusionModelSpec
from robobase.method.bc_runtime import bc_observation_layout
from robobase.method.jax_base import JaxMethodBase
from robobase.models.backbone import (
    DiffusionBackboneSpec,
    backbone_spec_from_cfg,
    build_diffusion_backbone,
    canonical_backbone_type,
)
from robobase.models.encoder import JaxResNetEncoder
from robobase.models.fusion import JaxFusionMultiCamFeature
from robobase.replay_buffer.replay_buffer import ReplayBuffer


FlowMatchingBackboneSpec = DiffusionBackboneSpec


@dataclass(frozen=True)
class FlowMatchingModelSpec:
    backbone: FlowMatchingBackboneSpec
    encoder_model: BCEncoderModelSpec | None
    view_fusion_model: BCViewFusionModelSpec | None
    use_lang_cond: bool = False
    lang_feature_dim: int = 512


@dataclass(frozen=True)
class FlowMatchingSpec:
    lr: float
    adaptive_lr: bool
    lr_schedule: str
    num_train_steps: int
    actor_grad_clip: float | None
    objective_type: str
    num_flow_steps: int
    sampler: str
    sample_schedule: str
    train_time_schedule: str
    horizon_dropout_lengths: tuple[int, ...] | None
    horizon_dropout_probs: tuple[float, ...] | None
    horizon_loss_weights: tuple[float, ...] | None
    use_ema: bool
    ema_decay: float
    weight_decay: float
    model: FlowMatchingModelSpec


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


def _sample_schedule_jump_point(sample_schedule: str) -> float | None:
    if sample_schedule in {"uniform", "linear"}:
        return None
    for prefix in ("front_loaded_", "nonuniform_"):
        if sample_schedule.startswith(prefix):
            value = sample_schedule[len(prefix):].replace("p", ".")
            try:
                jump_point = float(value)
            except ValueError:
                break
            if 0.0 < jump_point < 1.0:
                return jump_point
            break
    raise NotImplementedError(
        f"Unsupported Flow Matching sample_schedule '{sample_schedule}'."
    )


def _parse_schedule_float(value: str) -> float:
    return float(value.replace("p", "."))


def _train_time_beta_params(train_time_schedule: str) -> tuple[float, float] | None:
    if train_time_schedule in {"uniform", "linear"}:
        return None
    if train_time_schedule in {"beta", "u_shaped", "u_shaped_beta"}:
        return 0.5, 0.5
    for prefix in ("beta_", "u_shaped_beta_"):
        if not train_time_schedule.startswith(prefix):
            continue
        parts = train_time_schedule[len(prefix):].split("_")
        try:
            if len(parts) == 1:
                alpha = beta = _parse_schedule_float(parts[0])
            elif len(parts) == 2:
                alpha = _parse_schedule_float(parts[0])
                beta = _parse_schedule_float(parts[1])
            else:
                break
        except ValueError:
            break
        if alpha > 0.0 and beta > 0.0:
            return alpha, beta
        break
    raise NotImplementedError(
        f"Unsupported Flow Matching train_time_schedule '{train_time_schedule}'."
    )


def flow_matching_model_spec_from_cfg(cfg: DictConfig) -> FlowMatchingModelSpec:
    method_cfg = cfg.method
    backbone_cfg = method_cfg.get("backbone", None)
    if backbone_cfg is None:
        raise ValueError("Flow Matching requires a backbone config.")

    backbone_spec = backbone_spec_from_cfg(
        backbone_cfg,
        default_type="fully_connected",
        default_sequence_length=int(cfg.action_sequence),
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

    return FlowMatchingModelSpec(
        backbone=backbone_spec,
        encoder_model=encoder_model_spec,
        view_fusion_model=view_fusion_model_spec,
        use_lang_cond=bool(method_cfg.get("use_lang_cond", False)),
        lang_feature_dim=int(method_cfg.get("lang_feature_dim", 512)),
    )


def flow_matching_spec_from_cfg(cfg: DictConfig) -> FlowMatchingSpec:
    objective_cfg = cfg.method.get("objective", None)
    objective_type = (
        str(objective_cfg.get("type", "rectified_flow")).lower()
        if objective_cfg is not None
        else "rectified_flow"
    )
    num_flow_steps = (
        int(objective_cfg.get("num_flow_steps"))
        if objective_cfg is not None and objective_cfg.get("num_flow_steps") is not None
        else int(cfg.method.get("num_flow_steps", 5))
    )
    sampler = (
        str(objective_cfg.get("sampler", "euler")).lower()
        if objective_cfg is not None
        else str(cfg.method.get("sampler", "euler")).lower()
    )
    sample_schedule = (
        str(
            objective_cfg.get(
                "sample_schedule",
                cfg.method.get("sample_schedule", "uniform"),
            )
        ).lower()
        if objective_cfg is not None
        else str(cfg.method.get("sample_schedule", "uniform")).lower()
    )
    train_time_schedule = (
        str(
            objective_cfg.get(
                "train_time_schedule",
                cfg.method.get("train_time_schedule", "uniform"),
            )
        ).lower()
        if objective_cfg is not None
        else str(cfg.method.get("train_time_schedule", "uniform")).lower()
    )
    horizon_dropout_lengths = cfg.method.get("horizon_dropout_lengths", None)
    horizon_dropout_probs = cfg.method.get("horizon_dropout_probs", None)
    if horizon_dropout_lengths is not None:
        horizon_dropout_lengths = tuple(int(v) for v in horizon_dropout_lengths)
        if len(horizon_dropout_lengths) == 0:
            horizon_dropout_lengths = None
    if horizon_dropout_probs is not None:
        horizon_dropout_probs = tuple(float(v) for v in horizon_dropout_probs)
        if len(horizon_dropout_probs) == 0:
            horizon_dropout_probs = None
    horizon_loss_weights = cfg.method.get("horizon_loss_weights", None)
    if horizon_loss_weights is not None:
        horizon_loss_weights = tuple(float(v) for v in horizon_loss_weights)
        if len(horizon_loss_weights) == 0:
            horizon_loss_weights = None
    return FlowMatchingSpec(
        lr=float(cfg.method.lr),
        adaptive_lr=bool(cfg.method.adaptive_lr),
        lr_schedule=str(cfg.method.get("lr_schedule", "cosine")),
        num_train_steps=int(cfg.method.num_train_steps),
        actor_grad_clip=(
            None
            if cfg.method.actor_grad_clip is None
            else float(cfg.method.actor_grad_clip)
        ),
        objective_type=objective_type,
        num_flow_steps=num_flow_steps,
        sampler=sampler,
        sample_schedule=sample_schedule,
        train_time_schedule=train_time_schedule,
        horizon_dropout_lengths=horizon_dropout_lengths,
        horizon_dropout_probs=horizon_dropout_probs,
        horizon_loss_weights=horizon_loss_weights,
        use_ema=bool(cfg.method.get("use_ema", False)),
        ema_decay=float(cfg.method.get("ema_decay", 0.9999)),
        weight_decay=float(cfg.method.get("weight_decay", 1e-6)),
        model=flow_matching_model_spec_from_cfg(cfg),
    )


@dataclass(frozen=True)
class _BuiltFlowMatchingModel:
    backbone_model: object
    encoder_model: JaxResNetEncoder | None
    view_fusion_model: JaxFusionMultiCamFeature | None


def _build_encoder_and_fusion(
    *,
    model_spec: FlowMatchingModelSpec,
    observation_space: spaces.Dict,
    encoder_jit: bool,
) -> tuple[JaxResNetEncoder | None, JaxFusionMultiCamFeature | None, int]:
    obs_layout = bc_observation_layout(observation_space)
    encoder_model = None
    view_fusion_model = None

    if obs_layout.use_pixels:
        if model_spec.encoder_model is None:
            raise ValueError(
                "Pixel Flow Matching requires encoder_model in the model spec."
            )
        if model_spec.encoder_model.type != "resnet":
            raise NotImplementedError(
                "Unsupported Flow Matching encoder model type "
                f"'{model_spec.encoder_model.type}'."
            )
        if obs_layout.rgb_input_shape is None:
            raise ValueError("Pixel Flow Matching expected a valid RGB input shape.")
        if model_spec.encoder_model.use_plucker:
            if not model_spec.encoder_model.trainable:
                raise ValueError(
                    "Flow Matching encoder_model.use_plucker=true requires "
                    "encoder_model.trainable=true so the Plucker adapter is trained."
                )
            if not obs_layout.has_camera_conditioning:
                raise ValueError(
                    "Flow Matching encoder_model.use_plucker=true requires raymap or "
                    "camera parameter observations paired with every RGB observation."
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
                    "Multi-camera pixel Flow Matching requires view_fusion_model."
                )
            if model_spec.view_fusion_model.type != "multicam_feature":
                raise NotImplementedError(
                    "Unsupported Flow Matching view fusion model type "
                    f"'{model_spec.view_fusion_model.type}'."
                )
            view_fusion_model = JaxFusionMultiCamFeature.from_input_shape(
                input_shape=encoder_model.output_shape,
                mode=model_spec.view_fusion_model.mode,
            )
    elif model_spec.encoder_model is not None and model_spec.encoder_model.type != "resnet":
        raise NotImplementedError(
            "Unsupported Flow Matching encoder model type "
            f"'{model_spec.encoder_model.type}'."
        )

    rgb_latent_size = 0
    if encoder_model is not None:
        if view_fusion_model is not None:
            rgb_latent_size = int(view_fusion_model.output_shape[-1])
        else:
            rgb_latent_size = int(encoder_model.output_shape[-1])
    return encoder_model, view_fusion_model, rgb_latent_size


def _build_model(
    model_spec: FlowMatchingModelSpec,
    *,
    observation_space: spaces.Dict,
    action_space: spaces.Box,
    encoder_jit: bool,
) -> tuple[_BuiltFlowMatchingModel, int]:
    obs_layout = bc_observation_layout(observation_space)
    backbone_type = canonical_backbone_type(model_spec.backbone.type)
    encoder_model, view_fusion_model, rgb_latent_size = _build_encoder_and_fusion(
        model_spec=model_spec,
        observation_space=observation_space,
        encoder_jit=encoder_jit,
    )
    feature_dim = int(obs_layout.low_dim_size + rgb_latent_size)
    if model_spec.use_lang_cond:
        feature_dim += int(model_spec.lang_feature_dim)
    if backbone_type == "transformer" and not obs_layout.use_pixels:
        low_dim_state_spec = observation_space.spaces.get("low_dim_state")
        if low_dim_state_spec is not None:
            feature_dim = int(low_dim_state_spec.shape[-1])
            if model_spec.use_lang_cond:
                feature_dim += int(model_spec.lang_feature_dim)
    return _BuiltFlowMatchingModel(
        backbone_model=build_diffusion_backbone(
            model_spec.backbone,
            action_dim=int(action_space.shape[1]),
            sequence_length=int(action_space.shape[0]),
            condition_dim=feature_dim,
        ),
        encoder_model=encoder_model,
        view_fusion_model=view_fusion_model,
    ), feature_dim


class FlowMatching(JaxMethodBase):
    """Rectified Flow method using JAX/Flax backbones."""

    def __init__(
        self,
        lr: float,
        adaptive_lr: bool,
        num_train_steps: int,
        num_flow_steps: int,
        model: FlowMatchingModelSpec,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        num_train_envs: int,
        num_eval_envs: int,
        replay_alpha: float,
        replay_beta: float,
        frame_stack_on_channel: bool,
        intrinsic_reward_module=None,
        actor_grad_clip: Optional[float] = None,
        lr_schedule: str = "cosine",
        jit: bool = True,
        platform: str | None = None,
        seed: int = 0,
        is_rl: bool = False,
        use_ema: bool = False,
        ema_decay: float = 0.9999,
        weight_decay: float = 1e-6,
        objective_type: str = "rectified_flow",
        sampler: str = "euler",
        sample_schedule: str = "uniform",
        train_time_schedule: str = "uniform",
        horizon_dropout_lengths: tuple[int, ...] | None = None,
        horizon_dropout_probs: tuple[float, ...] | None = None,
        horizon_loss_weights: tuple[float, ...] | None = None,
        update_block_every_steps: int = 1,
    ):
        if not frame_stack_on_channel:
            raise NotImplementedError(
                "frame_stack_on_channel must be true for Flow Matching policies."
            )
        if str(objective_type).lower() not in {"rectified_flow", "flow_matching"}:
            raise NotImplementedError(
                "FlowMatching currently supports the Rectified Flow objective."
            )
        if str(sampler).lower() != "euler":
            raise NotImplementedError(
                "FlowMatching currently supports Euler sampling."
            )
        sample_schedule = str(sample_schedule).lower()
        sample_schedule_jump_point = _sample_schedule_jump_point(sample_schedule)
        train_time_schedule = str(train_time_schedule).lower()
        train_time_beta_params = _train_time_beta_params(train_time_schedule)
        if int(num_flow_steps) < 1:
            raise ValueError("num_flow_steps must be >= 1.")

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

        self.objective_type = str(objective_type).lower()
        self.sampler = str(sampler).lower()
        self.sample_schedule = sample_schedule
        self._sample_schedule_jump_point = sample_schedule_jump_point
        self.train_time_schedule = train_time_schedule
        self._train_time_beta_params = train_time_beta_params
        self._horizon_dropout_lengths = None
        self._horizon_dropout_probs = None
        self._horizon_loss_weights = None
        if horizon_dropout_lengths is not None:
            lengths = tuple(int(v) for v in horizon_dropout_lengths)
            if any(v < 1 or v > self.action_sequence for v in lengths):
                raise ValueError(
                    "horizon_dropout_lengths must be between 1 and action_sequence."
                )
            if horizon_dropout_probs is None:
                probs = tuple([1.0 / len(lengths)] * len(lengths))
            else:
                probs = tuple(float(v) for v in horizon_dropout_probs)
                if len(probs) != len(lengths):
                    raise ValueError(
                        "horizon_dropout_probs must match horizon_dropout_lengths."
                    )
                if any(v < 0.0 for v in probs) or sum(probs) <= 0.0:
                    raise ValueError(
                        "horizon_dropout_probs must be non-negative and sum > 0."
                    )
                prob_sum = sum(probs)
                probs = tuple(v / prob_sum for v in probs)
            self._horizon_dropout_lengths = jnp.asarray(lengths, dtype=jnp.int32)
            self._horizon_dropout_probs = jnp.asarray(probs, dtype=jnp.float32)
        if horizon_loss_weights is not None:
            weights = tuple(float(v) for v in horizon_loss_weights)
            if len(weights) != self.action_sequence:
                raise ValueError(
                    "horizon_loss_weights must have length equal to action_sequence."
                )
            if any(v < 0.0 for v in weights) or sum(weights) <= 0.0:
                raise ValueError(
                    "horizon_loss_weights must be non-negative and sum > 0."
                )
            weight_sum = sum(weights)
            weights = tuple(v / weight_sum for v in weights)
            self._horizon_loss_weights = jnp.asarray(weights, dtype=jnp.float32)
        self.num_flow_steps = int(num_flow_steps)
        # The diffusion-style SinusoidalPosEmb is tuned for the integer step
        # range (~[0, 1000]); CleanDiffuser feeds the integer diffusion-step
        # index to the network, not the continuous t in [0, 1]. Feeding raw
        # t in [0, 1] squashes the time embedding (it becomes nearly constant),
        # so the network cannot read the ODE time. Keep continuous t for the
        # interpolation/integration, but scale the *network* time input.
        self._time_scale = 1000.0
        self.model_spec = model
        self.use_lang_cond = bool(model.use_lang_cond)
        self.lang_feature_dim = int(model.lang_feature_dim) if self.use_lang_cond else 0
        self._condition_as_sequence = (
            canonical_backbone_type(self.model_spec.backbone.type) == "transformer"
        )
        self._init_cached_pixel_feature_key("flow_matching")

        built_model, feature_dim = _build_model(
            self.model_spec,
            observation_space=observation_space,
            action_space=action_space,
            encoder_jit=jit,
        )
        self.actor_model = built_model.backbone_model
        self.encoder = built_model.encoder_model
        self.view_fusion = built_model.view_fusion_model
        self._trainable_encoder = (
            self.encoder is not None
            and self.model_spec.encoder_model is not None
            and bool(self.model_spec.encoder_model.trainable)
        )
        self.rgb_latent_size = max(
            0, feature_dim - self.low_dim_size - self.lang_feature_dim
        )

        self.rng_key, init_key = jax.random.split(self.rng_key)
        dummy_actions = jnp.zeros(
            (1, self.action_sequence, self.action_dim), dtype=jnp.float32
        )
        dummy_timesteps = jnp.zeros((1,), dtype=jnp.float32)
        if feature_dim > 0 and self._condition_as_sequence and not self.use_pixels:
            dummy_features = jnp.zeros(
                (1, self.time_dim, feature_dim), dtype=jnp.float32
            )
        else:
            dummy_features = (
                jnp.zeros((1, feature_dim), dtype=jnp.float32)
                if feature_dim > 0
                else None
            )
        actor_params = self.actor_model.init(
            init_key, dummy_actions, dummy_timesteps, dummy_features
        )
        self.params = (
            {"actor": actor_params, "encoder": self.encoder.trainable_params}
            if self._trainable_encoder
            else actor_params
        )
        self.ema_params = self.params if self.use_ema else None

        self._ema_decay = ema_decay
        self._ema_min_decay = 0.0
        self._ema_update_after_step = 0
        self._ema_use_warmup = False
        self._ema_inv_gamma = 1.0
        self._ema_power = 0.75
        self._ema_optimization_step = 0

        learning_rate = lr
        if adaptive_lr:
            lr_schedule = str(lr_schedule).lower()
            if lr_schedule == "cosine":
                learning_rate = self.optax.cosine_decay_schedule(
                    init_value=lr,
                    decay_steps=self.num_train_steps,
                    alpha=0.0,
                )
            elif lr_schedule in {"warmup_cosine", "warmup-cosine"}:
                learning_rate = self.optax.warmup_cosine_decay_schedule(
                    init_value=0.0,
                    peak_value=lr,
                    warmup_steps=100,
                    decay_steps=self.num_train_steps,
                    end_value=0.0,
                )
            else:
                raise ValueError(f"Unknown flow matching lr_schedule '{lr_schedule}'.")
        transforms = []
        if actor_grad_clip is not None:
            transforms.append(self.optax.clip_by_global_norm(float(actor_grad_clip)))
        transforms.append(self.optax.adamw(learning_rate, weight_decay=weight_decay))
        self.optimizer = self.optax.chain(*transforms)
        self.opt_state = self.optimizer.init(self.params)

        update_fn = self._build_update_fn()
        update_many_fn = self._build_update_many_fn(update_fn)
        sample_fn = self._build_sample_fn()
        sample_from_noise_fn = self._build_sample_from_noise_fn()
        if self._jit_enabled:
            update_fn = jax.jit(update_fn)
            update_many_fn = jax.jit(update_many_fn)
            sample_fn = jax.jit(sample_fn)
            sample_from_noise_fn = jax.jit(sample_from_noise_fn)
        self._update_impl = update_fn
        self._update_many_impl = update_many_fn
        self._sample_impl = sample_fn
        self._sample_from_noise_impl = sample_from_noise_fn

        # Optional seed-aligned eval noise. When `_active_eval_seeds` is set
        # (list of env seeds in observation-batch-row order), `act` draws the
        # flow-matching initial noise deterministically from
        # (env_seed, per-seed episode step) instead of advancing the shared
        # rng_key batch-wise. This makes vectorized and serial evaluation
        # reproduce each other per seed. Disabled by default (None) so training
        # and ordinary eval behaviour is unchanged.
        self._eval_noise_base_key = jax.random.PRNGKey(0)
        self._eval_step_by_seed = {}
        self._active_eval_seeds = None

    def _ema_decay_value(self, optimization_step):
        step = jnp.maximum(0, optimization_step - self._ema_update_after_step - 1)
        if self._ema_use_warmup:
            decay = 1.0 - (1.0 + step / self._ema_inv_gamma) ** (-self._ema_power)
        else:
            decay = (1.0 + step) / (10.0 + step)
        decay = jnp.where(step <= 0, 0.0, decay)
        decay = jnp.minimum(decay, self._ema_decay)
        return jnp.maximum(decay, self._ema_min_decay)

    # ------------------------------------------------------------------
    # Language Conditioning
    # ------------------------------------------------------------------

    def _lang_token_rows(self, batch_or_obs: dict):
        return lang_token_rows(
            batch_or_obs,
            context="Language-conditioned flow matching",
        )

    def _encode_lang_tokens(self, tokens):
        return tokens_to_feature_jax(tokens, feature_dim=self.lang_feature_dim)

    def _extract_lang_features(self, batch_or_obs: dict):
        if not self.use_lang_cond:
            return None
        if "lang_features" in batch_or_obs:
            features = self._as_jax_array(
                lang_feature_rows(
                    batch_or_obs,
                    context="Language-conditioned flow matching",
                ),
                self.jnp.float32,
            )
            if features.shape[-1] != self.lang_feature_dim:
                raise ValueError(
                    "Language-conditioned flow matching expected lang_features "
                    f"with final dimension {self.lang_feature_dim}, got "
                    f"{features.shape[-1]}."
                )
            return features
        tokens = self._as_jax_array(
            self._lang_token_rows(batch_or_obs),
            self.jnp.float32,
        )
        return self._as_jax_array(
            self._encode_lang_tokens(tokens),
            self.jnp.float32,
        )

    def _prepare_obs_features(self, batch_or_obs: dict):
        if self._condition_as_sequence and not self.use_pixels:
            if "low_dim_state" not in batch_or_obs:
                raise ValueError(
                    "Transformer flow matching expects low_dim_state observations."
                )
            obs_features = self._as_jax_array(
                batch_or_obs["low_dim_state"], self.jnp.float32
            )
            if self.use_lang_cond:
                lang_features = self._extract_lang_features(batch_or_obs)
                lang_features = self.jnp.repeat(
                    lang_features[:, None, :],
                    obs_features.shape[1],
                    axis=1,
                )
                obs_features = self.jnp.concatenate(
                    [obs_features, lang_features],
                    axis=-1,
                )
            metrics = {}
            if self.logging:
                metrics["low_dim_state"] = np.asarray(
                    self.jax.device_get(obs_features[0, -1])
                )
            return obs_features, metrics
        obs_features, metrics = super()._prepare_obs_features(batch_or_obs)
        if self.use_lang_cond:
            obs_features = self.jnp.concatenate(
                [obs_features, self._extract_lang_features(batch_or_obs)],
                axis=-1,
            )
        return obs_features, metrics

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
                    "Trainable JAX flow-matching encoder requires raw RGB "
                    "observations; disable replay.cache_frozen_image_features."
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
        if self.use_lang_cond:
            inputs["lang"] = self._extract_lang_features(batch_or_obs)
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
        if "lang" in obs_inputs:
            features.append(obs_inputs["lang"])
        if not features:
            raise ValueError(
                "Flow matching requires at least one observation feature."
            )
        if len(features) == 1:
            return features[0]
        return jnp.concatenate(features, axis=-1)

    def _build_update_fn(self):
        optimizer = self.optimizer
        optax = self.optax
        use_ema = self.use_ema
        actor_model = self.actor_model
        time_scale = self._time_scale
        train_time_beta_params = self._train_time_beta_params
        horizon_dropout_lengths = self._horizon_dropout_lengths
        horizon_dropout_probs = self._horizon_dropout_probs
        horizon_loss_weights = self._horizon_loss_weights

        def update_fn(
            params, opt_state, rng_key, obs_inputs, actions,
            loss_coeff, action_pad_mask, ema_params, ema_optimization_step,
        ):
            source_key, time_key, dropout_key, next_key = jax.random.split(rng_key, 4)
            x1 = jax.random.normal(source_key, shape=actions.shape)
            if train_time_beta_params is None:
                t = jax.random.uniform(
                    time_key,
                    shape=(actions.shape[0],),
                    minval=0.0,
                    maxval=1.0,
                )
            else:
                alpha, beta = train_time_beta_params
                t = jax.random.beta(
                    time_key,
                    alpha,
                    beta,
                    shape=(actions.shape[0],),
                )
                t = jnp.clip(t, 1e-5, 1.0 - 1e-5)
            t_broadcast = t[:, None, None]
            xt = t_broadcast * x1 + (1.0 - t_broadcast) * actions
            target_velocity = actions - x1

            effective_action_pad_mask = action_pad_mask
            if horizon_dropout_lengths is not None:
                length_indices = jax.random.choice(
                    dropout_key,
                    horizon_dropout_lengths.shape[0],
                    shape=(actions.shape[0],),
                    p=horizon_dropout_probs,
                )
                sampled_lengths = horizon_dropout_lengths[length_indices]
                token_positions = jnp.arange(actions.shape[1], dtype=jnp.int32)
                dropout_mask = token_positions[None, :] >= sampled_lengths[:, None]
                if effective_action_pad_mask is None:
                    effective_action_pad_mask = dropout_mask
                else:
                    effective_action_pad_mask = jnp.logical_or(
                        effective_action_pad_mask, dropout_mask
                    )

            if effective_action_pad_mask is not None:
                valid_mask = jnp.logical_not(effective_action_pad_mask)
                xt = jnp.where(valid_mask[..., None], xt, actions)
                target_velocity = jnp.where(
                    valid_mask[..., None],
                    target_velocity,
                    0.0,
                )

            def loss_fn(current_params):
                obs_features = self._features_from_inputs(
                    current_params, obs_inputs
                )
                velocity_pred = actor_model.apply(
                    self._actor_params(current_params),
                    xt,
                    t * time_scale,
                    obs_features,
                )
                per_token_mse = jnp.square(velocity_pred - target_velocity)
                reduce_dims = tuple(range(1, per_token_mse.ndim))
                if horizon_loss_weights is not None:
                    action_reduce_dims = tuple(range(2, per_token_mse.ndim))
                    per_step_mse = per_token_mse.mean(axis=action_reduce_dims)
                    weights = horizon_loss_weights
                    if effective_action_pad_mask is None:
                        mse_loss = (per_step_mse * weights[None, :]).sum(axis=1)
                    else:
                        valid_mask = jnp.logical_not(effective_action_pad_mask).astype(
                            per_step_mse.dtype
                        )
                        weighted_mask = valid_mask * weights[None, :]
                        denom = jnp.clip(weighted_mask.sum(axis=1), min=1e-8)
                        mse_loss = (per_step_mse * weighted_mask).sum(axis=1) / denom
                elif effective_action_pad_mask is None:
                    mse_loss = per_token_mse.mean(axis=reduce_dims)
                else:
                    valid_mask = jnp.logical_not(effective_action_pad_mask).astype(
                        per_token_mse.dtype
                    )
                    while valid_mask.ndim < per_token_mse.ndim:
                        valid_mask = valid_mask[..., None]
                    valid_mask = jnp.broadcast_to(valid_mask, per_token_mse.shape)
                    masked_loss = per_token_mse * valid_mask
                    denom = jnp.clip(valid_mask.sum(axis=reduce_dims), min=1.0)
                    mse_loss = masked_loss.sum(axis=reduce_dims) / denom
                total_loss = (mse_loss * loss_coeff).mean()
                return total_loss, mse_loss

            (loss, mse_loss), grads = jax.value_and_grad(
                loss_fn, has_aux=True,
            )(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            new_ema_params = ema_params
            new_ema_step = ema_optimization_step
            if use_ema:
                new_ema_step = ema_optimization_step + 1
                decay = self._ema_decay_value(new_ema_step)
                one_minus_decay = 1.0 - decay
                new_ema_params = jax.tree.map(
                    lambda ema, param: ema - one_minus_decay * (ema - param),
                    ema_params, new_params,
                )

            new_pri = jnp.sqrt(mse_loss + 1e-10)
            max_pri = jnp.max(new_pri)
            normalized_pri = new_pri / jnp.where(max_pri > 0, max_pri, 1.0)
            return (
                new_params, new_opt_state, next_key, loss,
                normalized_pri, new_ema_params, new_ema_step,
            )

        return update_fn

    def _build_update_many_fn(self, update_fn):
        def update_many_fn(
            params, opt_state, rng_key, obs_inputs, actions,
            loss_coeff, action_pad_mask, ema_params, ema_optimization_step,
        ):
            def body_fn(carry, xs):
                current_params, current_opt_state, current_key, current_ema, current_ema_step = carry
                if action_pad_mask is None:
                    step_obs_inputs, step_actions, step_loss_coeff = xs
                    step_action_pad_mask = None
                else:
                    step_obs_inputs, step_actions, step_loss_coeff, step_action_pad_mask = xs
                (
                    next_params,
                    next_opt_state,
                    next_key,
                    loss,
                    priority,
                    next_ema,
                    next_ema_step,
                ) = update_fn(
                    current_params,
                    current_opt_state,
                    current_key,
                    step_obs_inputs,
                    step_actions,
                    step_loss_coeff,
                    step_action_pad_mask,
                    current_ema,
                    current_ema_step,
                )
                next_carry = (
                    next_params,
                    next_opt_state,
                    next_key,
                    next_ema,
                    next_ema_step,
                )
                return next_carry, (loss, priority)

            xs = (obs_inputs, actions, loss_coeff)
            if action_pad_mask is not None:
                xs = (*xs, action_pad_mask)
            (
                new_params,
                new_opt_state,
                new_key,
                new_ema,
                new_ema_step,
            ), (losses, priorities) = jax.lax.scan(
                body_fn,
                (params, opt_state, rng_key, ema_params, ema_optimization_step),
                xs,
            )
            return (
                new_params,
                new_opt_state,
                new_key,
                losses[-1],
                priorities[-1],
                new_ema,
                new_ema_step,
            )

        return update_many_fn

    def _build_sample_fn(self):
        actor_model = self.actor_model
        time_scale = self._time_scale
        sample_schedule = self._build_sample_schedule()

        def sample_fn(params, rng_key, obs_inputs):
            obs_features = self._features_from_inputs(params, obs_inputs)
            batch_size = obs_features.shape[0]
            sample = jax.random.normal(
                rng_key,
                shape=(batch_size, self.action_sequence, self.action_dim),
            )
            return self._integrate_sample(
                params, sample, obs_features, sample_schedule, actor_model, time_scale
            )

        return sample_fn

    def _build_sample_from_noise_fn(self):
        """Like `sample_fn` but integrates an externally supplied initial noise
        of shape (batch, action_sequence, action_dim). Used by seed-aligned
        eval so the noise can be keyed on (env_seed, episode_step)."""
        actor_model = self.actor_model
        time_scale = self._time_scale
        sample_schedule = self._build_sample_schedule()

        def sample_from_noise_fn(params, noise, obs_inputs):
            obs_features = self._features_from_inputs(params, obs_inputs)
            return self._integrate_sample(
                params, noise, obs_features, sample_schedule, actor_model, time_scale
            )

        return sample_from_noise_fn

    def _build_sample_schedule(self):
        if self._sample_schedule_jump_point is None:
            return jnp.linspace(
                0.0,
                1.0,
                self.num_flow_steps + 1,
                dtype=jnp.float32,
            )
        jump_point = float(self._sample_schedule_jump_point)
        return jnp.concatenate(
            [
                jnp.linspace(
                    0.0,
                    jump_point,
                    self.num_flow_steps,
                    dtype=jnp.float32,
                ),
                jnp.asarray([1.0], dtype=jnp.float32),
            ],
            axis=0,
        )

    def _integrate_sample(
        self, params, sample, obs_features, sample_schedule, actor_model, time_scale
    ):
        batch_size = obs_features.shape[0]

        def body_fn(index, current_sample):
            step = self.num_flow_steps - index
            t_value = sample_schedule[step]
            prev_t = sample_schedule[step - 1]
            delta_t = t_value - prev_t
            t_batch = jnp.full((batch_size,), t_value, dtype=jnp.float32)
            velocity = actor_model.apply(
                self._actor_params(params),
                current_sample,
                t_batch * time_scale,
                obs_features,
            )
            velocity = jnp.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)
            next_sample = current_sample + delta_t * velocity
            next_sample = jnp.nan_to_num(
                next_sample, nan=0.0, posinf=1.0, neginf=-1.0
            )
            return next_sample

        sample = jax.lax.fori_loop(0, self.num_flow_steps, body_fn, sample)
        sample = jnp.nan_to_num(sample, nan=0.0, posinf=1.0, neginf=-1.0)
        return jnp.clip(sample, -1.0, 1.0)

    def _fuse_multi_view(self, rgb_feats):
        if rgb_feats is None:
            return None
        rgb_feats = jnp.asarray(rgb_feats, dtype=jnp.float32)
        if self.view_fusion is not None:
            return self.view_fusion.apply({}, rgb_feats)
        return rgb_feats[:, 0]

    def act(self, observations: dict, step: int, eval_mode: bool):
        del step
        if self._trainable_encoder:
            obs_inputs = self._prepare_trainable_obs_inputs(observations)
        else:
            obs_inputs, _ = self._prepare_obs_features(observations)
        sample_params = self.ema_params if (eval_mode and self.use_ema) else self.params
        if eval_mode and self._active_eval_seeds is not None:
            noise = self._aligned_eval_noise(self._active_eval_seeds)
            actions = self._sample_from_noise_impl(sample_params, noise, obs_inputs)
        else:
            self.rng_key, sample_key = jax.random.split(self.rng_key)
            actions = self._sample_impl(sample_params, sample_key, obs_inputs)
        self._block(actions)
        return np.asarray(jax.device_get(actions), dtype=np.float32)

    def set_active_eval_seeds(self, seeds_in_batch_order):
        """Enable seed-aligned eval noise. `seeds_in_batch_order` is the env
        seed for each observation-batch row, in row order. Pass None to
        disable and restore the default shared-rng_key sampling."""
        if seeds_in_batch_order is None:
            self._active_eval_seeds = None
        else:
            self._active_eval_seeds = [int(s) for s in seeds_in_batch_order]

    def reset_aligned_eval_noise(self):
        """Clear per-seed episode-step counters (call once before an eval run)."""
        self._eval_step_by_seed = {}

    def _aligned_eval_noise(self, seeds):
        rows = []
        for s in seeds:
            s = int(s)
            t = self._eval_step_by_seed.get(s, 0)
            self._eval_step_by_seed[s] = t + 1
            row_key = jax.random.fold_in(
                jax.random.fold_in(self._eval_noise_base_key, s), t
            )
            rows.append(
                jax.random.normal(
                    row_key, shape=(self.action_sequence, self.action_dim)
                )
            )
        return jnp.stack(rows, axis=0)

    def _last_obs_inputs(self, obs_inputs):
        if isinstance(obs_inputs, dict):
            return self.jax.tree_util.tree_map(lambda value: value[-1], obs_inputs)
        return obs_inputs[-1]

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
        (
            self.params,
            self.opt_state,
            self.rng_key,
            actor_loss,
            new_priority,
            self.ema_params,
            self._ema_optimization_step,
        ) = self._update_impl(
            self.params,
            self.opt_state,
            self.rng_key,
            obs_inputs,
            actions,
            loss_coeff,
            action_pad_mask,
            self.ema_params,
            self._ema_optimization_step,
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

    def update_many(
        self,
        replay_iter: Iterator[dict],
        num_updates: int,
        replay_buffer: ReplayBuffer = None,
    ) -> dict[str, np.ndarray]:
        num_updates = int(num_updates)
        if num_updates <= 1 or self._uses_replay_priorities(replay_buffer):
            metrics = {}
            for _ in range(max(num_updates, 1)):
                metrics.update(self.update(replay_iter, 0, replay_buffer))
            return metrics

        obs_inputs_list = []
        actions = []
        loss_coeffs = []
        action_pad_masks = []
        has_action_pad_mask = None
        for _ in range(num_updates):
            batch = next(replay_iter)
            actions.append(self._as_jax_array(batch["action"], self.jnp.float32))
            loss_coeffs.append(self._loss_weights(batch))
            if self._trainable_encoder:
                obs = self._prepare_trainable_obs_inputs(batch)
            else:
                obs, _ = self._prepare_obs_features(batch)
            obs_inputs_list.append(obs)
            action_pad_mask = self._extract_action_pad_mask(batch)
            current_has_mask = action_pad_mask is not None
            if has_action_pad_mask is None:
                has_action_pad_mask = current_has_mask
            elif has_action_pad_mask != current_has_mask:
                raise ValueError(
                    "Cannot fuse updates with mixed action_pad_mask presence."
                )
            if current_has_mask:
                action_pad_masks.append(action_pad_mask)

        obs_inputs = self.jax.tree_util.tree_map(
            lambda *values: self.jnp.stack(values, axis=0),
            *obs_inputs_list,
        )
        actions = self.jnp.stack(actions, axis=0)
        loss_coeffs = self.jnp.stack(loss_coeffs, axis=0)
        action_pad_mask = (
            self.jnp.stack(action_pad_masks, axis=0)
            if has_action_pad_mask
            else None
        )

        start_time = time.perf_counter()
        (
            self.params,
            self.opt_state,
            self.rng_key,
            actor_loss,
            _new_priority,
            self.ema_params,
            self._ema_optimization_step,
        ) = self._update_many_impl(
            self.params,
            self.opt_state,
            self.rng_key,
            obs_inputs,
            actions,
            loss_coeffs,
            action_pad_mask,
            self.ema_params,
            self._ema_optimization_step,
        )
        if (
            self.logging
            or (self._update_step_count + num_updates) % self._update_block_every_steps
            == 0
        ):
            self._block(actor_loss)
        elapsed = time.perf_counter() - start_time
        self._update_step_count += num_updates
        self._first_update_completed = True

        if not self.logging:
            return {}
        metrics = {}
        self._maybe_log_update_metrics(
            metrics,
            actor_loss,
            self._last_obs_inputs(obs_inputs),
            elapsed / max(num_updates, 1),
        )
        return metrics

    def state_dict(self) -> dict:
        state = {"params": self._tree_to_numpy(self.params)}
        if self.encoder is not None:
            encoder_frozen_state = self.encoder.frozen_state_dict()
            if encoder_frozen_state:
                state["_encoder_frozen_state"] = self._tree_to_numpy(
                    encoder_frozen_state
                )
        if self.use_ema and self.ema_params is not None:
            state["_ema_params"] = self._tree_to_numpy(self.ema_params)
            state["_ema_state"] = {
                "decay": float(self._ema_decay),
                "min_decay": float(self._ema_min_decay),
                "optimization_step": int(self._ema_optimization_step),
                "update_after_step": int(self._ema_update_after_step),
                "use_ema_warmup": bool(self._ema_use_warmup),
                "inv_gamma": float(self._ema_inv_gamma),
                "power": float(self._ema_power),
            }
        return state

    def load_state_dict(self, state_dict: dict):
        self.params = self._tree_from_numpy(state_dict["params"])
        encoder_frozen_state = state_dict.get("_encoder_frozen_state")
        if self.encoder is not None and encoder_frozen_state is not None:
            self.encoder.load_frozen_state_dict(
                self._tree_from_numpy(encoder_frozen_state)
            )
        if not self.use_ema:
            return
        ema_params = state_dict.get("_ema_params")
        ema_state = state_dict.get("_ema_state")
        self.ema_params = (
            self._tree_from_numpy(ema_params) if ema_params is not None else self.params
        )
        if ema_state is None:
            self._ema_optimization_step = 0
            return
        self._ema_decay = float(ema_state.get("decay", self._ema_decay))
        self._ema_min_decay = float(ema_state.get("min_decay", self._ema_min_decay))
        self._ema_optimization_step = int(
            ema_state.get("optimization_step", self._ema_optimization_step)
        )
        self._ema_update_after_step = int(
            ema_state.get("update_after_step", self._ema_update_after_step)
        )
        self._ema_use_warmup = bool(
            ema_state.get("use_ema_warmup", self._ema_use_warmup)
        )
        self._ema_inv_gamma = float(ema_state.get("inv_gamma", self._ema_inv_gamma))
        self._ema_power = float(ema_state.get("power", self._ema_power))


__all__ = [
    "FlowMatching",
    "FlowMatchingBackboneSpec",
    "FlowMatchingModelSpec",
    "FlowMatchingSpec",
    "flow_matching_model_spec_from_cfg",
    "flow_matching_spec_from_cfg",
]
