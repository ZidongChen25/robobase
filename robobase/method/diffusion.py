from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Optional

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig

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


# ---------------------------------------------------------------------------
# Spec dataclasses
# ---------------------------------------------------------------------------

DiffusionActorModelSpec = DiffusionBackboneSpec


@dataclass(frozen=True)
class DiffusionModelSpec:
    actor_model: DiffusionBackboneSpec
    encoder_model: BCEncoderModelSpec | None
    view_fusion_model: BCViewFusionModelSpec | None
    backbone: DiffusionBackboneSpec | None = None
    use_lang_cond: bool = False
    lang_feature_dim: int = 512

    @property
    def resolved_backbone(self) -> DiffusionBackboneSpec:
        return self.backbone or self.actor_model


@dataclass(frozen=True)
class DiffusionSpec:
    lr: float
    adaptive_lr: bool
    lr_schedule: str
    num_train_steps: int
    actor_grad_clip: float | None
    objective_type: str
    num_diffusion_iters: int
    sampler: str
    use_ema: bool
    ema_decay: float
    ema_decay_schedule: str
    weight_decay: float
    model: DiffusionModelSpec


# ---------------------------------------------------------------------------
# Config parsers
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


def diffusion_model_spec_from_cfg(cfg: DictConfig) -> DiffusionModelSpec:
    method_cfg = cfg.method
    backbone_cfg = method_cfg.get("backbone", None)
    actor_model_cfg = method_cfg.get("actor_model", None)
    if backbone_cfg is None and actor_model_cfg is None:
        raise ValueError("Diffusion requires a backbone or actor_model config.")

    if backbone_cfg is not None:
        backbone_spec = backbone_spec_from_cfg(
            backbone_cfg,
            default_type="unet1d",
            default_sequence_length=int(cfg.action_sequence),
        )
    else:
        actor_model_type = _config_type(
            actor_model_cfg,
            default="conditional_unet1d",
            target_to_type={
                "robobase.models.diffusion.JaxConditionalUnet1D": "conditional_unet1d",
            },
        )
        backbone_spec = backbone_spec_from_cfg(
            actor_model_cfg,
            default_type=actor_model_type,
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

    return DiffusionModelSpec(
        actor_model=backbone_spec,
        encoder_model=encoder_model_spec,
        view_fusion_model=view_fusion_model_spec,
        backbone=backbone_spec if backbone_cfg is not None else None,
        use_lang_cond=bool(method_cfg.get("use_lang_cond", False)),
        lang_feature_dim=int(method_cfg.get("lang_feature_dim", 512)),
    )


def diffusion_spec_from_cfg(cfg: DictConfig) -> DiffusionSpec:
    objective_cfg = cfg.method.get("objective", None)
    objective_type = (
        str(objective_cfg.get("type", "ddpm")).lower()
        if objective_cfg is not None
        else "ddpm"
    )
    num_diffusion_iters = (
        int(objective_cfg.get("num_diffusion_iters"))
        if objective_cfg is not None and objective_cfg.get("num_diffusion_iters") is not None
        else int(cfg.method.num_diffusion_iters)
    )
    sampler = (
        str(objective_cfg.get("sampler", "ddim")).lower()
        if objective_cfg is not None
        else str(cfg.method.get("sampler", "ddim")).lower()
    )
    return DiffusionSpec(
        lr=float(cfg.method.lr),
        adaptive_lr=bool(cfg.method.adaptive_lr),
        lr_schedule=str(cfg.method.get("lr_schedule", "warmup_cosine")).lower(),
        num_train_steps=int(cfg.method.num_train_steps),
        actor_grad_clip=(
            None
            if cfg.method.actor_grad_clip is None
            else float(cfg.method.actor_grad_clip)
        ),
        objective_type=objective_type,
        num_diffusion_iters=num_diffusion_iters,
        sampler=sampler,
        use_ema=bool(cfg.method.get("use_ema", False)),
        ema_decay=float(cfg.method.get("ema_decay", 0.9999)),
        ema_decay_schedule=str(
            cfg.method.get("ema_decay_schedule", "diffusers")
        ).lower(),
        weight_decay=float(cfg.method.get("weight_decay", 1e-6)),
        model=diffusion_model_spec_from_cfg(cfg),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cosine_betas(num_diffusion_iters: int, *, max_beta: float = 0.999) -> np.ndarray:
    def alpha_bar(t: float) -> float:
        return np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2

    betas = []
    for index in range(num_diffusion_iters):
        t1 = index / num_diffusion_iters
        t2 = (index + 1) / num_diffusion_iters
        betas.append(min(1.0 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.asarray(betas, dtype=np.float32)


@dataclass(frozen=True)
class _BuiltDiffusionModel:
    actor_model: object
    encoder_model: JaxResNetEncoder | None
    view_fusion_model: JaxFusionMultiCamFeature | None


def _build_encoder_and_fusion(
    *,
    model_spec: DiffusionModelSpec,
    observation_space: spaces.Dict,
    encoder_jit: bool,
) -> tuple[JaxResNetEncoder | None, JaxFusionMultiCamFeature | None, int]:
    obs_layout = bc_observation_layout(observation_space)
    encoder_model = None
    view_fusion_model = None

    if obs_layout.use_pixels:
        if model_spec.encoder_model is None:
            raise ValueError(
                "Pixel diffusion requires encoder_model in the shared model spec."
            )
        if model_spec.encoder_model.type != "resnet":
            raise NotImplementedError(
                "Unsupported diffusion encoder model type "
                f"'{model_spec.encoder_model.type}'."
            )
        if obs_layout.rgb_input_shape is None:
            raise ValueError("Pixel diffusion expected a valid RGB input shape.")
        encoder_model = JaxResNetEncoder(
            input_shape=obs_layout.rgb_input_shape,
            model=model_spec.encoder_model.model,
            jit=encoder_jit,
        )
        if obs_layout.use_multicam_fusion:
            if model_spec.view_fusion_model is None:
                raise ValueError(
                    "Multi-camera pixel diffusion requires view_fusion_model."
                )
            if model_spec.view_fusion_model.type != "multicam_feature":
                raise NotImplementedError(
                    "Unsupported diffusion view fusion model type "
                    f"'{model_spec.view_fusion_model.type}'."
                )
            view_fusion_model = JaxFusionMultiCamFeature.from_input_shape(
                input_shape=encoder_model.output_shape,
                mode=model_spec.view_fusion_model.mode,
            )
    elif model_spec.encoder_model is not None and model_spec.encoder_model.type != "resnet":
        raise NotImplementedError(
            "Unsupported diffusion encoder model type "
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
    model_spec: DiffusionModelSpec,
    *,
    observation_space: spaces.Dict,
    action_space: spaces.Box,
    encoder_jit: bool,
) -> tuple[_BuiltDiffusionModel, int]:
    obs_layout = bc_observation_layout(observation_space)
    backbone_spec = model_spec.resolved_backbone
    backbone_type = canonical_backbone_type(backbone_spec.type)

    encoder_model, view_fusion_model, rgb_latent_size = _build_encoder_and_fusion(
        model_spec=model_spec,
        observation_space=observation_space,
        encoder_jit=encoder_jit,
    )
    lang_feature_dim = (
        int(model_spec.lang_feature_dim) if model_spec.use_lang_cond else 0
    )
    feature_dim = int(obs_layout.low_dim_size + rgb_latent_size + lang_feature_dim)
    if backbone_type == "transformer" and not obs_layout.use_pixels:
        low_dim_state_spec = observation_space.spaces.get("low_dim_state")
        if low_dim_state_spec is not None:
            feature_dim = int(low_dim_state_spec.shape[-1]) + lang_feature_dim
    return _BuiltDiffusionModel(
        actor_model=build_diffusion_backbone(
            backbone_spec,
            action_dim=int(action_space.shape[1]),
            sequence_length=int(action_space.shape[0]),
            condition_dim=feature_dim,
        ),
        encoder_model=encoder_model,
        view_fusion_model=view_fusion_model,
    ), feature_dim


# ---------------------------------------------------------------------------
# Diffusion class
# ---------------------------------------------------------------------------

class Diffusion(JaxMethodBase):
    """Diffusion policy implementation backed by JAX with Flax models."""

    def __init__(
        self,
        lr: float,
        adaptive_lr: bool,
        num_train_steps: int,
        num_diffusion_iters: int,
        model: DiffusionModelSpec,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        num_train_envs: int,
        num_eval_envs: int,
        replay_alpha: float,
        replay_beta: float,
        frame_stack_on_channel: bool,
        intrinsic_reward_module=None,
        actor_grad_clip: Optional[float] = None,
        lr_schedule: str = "warmup_cosine",
        jit: bool = True,
        platform: str | None = None,
        seed: int = 0,
        is_rl: bool = False,
        use_ema: bool = False,
        ema_decay: float = 0.9999,
        ema_decay_schedule: str = "diffusers",
        weight_decay: float = 1e-6,
        objective_type: str = "ddpm",
        sampler: str = "ddim",
        update_block_every_steps: int = 1,
    ):
        if not frame_stack_on_channel:
            raise NotImplementedError(
                "frame_stack_on_channel must be true for diffusion policies."
            )
        backbone_type = canonical_backbone_type(model.resolved_backbone.type)
        if (
            backbone_type == "unet1d"
            and len(action_space.shape) == 2
            and int(action_space.shape[0]) % 4 != 0
        ):
            raise ValueError(
                "Action sequence length has to be a multiple of 4 for diffusion model."
            )
        if str(objective_type).lower() != "ddpm":
            raise NotImplementedError(
                "Diffusion currently supports the DDPM noise-prediction objective."
            )
        if str(sampler).lower() not in {"ddim", "ddpm"}:
            raise NotImplementedError(
                "Diffusion currently supports DDIM and DDPM samplers."
            )

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

        self.num_diffusion_iters = int(num_diffusion_iters)
        self.objective_type = str(objective_type).lower()
        self.sampler = str(sampler).lower()
        self.model_spec = model
        self.use_lang_cond = bool(model.use_lang_cond)
        self.lang_feature_dim = int(model.lang_feature_dim) if self.use_lang_cond else 0
        self._clip_text_cache: dict[tuple[int, ...], np.ndarray] = {}
        self._backbone_type = backbone_type
        self._condition_as_sequence = backbone_type == "transformer"
        self._init_cached_pixel_feature_key("diffusion")

        # EMA config
        self._ema_decay = ema_decay
        self._ema_decay_schedule = str(ema_decay_schedule).lower()
        self._ema_min_decay = 0.0
        self._ema_update_after_step = 0
        self._ema_use_warmup = False
        self._ema_inv_gamma = 1.0
        self._ema_power = 0.75
        self._ema_optimization_step = 0

        # Build model
        built_model, feature_dim = _build_model(
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
        self.rgb_latent_size = max(
            0,
            feature_dim - self.low_dim_size - self.lang_feature_dim,
        )

        # Diffusion schedule
        betas = _cosine_betas(self.num_diffusion_iters)
        alphas_cumprod = np.cumprod(1.0 - betas)
        self.betas = jnp.asarray(betas)
        self.alphas_cumprod = jnp.asarray(alphas_cumprod, dtype=jnp.float32)
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)
        self.inference_timesteps = jnp.arange(
            self.num_diffusion_iters - 1, -1, -1, dtype=jnp.int32,
        )

        # Flax init with dummy inputs
        self.rng_key, init_key = jax.random.split(self.rng_key)
        dummy_actions = jnp.zeros(
            (1, self.action_sequence, self.action_dim), dtype=jnp.float32
        )
        dummy_timesteps = jnp.zeros((1,), dtype=jnp.int32)
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
            {
                "actor": actor_params,
                "encoder": self.encoder.trainable_params,
            }
            if self._trainable_encoder
            else actor_params
        )
        self.ema_params = self.params if self.use_ema else None

        # Optimizer
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
                raise ValueError(f"Unknown diffusion lr_schedule '{lr_schedule}'.")
        transforms = []
        if actor_grad_clip is not None:
            transforms.append(self.optax.clip_by_global_norm(float(actor_grad_clip)))
        transforms.append(self.optax.adamw(learning_rate, weight_decay=weight_decay))
        self.optimizer = self.optax.chain(*transforms)
        self.opt_state = self.optimizer.init(self.params)

        # JIT-compiled functions
        update_fn = self._build_update_fn()
        update_many_fn = self._build_update_many_fn(update_fn)
        sample_fn = self._build_sample_fn()
        if self._jit_enabled:
            update_fn = jax.jit(update_fn)
            update_many_fn = jax.jit(update_many_fn)
            sample_fn = jax.jit(sample_fn)
        self._update_impl = update_fn
        self._update_many_impl = update_many_fn
        self._sample_impl = sample_fn

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------

    def _ema_decay_value(self, optimization_step):
        if self._ema_decay_schedule == "constant":
            return jnp.asarray(self._ema_decay, dtype=jnp.float32)
        step = jnp.maximum(0, optimization_step - self._ema_update_after_step - 1)
        if self._ema_use_warmup:
            decay = 1.0 - (1.0 + step / self._ema_inv_gamma) ** (-self._ema_power)
        else:
            decay = (1.0 + step) / (10.0 + step)
        decay = jnp.where(step <= 0, 0.0, decay)
        decay = jnp.minimum(decay, self._ema_decay)
        return jnp.maximum(decay, self._ema_min_decay)

    def _prepare_obs_features(self, batch_or_obs: dict):
        if self._condition_as_sequence and not self.use_pixels:
            if "low_dim_state" not in batch_or_obs:
                raise ValueError("Transformer diffusion expects low_dim_state observations.")
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

    def _lang_token_rows(self, batch_or_obs: dict) -> np.ndarray:
        if "lang_tokens" not in batch_or_obs:
            raise ValueError(
                "Language-conditioned diffusion requires 'lang_tokens' observations."
            )
        tokens = np.asarray(batch_or_obs["lang_tokens"], dtype=np.int32)
        if tokens.ndim == 1:
            tokens = tokens[None, :]
        elif tokens.ndim >= 3:
            tokens = tokens.reshape(tokens.shape[0], -1, tokens.shape[-1])[:, -1, :]
        return tokens

    def _encode_clip_text(self, tokens: np.ndarray) -> np.ndarray:
        token_rows = np.asarray(tokens, dtype=np.int32)
        features = []
        missing = []
        missing_keys = []
        for row in token_rows:
            key = tuple(int(v) for v in row.tolist())
            cached = self._clip_text_cache.get(key)
            if cached is None:
                missing.append(row)
                missing_keys.append(key)
            features.append((key, cached))

        if missing:
            import torch
            import clip

            if not hasattr(self, "clip_model"):
                clip_model, _ = clip.load("ViT-B/32", device="cpu")
                clip_model.eval()
                for param in clip_model.parameters():
                    param.requires_grad_(False)
                self.__dict__["clip_model"] = clip_model

            with torch.no_grad():
                token_tensor = torch.as_tensor(
                    np.stack(missing),
                    dtype=torch.long,
                    device="cpu",
                )
                encoded = self.clip_model.encode_text(token_tensor).float().cpu().numpy()
            if encoded.shape[-1] != self.lang_feature_dim:
                raise ValueError(
                    "CLIP text feature dimension does not match "
                    f"method.lang_feature_dim: {encoded.shape[-1]} != "
                    f"{self.lang_feature_dim}."
                )
            for key, value in zip(missing_keys, encoded):
                self._clip_text_cache[key] = value.astype(np.float32, copy=False)

        return np.stack(
            [self._clip_text_cache[key] for key, _ in features],
            axis=0,
        ).astype(np.float32, copy=False)

    def _extract_lang_features(self, batch_or_obs: dict):
        if not self.use_lang_cond:
            return None
        return self._as_jax_array(
            self._encode_clip_text(self._lang_token_rows(batch_or_obs)),
            self.jnp.float32,
        )

    def _prepare_trainable_obs_inputs(self, batch_or_obs: dict):
        inputs = {}
        low_dim_obs = self._extract_low_dim_batch(batch_or_obs)
        if low_dim_obs is not None:
            inputs["low_dim"] = low_dim_obs
        if self.use_pixels:
            if self._has_cached_pixel_features(batch_or_obs):
                raise ValueError(
                    "Trainable JAX diffusion encoder requires raw RGB observations; "
                    "disable replay.cache_frozen_image_features."
                )
            rgb_obs, _ = self._extract_rgb_obs(batch_or_obs)
            inputs["rgb"] = rgb_obs
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
            )
            features.append(self._fuse_multi_view(rgb_feats))
        if "lang" in obs_inputs:
            features.append(obs_inputs["lang"])
        if not features:
            raise ValueError("Diffusion requires at least one observation feature.")
        if len(features) == 1:
            return features[0]
        return self.jnp.concatenate(features, axis=-1)

    def _obs_input_batch_size(self, obs_inputs) -> int:
        if isinstance(obs_inputs, dict):
            leaves = self.jax.tree_util.tree_leaves(obs_inputs)
            return int(leaves[0].shape[0])
        return int(obs_inputs.shape[0])

    def _last_obs_inputs(self, obs_inputs):
        if isinstance(obs_inputs, dict):
            return self.jax.tree_util.tree_map(lambda value: value[-1], obs_inputs)
        return obs_inputs[-1]

    # ------------------------------------------------------------------
    # Forward / update
    # ------------------------------------------------------------------

    def _build_update_fn(self):
        optimizer = self.optimizer
        optax = self.optax
        use_ema = self.use_ema
        actor_model = self.actor_model

        def update_fn(
            params, opt_state, rng_key, obs_inputs, actions,
            loss_coeff, action_pad_mask, ema_params, ema_optimization_step,
        ):
            noise_key, timestep_key, dropout_key, next_key = jax.random.split(rng_key, 4)
            noise = jax.random.normal(noise_key, shape=actions.shape)
            timesteps = jax.random.randint(
                timestep_key,
                shape=(actions.shape[0],),
                minval=0,
                maxval=self.num_diffusion_iters,
            )

            sqrt_alpha = self.sqrt_alphas_cumprod[timesteps][:, None, None]
            sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[timesteps][:, None, None]
            noisy_actions = sqrt_alpha * actions + sqrt_one_minus * noise

            if action_pad_mask is not None:
                valid_mask = jnp.logical_not(action_pad_mask)
                noisy_actions = jnp.where(valid_mask[..., None], noisy_actions, 0.0)
                noise = jnp.where(valid_mask[..., None], noise, 0.0)

            def loss_fn(current_params):
                obs_features = self._features_from_inputs(current_params, obs_inputs)
                actor_params = self._actor_params(current_params)
                if self._condition_as_sequence:
                    noise_pred = actor_model.apply(
                        actor_params,
                        noisy_actions,
                        timesteps,
                        obs_features,
                        train=True,
                        rngs={"dropout": dropout_key},
                    )
                else:
                    noise_pred = actor_model.apply(
                        actor_params, noisy_actions, timesteps, obs_features,
                    )
                per_token_mse = jnp.square(noise_pred - noise)
                reduce_dims = tuple(range(1, per_token_mse.ndim))
                if action_pad_mask is None:
                    mse_loss = per_token_mse.mean(axis=reduce_dims)
                else:
                    valid_mask = jnp.logical_not(action_pad_mask).astype(
                        per_token_mse.dtype
                    )
                    while valid_mask.ndim < per_token_mse.ndim:
                        valid_mask = valid_mask[..., None]
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

        def sample_fn(params, rng_key, obs_inputs):
            obs_features = self._features_from_inputs(params, obs_inputs)
            batch_size = obs_features.shape[0]
            init_key, loop_key = jax.random.split(rng_key)
            sample = jax.random.normal(
                init_key,
                shape=(batch_size, self.action_sequence, self.action_dim),
            )

            def _predict_noise(current_params, current_sample, timestep_batch):
                actor_params = self._actor_params(current_params)
                if self._condition_as_sequence:
                    return actor_model.apply(
                        actor_params,
                        current_sample,
                        timestep_batch,
                        obs_features,
                        train=False,
                    )
                return actor_model.apply(
                    actor_params,
                    current_sample,
                    timestep_batch,
                    obs_features,
                )

            def ddim_body_fn(index, current_sample):
                timestep = self.inference_timesteps[index]
                timestep_batch = jnp.full((batch_size,), timestep, dtype=jnp.int32)
                noise_pred = _predict_noise(params, current_sample, timestep_batch)
                alpha_t = self.alphas_cumprod[timestep]
                prev_index = jnp.maximum(timestep - 1, 0)
                alpha_prev = jnp.where(
                    timestep > 0,
                    self.alphas_cumprod[prev_index],
                    jnp.asarray(1.0, dtype=jnp.float32),
                )
                sqrt_alpha_t = jnp.sqrt(alpha_t)
                sqrt_one_minus_alpha_t = jnp.sqrt(jnp.maximum(1.0 - alpha_t, 0.0))
                pred_original_sample = (
                    current_sample - sqrt_one_minus_alpha_t * noise_pred
                ) / jnp.maximum(sqrt_alpha_t, 1e-12)
                pred_original_sample = jnp.clip(pred_original_sample, -1.0, 1.0)
                return (
                    jnp.sqrt(alpha_prev) * pred_original_sample
                    + jnp.sqrt(jnp.maximum(1.0 - alpha_prev, 0.0)) * noise_pred
                )

            def ddpm_body_fn(index, carry):
                current_key, current_sample = carry
                current_key, noise_key = jax.random.split(current_key)
                timestep = self.inference_timesteps[index]
                timestep_batch = jnp.full((batch_size,), timestep, dtype=jnp.int32)
                bar_alpha = self.alphas_cumprod[timestep]
                bar_alpha_prev = jnp.where(
                    timestep > 0,
                    self.alphas_cumprod[jnp.maximum(timestep - 1, 0)],
                    jnp.asarray(1.0, dtype=jnp.float32),
                )
                beta = self.betas[timestep]
                alpha = 1.0 - beta
                pred_noise = _predict_noise(params, current_sample, timestep_batch)

                sqrt_bar_alpha = jnp.sqrt(bar_alpha)
                sqrt_one_minus_bar_alpha = jnp.sqrt(
                    jnp.maximum(1.0 - bar_alpha, 1e-12)
                )
                upper_bound = (current_sample + sqrt_bar_alpha) / sqrt_one_minus_bar_alpha
                lower_bound = (current_sample - sqrt_bar_alpha) / sqrt_one_minus_bar_alpha
                pred_noise = jnp.clip(pred_noise, lower_bound, upper_bound)

                next_sample = (
                    current_sample
                    - beta / sqrt_one_minus_bar_alpha * pred_noise
                ) / jnp.sqrt(alpha)
                noise = jax.random.normal(noise_key, shape=current_sample.shape)
                sigma = jnp.sqrt(
                    jnp.maximum(beta * (1.0 - bar_alpha_prev) / (1.0 - bar_alpha), 0.0)
                )
                next_sample = jnp.where(
                    timestep > 0,
                    next_sample + sigma * noise,
                    next_sample,
                )
                return current_key, next_sample

            if self.sampler == "ddpm":
                _, sample = jax.lax.fori_loop(
                    0,
                    self.inference_timesteps.shape[0],
                    ddpm_body_fn,
                    (loop_key, sample),
                )
                return sample

            return jax.lax.fori_loop(
                0, self.inference_timesteps.shape[0], ddim_body_fn, sample,
            )

        return sample_fn

    # ------------------------------------------------------------------
    # Fusion override (Flax module)
    # ------------------------------------------------------------------

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
        self.rng_key, sample_key = jax.random.split(self.rng_key)
        sample_params = self.ema_params if (eval_mode and self.use_ema) else self.params
        actions = self._sample_impl(sample_params, sample_key, obs_inputs)
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
            self.params if self.ema_params is None else self.ema_params,
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
            self.params if self.ema_params is None else self.ema_params,
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

    # ------------------------------------------------------------------
    # State management (overrides for EMA)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        state = {"params": self._tree_to_numpy(self.params)}
        if self.use_ema and self.ema_params is not None:
            state["_ema_params"] = self._tree_to_numpy(self.ema_params)
            state["_ema_state"] = {
                "decay": float(self._ema_decay),
                "decay_schedule": self._ema_decay_schedule,
                "min_decay": float(self._ema_min_decay),
                "optimization_step": int(self._ema_optimization_step),
                "update_after_step": int(self._ema_update_after_step),
                "use_ema_warmup": bool(self._ema_use_warmup),
                "inv_gamma": float(self._ema_inv_gamma),
                "power": float(self._ema_power),
            }
        return state

    def load_state_dict(self, state_dict: dict):
        params = self._tree_from_numpy(state_dict["params"])
        if self._trainable_encoder and not (
            isinstance(params, dict) and "actor" in params
        ):
            params = {
                "actor": params,
                "encoder": self.encoder.trainable_params,
            }
        self.params = params
        if not self.use_ema:
            return
        ema_params = state_dict.get("_ema_params")
        ema_state = state_dict.get("_ema_state")
        if ema_params is not None:
            ema_params = self._tree_from_numpy(ema_params)
            if self._trainable_encoder and not (
                isinstance(ema_params, dict) and "actor" in ema_params
            ):
                ema_params = {
                    "actor": ema_params,
                    "encoder": self.encoder.trainable_params,
                }
            self.ema_params = ema_params
        else:
            self.ema_params = self.params
        if ema_state is None:
            self._ema_optimization_step = 0
            return
        self._ema_decay = float(ema_state.get("decay", self._ema_decay))
        self._ema_decay_schedule = str(
            ema_state.get("decay_schedule", self._ema_decay_schedule)
        )
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
    "DiffusionActorModelSpec",
    "DiffusionModelSpec",
    "DiffusionSpec",
    "Diffusion",
    "diffusion_model_spec_from_cfg",
    "diffusion_spec_from_cfg",
]
