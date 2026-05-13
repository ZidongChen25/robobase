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

from robobase.method.bc import BCEncoderModelSpec, BCViewFusionModelSpec
from robobase.method.bc_runtime import bc_observation_layout
from robobase.method.jax_base import JaxMethodBase
from robobase.models.backbone import (
    DiffusionBackboneSpec,
    backbone_spec_from_cfg,
    build_diffusion_backbone,
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


@dataclass(frozen=True)
class FlowMatchingSpec:
    lr: float
    adaptive_lr: bool
    num_train_steps: int
    actor_grad_clip: float | None
    objective_type: str
    num_flow_steps: int
    sampler: str
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
    return FlowMatchingSpec(
        lr=float(cfg.method.lr),
        adaptive_lr=bool(cfg.method.adaptive_lr),
        num_train_steps=int(cfg.method.num_train_steps),
        actor_grad_clip=(
            None
            if cfg.method.actor_grad_clip is None
            else float(cfg.method.actor_grad_clip)
        ),
        objective_type=objective_type,
        num_flow_steps=num_flow_steps,
        sampler=sampler,
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
        encoder_model = JaxResNetEncoder(
            input_shape=obs_layout.rgb_input_shape,
            model=model_spec.encoder_model.model,
            jit=encoder_jit,
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
    encoder_model, view_fusion_model, rgb_latent_size = _build_encoder_and_fusion(
        model_spec=model_spec,
        observation_space=observation_space,
        encoder_jit=encoder_jit,
    )
    feature_dim = int(obs_layout.low_dim_size + rgb_latent_size)
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
        jit: bool = True,
        platform: str | None = None,
        seed: int = 0,
        is_rl: bool = False,
        use_ema: bool = False,
        ema_decay: float = 0.9999,
        weight_decay: float = 1e-6,
        objective_type: str = "rectified_flow",
        sampler: str = "euler",
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
        self.num_flow_steps = int(num_flow_steps)
        self.model_spec = model
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
        self.rgb_latent_size = max(0, feature_dim - self.low_dim_size)

        self.rng_key, init_key = jax.random.split(self.rng_key)
        dummy_actions = jnp.zeros(
            (1, self.action_sequence, self.action_dim), dtype=jnp.float32
        )
        dummy_timesteps = jnp.zeros((1,), dtype=jnp.float32)
        dummy_features = (
            jnp.zeros((1, feature_dim), dtype=jnp.float32) if feature_dim > 0 else None
        )
        self.params = self.actor_model.init(
            init_key, dummy_actions, dummy_timesteps, dummy_features
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
            learning_rate = self.optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=lr,
                warmup_steps=100,
                decay_steps=self.num_train_steps,
                end_value=0.0,
            )
        transforms = []
        if actor_grad_clip is not None:
            transforms.append(self.optax.clip_by_global_norm(float(actor_grad_clip)))
        transforms.append(self.optax.adamw(learning_rate, weight_decay=weight_decay))
        self.optimizer = self.optax.chain(*transforms)
        self.opt_state = self.optimizer.init(self.params)

        update_fn = self._build_update_fn()
        sample_fn = self._build_sample_fn()
        if self._jit_enabled:
            update_fn = jax.jit(update_fn)
            sample_fn = jax.jit(sample_fn)
        self._update_impl = update_fn
        self._sample_impl = sample_fn

    def _ema_decay_value(self, optimization_step):
        step = jnp.maximum(0, optimization_step - self._ema_update_after_step - 1)
        if self._ema_use_warmup:
            decay = 1.0 - (1.0 + step / self._ema_inv_gamma) ** (-self._ema_power)
        else:
            decay = (1.0 + step) / (10.0 + step)
        decay = jnp.where(step <= 0, 0.0, decay)
        decay = jnp.minimum(decay, self._ema_decay)
        return jnp.maximum(decay, self._ema_min_decay)

    def _build_update_fn(self):
        optimizer = self.optimizer
        optax = self.optax
        use_ema = self.use_ema
        actor_model = self.actor_model

        def update_fn(
            params, opt_state, rng_key, obs_features, actions,
            loss_coeff, action_pad_mask, ema_params, ema_optimization_step,
        ):
            source_key, time_key, next_key = jax.random.split(rng_key, 3)
            x1 = jax.random.normal(source_key, shape=actions.shape)
            t = jax.random.uniform(
                time_key,
                shape=(actions.shape[0],),
                minval=0.0,
                maxval=1.0,
            )
            t_broadcast = t[:, None, None]
            xt = t_broadcast * x1 + (1.0 - t_broadcast) * actions
            target_velocity = actions - x1

            if action_pad_mask is not None:
                valid_mask = jnp.logical_not(action_pad_mask)
                xt = jnp.where(valid_mask[..., None], xt, actions)
                target_velocity = jnp.where(
                    valid_mask[..., None],
                    target_velocity,
                    0.0,
                )

            def loss_fn(current_params):
                velocity_pred = actor_model.apply(
                    current_params, xt, t, obs_features,
                )
                per_token_mse = jnp.square(velocity_pred - target_velocity)
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

    def _build_sample_fn(self):
        actor_model = self.actor_model
        sample_schedule = jnp.linspace(
            0.0,
            1.0,
            self.num_flow_steps + 1,
            dtype=jnp.float32,
        )

        def sample_fn(params, rng_key, obs_features):
            batch_size = obs_features.shape[0]
            sample = jax.random.normal(
                rng_key,
                shape=(batch_size, self.action_sequence, self.action_dim),
            )

            def body_fn(index, current_sample):
                step = self.num_flow_steps - index
                t_value = sample_schedule[step]
                prev_t = sample_schedule[step - 1]
                delta_t = t_value - prev_t
                t_batch = jnp.full((batch_size,), t_value, dtype=jnp.float32)
                velocity = actor_model.apply(
                    params, current_sample, t_batch, obs_features,
                )
                return current_sample + delta_t * velocity

            sample = jax.lax.fori_loop(0, self.num_flow_steps, body_fn, sample)
            return jnp.clip(sample, -1.0, 1.0)

        return sample_fn

    def _fuse_multi_view(self, rgb_feats):
        if rgb_feats is None:
            return None
        rgb_feats = jnp.asarray(rgb_feats, dtype=jnp.float32)
        if self.view_fusion is not None:
            return self.view_fusion.apply({}, rgb_feats)
        return rgb_feats[:, 0]

    def act(self, observations: dict, step: int, eval_mode: bool):
        del step
        obs_features, _ = self._prepare_obs_features(observations)
        self.rng_key, sample_key = jax.random.split(self.rng_key)
        sample_params = self.ema_params if (eval_mode and self.use_ema) else self.params
        actions = self._sample_impl(sample_params, sample_key, obs_features)
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
        obs_features, metrics = self._prepare_obs_features(batch)

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
            obs_features,
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
        self._maybe_log_update_metrics(metrics, actor_loss, obs_features, elapsed)
        self._first_update_completed = True
        return metrics

    def state_dict(self) -> dict:
        state = {"params": self._tree_to_numpy(self.params)}
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
