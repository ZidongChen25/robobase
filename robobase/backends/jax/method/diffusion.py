from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
from gymnasium import spaces

from robobase.backends.jax.models.diffusion import JaxConditionalUnet1D
from robobase.backends.jax.models.encoder import JaxResNetEncoder
from robobase.backends.jax.models.fusion import JaxFusionMultiCamFeature
from robobase.method.bc import BCEncoderModelSpec, BCViewFusionModelSpec
from robobase.method.bc_runtime import (
    bc_observation_layout,
    flatten_time_into_channel,
)
from robobase.method.diffusion import DiffusionModelSpec
from robobase.replay_buffer.prioritized_replay_buffer import PrioritizedReplayBuffer
from robobase.replay_buffer.replay_buffer import ReplayBuffer
from robobase.replay_buffer.vision_feature_cache import (
    cached_feature_observation_key,
)


def _maybe_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _stack_tensor_dictionary(tensor_dict: dict, axis: int) -> np.ndarray:
    return np.stack([_maybe_numpy(value) for value in tensor_dict.values()], axis=axis)


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
class _BuiltJaxDiffusionModel:
    actor_model: JaxConditionalUnet1D
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
                "Unsupported JAX diffusion encoder model type "
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
                    "Multi-camera pixel diffusion requires view_fusion_model in the shared model spec."
                )
            if model_spec.view_fusion_model.type != "multicam_feature":
                raise NotImplementedError(
                    "Unsupported JAX diffusion view fusion model type "
                    f"'{model_spec.view_fusion_model.type}'."
                )
            view_fusion_model = JaxFusionMultiCamFeature(
                input_shape=encoder_model.output_shape,
                mode=model_spec.view_fusion_model.mode,
            )
    elif model_spec.encoder_model is not None and model_spec.encoder_model.type != "resnet":
        raise NotImplementedError(
            "Unsupported JAX diffusion encoder model type "
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
) -> tuple[_BuiltJaxDiffusionModel, int]:
    obs_layout = bc_observation_layout(observation_space)
    actor_spec = model_spec.actor_model
    if actor_spec.type != "conditional_unet1d":
        raise NotImplementedError(
            f"Unsupported JAX diffusion actor model type '{actor_spec.type}'."
        )
    if actor_spec.sequence_length != int(action_space.shape[0]):
        raise ValueError(
            "Diffusion actor model sequence_length does not match the action space."
        )

    encoder_model, view_fusion_model, rgb_latent_size = _build_encoder_and_fusion(
        model_spec=model_spec,
        observation_space=observation_space,
        encoder_jit=encoder_jit,
    )
    feature_dim = int(obs_layout.low_dim_size + rgb_latent_size)
    return _BuiltJaxDiffusionModel(
        actor_model=JaxConditionalUnet1D(
            action_shape=tuple(int(v) for v in action_space.shape),
            feature_dim=feature_dim,
            diffusion_step_embed_dim=actor_spec.diffusion_step_embed_dim,
            down_dims=actor_spec.down_dims,
            kernel_size=actor_spec.kernel_size,
            n_groups=actor_spec.n_groups,
        ),
        encoder_model=encoder_model,
        view_fusion_model=view_fusion_model,
    ), feature_dim


class JaxDiffusion:
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
        jit: bool = True,
        platform: str | None = None,
        seed: int = 0,
        is_rl: bool = False,
        use_ema: bool = False,
    ):
        if intrinsic_reward_module is not None:
            raise NotImplementedError(
                "The JAX diffusion implementation does not support intrinsic rewards yet."
            )
        if not frame_stack_on_channel:
            raise NotImplementedError(
                "frame_stack_on_channel must be true for diffusion policies."
            )
        if len(action_space.shape) != 2:
            raise ValueError(
                "JaxDiffusion expects action_space.shape == (sequence_length, action_dim)."
            )
        if int(action_space.shape[0]) % 4 != 0:
            raise ValueError(
                "Action sequence length has to be a multiple of 4 for diffusion model."
            )

        import jax
        import jax.numpy as jnp
        import optax

        if platform:
            jax.config.update("jax_platform_name", str(platform))

        self.jax = jax
        self.jnp = jnp
        self.optax = optax

        self.lr = lr
        self.adaptive_lr = adaptive_lr
        self.num_train_steps = max(1, num_train_steps)
        self.num_diffusion_iters = int(num_diffusion_iters)
        self.model_spec = model
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_train_envs = num_train_envs
        self.num_eval_envs = num_eval_envs
        self.replay_alpha = replay_alpha
        self.replay_beta = replay_beta
        self.frame_stack_on_channel = frame_stack_on_channel
        self.actor_grad_clip = actor_grad_clip
        self.training = True
        self.logging = False
        self.is_rl = is_rl
        self.use_ema = use_ema
        self._eval_env_running = False
        self.backend_name = "jax"
        self._jit_enabled = jit
        self._first_update_completed = False
        self._update_step_count = 0
        self.obs_layout = bc_observation_layout(observation_space)
        self.time_dim = self.obs_layout.time_dim
        self.low_dim_size = self.obs_layout.low_dim_size
        self.use_pixels = self.obs_layout.use_pixels
        self.use_multicam_fusion = self.obs_layout.use_multicam_fusion
        self._rgb_batch_keys = tuple(self.obs_layout.rgb_keys)
        self._cached_pixel_feature_key = (
            cached_feature_observation_key("diffusion", "jax")
            if self.use_pixels
            else None
        )
        self.action_sequence = int(action_space.shape[0])
        self.action_dim = int(action_space.shape[1])
        self._ema_decay = 0.9999
        self._ema_min_decay = 0.0
        self._ema_update_after_step = 0
        self._ema_use_warmup = False
        self._ema_inv_gamma = 1.0
        self._ema_power = 0.75
        self._ema_optimization_step = 0

        built_model, feature_dim = _build_model(
            self.model_spec,
            observation_space=observation_space,
            action_space=action_space,
            encoder_jit=jit,
        )
        self.actor_model = built_model.actor_model
        self.encoder = built_model.encoder_model
        self.view_fusion = built_model.view_fusion_model
        self.rgb_latent_size = max(0, feature_dim - self.low_dim_size)

        betas = _cosine_betas(self.num_diffusion_iters)
        alphas_cumprod = np.cumprod(1.0 - betas)
        self.betas = jnp.asarray(betas)
        self.alphas_cumprod = jnp.asarray(alphas_cumprod, dtype=jnp.float32)
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)
        self.inference_timesteps = jnp.arange(
            self.num_diffusion_iters - 1,
            -1,
            -1,
            dtype=jnp.int32,
        )

        self.rng_key = jax.random.PRNGKey(int(seed))
        self.rng_key, init_key = jax.random.split(self.rng_key)
        self.params = self.actor_model.init_params(init_key)
        self.ema_params = self.params if self.use_ema else None

        learning_rate = lr
        if adaptive_lr:
            learning_rate = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=lr,
                warmup_steps=100,
                decay_steps=self.num_train_steps,
                end_value=0.0,
            )
        transforms = []
        if actor_grad_clip is not None:
            transforms.append(optax.clip_by_global_norm(float(actor_grad_clip)))
        transforms.append(optax.adamw(learning_rate, weight_decay=1e-6))
        self.optimizer = optax.chain(*transforms)
        self.opt_state = self.optimizer.init(self.params)

        update_fn = self._build_update_fn()
        sample_fn = self._build_sample_fn()
        if self._jit_enabled:
            update_fn = jax.jit(update_fn)
            sample_fn = jax.jit(sample_fn)
        self._update_impl = update_fn
        self._sample_impl = sample_fn

    def _ema_decay_value(self, optimization_step):
        step = self.jnp.maximum(
            0,
            optimization_step - self._ema_update_after_step - 1,
        )
        if self._ema_use_warmup:
            decay = 1.0 - (1.0 + step / self._ema_inv_gamma) ** (-self._ema_power)
        else:
            decay = (1.0 + step) / (10.0 + step)
        decay = self.jnp.where(step <= 0, 0.0, decay)
        decay = self.jnp.minimum(decay, self._ema_decay)
        return self.jnp.maximum(decay, self._ema_min_decay)

    def _build_update_fn(self):
        jnp = self.jnp
        optimizer = self.optimizer
        optax = self.optax
        use_ema = self.use_ema

        def update_fn(
            params,
            opt_state,
            rng_key,
            obs_features,
            actions,
            loss_coeff,
            action_pad_mask,
            ema_params,
            ema_optimization_step,
        ):
            noise_key, timestep_key, next_key = self.jax.random.split(rng_key, 3)
            noise = self.jax.random.normal(noise_key, shape=actions.shape)
            timesteps = self.jax.random.randint(
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
                noise_pred = self.actor_model.apply(
                    current_params,
                    noisy_actions,
                    timesteps,
                    obs_features,
                )
                per_token_mse = jnp.square(noise_pred - noise)
                reduce_dims = tuple(range(1, per_token_mse.ndim))
                if action_pad_mask is None:
                    mse_loss = per_token_mse.mean(axis=reduce_dims)
                else:
                    valid_mask = jnp.logical_not(action_pad_mask).astype(per_token_mse.dtype)
                    while valid_mask.ndim < per_token_mse.ndim:
                        valid_mask = valid_mask[..., None]
                    masked_loss = per_token_mse * valid_mask
                    denom = jnp.clip(valid_mask.sum(axis=reduce_dims), min=1.0)
                    mse_loss = masked_loss.sum(axis=reduce_dims) / denom
                total_loss = (mse_loss * loss_coeff).mean()
                return total_loss, mse_loss

            (loss, mse_loss), grads = self.jax.value_and_grad(
                loss_fn,
                has_aux=True,
            )(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            new_ema_params = ema_params
            new_ema_step = ema_optimization_step
            if use_ema:
                new_ema_step = ema_optimization_step + 1
                decay = self._ema_decay_value(new_ema_step)
                one_minus_decay = 1.0 - decay
                new_ema_params = self.jax.tree_util.tree_map(
                    lambda ema, param: ema - one_minus_decay * (ema - param),
                    ema_params,
                    new_params,
                )

            new_pri = jnp.sqrt(mse_loss + 1e-10)
            max_pri = jnp.max(new_pri)
            normalized_pri = new_pri / jnp.where(max_pri > 0, max_pri, 1.0)
            return (
                new_params,
                new_opt_state,
                next_key,
                loss,
                normalized_pri,
                new_ema_params,
                new_ema_step,
            )

        return update_fn

    def _build_sample_fn(self):
        def sample_fn(params, rng_key, obs_features):
            batch_size = obs_features.shape[0]
            sample = self.jax.random.normal(
                rng_key,
                shape=(batch_size, self.action_sequence, self.action_dim),
            )

            def body_fn(index, current_sample):
                timestep = self.inference_timesteps[index]
                timestep_batch = self.jnp.full(
                    (batch_size,),
                    timestep,
                    dtype=self.jnp.int32,
                )
                noise_pred = self.actor_model.apply(
                    params,
                    current_sample,
                    timestep_batch,
                    obs_features,
                )
                alpha_t = self.alphas_cumprod[timestep]
                prev_index = self.jnp.maximum(timestep - 1, 0)
                alpha_prev = self.jnp.where(
                    timestep > 0,
                    self.alphas_cumprod[prev_index],
                    self.jnp.asarray(1.0, dtype=self.jnp.float32),
                )
                sqrt_alpha_t = self.jnp.sqrt(alpha_t)
                sqrt_one_minus_alpha_t = self.jnp.sqrt(
                    self.jnp.maximum(1.0 - alpha_t, 0.0)
                )
                pred_original_sample = (
                    current_sample - sqrt_one_minus_alpha_t * noise_pred
                ) / self.jnp.maximum(sqrt_alpha_t, 1e-12)
                pred_original_sample = self.jnp.clip(pred_original_sample, -1.0, 1.0)
                return (
                    self.jnp.sqrt(alpha_prev) * pred_original_sample
                    + self.jnp.sqrt(self.jnp.maximum(1.0 - alpha_prev, 0.0))
                    * noise_pred
                )

            return self.jax.lax.fori_loop(
                0,
                self.inference_timesteps.shape[0],
                body_fn,
                sample,
            )

        return sample_fn

    def _block(self, *values):
        for value in values:
            self.jax.block_until_ready(value)

    def _extract_low_dim_batch(self, batch_or_obs: dict):
        if self.low_dim_size == 0 or "low_dim_state" not in batch_or_obs:
            return None
        low_dim_obs = _maybe_numpy(batch_or_obs["low_dim_state"]).astype(
            np.float32,
            copy=False,
        )
        low_dim_obs = flatten_time_into_channel(low_dim_obs)
        return low_dim_obs.reshape((low_dim_obs.shape[0], -1))

    def _extract_rgb_obs(self, batch_or_obs: dict):
        if not self.use_pixels:
            return None, {}
        rgb_obs_dict = {key: batch_or_obs[key] for key in self._rgb_batch_keys}
        metrics = {}
        if self.logging:
            metrics = {
                key: _maybe_numpy(value)[0, -1]
                for key, value in rgb_obs_dict.items()
            }
        rgb_obs = flatten_time_into_channel(
            _stack_tensor_dictionary(rgb_obs_dict, axis=1),
            has_view_axis=True,
        ).astype(np.float32, copy=False)
        return rgb_obs, metrics

    def _has_cached_pixel_features(self, batch_or_obs: dict) -> bool:
        return (
            self._cached_pixel_feature_key is not None
            and self._cached_pixel_feature_key in batch_or_obs
        )

    def _extract_cached_pixel_features(self, batch_or_obs: dict):
        if not self._has_cached_pixel_features(batch_or_obs):
            return None
        cached = _maybe_numpy(batch_or_obs[self._cached_pixel_feature_key]).astype(
            np.float32,
            copy=False,
        )
        cached = flatten_time_into_channel(cached)
        return cached.reshape((cached.shape[0], -1))

    def _extract_action_pad_mask(self, batch: dict):
        if "action_pad_mask" not in batch:
            return None
        return _maybe_numpy(batch["action_pad_mask"]).astype(np.bool_, copy=False)

    def _loss_weights(self, batch: dict) -> np.ndarray:
        if "sampling_probabilities" in batch:
            probs = _maybe_numpy(batch["sampling_probabilities"]).astype(
                np.float32,
                copy=False,
            )
            loss_weights = 1.0 / np.sqrt(probs + 1e-10)
            loss_weights = (loss_weights / np.max(loss_weights)) ** self.replay_beta
            return loss_weights.astype(np.float32, copy=False)
        batch_size = _maybe_numpy(batch["action"]).shape[0]
        return np.ones((batch_size,), dtype=np.float32)

    def _encode_pixels(self, rgb_obs):
        if self.encoder is None:
            return None
        return self.encoder.encode(rgb_obs)

    def _fuse_multi_view(self, rgb_feats):
        if rgb_feats is None:
            return None
        rgb_feats = self.jnp.asarray(rgb_feats, dtype=self.jnp.float32)
        if self.view_fusion is not None:
            return self.view_fusion.apply(self.jnp, rgb_feats)
        return rgb_feats[:, 0]

    def _combine_features(self, low_dim_obs, fused_view_feats):
        features = []
        if low_dim_obs is not None:
            features.append(self.jnp.asarray(low_dim_obs, dtype=self.jnp.float32))
        if fused_view_feats is not None:
            features.append(self.jnp.asarray(fused_view_feats, dtype=self.jnp.float32))
        if not features:
            raise ValueError(
                "Diffusion requires at least one observation feature source."
            )
        if len(features) == 1:
            return features[0]
        return self.jnp.concatenate(features, axis=-1)

    @property
    def eval_env_running(self):
        return self._eval_env_running

    def set_eval_env_running(self, value: bool):
        self._eval_env_running = value

    def train(self, training: bool):
        self.training = bool(training)

    def reset(self, step: int, agents_to_reset: list[int]):
        del step, agents_to_reset

    def act(self, observations: dict, step: int, eval_mode: bool):
        del step
        low_dim_obs = self._extract_low_dim_batch(observations)
        fused_view_feats = self._extract_cached_pixel_features(observations)
        if self.use_pixels and fused_view_feats is None:
            rgb_obs, _ = self._extract_rgb_obs(observations)
            fused_view_feats = self._fuse_multi_view(self._encode_pixels(rgb_obs))
        obs_features = self._combine_features(low_dim_obs, fused_view_feats)
        self.rng_key, sample_key = self.jax.random.split(self.rng_key)
        sample_params = self.ema_params if (eval_mode and self.use_ema) else self.params
        actions = self._sample_impl(sample_params, sample_key, obs_features)
        self._block(actions)
        return np.asarray(self.jax.device_get(actions), dtype=np.float32)

    def update(
        self,
        replay_iter: Iterator[dict],
        step: int,
        replay_buffer: ReplayBuffer = None,
    ) -> dict[str, np.ndarray]:
        del step
        batch = next(replay_iter)
        low_dim_obs = self._extract_low_dim_batch(batch)
        actions = _maybe_numpy(batch["action"]).astype(np.float32, copy=False)
        action_pad_mask = self._extract_action_pad_mask(batch)
        loss_coeff = self._loss_weights(batch)

        metrics = {}
        fused_view_feats = self._extract_cached_pixel_features(batch)
        if self.use_pixels and fused_view_feats is None:
            rgb_obs, pixel_metrics = self._extract_rgb_obs(batch)
            metrics.update(pixel_metrics)
            fused_view_feats = self._fuse_multi_view(self._encode_pixels(rgb_obs))

        obs_features = self._combine_features(low_dim_obs, fused_view_feats)

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
            self.jnp.asarray(obs_features),
            self.jnp.asarray(actions),
            self.jnp.asarray(loss_coeff),
            None if action_pad_mask is None else self.jnp.asarray(action_pad_mask),
            self.params if self.ema_params is None else self.ema_params,
            self._ema_optimization_step,
        )
        self._block(actor_loss, new_priority)
        elapsed = time.perf_counter() - start_time
        self._update_step_count += 1

        new_priority_np = np.asarray(self.jax.device_get(new_priority), dtype=np.float32)
        if isinstance(replay_buffer, PrioritizedReplayBuffer):
            replay_buffer.set_priority(
                indices=_maybe_numpy(batch["indices"]),
                priorities=new_priority_np ** self.replay_alpha,
            )

        if self.logging:
            metrics["actor_loss"] = float(np.asarray(self.jax.device_get(actor_loss)))
            metrics["backend/update_time_sec"] = elapsed
            metrics["backend/update_steps_per_second"] = obs_features.shape[0] / max(
                elapsed,
                1e-12,
            )
            if not self._first_update_completed:
                metrics["backend/first_update_time_sec"] = elapsed

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

    def checkpoint_state_dict(self) -> dict[str, dict]:
        return {
            "opt_state": self._tree_to_numpy(self.opt_state),
            "rng_key": np.asarray(self.rng_key),
            "update_step_count": int(self._update_step_count),
            "first_update_completed": bool(self._first_update_completed),
        }

    def load_checkpoint_state_dict(self, state_dict: dict[str, dict]):
        if "opt_state" in state_dict:
            self.opt_state = self._tree_from_numpy(state_dict["opt_state"])
        if "rng_key" in state_dict:
            self.rng_key = self.jnp.asarray(state_dict["rng_key"])
        self._update_step_count = int(state_dict.get("update_step_count", 0))
        self._first_update_completed = bool(
            state_dict.get("first_update_completed", False)
        )

    def _tree_to_numpy(self, tree):
        return self.jax.tree_util.tree_map(
            lambda x: x if x is None else np.asarray(self.jax.device_get(x)),
            tree,
        )

    def _tree_from_numpy(self, tree):
        return self.jax.tree_util.tree_map(self.jnp.asarray, tree)
