from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
from gymnasium import spaces

from robobase.backends.jax.models.encoder import JaxResNetEncoder
from robobase.backends.jax.models.fusion import JaxFusionMultiCamFeature
from robobase.backends.jax.models.fully_connected import JaxMLPWithSequenceOutput
from robobase.method.bc import BCModelSpec
from robobase.method.bc_runtime import (
    bc_actor_input_shapes,
    bc_observation_layout,
    flatten_time_into_channel,
)
from robobase.replay_buffer.prioritized_replay_buffer import PrioritizedReplayBuffer
from robobase.replay_buffer.replay_buffer import ReplayBuffer


def _maybe_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _extract_many(mapping: dict, pattern: str) -> dict:
    regex = re.compile(pattern)
    filtered = {}
    for key, value in mapping.items():
        if regex.search(key):
            filtered[key] = value
    if not filtered:
        raise ValueError(
            f"Couldn't find the regex key '{pattern}' in the mapping. "
            f"Available keys are: {list(mapping.keys())}"
        )
    return filtered


def _stack_tensor_dictionary(tensor_dict: dict, axis: int) -> np.ndarray:
    return np.stack([_maybe_numpy(value) for value in tensor_dict.values()], axis=axis)


@dataclass(frozen=True)
class _BuiltJaxBCModel:
    actor_model: JaxMLPWithSequenceOutput
    encoder_model: JaxResNetEncoder | None
    view_fusion_model: JaxFusionMultiCamFeature | None


def _build_model(
    model_spec: BCModelSpec,
    *,
    observation_space: spaces.Dict,
    action_space: spaces.Box,
    encoder_jit: bool,
) -> tuple[_BuiltJaxBCModel, int]:
    obs_layout = bc_observation_layout(observation_space)
    actor_spec = model_spec.actor_model
    if actor_spec.type != "mlp_bottleneck_sequence":
        raise NotImplementedError(
            f"Unsupported JAX BC actor model type '{actor_spec.type}'."
        )
    if actor_spec.output_sequence_length != int(action_space.shape[0]):
        raise ValueError(
            "BC actor model output_sequence_length does not match the action space."
        )
    if actor_spec.output_sequence_network_type not in {"rnn", "mlp"}:
        raise ValueError(
            "Unsupported JAX BC output_sequence_network_type "
            f"'{actor_spec.output_sequence_network_type}'."
        )

    encoder_model = None
    view_fusion_model = None

    if obs_layout.use_pixels:
        if model_spec.encoder_model is None:
            raise ValueError("Pixel BC requires encoder_model in the shared model spec.")
        if model_spec.encoder_model.type != "resnet":
            raise NotImplementedError(
                f"Unsupported JAX BC encoder model type '{model_spec.encoder_model.type}'."
            )
        if obs_layout.rgb_input_shape is None:
            raise ValueError("Pixel BC expected a valid RGB input shape.")
        encoder_model = JaxResNetEncoder(
            input_shape=obs_layout.rgb_input_shape,
            model=model_spec.encoder_model.model,
            jit=encoder_jit,
        )

        if obs_layout.use_multicam_fusion:
            if model_spec.view_fusion_model is None:
                raise ValueError(
                    "Multi-camera pixel BC requires view_fusion_model in the shared model spec."
                )
            if model_spec.view_fusion_model.type != "multicam_feature":
                raise NotImplementedError(
                    "Unsupported JAX BC view fusion model type "
                    f"'{model_spec.view_fusion_model.type}'."
                )
            view_fusion_model = JaxFusionMultiCamFeature(
                input_shape=encoder_model.output_shape,
                mode=model_spec.view_fusion_model.mode,
            )
    elif model_spec.encoder_model is not None and model_spec.encoder_model.type != "resnet":
        raise NotImplementedError(
            f"Unsupported JAX BC encoder model type '{model_spec.encoder_model.type}'."
        )

    rgb_latent_size = 0
    if encoder_model is not None:
        if view_fusion_model is not None:
            rgb_latent_size = int(view_fusion_model.output_shape[-1])
        else:
            rgb_latent_size = int(encoder_model.output_shape[-1])

    actor_input_shapes = bc_actor_input_shapes(
        low_dim_size=obs_layout.low_dim_size,
        rgb_latent_size=rgb_latent_size,
        frame_stack_on_channel=True,
        time_dim=obs_layout.time_dim,
    )
    feature_shape = actor_input_shapes["features"]
    input_dim = int(np.prod(feature_shape))

    return _BuiltJaxBCModel(
        actor_model=JaxMLPWithSequenceOutput(
            hidden_dims=actor_spec.hidden_dims,
            action_sequence=int(action_space.shape[0]),
            action_dim=int(action_space.shape[1]),
            output_sequence_network_type=actor_spec.output_sequence_network_type,
        ),
        encoder_model=encoder_model,
        view_fusion_model=view_fusion_model,
    ), input_dim


class JaxBC:
    """BC implementation backed by JAX.

    The actor and current pixel-encoder path are both JAX based. The pixel encoder
    remains frozen and currently supports the shared BC ResNet surface.
    """

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
    ):
        if intrinsic_reward_module is not None:
            raise NotImplementedError(
                "The JAX BC implementation does not support intrinsic rewards yet."
            )
        if len(action_space.shape) != 2:
            raise ValueError(
                "JaxBC expects action_space.shape == (sequence_length, action_dim)."
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
        self.action_sequence = int(action_space.shape[0])
        self.action_dim = int(action_space.shape[1])

        built_model, input_dim = _build_model(
            self.model_spec,
            observation_space=observation_space,
            action_space=action_space,
            encoder_jit=jit,
        )
        self.actor_model = built_model.actor_model
        self.encoder = built_model.encoder_model
        self.view_fusion = built_model.view_fusion_model
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

        self.rng_key = jax.random.PRNGKey(int(seed))
        self.params = self.actor_model.init_params(
            jax,
            jnp,
            self.rng_key,
            input_dim=input_dim,
        )

        learning_rate = lr
        if adaptive_lr:
            learning_rate = optax.cosine_decay_schedule(
                init_value=lr,
                decay_steps=self.num_train_steps,
            )
        transforms = []
        if actor_grad_clip is not None:
            transforms.append(optax.clip_by_global_norm(float(actor_grad_clip)))
        transforms.append(optax.adam(learning_rate))
        self.optimizer = optax.chain(*transforms)
        self.opt_state = self.optimizer.init(self.params)

        update_fn = self._build_update_fn()
        predict_fn = self._predict_impl
        if self._jit_enabled:
            update_fn = jax.jit(update_fn)
            predict_fn = jax.jit(predict_fn)
        self._update_impl = update_fn
        self._predict = predict_fn

    def _predict_impl(self, params, obs_features):
        return self.actor_model.apply(self.jax, self.jnp, params, obs_features)

    def _build_update_fn(self):
        jnp = self.jnp
        optimizer = self.optimizer
        optax = self.optax

        def update_fn(
            params,
            opt_state,
            obs_features,
            actions,
            loss_coeff,
            action_pad_mask,
        ):
            def loss_fn(current_params):
                action_pred = self._predict_impl(current_params, obs_features)
                per_token_mse = jnp.square(action_pred - actions)
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
                loss_fn, has_aux=True
            )(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            new_pri = jnp.sqrt(mse_loss + 1e-10)
            max_pri = jnp.max(new_pri)
            normalized_pri = new_pri / jnp.where(max_pri > 0, max_pri, 1.0)
            return new_params, new_opt_state, loss, normalized_pri

        return update_fn

    def _block(self, *values):
        for value in values:
            self.jax.block_until_ready(value)

    def _extract_low_dim_batch(self, batch_or_obs: dict):
        if self.low_dim_size == 0 or "low_dim_state" not in batch_or_obs:
            return None
        low_dim_obs = _maybe_numpy(batch_or_obs["low_dim_state"]).astype(
            np.float32, copy=False
        )
        low_dim_obs = flatten_time_into_channel(low_dim_obs)
        return low_dim_obs.reshape((low_dim_obs.shape[0], -1))

    def _extract_rgb_obs(self, batch_or_obs: dict):
        if not self.use_pixels:
            return None, {}
        if self.logging:
            rgb_obs_dict = _extract_many(batch_or_obs, r"rgb(?!.*?tp1)")
            metrics = {
                key: _maybe_numpy(value)[0, -1]
                for key, value in rgb_obs_dict.items()
            }
        else:
            rgb_obs_dict = _extract_many(batch_or_obs, r"rgb(?!.*?tp1)")
            metrics = {}
        rgb_obs = flatten_time_into_channel(
            _stack_tensor_dictionary(rgb_obs_dict, axis=1),
            has_view_axis=True,
        ).astype(np.float32, copy=False)
        return rgb_obs, metrics

    def _extract_action_pad_mask(self, batch: dict):
        if "action_pad_mask" not in batch:
            return None
        return _maybe_numpy(batch["action_pad_mask"]).astype(np.bool_, copy=False)

    def _loss_weights(self, batch: dict) -> np.ndarray:
        if "sampling_probabilities" in batch:
            probs = _maybe_numpy(batch["sampling_probabilities"]).astype(
                np.float32, copy=False
            )
            loss_weights = 1.0 / np.sqrt(probs + 1e-10)
            loss_weights = (loss_weights / np.max(loss_weights)) ** self.replay_beta
            return loss_weights.astype(np.float32, copy=False)
        batch_size = _maybe_numpy(batch["action"]).shape[0]
        return np.ones((batch_size,), dtype=np.float32)

    def _encode_pixels(self, rgb_obs):
        if self.encoder is None:
            return {}
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
            raise ValueError("BC requires at least one observation feature source.")
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
        del step, eval_mode
        low_dim_obs = self._extract_low_dim_batch(observations)
        rgb_obs, _ = self._extract_rgb_obs(observations) if self.use_pixels else (None, {})
        fused_view_feats = None
        if rgb_obs is not None:
            fused_view_feats = self._fuse_multi_view(self._encode_pixels(rgb_obs))
        obs_features = self._combine_features(low_dim_obs, fused_view_feats)
        actions = self._predict(self.params, obs_features)
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
        fused_view_feats = None
        if self.use_pixels:
            rgb_obs, pixel_metrics = self._extract_rgb_obs(batch)
            metrics.update(pixel_metrics)
            fused_view_feats = self._fuse_multi_view(self._encode_pixels(rgb_obs))

        obs_features = self._combine_features(low_dim_obs, fused_view_feats)

        start_time = time.perf_counter()
        (
            self.params,
            self.opt_state,
            actor_loss,
            new_priority,
        ) = self._update_impl(
            self.params,
            self.opt_state,
            self.jnp.asarray(obs_features),
            self.jnp.asarray(actions),
            self.jnp.asarray(loss_coeff),
            None if action_pad_mask is None else self.jnp.asarray(action_pad_mask),
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
                elapsed, 1e-12
            )
            if not self._first_update_completed:
                metrics["backend/first_update_time_sec"] = elapsed

        self._first_update_completed = True
        return metrics

    def state_dict(self) -> dict:
        return {"params": self._tree_to_numpy(self.params)}

    def load_state_dict(self, state_dict: dict):
        self.params = self._tree_from_numpy(state_dict["params"])

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
