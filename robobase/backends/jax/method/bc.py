from __future__ import annotations

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
)
from robobase.method.jax_base import JaxMethodBase
from robobase.method.jax_utils import maybe_numpy
from robobase.replay_buffer.replay_buffer import ReplayBuffer


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


class JaxBC(JaxMethodBase):
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

        self.params = self.actor_model.init_params(
            self.jax, self.jnp, self.rng_key, input_dim=input_dim,
        )

        # -- Optimizer --
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

        # -- JIT-compiled functions --
        update_fn = self._build_update_fn()
        predict_fn = self._predict_impl
        if self._jit_enabled:
            update_fn = self.jax.jit(update_fn)
            predict_fn = self.jax.jit(predict_fn)
        self._update_impl = update_fn
        self._predict = predict_fn

    # ------------------------------------------------------------------
    # Forward / update
    # ------------------------------------------------------------------

    def _predict_impl(self, params, obs_features):
        return self.actor_model.apply(self.jax, self.jnp, params, obs_features)

    def _build_update_fn(self):
        jnp = self.jnp
        optimizer = self.optimizer
        optax = self.optax

        def update_fn(
            params, opt_state, obs_features, actions, loss_coeff, action_pad_mask,
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
                loss_fn, has_aux=True,
            )(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            new_pri = jnp.sqrt(mse_loss + 1e-10)
            max_pri = jnp.max(new_pri)
            normalized_pri = new_pri / jnp.where(max_pri > 0, max_pri, 1.0)
            return new_params, new_opt_state, loss, normalized_pri

        return update_fn

    def act(self, observations: dict, step: int, eval_mode: bool):
        del step, eval_mode
        obs_features, _ = self._prepare_obs_features(observations)
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
        actions = maybe_numpy(batch["action"]).astype(np.float32, copy=False)
        action_pad_mask = self._extract_action_pad_mask(batch)
        loss_coeff = self._loss_weights(batch)
        obs_features, metrics = self._prepare_obs_features(batch)

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
        self._maybe_update_priorities(replay_buffer, batch, new_priority_np)
        self._maybe_log_update_metrics(metrics, actor_loss, obs_features, elapsed)
        self._first_update_completed = True
        return metrics
