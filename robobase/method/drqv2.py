"""DrQ-v2 implemented in JAX/Flax on RoboBase's shared RL runtime.

The update follows the reference implementation from Yarats et al.:
independent random-shift augmentation of current/next images, a deterministic
actor with scheduled Gaussian exploration, clipped target-policy noise, twin
critics, and an exponential-moving-average target critic.  RoboBase-specific
observation encoders, multi-camera fusion, replay weighting, demonstrations,
and checkpointing remain provided by the existing modular runtime.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig

from robobase import utils
from robobase.method.core import OffPolicyMethod
from robobase.method.rl_common import (
    JaxRLMethodBase,
    RLModelSpec,
    activation,
    random_shift_rgb,
    rl_model_spec_from_cfg,
    scale_unit_action,
    unscale_action,
)
from robobase.replay_buffer.replay_buffer import ReplayBuffer


@dataclass(frozen=True)
class DrQV2Spec:
    actor_lr: float
    critic_lr: float
    encoder_lr: float
    num_train_steps: int
    num_explore_steps: int
    critic_target_tau: float
    stddev_schedule: str
    stddev_clip: float
    use_augmentation: bool
    augmentation_pad: int
    num_critics: int
    feature_dim: int
    actor_uses_time: bool
    always_bootstrap: bool
    bc_lambda: float
    actor_grad_clip: float | None
    critic_grad_clip: float | None
    weight_decay: float
    model: RLModelSpec


def _optional_float(value) -> float | None:
    return None if value is None else float(value)


def drqv2_spec_from_cfg(cfg: DictConfig) -> DrQV2Spec:
    """Parse canonical and historical RoboBase DrQ-v2 configuration fields."""

    method = cfg.method
    if bool(method.get("distributional_critic", False)):
        raise NotImplementedError(
            "The initial JAX DrQ-v2 port implements the canonical scalar twin-Q "
            "critic. Historical DrQ-v2 distributional-critic launch configs "
            "require a separate, explicitly matched C51 implementation."
        )

    actor_lr = float(method.get("actor_lr", method.get("lr", 1e-4)))
    critic_lr = float(method.get("critic_lr", method.get("lr", 1e-4)))
    model_cfg = method.get("model", None)
    actor_model_cfg = method.get("actor_model", None)
    feature_dim = 50
    if actor_model_cfg is not None:
        feature_dim = int(actor_model_cfg.get("bottleneck_size", feature_dim))
    if model_cfg is not None:
        feature_dim = int(model_cfg.get("feature_dim", feature_dim))

    return DrQV2Spec(
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        encoder_lr=float(method.get("encoder_lr", critic_lr)),
        num_train_steps=int(method.get("num_train_steps", cfg.num_train_frames)),
        num_explore_steps=int(method.get("num_explore_steps", cfg.num_explore_steps)),
        critic_target_tau=float(method.get("critic_target_tau", 0.01)),
        stddev_schedule=str(method.get("stddev_schedule", "0.1")),
        stddev_clip=float(method.get("stddev_clip", 0.3)),
        use_augmentation=bool(method.get("use_augmentation", True)),
        augmentation_pad=int(method.get("augmentation_pad", 4)),
        num_critics=int(method.get("num_critics", 2)),
        feature_dim=feature_dim,
        actor_uses_time=bool(method.get("actor_uses_time", False)),
        always_bootstrap=bool(method.get("always_bootstrap", False)),
        bc_lambda=float(method.get("bc_lambda", 0.0)),
        actor_grad_clip=_optional_float(method.get("actor_grad_clip", None)),
        critic_grad_clip=_optional_float(method.get("critic_grad_clip", None)),
        weight_decay=float(method.get("weight_decay", 0.0)),
        model=rl_model_spec_from_cfg(cfg),
    )


def _dense(
    features: int,
    *,
    name: str,
    use_bias: bool,
    gain: float = 1.0,
) -> nn.Dense:
    return nn.Dense(
        features,
        use_bias=use_bias,
        kernel_init=nn.initializers.orthogonal(gain),
        bias_init=nn.initializers.zeros_init(),
        name=name,
    )


class DrQV2Actor(nn.Module):
    """Reference deterministic DrQ-v2 policy, expressed as a Flax module."""

    feature_dim: int
    hidden_dims: tuple[int, ...]
    action_dim: int
    activation_name: str = "relu"
    feature_norm: str = "layer"
    linear_bias: bool = True

    @nn.compact
    def __call__(self, features: jax.Array) -> jax.Array:
        x = features.astype(jnp.float32)
        if x.ndim > 2:
            x = x.reshape((x.shape[0], -1))
        x = _dense(
            self.feature_dim,
            name="trunk_dense",
            use_bias=self.linear_bias,
        )(x)
        if self.feature_norm == "layer":
            x = nn.LayerNorm(epsilon=1e-5, name="trunk_norm")(x)
        elif self.feature_norm not in {"none", "identity"}:
            raise ValueError(
                f"Unsupported DrQ-v2 feature norm {self.feature_norm!r}."
            )
        x = jnp.tanh(x)
        for index, width in enumerate(self.hidden_dims):
            x = _dense(
                width,
                name=f"policy_dense_{index}",
                use_bias=self.linear_bias,
            )(x)
            x = activation(x, self.activation_name)
        mean = _dense(
            self.action_dim,
            name="policy_out",
            use_bias=self.linear_bias,
        )(x)
        return jnp.tanh(mean)


class DrQV2Critic(nn.Module):
    """Configurable ensemble critic with the reference DrQ-v2 bottleneck."""

    feature_dim: int
    hidden_dims: tuple[int, ...]
    num_critics: int = 2
    activation_name: str = "relu"
    feature_norm: str = "layer"
    linear_bias: bool = True

    @nn.compact
    def __call__(self, features: jax.Array, actions: jax.Array) -> jax.Array:
        state = features.astype(jnp.float32)
        if state.ndim > 2:
            state = state.reshape((state.shape[0], -1))
        state = _dense(
            self.feature_dim,
            name="trunk_dense",
            use_bias=self.linear_bias,
        )(state)
        if self.feature_norm == "layer":
            state = nn.LayerNorm(epsilon=1e-5, name="trunk_norm")(state)
        elif self.feature_norm not in {"none", "identity"}:
            raise ValueError(
                f"Unsupported DrQ-v2 feature norm {self.feature_norm!r}."
            )
        state = jnp.tanh(state)
        state_action = jnp.concatenate([state, actions], axis=-1)

        q_values = []
        for critic_index in range(self.num_critics):
            value = state_action
            for layer_index, width in enumerate(self.hidden_dims):
                value = _dense(
                    width,
                    name=f"q{critic_index + 1}_dense_{layer_index}",
                    use_bias=self.linear_bias,
                )(value)
                value = activation(value, self.activation_name)
            value = _dense(
                1,
                name=f"q{critic_index + 1}_out",
                use_bias=self.linear_bias,
            )(value)
            q_values.append(value[..., 0])
        return jnp.stack(q_values, axis=-1)


def drqv2_sample_unit_action(
    mean: jax.Array,
    key: jax.Array,
    stddev: jax.Array | float,
    *,
    noise_clip: float | None,
    eps: float = 1e-6,
) -> jax.Array:
    """Sample the reference truncated-normal action in normalized coordinates."""

    noise = jax.random.normal(key, mean.shape, dtype=mean.dtype) * jnp.asarray(
        stddev,
        dtype=mean.dtype,
    )
    if noise_clip is not None:
        noise = jnp.clip(noise, -float(noise_clip), float(noise_clip))
    raw_action = mean + noise
    clipped_action = jnp.clip(raw_action, -1.0 + eps, 1.0 - eps)
    # Match the official straight-through clamp: bounded forward values while
    # preserving gradients through the unclipped policy sample.
    return raw_action + jax.lax.stop_gradient(clipped_action - raw_action)


def drqv2_td_target(
    rewards: jax.Array,
    discounts: jax.Array,
    bootstrap: jax.Array,
    next_q: jax.Array,
) -> jax.Array:
    """Canonical DrQ-v2 scalar Bellman target (with no entropy term)."""

    return rewards + bootstrap * discounts * next_q


class DrQV2(JaxRLMethodBase, OffPolicyMethod):
    """Modular, fully JIT-compiled JAX/Flax implementation of DrQ-v2."""

    def __init__(
        self,
        actor_lr: float,
        critic_lr: float,
        encoder_lr: float,
        num_train_steps: int,
        num_explore_steps: int,
        critic_target_tau: float,
        stddev_schedule: str,
        stddev_clip: float,
        use_augmentation: bool,
        augmentation_pad: int,
        num_critics: int,
        feature_dim: int,
        actor_uses_time: bool,
        always_bootstrap: bool,
        bc_lambda: float,
        weight_decay: float,
        model: RLModelSpec,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        num_train_envs: int,
        num_eval_envs: int,
        replay_alpha: float,
        replay_beta: float,
        frame_stack_on_channel: bool,
        intrinsic_reward_module=None,
        actor_grad_clip: Optional[float] = None,
        critic_grad_clip: Optional[float] = None,
        jit: bool = True,
        platform: str | None = None,
        seed: int = 0,
        update_block_every_steps: int = 1,
    ):
        super().__init__(
            lr=actor_lr,
            adaptive_lr=False,
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
            is_rl=True,
            update_block_every_steps=update_block_every_steps,
        )
        if self.action_sequence != 1:
            raise ValueError("Canonical DrQ-v2 requires action_sequence=1.")
        if num_critics < 2:
            raise ValueError("DrQ-v2 requires at least two critics.")
        if not 0.0 < critic_target_tau <= 1.0:
            raise ValueError("critic_target_tau must be in (0, 1].")
        if stddev_clip < 0.0:
            raise ValueError("stddev_clip must be non-negative.")
        if augmentation_pad < 0:
            raise ValueError("augmentation_pad must be non-negative.")
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive.")
        if bc_lambda < 0.0:
            raise ValueError("bc_lambda must be non-negative.")
        if model.norm not in {"layer", "none", "identity"}:
            raise ValueError(
                "DrQ-v2 model.norm must be 'layer', 'none', or 'identity'."
            )

        self.num_explore_steps = int(num_explore_steps)
        self.critic_target_tau = float(critic_target_tau)
        self.stddev_schedule = str(stddev_schedule)
        self.stddev_clip = float(stddev_clip)
        self.use_augmentation = bool(use_augmentation)
        self.augmentation_pad = int(augmentation_pad)
        self.num_critics = int(num_critics)
        self.feature_dim = int(feature_dim)
        self.actor_uses_time = bool(actor_uses_time)
        self.always_bootstrap = bool(always_bootstrap)
        self.bc_lambda = float(bc_lambda)
        self.critic_grad_clip = critic_grad_clip

        input_dim = self._setup_rl_features(model, seed=seed)
        if self.use_pixels and self.use_augmentation and not self._trainable_encoder:
            raise ValueError(
                "DrQ-v2 image augmentation requires a trainable encoder and raw "
                "RGB replay observations."
            )
        if (
            self.use_pixels
            and self.use_augmentation
            and model.encoder_model is not None
            and model.encoder_model.use_plucker
        ):
            raise ValueError(
                "DrQ-v2 random shifts do not transform Plucker rays. Disable "
                "use_augmentation or use an RGB-only encoder."
            )

        actor_input_dim = int(input_dim)
        if self._time_feature_dim and not self.actor_uses_time:
            actor_input_dim -= self._time_feature_dim
        if actor_input_dim < 1:
            raise ValueError("DrQ-v2 actor has no observation features.")
        self._actor_input_dim = actor_input_dim
        self.action_low, self.action_high = self._action_bounds()

        self.actor_model = DrQV2Actor(
            feature_dim=self.feature_dim,
            hidden_dims=model.hidden_dims,
            action_dim=self.action_dim,
            activation_name=model.activation,
            feature_norm=model.norm,
            linear_bias=model.linear_bias,
        )
        self.critic_model = DrQV2Critic(
            feature_dim=self.feature_dim,
            hidden_dims=model.hidden_dims,
            num_critics=self.num_critics,
            activation_name=model.activation,
            feature_norm=model.norm,
            linear_bias=model.linear_bias,
        )
        dummy_actor_features = jnp.zeros((1, actor_input_dim), dtype=jnp.float32)
        dummy_critic_features = jnp.zeros((1, input_dim), dtype=jnp.float32)
        dummy_action = jnp.zeros((1, self.action_dim), dtype=jnp.float32)
        actor_key, critic_key = jax.random.split(self.rng_key)
        actor_params = self.actor_model.init(actor_key, dummy_actor_features)
        critic_params = self.critic_model.init(
            critic_key,
            dummy_critic_features,
            dummy_action,
        )
        self.params = {"actor": actor_params, "critic": critic_params}
        if self._trainable_encoder:
            self.params["encoder"] = self._encoder_params
        self.target_critic_params = critic_params

        actor_transforms = []
        if actor_grad_clip is not None:
            actor_transforms.append(
                self.optax.clip_by_global_norm(float(actor_grad_clip))
            )
        actor_transforms.append(
            self.optax.adamw(float(actor_lr), weight_decay=float(weight_decay))
        )
        critic_transforms = []
        if critic_grad_clip is not None:
            critic_transforms.append(
                self.optax.clip_by_global_norm(float(critic_grad_clip))
            )
        critic_transforms.append(
            self.optax.adamw(float(critic_lr), weight_decay=float(weight_decay))
        )
        self.actor_optimizer = self.optax.chain(*actor_transforms)
        self.critic_optimizer = self.optax.chain(*critic_transforms)
        self.encoder_optimizer = None
        self.opt_state = {
            "actor": self.actor_optimizer.init(self.params["actor"]),
            "critic": self.critic_optimizer.init(self.params["critic"]),
        }
        if self._trainable_encoder:
            encoder_transforms = []
            if critic_grad_clip is not None:
                encoder_transforms.append(
                    self.optax.clip_by_global_norm(float(critic_grad_clip))
                )
            encoder_transforms.append(
                self.optax.adamw(
                    float(encoder_lr),
                    weight_decay=float(weight_decay),
                )
            )
            self.encoder_optimizer = self.optax.chain(*encoder_transforms)
            self.opt_state["encoder"] = self.encoder_optimizer.init(
                self.params["encoder"]
            )

        update_fn = self._build_update_fn()
        sample_fn = self._build_action_fn(deterministic=False)
        mean_fn = self._build_action_fn(deterministic=True)
        if self._jit_enabled:
            update_fn = jax.jit(update_fn)
            sample_fn = jax.jit(sample_fn)
            mean_fn = jax.jit(mean_fn)
        self._update_impl = update_fn
        self._sample_action = sample_fn
        self._mean_action = mean_fn

    def _actor_features(self, features: jax.Array) -> jax.Array:
        if not self.actor_uses_time and self._time_feature_dim:
            return features[..., : -self._time_feature_dim]
        return features

    def _build_action_fn(self, *, deterministic: bool):
        def action_fn(params, obs_inputs, key, stddev):
            features = self._rl_features(
                params.get("encoder", None),
                obs_inputs,
                stop_gradient=True,
            )
            mean = self.actor_model.apply(
                params["actor"],
                self._actor_features(features),
            )
            unit_action = (
                mean
                if deterministic
                else drqv2_sample_unit_action(
                    mean,
                    key,
                    stddev,
                    noise_clip=None,
                )
            )
            return scale_unit_action(
                unit_action,
                self.action_low,
                self.action_high,
            )

        return action_fn

    def _build_update_fn(self):
        actor_optimizer = self.actor_optimizer
        critic_optimizer = self.critic_optimizer
        encoder_optimizer = self.encoder_optimizer
        tau = self.critic_target_tau
        use_augmentation = self.use_augmentation
        augmentation_pad = self.augmentation_pad
        stddev_clip = self.stddev_clip
        bc_lambda = self.bc_lambda
        num_critics = self.num_critics

        def update_fn(
            params,
            target_critic_params,
            opt_state,
            obs_inputs,
            next_obs_inputs,
            actions,
            rewards,
            discounts,
            bootstrap,
            demos,
            loss_weights,
            stddev,
            key,
        ):
            (
                key,
                obs_augment_key,
                next_augment_key,
                next_action_key,
                actor_action_key,
            ) = jax.random.split(key, 5)
            if (
                use_augmentation
                and isinstance(obs_inputs, dict)
                and "rgb" in obs_inputs
            ):
                obs_inputs = dict(obs_inputs)
                next_obs_inputs = dict(next_obs_inputs)
                obs_inputs["rgb"] = random_shift_rgb(
                    obs_inputs["rgb"],
                    obs_augment_key,
                    augmentation_pad,
                )
                next_obs_inputs["rgb"] = random_shift_rgb(
                    next_obs_inputs["rgb"],
                    next_augment_key,
                    augmentation_pad,
                )

            def critic_loss_fn(critic_params, encoder_params):
                features = self._rl_features(encoder_params, obs_inputs)
                next_features = self._rl_features(
                    encoder_params,
                    next_obs_inputs,
                    stop_gradient=True,
                )
                next_mean = self.actor_model.apply(
                    params["actor"],
                    self._actor_features(next_features),
                )
                next_unit_action = drqv2_sample_unit_action(
                    next_mean,
                    next_action_key,
                    stddev,
                    noise_clip=stddev_clip,
                )
                next_action = scale_unit_action(
                    next_unit_action,
                    self.action_low,
                    self.action_high,
                )
                target_q_values = self.critic_model.apply(
                    target_critic_params,
                    next_features,
                    next_action,
                )
                target_q = drqv2_td_target(
                    rewards,
                    discounts,
                    bootstrap,
                    jnp.min(target_q_values, axis=-1),
                )
                target_q = jax.lax.stop_gradient(target_q)
                q_values = self.critic_model.apply(
                    critic_params,
                    features,
                    actions,
                )
                squared_error = jnp.square(q_values - target_q[:, None])
                # The reference implementation adds Q1 and Q2 MSE losses.
                per_sample_loss = jnp.sum(squared_error, axis=-1)
                priority_error = jnp.mean(squared_error, axis=-1)
                critic_loss = jnp.mean(per_sample_loss * loss_weights)
                return critic_loss, (
                    priority_error,
                    q_values,
                    target_q,
                    jax.lax.stop_gradient(features),
                )

            encoder_params = params.get("encoder", None)
            if self._trainable_encoder:
                (critic_loss, critic_aux), critic_grads = jax.value_and_grad(
                    critic_loss_fn,
                    argnums=(0, 1),
                    has_aux=True,
                )(params["critic"], encoder_params)
                critic_param_grads, encoder_grads = critic_grads
            else:
                def critic_only_loss(critic_params):
                    return critic_loss_fn(critic_params, None)

                (critic_loss, critic_aux), critic_param_grads = (
                    jax.value_and_grad(critic_only_loss, has_aux=True)(
                        params["critic"]
                    )
                )
                encoder_grads = None
            critic_updates, critic_opt_state = critic_optimizer.update(
                critic_param_grads,
                opt_state["critic"],
                params["critic"],
            )
            params = dict(params)
            params["critic"] = self.optax.apply_updates(
                params["critic"],
                critic_updates,
            )
            next_opt_state = dict(opt_state)
            next_opt_state["critic"] = critic_opt_state
            if self._trainable_encoder:
                encoder_updates, encoder_opt_state = encoder_optimizer.update(
                    encoder_grads,
                    opt_state["encoder"],
                    params["encoder"],
                )
                params["encoder"] = self.optax.apply_updates(
                    params["encoder"],
                    encoder_updates,
                )
                next_opt_state["encoder"] = encoder_opt_state

            # Reuse the pre-critic-step representation exactly as the
            # reference implementation does (and avoid a second CNN pass).
            critic_actor_features = critic_aux[3]
            actor_features = self._actor_features(critic_actor_features)
            replay_unit_actions = unscale_action(
                actions,
                self.action_low,
                self.action_high,
            )

            def actor_loss_fn(actor_params):
                mean = self.actor_model.apply(actor_params, actor_features)
                unit_action = drqv2_sample_unit_action(
                    mean,
                    actor_action_key,
                    stddev,
                    noise_clip=stddev_clip,
                )
                sampled_action = scale_unit_action(
                    unit_action,
                    self.action_low,
                    self.action_high,
                )
                q_values = self.critic_model.apply(
                    params["critic"],
                    critic_actor_features,
                    sampled_action,
                )
                rl_loss = jnp.mean(
                    -jnp.min(q_values, axis=-1) * loss_weights
                )
                per_demo_bc = jnp.sum(
                    jnp.square(mean - replay_unit_actions),
                    axis=-1,
                )
                demo_count = jnp.sum(demos)
                bc_loss = bc_lambda * jnp.where(
                    demo_count > 0,
                    jnp.sum(per_demo_bc * demos) / jnp.maximum(demo_count, 1.0),
                    0.0,
                )
                return rl_loss + bc_loss, (
                    rl_loss,
                    bc_loss,
                    jnp.mean(mean),
                )

            (actor_loss, actor_aux), actor_grads = jax.value_and_grad(
                actor_loss_fn,
                has_aux=True,
            )(params["actor"])
            actor_updates, actor_opt_state = actor_optimizer.update(
                actor_grads,
                opt_state["actor"],
                params["actor"],
            )
            params["actor"] = self.optax.apply_updates(
                params["actor"],
                actor_updates,
            )
            next_opt_state["actor"] = actor_opt_state

            target_critic_params = jax.tree.map(
                lambda target, online: (1.0 - tau) * target + tau * online,
                target_critic_params,
                params["critic"],
            )
            priority_error, q_values, target_q, _ = critic_aux
            priority = jnp.sqrt(priority_error + 1e-10)
            priority = priority / jnp.maximum(jnp.max(priority), 1e-10)
            actor_rl_loss, actor_bc_loss, mean_action = actor_aux
            metrics = {
                "actor_loss": actor_loss,
                "actor_rl_loss": actor_rl_loss,
                "actor_bc_loss": actor_bc_loss,
                "critic_loss": critic_loss,
                "critic_q": jnp.mean(q_values),
                "critic_target_q": jnp.mean(target_q),
                "mean_action": mean_action,
                "ratio_of_demos": jnp.mean(demos),
                "stddev": stddev,
            }
            for critic_index in range(num_critics):
                metrics[f"critic_q{critic_index + 1}"] = jnp.mean(
                    q_values[:, critic_index]
                )
            return (
                params,
                target_critic_params,
                next_opt_state,
                key,
                priority,
                metrics,
            )

        return update_fn

    def act(self, observations: dict, step: int, eval_mode: bool):
        if step < self.num_explore_steps and not eval_mode:
            low = np.asarray(self.action_space.low, dtype=np.float32)
            high = np.asarray(self.action_space.high, dtype=np.float32)
            return np.random.uniform(
                low,
                high,
                size=(self.num_train_envs,) + self.action_space.shape,
            ).astype(np.float32)

        obs_inputs = self._prepare_rl_obs_inputs(observations)
        stddev = np.float32(utils.schedule(self.stddev_schedule, step))
        if eval_mode:
            actions = self._mean_action(
                self.params,
                obs_inputs,
                self.rng_key,
                stddev,
            )
        else:
            self.rng_key, sample_key = jax.random.split(self.rng_key)
            actions = self._sample_action(
                self.params,
                obs_inputs,
                sample_key,
                stddev,
            )
        self._block(actions)
        actions = np.asarray(jax.device_get(actions), dtype=np.float32)
        return actions.reshape((actions.shape[0], 1, self.action_dim))

    def update(
        self,
        replay_iter: Iterator[dict],
        step: int,
        replay_buffer: ReplayBuffer = None,
    ) -> dict[str, np.ndarray]:
        batch = next(replay_iter)
        obs_inputs = self._prepare_rl_obs_inputs(batch)
        next_obs_inputs = self._next_rl_obs_inputs(batch)
        actions = self._as_jax_array(batch["action"], self.jnp.float32).reshape(
            (batch["action"].shape[0], -1)
        )
        rewards = self._as_jax_array(batch["reward"], self.jnp.float32).reshape(-1)
        discounts = self._as_jax_array(
            batch.get("discount", np.ones_like(batch["reward"])),
            self.jnp.float32,
        ).reshape(-1)
        terminal = self._as_jax_array(batch["terminal"], self.jnp.float32).reshape(-1)
        bootstrap = (
            self.jnp.ones_like(terminal)
            if self.always_bootstrap
            else 1.0 - terminal
        )
        demos = self._as_jax_array(
            batch.get("demo", np.zeros_like(batch["reward"])),
            self.jnp.float32,
        ).reshape(-1)
        loss_weights = self._loss_weights(batch)
        stddev = self.jnp.asarray(
            utils.schedule(self.stddev_schedule, step),
            dtype=self.jnp.float32,
        )

        start_time = time.perf_counter()
        (
            self.params,
            self.target_critic_params,
            self.opt_state,
            self.rng_key,
            priority,
            jax_metrics,
        ) = self._update_impl(
            self.params,
            self.target_critic_params,
            self.opt_state,
            obs_inputs,
            next_obs_inputs,
            actions,
            rewards,
            discounts,
            bootstrap,
            demos,
            loss_weights,
            stddev,
            self.rng_key,
        )
        uses_priorities = self._uses_replay_priorities(replay_buffer)
        if self._should_block_update(uses_priorities):
            self._block(jax_metrics["critic_loss"], priority)
        elapsed = time.perf_counter() - start_time
        self._update_step_count += 1
        if uses_priorities:
            self._maybe_update_priorities(
                replay_buffer,
                batch,
                np.asarray(jax.device_get(priority), dtype=np.float32),
            )
        metrics = {}
        if self.logging:
            metrics = {
                key: float(np.asarray(jax.device_get(value)))
                for key, value in jax_metrics.items()
            }
            metrics["backend/update_time_sec"] = elapsed
        self._first_update_completed = True
        return metrics

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["target_critic_params"] = self._tree_to_numpy(
            self.target_critic_params
        )
        if self.encoder is not None:
            state["encoder_state"] = self._tree_to_numpy(
                self.encoder.frozen_state_dict()
            )
        return state

    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict)
        self.target_critic_params = self._tree_from_numpy(
            state_dict.get("target_critic_params", self.params["critic"])
        )
        if self.encoder is not None:
            self.encoder.load_frozen_state_dict(state_dict.get("encoder_state"))


__all__ = [
    "DrQV2",
    "DrQV2Actor",
    "DrQV2Critic",
    "DrQV2Spec",
    "drqv2_sample_unit_action",
    "drqv2_spec_from_cfg",
    "drqv2_td_target",
]
