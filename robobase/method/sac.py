"""Soft Actor-Critic implemented entirely in JAX/Flax."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Optional

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig

from robobase.method.core import OffPolicyMethod
from robobase.method.rl_common import (
    GaussianActor,
    JaxRLMethodBase,
    RLModelSpec,
    TwinQCritic,
    rl_model_spec_from_cfg,
    scale_unit_action,
    squashed_normal_sample_and_log_prob,
)
from robobase.replay_buffer.replay_buffer import ReplayBuffer


@dataclass(frozen=True)
class SACSpec:
    actor_lr: float
    critic_lr: float
    alpha_lr: float
    num_train_steps: int
    num_explore_steps: int
    critic_target_tau: float
    init_temperature: float
    target_entropy: float | None
    actor_grad_clip: float | None
    critic_grad_clip: float | None
    weight_decay: float
    model: RLModelSpec


def sac_spec_from_cfg(cfg: DictConfig) -> SACSpec:
    method = cfg.method
    target_entropy = method.get("target_entropy", None)
    return SACSpec(
        actor_lr=float(method.get("actor_lr", 3e-4)),
        critic_lr=float(method.get("critic_lr", 3e-4)),
        alpha_lr=float(method.get("alpha_lr", 3e-4)),
        num_train_steps=int(method.get("num_train_steps", cfg.num_train_frames)),
        num_explore_steps=int(method.get("num_explore_steps", cfg.num_explore_steps)),
        critic_target_tau=float(method.get("critic_target_tau", 0.005)),
        init_temperature=float(method.get("init_temperature", 0.1)),
        target_entropy=None if target_entropy is None else float(target_entropy),
        actor_grad_clip=(
            None
            if method.get("actor_grad_clip", None) is None
            else float(method.actor_grad_clip)
        ),
        critic_grad_clip=(
            None
            if method.get("critic_grad_clip", None) is None
            else float(method.critic_grad_clip)
        ),
        weight_decay=float(method.get("weight_decay", 0.0)),
        model=rl_model_spec_from_cfg(cfg),
    )


class SAC(JaxRLMethodBase, OffPolicyMethod):
    """Continuous-action SAC with twin Q networks and automatic temperature."""

    def __init__(
        self,
        actor_lr: float,
        critic_lr: float,
        alpha_lr: float,
        num_train_steps: int,
        num_explore_steps: int,
        critic_target_tau: float,
        init_temperature: float,
        target_entropy: float | None,
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
            raise ValueError("SAC requires action_sequence=1.")
        if init_temperature <= 0:
            raise ValueError("init_temperature must be positive.")
        if not 0.0 < critic_target_tau <= 1.0:
            raise ValueError("critic_target_tau must be in (0, 1].")

        self.num_explore_steps = int(num_explore_steps)
        self.critic_target_tau = float(critic_target_tau)
        self.target_entropy = (
            -float(self.action_dim) if target_entropy is None else float(target_entropy)
        )
        self.critic_grad_clip = critic_grad_clip
        input_dim = self._setup_rl_features(model, seed=seed)
        self.action_low, self.action_high = self._action_bounds()

        self.actor_model = GaussianActor(
            hidden_dims=model.hidden_dims,
            action_dim=self.action_dim,
            activation_name=model.activation,
            state_dependent_std=True,
        )
        self.critic_model = TwinQCritic(
            hidden_dims=model.hidden_dims,
            activation_name=model.activation,
        )
        dummy_features = jnp.zeros((1, input_dim), dtype=jnp.float32)
        dummy_action = jnp.zeros((1, self.action_dim), dtype=jnp.float32)
        actor_key, critic_key = jax.random.split(self.rng_key)
        actor_params = self.actor_model.init(actor_key, dummy_features)
        critic_params = self.critic_model.init(
            critic_key,
            dummy_features,
            dummy_action,
        )
        self.params = {
            "actor": actor_params,
            "critic": critic_params,
            "log_alpha": jnp.asarray(np.log(init_temperature), dtype=jnp.float32),
        }
        if self._trainable_encoder:
            self.params["encoder"] = self._encoder_params
        self.target_critic_params = critic_params

        critic_transforms = []
        if critic_grad_clip is not None:
            critic_transforms.append(
                self.optax.clip_by_global_norm(float(critic_grad_clip))
            )
        critic_transforms.append(
            self.optax.adamw(float(critic_lr), weight_decay=float(weight_decay))
        )
        actor_transforms = []
        if actor_grad_clip is not None:
            actor_transforms.append(
                self.optax.clip_by_global_norm(float(actor_grad_clip))
            )
        actor_transforms.append(self.optax.adam(float(actor_lr)))
        self.critic_optimizer = self.optax.chain(*critic_transforms)
        self.actor_optimizer = self.optax.chain(*actor_transforms)
        self.alpha_optimizer = self.optax.adam(float(alpha_lr))
        self.opt_state = {
            "critic": self.critic_optimizer.init(self._critic_bundle(self.params)),
            "actor": self.actor_optimizer.init(self.params["actor"]),
            "alpha": self.alpha_optimizer.init(self.params["log_alpha"]),
        }

        update_fn = self._build_update_fn()
        sample_fn = self._build_sample_fn(deterministic=False)
        mean_fn = self._build_sample_fn(deterministic=True)
        if self._jit_enabled:
            update_fn = jax.jit(update_fn)
            sample_fn = jax.jit(sample_fn)
            mean_fn = jax.jit(mean_fn)
        self._update_impl = update_fn
        self._sample_action = sample_fn
        self._mean_action = mean_fn

    def _critic_bundle(self, params):
        bundle = {"critic": params["critic"]}
        if self._trainable_encoder:
            bundle["encoder"] = params["encoder"]
        return bundle

    def _encoder_from_bundle(self, bundle):
        return bundle.get("encoder", None) if self._trainable_encoder else None

    def _build_sample_fn(self, *, deterministic: bool):
        def sample_fn(params, obs_inputs, key):
            features = self._rl_features(
                params.get("encoder", None),
                obs_inputs,
                stop_gradient=True,
            )
            mean, log_std = self.actor_model.apply(params["actor"], features)
            unit_action, _ = squashed_normal_sample_and_log_prob(
                key,
                mean,
                log_std,
                deterministic=deterministic,
            )
            return scale_unit_action(unit_action, self.action_low, self.action_high)

        return sample_fn

    def _build_update_fn(self):
        critic_optimizer = self.critic_optimizer
        actor_optimizer = self.actor_optimizer
        alpha_optimizer = self.alpha_optimizer
        tau = self.critic_target_tau
        target_entropy = self.target_entropy

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
            loss_weights,
            key,
        ):
            key, next_action_key, actor_action_key = jax.random.split(key, 3)
            alpha = jax.lax.stop_gradient(jnp.exp(params["log_alpha"]))

            def critic_loss_fn(critic_bundle):
                encoder_params = self._encoder_from_bundle(critic_bundle)
                features = self._rl_features(encoder_params, obs_inputs)
                next_features = self._rl_features(
                    encoder_params,
                    next_obs_inputs,
                    stop_gradient=True,
                )
                next_mean, next_log_std = self.actor_model.apply(
                    params["actor"],
                    next_features,
                )
                next_unit_action, next_log_prob = (
                    squashed_normal_sample_and_log_prob(
                        next_action_key,
                        next_mean,
                        next_log_std,
                    )
                )
                next_action = scale_unit_action(
                    next_unit_action,
                    self.action_low,
                    self.action_high,
                )
                next_q = self.critic_model.apply(
                    target_critic_params,
                    next_features,
                    next_action,
                )
                target_q = rewards + bootstrap * discounts * (
                    jnp.min(next_q, axis=-1) - alpha * next_log_prob
                )
                target_q = jax.lax.stop_gradient(target_q)
                q_values = self.critic_model.apply(
                    critic_bundle["critic"],
                    features,
                    actions,
                )
                per_sample = jnp.mean(
                    jnp.square(q_values - target_q[:, None]),
                    axis=-1,
                )
                return jnp.mean(per_sample * loss_weights), (
                    per_sample,
                    jnp.mean(q_values),
                    jnp.mean(target_q),
                )

            critic_bundle = self._critic_bundle(params)
            (critic_loss, critic_aux), critic_grads = jax.value_and_grad(
                critic_loss_fn,
                has_aux=True,
            )(critic_bundle)
            critic_updates, critic_opt_state = critic_optimizer.update(
                critic_grads,
                opt_state["critic"],
                critic_bundle,
            )
            critic_bundle = self.optax.apply_updates(critic_bundle, critic_updates)
            params = dict(params)
            params["critic"] = critic_bundle["critic"]
            if self._trainable_encoder:
                params["encoder"] = critic_bundle["encoder"]

            actor_features = self._rl_features(
                params.get("encoder", None),
                obs_inputs,
                stop_gradient=True,
            )

            def actor_loss_fn(actor_params):
                mean, log_std = self.actor_model.apply(actor_params, actor_features)
                unit_action, log_prob = squashed_normal_sample_and_log_prob(
                    actor_action_key,
                    mean,
                    log_std,
                )
                sampled_action = scale_unit_action(
                    unit_action,
                    self.action_low,
                    self.action_high,
                )
                q_values = self.critic_model.apply(
                    params["critic"],
                    actor_features,
                    sampled_action,
                )
                loss = jnp.mean(alpha * log_prob - jnp.min(q_values, axis=-1))
                return loss, log_prob

            (actor_loss, log_prob), actor_grads = jax.value_and_grad(
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

            def alpha_loss_fn(log_alpha):
                return -jnp.mean(
                    log_alpha
                    * jax.lax.stop_gradient(log_prob + target_entropy)
                )

            alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss_fn)(
                params["log_alpha"]
            )
            alpha_updates, alpha_opt_state = alpha_optimizer.update(
                alpha_grad,
                opt_state["alpha"],
                params["log_alpha"],
            )
            params["log_alpha"] = self.optax.apply_updates(
                params["log_alpha"],
                alpha_updates,
            )
            target_critic_params = jax.tree.map(
                lambda target, online: (1.0 - tau) * target + tau * online,
                target_critic_params,
                params["critic"],
            )
            per_sample, mean_q, mean_target_q = critic_aux
            priority = jnp.sqrt(per_sample + 1e-10)
            priority = priority / jnp.maximum(jnp.max(priority), 1e-10)
            metrics = {
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "alpha_loss": alpha_loss,
                "alpha": jnp.exp(params["log_alpha"]),
                "actor_logprob": jnp.mean(log_prob),
                "critic_q": mean_q,
                "critic_target_q": mean_target_q,
            }
            return (
                params,
                target_critic_params,
                {
                    "critic": critic_opt_state,
                    "actor": actor_opt_state,
                    "alpha": alpha_opt_state,
                },
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
        self.rng_key, sample_key = jax.random.split(self.rng_key)
        sample_fn = self._mean_action if eval_mode else self._sample_action
        actions = sample_fn(self.params, obs_inputs, sample_key)
        self._block(actions)
        actions = np.asarray(jax.device_get(actions), dtype=np.float32)
        return actions.reshape((actions.shape[0], 1, self.action_dim))

    def update(
        self,
        replay_iter: Iterator[dict],
        step: int,
        replay_buffer: ReplayBuffer = None,
    ) -> dict[str, np.ndarray]:
        del step
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
        bootstrap = 1.0 - terminal
        loss_weights = self._loss_weights(batch)

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
            loss_weights,
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


__all__ = ["SAC", "SACSpec", "sac_spec_from_cfg"]
