"""Proximal Policy Optimization implemented entirely in JAX/Flax."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig

from robobase.method.core import OnPolicyMethod
from robobase.method.rl_common import (
    ActorCritic,
    JaxRLMethodBase,
    RLModelSpec,
    normal_entropy,
    normal_log_prob,
    rl_model_spec_from_cfg,
)
from robobase.replay_buffer.replay_buffer import ReplayBuffer


@dataclass(frozen=True)
class PPOSpec:
    lr: float
    num_train_steps: int
    rollout_steps: int
    batch_size: int
    num_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    clip_range_vf: float | None
    normalize_advantage: bool
    entropy_coef: float
    value_coef: float
    max_grad_norm: float | None
    target_kl: float | None
    model: RLModelSpec


def ppo_spec_from_cfg(cfg: DictConfig) -> PPOSpec:
    method = cfg.method
    clip_range_vf = method.get("clip_range_vf", None)
    target_kl = method.get("target_kl", None)
    return PPOSpec(
        lr=float(method.get("lr", 3e-4)),
        num_train_steps=int(method.get("num_train_steps", cfg.num_train_frames)),
        rollout_steps=int(method.get("rollout_steps", 2048)),
        batch_size=int(method.get("batch_size", 64)),
        num_epochs=int(method.get("num_epochs", 10)),
        gamma=float(method.get("gamma", cfg.replay.gamma)),
        gae_lambda=float(method.get("gae_lambda", 0.95)),
        clip_range=float(method.get("clip_range", 0.2)),
        clip_range_vf=(
            None if clip_range_vf is None else float(clip_range_vf)
        ),
        normalize_advantage=bool(method.get("normalize_advantage", True)),
        entropy_coef=float(method.get("entropy_coef", 0.0)),
        value_coef=float(method.get("value_coef", 0.5)),
        max_grad_norm=(
            None
            if method.get("max_grad_norm", None) is None
            else float(method.max_grad_norm)
        ),
        target_kl=None if target_kl is None else float(target_kl),
        model=rl_model_spec_from_cfg(cfg),
    )


def generalized_advantage_estimate(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE while bootstrapping time limits but ending recursion at resets."""

    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    next_values = np.asarray(next_values, dtype=np.float32)
    terminated = np.asarray(terminated, dtype=np.float32)
    truncated = np.asarray(truncated, dtype=np.float32)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    next_advantage = np.zeros(rewards.shape[1], dtype=np.float32)
    for step in reversed(range(rewards.shape[0])):
        bootstrap = 1.0 - terminated[step]
        episode_continues = 1.0 - np.maximum(
            terminated[step],
            truncated[step],
        )
        delta = (
            rewards[step]
            + gamma * bootstrap * next_values[step]
            - values[step]
        )
        next_advantage = (
            delta
            + gamma * gae_lambda * episode_continues * next_advantage
        )
        advantages[step] = next_advantage
    return advantages, advantages + values


def generalized_advantage_estimate_jax(
    rewards: jax.Array,
    values: jax.Array,
    next_values: jax.Array,
    terminated: jax.Array,
    truncated: jax.Array,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Device-side reverse scan used by the PPO runtime."""

    def scan_step(next_advantage, inputs):
        reward, value, next_value, terminal, truncation = inputs
        bootstrap = 1.0 - terminal.astype(jnp.float32)
        episode_continues = 1.0 - jnp.maximum(
            terminal.astype(jnp.float32),
            truncation.astype(jnp.float32),
        )
        delta = reward + gamma * bootstrap * next_value - value
        advantage = (
            delta
            + gamma * gae_lambda * episode_continues * next_advantage
        )
        return advantage, advantage

    _, reversed_advantages = jax.lax.scan(
        scan_step,
        jnp.zeros_like(rewards[0]),
        (rewards, values, next_values, terminated, truncated),
        reverse=True,
    )
    return reversed_advantages, reversed_advantages + values


class PPO(JaxRLMethodBase, OnPolicyMethod):
    """Clipped PPO matching Stable-Baselines3's continuous Box objective."""

    on_policy = True

    def __init__(
        self,
        lr: float,
        num_train_steps: int,
        rollout_steps: int,
        batch_size: int,
        num_epochs: int,
        gamma: float,
        gae_lambda: float,
        clip_range: float,
        clip_range_vf: float | None,
        normalize_advantage: bool,
        entropy_coef: float,
        value_coef: float,
        target_kl: float | None,
        model: RLModelSpec,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        num_train_envs: int,
        num_eval_envs: int,
        replay_alpha: float,
        replay_beta: float,
        frame_stack_on_channel: bool,
        intrinsic_reward_module=None,
        max_grad_norm: Optional[float] = 0.5,
        jit: bool = True,
        platform: str | None = None,
        seed: int = 0,
        update_block_every_steps: int = 1,
    ):
        super().__init__(
            lr=lr,
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
            actor_grad_clip=max_grad_norm,
            jit=jit,
            platform=platform,
            seed=seed,
            is_rl=True,
            update_block_every_steps=update_block_every_steps,
        )
        if self.action_sequence != 1:
            raise ValueError("PPO requires action_sequence=1.")
        if rollout_steps < 1 or batch_size < 2 or num_epochs < 1:
            raise ValueError(
                "PPO requires rollout_steps >= 1, batch_size >= 2, num_epochs >= 1."
            )
        rollout_size = int(rollout_steps) * int(num_train_envs)
        if rollout_size <= 1 and normalize_advantage:
            raise ValueError(
                "rollout_steps * num_train_envs must exceed one when normalizing advantages."
            )
        if batch_size > rollout_size:
            raise ValueError("PPO batch_size cannot exceed the rollout size.")
        if rollout_size % batch_size:
            raise ValueError(
                "For stable JIT shapes, rollout_steps * num_train_envs must be "
                "divisible by PPO batch_size."
            )
        if not 0.0 < gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("PPO gamma and gae_lambda are outside valid ranges.")
        if clip_range <= 0.0:
            raise ValueError("PPO clip_range must be positive.")

        self.rollout_steps = int(rollout_steps)
        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_range = float(clip_range)
        self.clip_range_vf = (
            None if clip_range_vf is None else float(clip_range_vf)
        )
        self.normalize_advantage = bool(normalize_advantage)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.target_kl = None if target_kl is None else float(target_kl)
        input_dim = self._setup_rl_features(model, seed=seed)
        self.action_low, self.action_high = self._action_bounds()
        self.policy_model = ActorCritic(
            hidden_dims=model.hidden_dims,
            action_dim=self.action_dim,
            activation_name=model.activation,
        )
        dummy_features = jnp.zeros((1, input_dim), dtype=jnp.float32)
        policy_params = self.policy_model.init(self.rng_key, dummy_features)
        self.params = {"policy": policy_params}
        if self._trainable_encoder:
            self.params["encoder"] = self._encoder_params
        transforms = []
        if max_grad_norm is not None:
            transforms.append(self.optax.clip_by_global_norm(float(max_grad_norm)))
        transforms.append(self.optax.adam(float(lr), eps=1e-5))
        self.optimizer = self.optax.chain(*transforms)
        self.opt_state = self.optimizer.init(self.params)

        predict_fn = self._build_predict_fn()
        value_fn = self._build_value_fn()
        update_fn = self._build_update_fn()
        if self._jit_enabled:
            predict_fn = jax.jit(predict_fn)
            value_fn = jax.jit(value_fn)
            update_fn = jax.jit(update_fn)
        self._predict_impl = predict_fn
        self._value_impl = value_fn
        self._minibatch_update_impl = update_fn
        self._rollout: list[dict[str, Any]] = []
        self._pending_transition: dict[str, Any] | None = None

    @property
    def rollout_ready(self) -> bool:
        return len(self._rollout) >= self.rollout_steps

    def _build_predict_fn(self):
        def predict_fn(params, obs_inputs, key, deterministic):
            features = self._rl_features(
                params.get("encoder", None),
                obs_inputs,
                stop_gradient=True,
            )
            mean, log_std, values = self.policy_model.apply(
                params["policy"],
                features,
            )
            noise = jax.random.normal(key, mean.shape)
            raw_action = jax.lax.cond(
                deterministic,
                lambda _: mean,
                lambda _: mean + jnp.exp(log_std) * noise,
                operand=None,
            )
            log_prob = normal_log_prob(raw_action, mean, log_std)
            return raw_action, log_prob, values

        return predict_fn

    def _build_value_fn(self):
        def value_fn(params, obs_inputs):
            features = self._rl_features(
                params.get("encoder", None),
                obs_inputs,
                stop_gradient=True,
            )
            return self.policy_model.apply(params["policy"], features)[2]

        return value_fn

    def _build_update_fn(self):
        optimizer = self.optimizer
        clip_range = self.clip_range
        clip_range_vf = self.clip_range_vf
        normalize_advantage = self.normalize_advantage
        entropy_coef = self.entropy_coef
        value_coef = self.value_coef

        def update_fn(
            params,
            opt_state,
            obs_inputs,
            raw_actions,
            old_log_prob,
            old_values,
            advantages,
            returns,
        ):
            def loss_fn(current_params):
                features = self._rl_features(
                    current_params.get("encoder", None),
                    obs_inputs,
                )
                mean, log_std, values = self.policy_model.apply(
                    current_params["policy"],
                    features,
                )
                log_prob = normal_log_prob(raw_actions, mean, log_std)
                normalized_advantages = advantages
                if normalize_advantage:
                    normalized_advantages = (
                        advantages - jnp.mean(advantages)
                    ) / (jnp.std(advantages) + 1e-8)
                log_ratio = log_prob - old_log_prob
                ratio = jnp.exp(log_ratio)
                policy_loss = -jnp.mean(
                    jnp.minimum(
                        normalized_advantages * ratio,
                        normalized_advantages
                        * jnp.clip(ratio, 1.0 - clip_range, 1.0 + clip_range),
                    )
                )
                if clip_range_vf is None:
                    value_prediction = values
                else:
                    value_prediction = old_values + jnp.clip(
                        values - old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = jnp.mean(jnp.square(returns - value_prediction))
                entropy = jnp.mean(normal_entropy(log_std))
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
                approx_kl = jnp.mean((jnp.exp(log_ratio) - 1.0) - log_ratio)
                clip_fraction = jnp.mean(
                    (jnp.abs(ratio - 1.0) > clip_range).astype(jnp.float32)
                )
                return loss, {
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy,
                    "approx_kl": approx_kl,
                    "clip_fraction": clip_fraction,
                }

            (loss, metrics), grads = jax.value_and_grad(
                loss_fn,
                has_aux=True,
            )(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            updated_params = self.optax.apply_updates(params, updates)
            should_apply = jnp.asarray(True)
            if self.target_kl is not None:
                should_apply = metrics["approx_kl"] <= 1.5 * self.target_kl
            params, opt_state = jax.lax.cond(
                should_apply,
                lambda _: (updated_params, new_opt_state),
                lambda _: (params, opt_state),
                operand=None,
            )
            metrics["loss"] = loss
            metrics["update_applied"] = should_apply.astype(jnp.float32)
            return params, opt_state, metrics

        return update_fn

    def act(self, observations: dict, step: int, eval_mode: bool):
        del step
        obs_inputs = self._prepare_rl_obs_inputs(observations)
        self.rng_key, sample_key = jax.random.split(self.rng_key)
        raw_action, log_prob, values = self._predict_impl(
            self.params,
            obs_inputs,
            sample_key,
            jnp.asarray(eval_mode),
        )
        executed_action = jnp.clip(raw_action, self.action_low, self.action_high)
        self._block(executed_action)
        if not eval_mode:
            if self._pending_transition is not None:
                raise RuntimeError(
                    "PPO act() was called twice without observe_transition()."
                )
            self._pending_transition = {
                "obs_inputs": obs_inputs,
                "raw_action": raw_action,
                "log_prob": log_prob,
                "value": values,
            }
        action = np.asarray(jax.device_get(executed_action), dtype=np.float32)
        return action.reshape((action.shape[0], 1, self.action_dim))

    def _bootstrap_observations(
        self,
        next_observations: dict[str, np.ndarray],
        terminated: np.ndarray,
        truncated: np.ndarray,
        next_info: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        observations = {
            key: np.array(value, copy=True)
            for key, value in next_observations.items()
        }
        final_observations = next_info.get("final_observation", None)
        final_mask = next_info.get("_final_observation", None)
        if final_observations is None:
            return observations
        for env_index in range(self.num_train_envs):
            if not truncated[env_index] or terminated[env_index]:
                continue
            if final_mask is not None and not bool(final_mask[env_index]):
                continue
            final_observation = final_observations[env_index]
            for key in observations:
                observations[key][env_index] = final_observation[key]
        return observations

    def observe_transition(
        self,
        *,
        rewards: np.ndarray,
        terminations: np.ndarray,
        truncations: np.ndarray,
        next_observations: dict[str, np.ndarray],
        next_info: dict[str, Any],
    ) -> None:
        if self._pending_transition is None:
            raise RuntimeError("PPO observe_transition() requires a preceding act().")
        bootstrap_observations = self._bootstrap_observations(
            next_observations,
            np.asarray(terminations, dtype=bool),
            np.asarray(truncations, dtype=bool),
            next_info,
        )
        next_inputs = self._prepare_rl_obs_inputs(bootstrap_observations)
        next_values = self._value_impl(self.params, next_inputs)
        self._block(next_values)
        transition = dict(self._pending_transition)
        transition.update(
            rewards=self._as_jax_array(rewards, self.jnp.float32),
            terminated=self._as_jax_array(terminations, self.jnp.bool_),
            truncated=self._as_jax_array(truncations, self.jnp.bool_),
            next_value=next_values,
        )
        self._rollout.append(transition)
        self._pending_transition = None
        if len(self._rollout) > self.rollout_steps:
            raise RuntimeError("PPO rollout exceeded rollout_steps without an update.")

    def _stack_rollout(self):
        stack = lambda *values: self.jnp.stack(values, axis=0)
        obs_inputs = self.jax.tree.map(
            stack,
            *(transition["obs_inputs"] for transition in self._rollout),
        )
        values = self.jnp.stack(
            [transition["value"] for transition in self._rollout], axis=0
        )
        next_values = self.jnp.stack(
            [transition["next_value"] for transition in self._rollout], axis=0
        )
        rewards = self.jnp.stack(
            [transition["rewards"] for transition in self._rollout], axis=0
        )
        terminated = self.jnp.stack(
            [transition["terminated"] for transition in self._rollout], axis=0
        )
        truncated = self.jnp.stack(
            [transition["truncated"] for transition in self._rollout], axis=0
        )
        advantages, returns = generalized_advantage_estimate_jax(
            rewards,
            values,
            next_values,
            terminated,
            truncated,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        rollout_size = self.rollout_steps * self.num_train_envs

        def flatten(value):
            return value.reshape((rollout_size,) + value.shape[2:])

        return {
            "obs_inputs": self.jax.tree.map(flatten, obs_inputs),
            "raw_action": flatten(
                self.jnp.stack(
                    [transition["raw_action"] for transition in self._rollout],
                    axis=0,
                )
            ),
            "log_prob": flatten(
                self.jnp.stack(
                    [transition["log_prob"] for transition in self._rollout],
                    axis=0,
                )
            ),
            "value": flatten(values),
            "advantage": advantages.reshape(-1),
            "return": returns.reshape(-1),
        }

    def update(
        self,
        replay_iter: Iterator[dict] | None,
        step: int,
        replay_buffer: ReplayBuffer = None,
    ) -> dict[str, np.ndarray]:
        del replay_iter, step, replay_buffer
        if not self.rollout_ready:
            return {}
        rollout = self._stack_rollout()
        rollout_size = self.rollout_steps * self.num_train_envs
        metric_sums = None
        metric_batches = 0
        updates = 0
        stop_early = False
        start_time = time.perf_counter()
        for _ in range(self.num_epochs):
            self.rng_key, permutation_key = jax.random.split(self.rng_key)
            permutation = jax.random.permutation(permutation_key, rollout_size)
            for start in range(0, rollout_size, self.batch_size):
                indices = self.jnp.asarray(
                    permutation[start : start + self.batch_size],
                    dtype=self.jnp.int32,
                )
                obs_batch = self.jax.tree.map(lambda value: value[indices], rollout["obs_inputs"])
                self.params, self.opt_state, jax_metrics = (
                    self._minibatch_update_impl(
                        self.params,
                        self.opt_state,
                        obs_batch,
                        rollout["raw_action"][indices],
                        rollout["log_prob"][indices],
                        rollout["value"][indices],
                        rollout["advantage"][indices],
                        rollout["return"][indices],
                    )
                )
                metric_sums = (
                    jax_metrics
                    if metric_sums is None
                    else self.jax.tree.map(
                        lambda total, value: total + value,
                        metric_sums,
                        jax_metrics,
                    )
                )
                metric_batches += 1
                if self.target_kl is not None:
                    approx_kl = float(
                        np.asarray(jax.device_get(jax_metrics["approx_kl"]))
                    )
                    if approx_kl > 1.5 * self.target_kl:
                        stop_early = True
                        break
                updates += 1
            if stop_early:
                break
        self._block(self.params)
        elapsed = time.perf_counter() - start_time
        old_values = np.asarray(jax.device_get(rollout["value"]))
        returns = np.asarray(jax.device_get(rollout["return"]))
        explained_variance = 1.0 - np.var(returns - old_values) / max(
            np.var(returns),
            1e-8,
        )
        self._rollout.clear()
        self._update_step_count += updates
        self._first_update_completed = True
        if not self.logging:
            return {}
        metrics = {
            key: float(np.asarray(jax.device_get(value))) / max(metric_batches, 1)
            for key, value in metric_sums.items()
        }
        metrics.update(
            explained_variance=float(explained_variance),
            epochs_completed=float(updates * self.batch_size / rollout_size),
            early_stop_kl=float(stop_early),
            **{"backend/update_time_sec": elapsed},
        )
        return metrics

    def checkpoint_state_dict(self) -> dict[str, Any]:
        state = super().checkpoint_state_dict()
        state["rollout"] = self._tree_to_numpy(self._rollout)
        state["pending_transition"] = self._tree_to_numpy(self._pending_transition)
        return state

    def load_checkpoint_state_dict(self, state_dict: dict[str, Any]):
        super().load_checkpoint_state_dict(state_dict)
        self._rollout = self._tree_from_numpy(state_dict.get("rollout", []))
        self._pending_transition = self._tree_from_numpy(
            state_dict.get("pending_transition", None)
        )

    def state_dict(self) -> dict:
        state = super().state_dict()
        if self.encoder is not None:
            state["encoder_state"] = self._tree_to_numpy(
                self.encoder.frozen_state_dict()
            )
        return state

    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict)
        if self.encoder is not None:
            self.encoder.load_frozen_state_dict(state_dict.get("encoder_state"))


__all__ = [
    "PPO",
    "PPOSpec",
    "generalized_advantage_estimate",
    "generalized_advantage_estimate_jax",
    "ppo_spec_from_cfg",
]
