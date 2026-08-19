"""Continuous-action coarse-to-fine Q-network implemented in pure JAX."""

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
from robobase.method.rl_common import JaxRLMethodBase, RLModelSpec, activation
from robobase.method.rl_common import rl_model_spec_from_cfg
from robobase.replay_buffer.replay_buffer import ReplayBuffer


@dataclass(frozen=True)
class CQNSpec:
    critic_lr: float
    num_train_steps: int
    num_explore_steps: int
    critic_target_tau: float
    critic_grad_clip: float | None
    weight_decay: float
    levels: int
    bins: int
    atoms: int
    v_min: float
    v_max: float
    critic_lambda: float
    centralized_critic: bool
    use_dueling: bool
    always_bootstrap: bool
    stddev_schedule: str
    bc_lambda: float
    bc_margin: float
    use_target_network_for_rollout: bool
    num_update_steps: int
    model: RLModelSpec


def cqn_spec_from_cfg(cfg: DictConfig) -> CQNSpec:
    method = cfg.method
    return CQNSpec(
        critic_lr=float(method.get("critic_lr", 1e-4)),
        num_train_steps=int(method.get("num_train_steps", cfg.num_train_frames)),
        num_explore_steps=int(method.get("num_explore_steps", cfg.num_explore_steps)),
        critic_target_tau=float(method.get("critic_target_tau", 0.02)),
        critic_grad_clip=(
            None
            if method.get("critic_grad_clip", None) is None
            else float(method.critic_grad_clip)
        ),
        weight_decay=float(method.get("weight_decay", 0.0)),
        levels=int(method.get("levels", 3)),
        bins=int(method.get("bins", 5)),
        atoms=int(method.get("atoms", 51)),
        v_min=float(method.get("v_min", 0.0)),
        v_max=float(method.get("v_max", 200.0)),
        critic_lambda=float(method.get("critic_lambda", 1.0)),
        centralized_critic=bool(method.get("centralized_critic", False)),
        use_dueling=bool(method.get("use_dueling", True)),
        always_bootstrap=bool(method.get("always_bootstrap", False)),
        stddev_schedule=str(method.get("stddev_schedule", "0.1")),
        bc_lambda=float(method.get("bc_lambda", 0.0)),
        bc_margin=float(method.get("bc_margin", 0.0)),
        use_target_network_for_rollout=bool(
            method.get("use_target_network_for_rollout", False)
        ),
        num_update_steps=int(method.get("num_update_steps", 1)),
        model=rl_model_spec_from_cfg(cfg),
    )


class CQNStream(nn.Module):
    hidden_dims: tuple[int, ...]
    output_shape: tuple[int, ...]
    activation_name: str = "silu"
    norm: str = "layer"
    linear_bias: bool = False

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        x = inputs
        for index, width in enumerate(self.hidden_dims):
            x = nn.Dense(
                width,
                use_bias=self.linear_bias,
                kernel_init=nn.initializers.orthogonal(),
                name=f"dense_{index}",
            )(x)
            if self.norm == "layer":
                x = nn.LayerNorm(epsilon=1e-5, name=f"norm_{index}")(x)
            elif self.norm not in {"none", "identity"}:
                raise ValueError(f"Unsupported CQN hidden norm '{self.norm}'.")
            x = activation(x, self.activation_name)
        output = nn.Dense(
            int(np.prod(self.output_shape)),
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="out",
        )(x)
        return output.reshape((-1, *self.output_shape))


class C2FDistributionalCritic(nn.Module):
    hidden_dims: tuple[int, ...]
    action_dim: int
    bins: int
    atoms: int
    activation_name: str = "silu"
    norm: str = "layer"
    linear_bias: bool = False
    use_dueling: bool = True

    @nn.compact
    def __call__(
        self,
        features: jax.Array,
        level_one_hot: jax.Array,
        low_high_midpoint: jax.Array,
    ) -> jax.Array:
        x = jnp.concatenate([features, level_one_hot, low_high_midpoint], axis=-1)
        advantages = CQNStream(
            hidden_dims=self.hidden_dims,
            output_shape=(self.action_dim, self.bins, self.atoms),
            activation_name=self.activation_name,
            norm=self.norm,
            linear_bias=self.linear_bias,
            name="advantage",
        )(x)
        if not self.use_dueling:
            return advantages
        values = CQNStream(
            hidden_dims=self.hidden_dims,
            output_shape=(self.action_dim, 1, self.atoms),
            activation_name=self.activation_name,
            norm=self.norm,
            linear_bias=self.linear_bias,
            name="value",
        )(x)
        return values + advantages - advantages.mean(axis=-2, keepdims=True)


def zoom_in(
    low: jax.Array,
    high: jax.Array,
    indices: jax.Array,
    bins: int,
    initial_low: jax.Array,
    initial_high: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    width = (high - low) / float(bins)
    new_low = low + width * indices.astype(low.dtype)
    new_high = new_low + width
    return jnp.maximum(initial_low, new_low), jnp.minimum(initial_high, new_high)


def encode_action(
    action: jax.Array,
    initial_low: jax.Array,
    initial_high: jax.Array,
    levels: int,
    bins: int,
) -> jax.Array:
    low = jnp.broadcast_to(initial_low, action.shape)
    high = jnp.broadcast_to(initial_high, action.shape)
    indices = []
    for _ in range(levels):
        width = jnp.maximum((high - low) / float(bins), 1e-8)
        index = jnp.floor((action - low) / width).astype(jnp.int32)
        index = jnp.clip(index, 0, bins - 1)
        indices.append(index)
        low, high = zoom_in(
            low,
            high,
            index,
            bins,
            initial_low,
            initial_high,
        )
    return jnp.stack(indices, axis=-2)


def decode_action(
    discrete_action: jax.Array,
    initial_low: jax.Array,
    initial_high: jax.Array,
    levels: int,
    bins: int,
) -> jax.Array:
    batch_shape = discrete_action.shape[:-2] + (initial_low.shape[-1],)
    low = jnp.broadcast_to(initial_low, batch_shape)
    high = jnp.broadcast_to(initial_high, batch_shape)
    for level in range(levels):
        low, high = zoom_in(
            low,
            high,
            discrete_action[..., level, :],
            bins,
            initial_low,
            initial_high,
        )
    return 0.5 * (low + high)


def project_categorical(
    probabilities: jax.Array,
    rewards: jax.Array,
    discounts: jax.Array,
    bootstrap: jax.Array,
    support: jax.Array,
) -> jax.Array:
    """C51 L2 projection for probabilities shaped ``[B, L, D, atoms]``."""

    atoms = support.shape[0]
    v_min = support[0]
    v_max = support[-1]
    delta = (v_max - v_min) / float(atoms - 1)
    target = rewards[:, None] + (
        bootstrap * discounts
    )[:, None] * support[None, :]
    target = jnp.clip(target, v_min, v_max)
    projected_index = (target - v_min) / delta
    lower = jnp.floor(projected_index).astype(jnp.int32)
    upper = jnp.ceil(projected_index).astype(jnp.int32)
    lower_weight = jnp.where(
        lower == upper,
        1.0,
        upper.astype(jnp.float32) - projected_index,
    )
    upper_weight = jnp.where(
        lower == upper,
        0.0,
        projected_index - lower.astype(jnp.float32),
    )
    batch, levels, action_dim, _ = probabilities.shape
    lower = jnp.broadcast_to(lower[:, None, None, :], probabilities.shape)
    upper = jnp.broadcast_to(upper[:, None, None, :], probabilities.shape)
    lower_weight = jnp.broadcast_to(
        lower_weight[:, None, None, :], probabilities.shape
    )
    upper_weight = jnp.broadcast_to(
        upper_weight[:, None, None, :], probabilities.shape
    )
    flat_probs = probabilities.reshape((-1, atoms))
    flat_lower = lower.reshape((-1, atoms))
    flat_upper = upper.reshape((-1, atoms))
    flat_lower_weight = lower_weight.reshape((-1, atoms))
    flat_upper_weight = upper_weight.reshape((-1, atoms))

    def project_one(prob, low_index, high_index, low_weight, high_weight):
        result = jnp.zeros((atoms,), dtype=prob.dtype)
        result = result.at[low_index].add(prob * low_weight)
        return result.at[high_index].add(prob * high_weight)

    projected = jax.vmap(project_one)(
        flat_probs,
        flat_lower,
        flat_upper,
        flat_lower_weight,
        flat_upper_weight,
    )
    return projected.reshape((batch, levels, action_dim, atoms))


class CQN(JaxRLMethodBase, OffPolicyMethod):
    """RoboBase coarse-to-fine distributional Q-learning for Box actions."""

    def __init__(
        self,
        critic_lr: float,
        num_train_steps: int,
        num_explore_steps: int,
        critic_target_tau: float,
        weight_decay: float,
        levels: int,
        bins: int,
        atoms: int,
        v_min: float,
        v_max: float,
        critic_lambda: float,
        centralized_critic: bool,
        use_dueling: bool,
        always_bootstrap: bool,
        stddev_schedule: str,
        bc_lambda: float,
        bc_margin: float,
        use_target_network_for_rollout: bool,
        num_update_steps: int,
        model: RLModelSpec,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        num_train_envs: int,
        num_eval_envs: int,
        replay_alpha: float,
        replay_beta: float,
        frame_stack_on_channel: bool,
        intrinsic_reward_module=None,
        critic_grad_clip: Optional[float] = None,
        jit: bool = True,
        platform: str | None = None,
        seed: int = 0,
        update_block_every_steps: int = 1,
    ):
        super().__init__(
            lr=critic_lr,
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
            actor_grad_clip=critic_grad_clip,
            jit=jit,
            platform=platform,
            seed=seed,
            is_rl=True,
            update_block_every_steps=update_block_every_steps,
        )
        if self.action_sequence != 1:
            raise ValueError("CQN requires action_sequence=1.")
        if levels < 1 or bins < 2:
            raise ValueError("CQN requires levels >= 1 and bins >= 2.")
        if atoms < 2 or v_max <= v_min:
            raise ValueError("CQN requires atoms >= 2 and v_max > v_min.")
        if not 0.0 < critic_target_tau <= 1.0:
            raise ValueError("critic_target_tau must be in (0, 1].")

        self.levels = int(levels)
        self.bins = int(bins)
        self.atoms = int(atoms)
        self.critic_target_tau = float(critic_target_tau)
        self.critic_lambda = float(critic_lambda)
        self.centralized_critic = bool(centralized_critic)
        self.always_bootstrap = bool(always_bootstrap)
        self.stddev_schedule = str(stddev_schedule)
        self.bc_lambda = float(bc_lambda)
        self.bc_margin = float(bc_margin)
        self.use_target_network_for_rollout = bool(use_target_network_for_rollout)
        self.num_update_steps = int(num_update_steps)
        self.critic_grad_clip = critic_grad_clip
        self.num_explore_steps = int(num_explore_steps)
        input_dim = self._setup_rl_features(model, seed=seed)
        self.action_low, self.action_high = self._action_bounds()
        self.support = jnp.linspace(v_min, v_max, atoms, dtype=jnp.float32)
        self.critic_model = C2FDistributionalCritic(
            hidden_dims=model.hidden_dims,
            action_dim=self.action_dim,
            bins=self.bins,
            atoms=self.atoms,
            activation_name=model.activation,
            norm=model.norm,
            linear_bias=model.linear_bias,
            use_dueling=bool(use_dueling),
        )
        dummy_features = jnp.zeros((1, input_dim), dtype=jnp.float32)
        dummy_level = jnp.zeros((1, self.levels), dtype=jnp.float32)
        dummy_midpoint = jnp.zeros((1, self.action_dim), dtype=jnp.float32)
        critic_params = self.critic_model.init(
            self.rng_key,
            dummy_features,
            dummy_level,
            dummy_midpoint,
        )
        self.params = {"critic": critic_params}
        if self._trainable_encoder:
            self.params["encoder"] = self._encoder_params
        self.target_critic_params = critic_params
        transforms = []
        if critic_grad_clip is not None:
            transforms.append(self.optax.clip_by_global_norm(float(critic_grad_clip)))
        transforms.append(
            self.optax.adamw(float(critic_lr), weight_decay=float(weight_decay))
        )
        self.optimizer = self.optax.chain(*transforms)
        self.opt_state = self.optimizer.init(self.params)
        update_fn = self._build_update_fn()
        action_fn = self._build_greedy_action_fn()
        if self._jit_enabled:
            update_fn = jax.jit(update_fn)
            action_fn = jax.jit(action_fn)
        self._update_impl = update_fn
        self._greedy_action_impl = action_fn

    def _critic_logits_per_level(self, critic_params, features, action):
        discrete_action = encode_action(
            action,
            self.action_low,
            self.action_high,
            self.levels,
            self.bins,
        )
        low = jnp.broadcast_to(self.action_low, action.shape)
        high = jnp.broadcast_to(self.action_high, action.shape)
        logits_per_level = []
        chosen_logits_per_level = []
        for level in range(self.levels):
            one_hot = jnp.broadcast_to(
                jax.nn.one_hot(level, self.levels, dtype=jnp.float32),
                (features.shape[0], self.levels),
            )
            logits = self.critic_model.apply(
                critic_params,
                features,
                one_hot,
                0.5 * (low + high),
            )
            index = discrete_action[:, level, :]
            selected = jnp.take_along_axis(
                logits,
                index[:, :, None, None],
                axis=-2,
            )[..., 0, :]
            logits_per_level.append(logits)
            chosen_logits_per_level.append(selected)
            low, high = zoom_in(
                low,
                high,
                index,
                self.bins,
                self.action_low,
                self.action_high,
            )
        return (
            jnp.stack(chosen_logits_per_level, axis=1),
            jnp.stack(logits_per_level, axis=1),
        )

    def _greedy_action(self, critic_params, features):
        batch_size = features.shape[0]
        low = jnp.broadcast_to(
            self.action_low,
            (batch_size, self.action_dim),
        )
        high = jnp.broadcast_to(
            self.action_high,
            (batch_size, self.action_dim),
        )
        selected = []
        for level in range(self.levels):
            one_hot = jnp.broadcast_to(
                jax.nn.one_hot(level, self.levels, dtype=jnp.float32),
                (batch_size, self.levels),
            )
            logits = self.critic_model.apply(
                critic_params,
                features,
                one_hot,
                0.5 * (low + high),
            )
            probabilities = jax.nn.softmax(logits, axis=-1)
            q_values = jnp.sum(probabilities * self.support, axis=-1)
            index = jnp.argmax(q_values, axis=-1)
            selected.append(index)
            low, high = zoom_in(
                low,
                high,
                index,
                self.bins,
                self.action_low,
                self.action_high,
            )
        return 0.5 * (low + high), jnp.stack(selected, axis=1)

    def _build_greedy_action_fn(self):
        def action_fn(params, target_critic_params, obs_inputs, use_target):
            features = self._rl_features(
                params.get("encoder", None),
                obs_inputs,
                stop_gradient=True,
            )
            critic_params = jax.lax.cond(
                use_target,
                lambda _: target_critic_params,
                lambda _: params["critic"],
                operand=None,
            )
            return self._greedy_action(critic_params, features)[0]

        return action_fn

    def _greedy_action_for_update(self, critic_params, features, action_key):
        del action_key
        return self._greedy_action(critic_params, features)

    def _next_action_key(self):
        # Plain CQN has deterministic argmax selection. Keep its RNG stream
        # unchanged while exposing a hook for CQN-AS random tie breaking.
        return jax.random.PRNGKey(0)

    def _augment_update_obs_inputs(self, obs_inputs, next_obs_inputs, key):
        return obs_inputs, next_obs_inputs, key

    def _build_update_fn(self):
        optimizer = self.optimizer
        tau = self.critic_target_tau

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
            demos,
            action_key,
        ):
            obs_inputs, next_obs_inputs, action_key = (
                self._augment_update_obs_inputs(
                    obs_inputs,
                    next_obs_inputs,
                    action_key,
                )
            )
            def loss_fn(current_params):
                encoder_params = current_params.get("encoder", None)
                features = self._rl_features(encoder_params, obs_inputs)
                next_features = self._rl_features(
                    encoder_params,
                    next_obs_inputs,
                    stop_gradient=True,
                )
                next_action, _ = self._greedy_action_for_update(
                    current_params["critic"],
                    next_features,
                    action_key,
                )
                target_logits, _ = self._critic_logits_per_level(
                    target_critic_params,
                    next_features,
                    next_action,
                )
                target_probabilities = jax.nn.softmax(target_logits, axis=-1)
                target_distribution = project_categorical(
                    target_probabilities,
                    rewards,
                    discounts,
                    bootstrap,
                    self.support,
                )
                if self.centralized_critic:
                    target_distribution = jnp.broadcast_to(
                        target_distribution.mean(axis=-2, keepdims=True),
                        target_distribution.shape,
                    )
                target_distribution = jax.lax.stop_gradient(target_distribution)
                chosen_logits, all_logits = self._critic_logits_per_level(
                    current_params["critic"],
                    features,
                    actions,
                )
                chosen_log_probabilities = jax.nn.log_softmax(
                    chosen_logits,
                    axis=-1,
                )
                chosen_probabilities = jax.nn.softmax(chosen_logits, axis=-1)
                all_probabilities = jax.nn.softmax(all_logits, axis=-1)
                per_sample = -jnp.sum(
                    target_distribution * chosen_log_probabilities,
                    axis=-1,
                ).mean(axis=(1, 2))
                critic_loss = self.critic_lambda * jnp.mean(
                    per_sample * loss_weights
                )

                if self.bc_lambda > 0.0:
                    chosen_cdf = jnp.cumsum(chosen_probabilities, axis=-1)
                    all_cdf = jnp.cumsum(all_probabilities, axis=-1)
                    fosd = jnp.maximum(
                        chosen_cdf[..., None, :] - all_cdf,
                        0.0,
                    ).sum(axis=-1).mean(axis=(1, 2, 3))
                    demo_count = jnp.maximum(jnp.sum(demos), 1.0)
                    critic_loss = critic_loss + self.bc_lambda * (
                        jnp.sum(fosd * demos) / demo_count
                    )
                    if self.bc_margin > 0.0:
                        all_q = jnp.sum(
                            all_probabilities * self.support,
                            axis=-1,
                        )
                        chosen_q = jnp.sum(
                            chosen_probabilities * self.support,
                            axis=-1,
                        )
                        margin = jnp.maximum(
                            self.bc_margin
                            - (chosen_q[..., None] - all_q),
                            0.0,
                        ).mean(axis=(1, 2, 3))
                        critic_loss = critic_loss + self.bc_lambda * (
                            jnp.sum(margin * demos) / demo_count
                        )
                entropy = -jnp.sum(
                    chosen_probabilities
                    * jnp.log(jnp.maximum(chosen_probabilities, 1e-9)),
                    axis=-1,
                ).mean()
                target_entropy = -jnp.sum(
                    target_distribution
                    * jnp.log(jnp.maximum(target_distribution, 1e-9)),
                    axis=-1,
                ).mean()
                return critic_loss, (per_sample, entropy, target_entropy)

            (critic_loss, aux), grads = jax.value_and_grad(
                loss_fn,
                has_aux=True,
            )(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = self.optax.apply_updates(params, updates)
            target_critic_params = jax.tree.map(
                lambda target, online: (1.0 - tau) * target + tau * online,
                target_critic_params,
                params["critic"],
            )
            per_sample, entropy, projected_entropy = aux
            priority = jnp.sqrt(per_sample + 1e-10)
            priority = priority / jnp.maximum(jnp.max(priority), 1e-10)
            return (
                params,
                target_critic_params,
                opt_state,
                priority,
                {
                    "critic_loss": critic_loss,
                    "entropy": entropy,
                    "target_entropy": projected_entropy,
                    "loss_coeff": jnp.mean(loss_weights),
                },
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
        action = self._greedy_action_impl(
            self.params,
            self.target_critic_params,
            obs_inputs,
            jnp.asarray(self.use_target_network_for_rollout),
        )
        if not eval_mode:
            self.rng_key, noise_key = jax.random.split(self.rng_key)
            stddev = float(utils.schedule(self.stddev_schedule, step))
            action = action + stddev * jax.random.normal(noise_key, action.shape)
            action = jnp.clip(action, self.action_low, self.action_high)
            discrete = encode_action(
                action,
                self.action_low,
                self.action_high,
                self.levels,
                self.bins,
            )
            action = decode_action(
                discrete,
                self.action_low,
                self.action_high,
                self.levels,
                self.bins,
            )
        self._block(action)
        action = np.asarray(jax.device_get(action), dtype=np.float32)
        return action.reshape((action.shape[0], 1, self.action_dim))

    def update(
        self,
        replay_iter: Iterator[dict],
        step: int,
        replay_buffer: ReplayBuffer = None,
    ) -> dict[str, np.ndarray]:
        update_steps = 1 if step == 0 else self.num_update_steps
        metrics = {}
        for _ in range(update_steps):
            batch = next(replay_iter)
            obs_inputs = self._prepare_rl_obs_inputs(batch)
            next_obs_inputs = self._next_rl_obs_inputs(batch)
            actions = self._as_jax_array(batch["action"], self.jnp.float32).reshape(
                (batch["action"].shape[0], -1)
            )
            rewards = self._as_jax_array(
                batch["reward"], self.jnp.float32
            ).reshape(-1)
            discounts = self._as_jax_array(
                batch.get("discount", np.ones_like(batch["reward"])),
                self.jnp.float32,
            ).reshape(-1)
            terminal = self._as_jax_array(
                batch["terminal"], self.jnp.float32
            ).reshape(-1)
            bootstrap = (
                jnp.ones_like(terminal) if self.always_bootstrap else 1.0 - terminal
            )
            loss_weights = self._loss_weights(batch)
            demos = self._as_jax_array(
                batch.get("demo", np.zeros_like(batch["reward"])),
                self.jnp.float32,
            ).reshape(-1)
            start_time = time.perf_counter()
            (
                self.params,
                self.target_critic_params,
                self.opt_state,
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
                demos,
                self._next_action_key(),
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
            if self.logging:
                metrics.update(
                    {
                        key: float(np.asarray(jax.device_get(value)))
                        for key, value in jax_metrics.items()
                    }
                )
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
    "CQN",
    "CQNSpec",
    "cqn_spec_from_cfg",
    "decode_action",
    "encode_action",
    "project_categorical",
    "zoom_in",
]
