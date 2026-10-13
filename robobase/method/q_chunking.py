"""Q-chunking with a flow behavior policy, implemented entirely in JAX."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig

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


_OFFICIAL_DENSE_INIT = nn.initializers.variance_scaling(
    1.0,
    "fan_avg",
    "uniform",
)


@dataclass(frozen=True)
class QChunkingSpec:
    actor_lr: float
    critic_lr: float
    num_train_steps: int
    num_explore_steps: int
    critic_target_tau: float
    flow_steps: int
    actor_num_samples: int
    q_aggregate: str
    actor_grad_clip: float | None
    critic_grad_clip: float | None
    weight_decay: float
    model: RLModelSpec
    feature_projection: bool = False
    feature_dim: int = 64
    rgb_encoder_layers: int = 2
    rgb_shift_pad: int = 0
    bc_demo_only: bool = False
    actor_use_low_dim: bool = True
    low_dim_mask_prob: float = 0.0
    low_dim_mask_keep_last: int = 0


def q_chunking_spec_from_cfg(cfg: DictConfig) -> QChunkingSpec:
    method = cfg.method
    q_aggregate = str(method.get("q_aggregate", "mean")).lower()
    if q_aggregate not in {"mean", "min"}:
        raise ValueError("method.q_aggregate must be 'mean' or 'min'.")
    return QChunkingSpec(
        actor_lr=float(method.get("actor_lr", 3e-4)),
        critic_lr=float(method.get("critic_lr", 3e-4)),
        num_train_steps=int(method.get("num_train_steps", cfg.num_train_frames)),
        num_explore_steps=int(method.get("num_explore_steps", cfg.num_explore_steps)),
        critic_target_tau=float(method.get("critic_target_tau", 0.005)),
        flow_steps=int(method.get("flow_steps", 10)),
        actor_num_samples=int(method.get("actor_num_samples", 32)),
        q_aggregate=q_aggregate,
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
        feature_projection=bool(method.get("feature_projection", False)),
        feature_dim=int(method.get("feature_dim", 64)),
        rgb_encoder_layers=int(method.get("rgb_encoder_layers", 2)),
        rgb_shift_pad=int(method.get("rgb_shift_pad", 0)),
        bc_demo_only=bool(method.get("bc_demo_only", False)),
        actor_use_low_dim=bool(method.get("actor_use_low_dim", True)),
        low_dim_mask_prob=float(method.get("low_dim_mask_prob", 0.0)),
        low_dim_mask_keep_last=int(method.get("low_dim_mask_keep_last", 0)),
    )


def validate_q_chunking_config(
    *,
    action_sequence: int,
    execution_length: int,
    replay_nstep: int,
    temporal_ensemble: bool,
    action_execution_start: int = 0,
) -> None:
    """Validate the replay and rollout contract required by Q-chunking."""

    if action_sequence < 2:
        raise ValueError("Q-chunking requires action_sequence >= 2.")
    if execution_length != 1:
        raise ValueError(
            "Q-chunking stores primitive executed actions and requires "
            "execution_length=1."
        )
    if replay_nstep != action_sequence:
        raise ValueError(
            "Q-chunking requires replay.nstep == action_sequence so the replay "
            "returns the matching discounted K-step backup."
        )
    if temporal_ensemble:
        raise ValueError(
            "Q-chunking performs open-loop chunk execution inside the agent; "
            "set root temporal_ensemble=false."
        )
    if action_execution_start != 0:
        raise ValueError("Q-chunking requires action_execution_start=0.")


def q_chunking_td_target(
    rewards: jax.Array,
    discounts: jax.Array,
    bootstrap: jax.Array,
    next_q: jax.Array,
) -> jax.Array:
    """Return the unbiased replay-provided K-step TD target."""

    return rewards + discounts * bootstrap * next_q


class QChunkFlowActor(nn.Module):
    hidden_dims: tuple[int, ...]
    action_dim: int
    activation_name: str = "relu"
    use_layer_norm: bool = False

    @nn.compact
    def __call__(
        self,
        features: jax.Array,
        actions: jax.Array,
        times: jax.Array,
    ) -> jax.Array:
        times = jnp.asarray(times, dtype=jnp.float32).reshape((-1, 1))
        x = jnp.concatenate(
            [
                features.astype(jnp.float32),
                actions.astype(jnp.float32),
                times,
            ],
            axis=-1,
        )
        for index, width in enumerate(self.hidden_dims):
            x = nn.Dense(
                width,
                kernel_init=_OFFICIAL_DENSE_INIT,
                name=f"dense_{index}",
            )(x)
            x = activation(x, self.activation_name)
            if self.use_layer_norm:
                x = nn.LayerNorm(name=f"norm_{index}")(x)
        return nn.Dense(
            self.action_dim,
            kernel_init=_OFFICIAL_DENSE_INIT,
            name="velocity",
        )(x)


class QChunkCritic(nn.Module):
    hidden_dims: tuple[int, ...]
    num_critics: int = 2
    activation_name: str = "relu"
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, features: jax.Array, actions: jax.Array) -> jax.Array:
        inputs = jnp.concatenate(
            [features.astype(jnp.float32), actions.astype(jnp.float32)],
            axis=-1,
        )
        values = []
        for critic_index in range(self.num_critics):
            x = inputs
            for layer_index, width in enumerate(self.hidden_dims):
                x = nn.Dense(
                    width,
                    kernel_init=_OFFICIAL_DENSE_INIT,
                    name=f"q{critic_index + 1}_dense_{layer_index}",
                )(x)
                x = activation(x, self.activation_name)
                if self.use_layer_norm:
                    x = nn.LayerNorm(name=f"q{critic_index + 1}_norm_{layer_index}")(x)
            values.append(
                nn.Dense(
                    1,
                    kernel_init=_OFFICIAL_DENSE_INIT,
                    name=f"q{critic_index + 1}_out",
                )(x)[..., 0]
            )
        return jnp.stack(values, axis=-1)


class QChunkFeatureTrunk(nn.Module):
    """Per-modality projection matching the CQN-AS pixel trunk.

    Raw flattened CNN features (~19k dims, |x|~1e-2) and standardized low-dim
    state (~240 dims, |x|~1, outliers >40) are projected separately to
    ``feature_dim`` with LayerNorm+tanh, so neither modality dominates the MLP.
    Input layout follows ``JaxRLMethodBase._rl_features``: [low_dim, rgb, time].
    """

    low_dim_size: int
    rgb_size: int
    time_size: int
    hidden_dim: int
    feature_dim: int = 64
    rgb_encoder_layers: int = 2
    activation_name: str = "gelu"

    @property
    def output_dim(self) -> int:
        return (
            (self.feature_dim if self.low_dim_size else 0)
            + (self.feature_dim if self.rgb_size else 0)
            + self.time_size
        )

    @nn.compact
    def __call__(self, features: jax.Array) -> jax.Array:
        features = features.astype(jnp.float32)
        low_dim = features[:, : self.low_dim_size]
        rgb = features[:, self.low_dim_size : self.low_dim_size + self.rgb_size]
        time = features[:, self.low_dim_size + self.rgb_size :]
        parts = []
        if self.rgb_size:
            for index in range(self.rgb_encoder_layers):
                rgb = nn.Dense(
                    self.hidden_dim,
                    use_bias=False,
                    kernel_init=nn.initializers.orthogonal(),
                    name=f"rgb_dense_{index}",
                )(rgb)
                rgb = nn.LayerNorm(name=f"rgb_norm_{index}")(rgb)
                rgb = activation(rgb, self.activation_name)
            rgb = nn.Dense(
                self.feature_dim,
                use_bias=False,
                kernel_init=nn.initializers.orthogonal(),
                name="rgb_projection",
            )(rgb)
            parts.append(jnp.tanh(nn.LayerNorm(name="rgb_projection_norm")(rgb)))
        if self.low_dim_size:
            low_dim = nn.Dense(
                self.feature_dim,
                use_bias=False,
                kernel_init=nn.initializers.orthogonal(),
                name="low_dim_projection",
            )(low_dim)
            parts.append(jnp.tanh(nn.LayerNorm(name="low_dim_norm")(low_dim)))
        if self.time_size:
            parts.append(time)
        return jnp.concatenate(parts, axis=-1)


class QChunking(JaxRLMethodBase, OffPolicyMethod):
    """Official-style Best-of-N Q-chunking adapted to RoboBase replay."""

    def _init_cached_pixel_feature_key(self, method_name: str) -> None:
        del method_name
        super()._init_cached_pixel_feature_key("q_chunking")

    def __init__(
        self,
        actor_lr: float,
        critic_lr: float,
        num_train_steps: int,
        num_explore_steps: int,
        critic_target_tau: float,
        flow_steps: int,
        actor_num_samples: int,
        q_aggregate: str,
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
        feature_projection: bool = False,
        feature_dim: int = 64,
        rgb_encoder_layers: int = 2,
        rgb_shift_pad: int = 0,
        bc_demo_only: bool = False,
        actor_use_low_dim: bool = True,
        low_dim_mask_prob: float = 0.0,
        low_dim_mask_keep_last: int = 0,
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
        if self.action_sequence < 2:
            raise ValueError("Q-chunking requires action_sequence >= 2.")
        if flow_steps < 1:
            raise ValueError("flow_steps must be positive.")
        if actor_num_samples < 1:
            raise ValueError("actor_num_samples must be positive.")
        if not 0.0 < critic_target_tau <= 1.0:
            raise ValueError("critic_target_tau must be in (0, 1].")
        if q_aggregate not in {"mean", "min"}:
            raise ValueError("q_aggregate must be 'mean' or 'min'.")

        self.num_explore_steps = int(num_explore_steps)
        self.critic_target_tau = float(critic_target_tau)
        self.flow_steps = int(flow_steps)
        self.actor_num_samples = int(actor_num_samples)
        self.q_aggregate = q_aggregate
        self.critic_grad_clip = critic_grad_clip
        input_dim = self._setup_rl_features(model, seed=seed)
        self.action_low, self.action_high = self._action_bounds()
        if rgb_shift_pad < 0:
            raise ValueError("rgb_shift_pad must be non-negative.")
        self.feature_projection = bool(feature_projection)
        self.rgb_shift_pad = int(rgb_shift_pad)
        self.bc_demo_only = bool(bc_demo_only)
        self.actor_use_low_dim = bool(actor_use_low_dim)
        if not 0.0 <= low_dim_mask_prob <= 1.0:
            raise ValueError("low_dim_mask_prob must be in [0, 1].")
        self.low_dim_mask_prob = float(low_dim_mask_prob)
        self.low_dim_mask_keep_last = int(low_dim_mask_keep_last)
        if self.low_dim_mask_prob > 0.0:
            if not self.low_dim_size or "low_dim_state" not in self.observation_space.spaces:
                raise ValueError("low_dim_mask_prob requires a low_dim_state observation.")
            self._low_dim_frame_dim = int(self.observation_space.spaces["low_dim_state"].shape[-1])
            if not 0 <= self.low_dim_mask_keep_last < self._low_dim_frame_dim:
                raise ValueError("low_dim_mask_keep_last must be in [0, low-dim frame width).")
        raw_input_dim = int(input_dim)
        self.trunk_model = None
        if self.feature_projection:
            if self.use_pixels and not self._trainable_encoder:
                raise ValueError(
                    "Q-chunking feature_projection needs the trainable-encoder "
                    "feature layout [low_dim, rgb, time]."
                )
            # Flattened (frame-stacked) low-dim width, as produced by _extract_low_dim_batch.
            low_width = int(self.low_dim_size)
            rgb_width = raw_input_dim - low_width - self._time_feature_dim
            if rgb_width < 0 or (not self.use_pixels and rgb_width != 0):
                raise ValueError(
                    f"Unexpected RL feature layout: total={raw_input_dim}, "
                    f"low_dim={low_width}, time={self._time_feature_dim}."
                )
            self.trunk_model = QChunkFeatureTrunk(
                low_dim_size=low_width,
                rgb_size=rgb_width,
                time_size=self._time_feature_dim,
                hidden_dim=int(model.hidden_dims[0]),
                feature_dim=int(feature_dim),
                rgb_encoder_layers=int(rgb_encoder_layers),
                activation_name=model.activation,
            )
            input_dim = self.trunk_model.output_dim
        flat_action_dim = self.action_sequence * self.action_dim

        self.actor_model = QChunkFlowActor(
            hidden_dims=model.hidden_dims,
            action_dim=flat_action_dim,
            activation_name=model.activation,
            use_layer_norm=False,
        )
        self.critic_model = QChunkCritic(
            hidden_dims=model.hidden_dims,
            num_critics=2,
            activation_name=model.activation,
            use_layer_norm=True,
        )
        dummy_features = jnp.zeros((1, input_dim), dtype=jnp.float32)
        dummy_actions = jnp.zeros((1, flat_action_dim), dtype=jnp.float32)
        dummy_times = jnp.zeros((1, 1), dtype=jnp.float32)
        self.rng_key, actor_key, critic_key = jax.random.split(self.rng_key, 3)
        actor_params = self.actor_model.init(
            actor_key,
            dummy_features,
            dummy_actions,
            dummy_times,
        )
        critic_params = self.critic_model.init(
            critic_key,
            dummy_features,
            dummy_actions,
        )
        self.params: dict[str, Any] = {
            "actor": actor_params,
            "critic": critic_params,
        }
        if self.feature_projection:
            self.rng_key, actor_trunk_key, critic_trunk_key = jax.random.split(
                self.rng_key, 3
            )
            dummy_raw = jnp.zeros((1, raw_input_dim), dtype=jnp.float32)
            self.params["actor_trunk"] = self.trunk_model.init(actor_trunk_key, dummy_raw)
            self.params["critic_trunk"] = self.trunk_model.init(
                critic_trunk_key, dummy_raw
            )
        if self._trainable_encoder:
            initial_encoder = jax.tree.map(jnp.array, self._encoder_params)
            self.params["actor_encoder"] = initial_encoder
            self.params["critic_encoder"] = self._independent_rl_encoder_params(
                seed=int(seed) + 1
            )

        self.target_critic_params = self._critic_bundle(self.params)
        actor_transforms = []
        if actor_grad_clip is not None:
            actor_transforms.append(
                self.optax.clip_by_global_norm(float(actor_grad_clip))
            )
        actor_transforms.append(
            self.optax.adamw(
                float(actor_lr),
                weight_decay=float(weight_decay),
            )
        )
        critic_transforms = []
        if critic_grad_clip is not None:
            critic_transforms.append(
                self.optax.clip_by_global_norm(float(critic_grad_clip))
            )
        critic_transforms.append(
            self.optax.adamw(
                float(critic_lr),
                weight_decay=float(weight_decay),
            )
        )
        self.actor_optimizer = self.optax.chain(*actor_transforms)
        self.critic_optimizer = self.optax.chain(*critic_transforms)
        self.opt_state = {
            "actor": self.actor_optimizer.init(self._actor_bundle(self.params)),
            "critic": self.critic_optimizer.init(self._critic_bundle(self.params)),
        }

        update_fn = self._build_update_fn()
        policy_fn = self._build_policy_fn()
        if self._jit_enabled:
            update_fn = jax.jit(update_fn)
            policy_fn = jax.jit(policy_fn)
        self._update_impl = update_fn
        self._policy_action_impl = policy_fn

        self._train_open_loop_plan = None
        self._train_open_loop_position = None
        self._train_open_loop_valid = None
        self._eval_open_loop_plan = None
        self._eval_open_loop_position = None
        self._eval_open_loop_valid = None
        self._last_selected_q = np.nan

    def _actor_bundle(self, params):
        bundle = {"actor": params["actor"]}
        if self._trainable_encoder:
            bundle["actor_encoder"] = params["actor_encoder"]
        if self.feature_projection:
            bundle["actor_trunk"] = params["actor_trunk"]
        return bundle

    def _critic_bundle(self, params):
        bundle = {"critic": params["critic"]}
        if self._trainable_encoder:
            bundle["critic_encoder"] = params["critic_encoder"]
        if self.feature_projection:
            bundle["critic_trunk"] = params["critic_trunk"]
        return bundle

    def _mask_low_dim(self, low_dim, key):
        """Per-sample update-time low-dim dropout (same semantics as CQN-AS Stage-160)."""
        batch = low_dim.shape[0]
        frame = self._low_dim_frame_dim
        frames = low_dim.reshape((batch, -1, frame))
        keep = self.low_dim_mask_keep_last
        keep_mask = jnp.concatenate(
            [jnp.zeros((frame - keep,), frames.dtype), jnp.ones((keep,), frames.dtype)]
        )
        drop = jax.random.bernoulli(key, self.low_dim_mask_prob, (batch, 1, 1))
        return jnp.where(drop, frames * keep_mask, frames).reshape(low_dim.shape)

    def _project(self, trunk_params, features, stop_gradient):
        if not self.feature_projection:
            return features
        features = self.trunk_model.apply(trunk_params, features)
        return jax.lax.stop_gradient(features) if stop_gradient else features

    def _actor_features(self, bundle, obs_inputs, *, stop_gradient=False):
        if not self.actor_use_low_dim:
            if not isinstance(obs_inputs, dict):
                raise ValueError("actor_use_low_dim=False needs the trainable-encoder input dict.")
            if "low_dim" in obs_inputs:
                # Same zeros at train and inference: the actor cannot shortcut on proprio.
                obs_inputs = dict(obs_inputs)
                obs_inputs["low_dim"] = jnp.zeros_like(obs_inputs["low_dim"])
        features = self._rl_features(
            bundle.get("actor_encoder", None),
            obs_inputs,
            stop_gradient=stop_gradient,
        )
        return self._project(bundle.get("actor_trunk"), features, stop_gradient)

    def _critic_features(self, bundle, obs_inputs, *, stop_gradient=False):
        features = self._rl_features(
            bundle.get("critic_encoder", None),
            obs_inputs,
            stop_gradient=stop_gradient,
        )
        return self._project(bundle.get("critic_trunk"), features, stop_gradient)

    def _aggregate_q(self, q_values):
        if self.q_aggregate == "min":
            return jnp.min(q_values, axis=-1)
        return jnp.mean(q_values, axis=-1)

    def _flow_candidates(
        self,
        actor_params,
        features,
        key,
        *,
        count: int,
    ):
        batch_size = features.shape[0]
        flat_action_dim = self.action_sequence * self.action_dim
        actions = jax.random.normal(
            key,
            (batch_size * count, flat_action_dim),
            dtype=jnp.float32,
        )
        repeated_features = jnp.repeat(features, count, axis=0)
        dt = 1.0 / float(self.flow_steps)
        for flow_step in range(self.flow_steps):
            times = jnp.full(
                (batch_size * count, 1),
                flow_step * dt,
                dtype=jnp.float32,
            )
            velocity = self.actor_model.apply(
                actor_params,
                repeated_features,
                actions,
                times,
            )
            actions = actions + dt * velocity
        return jnp.clip(actions, -1.0, 1.0).reshape(
            (batch_size, count, flat_action_dim)
        )

    def _best_of_n(
        self,
        actor_bundle,
        critic_bundle,
        actor_features,
        critic_features,
        key,
    ):
        candidates = self._flow_candidates(
            actor_bundle["actor"],
            actor_features,
            key,
            count=self.actor_num_samples,
        )
        batch_size, count, flat_action_dim = candidates.shape
        repeated_critic_features = jnp.repeat(critic_features, count, axis=0)
        q_values = self.critic_model.apply(
            critic_bundle["critic"],
            repeated_critic_features,
            candidates.reshape((batch_size * count, flat_action_dim)),
        ).reshape((batch_size, count, 2))
        scores = self._aggregate_q(q_values)
        best_indices = jnp.argmax(scores, axis=1)
        selected = jnp.take_along_axis(
            candidates,
            best_indices[:, None, None],
            axis=1,
        )[:, 0]
        selected_q = jnp.take_along_axis(
            scores,
            best_indices[:, None],
            axis=1,
        )[:, 0]
        return selected, selected_q

    def _build_policy_fn(self):
        def policy_fn(params, obs_inputs, key):
            actor_bundle = self._actor_bundle(params)
            critic_bundle = self._critic_bundle(params)
            actor_features = self._actor_features(
                actor_bundle,
                obs_inputs,
                stop_gradient=True,
            )
            critic_features = self._critic_features(
                critic_bundle,
                obs_inputs,
                stop_gradient=True,
            )
            return self._best_of_n(
                actor_bundle,
                critic_bundle,
                actor_features,
                critic_features,
                key,
            )

        return policy_fn

    def _build_update_fn(self):
        actor_optimizer = self.actor_optimizer
        critic_optimizer = self.critic_optimizer
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
            action_valid,
            loss_weights,
            bc_weights,
            key,
        ):
            if (
                self.rgb_shift_pad > 0
                and isinstance(obs_inputs, dict)
                and "rgb" in obs_inputs
            ):
                key, obs_shift_key, next_shift_key = jax.random.split(key, 3)
                obs_inputs = dict(obs_inputs)
                next_obs_inputs = dict(next_obs_inputs)
                obs_inputs["rgb"] = random_shift_rgb(
                    obs_inputs["rgb"], obs_shift_key, self.rgb_shift_pad
                )
                next_obs_inputs["rgb"] = random_shift_rgb(
                    next_obs_inputs["rgb"], next_shift_key, self.rgb_shift_pad
                )
            if (
                self.low_dim_mask_prob > 0.0
                and isinstance(obs_inputs, dict)
                and "low_dim" in obs_inputs
            ):
                key, mask_key, next_mask_key = jax.random.split(key, 3)
                obs_inputs = dict(obs_inputs)
                next_obs_inputs = dict(next_obs_inputs)
                obs_inputs["low_dim"] = self._mask_low_dim(obs_inputs["low_dim"], mask_key)
                next_obs_inputs["low_dim"] = self._mask_low_dim(
                    next_obs_inputs["low_dim"], next_mask_key
                )
            key, next_action_key, source_key, time_key = jax.random.split(key, 4)
            full_chunk_valid = jnp.all(action_valid, axis=1).astype(jnp.float32)
            critic_weights = loss_weights * full_chunk_valid

            actor_bundle = self._actor_bundle(params)
            online_critic_bundle = self._critic_bundle(params)
            next_actor_features = self._actor_features(
                actor_bundle,
                next_obs_inputs,
                stop_gradient=True,
            )
            next_online_critic_features = self._critic_features(
                online_critic_bundle,
                next_obs_inputs,
                stop_gradient=True,
            )
            next_actions, _ = self._best_of_n(
                actor_bundle,
                online_critic_bundle,
                next_actor_features,
                next_online_critic_features,
                next_action_key,
            )
            next_target_features = self._critic_features(
                target_critic_params,
                next_obs_inputs,
                stop_gradient=True,
            )
            next_q_values = self.critic_model.apply(
                target_critic_params["critic"],
                next_target_features,
                next_actions,
            )
            next_q = self._aggregate_q(next_q_values)
            target_q = jax.lax.stop_gradient(
                q_chunking_td_target(rewards, discounts, bootstrap, next_q)
            )

            def critic_loss_fn(critic_bundle):
                features = self._critic_features(
                    critic_bundle,
                    obs_inputs,
                )
                q_values = self.critic_model.apply(
                    critic_bundle["critic"],
                    features,
                    actions,
                )
                per_sample = jnp.mean(
                    jnp.square(q_values - target_q[:, None]),
                    axis=-1,
                )
                denominator = jnp.maximum(jnp.sum(critic_weights), 1.0)
                loss = jnp.sum(per_sample * critic_weights) / denominator
                return loss, (per_sample, q_values)

            (critic_loss, (per_sample, q_values)), critic_grads = jax.value_and_grad(
                critic_loss_fn, has_aux=True
            )(online_critic_bundle)
            critic_updates, critic_opt_state = critic_optimizer.update(
                critic_grads,
                opt_state["critic"],
                online_critic_bundle,
            )
            updated_critic_bundle = self.optax.apply_updates(
                online_critic_bundle,
                critic_updates,
            )

            unit_actions = actions.reshape(
                (actions.shape[0], self.action_sequence, self.action_dim)
            )

            def actor_loss_fn(current_actor_bundle):
                features = self._actor_features(
                    current_actor_bundle,
                    obs_inputs,
                )
                source = jax.random.normal(
                    source_key,
                    actions.shape,
                    dtype=jnp.float32,
                )
                times = jax.random.uniform(
                    time_key,
                    (actions.shape[0], 1),
                    dtype=jnp.float32,
                )
                interpolated = (1.0 - times) * source + times * actions
                target_velocity = actions - source
                predicted_velocity = self.actor_model.apply(
                    current_actor_bundle["actor"],
                    features,
                    interpolated,
                    times,
                )
                squared_error = jnp.square(
                    predicted_velocity - target_velocity
                ).reshape(unit_actions.shape)
                token_error = jnp.mean(squared_error, axis=-1)
                valid = action_valid.astype(jnp.float32)
                per_sample_actor = jnp.sum(token_error * valid, axis=1) / jnp.maximum(
                    jnp.sum(valid, axis=1),
                    1.0,
                )
                actor_denominator = jnp.maximum(jnp.sum(bc_weights), 1.0)
                loss = jnp.sum(per_sample_actor * bc_weights) / actor_denominator
                return loss, per_sample_actor

            current_actor_bundle = self._actor_bundle(params)
            (actor_loss, per_sample_actor), actor_grads = jax.value_and_grad(
                actor_loss_fn,
                has_aux=True,
            )(current_actor_bundle)
            actor_updates, actor_opt_state = actor_optimizer.update(
                actor_grads,
                opt_state["actor"],
                current_actor_bundle,
            )
            updated_actor_bundle = self.optax.apply_updates(
                current_actor_bundle,
                actor_updates,
            )

            params = dict(params)
            params.update(updated_actor_bundle)
            params.update(updated_critic_bundle)
            target_critic_params = jax.tree.map(
                lambda target, online: (1.0 - tau) * target + tau * online,
                target_critic_params,
                updated_critic_bundle,
            )
            priority = jnp.sqrt(per_sample + 1e-10)
            priority = priority / jnp.maximum(jnp.max(priority), 1e-10)
            metrics = {
                "actor_loss": actor_loss,
                "bc_flow_loss": actor_loss,
                "critic_loss": critic_loss,
                "critic_q": jnp.mean(q_values),
                "critic_target_q": jnp.mean(target_q),
                "chunk_valid_fraction": jnp.mean(full_chunk_valid),
                "action_valid_fraction": jnp.mean(action_valid),
                "actor_per_sample_loss": jnp.mean(per_sample_actor),
                "bc_weight_fraction": jnp.mean((bc_weights > 0).astype(jnp.float32)),
            }
            return (
                params,
                target_critic_params,
                {
                    "actor": actor_opt_state,
                    "critic": critic_opt_state,
                },
                key,
                priority,
                metrics,
            )

        return update_fn

    def _open_loop_state(self, *, eval_mode: bool, batch_size: int):
        prefix = "_eval" if eval_mode else "_train"
        plan_name = f"{prefix}_open_loop_plan"
        position_name = f"{prefix}_open_loop_position"
        valid_name = f"{prefix}_open_loop_valid"
        plan = getattr(self, plan_name)
        if plan is None or plan.shape[0] != batch_size:
            plan = np.zeros(
                (batch_size, self.action_sequence, self.action_dim),
                dtype=np.float32,
            )
            position = np.zeros((batch_size,), dtype=np.int32)
            valid = np.zeros((batch_size,), dtype=np.bool_)
            setattr(self, plan_name, plan)
            setattr(self, position_name, position)
            setattr(self, valid_name, valid)
        return (
            getattr(self, plan_name),
            getattr(self, position_name),
            getattr(self, valid_name),
        )

    def act(self, observations: dict, step: int, eval_mode: bool):
        batch_size = int(next(iter(observations.values())).shape[0])
        plan, position, valid = self._open_loop_state(
            eval_mode=eval_mode,
            batch_size=batch_size,
        )
        refresh = np.logical_or(~valid, position >= self.action_sequence)
        if np.any(refresh):
            self.rng_key, action_key = jax.random.split(self.rng_key)
            if step < self.num_explore_steps and not eval_mode:
                sampled = jax.random.uniform(
                    action_key,
                    (batch_size, self.action_sequence * self.action_dim),
                    minval=self.action_low,
                    maxval=self.action_high,
                )
                sampled_q = jnp.full((batch_size,), jnp.nan)
            else:
                obs_inputs = self._prepare_rl_obs_inputs(observations)
                unit_actions, sampled_q = self._policy_action_impl(
                    self.params,
                    obs_inputs,
                    action_key,
                )
                sampled = scale_unit_action(
                    unit_actions,
                    self.action_low,
                    self.action_high,
                )
            self._block(sampled)
            sampled = np.asarray(jax.device_get(sampled), dtype=np.float32).reshape(
                (batch_size, self.action_sequence, self.action_dim)
            )
            plan[refresh] = sampled[refresh]
            position[refresh] = 0
            valid[refresh] = True
            sampled_q_np = np.asarray(jax.device_get(sampled_q), dtype=np.float32)
            finite_q = sampled_q_np[np.isfinite(sampled_q_np)]
            if finite_q.size:
                self._last_selected_q = float(finite_q.mean())

        offsets = np.arange(self.action_sequence, dtype=np.int32)[None]
        indices = np.minimum(
            position[:, None] + offsets,
            self.action_sequence - 1,
        )
        action_chunk = np.take_along_axis(
            plan,
            indices[..., None],
            axis=1,
        ).copy()
        position += 1
        return action_chunk

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
        actions = self._as_jax_array(batch["action"], self.jnp.float32)
        expected_shape = (self.action_sequence, self.action_dim)
        if tuple(actions.shape[1:]) != expected_shape:
            raise ValueError(
                "Q-chunking replay action shape must be "
                f"[B, {expected_shape[0]}, {expected_shape[1]}], got "
                f"{tuple(actions.shape)}."
            )
        flat_actions = actions.reshape((actions.shape[0], -1))
        unit_actions = unscale_action(
            flat_actions,
            self.action_low,
            self.action_high,
        )
        rewards = self._as_jax_array(batch["reward"], self.jnp.float32).reshape(-1)
        discounts = self._as_jax_array(
            batch.get("discount", np.ones_like(batch["reward"])),
            self.jnp.float32,
        ).reshape(-1)
        terminal = self._as_jax_array(
            batch["terminal"],
            self.jnp.float32,
        ).reshape(-1)
        bootstrap = 1.0 - terminal
        pad_mask = self._extract_action_pad_mask(batch)
        if pad_mask is None:
            action_valid = jnp.ones(
                (actions.shape[0], self.action_sequence),
                dtype=jnp.bool_,
            )
        else:
            action_valid = jnp.logical_not(pad_mask)
        loss_weights = self._loss_weights(batch)
        bc_weights = loss_weights
        if self.bc_demo_only:
            if "demo" not in batch:
                raise ValueError("bc_demo_only requires a 'demo' flag in replay batches.")
            demo = self._as_jax_array(batch["demo"], self.jnp.float32).reshape(
                (actions.shape[0], -1)
            )[:, 0]
            bc_weights = loss_weights * (demo > 0).astype(self.jnp.float32)

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
            unit_actions,
            rewards,
            discounts,
            bootstrap,
            action_valid,
            loss_weights,
            bc_weights,
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

    def reset(self, step: int, agents_to_reset: list[int]):
        del step
        for agent_index in agents_to_reset:
            if agent_index < self.num_train_envs:
                if self._train_open_loop_plan is not None:
                    self._train_open_loop_plan[agent_index] = 0
                    self._train_open_loop_position[agent_index] = 0
                    self._train_open_loop_valid[agent_index] = False
                continue
            eval_index = agent_index - self.num_train_envs
            if (
                self._eval_open_loop_plan is not None
                and 0 <= eval_index < self._eval_open_loop_plan.shape[0]
            ):
                self._eval_open_loop_plan[eval_index] = 0
                self._eval_open_loop_position[eval_index] = 0
                self._eval_open_loop_valid[eval_index] = False

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["target_critic_params"] = self._tree_to_numpy(self.target_critic_params)
        if self.encoder is not None:
            state["encoder_state"] = self._tree_to_numpy(
                self.encoder.frozen_state_dict()
            )
        return state

    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict)
        self.target_critic_params = self._tree_from_numpy(
            state_dict.get(
                "target_critic_params",
                self._critic_bundle(self.params),
            )
        )
        if self.encoder is not None:
            self.encoder.load_frozen_state_dict(state_dict.get("encoder_state"))

    def rollout_diagnostics(self) -> dict[str, float]:
        return {"q_chunking_selected_q": float(self._last_selected_q)}


__all__ = [
    "QChunkCritic",
    "QChunkFlowActor",
    "QChunking",
    "QChunkingSpec",
    "q_chunking_spec_from_cfg",
    "q_chunking_td_target",
    "validate_q_chunking_config",
]
