"""Coarse-to-fine Q-Network with Action Sequence in pure JAX.

This module extends the local distributional CQN implementation with the
sequence critic from CQN-AS: every coarse-to-fine level predicts bins for all
future sequence positions in parallel, while a GRU shares information along
the sequence axis.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig

from robobase import utils
from robobase.method.cqn import CQN, CQNSpec, cqn_spec_from_cfg, encode_action, zoom_in
from robobase.method.rl_common import JaxRLMethodBase, RLModelSpec, activation


def random_shift_rgb(rgb: jax.Array, key: jax.Array, pad: int = 4) -> jax.Array:
    """Reference RandomShiftsAug for ``[batch, views, channels, H, W]`` RGB."""

    if pad <= 0:
        return rgb
    batch, views, channels, height, width = rgb.shape
    flat = rgb.reshape((batch * views, channels, height, width))
    flat = jnp.pad(
        flat,
        ((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode="edge",
    )
    shifts = jax.random.randint(
        key,
        (batch * views, 2),
        minval=0,
        maxval=2 * pad + 1,
    )

    def crop(image, shift):
        return jax.lax.dynamic_slice(
            image,
            (0, shift[0], shift[1]),
            (channels, height, width),
        )

    return jax.vmap(crop)(flat, shifts).reshape(rgb.shape)


@dataclass(frozen=True)
class CQNASpec(CQNSpec):
    """CQN hyperparameters plus the action-sequence architecture settings."""

    gru_layers: int
    temporal_ensemble: bool
    temporal_ensemble_replan_interval: int
    temporal_ensemble_gain: float
    tie_break_delta: float


def cqn_as_spec_from_cfg(cfg: DictConfig) -> CQNASpec:
    method = cfg.method
    base = cqn_spec_from_cfg(cfg)
    base_values = {field.name: getattr(base, field.name) for field in fields(CQNSpec)}
    return CQNASpec(
        **base_values,
        gru_layers=int(method.get("gru_layers", 1)),
        temporal_ensemble=bool(method.get("temporal_ensemble", True)),
        temporal_ensemble_replan_interval=int(
            method.get("temporal_ensemble_replan_interval", 1)
        ),
        temporal_ensemble_gain=float(method.get("temporal_ensemble_gain", 0.01)),
        tie_break_delta=float(method.get("tie_break_delta", 1e-4)),
    )


class C2FSequenceDistributionalCritic(nn.Module):
    """Official-style dueling CQN-AS critic with per-stream MLPs and GRUs."""

    hidden_dims: tuple[int, ...]
    action_sequence: int
    action_dim: int
    levels: int
    bins: int
    atoms: int
    low_dim_size: int = 0
    feature_dim: int = 64
    rgb_encoder_layers: int = 2
    gru_layers: int = 1
    activation_name: str = "silu"
    use_dueling: bool = True

    @nn.compact
    def __call__(
        self,
        features: jax.Array,
        level_one_hot: jax.Array,
        low_high_midpoint: jax.Array,
    ) -> jax.Array:
        """Return logits shaped ``[B, K, action_dim, bins, atoms]``."""

        batch_size = features.shape[0]
        dtype = features.dtype
        sequence_id = jnp.broadcast_to(
            jnp.eye(self.action_sequence, dtype=dtype)[None],
            (batch_size, self.action_sequence, self.action_sequence),
        )
        repeated_level = jnp.broadcast_to(
            level_one_hot[:, None, :],
            (batch_size, self.action_sequence, self.levels),
        )
        exact_pixel_arch = 0 < self.low_dim_size < features.shape[-1]

        def recurrent_stream(prefix: str) -> jax.Array:
            stream_features = features
            if exact_pixel_arch:
                low_dim = features[:, : self.low_dim_size]
                rgb = features[:, self.low_dim_size :]
                for index in range(self.rgb_encoder_layers):
                    rgb = nn.Dense(
                        self.hidden_dims[0],
                        use_bias=False,
                        kernel_init=nn.initializers.orthogonal(),
                        name=f"{prefix}_rgb_dense_{index}",
                    )(rgb)
                    rgb = nn.LayerNorm(name=f"{prefix}_rgb_norm_{index}")(rgb)
                    rgb = activation(rgb, self.activation_name)
                rgb = nn.Dense(
                    self.feature_dim,
                    use_bias=False,
                    kernel_init=nn.initializers.orthogonal(),
                    name=f"{prefix}_rgb_projection",
                )(rgb)
                rgb = nn.LayerNorm(name=f"{prefix}_rgb_projection_norm")(rgb)
                rgb = jnp.tanh(rgb)
                low_dim = nn.Dense(
                    self.feature_dim,
                    use_bias=False,
                    kernel_init=nn.initializers.orthogonal(),
                    name=f"{prefix}_low_dim_projection",
                )(low_dim)
                low_dim = nn.LayerNorm(name=f"{prefix}_low_dim_norm")(low_dim)
                low_dim = jnp.tanh(low_dim)
                stream_features = jnp.concatenate([rgb, low_dim], axis=-1)

            repeated_features = jnp.broadcast_to(
                stream_features[:, None, :],
                (batch_size, self.action_sequence, stream_features.shape[-1]),
            )
            x = jnp.concatenate(
                [
                    repeated_features,
                    low_high_midpoint,
                    sequence_id,
                    repeated_level,
                ],
                axis=-1,
            )
            for index, width in enumerate(self.hidden_dims):
                x = nn.Dense(
                    width,
                    use_bias=False,
                    kernel_init=nn.initializers.orthogonal(),
                    name=f"{prefix}_dense_{index}",
                )(x)
                x = nn.LayerNorm(name=f"{prefix}_norm_{index}")(x)
                x = activation(x, self.activation_name)

            hidden_size = self.hidden_dims[-1]
            ScanGRU = nn.scan(
                nn.GRUCell,
                variable_broadcast="params",
                split_rngs={"params": False},
                in_axes=1,
                out_axes=1,
            )
            for layer in range(self.gru_layers):
                initial_carry = jnp.zeros(
                    (batch_size, hidden_size),
                    dtype=x.dtype,
                )
                scan_gru = ScanGRU(
                    features=hidden_size,
                    name=f"{prefix}_gru_{layer}",
                )
                _, x = scan_gru(initial_carry, x)
            return x

        advantage_features = recurrent_stream("advantage")
        advantages = nn.Dense(
            self.action_dim * self.bins * self.atoms,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="advantage_head",
        )(advantage_features).reshape(
            (
                batch_size,
                self.action_sequence,
                self.action_dim,
                self.bins,
                self.atoms,
            )
        )
        if not self.use_dueling:
            return advantages

        value_features = recurrent_stream("value")
        values = nn.Dense(
            self.action_dim * self.atoms,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="value_head",
        )(value_features).reshape(
            (
                batch_size,
                self.action_sequence,
                self.action_dim,
                1,
                self.atoms,
            )
        )
        return values + advantages - advantages.mean(axis=-2, keepdims=True)


class CQNAS(CQN):
    """Distributional CQN-AS action-sequence agent.

    Temporal ensembling lives in the method rather than the environment wrapper.
    This lets exploration noise be applied to the actual ensembled action, as in
    the reference implementation.  By default a new plan is registered every
    primitive environment step.  A larger replan interval keeps executing the
    current plan between inference calls, then ensembles overlapping plans when
    a new one is registered.  The returned chunk stores the executed action at
    index zero; with ensembling disabled, one predicted plan is instead executed
    open-loop for K calls before it is refreshed.
    """

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
        gru_layers: int,
        temporal_ensemble: bool,
        temporal_ensemble_replan_interval: int,
        temporal_ensemble_gain: float,
        tie_break_delta: float,
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
        JaxRLMethodBase.__init__(
            self,
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
        if self.action_sequence < 2:
            raise ValueError("CQN-AS requires action_sequence >= 2.")
        if levels < 1 or bins < 2:
            raise ValueError("CQN-AS requires levels >= 1 and bins >= 2.")
        if atoms < 2 or v_max <= v_min:
            raise ValueError("CQN-AS requires atoms >= 2 and v_max > v_min.")
        if not model.hidden_dims:
            raise ValueError("CQN-AS requires at least one critic hidden layer.")
        if gru_layers < 1:
            raise ValueError("CQN-AS requires gru_layers >= 1.")
        if not 1 <= temporal_ensemble_replan_interval <= self.action_sequence:
            raise ValueError(
                "temporal_ensemble_replan_interval must be in "
                "[1, action_sequence]."
            )
        if not 0.0 < critic_target_tau <= 1.0:
            raise ValueError("critic_target_tau must be in (0, 1].")
        if temporal_ensemble_gain < 0.0:
            raise ValueError("temporal_ensemble_gain must be non-negative.")
        if tie_break_delta < 0.0:
            raise ValueError("tie_break_delta must be non-negative.")

        self.levels = int(levels)
        self.bins = int(bins)
        self.atoms = int(atoms)
        self.gru_layers = int(gru_layers)
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
        self.temporal_ensemble = bool(temporal_ensemble)
        self.temporal_ensemble_replan_interval = int(
            temporal_ensemble_replan_interval
        )
        self.temporal_ensemble_gain = float(temporal_ensemble_gain)
        self.tie_break_delta = float(tie_break_delta)
        self._seed = int(seed)

        input_dim = self._setup_rl_features(model, seed=seed)
        self.action_low, self.action_high = self._action_bounds()
        self._step_action_low = jnp.asarray(action_space.low[0], dtype=jnp.float32)
        self._step_action_high = jnp.asarray(action_space.high[0], dtype=jnp.float32)
        self.support = jnp.linspace(v_min, v_max, atoms, dtype=jnp.float32)
        self.critic_model = C2FSequenceDistributionalCritic(
            hidden_dims=model.hidden_dims,
            action_sequence=self.action_sequence,
            action_dim=self.action_dim,
            levels=self.levels,
            bins=self.bins,
            atoms=self.atoms,
            low_dim_size=(self.low_dim_size if self.use_pixels else 0),
            gru_layers=self.gru_layers,
            activation_name=model.activation,
            use_dueling=bool(use_dueling),
        )
        dummy_features = jnp.zeros((1, input_dim), dtype=jnp.float32)
        dummy_level = jnp.zeros((1, self.levels), dtype=jnp.float32)
        dummy_midpoint = jnp.zeros(
            (1, self.action_sequence, self.action_dim),
            dtype=jnp.float32,
        )
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

        self._train_action_history = None
        self._train_action_history_valid = None
        self._eval_action_history = None
        self._eval_action_history_valid = None
        self._train_open_loop_plan = None
        self._train_open_loop_position = None
        self._train_open_loop_valid = None
        self._eval_open_loop_plan = None
        self._eval_open_loop_position = None
        self._eval_open_loop_valid = None

    def _init_cached_pixel_feature_key(self, method_name: str) -> None:
        del method_name
        super()._init_cached_pixel_feature_key("cqn_as")

    @property
    def _flat_action_dim(self) -> int:
        return self.action_sequence * self.action_dim

    def _critic_logits_per_level(self, critic_params, features, action):
        flat_action = jnp.asarray(action, dtype=jnp.float32).reshape(
            (features.shape[0], self._flat_action_dim)
        )
        discrete_action = encode_action(
            flat_action,
            self.action_low,
            self.action_high,
            self.levels,
            self.bins,
        )
        low = jnp.broadcast_to(self.action_low, flat_action.shape)
        high = jnp.broadcast_to(self.action_high, flat_action.shape)
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
                (0.5 * (low + high)).reshape(
                    (features.shape[0], self.action_sequence, self.action_dim)
                ),
            )
            index = discrete_action[:, level, :]
            sequence_index = index.reshape(
                (features.shape[0], self.action_sequence, self.action_dim)
            )
            selected = jnp.take_along_axis(
                logits,
                sequence_index[..., None, None],
                axis=-2,
            )[..., 0, :]
            logits_per_level.append(
                logits.reshape(
                    (features.shape[0], self._flat_action_dim, self.bins, self.atoms)
                )
            )
            chosen_logits_per_level.append(
                selected.reshape(
                    (features.shape[0], self._flat_action_dim, self.atoms)
                )
            )
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

    def _greedy_action(self, critic_params, features, key=None):
        batch_size = features.shape[0]
        low = jnp.broadcast_to(
            self.action_low,
            (batch_size, self._flat_action_dim),
        )
        high = jnp.broadcast_to(
            self.action_high,
            (batch_size, self._flat_action_dim),
        )
        level_keys = [None] * self.levels
        if key is not None:
            level_keys = list(jax.random.split(key, self.levels))
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
                (0.5 * (low + high)).reshape(
                    (batch_size, self.action_sequence, self.action_dim)
                ),
            )
            probabilities = jax.nn.softmax(logits, axis=-1)
            q_values = jnp.sum(probabilities * self.support, axis=-1)
            index = jnp.argmax(q_values, axis=-1)
            level_key = level_keys[level]
            if level_key is not None:
                random_index = jax.random.randint(
                    level_key,
                    index.shape,
                    minval=0,
                    maxval=self.bins,
                )
                q_span = q_values.max(axis=-1) - q_values.min(axis=-1)
                index = jnp.where(
                    q_span < self.tie_break_delta,
                    random_index,
                    index,
                )
            selected.append(index)
            flat_index = index.reshape((batch_size, self._flat_action_dim))
            low, high = zoom_in(
                low,
                high,
                flat_index,
                self.bins,
                self.action_low,
                self.action_high,
            )
        action = (0.5 * (low + high)).reshape(
            (batch_size, self.action_sequence, self.action_dim)
        )
        return action, jnp.stack(selected, axis=1)

    def _build_greedy_action_fn(self):
        def action_fn(params, target_critic_params, obs_inputs, use_target, key):
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
            return self._greedy_action(critic_params, features, key=key)[0]

        return action_fn

    def _greedy_action_for_update(self, critic_params, features, action_key):
        return self._greedy_action(
            critic_params,
            features,
            key=action_key,
        )

    def _next_action_key(self):
        self.rng_key, action_key = jax.random.split(self.rng_key)
        return action_key

    def _augment_update_obs_inputs(self, obs_inputs, next_obs_inputs, key):
        if not isinstance(obs_inputs, dict) or "rgb" not in obs_inputs:
            return obs_inputs, next_obs_inputs, key
        augment_key, next_augment_key, action_key = jax.random.split(key, 3)
        obs_inputs = dict(obs_inputs)
        next_obs_inputs = dict(next_obs_inputs)
        obs_inputs["rgb"] = random_shift_rgb(obs_inputs["rgb"], augment_key)
        next_obs_inputs["rgb"] = random_shift_rgb(
            next_obs_inputs["rgb"], next_augment_key
        )
        return obs_inputs, next_obs_inputs, action_key

    def _history_for_mode(self, eval_mode: bool, batch_size: int):
        history_name = "_eval_action_history" if eval_mode else "_train_action_history"
        valid_name = (
            "_eval_action_history_valid"
            if eval_mode
            else "_train_action_history_valid"
        )
        history = getattr(self, history_name)
        if history is None or history.shape[0] != batch_size:
            history = np.zeros(
                (
                    batch_size,
                    self.action_sequence,
                    self.action_sequence,
                    self.action_dim,
                ),
                dtype=np.float32,
            )
            valid = np.zeros(
                (batch_size, self.action_sequence),
                dtype=np.bool_,
            )
            setattr(self, history_name, history)
            setattr(self, valid_name, valid)
        return history, getattr(self, valid_name)

    def _ensemble_current_action(
        self,
        action_chunk: np.ndarray,
        *,
        eval_mode: bool,
        register_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        if not self.temporal_ensemble:
            return action_chunk[:, 0].copy()

        history, valid = self._history_for_mode(eval_mode, action_chunk.shape[0])
        if register_mask is None:
            register_mask = np.ones((action_chunk.shape[0],), dtype=np.bool_)
        else:
            register_mask = np.asarray(register_mask, dtype=np.bool_)
            if register_mask.shape != (action_chunk.shape[0],):
                raise ValueError(
                    "register_mask must have shape (batch_size,), got "
                    f"{register_mask.shape}."
                )
        history[:, 1:] = history[:, :-1].copy()
        valid[:, 1:] = valid[:, :-1].copy()
        history[:, 0] = 0.0
        valid[:, 0] = False
        history[register_mask, 0] = action_chunk[register_mask]
        valid[register_mask, 0] = True

        ages = np.arange(self.action_sequence, dtype=np.int32)
        candidates = history[:, ages, ages, :]
        weights = np.exp(-self.temporal_ensemble_gain * ages).astype(np.float32)
        weights = weights[None, :] * valid.astype(np.float32)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
        return np.sum(candidates * weights[..., None], axis=1)

    def _temporal_replan_mask(
        self,
        *,
        eval_mode: bool,
        batch_size: int,
    ) -> np.ndarray:
        """Return which environments need a new plan before the next shift."""

        _, valid = self._history_for_mode(eval_mode, batch_size)
        has_plan = np.any(valid, axis=1)
        newest_plan_age = np.argmax(valid, axis=1)
        return np.logical_or(
            ~has_plan,
            newest_plan_age + 1 >= self.temporal_ensemble_replan_interval,
        )

    def _open_loop_action_chunk(
        self,
        new_action_chunk: np.ndarray,
        *,
        eval_mode: bool,
    ) -> np.ndarray:
        """Execute one cached plan for K steps when ensembling is disabled."""

        prefix = "_eval" if eval_mode else "_train"
        plan_name = f"{prefix}_open_loop_plan"
        position_name = f"{prefix}_open_loop_position"
        valid_name = f"{prefix}_open_loop_valid"
        plan = getattr(self, plan_name)
        if plan is None or plan.shape[0] != new_action_chunk.shape[0]:
            plan = np.zeros_like(new_action_chunk)
            position = np.zeros((new_action_chunk.shape[0],), dtype=np.int32)
            valid = np.zeros((new_action_chunk.shape[0],), dtype=np.bool_)
            setattr(self, plan_name, plan)
            setattr(self, position_name, position)
            setattr(self, valid_name, valid)
        else:
            position = getattr(self, position_name)
            valid = getattr(self, valid_name)

        refresh = np.logical_or(~valid, position >= self.action_sequence)
        plan[refresh] = new_action_chunk[refresh]
        position[refresh] = 0
        valid[refresh] = True

        offsets = np.arange(self.action_sequence, dtype=np.int32)[None, :]
        indices = np.minimum(
            position[:, None] + offsets,
            self.action_sequence - 1,
        )
        current_chunk = np.take_along_axis(plan, indices[..., None], axis=1).copy()
        position += 1
        return current_chunk

    def _open_loop_needs_refresh(self, *, eval_mode: bool, batch_size: int) -> bool:
        prefix = "_eval" if eval_mode else "_train"
        plan = getattr(self, f"{prefix}_open_loop_plan")
        if plan is None or plan.shape[0] != batch_size:
            return True
        position = getattr(self, f"{prefix}_open_loop_position")
        valid = getattr(self, f"{prefix}_open_loop_valid")
        return bool(np.any(np.logical_or(~valid, position >= self.action_sequence)))

    def act(self, observations: dict, step: int, eval_mode: bool):
        batch_size = int(next(iter(observations.values())).shape[0])
        register_mask = None
        if self.temporal_ensemble:
            register_mask = self._temporal_replan_mask(
                eval_mode=eval_mode,
                batch_size=batch_size,
            )
            needs_inference = bool(np.any(register_mask))
        else:
            needs_inference = self._open_loop_needs_refresh(
                eval_mode=eval_mode,
                batch_size=batch_size,
            )
        if needs_inference:
            obs_inputs = self._prepare_rl_obs_inputs(observations)
            self.rng_key, action_key = jax.random.split(self.rng_key)
            action = self._greedy_action_impl(
                self.params,
                self.target_critic_params,
                obs_inputs,
                jnp.asarray(self.use_target_network_for_rollout),
                action_key,
            )
            self._block(action)
            action_chunk = np.asarray(jax.device_get(action), dtype=np.float32)
        else:
            if self.temporal_ensemble:
                action_chunk = np.zeros(
                    (batch_size, self.action_sequence, self.action_dim),
                    dtype=np.float32,
                )
            else:
                prefix = "_eval" if eval_mode else "_train"
                action_chunk = getattr(self, f"{prefix}_open_loop_plan").copy()
        if self.temporal_ensemble:
            executed_action = self._ensemble_current_action(
                action_chunk,
                eval_mode=eval_mode,
                register_mask=register_mask,
            )
        else:
            action_chunk = self._open_loop_action_chunk(
                action_chunk,
                eval_mode=eval_mode,
            )
            executed_action = action_chunk[:, 0].copy()

        if not eval_mode:
            self.rng_key, noise_key = jax.random.split(self.rng_key)
            if step < self.num_explore_steps:
                executed_action = jax.random.uniform(
                    noise_key,
                    executed_action.shape,
                    minval=self._step_action_low,
                    maxval=self._step_action_high,
                )
            else:
                stddev = float(utils.schedule(self.stddev_schedule, step))
                executed_action = jnp.asarray(executed_action) + stddev * jax.random.normal(
                    noise_key,
                    executed_action.shape,
                )
                executed_action = jnp.clip(
                    executed_action,
                    self._step_action_low,
                    self._step_action_high,
                )
            executed_action = np.asarray(
                jax.device_get(executed_action),
                dtype=np.float32,
            )

        action_chunk = action_chunk.copy()
        action_chunk[:, 0] = executed_action
        return action_chunk

    def reset(self, step: int, agents_to_reset: list[int]):
        del step
        for agent_index in agents_to_reset:
            if agent_index < self.num_train_envs:
                if self._train_action_history is not None:
                    self._train_action_history[agent_index] = 0
                    self._train_action_history_valid[agent_index] = False
                if self._train_open_loop_plan is not None:
                    self._train_open_loop_plan[agent_index] = 0
                    self._train_open_loop_position[agent_index] = 0
                    self._train_open_loop_valid[agent_index] = False
                continue
            eval_index = agent_index - self.num_train_envs
            if (
                self._eval_action_history is not None
                and 0 <= eval_index < self._eval_action_history.shape[0]
            ):
                self._eval_action_history[eval_index] = 0
                self._eval_action_history_valid[eval_index] = False
            if (
                self._eval_open_loop_plan is not None
                and 0 <= eval_index < self._eval_open_loop_plan.shape[0]
            ):
                self._eval_open_loop_plan[eval_index] = 0
                self._eval_open_loop_position[eval_index] = 0
                self._eval_open_loop_valid[eval_index] = False


__all__ = [
    "C2FSequenceDistributionalCritic",
    "CQNAS",
    "CQNASpec",
    "cqn_as_spec_from_cfg",
]
