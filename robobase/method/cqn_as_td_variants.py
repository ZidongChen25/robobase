"""CQN-AS TD-target construction variants (R2 line: ``td-variants``).

This module isolates the research knobs that change **how the bootstrap target
of the single TD objective is built**, on top of the frozen official-fidelity
CQN-AS port in :mod:`robobase.method.cqn_as`:

``td_target_action_source``
    Which action is scored by the target critic at ``s_{t+1}``.

    * ``critic`` (legacy) -- Double-CQN greedy chunk from the online critic.
    * ``replay_next`` -- approximate SARSA: the executed replay chunk shifted
      by one step, i.e. the action sequence starting at ``t+1``.
    * ``critic_replay_max`` -- candidate max: score the greedy chunk and the
      exact ``action_tp1`` replay chunk with the deepest-level expected C51
      value and bootstrap from whichever is larger.

``critic_sequence_mode``
    ``full`` (legacy) trains the TD cross-entropy on every sequence token;
    ``effective_k0`` restricts it to the first (actually executed) token.

    NOTE -- deliberate divergence from the research monolith.  There
    ``_critic_training_slice`` (``cqn_as_research.py:3074-3077``) is only
    reached from the ``separate_bc_policy`` update path
    (``cqn_as_research.py:4909`` and ``4972-4973``); ``_build_update_fn``
    delegates to the plain-CQN update whenever ``separate_bc_policy`` is false
    (``cqn_as_research.py:4798-4802``) and that path never slices
    (no occurrence in ``cqn_research.py``).  So in the monolith
    ``effective_k0`` is silently inert on the single-objective critic path.
    This module implements the documented semantics on the pristine
    single-objective path, matching the same helper's use on the scalar-Q
    fork (``cqn_direct_q.py:740`` and ``912-913``).

``autoregressive_action_dims``
    Replaces the parallel per-dimension critic head with a causal correction:
    the head for action dimension ``d`` conditions on the already-selected
    dimensions ``< d`` (a pure-Q factorization, no imitation term).

Two further knobs of this line are **not** implemented here because they are
inseparably coupled to other R2 lines; see the module-level notes in
``COUPLED_OPTIONS`` and the agent report.  Requesting them raises ``ValueError``
naming the line that owns the missing head.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, fields
from typing import Iterator, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig

from robobase.method.cqn import encode_action, project_categorical, zoom_in
from robobase.method.cqn_as import (
    C2FSequenceDistributionalCritic,
    CQNAS,
    CQNASpec,
    cqn_as_spec_from_cfg,
)
from robobase.method.rl_common import JaxRLMethodBase, RLModelSpec, activation
from robobase.replay_buffer.replay_buffer import ReplayBuffer

# ``td_target_action_source`` values that need a head owned by another R2 line.
# Mapping: option -> (owning line, required parameter group).
#
# Coupling evidence in the research monolith (``cqn_as_research.py`` at ff9dfbf):
#   * ``bc_policy``   -- selected at 4882-4887, calls ``_policy_action``
#     (definition 3016-3060) which applies ``self.policy_model`` to
#     ``current_params["policy"]``.
#   * ``policy_value`` -- selected at 4888-4896, calls ``_policy_value_action``
#     (definition 3474-3560) which mixes the critic's normalized expected Q
#     with ``log_softmax`` of the same BC policy head, weighted by
#     ``td_target_policy_value_beta``.
#   * ``self.policy_model`` / ``params["policy"]`` only exist under
#     ``separate_bc_policy or use_frozen_support_mask`` (2620-2644), and the
#     constructor at 1600-1608 hard-requires ``separate_bc_policy=true`` for
#     both options.  ``separate_bc_policy`` belongs to the ``bc-policy`` line
#     and ``use_frozen_support_mask`` to ``frozen-support-mask``.
COUPLED_OPTIONS = {
    "bc_policy": ("bc-policy", "params['policy'] (separate BC policy head)"),
    "policy_value": (
        "bc-policy + flow-policy",
        "params['policy'] plus td_target_policy_value_beta",
    ),
}

IMPLEMENTED_TD_TARGET_ACTION_SOURCES = (
    "critic",
    "replay_next",
    "critic_replay_max",
)

CRITIC_SEQUENCE_MODES = ("full", "effective_k0")


def shift_replay_action_sequence(
    actions: jax.Array,
    action_sequence: int,
    action_dim: int,
) -> jax.Array:
    """Return the consecutive replay action sequence starting at t+1."""

    sequence = jnp.asarray(actions).reshape(
        (actions.shape[0], int(action_sequence), int(action_dim))
    )
    return jnp.concatenate(
        [sequence[:, 1:], sequence[:, -1:]],
        axis=1,
    )


class AutoregressiveActionCorrection(nn.Module):
    """Causal action-dimension correction for a parallel sequence critic."""

    hidden_dim: int
    action_sequence: int
    action_dim: int
    bins: int
    atoms: int
    activation_name: str = "silu"

    def setup(self):
        self.feature_projection = nn.Dense(
            self.hidden_dim,
            use_bias=False,
            kernel_init=nn.initializers.orthogonal(),
            name="feature_projection",
        )
        self.base_logit_projection = nn.Dense(
            self.hidden_dim,
            use_bias=False,
            kernel_init=nn.initializers.orthogonal(),
            name="base_logit_projection",
        )
        self.previous_action_projection = nn.Dense(
            self.hidden_dim,
            use_bias=False,
            kernel_init=nn.initializers.orthogonal(),
            name="previous_action_projection",
        )
        self.dimension_embedding = nn.Embed(
            num_embeddings=self.action_dim,
            features=self.hidden_dim,
            embedding_init=nn.initializers.normal(stddev=0.02),
            name="dimension_embedding",
        )
        self.input_norm = nn.LayerNorm(name="input_norm")
        self.gru = nn.GRUCell(
            features=self.hidden_dim,
            name="action_dimension_gru",
        )
        self.output_projection = nn.Dense(
            self.bins * self.atoms,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="output_projection",
        )

    def _step(
        self,
        carry: jax.Array,
        base_logits: jax.Array,
        feature_embedding: jax.Array,
        dimension: int,
        previous_action: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        batch_size, sequence_length = base_logits.shape[:2]
        repeated_feature_embedding = jnp.broadcast_to(
            feature_embedding[:, None, :],
            (batch_size, sequence_length, self.hidden_dim),
        )
        base_embedding = self.base_logit_projection(
            base_logits.reshape(
                (batch_size, sequence_length, self.bins * self.atoms)
            )
        )
        previous_action_embedding = self.previous_action_projection(
            previous_action[..., None]
        )
        dimension_embedding = self.dimension_embedding(
            jnp.asarray(dimension, dtype=jnp.int32)
        )
        inputs = (
            base_embedding
            + repeated_feature_embedding
            + previous_action_embedding
            + dimension_embedding
        )
        inputs = self.input_norm(inputs).reshape(
            (batch_size * sequence_length, self.hidden_dim)
        )
        inputs = activation(inputs, self.activation_name)
        flat_carry = carry.reshape(
            (batch_size * sequence_length, self.hidden_dim)
        )
        flat_carry, recurrent = self.gru(flat_carry, inputs)
        correction = self.output_projection(recurrent).reshape(
            (
                batch_size,
                sequence_length,
                self.bins,
                self.atoms,
            )
        )
        return (
            flat_carry.reshape(
                (batch_size, sequence_length, self.hidden_dim)
            ),
            correction,
        )

    def __call__(
        self,
        base_logits: jax.Array,
        features: jax.Array,
        action_context: jax.Array,
    ) -> jax.Array:
        """Teacher-force corrections using only preceding action dimensions."""

        batch_size = base_logits.shape[0]
        carry = jnp.zeros(
            (batch_size, self.action_sequence, self.hidden_dim),
            dtype=base_logits.dtype,
        )
        previous_action = jnp.zeros(
            (batch_size, self.action_sequence),
            dtype=base_logits.dtype,
        )
        feature_embedding = self.feature_projection(features)
        corrections = []
        for dimension in range(self.action_dim):
            carry, correction = self._step(
                carry,
                base_logits[:, :, dimension],
                feature_embedding,
                dimension,
                previous_action,
            )
            corrections.append(correction)
            previous_action = action_context[:, :, dimension]
        return jnp.stack(corrections, axis=2)

    def greedy_bins(
        self,
        base_logits: jax.Array,
        features: jax.Array,
        low: jax.Array,
        high: jax.Array,
        support: jax.Array,
        tie_break_delta: float,
        key: jax.Array | None = None,
    ) -> jax.Array:
        """Select bins causally, feeding each selected center to the next head."""

        batch_size = base_logits.shape[0]
        carry = jnp.zeros(
            (batch_size, self.action_sequence, self.hidden_dim),
            dtype=base_logits.dtype,
        )
        previous_action = jnp.zeros(
            (batch_size, self.action_sequence),
            dtype=base_logits.dtype,
        )
        feature_embedding = self.feature_projection(features)
        dimension_keys = [None] * self.action_dim
        if key is not None:
            dimension_keys = list(jax.random.split(key, self.action_dim))
        selected = []
        for dimension in range(self.action_dim):
            carry, correction = self._step(
                carry,
                base_logits[:, :, dimension],
                feature_embedding,
                dimension,
                previous_action,
            )
            logits = base_logits[:, :, dimension] + correction
            probabilities = jax.nn.softmax(logits, axis=-1)
            q_values = jnp.sum(probabilities * support, axis=-1)
            index = jnp.argmax(q_values, axis=-1)
            dimension_key = dimension_keys[dimension]
            if dimension_key is not None:
                random_index = jax.random.randint(
                    dimension_key,
                    index.shape,
                    minval=0,
                    maxval=self.bins,
                )
                q_span = q_values.max(axis=-1) - q_values.min(axis=-1)
                index = jnp.where(
                    q_span < tie_break_delta,
                    random_index,
                    index,
                )
            selected.append(index)
            bin_width = (
                high[:, :, dimension] - low[:, :, dimension]
            ) / self.bins
            previous_action = low[:, :, dimension] + (
                index.astype(base_logits.dtype) + 0.5
            ) * bin_width
        return jnp.stack(selected, axis=-1)


class AutoregressiveSequenceDistributionalCritic(nn.Module):
    """Pristine CQN-AS critic plus a causal action-dimension Q correction."""

    hidden_dims: tuple[int, ...]
    action_sequence: int
    action_dim: int
    levels: int
    bins: int
    atoms: int
    low_dim_size: int = 0
    gru_layers: int = 1
    activation_name: str = "silu"
    use_dueling: bool = True

    def setup(self):
        self.base_critic = C2FSequenceDistributionalCritic(
            hidden_dims=self.hidden_dims,
            action_sequence=self.action_sequence,
            action_dim=self.action_dim,
            levels=self.levels,
            bins=self.bins,
            atoms=self.atoms,
            low_dim_size=self.low_dim_size,
            gru_layers=self.gru_layers,
            activation_name=self.activation_name,
            use_dueling=self.use_dueling,
        )
        self.action_correction = AutoregressiveActionCorrection(
            hidden_dim=self.hidden_dims[-1],
            action_sequence=self.action_sequence,
            action_dim=self.action_dim,
            bins=self.bins,
            atoms=self.atoms,
            activation_name=self.activation_name,
        )

    def __call__(
        self,
        features: jax.Array,
        level_one_hot: jax.Array,
        low_high_midpoint: jax.Array,
        action_context: jax.Array,
    ) -> jax.Array:
        base_logits = self.base_critic(
            features,
            level_one_hot,
            low_high_midpoint,
        )
        correction = self.action_correction(
            base_logits,
            features,
            action_context,
        )
        return base_logits + correction

    def greedy_bins(
        self,
        features: jax.Array,
        level_one_hot: jax.Array,
        low_high_midpoint: jax.Array,
        low: jax.Array,
        high: jax.Array,
        support: jax.Array,
        tie_break_delta: float,
        key: jax.Array | None = None,
    ) -> jax.Array:
        base_logits = self.base_critic(
            features,
            level_one_hot,
            low_high_midpoint,
        )
        return self.action_correction.greedy_bins(
            base_logits,
            features,
            low,
            high,
            support,
            tie_break_delta,
            key,
        )


@dataclass(frozen=True)
class CQNASTdVariantsSpec(CQNASpec):
    """Official CQN-AS hyperparameters plus the TD-variant knobs."""

    td_target_action_source: str
    td_target_policy_value_beta: float | None
    critic_sequence_mode: str
    autoregressive_action_dims: bool


def cqn_as_td_variants_spec_from_cfg(cfg: DictConfig) -> CQNASTdVariantsSpec:
    method = cfg.method
    base = cqn_as_spec_from_cfg(cfg)
    base_values = {field.name: getattr(base, field.name) for field in fields(CQNASpec)}
    td_target_policy_value_beta = method.get("td_target_policy_value_beta", None)
    return CQNASTdVariantsSpec(
        **base_values,
        td_target_action_source=str(
            method.get("td_target_action_source", "critic")
        ).lower(),
        td_target_policy_value_beta=(
            None
            if td_target_policy_value_beta is None
            else float(td_target_policy_value_beta)
        ),
        critic_sequence_mode=str(
            method.get("critic_sequence_mode", "full")
        ).lower(),
        autoregressive_action_dims=bool(
            method.get("autoregressive_action_dims", False)
        ),
    )


class CQNASTdVariants(CQNAS):
    """CQN-AS with alternative TD-target construction knobs.

    With every flag at its default (``td_target_action_source='critic'``,
    ``critic_sequence_mode='full'``, ``autoregressive_action_dims=False``,
    ``td_target_policy_value_beta=None``) this class is numerically identical
    to the pristine :class:`robobase.method.cqn_as.CQNAS`: the extra hooks
    reduce to the pristine greedy selection, an identity slice and no extra
    metrics.
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
        *,
        td_target_action_source: str = "critic",
        td_target_policy_value_beta: float | None = None,
        critic_sequence_mode: str = "full",
        autoregressive_action_dims: bool = False,
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

        # --- td-variants line validation -------------------------------
        td_target_action_source = str(td_target_action_source).lower()
        if td_target_action_source in COUPLED_OPTIONS:
            owner, requirement = COUPLED_OPTIONS[td_target_action_source]
            raise ValueError(
                f"td_target_action_source={td_target_action_source!r} needs "
                f"{requirement}, which is owned by the '{owner}' research "
                "line; it cannot be built against the pristine critic alone."
            )
        if td_target_action_source not in IMPLEMENTED_TD_TARGET_ACTION_SOURCES:
            raise ValueError(
                "td_target_action_source must be one of "
                f"{set(IMPLEMENTED_TD_TARGET_ACTION_SOURCES)}."
            )
        if td_target_policy_value_beta is not None:
            raise ValueError(
                "td_target_policy_value_beta is only meaningful with "
                "td_target_action_source=policy_value, which needs the BC "
                "policy head owned by the 'bc-policy' line; leave it null."
            )
        critic_sequence_mode = str(critic_sequence_mode).lower()
        if critic_sequence_mode not in CRITIC_SEQUENCE_MODES:
            raise ValueError(
                "critic_sequence_mode must be one of {'full', 'effective_k0'}."
            )

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

        self.td_target_action_source = td_target_action_source
        self.td_target_policy_value_beta = td_target_policy_value_beta
        self.critic_sequence_mode = critic_sequence_mode
        self.autoregressive_action_dims = bool(autoregressive_action_dims)

        input_dim = self._setup_rl_features(model, seed=seed)
        self.action_low, self.action_high = self._action_bounds()
        self._step_action_low = jnp.asarray(action_space.low[0], dtype=jnp.float32)
        self._step_action_high = jnp.asarray(action_space.high[0], dtype=jnp.float32)
        self.support = jnp.linspace(v_min, v_max, atoms, dtype=jnp.float32)
        critic_model_type = (
            AutoregressiveSequenceDistributionalCritic
            if self.autoregressive_action_dims
            else C2FSequenceDistributionalCritic
        )
        self.critic_model = critic_model_type(
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
        if self.autoregressive_action_dims:
            critic_params = self.critic_model.init(
                self.rng_key,
                dummy_features,
                dummy_level,
                dummy_midpoint,
                dummy_midpoint,
            )
        else:
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

    # ------------------------------------------------------------------
    # Critic evaluation / action selection
    # ------------------------------------------------------------------
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
            index = discrete_action[:, level, :]
            midpoint = (0.5 * (low + high)).reshape(
                (features.shape[0], self.action_sequence, self.action_dim)
            )
            if self.autoregressive_action_dims:
                # Teacher forcing: the causal head for dimension d reads the
                # centre of the *chosen* cell of the preceding dimensions.
                bin_width = (high - low) / self.bins
                action_context = (
                    low + (index.astype(jnp.float32) + 0.5) * bin_width
                ).reshape(
                    (
                        features.shape[0],
                        self.action_sequence,
                        self.action_dim,
                    )
                )
                logits = self.critic_model.apply(
                    critic_params,
                    features,
                    one_hot,
                    midpoint,
                    action_context,
                )
            else:
                logits = self.critic_model.apply(
                    critic_params,
                    features,
                    one_hot,
                    midpoint,
                )
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
            level_key = level_keys[level]
            midpoint = (0.5 * (low + high)).reshape(
                (batch_size, self.action_sequence, self.action_dim)
            )
            if self.autoregressive_action_dims:
                index = self.critic_model.apply(
                    critic_params,
                    features,
                    one_hot,
                    midpoint,
                    low.reshape(
                        (batch_size, self.action_sequence, self.action_dim)
                    ),
                    high.reshape(
                        (batch_size, self.action_sequence, self.action_dim)
                    ),
                    self.support,
                    self.tie_break_delta,
                    level_key,
                    method=self.critic_model.greedy_bins,
                )
            else:
                logits = self.critic_model.apply(
                    critic_params,
                    features,
                    one_hot,
                    midpoint,
                )
                probabilities = jax.nn.softmax(logits, axis=-1)
                q_values = jnp.sum(probabilities * self.support, axis=-1)
                index = jnp.argmax(q_values, axis=-1)
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

    # ------------------------------------------------------------------
    # td-variants hooks
    # ------------------------------------------------------------------
    def _critic_training_slice(self, values):
        """Restrict the TD objective to the tokens it is allowed to train.

        ``values`` is laid out ``[batch, levels, action_sequence * action_dim,
        ...]``; ``effective_k0`` keeps only the first chunk token, i.e. the
        action that is actually executed after temporal ensembling.
        """

        if self.critic_sequence_mode == "effective_k0":
            return values[:, :, : self.action_dim]
        return values

    def _score_action_sequence_for_backup(self, critic_params, features, action):
        """Deepest-level mean expected C51 value for one complete chunk."""

        chosen_logits, _ = self._critic_logits_per_level(
            critic_params,
            features,
            action,
        )
        chosen_probabilities = jax.nn.softmax(chosen_logits, axis=-1)
        chosen_q = jnp.sum(
            chosen_probabilities * self.support,
            axis=-1,
        )
        return chosen_q[:, -1].mean(axis=-1)

    def _td_target_action_for_update(
        self,
        critic_params,
        features,
        replay_actions,
        replay_next_actions,
        demos,
        action_key,
    ):
        """Select the bootstrap action used by the single TD objective.

        ``demos`` is accepted for signature parity with the research hook
        (``cqn_as_research.py:3961-4044``) but unused here: the only consumer
        is ``demo_behavior_force_probability``, which the taxonomy assigns to
        the ``bc-policy`` line.  Its branch lives at
        ``cqn_as_research.py:4011-4027`` inside this same ``critic_replay_max``
        block and must be re-merged there if the two lines are ever composed.
        """

        del demos
        if self.td_target_action_source == "replay_next":
            # The replay sequence contains consecutive executed actions, so
            # shifting once supplies the action sequence at s_{t+1}.
            return (
                shift_replay_action_sequence(
                    replay_actions,
                    self.action_sequence,
                    self.action_dim,
                ),
                {},
            )
        if self.td_target_action_source == "critic_replay_max":
            greedy_action, _ = self._greedy_action_for_update(
                critic_params,
                features,
                action_key,
            )
            behavior_action = jnp.asarray(
                replay_next_actions,
                dtype=jnp.float32,
            ).reshape(
                (
                    replay_next_actions.shape[0],
                    self.action_sequence,
                    self.action_dim,
                )
            )
            greedy_score = self._score_action_sequence_for_backup(
                critic_params,
                features,
                greedy_action,
            )
            behavior_score = self._score_action_sequence_for_backup(
                critic_params,
                features,
                behavior_action,
            )
            behavior_selected = behavior_score >= greedy_score
            selected_action = jnp.where(
                behavior_selected[:, None, None],
                behavior_action,
                greedy_action,
            )
            return selected_action, {
                "behavior_selected": behavior_selected,
                "behavior_score": behavior_score,
                "greedy_score": greedy_score,
            }
        return self._greedy_action_for_update(
            critic_params,
            features,
            action_key,
        )

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def _build_update_fn(self):
        optimizer = self.optimizer
        tau = self.critic_target_tau
        use_replay_td_target = self.td_target_action_source == "replay_next"
        use_replay_candidate_target = (
            self.td_target_action_source == "critic_replay_max"
        )

        def update_fn(
            params,
            target_critic_params,
            opt_state,
            obs_inputs,
            next_obs_inputs,
            actions,
            next_actions,
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
                next_action, target_action_info = (
                    self._td_target_action_for_update(
                        current_params["critic"],
                        next_features,
                        actions,
                        next_actions,
                        demos,
                        action_key,
                    )
                )
                target_logits, _ = self._critic_logits_per_level(
                    target_critic_params,
                    next_features,
                    next_action,
                )
                target_logits = self._critic_training_slice(target_logits)
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
                chosen_logits = self._critic_training_slice(chosen_logits)
                all_logits = self._critic_training_slice(all_logits)
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
                return critic_loss, (
                    per_sample,
                    entropy,
                    target_entropy,
                    target_action_info,
                )

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
            per_sample, entropy, projected_entropy, target_action_info = aux
            priority = jnp.sqrt(per_sample + 1e-10)
            priority = priority / jnp.maximum(jnp.max(priority), 1e-10)
            metrics = {
                "critic_loss": critic_loss,
                "entropy": entropy,
                "target_entropy": projected_entropy,
                "loss_coeff": jnp.mean(loss_weights),
            }
            if use_replay_td_target:
                metrics["td_target_replay_next"] = jnp.asarray(
                    1.0,
                    dtype=jnp.float32,
                )
            if use_replay_candidate_target:
                selected = target_action_info["behavior_selected"].astype(
                    jnp.float32
                )
                behavior_score = target_action_info["behavior_score"]
                greedy_score = target_action_info["greedy_score"]
                metrics["behavior_candidate_fraction"] = jnp.mean(selected)
                metrics["behavior_candidate_score"] = jnp.mean(behavior_score)
                metrics["greedy_candidate_score"] = jnp.mean(greedy_score)
                metrics["behavior_minus_greedy_q"] = jnp.mean(
                    behavior_score - greedy_score
                )
            return (
                params,
                target_critic_params,
                opt_state,
                priority,
                metrics,
            )

        return update_fn

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
            requires_next_action = (
                self.td_target_action_source == "critic_replay_max"
            )
            if requires_next_action and "action_tp1" not in batch:
                raise KeyError(
                    "critic_replay_max requires replay.include_next_action=true "
                    "and an action_tp1 batch element."
                )
            next_action_values = batch.get("action_tp1", batch["action"])
            next_actions = self._as_jax_array(
                next_action_values,
                self.jnp.float32,
            ).reshape((next_action_values.shape[0], -1))
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
                next_actions,
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


__all__ = [
    "AutoregressiveActionCorrection",
    "AutoregressiveSequenceDistributionalCritic",
    "COUPLED_OPTIONS",
    "CQNASTdVariants",
    "CQNASTdVariantsSpec",
    "cqn_as_td_variants_spec_from_cfg",
    "shift_replay_action_sequence",
]
