"""CQN-AS twin-critic research line (Stage 32-35).

Extracted from the research monolith (`cqn_as_research.py`) as an isolated
subclass of the FROZEN pristine `robobase.method.cqn_as.CQNAS`.  This file owns
exactly four flags:

``pessimistic_twin_critic``
    Two independently initialized direct C51 critics trained by the same
    reward Bellman objective.  Candidate selection, bootstrap distributions and
    rollout bins use the lower expected value.
``auxiliary_td_loss_weight``
    Optional second reward-only C51 Bellman loss from the same replay start
    (matched 1-step + n-step); ``0.5 * (L_1step + L_Nstep)`` at weight 1.
``episodic_twin_head_exploration``
    Randomized value-function exploration: each training environment samples
    one of the twin heads at episode reset and follows it greedily for the
    whole episode.  Evaluation and Bellman targets stay pessimistic min-Q.
``twin_rollout_beam_width``
    Rollout-time top-two joint coarse-to-fine beam with complete-chunk Q
    reranking.  Width 1 is the exact legacy action path; the beam never touches
    a Bellman/MC target or a loss.

With every flag at its default the class is behaviourally identical to the
pristine `CQNAS`: `_build_update_fn`, `_build_greedy_action_fn`, `act` and
`update` all delegate straight to the pristine implementations, and the
parameter tree contains only the pristine ``critic``/``encoder`` subtrees.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, fields
from typing import Iterator, Optional

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig

from robobase import utils
from robobase.method.cqn import project_categorical, zoom_in
from robobase.method.cqn_as import (
    C2FSequenceDistributionalCritic,
    CQNAS,
    CQNASpec,
    cqn_as_spec_from_cfg,
    random_shift_rgb,
)
from robobase.method.rl_common import JaxRLMethodBase, RLModelSpec
from robobase.replay_buffer.replay_buffer import ReplayBuffer

__all__ = [
    "CQNASTwinCritic",
    "CQNASTwinCriticSpec",
    "cqn_as_twin_critic_spec_from_cfg",
    "pessimistic_categorical_q",
    "select_episodic_twin_actions",
    "top2_joint_beam",
]


def pessimistic_categorical_q(
    logits1: jax.Array,
    logits2: jax.Array,
    support: jax.Array,
) -> jax.Array:
    """Lower expected C51 value from two independently trained critics."""

    q1 = jnp.sum(jax.nn.softmax(logits1, axis=-1) * support, axis=-1)
    q2 = jnp.sum(jax.nn.softmax(logits2, axis=-1) * support, axis=-1)
    return jnp.minimum(q1, q2)


def select_episodic_twin_actions(
    action1: jax.Array,
    action2: jax.Array,
    head_indices: jax.Array,
) -> jax.Array:
    """Select one complete action chunk per environment and critic head."""

    choose_second = jnp.asarray(head_indices, dtype=jnp.int32) == 1
    return jnp.where(choose_second[:, None, None], action2, action1)


def top2_joint_beam(
    parent_scores: jax.Array,
    q_values: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Keep the best joint assignments from each factor's top-two bins.

    ``parent_scores`` is ``[B, W]`` and ``q_values`` is
    ``[B, W, F, bins]``.  Every parent defines a factorized binary search
    space from its two highest reward-Q bins.  A width-``W`` dynamic program
    keeps the globally best complete assignments without independently
    sampling factors.  Returned tensors are the new cumulative scores,
    source-parent indices, and complete bin assignments ``[B, W, F]``.

    This is an action maximization helper only.  It creates no target, loss,
    action-label gradient, or demonstration preference.
    """

    if q_values.ndim != 4:
        raise ValueError("q_values must have shape [B, W, F, bins].")
    if parent_scores.shape != q_values.shape[:2]:
        raise ValueError("parent_scores must match q_values [B, W].")
    if q_values.shape[-1] < 2:
        raise ValueError("top2_joint_beam requires at least two bins.")

    batch, width, factors, _ = q_values.shape
    top_values, top_indices = jax.lax.top_k(q_values, 2)
    best_values = top_values[..., 0]
    regrets = (best_values - top_values[..., 1]) / float(factors)
    second_indices = top_indices[..., 1]

    scores = parent_scores + jnp.mean(best_values, axis=-1)
    origins = jnp.broadcast_to(
        jnp.arange(width, dtype=jnp.int32)[None],
        (batch, width),
    )
    assignments = top_indices[..., 0]

    def expand_factor(carry, factor):
        current_scores, current_origins, current_assignments = carry
        origin_regrets = jnp.take_along_axis(
            regrets[:, :, factor], current_origins, axis=1
        )
        origin_seconds = jnp.take_along_axis(
            second_indices[:, :, factor], current_origins, axis=1
        )

        flipped_assignments = current_assignments.at[:, :, factor].set(
            origin_seconds
        )
        expanded_scores = jnp.concatenate(
            [current_scores, current_scores - origin_regrets], axis=1
        )
        expanded_origins = jnp.concatenate(
            [current_origins, current_origins], axis=1
        )
        expanded_assignments = jnp.concatenate(
            [current_assignments, flipped_assignments], axis=1
        )

        kept_scores, kept_positions = jax.lax.top_k(expanded_scores, width)
        kept_origins = jnp.take_along_axis(
            expanded_origins, kept_positions, axis=1
        )
        kept_assignments = jnp.take_along_axis(
            expanded_assignments,
            kept_positions[:, :, None],
            axis=1,
        )
        return (kept_scores, kept_origins, kept_assignments), None

    (scores, origins, assignments), _ = jax.lax.scan(
        expand_factor,
        (scores, origins, assignments),
        jnp.arange(factors, dtype=jnp.int32),
    )
    return scores, origins, assignments


@dataclass(frozen=True)
class CQNASTwinCriticSpec(CQNASpec):
    """Pristine CQN-AS hyperparameters plus the twin-critic settings."""

    pessimistic_twin_critic: bool
    auxiliary_td_loss_weight: float
    episodic_twin_head_exploration: bool
    twin_rollout_beam_width: int


def cqn_as_twin_critic_spec_from_cfg(cfg: DictConfig) -> CQNASTwinCriticSpec:
    method = cfg.method
    replay = cfg.get("replay", None)

    def replay_get(name, default):
        return default if replay is None else replay.get(name, default)

    pessimistic_twin_critic = bool(
        method.get("pessimistic_twin_critic", False)
    )
    auxiliary_td_loss_weight = float(
        method.get("auxiliary_td_loss_weight", 0.0)
    )
    if auxiliary_td_loss_weight < 0.0:
        raise ValueError("method.auxiliary_td_loss_weight must be non-negative.")
    if pessimistic_twin_critic and not bool(
        replay_get("include_next_action", False)
    ):
        # The twin Bellman target maximizes over {pessimistic greedy chunk,
        # replayed next chunk}; without action_tp1 the graph cannot be built.
        raise ValueError(
            "pessimistic_twin_critic requires replay.include_next_action=true "
            "(the twin target reads the replayed next action chunk)."
        )
    if auxiliary_td_loss_weight > 0.0:
        auxiliary_nstep = replay_get("auxiliary_nstep", None)
        auxiliary_violations = []
        if not pessimistic_twin_critic:
            auxiliary_violations.append("method.pessimistic_twin_critic=true")
        if int(replay_get("nstep", 1)) != 1:
            auxiliary_violations.append("replay.nstep=1")
        if auxiliary_nstep is None or int(auxiliary_nstep) <= 1:
            auxiliary_violations.append("replay.auxiliary_nstep > 1")
        if not bool(replay_get("include_tp1", True)):
            auxiliary_violations.append("replay.include_tp1=true")
        if not bool(replay_get("include_next_action", False)):
            auxiliary_violations.append("replay.include_next_action=true")
        if auxiliary_violations:
            raise ValueError(
                "auxiliary TD requires the matched 1-step + n-step twin-C51 "
                "path: " + "; ".join(auxiliary_violations)
            )
    base = cqn_as_spec_from_cfg(cfg)
    base_values = {
        field.name: getattr(base, field.name) for field in fields(CQNASpec)
    }
    return CQNASTwinCriticSpec(
        **base_values,
        pessimistic_twin_critic=pessimistic_twin_critic,
        auxiliary_td_loss_weight=auxiliary_td_loss_weight,
        episodic_twin_head_exploration=bool(
            method.get("episodic_twin_head_exploration", False)
        ),
        twin_rollout_beam_width=int(
            method.get("twin_rollout_beam_width", 1)
        ),
    )


class CQNASTwinCritic(CQNAS):
    """CQN-AS with pessimistic twin C51 critics and a joint rollout beam."""

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
        pessimistic_twin_critic: bool = False,
        auxiliary_td_loss_weight: float = 0.0,
        episodic_twin_head_exploration: bool = False,
        twin_rollout_beam_width: int = 1,
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

        # ---- twin-critic line validation (research parity) --------------
        if auxiliary_td_loss_weight < 0.0:
            raise ValueError(
                "auxiliary_td_loss_weight must be non-negative."
            )
        if auxiliary_td_loss_weight > 0.0 and not pessimistic_twin_critic:
            raise ValueError(
                "auxiliary_td_loss_weight > 0 requires "
                "pessimistic_twin_critic=true."
            )
        if pessimistic_twin_critic:
            twin_violations = []
            if use_dueling:
                twin_violations.append("use_dueling=false")
            if centralized_critic:
                twin_violations.append("centralized_critic=false")
            if twin_violations:
                raise ValueError(
                    "pessimistic_twin_critic requires the isolated "
                    "reward-only direct-C51 path: "
                    + "; ".join(twin_violations)
                )
        if episodic_twin_head_exploration:
            exploration_violations = []
            if not pessimistic_twin_critic:
                exploration_violations.append("pessimistic_twin_critic=true")
            if exploration_violations:
                raise ValueError(
                    "episodic_twin_head_exploration requires the isolated "
                    "twin-C51 exploration path: "
                    + "; ".join(exploration_violations)
                )
        if twin_rollout_beam_width < 1:
            raise ValueError("twin_rollout_beam_width must be at least 1.")
        if twin_rollout_beam_width > 1:
            beam_violations = []
            if pessimistic_twin_critic and not episodic_twin_head_exploration:
                beam_violations.append(
                    "episodic_twin_head_exploration=true"
                )
            if bins < 2:
                beam_violations.append("bins>=2")
            if beam_violations:
                raise ValueError(
                    "twin_rollout_beam_width > 1 requires a direct-C51 "
                    "critic rollout-search path: "
                    + "; ".join(beam_violations)
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

        self.pessimistic_twin_critic = bool(pessimistic_twin_critic)
        self.auxiliary_td_loss_weight = float(auxiliary_td_loss_weight)
        self.episodic_twin_head_exploration = bool(
            episodic_twin_head_exploration
        )
        self.twin_rollout_beam_width = int(twin_rollout_beam_width)
        # NumPy-only episode-head state.  Constructed unconditionally (as in
        # the research monolith) so a checkpoint round-trip is independent of
        # the flag, and so it never touches the JAX parameter tree or the JAX
        # RNG stream.
        self._episodic_twin_head_rng = np.random.default_rng(int(seed) + 157)
        self._episodic_twin_heads = np.full(
            (int(num_train_envs),), -1, dtype=np.int8
        )
        self._episodic_twin_head_assignments = np.zeros(
            (2,), dtype=np.int64
        )

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

        def init_critic(key):
            return self.critic_model.init(
                key,
                dummy_features,
                dummy_level,
                dummy_midpoint,
            )

        if self.pessimistic_twin_critic:
            self.rng_key, critic_key, critic2_key = jax.random.split(
                self.rng_key, 3
            )
            critic_params = init_critic(critic_key)
            critic2_params = init_critic(critic2_key)
            self.params = {
                "critic": critic_params,
                "critic2": critic2_params,
            }
        else:
            # Flags off: identical to the pristine class, including the RNG
            # stream (no split before the single critic init).
            critic_params = init_critic(self.rng_key)
            critic2_params = None
            self.params = {"critic": critic_params}
        if self._trainable_encoder:
            self.params["encoder"] = self._encoder_params
        self.target_critic_params = (
            (critic_params, critic2_params)
            if self.pessimistic_twin_critic
            else critic_params
        )

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
    # Rollout-time action selection
    # ------------------------------------------------------------------

    @property
    def _twin_action_path_enabled(self) -> bool:
        """True when act() must thread twin/beam state through the graph."""

        return bool(
            self.pessimistic_twin_critic or self.twin_rollout_beam_width > 1
        )

    def _pessimistic_greedy_action(
        self,
        critic1_params,
        critic2_params,
        features,
        key=None,
    ):
        """Coarse-to-fine argmax under min(E[Z1], E[Z2])."""

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
            midpoint = (0.5 * (low + high)).reshape(
                (batch_size, self.action_sequence, self.action_dim)
            )
            logits1 = self.critic_model.apply(
                critic1_params,
                features,
                one_hot,
                midpoint,
            )
            logits2 = self.critic_model.apply(
                critic2_params,
                features,
                one_hot,
                midpoint,
            )
            q_values = pessimistic_categorical_q(
                logits1,
                logits2,
                self.support,
            )
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
            low, high = zoom_in(
                low,
                high,
                index.reshape((batch_size, self._flat_action_dim)),
                self.bins,
                self.action_low,
                self.action_high,
            )
        action = (0.5 * (low + high)).reshape(
            (batch_size, self.action_sequence, self.action_dim)
        )
        return action, jnp.stack(selected, axis=1)

    def _score_action_sequence_for_backup(
        self,
        critic_params,
        features,
        action,
    ):
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

    def _pessimistic_score_action_sequence(
        self,
        critic1_params,
        critic2_params,
        features,
        action,
    ):
        score1 = self._score_action_sequence_for_backup(
            critic1_params,
            features,
            action,
        )
        score2 = self._score_action_sequence_for_backup(
            critic2_params,
            features,
            action,
        )
        return jnp.minimum(score1, score2), score1, score2

    def _joint_beam_action(
        self,
        critic1_params,
        features,
        critic2_params=None,
    ):
        """Top-two joint C2F beam followed by complete-chunk Q reranking.

        With one critic this is the coherent rollout search used by a sampled
        episode head.  With two critics, every bin score and the final chunk
        score use clipped twin expected Q.  The beam acts only at rollout;
        update-time Bellman maximization deliberately remains unchanged for
        the Stage-34 isolation.
        """

        batch_size = features.shape[0]
        beam_width = int(self.twin_rollout_beam_width)
        flat_dim = self._flat_action_dim
        low = jnp.broadcast_to(
            self.action_low,
            (batch_size, beam_width, flat_dim),
        )
        high = jnp.broadcast_to(
            self.action_high,
            (batch_size, beam_width, flat_dim),
        )
        beam_scores = jnp.full(
            (batch_size, beam_width),
            -jnp.inf,
            dtype=jnp.float32,
        ).at[:, 0].set(0.0)
        repeated_features = jnp.repeat(features, beam_width, axis=0)

        for level in range(self.levels):
            one_hot = jnp.broadcast_to(
                jax.nn.one_hot(level, self.levels, dtype=jnp.float32),
                (batch_size * beam_width, self.levels),
            )
            midpoint = (0.5 * (low + high)).reshape(
                (
                    batch_size * beam_width,
                    self.action_sequence,
                    self.action_dim,
                )
            )
            logits1 = self.critic_model.apply(
                critic1_params,
                repeated_features,
                one_hot,
                midpoint,
            )
            probabilities1 = jax.nn.softmax(logits1, axis=-1)
            q_values = jnp.sum(
                probabilities1 * self.support,
                axis=-1,
            )
            if critic2_params is not None:
                logits2 = self.critic_model.apply(
                    critic2_params,
                    repeated_features,
                    one_hot,
                    midpoint,
                )
                probabilities2 = jax.nn.softmax(logits2, axis=-1)
                q_values2 = jnp.sum(
                    probabilities2 * self.support,
                    axis=-1,
                )
                q_values = jnp.minimum(q_values, q_values2)
            q_values = q_values.reshape(
                (batch_size, beam_width, flat_dim, self.bins)
            )
            beam_scores, parent_indices, selected_indices = top2_joint_beam(
                beam_scores,
                q_values,
            )
            gather_indices = jnp.broadcast_to(
                parent_indices[:, :, None],
                (batch_size, beam_width, flat_dim),
            )
            parent_low = jnp.take_along_axis(
                low, gather_indices, axis=1
            )
            parent_high = jnp.take_along_axis(
                high, gather_indices, axis=1
            )
            low, high = zoom_in(
                parent_low.reshape((batch_size * beam_width, flat_dim)),
                parent_high.reshape((batch_size * beam_width, flat_dim)),
                selected_indices.reshape(
                    (batch_size * beam_width, flat_dim)
                ),
                self.bins,
                self.action_low,
                self.action_high,
            )
            low = low.reshape((batch_size, beam_width, flat_dim))
            high = high.reshape((batch_size, beam_width, flat_dim))

        chunks = (0.5 * (low + high)).reshape(
            (
                batch_size,
                beam_width,
                self.action_sequence,
                self.action_dim,
            )
        )
        flat_chunks = chunks.reshape(
            (
                batch_size * beam_width,
                self.action_sequence,
                self.action_dim,
            )
        )
        score1 = self._score_action_sequence_for_backup(
            critic1_params,
            repeated_features,
            flat_chunks,
        ).reshape((batch_size, beam_width))
        final_scores = score1
        if critic2_params is not None:
            score2 = self._score_action_sequence_for_backup(
                critic2_params,
                repeated_features,
                flat_chunks,
            ).reshape((batch_size, beam_width))
            final_scores = jnp.minimum(score1, score2)
        best = jnp.argmax(final_scores, axis=-1)
        selected = jnp.take_along_axis(
            chunks,
            best[:, None, None, None],
            axis=1,
        )[:, 0]
        return selected, final_scores

    def _build_greedy_action_fn(self):
        if not self._twin_action_path_enabled:
            # Flags off: the pristine five-argument action function, byte for
            # byte the pristine rollout path.
            return super()._build_greedy_action_fn()

        def action_fn(
            params,
            target_critic_params,
            obs_inputs,
            use_target,
            key,
            twin_head_indices,
        ):
            features = self._rl_features(
                params.get("encoder", None),
                obs_inputs,
                stop_gradient=True,
            )
            if self.pessimistic_twin_critic:
                critic1_params = jax.lax.cond(
                    use_target,
                    lambda _: target_critic_params[0],
                    lambda _: params["critic"],
                    operand=None,
                )
                critic2_params = jax.lax.cond(
                    use_target,
                    lambda _: target_critic_params[1],
                    lambda _: params["critic2"],
                    operand=None,
                )

                def pessimistic_action(_):
                    if self.twin_rollout_beam_width > 1:
                        return self._joint_beam_action(
                            critic1_params,
                            features,
                            critic2_params=critic2_params,
                        )[0]
                    return self._pessimistic_greedy_action(
                        critic1_params,
                        critic2_params,
                        features,
                        key=key,
                    )[0]

                if self.episodic_twin_head_exploration:
                    def sampled_head_action(_):
                        key1 = (
                            None
                            if key is None
                            else jax.random.fold_in(key, 3301)
                        )
                        key2 = (
                            None
                            if key is None
                            else jax.random.fold_in(key, 3302)
                        )
                        if self.twin_rollout_beam_width > 1:
                            action1 = self._joint_beam_action(
                                critic1_params,
                                features,
                            )[0]
                            action2 = self._joint_beam_action(
                                critic2_params,
                                features,
                            )[0]
                        else:
                            action1 = self._greedy_action(
                                critic1_params,
                                features,
                                key=key1,
                            )[0]
                            action2 = self._greedy_action(
                                critic2_params,
                                features,
                                key=key2,
                            )[0]
                        return select_episodic_twin_actions(
                            action1,
                            action2,
                            twin_head_indices,
                        )

                    return jax.lax.cond(
                        jnp.all(twin_head_indices < 0),
                        pessimistic_action,
                        sampled_head_action,
                        operand=None,
                    )
                return pessimistic_action(None)
            critic_params = jax.lax.cond(
                use_target,
                lambda _: target_critic_params,
                lambda _: params["critic"],
                operand=None,
            )
            if self.twin_rollout_beam_width > 1:
                return self._joint_beam_action(critic_params, features)[0]
            return self._greedy_action(critic_params, features, key=key)[0]

        return action_fn

    def act(self, observations: dict, step: int, eval_mode: bool):
        if not self._twin_action_path_enabled:
            # Flags off: pristine rollout, pristine RNG stream.
            return super().act(observations, step, eval_mode)

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
            twin_head_indices = np.full((batch_size,), -1, dtype=np.int32)
            if (
                self.episodic_twin_head_exploration
                and not eval_mode
                and step >= self.num_explore_steps
            ):
                if batch_size > self._episodic_twin_heads.shape[0]:
                    raise ValueError(
                        "Training action batch exceeds num_train_envs for "
                        "episodic twin-head exploration."
                    )
                missing = np.flatnonzero(
                    self._episodic_twin_heads[:batch_size] < 0
                )
                self._resample_episodic_twin_heads(missing.tolist())
                twin_head_indices = self._episodic_twin_heads[
                    :batch_size
                ].astype(np.int32, copy=True)
            action = self._greedy_action_impl(
                self.params,
                self.target_critic_params,
                obs_inputs,
                jnp.asarray(self.use_target_network_for_rollout),
                action_key,
                jnp.asarray(twin_head_indices, dtype=jnp.int32),
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

    # ------------------------------------------------------------------
    # Update graph
    # ------------------------------------------------------------------

    def _auxiliary_rl_obs_inputs(self, batch: dict):
        """Prepare the optional terminal-truncated auxiliary bootstrap state."""

        suffix = "_tp_aux"
        observation_keys = self.observation_space.keys()
        if self._has_cached_pixel_features(batch):
            observation_keys = tuple(
                key for key in observation_keys if key not in self._rgb_batch_keys
            )
        auxiliary_batch = {}
        missing = []
        for key in observation_keys:
            auxiliary_key = f"{key}{suffix}"
            if auxiliary_key not in batch:
                missing.append(auxiliary_key)
            else:
                auxiliary_batch[key] = batch[auxiliary_key]
        if self._has_cached_pixel_features(batch):
            cached_auxiliary_key = f"{self._cached_pixel_feature_key}{suffix}"
            if cached_auxiliary_key not in batch:
                missing.append(cached_auxiliary_key)
            else:
                auxiliary_batch[self._cached_pixel_feature_key] = batch[
                    cached_auxiliary_key
                ]
        if missing:
            raise KeyError(
                "auxiliary TD replay batch is missing: "
                + ", ".join(sorted(missing))
            )
        return self._prepare_rl_obs_inputs(auxiliary_batch)

    def _build_update_fn(self):
        if self.pessimistic_twin_critic:
            return self._build_pessimistic_twin_update_fn()
        # Beam width alone is a rollout-only maximizer: the update graph stays
        # exactly the pristine one.
        return super()._build_update_fn()

    def _build_pessimistic_twin_update_fn(self):
        """One clipped twin-C51 Bellman objective, with no policy loss."""

        optimizer = self.optimizer
        tau = self.critic_target_tau
        auxiliary_weight = float(self.auxiliary_td_loss_weight)
        use_auxiliary = auxiliary_weight > 0.0

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
            mc_returns,
            action_key,
            *auxiliary_args,
        ):
            if use_auxiliary:
                if len(auxiliary_args) != 5:
                    raise ValueError(
                        "auxiliary twin-C51 update requires next observations, "
                        "next actions, rewards, discounts, and bootstrap."
                    )
                (
                    auxiliary_next_obs_inputs,
                    auxiliary_next_actions,
                    auxiliary_rewards,
                    auxiliary_discounts,
                    auxiliary_bootstrap,
                ) = auxiliary_args
            elif auxiliary_args:
                raise ValueError(
                    "auxiliary arguments were provided with zero auxiliary weight."
                )

            obs_inputs, next_obs_inputs, action_key = (
                self._augment_update_obs_inputs(
                    obs_inputs,
                    next_obs_inputs,
                    action_key,
                )
            )
            if use_auxiliary and isinstance(auxiliary_next_obs_inputs, dict):
                auxiliary_next_obs_inputs = dict(auxiliary_next_obs_inputs)
                if "rgb" in auxiliary_next_obs_inputs:
                    auxiliary_next_obs_inputs["rgb"] = random_shift_rgb(
                        auxiliary_next_obs_inputs["rgb"],
                        jax.random.fold_in(action_key, 3401),
                    )

            def loss_fn(current_params):
                encoder_params = current_params.get("encoder", None)
                features = self._rl_features(encoder_params, obs_inputs)
                next_features = self._rl_features(
                    encoder_params,
                    next_obs_inputs,
                    stop_gradient=True,
                )
                auxiliary_next_features = None
                if use_auxiliary:
                    auxiliary_next_features = self._rl_features(
                        encoder_params,
                        auxiliary_next_obs_inputs,
                        stop_gradient=True,
                    )

                target1_params, target2_params = target_critic_params

                def horizon_target(
                    horizon_next_features,
                    horizon_next_actions,
                    horizon_rewards,
                    horizon_discounts,
                    horizon_bootstrap,
                    greedy_key,
                    force_fold,
                ):
                    greedy_action, _ = self._pessimistic_greedy_action(
                        current_params["critic"],
                        current_params["critic2"],
                        horizon_next_features,
                        key=greedy_key,
                    )
                    behavior_action = jnp.asarray(
                        horizon_next_actions,
                        dtype=jnp.float32,
                    ).reshape(
                        (
                            horizon_next_actions.shape[0],
                            self.action_sequence,
                            self.action_dim,
                        )
                    )
                    greedy_score, greedy_score1, greedy_score2 = (
                        self._pessimistic_score_action_sequence(
                            current_params["critic"],
                            current_params["critic2"],
                            horizon_next_features,
                            greedy_action,
                        )
                    )
                    behavior_score, behavior_score1, behavior_score2 = (
                        self._pessimistic_score_action_sequence(
                            current_params["critic"],
                            current_params["critic2"],
                            horizon_next_features,
                            behavior_action,
                        )
                    )
                    # Hard-wired ``td_target_action_source=critic_replay_max``:
                    # the research monolith validates that flag rather than
                    # branching on it inside this graph.
                    behavior_selected = behavior_score >= greedy_score
                    del force_fold
                    demo_behavior_forced = jnp.zeros_like(
                        demos,
                        dtype=jnp.bool_,
                    )
                    next_action = jnp.where(
                        behavior_selected[:, None, None],
                        behavior_action,
                        greedy_action,
                    )

                    target_logits1, _ = self._critic_logits_per_level(
                        target1_params,
                        horizon_next_features,
                        next_action,
                    )
                    target_logits2, _ = self._critic_logits_per_level(
                        target2_params,
                        horizon_next_features,
                        next_action,
                    )
                    target_probabilities1 = jax.nn.softmax(
                        target_logits1,
                        axis=-1,
                    )
                    target_probabilities2 = jax.nn.softmax(
                        target_logits2,
                        axis=-1,
                    )
                    target_q1 = jnp.sum(
                        target_probabilities1 * self.support,
                        axis=-1,
                    )[:, -1].mean(axis=-1)
                    target_q2 = jnp.sum(
                        target_probabilities2 * self.support,
                        axis=-1,
                    )[:, -1].mean(axis=-1)
                    target1_selected = target_q1 <= target_q2
                    target_probabilities = jnp.where(
                        target1_selected[:, None, None, None],
                        target_probabilities1,
                        target_probabilities2,
                    )
                    target_distribution = project_categorical(
                        target_probabilities,
                        horizon_rewards,
                        horizon_discounts,
                        horizon_bootstrap,
                        self.support,
                    )
                    bellman_q = jnp.sum(
                        target_distribution * self.support,
                        axis=-1,
                    )
                    # Hard-wired ``mc_lower_bound_target=true`` (same reason as
                    # the critic_replay_max hard-wiring above).
                    mc_distribution = project_categorical(
                        target_probabilities,
                        mc_returns,
                        jnp.zeros_like(horizon_discounts),
                        jnp.zeros_like(horizon_bootstrap),
                        self.support,
                    )
                    use_mc_mask = mc_returns[:, None, None] > bellman_q
                    target_distribution = jnp.where(
                        use_mc_mask[..., None],
                        mc_distribution,
                        target_distribution,
                    )
                    target_distribution = jax.lax.stop_gradient(
                        target_distribution
                    )
                    return target_distribution, (
                        use_mc_mask,
                        target1_selected,
                        behavior_selected,
                        demo_behavior_forced,
                        behavior_score,
                        greedy_score,
                        behavior_score1,
                        behavior_score2,
                        greedy_score1,
                        greedy_score2,
                    )

                target_distribution, horizon_diagnostics = horizon_target(
                    next_features,
                    next_actions,
                    rewards,
                    discounts,
                    bootstrap,
                    action_key,
                    3201,
                )
                (
                    use_mc_mask,
                    target1_selected,
                    behavior_selected,
                    demo_behavior_forced,
                    behavior_score,
                    greedy_score,
                    behavior_score1,
                    behavior_score2,
                    greedy_score1,
                    greedy_score2,
                ) = horizon_diagnostics

                chosen_logits1, _ = self._critic_logits_per_level(
                    current_params["critic"],
                    features,
                    actions,
                )
                chosen_logits2, _ = self._critic_logits_per_level(
                    current_params["critic2"],
                    features,
                    actions,
                )
                chosen_log_probabilities1 = jax.nn.log_softmax(
                    chosen_logits1,
                    axis=-1,
                )
                chosen_log_probabilities2 = jax.nn.log_softmax(
                    chosen_logits2,
                    axis=-1,
                )
                one_step_per_sample1 = -jnp.sum(
                    target_distribution * chosen_log_probabilities1,
                    axis=-1,
                ).mean(axis=(1, 2))
                one_step_per_sample2 = -jnp.sum(
                    target_distribution * chosen_log_probabilities2,
                    axis=-1,
                ).mean(axis=(1, 2))
                auxiliary_per_sample1 = jnp.zeros_like(one_step_per_sample1)
                auxiliary_per_sample2 = jnp.zeros_like(one_step_per_sample2)
                auxiliary_target_distribution = target_distribution
                auxiliary_diagnostics = horizon_diagnostics
                if use_auxiliary:
                    (
                        auxiliary_target_distribution,
                        auxiliary_diagnostics,
                    ) = horizon_target(
                        auxiliary_next_features,
                        auxiliary_next_actions,
                        auxiliary_rewards,
                        auxiliary_discounts,
                        auxiliary_bootstrap,
                        jax.random.fold_in(action_key, 3403),
                        3204,
                    )
                    auxiliary_per_sample1 = -jnp.sum(
                        auxiliary_target_distribution
                        * chosen_log_probabilities1,
                        axis=-1,
                    ).mean(axis=(1, 2))
                    auxiliary_per_sample2 = -jnp.sum(
                        auxiliary_target_distribution
                        * chosen_log_probabilities2,
                        axis=-1,
                    ).mean(axis=(1, 2))
                loss_normalizer = 1.0 + auxiliary_weight
                per_sample1 = (
                    one_step_per_sample1
                    + auxiliary_weight * auxiliary_per_sample1
                ) / loss_normalizer
                per_sample2 = (
                    one_step_per_sample2
                    + auxiliary_weight * auxiliary_per_sample2
                ) / loss_normalizer
                per_sample = 0.5 * (per_sample1 + per_sample2)
                critic1_loss = self.critic_lambda * jnp.mean(
                    per_sample1 * loss_weights
                )
                critic2_loss = self.critic_lambda * jnp.mean(
                    per_sample2 * loss_weights
                )
                critic_loss = 0.5 * (critic1_loss + critic2_loss)
                one_step_critic_loss = 0.5 * self.critic_lambda * (
                    jnp.mean(one_step_per_sample1 * loss_weights)
                    + jnp.mean(one_step_per_sample2 * loss_weights)
                )
                auxiliary_critic_loss = 0.5 * self.critic_lambda * (
                    jnp.mean(auxiliary_per_sample1 * loss_weights)
                    + jnp.mean(auxiliary_per_sample2 * loss_weights)
                )

                chosen_probabilities1 = jax.nn.softmax(
                    chosen_logits1,
                    axis=-1,
                )
                chosen_probabilities2 = jax.nn.softmax(
                    chosen_logits2,
                    axis=-1,
                )
                chosen_q1 = jnp.sum(
                    chosen_probabilities1 * self.support,
                    axis=-1,
                )
                chosen_q2 = jnp.sum(
                    chosen_probabilities2 * self.support,
                    axis=-1,
                )
                entropy1 = -jnp.sum(
                    chosen_probabilities1
                    * jnp.log(jnp.maximum(chosen_probabilities1, 1e-9)),
                    axis=-1,
                ).mean()
                entropy2 = -jnp.sum(
                    chosen_probabilities2
                    * jnp.log(jnp.maximum(chosen_probabilities2, 1e-9)),
                    axis=-1,
                ).mean()
                target_entropy = -jnp.sum(
                    target_distribution
                    * jnp.log(jnp.maximum(target_distribution, 1e-9)),
                    axis=-1,
                ).mean()
                auxiliary_target_entropy = -jnp.sum(
                    auxiliary_target_distribution
                    * jnp.log(
                        jnp.maximum(auxiliary_target_distribution, 1e-9)
                    ),
                    axis=-1,
                ).mean()
                target_entropy = (
                    target_entropy
                    + auxiliary_weight * auxiliary_target_entropy
                ) / (1.0 + auxiliary_weight)
                (
                    auxiliary_use_mc_mask,
                    auxiliary_target1_selected,
                    auxiliary_behavior_selected,
                    auxiliary_demo_behavior_forced,
                    auxiliary_behavior_score,
                    auxiliary_greedy_score,
                    _,
                    _,
                    _,
                    _,
                ) = auxiliary_diagnostics
                return critic_loss, (
                    per_sample,
                    critic1_loss,
                    critic2_loss,
                    0.5 * (entropy1 + entropy2),
                    target_entropy,
                    jnp.mean(use_mc_mask.astype(jnp.float32)),
                    jnp.mean(jnp.abs(chosen_q1 - chosen_q2)),
                    jnp.mean(target1_selected.astype(jnp.float32)),
                    behavior_selected,
                    demo_behavior_forced,
                    behavior_score,
                    greedy_score,
                    behavior_score1,
                    behavior_score2,
                    greedy_score1,
                    greedy_score2,
                    one_step_critic_loss,
                    auxiliary_critic_loss,
                    auxiliary_use_mc_mask,
                    auxiliary_target1_selected,
                    auxiliary_behavior_selected,
                    auxiliary_demo_behavior_forced,
                    auxiliary_behavior_score,
                    auxiliary_greedy_score,
                )

            (critic_loss, aux), grads = jax.value_and_grad(
                loss_fn,
                has_aux=True,
            )(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = self.optax.apply_updates(params, updates)
            target1_params, target2_params = target_critic_params
            target1_params = jax.tree.map(
                lambda target, online: (1.0 - tau) * target + tau * online,
                target1_params,
                params["critic"],
            )
            target2_params = jax.tree.map(
                lambda target, online: (1.0 - tau) * target + tau * online,
                target2_params,
                params["critic2"],
            )
            (
                per_sample,
                critic1_loss,
                critic2_loss,
                entropy,
                target_entropy,
                mc_lower_bound_fraction,
                twin_q_disagreement,
                target1_fraction,
                behavior_selected,
                demo_behavior_forced,
                behavior_score,
                greedy_score,
                behavior_score1,
                behavior_score2,
                greedy_score1,
                greedy_score2,
                one_step_critic_loss,
                auxiliary_critic_loss,
                auxiliary_use_mc_mask,
                auxiliary_target1_selected,
                auxiliary_behavior_selected,
                auxiliary_demo_behavior_forced,
                auxiliary_behavior_score,
                auxiliary_greedy_score,
            ) = aux
            priority = jnp.sqrt(per_sample + 1e-10)
            priority = priority / jnp.maximum(jnp.max(priority), 1e-10)
            metrics = {
                "critic_loss": critic_loss,
                "critic1_loss": critic1_loss,
                "critic2_loss": critic2_loss,
                "one_step_critic_loss": one_step_critic_loss,
                "auxiliary_critic_loss": auxiliary_critic_loss,
                "auxiliary_td_loss_weight": jnp.asarray(
                    auxiliary_weight, dtype=jnp.float32
                ),
                "entropy": entropy,
                "target_entropy": target_entropy,
                "loss_coeff": jnp.mean(loss_weights),
                "mc_lower_bound_fraction": mc_lower_bound_fraction,
                "mc_return_mean": jnp.mean(mc_returns),
                "behavior_candidate_fraction": jnp.mean(
                    behavior_selected.astype(jnp.float32)
                ),
                "demo_behavior_force_fraction": jnp.mean(
                    demo_behavior_forced.astype(jnp.float32)
                ),
                "demo_behavior_force_probability": jnp.asarray(
                    0.0,
                    dtype=jnp.float32,
                ),
                "behavior_candidate_score": jnp.mean(behavior_score),
                "greedy_candidate_score": jnp.mean(greedy_score),
                "behavior_minus_greedy_q": jnp.mean(
                    behavior_score - greedy_score
                ),
                "twin_q_disagreement": twin_q_disagreement,
                "twin_target1_fraction": target1_fraction,
                "behavior_critic_gap": jnp.mean(
                    jnp.abs(behavior_score1 - behavior_score2)
                ),
                "greedy_critic_gap": jnp.mean(
                    jnp.abs(greedy_score1 - greedy_score2)
                ),
                "auxiliary_mc_lower_bound_fraction": jnp.mean(
                    auxiliary_use_mc_mask.astype(jnp.float32)
                ),
                "auxiliary_twin_target1_fraction": jnp.mean(
                    auxiliary_target1_selected.astype(jnp.float32)
                ),
                "auxiliary_behavior_candidate_fraction": jnp.mean(
                    auxiliary_behavior_selected.astype(jnp.float32)
                ),
                "auxiliary_demo_behavior_force_fraction": jnp.mean(
                    auxiliary_demo_behavior_forced.astype(jnp.float32)
                ),
                "auxiliary_behavior_candidate_score": jnp.mean(
                    auxiliary_behavior_score
                ),
                "auxiliary_greedy_candidate_score": jnp.mean(
                    auxiliary_greedy_score
                ),
                "auxiliary_behavior_minus_greedy_q": jnp.mean(
                    auxiliary_behavior_score - auxiliary_greedy_score
                ),
            }
            return (
                params,
                (target1_params, target2_params),
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
        if not self.pessimistic_twin_critic:
            # Flags off (including beam-only): the pristine update loop.
            return super().update(replay_iter, step, replay_buffer)

        update_steps = 1 if step == 0 else self.num_update_steps
        metrics = {}
        for _ in range(update_steps):
            batch = next(replay_iter)
            obs_inputs = self._prepare_rl_obs_inputs(batch)
            next_obs_inputs = self._next_rl_obs_inputs(batch)
            actions = self._as_jax_array(batch["action"], self.jnp.float32).reshape(
                (batch["action"].shape[0], -1)
            )
            if "action_tp1" not in batch:
                raise KeyError(
                    "pessimistic_twin_critic requires "
                    "replay.include_next_action=true and an action_tp1 batch "
                    "element."
                )
            next_action_values = batch["action_tp1"]
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
            mc_returns = self._as_jax_array(
                batch.get("mc_return", np.zeros_like(batch["reward"])),
                self.jnp.float32,
            ).reshape(-1)
            auxiliary_args = ()
            if self.auxiliary_td_loss_weight > 0.0:
                required_auxiliary = (
                    "action_tp_aux",
                    "reward_aux",
                    "discount_aux",
                    "terminal_aux",
                )
                missing_auxiliary = [
                    name for name in required_auxiliary if name not in batch
                ]
                if missing_auxiliary:
                    raise KeyError(
                        "auxiliary-horizon targets require "
                        "replay.auxiliary_nstep; missing: "
                        + ", ".join(missing_auxiliary)
                    )
                auxiliary_next_obs_inputs = self._auxiliary_rl_obs_inputs(batch)
                auxiliary_action_values = batch["action_tp_aux"]
                auxiliary_next_actions = self._as_jax_array(
                    auxiliary_action_values,
                    self.jnp.float32,
                ).reshape((auxiliary_action_values.shape[0], -1))
                auxiliary_rewards = self._as_jax_array(
                    batch["reward_aux"], self.jnp.float32
                ).reshape(-1)
                auxiliary_discounts = self._as_jax_array(
                    batch["discount_aux"], self.jnp.float32
                ).reshape(-1)
                auxiliary_terminal = self._as_jax_array(
                    batch["terminal_aux"], self.jnp.float32
                ).reshape(-1)
                auxiliary_bootstrap = (
                    jnp.ones_like(auxiliary_terminal)
                    if self.always_bootstrap
                    else 1.0 - auxiliary_terminal
                )
                auxiliary_args = (
                    auxiliary_next_obs_inputs,
                    auxiliary_next_actions,
                    auxiliary_rewards,
                    auxiliary_discounts,
                    auxiliary_bootstrap,
                )
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
                mc_returns,
                self._next_action_key(),
                *auxiliary_args,
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

    # ------------------------------------------------------------------
    # Episode-head bookkeeping
    # ------------------------------------------------------------------

    def _resample_episodic_twin_heads(self, agent_indices: list[int]) -> None:
        if not self.episodic_twin_head_exploration or not agent_indices:
            return
        valid = np.asarray(
            [
                index
                for index in agent_indices
                if 0 <= index < self.num_train_envs
            ],
            dtype=np.int32,
        )
        if valid.size == 0:
            return
        sampled = self._episodic_twin_head_rng.integers(
            0,
            2,
            size=valid.size,
            dtype=np.int8,
        )
        self._episodic_twin_heads[valid] = sampled
        self._episodic_twin_head_assignments += np.bincount(
            sampled,
            minlength=2,
        ).astype(np.int64)

    def reset(self, step: int, agents_to_reset: list[int]):
        self._resample_episodic_twin_heads(agents_to_reset)
        super().reset(step, agents_to_reset)

    def checkpoint_state_dict(self) -> dict:
        state = super().checkpoint_state_dict()
        state.update(self._exploration_checkpoint_state())
        return state

    def load_checkpoint_state_dict(self, state_dict: dict):
        super().load_checkpoint_state_dict(state_dict)
        self._load_exploration_checkpoint_state(state_dict)

    def _exploration_checkpoint_state(self) -> dict:
        """NumPy episode-head RNG stream (and nothing else).

        Without this a resume restarts the generator from the seed while the
        original process had advanced it, so the head assignment sequence
        diverges from an uninterrupted run.
        """

        generator = getattr(self, "_episodic_twin_head_rng", None)
        if generator is None:
            return {}
        return {
            "episodic_twin_head_rng_state": generator.bit_generator.state,
        }

    def _load_exploration_checkpoint_state(self, state_dict: dict) -> None:
        stored = state_dict.get("episodic_twin_head_rng_state")
        generator = getattr(self, "_episodic_twin_head_rng", None)
        if stored is not None and generator is not None:
            generator.bit_generator.state = stored
        # Workspace snapshots do not carry environment state. A resumed
        # online phase therefore starts fresh episodes and samples fresh
        # episode heads on its first action rather than leaking old heads.
        episodic_heads = getattr(self, "_episodic_twin_heads", None)
        if episodic_heads is not None:
            episodic_heads.fill(-1)

    def rollout_diagnostics(self) -> dict[str, float]:
        head_assignments = getattr(
            self,
            "_episodic_twin_head_assignments",
            np.zeros((2,), dtype=np.int64),
        )
        assignments = int(head_assignments.sum())
        return {
            "episodic_twin_head_assignments": float(assignments),
            "episodic_twin_head0_rate": (
                float(head_assignments[0] / assignments) if assignments else 0.0
            ),
            "episodic_twin_head1_rate": (
                float(head_assignments[1] / assignments) if assignments else 0.0
            ),
        }
