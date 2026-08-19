"""Structured / bin-space exploration variant of the official CQN-AS.

Phase-R2 line 1 (``structured-exploration``) of ``CQN_REFACTOR_PLAN.md``.
Everything here is *rollout-path* research: the critic architecture, the
Bellman/BC objective and the update graph are the pristine official ones.
The only update-path change is the Stage-160 low-dim observation mask, which
is an augmentation applied to the update batch (never at ``act()``).

Mechanisms carried over from the research monolith
(``robobase/method/cqn_as_research.py``, base ``ff9dfbf``):

* ``structured_exploration_*`` -- post-ensemble single-coordinate cell-width
  intervention held coherent for ``horizon`` decisions.  Emits the
  ``structured_explore*`` replay extras.
* ``bin_flip_*`` -- open-loop (``temporal_ensemble=false``) coherent bin-space
  sibling flip applied to a whole freshly refreshed plan.
* ``bin_explore_*`` -- Stage-153 hierarchical epsilon-bin exploration that is
  compatible with the closed-loop temporal ensemble (per-level probabilities,
  optional annealing schedule, persistence across N fresh plans).  Emits the
  ``explored`` replay extra.
* ``low_dim_mask_*`` -- Stage-160 update-time low-dim observation dropout.
* ``post_ensemble_l1/l2_flip_*`` -- persistent single-dimension post-ensemble
  bin flips at one or two C2F granularities (train only).
* ``random_levels_from`` / ``level_override_mode`` /
  ``post_ensemble_random_keep_levels`` / ``post_ensemble_fixed_leaf`` --
  eval-only C2F resolution probes (spec-only knobs; no ``cqn_as.yaml`` entry
  for the last two in the research tree either, they are set by
  ``scripts/eval_cqn_as_snapshot_sweep.py``).

With every flag at its default the class is byte-for-byte the pristine
``CQNAS`` code path: no extra JAX RNG split, no extra parameter, no metric
change.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig

from robobase import utils
from robobase.method.cqn import zoom_in
from robobase.method.cqn_as import (
    CQNAS,
    CQNASpec,
    cqn_as_spec_from_cfg,
    random_shift_rgb,
)
from robobase.method.rl_common import RLModelSpec


@dataclass(frozen=True)
class CQNASStructuredExploreSpec(CQNASpec):
    """Official CQN-AS spec plus the structured-exploration knobs."""

    structured_exploration_prob: float
    structured_exploration_level: int
    structured_exploration_horizon: int
    bin_flip_prob: float
    bin_flip_level: int | None
    bin_explore_probs: tuple[float, ...] | None
    bin_explore_schedule: str | None
    bin_explore_persist_plans: int | None
    low_dim_mask_prob: float
    low_dim_mask_keep_last: int
    random_levels_from: int | None
    level_override_mode: str
    post_ensemble_random_keep_levels: int | None
    post_ensemble_fixed_leaf: int | None
    post_ensemble_l1_flip_prob: float
    post_ensemble_l2_flip_prob: float
    post_ensemble_l1_flip_horizon: int


def cqn_as_structured_explore_spec_from_cfg(
    cfg: DictConfig,
) -> CQNASStructuredExploreSpec:
    method = cfg.method
    base = cqn_as_spec_from_cfg(cfg)
    base_values = {field.name: getattr(base, field.name) for field in fields(CQNASpec)}
    return CQNASStructuredExploreSpec(
        **base_values,
        structured_exploration_prob=float(
            method.get("structured_exploration_prob", 0.0)
        ),
        structured_exploration_level=int(
            method.get("structured_exploration_level", 1)
        ),
        structured_exploration_horizon=int(
            method.get("structured_exploration_horizon", 1)
        ),
        bin_flip_prob=float(method.get("bin_flip_prob", 0.0)),
        bin_flip_level=(
            None
            if method.get("bin_flip_level", None) is None
            else int(method.get("bin_flip_level"))
        ),
        bin_explore_probs=(
            None
            if method.get("bin_explore_probs", None) is None
            else tuple(float(p) for p in method.get("bin_explore_probs"))
        ),
        bin_explore_schedule=(
            None
            if method.get("bin_explore_schedule", None) is None
            else str(method.get("bin_explore_schedule"))
        ),
        bin_explore_persist_plans=(
            None
            if method.get("bin_explore_persist_plans", None) is None
            else int(method.get("bin_explore_persist_plans"))
        ),
        low_dim_mask_prob=float(method.get("low_dim_mask_prob", 0.0)),
        low_dim_mask_keep_last=int(method.get("low_dim_mask_keep_last", 0)),
        random_levels_from=(
            None
            if method.get("random_levels_from", None) is None
            else int(method.random_levels_from)
        ),
        level_override_mode=str(method.get("level_override_mode", "random")),
        post_ensemble_random_keep_levels=(
            None
            if method.get("post_ensemble_random_keep_levels", None) is None
            else int(method.post_ensemble_random_keep_levels)
        ),
        post_ensemble_fixed_leaf=(
            None
            if method.get("post_ensemble_fixed_leaf", None) is None
            else int(method.post_ensemble_fixed_leaf)
        ),
        post_ensemble_l1_flip_prob=float(
            method.get("post_ensemble_l1_flip_prob", 0.0)
        ),
        post_ensemble_l2_flip_prob=float(
            method.get("post_ensemble_l2_flip_prob", 0.0)
        ),
        post_ensemble_l1_flip_horizon=int(
            method.get("post_ensemble_l1_flip_horizon", 4)
        ),
    )


class CQNASStructuredExplore(CQNAS):
    """CQN-AS with the structured / bin-space exploration research line."""

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
        structured_exploration_prob: float = 0.0,
        structured_exploration_level: int = 1,
        structured_exploration_horizon: int = 1,
        bin_flip_prob: float = 0.0,
        bin_flip_level: int | None = None,
        bin_explore_probs: tuple[float, ...] | None = None,
        bin_explore_schedule: str | None = None,
        bin_explore_persist_plans: int | None = None,
        low_dim_mask_prob: float = 0.0,
        low_dim_mask_keep_last: int = 0,
        random_levels_from: int | None = None,
        level_override_mode: str = "random",
        post_ensemble_random_keep_levels: int | None = None,
        post_ensemble_fixed_leaf: int | None = None,
        post_ensemble_l1_flip_prob: float = 0.0,
        post_ensemble_l2_flip_prob: float = 0.0,
        post_ensemble_l1_flip_horizon: int = 4,
    ):
        # ------------------------------------------------------------------
        # Validate + store the line's flags BEFORE the pristine constructor.
        # ``_build_update_fn`` / ``_build_greedy_action_fn`` capture ``self``
        # and read these attributes at trace time, so they must already be
        # their final values by the time the pristine ``__init__`` returns.
        # ------------------------------------------------------------------
        if not 0.0 <= structured_exploration_prob <= 1.0:
            raise ValueError("structured_exploration_prob must be in [0, 1].")
        if not 0 <= structured_exploration_level < levels:
            raise ValueError("structured_exploration_level must be in [0, levels).")
        if structured_exploration_horizon < 1:
            raise ValueError("structured_exploration_horizon must be at least 1.")
        self.structured_exploration_prob = float(structured_exploration_prob)
        self.structured_exploration_level = int(structured_exploration_level)
        self.structured_exploration_horizon = int(structured_exploration_horizon)

        if not 0.0 <= bin_flip_prob <= 1.0:
            raise ValueError("bin_flip_prob must be in [0, 1].")
        if bin_flip_prob > 0.0 and temporal_ensemble:
            raise ValueError(
                "bin_flip_prob > 0 requires method.temporal_ensemble=false: "
                "the flip is defined on open-loop chunk execution so the "
                "flipped plan is executed verbatim (cqn-flow.md 32.2)."
            )
        if bin_flip_level is not None and not 0 <= bin_flip_level < levels:
            raise ValueError("bin_flip_level must be in [0, levels).")
        self.bin_flip_prob = float(bin_flip_prob)
        self.bin_flip_level = (
            None if bin_flip_level is None else int(bin_flip_level)
        )

        # Stage-153 hierarchical epsilon-bin exploration (ensemble-safe).
        if bin_explore_probs is not None:
            probs = tuple(float(p) for p in bin_explore_probs)
            if len(probs) != levels:
                raise ValueError(
                    "bin_explore_probs must list one probability per level."
                )
            if any(not 0.0 <= p <= 1.0 for p in probs):
                raise ValueError("bin_explore_probs entries must be in [0, 1].")
            if bin_flip_prob > 0.0:
                raise ValueError(
                    "bin_explore_probs and bin_flip_prob are mutually "
                    "exclusive exploration mechanisms."
                )
            self.bin_explore_probs = probs
        else:
            self.bin_explore_probs = None
        # Stage-162: optional schedule multiplying every level's activation
        # probability (e.g. "linear(1.0,0.0,100000)" anneals exploration
        # away as TD takes over). None keeps the static probabilities.
        if bin_explore_schedule is not None and bin_explore_probs is None:
            raise ValueError("bin_explore_schedule requires bin_explore_probs.")
        self.bin_explore_schedule = (
            None if bin_explore_schedule is None else str(bin_explore_schedule)
        )
        self._bin_explore_scale = 1.0
        if bin_explore_persist_plans is not None and bin_explore_persist_plans < 1:
            raise ValueError("bin_explore_persist_plans must be >= 1.")
        self._bin_explore_persist_plans_arg = (
            None
            if bin_explore_persist_plans is None
            else int(bin_explore_persist_plans)
        )

        # Stage-160 random low-dim observation mask: during updates only,
        # zero every low-dim frame except its last ``keep_last`` dims with
        # this per-sample probability (act() is never masked).
        if not 0.0 <= low_dim_mask_prob <= 1.0:
            raise ValueError("low_dim_mask_prob must be in [0, 1].")
        if low_dim_mask_keep_last < 0:
            raise ValueError("low_dim_mask_keep_last must be non-negative.")
        self.low_dim_mask_prob = float(low_dim_mask_prob)
        self.low_dim_mask_keep_last = int(low_dim_mask_keep_last)
        self._low_dim_frame_dim = None
        if "low_dim_state" in observation_space.spaces:
            self._low_dim_frame_dim = int(
                observation_space["low_dim_state"].shape[-1]
            )
        if self.low_dim_mask_prob > 0.0:
            if self._low_dim_frame_dim is None:
                raise ValueError(
                    "low_dim_mask_prob requires a low_dim_state observation."
                )
            if self.low_dim_mask_keep_last >= self._low_dim_frame_dim:
                raise ValueError(
                    "low_dim_mask_keep_last must be smaller than the "
                    "low-dim frame size."
                )

        self.random_levels_from = random_levels_from
        self.level_override_mode = str(level_override_mode)
        self.post_ensemble_random_keep_levels = (
            None
            if post_ensemble_random_keep_levels is None
            else int(post_ensemble_random_keep_levels)
        )
        self.post_ensemble_fixed_leaf = (
            None
            if post_ensemble_fixed_leaf is None
            else int(post_ensemble_fixed_leaf)
        )
        self.post_ensemble_l1_flip_prob = float(post_ensemble_l1_flip_prob)
        self.post_ensemble_l2_flip_prob = float(post_ensemble_l2_flip_prob)
        self.post_ensemble_l1_flip_horizon = int(post_ensemble_l1_flip_horizon)

        super().__init__(
            critic_lr=critic_lr,
            num_train_steps=num_train_steps,
            num_explore_steps=num_explore_steps,
            critic_target_tau=critic_target_tau,
            weight_decay=weight_decay,
            levels=levels,
            bins=bins,
            atoms=atoms,
            v_min=v_min,
            v_max=v_max,
            critic_lambda=critic_lambda,
            centralized_critic=centralized_critic,
            use_dueling=use_dueling,
            always_bootstrap=always_bootstrap,
            stddev_schedule=stddev_schedule,
            bc_lambda=bc_lambda,
            bc_margin=bc_margin,
            use_target_network_for_rollout=use_target_network_for_rollout,
            num_update_steps=num_update_steps,
            gru_layers=gru_layers,
            temporal_ensemble=temporal_ensemble,
            temporal_ensemble_replan_interval=temporal_ensemble_replan_interval,
            temporal_ensemble_gain=temporal_ensemble_gain,
            tie_break_delta=tie_break_delta,
            model=model,
            observation_space=observation_space,
            action_space=action_space,
            num_train_envs=num_train_envs,
            num_eval_envs=num_eval_envs,
            replay_alpha=replay_alpha,
            replay_beta=replay_beta,
            frame_stack_on_channel=frame_stack_on_channel,
            intrinsic_reward_module=intrinsic_reward_module,
            critic_grad_clip=critic_grad_clip,
            jit=jit,
            platform=platform,
            seed=seed,
            update_block_every_steps=update_block_every_steps,
        )

        # ------------------------------------------------------------------
        # Runtime exploration state (needs ``self.action_sequence``).
        # ------------------------------------------------------------------
        # How many consecutive fresh plans a fired flip is re-applied to.
        # Default (None) = action_sequence, matching per-step replanning; with
        # sparser replan intervals set this so that persist_plans x
        # replan_interval keeps the intended window length in env steps.
        self.bin_explore_persist_plans = (
            int(self.action_sequence)
            if self._bin_explore_persist_plans_arg is None
            else int(self._bin_explore_persist_plans_arg)
        )

        self._bin_flip_remaining = np.zeros((int(num_train_envs),), dtype=np.int32)
        self._bin_flip_dimension = np.full(
            (int(num_train_envs),), -1, dtype=np.int16
        )
        self._bin_flip_delta_sequence = np.zeros(
            (int(num_train_envs), self.action_sequence), dtype=np.float32
        )
        self._bin_flip_rng = np.random.default_rng(int(seed) + 151)

        self._bin_explore_remaining = np.zeros(
            (int(num_train_envs),), dtype=np.int32
        )
        self._bin_explore_dimension = np.full(
            (int(num_train_envs),), -1, dtype=np.int16
        )
        self._bin_explore_level = np.full(
            (int(num_train_envs),), -1, dtype=np.int16
        )
        self._bin_explore_sibling = np.full(
            (int(num_train_envs),), -1, dtype=np.int16
        )
        self._bin_explore_rng = np.random.default_rng(int(seed) + 153)
        self._bin_explore_fired_total = 0
        self._bin_explore_applied_total = 0
        self._bin_explore_calls_total = 0
        self._bin_explore_mask_rows_total = 0
        self._act_train_calls_total = 0
        self._bin_explored_exec_remaining = np.zeros(
            (int(num_train_envs),), dtype=np.int32
        )
        self._last_bin_explore_applied = np.zeros(
            (int(num_train_envs),), dtype=np.bool_
        )
        self._last_bin_explored = np.zeros(
            (int(num_train_envs),), dtype=np.bool_
        )

        self._last_structured_exploration_mask = np.zeros(
            (int(num_train_envs),), dtype=np.bool_
        )
        self._last_structured_exploration_start = np.zeros(
            (int(num_train_envs),), dtype=np.bool_
        )
        self._last_structured_exploration_dimension = np.full(
            (int(num_train_envs),), -1, dtype=np.int16
        )
        self._last_structured_exploration_delta = np.zeros(
            (int(num_train_envs),), dtype=np.float32
        )
        self._last_structured_exploration_assignment_prob = np.ones(
            (int(num_train_envs),), dtype=np.float32
        )
        self._structured_exploration_remaining = np.zeros(
            (int(num_train_envs),), dtype=np.int32
        )
        self._structured_exploration_dimension = np.full(
            (int(num_train_envs),), -1, dtype=np.int16
        )
        self._structured_exploration_direction = np.zeros(
            (int(num_train_envs),), dtype=np.float32
        )
        self._structured_exploration_eligible = 0
        self._structured_exploration_applied = 0
        self._structured_exploration_starts = 0

    # ------------------------------------------------------------------
    # Action selection (pristine ``_greedy_action`` + eval-only level probe)
    # ------------------------------------------------------------------

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
            # Diagnostic (eval only, default off): replace the critic's bin
            # choice with a uniform draw (or the parent cell's centre) at this
            # level and every level below it.  This measures what the critic's
            # per-level ordering is worth on task success.
            if (
                self.random_levels_from is not None
                and level >= self.random_levels_from
            ):
                if self.level_override_mode == "middle":
                    # Deterministic centre bin == what an agent with this
                    # level deleted would emit (the parent cell's centre).
                    index = jnp.full_like(index, self.bins // 2)
                else:
                    if level_key is None:
                        raise ValueError(
                            "random_levels_from needs an rng key; it is a "
                            "diagnostic for eval-time action selection."
                        )
                    index = jax.random.randint(
                        level_key,
                        index.shape,
                        minval=0,
                        maxval=self.bins,
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
    # Update-batch augmentation (pristine random shift + Stage-160 mask)
    # ------------------------------------------------------------------

    def _mask_low_dim(self, low_dim: jax.Array, key: jax.Array) -> jax.Array:
        """Zero all but the last ``keep_last`` dims of each low-dim frame.

        Applied per sample with probability ``low_dim_mask_prob`` on update
        batches only, so the rollout policy always sees full observations.
        """

        batch = low_dim.shape[0]
        frame = self._low_dim_frame_dim
        frames = low_dim.reshape((batch, -1, frame))
        keep = self.low_dim_mask_keep_last
        keep_mask = jnp.concatenate(
            [
                jnp.zeros((frame - keep,), dtype=frames.dtype),
                jnp.ones((keep,), dtype=frames.dtype),
            ]
        )
        drop = jax.random.bernoulli(key, self.low_dim_mask_prob, (batch, 1, 1))
        masked = jnp.where(drop, frames * keep_mask, frames)
        return masked.reshape(low_dim.shape)

    def _augment_update_obs_inputs(self, obs_inputs, next_obs_inputs, key):
        if not isinstance(obs_inputs, dict):
            return obs_inputs, next_obs_inputs, key
        has_rgb = "rgb" in obs_inputs
        has_mask = (
            getattr(self, "low_dim_mask_prob", 0.0) > 0.0
            and "low_dim" in obs_inputs
        )
        if not has_rgb and not has_mask:
            return obs_inputs, next_obs_inputs, key
        obs_inputs = dict(obs_inputs)
        next_obs_inputs = dict(next_obs_inputs)
        if has_rgb:
            augment_key, next_augment_key, key = jax.random.split(key, 3)
            obs_inputs["rgb"] = random_shift_rgb(obs_inputs["rgb"], augment_key)
            next_obs_inputs["rgb"] = random_shift_rgb(
                next_obs_inputs["rgb"], next_augment_key
            )
        if has_mask:
            mask_key, next_mask_key, key = jax.random.split(key, 3)
            obs_inputs["low_dim"] = self._mask_low_dim(
                obs_inputs["low_dim"], mask_key
            )
            next_obs_inputs["low_dim"] = self._mask_low_dim(
                next_obs_inputs["low_dim"], next_mask_key
            )
        return obs_inputs, next_obs_inputs, key

    # ------------------------------------------------------------------
    # Structured (continuous cell-width) exploration
    # ------------------------------------------------------------------

    def _structured_exploration_action(self, executed_action, key):
        """Perturb one executed coordinate by one local C2F cell width.

        This runs after temporal ensembling, so replay stores exactly the
        action that was intervened on. Only one coordinate changes per selected
        environment step; the BC plan and all other coordinates stay intact.
        """

        probability = float(getattr(self, "structured_exploration_prob", 0.0))
        level = int(getattr(self, "structured_exploration_level", 1))
        action = jnp.asarray(executed_action, dtype=jnp.float32)
        mask_key, dimension_key, direction_key = jax.random.split(key, 3)
        explore_mask = (
            jax.random.uniform(mask_key, (action.shape[0],)) < probability
        )
        dimensions = jax.random.randint(
            dimension_key,
            (action.shape[0],),
            minval=0,
            maxval=self.action_dim,
        )
        directions = jnp.where(
            jax.random.bernoulli(direction_key, shape=(action.shape[0],)),
            1.0,
            -1.0,
        )
        cell_width = (self._step_action_high - self._step_action_low) / float(
            self.bins ** (level + 1)
        )
        row = jnp.arange(action.shape[0])
        candidate = action.at[row, dimensions].add(
            directions * cell_width[dimensions]
        )
        candidate = jnp.clip(
            candidate,
            self._step_action_low,
            self._step_action_high,
        )
        return jnp.where(explore_mask[:, None], candidate, action), explore_mask

    def _coherent_structured_exploration_action(self, executed_action, key):
        """Apply a randomized local intervention for one or more decisions.

        A new assignment is sampled only while an environment is inactive.
        Once started, its coordinate and direction are held fixed for
        ``structured_exploration_horizon`` calls.  This makes the perturbation
        survive action smoothing while keeping horizon=1 equivalent to the
        original independent one-step intervention.
        """

        action = jnp.asarray(executed_action, dtype=jnp.float32)
        batch_size = int(action.shape[0])
        if self._structured_exploration_remaining.shape != (batch_size,):
            self._structured_exploration_remaining = np.zeros(
                (batch_size,), dtype=np.int32
            )
            self._structured_exploration_dimension = np.full(
                (batch_size,), -1, dtype=np.int16
            )
            self._structured_exploration_direction = np.zeros(
                (batch_size,), dtype=np.float32
            )

        probability = float(self.structured_exploration_prob)
        mask_key, dimension_key, direction_key = jax.random.split(key, 3)
        start_draw = np.asarray(
            jax.device_get(
                jax.random.uniform(mask_key, (batch_size,)) < probability
            ),
            dtype=np.bool_,
        )
        sampled_dimensions = np.asarray(
            jax.device_get(
                jax.random.randint(
                    dimension_key,
                    (batch_size,),
                    minval=0,
                    maxval=self.action_dim,
                )
            ),
            dtype=np.int16,
        )
        sampled_directions = np.asarray(
            jax.device_get(
                jnp.where(
                    jax.random.bernoulli(direction_key, shape=(batch_size,)),
                    1.0,
                    -1.0,
                )
            ),
            dtype=np.float32,
        )
        was_active = self._structured_exploration_remaining > 0
        starts = np.logical_and(~was_active, start_draw)
        self._structured_exploration_remaining[starts] = int(
            self.structured_exploration_horizon
        )
        self._structured_exploration_dimension[starts] = sampled_dimensions[starts]
        self._structured_exploration_direction[starts] = sampled_directions[starts]
        active = self._structured_exploration_remaining > 0

        dimensions = self._structured_exploration_dimension.copy()
        safe_dimensions = np.maximum(dimensions, 0)
        directions = self._structured_exploration_direction.copy()
        cell_width = (self._step_action_high - self._step_action_low) / float(
            self.bins ** (self.structured_exploration_level + 1)
        )
        row = jnp.arange(batch_size)
        safe_dimensions_jax = jnp.asarray(safe_dimensions, dtype=jnp.int32)
        candidate = action.at[row, safe_dimensions_jax].add(
            jnp.asarray(directions) * cell_width[safe_dimensions_jax]
        )
        candidate = jnp.clip(
            candidate,
            self._step_action_low,
            self._step_action_high,
        )
        explored = jnp.where(jnp.asarray(active)[:, None], candidate, action)
        signed_delta = np.asarray(
            jax.device_get(
                explored[row, safe_dimensions_jax]
                - action[row, safe_dimensions_jax]
            ),
            dtype=np.float32,
        ).copy()
        signed_delta[~active] = 0.0
        dimensions[~active] = -1

        assignment_probability = np.ones((batch_size,), dtype=np.float32)
        assignment_probability[np.logical_and(~was_active, ~starts)] = (
            1.0 - probability
        )
        assignment_probability[starts] = probability / float(2 * self.action_dim)

        self._structured_exploration_remaining[active] -= 1
        finished = self._structured_exploration_remaining <= 0
        self._structured_exploration_dimension[finished] = -1
        self._structured_exploration_direction[finished] = 0.0
        return (
            explored,
            active,
            starts,
            dimensions,
            signed_delta,
            assignment_probability,
        )

    # ------------------------------------------------------------------
    # Bin-space exploration
    # ------------------------------------------------------------------

    def _apply_bin_flip(self, action_chunk: np.ndarray) -> np.ndarray:
        """Coherent bin-space exploration on a fresh open-loop plan.

        With probability ``bin_flip_prob`` per plan refresh: pick one action
        dimension and one coarse-to-fine level, move every sequence step's
        level-l bin for that dimension to one common random sibling cell, and
        keep the deeper-level sub-indices (inherit-refine).  Integer-cell
        shifts re-encode exactly to the flipped path, so the intervention is
        alias-free by construction (cqn-flow.md 32.2, arm B).
        """

        batch = action_chunk.shape[0]
        flipped = action_chunk.copy()
        self._bin_flip_remaining = np.zeros((batch,), dtype=np.int32)
        self._bin_flip_dimension = np.full((batch,), -1, dtype=np.int16)
        self._bin_flip_delta_sequence = np.zeros(
            (batch, self.action_sequence), dtype=np.float32
        )
        low = np.asarray(self._step_action_low, dtype=np.float64)
        high = np.asarray(self._step_action_high, dtype=np.float64)
        for row in range(batch):
            if self._bin_flip_rng.random() >= self.bin_flip_prob:
                continue
            dim = int(self._bin_flip_rng.integers(self.action_dim))
            level = (
                self.bin_flip_level
                if self.bin_flip_level is not None
                else int(self._bin_flip_rng.integers(self.levels))
            )
            width = (high[dim] - low[dim]) / float(self.bins ** (level + 1))
            values = flipped[row, :, dim].astype(np.float64)
            cell = np.floor((values - low[dim]) / max(width, 1e-8))
            cell = np.clip(cell, 0, self.bins ** (level + 1) - 1)
            within = values - (low[dim] + cell * width)
            parent = cell // self.bins
            local = cell % self.bins
            sibling = int(
                (local[0] + 1 + self._bin_flip_rng.integers(self.bins - 1))
                % self.bins
            )
            new_cell = parent * self.bins + sibling
            new_values = low[dim] + new_cell * width + within
            delta = (new_values - values).astype(np.float32)
            flipped[row, :, dim] = new_values.astype(np.float32)
            self._bin_flip_remaining[row] = self.action_sequence
            self._bin_flip_dimension[row] = dim
            self._bin_flip_delta_sequence[row] = delta
        return flipped

    def _apply_bin_explore(
        self, action_chunk: np.ndarray, register_mask: np.ndarray | None = None
    ) -> np.ndarray:
        """Hierarchical epsilon-bin exploration compatible with the
        closed-loop temporal ensemble (cqn-flow.md 35).

        ``register_mask`` marks which batch rows carry a *fresh plan* this
        step (temporal-ensemble replan mask). Rows outside the mask are
        left untouched: their chunk is not registered, so shifting it
        would do nothing, and advancing their persist counter would burn
        the exploration window on discarded plans (cqn-flow.md 48.2).

        Per fresh plan and per level l (checked coarse-to-fine, first
        firing wins), with probability ``bin_explore_probs[l]``: pick one
        action dimension and move its level-l bin to a common random
        sibling, keeping deeper local offsets (inherit-refine, same
        alias-free cell math as ``_apply_bin_flip``).  Unlike the
        open-loop flip, the shift is REDRAWN ONTO every fresh plan for
        the next ``action_sequence`` steps: a one-shot flip would be
        diluted to the newest-plan ensemble weight and never execute, so
        persistence is what lets the ensemble average actually reach the
        sibling cell while closed-loop correction stays active.
        """

        batch = action_chunk.shape[0]
        shifted = action_chunk.copy()
        self._bin_explore_calls_total = (
            getattr(self, "_bin_explore_calls_total", 0) + 1
        )
        self._bin_explore_mask_rows_total = getattr(
            self, "_bin_explore_mask_rows_total", 0
        ) + int(batch if register_mask is None else int(np.sum(register_mask)))
        if self._bin_explore_remaining.shape[0] != batch:
            self._bin_explore_remaining = np.zeros((batch,), dtype=np.int32)
            self._bin_explore_dimension = np.full((batch,), -1, dtype=np.int16)
            self._bin_explore_level = np.full((batch,), -1, dtype=np.int16)
            self._bin_explore_sibling = np.full((batch,), -1, dtype=np.int16)
        # Rows whose chunk actually received a sibling shift this call; act()
        # uses this to flag the executed steps of shifted registered plans
        # for explore-aware n-step truncation (cqn-flow.md 60).
        self._last_bin_explore_applied = np.zeros((batch,), dtype=np.bool_)
        low = np.asarray(self._step_action_low, dtype=np.float64)
        high = np.asarray(self._step_action_high, dtype=np.float64)
        scale = float(getattr(self, "_bin_explore_scale", 1.0))
        for row in range(batch):
            if register_mask is not None and not register_mask[row]:
                continue
            if self._bin_explore_remaining[row] == 0:
                for level, prob in enumerate(self.bin_explore_probs):
                    if self._bin_explore_rng.random() >= prob * scale:
                        continue
                    dim = int(self._bin_explore_rng.integers(self.action_dim))
                    width = (high[dim] - low[dim]) / float(
                        self.bins ** (level + 1)
                    )
                    value0 = float(action_chunk[row, 0, dim])
                    cell0 = int(
                        np.clip(
                            np.floor((value0 - low[dim]) / max(width, 1e-8)),
                            0,
                            self.bins ** (level + 1) - 1,
                        )
                    )
                    sibling = int(
                        (
                            cell0 % self.bins
                            + 1
                            + self._bin_explore_rng.integers(self.bins - 1)
                        )
                        % self.bins
                    )
                    self._bin_explore_remaining[row] = (
                        self.bin_explore_persist_plans
                    )
                    self._bin_explore_dimension[row] = dim
                    self._bin_explore_level[row] = level
                    self._bin_explore_sibling[row] = sibling
                    self._bin_explore_fired_total += 1
                    break
            if self._bin_explore_remaining[row] > 0:
                dim = int(self._bin_explore_dimension[row])
                level = int(self._bin_explore_level[row])
                sibling = int(self._bin_explore_sibling[row])
                width = (high[dim] - low[dim]) / float(self.bins ** (level + 1))
                values = shifted[row, :, dim].astype(np.float64)
                cell = np.floor((values - low[dim]) / max(width, 1e-8))
                cell = np.clip(cell, 0, self.bins ** (level + 1) - 1)
                within = values - (low[dim] + cell * width)
                parent = cell // self.bins
                new_cell = parent * self.bins + sibling
                new_values = np.clip(
                    low[dim] + new_cell * width + within,
                    low[dim],
                    high[dim],
                )
                shifted[row, :, dim] = new_values.astype(np.float32)
                self._bin_explore_remaining[row] -= 1
                self._last_bin_explore_applied[row] = True
                self._bin_explore_applied_total += 1
        return shifted

    # ------------------------------------------------------------------
    # Post-ensemble interventions
    # ------------------------------------------------------------------

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
        out = np.sum(candidates * weights[..., None], axis=1)
        if (not eval_mode) and (
            self.post_ensemble_l1_flip_prob > 0.0
            or self.post_ensemble_l2_flip_prob > 0.0
        ):
            out = self._post_ensemble_bin_flip(out)
        if eval_mode and self.post_ensemble_random_keep_levels is not None:
            out = self._post_ensemble_randomize(out)
        return out

    def _post_ensemble_bin_flip(self, action: np.ndarray) -> np.ndarray:
        """Train-time exploration: persistent single-dimension bin flips
        applied AFTER temporal ensembling, at one or two C2F granularities.

        Two independent event processes:
          L1 process (post_ensemble_l1_flip_prob): force one alternative L1
            bin inside the CURRENT L0 cell (L2 center) on one dimension.
          L2 process (post_ensemble_l2_flip_prob): force one alternative L2
            bin inside the CURRENT L1 cell on one dimension.
        Each event holds its flip for `horizon` steps so it survives plant
        smoothing. The flipped action is executed AND stored, so 1-step TD
        grounds the flipped bin with the realized outcome. Motivation and
        dose logic: sibling-randomization diagnostics measured which levels'
        content carries outcome (task-dependent), and single-factor
        interventions keep the return contrast attributable. Eval untouched.
        """

        B, D = action.shape
        lo = np.asarray(self.action_low, dtype=np.float32).reshape(
            self.action_sequence, self.action_dim
        )[0]
        hi = np.asarray(self.action_high, dtype=np.float32).reshape(
            self.action_sequence, self.action_dim
        )[0]
        w0 = (hi - lo) / float(self.bins)
        w1 = w0 / float(self.bins)
        w2 = w1 / float(self.bins)
        out = action
        for name, prob, parent_w, child_w in (
            ("l1", self.post_ensemble_l1_flip_prob, w0, w1),
            ("l2", self.post_ensemble_l2_flip_prob, w1, w2),
        ):
            if prob <= 0.0:
                continue
            key = "_%s_flip_state" % name
            st = getattr(self, key, None)
            if st is None or st["dim"].shape[0] != B:
                st = {
                    "dim": np.full((B,), -1, dtype=np.int64),
                    "bin": np.zeros((B,), dtype=np.int64),
                    "left": np.zeros((B,), dtype=np.int64),
                }
                setattr(self, key, st)
            start_ = (st["left"] <= 0) & (np.random.rand(B) < float(prob))
            n_new = int(start_.sum())
            if n_new:
                st["dim"][start_] = np.random.randint(0, D, size=n_new)
                st["bin"][start_] = np.random.randint(0, self.bins, size=n_new)
                st["left"][start_] = int(self.post_ensemble_l1_flip_horizon)
            active = st["left"] > 0
            if active.any():
                if out is action:
                    out = action.copy()
                rows = np.where(active)[0]
                dims = st["dim"][rows]
                parent = np.clip(
                    np.floor((out[rows, dims] - lo[dims]) / parent_w[dims]),
                    0,
                    (parent_w[dims] > 0).astype(np.int64) * 0
                    + int(round((hi[0] - lo[0]) / parent_w[0])) - 1,
                )
                out[rows, dims] = (
                    lo[dims]
                    + parent * parent_w[dims]
                    + (st["bin"][rows] + 0.5) * child_w[dims]
                )
                st["left"][active] -= 1
        return out

    def _post_ensemble_randomize(self, action: np.ndarray) -> np.ndarray:
        """Diagnostic: keep the ensembled action's true C2F prefix, randomize
        the rest.

        Applied AFTER temporal ensembling, so the perturbation reaches the
        environment at full strength instead of averaging out across the 16
        plans (which is what made the pre-ensemble ``random_levels_from``
        probe insensitive: independent per-plan jitter is mean-zero, while a
        genuinely better fine policy would be CORRELATED across plans and
        pass the weights-sum-to-1 average untouched). keep_levels=2 asks:
        given the true L1 cell of the executed action, does the exact L2
        position matter? Equal performance => the task does not need
        resolution below the kept level."""

        keep = int(self.post_ensemble_random_keep_levels)
        if not 1 <= keep < self.levels:
            raise ValueError(
                "post_ensemble_random_keep_levels must be in [1, levels-1]."
            )
        lo = np.asarray(self.action_low, dtype=np.float32).reshape(
            self.action_sequence, self.action_dim
        )[0]
        hi = np.asarray(self.action_high, dtype=np.float32).reshape(
            self.action_sequence, self.action_dim
        )[0]
        span = hi - lo
        parent_w = span / float(self.bins**keep)
        leaf_w = span / float(self.bins**self.levels)
        n_leaves = self.bins ** (self.levels - keep)
        parent_idx = np.clip(
            np.floor((action - lo) / parent_w),
            0,
            self.bins**keep - 1,
        )
        if self.post_ensemble_fixed_leaf is not None:
            # Constant leaf: a PERSISTENT within-cell offset, in contrast to
            # the iid random leaf which the plant low-pass filters. leaf 0 =
            # the lowest sub-cell of the kept parent cell.
            leaf = np.full(action.shape, int(self.post_ensemble_fixed_leaf))
        else:
            leaf = np.random.randint(0, n_leaves, size=action.shape)
        return (lo + parent_idx * parent_w + (leaf + 0.5) * leaf_w).astype(
            action.dtype
        )

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def act(self, observations: dict, step: int, eval_mode: bool):
        batch_size = int(next(iter(observations.values())).shape[0])
        if not eval_mode:
            self._act_train_calls_total = (
                getattr(self, "_act_train_calls_total", 0) + 1
            )
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
            if (
                self.bin_flip_prob > 0.0
                and not self.temporal_ensemble
                and not eval_mode
                and step >= self.num_explore_steps
            ):
                action_chunk = self._apply_bin_flip(action_chunk)
            if (
                self.bin_explore_probs is not None
                and not eval_mode
                and step >= self.num_explore_steps
            ):
                if self.bin_explore_schedule is not None:
                    self._bin_explore_scale = float(
                        utils.schedule(self.bin_explore_schedule, step)
                    )
                action_chunk = self._apply_bin_explore(action_chunk, register_mask)
        else:
            if self.temporal_ensemble:
                action_chunk = np.zeros(
                    (batch_size, self.action_sequence, self.action_dim),
                    dtype=np.float32,
                )
            else:
                prefix = "_eval" if eval_mode else "_train"
                action_chunk = getattr(self, f"{prefix}_open_loop_plan").copy()
        if not eval_mode and self.bin_explore_probs is not None:
            # Executed-step explored flags: a registered plan that carried a
            # sibling shift contributes replan_interval executed steps, so a
            # persist-2 shift marks 2 x replan_interval steps. The workspace
            # snapshots _last_bin_explored into the replay "explored" extra
            # for explore-aware n-step truncation (cqn-flow.md 60).
            if (
                not hasattr(self, "_bin_explored_exec_remaining")
                or self._bin_explored_exec_remaining.shape[0] != batch_size
            ):
                self._bin_explored_exec_remaining = np.zeros(
                    (batch_size,), dtype=np.int32
                )
            applied = getattr(self, "_last_bin_explore_applied", None)
            if needs_inference and applied is not None:
                registered = (
                    np.asarray(register_mask, dtype=bool)
                    if register_mask is not None
                    else np.ones((batch_size,), dtype=bool)
                )
                newly_shifted = registered & np.asarray(applied, dtype=bool)
                self._bin_explored_exec_remaining[newly_shifted] = int(
                    getattr(self, "temporal_ensemble_replan_interval", 1)
                )
            self._last_bin_explored = self._bin_explored_exec_remaining > 0
            self._bin_explored_exec_remaining = np.maximum(
                self._bin_explored_exec_remaining - 1, 0
            )
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
            structured_mask = np.zeros((batch_size,), dtype=np.bool_)
            structured_start = np.zeros((batch_size,), dtype=np.bool_)
            structured_dimension = np.full((batch_size,), -1, dtype=np.int16)
            structured_delta = np.zeros((batch_size,), dtype=np.float32)
            structured_assignment_prob = np.ones((batch_size,), dtype=np.float32)
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
                if self.structured_exploration_prob > 0.0:
                    self.rng_key, structured_key = jax.random.split(self.rng_key)
                    (
                        executed_action,
                        structured_mask,
                        structured_start,
                        structured_dimension,
                        structured_delta,
                        structured_assignment_prob,
                    ) = self._coherent_structured_exploration_action(
                        executed_action,
                        structured_key,
                    )
            executed_action = np.asarray(
                jax.device_get(executed_action),
                dtype=np.float32,
            )
            structured_mask = np.asarray(
                jax.device_get(structured_mask),
                dtype=np.bool_,
            )
            structured_start = np.asarray(
                jax.device_get(structured_start), dtype=np.bool_
            )
            self._last_structured_exploration_mask = structured_mask
            self._last_structured_exploration_start = structured_start
            self._last_structured_exploration_dimension = np.asarray(
                structured_dimension, dtype=np.int16
            )
            self._last_structured_exploration_delta = np.asarray(
                structured_delta, dtype=np.float32
            )
            self._last_structured_exploration_assignment_prob = np.asarray(
                structured_assignment_prob, dtype=np.float32
            )
            if self.bin_flip_prob > 0.0 and not self.temporal_ensemble:
                position = getattr(self, "_train_open_loop_position")
                active = self._bin_flip_remaining > 0
                flip_start = self._bin_flip_remaining == self.action_sequence
                token = np.clip(
                    np.asarray(position, dtype=np.int32) - 1,
                    0,
                    self.action_sequence - 1,
                )
                rows = np.arange(batch_size)
                delta = self._bin_flip_delta_sequence[rows, token]
                self._last_structured_exploration_mask = active.copy()
                self._last_structured_exploration_start = flip_start.copy()
                self._last_structured_exploration_dimension = np.where(
                    active, self._bin_flip_dimension, -1
                ).astype(np.int16)
                self._last_structured_exploration_delta = np.where(
                    active, delta, 0.0
                ).astype(np.float32)
                self._last_structured_exploration_assignment_prob = np.where(
                    flip_start,
                    self.bin_flip_prob,
                    np.where(active, 1.0, 1.0 - self.bin_flip_prob),
                ).astype(np.float32)
                self._bin_flip_remaining = np.maximum(
                    self._bin_flip_remaining - 1, 0
                )
            if (
                step >= self.num_explore_steps
                and self.structured_exploration_prob > 0.0
            ):
                self._structured_exploration_eligible += int(batch_size)
                self._structured_exploration_applied += int(structured_mask.sum())
                self._structured_exploration_starts += int(structured_start.sum())

        action_chunk = action_chunk.copy()
        action_chunk[:, 0] = executed_action
        return action_chunk

    # ------------------------------------------------------------------
    # Checkpointing / diagnostics / episode reset
    # ------------------------------------------------------------------

    def checkpoint_state_dict(self) -> dict:
        state = super().checkpoint_state_dict()
        state.update(self._exploration_checkpoint_state())
        return state

    def load_checkpoint_state_dict(self, state_dict: dict):
        super().load_checkpoint_state_dict(state_dict)
        self._load_exploration_checkpoint_state(state_dict)

    def _exploration_checkpoint_state(self) -> dict:
        """NumPy exploration RNG streams (and nothing else).

        Without these a resume restarts both generators from the seed while
        the original process had advanced them, so the exploration
        assignment sequence diverges from an uninterrupted run
        (cqn-flow.md 48.2). The ``_bin_explore_*`` persist windows are
        deliberately NOT checkpointed: workspace snapshots carry no env
        state, so a resume starts fresh episodes without an agent.reset()
        call, and a restored mid-episode window would leak into them --
        the exact cross-episode intervention that reset() forbids
        (cqn-flow.md 48.3).
        """
        state = {}
        for key, attribute in (
            ("bin_flip_rng_state", "_bin_flip_rng"),
            ("bin_explore_rng_state", "_bin_explore_rng"),
        ):
            generator = getattr(self, attribute, None)
            if generator is not None:
                state[key] = generator.bit_generator.state
        return state

    def _load_exploration_checkpoint_state(self, state_dict: dict) -> None:
        # Older snapshots predate these keys; keep their fresh-init behavior.
        stored = state_dict.get("bin_flip_rng_state")
        generator = getattr(self, "_bin_flip_rng", None)
        if stored is not None and generator is not None:
            generator.bit_generator.state = stored
        stored = state_dict.get("bin_explore_rng_state")
        generator = getattr(self, "_bin_explore_rng", None)
        if stored is not None and generator is not None:
            generator.bit_generator.state = stored
        # Snapshots from the short-lived 48.2 format may carry
        # bin_explore_{remaining,dimension,level,sibling}; ignore them so
        # resumed runs start their fresh episodes windowless.

    def rollout_diagnostics(self) -> dict[str, float]:
        eligible = int(getattr(self, "_structured_exploration_eligible", 0))
        applied = int(getattr(self, "_structured_exploration_applied", 0))
        diagnostics = {
            "structured_exploration_rate": (
                float(applied / eligible) if eligible else 0.0
            ),
            "structured_exploration_applied": float(applied),
            "structured_exploration_eligible": float(eligible),
            "structured_exploration_starts": float(
                getattr(self, "_structured_exploration_starts", 0)
            ),
            "bin_explore_fired_total": float(
                getattr(self, "_bin_explore_fired_total", 0)
            ),
            "bin_explore_applied_total": float(
                getattr(self, "_bin_explore_applied_total", 0)
            ),
            "bin_explore_calls_total": float(
                getattr(self, "_bin_explore_calls_total", 0)
            ),
            "act_train_calls_total": float(
                getattr(self, "_act_train_calls_total", 0)
            ),
            "bin_explore_mask_rows_total": float(
                getattr(self, "_bin_explore_mask_rows_total", 0)
            ),
        }
        return diagnostics

    def reset(self, step: int, agents_to_reset: list[int]):
        del step
        for agent_index in agents_to_reset:
            if agent_index < self.num_train_envs:
                if hasattr(self, "_structured_exploration_remaining"):
                    self._structured_exploration_remaining[agent_index] = 0
                    self._structured_exploration_dimension[agent_index] = -1
                    self._structured_exploration_direction[agent_index] = 0.0
                if (
                    hasattr(self, "_bin_explore_remaining")
                    and agent_index < self._bin_explore_remaining.shape[0]
                ):
                    # A persisted sibling shift is a within-episode
                    # intervention; never carry it into the next episode.
                    self._bin_explore_remaining[agent_index] = 0
                    self._bin_explore_dimension[agent_index] = -1
                    self._bin_explore_level[agent_index] = -1
                    self._bin_explore_sibling[agent_index] = -1
                if (
                    hasattr(self, "_bin_explored_exec_remaining")
                    and agent_index < self._bin_explored_exec_remaining.shape[0]
                ):
                    self._bin_explored_exec_remaining[agent_index] = 0
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
    "CQNASStructuredExplore",
    "CQNASStructuredExploreSpec",
    "cqn_as_structured_explore_spec_from_cfg",
]
