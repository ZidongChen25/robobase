"""Critic-only decoupled joint-chunk CQN.

The method deliberately separates two value functions:

* ``Q_H(s, a[t:t+H])`` scores a complete replay action chunk.
* ``Q_P(s, prefix)`` amortizes optimistic completion values and is queried by
  level-major, factor-autoregressive coarse-to-fine (C2F) beam search.

There is no actor or behavior-cloning loss.  Demonstration and online samples
use the same replay losses, and only the replay-selected C2F bins receive a
prefix distillation loss.
"""

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
from robobase.method.cqn import encode_action, zoom_in
from robobase.method.q_chunking import QChunkCritic, q_chunking_td_target
from robobase.method.rl_common import (
    JaxRLMethodBase,
    RLModelSpec,
    activation,
    rl_model_spec_from_cfg,
    scale_unit_action,
    unscale_action,
)
from robobase.replay_buffer.replay_buffer import ReplayBuffer


_DENSE_INIT = nn.initializers.variance_scaling(1.0, "fan_avg", "uniform")


@dataclass(frozen=True)
class DJCQNSpec:
    critic_lr: float
    num_train_steps: int
    num_explore_steps: int
    critic_target_tau: float
    critic_grad_clip: float | None
    weight_decay: float
    levels: int
    bins: int
    beam_width: int
    num_critics: int
    prefix_expectile: float
    q_aggregate: str
    eval_lcb_beta: float
    sibling_exploration_prob: float
    sibling_level: int
    model: RLModelSpec


def djcqn_spec_from_cfg(cfg: DictConfig) -> DJCQNSpec:
    method = cfg.method
    q_aggregate = str(method.get("q_aggregate", "min")).lower()
    if q_aggregate not in {"mean", "min"}:
        raise ValueError("method.q_aggregate must be 'mean' or 'min'.")
    return DJCQNSpec(
        critic_lr=float(method.get("critic_lr", 3e-4)),
        num_train_steps=int(method.get("num_train_steps", cfg.num_train_frames)),
        num_explore_steps=int(method.get("num_explore_steps", cfg.num_explore_steps)),
        critic_target_tau=float(method.get("critic_target_tau", 0.005)),
        critic_grad_clip=(
            None
            if method.get("critic_grad_clip", None) is None
            else float(method.critic_grad_clip)
        ),
        weight_decay=float(method.get("weight_decay", 0.0)),
        levels=int(method.get("levels", 3)),
        bins=int(method.get("bins", 5)),
        beam_width=int(method.get("beam_width", 8)),
        num_critics=int(method.get("num_critics", 5)),
        prefix_expectile=float(method.get("prefix_expectile", 0.8)),
        q_aggregate=q_aggregate,
        eval_lcb_beta=float(method.get("eval_lcb_beta", 1.0)),
        sibling_exploration_prob=float(
            method.get("sibling_exploration_prob", 0.0)
        ),
        sibling_level=int(method.get("sibling_level", -1)),
        model=rl_model_spec_from_cfg(cfg),
    )


def validate_djcqn_config(
    *,
    action_sequence: int,
    execution_length: int,
    replay_nstep: int,
    temporal_ensemble: bool,
    prefix_horizon: int = 1,
    action_execution_start: int = 0,
) -> None:
    """Validate the fixed-H replay and one-step replanning contract."""

    if action_sequence < 2:
        raise ValueError("DJCQN requires action_sequence >= 2.")
    if prefix_horizon != 1:
        raise ValueError("The initial DJCQN implementation requires prefix_horizon=1.")
    if execution_length != 1:
        raise ValueError("DJCQN requires execution_length=1 for one-step replanning.")
    if replay_nstep != action_sequence:
        raise ValueError(
            "DJCQN requires replay.nstep == action_sequence for its H-step target."
        )
    if temporal_ensemble:
        raise ValueError("DJCQN requires temporal_ensemble=false.")
    if action_execution_start != 0:
        raise ValueError("DJCQN requires action_execution_start=0.")


def absolute_topk(scores: jax.Array, width: int) -> tuple[jax.Array, jax.Array]:
    """Select beams from the current absolute prefix values, never a sum."""

    width = min(int(width), int(scores.shape[-1]))
    return jax.lax.top_k(scores, width)


def chosen_bin_upper_expectile_loss(
    bin_values: jax.Array,
    chosen_bins: jax.Array,
    targets: jax.Array,
    expectile: float,
) -> jax.Array:
    """Chosen-bin-only expectile loss.

    Args:
        bin_values: ``[..., critics, bins]`` prefix values.
        chosen_bins: ``[...]`` replay bin indices.
        targets: ``[..., critics]`` stopped full-chunk values.

    Sibling bins are absent from the objective and therefore have exactly zero
    derivative with respect to the output tensor.
    """

    chosen = jnp.take_along_axis(
        bin_values,
        chosen_bins[..., None, None],
        axis=-1,
    )[..., 0]
    residual = targets - chosen
    weights = jnp.where(residual >= 0.0, expectile, 1.0 - expectile)
    return weights * jnp.square(residual)


def rank_adjacent_sibling_disagreement(
    prefix_values: jax.Array,
    chosen_bins: jax.Array,
    *,
    level: int,
    factors: int,
    bins: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Choose the locally adjacent bin with largest ensemble disagreement.

    ``prefix_values`` and ``chosen_bins`` follow the teacher-forced query order
    ``level * factors + factor``.  The returned factor and signed offset are
    per batch row, so vector environments need not share one intervention.
    Boundary proposals reflect to the other adjacent bin rather than becoming
    a no-op.
    """

    start = int(level) * int(factors)
    stop = start + int(factors)
    level_values = prefix_values[:, start:stop]
    level_bins = chosen_bins[:, start:stop]
    offsets = jnp.asarray((-1, 1), dtype=jnp.int32)
    proposed = level_bins[..., None] + offsets
    fallback = level_bins[..., None] - offsets
    sibling_bins = jnp.where(
        jnp.logical_and(proposed >= 0, proposed < int(bins)),
        proposed,
        fallback,
    )
    sibling_bins = jnp.clip(sibling_bins, 0, int(bins) - 1)
    sibling_values = jnp.take_along_axis(
        level_values,
        sibling_bins[:, :, None, :],
        axis=-1,
    )
    disagreement = jnp.std(sibling_values, axis=-2)
    flat_choice = jnp.argmax(
        disagreement.reshape((disagreement.shape[0], -1)), axis=-1
    )
    factor = flat_choice // 2
    selected_sibling = jnp.take_along_axis(
        sibling_bins.reshape((sibling_bins.shape[0], -1)),
        flat_choice[:, None],
        axis=-1,
    )[:, 0]
    selected_original = jnp.take_along_axis(
        level_bins,
        factor[:, None],
        axis=-1,
    )[:, 0]
    selected_disagreement = jnp.take_along_axis(
        disagreement.reshape((disagreement.shape[0], -1)),
        flat_choice[:, None],
        axis=-1,
    )[:, 0]
    return factor, selected_sibling - selected_original, selected_disagreement


class PrefixC2FCritic(nn.Module):
    """Ensemble prefix critic used only through explicit C2F value queries."""

    hidden_dims: tuple[int, ...]
    factors: int
    levels: int
    bins: int
    num_critics: int
    activation_name: str = "gelu"
    use_layer_norm: bool = True

    @nn.compact
    def __call__(
        self,
        features: jax.Array,
        low: jax.Array,
        high: jax.Array,
        selected_values: jax.Array,
        selected_mask: jax.Array,
        level_indices: jax.Array,
        factor_indices: jax.Array,
    ) -> jax.Array:
        level_one_hot = jax.nn.one_hot(level_indices, self.levels)
        factor_one_hot = jax.nn.one_hot(factor_indices, self.factors)
        context = jnp.concatenate(
            [
                features.astype(jnp.float32),
                low.astype(jnp.float32),
                high.astype(jnp.float32),
                selected_values.astype(jnp.float32),
                selected_mask.astype(jnp.float32),
                level_one_hot.astype(jnp.float32),
                factor_one_hot.astype(jnp.float32),
            ],
            axis=-1,
        )
        outputs = []
        for critic_index in range(self.num_critics):
            x = context
            for layer_index, width in enumerate(self.hidden_dims):
                x = nn.Dense(
                    width,
                    kernel_init=_DENSE_INIT,
                    name=f"p{critic_index + 1}_dense_{layer_index}",
                )(x)
                x = activation(x, self.activation_name)
                if self.use_layer_norm:
                    x = nn.LayerNorm(
                        name=f"p{critic_index + 1}_norm_{layer_index}"
                    )(x)
            outputs.append(
                nn.Dense(
                    self.bins,
                    kernel_init=_DENSE_INIT,
                    name=f"p{critic_index + 1}_out",
                )(x)
            )
        return jnp.stack(outputs, axis=-2)


def _batched_gather_beams(values: jax.Array, indices: jax.Array) -> jax.Array:
    """Gather the beam axis (axis=1) independently for every batch row."""

    trailing = (1,) * (values.ndim - 2)
    gather_indices = indices.reshape(indices.shape + trailing)
    gather_indices = jnp.broadcast_to(
        gather_indices,
        indices.shape + values.shape[2:],
    )
    return jnp.take_along_axis(values, gather_indices, axis=1)


def _aggregate_prefix_values(
    values: jax.Array,
    *,
    head_indices: jax.Array | None,
    eval_lcb_beta: float | None,
) -> jax.Array:
    if head_indices is not None:
        gather_indices = jnp.broadcast_to(
            head_indices[:, None, None, None],
            values.shape[:2] + (1, values.shape[-1]),
        )
        return jnp.take_along_axis(
            values,
            gather_indices,
            axis=-2,
        )[..., 0, :]
    mean = jnp.mean(values, axis=-2)
    if eval_lcb_beta is None:
        return mean
    return mean - float(eval_lcb_beta) * jnp.std(values, axis=-2)


def c2f_prefix_beam_search(
    *,
    model: PrefixC2FCritic,
    params: Any,
    features: jax.Array,
    action_low: jax.Array,
    action_high: jax.Array,
    levels: int,
    bins: int,
    beam_width: int,
    head_indices: jax.Array | None,
    eval_lcb_beta: float | None,
    forced_sibling_level: int = -1,
    forced_sibling_factor: int = -1,
    forced_sibling_offset: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Select a primitive action with conditional prefix-value beam search.

    The beam score is replaced by the value of the newest prefix at every
    factor.  It is never added to a previous factor score.  A forced sibling
    is applied while constructing the prefix, so every later factor and finer
    level is queried again under the changed prefix.
    """

    batch_size = features.shape[0]
    factors = action_low.shape[0]
    width = int(beam_width)
    low = jnp.broadcast_to(action_low, (batch_size, 1, factors))
    high = jnp.broadcast_to(action_high, (batch_size, 1, factors))
    selected = 0.5 * (low + high)
    current_values = jnp.zeros((batch_size, 1), dtype=jnp.float32)
    current_head_values = jnp.zeros(
        (batch_size, 1, model.num_critics), dtype=jnp.float32
    )

    for level in range(int(levels)):
        selected_mask = jnp.zeros_like(selected)
        for factor in range(int(factors)):
            beam_count = low.shape[1]
            repeated_features = jnp.repeat(features, beam_count, axis=0)
            level_indices = jnp.full(
                (batch_size * beam_count,), level, dtype=jnp.int32
            )
            factor_indices = jnp.full(
                (batch_size * beam_count,), factor, dtype=jnp.int32
            )
            values = model.apply(
                params,
                repeated_features,
                low.reshape((batch_size * beam_count, factors)),
                high.reshape((batch_size * beam_count, factors)),
                selected.reshape((batch_size * beam_count, factors)),
                selected_mask.reshape((batch_size * beam_count, factors)),
                level_indices,
                factor_indices,
            ).reshape((batch_size, beam_count, model.num_critics, bins))
            scores = _aggregate_prefix_values(
                values,
                head_indices=head_indices,
                eval_lcb_beta=eval_lcb_beta,
            )

            interval_width = (high[..., factor] - low[..., factor]) / float(bins)
            candidate_bins = jnp.arange(bins, dtype=jnp.int32)
            candidate_centers = (
                low[..., factor, None]
                + (candidate_bins.astype(jnp.float32) + 0.5)
                * interval_width[..., None]
            )

            if (
                level == int(forced_sibling_level)
                and factor == int(forced_sibling_factor)
                and int(forced_sibling_offset) != 0
            ):
                # Follow the best current parent, then move locally to a sibling.
                flat_scores = scores.reshape((batch_size, -1))
                greedy_flat = jnp.argmax(flat_scores, axis=-1)
                parent = greedy_flat // bins
                greedy_bin = greedy_flat % bins
                proposed = greedy_bin + int(forced_sibling_offset)
                fallback = greedy_bin - int(forced_sibling_offset)
                sibling_bin = jnp.where(
                    jnp.logical_and(proposed >= 0, proposed < bins),
                    proposed,
                    fallback,
                )
                parent_indices = parent[:, None]
                low = _batched_gather_beams(low, parent_indices)
                high = _batched_gather_beams(high, parent_indices)
                selected = _batched_gather_beams(selected, parent_indices)
                selected_mask = _batched_gather_beams(
                    selected_mask, parent_indices
                )
                parent_values = _batched_gather_beams(values, parent_indices)
                parent_centers = _batched_gather_beams(
                    candidate_centers, parent_indices
                )
                chosen = sibling_bin[:, None]
                chosen_center = jnp.take_along_axis(
                    parent_centers, chosen[..., None], axis=-1
                )[..., 0]
                current_head_values = jnp.take_along_axis(
                    parent_values,
                    chosen[:, :, None, None],
                    axis=-1,
                )[..., 0]
                current_values = _aggregate_prefix_values(
                    parent_values,
                    head_indices=head_indices,
                    eval_lcb_beta=eval_lcb_beta,
                )
                current_values = jnp.take_along_axis(
                    current_values, chosen[..., None], axis=-1
                )[..., 0]
            else:
                flat_scores = scores.reshape((batch_size, -1))
                current_values, chosen_flat = absolute_topk(flat_scores, width)
                parent = chosen_flat // bins
                chosen = chosen_flat % bins
                low = _batched_gather_beams(low, parent)
                high = _batched_gather_beams(high, parent)
                selected = _batched_gather_beams(selected, parent)
                selected_mask = _batched_gather_beams(selected_mask, parent)
                parent_values = _batched_gather_beams(values, parent)
                parent_centers = _batched_gather_beams(candidate_centers, parent)
                chosen_center = jnp.take_along_axis(
                    parent_centers, chosen[..., None], axis=-1
                )[..., 0]
                current_head_values = jnp.take_along_axis(
                    parent_values,
                    chosen[:, :, None, None],
                    axis=-1,
                )[..., 0]

            selected = selected.at[..., factor].set(chosen_center)
            selected_mask = selected_mask.at[..., factor].set(1.0)
            factor_low = low[..., factor]
            factor_high = high[..., factor]
            zoomed_low, zoomed_high = zoom_in(
                factor_low,
                factor_high,
                chosen,
                bins,
                action_low[factor],
                action_high[factor],
            )
            low = low.at[..., factor].set(zoomed_low)
            high = high.at[..., factor].set(zoomed_high)

    best = jnp.argmax(current_values, axis=1)
    best_action = jnp.take_along_axis(
        selected,
        best[:, None, None],
        axis=1,
    )[:, 0]
    best_values = jnp.take_along_axis(
        current_head_values,
        best[:, None, None],
        axis=1,
    )[:, 0]
    best_score = jnp.take_along_axis(current_values, best[:, None], axis=1)[:, 0]
    return best_action, best_values, best_score


class DJCQN(JaxRLMethodBase, OffPolicyMethod):
    """Actor-free joint-chunk value learning with one-step C2F replanning."""

    def _init_cached_pixel_feature_key(self, method_name: str) -> None:
        del method_name
        super()._init_cached_pixel_feature_key("djcqn")

    def __init__(
        self,
        critic_lr: float,
        num_train_steps: int,
        num_explore_steps: int,
        critic_target_tau: float,
        levels: int,
        bins: int,
        beam_width: int,
        num_critics: int,
        prefix_expectile: float,
        q_aggregate: str,
        eval_lcb_beta: float,
        sibling_exploration_prob: float,
        sibling_level: int,
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
        if levels < 1 or bins < 2 or beam_width < 1:
            raise ValueError("DJCQN requires levels>=1, bins>=2, beam_width>=1.")
        if num_critics < 2:
            raise ValueError("DJCQN requires num_critics>=2 for bootstrap exploration.")
        if not 0.5 <= prefix_expectile < 1.0:
            raise ValueError("prefix_expectile must be in [0.5, 1).")
        if str(q_aggregate) not in {"mean", "min"}:
            raise ValueError("q_aggregate must be 'mean' or 'min'.")
        if float(eval_lcb_beta) < 0.0:
            raise ValueError("eval_lcb_beta must be non-negative.")
        if not 0.0 < critic_target_tau <= 1.0:
            raise ValueError("critic_target_tau must be in (0, 1].")
        if not 0.0 <= sibling_exploration_prob <= 1.0:
            raise ValueError("sibling_exploration_prob must be in [0, 1].")

        self.num_explore_steps = int(num_explore_steps)
        self.critic_target_tau = float(critic_target_tau)
        self.levels = int(levels)
        self.bins = int(bins)
        self.beam_width = int(beam_width)
        self.num_critics = int(num_critics)
        self.prefix_expectile = float(prefix_expectile)
        self.q_aggregate = str(q_aggregate)
        self.eval_lcb_beta = float(eval_lcb_beta)
        self.sibling_exploration_prob = float(sibling_exploration_prob)
        self.sibling_level = (
            self.levels - 1 if int(sibling_level) < 0 else int(sibling_level)
        )
        if not 0 <= self.sibling_level < self.levels:
            raise ValueError("sibling_level must identify a configured C2F level.")

        input_dim = self._setup_rl_features(model, seed=seed)
        bounds_low = np.asarray(action_space.low, dtype=np.float32)
        bounds_high = np.asarray(action_space.high, dtype=np.float32)
        if (
            not np.all(np.isfinite(bounds_low))
            or not np.all(np.isfinite(bounds_high))
            or np.any(bounds_high <= bounds_low)
        ):
            raise ValueError("DJCQN requires finite, ordered action bounds.")
        if not (
            np.allclose(bounds_low, bounds_low[0:1])
            and np.allclose(bounds_high, bounds_high[0:1])
        ):
            raise ValueError("DJCQN requires identical primitive bounds across H.")
        self.primitive_low = jnp.asarray(bounds_low[0])
        self.primitive_high = jnp.asarray(bounds_high[0])
        flat_chunk_dim = self.action_sequence * self.action_dim

        self.joint_model = QChunkCritic(
            hidden_dims=model.hidden_dims,
            num_critics=self.num_critics,
            activation_name=model.activation,
            use_layer_norm=True,
        )
        self.prefix_model = PrefixC2FCritic(
            hidden_dims=model.hidden_dims,
            factors=self.action_dim,
            levels=self.levels,
            bins=self.bins,
            num_critics=self.num_critics,
            activation_name=model.activation,
            use_layer_norm=True,
        )
        dummy_features = jnp.zeros((1, input_dim), dtype=jnp.float32)
        dummy_chunk = jnp.zeros((1, flat_chunk_dim), dtype=jnp.float32)
        dummy_factor = jnp.zeros((1, self.action_dim), dtype=jnp.float32)
        dummy_index = jnp.zeros((1,), dtype=jnp.int32)
        self.rng_key, joint_key, prefix_key = jax.random.split(self.rng_key, 3)
        params = {
            "joint": self.joint_model.init(joint_key, dummy_features, dummy_chunk),
            "prefix": self.prefix_model.init(
                prefix_key,
                dummy_features,
                dummy_factor - 1.0,
                dummy_factor + 1.0,
                dummy_factor,
                dummy_factor,
                dummy_index,
                dummy_index,
            ),
        }
        if self._trainable_encoder:
            params["encoder"] = jax.tree.map(jnp.array, self._encoder_params)
        self.params: dict[str, Any] = params
        self.target_params = jax.tree.map(jnp.array, params)

        transforms = []
        if critic_grad_clip is not None:
            transforms.append(self.optax.clip_by_global_norm(float(critic_grad_clip)))
        transforms.append(
            self.optax.adamw(float(critic_lr), weight_decay=float(weight_decay))
        )
        self.optimizer = self.optax.chain(*transforms)
        self.opt_state = self.optimizer.init(self.params)

        update_fn = self._build_update_fn()
        train_policy_fn = self._build_policy_fn(eval_mode=False)
        eval_policy_fn = self._build_policy_fn(eval_mode=True)
        sibling_rank_fn = self._build_sibling_rank_fn()
        if self._jit_enabled:
            update_fn = jax.jit(update_fn)
            train_policy_fn = jax.jit(train_policy_fn, static_argnums=(3, 4))
            eval_policy_fn = jax.jit(eval_policy_fn, static_argnums=(3, 4))
            sibling_rank_fn = jax.jit(sibling_rank_fn)
        self._update_impl = update_fn
        self._train_policy_impl = train_policy_fn
        self._eval_policy_impl = eval_policy_fn
        self._sibling_rank_impl = sibling_rank_fn

        self._train_episode_heads = np.full(
            (self.num_train_envs,), -1, dtype=np.int32
        )
        self._last_selected_q = np.nan
        self._last_sibling_explored = 0.0
        self._last_sibling_disagreement = np.nan
        self._last_sibling_factor = np.nan

    def _features(self, params, obs_inputs, *, stop_gradient=False):
        return self._rl_features(
            params.get("encoder", None), obs_inputs, stop_gradient=stop_gradient
        )

    def _aggregate_joint(self, values):
        if self.q_aggregate == "min":
            return jnp.min(values, axis=-1)
        return jnp.mean(values, axis=-1)

    def _select(
        self,
        params,
        features,
        *,
        head_indices,
        eval_mode,
        sibling_factor=-1,
        sibling_offset=0,
    ):
        return c2f_prefix_beam_search(
            model=self.prefix_model,
            params=params["prefix"],
            features=features,
            action_low=-jnp.ones_like(self.primitive_low),
            action_high=jnp.ones_like(self.primitive_high),
            levels=self.levels,
            bins=self.bins,
            beam_width=self.beam_width,
            head_indices=head_indices,
            eval_lcb_beta=self.eval_lcb_beta if eval_mode else None,
            forced_sibling_level=self.sibling_level if sibling_factor >= 0 else -1,
            forced_sibling_factor=int(sibling_factor),
            forced_sibling_offset=int(sibling_offset),
        )

    def _build_policy_fn(self, *, eval_mode: bool):
        def policy_fn(params, obs_inputs, head_indices, sibling_factor, sibling_offset):
            features = self._features(params, obs_inputs, stop_gradient=True)
            return self._select(
                params,
                features,
                head_indices=None if eval_mode else head_indices,
                eval_mode=eval_mode,
                sibling_factor=sibling_factor,
                sibling_offset=sibling_offset,
            )

        return policy_fn

    def _build_sibling_rank_fn(self):
        def sibling_rank_fn(params, obs_inputs, primitive_actions):
            features = self._features(params, obs_inputs, stop_gradient=True)
            prefix_values, chosen_bins = self._teacher_forced_prefix_values(
                params,
                features,
                primitive_actions,
            )
            return rank_adjacent_sibling_disagreement(
                prefix_values,
                chosen_bins,
                level=self.sibling_level,
                factors=self.action_dim,
                bins=self.bins,
            )

        return sibling_rank_fn

    def _teacher_forced_prefix_values(self, params, features, primitive_actions):
        encoded = encode_action(
            primitive_actions,
            -jnp.ones((self.action_dim,), dtype=jnp.float32),
            jnp.ones((self.action_dim,), dtype=jnp.float32),
            self.levels,
            self.bins,
        )
        batch_size = primitive_actions.shape[0]
        low = -jnp.ones((batch_size, self.action_dim), dtype=jnp.float32)
        high = jnp.ones((batch_size, self.action_dim), dtype=jnp.float32)
        selected = 0.5 * (low + high)
        outputs = []
        chosen_bins = []
        for level in range(self.levels):
            selected_mask = jnp.zeros_like(selected)
            for factor in range(self.action_dim):
                level_indices = jnp.full((batch_size,), level, dtype=jnp.int32)
                factor_indices = jnp.full((batch_size,), factor, dtype=jnp.int32)
                outputs.append(
                    self.prefix_model.apply(
                        params["prefix"],
                        features,
                        low,
                        high,
                        selected,
                        selected_mask,
                        level_indices,
                        factor_indices,
                    )
                )
                chosen = encoded[:, level, factor]
                chosen_bins.append(chosen)
                width = (high[:, factor] - low[:, factor]) / float(self.bins)
                center = low[:, factor] + (chosen.astype(jnp.float32) + 0.5) * width
                selected = selected.at[:, factor].set(center)
                selected_mask = selected_mask.at[:, factor].set(1.0)
                new_low, new_high = zoom_in(
                    low[:, factor],
                    high[:, factor],
                    chosen,
                    self.bins,
                    -1.0,
                    1.0,
                )
                low = low.at[:, factor].set(new_low)
                high = high.at[:, factor].set(new_high)
        return jnp.stack(outputs, axis=1), jnp.stack(chosen_bins, axis=1)

    def _build_update_fn(self):
        optimizer = self.optimizer
        tau = self.critic_target_tau

        def update_fn(
            params,
            target_params,
            opt_state,
            obs_inputs,
            next_obs_inputs,
            actions,
            rewards,
            discounts,
            bootstrap,
            action_valid,
            loss_weights,
            bootstrap_mask,
            key,
        ):
            del key
            full_chunk_valid = jnp.all(action_valid, axis=1).astype(jnp.float32)
            sample_weights = loss_weights * full_chunk_valid
            target_next_features = self._features(
                target_params, next_obs_inputs, stop_gradient=True
            )
            next_action, next_head_values, _ = self._select(
                target_params,
                target_next_features,
                head_indices=None,
                eval_mode=True,
                sibling_factor=-1,
                sibling_offset=0,
            )
            del next_action
            next_v = self._aggregate_joint(next_head_values)
            target_q = jax.lax.stop_gradient(
                q_chunking_td_target(rewards, discounts, bootstrap, next_v)
            )

            target_features = self._features(
                target_params, obs_inputs, stop_gradient=True
            )
            target_joint_values = jax.lax.stop_gradient(
                self.joint_model.apply(
                    target_params["joint"], target_features, actions
                )
            )
            primitive_actions = actions.reshape(
                (actions.shape[0], self.action_sequence, self.action_dim)
            )[:, 0]

            def loss_fn(current_params):
                features = self._features(current_params, obs_inputs)
                joint_values = self.joint_model.apply(
                    current_params["joint"], features, actions
                )
                joint_error = jnp.square(joint_values - target_q[:, None])
                masked_joint = joint_error * bootstrap_mask
                joint_per_sample = jnp.sum(masked_joint, axis=-1) / jnp.maximum(
                    jnp.sum(bootstrap_mask, axis=-1), 1.0
                )

                prefix_values, chosen_bins = self._teacher_forced_prefix_values(
                    current_params, features, primitive_actions
                )
                # [B, queries, critics, bins], [B, queries], [B, critics].
                prefix_error = chosen_bin_upper_expectile_loss(
                    prefix_values,
                    chosen_bins,
                    jnp.broadcast_to(
                        target_joint_values[:, None, :],
                        prefix_values.shape[:-1],
                    ),
                    self.prefix_expectile,
                )
                prefix_error = prefix_error * bootstrap_mask[:, None, :]
                prefix_per_sample = jnp.sum(prefix_error, axis=(1, 2)) / jnp.maximum(
                    jnp.sum(bootstrap_mask, axis=-1)
                    * float(self.levels * self.action_dim),
                    1.0,
                )
                denominator = jnp.maximum(jnp.sum(sample_weights), 1.0)
                joint_loss = jnp.sum(joint_per_sample * sample_weights) / denominator
                prefix_loss = jnp.sum(prefix_per_sample * sample_weights) / denominator
                total = joint_loss + prefix_loss
                return total, (
                    joint_loss,
                    prefix_loss,
                    joint_per_sample,
                    joint_values,
                )

            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            joint_loss, prefix_loss, per_sample, joint_values = aux
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = self.optax.apply_updates(params, updates)
            target_params = jax.tree.map(
                lambda target, online: (1.0 - tau) * target + tau * online,
                target_params,
                params,
            )
            priority = jnp.sqrt(per_sample + 1e-10)
            priority = priority / jnp.maximum(jnp.max(priority), 1e-10)
            metrics = {
                "critic_loss": loss,
                "joint_critic_loss": joint_loss,
                "prefix_critic_loss": prefix_loss,
                "joint_q": jnp.mean(joint_values),
                "joint_target_q": jnp.mean(target_q),
                "chunk_valid_fraction": jnp.mean(full_chunk_valid),
                "bootstrap_head_fraction": jnp.mean(bootstrap_mask),
            }
            return params, target_params, opt_state, priority, metrics

        return update_fn

    def _episode_heads(self, batch_size: int) -> np.ndarray:
        if batch_size > self.num_train_envs:
            raise ValueError(
                "Training action batch exceeds configured num_train_envs."
            )
        heads = self._train_episode_heads[:batch_size]
        missing = np.flatnonzero(heads < 0)
        if missing.size:
            self.rng_key, head_key = jax.random.split(self.rng_key)
            sampled = np.asarray(
                jax.device_get(
                    jax.random.randint(
                        head_key,
                        (missing.size,),
                        minval=0,
                        maxval=self.num_critics,
                    )
                ),
                dtype=np.int32,
            )
            heads[missing] = sampled
        return heads.copy()

    def act(self, observations: dict, step: int, eval_mode: bool):
        batch_size = int(next(iter(observations.values())).shape[0])
        obs_inputs = self._prepare_rl_obs_inputs(observations)
        explored = np.zeros((batch_size,), dtype=np.bool_)
        sibling_disagreement = np.full((batch_size,), np.nan, dtype=np.float32)
        sibling_factors = np.full((batch_size,), -1, dtype=np.int32)
        if eval_mode:
            heads = jnp.zeros((batch_size,), dtype=jnp.int32)
            unit_action, _, selected_q = self._eval_policy_impl(
                self.params, obs_inputs, heads, -1, 0
            )
        elif int(step) < self.num_explore_steps:
            self.rng_key, explore_key = jax.random.split(self.rng_key)
            unit_action = jax.random.uniform(
                explore_key,
                (batch_size, self.action_dim),
                minval=-1.0,
                maxval=1.0,
            )
            selected_q = jnp.full((batch_size,), jnp.nan, dtype=jnp.float32)
        else:
            heads_np = self._episode_heads(batch_size)
            heads = jnp.asarray(heads_np)
            unit_action, _, selected_q = self._train_policy_impl(
                self.params,
                obs_inputs,
                heads,
                -1,
                0,
            )
            if self.sibling_exploration_prob > 0.0:
                self.rng_key, explore_key = jax.random.split(self.rng_key)
                explored = np.asarray(
                    jax.device_get(
                        jax.random.bernoulli(
                            explore_key,
                            self.sibling_exploration_prob,
                            (batch_size,),
                        )
                    ),
                    dtype=np.bool_,
                )
                if np.any(explored):
                    factors, offsets, disagreement = self._sibling_rank_impl(
                        self.params,
                        obs_inputs,
                        unit_action,
                    )
                    self._block(factors, offsets, disagreement)
                    factors_np = np.asarray(jax.device_get(factors), dtype=np.int32)
                    offsets_np = np.asarray(jax.device_get(offsets), dtype=np.int32)
                    disagreement_np = np.asarray(
                        jax.device_get(disagreement), dtype=np.float32
                    )
                    unit_action_np = np.asarray(
                        jax.device_get(unit_action), dtype=np.float32
                    ).copy()
                    selected_q_np = np.asarray(
                        jax.device_get(selected_q), dtype=np.float32
                    ).copy()
                    sibling_factors[explored] = factors_np[explored]
                    sibling_disagreement[explored] = disagreement_np[explored]

                    intervention_pairs = np.unique(
                        np.stack(
                            [factors_np[explored], offsets_np[explored]], axis=-1
                        ),
                        axis=0,
                    )
                    for factor, offset in intervention_pairs:
                        rows = np.flatnonzero(
                            explored
                            & (factors_np == int(factor))
                            & (offsets_np == int(offset))
                        )
                        row_obs_inputs = jax.tree.map(
                            lambda value: value[rows], obs_inputs
                        )
                        forced_action, _, forced_q = self._train_policy_impl(
                            self.params,
                            row_obs_inputs,
                            heads[rows],
                            int(factor),
                            int(offset),
                        )
                        self._block(forced_action, forced_q)
                        unit_action_np[rows] = np.asarray(
                            jax.device_get(forced_action), dtype=np.float32
                        )
                        selected_q_np[rows] = np.asarray(
                            jax.device_get(forced_q), dtype=np.float32
                        )
                    unit_action = jnp.asarray(unit_action_np)
                    selected_q = jnp.asarray(selected_q_np)
        action = scale_unit_action(
            unit_action,
            self.primitive_low,
            self.primitive_high,
        )
        self._block(action, selected_q)
        action_np = np.asarray(jax.device_get(action), dtype=np.float32)
        selected_q_np = np.asarray(jax.device_get(selected_q), dtype=np.float32)
        if np.isfinite(selected_q_np).any():
            self._last_selected_q = float(
                selected_q_np[np.isfinite(selected_q_np)].mean()
            )
        else:
            self._last_selected_q = np.nan
        self._last_sibling_explored = float(np.mean(explored))
        self._last_sibling_disagreement = (
            float(np.nanmean(sibling_disagreement)) if np.any(explored) else np.nan
        )
        self._last_sibling_factor = (
            float(np.mean(sibling_factors[explored])) if np.any(explored) else np.nan
        )
        # ActionSequence executes only index zero. A neutral suffix makes it
        # explicit that the method replans at the next environment step; replay
        # assembles the actual consecutive H actions independently.
        command = np.zeros(
            (batch_size, self.action_sequence, self.action_dim), dtype=np.float32
        )
        command[:, 0] = action_np
        return command

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
                "DJCQN replay action shape must be "
                f"[B, {expected_shape[0]}, {expected_shape[1]}], got "
                f"{tuple(actions.shape)}."
            )
        flat_actions = actions.reshape((actions.shape[0], -1))
        action_low, action_high = self._action_bounds()
        unit_actions = unscale_action(flat_actions, action_low, action_high)
        rewards = self._as_jax_array(batch["reward"], self.jnp.float32).reshape(-1)
        discounts = self._as_jax_array(
            batch.get("discount", np.ones_like(batch["reward"])), self.jnp.float32
        ).reshape(-1)
        terminal = self._as_jax_array(batch["terminal"], self.jnp.float32).reshape(-1)
        bootstrap = 1.0 - terminal
        pad_mask = self._extract_action_pad_mask(batch)
        action_valid = (
            jnp.ones((actions.shape[0], self.action_sequence), dtype=jnp.bool_)
            if pad_mask is None
            else jnp.logical_not(pad_mask)
        )
        loss_weights = self._loss_weights(batch)
        self.rng_key, mask_key = jax.random.split(self.rng_key)
        bootstrap_mask = jax.random.bernoulli(
            mask_key, 0.8, (actions.shape[0], self.num_critics)
        ).astype(jnp.float32)
        # Ensure every sample supervises at least one head.
        empty = jnp.sum(bootstrap_mask, axis=-1) == 0
        bootstrap_mask = bootstrap_mask.at[:, 0].set(
            jnp.maximum(bootstrap_mask[:, 0], empty.astype(jnp.float32))
        )

        start_time = time.perf_counter()
        (
            self.params,
            self.target_params,
            self.opt_state,
            priority,
            jax_metrics,
        ) = self._update_impl(
            self.params,
            self.target_params,
            self.opt_state,
            obs_inputs,
            next_obs_inputs,
            unit_actions,
            rewards,
            discounts,
            bootstrap,
            action_valid,
            loss_weights,
            bootstrap_mask,
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
                name: float(np.asarray(jax.device_get(value)))
                for name, value in jax_metrics.items()
            }
            metrics["backend/update_time_sec"] = elapsed
        self._first_update_completed = True
        return metrics

    def reset(self, step: int, agents_to_reset: list[int]):
        del step
        for index in agents_to_reset:
            if 0 <= index < self.num_train_envs:
                self._train_episode_heads[index] = -1

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["target_params"] = self._tree_to_numpy(self.target_params)
        if self.encoder is not None:
            state["encoder_state"] = self._tree_to_numpy(
                self.encoder.frozen_state_dict()
            )
        return state

    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict)
        self.target_params = self._tree_from_numpy(
            state_dict.get("target_params", self.params)
        )
        # Workspace snapshots do not restore environment state.  A resumed
        # rollout therefore starts fresh episodes and must draw fresh heads;
        # only the RNG stream in checkpoint_state_dict persists across resume.
        self._train_episode_heads.fill(-1)
        if self.encoder is not None:
            self.encoder.load_frozen_state_dict(state_dict.get("encoder_state"))

    def rollout_diagnostics(self) -> dict[str, float]:
        return {
            "djcqn_selected_q": float(self._last_selected_q),
            "djcqn_sibling_explored": float(self._last_sibling_explored),
            "djcqn_sibling_disagreement": float(
                self._last_sibling_disagreement
            ),
            "djcqn_sibling_factor": float(self._last_sibling_factor),
        }


__all__ = [
    "DJCQN",
    "DJCQNSpec",
    "PrefixC2FCritic",
    "absolute_topk",
    "c2f_prefix_beam_search",
    "chosen_bin_upper_expectile_loss",
    "djcqn_spec_from_cfg",
    "rank_adjacent_sibling_disagreement",
    "validate_djcqn_config",
]
