"""Coarse-to-fine Q-Network with Action Sequence in pure JAX.

This module extends the local distributional CQN implementation with the
sequence critic from CQN-AS: every coarse-to-fine level predicts bins for all
future sequence positions in parallel, while a GRU shares information along
the sequence axis.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import time
from typing import Iterator, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig

from robobase import utils
from robobase.method.cqn import (
    CQN,
    CQNSpec,
    cqn_spec_from_cfg,
    encode_action,
    progress_shaped_rewards,
    project_categorical,
    zoom_in,
)
from robobase.method.rl_common import (
    JaxRLMethodBase,
    RLModelSpec,
    activation,
    random_shift_rgb,
)
from robobase.replay_buffer.replay_buffer import ReplayBuffer


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
class CQNASpec(CQNSpec):
    """CQN hyperparameters plus the action-sequence architecture settings."""

    demo_fosd: bool
    strict_demo_rl_only: bool
    autoregressive_action_dims: bool
    pessimistic_twin_critic: bool
    auxiliary_td_loss_weight: float
    episodic_twin_head_exploration: bool
    twin_rollout_beam_width: int
    dense_return_q_target: bool
    dense_return_positive_only: bool
    dense_return_expected_q_loss: bool
    dense_return_advantage_alpha: float
    dense_return_advantage_clip_ratio: float | None
    q_reward_scale: float
    dense_return_label_smoothing: float
    dense_return_floor_satisfaction_margin: float | None
    dense_return_relative_floor_margin: float | None
    return_gated_margin: float | None
    return_gated_margin_weight: float
    dense_return_finest_neighbor_weight: float
    episodic_success_q_target: bool
    ordered_success_return_mix: float
    sequence_aligned_mc_discount: float | None
    unseen_return_floor_weight: float
    unseen_return_floor_value: float
    unseen_return_floor_reduction: str
    unseen_return_floor_topk: int
    gru_layers: int
    temporal_ensemble: bool
    temporal_ensemble_replan_interval: int
    temporal_ensemble_gain: float
    tie_break_delta: float
    random_levels_from: int | None
    level_override_mode: str
    post_ensemble_random_keep_levels: int | None
    post_ensemble_fixed_leaf: int | None
    post_ensemble_l1_flip_prob: float
    post_ensemble_l2_flip_prob: float
    post_ensemble_l1_flip_horizon: int
    structured_exploration_prob: float
    structured_exploration_level: int
    structured_exploration_horizon: int
    separate_bc_policy: bool
    bc_policy_stop_gradient: bool
    distinct_policy_encoder: bool
    td_target_action_source: str
    demo_behavior_force_probability: float
    td_target_policy_value_beta: float | None
    critic_sequence_mode: str
    token_split_horizon_targets: bool
    token_split_boundary: int | None
    mc_return_weight: float
    mc_lower_bound_target: bool
    mc_return_stop_gradient_encoder: bool
    mc_return_value_only: bool
    policy_value_beta: float | None
    cv_rct_weight: float | None
    cv_rct_level: int | None
    cv_rct_baseline: str
    awr_beta: float | None
    awr_weight_max: float
    awr_expectile_tau: float
    progress_potential_weight: float
    progress_potential_schedule: str | None
    progress_head_weight: float
    progress_expectile_tau: float
    progress_success_gated: bool
    flow_policy: bool
    flow_policy_candidates: int
    flow_policy_steps: int
    flow_policy_lambda: float
    flow_policy_ema: float | None
    flow_policy_hidden_dims: tuple[int, ...] | None
    flow_policy_gru_layers: int | None
    coarse_flow: bool
    coarse_flow_pure: bool
    coarse_flow_selfdistill_weight: float | None
    coarse_flow_selfdistill_threshold: float
    bin_flip_prob: float
    bin_flip_level: int | None
    bin_explore_probs: tuple[float, ...] | None
    bin_explore_schedule: str | None
    bin_explore_persist_plans: int | None
    low_dim_mask_prob: float
    low_dim_mask_keep_last: int
    use_frozen_support_mask: bool
    support_mask_decode: bool
    support_mask_tau: float
    support_mask_freeze_step: int


def cqn_as_spec_from_cfg(cfg: DictConfig) -> CQNASpec:
    method = cfg.method
    auxiliary_td_loss_weight = float(
        method.get("auxiliary_td_loss_weight", 0.0)
    )
    if auxiliary_td_loss_weight < 0.0:
        raise ValueError("method.auxiliary_td_loss_weight must be non-negative.")
    if auxiliary_td_loss_weight > 0.0:
        auxiliary_nstep = cfg.replay.get("auxiliary_nstep", None)
        auxiliary_violations = []
        if not bool(method.get("pessimistic_twin_critic", False)):
            auxiliary_violations.append("method.pessimistic_twin_critic=true")
        if int(cfg.replay.get("nstep", 1)) != 1:
            auxiliary_violations.append("replay.nstep=1")
        if auxiliary_nstep is None or int(auxiliary_nstep) <= 1:
            auxiliary_violations.append("replay.auxiliary_nstep > 1")
        if not bool(cfg.replay.get("include_tp1", True)):
            auxiliary_violations.append("replay.include_tp1=true")
        if not bool(cfg.replay.get("include_next_action", False)):
            auxiliary_violations.append("replay.include_next_action=true")
        if auxiliary_violations:
            raise ValueError(
                "auxiliary TD requires the matched 1-step + n-step twin-C51 "
                "path: " + "; ".join(auxiliary_violations)
            )
    token_split_horizon_targets = bool(
        method.get("token_split_horizon_targets", False)
    )
    if token_split_horizon_targets:
        token_split_nstep = cfg.replay.get("auxiliary_nstep", None)
        token_split_violations = []
        if int(cfg.replay.get("nstep", 1)) != 1:
            token_split_violations.append("replay.nstep=1")
        if token_split_nstep is None or int(token_split_nstep) <= 1:
            token_split_violations.append("replay.auxiliary_nstep > 1")
        if not bool(cfg.replay.get("include_tp1", True)):
            token_split_violations.append("replay.include_tp1=true")
        if token_split_violations:
            raise ValueError(
                "token_split_horizon_targets requires the auxiliary-horizon "
                "replay fields: " + "; ".join(token_split_violations)
            )
    progress_enabled = (
        float(method.get("progress_potential_weight", 0.0)) > 0.0
        or float(method.get("progress_head_weight", 0.0)) > 0.0
    )
    if progress_enabled:
        # Only the canonical / separate-BC CQN-AS update graphs thread the
        # potential; every other graph would silently ignore it.
        method_name = str(method.get("name", "cqn_as")).lower()
        if method_name != "cqn_as":
            raise NotImplementedError(
                "progress shaping is implemented for method=cqn_as only; got "
                f"method.name={method_name}."
            )
        if bool(method.get("direct_scalar_q", False)):
            raise NotImplementedError(
                "progress shaping is not implemented on the direct scalar-Q "
                "update graph."
            )
        # The (t+1)/T label is only progress toward success when the demo
        # episode ends at its first success frame; 96% of untruncated BiGym
        # demo transitions sit in a post-success tail where the label is flat.
        if "env" in cfg:
            env_cfg = cfg.env
            if str(env_cfg.get("env_name", "")) == "bigym" and not bool(
                env_cfg.get("truncate_demo_at_success", False)
            ):
                raise ValueError(
                    "progress labels require env.truncate_demo_at_success="
                    "true; untruncated demo tails make (t+1)/T flat and "
                    "misleading."
                )
        from robobase.replay_buffer.bigym_lazy_replay import (
            lazy_replay_enabled,
        )

        if lazy_replay_enabled(cfg):
            raise ValueError(
                "progress labels require episode-backed replay; set "
                "lazy_replay.use=false."
            )
    strict_demo_rl_only = bool(method.get("strict_demo_rl_only", False))
    if strict_demo_rl_only:
        violations = []
        if bool(cfg.get("is_imitation_learning", False)):
            violations.append("is_imitation_learning must be false")
        if bool(cfg.get("use_self_imitation", False)) and not bool(
            method.get("strict_allow_reward_only_success_replay", False)
        ):
            violations.append(
                "use_self_imitation requires "
                "method.strict_allow_reward_only_success_replay=true"
            )
        forbidden_nonzero = {
            "bc_lambda": method.get("bc_lambda", 0.0),
            "bc_margin": method.get("bc_margin", 0.0),
            "causal_rct_weight": method.get("causal_rct_weight", 0.0),
        }
        for name, value in forbidden_nonzero.items():
            if float(value) != 0.0:
                violations.append(f"method.{name} must be 0")
        forbidden_true = (
            "demo_fosd",
            "separate_bc_policy",
            "bc_policy_stop_gradient",
            "distinct_policy_encoder",
            "flow_policy",
            "coarse_flow",
            "coarse_flow_pure",
            "freeze_bc_policy",
            "direct_scalar_q",
            "use_frozen_support_mask",
        )
        for name in forbidden_true:
            if bool(method.get(name, False)):
                violations.append(f"method.{name} must be false")
        forbidden_optional = (
            "bc_lambda_schedule",
            "td_target_policy_value_beta",
            "policy_value_beta",
            "cv_rct_weight",
            "awr_beta",
            "coarse_flow_selfdistill_weight",
        )
        for name in forbidden_optional:
            if method.get(name, None) is not None:
                violations.append(f"method.{name} must be null")
        td_target_action_source = str(
            method.get("td_target_action_source", "critic")
        ).lower()
        if td_target_action_source not in {
            "critic",
            "replay_next",
            "critic_replay_max",
        }:
            violations.append(
                "method.td_target_action_source must be critic, replay_next, "
                "or critic_replay_max"
            )
        if (
            td_target_action_source == "critic_replay_max"
            and not bool(cfg.replay.get("include_next_action", False))
        ):
            violations.append(
                "replay.include_next_action must be true for critic_replay_max"
            )
        demo_behavior_force_probability = float(
            method.get("demo_behavior_force_probability", 0.0)
        )
        if not 0.0 <= demo_behavior_force_probability <= 1.0:
            violations.append(
                "method.demo_behavior_force_probability must be in [0, 1]"
            )
        if (
            demo_behavior_force_probability > 0.0
            and td_target_action_source != "critic_replay_max"
        ):
            violations.append(
                "method.demo_behavior_force_probability > 0 requires "
                "td_target_action_source=critic_replay_max"
            )
        if violations:
            raise ValueError(
                "strict_demo_rl_only forbids imitation/non-RL paths: "
                + "; ".join(violations)
            )
    base = cqn_spec_from_cfg(cfg)
    base_values = {field.name: getattr(base, field.name) for field in fields(CQNSpec)}
    policy_value_beta = method.get("policy_value_beta", None)
    td_target_policy_value_beta = method.get(
        "td_target_policy_value_beta",
        None,
    )
    return CQNASpec(
        **base_values,
        demo_fosd=bool(method.get("demo_fosd", True)),
        strict_demo_rl_only=strict_demo_rl_only,
        autoregressive_action_dims=bool(
            method.get("autoregressive_action_dims", False)
        ),
        pessimistic_twin_critic=bool(
            method.get("pessimistic_twin_critic", False)
        ),
        auxiliary_td_loss_weight=auxiliary_td_loss_weight,
        episodic_twin_head_exploration=bool(
            method.get("episodic_twin_head_exploration", False)
        ),
        twin_rollout_beam_width=int(
            method.get("twin_rollout_beam_width", 1)
        ),
        dense_return_q_target=bool(
            method.get("dense_return_q_target", False)
        ),
        dense_return_positive_only=bool(
            method.get("dense_return_positive_only", False)
        ),
        dense_return_expected_q_loss=bool(
            method.get("dense_return_expected_q_loss", False)
        ),
        dense_return_advantage_alpha=float(
            method.get("dense_return_advantage_alpha", 0.0)
        ),
        dense_return_advantage_clip_ratio=(
            None
            if method.get("dense_return_advantage_clip_ratio", None) is None
            else float(method.dense_return_advantage_clip_ratio)
        ),
        q_reward_scale=float(method.get("q_reward_scale", 1.0)),
        dense_return_label_smoothing=float(
            method.get("dense_return_label_smoothing", 0.0)
        ),
        dense_return_floor_satisfaction_margin=(
            None
            if method.get("dense_return_floor_satisfaction_margin", None)
            is None
            else float(
                method.get("dense_return_floor_satisfaction_margin")
            )
        ),
        dense_return_relative_floor_margin=(
            None
            if method.get("dense_return_relative_floor_margin", None)
            is None
            else float(method.get("dense_return_relative_floor_margin"))
        ),
        return_gated_margin=(
            None
            if method.get("return_gated_margin", None) is None
            else float(method.get("return_gated_margin"))
        ),
        return_gated_margin_weight=float(
            method.get("return_gated_margin_weight", 0.0)
        ),
        dense_return_finest_neighbor_weight=float(
            method.get("dense_return_finest_neighbor_weight", 0.0)
        ),
        episodic_success_q_target=bool(
            method.get("episodic_success_q_target", False)
        ),
        ordered_success_return_mix=float(
            method.get("ordered_success_return_mix", 0.0)
        ),
        sequence_aligned_mc_discount=(
            None
            if method.get("sequence_aligned_mc_discount", None) is None
            else float(method.sequence_aligned_mc_discount)
        ),
        unseen_return_floor_weight=float(
            method.get("unseen_return_floor_weight", 0.0)
        ),
        unseen_return_floor_value=float(
            method.get("unseen_return_floor_value", 0.0)
        ),
        unseen_return_floor_reduction=str(
            method.get("unseen_return_floor_reduction", "mean")
        ).lower(),
        unseen_return_floor_topk=int(
            method.get("unseen_return_floor_topk", 1)
        ),
        gru_layers=int(method.get("gru_layers", 1)),
        temporal_ensemble=bool(method.get("temporal_ensemble", True)),
        temporal_ensemble_replan_interval=int(
            method.get("temporal_ensemble_replan_interval", 1)
        ),
        temporal_ensemble_gain=float(method.get("temporal_ensemble_gain", 0.01)),
        tie_break_delta=float(method.get("tie_break_delta", 1e-4)),
        random_levels_from=(
            None
            if method.get("random_levels_from", None) is None
            else int(method.random_levels_from)
        ),
        level_override_mode=str(
            method.get("level_override_mode", "random")
        ),
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
        structured_exploration_prob=float(
            method.get("structured_exploration_prob", 0.0)
        ),
        structured_exploration_level=int(
            method.get("structured_exploration_level", 1)
        ),
        structured_exploration_horizon=int(
            method.get("structured_exploration_horizon", 1)
        ),
        separate_bc_policy=bool(method.get("separate_bc_policy", False)),
        bc_policy_stop_gradient=bool(
            method.get("bc_policy_stop_gradient", False)
        ),
        distinct_policy_encoder=bool(
            method.get("distinct_policy_encoder", False)
        ),
        td_target_action_source=str(
            method.get("td_target_action_source", "critic")
        ).lower(),
        demo_behavior_force_probability=float(
            method.get("demo_behavior_force_probability", 0.0)
        ),
        td_target_policy_value_beta=(
            None
            if td_target_policy_value_beta is None
            else float(td_target_policy_value_beta)
        ),
        critic_sequence_mode=str(
            method.get("critic_sequence_mode", "full")
        ).lower(),
        token_split_horizon_targets=token_split_horizon_targets,
        token_split_boundary=(
            None
            if method.get("token_split_boundary", None) is None
            else int(method.get("token_split_boundary"))
        ),
        mc_return_weight=float(method.get("mc_return_weight", 0.0)),
        mc_lower_bound_target=bool(
            method.get("mc_lower_bound_target", False)
        ),
        mc_return_stop_gradient_encoder=bool(
            method.get("mc_return_stop_gradient_encoder", False)
        ),
        mc_return_value_only=bool(method.get("mc_return_value_only", False)),
        policy_value_beta=(
            None if policy_value_beta is None else float(policy_value_beta)
        ),
        cv_rct_weight=(
            None
            if method.get("cv_rct_weight", None) is None
            else float(method.get("cv_rct_weight"))
        ),
        cv_rct_level=(
            None
            if method.get("cv_rct_level", None) is None
            else int(method.get("cv_rct_level"))
        ),
        cv_rct_baseline=str(method.get("cv_rct_baseline", "target_q")).lower(),
        awr_beta=(
            None
            if method.get("awr_beta", None) is None
            else float(method.get("awr_beta"))
        ),
        awr_weight_max=float(method.get("awr_weight_max", 10.0)),
        awr_expectile_tau=float(method.get("awr_expectile_tau", 0.7)),
        progress_potential_weight=float(
            method.get("progress_potential_weight", 0.0)
        ),
        progress_potential_schedule=(
            None
            if method.get("progress_potential_schedule", None) is None
            else str(method.get("progress_potential_schedule"))
        ),
        progress_head_weight=float(method.get("progress_head_weight", 0.0)),
        progress_expectile_tau=float(
            method.get("progress_expectile_tau", 0.9)
        ),
        progress_success_gated=bool(
            method.get("progress_success_gated", True)
        ),
        flow_policy=bool(method.get("flow_policy", False)),
        flow_policy_candidates=int(method.get("flow_policy_candidates", 8)),
        flow_policy_steps=int(method.get("flow_policy_steps", 8)),
        flow_policy_lambda=float(method.get("flow_policy_lambda", 1.0)),
        flow_policy_ema=(
            None
            if method.get("flow_policy_ema", None) is None
            else float(method.get("flow_policy_ema"))
        ),
        flow_policy_hidden_dims=(
            None
            if method.get("flow_policy_hidden_dims", None) is None
            else tuple(
                int(v) for v in method.get("flow_policy_hidden_dims")
            )
        ),
        flow_policy_gru_layers=(
            None
            if method.get("flow_policy_gru_layers", None) is None
            else int(method.get("flow_policy_gru_layers"))
        ),
        coarse_flow=bool(method.get("coarse_flow", False)),
        coarse_flow_pure=bool(method.get("coarse_flow_pure", False)),
        coarse_flow_selfdistill_weight=(
            None
            if method.get("coarse_flow_selfdistill_weight", None) is None
            else float(method.get("coarse_flow_selfdistill_weight"))
        ),
        coarse_flow_selfdistill_threshold=float(
            method.get("coarse_flow_selfdistill_threshold", 0.5)
        ),
        bin_flip_prob=float(method.get("bin_flip_prob", 0.0)),
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
        low_dim_mask_keep_last=int(
            method.get("low_dim_mask_keep_last", 0)
        ),
        bin_flip_level=(
            None
            if method.get("bin_flip_level", None) is None
            else int(method.get("bin_flip_level"))
        ),
        use_frozen_support_mask=bool(
            method.get("use_frozen_support_mask", False)
        ),
        support_mask_decode=bool(method.get("support_mask_decode", True)),
        support_mask_tau=float(method.get("support_mask_tau", 0.3)),
        support_mask_freeze_step=int(
            method.get("support_mask_freeze_step", 10000)
        ),
    )


def action_centered_moment_loss(
    treatment_effect,
    outcome,
    treated,
    propensity,
    valid,
    sample_weight,
):
    """Action-centered squared loss with the outcome-only constant removed.

    For randomized ``Z ~ Bernoulli(p)``, the conditional population minimizer
    is ``E[Y(1) - Y(0) | state, proposed_action]`` even when the state-only
    baseline outcome is arbitrarily complex:

    ``p(1-p) tau^2 - 2 (Z-p) Y tau``.

    Replacing ``Y`` with ``Y - b(pre-treatment covariates)`` keeps the
    minimizer unchanged because ``E[(Z-p) b] = 0`` under randomization, while
    shrinking the gradient variance by the measured 10-15x factor when ``b``
    is an MC-calibrated value baseline (cqn-flow.md section 22).
    """

    tau = jnp.asarray(treatment_effect, dtype=jnp.float32)
    y = jnp.asarray(outcome, dtype=jnp.float32)
    z = jnp.asarray(treated, dtype=jnp.float32)
    p = jnp.asarray(propensity, dtype=jnp.float32)
    mask = jnp.asarray(valid, dtype=jnp.float32)
    weight = jnp.asarray(sample_weight, dtype=jnp.float32) * mask
    per_sample = (
        p * (1.0 - p) * jnp.square(tau)
        - 2.0 * (z - p) * y * tau
    )
    return jnp.sum(weight * per_sample) / jnp.maximum(
        jnp.sum(weight),
        1.0,
    )


class ExpectileValueHead(nn.Module):
    """Scalar state-value head for IQL-style expectile regression.

    Reads (stop-gradient) encoder features only; it never queries actions, so
    it cannot leak counterfactual claims into the behavior policy.  Used by
    the AWR-weighted BC path (cqn-flow.md section 26.2).
    """

    hidden_dims: tuple[int, ...]
    activation_name: str = "silu"

    @nn.compact
    def __call__(self, features: jax.Array) -> jax.Array:
        x = features
        for index, width in enumerate(self.hidden_dims):
            x = nn.Dense(
                width,
                use_bias=False,
                kernel_init=nn.initializers.orthogonal(),
                name=f"value_dense_{index}",
            )(x)
            x = nn.LayerNorm(name=f"value_norm_{index}")(x)
            x = activation(x, self.activation_name)
        value = nn.Dense(
            1,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="value_out",
        )(x)
        return value[..., 0]


class FlowPolicyHead(nn.Module):
    """Conditional flow-matching velocity field over action chunks.

    Behavior-side flow for the flow+CQN line (cqn-flow.md section 29): the
    flow proposes demonstration-style chunks; the calibrated critic only
    reranks among them, so Q is queried exactly on the manifold where
    Stage-142 measured it to be reliable.  Forward-time convention:
    ``x_t = (1-t) x0 + t x1`` with target velocity ``x1 - x0``.

    v1b (cqn-flow.md 29.6): velocity factorized per sequence step.  v1c
    (29.7): the head additionally mirrors the categorical policy tower --
    per-stream rgb/low-dim projections and a GRU along the sequence --
    because the flat raw-feature MLP was the measured sampler bottleneck
    (flow BC alone 8% vs categorical BC 62%+).

    Coarse-flow mode (cqn-flow.md 34) additionally passes ``bin_context``
    -- the critic-selected cell's per-level bin one-hots plus normalized
    cell center, per sequence step -- and the field then models the
    within-cell residual in [-1, 1] coordinates instead of the full
    action.  ``bin_context=None`` keeps the legacy parameter shapes.
    """

    hidden_dims: tuple[int, ...]
    action_sequence: int
    action_dim: int
    low_dim_size: int = 0
    feature_dim: int = 64
    rgb_encoder_layers: int = 2
    gru_layers: int = 1
    activation_name: str = "silu"

    @nn.compact
    def __call__(
        self,
        features: jax.Array,
        x_t: jax.Array,
        time: jax.Array,
        bin_context: jax.Array | None = None,
    ) -> jax.Array:
        batch = features.shape[0]
        stream_features = features
        if 0 < self.low_dim_size < features.shape[-1]:
            low_dim = features[:, : self.low_dim_size]
            rgb = features[:, self.low_dim_size :]
            for index in range(self.rgb_encoder_layers):
                rgb = nn.Dense(
                    self.hidden_dims[0],
                    use_bias=False,
                    kernel_init=nn.initializers.orthogonal(),
                    name=f"flow_rgb_dense_{index}",
                )(rgb)
                rgb = nn.LayerNorm(name=f"flow_rgb_norm_{index}")(rgb)
                rgb = activation(rgb, self.activation_name)
            rgb = nn.Dense(
                self.feature_dim,
                use_bias=False,
                kernel_init=nn.initializers.orthogonal(),
                name="flow_rgb_projection",
            )(rgb)
            rgb = nn.LayerNorm(name="flow_rgb_projection_norm")(rgb)
            rgb = jnp.tanh(rgb)
            low_dim = nn.Dense(
                self.feature_dim,
                use_bias=False,
                kernel_init=nn.initializers.orthogonal(),
                name="flow_low_dim_projection",
            )(low_dim)
            low_dim = nn.LayerNorm(name="flow_low_dim_norm")(low_dim)
            low_dim = jnp.tanh(low_dim)
            stream_features = jnp.concatenate([rgb, low_dim], axis=-1)

        time = jnp.reshape(time, (-1, 1)).astype(jnp.float32)
        two_pi = 2.0 * jnp.pi
        time_embedding = jnp.concatenate(
            [
                time,
                jnp.sin(two_pi * time),
                jnp.cos(two_pi * time),
                jnp.sin(2.0 * two_pi * time),
                jnp.cos(2.0 * two_pi * time),
            ],
            axis=-1,
        )
        steps = x_t.reshape(
            (batch, self.action_sequence, self.action_dim)
        )
        sequence_one_hot = jnp.broadcast_to(
            jnp.eye(self.action_sequence, dtype=jnp.float32)[None],
            (batch, self.action_sequence, self.action_sequence),
        )
        parts = [
            jnp.broadcast_to(
                stream_features[:, None, :],
                (
                    batch,
                    self.action_sequence,
                    stream_features.shape[-1],
                ),
            ),
            steps,
            jnp.broadcast_to(
                time_embedding[:, None, :],
                (
                    batch,
                    self.action_sequence,
                    time_embedding.shape[-1],
                ),
            ),
            sequence_one_hot,
        ]
        if bin_context is not None:
            parts.append(bin_context.astype(jnp.float32))
        x = jnp.concatenate(parts, axis=-1)
        for index, width in enumerate(self.hidden_dims):
            x = nn.Dense(
                width,
                use_bias=False,
                kernel_init=nn.initializers.orthogonal(),
                name=f"flow_dense_{index}",
            )(x)
            x = nn.LayerNorm(name=f"flow_norm_{index}")(x)
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
                (batch, hidden_size),
                dtype=x.dtype,
            )
            scan_gru = ScanGRU(
                features=hidden_size,
                name=f"flow_gru_{layer}",
            )
            _, x = scan_gru(initial_carry, x)
        velocity = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="flow_velocity_out",
        )(x)
        return velocity.reshape(
            (batch, self.action_sequence * self.action_dim)
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
        return_streams: bool = False,
    ) -> jax.Array | tuple[jax.Array, jax.Array, jax.Array]:
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
        centered_advantages = advantages - advantages.mean(
            axis=-2,
            keepdims=True,
        )
        combined = values + centered_advantages
        if return_streams:
            return combined, values, centered_advantages
        return combined


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
    """Legacy CQN-AS critic plus a causal action-dimension Q correction."""

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
        return_streams: bool = False,
    ) -> jax.Array | tuple[jax.Array, jax.Array, jax.Array]:
        base_logits, values, _ = self.base_critic(
            features,
            level_one_hot,
            low_high_midpoint,
            return_streams=True,
        )
        correction = self.action_correction(
            base_logits,
            features,
            action_context,
        )
        combined = base_logits + correction
        if return_streams:
            return combined, values, combined - values
        return combined

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
        structured_exploration_prob: float,
        structured_exploration_level: int,
        structured_exploration_horizon: int,
        separate_bc_policy: bool,
        bc_policy_stop_gradient: bool,
        distinct_policy_encoder: bool,
        td_target_action_source: str,
        demo_behavior_force_probability: float,
        td_target_policy_value_beta: float | None,
        critic_sequence_mode: str,
        mc_return_weight: float,
        mc_lower_bound_target: bool,
        mc_return_stop_gradient_encoder: bool,
        mc_return_value_only: bool,
        policy_value_beta: float | None,
        strict_demo_rl_only: bool,
        autoregressive_action_dims: bool,
        pessimistic_twin_critic: bool,
        auxiliary_td_loss_weight: float,
        episodic_twin_head_exploration: bool,
        twin_rollout_beam_width: int,
        dense_return_q_target: bool,
        dense_return_positive_only: bool,
        dense_return_expected_q_loss: bool,
        dense_return_advantage_alpha: float,
        dense_return_advantage_clip_ratio: float | None,
        q_reward_scale: float,
        dense_return_label_smoothing: float,
        dense_return_floor_satisfaction_margin: float | None,
        dense_return_relative_floor_margin: float | None,
        return_gated_margin: float | None,
        return_gated_margin_weight: float,
        dense_return_finest_neighbor_weight: float,
        episodic_success_q_target: bool,
        ordered_success_return_mix: float,
        sequence_aligned_mc_discount: float | None,
        unseen_return_floor_weight: float,
        unseen_return_floor_value: float,
        unseen_return_floor_reduction: str,
        unseen_return_floor_topk: int,
        model: RLModelSpec,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        num_train_envs: int,
        num_eval_envs: int,
        replay_alpha: float,
        replay_beta: float,
        frame_stack_on_channel: bool,
        intrinsic_reward_module=None,
        token_split_horizon_targets: bool = False,
        token_split_boundary: int | None = None,
        demo_fosd: bool = True,
        critic_grad_clip: Optional[float] = None,
        jit: bool = True,
        platform: str | None = None,
        seed: int = 0,
        update_block_every_steps: int = 1,
        cv_rct_weight: float | None = None,
        cv_rct_level: int | None = None,
        cv_rct_baseline: str = "target_q",
        awr_beta: float | None = None,
        awr_weight_max: float = 10.0,
        awr_expectile_tau: float = 0.7,
        progress_potential_weight: float = 0.0,
        progress_potential_schedule: str | None = None,
        progress_head_weight: float = 0.0,
        progress_expectile_tau: float = 0.9,
        progress_success_gated: bool = True,
        flow_policy: bool = False,
        flow_policy_candidates: int = 8,
        flow_policy_steps: int = 8,
        flow_policy_lambda: float = 1.0,
        flow_policy_ema: float | None = None,
        flow_policy_hidden_dims: tuple[int, ...] | None = None,
        flow_policy_gru_layers: int | None = None,
        coarse_flow: bool = False,
        coarse_flow_pure: bool = False,
        coarse_flow_selfdistill_weight: float | None = None,
        coarse_flow_selfdistill_threshold: float = 0.5,
        bc_lambda_schedule: str | None = None,
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
        use_frozen_support_mask: bool = False,
        support_mask_decode: bool = True,
        support_mask_tau: float = 0.3,
        support_mask_freeze_step: int = 10000,
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
        if self.action_sequence < 1:
            raise ValueError("CQN-AS requires action_sequence >= 1.")
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
        if not 0.0 <= structured_exploration_prob <= 1.0:
            raise ValueError("structured_exploration_prob must be in [0, 1].")
        if not 0 <= structured_exploration_level < levels:
            raise ValueError(
                "structured_exploration_level must be in [0, levels)."
            )
        if structured_exploration_horizon < 1:
            raise ValueError(
                "structured_exploration_horizon must be at least 1."
            )
        td_target_action_source = str(td_target_action_source).lower()
        if td_target_action_source not in {
            "critic",
            "replay_next",
            "critic_replay_max",
            "bc_policy",
            "policy_value",
        }:
            raise ValueError(
                "td_target_action_source must be one of "
                "{'critic', 'replay_next', 'critic_replay_max', "
                "'bc_policy', 'policy_value'}."
            )
        critic_sequence_mode = str(critic_sequence_mode).lower()
        if critic_sequence_mode not in {"full", "effective_k0"}:
            raise ValueError(
                "critic_sequence_mode must be one of {'full', 'effective_k0'}."
            )
        token_split_horizon_targets = bool(token_split_horizon_targets)
        if token_split_horizon_targets:
            token_split_violations = []
            if pessimistic_twin_critic:
                token_split_violations.append("pessimistic_twin_critic=false")
            if float(auxiliary_td_loss_weight) > 0.0:
                token_split_violations.append("auxiliary_td_loss_weight=0")
            if separate_bc_policy:
                token_split_violations.append("separate_bc_policy=false")
            if str(td_target_action_source).lower() != "critic":
                token_split_violations.append("td_target_action_source=critic")
            if dense_return_q_target:
                token_split_violations.append("dense_return_q_target=false")
            if episodic_success_q_target:
                token_split_violations.append(
                    "episodic_success_q_target=false"
                )
            if mc_lower_bound_target:
                token_split_violations.append("mc_lower_bound_target=false")
            if sequence_aligned_mc_discount is not None:
                token_split_violations.append(
                    "sequence_aligned_mc_discount=null"
                )
            if critic_sequence_mode != "full":
                token_split_violations.append("critic_sequence_mode=full")
            if int(self.action_sequence) < 2:
                token_split_violations.append("action_sequence >= 2")
            if token_split_violations:
                raise ValueError(
                    "token_split_horizon_targets requires: "
                    + "; ".join(token_split_violations)
                )
            if token_split_boundary is None:
                raise ValueError(
                    "token_split_horizon_targets requires an explicit "
                    "token_split_boundary (1-based token index at or below "
                    "which the exact legacy 1-step backup is kept)."
                )
            token_split_boundary = int(token_split_boundary)
            if not 1 <= token_split_boundary < int(self.action_sequence):
                raise ValueError(
                    "token_split_boundary must lie in [1, action_sequence)."
                )
        if (
            not separate_bc_policy
            and td_target_action_source
            not in {"critic", "replay_next", "critic_replay_max"}
        ):
            raise ValueError(
                "td_target_action_source requires separate_bc_policy=true "
                "unless it is critic, replay_next, or critic_replay_max."
            )
        if separate_bc_policy and td_target_action_source == "critic_replay_max":
            raise ValueError(
                "critic_replay_max is implemented only for the single-objective "
                "critic path."
            )
        if not 0.0 <= demo_behavior_force_probability <= 1.0:
            raise ValueError(
                "demo_behavior_force_probability must be in [0, 1]."
            )
        if (
            demo_behavior_force_probability > 0.0
            and td_target_action_source != "critic_replay_max"
        ):
            raise ValueError(
                "demo_behavior_force_probability > 0 requires "
                "td_target_action_source=critic_replay_max."
            )
        if (
            td_target_policy_value_beta is not None
            and td_target_policy_value_beta < 0.0
        ):
            raise ValueError(
                "td_target_policy_value_beta must be non-negative or null."
            )
        if (
            td_target_action_source == "policy_value"
            and td_target_policy_value_beta is None
        ):
            raise ValueError(
                "td_target_action_source=policy_value requires "
                "td_target_policy_value_beta."
            )
        if (
            td_target_action_source != "policy_value"
            and td_target_policy_value_beta is not None
        ):
            raise ValueError(
                "td_target_policy_value_beta is only valid when "
                "td_target_action_source=policy_value."
            )
        if separate_bc_policy and bc_lambda <= 0.0:
            raise ValueError("separate_bc_policy=true requires bc_lambda > 0.")
        if autoregressive_action_dims and separate_bc_policy:
            raise ValueError(
                "autoregressive_action_dims requires "
                "separate_bc_policy=false."
            )
        if dense_return_q_target and separate_bc_policy:
            raise ValueError(
                "dense_return_q_target requires separate_bc_policy=false."
            )
        if dense_return_positive_only and not dense_return_q_target:
            raise ValueError(
                "dense_return_positive_only requires "
                "dense_return_q_target=true."
            )
        if dense_return_positive_only and not mc_lower_bound_target:
            raise ValueError(
                "dense_return_positive_only requires "
                "mc_lower_bound_target=true so completed returns are present."
            )
        if use_frozen_support_mask:
            support_mask_violations = []
            if separate_bc_policy:
                support_mask_violations.append("separate_bc_policy=false")
            if autoregressive_action_dims:
                support_mask_violations.append(
                    "autoregressive_action_dims=false"
                )
            if pessimistic_twin_critic:
                support_mask_violations.append(
                    "pessimistic_twin_critic=false"
                )
            if twin_rollout_beam_width != 1:
                support_mask_violations.append("twin_rollout_beam_width=1")
            if coarse_flow:
                support_mask_violations.append("coarse_flow=false")
            if dense_return_expected_q_loss:
                support_mask_violations.append(
                    "dense_return_expected_q_loss=false"
                )
            if support_mask_violations:
                raise ValueError(
                    "use_frozen_support_mask requires the canonical "
                    "single-critic decode path: "
                    + "; ".join(support_mask_violations)
                )
        if not 0.0 < support_mask_tau <= 1.0:
            raise ValueError("support_mask_tau must be in (0, 1].")
        if support_mask_freeze_step < 0:
            raise ValueError(
                "support_mask_freeze_step must be non-negative."
            )
        if dense_return_expected_q_loss and not dense_return_q_target:
            raise ValueError(
                "dense_return_expected_q_loss requires "
                "dense_return_q_target=true."
            )
        if not 0.0 <= dense_return_advantage_alpha < 1.0:
            raise ValueError(
                "dense_return_advantage_alpha must be in [0, 1)."
            )
        if (
            dense_return_advantage_alpha > 0.0
            and not dense_return_q_target
        ):
            raise ValueError(
                "dense_return_advantage_alpha requires "
                "dense_return_q_target=true."
            )
        if (
            dense_return_advantage_alpha > 0.0
            and dense_return_expected_q_loss
        ):
            raise ValueError(
                "dense_return_advantage_alpha requires "
                "dense_return_expected_q_loss=false."
            )
        if dense_return_advantage_clip_ratio is not None:
            if not 0.0 < dense_return_advantage_clip_ratio < 1.0:
                raise ValueError(
                    "dense_return_advantage_clip_ratio must be in (0, 1)."
                )
            if dense_return_advantage_alpha <= 0.0:
                raise ValueError(
                    "dense_return_advantage_clip_ratio requires "
                    "dense_return_advantage_alpha > 0."
                )
        if q_reward_scale <= 0.0:
            raise ValueError("q_reward_scale must be positive.")
        if q_reward_scale != 1.0:
            if not dense_return_q_target:
                raise ValueError(
                    "q_reward_scale != 1 requires "
                    "dense_return_q_target=true."
                )
            if not mc_lower_bound_target:
                raise ValueError(
                    "q_reward_scale != 1 requires "
                    "mc_lower_bound_target=true."
                )
            if episodic_success_q_target:
                raise ValueError(
                    "q_reward_scale != 1 is incompatible with "
                    "episodic_success_q_target."
                )
            if ordered_success_return_mix > 0.0:
                raise ValueError(
                    "q_reward_scale != 1 is incompatible with "
                    "ordered_success_return_mix."
                )
            if sequence_aligned_mc_discount is not None:
                raise ValueError(
                    "q_reward_scale != 1 is incompatible with "
                    "sequence_aligned_mc_discount."
                )
            if not v_min <= q_reward_scale <= v_max:
                raise ValueError(
                    "q_reward_scale terminal target must lie on "
                    "the C51 support."
                )
        if not 0.0 <= dense_return_label_smoothing < 1.0:
            raise ValueError(
                "dense_return_label_smoothing must be in [0, 1)."
            )
        if dense_return_floor_satisfaction_margin is not None:
            if dense_return_floor_satisfaction_margin < 0.0:
                raise ValueError(
                    "dense_return_floor_satisfaction_margin must be "
                    "nonnegative."
                )
            if not dense_return_q_target:
                raise ValueError(
                    "dense_return_floor_satisfaction_margin requires "
                    "dense_return_q_target=true."
                )
            if dense_return_label_smoothing > 0.0:
                raise ValueError(
                    "dense_return_floor_satisfaction_margin is "
                    "incompatible with dense_return_label_smoothing."
                )
            if dense_return_advantage_alpha > 0.0:
                raise ValueError(
                    "dense_return_floor_satisfaction_margin is "
                    "incompatible with dense_return_advantage_alpha."
                )
        if (
            dense_return_label_smoothing > 0.0
            and not dense_return_q_target
        ):
            raise ValueError(
                "dense_return_label_smoothing requires "
                "dense_return_q_target=true."
            )
        if dense_return_relative_floor_margin is not None:
            if dense_return_relative_floor_margin <= 0.0:
                raise ValueError(
                    "dense_return_relative_floor_margin must be positive."
                )
            if not dense_return_q_target:
                raise ValueError(
                    "dense_return_relative_floor_margin requires "
                    "dense_return_q_target=true."
                )
            if dense_return_floor_satisfaction_margin is not None:
                raise ValueError(
                    "dense_return_relative_floor_margin is incompatible "
                    "with dense_return_floor_satisfaction_margin."
                )
            if dense_return_advantage_alpha > 0.0:
                raise ValueError(
                    "dense_return_relative_floor_margin is incompatible "
                    "with dense_return_advantage_alpha."
                )
        if return_gated_margin is not None:
            if return_gated_margin <= 0.0:
                raise ValueError("return_gated_margin must be positive.")
            if return_gated_margin_weight <= 0.0:
                raise ValueError(
                    "return_gated_margin requires a positive "
                    "return_gated_margin_weight."
                )
            if not dense_return_q_target:
                raise ValueError(
                    "return_gated_margin requires dense_return_q_target=true."
                )
            if not mc_lower_bound_target:
                raise ValueError(
                    "return_gated_margin requires mc_lower_bound_target=true."
                )
        if not 0.0 <= dense_return_finest_neighbor_weight <= 1.0:
            raise ValueError(
                "dense_return_finest_neighbor_weight must be in [0, 1]."
            )
        if (
            dense_return_finest_neighbor_weight > 0.0
            and not dense_return_q_target
        ):
            raise ValueError(
                "dense_return_finest_neighbor_weight requires "
                "dense_return_q_target=true."
            )
        if (
            dense_return_expected_q_loss
            and dense_return_finest_neighbor_weight > 0.0
        ):
            raise ValueError(
                "dense_return_expected_q_loss requires "
                "dense_return_finest_neighbor_weight=0."
            )
        if episodic_success_q_target and not dense_return_q_target:
            raise ValueError(
                "episodic_success_q_target requires "
                "dense_return_q_target=true."
            )
        if episodic_success_q_target and mc_lower_bound_target:
            raise ValueError(
                "episodic_success_q_target replaces the discounted "
                "MC-lower-bound/Bellman target; set "
                "mc_lower_bound_target=false."
            )
        if episodic_success_q_target and mc_return_weight != 0.0:
            raise ValueError(
                "episodic_success_q_target is the complete Q objective; "
                "set mc_return_weight=0."
            )
        if (
            episodic_success_q_target
            and unseen_return_floor_value != 0.0
        ):
            raise ValueError(
                "episodic_success_q_target requires "
                "unseen_return_floor_value=0."
            )
        if episodic_success_q_target and not (
            v_min <= 0.0 <= 1.0 <= v_max
        ):
            raise ValueError(
                "episodic_success_q_target requires C51 support to "
                "contain both 0 and 1."
            )
        if not 0.0 <= ordered_success_return_mix <= 1.0:
            raise ValueError(
                "ordered_success_return_mix must be in [0, 1]."
            )
        if (
            ordered_success_return_mix > 0.0
            and not mc_lower_bound_target
        ):
            raise ValueError(
                "ordered_success_return_mix requires "
                "mc_lower_bound_target=true."
            )
        if (
            ordered_success_return_mix > 0.0
            and not dense_return_q_target
        ):
            raise ValueError(
                "ordered_success_return_mix requires "
                "dense_return_q_target=true."
            )
        if sequence_aligned_mc_discount is not None:
            if not 0.0 < sequence_aligned_mc_discount <= 1.0:
                raise ValueError(
                    "sequence_aligned_mc_discount must be in (0, 1]."
                )
            if not mc_lower_bound_target:
                raise ValueError(
                    "sequence_aligned_mc_discount requires "
                    "mc_lower_bound_target=true."
                )
            if not dense_return_q_target:
                raise ValueError(
                    "sequence_aligned_mc_discount requires "
                    "dense_return_q_target=true."
                )
            if ordered_success_return_mix > 0.0:
                raise ValueError(
                    "sequence_aligned_mc_discount and "
                    "ordered_success_return_mix are mutually exclusive."
                )
            if critic_sequence_mode != "full":
                raise ValueError(
                    "sequence_aligned_mc_discount requires "
                    "critic_sequence_mode=full."
                )
        if distinct_policy_encoder and not separate_bc_policy:
            raise ValueError(
                "distinct_policy_encoder=true requires separate_bc_policy=true."
            )
        if policy_value_beta is not None and policy_value_beta < 0.0:
            raise ValueError(
                "policy_value_beta must be non-negative or null."
            )
        if policy_value_beta is not None and not separate_bc_policy:
            raise ValueError(
                "policy_value_beta requires separate_bc_policy=true."
            )
        if mc_return_weight < 0.0:
            raise ValueError("mc_return_weight must be non-negative.")
        if auxiliary_td_loss_weight < 0.0:
            raise ValueError(
                "auxiliary_td_loss_weight must be non-negative."
            )
        if auxiliary_td_loss_weight > 0.0 and not pessimistic_twin_critic:
            raise ValueError(
                "auxiliary_td_loss_weight > 0 requires "
                "pessimistic_twin_critic=true."
            )
        if mc_lower_bound_target and separate_bc_policy:
            raise ValueError(
                "mc_lower_bound_target requires the canonical critic path "
                "(separate_bc_policy=false)."
            )
        if unseen_return_floor_weight < 0.0:
            raise ValueError(
                "unseen_return_floor_weight must be non-negative."
            )
        if dense_return_q_target and unseen_return_floor_weight != 0.0:
            raise ValueError(
                "dense_return_q_target is the complete Q objective and "
                "requires unseen_return_floor_weight=0."
            )
        if not v_min <= unseen_return_floor_value <= v_max:
            raise ValueError(
                "unseen_return_floor_value must lie on the C51 support."
            )
        unseen_return_floor_reduction = str(
            unseen_return_floor_reduction
        ).lower()
        if unseen_return_floor_reduction not in {"mean", "max", "topk"}:
            raise ValueError(
                "unseen_return_floor_reduction must be one of "
                "{'mean', 'max', 'topk'}."
            )
        if not 1 <= unseen_return_floor_topk < bins:
            raise ValueError(
                "unseen_return_floor_topk must be in [1, bins - 1]."
            )
        # Canonical (non-decoupled) MC anchor is a deliberate Stage-147 arm:
        # it calibrates the same Q head that drives behavior.  The historical
        # decoupling requirement is therefore relaxed; interpretation caveats
        # live in cqn-flow.md section 28.6.
        self._canonical_mc_anchor = bool(
            mc_return_weight > 0.0 and not separate_bc_policy
        )
        self.mc_lower_bound_target = bool(mc_lower_bound_target)
        self.episodic_success_q_target = bool(
            episodic_success_q_target
        )
        self.ordered_success_return_mix = float(
            ordered_success_return_mix
        )
        self.sequence_aligned_mc_discount = (
            None
            if sequence_aligned_mc_discount is None
            else float(sequence_aligned_mc_discount)
        )
        self._uses_canonical_mc_returns = bool(
            self._canonical_mc_anchor
            or self.mc_lower_bound_target
            or self.episodic_success_q_target
        )
        if mc_return_weight > 0.0 and mc_return_value_only and not use_dueling:
            raise ValueError(
                "mc_return_value_only=true requires use_dueling=true."
            )
        if pessimistic_twin_critic:
            twin_violations = []
            if not strict_demo_rl_only:
                twin_violations.append("strict_demo_rl_only=true")
            if use_dueling:
                twin_violations.append("use_dueling=false")
            if autoregressive_action_dims:
                twin_violations.append("autoregressive_action_dims=false")
            if td_target_action_source != "critic_replay_max":
                twin_violations.append(
                    "td_target_action_source=critic_replay_max"
                )
            if not mc_lower_bound_target:
                twin_violations.append("mc_lower_bound_target=true")
            if dense_return_q_target:
                twin_violations.append("dense_return_q_target=false")
            if unseen_return_floor_weight != 0.0:
                twin_violations.append("unseen_return_floor_weight=0")
            if episodic_success_q_target:
                twin_violations.append("episodic_success_q_target=false")
            if ordered_success_return_mix != 0.0:
                twin_violations.append("ordered_success_return_mix=0")
            if sequence_aligned_mc_discount is not None:
                twin_violations.append("sequence_aligned_mc_discount=null")
            if mc_return_weight != 0.0:
                twin_violations.append("mc_return_weight=0")
            if centralized_critic:
                twin_violations.append("centralized_critic=false")
            if critic_sequence_mode != "full":
                twin_violations.append("critic_sequence_mode=full")
            if q_reward_scale != 1.0:
                twin_violations.append("q_reward_scale=1")
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
            if structured_exploration_prob != 0.0:
                exploration_violations.append("structured_exploration_prob=0")
            if bin_flip_prob != 0.0:
                exploration_violations.append("bin_flip_prob=0")
            if bin_explore_probs is not None:
                exploration_violations.append("bin_explore_probs=null")
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
            if separate_bc_policy:
                beam_violations.append("separate_bc_policy=false")
            if coarse_flow:
                beam_violations.append("coarse_flow=false")
            if autoregressive_action_dims:
                beam_violations.append("autoregressive_action_dims=false")
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
        self.demo_fosd = bool(demo_fosd)
        self.strict_demo_rl_only = bool(strict_demo_rl_only)
        self.autoregressive_action_dims = bool(
            autoregressive_action_dims
        )
        self.pessimistic_twin_critic = bool(pessimistic_twin_critic)
        self.auxiliary_td_loss_weight = float(auxiliary_td_loss_weight)
        self.episodic_twin_head_exploration = bool(
            episodic_twin_head_exploration
        )
        self.twin_rollout_beam_width = int(twin_rollout_beam_width)
        self.dense_return_q_target = bool(dense_return_q_target)
        self.dense_return_positive_only = bool(
            dense_return_positive_only
        )
        self.dense_return_expected_q_loss = bool(
            dense_return_expected_q_loss
        )
        self.dense_return_advantage_alpha = float(
            dense_return_advantage_alpha
        )
        self.dense_return_advantage_clip_ratio = (
            None
            if dense_return_advantage_clip_ratio is None
            else float(dense_return_advantage_clip_ratio)
        )
        self.q_reward_scale = float(q_reward_scale)
        self.dense_return_label_smoothing = float(
            dense_return_label_smoothing
        )
        self.dense_return_floor_satisfaction_margin = (
            None
            if dense_return_floor_satisfaction_margin is None
            else float(dense_return_floor_satisfaction_margin)
        )
        self.dense_return_relative_floor_margin = (
            None
            if dense_return_relative_floor_margin is None
            else float(dense_return_relative_floor_margin)
        )
        self.return_gated_margin = (
            None if return_gated_margin is None else float(return_gated_margin)
        )
        self.return_gated_margin_weight = float(return_gated_margin_weight)
        self.dense_return_finest_neighbor_weight = float(
            dense_return_finest_neighbor_weight
        )
        self.unseen_return_floor_weight = float(
            unseen_return_floor_weight
        )
        self.unseen_return_floor_value = float(
            unseen_return_floor_value
        )
        self.unseen_return_floor_reduction = (
            unseen_return_floor_reduction
        )
        self.unseen_return_floor_topk = int(
            unseen_return_floor_topk
        )
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
        self.random_levels_from = random_levels_from
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
        self.level_override_mode = str(level_override_mode)
        self.structured_exploration_prob = float(
            structured_exploration_prob
        )
        self.structured_exploration_level = int(
            structured_exploration_level
        )
        self.structured_exploration_horizon = int(
            structured_exploration_horizon
        )
        self.separate_bc_policy = bool(separate_bc_policy)
        self.bc_policy_stop_gradient = bool(bc_policy_stop_gradient)
        self.distinct_policy_encoder = bool(distinct_policy_encoder)
        self.use_frozen_support_mask = bool(use_frozen_support_mask)
        self.support_mask_decode = bool(support_mask_decode)
        self.support_mask_tau = float(support_mask_tau)
        self.support_mask_freeze_step = int(support_mask_freeze_step)
        self.td_target_action_source = td_target_action_source
        self.demo_behavior_force_probability = float(
            demo_behavior_force_probability
        )
        self.td_target_policy_value_beta = (
            None
            if td_target_policy_value_beta is None
            else float(td_target_policy_value_beta)
        )
        self.critic_sequence_mode = critic_sequence_mode
        self.token_split_horizon_targets = token_split_horizon_targets
        self.token_split_boundary = (
            None if token_split_boundary is None else int(token_split_boundary)
        )
        self.bc_lambda_schedule = (
            None if bc_lambda_schedule is None else str(bc_lambda_schedule)
        )
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
        self._bin_flip_remaining = np.zeros(
            (int(num_train_envs),), dtype=np.int32
        )
        self._bin_flip_dimension = np.full(
            (int(num_train_envs),), -1, dtype=np.int16
        )
        self._bin_flip_delta_sequence = np.zeros(
            (int(num_train_envs), self.action_sequence), dtype=np.float32
        )
        self._bin_flip_rng = np.random.default_rng(int(seed) + 151)
        self._episodic_twin_head_rng = np.random.default_rng(int(seed) + 157)
        self._episodic_twin_heads = np.full(
            (int(num_train_envs),), -1, dtype=np.int8
        )
        self._episodic_twin_head_assignments = np.zeros(
            (2,), dtype=np.int64
        )
        # Stage-153 hierarchical epsilon-bin exploration (ensemble-safe).
        if bin_explore_probs is not None:
            probs = tuple(float(p) for p in bin_explore_probs)
            if len(probs) != levels:
                raise ValueError(
                    "bin_explore_probs must list one probability per level."
                )
            if any(not 0.0 <= p <= 1.0 for p in probs):
                raise ValueError(
                    "bin_explore_probs entries must be in [0, 1]."
                )
            if bin_flip_prob > 0.0:
                raise ValueError(
                    "bin_explore_probs and bin_flip_prob are mutually "
                    "exclusive exploration mechanisms."
                )
            self.bin_explore_probs = probs
        else:
            self.bin_explore_probs = None
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
        # Stage-162: optional schedule multiplying every level's activation
        # probability (e.g. "linear(1.0,0.0,100000)" anneals exploration
        # away as TD takes over). None keeps the static probabilities.
        if bin_explore_schedule is not None and bin_explore_probs is None:
            raise ValueError(
                "bin_explore_schedule requires bin_explore_probs."
            )
        self.bin_explore_schedule = (
            None if bin_explore_schedule is None else str(bin_explore_schedule)
        )
        self._bin_explore_scale = 1.0
        # How many consecutive fresh plans a fired flip is re-applied to.
        # Default (None) = action_sequence, matching per-step replanning;
        # with sparser replan intervals set this so that persist_plans x
        # replan_interval keeps the intended window length in env steps.
        if bin_explore_persist_plans is not None and bin_explore_persist_plans < 1:
            raise ValueError("bin_explore_persist_plans must be >= 1.")
        self.bin_explore_persist_plans = (
            int(self.action_sequence)
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
        self.mc_return_weight = float(mc_return_weight)
        self.mc_return_stop_gradient_encoder = bool(
            mc_return_stop_gradient_encoder
        )
        self.mc_return_value_only = bool(mc_return_value_only)
        self.policy_value_beta = (
            None if policy_value_beta is None else float(policy_value_beta)
        )
        cv_rct_baseline = str(cv_rct_baseline).lower()
        if cv_rct_baseline not in {"target_q", "none"}:
            raise ValueError("cv_rct_baseline must be 'target_q' or 'none'.")
        if cv_rct_weight is not None:
            if float(cv_rct_weight) < 0.0:
                raise ValueError("cv_rct_weight must be non-negative.")
            if not separate_bc_policy:
                raise ValueError(
                    "cv_rct_weight requires separate_bc_policy=true."
                )
            if not structured_exploration_prob > 0.0:
                raise ValueError(
                    "cv_rct_weight requires a randomized structured "
                    "exploration policy (structured_exploration_prob > 0)."
                )
        self.cv_rct_weight = (
            None if cv_rct_weight is None else float(cv_rct_weight)
        )
        resolved_cv_level = (
            int(structured_exploration_level)
            if cv_rct_level is None
            else int(cv_rct_level)
        )
        if not 0 <= resolved_cv_level < levels:
            raise ValueError("cv_rct_level must be in [0, levels).")
        self.cv_rct_level = resolved_cv_level
        self.cv_rct_baseline = cv_rct_baseline
        if awr_beta is not None:
            if float(awr_beta) <= 0.0:
                raise ValueError("awr_beta must be positive.")
            if not separate_bc_policy:
                raise ValueError("awr_beta requires separate_bc_policy=true.")
        if awr_weight_max <= 0.0:
            raise ValueError("awr_weight_max must be positive.")
        if not 0.0 < awr_expectile_tau < 1.0:
            raise ValueError("awr_expectile_tau must be in (0, 1).")
        self.awr_beta = None if awr_beta is None else float(awr_beta)
        self.awr_weight_max = float(awr_weight_max)
        self.awr_expectile_tau = float(awr_expectile_tau)
        # ---- Progress-potential shaping (Ng et al. 1999 potential form) ----
        # Phi is a state-only auxiliary head; the potential enters the C51
        # target's reward scalar only.  Replay rewards and mc_returns stay raw
        # so no stored quantity depends on lambda or gamma.
        progress_potential_weight = float(progress_potential_weight)
        progress_head_weight = float(progress_head_weight)
        if progress_potential_weight < 0.0:
            raise ValueError(
                "progress_potential_weight must be non-negative."
            )
        if progress_head_weight < 0.0:
            raise ValueError("progress_head_weight must be non-negative.")
        if not 0.0 < float(progress_expectile_tau) < 1.0:
            raise ValueError("progress_expectile_tau must be in (0, 1).")
        if progress_potential_schedule is not None:
            # Fail fast on an unparsable schedule string instead of at the
            # first update.
            utils.schedule(str(progress_potential_schedule), 0)
            if progress_potential_weight <= 0.0:
                raise ValueError(
                    "progress_potential_schedule requires "
                    "progress_potential_weight > 0 (the weight is the "
                    "schedule's enable gate and its bound check)."
                )
        self.progress_potential_weight = progress_potential_weight
        self.progress_potential_schedule = (
            None
            if progress_potential_schedule is None
            else str(progress_potential_schedule)
        )
        self.progress_head_weight = progress_head_weight
        self.progress_expectile_tau = float(progress_expectile_tau)
        self.progress_success_gated = bool(progress_success_gated)
        # The head is instantiated whenever either consumer needs it; unlike
        # awr_beta it is NOT restricted to the separate_bc_policy platform.
        self.progress_head_enabled = bool(
            progress_head_weight > 0.0 or progress_potential_weight > 0.0
        )
        self.progress_shaping_enabled = progress_potential_weight > 0.0
        if self.progress_head_enabled:
            if self.pessimistic_twin_critic:
                raise NotImplementedError(
                    "progress shaping is not implemented on the "
                    "pessimistic_twin_critic update graph."
                )
            if bool(getattr(self, "direct_scalar_q", False)):
                raise NotImplementedError(
                    "progress shaping is not implemented on the direct "
                    "scalar-Q update graph."
                )
        if self.progress_shaping_enabled:
            # Raw Monte-Carlo consumers assume UNSHAPED {0,1}-scale returns.
            # Mixing them with shaped Bellman targets silently inverts the
            # lower-bound mask, so refuse instead of guessing the shift.
            mc_conflicts = [
                name
                for name, active in (
                    ("mc_lower_bound_target", self.mc_lower_bound_target),
                    (
                        "episodic_success_q_target",
                        self.episodic_success_q_target,
                    ),
                    (
                        "ordered_success_return_mix",
                        self.ordered_success_return_mix > 0.0,
                    ),
                )
                if active
            ]
            if mc_conflicts:
                raise ValueError(
                    "progress_potential_weight > 0 cannot be combined with "
                    "raw Monte-Carlo targets ("
                    + ", ".join(mc_conflicts)
                    + "); the shifted-MC variant is not implemented."
                )
            # C51 support headroom: the largest shaped reward scalar is
            # (max sparse return + lambda) * q_reward_scale and
            # project_categorical clips silently at v_max.
            shaped_bound = (1.0 + progress_potential_weight) * float(
                q_reward_scale
            )
            if shaped_bound > float(v_max) + 1e-6:
                raise ValueError(
                    "shaped target bound (1 + progress_potential_weight) * "
                    f"q_reward_scale = {shaped_bound:.4f} exceeds v_max="
                    f"{float(v_max):.4f}; C51 projection would clip silently."
                )
        if flow_policy and not separate_bc_policy:
            raise ValueError("flow_policy requires separate_bc_policy=true.")
        if flow_policy_candidates < 1:
            raise ValueError("flow_policy_candidates must be at least 1.")
        if flow_policy_steps < 1:
            raise ValueError("flow_policy_steps must be at least 1.")
        if flow_policy_lambda < 0.0:
            raise ValueError("flow_policy_lambda must be non-negative.")
        self.flow_policy = bool(flow_policy)
        self.flow_policy_candidates = int(flow_policy_candidates)
        self.flow_policy_steps = int(flow_policy_steps)
        self.flow_policy_lambda = float(flow_policy_lambda)
        if flow_policy_ema is not None and not 0.0 < flow_policy_ema < 1.0:
            raise ValueError("flow_policy_ema must be in (0, 1).")
        self.flow_policy_ema = (
            None if flow_policy_ema is None else float(flow_policy_ema)
        )
        self.flow_policy_ema_params = None
        self.flow_policy_hidden_dims = (
            None
            if flow_policy_hidden_dims is None
            else tuple(int(v) for v in flow_policy_hidden_dims)
        )
        self.flow_policy_gru_layers = (
            None
            if flow_policy_gru_layers is None
            else int(flow_policy_gru_layers)
        )
        # Stage-152 coarse-flow (cqn-flow.md 34): the canonical critic keeps
        # the coarse bin decision (where sibling bins have data support and
        # TD counterfactuals are identifiable) and a bin-conditioned flow
        # head supplies the continuous within-cell residual.
        if coarse_flow and separate_bc_policy:
            raise ValueError(
                "coarse_flow runs on the canonical platform; "
                "set separate_bc_policy=false."
            )
        if coarse_flow and flow_policy:
            raise ValueError(
                "coarse_flow and flow_policy (decoupled rerank) are "
                "mutually exclusive."
            )
        if coarse_flow_selfdistill_weight is not None and (
            coarse_flow_selfdistill_weight < 0.0
        ):
            raise ValueError(
                "coarse_flow_selfdistill_weight must be non-negative."
            )
        if coarse_flow_pure and not coarse_flow:
            raise ValueError("coarse_flow_pure requires coarse_flow=true.")
        self.coarse_flow = bool(coarse_flow)
        # Stage-155 no-selection control: the flow models the FULL action
        # chunk (no bin context, no critic argmax at rollout).  Everything
        # else -- critic training, encoder, flow capacity, EMA -- stays
        # matched, so (CCFF - pure) isolates the coarse selection
        # mechanism as a whole.
        self.coarse_flow_pure = bool(coarse_flow_pure)
        self.coarse_flow_selfdistill_weight = (
            None
            if coarse_flow_selfdistill_weight is None
            else float(coarse_flow_selfdistill_weight)
        )
        self.coarse_flow_selfdistill_threshold = float(
            coarse_flow_selfdistill_threshold
        )
        self._seed = int(seed)
        self._last_structured_exploration_mask = np.zeros(
            (int(num_train_envs),),
            dtype=np.bool_,
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
        def init_critic(key):
            if self.autoregressive_action_dims:
                return self.critic_model.init(
                    key,
                    dummy_features,
                    dummy_level,
                    dummy_midpoint,
                    dummy_midpoint,
                )
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
            critic_params = init_critic(self.rng_key)
            critic2_params = None
            self.params = {"critic": critic_params}
        if self.separate_bc_policy or self.use_frozen_support_mask:
            # The policy has its own coarse-to-fine bin logits.  It deliberately
            # has no value atoms: demo CE can train this head without changing
            # the critic's return distribution or action ranking.  The frozen
            # support mask reuses the same head as a standalone per-level
            # bin-probability model on the canonical critic path.
            self.policy_model = C2FSequenceDistributionalCritic(
                hidden_dims=model.hidden_dims,
                action_sequence=self.action_sequence,
                action_dim=self.action_dim,
                levels=self.levels,
                bins=self.bins,
                atoms=1,
                low_dim_size=(self.low_dim_size if self.use_pixels else 0),
                gru_layers=self.gru_layers,
                activation_name=model.activation,
                use_dueling=False,
            )
            self.rng_key, policy_key = jax.random.split(self.rng_key)
            self.params["policy"] = self.policy_model.init(
                policy_key,
                dummy_features,
                dummy_level,
                dummy_midpoint,
            )
            if self.awr_beta is not None:
                self.expectile_value_model = ExpectileValueHead(
                    hidden_dims=model.hidden_dims,
                    activation_name=model.activation,
                )
                self.rng_key, value_key = jax.random.split(self.rng_key)
                self.params["expectile_value"] = (
                    self.expectile_value_model.init(
                        value_key,
                        dummy_features,
                    )
                )
            if self.flow_policy:
                self.flow_policy_model = FlowPolicyHead(
                    hidden_dims=(
                        self.flow_policy_hidden_dims
                        if self.flow_policy_hidden_dims is not None
                        else model.hidden_dims
                    ),
                    action_sequence=self.action_sequence,
                    action_dim=self.action_dim,
                    low_dim_size=(
                        self.low_dim_size if self.use_pixels else 0
                    ),
                    gru_layers=(
                        self.flow_policy_gru_layers
                        if self.flow_policy_gru_layers is not None
                        else self.gru_layers
                    ),
                    activation_name=model.activation,
                )
                self.rng_key, flow_key = jax.random.split(self.rng_key)
                self.params["flow_policy"] = self.flow_policy_model.init(
                    flow_key,
                    dummy_features,
                    jnp.zeros(
                        (1, self._flat_action_dim), dtype=jnp.float32
                    ),
                    jnp.zeros((1,), dtype=jnp.float32),
                )
                if self.flow_policy_ema is not None:
                    self.flow_policy_ema_params = jax.tree.map(
                        jnp.array,
                        self.params["flow_policy"],
                    )
        self.progress_value_model = None
        if self.progress_head_enabled:
            # Same scalar state-value head as AWR, but on the canonical
            # platform too: it reads stop-gradient features and never an
            # action, so it adds no action-label objective.
            self.progress_value_model = ExpectileValueHead(
                hidden_dims=model.hidden_dims,
                activation_name=model.activation,
            )
            # fold_in rather than split: adding the head must not consume the
            # rollout/update RNG stream, so a progress-enabled arm keeps the
            # exact legacy action keys and stays comparable to its control.
            self.params["progress_value"] = self.progress_value_model.init(
                jax.random.fold_in(self.rng_key, 0x9209),
                dummy_features,
            )
        if self.coarse_flow:
            self.flow_policy_model = FlowPolicyHead(
                hidden_dims=(
                    self.flow_policy_hidden_dims
                    if self.flow_policy_hidden_dims is not None
                    else model.hidden_dims
                ),
                action_sequence=self.action_sequence,
                action_dim=self.action_dim,
                low_dim_size=(
                    self.low_dim_size if self.use_pixels else 0
                ),
                gru_layers=(
                    self.flow_policy_gru_layers
                    if self.flow_policy_gru_layers is not None
                    else self.gru_layers
                ),
                activation_name=model.activation,
            )
            bin_context_dim = (
                self.levels * self.bins * self.action_dim + self.action_dim
            )
            dummy_bin_context = (
                None
                if self.coarse_flow_pure
                else jnp.zeros(
                    (1, self.action_sequence, bin_context_dim),
                    dtype=jnp.float32,
                )
            )
            self.rng_key, flow_key = jax.random.split(self.rng_key)
            self.params["flow_policy"] = self.flow_policy_model.init(
                flow_key,
                dummy_features,
                jnp.zeros(
                    (1, self._flat_action_dim), dtype=jnp.float32
                ),
                jnp.zeros((1,), dtype=jnp.float32),
                bin_context=dummy_bin_context,
            )
            if self.flow_policy_ema is not None:
                self.flow_policy_ema_params = jax.tree.map(
                    jnp.array,
                    self.params["flow_policy"],
                )
        if self._trainable_encoder:
            self.params["encoder"] = self._encoder_params
            if self.distinct_policy_encoder:
                # JAX arrays are immutable, so duplicating the leaves in the
                # parameter tree is sufficient to give the policy tower its
                # own optimizer state and gradient path. Both towers start
                # from the same visual initialization for a matched ablation.
                self.params["policy_encoder"] = jax.tree.map(
                    lambda value: jnp.array(value),
                    self._encoder_params,
                )
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

    def _init_cached_pixel_feature_key(self, method_name: str) -> None:
        del method_name
        super()._init_cached_pixel_feature_key("cqn_as")

    @property
    def _flat_action_dim(self) -> int:
        return self.action_sequence * self.action_dim

    def _critic_logits_per_level(
        self,
        critic_params,
        features,
        action,
        *,
        return_components: bool = False,
    ):
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
        value_logits_per_level = []
        chosen_advantage_logits_per_level = []
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
                bin_width = (high - low) / self.bins
                action_context = (
                    low
                    + (index.astype(jnp.float32) + 0.5) * bin_width
                ).reshape(
                    (
                        features.shape[0],
                        self.action_sequence,
                        self.action_dim,
                    )
                )
                model_output = self.critic_model.apply(
                    critic_params,
                    features,
                    one_hot,
                    midpoint,
                    action_context,
                    return_streams=return_components,
                )
            else:
                model_output = self.critic_model.apply(
                    critic_params,
                    features,
                    one_hot,
                    midpoint,
                    return_streams=return_components,
                )
            if return_components:
                logits, values, centered_advantages = model_output
            else:
                logits = model_output
            sequence_index = index.reshape(
                (features.shape[0], self.action_sequence, self.action_dim)
            )
            selected = jnp.take_along_axis(
                logits,
                sequence_index[..., None, None],
                axis=-2,
            )[..., 0, :]
            if return_components:
                selected_advantages = jnp.take_along_axis(
                    centered_advantages,
                    sequence_index[..., None, None],
                    axis=-2,
                )[..., 0, :]
                value_logits_per_level.append(
                    values[..., 0, :].reshape(
                        (features.shape[0], self._flat_action_dim, self.atoms)
                    )
                )
                chosen_advantage_logits_per_level.append(
                    selected_advantages.reshape(
                        (features.shape[0], self._flat_action_dim, self.atoms)
                    )
                )
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
        result = (
            jnp.stack(chosen_logits_per_level, axis=1),
            jnp.stack(logits_per_level, axis=1),
        )
        if return_components:
            return result + (
                jnp.stack(value_logits_per_level, axis=1),
                jnp.stack(chosen_advantage_logits_per_level, axis=1),
            )
        return result

    def _policy_logits_per_level(self, policy_params, features, action):
        """Return BC bin logits and encoded expert bins at every C2F level."""

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
        for level in range(self.levels):
            one_hot = jnp.broadcast_to(
                jax.nn.one_hot(level, self.levels, dtype=jnp.float32),
                (features.shape[0], self.levels),
            )
            logits = self.policy_model.apply(
                policy_params,
                features,
                one_hot,
                (0.5 * (low + high)).reshape(
                    (features.shape[0], self.action_sequence, self.action_dim)
                ),
            )[..., 0]
            logits_per_level.append(
                logits.reshape(
                    (features.shape[0], self._flat_action_dim, self.bins)
                )
            )
            low, high = zoom_in(
                low,
                high,
                discrete_action[:, level, :],
                self.bins,
                self.action_low,
                self.action_high,
            )
        return jnp.stack(logits_per_level, axis=1), discrete_action

    def _support_mask_bins(self, policy_params, features, one_hot, midpoint):
        """Admissible bins from the frozen behavior head at one C2F level."""

        logits = self.policy_model.apply(
            jax.lax.stop_gradient(policy_params),
            features,
            one_hot,
            midpoint,
        )[..., 0]
        probabilities = jax.nn.softmax(logits, axis=-1)
        return probabilities >= (
            self.support_mask_tau
            * jnp.max(probabilities, axis=-1, keepdims=True)
        )

    def _support_mask_for_actions(self, policy_params, features, actions):
        """Per-level admissible bins along the executed action's zoom path."""

        policy_logits, _ = self._policy_logits_per_level(
            jax.lax.stop_gradient(policy_params),
            features,
            actions,
        )
        probabilities = jax.nn.softmax(policy_logits, axis=-1)
        return probabilities >= (
            self.support_mask_tau
            * jnp.max(probabilities, axis=-1, keepdims=True)
        )

    def _support_mask_ce_loss(self, policy_params, features, actions, demos):
        """Demo-masked CE that trains only the bin-probability head."""

        policy_logits, expert_bins = self._policy_logits_per_level(
            policy_params,
            features,
            actions,
        )
        policy_log_probabilities = jax.nn.log_softmax(
            policy_logits,
            axis=-1,
        )
        expert_log_probabilities = jnp.take_along_axis(
            policy_log_probabilities,
            expert_bins[..., None],
            axis=-1,
        )[..., 0]
        per_sample = -expert_log_probabilities.mean(axis=(1, 2))
        demo_count = jnp.maximum(jnp.sum(demos), 1.0)
        return jnp.sum(per_sample * demos) / demo_count

    def _policy_action(self, policy_params, features, key=None):
        """Autoregress over C2F levels using the independent BC policy head."""

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
            logits = self.policy_model.apply(
                policy_params,
                features,
                one_hot,
                (0.5 * (low + high)).reshape(
                    (batch_size, self.action_sequence, self.action_dim)
                ),
            )[..., 0]
            index = jnp.argmax(logits, axis=-1)
            level_key = level_keys[level]
            if level_key is not None:
                random_index = jax.random.randint(
                    level_key,
                    index.shape,
                    minval=0,
                    maxval=self.bins,
                )
                logit_span = logits.max(axis=-1) - logits.min(axis=-1)
                index = jnp.where(
                    logit_span < self.tie_break_delta,
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

    def _critic_training_slice(self, values):
        if self.critic_sequence_mode == "effective_k0":
            return values[:, :, : self.action_dim]
        return values

    def _greedy_action(
        self,
        critic_params,
        features,
        key=None,
        policy_params=None,
    ):
        batch_size = features.shape[0]
        use_support_mask = (
            self.use_frozen_support_mask and policy_params is not None
        )
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
                q_values = jnp.sum(
                    probabilities * self.support,
                    axis=-1,
                )
                admissible = None
                if use_support_mask:
                    admissible = self._support_mask_bins(
                        policy_params,
                        features,
                        one_hot,
                        midpoint,
                    )
                    q_values = jnp.where(admissible, q_values, -jnp.inf)
                index = jnp.argmax(q_values, axis=-1)
                if level_key is not None:
                    if admissible is None:
                        random_index = jax.random.randint(
                            level_key,
                            index.shape,
                            minval=0,
                            maxval=self.bins,
                        )
                        q_span = (
                            q_values.max(axis=-1)
                            - q_values.min(axis=-1)
                        )
                    else:
                        random_index = jax.random.categorical(
                            level_key,
                            jnp.where(admissible, 0.0, -jnp.inf),
                            axis=-1,
                        )
                        q_span = q_values.max(axis=-1) - jnp.min(
                            jnp.where(admissible, q_values, jnp.inf),
                            axis=-1,
                        )
                    index = jnp.where(
                        q_span < self.tie_break_delta,
                        random_index,
                        index,
                    )
                # Diagnostic (eval only, default off): replace the critic's
                # bin choice with a uniform draw at this level and below.
                # The powered sibling probe (08-04) put the critic's ordering
                # at chance on the finest C2F level (sign accuracy 0.491/0.500
                # at level 2) while forcing a different fine bin still moved
                # realized return (regret 0.052-0.065 on 57% of states). This
                # knob measures what that unexploited ordering is actually
                # worth on task success: if randomizing the fine levels costs
                # nothing, fixing their ordering cannot buy anything either.
                if (
                    self.random_levels_from is not None
                    and level >= self.random_levels_from
                ):
                    if self.level_override_mode == "middle":
                        # Deterministic centre bin == what an agent with this
                        # level deleted would emit (the parent cell's centre).
                        # Comparing against "random" separates two things the
                        # fine levels could be doing: making a decision, or
                        # merely dithering inside the parent cell.
                        if admissible is None:
                            index = jnp.full_like(index, self.bins // 2)
                        else:
                            centre_distance = jnp.abs(
                                jnp.arange(self.bins) - self.bins // 2
                            ).astype(jnp.float32)
                            index = jnp.argmax(
                                jnp.where(
                                    admissible,
                                    -centre_distance,
                                    -jnp.inf,
                                ),
                                axis=-1,
                            )
                    else:
                        if level_key is None:
                            raise ValueError(
                                "random_levels_from needs an rng key; it is a "
                                "diagnostic for eval-time action selection."
                            )
                        if admissible is None:
                            index = jax.random.randint(
                                level_key,
                                index.shape,
                                minval=0,
                                maxval=self.bins,
                            )
                        else:
                            index = jax.random.categorical(
                                level_key,
                                jnp.where(admissible, 0.0, -jnp.inf),
                                axis=-1,
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

    def _policy_value_action(
        self,
        critic_params,
        value_features: jax.Array,
        policy_params,
        policy_features: jax.Array,
        key: jax.Array | None = None,
        *,
        policy_value_beta: float | None = None,
    ):
        """Select bins with normalized direct C51 Q plus the BC log prior."""

        resolved_beta = (
            self.policy_value_beta
            if policy_value_beta is None
            else float(policy_value_beta)
        )
        if resolved_beta is None:
            raise ValueError(
                "_policy_value_action requires policy_value_beta"
            )
        batch_size = value_features.shape[0]
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
            critic_logits = self.critic_model.apply(
                critic_params,
                value_features,
                one_hot,
                midpoint,
            )
            q_values = jnp.sum(
                jax.nn.softmax(critic_logits, axis=-1) * self.support,
                axis=-1,
            )
            policy_logits = self.policy_model.apply(
                policy_params,
                policy_features,
                one_hot,
                midpoint,
            )[..., 0]
            centered_q = q_values - q_values.mean(axis=-1, keepdims=True)
            q_scale = jnp.sqrt(
                jnp.mean(jnp.square(centered_q), axis=-1, keepdims=True)
                + 1e-6
            )
            score = centered_q / q_scale + (
                resolved_beta
                * jax.nn.log_softmax(policy_logits, axis=-1)
            )
            index = jnp.argmax(score, axis=-1)
            level_key = level_keys[level]
            if level_key is not None:
                random_index = jax.random.randint(
                    level_key,
                    index.shape,
                    minval=0,
                    maxval=self.bins,
                )
                score_span = score.max(axis=-1) - score.min(axis=-1)
                index = jnp.where(
                    score_span < self.tie_break_delta,
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

    def _flow_policy_sample(
        self,
        flow_params,
        features: jax.Array,
        key: jax.Array,
        candidates: int,
    ) -> jax.Array:
        """Euler-integrate the flow head into [B, M, K, D] action chunks."""

        batch = features.shape[0]
        m = int(candidates)
        x = jax.random.normal(
            key,
            (batch * m, self._flat_action_dim),
            dtype=jnp.float32,
        )
        repeated_features = jnp.repeat(features, m, axis=0)
        dt = 1.0 / float(self.flow_policy_steps)
        for step in range(self.flow_policy_steps):
            t = jnp.full((batch * m,), step * dt, dtype=jnp.float32)
            velocity = self.flow_policy_model.apply(
                flow_params,
                repeated_features,
                x,
                t,
            )
            x = x + dt * velocity
        x = jnp.clip(x, self.action_low, self.action_high)
        return x.reshape(
            (batch, m, self.action_sequence, self.action_dim)
        )

    def _flow_rerank_action(
        self,
        critic_params,
        value_features: jax.Array,
        chunks: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Score [B, M, K, D] candidates with the critic; pick the argmax.

        Score = deepest-level expected Q along each chunk's own zoom path,
        averaged over sequence and action dimensions.  Candidates are flow-BC
        samples, so every query stays on the manifold where the critic's
        calibration was measured (cqn-flow.md sections 24 and 29).
        """

        batch, m = chunks.shape[0], chunks.shape[1]
        flat = chunks.reshape((batch * m, self._flat_action_dim))
        repeated_features = jnp.repeat(value_features, m, axis=0)
        chosen_logits, _ = self._critic_logits_per_level(
            critic_params,
            repeated_features,
            flat,
        )
        probabilities = jax.nn.softmax(chosen_logits, axis=-1)
        q = jnp.sum(probabilities * self.support, axis=-1)
        scores = q[:, -1, :].mean(axis=-1).reshape((batch, m))
        best = jnp.argmax(scores, axis=-1)
        selected = jnp.take_along_axis(
            chunks,
            best[:, None, None, None],
            axis=1,
        )[:, 0]
        return selected, scores

    def _coarse_flow_cell(
        self,
        indices: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Cell context for the bin-conditioned flow head.

        ``indices`` holds per-level bin choices, either ``[B, L, K, D]``
        (from ``_greedy_action``) or ``[B, L, K*D]`` (from
        ``encode_action``).  Returns ``(bin_context, cell_low,
        cell_width)`` where ``bin_context`` is ``[B, K, L*D*bins + D]``
        (per-level one-hots plus the normalized cell center) and the cell
        bounds are flat ``[B, K*D]`` arrays in action units.
        """

        batch = indices.shape[0]
        flat = self._flat_action_dim
        idx = indices.reshape((batch, self.levels, flat))
        low = jnp.broadcast_to(self.action_low, (batch, flat))
        high = jnp.broadcast_to(self.action_high, (batch, flat))
        one_hots = []
        for level in range(self.levels):
            one_hots.append(
                jax.nn.one_hot(idx[:, level], self.bins, dtype=jnp.float32)
            )
            low, high = zoom_in(
                low,
                high,
                idx[:, level],
                self.bins,
                self.action_low,
                self.action_high,
            )
        cell_width = jnp.maximum(high - low, 1e-8)
        center = 0.5 * (low + high)
        span = jnp.maximum(self.action_high - self.action_low, 1e-8)
        center_context = (
            2.0 * (center - self.action_low) / span - 1.0
        ).reshape((batch, self.action_sequence, self.action_dim))
        one_hot_context = (
            jnp.stack(one_hots, axis=1)
            .reshape(
                (
                    batch,
                    self.levels,
                    self.action_sequence,
                    self.action_dim,
                    self.bins,
                )
            )
            .transpose((0, 2, 1, 3, 4))
            .reshape(
                (
                    batch,
                    self.action_sequence,
                    self.levels * self.action_dim * self.bins,
                )
            )
        )
        bin_context = jnp.concatenate(
            [one_hot_context, center_context], axis=-1
        )
        return bin_context, low, cell_width

    def _coarse_flow_action(
        self,
        flow_params,
        features: jax.Array,
        indices: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        """Euler-integrate the within-cell residual and decode to actions.

        The flow works in [-1, 1] cell coordinates, so its output can
        never leave the critic-selected cell: the critic keeps full
        authority at the (identifiable) bin resolution while the flow
        only supplies the continuous precision the bin center lacks.
        """

        batch = features.shape[0]
        if indices is None:
            # Stage-155 no-selection control: the "cell" is the whole
            # action range and there is no conditioning.
            bin_context = None
            cell_low = jnp.broadcast_to(
                self.action_low, (batch, self._flat_action_dim)
            )
            cell_width = jnp.broadcast_to(
                self.action_high - self.action_low,
                (batch, self._flat_action_dim),
            )
        else:
            bin_context, cell_low, cell_width = self._coarse_flow_cell(
                indices
            )
        x = jax.random.normal(
            key,
            (batch, self._flat_action_dim),
            dtype=jnp.float32,
        )
        dt = 1.0 / float(self.flow_policy_steps)
        for step in range(self.flow_policy_steps):
            t = jnp.full((batch,), step * dt, dtype=jnp.float32)
            velocity = self.flow_policy_model.apply(
                flow_params,
                features,
                x,
                t,
                bin_context=bin_context,
            )
            x = x + dt * velocity
        residual = jnp.clip(x, -1.0, 1.0)
        action = cell_low + (residual + 1.0) * 0.5 * cell_width
        return action.reshape(
            (batch, self.action_sequence, self.action_dim)
        )

    def _build_greedy_action_fn(self):
        def action_fn(
            params,
            target_critic_params,
            obs_inputs,
            use_target,
            key,
            twin_head_indices,
        ):
            if self.separate_bc_policy:
                policy_encoder_params = params.get("encoder", None)
                if self.distinct_policy_encoder:
                    policy_encoder_params = params.get("policy_encoder", None)
                policy_features = self._rl_features(
                    policy_encoder_params,
                    obs_inputs,
                    stop_gradient=True,
                )
                if getattr(self, "flow_policy", False):
                    value_features = self._rl_features(
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
                    # When EMA is enabled the act() call site substitutes
                    # the EMA weights into params["flow_policy"], so they
                    # arrive as a traced argument rather than a jit-baked
                    # closure constant.
                    chunks = self._flow_policy_sample(
                        params["flow_policy"],
                        policy_features,
                        key,
                        self.flow_policy_candidates,
                    )
                    selected, _ = self._flow_rerank_action(
                        critic_params,
                        value_features,
                        chunks,
                    )
                    return selected
                if self.policy_value_beta is not None:
                    value_features = self._rl_features(
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
                    return self._policy_value_action(
                        critic_params,
                        value_features,
                        params["policy"],
                        policy_features,
                        key,
                    )[0]
                return self._policy_action(
                    params["policy"], policy_features, key=key
                )[0]
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

                if getattr(self, "episodic_twin_head_exploration", False):
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
            if getattr(self, "coarse_flow", False):
                if getattr(self, "coarse_flow_pure", False):
                    return self._coarse_flow_action(
                        params["flow_policy"],
                        features,
                        None,
                        key,
                    )
                tie_key = None
                flow_key = key
                if key is not None:
                    tie_key, flow_key = jax.random.split(key)
                _, indices = self._greedy_action(
                    critic_params, features, key=tie_key
                )
                return self._coarse_flow_action(
                    params["flow_policy"],
                    features,
                    indices,
                    flow_key,
                )
            if self.twin_rollout_beam_width > 1:
                return self._joint_beam_action(critic_params, features)[0]
            return self._greedy_action(
                critic_params,
                features,
                key=key,
                policy_params=(
                    params["policy"]
                    if self.use_frozen_support_mask
                    and self.support_mask_decode
                    else None
                ),
            )[0]

        return action_fn

    def _greedy_action_for_update(
        self,
        critic_params,
        features,
        action_key,
        policy_params=None,
    ):
        if policy_params is None:
            # Subclasses override _greedy_action without the support-mask
            # keyword; only thread it when a mask head is actually supplied.
            return self._greedy_action(
                critic_params,
                features,
                key=action_key,
            )
        return self._greedy_action(
            critic_params,
            features,
            key=action_key,
            policy_params=policy_params,
        )

    def _td_target_action_for_update(
        self,
        critic_params,
        features,
        replay_actions,
        replay_next_actions,
        demos,
        action_key,
        policy_params=None,
    ):
        if self.td_target_action_source == "replay_next":
            # This hook is used by the no-policy/single-objective parent update
            # path.  The replay sequence contains consecutive executed actions,
            # so shifting once supplies the action sequence at s_{t+1}.
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
                policy_params=policy_params,
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
            if self.demo_behavior_force_probability > 0.0:
                force_key = jax.random.fold_in(action_key, 2601)
                demo_behavior_forced = (
                    demos >= 0.5
                ) & jax.random.bernoulli(
                    force_key,
                    self.demo_behavior_force_probability,
                    shape=demos.shape,
                )
                behavior_selected = (
                    behavior_selected | demo_behavior_forced
                )
            else:
                demo_behavior_forced = jnp.zeros_like(
                    demos,
                    dtype=jnp.bool_,
                )
            selected_action = jnp.where(
                behavior_selected[:, None, None],
                behavior_action,
                greedy_action,
            )
            return selected_action, {
                "behavior_selected": behavior_selected,
                "behavior_score": behavior_score,
                "greedy_score": greedy_score,
                "demo_behavior_forced": demo_behavior_forced,
            }
        return self._greedy_action_for_update(
            critic_params,
            features,
            action_key,
            policy_params=policy_params,
        )

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

    def _next_action_key(self):
        self.rng_key, action_key = jax.random.split(self.rng_key)
        return action_key

    def _structured_exploration_action(self, executed_action, key):
        """Perturb one executed coordinate by one local C2F cell width.

        This runs after temporal ensembling, so replay stores exactly the
        action that was intervened on. Only one coordinate changes per selected
        environment step; the BC plan and all other coordinates stay intact.
        """

        probability = float(
            getattr(self, "structured_exploration_prob", 0.0)
        )
        level = int(getattr(self, "structured_exploration_level", 1))
        action = jnp.asarray(executed_action, dtype=jnp.float32)
        mask_key, dimension_key, direction_key = jax.random.split(key, 3)
        explore_mask = jax.random.uniform(
            mask_key,
            (action.shape[0],),
        ) < probability
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
                    jax.random.bernoulli(
                        direction_key, shape=(batch_size,)
                    ),
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
        self._structured_exploration_dimension[starts] = (
            sampled_dimensions[starts]
        )
        self._structured_exploration_direction[starts] = sampled_directions[
            starts
        ]
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
        assignment_probability[starts] = probability / float(
            2 * self.action_dim
        )

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
        drop = jax.random.bernoulli(
            key, self.low_dim_mask_prob, (batch, 1, 1)
        )
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
            obs_inputs["rgb"] = random_shift_rgb(
                obs_inputs["rgb"], augment_key
            )
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
                if (
                    getattr(self, "low_dim_mask_prob", 0.0) > 0.0
                    and "low_dim" in auxiliary_next_obs_inputs
                ):
                    auxiliary_next_obs_inputs["low_dim"] = self._mask_low_dim(
                        auxiliary_next_obs_inputs["low_dim"],
                        jax.random.fold_in(action_key, 3402),
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
                    behavior_selected = behavior_score >= greedy_score
                    if self.demo_behavior_force_probability > 0.0:
                        force_key = jax.random.fold_in(action_key, force_fold)
                        demo_behavior_forced = (
                            demos >= 0.5
                        ) & jax.random.bernoulli(
                            force_key,
                            self.demo_behavior_force_probability,
                            shape=demos.shape,
                        )
                        behavior_selected = (
                            behavior_selected | demo_behavior_forced
                        )
                    else:
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
                    self.demo_behavior_force_probability,
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

    def _build_update_fn(self):
        if self.pessimistic_twin_critic:
            return self._build_pessimistic_twin_update_fn()
        if not getattr(self, "separate_bc_policy", False):
            return super()._build_update_fn()

        optimizer = self.optimizer
        tau = self.critic_target_tau
        use_cv_rct = getattr(self, "cv_rct_weight", None) is not None
        use_progress_head = bool(getattr(self, "progress_head_enabled", False))
        use_progress_shaping = bool(
            getattr(self, "progress_shaping_enabled", False)
        )
        progress_head_weight = float(
            getattr(self, "progress_head_weight", 0.0)
        )
        progress_expectile_tau = float(
            getattr(self, "progress_expectile_tau", 0.9)
        )
        progress_success_gated = bool(
            getattr(self, "progress_success_gated", True)
        )

        def update_impl(
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
            mc_returns,
            structured,
            progress_labels,
            progress_valid,
            progress_lambda,
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
                policy_features = features
                next_policy_features = next_features
                if self.distinct_policy_encoder:
                    policy_encoder_params = current_params.get(
                        "policy_encoder", None
                    )
                    policy_features = self._rl_features(
                        policy_encoder_params,
                        obs_inputs,
                    )
                    next_policy_features = self._rl_features(
                        policy_encoder_params,
                        next_obs_inputs,
                        stop_gradient=True,
                    )

                if self.td_target_action_source == "replay_next":
                    # Replay sequences are assembled from the actions that were
                    # actually executed at consecutive environment steps.  The
                    # shifted first token is therefore a_{t+1}, including the
                    # temporal ensemble, rather than a newly predicted raw plan.
                    next_action = shift_replay_action_sequence(
                        actions,
                        self.action_sequence,
                        self.action_dim,
                    )
                elif self.td_target_action_source == "bc_policy":
                    next_action, _ = self._policy_action(
                        current_params["policy"],
                        next_policy_features,
                        key=action_key,
                    )
                elif self.td_target_action_source == "policy_value":
                    next_action, _ = self._policy_value_action(
                        current_params["critic"],
                        next_features,
                        current_params["policy"],
                        next_policy_features,
                        key=action_key,
                        policy_value_beta=self.td_target_policy_value_beta,
                    )
                else:
                    next_action, _ = self._greedy_action(
                        current_params["critic"],
                        next_features,
                        key=action_key,
                    )

                target_logits, _ = self._critic_logits_per_level(
                    target_critic_params,
                    next_features,
                    next_action,
                )
                target_logits = self._critic_training_slice(target_logits)
                target_probabilities = jax.nn.softmax(target_logits, axis=-1)
                progress_zero = jnp.asarray(0.0, dtype=jnp.float32)
                progress_phi_raw = jnp.zeros_like(rewards)
                progress_phi = jnp.zeros_like(rewards)
                progress_phi_next = jnp.zeros_like(rewards)
                if use_progress_head:
                    progress_phi_raw = self.progress_value_model.apply(
                        current_params["progress_value"],
                        jax.lax.stop_gradient(features),
                    )
                    progress_phi = jax.lax.stop_gradient(
                        jnp.clip(progress_phi_raw, 0.0, 1.0)
                    )
                    progress_phi_next = jax.lax.stop_gradient(
                        jnp.clip(
                            self.progress_value_model.apply(
                                current_params["progress_value"],
                                next_features,
                            ),
                            0.0,
                            1.0,
                        )
                    )
                shaped_rewards = rewards
                progress_clip_fraction = progress_zero
                if use_progress_shaping:
                    shaped_rewards = progress_shaped_rewards(
                        rewards,
                        discounts,
                        bootstrap,
                        progress_phi,
                        progress_phi_next,
                        progress_lambda,
                    )
                    shaped_atom_targets = shaped_rewards[:, None] + (
                        bootstrap * discounts
                    )[:, None] * self.support[None, :]
                    progress_clip_fraction = jnp.mean(
                        (
                            (shaped_atom_targets < self.support[0])
                            | (shaped_atom_targets > self.support[-1])
                        ).astype(jnp.float32)
                    )
                target_distribution = project_categorical(
                    target_probabilities,
                    shaped_rewards,
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
                per_sample = -jnp.sum(
                    target_distribution * chosen_log_probabilities,
                    axis=-1,
                ).mean(axis=(1, 2))
                td_critic_loss = self.critic_lambda * jnp.mean(
                    per_sample * loss_weights
                )

                # ``mc_returns`` is discounted reward-to-go computed only after
                # the full episode has finished. Project the scalar onto the
                # fixed C51 support and supervise the replayed effective action.
                # This is return regression, not action imitation or max-Q
                # bootstrapping.
                mc_target_distribution = project_categorical(
                    target_probabilities,
                    mc_returns,
                    jnp.zeros_like(discounts),
                    jnp.zeros_like(bootstrap),
                    self.support,
                )
                mc_target_distribution = jax.lax.stop_gradient(
                    mc_target_distribution
                )
                mc_chosen_log_probabilities = chosen_log_probabilities
                if self.mc_return_value_only:
                    # This blocks the direct MC gradient to advantage-stream
                    # parameters. Since distributional dueling combines atom
                    # logits before softmax, changing value logits (or shared
                    # encoder features) can still change expected-Q ranking.
                    mc_features = features
                    if self.mc_return_stop_gradient_encoder:
                        mc_features = jax.lax.stop_gradient(mc_features)
                    (
                        _,
                        _,
                        mc_value_logits,
                        mc_advantage_logits,
                    ) = self._critic_logits_per_level(
                        current_params["critic"],
                        mc_features,
                        actions,
                        return_components=True,
                    )
                    mc_value_logits = self._critic_training_slice(
                        mc_value_logits
                    )
                    mc_advantage_logits = self._critic_training_slice(
                        mc_advantage_logits
                    )
                    mc_chosen_log_probabilities = jax.nn.log_softmax(
                        mc_value_logits
                        + jax.lax.stop_gradient(mc_advantage_logits),
                        axis=-1,
                    )
                elif self.mc_return_stop_gradient_encoder:
                    mc_chosen_logits, _ = self._critic_logits_per_level(
                        current_params["critic"],
                        jax.lax.stop_gradient(features),
                        actions,
                    )
                    mc_chosen_logits = self._critic_training_slice(
                        mc_chosen_logits
                    )
                    mc_chosen_log_probabilities = jax.nn.log_softmax(
                        mc_chosen_logits,
                        axis=-1,
                    )
                mc_per_sample = -jnp.sum(
                    mc_target_distribution * mc_chosen_log_probabilities,
                    axis=-1,
                ).mean(axis=(1, 2))
                mc_return_loss = self.mc_return_weight * jnp.mean(
                    mc_per_sample * loss_weights
                )
                critic_loss = td_critic_loss + mc_return_loss

                # Control-variate-adjusted causal RCT loss (Stage-141).
                # The unadjusted moment loss is statistically dead here:
                # between-state RTG variance (sd~0.18-0.29) needs 5e4-2.7e6
                # samples at measured effect sizes, while a pre-treatment
                # value baseline shrinks variance 10-15x (cqn-flow.md sec 22).
                zero = jnp.asarray(0.0, dtype=jnp.float32)
                cv_rct_loss = zero
                cv_rct_moment = zero
                cv_valid_fraction = zero
                cv_treated_fraction = zero
                cv_tau_abs_mean = zero
                cv_outcome_adj_std = zero
                if structured is not None:
                    (
                        se_start,
                        se_dimension,
                        se_delta,
                        se_assignment_prob,
                    ) = structured
                    treated = se_start > 0.5
                    recorded_dimension = jnp.asarray(
                        se_dimension, dtype=jnp.int32
                    )
                    recorded_delta = jnp.asarray(
                        se_delta, dtype=jnp.float32
                    )
                    assignment_probability = jnp.asarray(
                        se_assignment_prob, dtype=jnp.float32
                    )
                    # Randomization only holds for online rollouts; demo
                    # transitions (assignment_prob stored as 1.0) and any
                    # future relabeled demos are excluded explicitly.
                    not_demo = demos < 0.5
                    valid_treatment = (
                        treated
                        & not_demo
                        & (recorded_dimension >= 0)
                        & (recorded_dimension < self.action_dim)
                        & (assignment_probability < 1.0)
                    )
                    valid_control = (
                        (~treated)
                        & not_demo
                        & (recorded_dimension < 0)
                        & (jnp.abs(recorded_delta) <= 1e-8)
                        & (assignment_probability < 1.0)
                    )
                    causal_valid = valid_treatment | valid_control

                    causal_key = jax.random.fold_in(action_key, 141)
                    dimension_key, direction_key = jax.random.split(
                        causal_key
                    )
                    batch = actions.shape[0]
                    sampled_dimension = jax.random.randint(
                        dimension_key,
                        (batch,),
                        minval=0,
                        maxval=self.action_dim,
                    )
                    sampled_direction = jnp.where(
                        jax.random.bernoulli(
                            direction_key, shape=(batch,)
                        ),
                        1.0,
                        -1.0,
                    )
                    intervention_dimension = jnp.where(
                        treated,
                        jnp.maximum(recorded_dimension, 0),
                        sampled_dimension,
                    )
                    cell_width = (
                        self._step_action_high - self._step_action_low
                    ) / float(self.bins ** (self.cv_rct_level + 1))
                    proposed_delta = jnp.where(
                        treated,
                        recorded_delta,
                        sampled_direction
                        * cell_width[intervention_dimension],
                    )
                    action_sequence = actions.reshape(
                        (batch, self.action_sequence, self.action_dim)
                    )
                    row = jnp.arange(batch)
                    # Treated: recover the pre-treatment proposal by undoing
                    # the recorded delta.  Control: apply a pseudo-delta.
                    counterfactual_sequence = action_sequence.at[
                        row, 0, intervention_dimension
                    ].add(
                        jnp.where(
                            treated, -proposed_delta, proposed_delta
                        )
                    )
                    counterfactual_sequence = jnp.clip(
                        counterfactual_sequence,
                        self._step_action_low,
                        self._step_action_high,
                    )
                    counterfactual_flat = counterfactual_sequence.reshape(
                        (batch, -1)
                    )

                    def deepest_dim_q(logits):
                        sliced = self._critic_training_slice(logits)
                        probabilities = jax.nn.softmax(sliced, axis=-1)
                        q = jnp.sum(
                            probabilities * self.support, axis=-1
                        )
                        return jnp.take_along_axis(
                            q[:, -1, :],
                            intervention_dimension[:, None],
                            axis=1,
                        )[:, 0]

                    cf_logits, _ = self._critic_logits_per_level(
                        current_params["critic"],
                        features,
                        counterfactual_flat,
                    )
                    q_cf_online = deepest_dim_q(cf_logits)
                    # chosen_probabilities is already training-sliced.
                    chosen_q_expected = jnp.sum(
                        chosen_probabilities * self.support, axis=-1
                    )
                    q_exec_online = jnp.take_along_axis(
                        chosen_q_expected[:, -1, :],
                        intervention_dimension[:, None],
                        axis=1,
                    )[:, 0]
                    # tau = Q(intervened action) - Q(non-intervened action).
                    treatment_effect = jnp.where(
                        treated,
                        q_exec_online - q_cf_online,
                        q_cf_online - q_exec_online,
                    )

                    outcome = mc_returns
                    if self.cv_rct_baseline == "target_q":
                        # Pre-treatment covariate: target-critic value of the
                        # non-intervened action.  Never the executed action of
                        # a treated sample -- that would absorb the effect.
                        base_flat = jnp.where(
                            treated[:, None],
                            counterfactual_flat,
                            actions,
                        )
                        base_logits, _ = self._critic_logits_per_level(
                            target_critic_params,
                            jax.lax.stop_gradient(features),
                            base_flat,
                        )
                        baseline = jax.lax.stop_gradient(
                            deepest_dim_q(base_logits)
                        )
                        outcome = mc_returns - baseline
                    propensity = float(self.structured_exploration_prob)
                    cv_rct_moment = action_centered_moment_loss(
                        treatment_effect,
                        outcome,
                        treated,
                        propensity,
                        causal_valid,
                        loss_weights,
                    )
                    cv_rct_loss = (
                        jnp.asarray(
                            self.cv_rct_weight, dtype=jnp.float32
                        )
                        * cv_rct_moment
                    )
                    valid_count = jnp.maximum(
                        jnp.sum(causal_valid.astype(jnp.float32)), 1.0
                    )
                    cv_valid_fraction = jnp.mean(
                        causal_valid.astype(jnp.float32)
                    )
                    cv_treated_fraction = (
                        jnp.sum(
                            (causal_valid & treated).astype(jnp.float32)
                        )
                        / valid_count
                    )
                    cv_tau_abs_mean = (
                        jnp.sum(
                            jnp.abs(treatment_effect)
                            * causal_valid.astype(jnp.float32)
                        )
                        / valid_count
                    )
                    valid_outcome = jnp.where(causal_valid, outcome, 0.0)
                    outcome_mean = (
                        jnp.sum(valid_outcome) / valid_count
                    )
                    cv_outcome_adj_std = jnp.sqrt(
                        jnp.maximum(
                            jnp.sum(
                                jnp.where(
                                    causal_valid,
                                    jnp.square(outcome - outcome_mean),
                                    0.0,
                                )
                            )
                            / valid_count,
                            0.0,
                        )
                    )
                critic_loss = critic_loss + cv_rct_loss

                if self.bc_policy_stop_gradient:
                    policy_features = jax.lax.stop_gradient(policy_features)
                policy_logits, expert_bins = self._policy_logits_per_level(
                    current_params["policy"],
                    policy_features,
                    actions,
                )
                policy_log_probabilities = jax.nn.log_softmax(
                    policy_logits,
                    axis=-1,
                )
                expert_log_probabilities = jnp.take_along_axis(
                    policy_log_probabilities,
                    expert_bins[..., None],
                    axis=-1,
                )[..., 0]
                policy_per_sample = -expert_log_probabilities.mean(axis=(1, 2))
                demo_count = jnp.maximum(jnp.sum(demos), 1.0)
                awr_zero = jnp.asarray(0.0, dtype=jnp.float32)
                awr_value_loss = awr_zero
                awr_value_mean = awr_zero
                awr_weight_mean = awr_zero
                awr_weight_ess = awr_zero
                if self.awr_beta is not None:
                    # IQL-style expectile state value on executed transitions
                    # only; features are stop-gradient so the value objective
                    # cannot disturb the shared visual representation.
                    state_value = self.expectile_value_model.apply(
                        current_params["expectile_value"],
                        jax.lax.stop_gradient(features),
                    )
                    value_error = mc_returns - state_value
                    expectile_weight = jnp.where(
                        value_error < 0.0,
                        1.0 - self.awr_expectile_tau,
                        self.awr_expectile_tau,
                    )
                    awr_value_loss = jnp.mean(
                        expectile_weight * jnp.square(value_error)
                    )
                    awr_value_mean = jnp.mean(state_value)
                    # Advantage-weighted BC over demo AND online transitions:
                    # completed-return advantage suppresses failed rollouts,
                    # no counterfactual (unexecuted-action) query is made.
                    awr_weights = jax.lax.stop_gradient(
                        jnp.clip(
                            jnp.exp(value_error / self.awr_beta),
                            0.0,
                            self.awr_weight_max,
                        )
                    )
                    weight_sum = jnp.maximum(jnp.sum(awr_weights), 1e-6)
                    policy_ce = (
                        jnp.sum(policy_per_sample * awr_weights) / weight_sum
                    )
                    awr_weight_mean = jnp.mean(awr_weights)
                    awr_weight_ess = jnp.square(weight_sum) / (
                        jnp.maximum(jnp.sum(jnp.square(awr_weights)), 1e-6)
                        * awr_weights.shape[0]
                    )
                else:
                    policy_ce = (
                        jnp.sum(policy_per_sample * demos) / demo_count
                    )
                policy_loss = self.bc_lambda * policy_ce
                flow_policy_loss = jnp.asarray(0.0, dtype=jnp.float32)
                if self.flow_policy:
                    # Conditional flow matching on demonstration chunks only,
                    # matching the demo-only convention of the categorical CE
                    # (Stage-145b showed cloning online rollouts is harmful).
                    flow_key = jax.random.fold_in(action_key, 146)
                    noise_key, time_key = jax.random.split(flow_key)
                    x1 = actions
                    x0 = jax.random.normal(
                        noise_key, x1.shape, dtype=jnp.float32
                    )
                    t = jax.random.uniform(
                        time_key, (x1.shape[0],), dtype=jnp.float32
                    )
                    x_t = (1.0 - t[:, None]) * x0 + t[:, None] * x1
                    predicted_velocity = self.flow_policy_model.apply(
                        current_params["flow_policy"],
                        policy_features,
                        x_t,
                        t,
                    )
                    flow_per_sample = jnp.square(
                        predicted_velocity - (x1 - x0)
                    ).mean(axis=-1)
                    flow_policy_loss = self.flow_policy_lambda * (
                        jnp.sum(flow_per_sample * demos) / demo_count
                    )
                progress_head_loss = progress_zero
                progress_value_mean = progress_zero
                progress_valid_fraction = progress_zero
                if use_progress_head:
                    progress_error = progress_labels - progress_phi_raw
                    progress_expectile_weight = jnp.where(
                        progress_error < 0.0,
                        1.0 - progress_expectile_tau,
                        progress_expectile_tau,
                    )
                    progress_mask = (
                        progress_valid
                        if progress_success_gated
                        else jnp.ones_like(progress_valid)
                    )
                    progress_valid_fraction = jnp.mean(progress_mask)
                    progress_head_loss = progress_head_weight * (
                        jnp.sum(
                            progress_expectile_weight
                            * jnp.square(progress_error)
                            * progress_mask
                        )
                        / jnp.maximum(jnp.sum(progress_mask), 1.0)
                    )
                    progress_value_mean = jnp.mean(progress_phi)
                total_loss = (
                    critic_loss
                    + policy_loss
                    + awr_value_loss
                    + flow_policy_loss
                    + progress_head_loss
                )

                policy_correct = (
                    jnp.argmax(policy_logits, axis=-1) == expert_bins
                ).astype(jnp.float32).mean(axis=(1, 2))
                policy_demo_top1 = jnp.sum(policy_correct * demos) / demo_count
                policy_probabilities = jax.nn.softmax(policy_logits, axis=-1)
                policy_entropy = -jnp.sum(
                    policy_probabilities
                    * jnp.log(jnp.maximum(policy_probabilities, 1e-9)),
                    axis=-1,
                ).mean()
                critic_entropy = -jnp.sum(
                    chosen_probabilities
                    * jnp.log(jnp.maximum(chosen_probabilities, 1e-9)),
                    axis=-1,
                ).mean()
                target_entropy = -jnp.sum(
                    target_distribution
                    * jnp.log(jnp.maximum(target_distribution, 1e-9)),
                    axis=-1,
                ).mean()
                all_probabilities = jax.nn.softmax(all_logits, axis=-1)
                all_q = jnp.sum(all_probabilities * self.support, axis=-1)
                critic_q_span = (all_q.max(axis=-1) - all_q.min(axis=-1)).mean()
                chosen_q = jnp.sum(chosen_probabilities * self.support, axis=-1)
                mc_return_mae = jnp.mean(
                    jnp.abs(chosen_q - mc_returns[:, None, None])
                )
                return total_loss, (
                    per_sample,
                    critic_loss,
                    td_critic_loss,
                    mc_return_loss,
                    mc_return_mae,
                    policy_loss,
                    policy_ce,
                    policy_demo_top1,
                    policy_entropy,
                    critic_entropy,
                    target_entropy,
                    critic_q_span,
                    cv_rct_loss,
                    cv_rct_moment,
                    cv_valid_fraction,
                    cv_treated_fraction,
                    cv_tau_abs_mean,
                    cv_outcome_adj_std,
                    awr_value_loss,
                    awr_value_mean,
                    awr_weight_mean,
                    awr_weight_ess,
                    flow_policy_loss,
                    progress_head_loss,
                    progress_value_mean,
                    progress_valid_fraction,
                    progress_clip_fraction,
                )

            (total_loss, aux), grads = jax.value_and_grad(
                loss_fn,
                has_aux=True,
            )(params)
            policy_encoder_grad_norm = (
                self.optax.tree.norm(grads["policy_encoder"])
                if "policy_encoder" in grads
                else jnp.asarray(0.0, dtype=total_loss.dtype)
            )
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = self.optax.apply_updates(params, updates)
            target_critic_params = jax.tree.map(
                lambda target, online: (1.0 - tau) * target + tau * online,
                target_critic_params,
                params["critic"],
            )
            (
                per_sample,
                critic_loss,
                td_critic_loss,
                mc_return_loss,
                mc_return_mae,
                policy_loss,
                policy_ce,
                policy_demo_top1,
                policy_entropy,
                critic_entropy,
                projected_entropy,
                critic_q_span,
                cv_rct_loss,
                cv_rct_moment,
                cv_valid_fraction,
                cv_treated_fraction,
                cv_tau_abs_mean,
                cv_outcome_adj_std,
                awr_value_loss,
                awr_value_mean,
                awr_weight_mean,
                awr_weight_ess,
                flow_policy_loss,
                progress_head_loss,
                progress_value_mean,
                progress_valid_fraction,
                progress_clip_fraction,
            ) = aux
            priority = jnp.sqrt(per_sample + 1e-10)
            priority = priority / jnp.maximum(jnp.max(priority), 1e-10)
            progress_metrics = {}
            if use_progress_head:
                progress_metrics["progress_head_loss"] = progress_head_loss
                progress_metrics["progress_head_value_mean"] = (
                    progress_value_mean
                )
                progress_metrics["progress_label_mean"] = jnp.mean(
                    progress_labels
                )
                progress_metrics["progress_valid_fraction"] = (
                    progress_valid_fraction
                )
            if use_progress_shaping:
                progress_metrics["progress_potential_lambda"] = jnp.asarray(
                    progress_lambda,
                    dtype=jnp.float32,
                )
                progress_metrics["progress_shaping_clip_frac"] = (
                    progress_clip_fraction
                )
            return (
                params,
                target_critic_params,
                opt_state,
                priority,
                {
                    **progress_metrics,
                    "critic_loss": critic_loss,
                    "td_critic_loss": td_critic_loss,
                    "mc_return_loss": mc_return_loss,
                    "mc_return_mae": mc_return_mae,
                    "mc_return_mean": jnp.mean(mc_returns),
                    "policy_bc_loss": policy_loss,
                    "policy_ce": policy_ce,
                    "total_loss": total_loss,
                    "policy_demo_top1": policy_demo_top1,
                    "policy_entropy": policy_entropy,
                    "policy_encoder_grad_norm": policy_encoder_grad_norm,
                    "entropy": critic_entropy,
                    "target_entropy": projected_entropy,
                    "critic_q_span": critic_q_span,
                    "loss_coeff": jnp.mean(loss_weights),
                    "cv_rct_loss": cv_rct_loss,
                    "cv_rct_moment_loss": cv_rct_moment,
                    "cv_rct_valid_fraction": cv_valid_fraction,
                    "cv_rct_treated_fraction": cv_treated_fraction,
                    "cv_rct_tau_abs_mean": cv_tau_abs_mean,
                    "cv_rct_outcome_adj_std": cv_outcome_adj_std,
                    "awr_value_loss": awr_value_loss,
                    "awr_value_mean": awr_value_mean,
                    "awr_weight_mean": awr_weight_mean,
                    "awr_weight_ess": awr_weight_ess,
                    "flow_policy_loss": flow_policy_loss,
                },
            )

        def split_progress_args(args):
            # (progress, progress_valid, lambda) are threaded immediately
            # before action_key; every other configuration passes none and
            # gets the exact legacy zero-potential graph.
            if not use_progress_head:
                rewards = args[6]
                return args, (
                    jnp.zeros_like(rewards),
                    jnp.zeros_like(rewards),
                    jnp.asarray(0.0, dtype=jnp.float32),
                )
            (*rest, labels, valid, weight, action_key) = args
            return (*rest, action_key), (labels, valid, weight)

        if use_cv_rct:

            def update_fn(*args):
                args, progress_args = split_progress_args(args)
                (
                    *core,
                    mc_returns,
                    structured_explore_start,
                    structured_explore_dimension,
                    structured_explore_delta,
                    structured_explore_assignment_prob,
                    action_key,
                ) = args
                return update_impl(
                    *core,
                    mc_returns,
                    (
                        structured_explore_start,
                        structured_explore_dimension,
                        structured_explore_delta,
                        structured_explore_assignment_prob,
                    ),
                    *progress_args,
                    action_key,
                )

        else:

            def update_fn(*args):
                args, progress_args = split_progress_args(args)
                (*core, mc_returns, action_key) = args
                return update_impl(
                    *core,
                    mc_returns,
                    None,
                    *progress_args,
                    action_key,
                )

        return update_fn

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
        out = np.sum(candidates * weights[..., None], axis=1)
        if (not eval_mode) and (
            getattr(self, "post_ensemble_l1_flip_prob", 0.0) > 0.0
            or getattr(self, "post_ensemble_l2_flip_prob", 0.0) > 0.0
        ):
            out = self._post_ensemble_bin_flip(out)
        if eval_mode and getattr(self, "log_ensemble_consensus", False):
            self._accumulate_ensemble_consensus(candidates, valid)
        pre = out
        if (
            eval_mode
            and getattr(self, "post_ensemble_random_keep_levels", None)
            is not None
        ):
            out = self._post_ensemble_randomize(out)
        if eval_mode and getattr(self, "log_executed_actions", False):
            trace = getattr(self, "_action_trace", None)
            if trace is None:
                trace = []
                self._action_trace = trace
            trace.append((pre[0].copy(), out[0].copy()))
        return out

    def _accumulate_ensemble_consensus(self, candidates, valid) -> None:
        """Diagnostic: how consistent are the 16 plans' votes for the
        CURRENT step's action, measured on the L0 grid?

        candidates: (B, K, D) -- plan age k's prediction for the current step.
        valid:      (B, K)    -- which history slots hold a real plan.
        Accumulates: exact = votes agreeing with the modal L0 bin;
        adjacent = votes within +-1 bin of the mode; full = (dim, step)
        entries where ALL valid plans agree."""

        lo = np.asarray(self.action_low, dtype=np.float32).reshape(
            self.action_sequence, self.action_dim
        )[0]
        hi = np.asarray(self.action_high, dtype=np.float32).reshape(
            self.action_sequence, self.action_dim
        )[0]
        w0 = (hi - lo) / float(self.bins)
        B, K, D = candidates.shape
        bins0 = np.clip(
            np.floor((candidates - lo[None, None, :]) / w0[None, None, :]),
            0,
            self.bins - 1,
        ).astype(np.int64)
        v = valid.astype(np.float32)  # (B, K)
        counts = np.zeros((B, self.bins, D), dtype=np.float32)
        for k in range(K):
            np.add.at(
                counts,
                (np.arange(B)[:, None], bins0[:, k, :], np.arange(D)[None, :]),
                v[:, k, None],
            )
        nvalid = v.sum(axis=1)  # (B,)
        modal = counts.argmax(axis=1)  # (B, D)
        modal_count = counts.max(axis=1)  # (B, D)
        below = np.take_along_axis(
            counts, np.clip(modal - 1, 0, self.bins - 1)[:, None, :], axis=1
        )[:, 0, :]
        above = np.take_along_axis(
            counts, np.clip(modal + 1, 0, self.bins - 1)[:, None, :], axis=1
        )[:, 0, :]
        below = np.where(modal - 1 < 0, 0.0, below)
        above = np.where(modal + 1 > self.bins - 1, 0.0, above)
        adj_count = modal_count + below + above
        ok = nvalid > 0
        st = getattr(self, "_ensemble_consensus_stats", None)
        if st is None:
            st = {"votes": 0.0, "exact": 0.0, "adjacent": 0.0,
                  "entries": 0.0, "full": 0.0}
            self._ensemble_consensus_stats = st
        st["votes"] += float((nvalid[:, None] * np.ones((1, D)))[ok].sum())
        st["exact"] += float(modal_count[ok].sum())
        st["adjacent"] += float(adj_count[ok].sum())
        st["entries"] += float(ok.sum() * D)
        st["full"] += float((modal_count[ok] == nvalid[ok, None]).sum())

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
        return (
            lo
            + parent_idx * parent_w
            + (leaf + 0.5) * leaf_w
        ).astype(action.dtype)

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
        ) + int(
            batch
            if register_mask is None
            else int(np.sum(register_mask))
        )
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
                width = (high[dim] - low[dim]) / float(
                    self.bins ** (level + 1)
                )
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
            twin_head_indices = np.full((batch_size,), -1, dtype=np.int32)
            if (
                getattr(self, "episodic_twin_head_exploration", False)
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
            rollout_params = self.params
            if (
                getattr(self, "flow_policy", False)
                or getattr(self, "coarse_flow", False)
            ) and getattr(self, "flow_policy_ema", None) is not None:
                # Same pytree structure, EMA leaves: no jit retrace, and the
                # EMA weights flow in as traced arguments.
                rollout_params = {
                    **self.params,
                    "flow_policy": self.flow_policy_ema_params,
                }
            action = self._greedy_action_impl(
                rollout_params,
                self.target_critic_params,
                obs_inputs,
                jnp.asarray(self.use_target_network_for_rollout),
                action_key,
                jnp.asarray(twin_head_indices, dtype=jnp.int32),
            )
            self._block(action)
            action_chunk = np.asarray(jax.device_get(action), dtype=np.float32)
            # Freshest greedy plan (pre-ensemble, pre-exploration), exposed for
            # external value-steered selection (scripts/eval_policy_qselect.py).
            self._last_plan_chunk = action_chunk.copy()
            if (
                getattr(self, "bin_flip_prob", 0.0) > 0.0
                and not self.temporal_ensemble
                and not eval_mode
                and step >= self.num_explore_steps
            ):
                action_chunk = self._apply_bin_flip(action_chunk)
            if (
                getattr(self, "bin_explore_probs", None) is not None
                and not eval_mode
                and step >= self.num_explore_steps
            ):
                if getattr(self, "bin_explore_schedule", None) is not None:
                    self._bin_explore_scale = float(
                        utils.schedule(self.bin_explore_schedule, step)
                    )
                action_chunk = self._apply_bin_explore(
                    action_chunk, register_mask
                )
        else:
            if self.temporal_ensemble:
                action_chunk = np.zeros(
                    (batch_size, self.action_sequence, self.action_dim),
                    dtype=np.float32,
                )
            else:
                prefix = "_eval" if eval_mode else "_train"
                action_chunk = getattr(self, f"{prefix}_open_loop_plan").copy()
        if not eval_mode and getattr(self, "bin_explore_probs", None) is not None:
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
            structured_dimension = np.full(
                (batch_size,), -1, dtype=np.int16
            )
            structured_delta = np.zeros((batch_size,), dtype=np.float32)
            structured_assignment_prob = np.ones(
                (batch_size,), dtype=np.float32
            )
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
                executed_action = (
                    jnp.asarray(executed_action)
                    + stddev
                    * jax.random.normal(
                        noise_key,
                        executed_action.shape,
                    )
                )
                executed_action = jnp.clip(
                    executed_action,
                    self._step_action_low,
                    self._step_action_high,
                )
                if getattr(self, "structured_exploration_prob", 0.0) > 0.0:
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
            if (
                getattr(self, "bin_flip_prob", 0.0) > 0.0
                and not self.temporal_ensemble
            ):
                position = getattr(self, "_train_open_loop_position")
                active = self._bin_flip_remaining > 0
                flip_start = (
                    self._bin_flip_remaining == self.action_sequence
                )
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
                self._last_structured_exploration_assignment_prob = (
                    np.where(
                        flip_start,
                        self.bin_flip_prob,
                        np.where(active, 1.0, 1.0 - self.bin_flip_prob),
                    ).astype(np.float32)
                )
                self._bin_flip_remaining = np.maximum(
                    self._bin_flip_remaining - 1, 0
                )
            if (
                step >= self.num_explore_steps
                and getattr(self, "structured_exploration_prob", 0.0) > 0.0
            ):
                self._structured_exploration_eligible += int(batch_size)
                self._structured_exploration_applied += int(
                    structured_mask.sum()
                )
                self._structured_exploration_starts += int(
                    structured_start.sum()
                )

        action_chunk = action_chunk.copy()
        action_chunk[:, 0] = executed_action
        return action_chunk

    def state_dict(self) -> dict:
        state = super().state_dict()
        if self.flow_policy_ema_params is not None:
            state["flow_policy_ema_params"] = self._tree_to_numpy(
                self.flow_policy_ema_params
            )
        return state

    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict)
        if getattr(self, "flow_policy_ema", None) is not None:
            stored = state_dict.get("flow_policy_ema_params")
            self.flow_policy_ema_params = (
                self._tree_from_numpy(stored)
                if stored is not None
                else jax.tree.map(jnp.array, self.params["flow_policy"])
            )

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
        call, and a restored mid-episode window would leak into them —
        the exact cross-episode intervention that reset() forbids
        (cqn-flow.md 48.3).
        """
        state = {}
        for key, attribute in (
            ("bin_flip_rng_state", "_bin_flip_rng"),
            ("bin_explore_rng_state", "_bin_explore_rng"),
            ("episodic_twin_head_rng_state", "_episodic_twin_head_rng"),
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
        # Snapshots from the short-lived 48.2 format may carry
        # bin_explore_{remaining,dimension,level,sibling}; ignore them so
        # resumed runs start their fresh episodes windowless.

    def rollout_diagnostics(self) -> dict[str, float]:
        eligible = int(
            getattr(self, "_structured_exploration_eligible", 0)
        )
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
        }
        diagnostics.update(
            {
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
        )
        head_assignments = getattr(
            self,
            "_episodic_twin_head_assignments",
            np.zeros((2,), dtype=np.int64),
        )
        assignments = int(head_assignments.sum())
        diagnostics.update(
            {
                "episodic_twin_head_assignments": float(assignments),
                "episodic_twin_head0_rate": (
                    float(head_assignments[0] / assignments)
                    if assignments
                    else 0.0
                ),
                "episodic_twin_head1_rate": (
                    float(head_assignments[1] / assignments)
                    if assignments
                    else 0.0
                ),
            }
        )
        return diagnostics

    def update(
        self,
        replay_iter: Iterator[dict],
        step: int,
        replay_buffer: ReplayBuffer = None,
    ) -> dict[str, np.ndarray]:
        """Update CQN-AS, including MC returns for the decoupled critic path."""

        if not getattr(self, "separate_bc_policy", False):
            metrics = super().update(replay_iter, step, replay_buffer)
            if (
                getattr(self, "coarse_flow", False)
                and self.flow_policy_ema is not None
            ):
                decay = self.flow_policy_ema
                self.flow_policy_ema_params = jax.tree.map(
                    lambda ema, online: decay * ema + (1.0 - decay) * online,
                    self.flow_policy_ema_params,
                    self.params["flow_policy"],
                )
            return metrics

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
            mc_returns = self._as_jax_array(
                batch.get("mc_return", np.zeros_like(batch["reward"])),
                self.jnp.float32,
            ).reshape(-1)
            direct_q_extra_args = ()
            if (
                getattr(self, "direct_scalar_q", False)
                or getattr(self, "cv_rct_weight", None) is not None
            ):
                direct_q_extra_args = (
                    self._as_jax_array(
                        batch.get(
                            "structured_explore_start",
                            np.zeros_like(batch["reward"]),
                        ),
                        self.jnp.float32,
                    ).reshape(-1),
                    self._as_jax_array(
                        batch.get(
                            "structured_explore_dimension",
                            np.full_like(batch["reward"], -1),
                        ),
                        self.jnp.int32,
                    ).reshape(-1),
                    self._as_jax_array(
                        batch.get(
                            "structured_explore_delta",
                            np.zeros_like(batch["reward"]),
                        ),
                        self.jnp.float32,
                    ).reshape(-1),
                    self._as_jax_array(
                        batch.get(
                            "structured_explore_assignment_prob",
                            np.ones_like(batch["reward"]),
                        ),
                        self.jnp.float32,
                    ).reshape(-1),
                )
            progress_args = ()
            if getattr(self, "progress_head_enabled", False):
                progress_args = self._progress_update_args(batch, step)
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
                mc_returns,
                *direct_q_extra_args,
                *progress_args,
                self._next_action_key(),
            )
            if (
                getattr(self, "flow_policy", False)
                and self.flow_policy_ema is not None
            ):
                decay = self.flow_policy_ema
                self.flow_policy_ema_params = jax.tree.map(
                    lambda ema, online: decay * ema + (1.0 - decay) * online,
                    self.flow_policy_ema_params,
                    self.params["flow_policy"],
                )
            uses_priorities = self._uses_replay_priorities(replay_buffer)
            if self._should_block_update(uses_priorities):
                self._block(jax_metrics["critic_loss"], priority)
            committed_metric = jax_metrics.get(
                "nan_diag/update_committed", None
            )
            update_committed = True
            if committed_metric is not None:
                update_committed = bool(
                    float(np.asarray(jax.device_get(committed_metric))) >= 0.5
                )
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
            elif not update_committed:
                metrics.update(
                    {
                        key: float(np.asarray(jax.device_get(value)))
                        for key, value in jax_metrics.items()
                    }
                )
        self._first_update_completed = True
        return metrics

    def _resample_episodic_twin_heads(self, agent_indices: list[int]) -> None:
        if (
            not getattr(self, "episodic_twin_head_exploration", False)
            or not agent_indices
        ):
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
        del step
        self._resample_episodic_twin_heads(agents_to_reset)
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
                    and agent_index
                    < self._bin_explored_exec_remaining.shape[0]
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
    "AutoregressiveActionCorrection",
    "AutoregressiveSequenceDistributionalCritic",
    "C2FSequenceDistributionalCritic",
    "CQNAS",
    "CQNASpec",
    "cqn_as_spec_from_cfg",
    "select_episodic_twin_actions",
    "top2_joint_beam",
]
