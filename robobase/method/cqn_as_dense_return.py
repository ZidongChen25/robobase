"""CQN-AS dense-return variant (the no-BC research line's return machinery).

This module isolates ONE research line out of the pre-refactor monolith
(``cqn_as_research.py`` / ``cqn_research.py``): the reward/return side of the
"no-BC" (``strict_demo_rl_only``) programme.  It subclasses the FROZEN pristine
:class:`robobase.method.cqn_as.CQNAS` and overrides ``_build_update_fn`` /
``update`` with a copy of the pristine :class:`robobase.method.cqn.CQN` bodies
plus only this line's changes.

Flags owned here (see ``robobase/cfgs/method/cqn_as_dense_return.yaml``):

``dense_return_q_target``
    Single dense distributional-Q objective: the replayed bin gets its
    reward/Bellman target and every other bin gets the task-minimum return.
``dense_return_positive_only``
    Restrict the dense target to transitions from completed positive-return
    trajectories; zero-return transitions keep canonical chosen-action C51 TD.
``dense_return_expected_q_loss``
    Regress only the expectation of that same dense target (scalar MSE)
    instead of its full categorical shape.
``dense_return_advantage_alpha`` / ``dense_return_advantage_clip_ratio``
    Baird advantage-learning operator applied inside the same categorical
    target, optionally clipped to near-greedy bins.
``dense_return_label_smoothing``
    Convex mix of every bin target with the uniform distribution, giving the
    categorical cross-entropy a finite logit optimum.
``dense_return_floor_satisfaction_margin``
    Satisficing floor: suspend the CE on floor-targeted bins already within a
    margin of the floor (target-conditioned, never label-conditioned).
``dense_return_relative_floor_margin``
    Supervise counterfactual bins to ``max(E[chosen] - m, floor)`` instead of
    the absolute floor.
``dense_return_finest_neighbor_weight``
    Local-Q kernel: immediate neighbour bins at the finest C2F level receive a
    convex mixture of the chosen and floor distributions.
``return_gated_margin`` / ``return_gated_margin_weight``
    One-sided expected-Q hinge keeping the executed bin ahead of its siblings,
    gated purely by measured positive completed return.
``q_reward_scale``
    Positive affine reward scale inside the Bellman target.
``episodic_success_q_target``
    Binary episodic Monte-Carlo control: the replayed bin gets 1 iff its
    completed trajectory succeeded, 0 otherwise.
``ordered_success_return_mix`` / ``sequence_aligned_mc_discount``
    Return transforms that only ever fed the MC-lower-bound target; see the
    coupling note below.
``unseen_return_floor_*``
    Q-Transformer-style conservative regression of replay-unseen bins to a
    valid minimum return (mean / max / topk reduction).
``strict_demo_rl_only`` / ``strict_allow_reward_only_success_replay``
    Config-composition audit guard for demo-driven RL with no imitation path.

Anti-imitation invariant preserved by every term here: on a zero-return sample
the chosen bin's target equals the floor, so both the loss and its logit
gradients are exactly invariant to the recorded action label.

COUPLING (mc-rct line).  ``ordered_success_return_mix`` and
``sequence_aligned_mc_discount`` are return transforms whose ONLY consumer in
the monolith is the ``mc_lower_bound_target`` ``max(TD, MC)`` branch
(``cqn_research.py`` lines 1481-1528), which belongs to the ``mc-rct`` line.
The pure transforms are exported here (they are return machinery); the flags
are validated and REFUSED rather than silently ignored, and no ``mc_*`` flag is
absorbed.  ``episodic_success_q_target``, ``dense_return_positive_only`` and
``return_gated_margin`` only need the completed-episode return VALUE from
replay (``mc_return``), not any mc-rct flag, so they are implemented in full.
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

from robobase.method.cqn import encode_action, project_categorical
from robobase.method.cqn_as import CQNAS, CQNASpec, cqn_as_spec_from_cfg
from robobase.method.rl_common import RLModelSpec
from robobase.replay_buffer.replay_buffer import ReplayBuffer


# ---------------------------------------------------------------------------
# Return machinery (verbatim from the research monolith, cqn_research.py)
# ---------------------------------------------------------------------------


def shift_categorical_distribution(
    probabilities: jax.Array,
    shifts: jax.Array,
    support: jax.Array,
) -> jax.Array:
    """Translate arbitrary C51 distributions by one scalar per distribution."""

    probabilities = jnp.asarray(probabilities, dtype=jnp.float32)
    shifts = jnp.asarray(shifts, dtype=jnp.float32)
    if probabilities.shape[:-1] != shifts.shape:
        raise ValueError(
            "categorical shifts must match every non-atom dimension"
        )
    atoms = int(probabilities.shape[-1])
    flat_probabilities = probabilities.reshape((-1, 1, 1, atoms))
    flat_shifts = shifts.reshape((-1,))
    shifted = project_categorical(
        flat_probabilities,
        flat_shifts,
        jnp.ones_like(flat_shifts),
        jnp.ones_like(flat_shifts),
        support,
    )
    return shifted.reshape(probabilities.shape)


def advantage_learning_target_shift(
    q_values: jax.Array,
    q_lower: float | jax.Array,
    alpha: float,
    clip_ratio: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Return advantage-learning shifts and their active-bin mask."""

    values = jnp.asarray(q_values, dtype=jnp.float32)
    state_value = jnp.max(values, axis=-1, keepdims=True)
    disadvantage = state_value - values
    if clip_ratio is None:
        active = jnp.ones_like(values, dtype=bool)
    else:
        lower = jnp.asarray(q_lower, dtype=values.dtype)
        denominator = jnp.maximum(
            state_value - lower,
            jnp.finfo(values.dtype).eps,
        )
        relative_value = (values - lower) / denominator
        active = relative_value >= float(clip_ratio)
    shifts = -float(alpha) * disadvantage * active.astype(values.dtype)
    return shifts, active


def sequence_aligned_sparse_returns(
    mc_returns: jax.Array,
    action_sequence: int,
    action_dim: int,
    discount: float,
) -> jax.Array:
    """Recover each future token's return for a sparse terminal reward.

    Exported for completeness: in the monolith this transform is consumed only
    by the mc-rct ``mc_lower_bound_target`` branch, so
    :class:`CQNASDenseReturn` refuses ``sequence_aligned_mc_discount`` rather
    than absorbing that foreign flag.
    """

    values = jnp.asarray(mc_returns, dtype=jnp.float32)
    token_offsets = jnp.repeat(
        jnp.arange(int(action_sequence), dtype=jnp.float32),
        int(action_dim),
    )
    discount_powers = jnp.power(
        jnp.asarray(discount, dtype=jnp.float32),
        token_offsets,
    )
    aligned = values[:, None] / discount_powers[None, :]
    return jnp.where(
        values[:, None] > 0.0,
        jnp.clip(aligned, 0.0, 1.0),
        0.0,
    )


def unseen_return_floor_loss(
    all_logits: jax.Array,
    discrete_action: jax.Array,
    support: jax.Array,
    floor_value: float,
    reduction: str = "mean",
    topk: int = 1,
    support_mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Regress only replay-unseen bins to a valid minimum return.

    ``all_logits`` is ``[B, L, D, bins, atoms]`` and ``discrete_action`` is
    ``[B, L, D]``.  The replayed bin is explicitly masked out: it is trained
    only by the Bellman/return objective.  This is the squared conservative-Q
    regularizer used by Q-Transformer, adapted to CQN's C51 expected values;
    it is not an action likelihood, classification, or margin loss.  With a
    ``support_mask`` of admissible bins ``[B, L, D, bins]``, the floored set
    shrinks to unseen AND out-of-support bins; in-support unexecuted
    siblings are left entirely to the TD objective.
    """

    all_probabilities = jax.nn.softmax(all_logits, axis=-1)
    all_q = jnp.sum(all_probabilities * support, axis=-1)
    chosen_mask = jax.nn.one_hot(
        discrete_action,
        all_logits.shape[-2],
        dtype=all_q.dtype,
    )
    unseen_mask = 1.0 - chosen_mask
    if support_mask is not None:
        unseen_mask = unseen_mask * (
            1.0 - support_mask.astype(all_q.dtype)
        )
    reduction = str(reduction).lower()
    if reduction == "mean":
        reduce_axes = tuple(range(1, unseen_mask.ndim))
        unseen_count = jnp.maximum(
            jnp.sum(unseen_mask, axis=reduce_axes),
            1.0,
        )
        per_sample_loss = (
            jnp.sum(
                jnp.square(all_q - float(floor_value)) * unseen_mask,
                axis=reduce_axes,
            )
            / unseen_count
        )
        per_sample_unseen_q = (
            jnp.sum(all_q * unseen_mask, axis=reduce_axes) / unseen_count
        )
    elif reduction in {"max", "topk"}:
        # Greedy control depends on the single largest competing bin, not the
        # average unseen value. Apply the same absolute reward floor to the
        # requested upper tail independently for every C2F action head.
        tail_count = 1 if reduction == "max" else int(topk)
        if not 1 <= tail_count < all_logits.shape[-2]:
            raise ValueError(
                "unseen return floor topk must be in [1, bins - 1]"
            )
        sorted_unseen_q = jnp.sort(
            jnp.where(unseen_mask.astype(bool), all_q, -jnp.inf),
            axis=-1,
        )
        tail_unseen_q = sorted_unseen_q[..., -tail_count:]
        if support_mask is not None:
            tail_unseen_q = jnp.where(
                jnp.isfinite(tail_unseen_q),
                tail_unseen_q,
                float(floor_value),
            )
        reduce_axes = tuple(range(1, tail_unseen_q.ndim))
        per_sample_loss = jnp.mean(
            jnp.square(tail_unseen_q - float(floor_value)),
            axis=reduce_axes,
        )
        per_sample_unseen_q = jnp.mean(
            tail_unseen_q,
            axis=reduce_axes,
        )
    else:
        raise ValueError(
            "unseen return floor reduction must be one of "
            "{'mean', 'max', 'topk'}"
        )
    return per_sample_loss, per_sample_unseen_q


def dense_return_distributional_loss(
    all_logits: jax.Array,
    discrete_action: jax.Array,
    chosen_target_distribution: jax.Array,
    support: jax.Array,
    floor_value: float,
    finest_neighbor_weight: float = 0.0,
    advantage_alpha: float = 0.0,
    advantage_clip_ratio: float | None = None,
    label_smoothing: float = 0.0,
    floor_satisfaction_margin: float | None = None,
    relative_floor_margin: float | None = None,
    support_mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Regress every action bin to an absolute return distribution.

    The replayed bin receives its reward/Bellman-derived C51 target; every
    other bin normally receives a point mass at the task's valid minimum
    return. With a positive ``finest_neighbor_weight``, immediate neighboring
    bins at only the finest C2F level receive the corresponding convex mixture
    of chosen and floor distributions. This is a local continuous-action Q
    smoothness prior; coarse neighbors remain at the floor because they need
    not denote a small action perturbation. Summing over bins preserves the
    canonical chosen-action TD gradient while adding one return target for
    each counterfactual bin.

    When ``chosen_target_distribution`` is itself the floor point mass, every
    bin has the same target. The loss and its logit gradient are then exactly
    invariant to ``discrete_action``; action identity carries no signal
    without a return difference. A positive ``advantage_alpha`` applies
    Baird's single-objective advantage-learning operator uniformly to every
    bin by translating its base return distribution by
    ``-alpha * (max_b Q(s,b) - Q(s,a))``. The shift is stop-gradient.
    With a clipping ratio, the shift is applied only to bins whose
    support-relative value is sufficiently close to the current maximum.
    With a ``support_mask`` of admissible bins ``[B, L, D, bins]``,
    in-support unexecuted siblings drop out of the loss entirely: they
    receive no floor target and stay under TD control, while out-of-support
    bins keep the floor regression.
    """

    atoms = int(support.shape[0])
    v_min = support[0]
    v_max = support[-1]
    delta = (v_max - v_min) / float(atoms - 1)
    projected = jnp.clip(
        (jnp.asarray(floor_value, dtype=all_logits.dtype) - v_min) / delta,
        0.0,
        float(atoms - 1),
    )
    lower = jnp.floor(projected).astype(jnp.int32)
    upper = jnp.ceil(projected).astype(jnp.int32)
    lower_weight = jnp.where(
        lower == upper,
        1.0,
        upper.astype(all_logits.dtype) - projected,
    )
    upper_weight = jnp.where(
        lower == upper,
        0.0,
        projected - lower.astype(all_logits.dtype),
    )
    floor_distribution = jnp.zeros(
        (atoms,),
        dtype=all_logits.dtype,
    )
    floor_distribution = floor_distribution.at[lower].add(lower_weight)
    floor_distribution = floor_distribution.at[upper].add(upper_weight)

    if relative_floor_margin is not None:
        # Relative floor: supervise counterfactual bins to "slightly
        # worse than the executed action" instead of the absolute task
        # minimum. The generalized lesson across nearby (off-manifold)
        # states becomes a smooth preference field with a small
        # runner-up gap rather than "everything here is worthless" --
        # the measured cause of off-manifold value collapse. At a
        # zero-return sample the shifted value clips back to the floor,
        # so every bin's target coincides and the exact action-label
        # invariance is preserved.
        chosen_expected = jnp.sum(
            jax.lax.stop_gradient(chosen_target_distribution) * support,
            axis=-1,
        )
        rel_value = jnp.clip(
            chosen_expected - relative_floor_margin,
            jnp.asarray(floor_value, dtype=all_logits.dtype),
            v_max,
        )
        rel_projected = jnp.clip(
            (rel_value - v_min) / delta,
            0.0,
            float(atoms - 1),
        )
        rel_lower = jnp.floor(rel_projected).astype(jnp.int32)
        rel_upper = jnp.ceil(rel_projected).astype(jnp.int32)
        rel_lower_weight = jnp.where(
            rel_lower == rel_upper,
            1.0,
            rel_upper.astype(all_logits.dtype) - rel_projected,
        )
        rel_upper_weight = jnp.where(
            rel_lower == rel_upper,
            0.0,
            rel_projected - rel_lower.astype(all_logits.dtype),
        )
        one_hot_lower = jax.nn.one_hot(
            rel_lower, atoms, dtype=all_logits.dtype
        )
        one_hot_upper = jax.nn.one_hot(
            rel_upper, atoms, dtype=all_logits.dtype
        )
        floor_distribution = (
            rel_lower_weight[..., None] * one_hot_lower
            + rel_upper_weight[..., None] * one_hot_upper
        )[..., None, :]

    chosen_mask = jax.nn.one_hot(
        discrete_action,
        all_logits.shape[-2],
        dtype=all_logits.dtype,
    )
    bin_index = jnp.arange(
        all_logits.shape[-2],
        dtype=discrete_action.dtype,
    )
    neighbor_mask = (
        jnp.abs(bin_index - discrete_action[..., None]) == 1
    ).astype(all_logits.dtype)
    finest_level_mask = jax.nn.one_hot(
        all_logits.shape[1] - 1,
        all_logits.shape[1],
        dtype=all_logits.dtype,
    )[None, :, None, None]
    neighbor_mask = neighbor_mask * finest_level_mask
    kernel_weight = chosen_mask + (
        jnp.asarray(finest_neighbor_weight, dtype=all_logits.dtype)
        * neighbor_mask
    )
    targets = (
        kernel_weight[..., None] * chosen_target_distribution[..., None, :]
        + (1.0 - kernel_weight[..., None]) * floor_distribution
    )
    all_probabilities = jax.nn.softmax(all_logits, axis=-1)
    all_q = jnp.sum(all_probabilities * support, axis=-1)
    if advantage_alpha > 0.0:
        target_shift, _ = advantage_learning_target_shift(
            jax.lax.stop_gradient(all_q),
            support[0],
            advantage_alpha,
            advantage_clip_ratio,
        )
        targets = shift_categorical_distribution(
            targets,
            target_shift,
            support,
        )
        targets = jax.lax.stop_gradient(targets)
    if label_smoothing > 0.0:
        # A convex mix with the uniform distribution gives the categorical
        # cross-entropy a finite logit optimum, so training self-terminates
        # instead of sharpening point-mass targets indefinitely. Uniform
        # mass over a symmetric support has expectation zero, the same as
        # the floor, so bin expectations keep their ordering and the
        # zero-return action-label invariance is preserved exactly (the
        # mix is applied identically to every bin's target).
        targets = (
            (1.0 - label_smoothing) * targets
            + label_smoothing / float(atoms)
        )
    per_bin_loss = -jnp.sum(
        targets * jax.nn.log_softmax(all_logits, axis=-1),
        axis=-1,
    )
    if floor_satisfaction_margin is not None:
        # Satisficing floor: suspend the cross-entropy on any bin whose
        # target IS the floor distribution and whose expected value
        # already sits within the margin of the floor. The rule is
        # target-conditioned, not label-conditioned: on a zero-return
        # sample the replayed bin's target equals the floor, so every
        # bin follows the identical value-conditioned rule and the exact
        # action-label invariance is preserved. Positive-return chosen
        # bins are never suspended. This gives the floor term a
        # margin-style finite optimum (training self-terminates once
        # constraints are satisfied) without injecting entropy into the
        # bootstrap path.
        is_floor_target = (
            jnp.max(
                jnp.abs(targets - floor_distribution),
                axis=-1,
            )
            < 1e-6
        )
        probabilities = jax.lax.stop_gradient(
            jax.nn.softmax(all_logits, axis=-1)
        )
        expected_q = jnp.sum(probabilities * support, axis=-1)
        satisfied = is_floor_target & (
            expected_q
            <= (
                jnp.asarray(floor_value, dtype=all_logits.dtype)
                + floor_satisfaction_margin
            )
        )
        per_bin_loss = per_bin_loss * (
            1.0 - satisfied.astype(per_bin_loss.dtype)
        )
    if support_mask is not None:
        floor_exempt = support_mask.astype(per_bin_loss.dtype) * (
            1.0 - chosen_mask
        )
        per_bin_loss = per_bin_loss * (1.0 - floor_exempt)
    per_sample_loss = per_bin_loss.sum(axis=-1).mean(axis=(1, 2))

    chosen_q = jnp.sum(all_q * chosen_mask, axis=-1).mean(axis=(1, 2))
    unseen_mask = 1.0 - chosen_mask
    unseen_axes = tuple(range(1, unseen_mask.ndim))
    unseen_count = jnp.maximum(
        jnp.sum(unseen_mask, axis=unseen_axes),
        1.0,
    )
    unseen_q = (
        jnp.sum(all_q * unseen_mask, axis=unseen_axes) / unseen_count
    )
    return per_sample_loss, chosen_q, unseen_q


def return_gated_margin_loss(
    all_logits: jax.Array,
    discrete_action: jax.Array,
    support: jax.Array,
    margin: float,
    positive_mask: jax.Array,
) -> jax.Array:
    """One-sided expected-Q hinge keeping the executed bin ahead by a
    margin, gated PURELY by measured positive return.

    For every head of a transition whose completed return is positive,
    penalise ``relu(Q_b - (Q_chosen - margin))`` on the non-executed
    bins. The gate reads reward-to-go only -- never a demonstration
    flag -- and applies identically to successful online transitions.
    On zero-return samples the term vanishes exactly, so the loss and
    its gradients are invariant to the recorded action label there (the
    project's operational anti-imitation test). Unlike an absolute
    floor, the hinge constrains RELATIVE separation, so it survives
    global level drift.
    """

    probabilities = jax.nn.softmax(all_logits, axis=-1)
    all_q = jnp.sum(probabilities * support, axis=-1)
    chosen_mask = jax.nn.one_hot(
        discrete_action, all_logits.shape[-2], dtype=all_logits.dtype
    )
    chosen_q = jnp.sum(all_q * chosen_mask, axis=-1, keepdims=True)
    violation = jax.nn.relu(all_q - (chosen_q - margin))
    violation = violation * (1.0 - chosen_mask)
    per_sample = violation.sum(axis=-1).mean(axis=(1, 2))
    return per_sample * positive_mask.astype(per_sample.dtype)


def dense_return_expected_q_loss(
    all_logits: jax.Array,
    discrete_action: jax.Array,
    chosen_target_distribution: jax.Array,
    support: jax.Array,
    floor_value: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Regress every action bin only on the expected return used for control.

    The replayed bin receives the expectation of its Bellman/MC categorical
    target and every counterfactual bin receives the task-valid floor.  When
    the chosen expectation equals the floor, all bin targets are identical,
    so both loss and logit gradient are independent of the replay action.
    """

    all_probabilities = jax.nn.softmax(all_logits, axis=-1)
    all_q = jnp.sum(all_probabilities * support, axis=-1)
    chosen_target_q = jnp.sum(
        chosen_target_distribution * support,
        axis=-1,
    )
    chosen_mask = jax.nn.one_hot(
        discrete_action,
        all_logits.shape[-2],
        dtype=all_logits.dtype,
    )
    floor = jnp.asarray(floor_value, dtype=all_logits.dtype)
    targets = (
        chosen_mask * chosen_target_q[..., None]
        + (1.0 - chosen_mask) * floor
    )
    per_bin_loss = 0.5 * jnp.square(all_q - targets)
    per_sample_loss = per_bin_loss.sum(axis=-1).mean(axis=(1, 2))

    chosen_q = jnp.sum(all_q * chosen_mask, axis=-1).mean(axis=(1, 2))
    unseen_mask = 1.0 - chosen_mask
    unseen_axes = tuple(range(1, unseen_mask.ndim))
    unseen_count = jnp.maximum(
        jnp.sum(unseen_mask, axis=unseen_axes),
        1.0,
    )
    unseen_q = (
        jnp.sum(all_q * unseen_mask, axis=unseen_axes) / unseen_count
    )
    return per_sample_loss, chosen_q, unseen_q


def episodic_success_returns(mc_returns: jax.Array) -> jax.Array:
    """Map completed sparse-task returns to undiscounted success outcomes.

    A positive reward-to-go means that the completed trajectory eventually
    succeeded, regardless of how much discount attenuated that terminal bit.
    Zero or negative return maps to failure. No demonstration identity or
    action label is consulted.
    """

    values = jnp.asarray(mc_returns)
    return (values > 0.0).astype(jnp.float32)


def ordered_success_returns(
    mc_returns: jax.Array,
    success_mix: float,
) -> jax.Array:
    """Lift positive discounted returns while preserving their ordering.

    Exported for completeness: in the monolith this transform is consumed only
    by the mc-rct ``mc_lower_bound_target`` branch, so
    :class:`CQNASDenseReturn` refuses ``ordered_success_return_mix`` rather
    than absorbing that foreign flag.
    """

    values = jnp.asarray(mc_returns, dtype=jnp.float32)
    positive_values = jnp.where(values > 0.0, values, 0.0)
    success = episodic_success_returns(values)
    mix = jnp.asarray(success_mix, dtype=jnp.float32)
    return (1.0 - mix) * positive_values + mix * success


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CQNASDenseReturnSpec(CQNASpec):
    """Pristine CQN-AS hyperparameters plus the dense-return line's flags."""

    strict_demo_rl_only: bool
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


def _validate_strict_demo_rl_only(cfg: DictConfig) -> bool:
    """Audit guard for demo-driven RL experiments with no imitation path.

    Copied from ``cqn_as_research.cqn_as_spec_from_cfg``.  Every lookup uses
    ``.get`` with the inert default, so keys owned by other research lines are
    still rejected when someone adds them to a dense-return config.
    """

    method = cfg.method
    strict_demo_rl_only = bool(method.get("strict_demo_rl_only", False))
    if not strict_demo_rl_only:
        return False
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
    return True


def cqn_as_dense_return_spec_from_cfg(cfg: DictConfig) -> CQNASDenseReturnSpec:
    method = cfg.method
    strict_demo_rl_only = _validate_strict_demo_rl_only(cfg)
    base = cqn_as_spec_from_cfg(cfg)
    base_values = {
        field.name: getattr(base, field.name) for field in fields(CQNASpec)
    }
    return CQNASDenseReturnSpec(
        **base_values,
        strict_demo_rl_only=strict_demo_rl_only,
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
        ),
        unseen_return_floor_topk=int(
            method.get("unseen_return_floor_topk", 1)
        ),
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class CQNASDenseReturn(CQNAS):
    """Pristine CQN-AS plus the dense-return / conservative-floor objective.

    Every flag defaults to OFF; with the defaults, ``_build_update_fn`` reduces
    to the pristine :meth:`robobase.method.cqn.CQN._build_update_fn` body and
    ``update`` to the pristine :meth:`robobase.method.cqn.CQN.update` body
    (plus one unused zero-valued ``mc_returns`` argument).
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
        strict_demo_rl_only: bool = False,
        dense_return_q_target: bool = False,
        dense_return_positive_only: bool = False,
        dense_return_expected_q_loss: bool = False,
        dense_return_advantage_alpha: float = 0.0,
        dense_return_advantage_clip_ratio: float | None = None,
        q_reward_scale: float = 1.0,
        dense_return_label_smoothing: float = 0.0,
        dense_return_floor_satisfaction_margin: float | None = None,
        dense_return_relative_floor_margin: float | None = None,
        return_gated_margin: float | None = None,
        return_gated_margin_weight: float = 0.0,
        dense_return_finest_neighbor_weight: float = 0.0,
        episodic_success_q_target: bool = False,
        ordered_success_return_mix: float = 0.0,
        sequence_aligned_mc_discount: float | None = None,
        unseen_return_floor_weight: float = 0.0,
        unseen_return_floor_value: float = 0.0,
        unseen_return_floor_reduction: str = "mean",
        unseen_return_floor_topk: int = 1,
    ):
        # -- validation (copied from cqn_as_research.CQNAS.__init__, with the
        #    ``mc_lower_bound_target`` clauses handled per the coupling note in
        #    this module's docstring) --
        if dense_return_positive_only and not dense_return_q_target:
            raise ValueError(
                "dense_return_positive_only requires "
                "dense_return_q_target=true."
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
        if ordered_success_return_mix > 0.0:
            # Coupling: the only consumer of the ordered-success transform is
            # the mc-rct ``mc_lower_bound_target`` max(TD, MC) branch, which
            # this variant deliberately does not implement.
            raise ValueError(
                "ordered_success_return_mix requires "
                "mc_lower_bound_target=true, which lives in the mc-rct "
                "variant; CQNASDenseReturn does not implement the "
                "MC-lower-bound target."
            )
        if sequence_aligned_mc_discount is not None:
            if not 0.0 < sequence_aligned_mc_discount <= 1.0:
                raise ValueError(
                    "sequence_aligned_mc_discount must be in (0, 1]."
                )
            # Same coupling as ordered_success_return_mix.
            raise ValueError(
                "sequence_aligned_mc_discount requires "
                "mc_lower_bound_target=true, which lives in the mc-rct "
                "variant; CQNASDenseReturn does not implement the "
                "MC-lower-bound target."
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

        # ``CQNAS.__init__`` calls ``self._build_update_fn()``, so every flag
        # the closure reads must already be bound.
        self.strict_demo_rl_only = bool(strict_demo_rl_only)
        self.dense_return_q_target = bool(dense_return_q_target)
        self.dense_return_positive_only = bool(dense_return_positive_only)
        self.dense_return_expected_q_loss = bool(dense_return_expected_q_loss)
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
        self.episodic_success_q_target = bool(episodic_success_q_target)
        self.ordered_success_return_mix = float(ordered_success_return_mix)
        self.sequence_aligned_mc_discount = (
            None
            if sequence_aligned_mc_discount is None
            else float(sequence_aligned_mc_discount)
        )
        self.unseen_return_floor_weight = float(unseen_return_floor_weight)
        self.unseen_return_floor_value = float(unseen_return_floor_value)
        self.unseen_return_floor_reduction = unseen_return_floor_reduction
        self.unseen_return_floor_topk = int(unseen_return_floor_topk)
        # Completed-episode returns are needed by the three terms that gate on
        # measured reward-to-go. In the monolith this data dependency was
        # spelled as a config guard (``mc_lower_bound_target=true``); here it
        # is a direct replay-element requirement, checked in ``update``.
        self._uses_mc_returns = bool(
            self.episodic_success_q_target
            or self.dense_return_positive_only
            or (
                self.return_gated_margin is not None
                and self.return_gated_margin_weight > 0.0
            )
        )

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
            temporal_ensemble_replan_interval=(
                temporal_ensemble_replan_interval
            ),
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

    # -- update path: pristine ``CQN._build_update_fn`` + dense-return terms --

    def _build_update_fn(self):
        optimizer = self.optimizer
        tau = self.critic_target_tau

        # dense-return line configuration, resolved once outside the trace
        q_reward_scale = float(self.q_reward_scale)
        use_dense_return_target = bool(self.dense_return_q_target)
        dense_return_positive_only = bool(self.dense_return_positive_only)
        use_dense_expected_q_target = bool(self.dense_return_expected_q_loss)
        dense_return_advantage_alpha = float(
            self.dense_return_advantage_alpha
        )
        dense_return_advantage_clip_ratio = (
            self.dense_return_advantage_clip_ratio
        )
        dense_return_finest_neighbor_weight = float(
            self.dense_return_finest_neighbor_weight
        )
        dense_return_label_smoothing = float(
            self.dense_return_label_smoothing
        )
        dense_return_floor_satisfaction_margin = (
            self.dense_return_floor_satisfaction_margin
        )
        dense_return_relative_floor_margin = (
            self.dense_return_relative_floor_margin
        )
        return_gated_margin = self.return_gated_margin
        return_gated_margin_weight = float(self.return_gated_margin_weight)
        use_episodic_success_target = bool(self.episodic_success_q_target)
        return_floor_weight = float(self.unseen_return_floor_weight)
        use_return_floor = return_floor_weight > 0.0
        return_floor_value = float(self.unseen_return_floor_value)
        return_floor_reduction = str(self.unseen_return_floor_reduction)
        return_floor_topk = int(self.unseen_return_floor_topk)

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
            mc_returns,
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
                    rewards * q_reward_scale,
                    discounts,
                    bootstrap,
                    self.support,
                )
                if self.centralized_critic:
                    target_distribution = jnp.broadcast_to(
                        target_distribution.mean(axis=-2, keepdims=True),
                        target_distribution.shape,
                    )
                if use_episodic_success_target:
                    # Sparse-success Monte-Carlo control. Completed-trajectory
                    # reward, not demo identity, supplies the only positive
                    # signal. A failed demo and failed online trajectory both
                    # map to zero; any successful trajectory maps to one.
                    episodic_success = episodic_success_returns(mc_returns)
                    target_distribution = project_categorical(
                        target_probabilities,
                        episodic_success,
                        jnp.zeros_like(discounts),
                        jnp.zeros_like(bootstrap),
                        self.support,
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
                canonical_per_sample = -jnp.sum(
                    target_distribution * chosen_log_probabilities,
                    axis=-1,
                ).mean(axis=(1, 2))

                dense_chosen_q = jnp.asarray(0.0, dtype=jnp.float32)
                dense_unseen_q = jnp.asarray(0.0, dtype=jnp.float32)
                dense_positive_fraction = jnp.asarray(0.0, dtype=jnp.float32)
                if use_dense_return_target:
                    flat_actions = actions.reshape((actions.shape[0], -1))
                    discrete_action = encode_action(
                        flat_actions,
                        self.action_low,
                        self.action_high,
                        self.levels,
                        self.bins,
                    )
                    discrete_action = discrete_action[
                        :, :, : all_logits.shape[2]
                    ]
                    if use_dense_expected_q_target:
                        (
                            dense_per_sample,
                            dense_chosen_q,
                            dense_unseen_q,
                        ) = dense_return_expected_q_loss(
                            all_logits,
                            discrete_action,
                            target_distribution,
                            self.support,
                            return_floor_value,
                        )
                    else:
                        (
                            dense_per_sample,
                            dense_chosen_q,
                            dense_unseen_q,
                        ) = dense_return_distributional_loss(
                            all_logits,
                            discrete_action,
                            target_distribution,
                            self.support,
                            return_floor_value,
                            dense_return_finest_neighbor_weight,
                            dense_return_advantage_alpha,
                            dense_return_advantage_clip_ratio,
                            dense_return_label_smoothing,
                            dense_return_floor_satisfaction_margin,
                            dense_return_relative_floor_margin,
                        )
                    if dense_return_positive_only:
                        dense_positive_mask = mc_returns > 0.0
                        per_sample = jnp.where(
                            dense_positive_mask,
                            dense_per_sample,
                            canonical_per_sample,
                        )
                        dense_positive_fraction = jnp.mean(
                            dense_positive_mask.astype(jnp.float32)
                        )
                    else:
                        per_sample = dense_per_sample
                    if (
                        return_gated_margin is not None
                        and return_gated_margin_weight > 0.0
                    ):
                        per_sample = per_sample + (
                            return_gated_margin_weight
                            * return_gated_margin_loss(
                                all_logits,
                                discrete_action,
                                self.support,
                                return_gated_margin,
                                mc_returns > 0.0,
                            )
                        )
                else:
                    per_sample = canonical_per_sample
                critic_loss = self.critic_lambda * jnp.mean(
                    per_sample * loss_weights
                )
                dense_return_q_loss = jnp.where(
                    use_dense_return_target,
                    critic_loss,
                    jnp.asarray(0.0, dtype=jnp.float32),
                )

                return_floor_loss = jnp.asarray(0.0, dtype=jnp.float32)
                unseen_q_mean = jnp.mean(dense_unseen_q)
                chosen_q_mean = jnp.mean(dense_chosen_q)
                if use_return_floor:
                    flat_actions = actions.reshape((actions.shape[0], -1))
                    discrete_action = encode_action(
                        flat_actions,
                        self.action_low,
                        self.action_high,
                        self.levels,
                        self.bins,
                    )
                    # CQN-AS can train only the actually executed k=0 token.
                    discrete_action = discrete_action[
                        :, :, : all_logits.shape[2]
                    ]
                    floor_per_sample, unseen_q_per_sample = (
                        unseen_return_floor_loss(
                            all_logits,
                            discrete_action,
                            self.support,
                            return_floor_value,
                            reduction=return_floor_reduction,
                            topk=return_floor_topk,
                        )
                    )
                    return_floor_loss = return_floor_weight * jnp.mean(
                        floor_per_sample * loss_weights
                    )
                    critic_loss = critic_loss + return_floor_loss
                    unseen_q_mean = jnp.mean(unseen_q_per_sample)
                    chosen_q_mean = jnp.mean(
                        jnp.sum(
                            chosen_probabilities * self.support,
                            axis=-1,
                        )
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
                    return_floor_loss,
                    unseen_q_mean,
                    chosen_q_mean,
                    dense_return_q_loss,
                    dense_positive_fraction,
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
            (
                per_sample,
                entropy,
                projected_entropy,
                return_floor_loss,
                unseen_q_mean,
                chosen_q_mean,
                dense_return_q_loss,
                dense_positive_fraction,
            ) = aux
            priority = jnp.sqrt(per_sample + 1e-10)
            priority = priority / jnp.maximum(jnp.max(priority), 1e-10)
            metrics = {
                "critic_loss": critic_loss,
                "entropy": entropy,
                "target_entropy": projected_entropy,
                "loss_coeff": jnp.mean(loss_weights),
            }
            if q_reward_scale != 1.0:
                metrics["q_reward_scale"] = jnp.asarray(
                    q_reward_scale,
                    dtype=jnp.float32,
                )
                metrics["scaled_mc_return_mean"] = (
                    jnp.mean(mc_returns) * q_reward_scale
                )
            if dense_return_advantage_alpha > 0.0:
                metrics["dense_return_advantage_alpha"] = jnp.asarray(
                    dense_return_advantage_alpha,
                    dtype=jnp.float32,
                )
            if dense_return_advantage_clip_ratio is not None:
                metrics["dense_return_advantage_clip_ratio"] = jnp.asarray(
                    dense_return_advantage_clip_ratio,
                    dtype=jnp.float32,
                )
            if use_episodic_success_target:
                metrics["episodic_success_fraction"] = jnp.mean(
                    episodic_success_returns(mc_returns)
                )
            if use_return_floor:
                metrics["unseen_return_floor_loss"] = return_floor_loss
            if use_return_floor or use_dense_return_target:
                metrics["unseen_q_mean"] = unseen_q_mean
                metrics["chosen_q_mean"] = chosen_q_mean
                metrics["chosen_unseen_q_gap"] = (
                    chosen_q_mean - unseen_q_mean
                )
            if use_dense_return_target:
                metrics["dense_return_q_loss"] = dense_return_q_loss
            if dense_return_positive_only:
                metrics["dense_return_positive_fraction"] = (
                    dense_positive_fraction
                )
            if use_dense_expected_q_target:
                metrics["dense_return_expected_q_target"] = jnp.asarray(
                    1.0,
                    dtype=jnp.float32,
                )
            return (
                params,
                target_critic_params,
                opt_state,
                priority,
                metrics,
            )

        return update_fn

    # -- ``CQN.update`` + the completed-episode return element --

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
            if self._uses_mc_returns and "mc_return" not in batch:
                raise KeyError(
                    "episodic_success_q_target / dense_return_positive_only / "
                    "return_gated_margin gate on the completed-episode return; "
                    "the replay batch must provide an 'mc_return' element."
                )
            mc_returns = self._as_jax_array(
                batch.get("mc_return", np.zeros_like(batch["reward"])),
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
                mc_returns,
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
    "CQNASDenseReturn",
    "CQNASDenseReturnSpec",
    "advantage_learning_target_shift",
    "cqn_as_dense_return_spec_from_cfg",
    "dense_return_distributional_loss",
    "dense_return_expected_q_loss",
    "episodic_success_returns",
    "ordered_success_returns",
    "return_gated_margin_loss",
    "sequence_aligned_sparse_returns",
    "shift_categorical_distribution",
    "unseen_return_floor_loss",
]
