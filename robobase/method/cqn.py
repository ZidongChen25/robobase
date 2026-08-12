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
from robobase.method.rl_common import random_shift_rgb
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
    bc_lambda_schedule: str | None
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
        bc_lambda_schedule=(
            None
            if method.get("bc_lambda_schedule", None) is None
            else str(method.get("bc_lambda_schedule"))
        ),
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


def categorical_point_mass(
    values: jax.Array,
    support: jax.Array,
) -> jax.Array:
    """Project an arbitrary tensor of scalar values onto a C51 support."""

    values = jnp.asarray(values, dtype=jnp.float32)
    atoms = int(support.shape[0])
    v_min = support[0]
    v_max = support[-1]
    delta = (v_max - v_min) / float(atoms - 1)
    projected = jnp.clip((values - v_min) / delta, 0.0, float(atoms - 1))
    lower = jnp.floor(projected).astype(jnp.int32)
    upper = jnp.ceil(projected).astype(jnp.int32)
    lower_weight = jnp.where(
        lower == upper,
        1.0,
        upper.astype(jnp.float32) - projected,
    )
    upper_weight = jnp.where(
        lower == upper,
        0.0,
        projected - lower.astype(jnp.float32),
    )
    return (
        jax.nn.one_hot(lower, atoms, dtype=jnp.float32)
        * lower_weight[..., None]
        + jax.nn.one_hot(upper, atoms, dtype=jnp.float32)
        * upper_weight[..., None]
    )


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
    """Recover each future token's return for a sparse terminal reward."""

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
) -> tuple[jax.Array, jax.Array]:
    """Regress only replay-unseen bins to a valid minimum return.

    ``all_logits`` is ``[B, L, D, bins, atoms]`` and ``discrete_action`` is
    ``[B, L, D]``.  The replayed bin is explicitly masked out: it is trained
    only by the Bellman/return objective.  This is the squared conservative-Q
    regularizer used by Q-Transformer, adapted to CQN's C51 expected values;
    it is not an action likelihood, classification, or margin loss.
    """

    all_probabilities = jax.nn.softmax(all_logits, axis=-1)
    all_q = jnp.sum(all_probabilities * support, axis=-1)
    chosen_mask = jax.nn.one_hot(
        discrete_action,
        all_logits.shape[-2],
        dtype=all_q.dtype,
    )
    unseen_mask = 1.0 - chosen_mask
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
        # runner-up gap rather than "everything here is worthless" —
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
    bins. The gate reads reward-to-go only — never a demonstration
    flag — and applies identically to successful online transitions.
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
    """Lift positive discounted returns while preserving their ordering."""

    values = jnp.asarray(mc_returns, dtype=jnp.float32)
    positive_values = jnp.where(values > 0.0, values, 0.0)
    success = episodic_success_returns(values)
    mix = jnp.asarray(success_mix, dtype=jnp.float32)
    return (1.0 - mix) * positive_values + mix * success


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
        bc_lambda_schedule: str | None = None,
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
        self.bc_lambda_schedule = (
            None if bc_lambda_schedule is None else str(bc_lambda_schedule)
        )
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

        Plain CQN ignores replay actions and performs its usual greedy
        Double-CQN selection.  CQN-AS overrides this hook for replay-SARSA
        without duplicating the no-policy update path.
        """

        del replay_actions, replay_next_actions, demos
        return self._greedy_action_for_update(
            critic_params,
            features,
            action_key,
        )

    def _next_action_key(self):
        # Plain CQN has deterministic argmax selection. Keep its RNG stream
        # unchanged while exposing a hook for CQN-AS random tie breaking.
        return jax.random.PRNGKey(0)

    def _augment_update_obs_inputs(self, obs_inputs, next_obs_inputs, key):
        return obs_inputs, next_obs_inputs, key

    def _build_update_fn(self):
        optimizer = self.optimizer
        tau = self.critic_target_tau
        use_canonical_mc = bool(getattr(self, "_canonical_mc_anchor", False))
        use_mc_lower_bound = bool(
            getattr(self, "mc_lower_bound_target", False)
        )
        use_episodic_success_target = bool(
            getattr(self, "episodic_success_q_target", False)
        )
        ordered_success_return_mix = float(
            getattr(self, "ordered_success_return_mix", 0.0)
        )
        use_ordered_success_return = ordered_success_return_mix > 0.0
        sequence_aligned_mc_discount = getattr(
            self,
            "sequence_aligned_mc_discount",
            None,
        )
        use_sequence_aligned_mc = sequence_aligned_mc_discount is not None
        use_mc_returns = (
            use_canonical_mc
            or use_mc_lower_bound
            or use_episodic_success_target
        )
        return_floor_weight = float(
            getattr(self, "unseen_return_floor_weight", 0.0)
        )
        use_return_floor = return_floor_weight > 0.0
        use_dense_return_target = bool(
            getattr(self, "dense_return_q_target", False)
        )
        dense_return_positive_only = bool(
            getattr(self, "dense_return_positive_only", False)
        )
        use_dense_expected_q_target = bool(
            getattr(self, "dense_return_expected_q_loss", False)
        )
        dense_return_advantage_alpha = float(
            getattr(self, "dense_return_advantage_alpha", 0.0)
        )
        dense_return_advantage_clip_ratio = getattr(
            self,
            "dense_return_advantage_clip_ratio",
            None,
        )
        if dense_return_advantage_clip_ratio is not None:
            dense_return_advantage_clip_ratio = float(
                dense_return_advantage_clip_ratio
            )
        q_reward_scale = float(getattr(self, "q_reward_scale", 1.0))
        use_replay_td_target = (
            str(getattr(self, "td_target_action_source", "critic")).lower()
            == "replay_next"
        )
        use_replay_candidate_target = (
            str(getattr(self, "td_target_action_source", "critic")).lower()
            == "critic_replay_max"
        )
        dense_return_finest_neighbor_weight = float(
            getattr(self, "dense_return_finest_neighbor_weight", 0.0)
        )
        dense_return_label_smoothing = float(
            getattr(self, "dense_return_label_smoothing", 0.0)
        )
        dense_return_floor_satisfaction_margin = getattr(
            self,
            "dense_return_floor_satisfaction_margin",
            None,
        )
        dense_return_relative_floor_margin = getattr(
            self,
            "dense_return_relative_floor_margin",
            None,
        )
        if dense_return_relative_floor_margin is not None:
            dense_return_relative_floor_margin = float(
                dense_return_relative_floor_margin
            )
        return_gated_margin = getattr(self, "return_gated_margin", None)
        if return_gated_margin is not None:
            return_gated_margin = float(return_gated_margin)
        return_gated_margin_weight = float(
            getattr(self, "return_gated_margin_weight", 0.0)
        )
        if dense_return_floor_satisfaction_margin is not None:
            dense_return_floor_satisfaction_margin = float(
                dense_return_floor_satisfaction_margin
            )
        return_floor_value = float(
            getattr(self, "unseen_return_floor_value", 0.0)
        )
        return_floor_reduction = str(
            getattr(self, "unseen_return_floor_reduction", "mean")
        ).lower()
        return_floor_topk = int(
            getattr(self, "unseen_return_floor_topk", 1)
        )
        use_bc_schedule = (
            getattr(self, "bc_lambda_schedule", None) is not None
        )
        use_coarse_flow = bool(getattr(self, "coarse_flow", False))

        def array_all_finite(value):
            value = jnp.asarray(value)
            return jnp.all(jnp.isfinite(value))

        def array_max_abs_finite(value):
            value = jnp.asarray(value)
            finite_abs = jnp.where(jnp.isfinite(value), jnp.abs(value), 0.0)
            return jnp.max(finite_abs)

        def tree_all_finite(tree):
            return jnp.all(
                jnp.stack(
                    [array_all_finite(leaf) for leaf in jax.tree.leaves(tree)]
                )
            )

        def tree_max_abs_finite(tree):
            return jnp.max(
                jnp.stack(
                    [
                        array_max_abs_finite(leaf)
                        for leaf in jax.tree.leaves(tree)
                    ]
                )
            )

        use_token_split = bool(
            getattr(self, "token_split_horizon_targets", False)
        )
        token_split_boundary = getattr(self, "token_split_boundary", None)

        def update_impl(
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
            bc_weight,
            action_key,
            aux_next_obs_inputs=None,
            aux_next_actions=None,
            aux_rewards=None,
            aux_discounts=None,
            aux_bootstrap=None,
        ):
            obs_inputs, next_obs_inputs, action_key = (
                self._augment_update_obs_inputs(
                    obs_inputs,
                    next_obs_inputs,
                    action_key,
                )
            )
            if use_token_split and isinstance(aux_next_obs_inputs, dict):
                aux_next_obs_inputs = dict(aux_next_obs_inputs)
                if "rgb" in aux_next_obs_inputs:
                    aux_next_obs_inputs["rgb"] = random_shift_rgb(
                        aux_next_obs_inputs["rgb"],
                        jax.random.fold_in(action_key, 4243),
                    )
                if (
                    getattr(self, "low_dim_mask_prob", 0.0) > 0.0
                    and "low_dim" in aux_next_obs_inputs
                ):
                    aux_next_obs_inputs["low_dim"] = self._mask_low_dim(
                        aux_next_obs_inputs["low_dim"],
                        jax.random.fold_in(action_key, 4244),
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
                token_split_aux_fraction = jnp.asarray(0.0, dtype=jnp.float32)
                token_split_aux_reward_mean = jnp.asarray(
                    0.0, dtype=jnp.float32
                )
                if use_token_split:
                    aux_features = self._rl_features(
                        encoder_params,
                        aux_next_obs_inputs,
                        stop_gradient=True,
                    )
                    aux_next_action, _ = self._td_target_action_for_update(
                        current_params["critic"],
                        aux_features,
                        actions,
                        aux_next_actions,
                        demos,
                        jax.random.fold_in(action_key, 4245),
                    )
                    aux_target_logits, _ = self._critic_logits_per_level(
                        target_critic_params,
                        aux_features,
                        aux_next_action,
                    )
                    aux_target_probabilities = jax.nn.softmax(
                        aux_target_logits,
                        axis=-1,
                    )
                    aux_target_distribution = project_categorical(
                        aux_target_probabilities,
                        aux_rewards * q_reward_scale,
                        aux_discounts,
                        aux_bootstrap,
                        self.support,
                    )
                    if self.centralized_critic:
                        aux_target_distribution = jnp.broadcast_to(
                            aux_target_distribution.mean(
                                axis=-2, keepdims=True
                            ),
                            aux_target_distribution.shape,
                        )
                    # The D axis is laid out [token 0 dims..., token 1 dims,
                    # ...]; tokens whose 1-based index exceeds the boundary
                    # regress to the long-horizon (auxiliary_nstep) backup,
                    # the rest keep the exact legacy 1-step backup.
                    token_index = (
                        jnp.arange(target_distribution.shape[2])
                        // self.action_dim
                    )
                    aux_token_mask = (
                        token_index + 1
                    ) > int(token_split_boundary)
                    target_distribution = jnp.where(
                        aux_token_mask[None, None, :, None],
                        aux_target_distribution,
                        target_distribution,
                    )
                    token_split_aux_fraction = jnp.mean(
                        aux_token_mask.astype(jnp.float32)
                    )
                    token_split_aux_reward_mean = jnp.mean(aux_rewards)
                mc_lower_bound_fraction = jnp.asarray(
                    0.0,
                    dtype=jnp.float32,
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
                elif use_mc_lower_bound:
                    mc_target_returns = mc_returns * q_reward_scale
                    if use_ordered_success_return:
                        mc_target_returns = ordered_success_returns(
                            mc_returns,
                            ordered_success_return_mix,
                        )
                    bellman_q = jnp.sum(
                        target_distribution * self.support,
                        axis=-1,
                    )
                    if use_sequence_aligned_mc:
                        aligned_returns = sequence_aligned_sparse_returns(
                            mc_returns,
                            self.action_sequence,
                            self.action_dim,
                            sequence_aligned_mc_discount,
                        )
                        mc_distribution = categorical_point_mass(
                            aligned_returns[:, None, :],
                            self.support,
                        )
                        mc_distribution = jnp.broadcast_to(
                            mc_distribution,
                            target_distribution.shape,
                        )
                        use_mc_mask = (
                            aligned_returns[:, None, :] > bellman_q
                        )
                    else:
                        mc_distribution = project_categorical(
                            target_probabilities,
                            mc_target_returns,
                            jnp.zeros_like(discounts),
                            jnp.zeros_like(bootstrap),
                            self.support,
                        )
                        use_mc_mask = (
                            mc_target_returns[:, None, None] > bellman_q
                        )
                    target_distribution = jnp.where(
                        use_mc_mask[..., None],
                        mc_distribution,
                        target_distribution,
                    )
                    mc_lower_bound_fraction = jnp.mean(
                        use_mc_mask.astype(jnp.float32)
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

                bc_fosd_term = jnp.asarray(0.0, dtype=jnp.float32)
                bc_margin_term = jnp.asarray(0.0, dtype=jnp.float32)
                if self.bc_lambda > 0.0 or use_bc_schedule:
                    demo_count = jnp.maximum(jnp.sum(demos), 1.0)
                    # CQN-AS historically uses both FOSD and an expected-Q
                    # margin.  Keep that behavior by default, while allowing
                    # a margin-only baseline for objective-matched flow
                    # experiments.
                    if getattr(self, "demo_fosd", True):
                        chosen_cdf = jnp.cumsum(chosen_probabilities, axis=-1)
                        all_cdf = jnp.cumsum(all_probabilities, axis=-1)
                        fosd = jnp.maximum(
                            chosen_cdf[..., None, :] - all_cdf,
                            0.0,
                        ).sum(axis=-1).mean(axis=(1, 2, 3))
                        bc_fosd_term = bc_weight * (
                            jnp.sum(fosd * demos) / demo_count
                        )
                        critic_loss = critic_loss + bc_fosd_term
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
                        bc_margin_term = bc_weight * (
                            jnp.sum(margin * demos) / demo_count
                        )
                        critic_loss = critic_loss + bc_margin_term

                # BC-anchor diagnostics (cqn-flow.md sec 64). The margin hinge
                # implements the constraint Q(a_demo) >= max sibling + m, so
                # what carries meaning across tasks is how often that
                # constraint binds and whether it holds behaviorally -- not
                # lambda's numeric value, which only sets a force whose
                # counterpart (the TD force) is scaled by reward density,
                # horizon and Q range.
                diag_norm = jnp.maximum(jnp.sum(demos), 1.0)
                diag_all_q = jnp.sum(all_probabilities * self.support, axis=-1)
                diag_chosen_q = jnp.sum(
                    chosen_probabilities * self.support, axis=-1
                )
                diag_gap = diag_chosen_q - jnp.max(diag_all_q, axis=-1)
                diag_sibling = (
                    jnp.abs(diag_chosen_q[..., None] - diag_all_q) > 1e-9
                )
                diag_violating = (
                    (self.bc_margin - (diag_chosen_q[..., None] - diag_all_q))
                    > 0.0
                ) & diag_sibling
                diag_binding = jnp.sum(
                    diag_violating.astype(jnp.float32), axis=(1, 2, 3)
                ) / jnp.maximum(
                    jnp.sum(diag_sibling.astype(jnp.float32), axis=(1, 2, 3)),
                    1.0,
                )
                bc_diagnostics = {
                    "bc_weight": jnp.asarray(bc_weight, dtype=jnp.float32),
                    "bc_agreement": jnp.sum(
                        (diag_gap >= -1e-6)
                        .astype(jnp.float32)
                        .mean(axis=(1, 2))
                        * demos
                    )
                    / diag_norm,
                    "bc_binding_rate": jnp.sum(diag_binding * demos)
                    / diag_norm,
                    "bc_margin_gap": jnp.sum(
                        diag_gap.mean(axis=(1, 2)) * demos
                    )
                    / diag_norm,
                    "bc_sibling_q_span": jnp.sum(
                        (
                            jnp.max(diag_all_q, axis=-1)
                            - jnp.min(diag_all_q, axis=-1)
                        ).mean(axis=(1, 2))
                        * demos
                    )
                    / diag_norm,
                    "bc_online_agreement": jnp.sum(
                        (diag_gap >= -1e-6)
                        .astype(jnp.float32)
                        .mean(axis=(1, 2))
                        * (1.0 - demos)
                    )
                    / jnp.maximum(jnp.sum(1.0 - demos), 1.0),
                }
                mc_zero = jnp.asarray(0.0, dtype=jnp.float32)
                mc_return_loss = mc_zero
                mc_return_mae = mc_zero
                if use_canonical_mc:
                    # Stage-147: completed-episode discounted return projected
                    # onto the fixed C51 support supervises the executed
                    # action's own Q head.  Return regression, not imitation.
                    mc_target = project_categorical(
                        jax.nn.softmax(chosen_logits, axis=-1),
                        mc_returns,
                        jnp.zeros_like(discounts),
                        jnp.zeros_like(bootstrap),
                        self.support,
                    )
                    mc_target = jax.lax.stop_gradient(mc_target)
                    mc_per_sample = -jnp.sum(
                        mc_target * chosen_log_probabilities,
                        axis=-1,
                    ).mean(axis=(1, 2))
                    mc_return_loss = float(
                        self.mc_return_weight
                    ) * jnp.mean(mc_per_sample * loss_weights)
                    critic_loss = critic_loss + mc_return_loss
                    chosen_expected_q = jnp.sum(
                        jax.nn.softmax(chosen_logits, axis=-1)
                        * self.support,
                        axis=-1,
                    )
                    mc_return_mae = jnp.mean(
                        jnp.abs(
                            chosen_expected_q
                            - mc_returns[:, None, None]
                        )
                    )
                coarse_flow_loss = jnp.asarray(0.0, dtype=jnp.float32)
                if use_coarse_flow:
                    # Stage-152 coarse-flow: bin-conditioned CFM on the
                    # within-cell residual of the recorded action.  Features
                    # are stop-gradient and the flow head has its own
                    # parameters, so the critic's gradients are exactly the
                    # legacy ones.
                    flow_features = jax.lax.stop_gradient(features)
                    cfm_key = jax.random.fold_in(action_key, 152)
                    noise_key, time_key = jax.random.split(cfm_key)
                    if getattr(self, "coarse_flow_pure", False):
                        # Stage-155 no-selection control: full-range
                        # coordinates, no conditioning.
                        bin_context = None
                        cell_low = jnp.broadcast_to(
                            self.action_low, actions.shape
                        )
                        cell_width = jnp.broadcast_to(
                            self.action_high - self.action_low,
                            actions.shape,
                        )
                    else:
                        cell_indices = encode_action(
                            actions,
                            self.action_low,
                            self.action_high,
                            self.levels,
                            self.bins,
                        )
                        bin_context, cell_low, cell_width = (
                            self._coarse_flow_cell(cell_indices)
                        )
                    u1 = jnp.clip(
                        2.0 * (actions - cell_low) / cell_width - 1.0,
                        -1.0,
                        1.0,
                    )
                    x0 = jax.random.normal(
                        noise_key, u1.shape, dtype=jnp.float32
                    )
                    t = jax.random.uniform(
                        time_key, (u1.shape[0],), dtype=jnp.float32
                    )
                    x_t = (1.0 - t[:, None]) * x0 + t[:, None] * u1
                    predicted_velocity = self.flow_policy_model.apply(
                        current_params["flow_policy"],
                        flow_features,
                        x_t,
                        t,
                        bin_context=bin_context,
                    )
                    flow_per_sample = jnp.square(
                        predicted_velocity - (u1 - x0)
                    ).mean(axis=-1)
                    flow_weights = demos
                    if self.coarse_flow_selfdistill_weight is not None:
                        # mc_returns are zeros unless the canonical MC anchor
                        # supplies completed-episode returns, in which case
                        # high-return online chunks join the flow's training
                        # set with a reduced weight.
                        qualified = (
                            (
                                mc_returns
                                >= self.coarse_flow_selfdistill_threshold
                            )
                            & (demos < 0.5)
                        ).astype(jnp.float32)
                        flow_weights = (
                            demos
                            + self.coarse_flow_selfdistill_weight * qualified
                        )
                    coarse_flow_loss = self.flow_policy_lambda * (
                        jnp.sum(flow_per_sample * flow_weights)
                        / jnp.maximum(jnp.sum(flow_weights), 1.0)
                    )
                    critic_loss = critic_loss + coarse_flow_loss
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
                nan_diag = {
                    "features_all_finite": array_all_finite(features),
                    "next_features_all_finite": array_all_finite(next_features),
                    "target_logits_all_finite": array_all_finite(target_logits),
                    "target_probabilities_all_finite": array_all_finite(
                        target_probabilities
                    ),
                    "target_distribution_all_finite": array_all_finite(
                        target_distribution
                    ),
                    "chosen_logits_all_finite": array_all_finite(chosen_logits),
                    "all_logits_all_finite": array_all_finite(all_logits),
                    "chosen_log_probabilities_all_finite": array_all_finite(
                        chosen_log_probabilities
                    ),
                    "canonical_per_sample_all_finite": array_all_finite(
                        canonical_per_sample
                    ),
                    "bc_fosd_term_all_finite": array_all_finite(bc_fosd_term),
                    "bc_margin_term_all_finite": array_all_finite(bc_margin_term),
                    "loss_all_finite": array_all_finite(critic_loss),
                    "features_max_abs_finite": array_max_abs_finite(features),
                    "next_features_max_abs_finite": array_max_abs_finite(
                        next_features
                    ),
                    "target_logits_max_abs_finite": array_max_abs_finite(
                        target_logits
                    ),
                    "chosen_logits_max_abs_finite": array_max_abs_finite(
                        chosen_logits
                    ),
                }
                return critic_loss, (
                    per_sample,
                    entropy,
                    target_entropy,
                    return_floor_loss,
                    unseen_q_mean,
                    chosen_q_mean,
                    mc_lower_bound_fraction,
                    mc_return_loss,
                    mc_return_mae,
                    coarse_flow_loss,
                    dense_return_q_loss,
                    dense_positive_fraction,
                    token_split_aux_fraction,
                    token_split_aux_reward_mean,
                    target_action_info,
                    bc_diagnostics,
                    nan_diag,
                )

            pre_params = params
            pre_target_critic_params = target_critic_params
            pre_opt_state = opt_state
            (critic_loss, aux), grads = jax.value_and_grad(
                loss_fn,
                has_aux=True,
            )(params)
            updates, candidate_opt_state = optimizer.update(
                grads, opt_state, params
            )
            candidate_params = self.optax.apply_updates(params, updates)
            candidate_target_critic_params = jax.tree.map(
                lambda target, online: (1.0 - tau) * target + tau * online,
                target_critic_params,
                candidate_params["critic"],
            )
            (
                per_sample,
                entropy,
                projected_entropy,
                return_floor_loss,
                unseen_q_mean,
                chosen_q_mean,
                mc_lower_bound_fraction,
                mc_return_loss,
                mc_return_mae,
                coarse_flow_loss,
                dense_return_q_loss,
                dense_positive_fraction,
                token_split_aux_fraction,
                token_split_aux_reward_mean,
                target_action_info,
                bc_diagnostics,
                nan_diag,
            ) = aux
            update_diag = {
                "pre_params_all_finite": tree_all_finite(pre_params),
                "pre_target_all_finite": tree_all_finite(
                    pre_target_critic_params
                ),
                "pre_opt_state_all_finite": tree_all_finite(pre_opt_state),
                "grads_all_finite": tree_all_finite(grads),
                "updates_all_finite": tree_all_finite(updates),
                "candidate_opt_state_all_finite": tree_all_finite(
                    candidate_opt_state
                ),
                "candidate_params_all_finite": tree_all_finite(candidate_params),
                "candidate_target_all_finite": tree_all_finite(
                    candidate_target_critic_params
                ),
                "grads_max_abs_finite": tree_max_abs_finite(grads),
                "updates_max_abs_finite": tree_max_abs_finite(updates),
            }
            finite_flags = [
                value
                for key, value in {**nan_diag, **update_diag}.items()
                if key.endswith("_all_finite")
            ]
            update_all_finite = jnp.all(jnp.stack(finite_flags))
            # Preserve the last known-good state on the first bad update. For
            # a finite update, selecting the candidate is bit-identical to the
            # pre-instrumentation return path.
            params = jax.tree.map(
                lambda old, new: jnp.where(update_all_finite, new, old),
                pre_params,
                candidate_params,
            )
            target_critic_params = jax.tree.map(
                lambda old, new: jnp.where(update_all_finite, new, old),
                pre_target_critic_params,
                candidate_target_critic_params,
            )
            opt_state = jax.tree.map(
                lambda old, new: jnp.where(update_all_finite, new, old),
                pre_opt_state,
                candidate_opt_state,
            )
            priority = jnp.sqrt(per_sample + 1e-10)
            priority = priority / jnp.maximum(jnp.max(priority), 1e-10)
            priority = jnp.where(update_all_finite, priority, 0.0)
            metrics = {
                "critic_loss": critic_loss,
                "entropy": entropy,
                "target_entropy": projected_entropy,
                "loss_coeff": jnp.mean(loss_weights),
                "nan_diag/update_committed": update_all_finite.astype(jnp.float32),
                **{f"nan_diag/{key}": value for key, value in nan_diag.items()},
                **{f"nan_diag/{key}": value for key, value in update_diag.items()},
                **bc_diagnostics,
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
                metrics["behavior_candidate_score"] = jnp.mean(
                    behavior_score
                )
                metrics["greedy_candidate_score"] = jnp.mean(greedy_score)
                metrics["behavior_minus_greedy_q"] = jnp.mean(
                    behavior_score - greedy_score
                )
                if "demo_behavior_forced" in target_action_info:
                    metrics["demo_behavior_force_fraction"] = jnp.mean(
                        target_action_info[
                            "demo_behavior_forced"
                        ].astype(jnp.float32)
                    )
                    metrics["demo_behavior_force_probability"] = jnp.asarray(
                        self.demo_behavior_force_probability,
                        dtype=jnp.float32,
                    )
            if use_canonical_mc:
                metrics["mc_return_loss"] = mc_return_loss
                metrics["mc_return_mae"] = mc_return_mae
            if use_mc_lower_bound:
                metrics["mc_lower_bound_fraction"] = (
                    mc_lower_bound_fraction
                )
                metrics["mc_return_mean"] = jnp.mean(mc_returns)
            if use_token_split:
                metrics["token_split_aux_fraction"] = token_split_aux_fraction
                metrics["token_split_aux_reward_mean"] = (
                    token_split_aux_reward_mean
                )
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
            if use_ordered_success_return:
                metrics["ordered_success_return_mean"] = jnp.mean(
                    ordered_success_returns(
                        mc_returns,
                        ordered_success_return_mix,
                    )
                )
            if use_sequence_aligned_mc:
                metrics["sequence_aligned_mc_return_mean"] = jnp.mean(
                    sequence_aligned_sparse_returns(
                        mc_returns,
                        self.action_sequence,
                        self.action_dim,
                        sequence_aligned_mc_discount,
                    )
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
            if use_coarse_flow:
                metrics["coarse_flow_loss"] = coarse_flow_loss
            return (
                params,
                target_critic_params,
                opt_state,
                priority,
                metrics,
            )

        base_bc_weight = float(self.bc_lambda)

        def split_aux_tail(args):
            # token-split forwards five auxiliary-horizon tensors after
            # action_key; every other configuration passes none.
            if not use_token_split:
                return args, ()
            return args[:-5], args[-5:]

        def call_impl(core_args, mc_returns, bc_weight, action_key, aux_tail):
            return update_impl(
                *core_args, mc_returns, bc_weight, action_key, *aux_tail
            )

        if use_mc_returns and use_bc_schedule:

            def update_fn(*args):
                args, aux_tail = split_aux_tail(args)
                (*core, mc_returns, bc_weight, action_key) = args
                return call_impl(
                    core, mc_returns, bc_weight, action_key, aux_tail
                )

        elif use_mc_returns:

            def update_fn(*args):
                args, aux_tail = split_aux_tail(args)
                (*core, mc_returns, action_key) = args
                return call_impl(
                    core, mc_returns, base_bc_weight, action_key, aux_tail
                )

        elif use_bc_schedule:

            def update_fn(*args):
                args, aux_tail = split_aux_tail(args)
                (*core, bc_weight, action_key) = args
                rewards = core[7]
                return call_impl(
                    core,
                    jnp.zeros_like(rewards),
                    bc_weight,
                    action_key,
                    aux_tail,
                )

        else:

            def update_fn(*args):
                args, aux_tail = split_aux_tail(args)
                (*core, action_key) = args
                rewards = core[7]
                return call_impl(
                    core,
                    jnp.zeros_like(rewards),
                    base_bc_weight,
                    action_key,
                    aux_tail,
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
                str(
                    getattr(self, "td_target_action_source", "critic")
                ).lower()
                == "critic_replay_max"
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
            canonical_mc_args = ()
            if getattr(self, "_uses_canonical_mc_returns", False):
                canonical_mc_args = (
                    self._as_jax_array(
                        batch.get(
                            "mc_return",
                            np.zeros_like(batch["reward"]),
                        ),
                        self.jnp.float32,
                    ).reshape(-1),
                )
            if getattr(self, "bc_lambda_schedule", None) is not None:
                canonical_mc_args = canonical_mc_args + (
                    float(utils.schedule(self.bc_lambda_schedule, step)),
                )
            auxiliary_args = ()
            needs_aux_horizon = (
                float(getattr(self, "auxiliary_td_loss_weight", 0.0)) > 0.0
                or bool(getattr(self, "token_split_horizon_targets", False))
            )
            if needs_aux_horizon:
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
                *canonical_mc_args,
                self._next_action_key(),
                *auxiliary_args,
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
    "advantage_learning_target_shift",
    "categorical_point_mass",
    "dense_return_distributional_loss",
    "dense_return_expected_q_loss",
    "episodic_success_returns",
    "ordered_success_returns",
    "cqn_spec_from_cfg",
    "decode_action",
    "encode_action",
    "project_categorical",
    "sequence_aligned_sparse_returns",
    "shift_categorical_distribution",
    "zoom_in",
]
