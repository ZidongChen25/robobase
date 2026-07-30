"""CQN-AS critics whose value prediction is learned with flow matching.

The method keeps CQN-AS action discretisation and rollout behaviour, but
replaces the direct C51 head with a conditional value-flow field.  The field
can transport a 51-dimensional categorical-logit state (``categorical``), a
one-dimensional expected-Q state (``scalar``), or one-dimensional stochastic
return samples (``return_sample``).  At each coarse-to-fine level all action
bins and all source samples are evaluated in one batched model call; only the
levels and Euler steps remain sequential.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable, Literal, NamedTuple, Optional

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
from robobase.method.flow_sources import linear_flow_training_pair
from robobase.method.rl_common import JaxRLMethodBase, RLModelSpec, activation
from robobase.models.backbones.common import SinusoidalPosEmb


ValueMode = Literal["categorical", "scalar", "return_sample"]
FlowSourceType = Literal["gaussian", "uniform"]
ReturnSampleAggregation = Literal["mean", "entropic", "truncated_mean"]
ScalarValueEmbedding = Literal["raw", "hl_gauss"]
TimeEmbeddingType = Literal["sinusoidal", "fourier", "raw"]
CriticArchitecture = Literal["flow_q", "flow_v_direct_a"]


def aggregate_return_samples(
    samples: jax.Array,
    *,
    aggregation: ReturnSampleAggregation,
    temperature: float,
    truncate_top: int = 0,
    sample_axis: int = 1,
) -> jax.Array:
    """Aggregate return samples into an action score.

    ``entropic`` implements EVOR's regularized value

        eta * log(mean(exp(return / eta))).

    The log-mean-exp form is evaluated stably and stays exactly equal to the
    only sample when ``R=1``. ``truncated_mean`` implements FlowCritic's
    pessimistic readout by sorting return samples and discarding the largest
    ``truncate_top`` values before averaging.
    """

    values = jnp.asarray(samples)
    aggregation = str(aggregation).lower()
    if aggregation == "mean":
        return values.mean(axis=sample_axis)
    if aggregation == "truncated_mean":
        truncate_top = int(truncate_top)
        if truncate_top < 0:
            raise ValueError("truncate_top must be non-negative.")
        sample_count = values.shape[sample_axis]
        if sample_count < 1:
            raise ValueError("return sample axis must be non-empty.")
        if truncate_top >= sample_count:
            raise ValueError(
                "truncate_top must be smaller than the return sample count."
            )
        sorted_values = jnp.sort(values, axis=sample_axis)
        keep = jnp.arange(sample_count - truncate_top)
        return jnp.take(sorted_values, keep, axis=sample_axis).mean(
            axis=sample_axis
        )
    if aggregation != "entropic":
        raise ValueError(
            "aggregation must be 'mean', 'entropic', or 'truncated_mean'."
        )
    if temperature <= 0.0:
        raise ValueError("entropic return temperature must be positive.")
    sample_count = values.shape[sample_axis]
    if sample_count < 1:
        raise ValueError("return sample axis must be non-empty.")
    eta = jnp.asarray(temperature, dtype=values.dtype)
    return eta * (
        jax.scipy.special.logsumexp(values / eta, axis=sample_axis)
        - jnp.log(jnp.asarray(sample_count, dtype=values.dtype))
    )


class PCBFTrainingPair(NamedTuple):
    """Successor/current paths and the detached PCBF velocity target."""

    successor_sample: jax.Array
    current_sample: jax.Array
    target_velocity: jax.Array


class EVORTDTrainingPair(NamedTuple):
    """Current interpolant and detached EVOR velocity-space TD target."""

    current_sample: jax.Array
    target_velocity: jax.Array


def evor_velocity_td_pair(
    source: jax.Array,
    current_endpoint: jax.Array,
    reward: jax.Array,
    effective_discount: jax.Array,
    next_velocity: jax.Array,
    tau: jax.Array,
) -> EVORTDTrainingPair:
    """Construct EVOR Equation 35/36 in repository reverse-time coordinates.

    EVOR uses forward time ``t`` with source at zero.  This repository uses
    ``tau = 1 - t`` with source at one, so the same linear interpolant is
    produced by :func:`linear_flow_training_pair`.  The velocity orientation
    remains endpoint-minus-source under the repository's positive integration
    convention.
    """

    source = jnp.asarray(source)
    current_endpoint = jnp.asarray(current_endpoint, dtype=source.dtype)
    next_velocity = jnp.asarray(next_velocity, dtype=source.dtype)
    if source.shape != current_endpoint.shape:
        raise ValueError("source and current_endpoint shapes must match.")
    if source.shape != next_velocity.shape:
        raise ValueError("source and next_velocity shapes must match.")
    if source.ndim < 2:
        raise ValueError("EVOR tensors require a leading batch axis.")
    reward = jnp.asarray(reward, dtype=source.dtype)
    effective_discount = jnp.asarray(
        effective_discount, dtype=source.dtype
    )
    if reward.shape != (source.shape[0],):
        raise ValueError("reward must have shape [batch].")
    if effective_discount.shape != reward.shape:
        raise ValueError("effective_discount must match reward shape.")
    batch_broadcast = (source.shape[0], *((1,) * (source.ndim - 1)))
    reward = reward.reshape(batch_broadcast)
    effective_discount = effective_discount.reshape(batch_broadcast)
    current_sample = linear_flow_training_pair(
        source,
        current_endpoint,
        tau,
    ).sample
    target_velocity = reward + effective_discount * next_velocity
    return EVORTDTrainingPair(
        current_sample=current_sample,
        target_velocity=jax.lax.stop_gradient(target_velocity),
    )


class FlowIQNTrainingPair(NamedTuple):
    """Rank-coupled scalar source, target, and explicit source quantile."""

    source: jax.Array
    target: jax.Array
    source_quantile: jax.Array


def quantile_couple_return_samples(
    source: jax.Array,
    target: jax.Array,
    *,
    source_min: float,
    source_max: float,
    sample_axis: int = 1,
) -> FlowIQNTrainingPair:
    """Apply FlowIQN's per-condition one-dimensional monotone coupling.

    The source and Bellman samples must already have matching axes. Sorting is
    performed only along the return-sample axis, so states, C2F levels,
    sequence positions, action dimensions, and queried bins never mix.
    """

    if source_max <= source_min:
        raise ValueError("source_max must be greater than source_min.")
    source = jnp.asarray(source)
    target = jnp.asarray(target, dtype=source.dtype)
    if source.shape != target.shape:
        raise ValueError("FlowIQN source and target shapes must match.")
    if source.shape[-1] != 1:
        raise ValueError("FlowIQN is defined for scalar return samples.")
    if not -source.ndim <= sample_axis < source.ndim:
        raise ValueError("sample_axis is outside the source rank.")
    sorted_source = jnp.sort(source, axis=sample_axis)
    sorted_target = jnp.sort(target, axis=sample_axis)
    source_quantile = (sorted_source - float(source_min)) / float(
        source_max - source_min
    )
    source_quantile = jnp.clip(source_quantile, 0.0, 1.0)
    return FlowIQNTrainingPair(
        source=sorted_source,
        target=sorted_target,
        source_quantile=source_quantile,
    )


def quantile_huber_endpoint_loss(
    predicted_quantiles: jax.Array,
    target_particles: jax.Array,
    quantile_levels: jax.Array,
    *,
    kappa: float = 1.0,
) -> jax.Array:
    """Return the all-pairs quantile-Huber loss for each batch element.

    ``predicted_quantiles`` and ``quantile_levels`` use
    ``[batch, online_samples, ...]`` while ``target_particles`` uses
    ``[batch, target_samples, ...]``.  Every predicted quantile is compared
    with every Bellman target particle, as in IQN/DBC, rather than being
    assigned a single empirical order statistic.
    """

    predicted_quantiles = jnp.asarray(predicted_quantiles)
    target_particles = jnp.asarray(
        target_particles, dtype=predicted_quantiles.dtype
    )
    quantile_levels = jnp.asarray(
        quantile_levels, dtype=predicted_quantiles.dtype
    )
    if predicted_quantiles.ndim < 2:
        raise ValueError("predicted_quantiles must include batch and sample axes.")
    if predicted_quantiles.shape != quantile_levels.shape:
        raise ValueError(
            "quantile_levels must have the same shape as predicted_quantiles."
        )
    if (
        target_particles.ndim != predicted_quantiles.ndim
        or target_particles.shape[0] != predicted_quantiles.shape[0]
        or target_particles.shape[2:] != predicted_quantiles.shape[2:]
    ):
        raise ValueError(
            "target_particles must match the prediction batch and event axes."
        )
    if kappa <= 0.0:
        raise ValueError("kappa must be positive.")

    errors = (
        target_particles[:, None, ...]
        - predicted_quantiles[:, :, None, ...]
    )
    absolute_errors = jnp.abs(errors)
    huber = jnp.where(
        absolute_errors <= float(kappa),
        0.5 * jnp.square(errors),
        float(kappa) * (absolute_errors - 0.5 * float(kappa)),
    )
    quantiles = quantile_levels[:, :, None, ...]
    asymmetric_weight = jnp.abs(
        quantiles - (errors < 0.0).astype(predicted_quantiles.dtype)
    )
    loss = asymmetric_weight * huber / float(kappa)
    return loss.mean(axis=tuple(range(1, loss.ndim)))


def path_coupled_bellman_flow_pair(
    source: jax.Array,
    next_endpoint: jax.Array,
    reward: jax.Array,
    effective_discount: jax.Array,
    bootstrap: jax.Array,
    forward_time: jax.Array,
    next_velocity: jax.Array,
    *,
    control_lambda: float,
) -> PCBFTrainingPair:
    """Construct the source-consistent path and control-variate PCBF target.

    This helper uses the paper's forward time convention: ``t=0`` is the
    source and ``t=1`` is the return endpoint.  Callers are responsible for
    expanding scalar batch fields so they broadcast over flow sample, action,
    bin, and value axes.  Successor-network quantities are explicitly detached
    here, which keeps the EMA field a target rather than a second trainable
    branch of the loss.
    """

    if control_lambda < 0.0:
        raise ValueError("PCBF control_lambda must be non-negative.")
    source = jnp.asarray(source)
    next_endpoint = jax.lax.stop_gradient(jnp.asarray(next_endpoint))
    next_velocity = jax.lax.stop_gradient(jnp.asarray(next_velocity))
    reward = jnp.asarray(reward, dtype=source.dtype)
    effective_discount = jnp.asarray(effective_discount, dtype=source.dtype)
    bootstrap = jnp.asarray(bootstrap, dtype=source.dtype)
    forward_time = jnp.asarray(forward_time, dtype=source.dtype)

    successor_sample = (
        (1.0 - forward_time) * source + forward_time * next_endpoint
    )
    current_sample = (
        forward_time * reward
        + effective_discount * successor_sample
        + (1.0 - forward_time) * (1.0 - effective_discount) * source
    )
    sample_target = reward + effective_discount * next_endpoint - source
    correction = next_velocity - (next_endpoint - source)
    target_velocity = sample_target + (
        float(control_lambda) * bootstrap * correction
    )
    return PCBFTrainingPair(
        successor_sample=jax.lax.stop_gradient(successor_sample),
        current_sample=jax.lax.stop_gradient(current_sample),
        target_velocity=jax.lax.stop_gradient(target_velocity),
    )


def centered_log_probabilities(
    probabilities: jax.Array,
    *,
    epsilon: float = 1e-8,
) -> jax.Array:
    """Map a PMF to the zero-mean representative of its logit equivalence class."""

    probabilities = jnp.asarray(probabilities)
    probabilities = jnp.maximum(probabilities, epsilon)
    probabilities = probabilities / probabilities.sum(axis=-1, keepdims=True)
    logits = jnp.log(probabilities)
    return logits - logits.mean(axis=-1, keepdims=True)


def flow_logits_to_probabilities(logits: jax.Array) -> jax.Array:
    """Interpret a categorical value-flow endpoint as an atom PMF."""

    return jax.nn.softmax(logits, axis=-1)


def expected_q(probabilities: jax.Array, support: jax.Array) -> jax.Array:
    """Return the categorical expectation while preserving leading axes."""

    return jnp.sum(probabilities * jnp.asarray(support), axis=-1)


class SupportedLCBGateResult(NamedTuple):
    """Per-bin support/pessimism diagnostics for a behavior-safe override."""

    indices: jax.Array
    bc_indices: jax.Array
    override_mask: jax.Array
    lcb_delta: jax.Array
    support_mask: jax.Array


class SupportedLCBPlanResult(NamedTuple):
    """One behavior-safe plan intervention and its selection diagnostics."""

    action: jax.Array
    candidate_indices: jax.Array
    bc_indices: jax.Array
    eligible_override_mask: jax.Array
    applied_override: jax.Array
    selected_dimension: jax.Array
    selected_lcb_delta: jax.Array


def supported_lcb_action_indices(
    policy_logits: jax.Array,
    advantage_ensemble: jax.Array,
    *,
    lcb_scale: float,
    min_lcb_margin: float,
    max_bc_logprob_drop: float,
) -> SupportedLCBGateResult:
    """Select bins only when support and independent-critic LCB both pass.

    ``advantage_ensemble`` has one extra leading ensemble axis relative to
    ``policy_logits``.  Every member is converted to a delta against the
    behavior-policy argmax bin.  A candidate may replace that bin only if its
    behavior log-probability drop is within ``max_bc_logprob_drop`` and

        mean(delta) - lcb_scale * std(delta) > min_lcb_margin.

    Otherwise the returned index is exactly the BC argmax.  This hard fallback
    is deliberately different from a global ``A + beta * log pi`` blend.
    """

    policy_logits = jnp.asarray(policy_logits)
    advantage_ensemble = jnp.asarray(advantage_ensemble)
    if policy_logits.ndim < 1:
        raise ValueError("policy_logits must have a bin axis.")
    if advantage_ensemble.ndim != policy_logits.ndim + 1:
        raise ValueError(
            "advantage_ensemble must add exactly one leading ensemble axis."
        )
    if advantage_ensemble.shape[1:] != policy_logits.shape:
        raise ValueError(
            "advantage_ensemble trailing shape must match policy_logits."
        )
    if advantage_ensemble.shape[0] < 2:
        raise ValueError("LCB gating requires at least two independent critics.")
    if lcb_scale < 0.0:
        raise ValueError("lcb_scale must be non-negative.")
    if min_lcb_margin < 0.0:
        raise ValueError("min_lcb_margin must be non-negative.")
    if max_bc_logprob_drop < 0.0:
        raise ValueError("max_bc_logprob_drop must be non-negative.")

    bc_indices = jnp.argmax(policy_logits, axis=-1)
    bc_index = bc_indices[None, ..., None]
    bc_advantage = jnp.take_along_axis(
        advantage_ensemble,
        bc_index,
        axis=-1,
    )
    advantage_delta = advantage_ensemble - bc_advantage
    lcb_delta = advantage_delta.mean(axis=0) - float(lcb_scale) * (
        advantage_delta.std(axis=0)
    )

    log_probabilities = jax.nn.log_softmax(policy_logits, axis=-1)
    bc_log_probability = jnp.take_along_axis(
        log_probabilities,
        bc_indices[..., None],
        axis=-1,
    )
    log_probability_drop = bc_log_probability - log_probabilities
    support_mask = log_probability_drop <= (
        float(max_bc_logprob_drop) + 1e-7
    )
    supported_lcb = jnp.where(support_mask, lcb_delta, -jnp.inf)
    candidate_indices = jnp.argmax(supported_lcb, axis=-1)
    candidate_lcb = jnp.take_along_axis(
        supported_lcb,
        candidate_indices[..., None],
        axis=-1,
    )[..., 0]
    override_mask = jnp.logical_and(
        candidate_indices != bc_indices,
        candidate_lcb > float(min_lcb_margin),
    )
    indices = jnp.where(override_mask, candidate_indices, bc_indices)
    return SupportedLCBGateResult(
        indices=indices,
        bc_indices=bc_indices,
        override_mask=override_mask,
        lcb_delta=lcb_delta,
        support_mask=support_mask,
    )


def sibling_bin_candidate_plans(
    baseline_plan: jax.Array,
    action_low: jax.Array,
    action_high: jax.Array,
    *,
    bins: int,
    force_level: int,
    intervention_horizon: int,
) -> tuple[jax.Array, jax.Array]:
    """Vectorize the branch-oracle sibling intervention over all dimensions.

    Returns candidate plans with shape ``[B, D, bins, K, D]`` and the applied
    delta with shape ``[B, D, bins]``.  The construction intentionally matches
    ``finetune_cqn_branch_oracle._sibling_candidate_plans``: one level's fixed
    prefix is retained and the selected child's center delta is repeated over
    the first ``intervention_horizon`` plan tokens.
    """

    baseline_plan = jnp.asarray(baseline_plan, dtype=jnp.float32)
    action_low = jnp.asarray(action_low, dtype=jnp.float32)
    action_high = jnp.asarray(action_high, dtype=jnp.float32)
    if baseline_plan.ndim != 3:
        raise ValueError("baseline_plan must have shape [B, K, D].")
    if action_low.shape != (baseline_plan.shape[-1],):
        raise ValueError("action_low must have one bound per action dimension.")
    if action_high.shape != action_low.shape:
        raise ValueError("action_high shape must match action_low.")
    if bins < 2:
        raise ValueError("bins must be at least two.")
    if force_level < 0:
        raise ValueError("force_level must be non-negative.")
    if not 1 <= intervention_horizon <= baseline_plan.shape[1]:
        raise ValueError(
            "intervention_horizon must be in [1, action_sequence]."
        )

    batch_size, action_sequence, action_dim = baseline_plan.shape
    prefix_count = int(bins) ** int(force_level)
    prefix_width = (action_high - action_low) / float(prefix_count)
    baseline_action = baseline_plan[:, 0, :]
    prefix_index = jnp.clip(
        jnp.floor(
            (baseline_action - action_low[None]) / prefix_width[None]
        ),
        0,
        prefix_count - 1,
    )
    prefix_low = action_low[None] + prefix_index * prefix_width[None]
    child_width = prefix_width / float(bins)
    centers = prefix_low[..., None] + (
        jnp.arange(bins, dtype=jnp.float32)[None, None, :] + 0.5
    ) * child_width[None, :, None]
    centers = jnp.clip(
        centers,
        action_low[None, :, None],
        action_high[None, :, None],
    )
    deltas = centers - baseline_action[..., None]

    sequence_mask = (
        jnp.arange(action_sequence) < int(intervention_horizon)
    ).astype(jnp.float32)
    dimension_mask = jnp.eye(action_dim, dtype=jnp.float32)
    perturbation = (
        deltas[..., None, None]
        * sequence_mask[None, None, None, :, None]
        * dimension_mask[None, :, None, None, :]
    )
    candidates = baseline_plan[:, None, None, :, :] + perturbation
    candidates = jnp.clip(
        candidates,
        action_low[None, None, None, None, :],
        action_high[None, None, None, None, :],
    )
    if candidates.shape != (
        batch_size,
        action_dim,
        bins,
        action_sequence,
        action_dim,
    ):
        raise AssertionError("unexpected sibling candidate plan shape")
    return candidates, deltas


def select_single_supported_lcb_plan(
    baseline_plan: jax.Array,
    candidate_plans: jax.Array,
    policy_candidate_scores: jax.Array,
    advantage_ensemble: jax.Array,
    *,
    lcb_scale: float,
    min_lcb_margin: float,
    max_bc_logprob_drop: float,
) -> SupportedLCBPlanResult:
    """Apply at most one supported dimension override to a behavior plan."""

    baseline_plan = jnp.asarray(baseline_plan)
    candidate_plans = jnp.asarray(candidate_plans)
    policy_candidate_scores = jnp.asarray(policy_candidate_scores)
    advantage_ensemble = jnp.asarray(advantage_ensemble)
    if baseline_plan.ndim != 3:
        raise ValueError("baseline_plan must have shape [B, K, D].")
    expected_candidate_shape = (
        baseline_plan.shape[0],
        baseline_plan.shape[-1],
        policy_candidate_scores.shape[-1],
        baseline_plan.shape[1],
        baseline_plan.shape[-1],
    )
    if candidate_plans.shape != expected_candidate_shape:
        raise ValueError(
            "candidate_plans must have shape [B, D, bins, K, D]."
        )
    if policy_candidate_scores.shape[:2] != (
        baseline_plan.shape[0],
        baseline_plan.shape[-1],
    ):
        raise ValueError(
            "policy_candidate_scores must have shape [B, D, bins]."
        )
    if advantage_ensemble.shape[1:] != policy_candidate_scores.shape:
        raise ValueError(
            "advantage_ensemble must have shape [M, B, D, bins]."
        )

    gate = supported_lcb_action_indices(
        policy_candidate_scores,
        advantage_ensemble,
        lcb_scale=lcb_scale,
        min_lcb_margin=min_lcb_margin,
        max_bc_logprob_drop=max_bc_logprob_drop,
    )
    selected_lcb_per_dimension = jnp.take_along_axis(
        gate.lcb_delta,
        gate.indices[..., None],
        axis=-1,
    )[..., 0]
    eligible_lcb = jnp.where(
        gate.override_mask,
        selected_lcb_per_dimension,
        -jnp.inf,
    )
    selected_dimension = jnp.argmax(eligible_lcb, axis=-1)
    applied_override = jnp.any(gate.override_mask, axis=-1)
    batch_index = jnp.arange(baseline_plan.shape[0])
    selected_bin = gate.indices[batch_index, selected_dimension]
    selected_plan = candidate_plans[
        batch_index,
        selected_dimension,
        selected_bin,
    ]
    action = jnp.where(
        applied_override[:, None, None],
        selected_plan,
        baseline_plan,
    )
    selected_lcb_delta = jnp.where(
        applied_override,
        eligible_lcb[batch_index, selected_dimension],
        0.0,
    )
    selected_dimension = jnp.where(
        applied_override,
        selected_dimension,
        -1,
    )
    return SupportedLCBPlanResult(
        action=action,
        candidate_indices=gate.indices,
        bc_indices=gate.bc_indices,
        eligible_override_mask=gate.override_mask,
        applied_override=applied_override,
        selected_dimension=selected_dimension,
        selected_lcb_delta=selected_lcb_delta,
    )


def hl_gauss_encode(
    values: jax.Array,
    *,
    v_min: float,
    v_max: float,
    bins: int = 51,
    sigma: float = 16.0,
) -> jax.Array:
    """Encode scalar flow states with the HL-Gauss representation used by floq.

    ``bins`` denotes Gaussian-CDF boundary points, so the returned feature axis
    has ``bins - 1`` entries. ``sigma`` is measured in boundary-bin widths,
    matching the public floq implementation.  The small denominator floor is
    important when an intermediate flow state briefly leaves the value range.
    """

    if bins < 2:
        raise ValueError("HL-Gauss requires at least two boundary bins.")
    if v_max <= v_min:
        raise ValueError("HL-Gauss requires v_max > v_min.")
    if sigma <= 0.0:
        raise ValueError("HL-Gauss sigma must be positive.")
    values = jnp.asarray(values)
    if values.shape[-1:] == (1,):
        values = values[..., 0]
    # Flow trajectories can briefly overshoot the configured support.  Saturate
    # the encoding at the edge instead of returning an all-zero feature when
    # both Gaussian CDF endpoints round to the same floating-point value.
    values = jnp.clip(values, v_min, v_max)
    support = jnp.linspace(v_min, v_max, bins, dtype=values.dtype)
    bin_width = support[1] - support[0]
    scaled = (support - values[..., None]) / (
        jnp.sqrt(jnp.asarray(2.0, dtype=values.dtype)) * sigma * bin_width
    )
    cdf = jax.scipy.special.erf(scaled)
    normalizer = jnp.maximum(cdf[..., -1:] - cdf[..., :1], 1e-6)
    probabilities = (cdf[..., 1:] - cdf[..., :-1]) / normalizer
    probabilities = jnp.maximum(probabilities, 0.0)
    return probabilities / jnp.maximum(
        probabilities.sum(axis=-1, keepdims=True), 1e-6
    )


def categorical_cross_entropy(
    target_probabilities: jax.Array,
    predicted_logits: jax.Array,
) -> jax.Array:
    """Categorical cross entropy with the atom axis reduced."""

    return -jnp.sum(
        target_probabilities * jax.nn.log_softmax(predicted_logits, axis=-1),
        axis=-1,
    )


def scalar_to_categorical(
    values: jax.Array,
    support: jax.Array,
) -> jax.Array:
    """Project arbitrary-shaped scalar targets onto a uniformly spaced support."""

    values = jnp.asarray(values)
    support = jnp.asarray(support, dtype=values.dtype)
    if support.ndim != 1 or support.shape[0] < 2:
        raise ValueError("support must be a one-dimensional array with >= 2 atoms.")
    v_min = support[0]
    v_max = support[-1]
    delta = (v_max - v_min) / float(support.shape[0] - 1)
    clipped = jnp.clip(values, v_min, v_max)
    projected_index = (clipped - v_min) / delta
    lower = jnp.floor(projected_index).astype(jnp.int32)
    upper = jnp.ceil(projected_index).astype(jnp.int32)
    lower_weight = jnp.where(
        lower == upper,
        1.0,
        upper.astype(values.dtype) - projected_index,
    )
    upper_weight = jnp.where(
        lower == upper,
        0.0,
        projected_index - lower.astype(values.dtype),
    )
    atoms = support.shape[0]
    return (
        jax.nn.one_hot(lower, atoms, dtype=values.dtype) * lower_weight[..., None]
        + jax.nn.one_hot(upper, atoms, dtype=values.dtype)
        * upper_weight[..., None]
    )


def demo_margin_per_sample(
    all_q: jax.Array,
    chosen_q: jax.Array,
    *,
    margin: float,
) -> jax.Array:
    """CQN-AS large-margin imitation loss before applying the demo mask."""

    losses = jnp.maximum(margin - (chosen_q[..., None] - all_q), 0.0)
    return losses.mean(axis=tuple(range(1, losses.ndim)))


def demo_fosd_per_sample(
    chosen_probabilities: jax.Array,
    all_probabilities: jax.Array,
) -> jax.Array:
    """Original CQN first-order-stochastic-dominance demo penalty."""

    chosen_cdf = jnp.cumsum(chosen_probabilities, axis=-1)
    all_cdf = jnp.cumsum(all_probabilities, axis=-1)
    losses = jnp.maximum(chosen_cdf[..., None, :] - all_cdf, 0.0).sum(axis=-1)
    return losses.mean(axis=tuple(range(1, losses.ndim)))


def source_bin_flip_rate_per_sample(all_q_samples: jax.Array) -> jax.Array:
    """Fraction of source-wise bin choices disagreeing with the mean-Q bin.

    ``all_q_samples`` is shaped ``[B,R,...,N]``.  This is a cheap diagnostic
    at a fixed replay/demo condition and C2F level; it is not the flip rate of
    a complete autoregressive rollout, whose later zoom conditions can differ.
    """

    all_q_samples = jnp.asarray(all_q_samples)
    if all_q_samples.ndim < 3:
        raise ValueError("all_q_samples must have [B,R,...,N] axes.")
    mean_choice = jnp.argmax(all_q_samples.mean(axis=1), axis=-1)
    source_choice = jnp.argmax(all_q_samples, axis=-1)
    disagreement = source_choice != mean_choice[:, None]
    return disagreement.mean(axis=tuple(range(1, disagreement.ndim)))


def integrate_value_flow(
    velocity_fn: Callable[[jax.Array, jax.Array], jax.Array],
    source: jax.Array,
    *,
    num_flow_steps: int,
    end_tau: jax.Array | float = 0.0,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> jax.Array:
    """Euler-integrate from source (tau=1) to ``end_tau``.

    The Python loop is intentionally static so JAX unrolls the small number of
    configured flow steps.  Every other axis -- batch, source sample, sequence,
    action dimension, bin, and value dimension -- remains vectorised.
    """

    if num_flow_steps < 1:
        raise ValueError("num_flow_steps must be positive.")
    if (clip_min is None) != (clip_max is None):
        raise ValueError("clip_min and clip_max must be set together.")
    if clip_min is not None and clip_max is not None and clip_max <= clip_min:
        raise ValueError("clip_max must be greater than clip_min.")
    value = jnp.asarray(source)
    end_tau = jnp.asarray(end_tau, dtype=value.dtype)
    if end_tau.ndim > value.ndim:
        raise ValueError("end_tau rank cannot exceed the flow-state rank.")
    broadcast_shape = (*end_tau.shape, *((1,) * (value.ndim - end_tau.ndim)))
    step_size = (jnp.ones_like(end_tau) - end_tau) / float(num_flow_steps)
    value_step_size = step_size.reshape(broadcast_shape)
    for step in range(num_flow_steps):
        tau = jnp.ones_like(end_tau) - step * step_size
        value = value + value_step_size * velocity_fn(value, tau)
        if clip_min is not None and clip_max is not None:
            value = jnp.clip(value, clip_min, clip_max)
    return value


def integrate_value_flow_trajectory(
    velocity_fn: Callable[[jax.Array, jax.Array], jax.Array],
    source: jax.Array,
    *,
    num_flow_steps: int,
    end_tau: jax.Array | float = 0.0,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> jax.Array:
    """Euler-integrate a value flow and retain every intermediate state.

    The returned array has a leading axis of length ``num_flow_steps + 1``.
    Entry zero is the exact source and the final entry is bit-for-bit the same
    update recurrence as :func:`integrate_value_flow`.  This read-only helper
    supports FLOQ collapse/utilization diagnostics; training continues to use
    the endpoint-only integrator.
    """

    if num_flow_steps < 1:
        raise ValueError("num_flow_steps must be positive.")
    if (clip_min is None) != (clip_max is None):
        raise ValueError("clip_min and clip_max must be set together.")
    if clip_min is not None and clip_max is not None and clip_max <= clip_min:
        raise ValueError("clip_max must be greater than clip_min.")
    value = jnp.asarray(source)
    end_tau = jnp.asarray(end_tau, dtype=value.dtype)
    if end_tau.ndim > value.ndim:
        raise ValueError("end_tau rank cannot exceed the flow-state rank.")
    broadcast_shape = (*end_tau.shape, *((1,) * (value.ndim - end_tau.ndim)))
    step_size = (jnp.ones_like(end_tau) - end_tau) / float(num_flow_steps)
    value_step_size = step_size.reshape(broadcast_shape)
    trajectory = [value]
    for step in range(num_flow_steps):
        tau = jnp.ones_like(end_tau) - step * step_size
        value = value + value_step_size * velocity_fn(value, tau)
        if clip_min is not None and clip_max is not None:
            value = jnp.clip(value, clip_min, clip_max)
        trajectory.append(value)
    return jnp.stack(trajectory, axis=0)


def scalar_flow_trajectory_diagnostics(
    trajectory: jax.Array,
    *,
    epsilon: float = 1e-6,
) -> dict[str, jax.Array]:
    """Quantify whether a scalar flow uses nontrivial iterative computation.

    ``trajectory`` must have shape ``[steps + 1, batch, sources, ..., 1]``.
    Curvature is measured against the straight chord joining each exact source
    to its own final endpoint.  Source contraction reports how strongly the
    endpoint suppresses initial-noise variation.  These are mechanistic
    diagnostics only: neither quantity is a policy-quality metric.
    """

    trajectory = jnp.asarray(trajectory)
    if trajectory.ndim < 4 or trajectory.shape[-1] != 1:
        raise ValueError(
            "trajectory must have shape [steps + 1, batch, sources, ..., 1]."
        )
    if trajectory.shape[0] < 3:
        raise ValueError("curvature diagnostics require at least two flow steps.")
    if trajectory.shape[2] < 2:
        raise ValueError("source diagnostics require at least two sources.")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")

    source = trajectory[0]
    endpoint = trajectory[-1]
    fractions = jnp.linspace(
        0.0,
        1.0,
        trajectory.shape[0],
        dtype=trajectory.dtype,
    ).reshape((trajectory.shape[0], *((1,) * (trajectory.ndim - 1))))
    chord = source[None] + fractions * (endpoint - source)[None]
    residual = trajectory[1:-1] - chord[1:-1]
    curvature_abs_mean = jnp.mean(jnp.abs(residual))
    curvature_rms = jnp.sqrt(jnp.mean(jnp.square(residual)))
    displacement_rms = jnp.sqrt(
        jnp.mean(jnp.square(endpoint - source))
    )

    source_std = jnp.std(source[..., 0], axis=1)
    endpoint_std = jnp.std(endpoint[..., 0], axis=1)
    source_std_mean = jnp.mean(source_std)
    endpoint_std_mean = jnp.mean(endpoint_std)

    increments = jnp.diff(trajectory, axis=0)
    mean_increment = jnp.mean(increments, axis=0, keepdims=True)
    increment_variation_rms = jnp.sqrt(
        jnp.mean(jnp.square(increments - mean_increment))
    )
    increment_rms = jnp.sqrt(jnp.mean(jnp.square(increments)))

    return {
        "curvature_abs_mean": curvature_abs_mean,
        "curvature_rms": curvature_rms,
        "normalized_curvature_rms": curvature_rms
        / (displacement_rms + float(epsilon)),
        "displacement_rms": displacement_rms,
        "source_std_mean": source_std_mean,
        "endpoint_std_mean": endpoint_std_mean,
        "source_contraction_ratio": endpoint_std_mean
        / (source_std_mean + float(epsilon)),
        "increment_variation_rms": increment_variation_rms,
        "normalized_increment_variation": increment_variation_rms
        / (increment_rms + float(epsilon)),
    }


def integrate_value_flow_with_source_jvp(
    velocity_fn: Callable[[jax.Array, jax.Array], jax.Array],
    source: jax.Array,
    *,
    source_tangent: jax.Array | None = None,
    num_flow_steps: int,
    end_tau: jax.Array | float = 0.0,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Euler-integrate a value flow and its source Jacobian-vector product.

    This is the reverse-time equivalent of Value Flows' flow-derivative ODE.
    The tangent follows ``d phi(source) / d source`` without differentiating
    through the whole solver.  As in the official implementation, optional
    support clipping applies to the transported return but not to the tangent.
    """

    if num_flow_steps < 1:
        raise ValueError("num_flow_steps must be positive.")
    if (clip_min is None) != (clip_max is None):
        raise ValueError("clip_min and clip_max must be set together.")
    if clip_min is not None and clip_max is not None and clip_max <= clip_min:
        raise ValueError("clip_max must be greater than clip_min.")
    value = jnp.asarray(source)
    tangent = (
        jnp.ones_like(value)
        if source_tangent is None
        else jnp.asarray(source_tangent, dtype=value.dtype)
    )
    if tangent.shape != value.shape:
        raise ValueError("source_tangent must have the same shape as source.")
    end_tau = jnp.asarray(end_tau, dtype=value.dtype)
    if end_tau.ndim > value.ndim:
        raise ValueError("end_tau rank cannot exceed the flow-state rank.")
    broadcast_shape = (*end_tau.shape, *((1,) * (value.ndim - end_tau.ndim)))
    step_size = (jnp.ones_like(end_tau) - end_tau) / float(num_flow_steps)
    value_step_size = step_size.reshape(broadcast_shape)
    for step in range(num_flow_steps):
        tau = jnp.ones_like(end_tau) - step * step_size
        velocity, tangent_velocity = jax.jvp(
            lambda current: velocity_fn(current, tau),
            (value,),
            (tangent,),
        )
        value = value + value_step_size * velocity
        tangent = tangent + value_step_size * tangent_velocity
        if clip_min is not None and clip_max is not None:
            value = jnp.clip(value, clip_min, clip_max)
    return value, tangent


@dataclass(frozen=True)
class CQNFlowSpec(CQNASpec):
    """CQN-AS settings plus conditional value-flow hyperparameters."""

    value_mode: str
    num_flow_steps: int
    num_flow_samples: int
    num_target_flow_samples: int
    num_action_flow_samples: int
    flow_source_type: str
    flow_source_std: float
    flow_source_min: float | None
    flow_source_max: float | None
    antithetic_flow_sources: bool
    fixed_action_flow_sources: bool
    action_flow_quantile_grid: bool
    flow_iqn_quantile_coupling: bool
    quantile_endpoint_lambda: float
    quantile_huber_kappa: float
    return_sample_aggregation: str
    return_sample_temperature: float
    return_sample_truncate_top: int
    flow_q_action_readout: bool
    atom_ce_lambda: float
    bcfm_lambda: float
    dcfm_lambda: float
    evor_td_lambda: float
    confidence_weight_temp: float | None
    pcbf_loss_coeff: float
    pcbf_lambda: float
    endpoint_q_lambda: float
    source_consistency_lambda: float
    flow_distill_lambda: float
    flow_distill_action_readout: bool
    demo_flow_steps: int | None
    query_hidden_dim: int
    time_embedding_type: str
    time_embed_dim: int
    time_scale: float
    clip_scalar_targets: bool
    clip_flow_trajectory: bool
    scalar_value_embedding: str
    scalar_embed_bins: int
    scalar_embed_sigma: float
    critic_architecture: str
    advantage_c51_lambda: float
    advantage_q_lambda: float
    causal_branch_cache: str | None
    causal_branch_weight: float
    causal_branch_delta_weight: float
    causal_branch_temperature: float
    causal_branch_batch_size: int
    causal_branch_level: int
    freeze_bc_policy: bool
    bc_policy_mode: str


def cqn_flow_spec_from_cfg(cfg: DictConfig) -> CQNFlowSpec:
    method = cfg.method
    base = cqn_as_spec_from_cfg(cfg)
    base_values = {field.name: getattr(base, field.name) for field in fields(CQNASpec)}
    num_flow_samples = int(method.get("num_flow_samples", 2))
    source_min = method.get("flow_source_min", None)
    source_max = method.get("flow_source_max", None)
    demo_flow_steps = method.get("demo_flow_steps", None)
    causal_branch_cache = method.get("causal_branch_cache", None)
    return CQNFlowSpec(
        **base_values,
        value_mode=str(method.get("value_mode", "categorical")).lower(),
        num_flow_steps=int(method.get("num_flow_steps", 2)),
        num_flow_samples=num_flow_samples,
        num_target_flow_samples=int(
            method.get("num_target_flow_samples", num_flow_samples)
        ),
        num_action_flow_samples=int(
            method.get("num_action_flow_samples", num_flow_samples)
        ),
        flow_source_type=str(method.get("flow_source_type", "gaussian")).lower(),
        flow_source_std=float(method.get("flow_source_std", 1.0)),
        flow_source_min=None if source_min is None else float(source_min),
        flow_source_max=None if source_max is None else float(source_max),
        antithetic_flow_sources=bool(
            method.get("antithetic_flow_sources", False)
        ),
        fixed_action_flow_sources=bool(
            method.get("fixed_action_flow_sources", False)
        ),
        action_flow_quantile_grid=bool(
            method.get("action_flow_quantile_grid", False)
        ),
        flow_iqn_quantile_coupling=bool(
            method.get("flow_iqn_quantile_coupling", False)
        ),
        quantile_endpoint_lambda=float(
            method.get("quantile_endpoint_lambda", 0.0)
        ),
        quantile_huber_kappa=float(
            method.get("quantile_huber_kappa", 1.0)
        ),
        return_sample_aggregation=str(
            method.get("return_sample_aggregation", "mean")
        ).lower(),
        return_sample_temperature=float(
            method.get("return_sample_temperature", 1.0)
        ),
        return_sample_truncate_top=int(
            method.get("return_sample_truncate_top", 0)
        ),
        flow_q_action_readout=bool(
            method.get("flow_q_action_readout", False)
        ),
        atom_ce_lambda=float(method.get("atom_ce_lambda", 1.0)),
        bcfm_lambda=float(method.get("bcfm_lambda", 1.0)),
        dcfm_lambda=float(method.get("dcfm_lambda", 0.0)),
        evor_td_lambda=float(method.get("evor_td_lambda", 0.0)),
        confidence_weight_temp=(
            None
            if method.get("confidence_weight_temp", None) is None
            else float(method.confidence_weight_temp)
        ),
        pcbf_loss_coeff=float(method.get("pcbf_loss_coeff", 0.0)),
        pcbf_lambda=float(method.get("pcbf_lambda", 0.0)),
        endpoint_q_lambda=float(method.get("endpoint_q_lambda", 0.0)),
        source_consistency_lambda=float(
            method.get("source_consistency_lambda", 0.0)
        ),
        flow_distill_lambda=float(
            method.get("flow_distill_lambda", 0.0)
        ),
        flow_distill_action_readout=bool(
            method.get("flow_distill_action_readout", False)
        ),
        demo_flow_steps=(
            None if demo_flow_steps is None else int(demo_flow_steps)
        ),
        query_hidden_dim=int(method.get("query_hidden_dim", 128)),
        time_embedding_type=str(
            method.get("time_embedding_type", "sinusoidal")
        ).lower(),
        time_embed_dim=int(method.get("time_embed_dim", 32)),
        time_scale=float(method.get("time_scale", 1000.0)),
        clip_scalar_targets=bool(method.get("clip_scalar_targets", True)),
        clip_flow_trajectory=bool(method.get("clip_flow_trajectory", False)),
        scalar_value_embedding=str(
            method.get("scalar_value_embedding", "raw")
        ).lower(),
        scalar_embed_bins=int(method.get("scalar_embed_bins", 51)),
        scalar_embed_sigma=float(method.get("scalar_embed_sigma", 16.0)),
        critic_architecture=str(
            method.get("critic_architecture", "flow_q")
        ).lower(),
        advantage_c51_lambda=float(
            method.get("advantage_c51_lambda", 1.0)
        ),
        advantage_q_lambda=float(method.get("advantage_q_lambda", 1.0)),
        causal_branch_cache=(
            None
            if causal_branch_cache is None
            else str(causal_branch_cache)
        ),
        causal_branch_weight=float(
            method.get("causal_branch_weight", 0.0)
        ),
        causal_branch_delta_weight=float(
            method.get("causal_branch_delta_weight", 10.0)
        ),
        causal_branch_temperature=float(
            method.get("causal_branch_temperature", 0.05)
        ),
        causal_branch_batch_size=int(
            method.get("causal_branch_batch_size", 32)
        ),
        causal_branch_level=int(method.get("causal_branch_level", 1)),
        freeze_bc_policy=bool(method.get("freeze_bc_policy", False)),
        bc_policy_mode=str(
            method.get("bc_policy_mode", "behavior_logits")
        ).lower(),
    )


class C2FSequenceFlowCritic(nn.Module):
    """Shared sequence trunk and vectorised conditional value-flow head.

    Inputs use ``[B,R,K,D,N,V]`` for the flow state.  ``R`` source samples and
    ``N`` bins are explicit batch axes, so neither produces an extra Python
    loop or another image-encoder call.
    """

    hidden_dims: tuple[int, ...]
    query_hidden_dim: int
    time_embed_dim: int
    action_sequence: int
    action_dim: int
    levels: int
    bins: int
    value_dim: int
    scalar_value_embedding: str = "raw"
    scalar_embed_bins: int = 51
    scalar_embed_sigma: float = 16.0
    value_min: float = -2.0
    value_max: float = 2.0
    time_embedding_type: str = "sinusoidal"
    time_scale: float = 1000.0
    low_dim_size: int = 0
    feature_dim: int = 64
    rgb_encoder_layers: int = 2
    gru_layers: int = 1
    activation_name: str = "silu"
    use_dueling: bool = True
    value_only: bool = False
    quantile_conditioning: bool = False

    @nn.compact
    def __call__(
        self,
        features: jax.Array,
        level_one_hot: jax.Array,
        low_high_midpoint: jax.Array,
        interval_half_width: jax.Array,
        candidate_bins: jax.Array,
        candidate_centers: jax.Array,
        flow_values: jax.Array,
        tau: jax.Array,
        source_quantiles: jax.Array | None = None,
    ) -> jax.Array:
        batch_size, source_samples, sequence, action_dim, query_bins, value_dim = (
            flow_values.shape
        )
        if sequence != self.action_sequence or action_dim != self.action_dim:
            raise ValueError("flow_values sequence/action axes do not match the model.")
        if value_dim != self.value_dim:
            raise ValueError("flow_values value axis does not match value_dim.")

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
        stream_features = features
        if exact_pixel_arch:
            low_dim = features[:, : self.low_dim_size]
            rgb = features[:, self.low_dim_size :]
            for index in range(self.rgb_encoder_layers):
                rgb = nn.Dense(
                    self.hidden_dims[0],
                    use_bias=False,
                    kernel_init=nn.initializers.orthogonal(),
                    name=f"context_rgb_dense_{index}",
                )(rgb)
                rgb = nn.LayerNorm(name=f"context_rgb_norm_{index}")(rgb)
                rgb = activation(rgb, self.activation_name)
            rgb = nn.Dense(
                self.feature_dim,
                use_bias=False,
                kernel_init=nn.initializers.orthogonal(),
                name="context_rgb_projection",
            )(rgb)
            rgb = jnp.tanh(nn.LayerNorm(name="context_rgb_projection_norm")(rgb))
            low_dim = nn.Dense(
                self.feature_dim,
                use_bias=False,
                kernel_init=nn.initializers.orthogonal(),
                name="context_low_dim_projection",
            )(low_dim)
            low_dim = jnp.tanh(nn.LayerNorm(name="context_low_dim_norm")(low_dim))
            stream_features = jnp.concatenate([rgb, low_dim], axis=-1)

        repeated_features = jnp.broadcast_to(
            stream_features[:, None, :],
            (batch_size, self.action_sequence, stream_features.shape[-1]),
        )
        context = jnp.concatenate(
            [
                repeated_features,
                low_high_midpoint,
                interval_half_width,
                sequence_id,
                repeated_level,
            ],
            axis=-1,
        )
        for index, width in enumerate(self.hidden_dims):
            context = nn.Dense(
                width,
                use_bias=False,
                kernel_init=nn.initializers.orthogonal(),
                name=f"context_dense_{index}",
            )(context)
            context = nn.LayerNorm(name=f"context_norm_{index}")(context)
            context = activation(context, self.activation_name)

        hidden_size = self.hidden_dims[-1]
        ScanGRU = nn.scan(
            nn.GRUCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )
        for layer in range(self.gru_layers):
            carry = jnp.zeros((batch_size, hidden_size), dtype=context.dtype)
            scan_gru = ScanGRU(features=hidden_size, name=f"context_gru_{layer}")
            _, context = scan_gru(carry, context)

        context = nn.Dense(
            self.query_hidden_dim,
            kernel_init=nn.initializers.orthogonal(),
            name="context_projection",
        )(context)[:, None, :, None, None, :]

        candidate_condition = None
        if not self.value_only:
            candidate_bins = jnp.asarray(candidate_bins, dtype=jnp.int32)
            candidate_centers = jnp.asarray(candidate_centers, dtype=dtype)
            dim_id = jnp.broadcast_to(
                jnp.eye(self.action_dim, dtype=dtype)[None, None, :, None, :],
                (
                    batch_size,
                    self.action_sequence,
                    self.action_dim,
                    query_bins,
                    self.action_dim,
                ),
            )
            bin_id = jax.nn.one_hot(candidate_bins, self.bins, dtype=dtype)
            local_width = jnp.broadcast_to(
                interval_half_width[..., None, None],
                (*candidate_centers.shape, 1),
            )
            candidate_condition = jnp.concatenate(
                [
                    candidate_centers[..., None],
                    local_width,
                    dim_id,
                    bin_id,
                ],
                axis=-1,
            )
            candidate_condition = nn.Dense(
                self.query_hidden_dim,
                kernel_init=nn.initializers.orthogonal(),
                name="candidate_projection",
            )(candidate_condition)[:, None, ...]

        tau = jnp.asarray(tau, dtype=dtype)
        if tau.ndim == 0:
            tau = jnp.broadcast_to(tau, (batch_size, source_samples))
        elif tau.ndim == 1:
            tau = jnp.broadcast_to(tau[:, None], (batch_size, source_samples))
        else:
            tau = jnp.broadcast_to(tau, (batch_size, source_samples))
        if self.time_embedding_type == "fourier":
            # Paper-faithful floq convention uses t=0 at the source, whereas
            # this repository calls that endpoint tau=1.
            forward_time = jnp.ones_like(tau) - tau
            frequencies = jnp.arange(
                1, self.time_embed_dim + 1, dtype=dtype
            )
            time_features = jnp.cos(
                jnp.pi
                * forward_time[..., None]
                * frequencies[None, None, :]
            )
        elif self.time_embedding_type == "raw":
            time_features = (jnp.ones_like(tau) - tau)[..., None]
        else:
            time_features = SinusoidalPosEmb(
                self.time_embed_dim,
                name="time_embedding",
            )((tau * self.time_scale).reshape((-1,))).reshape(
                (batch_size, source_samples, self.time_embed_dim)
            )
        time_features = nn.Dense(
            self.query_hidden_dim,
            kernel_init=nn.initializers.orthogonal(),
            name="time_projection",
        )(time_features)[:, :, None, None, None, :]

        value_inputs = flow_values
        if self.value_dim == 1 and self.scalar_value_embedding == "hl_gauss":
            value_inputs = hl_gauss_encode(
                flow_values,
                v_min=self.value_min,
                v_max=self.value_max,
                bins=self.scalar_embed_bins,
                sigma=self.scalar_embed_sigma,
            )
        value_features = nn.Dense(
            self.query_hidden_dim,
            kernel_init=nn.initializers.orthogonal(),
            name="value_projection",
        )(value_inputs)
        quantile_features = 0.0
        if self.quantile_conditioning:
            if source_quantiles is None:
                raise ValueError(
                    "quantile-conditioned critic requires source_quantiles."
                )
            source_quantiles = jnp.asarray(source_quantiles, dtype=dtype)
            expected_quantile_shape = (*flow_values.shape[:-1], 1)
            if source_quantiles.shape != expected_quantile_shape:
                raise ValueError(
                    "source_quantiles must match flow-value leading axes."
                )
            quantile_features = nn.Dense(
                self.query_hidden_dim,
                kernel_init=nn.initializers.orthogonal(),
                name="quantile_projection",
            )(source_quantiles)
        if self.value_only:
            # This branch is structurally independent of candidate bin/center.
            # With common source samples across bins, all candidates therefore
            # receive exactly the same state/prefix baseline.
            value_query = (
                context + time_features + value_features + quantile_features
            )
            value_query = activation(
                nn.LayerNorm(name="value_query_norm")(value_query),
                self.activation_name,
            )
            return nn.Dense(
                self.value_dim,
                kernel_init=nn.initializers.zeros_init(),
                bias_init=nn.initializers.zeros_init(),
                name="value_velocity_head",
            )(value_query)

        query = (
            context
            + candidate_condition
            + time_features
            + value_features
            + quantile_features
        )
        query = nn.LayerNorm(name="query_norm")(query)
        query = activation(query, self.activation_name)
        query = nn.Dense(
            self.query_hidden_dim,
            kernel_init=nn.initializers.orthogonal(),
            name="query_dense",
        )(query)
        query = activation(query, self.activation_name)
        advantage = nn.Dense(
            self.value_dim,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="velocity_head",
        )(query)
        if not self.use_dueling:
            return advantage

        # A candidate-independent residual provides a dueling-style state-value
        # path without subtracting a query-set mean.  That keeps one bin's
        # vector field invariant whether it is queried alone or with all bins.
        value_query = (
            context + time_features + value_features + quantile_features
        )
        value_query = activation(
            nn.LayerNorm(name="value_query_norm")(value_query),
            self.activation_name,
        )
        value = nn.Dense(
            self.value_dim,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="value_velocity_head",
        )(value_query)
        return value + advantage


class CQNFlowAS(CQNAS):
    """CQN-AS with categorical, expected-value, or return-sample flows."""

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
        td_target_policy_value_beta: float | None,
        critic_sequence_mode: str,
        mc_return_weight: float,
        mc_return_stop_gradient_encoder: bool,
        mc_return_value_only: bool,
        value_mode: str,
        num_flow_steps: int,
        num_flow_samples: int,
        num_target_flow_samples: int,
        num_action_flow_samples: int,
        flow_source_type: str,
        flow_source_std: float,
        flow_source_min: float | None,
        flow_source_max: float | None,
        antithetic_flow_sources: bool,
        fixed_action_flow_sources: bool,
        action_flow_quantile_grid: bool,
        flow_iqn_quantile_coupling: bool,
        quantile_endpoint_lambda: float,
        quantile_huber_kappa: float,
        return_sample_aggregation: str,
        return_sample_temperature: float,
        return_sample_truncate_top: int,
        flow_q_action_readout: bool,
        atom_ce_lambda: float,
        bcfm_lambda: float,
        dcfm_lambda: float,
        evor_td_lambda: float,
        confidence_weight_temp: float | None,
        pcbf_loss_coeff: float,
        pcbf_lambda: float,
        endpoint_q_lambda: float,
        source_consistency_lambda: float,
        flow_distill_lambda: float,
        flow_distill_action_readout: bool,
        demo_flow_steps: int | None,
        demo_fosd: bool,
        query_hidden_dim: int,
        time_embedding_type: str,
        time_embed_dim: int,
        time_scale: float,
        clip_scalar_targets: bool,
        clip_flow_trajectory: bool,
        scalar_value_embedding: str,
        scalar_embed_bins: int,
        scalar_embed_sigma: float,
        critic_architecture: str,
        advantage_c51_lambda: float,
        advantage_q_lambda: float,
        causal_branch_cache: str | None,
        causal_branch_weight: float,
        causal_branch_delta_weight: float,
        causal_branch_temperature: float,
        causal_branch_batch_size: int,
        causal_branch_level: int,
        policy_value_beta: float | None,
        freeze_bc_policy: bool,
        bc_policy_mode: str,
        model: RLModelSpec,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        num_train_envs: int,
        num_eval_envs: int,
        replay_alpha: float,
        replay_beta: float,
        frame_stack_on_channel: bool,
        intrinsic_reward_module=None,
        demo_batch_size: Optional[int] = None,
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
        value_mode = str(value_mode).lower()
        flow_source_type = str(flow_source_type).lower()
        scalar_value_embedding = str(scalar_value_embedding).lower()
        time_embedding_type = str(time_embedding_type).lower()
        critic_architecture = str(critic_architecture).lower()
        bc_policy_mode = str(bc_policy_mode).lower()
        return_sample_aggregation = str(return_sample_aggregation).lower()
        if self.action_sequence < 2:
            raise ValueError("CQN-Flow requires action_sequence >= 2.")
        if levels < 1 or bins < 2:
            raise ValueError("CQN-Flow requires levels >= 1 and bins >= 2.")
        if atoms < 2 or v_max <= v_min:
            raise ValueError("CQN-Flow requires atoms >= 2 and v_max > v_min.")
        if value_mode not in {"categorical", "scalar", "return_sample"}:
            raise ValueError(
                "value_mode must be 'categorical', 'scalar', or "
                "'return_sample'."
            )
        if critic_architecture not in {"flow_q", "flow_v_direct_a"}:
            raise ValueError(
                "critic_architecture must be 'flow_q' or 'flow_v_direct_a'."
            )
        if not model.hidden_dims:
            raise ValueError("CQN-Flow requires at least one critic hidden layer.")
        if gru_layers < 1:
            raise ValueError("CQN-Flow requires gru_layers >= 1.")
        if not 1 <= temporal_ensemble_replan_interval <= self.action_sequence:
            raise ValueError(
                "temporal_ensemble_replan_interval must be in "
                "[1, action_sequence]."
            )
        if (
            num_flow_steps < 1
            or num_flow_samples < 1
            or num_target_flow_samples < 1
            or num_action_flow_samples < 1
        ):
            raise ValueError("Flow steps and source samples must be positive.")
        if demo_flow_steps is not None and not (
            1 <= demo_flow_steps <= num_flow_steps
        ):
            raise ValueError(
                "demo_flow_steps must be in [1, num_flow_steps] or null."
            )
        if flow_source_type not in {"gaussian", "uniform"}:
            raise ValueError("flow_source_type must be 'gaussian' or 'uniform'.")
        if flow_source_type == "gaussian" and flow_source_std <= 0.0:
            raise ValueError("flow_source_std must be positive.")
        if return_sample_aggregation not in {
            "mean",
            "entropic",
            "truncated_mean",
        }:
            raise ValueError(
                "return_sample_aggregation must be 'mean', 'entropic', or "
                "'truncated_mean'."
            )
        if return_sample_temperature <= 0.0:
            raise ValueError("return_sample_temperature must be positive.")
        if return_sample_truncate_top < 0:
            raise ValueError(
                "return_sample_truncate_top must be non-negative."
            )
        if (
            return_sample_aggregation == "truncated_mean"
            and return_sample_truncate_top
            >= min(
                num_flow_samples,
                num_target_flow_samples,
                num_action_flow_samples,
            )
        ):
            raise ValueError(
                "return_sample_truncate_top must be smaller than every "
                "configured return sample count."
            )
        if (
            return_sample_aggregation != "mean"
            and value_mode != "return_sample"
        ):
            raise ValueError(
                "Non-mean return aggregation requires "
                "value_mode=return_sample."
            )
        if (flow_source_min is None) != (flow_source_max is None):
            raise ValueError(
                "flow_source_min and flow_source_max must be set together."
            )
        if (
            flow_source_min is not None
            and flow_source_max is not None
            and flow_source_max <= flow_source_min
        ):
            raise ValueError("flow_source_max must be greater than flow_source_min.")
        if atom_ce_lambda < 0.0:
            raise ValueError("atom_ce_lambda must be non-negative.")
        if (
            confidence_weight_temp is not None
            and confidence_weight_temp <= 0.0
        ):
            raise ValueError("confidence_weight_temp must be positive or null.")
        if min(
            bcfm_lambda,
            dcfm_lambda,
            evor_td_lambda,
            pcbf_loss_coeff,
            pcbf_lambda,
            quantile_endpoint_lambda,
            endpoint_q_lambda,
            source_consistency_lambda,
            flow_distill_lambda,
        ) < 0.0:
            raise ValueError("Flow objective coefficients must be non-negative.")
        if quantile_huber_kappa <= 0.0:
            raise ValueError("quantile_huber_kappa must be positive.")
        if value_mode != "categorical" and atom_ce_lambda != 0.0:
            raise ValueError(
                "Scalar CQN-Flow has no atom CE; set atom_ce_lambda=0."
            )
        if value_mode != "categorical" and demo_fosd:
            raise ValueError(
                "Scalar CQN-Flow has no return CDF; set demo_fosd=false."
            )
        if value_mode != "return_sample" and dcfm_lambda != 0.0:
            raise ValueError(
                "DCFM is only defined for value_mode=return_sample; set "
                "dcfm_lambda=0."
            )
        if evor_td_lambda > 0.0:
            if value_mode != "return_sample":
                raise ValueError(
                    "EVOR FlowTD requires value_mode=return_sample."
                )
            if flow_source_type != "gaussian":
                raise ValueError("EVOR FlowTD requires a Gaussian source.")
            if critic_architecture != "flow_q":
                raise ValueError(
                    "EVOR FlowTD requires critic_architecture=flow_q."
                )
            if (
                not separate_bc_policy
                or str(td_target_action_source).lower() != "bc_policy"
            ):
                raise ValueError(
                    "EVOR FlowTD requires a separate BC policy and "
                    "td_target_action_source=bc_policy."
                )
            if (
                bcfm_lambda != 0.0
                or dcfm_lambda != 0.0
                or pcbf_loss_coeff != 0.0
                or flow_iqn_quantile_coupling
            ):
                raise ValueError(
                    "EVOR FlowTD is an isolated objective; disable BCFM, "
                    "DCFM, PCBF, and FlowIQN coupling."
                )
            if confidence_weight_temp is not None:
                raise ValueError(
                    "EVOR FlowTD does not use Value-Flows confidence weights."
                )
        if (
            confidence_weight_temp is not None
            and value_mode != "return_sample"
        ):
            raise ValueError(
                "Value-Flow confidence weighting requires "
                "value_mode=return_sample."
            )
        if (
            confidence_weight_temp is not None
            and flow_source_type != "gaussian"
        ):
            raise ValueError(
                "Value-Flow confidence weighting requires a Gaussian source."
            )
        if confidence_weight_temp is not None and critic_architecture != "flow_q":
            raise ValueError(
                "Value-Flow confidence weighting requires "
                "critic_architecture=flow_q."
            )
        if value_mode != "return_sample" and (
            pcbf_loss_coeff != 0.0 or pcbf_lambda != 0.0
        ):
            raise ValueError(
                "PCBF is only defined for value_mode=return_sample; set "
                "pcbf_loss_coeff=0 and pcbf_lambda=0."
            )
        if flow_iqn_quantile_coupling:
            if value_mode != "return_sample":
                raise ValueError(
                    "FlowIQN quantile coupling requires "
                    "value_mode=return_sample."
                )
            if flow_source_type != "uniform":
                raise ValueError(
                    "FlowIQN quantile coupling requires a uniform source."
                )
            if antithetic_flow_sources:
                raise ValueError(
                    "FlowIQN uses independently sampled source quantiles; "
                    "set antithetic_flow_sources=false."
                )
            if num_target_flow_samples != num_flow_samples:
                raise ValueError(
                    "FlowIQN requires equal train and target sample counts."
                )
            if bcfm_lambda <= 0.0:
                raise ValueError("FlowIQN requires bcfm_lambda > 0.")
            if critic_architecture != "flow_q":
                raise ValueError(
                    "FlowIQN quantile coupling requires critic_architecture="
                    "flow_q."
                )
            if dcfm_lambda != 0.0 or pcbf_loss_coeff != 0.0:
                raise ValueError(
                    "FlowIQN is a quantile-coupled CFM objective; disable "
                    "DCFM and PCBF."
                )
        if (
            quantile_endpoint_lambda > 0.0
            and not flow_iqn_quantile_coupling
        ):
            raise ValueError(
                "All-pairs endpoint quantile regression requires "
                "flow_iqn_quantile_coupling=true."
            )
        if action_flow_quantile_grid and not flow_iqn_quantile_coupling:
            raise ValueError(
                "A deterministic action quantile grid requires "
                "flow_iqn_quantile_coupling=true."
            )
        if value_mode == "return_sample" and source_consistency_lambda != 0.0:
            raise ValueError(
                "Return-sample flow variance is intentional; set "
                "source_consistency_lambda=0."
            )
        if flow_distill_lambda > 0.0 and value_mode != "scalar":
            raise ValueError(
                "FLOQ's distilled scalar readout requires value_mode=scalar."
            )
        if flow_distill_lambda > 0.0 and critic_architecture != "flow_q":
            raise ValueError(
                "The distilled scalar readout requires "
                "critic_architecture=flow_q."
            )
        if flow_distill_action_readout and flow_distill_lambda <= 0.0:
            raise ValueError(
                "flow_distill_action_readout=true requires "
                "flow_distill_lambda > 0."
            )
        if flow_q_action_readout and critic_architecture != "flow_q":
            raise ValueError(
                "flow_q_action_readout=true requires "
                "critic_architecture=flow_q."
            )
        if (
            value_mode == "return_sample"
            and num_target_flow_samples != num_flow_samples
        ):
            raise ValueError(
                "return_sample requires num_target_flow_samples == "
                "num_flow_samples so BCFM can couple the same base noise."
            )
        if scalar_value_embedding not in {"raw", "hl_gauss"}:
            raise ValueError(
                "scalar_value_embedding must be 'raw' or 'hl_gauss'."
            )
        if value_mode == "categorical" and scalar_value_embedding != "raw":
            raise ValueError(
                "HL-Gauss embeds scalar interpolants only; use "
                "scalar_value_embedding=raw for categorical flow."
            )
        if scalar_embed_bins < 2 or scalar_embed_sigma <= 0.0:
            raise ValueError(
                "scalar_embed_bins must be >= 2 and scalar_embed_sigma positive."
            )
        if time_embedding_type not in {"sinusoidal", "fourier", "raw"}:
            raise ValueError(
                "time_embedding_type must be 'sinusoidal', 'fourier', or 'raw'."
            )
        if query_hidden_dim < 1 or time_embed_dim < 2:
            raise ValueError(
                "query_hidden_dim must be positive and time_embed_dim must be >= 2."
            )
        if time_scale <= 0.0:
            raise ValueError("time_scale must be positive.")
        if not 0.0 < critic_target_tau <= 1.0:
            raise ValueError("critic_target_tau must be in (0, 1].")
        if temporal_ensemble_gain < 0.0 or tie_break_delta < 0.0:
            raise ValueError("Temporal/tie-break coefficients must be non-negative.")
        if not 0.0 <= structured_exploration_prob <= 1.0:
            raise ValueError("structured_exploration_prob must be in [0, 1].")
        if (
            structured_exploration_prob > 0.0
            and not 0 <= structured_exploration_level < levels
        ):
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
            "bc_policy",
            "policy_value",
        }:
            raise ValueError(
                "td_target_action_source must be one of "
                "{'critic', 'replay_next', 'bc_policy', 'policy_value'}."
            )
        critic_sequence_mode = str(critic_sequence_mode).lower()
        if critic_sequence_mode not in {"full", "effective_k0"}:
            raise ValueError(
                "critic_sequence_mode must be one of {'full', 'effective_k0'}."
            )
        if not separate_bc_policy and td_target_action_source != "critic":
            raise ValueError(
                "td_target_action_source requires separate_bc_policy=true "
                "unless it is 'critic'."
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
        if distinct_policy_encoder and not separate_bc_policy:
            raise ValueError(
                "distinct_policy_encoder=true requires separate_bc_policy=true."
            )
        if freeze_bc_policy and not separate_bc_policy:
            raise ValueError(
                "freeze_bc_policy=true requires separate_bc_policy=true."
            )
        if freeze_bc_policy and not distinct_policy_encoder:
            raise ValueError(
                "freeze_bc_policy=true requires distinct_policy_encoder=true "
                "so critic updates cannot change BC visual features."
            )
        if bc_policy_mode not in {"behavior_logits", "legacy_c51"}:
            raise ValueError(
                "bc_policy_mode must be behavior_logits or legacy_c51."
            )
        if bc_policy_mode == "legacy_c51" and not freeze_bc_policy:
            raise ValueError(
                "bc_policy_mode=legacy_c51 is an imported deployment policy "
                "and requires freeze_bc_policy=true."
            )
        if mc_return_weight < 0.0:
            raise ValueError("mc_return_weight must be non-negative.")
        if mc_return_weight > 0.0 and not separate_bc_policy:
            raise ValueError(
                "mc_return_weight > 0 requires separate_bc_policy=true."
            )
        if mc_return_stop_gradient_encoder:
            raise ValueError(
                "CQN-Flow does not yet support MC-only encoder detachment; "
                "set mc_return_stop_gradient_encoder=false."
            )
        if mc_return_value_only:
            raise ValueError(
                "mc_return_value_only is specific to categorical dueling "
                "logits; set it false for CQN-Flow."
            )
        if demo_batch_size is not None and demo_batch_size < 0:
            raise ValueError("demo_batch_size must be non-negative or None.")
        if min(advantage_c51_lambda, advantage_q_lambda) < 0.0:
            raise ValueError(
                "Hybrid advantage loss coefficients must be non-negative."
            )
        if causal_branch_weight < 0.0:
            raise ValueError("causal_branch_weight must be non-negative.")
        if causal_branch_delta_weight < 0.0:
            raise ValueError(
                "causal_branch_delta_weight must be non-negative."
            )
        if causal_branch_temperature <= 0.0:
            raise ValueError("causal_branch_temperature must be positive.")
        if causal_branch_batch_size < 2:
            raise ValueError("causal_branch_batch_size must be at least 2.")
        if (
            causal_branch_weight > 0.0
            and not 0 <= causal_branch_level < levels
        ):
            raise ValueError(
                "causal_branch_level must be in [0, levels)."
            )
        if causal_branch_weight > 0.0 and causal_branch_cache is None:
            raise ValueError(
                "causal_branch_weight > 0 requires causal_branch_cache."
            )
        if policy_value_beta is not None and policy_value_beta < 0.0:
            raise ValueError("policy_value_beta must be non-negative or null.")
        if critic_architecture == "flow_v_direct_a":
            if td_target_action_source == "policy_value":
                raise ValueError(
                    "policy_value TD targets currently require "
                    "critic_architecture=flow_q."
                )
            if value_mode != "scalar":
                raise ValueError(
                    "flow_v_direct_a requires value_mode=scalar."
                )
            if not separate_bc_policy:
                raise ValueError(
                    "flow_v_direct_a requires separate_bc_policy=true."
                )
            if mc_return_weight <= 0.0:
                raise ValueError(
                    "flow_v_direct_a requires mc_return_weight > 0 so Flow-V "
                    "has a completed-return state baseline target."
                )
            if advantage_c51_lambda == 0.0 and advantage_q_lambda == 0.0:
                raise ValueError(
                    "flow_v_direct_a requires at least one advantage loss."
                )
            if dcfm_lambda != 0.0 or pcbf_loss_coeff != 0.0:
                raise ValueError(
                    "flow_v_direct_a currently supports BCFM only; disable "
                    "DCFM and PCBF."
                )
        elif causal_branch_weight > 0.0:
            raise ValueError(
                "causal branch supervision currently requires "
                "critic_architecture=flow_v_direct_a."
            )
        if (
            policy_value_beta is not None
            and not (
                critic_architecture == "flow_v_direct_a"
                or flow_distill_action_readout
                or flow_q_action_readout
            )
        ):
            raise ValueError(
                "policy_value_beta requires flow_v_direct_a, "
                "flow_distill_action_readout=true, or "
                "flow_q_action_readout=true."
            )
        if policy_value_beta is not None and not separate_bc_policy:
            raise ValueError(
                "policy_value_beta requires separate_bc_policy=true."
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
        self.demo_batch_size = (
            None if demo_batch_size in {None, 0} else int(demo_batch_size)
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
        self.structured_exploration_prob = float(
            structured_exploration_prob
        )
        self.structured_exploration_level = min(
            max(int(structured_exploration_level), 0),
            int(levels) - 1,
        )
        self.structured_exploration_horizon = int(
            structured_exploration_horizon
        )
        self.separate_bc_policy = bool(separate_bc_policy)
        self.bc_policy_stop_gradient = bool(bc_policy_stop_gradient)
        self.distinct_policy_encoder = bool(distinct_policy_encoder)
        self.td_target_action_source = td_target_action_source
        self.td_target_policy_value_beta = (
            None
            if td_target_policy_value_beta is None
            else float(td_target_policy_value_beta)
        )
        self.critic_sequence_mode = critic_sequence_mode
        self.mc_return_weight = float(mc_return_weight)
        self.mc_return_stop_gradient_encoder = bool(
            mc_return_stop_gradient_encoder
        )
        self.mc_return_value_only = bool(mc_return_value_only)
        self._seed = int(seed)
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

        self.value_mode: ValueMode = value_mode
        self.value_dim = self.atoms if value_mode == "categorical" else 1
        self.num_flow_steps = int(num_flow_steps)
        self.num_flow_samples = int(num_flow_samples)
        self.num_target_flow_samples = int(num_target_flow_samples)
        self.num_action_flow_samples = int(num_action_flow_samples)
        self.flow_source_type: FlowSourceType = flow_source_type
        self.flow_source_std = float(flow_source_std)
        self.flow_source_min = float(
            0.1 * v_min if flow_source_min is None else flow_source_min
        )
        self.flow_source_max = float(
            0.1 * v_max if flow_source_max is None else flow_source_max
        )
        if self.flow_source_max <= self.flow_source_min:
            raise ValueError(
                "Derived uniform source bounds are invalid; set explicit "
                "flow_source_min/max."
            )
        self.antithetic_flow_sources = bool(antithetic_flow_sources)
        self.fixed_action_flow_sources = bool(fixed_action_flow_sources)
        self.action_flow_quantile_grid = bool(action_flow_quantile_grid)
        self.flow_iqn_quantile_coupling = bool(
            flow_iqn_quantile_coupling
        )
        self.quantile_endpoint_lambda = float(
            quantile_endpoint_lambda
        )
        self.quantile_huber_kappa = float(quantile_huber_kappa)
        self.return_sample_aggregation: ReturnSampleAggregation = (
            return_sample_aggregation
        )
        self.return_sample_temperature = float(return_sample_temperature)
        self.return_sample_truncate_top = int(
            return_sample_truncate_top
        )
        self.flow_q_action_readout = bool(flow_q_action_readout)
        self.atom_ce_lambda = float(atom_ce_lambda)
        self.bcfm_lambda = float(bcfm_lambda)
        self.dcfm_lambda = float(dcfm_lambda)
        self.evor_td_lambda = float(evor_td_lambda)
        self.confidence_weight_temp = (
            None
            if confidence_weight_temp is None
            else float(confidence_weight_temp)
        )
        self.pcbf_loss_coeff = float(pcbf_loss_coeff)
        self.pcbf_lambda = float(pcbf_lambda)
        self.endpoint_q_lambda = float(endpoint_q_lambda)
        self.source_consistency_lambda = float(source_consistency_lambda)
        self.flow_distill_lambda = float(flow_distill_lambda)
        self.flow_distill_action_readout = bool(
            flow_distill_action_readout
        )
        self.demo_flow_steps = (
            self.num_flow_steps
            if demo_flow_steps is None
            else int(demo_flow_steps)
        )
        self.demo_fosd = bool(demo_fosd)
        self.query_hidden_dim = int(query_hidden_dim)
        self.time_embedding_type: TimeEmbeddingType = time_embedding_type
        self.time_embed_dim = int(time_embed_dim)
        self.time_scale = float(time_scale)
        self.clip_scalar_targets = bool(clip_scalar_targets)
        self.clip_flow_trajectory = bool(clip_flow_trajectory)
        self.scalar_value_embedding: ScalarValueEmbedding = scalar_value_embedding
        self.scalar_embed_bins = int(scalar_embed_bins)
        self.scalar_embed_sigma = float(scalar_embed_sigma)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.critic_architecture: CriticArchitecture = critic_architecture
        self.hybrid_flow_v_direct_a = (
            self.critic_architecture == "flow_v_direct_a"
        )
        self.advantage_c51_lambda = float(advantage_c51_lambda)
        self.advantage_q_lambda = float(advantage_q_lambda)
        self.causal_branch_cache = causal_branch_cache
        self.causal_branch_weight = float(causal_branch_weight)
        self.causal_branch_delta_weight = float(
            causal_branch_delta_weight
        )
        self.causal_branch_temperature = float(
            causal_branch_temperature
        )
        self.causal_branch_batch_size = int(causal_branch_batch_size)
        self.causal_branch_level = int(causal_branch_level)
        self.policy_value_beta = (
            None if policy_value_beta is None else float(policy_value_beta)
        )
        self.freeze_bc_policy = bool(freeze_bc_policy)
        self.bc_policy_mode = bc_policy_mode

        input_dim = self._setup_rl_features(model, seed=seed)
        self._load_causal_branch_cache(input_dim)
        self.action_low, self.action_high = self._action_bounds()
        self._step_action_low = jnp.asarray(action_space.low[0], dtype=jnp.float32)
        self._step_action_high = jnp.asarray(action_space.high[0], dtype=jnp.float32)
        self.support = jnp.linspace(v_min, v_max, atoms, dtype=jnp.float32)
        self.critic_model = C2FSequenceFlowCritic(
            hidden_dims=model.hidden_dims,
            query_hidden_dim=self.query_hidden_dim,
            time_embed_dim=self.time_embed_dim,
            time_embedding_type=self.time_embedding_type,
            time_scale=self.time_scale,
            action_sequence=self.action_sequence,
            action_dim=self.action_dim,
            levels=self.levels,
            bins=self.bins,
            value_dim=self.value_dim,
            scalar_value_embedding=self.scalar_value_embedding,
            scalar_embed_bins=self.scalar_embed_bins,
            scalar_embed_sigma=self.scalar_embed_sigma,
            value_min=self.v_min,
            value_max=self.v_max,
            low_dim_size=(self.low_dim_size if self.use_pixels else 0),
            gru_layers=self.gru_layers,
            activation_name=model.activation,
            use_dueling=bool(use_dueling),
            value_only=self.hybrid_flow_v_direct_a,
            quantile_conditioning=self.flow_iqn_quantile_coupling,
        )
        dummy_features = jnp.zeros((1, input_dim), dtype=jnp.float32)
        dummy_level = jnp.zeros((1, self.levels), dtype=jnp.float32)
        dummy_interval = jnp.zeros(
            (1, self.action_sequence, self.action_dim), dtype=jnp.float32
        )
        dummy_bins = jnp.broadcast_to(
            jnp.arange(self.bins, dtype=jnp.int32)[None, None, None, :],
            (1, self.action_sequence, self.action_dim, self.bins),
        )
        dummy_values = jnp.zeros(
            (
                1,
                1,
                self.action_sequence,
                self.action_dim,
                self.bins,
                self.value_dim,
            ),
            dtype=jnp.float32,
        )
        critic_params = self.critic_model.init(
            self.rng_key,
            dummy_features,
            dummy_level,
            dummy_interval,
            dummy_interval,
            dummy_bins,
            dummy_bins.astype(jnp.float32),
            dummy_values,
            jnp.ones((1, 1), dtype=jnp.float32),
            (
                jnp.zeros_like(dummy_values[..., :1])
                if self.flow_iqn_quantile_coupling
                else None
            ),
        )
        self.params = {"critic": critic_params}
        advantage_params = None
        if self.hybrid_flow_v_direct_a:
            # A direct categorical residual head handles all within-state bin
            # ranking. Flow-V is candidate-independent and supplies only the
            # state/prefix baseline.
            self.advantage_model = C2FSequenceDistributionalCritic(
                hidden_dims=model.hidden_dims,
                action_sequence=self.action_sequence,
                action_dim=self.action_dim,
                levels=self.levels,
                bins=self.bins,
                atoms=self.atoms,
                low_dim_size=(self.low_dim_size if self.use_pixels else 0),
                gru_layers=self.gru_layers,
                activation_name=model.activation,
                use_dueling=False,
            )
            self.rng_key, advantage_key = jax.random.split(self.rng_key)
            advantage_params = self.advantage_model.init(
                advantage_key,
                dummy_features,
                dummy_level,
                dummy_interval,
            )
            self.params["advantage"] = advantage_params
        if self.flow_distill_lambda > 0.0:
            # FLOQ keeps a cheap scalar critic beside the velocity field.  It
            # is conditioned on the same state/image features, C2F level, zoom
            # prefix, sequence position, action dimension, and candidate bin.
            # Unlike the flow field, action selection needs only one head call
            # per level and no source sampling or Euler integration.
            self.flow_distill_readout_model = (
                C2FSequenceDistributionalCritic(
                    hidden_dims=model.hidden_dims,
                    action_sequence=self.action_sequence,
                    action_dim=self.action_dim,
                    levels=self.levels,
                    bins=self.bins,
                    atoms=1,
                    low_dim_size=(
                        self.low_dim_size if self.use_pixels else 0
                    ),
                    gru_layers=self.gru_layers,
                    activation_name=model.activation,
                    use_dueling=bool(use_dueling),
                )
            )
            self.rng_key, readout_key = jax.random.split(self.rng_key)
            self.params["flow_distill_readout"] = (
                self.flow_distill_readout_model.init(
                    readout_key,
                    dummy_features,
                    dummy_level,
                    dummy_interval,
                )
            )
        if self.separate_bc_policy:
            policy_atoms = (
                self.atoms if self.bc_policy_mode == "legacy_c51" else 1
            )
            self.policy_model = C2FSequenceDistributionalCritic(
                hidden_dims=model.hidden_dims,
                action_sequence=self.action_sequence,
                action_dim=self.action_dim,
                levels=self.levels,
                bins=self.bins,
                atoms=policy_atoms,
                low_dim_size=(self.low_dim_size if self.use_pixels else 0),
                gru_layers=self.gru_layers,
                activation_name=model.activation,
                use_dueling=(
                    bool(use_dueling)
                    if self.bc_policy_mode == "legacy_c51"
                    else False
                ),
            )
            self.rng_key, policy_key = jax.random.split(self.rng_key)
            self.params["policy"] = self.policy_model.init(
                policy_key,
                dummy_features,
                dummy_level,
                dummy_interval,
            )
        if self._trainable_encoder:
            self.params["encoder"] = self._encoder_params
            if self.distinct_policy_encoder:
                self.params["policy_encoder"] = jax.tree.map(
                    lambda value: jnp.array(value),
                    self._encoder_params,
                )
        if self.hybrid_flow_v_direct_a:
            self.target_critic_params = {
                "critic": critic_params,
                "advantage": advantage_params,
            }
        else:
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

    @property
    def _flat_action_dim(self) -> int:
        return self.action_sequence * self.action_dim

    def _policy_bin_scores(
        self,
        policy_params,
        features: jax.Array,
        level_one_hot: jax.Array,
        midpoint: jax.Array,
    ) -> jax.Array:
        """Return behavior scores for atoms=1 BC or imported legacy C51."""

        outputs = self.policy_model.apply(
            policy_params,
            features,
            level_one_hot,
            midpoint,
        )
        if self.bc_policy_mode == "legacy_c51":
            return expected_q(jax.nn.softmax(outputs, axis=-1), self.support)
        return outputs[..., 0]

    def _policy_logits_per_level(self, policy_params, features, action):
        """Return behavior-bin scores and encoded bins along a zoom path."""

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
        scores_per_level = []
        for level in range(self.levels):
            one_hot = jnp.broadcast_to(
                jax.nn.one_hot(level, self.levels, dtype=jnp.float32),
                (features.shape[0], self.levels),
            )
            scores = self._policy_bin_scores(
                policy_params,
                features,
                one_hot,
                (0.5 * (low + high)).reshape(
                    (
                        features.shape[0],
                        self.action_sequence,
                        self.action_dim,
                    )
                ),
            )
            scores_per_level.append(
                scores.reshape(
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
        return jnp.stack(scores_per_level, axis=1), discrete_action

    def _policy_action(self, policy_params, features, key=None):
        """Autoregress with a BC-logit or exactly imported legacy C51 policy."""

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
            scores = self._policy_bin_scores(
                policy_params,
                features,
                one_hot,
                (0.5 * (low + high)).reshape(
                    (batch_size, self.action_sequence, self.action_dim)
                ),
            )
            index = jnp.argmax(scores, axis=-1)
            level_key = level_keys[level]
            if level_key is not None:
                random_index = jax.random.randint(
                    level_key,
                    index.shape,
                    minval=0,
                    maxval=self.bins,
                )
                score_span = scores.max(axis=-1) - scores.min(axis=-1)
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

    def _sequence_training_slice(self, values, *, sequence_axis: int):
        """Restrict critic supervision to the actually executed k=0 token."""

        if self.critic_sequence_mode == "full":
            return values
        index = [slice(None)] * values.ndim
        index[int(sequence_axis)] = slice(0, 1)
        return values[tuple(index)]

    def _level_condition(
        self,
        low: jax.Array,
        high: jax.Array,
        level: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        batch_size = low.shape[0]
        midpoint = (0.5 * (low + high)).reshape(
            (batch_size, self.action_sequence, self.action_dim)
        )
        half_width = (0.5 * (high - low)).reshape(
            (batch_size, self.action_sequence, self.action_dim)
        )
        level_one_hot = jnp.broadcast_to(
            jax.nn.one_hot(level, self.levels, dtype=jnp.float32),
            (batch_size, self.levels),
        )
        candidate_bins = jnp.broadcast_to(
            jnp.arange(self.bins, dtype=jnp.int32)[None, None, None, :],
            (batch_size, self.action_sequence, self.action_dim, self.bins),
        )
        low_chunk = low.reshape((batch_size, self.action_sequence, self.action_dim))
        width = (high - low).reshape(
            (batch_size, self.action_sequence, self.action_dim)
        ) / float(self.bins)
        candidate_centers = low_chunk[..., None] + width[..., None] * (
            candidate_bins.astype(jnp.float32) + 0.5
        )
        return (
            level_one_hot,
            midpoint,
            half_width,
            candidate_bins,
            candidate_centers,
        )

    def _flow_source(
        self,
        key: jax.Array,
        batch_size: int,
        query_bins: int | None = None,
        num_samples: int | None = None,
    ) -> jax.Array:
        # Common random numbers across bins make within-level argmax comparisons
        # substantially less noisy than independent per-bin sources.
        query_bins = self.bins if query_bins is None else int(query_bins)
        num_samples = (
            self.num_flow_samples if num_samples is None else int(num_samples)
        )
        sampled_count = (
            (num_samples + 1) // 2
            if self.antithetic_flow_sources
            else num_samples
        )
        sampled_shape = (
            batch_size,
            sampled_count,
            self.action_sequence,
            self.action_dim,
            1,
            self.value_dim,
        )
        if self.flow_source_type == "gaussian":
            sampled = self.flow_source_std * jax.random.normal(
                key, sampled_shape, dtype=jnp.float32
            )
            mirrored = -sampled[:, : num_samples - sampled_count]
        else:
            sampled = jax.random.uniform(
                key,
                sampled_shape,
                minval=self.flow_source_min,
                maxval=self.flow_source_max,
                dtype=jnp.float32,
            )
            mirrored = (
                self.flow_source_min
                + self.flow_source_max
                - sampled[:, : num_samples - sampled_count]
            )
        source = jnp.concatenate([sampled, mirrored], axis=1)
        if self.value_mode == "categorical":
            source = source - source.mean(axis=-1, keepdims=True)
        return jnp.broadcast_to(
            source,
            (
                batch_size,
                num_samples,
                self.action_sequence,
                self.action_dim,
                query_bins,
                self.value_dim,
            ),
        )

    def _flow_source_quantiles(self, source: jax.Array) -> jax.Array | None:
        if not self.flow_iqn_quantile_coupling:
            return None
        if self.flow_source_type != "uniform":
            raise ValueError("FlowIQN source quantiles require uniform noise.")
        quantiles = (
            jnp.asarray(source) - float(self.flow_source_min)
        ) / float(self.flow_source_max - self.flow_source_min)
        return jnp.clip(quantiles[..., :1], 0.0, 1.0)

    def _action_flow_source(
        self,
        key: jax.Array,
        batch_size: int,
        query_bins: int,
        *,
        num_samples: int | None = None,
    ) -> jax.Array:
        """Return the source bank used only for action ranking.

        FlowIQN scalarizes its learned return distribution on the deterministic
        midpoint grid ``tau_k = (k - 0.5) / K``.  Keeping this separate from
        :meth:`_flow_source` is important: Bellman targets and current training
        paths still require fresh random quantiles.
        """

        num_samples = (
            self.num_action_flow_samples
            if num_samples is None
            else int(num_samples)
        )
        if not self.action_flow_quantile_grid:
            return self._flow_source(
                key,
                batch_size,
                query_bins,
                num_samples=num_samples,
            )
        quantiles = (
            jnp.arange(num_samples, dtype=jnp.float32) + 0.5
        ) / float(num_samples)
        source = self.flow_source_min + quantiles * (
            self.flow_source_max - self.flow_source_min
        )
        source = source.reshape((1, num_samples, 1, 1, 1, 1))
        return jnp.broadcast_to(
            source,
            (
                batch_size,
                num_samples,
                self.action_sequence,
                self.action_dim,
                query_bins,
                self.value_dim,
            ),
        )

    def _integrate_level(
        self,
        critic_params,
        features: jax.Array,
        condition,
        key: jax.Array,
        *,
        source: jax.Array | None = None,
        num_samples: int | None = None,
        num_flow_steps: int | None = None,
        end_tau: jax.Array | float = 0.0,
        action_ranking: bool = False,
    ) -> jax.Array:
        level_one_hot, midpoint, half_width, candidate_bins, centers = condition
        if source is None:
            source_fn = (
                self._action_flow_source
                if action_ranking
                else self._flow_source
            )
            source = source_fn(
                key,
                features.shape[0],
                candidate_bins.shape[-1],
                num_samples=(
                    self.num_action_flow_samples
                    if num_samples is None
                    else num_samples
                ),
            )
        source_quantiles = self._flow_source_quantiles(source)

        def velocity_fn(values, tau):
            velocity = self.critic_model.apply(
                critic_params,
                features,
                level_one_hot,
                midpoint,
                half_width,
                candidate_bins,
                centers,
                values,
                tau,
                source_quantiles,
            )
            if self.value_mode == "categorical":
                velocity = velocity - velocity.mean(axis=-1, keepdims=True)
            return velocity

        endpoint = integrate_value_flow(
            velocity_fn,
            source,
            num_flow_steps=(
                self.num_flow_steps
                if num_flow_steps is None
                else int(num_flow_steps)
            ),
            end_tau=end_tau,
            clip_min=(
                self.v_min
                if self.value_mode != "categorical"
                and self.clip_flow_trajectory
                else None
            ),
            clip_max=(
                self.v_max
                if self.value_mode != "categorical"
                and self.clip_flow_trajectory
                else None
            ),
        )
        if self.value_mode == "categorical":
            endpoint = endpoint - endpoint.mean(axis=-1, keepdims=True)
        return endpoint

    def _integrate_level_trajectory(
        self,
        critic_params,
        features: jax.Array,
        condition,
        *,
        source: jax.Array,
        num_flow_steps: int | None = None,
        end_tau: jax.Array | float = 0.0,
    ) -> jax.Array:
        """Read-only counterpart of ``_integrate_level`` retaining its path."""

        (
            level_one_hot,
            midpoint,
            half_width,
            candidate_bins,
            centers,
        ) = condition
        source_quantiles = self._flow_source_quantiles(source)

        def velocity_fn(values, tau):
            velocity = self.critic_model.apply(
                critic_params,
                features,
                level_one_hot,
                midpoint,
                half_width,
                candidate_bins,
                centers,
                values,
                tau,
                source_quantiles,
            )
            if self.value_mode == "categorical":
                velocity = velocity - velocity.mean(axis=-1, keepdims=True)
            return velocity

        return integrate_value_flow_trajectory(
            velocity_fn,
            source,
            num_flow_steps=(
                self.num_flow_steps
                if num_flow_steps is None
                else int(num_flow_steps)
            ),
            end_tau=end_tau,
            clip_min=(
                self.v_min
                if self.value_mode != "categorical"
                and self.clip_flow_trajectory
                else None
            ),
            clip_max=(
                self.v_max
                if self.value_mode != "categorical"
                and self.clip_flow_trajectory
                else None
            ),
        )

    def _value_flow_confidence_weights(
        self,
        target_critic_params,
        features: jax.Array,
        actions: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Estimate official Value-Flows return std and confidence weights.

        A single Gaussian source is transported through the EMA critic for
        each replay-selected C2F condition.  The source JVP is integrated in
        parallel with the return, then averaged over the trained sequence,
        action dimensions, and C2F levels to produce one transition weight.
        """

        if self.confidence_weight_temp is None:
            ones = jnp.ones((features.shape[0],), dtype=features.dtype)
            zeros = jnp.zeros_like(ones)
            return ones, zeros
        batch_size = features.shape[0]
        flat_action = jnp.asarray(actions, dtype=jnp.float32).reshape(
            (batch_size, self._flat_action_dim)
        )
        discrete_action = encode_action(
            flat_action,
            self.action_low,
            self.action_high,
            self.levels,
            self.bins,
        ).reshape(
            (batch_size, self.levels, self.action_sequence, self.action_dim)
        )
        low = jnp.broadcast_to(self.action_low, flat_action.shape)
        high = jnp.broadcast_to(self.action_high, flat_action.shape)
        source_keys = list(jax.random.split(key, self.levels))
        per_level_stds = []
        stopped_features = jax.lax.stop_gradient(features)
        for level in range(self.levels):
            index = discrete_action[:, level]
            condition = self._select_condition(
                self._level_condition(low, high, level),
                index,
            )
            (
                level_one_hot,
                midpoint,
                half_width,
                candidate_bins,
                centers,
            ) = condition
            source = self._flow_source(
                source_keys[level],
                batch_size,
                1,
                num_samples=1,
            )

            def velocity_fn(values, tau):
                return self.critic_model.apply(
                    target_critic_params,
                    stopped_features,
                    level_one_hot,
                    midpoint,
                    half_width,
                    candidate_bins,
                    centers,
                    values,
                    tau,
                    None,
                )

            _, source_jvp = integrate_value_flow_with_source_jvp(
                velocity_fn,
                source,
                source_tangent=(
                    jnp.ones_like(source) * float(self.flow_source_std)
                ),
                num_flow_steps=self.num_flow_steps,
                clip_min=(
                    self.v_min if self.clip_flow_trajectory else None
                ),
                clip_max=(
                    self.v_max if self.clip_flow_trajectory else None
                ),
            )
            source_jvp = self._sequence_training_slice(
                source_jvp,
                sequence_axis=2,
            )
            per_level_stds.append(
                jnp.abs(source_jvp[..., 0]).mean(axis=(1, 2, 3, 4))
            )
            low, high = zoom_in(
                low,
                high,
                index.reshape((batch_size, self._flat_action_dim)),
                self.bins,
                self.action_low,
                self.action_high,
            )
        return_std = jnp.stack(per_level_stds, axis=1).mean(axis=1)
        safe_std = jnp.maximum(return_std, 1e-8)
        weights = (
            jax.nn.sigmoid(
                -float(self.confidence_weight_temp) / safe_std
            )
            + 0.5
        )
        return (
            jax.lax.stop_gradient(weights),
            jax.lax.stop_gradient(return_std),
        )

    @staticmethod
    def _select_condition(condition, index: jax.Array):
        """Gather one query per sequence position and action dimension."""

        level_one_hot, midpoint, half_width, candidate_bins, centers = condition
        selected_bins = jnp.take_along_axis(
            candidate_bins,
            index[..., None],
            axis=-1,
        )
        selected_centers = jnp.take_along_axis(
            centers,
            index[..., None],
            axis=-1,
        )
        return (
            level_one_hot,
            midpoint,
            half_width,
            selected_bins,
            selected_centers,
        )

    def _endpoint_q_samples(self, endpoints: jax.Array) -> jax.Array:
        if self.value_mode == "categorical":
            return expected_q(
                flow_logits_to_probabilities(endpoints), self.support
            )
        return endpoints[..., 0]

    def _endpoint_q(self, endpoints: jax.Array) -> jax.Array:
        samples = self._endpoint_q_samples(endpoints)
        if self.value_mode == "return_sample":
            return aggregate_return_samples(
                samples,
                aggregation=self.return_sample_aggregation,
                temperature=self.return_sample_temperature,
                truncate_top=self.return_sample_truncate_top,
                sample_axis=1,
            )
        return samples.mean(axis=1)

    def _load_causal_branch_cache(self, input_dim: int) -> None:
        """Load the frozen-feature simulator branch buffer, if configured."""

        self._causal_branch_features = None
        self._causal_branch_actions = None
        self._causal_branch_returns = None
        self._causal_branch_dimensions = None
        self._causal_branch_informative_indices = None
        self._causal_branch_pair_left = None
        self._causal_branch_pair_right = None
        if self.causal_branch_weight <= 0.0:
            return

        cache_path = Path(str(self.causal_branch_cache)).expanduser().resolve()
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"causal branch cache does not exist: {cache_path}"
            )
        with np.load(cache_path, allow_pickle=False) as payload:
            required = {
                "train_features",
                "train_actions",
                "train_returns",
                "train_action_dimensions",
            }
            missing = required.difference(payload.files)
            if missing:
                raise ValueError(
                    f"causal branch cache is missing arrays: {sorted(missing)}"
                )
            features = np.asarray(payload["train_features"], np.float32)
            actions = np.asarray(payload["train_actions"], np.float32)
            returns = np.asarray(payload["train_returns"], np.float32)
            dimensions = np.asarray(
                payload["train_action_dimensions"], np.int32
            )

        expected_actions = (
            features.shape[0],
            self.bins,
            self.action_sequence,
            self.action_dim,
        )
        if features.ndim != 2 or features.shape[1] != int(input_dim):
            raise ValueError(
                "causal branch features must have shape "
                f"[N, {input_dim}], got {features.shape}"
            )
        if actions.shape != expected_actions:
            raise ValueError(
                "causal branch actions must have shape "
                f"{expected_actions}, got {actions.shape}"
            )
        if returns.shape != (features.shape[0], self.bins):
            raise ValueError(
                "causal branch returns must have shape "
                f"{(features.shape[0], self.bins)}, got {returns.shape}"
            )
        if dimensions.shape != (features.shape[0],):
            raise ValueError(
                "causal branch action dimensions must have one value per state"
            )
        if np.any(dimensions < 0) or np.any(dimensions >= self.action_dim):
            raise ValueError(
                "causal branch action dimensions are outside the primitive "
                "action space"
            )
        if not (
            np.all(np.isfinite(features))
            and np.all(np.isfinite(actions))
            and np.all(np.isfinite(returns))
        ):
            raise ValueError("causal branch cache contains non-finite values")
        informative = np.flatnonzero(np.ptp(returns, axis=1) > 1e-12)
        if informative.size == 0:
            raise ValueError(
                "causal branch cache contains no informative return contrast"
            )

        pair_left, pair_right = np.triu_indices(self.bins, k=1)
        self._causal_branch_features = jnp.asarray(features)
        self._causal_branch_actions = jnp.asarray(actions)
        self._causal_branch_returns = jnp.asarray(returns)
        self._causal_branch_dimensions = jnp.asarray(dimensions)
        self._causal_branch_informative_indices = jnp.asarray(
            informative, dtype=jnp.int32
        )
        self._causal_branch_pair_left = jnp.asarray(
            pair_left, dtype=jnp.int32
        )
        self._causal_branch_pair_right = jnp.asarray(
            pair_right, dtype=jnp.int32
        )

    def _causal_branch_objective(
        self,
        advantage_params,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Train direct-A from exact same-state sibling return contrasts."""

        if self.causal_branch_weight <= 0.0:
            zero = jnp.asarray(0.0, dtype=jnp.float32)
            return zero, zero, zero, zero

        informative_key, all_key = jax.random.split(key)
        informative_count = max(1, self.causal_branch_batch_size // 2)
        all_count = self.causal_branch_batch_size - informative_count
        informative_positions = jax.random.randint(
            informative_key,
            (informative_count,),
            minval=0,
            maxval=self._causal_branch_informative_indices.shape[0],
        )
        indices = self._causal_branch_informative_indices[
            informative_positions
        ]
        if all_count:
            all_indices = jax.random.randint(
                all_key,
                (all_count,),
                minval=0,
                maxval=self._causal_branch_features.shape[0],
            )
            indices = jnp.concatenate([indices, all_indices], axis=0)

        features = self._causal_branch_features[indices]
        actions = self._causal_branch_actions[indices]
        returns = self._causal_branch_returns[indices]
        dimensions = self._causal_branch_dimensions[indices]
        candidate_count = actions.shape[1]
        repeated_features = jnp.repeat(features, candidate_count, axis=0)
        flat_actions = actions.reshape(
            (
                self.causal_branch_batch_size * candidate_count,
                self.action_sequence,
                self.action_dim,
            )
        )
        _, _, chosen_advantage, _ = self._advantage_outputs_per_level(
            advantage_params,
            repeated_features,
            flat_actions,
        )
        chosen_advantage = chosen_advantage.reshape(
            (
                self.causal_branch_batch_size,
                candidate_count,
                self.levels,
                self.action_sequence,
                self.action_dim,
            )
        )
        level_values = chosen_advantage[
            :, :, self.causal_branch_level, 0, :
        ]
        gather_dimensions = jnp.broadcast_to(
            dimensions[:, None, None],
            (self.causal_branch_batch_size, candidate_count, 1),
        )
        q_values = jnp.take_along_axis(
            level_values,
            gather_dimensions,
            axis=-1,
        )[..., 0]

        left = self._causal_branch_pair_left
        right = self._causal_branch_pair_right
        return_delta = returns[:, left] - returns[:, right]
        labels = jnp.sign(return_delta)
        informative_mask = (
            jnp.abs(return_delta) > 1e-12
        ).astype(jnp.float32)
        q_delta = q_values[:, left] - q_values[:, right]
        pair_loss = jax.nn.softplus(
            -labels * q_delta / self.causal_branch_temperature
        )
        denominator = jnp.maximum(jnp.sum(informative_mask), 1.0)
        ranking_loss = (
            jnp.sum(pair_loss * informative_mask) / denominator
        )

        delta_error = q_delta - return_delta
        abs_delta_error = jnp.abs(delta_error)
        delta_loss = jnp.mean(
            jnp.where(
                abs_delta_error <= 1.0,
                0.5 * jnp.square(delta_error),
                abs_delta_error - 0.5,
            )
        )
        accuracy = jnp.sum(
            (jnp.sign(q_delta) == labels).astype(jnp.float32)
            * informative_mask
        ) / denominator
        q_span = jnp.mean(jnp.ptp(q_values, axis=1))
        return ranking_loss, delta_loss, accuracy, q_span

    def _advantage_level(
        self,
        advantage_params,
        features: jax.Array,
        level: int,
        low: jax.Array,
        high: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Return direct-C51 logits and zero-mean expected advantages."""

        level_one_hot = jnp.broadcast_to(
            jax.nn.one_hot(level, self.levels, dtype=jnp.float32),
            (features.shape[0], self.levels),
        )
        midpoint = (0.5 * (low + high)).reshape(
            (features.shape[0], self.action_sequence, self.action_dim)
        )
        logits = self.advantage_model.apply(
            advantage_params,
            features,
            level_one_hot,
            midpoint,
        )
        raw_expected = expected_q(jax.nn.softmax(logits, axis=-1), self.support)
        centered_advantage = raw_expected - raw_expected.mean(
            axis=-1,
            keepdims=True,
        )
        return logits, centered_advantage

    def _advantage_outputs_per_level(
        self,
        advantage_params,
        features: jax.Array,
        action: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Evaluate the direct residual head along a replayed C2F zoom path.

        Returns chosen logits, all logits, chosen centered expectations, and
        all centered expectations with level kept as an explicit axis.
        """

        batch_size = features.shape[0]
        flat_action = jnp.asarray(action, dtype=jnp.float32).reshape(
            (batch_size, self._flat_action_dim)
        )
        discrete_action = encode_action(
            flat_action,
            self.action_low,
            self.action_high,
            self.levels,
            self.bins,
        ).reshape(
            (batch_size, self.levels, self.action_sequence, self.action_dim)
        )
        low = jnp.broadcast_to(self.action_low, flat_action.shape)
        high = jnp.broadcast_to(self.action_high, flat_action.shape)
        chosen_logits_per_level = []
        all_logits_per_level = []
        chosen_advantage_per_level = []
        all_advantage_per_level = []
        for level in range(self.levels):
            logits, centered_advantage = self._advantage_level(
                advantage_params,
                features,
                level,
                low,
                high,
            )
            index = discrete_action[:, level]
            chosen_logits = jnp.take_along_axis(
                logits,
                index[..., None, None],
                axis=-2,
            )[..., 0, :]
            chosen_advantage = jnp.take_along_axis(
                centered_advantage,
                index[..., None],
                axis=-1,
            )[..., 0]
            chosen_logits_per_level.append(chosen_logits)
            all_logits_per_level.append(logits)
            chosen_advantage_per_level.append(chosen_advantage)
            all_advantage_per_level.append(centered_advantage)
            low, high = zoom_in(
                low,
                high,
                index.reshape((batch_size, self._flat_action_dim)),
                self.bins,
                self.action_low,
                self.action_high,
            )
        return (
            jnp.stack(chosen_logits_per_level, axis=1),
            jnp.stack(all_logits_per_level, axis=1),
            jnp.stack(chosen_advantage_per_level, axis=1),
            jnp.stack(all_advantage_per_level, axis=1),
        )

    def _flow_distill_level(
        self,
        readout_params,
        features: jax.Array,
        level: int,
        low: jax.Array,
        high: jax.Array,
    ) -> jax.Array:
        """Return the FLOQ-distilled scalar value of every candidate bin."""

        if self.flow_distill_lambda <= 0.0:
            raise ValueError("The flow-distilled readout is not enabled.")
        level_one_hot = jnp.broadcast_to(
            jax.nn.one_hot(level, self.levels, dtype=jnp.float32),
            (features.shape[0], self.levels),
        )
        midpoint = (0.5 * (low + high)).reshape(
            (features.shape[0], self.action_sequence, self.action_dim)
        )
        return self.flow_distill_readout_model.apply(
            readout_params,
            features,
            level_one_hot,
            midpoint,
        )[..., 0]

    def _flow_distill_outputs_per_level(
        self,
        readout_params,
        features: jax.Array,
        action: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Evaluate chosen and all-bin scalar readouts on a replay zoom path."""

        batch_size = features.shape[0]
        flat_action = jnp.asarray(action, dtype=jnp.float32).reshape(
            (batch_size, self._flat_action_dim)
        )
        discrete_action = encode_action(
            flat_action,
            self.action_low,
            self.action_high,
            self.levels,
            self.bins,
        ).reshape(
            (batch_size, self.levels, self.action_sequence, self.action_dim)
        )
        low = jnp.broadcast_to(self.action_low, flat_action.shape)
        high = jnp.broadcast_to(self.action_high, flat_action.shape)
        chosen_per_level = []
        all_per_level = []
        for level in range(self.levels):
            all_q = self._flow_distill_level(
                readout_params,
                features,
                level,
                low,
                high,
            )
            index = discrete_action[:, level]
            chosen_q = jnp.take_along_axis(
                all_q,
                index[..., None],
                axis=-1,
            )[..., 0]
            chosen_per_level.append(chosen_q)
            all_per_level.append(all_q)
            low, high = zoom_in(
                low,
                high,
                index.reshape((batch_size, self._flat_action_dim)),
                self.bins,
                self.action_low,
                self.action_high,
            )
        return (
            jnp.stack(chosen_per_level, axis=1),
            jnp.stack(all_per_level, axis=1),
        )

    def _flow_distill_greedy_action(
        self,
        readout_params,
        features: jax.Array,
        key: jax.Array | None = None,
    ):
        """Select all C2F bins with the cheap online distilled Q readout."""

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
            q_values = self._flow_distill_level(
                readout_params,
                features,
                level,
                low,
                high,
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

    def _flow_distill_policy_value_action(
        self,
        readout_params,
        value_features: jax.Array,
        policy_params,
        policy_features: jax.Array,
        key: jax.Array | None = None,
    ):
        """Combine normalized distilled Q with the independent BC log prior."""

        if self.policy_value_beta is None:
            raise ValueError(
                "_flow_distill_policy_value_action requires "
                "policy_value_beta"
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
            policy_logits = self.policy_model.apply(
                policy_params,
                policy_features,
                one_hot,
                midpoint,
            )[..., 0]
            q_values = self._flow_distill_level(
                readout_params,
                value_features,
                level,
                low,
                high,
            )
            centered_q = q_values - q_values.mean(axis=-1, keepdims=True)
            q_scale = jnp.sqrt(
                jnp.mean(jnp.square(centered_q), axis=-1, keepdims=True)
                + 1e-6
            )
            score = centered_q / q_scale + (
                self.policy_value_beta
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

    def _hybrid_greedy_action(
        self,
        flow_params,
        advantage_params,
        features: jax.Array,
        key: jax.Array | None = None,
    ):
        """Select C2F bins with Flow-V plus centered direct-C51 advantage."""

        batch_size = features.shape[0]
        if key is None:
            key = jax.random.PRNGKey(0)
        level_keys = list(jax.random.split(key, 2 * self.levels))
        low = jnp.broadcast_to(
            self.action_low, (batch_size, self._flat_action_dim)
        )
        high = jnp.broadcast_to(
            self.action_high, (batch_size, self._flat_action_dim)
        )
        selected = []
        for level in range(self.levels):
            flow_key = level_keys[2 * level]
            if self.fixed_action_flow_sources:
                flow_key = jax.random.fold_in(
                    jax.random.PRNGKey(self._seed + 1729), level
                )
            flow_endpoint = self._integrate_level(
                flow_params,
                features,
                self._level_condition(low, high, level),
                flow_key,
                action_ranking=True,
            )
            flow_v = self._endpoint_q(flow_endpoint)
            _, centered_advantage = self._advantage_level(
                advantage_params,
                features,
                level,
                low,
                high,
            )
            q_values = flow_v + centered_advantage
            index = jnp.argmax(q_values, axis=-1)
            random_index = jax.random.randint(
                level_keys[2 * level + 1],
                index.shape,
                minval=0,
                maxval=self.bins,
            )
            q_span = q_values.max(axis=-1) - q_values.min(axis=-1)
            index = jnp.where(q_span < self.tie_break_delta, random_index, index)
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

    def _hybrid_policy_value_action(
        self,
        advantage_params,
        value_features: jax.Array,
        policy_params,
        policy_features: jax.Array,
        key: jax.Array | None = None,
    ):
        """Choose C2F bins with normalized direct-A plus the BC log prior."""

        if self.policy_value_beta is None:
            raise ValueError(
                "_hybrid_policy_value_action requires policy_value_beta"
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
            policy_logits = self.policy_model.apply(
                policy_params,
                policy_features,
                one_hot,
                midpoint,
            )[..., 0]
            _, centered_advantage = self._advantage_level(
                advantage_params,
                value_features,
                level,
                low,
                high,
            )
            advantage_scale = jnp.sqrt(
                jnp.mean(jnp.square(centered_advantage), axis=-1, keepdims=True)
                + 1e-6
            )
            normalized_advantage = centered_advantage / advantage_scale
            score = normalized_advantage + (
                self.policy_value_beta
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

    def _supported_lcb_policy_plan(
        self,
        advantage_params_ensemble,
        value_features: jax.Array,
        policy_params,
        policy_features: jax.Array,
        baseline_plan: jax.Array,
        *,
        force_level: int,
        intervention_horizon: int,
        lcb_scale: float,
        min_lcb_margin: float,
        max_bc_logprob_drop: float,
    ) -> SupportedLCBPlanResult:
        """Score exact branch-oracle candidates and conservatively edit BC."""

        if not self.hybrid_flow_v_direct_a:
            raise ValueError(
                "supported LCB sidecars require flow_v_direct_a architecture."
            )
        if not self.separate_bc_policy:
            raise ValueError("supported LCB sidecars require a BC policy head.")
        if not 0 <= force_level < self.levels:
            raise ValueError("force_level must be in [0, levels).")

        baseline_plan = jnp.asarray(baseline_plan, dtype=jnp.float32)
        candidate_plans, _ = sibling_bin_candidate_plans(
            baseline_plan,
            jnp.asarray(self._step_action_low),
            jnp.asarray(self._step_action_high),
            bins=self.bins,
            force_level=force_level,
            intervention_horizon=intervention_horizon,
        )
        batch_size = baseline_plan.shape[0]
        candidate_count = self.action_dim * self.bins
        flat_candidates = candidate_plans.reshape(
            (
                batch_size * candidate_count,
                self.action_sequence,
                self.action_dim,
            )
        )
        repeated_value_features = jnp.repeat(
            value_features,
            candidate_count,
            axis=0,
        )
        repeated_policy_features = jnp.repeat(
            policy_features,
            candidate_count,
            axis=0,
        )

        policy_logits, encoded_bins = self._policy_logits_per_level(
            policy_params,
            repeated_policy_features,
            flat_candidates,
        )
        policy_selected_log_probability = jnp.take_along_axis(
            jax.nn.log_softmax(policy_logits, axis=-1),
            encoded_bins[..., None],
            axis=-1,
        )[..., 0]
        policy_selected_log_probability = (
            policy_selected_log_probability.reshape(
                (
                    batch_size,
                    self.action_dim,
                    self.bins,
                    self.levels,
                    self._flat_action_dim,
                )
            )[..., force_level, :]
        )
        dimension_index = jnp.broadcast_to(
            jnp.arange(self.action_dim)[None, :, None, None],
            (
                batch_size,
                self.action_dim,
                self.bins,
                1,
            ),
        )
        policy_candidate_scores = jnp.take_along_axis(
            policy_selected_log_probability,
            dimension_index,
            axis=-1,
        )[..., 0]

        def member_scores(advantage_params):
            _, _, chosen_advantage, _ = self._advantage_outputs_per_level(
                advantage_params,
                repeated_value_features,
                flat_candidates,
            )
            return chosen_advantage

        ensemble_scores = jax.vmap(member_scores)(
            advantage_params_ensemble
        ).reshape(
            (
                -1,
                batch_size,
                self.action_dim,
                self.bins,
                self.levels,
                self._flat_action_dim,
            )
        )[..., force_level, :]
        ensemble_dimension_index = jnp.broadcast_to(
            dimension_index[None],
            (
                ensemble_scores.shape[0],
                batch_size,
                self.action_dim,
                self.bins,
                1,
            ),
        )
        advantage_scores = jnp.take_along_axis(
            ensemble_scores,
            ensemble_dimension_index,
            axis=-1,
        )[..., 0]
        return select_single_supported_lcb_plan(
            baseline_plan,
            candidate_plans,
            policy_candidate_scores,
            advantage_scores,
            lcb_scale=lcb_scale,
            min_lcb_margin=min_lcb_margin,
            max_bc_logprob_drop=max_bc_logprob_drop,
        )

    def _greedy_action(
        self,
        critic_params,
        features,
        key=None,
        *,
        advantage_params=None,
    ):
        if self.hybrid_flow_v_direct_a:
            if advantage_params is None:
                if (
                    isinstance(critic_params, dict)
                    and "critic" in critic_params
                    and "advantage" in critic_params
                ):
                    advantage_params = critic_params["advantage"]
                    critic_params = critic_params["critic"]
                else:
                    raise ValueError(
                        "Hybrid greedy action requires direct advantage params."
                    )
            return self._hybrid_greedy_action(
                critic_params,
                advantage_params,
                features,
                key,
            )
        batch_size = features.shape[0]
        if key is None:
            key = jax.random.PRNGKey(0)
        level_keys = list(jax.random.split(key, 2 * self.levels))
        low = jnp.broadcast_to(
            self.action_low, (batch_size, self._flat_action_dim)
        )
        high = jnp.broadcast_to(
            self.action_high, (batch_size, self._flat_action_dim)
        )
        selected = []
        for level in range(self.levels):
            flow_key = level_keys[2 * level]
            if self.fixed_action_flow_sources:
                flow_key = jax.random.fold_in(
                    jax.random.PRNGKey(self._seed + 1729), level
                )
            endpoints = self._integrate_level(
                critic_params,
                features,
                self._level_condition(low, high, level),
                flow_key,
                action_ranking=True,
            )
            q_values = self._endpoint_q(endpoints)
            index = jnp.argmax(q_values, axis=-1)
            random_index = jax.random.randint(
                level_keys[2 * level + 1],
                index.shape,
                minval=0,
                maxval=self.bins,
            )
            q_span = q_values.max(axis=-1) - q_values.min(axis=-1)
            index = jnp.where(q_span < self.tie_break_delta, random_index, index)
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

    def _flow_q_policy_value_action(
        self,
        critic_params,
        value_features: jax.Array,
        policy_params,
        policy_features: jax.Array,
        key: jax.Array | None = None,
        *,
        policy_value_beta: float | None = None,
    ):
        """Combine normalized integrated flow-Q with an independent BC prior.

        For ``value_mode=return_sample``, ``_endpoint_q`` applies the configured
        mean or EVOR-style entropic aggregation before the bin scores are
        normalized.  All action dimensions, bins, and return sources remain
        parallel inside each C2F level.
        """

        resolved_beta = (
            self.policy_value_beta
            if policy_value_beta is None
            else float(policy_value_beta)
        )
        if resolved_beta is None:
            raise ValueError(
                "_flow_q_policy_value_action requires policy_value_beta"
            )
        batch_size = value_features.shape[0]
        if key is None:
            key = jax.random.PRNGKey(0)
        level_keys = list(jax.random.split(key, 2 * self.levels))
        low = jnp.broadcast_to(
            self.action_low,
            (batch_size, self._flat_action_dim),
        )
        high = jnp.broadcast_to(
            self.action_high,
            (batch_size, self._flat_action_dim),
        )
        selected = []
        for level in range(self.levels):
            one_hot = jnp.broadcast_to(
                jax.nn.one_hot(level, self.levels, dtype=jnp.float32),
                (batch_size, self.levels),
            )
            midpoint = (0.5 * (low + high)).reshape(
                (batch_size, self.action_sequence, self.action_dim)
            )
            policy_scores = self._policy_bin_scores(
                policy_params,
                policy_features,
                one_hot,
                midpoint,
            )
            flow_key = level_keys[2 * level]
            if self.fixed_action_flow_sources:
                flow_key = jax.random.fold_in(
                    jax.random.PRNGKey(self._seed + 1729), level
                )
            endpoints = self._integrate_level(
                critic_params,
                value_features,
                self._level_condition(low, high, level),
                flow_key,
                action_ranking=True,
            )
            q_values = self._endpoint_q(endpoints)
            centered_q = q_values - q_values.mean(axis=-1, keepdims=True)
            q_scale = jnp.sqrt(
                jnp.mean(jnp.square(centered_q), axis=-1, keepdims=True)
                + 1e-6
            )
            score = centered_q / q_scale + (
                resolved_beta
                * jax.nn.log_softmax(policy_scores, axis=-1)
            )
            index = jnp.argmax(score, axis=-1)
            random_index = jax.random.randint(
                level_keys[2 * level + 1],
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

    def _build_greedy_action_fn(self):
        """Build rollout inference with optional strict policy/value towers."""

        def action_fn(params, target_critic_params, obs_inputs, use_target, key):
            if self.flow_distill_action_readout:
                value_features = self._rl_features(
                    params.get("encoder", None),
                    obs_inputs,
                    stop_gradient=True,
                )
                if (
                    self.separate_bc_policy
                    and self.policy_value_beta is not None
                ):
                    policy_encoder_params = params.get("encoder", None)
                    if self.distinct_policy_encoder:
                        policy_encoder_params = params.get(
                            "policy_encoder", None
                        )
                    policy_features = self._rl_features(
                        policy_encoder_params,
                        obs_inputs,
                        stop_gradient=True,
                    )
                    return self._flow_distill_policy_value_action(
                        params["flow_distill_readout"],
                        value_features,
                        params["policy"],
                        policy_features,
                        key,
                    )[0]
                return self._flow_distill_greedy_action(
                    params["flow_distill_readout"],
                    value_features,
                    key,
                )[0]
            if self.flow_q_action_readout:
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
                if (
                    self.separate_bc_policy
                    and self.policy_value_beta is not None
                ):
                    policy_encoder_params = params.get("encoder", None)
                    if self.distinct_policy_encoder:
                        policy_encoder_params = params.get(
                            "policy_encoder", None
                        )
                    policy_features = self._rl_features(
                        policy_encoder_params,
                        obs_inputs,
                        stop_gradient=True,
                    )
                    return self._flow_q_policy_value_action(
                        critic_params,
                        value_features,
                        params["policy"],
                        policy_features,
                        key,
                    )[0]
                return self._greedy_action(
                    critic_params,
                    value_features,
                    key=key,
                )[0]
            if self.separate_bc_policy:
                policy_encoder_params = params.get("encoder", None)
                if self.distinct_policy_encoder:
                    policy_encoder_params = params.get("policy_encoder", None)
                policy_features = self._rl_features(
                    policy_encoder_params,
                    obs_inputs,
                    stop_gradient=True,
                )
                if (
                    self.hybrid_flow_v_direct_a
                    and self.policy_value_beta is not None
                ):
                    value_features = self._rl_features(
                        params.get("encoder", None),
                        obs_inputs,
                        stop_gradient=True,
                    )
                    advantage_params = jax.lax.cond(
                        use_target,
                        lambda _: target_critic_params["advantage"],
                        lambda _: params["advantage"],
                        operand=None,
                    )
                    return self._hybrid_policy_value_action(
                        advantage_params,
                        value_features,
                        params["policy"],
                        policy_features,
                        key,
                    )[0]
                return self._policy_action(
                    params["policy"],
                    policy_features,
                    key=key,
                )[0]

            features = self._rl_features(
                params.get("encoder", None),
                obs_inputs,
                stop_gradient=True,
            )
            if self.hybrid_flow_v_direct_a:
                flow_params = jax.lax.cond(
                    use_target,
                    lambda _: target_critic_params["critic"],
                    lambda _: params["critic"],
                    operand=None,
                )
                advantage_params = jax.lax.cond(
                    use_target,
                    lambda _: target_critic_params["advantage"],
                    lambda _: params["advantage"],
                    operand=None,
                )
                return self._hybrid_greedy_action(
                    flow_params,
                    advantage_params,
                    features,
                    key,
                )[0]
            critic_params = jax.lax.cond(
                use_target,
                lambda _: target_critic_params,
                lambda _: params["critic"],
                operand=None,
            )
            return self._greedy_action(critic_params, features, key=key)[0]

        return action_fn

    def _flow_utilization_probe_from_features(
        self,
        critic_params,
        features: jax.Array,
        action: jax.Array,
        key: jax.Array,
        *,
        num_source_samples: int,
        step_counts: tuple[int, ...],
    ) -> dict[str, jax.Array]:
        """Measure flow curvature and sensitivity to integration depth.

        Every step-count comparison reuses the exact same source bank and the
        exact same action-conditioned C2F intervals.  The action path is fixed
        externally rather than selected by the critic, so this probe cannot
        improve a checkpoint's task score or leak into checkpoint selection.
        """

        if self.value_mode not in {"scalar", "return_sample"}:
            raise ValueError(
                "flow-utilization diagnostics require scalar or return-sample "
                "value flows."
            )
        if self.hybrid_flow_v_direct_a:
            raise ValueError(
                "flow-utilization diagnostics reject flow-V/direct-A hybrids "
                "because direct-A would confound action rankings."
            )
        if self.num_flow_steps < 2:
            raise ValueError(
                "flow-utilization curvature requires at least two configured "
                "flow steps."
            )
        if num_source_samples < 2:
            raise ValueError("num_source_samples must be at least 2.")
        if not step_counts or any(int(count) < 1 for count in step_counts):
            raise ValueError("step_counts must contain positive integers.")

        features = jnp.asarray(features)
        if features.ndim != 2 or features.shape[0] < 1:
            raise ValueError("features must be a non-empty [B, F] array.")
        batch_size = features.shape[0]
        flat_action = jnp.asarray(action, dtype=jnp.float32).reshape(
            (batch_size, self._flat_action_dim)
        )
        discrete_action = encode_action(
            flat_action,
            self.action_low,
            self.action_high,
            self.levels,
            self.bins,
        ).reshape(
            (batch_size, self.levels, self.action_sequence, self.action_dim)
        )
        low = jnp.broadcast_to(self.action_low, flat_action.shape)
        high = jnp.broadcast_to(self.action_high, flat_action.shape)
        level_keys = jax.random.split(key, self.levels)

        trajectory_metrics = {
            name: []
            for name in (
                "curvature_abs_mean",
                "curvature_rms",
                "normalized_curvature_rms",
                "displacement_rms",
                "source_std_mean",
                "endpoint_std_mean",
                "source_contraction_ratio",
                "increment_variation_rms",
                "normalized_increment_variation",
            )
        }
        step_ranking_agreement = []
        step_q_rmse = []
        step_normalized_q_rmse = []
        configured_q_span = []

        for level in range(self.levels):
            condition = self._level_condition(low, high, level)
            source = self._flow_source(
                level_keys[level],
                batch_size,
                self.bins,
                num_samples=num_source_samples,
            )
            trajectory = self._integrate_level_trajectory(
                critic_params,
                features,
                condition,
                source=source,
            )
            level_metrics = scalar_flow_trajectory_diagnostics(trajectory)
            for name, values in trajectory_metrics.items():
                values.append(level_metrics[name])

            configured_q = self._endpoint_q(trajectory[-1])
            configured_rank = jnp.argmax(configured_q, axis=-1)
            q_span = (
                configured_q.max(axis=-1) - configured_q.min(axis=-1)
            ).mean()
            configured_q_span.append(q_span)
            level_ranking_agreement = []
            level_q_rmse = []
            level_normalized_q_rmse = []
            for step_count in step_counts:
                if int(step_count) == self.num_flow_steps:
                    endpoint = trajectory[-1]
                else:
                    endpoint = self._integrate_level(
                        critic_params,
                        features,
                        condition,
                        level_keys[level],
                        source=source,
                        num_samples=num_source_samples,
                        num_flow_steps=int(step_count),
                    )
                step_q = self._endpoint_q(endpoint)
                rmse = jnp.sqrt(jnp.mean(jnp.square(step_q - configured_q)))
                level_ranking_agreement.append(
                    jnp.mean(
                        jnp.argmax(step_q, axis=-1) == configured_rank,
                        dtype=jnp.float32,
                    )
                )
                level_q_rmse.append(rmse)
                level_normalized_q_rmse.append(rmse / (q_span + 1e-6))
            step_ranking_agreement.append(
                jnp.stack(level_ranking_agreement)
            )
            step_q_rmse.append(jnp.stack(level_q_rmse))
            step_normalized_q_rmse.append(
                jnp.stack(level_normalized_q_rmse)
            )

            index = discrete_action[:, level]
            low, high = zoom_in(
                low,
                high,
                index.reshape((batch_size, self._flat_action_dim)),
                self.bins,
                self.action_low,
                self.action_high,
            )

        stacked_trajectory_metrics = {
            f"per_level_{name}": jnp.stack(values)
            for name, values in trajectory_metrics.items()
        }
        per_level_step_ranking_agreement = jnp.stack(
            step_ranking_agreement
        )
        per_level_step_q_rmse = jnp.stack(step_q_rmse)
        per_level_step_normalized_q_rmse = jnp.stack(
            step_normalized_q_rmse
        )
        return {
            **stacked_trajectory_metrics,
            "configured_num_flow_steps": jnp.asarray(
                self.num_flow_steps, dtype=jnp.int32
            ),
            "step_counts": jnp.asarray(step_counts, dtype=jnp.int32),
            "condition_action": flat_action.reshape(
                (batch_size, self.action_sequence, self.action_dim)
            ),
            "per_level_configured_q_span": jnp.stack(configured_q_span),
            "per_level_step_ranking_agreement": (
                per_level_step_ranking_agreement
            ),
            "per_level_step_q_rmse": per_level_step_q_rmse,
            "per_level_step_normalized_q_rmse": (
                per_level_step_normalized_q_rmse
            ),
            "mean_normalized_curvature_rms": jnp.mean(
                stacked_trajectory_metrics[
                    "per_level_normalized_curvature_rms"
                ]
            ),
            "mean_source_contraction_ratio": jnp.mean(
                stacked_trajectory_metrics[
                    "per_level_source_contraction_ratio"
                ]
            ),
            "mean_step_ranking_agreement": jnp.mean(
                per_level_step_ranking_agreement, axis=0
            ),
            "mean_step_q_rmse": jnp.mean(per_level_step_q_rmse, axis=0),
            "mean_step_normalized_q_rmse": jnp.mean(
                per_level_step_normalized_q_rmse, axis=0
            ),
        }

    def flow_utilization_probe(
        self,
        observations: dict,
        actions: jax.Array | np.ndarray | None = None,
        *,
        num_source_samples: int = 8,
        step_counts: tuple[int, ...] = (1, 2, 4, 8),
        seed: int = 0,
        use_target_network: bool | None = None,
    ) -> dict[str, jax.Array]:
        """Run a read-only FLOQ flow-depth/collapse diagnostic.

        When ``actions`` is omitted, conditions come from the independent BC
        policy.  Explicit replay/demo actions may instead be supplied.  The
        method does not mutate the agent RNG or action history and its outputs
        must not be used to select checkpoints or policy-value mixing weights.
        """

        if self.value_mode not in {"scalar", "return_sample"}:
            raise ValueError(
                "flow-utilization diagnostics require scalar or return-sample "
                "value flows."
            )
        if self.hybrid_flow_v_direct_a:
            raise ValueError(
                "flow-utilization diagnostics reject flow-V/direct-A hybrids."
            )
        if self.num_flow_steps < 2:
            raise ValueError(
                "flow-utilization curvature requires at least two configured "
                "flow steps."
            )
        if num_source_samples < 2:
            raise ValueError("num_source_samples must be at least 2.")
        step_counts = tuple(int(count) for count in step_counts)
        if not step_counts or any(count < 1 for count in step_counts):
            raise ValueError("step_counts must contain positive integers.")
        if len(set(step_counts)) != len(step_counts):
            raise ValueError("step_counts must not contain duplicates.")

        obs_inputs = self._prepare_rl_obs_inputs(observations)
        features = self._rl_features(
            self.params.get("encoder", None),
            obs_inputs,
            stop_gradient=True,
        )
        if actions is None:
            if not self.separate_bc_policy:
                raise ValueError(
                    "actions=None requires an independent BC policy; pass "
                    "explicit replay/demo actions otherwise."
                )
            policy_encoder_params = self.params.get("encoder", None)
            if self.distinct_policy_encoder:
                policy_encoder_params = self.params.get(
                    "policy_encoder", None
                )
            policy_features = self._rl_features(
                policy_encoder_params,
                obs_inputs,
                stop_gradient=True,
            )
            actions = self._policy_action(
                self.params["policy"],
                policy_features,
                key=None,
            )[0]
        else:
            actions = jnp.asarray(actions, dtype=jnp.float32)
        expected_action_shape = (
            features.shape[0],
            self.action_sequence,
            self.action_dim,
        )
        if actions.shape != expected_action_shape:
            raise ValueError(
                "actions must have shape "
                f"{expected_action_shape}, got {actions.shape}."
            )

        if use_target_network is None:
            use_target_network = self.use_target_network_for_rollout
        critic_params = (
            self.target_critic_params
            if bool(use_target_network)
            else self.params["critic"]
        )
        key = jax.random.PRNGKey(int(seed))

        def probe_fn(params, encoded_features, condition_actions, probe_key):
            return self._flow_utilization_probe_from_features(
                params,
                encoded_features,
                condition_actions,
                probe_key,
                num_source_samples=num_source_samples,
                step_counts=step_counts,
            )

        if self._jit_enabled:
            probe_fn = jax.jit(probe_fn)
        metrics = probe_fn(critic_params, features, actions, key)
        return jax.block_until_ready(metrics)

    def _source_resampling_ranking_probe_from_features(
        self,
        critic_params,
        features: jax.Array,
        key: jax.Array,
        *,
        num_source_draws: int,
        num_action_flow_samples: int = 1,
    ) -> dict[str, jax.Array]:
        """Measure action-bin stability under independent source groups.

        Each observation is expanded into ``num_source_draws`` independent
        coarse-to-fine rollouts.  Each rollout averages
        ``num_action_flow_samples`` source endpoints before selecting a bin,
        exactly matching the deployed Monte Carlo readout for a requested
        ``R_action``.  Unlike :meth:`_greedy_action`, sources are always sampled
        from ``key`` and passed explicitly to the integrator;
        ``fixed_action_flow_sources`` therefore has no effect on this probe.
        Tie-breaking noise is intentionally omitted so every reported flip is
        attributable to the value-flow ranking itself.

        Later levels follow each draw's own zoom path.  Their source-Q spread
        consequently measures complete rollout instability (source variation
        plus any earlier path divergence), rather than a fixed-condition local
        sensitivity.
        """

        if num_source_draws < 2:
            raise ValueError("num_source_draws must be at least 2.")
        if num_action_flow_samples < 1:
            raise ValueError("num_action_flow_samples must be at least 1.")
        features = jnp.asarray(features)
        if features.ndim != 2 or features.shape[0] < 1:
            raise ValueError("features must be a non-empty [B, F] array.")
        flow_params = critic_params
        advantage_params = None
        if self.hybrid_flow_v_direct_a:
            flow_params = critic_params["critic"]
            advantage_params = critic_params["advantage"]

        batch_size = features.shape[0]
        expanded_features = jnp.broadcast_to(
            features[:, None, :],
            (batch_size, num_source_draws, features.shape[-1]),
        ).reshape((batch_size * num_source_draws, features.shape[-1]))
        low = jnp.broadcast_to(
            self.action_low,
            (batch_size * num_source_draws, self._flat_action_dim),
        )
        high = jnp.broadcast_to(
            self.action_high,
            (batch_size * num_source_draws, self._flat_action_dim),
        )

        level_keys = jax.random.split(key, self.levels)
        selected_per_level = []
        agreement_per_level = []
        q_span_per_level = []
        top2_gap_per_level = []
        source_q_std_per_level = []
        rank_snr_per_level = []
        for level in range(self.levels):
            condition = self._level_condition(low, high, level)
            # The expanded batch axis is the independent source-group axis.
            # Sources are passed explicitly so configured fixed rollout banks
            # cannot couple otherwise independent diagnostic groups.
            source = self._flow_source(
                level_keys[level],
                batch_size * num_source_draws,
                self.bins,
                num_samples=num_action_flow_samples,
            )
            endpoints = self._integrate_level(
                flow_params,
                expanded_features,
                condition,
                level_keys[level],
                source=source,
                num_samples=num_action_flow_samples,
            )
            q_values = self._endpoint_q_samples(endpoints).mean(axis=1).reshape(
                (
                    batch_size,
                    num_source_draws,
                    self.action_sequence,
                    self.action_dim,
                    self.bins,
                )
            )
            if self.hybrid_flow_v_direct_a:
                _, centered_advantage = self._advantage_level(
                    advantage_params,
                    expanded_features,
                    level,
                    low,
                    high,
                )
                q_values = q_values + centered_advantage.reshape(
                    (
                        batch_size,
                        num_source_draws,
                        self.action_sequence,
                        self.action_dim,
                        self.bins,
                    )
                )
            selected = jnp.argmax(q_values, axis=-1)
            selected_per_level.append(selected)

            bin_frequency = jax.nn.one_hot(
                selected,
                self.bins,
                dtype=jnp.float32,
            ).mean(axis=1)
            agreement_per_level.append(
                jnp.max(bin_frequency, axis=-1).mean()
            )
            q_span_per_level.append(
                (q_values.max(axis=-1) - q_values.min(axis=-1)).mean()
            )
            top_values, _ = jax.lax.top_k(q_values, 2)
            mean_top2_gap = (top_values[..., 0] - top_values[..., 1]).mean(
                axis=1
            )
            top2_gap_per_level.append(mean_top2_gap.mean())
            per_axis_source_q_std = q_values.std(axis=1).mean(axis=-1)
            source_q_std_per_level.append(per_axis_source_q_std.mean())
            rank_snr_per_level.append(
                (mean_top2_gap / (per_axis_source_q_std + 1e-6)).mean()
            )

            flat_selected = selected.reshape(
                (batch_size * num_source_draws, self._flat_action_dim)
            )
            low, high = zoom_in(
                low,
                high,
                flat_selected,
                self.bins,
                self.action_low,
                self.action_high,
            )

        actions = (0.5 * (low + high)).reshape(
            (
                batch_size,
                num_source_draws,
                self.action_sequence,
                self.action_dim,
            )
        )
        action_std = actions.std(axis=1)
        agreement = jnp.stack(agreement_per_level)
        selected_bins = jnp.stack(selected_per_level, axis=2)
        return {
            "per_level_bin_agreement": agreement,
            "per_level_bin_flip_rate": 1.0 - agreement,
            "per_level_q_span": jnp.stack(q_span_per_level),
            "per_level_top2_gap": jnp.stack(top2_gap_per_level),
            "per_level_source_q_std": jnp.stack(source_q_std_per_level),
            "per_level_rank_snr": jnp.stack(rank_snr_per_level),
            "action_source_std_mean": action_std.mean(),
            "action_source_std_max": action_std.max(),
            "action_mean": actions.mean(axis=1),
            "action_source_std": action_std,
            "selected_bins": selected_bins,
        }

    def source_resampling_ranking_probe(
        self,
        observations: dict,
        *,
        num_source_draws: int,
        num_action_flow_samples: int = 1,
        seed: int = 0,
        use_target_network: bool | None = None,
    ) -> dict[str, jax.Array]:
        """Run a read-only source-resampling ranking diagnostic.

        ``observations`` uses the same batched raw/image observation format as
        :meth:`act`.  The method does not advance the agent RNG, mutate action
        history, or alter training/rollout source configuration.
        """

        if num_source_draws < 2:
            raise ValueError("num_source_draws must be at least 2.")
        if num_action_flow_samples < 1:
            raise ValueError("num_action_flow_samples must be at least 1.")
        obs_inputs = self._prepare_rl_obs_inputs(observations)
        features = self._rl_features(
            self.params.get("encoder", None),
            obs_inputs,
            stop_gradient=True,
        )
        if use_target_network is None:
            use_target_network = self.use_target_network_for_rollout
        critic_params = (
            self.target_critic_params
            if bool(use_target_network)
            else (
                {
                    "critic": self.params["critic"],
                    "advantage": self.params["advantage"],
                }
                if self.hybrid_flow_v_direct_a
                else self.params["critic"]
            )
        )
        key = jax.random.PRNGKey(int(seed))

        def probe_fn(params, encoded_features, probe_key):
            return self._source_resampling_ranking_probe_from_features(
                params,
                encoded_features,
                probe_key,
                num_source_draws=num_source_draws,
                num_action_flow_samples=num_action_flow_samples,
            )

        if self._jit_enabled:
            probe_fn = jax.jit(probe_fn)
        metrics = probe_fn(critic_params, features, key)
        return jax.block_until_ready(metrics)

    def _endpoints_per_level(
        self,
        critic_params,
        features: jax.Array,
        action: jax.Array,
        key: jax.Array,
        *,
        num_flow_steps: int | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Return replay-chosen and all-bin flow readouts at every level.

        ``num_flow_steps=None`` uses the configured endpoint integrator.  A
        value of one is the ``Q_v0`` proxy ``source + v(source, t=0)`` used by
        stable return-flow demonstration imitation.
        """

        batch_size = features.shape[0]
        flat_action = jnp.asarray(action, dtype=jnp.float32).reshape(
            (batch_size, self._flat_action_dim)
        )
        discrete_action = encode_action(
            flat_action,
            self.action_low,
            self.action_high,
            self.levels,
            self.bins,
        ).reshape(
            (batch_size, self.levels, self.action_sequence, self.action_dim)
        )
        low = jnp.broadcast_to(self.action_low, flat_action.shape)
        high = jnp.broadcast_to(self.action_high, flat_action.shape)
        level_keys = list(jax.random.split(key, self.levels))
        all_endpoints = []
        chosen_endpoints = []
        for level in range(self.levels):
            endpoints = self._integrate_level(
                critic_params,
                features,
                self._level_condition(low, high, level),
                level_keys[level],
                num_flow_steps=num_flow_steps,
            )
            index = discrete_action[:, level]
            chosen = jnp.take_along_axis(
                endpoints,
                index[:, None, ..., None, None],
                axis=-2,
            )[..., 0, :]
            all_endpoints.append(endpoints)
            chosen_endpoints.append(chosen)
            low, high = zoom_in(
                low,
                high,
                index.reshape((batch_size, self._flat_action_dim)),
                self.bins,
                self.action_low,
                self.action_high,
            )
        # chosen: [B,R,L,K,D,V], all: [B,R,L,K,D,N,V]
        return (
            jnp.stack(chosen_endpoints, axis=2),
            jnp.stack(all_endpoints, axis=2),
        )

    def _q_values_per_level(
        self,
        critic_params,
        features: jax.Array,
        action: jax.Array,
        key: jax.Array,
        *,
        num_flow_steps: int | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Return scalar chosen/all-bin Q, including direct-A when enabled."""

        flow_params = critic_params
        advantage_params = None
        if self.hybrid_flow_v_direct_a:
            flow_params = critic_params["critic"]
            advantage_params = critic_params["advantage"]
        chosen_endpoints, all_endpoints = self._endpoints_per_level(
            flow_params,
            features,
            action,
            key,
            num_flow_steps=num_flow_steps,
        )
        chosen_q = self._endpoint_q(chosen_endpoints)
        all_q = self._endpoint_q(all_endpoints)
        if self.hybrid_flow_v_direct_a:
            _, _, chosen_advantage, all_advantage = (
                self._advantage_outputs_per_level(
                    advantage_params,
                    features,
                    action,
                )
            )
            chosen_q = chosen_q + chosen_advantage
            all_q = all_q + all_advantage
        return chosen_q, all_q

    def _selected_endpoints_per_level(
        self,
        critic_params,
        features: jax.Array,
        action: jax.Array,
        key: jax.Array,
        *,
        num_samples: int | None = None,
    ) -> jax.Array:
        """Integrate only the action-selected bin at every zoom level."""

        batch_size = features.shape[0]
        flat_action = jnp.asarray(action, dtype=jnp.float32).reshape(
            (batch_size, self._flat_action_dim)
        )
        discrete_action = encode_action(
            flat_action,
            self.action_low,
            self.action_high,
            self.levels,
            self.bins,
        ).reshape(
            (batch_size, self.levels, self.action_sequence, self.action_dim)
        )
        low = jnp.broadcast_to(self.action_low, flat_action.shape)
        high = jnp.broadcast_to(self.action_high, flat_action.shape)
        level_keys = list(jax.random.split(key, self.levels))
        chosen_endpoints = []
        for level in range(self.levels):
            index = discrete_action[:, level]
            condition = self._select_condition(
                self._level_condition(low, high, level),
                index,
            )
            endpoint = self._integrate_level(
                critic_params,
                features,
                condition,
                level_keys[level],
                num_samples=(
                    self.num_action_flow_samples
                    if num_samples is None
                    else num_samples
                ),
            )[..., 0, :]
            chosen_endpoints.append(endpoint)
            low, high = zoom_in(
                low,
                high,
                index.reshape((batch_size, self._flat_action_dim)),
                self.bins,
                self.action_low,
                self.action_high,
            )
        return jnp.stack(chosen_endpoints, axis=2)

    def _selected_endpoints_and_quantiles_per_level(
        self,
        critic_params,
        features: jax.Array,
        action: jax.Array,
        key: jax.Array,
        *,
        num_samples: int | None = None,
    ) -> tuple[jax.Array, jax.Array | None]:
        """Integrate selected bins and retain their source quantile labels."""

        batch_size = features.shape[0]
        sample_count = (
            self.num_action_flow_samples
            if num_samples is None
            else int(num_samples)
        )
        flat_action = jnp.asarray(action, dtype=jnp.float32).reshape(
            (batch_size, self._flat_action_dim)
        )
        discrete_action = encode_action(
            flat_action,
            self.action_low,
            self.action_high,
            self.levels,
            self.bins,
        ).reshape(
            (batch_size, self.levels, self.action_sequence, self.action_dim)
        )
        low = jnp.broadcast_to(self.action_low, flat_action.shape)
        high = jnp.broadcast_to(self.action_high, flat_action.shape)
        level_keys = list(jax.random.split(key, self.levels))
        chosen_endpoints = []
        source_quantiles = []
        for level in range(self.levels):
            index = discrete_action[:, level]
            condition = self._select_condition(
                self._level_condition(low, high, level),
                index,
            )
            source = self._flow_source(
                level_keys[level],
                batch_size,
                1,
                num_samples=sample_count,
            )
            quantiles = self._flow_source_quantiles(source)
            endpoint = self._integrate_level(
                critic_params,
                features,
                condition,
                level_keys[level],
                source=source,
                num_samples=sample_count,
            )[..., 0, :]
            chosen_endpoints.append(endpoint)
            if quantiles is not None:
                source_quantiles.append(quantiles[..., 0, 0])
            low, high = zoom_in(
                low,
                high,
                index.reshape((batch_size, self._flat_action_dim)),
                self.bins,
                self.action_low,
                self.action_high,
            )
        stacked_quantiles = (
            jnp.stack(source_quantiles, axis=2)
            if source_quantiles
            else None
        )
        return jnp.stack(chosen_endpoints, axis=2), stacked_quantiles

    def _target_values(
        self,
        target_critic_params,
        next_features,
        next_action,
        rewards,
        discounts,
        bootstrap,
        key,
        *,
        target_endpoints=None,
    ):
        if target_endpoints is None:
            target_endpoints = self._selected_endpoints_per_level(
                target_critic_params,
                next_features,
                next_action,
                key,
                num_samples=self.num_target_flow_samples,
            )
        if self.value_mode == "categorical":
            next_probabilities = flow_logits_to_probabilities(
                target_endpoints
            ).mean(axis=1)
            flat_probabilities = next_probabilities.reshape(
                (
                    next_probabilities.shape[0],
                    self.levels,
                    self._flat_action_dim,
                    self.atoms,
                )
            )
            target_distribution = project_categorical(
                flat_probabilities,
                rewards,
                discounts,
                bootstrap,
                self.support,
            ).reshape(
                (
                    next_probabilities.shape[0],
                    self.levels,
                    self.action_sequence,
                    self.action_dim,
                    self.atoms,
                )
            )
            if self.centralized_critic:
                target_distribution = jnp.broadcast_to(
                    target_distribution.mean(axis=(2, 3), keepdims=True),
                    target_distribution.shape,
                )
            return jax.lax.stop_gradient(target_distribution)

        if self.value_mode == "return_sample":
            next_returns = target_endpoints[..., 0]
            target_returns = rewards[:, None, None, None, None] + (
                discounts * bootstrap
            )[:, None, None, None, None] * next_returns
            if self.clip_scalar_targets:
                target_returns = jnp.clip(
                    target_returns, self.v_min, self.v_max
                )
            if self.centralized_critic:
                target_returns = jnp.broadcast_to(
                    target_returns.mean(axis=(3, 4), keepdims=True),
                    target_returns.shape,
                )
            return jax.lax.stop_gradient(target_returns)

        next_q = target_endpoints[..., 0].mean(axis=1)
        target_q = rewards[:, None, None, None] + (
            discounts * bootstrap
        )[:, None, None, None] * next_q
        if self.clip_scalar_targets:
            target_q = jnp.clip(target_q, self.v_min, self.v_max)
        if self.centralized_critic:
            target_q = jnp.broadcast_to(
                target_q.mean(axis=(2, 3), keepdims=True), target_q.shape
            )
        return jax.lax.stop_gradient(target_q)

    def _hybrid_target_values(
        self,
        target_params,
        next_features: jax.Array,
        next_action: jax.Array,
        rewards: jax.Array,
        discounts: jax.Array,
        bootstrap: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        """Bellman target for ``Flow-V + centered direct-C51 advantage``."""

        target_v_endpoints = self._selected_endpoints_per_level(
            target_params["critic"],
            next_features,
            next_action,
            key,
            num_samples=self.num_target_flow_samples,
        )
        target_v = self._endpoint_q(target_v_endpoints)
        _, _, target_advantage, _ = self._advantage_outputs_per_level(
            target_params["advantage"],
            next_features,
            next_action,
        )
        next_q = target_v + target_advantage
        target_q = rewards[:, None, None, None] + (
            discounts * bootstrap
        )[:, None, None, None] * next_q
        if self.clip_scalar_targets:
            target_q = jnp.clip(target_q, self.v_min, self.v_max)
        if self.centralized_critic:
            target_q = jnp.broadcast_to(
                target_q.mean(axis=(2, 3), keepdims=True),
                target_q.shape,
            )
        return jax.lax.stop_gradient(target_q)

    def _flow_matching_loss(
        self,
        critic_params,
        features,
        actions,
        targets,
        source_key,
        time_key,
    ):
        batch_size = features.shape[0]
        flat_action = jnp.asarray(actions, dtype=jnp.float32).reshape(
            (batch_size, self._flat_action_dim)
        )
        discrete_action = encode_action(
            flat_action,
            self.action_low,
            self.action_high,
            self.levels,
            self.bins,
        ).reshape(
            (batch_size, self.levels, self.action_sequence, self.action_dim)
        )
        low = jnp.broadcast_to(self.action_low, flat_action.shape)
        high = jnp.broadcast_to(self.action_high, flat_action.shape)
        source_keys = list(jax.random.split(source_key, self.levels))
        time_keys = list(jax.random.split(time_key, self.levels))
        per_level = []
        for level in range(self.levels):
            index = discrete_action[:, level]
            condition = self._select_condition(
                self._level_condition(low, high, level),
                index,
            )
            source = self._flow_source(
                source_keys[level],
                batch_size,
                1,
                num_samples=self.num_flow_samples,
            )
            if self.value_mode == "categorical":
                target = centered_log_probabilities(targets[:, level])
                target = target[:, None, ..., None, :]
                target = jnp.broadcast_to(
                    target,
                    (
                        batch_size,
                        self.num_flow_samples,
                        self.action_sequence,
                        self.action_dim,
                        1,
                        self.value_dim,
                    ),
                )
            elif self.value_mode == "return_sample":
                target = targets[:, :, level, ..., None, None]
            else:
                target = targets[:, level, ..., None]
                target = target[:, None, ..., None, :]
                target = jnp.broadcast_to(
                    target,
                    (
                        batch_size,
                        self.num_flow_samples,
                        self.action_sequence,
                        self.action_dim,
                        1,
                        self.value_dim,
                    ),
                )
            source_quantiles = None
            if self.flow_iqn_quantile_coupling:
                coupled = quantile_couple_return_samples(
                    source,
                    target,
                    source_min=self.flow_source_min,
                    source_max=self.flow_source_max,
                    sample_axis=1,
                )
                source = coupled.source
                target = coupled.target
                source_quantiles = coupled.source_quantile
            tau = jax.random.uniform(
                time_keys[level],
                (batch_size, self.num_flow_samples),
                minval=0.0,
                maxval=1.0,
                dtype=jnp.float32,
            )
            pair = linear_flow_training_pair(source, target, tau)
            prediction = self.critic_model.apply(
                critic_params,
                features,
                *condition,
                pair.sample,
                tau,
                source_quantiles,
            )
            selected_prediction = prediction[..., 0, :]
            if self.value_mode == "categorical":
                selected_prediction = selected_prediction - (
                    selected_prediction.mean(axis=-1, keepdims=True)
                )
            selected_velocity = pair.target_velocity[..., 0, :]
            squared_error = jnp.square(selected_prediction - selected_velocity)
            squared_error = self._sequence_training_slice(
                squared_error,
                sequence_axis=2,
            )
            per_level.append(squared_error.mean(axis=(1, 2, 3, 4)))

            low, high = zoom_in(
                low,
                high,
                index.reshape((batch_size, self._flat_action_dim)),
                self.bins,
                self.action_low,
                self.action_high,
            )

        per_sample = jnp.stack(per_level, axis=1).mean(axis=1)
        return per_sample

    def _target_expected_q(self, targets: jax.Array) -> jax.Array:
        if self.value_mode == "categorical":
            return expected_q(targets, self.support)
        if self.value_mode == "return_sample":
            return targets.mean(axis=1)
        return targets

    def _dcfm_loss(
        self,
        critic_params,
        target_critic_params,
        features,
        next_features,
        actions,
        next_actions,
        rewards,
        discounts,
        bootstrap,
        source_key,
        time_key,
    ):
        """Value Flows distributional consistency loss for scalar returns.

        In repository time coordinates, tau=1 is Gaussian noise and tau=0 is
        the return endpoint. The same base noise and time are used by BCFM and
        DCFM. Following the paper and official implementation, discounting is
        applied to the current-flow input; the target vector field is not
        multiplied by an additional discount factor.
        """

        if self.value_mode != "return_sample":
            raise ValueError("DCFM requires value_mode=return_sample.")
        batch_size = features.shape[0]
        current_flat = jnp.asarray(actions, dtype=jnp.float32).reshape(
            (batch_size, self._flat_action_dim)
        )
        next_flat = jnp.asarray(next_actions, dtype=jnp.float32).reshape(
            (batch_size, self._flat_action_dim)
        )
        current_indices = encode_action(
            current_flat,
            self.action_low,
            self.action_high,
            self.levels,
            self.bins,
        ).reshape(
            (batch_size, self.levels, self.action_sequence, self.action_dim)
        )
        next_indices = encode_action(
            next_flat,
            self.action_low,
            self.action_high,
            self.levels,
            self.bins,
        ).reshape(
            (batch_size, self.levels, self.action_sequence, self.action_dim)
        )
        current_low = jnp.broadcast_to(self.action_low, current_flat.shape)
        current_high = jnp.broadcast_to(self.action_high, current_flat.shape)
        next_low = jnp.broadcast_to(self.action_low, next_flat.shape)
        next_high = jnp.broadcast_to(self.action_high, next_flat.shape)
        source_keys = list(jax.random.split(source_key, self.levels))
        time_keys = list(jax.random.split(time_key, self.levels))
        discount_mask = (discounts * bootstrap)[
            :, None, None, None, None, None
        ]
        valid_dcfm = bootstrap[:, None, None, None, None, None]
        reward_values = rewards[:, None, None, None, None, None]
        per_level = []
        for level in range(self.levels):
            current_index = current_indices[:, level]
            next_index = next_indices[:, level]
            current_condition = self._select_condition(
                self._level_condition(current_low, current_high, level),
                current_index,
            )
            next_condition = self._select_condition(
                self._level_condition(next_low, next_high, level),
                next_index,
            )
            source = self._flow_source(
                source_keys[level],
                batch_size,
                1,
                num_samples=self.num_flow_samples,
            )
            tau = jax.random.uniform(
                time_keys[level],
                (batch_size, self.num_flow_samples),
                minval=0.0,
                maxval=1.0,
                dtype=jnp.float32,
            )
            partial_next_return = self._integrate_level(
                target_critic_params,
                next_features,
                next_condition,
                source_keys[level],
                source=source,
                end_tau=tau,
            )
            current_return = reward_values + discount_mask * partial_next_return
            if self.centralized_critic:
                current_return = jnp.broadcast_to(
                    current_return.mean(axis=(2, 3), keepdims=True),
                    current_return.shape,
                )
            current_velocity = self.critic_model.apply(
                critic_params,
                features,
                *current_condition,
                current_return,
                tau,
            )
            target_velocity = self.critic_model.apply(
                target_critic_params,
                next_features,
                *next_condition,
                partial_next_return,
                tau,
            )
            if self.centralized_critic:
                target_velocity = jnp.broadcast_to(
                    target_velocity.mean(axis=(2, 3), keepdims=True),
                    target_velocity.shape,
                )
            squared_error = jnp.square(
                current_velocity - jax.lax.stop_gradient(target_velocity)
            ) * valid_dcfm
            squared_error = self._sequence_training_slice(
                squared_error,
                sequence_axis=2,
            )
            per_level.append(squared_error.mean(axis=(1, 2, 3, 4, 5)))

            current_low, current_high = zoom_in(
                current_low,
                current_high,
                current_index.reshape((batch_size, self._flat_action_dim)),
                self.bins,
                self.action_low,
                self.action_high,
            )
            next_low, next_high = zoom_in(
                next_low,
                next_high,
                next_index.reshape((batch_size, self._flat_action_dim)),
                self.bins,
                self.action_low,
                self.action_high,
            )
        return jnp.stack(per_level, axis=1).mean(axis=1)

    def _evor_td_loss(
        self,
        critic_params,
        target_critic_params,
        features,
        next_features,
        actions,
        next_actions,
        rewards,
        discounts,
        bootstrap,
        mc_returns,
        source_key,
        time_key,
    ):
        """EVOR velocity-space TD objective (paper Equations 35 and 36).

        The completed trajectory return supplies the paper-valid offline
        reward-to-go sample ``z1`` and fresh Gaussian noise supplies ``z0``.
        The online velocity at their linear interpolant is regressed to
        ``r + gamma * target_next_velocity`` evaluated at the exact same
        scalar ``z_t`` and time.  The next action is supplied by the
        independent BC policy before entering this helper.  This uses the
        data-endpoint option described after EVOR Equation 16 and avoids an
        unanchored target-flow endpoint during sparse-reward cold start.
        """

        if self.value_mode != "return_sample":
            raise ValueError("EVOR FlowTD requires value_mode=return_sample.")
        batch_size = features.shape[0]
        current_flat = jnp.asarray(actions, dtype=jnp.float32).reshape(
            (batch_size, self._flat_action_dim)
        )
        next_flat = jnp.asarray(next_actions, dtype=jnp.float32).reshape(
            (batch_size, self._flat_action_dim)
        )
        current_indices = encode_action(
            current_flat,
            self.action_low,
            self.action_high,
            self.levels,
            self.bins,
        ).reshape(
            (batch_size, self.levels, self.action_sequence, self.action_dim)
        )
        next_indices = encode_action(
            next_flat,
            self.action_low,
            self.action_high,
            self.levels,
            self.bins,
        ).reshape(
            (batch_size, self.levels, self.action_sequence, self.action_dim)
        )
        current_low = jnp.broadcast_to(
            self.action_low, current_flat.shape
        )
        current_high = jnp.broadcast_to(
            self.action_high, current_flat.shape
        )
        next_low = jnp.broadcast_to(self.action_low, next_flat.shape)
        next_high = jnp.broadcast_to(self.action_high, next_flat.shape)
        source_keys = list(jax.random.split(source_key, self.levels))
        time_keys = list(jax.random.split(time_key, self.levels))
        effective_discount = discounts * bootstrap
        mc_returns = jnp.asarray(mc_returns, dtype=jnp.float32)
        if mc_returns.shape != (batch_size,):
            raise ValueError("EVOR mc_returns must have shape [batch].")
        per_level = []

        for level in range(self.levels):
            current_index = current_indices[:, level]
            next_index = next_indices[:, level]
            current_condition = self._select_condition(
                self._level_condition(current_low, current_high, level),
                current_index,
            )
            next_condition = self._select_condition(
                self._level_condition(next_low, next_high, level),
                next_index,
            )
            source = self._flow_source(
                source_keys[level],
                batch_size,
                1,
                num_samples=self.num_flow_samples,
            )
            current_endpoint = jax.lax.stop_gradient(
                jnp.broadcast_to(
                    mc_returns[:, None, None, None, None, None],
                    source.shape,
                )
            )
            tau = jax.random.uniform(
                time_keys[level],
                (batch_size, self.num_flow_samples),
                minval=0.0,
                maxval=1.0,
                dtype=jnp.float32,
            )
            current_sample = linear_flow_training_pair(
                source,
                current_endpoint,
                tau,
            ).sample
            next_velocity = self.critic_model.apply(
                target_critic_params,
                next_features,
                *next_condition,
                current_sample,
                tau,
                None,
            )
            pair = evor_velocity_td_pair(
                source,
                current_endpoint,
                rewards,
                effective_discount,
                next_velocity,
                tau,
            )
            target_velocity = pair.target_velocity
            if self.centralized_critic:
                target_velocity = jnp.broadcast_to(
                    target_velocity.mean(axis=(2, 3), keepdims=True),
                    target_velocity.shape,
                )
            prediction = self.critic_model.apply(
                critic_params,
                features,
                *current_condition,
                pair.current_sample,
                tau,
                None,
            )
            squared_error = jnp.square(prediction - target_velocity)
            squared_error = self._sequence_training_slice(
                squared_error,
                sequence_axis=2,
            )
            per_level.append(
                squared_error.mean(axis=(1, 2, 3, 4, 5))
            )

            current_low, current_high = zoom_in(
                current_low,
                current_high,
                current_index.reshape(
                    (batch_size, self._flat_action_dim)
                ),
                self.bins,
                self.action_low,
                self.action_high,
            )
            next_low, next_high = zoom_in(
                next_low,
                next_high,
                next_index.reshape(
                    (batch_size, self._flat_action_dim)
                ),
                self.bins,
                self.action_low,
                self.action_high,
            )

        return jnp.stack(per_level, axis=1).mean(axis=1)

    def _pcbf_loss(
        self,
        critic_params,
        target_critic_params,
        features,
        next_features,
        actions,
        next_actions,
        rewards,
        discounts,
        bootstrap,
        source_key,
        time_key,
        *,
        next_endpoints=None,
    ):
        """Path-Coupled Bellman Flows loss for scalar return samples.

        PCBF uses the same base sample for the current and successor paths.
        The paper uses forward time ``t`` (source=0, endpoint=1), while the
        rest of this repository uses ``tau=1-t``.  The critic therefore still
        receives ``tau`` and handles its own forward-time embedding.
        """

        if self.value_mode != "return_sample":
            raise ValueError("PCBF requires value_mode=return_sample.")
        batch_size = features.shape[0]
        current_flat = jnp.asarray(actions, dtype=jnp.float32).reshape(
            (batch_size, self._flat_action_dim)
        )
        next_flat = jnp.asarray(next_actions, dtype=jnp.float32).reshape(
            (batch_size, self._flat_action_dim)
        )
        current_indices = encode_action(
            current_flat,
            self.action_low,
            self.action_high,
            self.levels,
            self.bins,
        ).reshape(
            (batch_size, self.levels, self.action_sequence, self.action_dim)
        )
        next_indices = encode_action(
            next_flat,
            self.action_low,
            self.action_high,
            self.levels,
            self.bins,
        ).reshape(
            (batch_size, self.levels, self.action_sequence, self.action_dim)
        )
        current_low = jnp.broadcast_to(self.action_low, current_flat.shape)
        current_high = jnp.broadcast_to(self.action_high, current_flat.shape)
        next_low = jnp.broadcast_to(self.action_low, next_flat.shape)
        next_high = jnp.broadcast_to(self.action_high, next_flat.shape)
        source_keys = list(jax.random.split(source_key, self.levels))
        time_keys = list(jax.random.split(time_key, self.levels))
        reward_values = rewards[:, None, None, None, None, None]
        effective_discount = (discounts * bootstrap)[
            :, None, None, None, None, None
        ]
        bootstrap_values = bootstrap[:, None, None, None, None, None]
        per_level = []
        for level in range(self.levels):
            current_index = current_indices[:, level]
            next_index = next_indices[:, level]
            current_condition = self._select_condition(
                self._level_condition(current_low, current_high, level),
                current_index,
            )
            next_condition = self._select_condition(
                self._level_condition(next_low, next_high, level),
                next_index,
            )
            source = self._flow_source(
                source_keys[level],
                batch_size,
                1,
                num_samples=self.num_flow_samples,
            )
            repo_tau = jax.random.uniform(
                time_keys[level],
                (batch_size, self.num_flow_samples),
                minval=0.0,
                maxval=1.0,
                dtype=jnp.float32,
            )
            forward_time = (1.0 - repo_tau)[..., None, None, None, None]
            if next_endpoints is None:
                next_endpoint = self._integrate_level(
                    target_critic_params,
                    next_features,
                    next_condition,
                    source_keys[level],
                    source=source,
                    end_tau=0.0,
                )
            else:
                # _selected_endpoints_per_level removes the singleton selected
                # bin axis; restore it for vector-field evaluation.
                next_endpoint = next_endpoints[:, :, level, ..., None, :]
            if self.centralized_critic:
                next_endpoint = jnp.broadcast_to(
                    next_endpoint.mean(axis=(2, 3), keepdims=True),
                    next_endpoint.shape,
                )
            next_endpoint = jax.lax.stop_gradient(next_endpoint)
            successor_sample = (
                (1.0 - forward_time) * source
                + forward_time * next_endpoint
            )
            next_velocity = self.critic_model.apply(
                target_critic_params,
                next_features,
                *next_condition,
                successor_sample,
                repo_tau,
            )
            pair = path_coupled_bellman_flow_pair(
                source,
                next_endpoint,
                reward_values,
                effective_discount,
                bootstrap_values,
                forward_time,
                next_velocity,
                control_lambda=self.pcbf_lambda,
            )
            current_velocity = self.critic_model.apply(
                critic_params,
                features,
                *current_condition,
                pair.current_sample,
                repo_tau,
            )
            squared_error = jnp.square(
                current_velocity - pair.target_velocity
            )
            squared_error = self._sequence_training_slice(
                squared_error,
                sequence_axis=2,
            )
            per_level.append(squared_error.mean(axis=(1, 2, 3, 4, 5)))

            current_low, current_high = zoom_in(
                current_low,
                current_high,
                current_index.reshape((batch_size, self._flat_action_dim)),
                self.bins,
                self.action_low,
                self.action_high,
            )
            next_low, next_high = zoom_in(
                next_low,
                next_high,
                next_index.reshape((batch_size, self._flat_action_dim)),
                self.bins,
                self.action_low,
                self.action_high,
            )
        return jnp.stack(per_level, axis=1).mean(axis=1)

    def _demo_losses(self, all_endpoints, chosen_endpoints):
        all_q_samples = self._endpoint_q_samples(all_endpoints)
        chosen_q_samples = self._endpoint_q_samples(chosen_endpoints)
        all_q = all_q_samples.mean(axis=1)
        chosen_q = chosen_q_samples.mean(axis=1)
        if self.value_mode == "categorical":
            all_probabilities = flow_logits_to_probabilities(all_endpoints).mean(axis=1)
            chosen_probabilities = flow_logits_to_probabilities(
                chosen_endpoints
            ).mean(axis=1)
            fosd = (
                demo_fosd_per_sample(chosen_probabilities, all_probabilities)
                if self.demo_fosd
                else jnp.zeros((all_q.shape[0],), dtype=all_q.dtype)
            )
        else:
            fosd = jnp.zeros((all_q.shape[0],), dtype=all_q.dtype)
        margin = (
            demo_margin_per_sample(all_q, chosen_q, margin=self.bc_margin)
            if self.bc_margin > 0.0
            else jnp.zeros_like(fosd)
        )
        # Everything below is logging-only.  Detach it explicitly rather than
        # relying on has_aux to prune every shared cotangent.  In particular,
        # d(std)/dx is undefined at exactly zero source variance and previously
        # contaminated an otherwise finite demo-margin update.
        diagnostic_all_q_samples = jax.lax.stop_gradient(all_q_samples)
        diagnostic_chosen_q = jax.lax.stop_gradient(chosen_q)
        diagnostic_all_q = jax.lax.stop_gradient(all_q)
        optimality_gap = diagnostic_chosen_q - diagnostic_all_q.max(axis=-1)
        max_q = diagnostic_all_q.max(axis=-1, keepdims=True)
        max_ties = jnp.sum(
            jnp.abs(diagnostic_all_q - max_q) <= 1e-6,
            axis=-1,
        )
        # Credit ties fractionally. At the all-equal initialization this is
        # exactly 1 / bins, rather than the misleading 100% from "tied max".
        expert_top1 = jnp.where(
            optimality_gap >= -1e-6,
            1.0 / jnp.maximum(max_ties, 1),
            0.0,
        ).mean(
            axis=tuple(range(1, optimality_gap.ndim))
        )
        expert_q_gap = optimality_gap.mean(
            axis=tuple(range(1, optimality_gap.ndim))
        )
        q_span = (
            diagnostic_all_q.max(axis=-1) - diagnostic_all_q.min(axis=-1)
        ).mean(
            axis=tuple(range(1, diagnostic_all_q.ndim - 1))
        )
        top_values, _ = jax.lax.top_k(diagnostic_all_q, 2)
        top2_gap = (top_values[..., 0] - top_values[..., 1]).mean(
            axis=tuple(range(1, top_values.ndim - 1))
        )
        all_bin_source_std = diagnostic_all_q_samples.std(axis=1).mean(
            axis=tuple(range(1, diagnostic_all_q_samples.ndim - 1))
        )
        rank_snr = top2_gap / (all_bin_source_std + 1e-6)
        source_bin_flip_rate = source_bin_flip_rate_per_sample(
            diagnostic_all_q_samples
        )
        return (
            fosd,
            margin,
            expert_top1,
            expert_q_gap,
            q_span,
            top2_gap,
            all_bin_source_std,
            rank_snr,
            source_bin_flip_rate,
        )

    def _build_hybrid_update_fn(self):
        """Train Flow-V and a direct categorical advantage as separate roles."""

        optimizer = self.optimizer
        target_tau = self.critic_target_tau

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
            obs_inputs, next_obs_inputs, action_key = self._augment_update_obs_inputs(
                obs_inputs,
                next_obs_inputs,
                action_key,
            )
            (
                selection_key,
                target_source_key,
                train_source_key,
                time_key,
                endpoint_key,
                branch_key,
            ) = jax.random.split(action_key, 6)

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
                    action_sequence = actions.reshape(
                        (
                            actions.shape[0],
                            self.action_sequence,
                            self.action_dim,
                        )
                    )
                    next_action = jnp.concatenate(
                        [action_sequence[:, 1:], action_sequence[:, -1:]],
                        axis=1,
                    )
                elif self.td_target_action_source == "bc_policy":
                    next_action, _ = self._policy_action(
                        current_params["policy"],
                        next_policy_features,
                        key=selection_key,
                    )
                else:
                    next_action, _ = self._hybrid_greedy_action(
                        current_params["critic"],
                        current_params["advantage"],
                        next_features,
                        selection_key,
                    )

                target_q = self._hybrid_target_values(
                    target_critic_params,
                    next_features,
                    next_action,
                    rewards,
                    discounts,
                    bootstrap,
                    target_source_key,
                )

                # Flow-V is deliberately not given the replayed candidate bin.
                # Completed return supplies a behavior-state/prefix baseline;
                # action-dependent Bellman residuals are left to direct-A.
                flow_v_targets = jnp.broadcast_to(
                    mc_returns[:, None, None, None],
                    target_q.shape,
                )
                flow_per_sample = self._flow_matching_loss(
                    current_params["critic"],
                    features,
                    actions,
                    flow_v_targets,
                    train_source_key,
                    time_key,
                )
                bcfm_loss = jnp.mean(flow_per_sample * loss_weights)

                chosen_v_endpoints = self._selected_endpoints_per_level(
                    current_params["critic"],
                    features,
                    actions,
                    endpoint_key,
                    num_samples=self.num_flow_samples,
                )
                flow_v_samples = self._endpoint_q_samples(chosen_v_endpoints)
                flow_v_samples = self._sequence_training_slice(
                    flow_v_samples,
                    sequence_axis=3,
                )
                flow_v = flow_v_samples.mean(axis=1)
                target_q_train = self._sequence_training_slice(
                    target_q,
                    sequence_axis=2,
                )
                flow_v_targets_train = self._sequence_training_slice(
                    flow_v_targets,
                    sequence_axis=2,
                )

                (
                    chosen_advantage_logits,
                    _,
                    chosen_advantage,
                    all_advantage,
                ) = self._advantage_outputs_per_level(
                    current_params["advantage"],
                    features,
                    actions,
                )
                chosen_advantage_logits = self._sequence_training_slice(
                    chosen_advantage_logits,
                    sequence_axis=2,
                )
                chosen_advantage = self._sequence_training_slice(
                    chosen_advantage,
                    sequence_axis=2,
                )
                all_advantage = self._sequence_training_slice(
                    all_advantage,
                    sequence_axis=2,
                )

                residual_target = jax.lax.stop_gradient(
                    target_q_train - flow_v
                )
                residual_distribution = scalar_to_categorical(
                    residual_target,
                    self.support,
                )
                advantage_ce = categorical_cross_entropy(
                    residual_distribution,
                    chosen_advantage_logits,
                )
                advantage_ce_per_sample = advantage_ce.mean(axis=(1, 2, 3))
                advantage_c51_loss = jnp.mean(
                    advantage_ce_per_sample * loss_weights
                )

                # Stop Flow-V here so the direct head cannot delegate ranking
                # back to the large baseline. Centering couples every bin and
                # fixes the additive decomposition's otherwise free offset.
                hybrid_q = jax.lax.stop_gradient(flow_v) + chosen_advantage
                advantage_q_error = jnp.square(hybrid_q - target_q_train)
                advantage_q_per_sample = advantage_q_error.mean(
                    axis=(1, 2, 3)
                )
                advantage_q_loss = jnp.mean(
                    advantage_q_per_sample * loss_weights
                )

                flow_v_error = jnp.square(flow_v - flow_v_targets_train)
                flow_v_per_sample = flow_v_error.mean(axis=(1, 2, 3))
                endpoint_q_loss = jnp.mean(
                    flow_v_per_sample * loss_weights
                )
                source_variance_per_sample = flow_v_samples.var(axis=1).mean(
                    axis=(1, 2, 3)
                )
                source_consistency_loss = jnp.mean(
                    source_variance_per_sample * loss_weights
                )
                td_critic_loss = self.critic_lambda * (
                    self.bcfm_lambda * bcfm_loss
                    + self.advantage_c51_lambda * advantage_c51_loss
                    + self.advantage_q_lambda * advantage_q_loss
                    + self.endpoint_q_lambda * endpoint_q_loss
                    + self.source_consistency_lambda
                    * source_consistency_loss
                )
                mc_return_loss = self.mc_return_weight * endpoint_q_loss
                (
                    causal_branch_ranking_loss,
                    causal_branch_delta_loss,
                    causal_branch_accuracy,
                    causal_branch_q_span,
                ) = self._causal_branch_objective(
                    current_params["advantage"],
                    branch_key,
                )
                causal_branch_loss = self.causal_branch_weight * (
                    causal_branch_ranking_loss
                    + self.causal_branch_delta_weight
                    * causal_branch_delta_loss
                )
                critic_loss = (
                    td_critic_loss + mc_return_loss + causal_branch_loss
                )

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
                policy_per_sample = -expert_log_probabilities.mean(
                    axis=(1, 2)
                )
                demo_count = jnp.maximum(jnp.sum(demos), 1.0)
                policy_ce = jnp.sum(policy_per_sample * demos) / demo_count
                policy_loss = self.bc_lambda * policy_ce
                optimized_policy_loss = (
                    jax.lax.stop_gradient(policy_loss)
                    if self.freeze_bc_policy
                    else policy_loss
                )
                total_loss = critic_loss + optimized_policy_loss

                policy_correct = (
                    jnp.argmax(policy_logits, axis=-1) == expert_bins
                ).astype(jnp.float32).mean(axis=(1, 2))
                policy_demo_top1 = (
                    jnp.sum(policy_correct * demos) / demo_count
                )
                policy_probabilities = jax.nn.softmax(
                    policy_logits,
                    axis=-1,
                )
                policy_entropy = -jnp.sum(
                    policy_probabilities
                    * jnp.log(jnp.maximum(policy_probabilities, 1e-9)),
                    axis=-1,
                ).mean()
                advantage_span = (
                    all_advantage.max(axis=-1)
                    - all_advantage.min(axis=-1)
                ).mean()
                return total_loss, (
                    flow_per_sample,
                    advantage_ce_per_sample,
                    advantage_q_per_sample,
                    bcfm_loss,
                    advantage_c51_loss,
                    advantage_q_loss,
                    endpoint_q_loss,
                    source_consistency_loss,
                    td_critic_loss,
                    mc_return_loss,
                    policy_loss,
                    policy_ce,
                    policy_demo_top1,
                    policy_entropy,
                    causal_branch_loss,
                    causal_branch_ranking_loss,
                    causal_branch_delta_loss,
                    causal_branch_accuracy,
                    causal_branch_q_span,
                    flow_v.mean(),
                    flow_v.min(),
                    flow_v.max(),
                    target_q_train.mean(),
                    hybrid_q.mean(),
                    jnp.abs(chosen_advantage).mean(),
                    advantage_span,
                    jnp.mean(
                        jnp.abs(
                            flow_v
                            - mc_returns[:, None, None, None]
                        )
                    ),
                )

            (total_loss, aux), grads = jax.value_and_grad(
                loss_fn,
                has_aux=True,
            )(params)

            def tree_norm_or_zero(name):
                return (
                    self.optax.tree.norm(grads[name])
                    if name in grads
                    else jnp.asarray(0.0, dtype=total_loss.dtype)
                )

            grad_norm = self.optax.tree.norm(grads)
            flow_grad_norm = tree_norm_or_zero("critic")
            advantage_grad_norm = tree_norm_or_zero("advantage")
            encoder_grad_norm = tree_norm_or_zero("encoder")
            policy_encoder_grad_norm = tree_norm_or_zero("policy_encoder")

            def nonfinite_fraction(tree):
                leaves = jax.tree.leaves(tree)
                nonfinite = sum(
                    jnp.sum(~jnp.isfinite(leaf)) for leaf in leaves
                )
                size = sum(leaf.size for leaf in leaves)
                return nonfinite.astype(jnp.float32) / float(max(size, 1))

            flow_nonfinite = nonfinite_fraction(grads["critic"])
            advantage_nonfinite = nonfinite_fraction(grads["advantage"])
            updates, opt_state = optimizer.update(grads, opt_state, params)
            update_norm = self.optax.tree.norm(updates)
            updated_params = self.optax.apply_updates(params, updates)
            if self.freeze_bc_policy:
                updated_params = {
                    **updated_params,
                    "policy": params["policy"],
                    "policy_encoder": params["policy_encoder"],
                }
            params = updated_params
            online_target_params = {
                "critic": params["critic"],
                "advantage": params["advantage"],
            }
            target_critic_params = jax.tree.map(
                lambda target, online: (1.0 - target_tau) * target
                + target_tau * online,
                target_critic_params,
                online_target_params,
            )
            (
                flow_per_sample,
                advantage_ce_per_sample,
                advantage_q_per_sample,
                bcfm_loss,
                advantage_c51_loss,
                advantage_q_loss,
                endpoint_q_loss,
                source_consistency_loss,
                td_critic_loss,
                mc_return_loss,
                policy_loss,
                policy_ce,
                policy_demo_top1,
                policy_entropy,
                causal_branch_loss,
                causal_branch_ranking_loss,
                causal_branch_delta_loss,
                causal_branch_accuracy,
                causal_branch_q_span,
                flow_v_mean,
                flow_v_min,
                flow_v_max,
                target_q_mean,
                hybrid_q_mean,
                advantage_abs_mean,
                advantage_span,
                mc_return_mae,
            ) = aux
            critic_loss = (
                td_critic_loss + mc_return_loss + causal_branch_loss
            )
            priority_error = (
                self.bcfm_lambda * flow_per_sample
                + self.advantage_c51_lambda * advantage_ce_per_sample
                + self.advantage_q_lambda * advantage_q_per_sample
            )
            priority = jnp.sqrt(
                jnp.maximum(priority_error, 0.0) + 1e-10
            )
            priority = priority / jnp.maximum(jnp.max(priority), 1e-10)
            return (
                params,
                target_critic_params,
                opt_state,
                priority,
                {
                    "critic_loss": critic_loss,
                    "td_critic_loss": td_critic_loss,
                    "mc_return_loss": mc_return_loss,
                    "mc_return_mae": mc_return_mae,
                    "mc_return_mean": jnp.mean(mc_returns),
                    "policy_bc_loss": policy_loss,
                    "policy_ce": policy_ce,
                    "policy_demo_top1": policy_demo_top1,
                    "policy_entropy": policy_entropy,
                    "causal_branch_loss": causal_branch_loss,
                    "causal_branch_ranking_loss": (
                        causal_branch_ranking_loss
                    ),
                    "causal_branch_delta_loss": causal_branch_delta_loss,
                    "causal_branch_pairwise_accuracy": (
                        causal_branch_accuracy
                    ),
                    "causal_branch_q_span": causal_branch_q_span,
                    "total_loss": total_loss,
                    "critic_grad_norm": grad_norm,
                    "flow_critic_grad_norm": flow_grad_norm,
                    "advantage_grad_norm": advantage_grad_norm,
                    "encoder_grad_norm": encoder_grad_norm,
                    "policy_encoder_grad_norm": policy_encoder_grad_norm,
                    "flow_critic_grad_nonfinite_fraction": flow_nonfinite,
                    "advantage_grad_nonfinite_fraction": advantage_nonfinite,
                    "critic_update_norm": update_norm,
                    "flow_loss": bcfm_loss,
                    "bcfm_loss": bcfm_loss,
                    "advantage_c51_loss": advantage_c51_loss,
                    "advantage_q_loss": advantage_q_loss,
                    "endpoint_q_loss": endpoint_q_loss,
                    "source_consistency_loss": source_consistency_loss,
                    "source_q_std": jnp.sqrt(
                        jnp.maximum(source_consistency_loss, 0.0)
                    ),
                    "flow_v_mean": flow_v_mean,
                    "flow_v_min": flow_v_min,
                    "flow_v_max": flow_v_max,
                    "target_q_mean": target_q_mean,
                    "hybrid_q_mean": hybrid_q_mean,
                    "advantage_abs_mean": advantage_abs_mean,
                    "advantage_q_span": advantage_span,
                    "loss_coeff": jnp.mean(loss_weights),
                },
            )

        return update_fn

    def _build_update_fn(self):
        if self.hybrid_flow_v_direct_a:
            return self._build_hybrid_update_fn()

        optimizer = self.optimizer
        target_tau = self.critic_target_tau

        def core_update_fn(
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
            obs_inputs, next_obs_inputs, action_key = self._augment_update_obs_inputs(
                obs_inputs, next_obs_inputs, action_key
            )
            (
                selection_key,
                target_source_key,
                train_source_key,
                time_key,
                endpoint_key,
                demo_key,
            ) = jax.random.split(action_key, 6)

            def loss_fn(current_params):
                encoder_params = current_params.get("encoder", None)
                features = self._rl_features(encoder_params, obs_inputs)
                next_features = self._rl_features(
                    encoder_params, next_obs_inputs, stop_gradient=True
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
                    action_sequence = actions.reshape(
                        (
                            actions.shape[0],
                            self.action_sequence,
                            self.action_dim,
                        )
                    )
                    next_action = jnp.concatenate(
                        [action_sequence[:, 1:], action_sequence[:, -1:]],
                        axis=1,
                    )
                elif self.td_target_action_source == "bc_policy":
                    next_action, _ = self._policy_action(
                        current_params["policy"],
                        next_policy_features,
                        key=selection_key,
                    )
                elif self.td_target_action_source == "policy_value":
                    next_action, _ = self._flow_q_policy_value_action(
                        current_params["critic"],
                        next_features,
                        current_params["policy"],
                        next_policy_features,
                        selection_key,
                        policy_value_beta=(
                            self.td_target_policy_value_beta
                        ),
                    )
                elif self.flow_distill_action_readout:
                    next_action, _ = self._flow_distill_greedy_action(
                        current_params["flow_distill_readout"],
                        next_features,
                        selection_key,
                    )
                else:
                    next_action, _ = self._greedy_action_for_update(
                        current_params["critic"],
                        next_features,
                        selection_key,
                    )
                target_endpoints = None
                if self.pcbf_loss_coeff > 0.0:
                    target_endpoints = self._selected_endpoints_per_level(
                        target_critic_params,
                        next_features,
                        next_action,
                        target_source_key,
                        num_samples=self.num_target_flow_samples,
                    )
                targets = self._target_values(
                    target_critic_params,
                    next_features,
                    next_action,
                    rewards,
                    discounts,
                    bootstrap,
                    target_source_key,
                    target_endpoints=target_endpoints,
                )
                # Value Flows requires exact same-noise coupling between the
                # bootstrapped target sample and current BCFM interpolation.
                # Expected-value floq instead uses independent target-MC and
                # current-flow source samples.
                flow_source_key = (
                    train_source_key
                    if self.flow_iqn_quantile_coupling
                    else (
                        target_source_key
                        if self.value_mode == "return_sample"
                        else train_source_key
                    )
                )
                (
                    confidence_weights,
                    confidence_return_std,
                ) = self._value_flow_confidence_weights(
                    target_critic_params,
                    features,
                    actions,
                    jax.random.fold_in(target_source_key, 907),
                )
                flow_loss_weights = loss_weights * confidence_weights
                per_sample = jnp.zeros_like(loss_weights)
                if self.bcfm_lambda > 0.0:
                    per_sample = self._flow_matching_loss(
                        current_params["critic"],
                        features,
                        actions,
                        targets,
                        flow_source_key,
                        time_key,
                    )
                bcfm_loss = jnp.mean(per_sample * flow_loss_weights)
                dcfm_per_sample = jnp.zeros_like(per_sample)
                if self.dcfm_lambda > 0.0:
                    dcfm_per_sample = self._dcfm_loss(
                        current_params["critic"],
                        target_critic_params,
                        features,
                        next_features,
                        actions,
                        next_action,
                        rewards,
                        discounts,
                        bootstrap,
                        target_source_key,
                        time_key,
                    )
                dcfm_loss = jnp.mean(dcfm_per_sample * flow_loss_weights)
                evor_per_sample = jnp.zeros_like(per_sample)
                if self.evor_td_lambda > 0.0:
                    evor_per_sample = self._evor_td_loss(
                        current_params["critic"],
                        target_critic_params,
                        features,
                        next_features,
                        actions,
                        next_action,
                        rewards,
                        discounts,
                        bootstrap,
                        mc_returns,
                        jax.random.fold_in(target_source_key, 1701),
                        jax.random.fold_in(time_key, 1701),
                    )
                evor_td_loss = jnp.mean(
                    evor_per_sample * loss_weights
                )
                pcbf_per_sample = jnp.zeros_like(per_sample)
                if self.pcbf_loss_coeff > 0.0:
                    pcbf_per_sample = self._pcbf_loss(
                        current_params["critic"],
                        target_critic_params,
                        features,
                        next_features,
                        actions,
                        next_action,
                        rewards,
                        discounts,
                        bootstrap,
                        target_source_key,
                        time_key,
                        next_endpoints=target_endpoints,
                    )
                pcbf_loss = jnp.mean(pcbf_per_sample * flow_loss_weights)
                ce_per_sample = jnp.zeros_like(per_sample)
                fosd = jnp.zeros_like(per_sample)
                margin = jnp.zeros_like(per_sample)
                expert_top1 = jnp.zeros_like(per_sample)
                expert_q_gap = jnp.zeros_like(per_sample)
                demo_q_span = jnp.zeros_like(per_sample)
                demo_top2_gap = jnp.zeros_like(per_sample)
                demo_source_q_std = jnp.zeros_like(per_sample)
                demo_rank_snr = jnp.zeros_like(per_sample)
                demo_source_bin_flip_rate = jnp.zeros_like(per_sample)
                if self.quantile_endpoint_lambda > 0.0:
                    (
                        chosen_endpoints,
                        chosen_endpoint_quantiles,
                    ) = self._selected_endpoints_and_quantiles_per_level(
                        current_params["critic"],
                        features,
                        actions,
                        endpoint_key,
                        num_samples=self.num_flow_samples,
                    )
                else:
                    chosen_endpoints = self._selected_endpoints_per_level(
                        current_params["critic"],
                        features,
                        actions,
                        endpoint_key,
                        num_samples=self.num_flow_samples,
                    )
                    chosen_endpoint_quantiles = None
                if self.value_mode == "categorical":
                    ce = categorical_cross_entropy(
                        targets[:, None], chosen_endpoints
                    )
                    ce = self._sequence_training_slice(
                        ce,
                        sequence_axis=3,
                    )
                    ce_per_sample = ce.mean(axis=(1, 2, 3, 4))
                endpoint_ce = jnp.mean(ce_per_sample * loss_weights)
                atom_ce_loss = (
                    self.atom_ce_lambda * endpoint_ce
                    if self.atom_ce_lambda > 0.0
                    else jnp.asarray(0.0, dtype=bcfm_loss.dtype)
                )
                endpoint_q_samples = self._endpoint_q_samples(chosen_endpoints)
                endpoint_q_samples = self._sequence_training_slice(
                    endpoint_q_samples,
                    sequence_axis=3,
                )
                endpoint_q = endpoint_q_samples.mean(axis=1)
                target_q = self._target_expected_q(targets)
                target_q = self._sequence_training_slice(
                    target_q,
                    sequence_axis=2,
                )
                endpoint_q_per_sample = jnp.square(
                    endpoint_q - target_q
                ).mean(axis=(1, 2, 3))
                endpoint_q_loss = jnp.mean(
                    endpoint_q_per_sample * loss_weights
                )
                quantile_endpoint_per_sample = jnp.zeros_like(per_sample)
                quantile_endpoint_loss = jnp.asarray(
                    0.0, dtype=bcfm_loss.dtype
                )
                if self.quantile_endpoint_lambda > 0.0:
                    if chosen_endpoint_quantiles is None:
                        raise ValueError(
                            "Quantile endpoint loss requires source quantiles."
                        )
                    predicted_quantiles = self._sequence_training_slice(
                        chosen_endpoints[..., 0],
                        sequence_axis=3,
                    )
                    target_particles = self._sequence_training_slice(
                        targets,
                        sequence_axis=3,
                    )
                    endpoint_quantile_levels = self._sequence_training_slice(
                        chosen_endpoint_quantiles,
                        sequence_axis=3,
                    )
                    quantile_endpoint_per_sample = (
                        quantile_huber_endpoint_loss(
                            predicted_quantiles,
                            target_particles,
                            endpoint_quantile_levels,
                            kappa=self.quantile_huber_kappa,
                        )
                    )
                    quantile_endpoint_loss = jnp.mean(
                        quantile_endpoint_per_sample * loss_weights
                    )
                source_q_variance_per_sample = endpoint_q_samples.var(
                    axis=1
                ).mean(axis=(1, 2, 3))
                source_consistency_loss = jnp.mean(
                    source_q_variance_per_sample * loss_weights
                )
                flow_distill_per_sample = jnp.zeros_like(per_sample)
                flow_distill_loss = jnp.asarray(
                    0.0, dtype=bcfm_loss.dtype
                )
                flow_distill_mae = jnp.asarray(
                    0.0, dtype=bcfm_loss.dtype
                )
                flow_distill_q_span = jnp.asarray(
                    0.0, dtype=bcfm_loss.dtype
                )
                if self.flow_distill_lambda > 0.0:
                    # Match official FLOQ semantics: the scalar critic learns
                    # the mean return integrated by the *online* flow, and the
                    # target is detached so this auxiliary loss cannot turn
                    # into another endpoint loss on the velocity field.  The
                    # readout consumes detached shared features as the local
                    # memory-efficient analogue of FLOQ's separate encoder.
                    (
                        distilled_chosen_q,
                        distilled_all_q,
                    ) = self._flow_distill_outputs_per_level(
                        current_params["flow_distill_readout"],
                        jax.lax.stop_gradient(features),
                        actions,
                    )
                    distilled_chosen_q = self._sequence_training_slice(
                        distilled_chosen_q,
                        sequence_axis=2,
                    )
                    distilled_all_q = self._sequence_training_slice(
                        distilled_all_q,
                        sequence_axis=2,
                    )
                    distilled_target = jax.lax.stop_gradient(endpoint_q)
                    distill_error = distilled_chosen_q - distilled_target
                    flow_distill_per_sample = jnp.square(
                        distill_error
                    ).mean(axis=(1, 2, 3))
                    flow_distill_loss = jnp.mean(
                        flow_distill_per_sample * loss_weights
                    )
                    flow_distill_mae = jnp.mean(jnp.abs(distill_error))
                    flow_distill_q_span = jnp.mean(
                        distilled_all_q.max(axis=-1)
                        - distilled_all_q.min(axis=-1)
                    )
                td_critic_loss = self.critic_lambda * (
                    self.bcfm_lambda * bcfm_loss
                    + self.dcfm_lambda * dcfm_loss
                    + self.evor_td_lambda * evor_td_loss
                    + self.pcbf_loss_coeff * pcbf_loss
                    + atom_ce_loss
                    + self.quantile_endpoint_lambda
                    * quantile_endpoint_loss
                    + self.endpoint_q_lambda * endpoint_q_loss
                    + self.source_consistency_lambda
                    * source_consistency_loss
                    + self.flow_distill_lambda * flow_distill_loss
                )
                if self.value_mode == "categorical":
                    # Match direct C51's completed-return target in
                    # distribution space.  With zero bootstrap the projection
                    # is the two-atom interpolation of the scalar return; the
                    # input probabilities only provide the required normalized
                    # carrier shape.
                    flat_targets = targets.reshape(
                        (
                            targets.shape[0],
                            targets.shape[1],
                            self._flat_action_dim,
                            self.atoms,
                        )
                    )
                    mc_target_distribution = project_categorical(
                        flat_targets,
                        mc_returns,
                        jnp.zeros_like(discounts),
                        jnp.zeros_like(bootstrap),
                        self.support,
                    ).reshape(targets.shape)
                    mc_ce = categorical_cross_entropy(
                        mc_target_distribution[:, None],
                        chosen_endpoints,
                    )
                    mc_ce = self._sequence_training_slice(
                        mc_ce,
                        sequence_axis=3,
                    )
                    mc_per_sample = mc_ce.mean(axis=(1, 2, 3, 4))
                else:
                    mc_per_sample = jnp.square(
                        endpoint_q - mc_returns[:, None, None, None]
                    ).mean(axis=(1, 2, 3))
                mc_return_loss = self.mc_return_weight * jnp.mean(
                    mc_per_sample * loss_weights
                )
                mc_return_mae = jnp.mean(
                    jnp.abs(
                        endpoint_q - mc_returns[:, None, None, None]
                    )
                )
                critic_loss = td_critic_loss + mc_return_loss
                policy_loss = jnp.asarray(0.0, dtype=critic_loss.dtype)
                policy_ce = jnp.asarray(0.0, dtype=critic_loss.dtype)
                policy_demo_top1 = jnp.asarray(
                    0.0, dtype=critic_loss.dtype
                )
                policy_entropy = jnp.asarray(0.0, dtype=critic_loss.dtype)

                if self.bc_lambda > 0.0 and not self.separate_bc_policy:
                    def endpoint_demo_losses(
                        demo_features,
                        demo_actions,
                        start,
                    ):
                        demo_chosen, demo_all = self._endpoints_per_level(
                            current_params["critic"],
                            demo_features,
                            demo_actions,
                            demo_key,
                            num_flow_steps=self.demo_flow_steps,
                        )
                        (
                            sub_fosd,
                            sub_margin,
                            sub_top1,
                            sub_q_gap,
                            sub_q_span,
                            sub_top2_gap,
                            sub_source_q_std,
                            sub_rank_snr,
                            sub_source_bin_flip_rate,
                        ) = self._demo_losses(
                            demo_all,
                            demo_chosen,
                        )
                        full_fosd = jnp.zeros_like(per_sample).at[start:].set(
                            sub_fosd
                        )
                        full_margin = jnp.zeros_like(per_sample).at[start:].set(
                            sub_margin
                        )
                        full_top1 = jnp.zeros_like(per_sample).at[start:].set(
                            sub_top1
                        )
                        full_q_gap = jnp.zeros_like(per_sample).at[start:].set(
                            sub_q_gap
                        )
                        full_q_span = jnp.zeros_like(per_sample).at[start:].set(
                            sub_q_span
                        )
                        full_top2_gap = jnp.zeros_like(per_sample).at[start:].set(
                            sub_top2_gap
                        )
                        full_source_q_std = jnp.zeros_like(per_sample).at[
                            start:
                        ].set(sub_source_q_std)
                        full_rank_snr = jnp.zeros_like(per_sample).at[start:].set(
                            sub_rank_snr
                        )
                        full_source_bin_flip_rate = jnp.zeros_like(
                            per_sample
                        ).at[start:].set(sub_source_bin_flip_rate)
                        return (
                            full_fosd,
                            full_margin,
                            full_top1,
                            full_q_gap,
                            full_q_span,
                            full_top2_gap,
                            full_source_q_std,
                            full_rank_snr,
                            full_source_bin_flip_rate,
                        )

                    def calculate_demo_losses(_):
                        if self.demo_batch_size is None:
                            return endpoint_demo_losses(features, actions, 0)
                        demo_rows = min(
                            self.demo_batch_size,
                            features.shape[0],
                        )
                        start = features.shape[0] - demo_rows

                        def all_rows(_):
                            return endpoint_demo_losses(features, actions, 0)

                        def appended_demo_rows(_):
                            return endpoint_demo_losses(
                                features[-demo_rows:],
                                actions[-demo_rows:],
                                start,
                            )

                        # Self-imitation samples can also appear in the online
                        # replay prefix. Preserve their loss exactly when they
                        # are present; otherwise avoid all-bin BPTT for rows
                        # known to be ordinary replay.
                        return jax.lax.cond(
                            jnp.any(demos[:start] > 0.0),
                            all_rows,
                            appended_demo_rows,
                            operand=None,
                        )

                    def zero_demo_losses(_):
                        zeros = jnp.zeros_like(per_sample)
                        return (zeros,) * 9

                    (
                        fosd,
                        margin,
                        expert_top1,
                        expert_q_gap,
                        demo_q_span,
                        demo_top2_gap,
                        demo_source_q_std,
                        demo_rank_snr,
                        demo_source_bin_flip_rate,
                    ) = jax.lax.cond(
                        jnp.any(demos > 0.0),
                        calculate_demo_losses,
                        zero_demo_losses,
                        operand=None,
                    )
                    demo_count = jnp.maximum(jnp.sum(demos), 1.0)
                    critic_loss = critic_loss + self.bc_lambda * (
                        jnp.sum((fosd + margin) * demos) / demo_count
                    )

                if self.separate_bc_policy:
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
                    policy_per_sample = -expert_log_probabilities.mean(
                        axis=(1, 2)
                    )
                    demo_count = jnp.maximum(jnp.sum(demos), 1.0)
                    policy_ce = jnp.sum(policy_per_sample * demos) / demo_count
                    policy_loss = self.bc_lambda * policy_ce
                    policy_correct = (
                        jnp.argmax(policy_logits, axis=-1) == expert_bins
                    ).astype(jnp.float32).mean(axis=(1, 2))
                    policy_demo_top1 = (
                        jnp.sum(policy_correct * demos) / demo_count
                    )
                    policy_probabilities = jax.nn.softmax(
                        policy_logits,
                        axis=-1,
                    )
                    policy_entropy = -jnp.sum(
                        policy_probabilities
                        * jnp.log(jnp.maximum(policy_probabilities, 1e-9)),
                        axis=-1,
                    ).mean()

                optimized_policy_loss = (
                    jax.lax.stop_gradient(policy_loss)
                    if self.freeze_bc_policy
                    else policy_loss
                )
                total_loss = critic_loss + optimized_policy_loss

                if self.value_mode == "categorical":
                    target_entropy_terms = -jnp.sum(
                        targets * jnp.log(jnp.maximum(targets, 1e-9)), axis=-1
                    )
                    target_entropy = jnp.mean(
                        target_entropy_terms.mean(axis=(1, 2, 3))
                        * loss_weights
                    )
                    predicted_probabilities = flow_logits_to_probabilities(
                        chosen_endpoints
                    )
                    predicted_entropy_terms = -jnp.sum(
                        predicted_probabilities
                        * jnp.log(jnp.maximum(predicted_probabilities, 1e-9)),
                        axis=-1,
                    )
                    predicted_entropy = jnp.mean(
                        predicted_entropy_terms.mean(axis=(1, 2, 3, 4))
                        * loss_weights
                    )
                    target_logits = centered_log_probabilities(targets)
                    target_logit_rms = jnp.sqrt(jnp.mean(jnp.square(target_logits)))
                    target_logit_abs_max = jnp.max(jnp.abs(target_logits))
                    target_probability_floor_fraction = jnp.mean(
                        targets <= 1e-8
                    )
                else:
                    target_entropy = jnp.asarray(0.0, dtype=features.dtype)
                    predicted_entropy = jnp.asarray(0.0, dtype=features.dtype)
                    target_logit_rms = jnp.asarray(0.0, dtype=features.dtype)
                    target_logit_abs_max = jnp.asarray(0.0, dtype=features.dtype)
                    target_probability_floor_fraction = jnp.asarray(
                        0.0, dtype=features.dtype
                    )
                endpoint_kl = jnp.maximum(endpoint_ce - target_entropy, 0.0)
                source_q_std = jnp.sqrt(
                    jnp.maximum(source_consistency_loss, 0.0)
                )
                target_return_std = (
                    targets.std(axis=1).mean()
                    if self.value_mode == "return_sample"
                    else jnp.asarray(0.0, dtype=features.dtype)
                )
                return total_loss, (
                    per_sample,
                    dcfm_per_sample,
                    evor_per_sample,
                    pcbf_per_sample,
                    quantile_endpoint_per_sample,
                    bcfm_loss,
                    dcfm_loss,
                    evor_td_loss,
                    pcbf_loss,
                    atom_ce_loss,
                    endpoint_ce,
                    quantile_endpoint_loss,
                    endpoint_q_loss,
                    source_consistency_loss,
                    source_q_std,
                    flow_distill_loss,
                    flow_distill_mae,
                    flow_distill_q_span,
                    endpoint_q.mean(),
                    endpoint_q.min(),
                    endpoint_q.max(),
                    target_q.mean(),
                    target_return_std,
                    fosd,
                    margin,
                    expert_top1,
                    expert_q_gap,
                    demo_q_span,
                    demo_top2_gap,
                    demo_source_q_std,
                    demo_rank_snr,
                    demo_source_bin_flip_rate,
                    target_entropy,
                    predicted_entropy,
                    endpoint_kl,
                    target_logit_rms,
                    target_logit_abs_max,
                    target_probability_floor_fraction,
                    confidence_weights,
                    confidence_return_std,
                    td_critic_loss,
                    mc_return_loss,
                    mc_return_mae,
                    policy_loss,
                    policy_ce,
                    policy_demo_top1,
                    policy_entropy,
                )

            (total_loss, aux), grads = jax.value_and_grad(
                loss_fn, has_aux=True
            )(params)
            # Keep the pre-transform norm visible: PCBF in particular can
            # develop a finite-loss / exploding-gradient failure between log
            # intervals.  The optimizer may clip this value, but logging the
            # raw norm is what lets a short correctness gate detect it.
            grad_norm = self.optax.tree.norm(grads)
            critic_grads = grads["critic"]
            critic_grad_norm = self.optax.tree.norm(critic_grads)
            flow_distill_readout_grad_norm = (
                self.optax.tree.norm(grads["flow_distill_readout"])
                if "flow_distill_readout" in grads
                else jnp.asarray(0.0, dtype=grad_norm.dtype)
            )
            encoder_grad_norm = (
                self.optax.tree.norm(grads["encoder"])
                if "encoder" in grads
                else jnp.asarray(0.0, dtype=grad_norm.dtype)
            )
            policy_encoder_grad_norm = (
                self.optax.tree.norm(grads["policy_encoder"])
                if "policy_encoder" in grads
                else jnp.asarray(0.0, dtype=grad_norm.dtype)
            )
            velocity_head_grads = {
                name: value
                for name, value in critic_grads["params"].items()
                if "velocity_head" in name
            }
            velocity_head_grad_norm = self.optax.tree.norm(velocity_head_grads)

            def nonfinite_fraction(tree):
                leaves = jax.tree.leaves(tree)
                nonfinite = sum(jnp.sum(~jnp.isfinite(leaf)) for leaf in leaves)
                size = sum(leaf.size for leaf in leaves)
                return nonfinite.astype(jnp.float32) / float(max(size, 1))

            critic_grad_nonfinite_fraction = nonfinite_fraction(critic_grads)
            encoder_grad_nonfinite_fraction = (
                nonfinite_fraction(grads["encoder"])
                if "encoder" in grads
                else jnp.asarray(0.0, dtype=jnp.float32)
            )
            updates, opt_state = optimizer.update(grads, opt_state, params)
            update_norm = self.optax.tree.norm(updates)
            updated_params = self.optax.apply_updates(params, updates)
            if self.freeze_bc_policy:
                updated_params = {
                    **updated_params,
                    "policy": params["policy"],
                    "policy_encoder": params["policy_encoder"],
                }
            params = updated_params
            target_critic_params = jax.tree.map(
                lambda target, online: (1.0 - target_tau) * target
                + target_tau * online,
                target_critic_params,
                params["critic"],
            )
            (
                per_sample,
                dcfm_per_sample,
                evor_per_sample,
                pcbf_per_sample,
                quantile_endpoint_per_sample,
                bcfm_loss,
                dcfm_loss,
                evor_td_loss,
                pcbf_loss,
                atom_ce_loss,
                endpoint_ce,
                quantile_endpoint_loss,
                endpoint_q_loss,
                source_consistency_loss,
                source_q_std,
                flow_distill_loss,
                flow_distill_mae,
                flow_distill_q_span,
                endpoint_q_mean,
                endpoint_q_min,
                endpoint_q_max,
                target_q_mean,
                target_return_std,
                fosd,
                margin,
                expert_top1,
                expert_q_gap,
                demo_q_span,
                demo_top2_gap,
                demo_source_q_std,
                demo_rank_snr,
                demo_source_bin_flip_rate,
                entropy,
                predicted_entropy,
                endpoint_kl,
                target_logit_rms,
                target_logit_abs_max,
                target_probability_floor_fraction,
                confidence_weights,
                confidence_return_std,
                td_critic_loss,
                mc_return_loss,
                mc_return_mae,
                policy_loss,
                policy_ce,
                policy_demo_top1,
                policy_entropy,
            ) = aux
            critic_loss = td_critic_loss + mc_return_loss
            priority_error = (
                self.bcfm_lambda * per_sample
                + self.dcfm_lambda * dcfm_per_sample
                + self.evor_td_lambda * evor_per_sample
                + self.pcbf_loss_coeff * pcbf_per_sample
                + self.quantile_endpoint_lambda
                * quantile_endpoint_per_sample
            )
            priority = jnp.sqrt(jnp.maximum(priority_error, 0.0) + 1e-10)
            priority = priority / jnp.maximum(jnp.max(priority), 1e-10)
            demo_count = jnp.maximum(jnp.sum(demos), 1.0)
            return (
                params,
                target_critic_params,
                opt_state,
                priority,
                {
                    "critic_loss": critic_loss,
                    "td_critic_loss": td_critic_loss,
                    "mc_return_loss": mc_return_loss,
                    "mc_return_mae": mc_return_mae,
                    "mc_return_mean": jnp.mean(mc_returns),
                    "policy_bc_loss": policy_loss,
                    "policy_ce": policy_ce,
                    "policy_demo_top1": policy_demo_top1,
                    "policy_entropy": policy_entropy,
                    "total_loss": total_loss,
                    "critic_grad_norm": grad_norm,
                    "flow_critic_grad_norm": critic_grad_norm,
                    "flow_distill_readout_grad_norm": (
                        flow_distill_readout_grad_norm
                    ),
                    "encoder_grad_norm": encoder_grad_norm,
                    "policy_encoder_grad_norm": policy_encoder_grad_norm,
                    "velocity_head_grad_norm": velocity_head_grad_norm,
                    "flow_critic_grad_nonfinite_fraction": (
                        critic_grad_nonfinite_fraction
                    ),
                    "encoder_grad_nonfinite_fraction": (
                        encoder_grad_nonfinite_fraction
                    ),
                    "critic_update_norm": update_norm,
                    # Keep flow_loss as a compatibility alias for the original
                    # endpoint/bootstrapped conditional FM objective.
                    "flow_loss": bcfm_loss,
                    "bcfm_loss": bcfm_loss,
                    "dcfm_loss": dcfm_loss,
                    "evor_td_loss": evor_td_loss,
                    "pcbf_loss": pcbf_loss,
                    "atom_ce_loss": atom_ce_loss,
                    "endpoint_ce": endpoint_ce,
                    "endpoint_kl": endpoint_kl,
                    "quantile_endpoint_loss": quantile_endpoint_loss,
                    "endpoint_q_loss": endpoint_q_loss,
                    "source_consistency_loss": source_consistency_loss,
                    "source_q_std": source_q_std,
                    "flow_distill_loss": flow_distill_loss,
                    "flow_distill_mae": flow_distill_mae,
                    "flow_distill_q_span": flow_distill_q_span,
                    "endpoint_q_mean": endpoint_q_mean,
                    "endpoint_q_min": endpoint_q_min,
                    "endpoint_q_max": endpoint_q_max,
                    "target_q_mean": target_q_mean,
                    "target_return_std": target_return_std,
                    "confidence_weight_mean": jnp.mean(confidence_weights),
                    "confidence_weight_min": jnp.min(confidence_weights),
                    "confidence_weight_max": jnp.max(confidence_weights),
                    "confidence_weight_std": jnp.std(confidence_weights),
                    "confidence_return_std_mean": jnp.mean(
                        confidence_return_std
                    ),
                    "confidence_return_std_min": jnp.min(
                        confidence_return_std
                    ),
                    "confidence_return_std_max": jnp.max(
                        confidence_return_std
                    ),
                    "demo_fosd_loss": jnp.sum(fosd * demos) / demo_count,
                    "demo_margin_loss": jnp.sum(margin * demos) / demo_count,
                    "demo_expert_top1": (
                        jnp.sum(expert_top1 * demos) / demo_count
                    ),
                    "demo_expert_q_gap": (
                        jnp.sum(expert_q_gap * demos) / demo_count
                    ),
                    "demo_q_span": jnp.sum(demo_q_span * demos) / demo_count,
                    "demo_top2_gap": (
                        jnp.sum(demo_top2_gap * demos) / demo_count
                    ),
                    "demo_source_q_std": (
                        jnp.sum(demo_source_q_std * demos) / demo_count
                    ),
                    "demo_rank_snr": (
                        jnp.sum(demo_rank_snr * demos) / demo_count
                    ),
                    # Fixed-condition/level diagnostic, not full rollout flips.
                    "demo_source_bin_flip_rate": (
                        jnp.sum(demo_source_bin_flip_rate * demos) / demo_count
                    ),
                    "target_entropy": entropy,
                    "predicted_entropy": predicted_entropy,
                    "target_logit_rms": target_logit_rms,
                    "target_logit_abs_max": target_logit_abs_max,
                    "target_probability_floor_fraction": (
                        target_probability_floor_fraction
                    ),
                    "loss_coeff": jnp.mean(loss_weights),
                },
            )

        if self.separate_bc_policy:
            return core_update_fn

        def legacy_update_fn(
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
            return core_update_fn(
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
                jnp.zeros_like(rewards),
                action_key,
            )

        return legacy_update_fn


__all__ = [
    "C2FSequenceFlowCritic",
    "CQNFlowAS",
    "CQNFlowSpec",
    "EVORTDTrainingPair",
    "FlowIQNTrainingPair",
    "PCBFTrainingPair",
    "categorical_cross_entropy",
    "centered_log_probabilities",
    "cqn_flow_spec_from_cfg",
    "demo_fosd_per_sample",
    "demo_margin_per_sample",
    "evor_velocity_td_pair",
    "expected_q",
    "flow_logits_to_probabilities",
    "hl_gauss_encode",
    "integrate_value_flow",
    "integrate_value_flow_with_source_jvp",
    "path_coupled_bellman_flow_pair",
    "quantile_couple_return_samples",
    "quantile_huber_endpoint_loss",
    "scalar_to_categorical",
    "source_bin_flip_rate_per_sample",
    "sibling_bin_candidate_plans",
    "select_single_supported_lcb_plan",
    "supported_lcb_action_indices",
]
