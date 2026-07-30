"""Pure-JAX source paths shared by flow-based imitation methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np


Array = jax.Array
ScheduleKind = Literal["hard", "linear", "cosine"]


class FlowTrainingPair(NamedTuple):
    """A noised model input and the velocity used to supervise it."""

    sample: Array
    target_velocity: Array
    schedule_channel: Array | None = None


class FlowInferenceSource(NamedTuple):
    """A source presented to the model during one inference step."""

    sample: Array
    schedule_channel: Array | None = None


class FlowSource(Protocol):
    """Common pure-JAX source/path contract used by flow policies."""

    def build_training_pair(
        self,
        key: Array,
        target: Array,
        tau: Array | float,
        *,
        source: Array | None = None,
        omega: Array | float | None = None,
        dt: Array | float | None = None,
    ) -> FlowTrainingPair: ...

    def build_inference_source(
        self,
        key: Array,
        *,
        shape: tuple[int, ...] | None = None,
        dtype=jnp.float32,
        source: Array | None = None,
        reference: Array | None = None,
        omega: Array | float | None = None,
    ) -> FlowInferenceSource: ...


def _time_like_actions(time: Array | float, actions: Array) -> Array:
    """Broadcast a scalar or leading-batch time over action horizon/features."""
    time = jnp.asarray(time, dtype=actions.dtype)
    if time.ndim > actions.ndim:
        raise ValueError(f"time rank {time.ndim} exceeds action rank {actions.ndim}.")
    return jnp.reshape(time, (*time.shape, *((1,) * (actions.ndim - time.ndim))))


def _schedule_channel(omega: Array | float, actions: Array) -> Array:
    """Normalize a horizon-wise paper omega to ``(..., horizon, 1)``."""
    if actions.ndim < 2:
        raise ValueError("actions must have at least horizon and feature axes.")

    omega = jnp.asarray(omega, dtype=actions.dtype)
    channel_shape = (*actions.shape[:-1], 1)
    if omega.ndim == 0:
        omega = jnp.broadcast_to(omega, channel_shape)
    else:
        if omega.ndim == 1 and omega.shape[0] == actions.shape[-2]:
            omega = omega[..., None]
        elif omega.ndim == actions.ndim - 1:
            omega = omega[..., None]
        try:
            omega = jnp.broadcast_to(omega, channel_shape)
        except ValueError as exc:
            raise ValueError(
                "omega must be scalar or broadcastable to the action horizon "
                f"shape {channel_shape}; got {omega.shape}."
            ) from exc
    return jnp.ones_like(omega) - omega


def linear_flow_training_pair(
    source: Array,
    target: Array,
    tau: Array | float,
) -> FlowTrainingPair:
    """Construct the repository's reverse-time linear flow path.

    ``tau=1`` is the source and ``tau=0`` is the target. Sampling integrates
    with positive ``target - source`` velocity while tau decreases.
    """
    source = jnp.asarray(source)
    target = jnp.asarray(target, dtype=source.dtype)
    if source.shape != target.shape:
        raise ValueError(
            f"source and target shapes must match; got {source.shape} and "
            f"{target.shape}."
        )
    tau = _time_like_actions(tau, target)
    return FlowTrainingPair(
        sample=tau * source + (jnp.ones_like(tau) - tau) * target,
        target_velocity=target - source,
    )


def gaussian_flow_training_pair(
    key: Array,
    target: Array,
    tau: Array | float,
) -> FlowTrainingPair:
    """Construct a standard Gaussian-to-data flow-matching pair."""
    target = jnp.asarray(target)
    source = jax.random.normal(key, target.shape, dtype=target.dtype)
    return linear_flow_training_pair(source, target, tau)


def a2a_flow_training_pair(
    source_actions: Array,
    target_actions: Array,
    tau: Array | float,
) -> FlowTrainingPair:
    """Construct an action-to-action pair on the common linear path."""
    return linear_flow_training_pair(source_actions, target_actions, tau)


def legato_schedule(
    horizon: int,
    delay: Array | int,
    ramp: Array | int,
    *,
    start: Array | int = 0,
    kind: ScheduleKind = "linear",
    dtype=jnp.float32,
) -> Array:
    """Build the paper's horizon-wise guidance strength ``omega``.

    ``start``, ``delay``, and ``ramp`` may be scalars or broadcastable batch
    arrays. The result has shape ``broadcast(start, delay, ramp).shape +
    (horizon, 1)``. Paper convention is preserved: one means fully guided and
    zero means unguided. Linear and cosine schedules are one at the start of
    the ramp and reach zero at its continuous endpoint ``start + delay +
    ramp``.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be positive; got {horizon}.")
    if kind not in {"hard", "linear", "cosine"}:
        raise ValueError(
            f"Unsupported Legato schedule kind {kind!r}; expected hard, "
            "linear, or cosine."
        )

    concrete_values = (start, delay, ramp)
    if not any(isinstance(value, jax.core.Tracer) for value in concrete_values):
        start_np, delay_np, ramp_np = np.broadcast_arrays(
            *(np.asarray(value) for value in concrete_values)
        )
        if (
            np.any(start_np < 0)
            or np.any(delay_np < 0)
            or np.any(ramp_np < 0)
            or np.any(start_np + delay_np + ramp_np > horizon)
        ):
            raise ValueError(
                "Legato schedule requires non-negative start/delay/ramp and "
                "start + delay + ramp <= horizon."
            )

    start, delay, ramp = jnp.broadcast_arrays(
        jnp.asarray(start, dtype=dtype),
        jnp.asarray(delay, dtype=dtype),
        jnp.asarray(ramp, dtype=dtype),
    )
    valid_bounds = (
        (start >= 0) & (delay >= 0) & (ramp >= 0) & (start + delay + ramp <= horizon)
    )
    position = jnp.arange(horizon, dtype=dtype)
    position = jnp.reshape(position, (*((1,) * start.ndim), horizon))
    start = start[..., None]
    delay = delay[..., None]
    ramp = ramp[..., None]
    relative_position = position - start
    in_prefix = (relative_position >= 0) & (relative_position < delay)

    if kind == "hard":
        omega = in_prefix.astype(dtype)
    else:
        safe_ramp = jnp.maximum(ramp, jnp.asarray(1, dtype=dtype))
        progress = jnp.clip((relative_position - delay) / safe_ramp, 0.0, 1.0)
        if kind == "linear":
            decay = 1.0 - progress
        else:
            decay = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
        in_ramp = (relative_position >= delay) & (relative_position < delay + ramp)
        omega = jnp.where(in_prefix, 1.0, jnp.where(in_ramp, decay, 0.0))
    # Traced values cannot raise a Python exception without requiring callers
    # to use checkify. Surface invalid dynamic bounds as an explicit sentinel;
    # FlowMatching validates all configured sampling bounds before tracing.
    omega = jnp.where(valid_bounds[..., None], omega, jnp.nan)
    return omega[..., None]


def legato_training_pair(
    target: Array,
    noise: Array,
    tau: Array | float,
    omega: Array | float,
    dt: Array | float,
) -> FlowTrainingPair:
    """Construct the Legato v2 training path in repository time convention.

    The paper uses data time ``t`` from zero to one. Here ``tau = 1 - t`` so
    tau runs from the effective source to data, matching ``flow_matching.py``.
    """
    target = jnp.asarray(target)
    noise = jnp.asarray(noise, dtype=target.dtype)
    if target.shape != noise.shape:
        raise ValueError(
            f"target and noise shapes must match; got {target.shape} and {noise.shape}."
        )

    schedule_channel = _schedule_channel(omega, target)
    omega_actions = jnp.broadcast_to(
        jnp.ones_like(schedule_channel) - schedule_channel,
        target.shape,
    )
    tau_actions = _time_like_actions(tau, target)
    dt = jnp.asarray(dt, dtype=target.dtype)
    effective_source = omega_actions * target + (1.0 - omega_actions) * noise
    sample = tau_actions * effective_source + (1.0 - tau_actions) * target
    kappa = omega_actions / dt
    target_velocity = (1.0 - kappa * tau_actions) * (target - noise)
    return FlowTrainingPair(sample, target_velocity, schedule_channel)


def legato_inference_source(
    sample: Array,
    reference_actions: Array,
    omega: Array | float,
) -> FlowInferenceSource:
    """Apply per-step Legato guidance before evaluating the velocity model."""
    sample = jnp.asarray(sample)
    reference_actions = jnp.asarray(reference_actions, dtype=sample.dtype)
    if sample.shape != reference_actions.shape:
        raise ValueError(
            "sample and reference action shapes must match; got "
            f"{sample.shape} and {reference_actions.shape}."
        )
    schedule_channel = _schedule_channel(omega, sample)
    omega_actions = jnp.broadcast_to(
        jnp.ones_like(schedule_channel) - schedule_channel,
        sample.shape,
    )
    guided_sample = (1.0 - omega_actions) * sample + omega_actions * reference_actions
    return FlowInferenceSource(guided_sample, schedule_channel)


def guided_euler_step(
    sample: Array,
    reference_actions: Array,
    velocity: Array,
    omega: Array | float,
    dt: Array | float,
) -> Array:
    """Apply Legato guidance and one positive-velocity Euler update."""
    source = legato_inference_source(sample, reference_actions, omega)
    velocity = jnp.asarray(velocity, dtype=source.sample.dtype)
    if velocity.shape != source.sample.shape:
        raise ValueError(
            f"velocity shape must match sample; got {velocity.shape} and "
            f"{source.sample.shape}."
        )
    return source.sample + jnp.asarray(dt, dtype=source.sample.dtype) * velocity


@dataclass(frozen=True)
class GaussianFlowSource:
    """Standard Gaussian source using the repository's linear path."""

    def build_training_pair(
        self,
        key,
        target,
        tau,
        *,
        source=None,
        omega=None,
        dt=None,
    ):
        del source, omega, dt
        return gaussian_flow_training_pair(key, target, tau)

    def build_inference_source(
        self,
        key,
        *,
        shape=None,
        dtype=jnp.float32,
        source=None,
        reference=None,
        omega=None,
    ):
        del source, reference, omega
        if shape is None:
            raise ValueError("Gaussian inference requires shape.")
        return FlowInferenceSource(jax.random.normal(key, shape, dtype=dtype))


@dataclass(frozen=True)
class A2AFlowSource:
    """Deterministic encoded-action source for Action-to-Action FM."""

    def build_training_pair(
        self,
        key,
        target,
        tau,
        *,
        source=None,
        omega=None,
        dt=None,
    ):
        del key, omega, dt
        if source is None:
            raise ValueError("A2A training requires encoded source actions.")
        return a2a_flow_training_pair(source, target, tau)

    def build_inference_source(
        self,
        key,
        *,
        shape=None,
        dtype=jnp.float32,
        source=None,
        reference=None,
        omega=None,
    ):
        del key, shape, dtype, reference, omega
        if source is None:
            raise ValueError("A2A inference requires encoded source actions.")
        return FlowInferenceSource(jnp.asarray(source))


@dataclass(frozen=True)
class LegatoFlowSource:
    """Legato v2 source/path with paper or public-code target semantics."""

    target_mode: Literal["paper_minus", "public_kinetix_plus"] = "paper_minus"

    def build_training_pair(
        self,
        key,
        target,
        tau,
        *,
        source=None,
        omega=None,
        dt=None,
    ):
        del source
        if omega is None or dt is None:
            raise ValueError("Legato training requires omega and dt.")
        noise = jax.random.normal(key, target.shape, dtype=target.dtype)
        pair = legato_training_pair(target, noise, tau, omega, dt)
        if self.target_mode == "paper_minus":
            return pair
        tau_actions = _time_like_actions(tau, target)
        omega_actions = jnp.broadcast_to(1.0 - pair.schedule_channel, target.shape)
        public_target = (
            1.0 + omega_actions / jnp.asarray(dt, dtype=target.dtype) * tau_actions
        ) * (target - noise)
        return FlowTrainingPair(pair.sample, public_target, pair.schedule_channel)

    def build_inference_source(
        self,
        key,
        *,
        shape=None,
        dtype=jnp.float32,
        source=None,
        reference=None,
        omega=None,
    ):
        del source
        if shape is None or reference is None or omega is None:
            raise ValueError("Legato inference requires shape, reference, and omega.")
        noise = jax.random.normal(key, shape, dtype=dtype)
        return legato_inference_source(noise, reference, omega)


__all__ = [
    "A2AFlowSource",
    "FlowInferenceSource",
    "FlowSource",
    "FlowTrainingPair",
    "GaussianFlowSource",
    "LegatoFlowSource",
    "a2a_flow_training_pair",
    "gaussian_flow_training_pair",
    "guided_euler_step",
    "legato_inference_source",
    "legato_schedule",
    "legato_training_pair",
    "linear_flow_training_pair",
]
