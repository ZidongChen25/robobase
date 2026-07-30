"""Delay-aware BiGym rollout wrapper around the pinned official JAX core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

from flax import nnx
import jax
import jax.numpy as jnp

from benchmarks.official_bigym.legato_upstream import load_upstream_module


PolicyMode = Literal["vanilla", "rtc", "legato"]


@dataclass(frozen=True)
class OfficialPolicyConfig:
    """Shared architecture plus official Kinetix rollout settings."""

    action_horizon: int = 8
    execute_horizon: int = 4
    inference_delay: int = 0
    num_flow_steps: int = 5
    channel_dim: int = 256
    channel_hidden_dim: int = 512
    token_hidden_dim: int = 64
    num_layers: int = 4
    warmup_min: int = 0
    warmup_max: int = 4
    warmup_sampling: str = "exp"
    rtc_prefix_schedule: str = "exp"
    rtc_max_guidance_weight: float = 5.0

    def __post_init__(self) -> None:
        positive = {
            "action_horizon": self.action_horizon,
            "execute_horizon": self.execute_horizon,
            "num_flow_steps": self.num_flow_steps,
            "channel_dim": self.channel_dim,
            "channel_hidden_dim": self.channel_hidden_dim,
            "token_hidden_dim": self.token_hidden_dim,
            "num_layers": self.num_layers,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Configuration values must be positive: {invalid}.")
        if not 0 <= self.inference_delay <= self.execute_horizon:
            raise ValueError("inference_delay must lie in [0, execute_horizon].")
        if self.execute_horizon + self.inference_delay > self.action_horizon:
            raise ValueError(
                "Official delayed execution requires execute_horizon + "
                "inference_delay <= action_horizon."
            )
        if not 0 <= self.warmup_min <= self.warmup_max <= self.action_horizon:
            raise ValueError("warmup bounds must lie within the action horizon.")
        if self.warmup_sampling not in {"bell", "uniform", "exp"}:
            raise ValueError("warmup_sampling must be bell, uniform, or exp.")
        if self.rtc_prefix_schedule not in {"linear", "exp", "ones", "zeros"}:
            raise ValueError("Unsupported official RTC prefix schedule.")
        if self.rtc_max_guidance_weight <= 0:
            raise ValueError("rtc_max_guidance_weight must be positive.")


class DelayState(NamedTuple):
    """Previously generated chunk aligned to the current control time."""

    previous_chunk: jax.Array
    valid: jax.Array


class PolicyPrediction(NamedTuple):
    execute_actions: jax.Array
    generated_chunk: jax.Array
    next_state: DelayState


def initial_delay_state(
    batch_size: int,
    action_horizon: int,
    action_dim: int,
    *,
    dtype=jnp.float32,
) -> DelayState:
    if min(batch_size, action_horizon, action_dim) <= 0:
        raise ValueError("Delay-state dimensions must be positive.")
    return DelayState(
        jnp.zeros((batch_size, action_horizon, action_dim), dtype=dtype),
        jnp.zeros((batch_size,), dtype=jnp.bool_),
    )


def shift_generated_chunk(chunk: jax.Array, execute_horizon: int) -> jax.Array:
    """Match the official evaluator's truncate-and-zero-pad state update."""
    if chunk.ndim != 3:
        raise ValueError(f"chunk must have shape [B,H,A], got {chunk.shape}.")
    horizon = chunk.shape[1]
    if not 0 < execute_horizon <= horizon:
        raise ValueError("execute_horizon must lie in [1, action_horizon].")
    padding = jnp.zeros_like(chunk[:, :execute_horizon])
    return jnp.concatenate([chunk[:, execute_horizon:], padding], axis=1)


def merge_delayed_execution(
    previous_chunk: jax.Array,
    generated_chunk: jax.Array,
    *,
    inference_delay: int,
    execute_horizon: int,
) -> jax.Array:
    """Execute delayed actions from the old chunk and the rest from the new one."""
    if previous_chunk.shape != generated_chunk.shape or previous_chunk.ndim != 3:
        raise ValueError("previous and generated chunks must have equal [B,H,A] shape.")
    if not 0 <= inference_delay <= execute_horizon <= generated_chunk.shape[1]:
        raise ValueError("Invalid inference/execute horizons.")
    return jnp.concatenate(
        [
            previous_chunk[:, :inference_delay],
            generated_chunk[:, inference_delay:execute_horizon],
        ],
        axis=1,
    )


class OfficialBigymPolicy:
    """Run the unmodified official model/loss behind a BiGym delay boundary."""

    def __init__(
        self,
        *,
        mode: PolicyMode,
        obs_dim: int,
        action_dim: int,
        config: OfficialPolicyConfig,
        seed: int = 0,
        verify_upstream: bool = True,
    ):
        if mode not in {"vanilla", "rtc", "legato"}:
            raise ValueError(f"Unsupported policy mode: {mode!r}.")
        if obs_dim <= 0 or action_dim <= 0:
            raise ValueError("obs_dim and action_dim must be positive.")
        self.mode = mode
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.config = config
        upstream_kind = "legato" if mode == "legato" else "vanilla"
        self.upstream = load_upstream_module(
            upstream_kind, verify_commit=verify_upstream
        )
        common = dict(
            channel_dim=config.channel_dim,
            channel_hidden_dim=config.channel_hidden_dim,
            token_hidden_dim=config.token_hidden_dim,
            num_layers=config.num_layers,
            action_chunk_size=config.action_horizon,
        )
        if mode == "legato":
            model_config = self.upstream.ModelConfig(
                **common,
                warmup_min=config.warmup_min,
                warmup_max=config.warmup_max,
                inference_num_steps=config.num_flow_steps,
                warmup_sampling=config.warmup_sampling,
            )
        else:
            model_config = self.upstream.ModelConfig(**common, simulated_delay=None)
        self.policy = self.upstream.FlowPolicy(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            config=model_config,
            rngs=nnx.Rngs(seed),
        )

        steps = config.num_flow_steps
        if mode == "vanilla":
            self._sample = nnx.jit(
                lambda policy, key, obs: policy.action(key, obs, steps)
            )
        elif mode == "rtc":
            delay = config.inference_delay
            prefix_horizon = config.action_horizon - config.execute_horizon
            schedule = config.rtc_prefix_schedule
            max_weight = config.rtc_max_guidance_weight
            self._sample = nnx.jit(
                lambda policy, key, obs, previous: policy.realtime_action(
                    key,
                    obs,
                    steps,
                    previous,
                    delay,
                    prefix_horizon,
                    schedule,
                    max_weight,
                )
            )
        else:
            delay = config.inference_delay
            self._sample = nnx.jit(
                lambda policy, key, obs, previous: policy.action_legato(
                    key, obs, steps, previous, delay
                )
            )
        self._bootstrap = nnx.jit(
            lambda policy, key, obs: policy.action(key, obs, steps)
        )

    @property
    def upstream_model_kind(self) -> str:
        return "legato" if self.mode == "legato" else "vanilla"

    def training_loss(
        self, key: jax.Array, features: jax.Array, action_chunks: jax.Array
    ) -> jax.Array:
        """Call the official loss verbatim, including its public plus-sign target."""
        features = jnp.asarray(features, dtype=jnp.float32)
        action_chunks = jnp.asarray(action_chunks, dtype=jnp.float32)
        self._validate_inputs(features, action_chunks)
        return self.policy.loss(key, features, action_chunks)

    def bootstrap(self, key: jax.Array, features: jax.Array) -> DelayState:
        """Generate the initial chunk before the first asynchronous policy call."""
        features = self._validate_features(features)
        chunk = self._bootstrap(self.policy, key, features)
        return DelayState(
            chunk,
            jnp.ones((features.shape[0],), dtype=jnp.bool_),
        )

    def predict(
        self,
        key: jax.Array,
        features: jax.Array,
        state: DelayState,
    ) -> PolicyPrediction:
        """Generate one chunk and return exactly the actions executed this cycle."""
        features = self._validate_features(features)
        expected = (
            features.shape[0],
            self.config.action_horizon,
            self.action_dim,
        )
        if state.previous_chunk.shape != expected or state.valid.shape != expected[:1]:
            raise ValueError(
                f"Delay state must have shapes {expected} and {expected[:1]}."
            )
        if self.mode == "vanilla":
            generated = self._sample(self.policy, key, features)
        else:
            generated = self._sample(
                self.policy, key, features, state.previous_chunk
            )
        execute = merge_delayed_execution(
            state.previous_chunk,
            generated,
            inference_delay=self.config.inference_delay,
            execute_horizon=self.config.execute_horizon,
        )
        next_state = DelayState(
            shift_generated_chunk(generated, self.config.execute_horizon),
            state.valid,
        )
        return PolicyPrediction(execute, generated, next_state)

    def reset(self, state: DelayState, reset_mask: jax.Array) -> DelayState:
        reset_mask = jnp.asarray(reset_mask, dtype=jnp.bool_)
        if reset_mask.shape != state.valid.shape:
            raise ValueError("reset_mask must match the delay-state batch axis.")
        return DelayState(
            jnp.where(reset_mask[:, None, None], 0.0, state.previous_chunk),
            jnp.where(reset_mask, False, state.valid),
        )

    def _validate_features(self, features: jax.Array) -> jax.Array:
        features = jnp.asarray(features, dtype=jnp.float32)
        if features.ndim != 2 or features.shape[1] != self.obs_dim:
            raise ValueError(
                f"features must have shape [B,{self.obs_dim}], got {features.shape}."
            )
        return features

    def _validate_inputs(
        self, features: jax.Array, action_chunks: jax.Array
    ) -> None:
        features = self._validate_features(features)
        expected = (
            features.shape[0],
            self.config.action_horizon,
            self.action_dim,
        )
        if action_chunks.shape != expected or action_chunks.dtype != jnp.float32:
            raise ValueError(
                f"action_chunks must be float32 with shape {expected}; got "
                f"{action_chunks.shape}/{action_chunks.dtype}."
            )


__all__ = [
    "DelayState",
    "OfficialBigymPolicy",
    "OfficialPolicyConfig",
    "PolicyPrediction",
    "initial_delay_state",
    "merge_delayed_execution",
    "shift_generated_chunk",
]
