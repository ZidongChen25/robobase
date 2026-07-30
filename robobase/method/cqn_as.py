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
    project_categorical,
    zoom_in,
)
from robobase.method.rl_common import JaxRLMethodBase, RLModelSpec, activation
from robobase.replay_buffer.replay_buffer import ReplayBuffer


def random_shift_rgb(rgb: jax.Array, key: jax.Array, pad: int = 4) -> jax.Array:
    """Reference RandomShiftsAug for ``[batch, views, channels, H, W]`` RGB."""

    if pad <= 0:
        return rgb
    batch, views, channels, height, width = rgb.shape
    flat = rgb.reshape((batch * views, channels, height, width))
    flat = jnp.pad(
        flat,
        ((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode="edge",
    )
    shifts = jax.random.randint(
        key,
        (batch * views, 2),
        minval=0,
        maxval=2 * pad + 1,
    )

    def crop(image, shift):
        return jax.lax.dynamic_slice(
            image,
            (0, shift[0], shift[1]),
            (channels, height, width),
        )

    return jax.vmap(crop)(flat, shifts).reshape(rgb.shape)


@dataclass(frozen=True)
class CQNASpec(CQNSpec):
    """CQN hyperparameters plus the action-sequence architecture settings."""

    demo_fosd: bool
    gru_layers: int
    temporal_ensemble: bool
    temporal_ensemble_replan_interval: int
    temporal_ensemble_gain: float
    tie_break_delta: float
    structured_exploration_prob: float
    structured_exploration_level: int
    structured_exploration_horizon: int
    separate_bc_policy: bool
    bc_policy_stop_gradient: bool
    distinct_policy_encoder: bool
    td_target_action_source: str
    td_target_policy_value_beta: float | None
    critic_sequence_mode: str
    mc_return_weight: float
    mc_return_stop_gradient_encoder: bool
    mc_return_value_only: bool
    policy_value_beta: float | None
    cv_rct_weight: float | None
    cv_rct_level: int | None
    cv_rct_baseline: str
    awr_beta: float | None
    awr_weight_max: float
    awr_expectile_tau: float
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


def cqn_as_spec_from_cfg(cfg: DictConfig) -> CQNASpec:
    method = cfg.method
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
        gru_layers=int(method.get("gru_layers", 1)),
        temporal_ensemble=bool(method.get("temporal_ensemble", True)),
        temporal_ensemble_replan_interval=int(
            method.get("temporal_ensemble_replan_interval", 1)
        ),
        temporal_ensemble_gain=float(method.get("temporal_ensemble_gain", 0.01)),
        tie_break_delta=float(method.get("tie_break_delta", 1e-4)),
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
        td_target_policy_value_beta=(
            None
            if td_target_policy_value_beta is None
            else float(td_target_policy_value_beta)
        ),
        critic_sequence_mode=str(
            method.get("critic_sequence_mode", "full")
        ).lower(),
        mc_return_weight=float(method.get("mc_return_weight", 0.0)),
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
        td_target_policy_value_beta: float | None,
        critic_sequence_mode: str,
        mc_return_weight: float,
        mc_return_stop_gradient_encoder: bool,
        mc_return_value_only: bool,
        policy_value_beta: float | None,
        model: RLModelSpec,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        num_train_envs: int,
        num_eval_envs: int,
        replay_alpha: float,
        replay_beta: float,
        frame_stack_on_channel: bool,
        intrinsic_reward_module=None,
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
                "td_target_action_source requires separate_bc_policy=true unless "
                "it is 'critic'."
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
        # Canonical (non-decoupled) MC anchor is a deliberate Stage-147 arm:
        # it calibrates the same Q head that drives behavior.  The historical
        # decoupling requirement is therefore relaxed; interpretation caveats
        # live in cqn-flow.md section 28.6.
        self._canonical_mc_anchor = bool(
            mc_return_weight > 0.0 and not separate_bc_policy
        )
        if mc_return_weight > 0.0 and mc_return_value_only and not use_dueling:
            raise ValueError(
                "mc_return_value_only=true requires use_dueling=true."
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
        self.structured_exploration_level = int(
            structured_exploration_level
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
        critic_params = self.critic_model.init(
            self.rng_key,
            dummy_features,
            dummy_level,
            dummy_midpoint,
        )
        self.params = {"critic": critic_params}
        if self.separate_bc_policy:
            # The policy has its own coarse-to-fine bin logits.  It deliberately
            # has no value atoms: demo CE can train this head without changing
            # the critic's return distribution or action ranking.
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
            model_output = self.critic_model.apply(
                critic_params,
                features,
                one_hot,
                (0.5 * (low + high)).reshape(
                    (features.shape[0], self.action_sequence, self.action_dim)
                ),
                return_streams=return_components,
            )
            if return_components:
                logits, values, centered_advantages = model_output
            else:
                logits = model_output
            index = discrete_action[:, level, :]
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
        def action_fn(params, target_critic_params, obs_inputs, use_target, key):
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
            return self._greedy_action(critic_params, features, key=key)[0]

        return action_fn

    def _greedy_action_for_update(self, critic_params, features, action_key):
        return self._greedy_action(
            critic_params,
            features,
            key=action_key,
        )

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

    def _build_update_fn(self):
        if not getattr(self, "separate_bc_policy", False):
            return super()._build_update_fn()

        optimizer = self.optimizer
        tau = self.critic_target_tau
        use_cv_rct = getattr(self, "cv_rct_weight", None) is not None

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
                    action_sequence = actions.reshape(
                        (actions.shape[0], self.action_sequence, self.action_dim)
                    )
                    # Replay sequences are assembled from the actions that were
                    # actually executed at consecutive environment steps.  The
                    # shifted first token is therefore a_{t+1}, including the
                    # temporal ensemble, rather than a newly predicted raw plan.
                    next_action = jnp.concatenate(
                        [action_sequence[:, 1:], action_sequence[:, -1:]],
                        axis=1,
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
                total_loss = (
                    critic_loss
                    + policy_loss
                    + awr_value_loss
                    + flow_policy_loss
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
            ) = aux
            priority = jnp.sqrt(per_sample + 1e-10)
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

        if use_cv_rct:

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
                structured_explore_start,
                structured_explore_dimension,
                structured_explore_delta,
                structured_explore_assignment_prob,
                action_key,
            ):
                return update_impl(
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
                    (
                        structured_explore_start,
                        structured_explore_dimension,
                        structured_explore_delta,
                        structured_explore_assignment_prob,
                    ),
                    action_key,
                )

        else:

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
                return update_impl(
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
                    None,
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
        return np.sum(candidates * weights[..., None], axis=1)

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

    def _apply_bin_explore(self, action_chunk: np.ndarray) -> np.ndarray:
        """Hierarchical epsilon-bin exploration compatible with the
        closed-loop temporal ensemble (cqn-flow.md 35).

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
        if self._bin_explore_remaining.shape[0] != batch:
            self._bin_explore_remaining = np.zeros((batch,), dtype=np.int32)
            self._bin_explore_dimension = np.full((batch,), -1, dtype=np.int16)
            self._bin_explore_level = np.full((batch,), -1, dtype=np.int16)
            self._bin_explore_sibling = np.full((batch,), -1, dtype=np.int16)
        low = np.asarray(self._step_action_low, dtype=np.float64)
        high = np.asarray(self._step_action_high, dtype=np.float64)
        scale = float(getattr(self, "_bin_explore_scale", 1.0))
        for row in range(batch):
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
        return shifted

    def act(self, observations: dict, step: int, eval_mode: bool):
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
            )
            self._block(action)
            action_chunk = np.asarray(jax.device_get(action), dtype=np.float32)
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
                action_chunk = self._apply_bin_explore(action_chunk)
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

    def rollout_diagnostics(self) -> dict[str, float]:
        eligible = int(
            getattr(self, "_structured_exploration_eligible", 0)
        )
        applied = int(getattr(self, "_structured_exploration_applied", 0))
        return {
            "structured_exploration_rate": (
                float(applied / eligible) if eligible else 0.0
            ),
            "structured_exploration_applied": float(applied),
            "structured_exploration_eligible": float(eligible),
            "structured_exploration_starts": float(
                getattr(self, "_structured_exploration_starts", 0)
            ),
        }

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

    def reset(self, step: int, agents_to_reset: list[int]):
        del step
        for agent_index in agents_to_reset:
            if agent_index < self.num_train_envs:
                if hasattr(self, "_structured_exploration_remaining"):
                    self._structured_exploration_remaining[agent_index] = 0
                    self._structured_exploration_dimension[agent_index] = -1
                    self._structured_exploration_direction[agent_index] = 0.0
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
    "C2FSequenceDistributionalCritic",
    "CQNAS",
    "CQNASpec",
    "cqn_as_spec_from_cfg",
]
