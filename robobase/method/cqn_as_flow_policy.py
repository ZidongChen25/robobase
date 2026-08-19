"""Flow-policy CQN-AS variant (R2 line 8: ``flow_policy*`` / ``coarse_flow*``).

This module isolates the in-monolith flow-policy research line from the frozen
official CQN-AS port.  It is unrelated to ``robobase/method/cqn_flow.py``
(``CQNFlowAS``), which is a separate legacy method and stays untouched.

Two mechanisms live here (cqn-flow.md sections 29 and 34):

* **Decoupled flow rerank** (``flow_policy``, Stage-146): a behavior-side
  conditional flow proposes ``flow_policy_candidates`` demonstration-style
  chunks and the calibrated critic reranks them, so Q is only queried on the
  manifold where its calibration was measured.  The research implementation
  gates this on ``separate_bc_policy=true`` -- the flow replaces the separate
  categorical BC policy as rollout behavior and its conditional-flow-matching
  loss is computed inside the separate-BC-policy update function.  That
  substrate belongs to the ``bc-policy`` line, so this class ships the two
  reusable, independently testable pieces (:meth:`_flow_policy_sample` and
  :meth:`_flow_rerank_action`) and raises the research error message when the
  flag is switched on.  See the coupling list below.

* **Coarse flow** (``coarse_flow``, Stage-152/155): fully implemented here.
  The canonical critic keeps the bin decision at the coarse resolution where
  sibling bins have data support (so TD counterfactuals stay identifiable) and
  a bin-conditioned flow head models the continuous within-cell residual in
  [-1, 1] cell coordinates.  ``coarse_flow_pure`` (Stage-155) is the
  no-selection control: the flow models the full action chunk with no bin
  conditioning and the critic never touches the rollout action.

Coupling (documented, never hacked around):

* ``flow_policy`` -> ``separate_bc_policy`` (bc-policy line).  Research:
  ``cqn_as_research.py:2467`` raises ``"flow_policy requires
  separate_bc_policy=true."``; the CFM loss lives at
  ``cqn_as_research.py:5329-5355`` inside the separate-BC-policy update
  function (it needs ``policy_features`` and ``demo_count`` from that path);
  the rollout branch is ``cqn_as_research.py:3769-3796``.
* ``policy_value_beta`` -> ``separate_bc_policy`` (bc-policy line).  Research:
  ``cqn_as_research.py:1943`` raises ``"policy_value_beta requires
  separate_bc_policy=true."``; the selector ``_policy_value_action``
  (``cqn_as_research.py:3474-3566``) reads ``self.policy_model`` and
  ``params["policy"]``, both created only under ``separate_bc_policy``.
  The td-variants line's ``td_target_policy_value_beta`` calls the *same*
  selector (``cqn_as_research.py:4895``); this class reads neither flag.
* ``coarse_flow_selfdistill_weight`` -> the ``mc_return`` replay element
  (mc-rct line).  See :meth:`update` for the exact reproduction rule.

All flags default to OFF; with the defaults this class builds no flow
parameters, keeps the pristine parameter tree, and delegates
``_build_update_fn`` / ``_build_greedy_action_fn`` / ``update`` straight to
the frozen official implementation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, fields
from typing import Iterator, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig

from robobase.method.cqn import encode_action, project_categorical, zoom_in
from robobase.method.cqn_as import CQNAS, CQNASpec, cqn_as_spec_from_cfg
from robobase.method.rl_common import RLModelSpec, activation
from robobase.replay_buffer.replay_buffer import ReplayBuffer


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


@dataclass(frozen=True)
class CQNASFlowPolicySpec(CQNASpec):
    """CQN-AS hyperparameters plus the flow-policy line's flags."""

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
    policy_value_beta: float | None


def cqn_as_flow_policy_spec_from_cfg(cfg: DictConfig) -> CQNASFlowPolicySpec:
    method = cfg.method
    base = cqn_as_spec_from_cfg(cfg)
    base_values = {field.name: getattr(base, field.name) for field in fields(CQNASpec)}
    policy_value_beta = method.get("policy_value_beta", None)
    return CQNASFlowPolicySpec(
        **base_values,
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
        policy_value_beta=(
            None if policy_value_beta is None else float(policy_value_beta)
        ),
    )


class CQNASFlowPolicy(CQNAS):
    """CQN-AS with the flow-policy / coarse-flow research line.

    With every line flag at its default the class is the frozen official
    :class:`~robobase.method.cqn_as.CQNAS`: no ``flow_policy`` parameter
    group is created, ``_build_update_fn`` / ``_build_greedy_action_fn`` /
    ``update`` are the pristine implementations, and no extra RNG is drawn.
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
        policy_value_beta: float | None = None,
    ):
        # --- line flags are validated and stored BEFORE the pristine
        # __init__ runs, because CQNAS.__init__ calls _build_update_fn() and
        # _build_greedy_action_fn(), which branch on them.
        if flow_policy:
            # Coupling (bc-policy line): the decoupled rerank platform replaces
            # the separate categorical BC policy as rollout behavior and its
            # CFM loss lives in the separate-BC-policy update function.  The
            # research class raises exactly this for any configuration this
            # class can express (cqn_as_research.py:2467).
            raise ValueError(
                "flow_policy requires separate_bc_policy=true, which lives in "
                "the bc-policy variant; CQNASFlowPolicy implements the "
                "canonical-platform coarse_flow mechanism plus the reusable "
                "_flow_policy_sample/_flow_rerank_action helpers."
            )
        if policy_value_beta is not None:
            # Coupling (bc-policy line): the selector needs the separate BC
            # policy head (self.policy_model / params["policy"]).
            raise ValueError(
                "policy_value_beta requires separate_bc_policy=true, which "
                "lives in the bc-policy variant."
            )
        if flow_policy_candidates < 1:
            raise ValueError("flow_policy_candidates must be at least 1.")
        if flow_policy_steps < 1:
            raise ValueError("flow_policy_steps must be at least 1.")
        if flow_policy_lambda < 0.0:
            raise ValueError("flow_policy_lambda must be non-negative.")
        if flow_policy_ema is not None and not 0.0 < flow_policy_ema < 1.0:
            raise ValueError("flow_policy_ema must be in (0, 1).")
        if coarse_flow_selfdistill_weight is not None and (
            coarse_flow_selfdistill_weight < 0.0
        ):
            raise ValueError(
                "coarse_flow_selfdistill_weight must be non-negative."
            )
        if coarse_flow_pure and not coarse_flow:
            raise ValueError("coarse_flow_pure requires coarse_flow=true.")

        self.flow_policy = bool(flow_policy)
        self.flow_policy_candidates = int(flow_policy_candidates)
        self.flow_policy_steps = int(flow_policy_steps)
        self.flow_policy_lambda = float(flow_policy_lambda)
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
        self.coarse_flow = bool(coarse_flow)
        # Stage-155 no-selection control: the flow models the FULL action
        # chunk (no bin context, no critic argmax at rollout).  Everything
        # else -- critic training, encoder, flow capacity, EMA -- stays
        # matched, so (CCFF - pure) isolates the coarse selection mechanism.
        self.coarse_flow_pure = bool(coarse_flow_pure)
        self.coarse_flow_selfdistill_weight = (
            None
            if coarse_flow_selfdistill_weight is None
            else float(coarse_flow_selfdistill_weight)
        )
        self.coarse_flow_selfdistill_threshold = float(
            coarse_flow_selfdistill_threshold
        )
        # Self-distillation only exists inside the coarse-flow loss, so the
        # ``mc_return`` batch element is threaded only when both are on.
        # Research ignores the weight when coarse_flow is off
        # (``use_coarse_flow`` gates the whole block, cqn_research.py:1853).
        self._coarse_flow_uses_mc_returns = bool(
            self.coarse_flow and self.coarse_flow_selfdistill_weight is not None
        )
        self.policy_value_beta = None
        self.flow_policy_model = None

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

        if not self.coarse_flow:
            # Flags off: pristine parameter tree, optimizer state, update fn,
            # action fn and RNG stream, untouched.
            return

        # Pristine CQNAS.__init__ draws no RNG after the critic init, so
        # splitting here reproduces the research ordering exactly
        # (cqn_as_research.py:2735).
        self.flow_policy_model = FlowPolicyHead(
            hidden_dims=(
                self.flow_policy_hidden_dims
                if self.flow_policy_hidden_dims is not None
                else model.hidden_dims
            ),
            action_sequence=self.action_sequence,
            action_dim=self.action_dim,
            low_dim_size=(self.low_dim_size if self.use_pixels else 0),
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
        dummy_features = jnp.zeros(
            (1, self._rl_feature_dim), dtype=jnp.float32
        )
        self.rng_key, flow_key = jax.random.split(self.rng_key)
        self.params["flow_policy"] = self.flow_policy_model.init(
            flow_key,
            dummy_features,
            jnp.zeros((1, self._flat_action_dim), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
            bin_context=dummy_bin_context,
        )
        if self.flow_policy_ema is not None:
            self.flow_policy_ema_params = jax.tree.map(
                jnp.array,
                self.params["flow_policy"],
            )

        # The flow parameters joined the tree after CQNAS.__init__ built the
        # optimizer state and the jitted callables, so both are rebuilt here.
        self.opt_state = self.optimizer.init(self.params)
        update_fn = self._build_update_fn()
        action_fn = self._build_greedy_action_fn()
        if self._jit_enabled:
            update_fn = jax.jit(update_fn)
            action_fn = jax.jit(action_fn)
        self._update_impl = update_fn
        if self.flow_policy_ema is None:
            self._greedy_action_impl = action_fn
        else:
            # Research substitutes the EMA weights into params["flow_policy"]
            # at the act() call site (cqn_as_research.py:6115-6126) so they
            # arrive as traced arguments of the already-jitted callable and
            # never trigger a retrace.  Wrapping the compiled function here is
            # equivalent -- act() is the only caller of _greedy_action_impl --
            # and avoids duplicating the pristine act() body.
            def rollout_action_fn(params, *rest):
                return action_fn(
                    {**params, "flow_policy": self.flow_policy_ema_params},
                    *rest,
                )

            self._greedy_action_impl = rollout_action_fn

    # ------------------------------------------------------------------
    # Flow sampling / reranking (cqn-flow.md 29)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Coarse flow (cqn-flow.md 34 / Stage-152, Stage-155)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------
    def _build_greedy_action_fn(self):
        if not self.coarse_flow:
            return super()._build_greedy_action_fn()

        # Copy of the pristine CQNAS action_fn with the coarse-flow branch
        # from cqn_as_research.py:3902-3921 applied.
        def action_fn(params, target_critic_params, obs_inputs, use_target, key):
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
            if self.coarse_flow_pure:
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

        return action_fn

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _build_update_fn(self):
        if not self.coarse_flow:
            return super()._build_update_fn()

        # Copy of the pristine CQN._build_update_fn body with the Stage-152
        # coarse-flow CFM block from cqn_research.py:1852-1927 applied.
        optimizer = self.optimizer
        tau = self.critic_target_tau
        use_selfdistill = self._coarse_flow_uses_mc_returns

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
                chosen_log_probabilities = jax.nn.log_softmax(
                    chosen_logits,
                    axis=-1,
                )
                chosen_probabilities = jax.nn.softmax(chosen_logits, axis=-1)
                all_probabilities = jax.nn.softmax(all_logits, axis=-1)
                per_sample = -jnp.sum(
                    target_distribution * chosen_log_probabilities,
                    axis=-1,
                ).mean(axis=(1, 2))
                critic_loss = self.critic_lambda * jnp.mean(
                    per_sample * loss_weights
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
                # Stage-152 coarse-flow: bin-conditioned CFM on the
                # within-cell residual of the recorded action.  Features
                # are stop-gradient and the flow head has its own
                # parameters, so the critic's gradients are exactly the
                # legacy ones.
                flow_features = jax.lax.stop_gradient(features)
                cfm_key = jax.random.fold_in(action_key, 152)
                noise_key, time_key = jax.random.split(cfm_key)
                if self.coarse_flow_pure:
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
                if use_selfdistill:
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
                return critic_loss, (
                    per_sample,
                    entropy,
                    target_entropy,
                    coarse_flow_loss,
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
            per_sample, entropy, projected_entropy, coarse_flow_loss = aux
            priority = jnp.sqrt(per_sample + 1e-10)
            priority = priority / jnp.maximum(jnp.max(priority), 1e-10)
            return (
                params,
                target_critic_params,
                opt_state,
                priority,
                {
                    "critic_loss": critic_loss,
                    "entropy": entropy,
                    "target_entropy": projected_entropy,
                    "loss_coeff": jnp.mean(loss_weights),
                    "coarse_flow_loss": coarse_flow_loss,
                },
            )

        if use_selfdistill:

            def update_fn(*args):
                (*core, mc_returns, action_key) = args
                return update_impl(*core, mc_returns, action_key)

        else:

            def update_fn(*args):
                (*core, action_key) = args
                rewards = core[6]
                return update_impl(
                    *core,
                    jnp.zeros_like(rewards),
                    action_key,
                )

        return update_fn

    def update(
        self,
        replay_iter: Iterator[dict],
        step: int,
        replay_buffer: ReplayBuffer = None,
    ) -> dict[str, np.ndarray]:
        if self._coarse_flow_uses_mc_returns:
            metrics = self._update_with_mc_returns(
                replay_iter, step, replay_buffer
            )
        else:
            metrics = super().update(replay_iter, step, replay_buffer)
        if self.coarse_flow and self.flow_policy_ema is not None:
            decay = self.flow_policy_ema
            self.flow_policy_ema_params = jax.tree.map(
                lambda ema, online: decay * ema + (1.0 - decay) * online,
                self.flow_policy_ema_params,
                self.params["flow_policy"],
            )
        return metrics

    def _update_with_mc_returns(
        self,
        replay_iter: Iterator[dict],
        step: int,
        replay_buffer: ReplayBuffer = None,
    ) -> dict[str, np.ndarray]:
        """Pristine ``CQN.update`` plus the ``mc_return`` batch element.

        Coupling (mc-rct line): the research monolith threads ``mc_return``
        into the canonical update only when an MC anchor flag is set
        (``cqn_research.py:1109`` ``use_mc_returns``), so self-distillation is
        inert without one.  The reachable behaviour is identical here: the
        ``mc_return`` replay element is only registered by
        ``workspace._mc_return_anchor_enabled`` (``workspace.py:303``), which
        requires ``mc_return_weight > 0`` / ``mc_lower_bound_target`` /
        ``episodic_success_q_target``, so ``batch.get("mc_return", zeros)``
        returns zeros in exactly the configurations where research passes
        zeros.  Note the same helper also gates on the method NAME being
        ``cqn_as``/``cqn_flow``, so under the ``cqn_as_flow_policy`` name the
        element is never registered until that shared file is extended.
        """

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

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
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


__all__ = [
    "CQNASFlowPolicy",
    "CQNASFlowPolicySpec",
    "FlowPolicyHead",
    "cqn_as_flow_policy_spec_from_cfg",
]
