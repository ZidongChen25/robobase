"""CQN-AS research line ``mc-rct``: Monte-Carlo return targets + RCT probes.

This variant subclasses the FROZEN pristine :class:`robobase.method.cqn_as.CQNAS`
and adds exactly one research line:

* ``mc_lower_bound_target`` -- replace a *lower* Bellman target with the
  completed-episode Monte-Carlo return of the same executed action.  This is
  the Q-Transformer ``max(TD, MC)`` lower bound: one Q-learning target, not an
  auxiliary imitation loss.  The composition is per (sample, level, dim): the
  MC point mass wins only where ``mc_return > E[target_distribution]``.
* ``mc_return_weight`` -- an *additive* auxiliary C51 cross-entropy from the
  executed action's own head onto the completed-episode return.
* ``mc_return_stop_gradient_encoder`` -- route that auxiliary loss through a
  stop-gradient copy of the features, so MC calibrates the critic head without
  rewriting a representation trained by another objective.
* ``mc_return_value_only`` -- send the auxiliary MC gradient through the
  dueling *value* stream only, stop-gradient-ing the current action's centered
  advantage logits.

Randomized-control (RCT) probes: ``causal_rct_*`` and ``cv_rct_*`` are accepted
and validated here for configuration fidelity but are NOT implementable on this
class -- both are inseparably owned by other lines.  See the module-level
``COUPLING`` note below and the line report.

COUPLING
--------
``causal_rct_weight`` / ``causal_rct_level``
    Declared in ``robobase/cfgs/method/cqn_as.yaml`` but never read by
    ``cqn_as_research.CQNAS``.  The only implementation lives in
    ``robobase/method/cqn_direct_q.py`` (``CQNDirectQAS``, direct-scalar-Q
    line), which the refactor plan already treats as a separate module.  It
    additionally requires ``structured_exploration_prob`` (structured-
    exploration line).
``cv_rct_weight`` / ``cv_rct_level`` / ``cv_rct_baseline``
    Implemented ONLY inside the ``separate_bc_policy`` branch of
    ``cqn_as_research.CQNAS._build_update_fn`` (bc-policy line) and reads the
    ``structured_exploration`` replay tuple plus ``_critic_training_slice``
    (td-variants line).  ``cqn_as_research.py`` itself enforces
    ``cv_rct_weight requires separate_bc_policy=true``, so the flag is
    unreachable from a canonical-critic class such as this one.
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
from robobase.method.rl_common import JaxRLMethodBase, RLModelSpec, activation
from robobase.replay_buffer.replay_buffer import ReplayBuffer


class C2FSequenceDistributionalCriticStreams(nn.Module):
    """Pristine CQN-AS critic plus an opt-in dueling-stream decomposition.

    Byte-for-byte the pristine ``C2FSequenceDistributionalCritic`` (same
    submodule names, therefore an identical parameter tree and identical init
    RNG consumption) except for ``return_streams``.  With ``return_streams``
    false the returned logits use the *pristine* association
    ``values + advantages - advantages.mean(...)``, so the default path stays
    numerically identical to the frozen class.
    """

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
    ) -> jax.Array:
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
            if return_streams:
                raise ValueError(
                    "return_streams requires use_dueling=true; there is no "
                    "value stream to separate."
                )
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
        if return_streams:
            centered_advantages = advantages - advantages.mean(
                axis=-2,
                keepdims=True,
            )
            return values + centered_advantages, values, centered_advantages
        return values + advantages - advantages.mean(axis=-2, keepdims=True)


@dataclass(frozen=True)
class CQNASMcRctSpec(CQNASpec):
    """Pristine CQN-AS hyperparameters plus the ``mc-rct`` line."""

    mc_return_weight: float
    mc_lower_bound_target: bool
    mc_return_stop_gradient_encoder: bool
    mc_return_value_only: bool
    causal_rct_weight: float
    causal_rct_level: int | None
    cv_rct_weight: float | None
    cv_rct_level: int | None
    cv_rct_baseline: str


def cqn_as_mc_rct_spec_from_cfg(cfg: DictConfig) -> CQNASMcRctSpec:
    method = cfg.method
    base = cqn_as_spec_from_cfg(cfg)
    base_values = {field.name: getattr(base, field.name) for field in fields(CQNASpec)}
    return CQNASMcRctSpec(
        **base_values,
        mc_return_weight=float(method.get("mc_return_weight", 0.0)),
        mc_lower_bound_target=bool(method.get("mc_lower_bound_target", False)),
        mc_return_stop_gradient_encoder=bool(
            method.get("mc_return_stop_gradient_encoder", False)
        ),
        mc_return_value_only=bool(method.get("mc_return_value_only", False)),
        causal_rct_weight=float(method.get("causal_rct_weight", 0.0)),
        causal_rct_level=(
            None
            if method.get("causal_rct_level", None) is None
            else int(method.get("causal_rct_level"))
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
    )


class CQNASMcRct(CQNAS):
    """CQN-AS with Monte-Carlo return targets and randomized-control probes."""

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
        mc_return_weight: float = 0.0,
        mc_lower_bound_target: bool = False,
        mc_return_stop_gradient_encoder: bool = False,
        mc_return_value_only: bool = False,
        causal_rct_weight: float = 0.0,
        causal_rct_level: int | None = None,
        cv_rct_weight: float | None = None,
        cv_rct_level: int | None = None,
        cv_rct_baseline: str = "target_q",
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

        # --- mc-rct line validation (mirrors cqn_as_research.CQNAS) ---------
        if mc_return_weight < 0.0:
            raise ValueError("mc_return_weight must be non-negative.")
        if mc_return_weight > 0.0 and mc_return_value_only and not use_dueling:
            raise ValueError(
                "mc_return_value_only=true requires use_dueling=true."
            )
        if causal_rct_weight < 0.0:
            raise ValueError("causal_rct_weight must be non-negative.")
        resolved_causal_level = (
            levels - 1 if causal_rct_level is None else int(causal_rct_level)
        )
        if not 0 <= resolved_causal_level < levels:
            raise ValueError("causal_rct_level must be in [0, levels).")
        if causal_rct_weight > 0.0:
            # The direct-scalar-Q line owns this loss; see module COUPLING.
            raise ValueError(
                "causal_rct_weight > 0 is implemented only by "
                "robobase.method.cqn_direct_q.CQNDirectQAS (direct-scalar-Q "
                "line) and additionally requires structured_exploration_prob "
                "from the structured-exploration line. It is not available on "
                "cqn_as_mc_rct."
            )
        cv_rct_baseline = str(cv_rct_baseline).lower()
        if cv_rct_baseline not in {"target_q", "none"}:
            raise ValueError("cv_rct_baseline must be 'target_q' or 'none'.")
        resolved_cv_level = 0 if cv_rct_level is None else int(cv_rct_level)
        if not 0 <= resolved_cv_level < levels:
            raise ValueError("cv_rct_level must be in [0, levels).")
        if cv_rct_weight is not None:
            if float(cv_rct_weight) < 0.0:
                raise ValueError("cv_rct_weight must be non-negative.")
            # The bc-policy line owns this loss; see module COUPLING.
            raise ValueError(
                "cv_rct_weight requires separate_bc_policy=true (bc-policy "
                "line) and a randomized structured exploration policy "
                "(structured-exploration line); the canonical-critic "
                "cqn_as_mc_rct class provides neither."
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
        self._seed = int(seed)

        self.mc_return_weight = float(mc_return_weight)
        self.mc_lower_bound_target = bool(mc_lower_bound_target)
        self.mc_return_stop_gradient_encoder = bool(
            mc_return_stop_gradient_encoder
        )
        self.mc_return_value_only = bool(mc_return_value_only)
        self.causal_rct_weight = float(causal_rct_weight)
        self.causal_rct_level = int(resolved_causal_level)
        self.cv_rct_weight = (
            None if cv_rct_weight is None else float(cv_rct_weight)
        )
        self.cv_rct_level = int(resolved_cv_level)
        self.cv_rct_baseline = cv_rct_baseline
        # Names kept identical to the research monolith so workspace-side
        # replay-element gating (``_mc_return_anchor_enabled``) and existing
        # diagnostics keep working unchanged.
        self._canonical_mc_anchor = bool(self.mc_return_weight > 0.0)
        self._uses_canonical_mc_returns = bool(
            self._canonical_mc_anchor or self.mc_lower_bound_target
        )

        input_dim = self._setup_rl_features(model, seed=seed)
        self.action_low, self.action_high = self._action_bounds()
        self._step_action_low = jnp.asarray(action_space.low[0], dtype=jnp.float32)
        self._step_action_high = jnp.asarray(action_space.high[0], dtype=jnp.float32)
        self.support = jnp.linspace(v_min, v_max, atoms, dtype=jnp.float32)
        self.critic_model = C2FSequenceDistributionalCriticStreams(
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

    def _critic_logits_per_level(
        self,
        critic_params,
        features,
        action,
        *,
        return_components: bool = False,
    ):
        """Pristine CQN-AS body, plus an opt-in dueling-stream decomposition."""

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

    def _build_update_fn(self):
        """Pristine ``CQN._build_update_fn`` plus the mc-rct MC targets."""

        optimizer = self.optimizer
        tau = self.critic_target_tau
        use_canonical_mc = bool(self._canonical_mc_anchor)
        use_mc_lower_bound = bool(self.mc_lower_bound_target)
        use_mc_returns = bool(self._uses_canonical_mc_returns)

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
                # ``max(TD, MC)`` lower-bound target composition. The MC point
                # mass replaces the Bellman target ONLY where the completed
                # episode return exceeds the bootstrapped expectation, per
                # (sample, level, action-dim). It is a Q-learning target, not
                # an added loss term.
                mc_lower_bound_fraction = jnp.asarray(0.0, dtype=jnp.float32)
                if use_mc_lower_bound:
                    bellman_q = jnp.sum(
                        target_distribution * self.support,
                        axis=-1,
                    )
                    mc_distribution = project_categorical(
                        target_probabilities,
                        mc_returns,
                        jnp.zeros_like(discounts),
                        jnp.zeros_like(bootstrap),
                        self.support,
                    )
                    use_mc_mask = mc_returns[:, None, None] > bellman_q
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

                # Auxiliary MC anchor: C51 cross-entropy from the executed
                # action's own head onto the completed-episode return.
                mc_zero = jnp.asarray(0.0, dtype=jnp.float32)
                mc_return_loss = mc_zero
                mc_return_mae = mc_zero
                if use_canonical_mc:
                    mc_target = project_categorical(
                        jax.nn.softmax(chosen_logits, axis=-1),
                        mc_returns,
                        jnp.zeros_like(discounts),
                        jnp.zeros_like(bootstrap),
                        self.support,
                    )
                    mc_target = jax.lax.stop_gradient(mc_target)
                    mc_chosen_log_probabilities = chosen_log_probabilities
                    if self.mc_return_value_only:
                        # Block the direct MC gradient into advantage-stream
                        # parameters. Distributional dueling still adds the
                        # atom logits before the softmax, so value logits (and
                        # shared encoder features) can move expected-Q ranking.
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
                        mc_chosen_log_probabilities = jax.nn.log_softmax(
                            mc_chosen_logits,
                            axis=-1,
                        )
                    mc_per_sample = -jnp.sum(
                        mc_target * mc_chosen_log_probabilities,
                        axis=-1,
                    ).mean(axis=(1, 2))
                    mc_return_loss = float(
                        self.mc_return_weight
                    ) * jnp.mean(mc_per_sample * loss_weights)
                    critic_loss = critic_loss + mc_return_loss
                    chosen_expected_q = jnp.sum(
                        jax.nn.softmax(chosen_logits, axis=-1) * self.support,
                        axis=-1,
                    )
                    mc_return_mae = jnp.mean(
                        jnp.abs(
                            chosen_expected_q
                            - mc_returns[:, None, None]
                        )
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
                    mc_return_loss,
                    mc_return_mae,
                    mc_lower_bound_fraction,
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
                mc_return_loss,
                mc_return_mae,
                mc_lower_bound_fraction,
            ) = aux
            priority = jnp.sqrt(per_sample + 1e-10)
            priority = priority / jnp.maximum(jnp.max(priority), 1e-10)
            metrics = {
                "critic_loss": critic_loss,
                "entropy": entropy,
                "target_entropy": projected_entropy,
                "loss_coeff": jnp.mean(loss_weights),
            }
            if use_canonical_mc:
                metrics["mc_return_loss"] = mc_return_loss
                metrics["mc_return_mae"] = mc_return_mae
            if use_mc_lower_bound:
                metrics["mc_lower_bound_fraction"] = mc_lower_bound_fraction
                metrics["mc_return_mean"] = jnp.mean(mc_returns)
            return (
                params,
                target_critic_params,
                opt_state,
                priority,
                metrics,
            )

        if use_mc_returns:

            def update_fn(*args):
                (*core, mc_returns, action_key) = args
                return update_impl(*core, mc_returns, action_key)

        else:
            # Keep the pristine 12-argument calling convention exactly when
            # the line is off, so the traced graph matches the frozen class.
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
        """Pristine ``CQN.update`` plus the optional ``mc_return`` element."""

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
            mc_args = ()
            if self._uses_canonical_mc_returns:
                mc_args = (
                    self._as_jax_array(
                        batch.get(
                            "mc_return",
                            np.zeros_like(batch["reward"]),
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
                *mc_args,
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
    "C2FSequenceDistributionalCriticStreams",
    "CQNASMcRct",
    "CQNASMcRctSpec",
    "cqn_as_mc_rct_spec_from_cfg",
]
