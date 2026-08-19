"""FS-CQN: frozen-support mask on the pristine CQN-AS graph.

Research line ``frozen-support-mask`` (proposal ``reports/fscqn_proposal_
20260814.md``, introduced by commit ``2befcab``; ``support_mask_decode``
added by ``3016806``).  A second coarse-to-fine head with a single "atom"
per bin is trained by demo cross-entropy alongside the critic and frozen
after ``support_mask_freeze_step`` *gradient updates* (the offline/online
seam).  Per C2F level it defines the admissible bin set

    A(s, level) = {b : pi_b >= support_mask_tau * max_b' pi_b'}

and the critic's coarse-to-fine argmax is restricted to ``A``:

* the TD-target (bootstrap) argmax is always masked while
  ``use_frozen_support_mask`` is true;
* the rollout/decode argmax is masked only when ``support_mask_decode``
  is also true (Wave-10 arm A "FS-CQN-TM" runs target-mask-only, after
  the 08-15 measurement showed a frozen decode mask is actively harmful
  on a drifted critic while the masked target still protects the value
  surface).

With ``use_frozen_support_mask=false`` the class is the pristine
``CQNAS``: no ``policy`` entry in ``params``, no extra metrics, no change
to the update graph or the RNG stream.

Every overridden method is a copy of the pristine body from
``robobase/method/cqn.py`` / ``robobase/method/cqn_as.py`` with only the
support-mask edits applied.  Nothing here imports from ``*_research``.

Not reproducible on the pristine base (documented, not hacked around):
the research line also gates the ``dense-return`` line's floor losses so
that in-support unexecuted siblings receive no floor gradient.  Those
losses do not exist in pristine CQN, so this module ships the three
gating primitives (``support_gated_unseen_mask``,
``support_gated_tail_unseen_q``, ``support_gated_per_bin_loss``) that
express exactly that edit, for the dense-return line to call.
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

from robobase.method.cqn import encode_action, project_categorical, zoom_in
from robobase.method.cqn_as import (
    C2FSequenceDistributionalCritic,
    CQNAS,
    CQNASpec,
    cqn_as_spec_from_cfg,
)
from robobase.method.rl_common import JaxRLMethodBase, RLModelSpec
from robobase.replay_buffer.replay_buffer import ReplayBuffer


@dataclass(frozen=True)
class CQNASFrozenSupportMaskSpec(CQNASpec):
    """CQN-AS hyperparameters plus the frozen-support-mask settings."""

    use_frozen_support_mask: bool
    support_mask_decode: bool
    support_mask_tau: float
    support_mask_freeze_step: int


def cqn_as_fscqn_spec_from_cfg(cfg: DictConfig) -> CQNASFrozenSupportMaskSpec:
    method = cfg.method
    base = cqn_as_spec_from_cfg(cfg)
    base_values = {
        field.name: getattr(base, field.name) for field in fields(CQNASpec)
    }
    return CQNASFrozenSupportMaskSpec(
        **base_values,
        use_frozen_support_mask=bool(
            method.get("use_frozen_support_mask", False)
        ),
        support_mask_decode=bool(method.get("support_mask_decode", True)),
        support_mask_tau=float(method.get("support_mask_tau", 0.3)),
        support_mask_freeze_step=int(
            method.get("support_mask_freeze_step", 10000)
        ),
    )


def support_gated_unseen_mask(
    unseen_mask: jax.Array,
    support_mask: jax.Array | None,
) -> jax.Array:
    """Shrink an unseen-bin mask to unseen AND out-of-support bins.

    ``unseen_mask`` is ``[B, L, D, bins]`` float; ``support_mask`` is the
    boolean admissible set of the same shape.  In-support unexecuted
    siblings are left entirely to the TD objective.  This is the first of
    the three edits the FS-CQN line makes to the dense-return line's floor
    losses (``unseen_return_floor_loss``).
    """

    if support_mask is None:
        return unseen_mask
    return unseen_mask * (1.0 - support_mask.astype(unseen_mask.dtype))


def support_gated_tail_unseen_q(
    tail_unseen_q: jax.Array,
    floor_value: float,
    support_mask: jax.Array | None,
) -> jax.Array:
    """Replace non-finite tail entries with the floor value.

    Second FS-CQN edit to ``unseen_return_floor_loss``: with a support
    mask the ``topk``/``max`` reductions can select a sentinel entry when
    every bin is admissible (no out-of-support bin is left to floor).  The
    sentinel is rewritten to ``floor_value`` so the squared term is 0
    instead of ``inf``/``nan``.
    """

    if support_mask is None:
        return tail_unseen_q
    return jnp.where(
        jnp.isfinite(tail_unseen_q),
        tail_unseen_q,
        float(floor_value),
    )


def support_gated_per_bin_loss(
    per_bin_loss: jax.Array,
    chosen_mask: jax.Array,
    support_mask: jax.Array | None,
) -> jax.Array:
    """Zero the dense floor term on in-support unexecuted siblings.

    Third FS-CQN edit, applied to the dense-return line's
    ``dense_return_distributional_loss`` immediately before its per-sample
    reduction.  The executed bin keeps its return target; out-of-support
    bins keep the floor regression.
    """

    if support_mask is None:
        return per_bin_loss
    floor_exempt = support_mask.astype(per_bin_loss.dtype) * (
        1.0 - chosen_mask
    )
    return per_bin_loss * (1.0 - floor_exempt)


class CQNASFrozenSupportMask(CQNAS):
    """CQN-AS whose C2F argmax is restricted to a frozen demo-CE support."""

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
        # --- FS-CQN validation (research parity, cqn_as.py @ 2befcab). ---
        if use_frozen_support_mask:
            self._validate_support_mask_path()
        if not 0.0 < support_mask_tau <= 1.0:
            raise ValueError("support_mask_tau must be in (0, 1].")
        if support_mask_freeze_step < 0:
            raise ValueError(
                "support_mask_freeze_step must be non-negative."
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
        # --- FS-CQN state. ---
        self.use_frozen_support_mask = bool(use_frozen_support_mask)
        self.support_mask_decode = bool(support_mask_decode)
        self.support_mask_tau = float(support_mask_tau)
        self.support_mask_freeze_step = int(support_mask_freeze_step)
        self.policy_model = None

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
        if self.use_frozen_support_mask:
            # The support head has its own coarse-to-fine bin logits.  It
            # deliberately has no value atoms: demo CE can train this head
            # without changing the critic's return distribution or action
            # ranking.  The RNG stream matches the research monolith: the
            # critic is initialized from ``self.rng_key`` unchanged, then a
            # single split feeds the head.
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

    def _validate_support_mask_path(self) -> None:
        """Reject decode paths the mask does not cover.

        The mask only understands the canonical single-critic, non
        autoregressive coarse-to-fine decode.  None of the offending flags
        exist on this class, so the checks read through ``getattr``: they
        fire for any subclass/mixin that combines this line with the
        ``bc-policy``, ``td-variants``, ``twin-critic``, ``flow-policy`` or
        ``dense-return`` lines and turns one of them on before calling
        ``super().__init__``.
        """

        violations = []
        if bool(getattr(self, "separate_bc_policy", False)):
            violations.append("separate_bc_policy=false")
        if bool(getattr(self, "autoregressive_action_dims", False)):
            violations.append("autoregressive_action_dims=false")
        if bool(getattr(self, "pessimistic_twin_critic", False)):
            violations.append("pessimistic_twin_critic=false")
        if int(getattr(self, "twin_rollout_beam_width", 1)) != 1:
            violations.append("twin_rollout_beam_width=1")
        if bool(getattr(self, "coarse_flow", False)):
            violations.append("coarse_flow=false")
        if bool(getattr(self, "dense_return_expected_q_loss", False)):
            violations.append("dense_return_expected_q_loss=false")
        if violations:
            raise ValueError(
                "use_frozen_support_mask requires the canonical "
                "single-critic decode path: " + "; ".join(violations)
            )

    def _policy_logits_per_level(self, policy_params, features, action):
        """Return support-head bin logits and encoded expert bins per level."""

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
            midpoint = (0.5 * (low + high)).reshape(
                (batch_size, self.action_sequence, self.action_dim)
            )
            logits = self.critic_model.apply(
                critic_params,
                features,
                one_hot,
                midpoint,
            )
            probabilities = jax.nn.softmax(logits, axis=-1)
            q_values = jnp.sum(probabilities * self.support, axis=-1)
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
            level_key = level_keys[level]
            if level_key is not None:
                if admissible is None:
                    random_index = jax.random.randint(
                        level_key,
                        index.shape,
                        minval=0,
                        maxval=self.bins,
                    )
                    q_span = q_values.max(axis=-1) - q_values.min(axis=-1)
                else:
                    # Tie breaking stays inside the admissible set: draw
                    # uniformly over it and measure the span over it too,
                    # so masked-out bins can neither be sampled nor make
                    # the span look wide.
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

    def _build_greedy_action_fn(self):
        mask_decode = bool(self.use_frozen_support_mask) and bool(
            self.support_mask_decode
        )

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
            if mask_decode:
                return self._greedy_action(
                    critic_params,
                    features,
                    key=key,
                    policy_params=params["policy"],
                )[0]
            return self._greedy_action(critic_params, features, key=key)[0]

        return action_fn

    def _greedy_action_for_update(
        self,
        critic_params,
        features,
        action_key,
        policy_params=None,
    ):
        if policy_params is None:
            # Keep the pristine call shape when no mask head is supplied so
            # any other override of ``_greedy_action`` stays compatible.
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

    def _build_update_fn(self):
        optimizer = self.optimizer
        tau = self.critic_target_tau
        use_support_mask_head = bool(
            getattr(self, "use_frozen_support_mask", False)
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
            support_mask_ce_weight,
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
                if use_support_mask_head:
                    # The bootstrap argmax is masked whenever the head
                    # exists, independently of ``support_mask_decode``.
                    next_action, _ = self._greedy_action_for_update(
                        current_params["critic"],
                        next_features,
                        action_key,
                        policy_params=current_params["policy"],
                    )
                else:
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
                support_mask_ce_loss = jnp.asarray(0.0, dtype=jnp.float32)
                support_mask_width = jnp.asarray(0.0, dtype=jnp.float32)
                if use_support_mask_head:
                    # Diagnostic only: booleans carry no cotangent, and the
                    # head/encoder are stop-gradient here, so the width has
                    # no effect on critic_loss or on any gradient.
                    support_mask = self._support_mask_for_actions(
                        current_params["policy"],
                        jax.lax.stop_gradient(features),
                        actions,
                    )[:, :, : all_logits.shape[2]]
                    support_mask_width = jnp.mean(
                        support_mask.astype(jnp.float32)
                    )
                    # CE trains the bin head only: the encoder features are
                    # stop-gradient, and the head has no path into the
                    # critic's return distribution.
                    support_mask_ce_loss = support_mask_ce_weight * (
                        self._support_mask_ce_loss(
                            current_params["policy"],
                            jax.lax.stop_gradient(features),
                            actions,
                            demos,
                        )
                    )
                    critic_loss = critic_loss + support_mask_ce_loss

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
                if use_support_mask_head:
                    return critic_loss, (
                        per_sample,
                        entropy,
                        target_entropy,
                        support_mask_ce_loss,
                        support_mask_width,
                    )
                return critic_loss, (per_sample, entropy, target_entropy)

            (critic_loss, aux), grads = jax.value_and_grad(
                loss_fn,
                has_aux=True,
            )(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            pre_params = params
            params = self.optax.apply_updates(params, updates)
            if use_support_mask_head:
                # Freeze the head at the seam.  A zero CE weight is not
                # enough: adamw's decoupled weight decay and the surviving
                # adam moments would keep moving the frozen head, so the
                # pre-update leaves are restored outright.
                params = dict(params)
                params["policy"] = jax.tree.map(
                    lambda frozen, trained: jnp.where(
                        support_mask_ce_weight > 0.0,
                        trained,
                        frozen,
                    ),
                    pre_params["policy"],
                    params["policy"],
                )
            target_critic_params = jax.tree.map(
                lambda target, online: (1.0 - tau) * target + tau * online,
                target_critic_params,
                params["critic"],
            )
            if use_support_mask_head:
                (
                    per_sample,
                    entropy,
                    projected_entropy,
                    support_mask_ce_loss,
                    support_mask_width,
                ) = aux
            else:
                per_sample, entropy, projected_entropy = aux
            priority = jnp.sqrt(per_sample + 1e-10)
            priority = priority / jnp.maximum(jnp.max(priority), 1e-10)
            metrics = {
                "critic_loss": critic_loss,
                "entropy": entropy,
                "target_entropy": projected_entropy,
                "loss_coeff": jnp.mean(loss_weights),
            }
            if use_support_mask_head:
                metrics["support_mask_ce_loss"] = support_mask_ce_loss
                metrics["support_mask_ce_weight"] = support_mask_ce_weight
                metrics["support_mask_width"] = support_mask_width
            return (
                params,
                target_critic_params,
                opt_state,
                priority,
                metrics,
            )

        if use_support_mask_head:
            # The frozen-support CE gate is threaded immediately before
            # action_key; without the head the signature is pristine.
            return update_impl

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
                jnp.asarray(0.0, dtype=jnp.float32),
                action_key,
            )

        return update_fn

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
            support_mask_args = ()
            if self.use_frozen_support_mask:
                # Freeze counts gradient updates, not environment steps, so
                # the head stops exactly at the offline/online seam even
                # though pretraining passes step=0 for every update.
                support_mask_args = (
                    jnp.asarray(
                        1.0
                        if self._update_step_count
                        < int(self.support_mask_freeze_step)
                        else 0.0,
                        dtype=jnp.float32,
                    ),
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
                *support_mask_args,
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
    "CQNASFrozenSupportMask",
    "CQNASFrozenSupportMaskSpec",
    "cqn_as_fscqn_spec_from_cfg",
    "support_gated_per_bin_loss",
    "support_gated_tail_unseen_q",
    "support_gated_unseen_mask",
]
