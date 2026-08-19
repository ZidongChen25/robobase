"""CQN-AS research line: separate / frozen behaviour-cloning policy.

R2 extraction of the ``bc-policy`` flag family from the research monolith
(``robobase/method/cqn_as_research.py``).  The line has two independent
mechanisms that share the "behaviour policy" theme:

1. **Decoupled BC policy** (``separate_bc_policy``).  A dedicated
   coarse-to-fine *bin* head (``params["policy"]``, ``atoms=1``, no dueling)
   is trained with demo cross-entropy while the C51 critic keeps a pure TD
   objective — the legacy FOSD/margin BC terms are removed from the critic
   loss and ``bc_lambda`` becomes the weight of the policy CE instead.  The
   BC head, not the critic, drives the rollout in ``act()``.
   ``bc_policy_stop_gradient`` cuts the BC gradient into the shared visual
   tower; ``distinct_policy_encoder`` instead gives the BC head its own
   trainable encoder (``params["policy_encoder"]``), which is the strict
   value/imitation isolation switch.

2. **Demo behaviour forcing** (``demo_behavior_force_probability``).  On the
   *single-objective* critic path (``separate_bc_policy=false``) this forces
   the reward-based Bellman bootstrap action of demo samples to the exact
   recorded ``action_tp1`` chunk with probability ``p``.  It is inert
   without its carrier ``td_target_action_source=critic_replay_max``, so the
   minimal candidate-max target selection is reproduced here; see the module
   notes below.

Frozen-policy import (``freeze_bc_policy`` / ``bc_policy_mode`` /
``frozen_policy_snapshot``) is NOT part of the CQN-AS graph at all: those
three flags are read only by ``robobase/factory.py`` on the
``direct_scalar_q`` branch and by ``robobase/method/cqn_direct_q.py`` /
``cqn_flow.py``.  They are accepted here purely so a mis-wired config fails
loudly instead of silently doing nothing.

Coupling (documented, not absorbed):
* ``td_target_action_source`` belongs to the ``td-variants`` line.  Only the
  two values this line needs are implemented — ``critic`` (pristine) and
  ``critic_replay_max`` (the carrier for demo forcing).  ``replay_next``,
  ``bc_policy`` and ``policy_value`` raise ``NotImplementedError``.
* ``critic_sequence_mode=effective_k0``, ``mc_return_weight``, ``awr_beta``,
  ``flow_policy``, ``policy_value_beta``, ``cv_rct_weight`` and the progress
  head all ride on the research separate-BC update graph but belong to other
  lines; they are omitted here.
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
from robobase.method.rl_common import RLModelSpec
from robobase.replay_buffer.replay_buffer import ReplayBuffer

_TD_TARGET_SOURCES = ("critic", "critic_replay_max")
_UNSUPPORTED_TD_TARGET_SOURCES = ("replay_next", "bc_policy", "policy_value")


@dataclass(frozen=True)
class CQNASBcPolicySpec(CQNASpec):
    """Pristine CQN-AS hyperparameters plus the bc-policy flag family."""

    separate_bc_policy: bool
    bc_policy_stop_gradient: bool
    distinct_policy_encoder: bool
    td_target_action_source: str
    demo_behavior_force_probability: float
    freeze_bc_policy: bool
    bc_policy_mode: str
    frozen_policy_snapshot: str | None


def cqn_as_bc_policy_spec_from_cfg(cfg: DictConfig) -> CQNASBcPolicySpec:
    method = cfg.method
    base = cqn_as_spec_from_cfg(cfg)
    base_values = {
        field.name: getattr(base, field.name) for field in fields(CQNASpec)
    }
    frozen_policy_snapshot = method.get("frozen_policy_snapshot", None)
    return CQNASBcPolicySpec(
        **base_values,
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
        freeze_bc_policy=bool(method.get("freeze_bc_policy", False)),
        bc_policy_mode=str(
            method.get("bc_policy_mode", "behavior_logits")
        ).lower(),
        frozen_policy_snapshot=(
            None
            if frozen_policy_snapshot is None
            else str(frozen_policy_snapshot)
        ),
    )


class CQNASBcPolicy(CQNAS):
    """CQN-AS with an optional separate BC policy and demo behaviour forcing.

    With every flag at its default the class is the pristine ``CQNAS``: the
    update function and the greedy-action function fall straight through to
    the frozen implementations and no extra parameters are created.
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
        separate_bc_policy: bool = False,
        bc_policy_stop_gradient: bool = False,
        distinct_policy_encoder: bool = False,
        td_target_action_source: str = "critic",
        demo_behavior_force_probability: float = 0.0,
        freeze_bc_policy: bool = False,
        bc_policy_mode: str = "behavior_logits",
        frozen_policy_snapshot: str | None = None,
    ):
        separate_bc_policy = bool(separate_bc_policy)
        distinct_policy_encoder = bool(distinct_policy_encoder)
        td_target_action_source = str(td_target_action_source).lower()
        demo_behavior_force_probability = float(
            demo_behavior_force_probability
        )
        bc_policy_mode = str(bc_policy_mode).lower()

        # --- flags this variant deliberately does not implement -----------
        if bool(freeze_bc_policy) or frozen_policy_snapshot is not None:
            raise NotImplementedError(
                "freeze_bc_policy / frozen_policy_snapshot are consumed by "
                "robobase.factory on the direct_scalar_q branch and by "
                "cqn_direct_q.CQNDirectQAS / cqn_flow.CQNFlowAS; the CQN-AS "
                "update graph never reads them."
            )
        if bc_policy_mode not in {"behavior_logits", "legacy_c51"}:
            raise ValueError(
                "bc_policy_mode must be behavior_logits or legacy_c51."
            )
        if bc_policy_mode != "behavior_logits":
            raise NotImplementedError(
                "bc_policy_mode=legacy_c51 is a cqn_direct_q / cqn_flow "
                "frozen-policy import mode and is out of scope for CQN-AS."
            )
        if td_target_action_source in _UNSUPPORTED_TD_TARGET_SOURCES:
            raise NotImplementedError(
                f"td_target_action_source={td_target_action_source} belongs "
                "to the td-variants research line; this variant implements "
                "only " + ", ".join(_TD_TARGET_SOURCES) + "."
            )
        if td_target_action_source not in _TD_TARGET_SOURCES:
            raise ValueError(
                "td_target_action_source must be one of "
                f"{set(_TD_TARGET_SOURCES)}."
            )

        # --- research-era cross-flag validation (cqn_as_research.py) ------
        if separate_bc_policy and td_target_action_source == (
            "critic_replay_max"
        ):
            raise ValueError(
                "critic_replay_max is implemented only for the "
                "single-objective critic path."
            )
        if separate_bc_policy and float(bc_lambda) <= 0.0:
            raise ValueError(
                "separate_bc_policy=true requires bc_lambda > 0."
            )
        if distinct_policy_encoder and not separate_bc_policy:
            raise ValueError(
                "distinct_policy_encoder=true requires "
                "separate_bc_policy=true."
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

        # Set before ``super().__init__`` so the update/greedy-action factories
        # it calls already see the correct branch.
        self.separate_bc_policy = separate_bc_policy
        self.bc_policy_stop_gradient = bool(bc_policy_stop_gradient)
        self.distinct_policy_encoder = distinct_policy_encoder
        self.td_target_action_source = td_target_action_source
        self.demo_behavior_force_probability = (
            demo_behavior_force_probability
        )
        self.freeze_bc_policy = False
        self.bc_policy_mode = bc_policy_mode
        self.frozen_policy_snapshot = None
        self.policy_model = None

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

        if self.separate_bc_policy:
            self._setup_bc_policy_head(model)

    # ------------------------------------------------------------------
    # Parameter construction
    # ------------------------------------------------------------------
    def _setup_bc_policy_head(self, model: RLModelSpec) -> None:
        """Add the BC bin head (and optional policy tower) to ``self.params``.

        The pristine ``__init__`` consumes ``self.rng_key`` for the critic
        init without splitting it, exactly like the research monolith, so
        splitting once here reproduces the research policy-init key stream.
        """

        # No value atoms: demo CE can train this head without touching the
        # critic's return distribution or its action ranking.
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
        dummy_features = jnp.zeros(
            (1, self._rl_feature_dim), dtype=jnp.float32
        )
        dummy_level = jnp.zeros((1, self.levels), dtype=jnp.float32)
        dummy_midpoint = jnp.zeros(
            (1, self.action_sequence, self.action_dim),
            dtype=jnp.float32,
        )
        self.rng_key, policy_key = jax.random.split(self.rng_key)
        self.params["policy"] = self.policy_model.init(
            policy_key,
            dummy_features,
            dummy_level,
            dummy_midpoint,
        )
        if self._trainable_encoder and self.distinct_policy_encoder:
            # JAX arrays are immutable, so duplicating the leaves is enough to
            # give the policy tower its own optimizer state and gradient path.
            # Both towers start from the same visual initialization.
            self.params["policy_encoder"] = jax.tree.map(
                lambda value: jnp.array(value),
                self._encoder_params,
            )
        # The optimizer was initialised on the critic-only tree.
        self.opt_state = self.optimizer.init(self.params)

    # ------------------------------------------------------------------
    # BC policy head helpers (new code; no pristine counterpart)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Demo behaviour forcing (carrier: td_target_action_source)
    # ------------------------------------------------------------------
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

    def _td_target_action_for_update(
        self,
        critic_params,
        features,
        replay_actions,
        replay_next_actions,
        demos,
        action_key,
    ):
        """Bootstrap action for the single TD objective.

        ``critic`` is the pristine Double-CQN greedy selection.
        ``critic_replay_max`` compares that greedy chunk with the exact
        recorded ``action_tp1`` chunk by deepest-level Q, and
        ``demo_behavior_force_probability`` additionally forces the recorded
        behaviour chunk on demo samples.  Neither adds an imitation term:
        only the reward-based Bellman bootstrap action changes.
        """

        del replay_actions
        if self.td_target_action_source != "critic_replay_max":
            return (
                self._greedy_action_for_update(
                    critic_params,
                    features,
                    action_key,
                )[0],
                {},
            )
        greedy_action, _ = self._greedy_action_for_update(
            critic_params,
            features,
            action_key,
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
            demo_behavior_forced = (demos >= 0.5) & jax.random.bernoulli(
                force_key,
                self.demo_behavior_force_probability,
                shape=demos.shape,
            )
            behavior_selected = behavior_selected | demo_behavior_forced
        else:
            demo_behavior_forced = jnp.zeros_like(demos, dtype=jnp.bool_)
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

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------
    def _build_greedy_action_fn(self):
        # Pristine ``CQNAS._build_greedy_action_fn`` plus the separate-BC
        # branch: the BC head, not the critic, controls the robot.
        def action_fn(
            params,
            target_critic_params,
            obs_inputs,
            use_target,
            key,
        ):
            if getattr(self, "separate_bc_policy", False):
                policy_encoder_params = params.get("encoder", None)
                if self.distinct_policy_encoder:
                    policy_encoder_params = params.get("policy_encoder", None)
                policy_features = self._rl_features(
                    policy_encoder_params,
                    obs_inputs,
                    stop_gradient=True,
                )
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
            return self._greedy_action(critic_params, features, key=key)[0]

        return action_fn

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def _build_update_fn(self):
        if getattr(self, "separate_bc_policy", False):
            return self._build_separate_bc_update_fn()
        if (
            getattr(self, "td_target_action_source", "critic")
            == "critic_replay_max"
        ):
            return self._build_candidate_target_update_fn()
        return super()._build_update_fn()

    def _build_separate_bc_update_fn(self):
        """Pristine TD critic + an independent demo-CE bin policy."""

        optimizer = self.optimizer
        tau = self.critic_target_tau

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
                if self.distinct_policy_encoder:
                    policy_features = self._rl_features(
                        current_params.get("policy_encoder", None),
                        obs_inputs,
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
                target_distribution = jax.lax.stop_gradient(
                    target_distribution
                )
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
                per_sample = -jnp.sum(
                    target_distribution * chosen_log_probabilities,
                    axis=-1,
                ).mean(axis=(1, 2))
                # TD only: the legacy FOSD/margin BC terms of the pristine
                # critic loss move to the policy head below.
                critic_loss = self.critic_lambda * jnp.mean(
                    per_sample * loss_weights
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
                total_loss = critic_loss + policy_loss

                policy_correct = (
                    (jnp.argmax(policy_logits, axis=-1) == expert_bins)
                    .astype(jnp.float32)
                    .mean(axis=(1, 2))
                )
                policy_demo_top1 = (
                    jnp.sum(policy_correct * demos) / demo_count
                )
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
                critic_q_span = (
                    all_q.max(axis=-1) - all_q.min(axis=-1)
                ).mean()
                return total_loss, (
                    per_sample,
                    critic_loss,
                    policy_loss,
                    policy_ce,
                    policy_demo_top1,
                    policy_entropy,
                    critic_entropy,
                    target_entropy,
                    critic_q_span,
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
                policy_loss,
                policy_ce,
                policy_demo_top1,
                policy_entropy,
                critic_entropy,
                projected_entropy,
                critic_q_span,
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
                    "td_critic_loss": critic_loss,
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
                },
            )

        return update_fn

    def _build_candidate_target_update_fn(self):
        """Pristine update graph with the candidate-max bootstrap action.

        Byte-for-byte the pristine ``CQN._build_update_fn`` body except that
        the bootstrap action comes from ``_td_target_action_for_update`` (so
        ``demo_behavior_force_probability`` can act) and that the extra
        ``next_actions`` replay chunk is threaded in.
        """

        optimizer = self.optimizer
        tau = self.critic_target_tau

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
                target_distribution = jax.lax.stop_gradient(
                    target_distribution
                )
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
                    fosd = (
                        jnp.maximum(
                            chosen_cdf[..., None, :] - all_cdf,
                            0.0,
                        )
                        .sum(axis=-1)
                        .mean(axis=(1, 2, 3))
                    )
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
                            self.bc_margin - (chosen_q[..., None] - all_q),
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
                    target_action_info,
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
                target_action_info,
            ) = aux
            priority = jnp.sqrt(per_sample + 1e-10)
            priority = priority / jnp.maximum(jnp.max(priority), 1e-10)
            selected = target_action_info["behavior_selected"].astype(
                jnp.float32
            )
            behavior_score = target_action_info["behavior_score"]
            greedy_score = target_action_info["greedy_score"]
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
                    "behavior_candidate_fraction": jnp.mean(selected),
                    "behavior_candidate_score": jnp.mean(behavior_score),
                    "greedy_candidate_score": jnp.mean(greedy_score),
                    "behavior_minus_greedy_q": jnp.mean(
                        behavior_score - greedy_score
                    ),
                    "demo_behavior_force_fraction": jnp.mean(
                        target_action_info["demo_behavior_forced"].astype(
                            jnp.float32
                        )
                    ),
                    "demo_behavior_force_probability": jnp.asarray(
                        self.demo_behavior_force_probability,
                        dtype=jnp.float32,
                    ),
                },
            )

        return update_fn

    def update(
        self,
        replay_iter: Iterator[dict],
        step: int,
        replay_buffer: ReplayBuffer = None,
    ) -> dict[str, np.ndarray]:
        """Pristine update loop; the candidate-max path also feeds a_{t+1}.

        ``separate_bc_policy`` keeps the pristine ``_update_impl`` signature,
        so only ``critic_replay_max`` needs its own loop.
        """

        if self.td_target_action_source != "critic_replay_max":
            return super().update(replay_iter, step, replay_buffer)

        update_steps = 1 if step == 0 else self.num_update_steps
        metrics = {}
        for _ in range(update_steps):
            batch = next(replay_iter)
            obs_inputs = self._prepare_rl_obs_inputs(batch)
            next_obs_inputs = self._next_rl_obs_inputs(batch)
            actions = self._as_jax_array(
                batch["action"], self.jnp.float32
            ).reshape((batch["action"].shape[0], -1))
            if "action_tp1" not in batch:
                raise KeyError(
                    "critic_replay_max requires replay.include_next_action="
                    "true and an action_tp1 batch element."
                )
            next_action_values = batch["action_tp1"]
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
                jnp.ones_like(terminal)
                if self.always_bootstrap
                else 1.0 - terminal
            )
            loss_weights = self._loss_weights(batch)
            demos = self._as_jax_array(
                batch.get("demo", np.zeros_like(batch["reward"])),
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
                next_actions,
                rewards,
                discounts,
                bootstrap,
                loss_weights,
                demos,
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
    "CQNASBcPolicy",
    "CQNASBcPolicySpec",
    "cqn_as_bc_policy_spec_from_cfg",
]
