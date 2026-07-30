"""Monolithic scalar-Q control for the CQN-AS/FLOQ comparison.

This agent deliberately keeps the strict CQN-AS two-tower data and policy
protocol while replacing the categorical/flow value objective with a direct
scalar Bellman regression.  It is a mechanism control: state/image features,
C2F action-bin conditioning, replay actions, target-action source, MC-return
anchor, update-to-data ratio, and the independent BC policy remain unchanged.
"""

from __future__ import annotations

import copy
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp

from robobase.method.cqn import encode_action, zoom_in
from robobase.method.cqn_as import (
    C2FSequenceDistributionalCritic,
    CQNAS,
    action_centered_moment_loss,
)


class CQNDirectQAS(CQNAS):
    """CQN-AS with a candidate-conditioned monolithic scalar-Q critic."""

    def __init__(
        self,
        *,
        direct_q_loss: str = "mse",
        direct_q_huber_delta: float = 1.0,
        causal_rct_weight: float = 0.0,
        causal_rct_level: int | None = None,
        freeze_bc_policy: bool = False,
        bc_policy_mode: str = "behavior_logits",
        frozen_policy_snapshot: str | None = None,
        **kwargs,
    ):
        direct_q_loss = str(direct_q_loss).lower()
        if direct_q_loss not in {"mse", "huber"}:
            raise ValueError("direct_q_loss must be 'mse' or 'huber'.")
        if direct_q_huber_delta <= 0.0:
            raise ValueError("direct_q_huber_delta must be positive.")
        if causal_rct_weight < 0.0:
            raise ValueError("causal_rct_weight must be non-negative.")
        resolved_causal_level = (
            int(kwargs.get("structured_exploration_level", 1))
            if causal_rct_level is None
            else int(causal_rct_level)
        )
        if not 0 <= resolved_causal_level < int(kwargs["levels"]):
            raise ValueError("causal_rct_level must be in [0, levels).")
        if causal_rct_weight > 0.0:
            exploration_probability = float(
                kwargs.get("structured_exploration_prob", 0.0)
            )
            if not 0.0 < exploration_probability < 1.0:
                raise ValueError(
                    "causal RCT loss requires structured_exploration_prob "
                    "strictly between zero and one."
                )
            if int(kwargs.get("structured_exploration_horizon", 1)) != 1:
                raise ValueError(
                    "causal RCT loss requires a one-step randomized "
                    "intervention."
                )
            if float(kwargs.get("mc_return_weight", 0.0)) <= 0.0:
                raise ValueError(
                    "causal RCT loss requires completed Monte-Carlo returns."
                )
            if str(kwargs.get("critic_sequence_mode", "")).lower() != (
                "effective_k0"
            ):
                raise ValueError(
                    "causal RCT loss requires critic_sequence_mode=effective_k0."
                )
        if not bool(kwargs.get("separate_bc_policy", False)):
            raise ValueError(
                "CQNDirectQAS requires separate_bc_policy=true so value "
                "quality is not confounded with the behavior head."
            )
        if bool(kwargs.get("demo_fosd", False)):
            raise ValueError(
                "CQNDirectQAS has no categorical CDF; set demo_fosd=false."
            )
        if bool(kwargs.get("mc_return_stop_gradient_encoder", False)):
            raise ValueError(
                "CQNDirectQAS currently requires "
                "mc_return_stop_gradient_encoder=false."
            )
        if bool(kwargs.get("mc_return_value_only", False)):
            raise ValueError(
                "CQNDirectQAS has one scalar head; set "
                "mc_return_value_only=false."
            )
        bc_policy_mode = str(bc_policy_mode).lower()
        if bc_policy_mode not in {"behavior_logits", "legacy_c51"}:
            raise ValueError(
                "bc_policy_mode must be behavior_logits or legacy_c51."
            )
        if freeze_bc_policy and not bool(
            kwargs.get("distinct_policy_encoder", False)
        ):
            raise ValueError(
                "freeze_bc_policy=true requires "
                "distinct_policy_encoder=true."
            )
        if bc_policy_mode == "legacy_c51" and not freeze_bc_policy:
            raise ValueError(
                "bc_policy_mode=legacy_c51 requires freeze_bc_policy=true."
            )
        if freeze_bc_policy and frozen_policy_snapshot is None:
            raise ValueError(
                "freeze_bc_policy=true requires frozen_policy_snapshot."
            )
        if frozen_policy_snapshot is not None and not freeze_bc_policy:
            raise ValueError(
                "frozen_policy_snapshot requires freeze_bc_policy=true."
            )

        model = kwargs["model"]
        use_dueling = bool(kwargs["use_dueling"])
        seed = int(kwargs.get("seed", 0))
        super().__init__(**kwargs)

        self.direct_scalar_q = True
        self.direct_q_loss = direct_q_loss
        self.direct_q_huber_delta = float(direct_q_huber_delta)
        self.causal_rct_weight = float(causal_rct_weight)
        self.causal_rct_level = int(resolved_causal_level)
        self.freeze_bc_policy = bool(freeze_bc_policy)
        self.bc_policy_mode = bc_policy_mode
        self.frozen_policy_snapshot = (
            None
            if frozen_policy_snapshot is None
            else str(Path(frozen_policy_snapshot).expanduser().resolve())
        )
        dummy_features = jnp.zeros(
            (1, int(self._rl_feature_dim)),
            dtype=jnp.float32,
        )
        dummy_level = jnp.zeros((1, self.levels), dtype=jnp.float32)
        dummy_midpoint = jnp.zeros(
            (1, self.action_sequence, self.action_dim),
            dtype=jnp.float32,
        )
        if self.bc_policy_mode == "legacy_c51":
            # Match the original CQN-AS target critic exactly so a
            # validation-selected clean behavior checkpoint can be imported
            # without projecting its 51-atom distribution into a new BC head.
            self.policy_model = C2FSequenceDistributionalCritic(
                hidden_dims=model.hidden_dims,
                action_sequence=self.action_sequence,
                action_dim=self.action_dim,
                levels=self.levels,
                bins=self.bins,
                atoms=self.atoms,
                low_dim_size=(
                    self.low_dim_size if self.use_pixels else 0
                ),
                gru_layers=self.gru_layers,
                activation_name=model.activation,
                use_dueling=use_dueling,
            )
            policy_key = jax.random.fold_in(
                jax.random.PRNGKey(seed),
                0xC51BC,
            )
            self.params = {
                **self.params,
                "policy": self.policy_model.init(
                    policy_key,
                    dummy_features,
                    dummy_level,
                    dummy_midpoint,
                ),
            }
        self.critic_model = C2FSequenceDistributionalCritic(
            hidden_dims=model.hidden_dims,
            action_sequence=self.action_sequence,
            action_dim=self.action_dim,
            levels=self.levels,
            bins=self.bins,
            atoms=1,
            low_dim_size=(self.low_dim_size if self.use_pixels else 0),
            gru_layers=self.gru_layers,
            activation_name=model.activation,
            use_dueling=use_dueling,
        )
        direct_key = jax.random.fold_in(
            jax.random.PRNGKey(seed),
            0xD1EC7,
        )
        critic_params = self.critic_model.init(
            direct_key,
            dummy_features,
            dummy_level,
            dummy_midpoint,
        )
        self.params = {**self.params, "critic": critic_params}
        if self.frozen_policy_snapshot is not None:
            self.params = self._import_frozen_policy(
                self.params,
                Path(self.frozen_policy_snapshot),
            )
        self.target_critic_params = critic_params
        self.opt_state = self.optimizer.init(self.params)

        update_fn = self._build_direct_q_update_fn()
        action_fn = self._build_direct_q_action_fn()
        if self._jit_enabled:
            update_fn = jax.jit(update_fn)
            action_fn = jax.jit(action_fn)
        self._update_impl = update_fn
        self._greedy_action_impl = action_fn

    @staticmethod
    def _require_tree_compatible(name, source, target) -> None:
        if jax.tree.structure(source) != jax.tree.structure(target):
            raise ValueError(f"{name} parameter structures do not match.")
        source_shapes = tuple(
            tuple(value.shape) for value in jax.tree.leaves(source)
        )
        target_shapes = tuple(
            tuple(value.shape) for value in jax.tree.leaves(target)
        )
        if source_shapes != target_shapes:
            raise ValueError(f"{name} parameter shapes do not match.")

    def _import_frozen_policy(self, params, snapshot: Path):
        """Import a clean CQN-AS target critic and visual encoder bitwise."""

        if not snapshot.is_file():
            raise FileNotFoundError(snapshot)
        with snapshot.open("rb") as handle:
            payload = pickle.load(handle)
        legacy_state = payload.get("agent")
        if not isinstance(legacy_state, dict):
            raise ValueError("frozen policy snapshot has no agent state.")
        legacy_params = legacy_state.get("params")
        if not isinstance(legacy_params, dict):
            raise ValueError("frozen policy snapshot has no parameter tree.")
        legacy_policy = legacy_state.get(
            "target_critic_params",
            legacy_params.get("critic"),
        )
        if legacy_policy is None:
            raise ValueError("frozen policy snapshot has no CQN critic.")
        self._require_tree_compatible(
            "legacy CQN critic -> frozen policy",
            legacy_policy,
            params["policy"],
        )
        imported = {
            **params,
            "policy": copy.deepcopy(legacy_policy),
        }
        if "policy_encoder" in params:
            legacy_encoder = legacy_params.get("encoder")
            if legacy_encoder is None:
                raise ValueError(
                    "frozen pixel policy snapshot has no encoder."
                )
            self._require_tree_compatible(
                "legacy encoder -> frozen policy encoder",
                legacy_encoder,
                params["policy_encoder"],
            )
            imported["policy_encoder"] = copy.deepcopy(legacy_encoder)
        if "encoder" in params:
            legacy_encoder = legacy_params.get("encoder")
            if legacy_encoder is None:
                raise ValueError(
                    "frozen pixel policy snapshot has no encoder."
                )
            self._require_tree_compatible(
                "legacy encoder -> value encoder",
                legacy_encoder,
                params["encoder"],
            )
            imported["encoder"] = copy.deepcopy(legacy_encoder)
        return imported

    def _policy_bin_scores(
        self,
        policy_params,
        features: jax.Array,
        level_one_hot: jax.Array,
        midpoint: jax.Array,
    ) -> jax.Array:
        """Return atoms=1 BC scores or legacy-C51 expected return."""

        outputs = self.policy_model.apply(
            policy_params,
            features,
            level_one_hot,
            midpoint,
        )
        if self.bc_policy_mode == "legacy_c51":
            return jnp.sum(
                jax.nn.softmax(outputs, axis=-1) * self.support,
                axis=-1,
            )
        return outputs[..., 0]

    def _policy_logits_per_level(self, policy_params, features, action):
        """Return frozen-policy bin scores along the replay zoom path."""

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
        )
        low = jnp.broadcast_to(self.action_low, flat_action.shape)
        high = jnp.broadcast_to(self.action_high, flat_action.shape)
        scores_per_level = []
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
                    (
                        batch_size,
                        self.action_sequence,
                        self.action_dim,
                    )
                ),
            )
            scores_per_level.append(
                scores.reshape(
                    (batch_size, self._flat_action_dim, self.bins)
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
        """Autoregress with the imported clean CQN-AS behavior policy."""

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
                    (
                        batch_size,
                        self.action_sequence,
                        self.action_dim,
                    )
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

    def _direct_q_per_level(
        self,
        critic_params,
        features: jax.Array,
        action: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Return replay-chosen and all-bin scalar Q at every C2F level."""

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
            one_hot = jnp.broadcast_to(
                jax.nn.one_hot(level, self.levels, dtype=jnp.float32),
                (batch_size, self.levels),
            )
            q_values = self.critic_model.apply(
                critic_params,
                features,
                one_hot,
                (0.5 * (low + high)).reshape(
                    (batch_size, self.action_sequence, self.action_dim)
                ),
            )[..., 0]
            index = discrete_action[:, level]
            chosen = jnp.take_along_axis(
                q_values,
                index[..., None],
                axis=-1,
            )[..., 0]
            chosen_per_level.append(
                chosen.reshape((batch_size, self._flat_action_dim))
            )
            all_per_level.append(
                q_values.reshape(
                    (batch_size, self._flat_action_dim, self.bins)
                )
            )
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

    def _direct_q_action(
        self,
        critic_params,
        value_features: jax.Array,
        *,
        key: jax.Array | None,
        policy_params=None,
        policy_features: jax.Array | None = None,
        policy_value_beta: float | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Autoregress with scalar Q, optionally regularized by the BC prior."""

        use_policy_prior = policy_params is not None
        resolved_beta = (
            self.policy_value_beta
            if policy_value_beta is None
            else float(policy_value_beta)
        )
        if use_policy_prior and (
            policy_features is None or resolved_beta is None
        ):
            raise ValueError(
                "policy/value action selection requires policy features "
                "and a finite policy_value_beta."
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
            q_values = self.critic_model.apply(
                critic_params,
                value_features,
                one_hot,
                midpoint,
            )[..., 0]
            score = q_values
            if use_policy_prior:
                policy_logits = self._policy_bin_scores(
                    policy_params,
                    policy_features,
                    one_hot,
                    midpoint,
                )
                centered_q = q_values - q_values.mean(
                    axis=-1,
                    keepdims=True,
                )
                q_scale = jnp.sqrt(
                    jnp.mean(
                        jnp.square(centered_q),
                        axis=-1,
                        keepdims=True,
                    )
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

    def _build_direct_q_action_fn(self):
        def action_fn(
            params,
            target_critic_params,
            obs_inputs,
            use_target,
            key,
        ):
            policy_encoder_params = params.get("encoder", None)
            if self.distinct_policy_encoder:
                policy_encoder_params = params.get("policy_encoder", None)
            policy_features = self._rl_features(
                policy_encoder_params,
                obs_inputs,
                stop_gradient=True,
            )
            if self.policy_value_beta is None:
                return self._policy_action(
                    params["policy"],
                    policy_features,
                    key=key,
                )[0]

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
            return self._direct_q_action(
                critic_params,
                value_features,
                key=key,
                policy_params=params["policy"],
                policy_features=policy_features,
            )[0]

        return action_fn

    def _regression_loss(self, error: jax.Array) -> jax.Array:
        if self.direct_q_loss == "mse":
            return jnp.square(error)
        absolute = jnp.abs(error)
        delta = self.direct_q_huber_delta
        quadratic = jnp.minimum(absolute, delta)
        linear = absolute - quadratic
        return 0.5 * jnp.square(quadratic) + delta * linear

    def _build_direct_q_update_fn(self):
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
            structured_explore_start,
            structured_explore_dimension,
            structured_explore_delta,
            structured_explore_assignment_prob,
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
                        "policy_encoder",
                        None,
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
                        key=action_key,
                    )
                elif self.td_target_action_source == "policy_value":
                    next_action, _ = self._direct_q_action(
                        current_params["critic"],
                        next_features,
                        key=action_key,
                        policy_params=current_params["policy"],
                        policy_features=next_policy_features,
                        policy_value_beta=(
                            self.td_target_policy_value_beta
                        ),
                    )
                else:
                    next_action, _ = self._direct_q_action(
                        current_params["critic"],
                        next_features,
                        key=action_key,
                    )

                next_q, _ = self._direct_q_per_level(
                    target_critic_params,
                    next_features,
                    next_action,
                )
                next_q = self._critic_training_slice(next_q)
                target_q = rewards[:, None, None] + (
                    discounts * bootstrap
                )[:, None, None] * next_q
                if self.centralized_critic:
                    target_q = jnp.broadcast_to(
                        target_q.mean(axis=-1, keepdims=True),
                        target_q.shape,
                    )
                target_q = jax.lax.stop_gradient(target_q)

                chosen_q_full, all_q_full = self._direct_q_per_level(
                    current_params["critic"],
                    features,
                    actions,
                )
                if self.causal_rct_weight > 0.0:
                    treated = structured_explore_start > 0.5
                    recorded_dimension = jnp.asarray(
                        structured_explore_dimension,
                        dtype=jnp.int32,
                    )
                    recorded_delta = jnp.asarray(
                        structured_explore_delta,
                        dtype=jnp.float32,
                    )
                    assignment_probability = jnp.asarray(
                        structured_explore_assignment_prob,
                        dtype=jnp.float32,
                    )
                    valid_treatment = (
                        treated
                        & (recorded_dimension >= 0)
                        & (recorded_dimension < self.action_dim)
                        & (assignment_probability < 1.0)
                    )
                    valid_control = (
                        (~treated)
                        & (recorded_dimension < 0)
                        & (jnp.abs(recorded_delta) <= 1e-8)
                        & (assignment_probability < 1.0)
                    )
                    causal_valid = valid_treatment | valid_control

                    dimension_key, direction_key = jax.random.split(
                        action_key
                    )
                    sampled_dimension = jax.random.randint(
                        dimension_key,
                        (actions.shape[0],),
                        minval=0,
                        maxval=self.action_dim,
                    )
                    sampled_direction = jnp.where(
                        jax.random.bernoulli(
                            direction_key,
                            shape=(actions.shape[0],),
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
                    ) / float(self.bins ** (self.causal_rct_level + 1))
                    proposed_delta = jnp.where(
                        treated,
                        recorded_delta,
                        sampled_direction
                        * cell_width[intervention_dimension],
                    )
                    action_sequence = actions.reshape(
                        (
                            actions.shape[0],
                            self.action_sequence,
                            self.action_dim,
                        )
                    )
                    row = jnp.arange(actions.shape[0])
                    counterfactual_action = action_sequence.at[
                        row,
                        0,
                        intervention_dimension,
                    ].add(
                        jnp.where(
                            treated,
                            -proposed_delta,
                            proposed_delta,
                        )
                    )
                    counterfactual_action = jnp.clip(
                        counterfactual_action,
                        self._step_action_low,
                        self._step_action_high,
                    )
                    counterfactual_q, _ = self._direct_q_per_level(
                        current_params["critic"],
                        features,
                        counterfactual_action,
                    )
                    observed_score = chosen_q_full[
                        row,
                        self.causal_rct_level,
                        intervention_dimension,
                    ]
                    counterfactual_score = counterfactual_q[
                        row,
                        self.causal_rct_level,
                        intervention_dimension,
                    ]
                    treatment_effect = jnp.where(
                        treated,
                        observed_score - counterfactual_score,
                        counterfactual_score - observed_score,
                    )
                    propensity = float(self.structured_exploration_prob)
                    causal_rct_moment_loss = action_centered_moment_loss(
                        treatment_effect,
                        mc_returns,
                        treated,
                        propensity,
                        causal_valid,
                        loss_weights,
                    )
                    causal_rct_loss = (
                        self.causal_rct_weight * causal_rct_moment_loss
                    )
                    expected_assignment_probability = jnp.where(
                        treated,
                        propensity / float(2 * self.action_dim),
                        1.0 - propensity,
                    )
                    assignment_error = jnp.where(
                        causal_valid,
                        jnp.abs(
                            assignment_probability
                            - expected_assignment_probability
                        ),
                        0.0,
                    )
                    valid_count = jnp.maximum(
                        jnp.sum(causal_valid.astype(jnp.float32)),
                        1.0,
                    )
                    causal_valid_fraction = jnp.mean(
                        causal_valid.astype(jnp.float32)
                    )
                    causal_treated_fraction = jnp.sum(
                        (
                            causal_valid & treated
                        ).astype(jnp.float32)
                    ) / valid_count
                    causal_tau_abs_mean = jnp.sum(
                        jnp.abs(treatment_effect)
                        * causal_valid.astype(jnp.float32)
                    ) / valid_count
                    causal_assignment_error_max = jnp.max(
                        assignment_error
                    )
                else:
                    zero = jnp.asarray(0.0, dtype=features.dtype)
                    causal_rct_moment_loss = zero
                    causal_rct_loss = zero
                    causal_valid_fraction = zero
                    causal_treated_fraction = zero
                    causal_tau_abs_mean = zero
                    causal_assignment_error_max = zero

                chosen_q = self._critic_training_slice(chosen_q_full)
                all_q = self._critic_training_slice(all_q_full)
                td_error = chosen_q - target_q
                td_per_sample = self._regression_loss(td_error).mean(
                    axis=(1, 2)
                )
                td_critic_loss = self.critic_lambda * jnp.mean(
                    td_per_sample * loss_weights
                )

                mc_error = (
                    chosen_q - mc_returns[:, None, None]
                )
                mc_per_sample = self._regression_loss(mc_error).mean(
                    axis=(1, 2)
                )
                mc_return_loss = self.mc_return_weight * jnp.mean(
                    mc_per_sample * loss_weights
                )
                critic_loss = (
                    td_critic_loss + mc_return_loss + causal_rct_loss
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
                q_span = (
                    all_q.max(axis=-1) - all_q.min(axis=-1)
                ).mean()
                return total_loss, (
                    td_per_sample,
                    td_critic_loss,
                    mc_return_loss,
                    jnp.mean(jnp.abs(mc_error)),
                    policy_loss,
                    policy_ce,
                    policy_demo_top1,
                    policy_entropy,
                    q_span,
                    chosen_q.mean(),
                    target_q.mean(),
                    causal_rct_loss,
                    causal_rct_moment_loss,
                    causal_valid_fraction,
                    causal_treated_fraction,
                    causal_tau_abs_mean,
                    causal_assignment_error_max,
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

            def nonfinite_fraction(tree):
                leaves = jax.tree.leaves(tree)
                nonfinite = sum(
                    jnp.sum(~jnp.isfinite(leaf)) for leaf in leaves
                )
                size = sum(leaf.size for leaf in leaves)
                return nonfinite.astype(jnp.float32) / float(max(size, 1))

            grad_norm = self.optax.tree.norm(grads)
            critic_grad_norm = tree_norm_or_zero("critic")
            encoder_grad_norm = tree_norm_or_zero("encoder")
            policy_grad_norm = tree_norm_or_zero("policy")
            policy_encoder_grad_norm = tree_norm_or_zero("policy_encoder")
            critic_nonfinite = nonfinite_fraction(grads["critic"])
            updates, opt_state = optimizer.update(grads, opt_state, params)
            update_norm = self.optax.tree.norm(updates)
            updated_params = self.optax.apply_updates(params, updates)
            if self.freeze_bc_policy:
                updated_params = {
                    **updated_params,
                    "policy": params["policy"],
                }
                if "policy_encoder" in params:
                    updated_params["policy_encoder"] = params[
                        "policy_encoder"
                    ]
            params = updated_params
            target_critic_params = jax.tree.map(
                lambda target, online: (1.0 - target_tau) * target
                + target_tau * online,
                target_critic_params,
                params["critic"],
            )
            (
                td_per_sample,
                td_critic_loss,
                mc_return_loss,
                mc_return_mae,
                policy_loss,
                policy_ce,
                policy_demo_top1,
                policy_entropy,
                q_span,
                q_mean,
                target_q_mean,
                causal_rct_loss,
                causal_rct_moment_loss,
                causal_valid_fraction,
                causal_treated_fraction,
                causal_tau_abs_mean,
                causal_assignment_error_max,
            ) = aux
            critic_loss = td_critic_loss + mc_return_loss + causal_rct_loss
            priority = jnp.sqrt(jnp.maximum(td_per_sample, 0.0) + 1e-10)
            priority = priority / jnp.maximum(jnp.max(priority), 1e-10)
            return (
                params,
                target_critic_params,
                opt_state,
                priority,
                {
                    "critic_loss": critic_loss,
                    "td_critic_loss": td_critic_loss,
                    "direct_q_loss": td_critic_loss,
                    "mc_return_loss": mc_return_loss,
                    "mc_return_mae": mc_return_mae,
                    "mc_return_mean": jnp.mean(mc_returns),
                    "causal_rct_loss": causal_rct_loss,
                    "causal_rct_moment_loss": causal_rct_moment_loss,
                    "causal_rct_valid_fraction": causal_valid_fraction,
                    "causal_rct_treated_fraction": causal_treated_fraction,
                    "causal_rct_tau_abs_mean": causal_tau_abs_mean,
                    "causal_rct_assignment_error_max": (
                        causal_assignment_error_max
                    ),
                    "policy_bc_loss": policy_loss,
                    "policy_ce": policy_ce,
                    "policy_demo_top1": policy_demo_top1,
                    "policy_entropy": policy_entropy,
                    "policy_grad_norm": policy_grad_norm,
                    "total_loss": total_loss,
                    "critic_grad_norm": grad_norm,
                    "direct_q_grad_norm": critic_grad_norm,
                    "encoder_grad_norm": encoder_grad_norm,
                    "policy_encoder_grad_norm": policy_encoder_grad_norm,
                    "direct_q_grad_nonfinite_fraction": critic_nonfinite,
                    "critic_update_norm": update_norm,
                    "critic_q_span": q_span,
                    "direct_q_mean": q_mean,
                    "target_q_mean": target_q_mean,
                    "loss_coeff": jnp.mean(loss_weights),
                },
            )

        return update_fn


__all__ = ["CQNDirectQAS", "action_centered_moment_loss"]
