"""Progress-potential shaping on top of the frozen official CQN-AS port.

Research line ``progress-shaping`` (introduced by commit ``28e88f4``, designed
in ``reports/progress_shaping_impl_20260818.md``).  The mechanism is the Ng et
al. (1999) potential form applied to the C51 target's reward scalar only::

    r~ = r + lambda * (gamma_n * bootstrap * Phi(s') - Phi(s))
    target_z = r~ + gamma_n * bootstrap * z

``Phi(s) = clip(V_prog(s), 0, 1)`` where ``V_prog`` is a state-only expectile
head reading **stop-gradient** encoder features; it never sees an action, so it
adds no action-label objective.  ``bootstrap`` (``1 - terminal`` unless
``always_bootstrap``) supplies ``Phi(terminal) = 0`` for free, so a terminal
success target is exactly ``1 - lambda * Phi(s_{T-1})``.  ``discounts`` is the
*same* ``gamma^n`` array the C51 projection multiplies into the support, so the
shaping telescopes exactly at n-step granularity.

``Phi`` is trained by expectile regression (``progress_expectile_tau``) onto the
**raw** stored label ``p_t = (t+1)/T``, optionally masked to successful
episodes / genuine demos (``progress_success_gated``).  Nothing gamma- or
lambda-dependent is ever written to replay.

Flags (all default OFF / legacy):

* ``progress_potential_weight``  -- lambda; also the enable gate for the
  target-side shaping and for the C51 support bound check.
* ``progress_potential_schedule`` -- optional ``"linear(a,b,n)"`` string
  replacing the constant lambda (``utils.schedule``).
* ``progress_head_weight``       -- weight of the expectile regression loss.
* ``progress_expectile_tau``     -- tau in ``|tau - 1{u<0}| * u^2``.
* ``progress_success_gated``     -- regress only where ``progress_valid == 1``.

With both weights at zero the head is not created at all, so the parameter
tree, the RNG stream, the traced graph and the metric set are bit-identical to
the pristine :class:`robobase.method.cqn_as.CQNAS`.
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

from robobase import utils
from robobase.method.cqn import project_categorical
from robobase.method.cqn_as import CQNAS, CQNASpec, cqn_as_spec_from_cfg
from robobase.method.rl_common import RLModelSpec, activation
from robobase.replay_buffer.replay_buffer import ReplayBuffer


def progress_shaped_rewards(
    rewards: jax.Array,
    discounts: jax.Array,
    bootstrap: jax.Array,
    phi: jax.Array,
    phi_next: jax.Array,
    weight: jax.Array | float,
) -> jax.Array:
    """Ng-form potential shaping of the per-transition reward scalar.

    ``r~ = r + lambda * (gamma * bootstrap * Phi(s') - Phi(s))``.

    ``bootstrap`` supplies ``Phi(terminal) = 0`` for free, so a terminal
    success target becomes exactly ``1 - lambda * Phi(s_T-1)``.  ``discounts``
    must be the per-transition discount actually multiplied into the C51
    target (``gamma^n`` for an n-step backup), otherwise the shaping does not
    telescope and degenerates into an arbitrary dense reward.
    """

    return rewards + weight * (discounts * bootstrap * phi_next - phi)


class ExpectileValueHead(nn.Module):
    """Scalar state-value head for IQL-style expectile regression.

    Reads (stop-gradient) encoder features only; it never queries actions, so
    it cannot leak counterfactual claims into the behavior policy.
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


@dataclass(frozen=True)
class CQNASProgressShapingSpec(CQNASpec):
    """Pristine CQN-AS spec plus the progress-shaping knobs."""

    progress_potential_weight: float
    progress_potential_schedule: str | None
    progress_head_weight: float
    progress_expectile_tau: float
    progress_success_gated: bool


# NOTE (workspace coupling): the ``progress`` / ``progress_valid`` replay
# elements are registered workspace-side by
# ``robobase/workspace.py::_progress_label_enabled``, which gates on
# ``method_name in {"cqn_as", "cqn_flow"}``.  That set must learn
# ``"cqn_as_progress_shaping"`` or this class raises ``KeyError`` at the first
# update. Agents may not edit workspace.py; see the R2 report.


def cqn_as_progress_shaping_spec_from_cfg(
    cfg: DictConfig,
) -> CQNASProgressShapingSpec:
    method = cfg.method
    base = cqn_as_spec_from_cfg(cfg)
    base_values = {
        field.name: getattr(base, field.name) for field in fields(CQNASpec)
    }
    progress_potential_weight = float(
        method.get("progress_potential_weight", 0.0)
    )
    progress_head_weight = float(method.get("progress_head_weight", 0.0))
    progress_enabled = (
        progress_potential_weight > 0.0 or progress_head_weight > 0.0
    )
    if progress_enabled:
        # Only this variant's update graph threads the potential; every other
        # graph would silently ignore it.
        method_name = str(method.get("name", "cqn_as_progress_shaping")).lower()
        if method_name not in {"cqn_as", "cqn_as_progress_shaping"}:
            raise NotImplementedError(
                "progress shaping is implemented for "
                "method=cqn_as_progress_shaping only; got "
                f"method.name={method_name}."
            )
        if bool(method.get("direct_scalar_q", False)):
            raise NotImplementedError(
                "progress shaping is not implemented on the direct scalar-Q "
                "update graph."
            )
        # The (t+1)/T label is only progress toward success when the demo
        # episode ends at its first success frame; 96% of untruncated BiGym
        # demo transitions sit in a post-success tail where the label is flat.
        if "env" in cfg:
            env_cfg = cfg.env
            if str(env_cfg.get("env_name", "")) == "bigym" and not bool(
                env_cfg.get("truncate_demo_at_success", False)
            ):
                raise ValueError(
                    "progress labels require env.truncate_demo_at_success="
                    "true; untruncated demo tails make (t+1)/T flat and "
                    "misleading."
                )
        from robobase.replay_buffer.bigym_lazy_replay import (
            lazy_replay_enabled,
        )

        if lazy_replay_enabled(cfg):
            raise ValueError(
                "progress labels require episode-backed replay; set "
                "lazy_replay.use=false."
            )
    return CQNASProgressShapingSpec(
        **base_values,
        progress_potential_weight=progress_potential_weight,
        progress_potential_schedule=(
            None
            if method.get("progress_potential_schedule", None) is None
            else str(method.get("progress_potential_schedule"))
        ),
        progress_head_weight=progress_head_weight,
        progress_expectile_tau=float(method.get("progress_expectile_tau", 0.9)),
        progress_success_gated=bool(method.get("progress_success_gated", True)),
    )


class CQNASProgressShaping(CQNAS):
    """CQN-AS with an auxiliary progress head and an Ng-form target potential.

    Every override below is a copy of the pristine method body
    (``robobase/method/cqn.py`` for ``_build_update_fn`` / ``update``) with the
    progress-shaping block applied on top; with the flags at their defaults the
    Python-level gates collapse the copy back to the pristine computation.
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
        progress_potential_weight: float = 0.0,
        progress_potential_schedule: str | None = None,
        progress_head_weight: float = 0.0,
        progress_expectile_tau: float = 0.9,
        progress_success_gated: bool = True,
    ):
        # ---- Progress-potential shaping (Ng et al. 1999 potential form) ----
        # Phi is a state-only auxiliary head; the potential enters the C51
        # target's reward scalar only.  Replay rewards stay raw so no stored
        # quantity depends on lambda or gamma.  The value checks run before
        # ``super().__init__`` so an unparsable schedule or an out-of-range tau
        # fails at construction rather than at the first update, and the gate
        # attributes are published before the pristine ``__init__`` calls
        # ``self._build_update_fn()``.
        progress_potential_weight = float(progress_potential_weight)
        progress_head_weight = float(progress_head_weight)
        if progress_potential_weight < 0.0:
            raise ValueError("progress_potential_weight must be non-negative.")
        if progress_head_weight < 0.0:
            raise ValueError("progress_head_weight must be non-negative.")
        if not 0.0 < float(progress_expectile_tau) < 1.0:
            raise ValueError("progress_expectile_tau must be in (0, 1).")
        if progress_potential_schedule is not None:
            utils.schedule(str(progress_potential_schedule), 0)
            if progress_potential_weight <= 0.0:
                raise ValueError(
                    "progress_potential_schedule requires "
                    "progress_potential_weight > 0 (the weight is the "
                    "schedule's enable gate and its bound check)."
                )
        self.progress_potential_weight = progress_potential_weight
        self.progress_potential_schedule = (
            None
            if progress_potential_schedule is None
            else str(progress_potential_schedule)
        )
        self.progress_head_weight = progress_head_weight
        self.progress_expectile_tau = float(progress_expectile_tau)
        self.progress_success_gated = bool(progress_success_gated)
        # The head is instantiated whenever either consumer needs it.
        self.progress_head_enabled = bool(
            progress_head_weight > 0.0 or progress_potential_weight > 0.0
        )
        self.progress_shaping_enabled = progress_potential_weight > 0.0
        self.progress_value_model = None

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
            temporal_ensemble_replan_interval=temporal_ensemble_replan_interval,
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

        if self.progress_head_enabled:
            # Guards against update graphs this line never wired.  None of
            # these attributes exist on the pristine class, so the checks are
            # inert here and only bite if a future composition sets them.
            if bool(getattr(self, "pessimistic_twin_critic", False)):
                raise NotImplementedError(
                    "progress shaping is not implemented on the "
                    "pessimistic_twin_critic update graph."
                )
            if bool(getattr(self, "direct_scalar_q", False)):
                raise NotImplementedError(
                    "progress shaping is not implemented on the direct "
                    "scalar-Q update graph."
                )
        if self.progress_shaping_enabled:
            # Raw Monte-Carlo consumers assume UNSHAPED {0,1}-scale returns.
            # Mixing them with shaped Bellman targets silently inverts the
            # lower-bound mask, so refuse instead of guessing the shift.
            mc_conflicts = [
                name
                for name, active in (
                    (
                        "mc_lower_bound_target",
                        bool(getattr(self, "mc_lower_bound_target", False)),
                    ),
                    (
                        "episodic_success_q_target",
                        bool(getattr(self, "episodic_success_q_target", False)),
                    ),
                    (
                        "ordered_success_return_mix",
                        float(getattr(self, "ordered_success_return_mix", 0.0))
                        > 0.0,
                    ),
                )
                if active
            ]
            if mc_conflicts:
                raise ValueError(
                    "progress_potential_weight > 0 cannot be combined with "
                    "raw Monte-Carlo targets ("
                    + ", ".join(mc_conflicts)
                    + "); the shifted-MC variant is not implemented."
                )
            # C51 support headroom: the largest shaped reward scalar is
            # (max sparse return + lambda) * q_reward_scale and
            # project_categorical clips silently at v_max.  The pristine graph
            # has no q_reward_scale, i.e. it is identically 1.0.
            shaped_bound = (1.0 + progress_potential_weight) * float(
                getattr(self, "q_reward_scale", 1.0)
            )
            if shaped_bound > float(v_max) + 1e-6:
                raise ValueError(
                    "shaped target bound (1 + progress_potential_weight) * "
                    f"q_reward_scale = {shaped_bound:.4f} exceeds v_max="
                    f"{float(v_max):.4f}; C51 projection would clip silently."
                )

        if self.progress_head_enabled:
            # Scalar state-value head on the canonical CQN-AS platform: it
            # reads stop-gradient features and never an action, so it adds no
            # action-label objective.
            self.progress_value_model = ExpectileValueHead(
                hidden_dims=model.hidden_dims,
                activation_name=model.activation,
            )
            # fold_in rather than split: adding the head must not consume the
            # rollout/update RNG stream, so a progress-enabled arm keeps the
            # exact legacy action keys and stays comparable to its control.
            self.params["progress_value"] = self.progress_value_model.init(
                jax.random.fold_in(self.rng_key, 0x9209),
                jnp.zeros((1, self._rl_feature_dim), dtype=jnp.float32),
            )
            # The pristine ``__init__`` sized the optimizer state before the
            # head existed; re-initialising is deterministic (adamw zeros) and
            # only adds the new parameter group.
            self.opt_state = self.optimizer.init(self.params)

    # ------------------------------------------------------------------
    # Update graph (copy of ``CQN._build_update_fn`` + the progress block)
    # ------------------------------------------------------------------
    def _build_update_fn(self):
        optimizer = self.optimizer
        tau = self.critic_target_tau
        # ``use_progress_head`` gates the whole block (parameter group + batch
        # elements + metrics); ``use_progress_shaping`` additionally rewrites
        # the target reward scalar. Both false is the exact legacy graph.
        use_progress_head = bool(getattr(self, "progress_head_enabled", False))
        use_progress_shaping = bool(
            getattr(self, "progress_shaping_enabled", False)
        )
        progress_head_weight = float(getattr(self, "progress_head_weight", 0.0))
        progress_expectile_tau = float(
            getattr(self, "progress_expectile_tau", 0.9)
        )
        progress_success_gated = bool(
            getattr(self, "progress_success_gated", True)
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
            progress_labels,
            progress_valid,
            progress_lambda,
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
                # Ng-form potential shaping. ``progress_phi_raw`` is the
                # unclipped head output kept for the regression loss; the
                # potential itself is clipped into [0, 1] and stop-gradient so
                # the shaping term is a pure per-state constant.
                progress_zero = jnp.asarray(0.0, dtype=jnp.float32)
                progress_phi_raw = jnp.zeros_like(rewards)
                progress_phi = jnp.zeros_like(rewards)
                progress_phi_next = jnp.zeros_like(rewards)
                if use_progress_head:
                    progress_phi_raw = self.progress_value_model.apply(
                        current_params["progress_value"],
                        jax.lax.stop_gradient(features),
                    )
                    progress_phi = jax.lax.stop_gradient(
                        jnp.clip(progress_phi_raw, 0.0, 1.0)
                    )
                    progress_phi_next = jax.lax.stop_gradient(
                        jnp.clip(
                            self.progress_value_model.apply(
                                current_params["progress_value"],
                                next_features,
                            ),
                            0.0,
                            1.0,
                        )
                    )
                shaped_rewards = rewards
                progress_clip_fraction = progress_zero
                if use_progress_shaping:
                    shaped_rewards = progress_shaped_rewards(
                        rewards,
                        discounts,
                        bootstrap,
                        progress_phi,
                        progress_phi_next,
                        progress_lambda,
                    )
                    # project_categorical clips silently at the support edges
                    # and clipping breaks the telescope, so measure it.
                    shaped_atom_targets = shaped_rewards[:, None] + (
                        bootstrap * discounts
                    )[:, None] * self.support[None, :]
                    progress_clip_fraction = jnp.mean(
                        (
                            (shaped_atom_targets < self.support[0])
                            | (shaped_atom_targets > self.support[-1])
                        ).astype(jnp.float32)
                    )
                target_distribution = project_categorical(
                    target_probabilities,
                    shaped_rewards,
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
                progress_head_loss = progress_zero
                progress_value_mean = progress_zero
                progress_valid_fraction = progress_zero
                if use_progress_head:
                    # Expectile regression of the state-only head onto the raw
                    # (t+1)/T replay label.  tau > 0.5 fits the optimistic
                    # envelope, absorbing the fact that a time index is a
                    # task clock rather than a task state.
                    progress_error = progress_labels - progress_phi_raw
                    expectile_weight = jnp.where(
                        progress_error < 0.0,
                        1.0 - progress_expectile_tau,
                        progress_expectile_tau,
                    )
                    progress_mask = (
                        progress_valid
                        if progress_success_gated
                        else jnp.ones_like(progress_valid)
                    )
                    progress_valid_fraction = jnp.mean(progress_mask)
                    progress_head_loss = progress_head_weight * (
                        jnp.sum(
                            expectile_weight
                            * jnp.square(progress_error)
                            * progress_mask
                        )
                        / jnp.maximum(jnp.sum(progress_mask), 1.0)
                    )
                    progress_value_mean = jnp.mean(progress_phi)
                    if progress_head_weight > 0.0:
                        critic_loss = critic_loss + progress_head_loss
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
                    progress_head_loss,
                    progress_value_mean,
                    progress_valid_fraction,
                    progress_clip_fraction,
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
                progress_head_loss,
                progress_value_mean,
                progress_valid_fraction,
                progress_clip_fraction,
            ) = aux
            priority = jnp.sqrt(per_sample + 1e-10)
            priority = priority / jnp.maximum(jnp.max(priority), 1e-10)
            metrics = {
                "critic_loss": critic_loss,
                "entropy": entropy,
                "target_entropy": projected_entropy,
                "loss_coeff": jnp.mean(loss_weights),
            }
            if use_progress_head:
                metrics["progress_head_loss"] = progress_head_loss
                metrics["progress_head_value_mean"] = progress_value_mean
                metrics["progress_label_mean"] = jnp.mean(progress_labels)
                metrics["progress_valid_fraction"] = progress_valid_fraction
            if use_progress_shaping:
                metrics["progress_potential_lambda"] = jnp.asarray(
                    progress_lambda,
                    dtype=jnp.float32,
                )
                metrics["progress_shaping_clip_frac"] = progress_clip_fraction
            return (
                params,
                target_critic_params,
                opt_state,
                priority,
                metrics,
            )

        def split_progress_args(args):
            # (progress, progress_valid, lambda) are threaded immediately
            # before action_key; every other configuration passes none and
            # gets the exact legacy zero-potential graph.
            if not use_progress_head:
                rewards = args[6]
                return args, (
                    jnp.zeros_like(rewards),
                    jnp.zeros_like(rewards),
                    jnp.asarray(0.0, dtype=jnp.float32),
                )
            (*rest, labels, valid, weight, action_key) = args
            return (*rest, action_key), (labels, valid, weight)

        def update_fn(*args):
            args, progress_args = split_progress_args(args)
            (*core, action_key) = args
            return update_impl(*core, *progress_args, action_key)

        return update_fn

    def _progress_update_args(self, batch: dict, step: int) -> tuple:
        """Raw progress labels plus the current shaping lambda.

        The label is the stored time index ``(t+1)/T``; nothing gamma- or
        lambda-dependent is ever read from replay, so a lambda sweep never
        invalidates the shared demo cache.
        """

        missing = [
            name
            for name in ("progress", "progress_valid")
            if name not in batch
        ]
        if missing:
            raise KeyError(
                "progress shaping requires episode-backed replay elements; "
                "missing: " + ", ".join(missing)
            )
        progress_labels = self._as_jax_array(
            batch["progress"], self.jnp.float32
        ).reshape(-1)
        progress_valid = self._as_jax_array(
            batch["progress_valid"], self.jnp.float32
        ).reshape(-1)
        schedule = getattr(self, "progress_potential_schedule", None)
        weight = float(getattr(self, "progress_potential_weight", 0.0))
        if schedule is not None:
            weight = float(utils.schedule(schedule, step))
        return (
            progress_labels,
            progress_valid,
            jnp.asarray(weight, dtype=jnp.float32),
        )

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
            progress_args = ()
            if self.progress_head_enabled:
                progress_args = self._progress_update_args(batch, step)
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
                *progress_args,
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
    "CQNASProgressShaping",
    "CQNASProgressShapingSpec",
    "ExpectileValueHead",
    "cqn_as_progress_shaping_spec_from_cfg",
    "progress_shaped_rewards",
]
