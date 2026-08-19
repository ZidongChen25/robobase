"""Advantage-weighted regression (AWR) auxiliary for CQN-AS.

Research line 7 of ``CQN_REFACTOR_PLAN.md``: flags ``awr_beta``,
``awr_weight_max`` and ``awr_expectile_tau``.  Route-(c) "support-only
improvement" from ``cqn-flow.md`` section 26.2, shipped as the Stage-145 gate
``robobase/cfgs/launch/cqn_as_pixel_bigym_stage145_awr_gate.yaml``.

The line has two pieces, both transcribed from the research monolith
``robobase/method/cqn_as_research.py`` lines 5282-5327:

1. **IQL-style expectile state value.**  A scalar head ``V(s)``
   (:class:`ExpectileValueHead`) reads *stop-gradient* encoder features -- it
   never queries an action, so it cannot leak counterfactual (unexecuted
   action) claims -- and is regressed onto the replayed Monte-Carlo return
   ``mc_return`` with the asymmetric expectile loss
   ``E[|tau - 1{u<0}| u^2]``, ``u = mc_return - V(s)``.  Its loss is added to
   the differentiated objective; ``critic_loss`` keeps reporting the critic
   term alone, exactly as in the research monolith.

2. **Advantage-weighted behavior cloning.**  ``w = clip(exp(u / beta), 0,
   awr_weight_max)`` (stop-gradient) replaces the demonstration mask in the
   behavior-cloning objective, so the BC term is a self-normalised weighted
   mean ``sum(x * w) / max(sum(w), 1e-6)`` over demo **and** online
   transitions instead of a demo-only mean.  Failed rollouts are suppressed by
   their own completed return rather than excluded by provenance.

Coupling (Phase R2 coupling protocol, see the module's hand-off report)
---------------------------------------------------------------------
In the research monolith piece 2 reweights the cross-entropy of the *separate
BC policy head* owned by the ``bc-policy`` line, and ``__init__`` hard-refuses
``awr_beta`` without it::

    cqn_as_research.py:2370-2371
        if not separate_bc_policy:
            raise ValueError("awr_beta requires separate_bc_policy=true.")

That head does not exist on the pristine critic, and the whole AWR block lives
inside ``_build_separate_policy_update_fn`` (research gate at
``cqn_as_research.py:4801``), a different update graph and a different
parameter tree.  Reproducing it here would mean absorbing the ``bc-policy``
line wholesale, which the refactor forbids.  This file therefore keeps piece 1
byte-for-byte and applies the *identical* weighting transform of piece 2 to the
pristine critic's OWN behaviour-cloning objective: the demo-masked FOSD and
margin terms of ``CQN._build_update_fn`` (``robobase/method/cqn.py`` lines
581-608).  Concretely ``sum(x * demos) / max(sum(demos), 1)`` becomes
``sum(x * w) / max(sum(w), 1e-6)`` -- the same estimator the research code
applies to ``policy_per_sample``.  Because the pristine BC objective is the
consumer of the weights, ``awr_beta`` requires ``bc_lambda > 0`` here, which is
the transitive form of the research requirement (``separate_bc_policy=true``
itself requires ``bc_lambda > 0``, ``cqn_as_research.py:1649-1650``).

``mc_return`` plumbing
---------------------
The expectile target is read straight off the replay batch with
``batch.get("mc_return", zeros)``, exactly as the research ``update()`` does
(``cqn_as_research.py:6507-6510``).  Storage of that replay element is decided
by ``robobase.workspace._mc_return_anchor_enabled``, which keys off the
``mc-rct`` line's flags (``mc_return_weight`` / ``mc_lower_bound_target`` /
``episodic_success_q_target``) and off the method *name*.  ``awr_beta`` is not
one of its conditions, so an AWR-only configuration silently regresses ``V``
towards zeros; the Stage-145 gate works around this by also setting
``mc_return_weight: 0.1``.  This is research-era behaviour and is reproduced
unchanged here -- registering ``cqn_as_awr`` must extend that helper.
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

from robobase.method.cqn import project_categorical
from robobase.method.cqn_as import CQNAS, CQNASpec, cqn_as_spec_from_cfg
from robobase.method.rl_common import RLModelSpec, activation
from robobase.replay_buffer.replay_buffer import ReplayBuffer

__all__ = [
    "CQNASAwr",
    "CQNASAwrSpec",
    "ExpectileValueHead",
    "cqn_as_awr_spec_from_cfg",
]


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


@dataclass(frozen=True)
class CQNASAwrSpec(CQNASpec):
    """CQN-AS hyperparameters plus the AWR auxiliary settings."""

    awr_beta: float | None
    awr_weight_max: float
    awr_expectile_tau: float


def cqn_as_awr_spec_from_cfg(cfg: DictConfig) -> CQNASAwrSpec:
    method = cfg.method
    base = cqn_as_spec_from_cfg(cfg)
    base_values = {field.name: getattr(base, field.name) for field in fields(CQNASpec)}
    return CQNASAwrSpec(
        **base_values,
        awr_beta=(
            None
            if method.get("awr_beta", None) is None
            else float(method.get("awr_beta"))
        ),
        awr_weight_max=float(method.get("awr_weight_max", 10.0)),
        awr_expectile_tau=float(method.get("awr_expectile_tau", 0.7)),
    )


class CQNASAwr(CQNAS):
    """CQN-AS with the advantage-weighted-regression auxiliary.

    ``awr_beta=None`` (the default) is the pristine :class:`CQNAS` graph, RNG
    stream and parameter tree, bit for bit: no extra head is created and the
    update function is the pristine one with an unused ``mc_returns``
    argument threaded through.
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
        awr_beta: float | None = None,
        awr_weight_max: float = 10.0,
        awr_expectile_tau: float = 0.7,
    ):
        # Mirrors cqn_as_research.py:2367-2378.  ``separate_bc_policy`` does not
        # exist on the pristine platform; ``bc_lambda > 0`` is the transitive
        # requirement (see the module docstring).
        if awr_beta is not None:
            if float(awr_beta) <= 0.0:
                raise ValueError("awr_beta must be positive.")
            if float(bc_lambda) <= 0.0:
                raise ValueError(
                    "awr_beta requires bc_lambda > 0: the advantage weights "
                    "replace the demonstration mask in the critic's "
                    "behavior-cloning terms, which are inactive when "
                    "bc_lambda=0."
                )
        if awr_weight_max <= 0.0:
            raise ValueError("awr_weight_max must be positive.")
        if not 0.0 < awr_expectile_tau < 1.0:
            raise ValueError("awr_expectile_tau must be in (0, 1).")
        self.awr_beta = None if awr_beta is None else float(awr_beta)
        self.awr_weight_max = float(awr_weight_max)
        self.awr_expectile_tau = float(awr_expectile_tau)
        self.expectile_value_model = None

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

        if self.awr_beta is not None:
            # cqn_as_research.py:2645-2656.  ``split`` (not ``fold_in``) matches
            # the research monolith, so an AWR arm's rollout/update RNG stream
            # is shifted relative to its flags-off control -- as it was in the
            # Stage-145 waves.
            self.expectile_value_model = ExpectileValueHead(
                hidden_dims=model.hidden_dims,
                activation_name=model.activation,
            )
            self.rng_key, value_key = jax.random.split(self.rng_key)
            self.params["expectile_value"] = self.expectile_value_model.init(
                value_key,
                jnp.zeros((1, self._rl_feature_dim), dtype=jnp.float32),
            )
            # The head joins the single shared optimizer, exactly as in the
            # research monolith (one adamw over the whole params dict).
            self.opt_state = self.optimizer.init(self.params)

    def _build_update_fn(self):
        """Pristine ``CQN._build_update_fn`` plus the AWR auxiliary.

        Copied from ``robobase/method/cqn.py`` lines 505-648; the additions are
        the ``mc_returns`` argument, the expectile-value block
        (``cqn_as_research.py:5287-5323``) and the AWR-weighted behaviour
        cloning that replaces the demo mask.  With ``awr_beta=None`` every
        added branch is a Python-level ``if`` that is not taken, so the traced
        graph is the pristine one.
        """

        optimizer = self.optimizer
        tau = self.critic_target_tau
        use_awr = self.awr_beta is not None

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

                awr_zero = jnp.asarray(0.0, dtype=jnp.float32)
                awr_value_loss = awr_zero
                awr_value_mean = awr_zero
                awr_weight_mean = awr_zero
                awr_weight_ess = awr_zero
                awr_weights = None
                weight_sum = None
                if use_awr:
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
                    awr_weight_mean = jnp.mean(awr_weights)
                    awr_weight_ess = jnp.square(weight_sum) / (
                        jnp.maximum(jnp.sum(jnp.square(awr_weights)), 1e-6)
                        * awr_weights.shape[0]
                    )

                if self.bc_lambda > 0.0:
                    chosen_cdf = jnp.cumsum(chosen_probabilities, axis=-1)
                    all_cdf = jnp.cumsum(all_probabilities, axis=-1)
                    fosd = jnp.maximum(
                        chosen_cdf[..., None, :] - all_cdf,
                        0.0,
                    ).sum(axis=-1).mean(axis=(1, 2, 3))
                    if use_awr:
                        critic_loss = critic_loss + self.bc_lambda * (
                            jnp.sum(fosd * awr_weights) / weight_sum
                        )
                    else:
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
                        if use_awr:
                            critic_loss = critic_loss + self.bc_lambda * (
                                jnp.sum(margin * awr_weights) / weight_sum
                            )
                        else:
                            demo_count = jnp.maximum(jnp.sum(demos), 1.0)
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
                # ``critic_loss`` stays the reported critic term; only the
                # differentiated objective carries the auxiliary value loss
                # (cqn_as_research.py:5381-5387 / 5520).
                total_loss = critic_loss
                if use_awr:
                    total_loss = critic_loss + awr_value_loss
                return total_loss, (
                    per_sample,
                    critic_loss,
                    entropy,
                    target_entropy,
                    awr_value_loss,
                    awr_value_mean,
                    awr_weight_mean,
                    awr_weight_ess,
                )

            (_, aux), grads = jax.value_and_grad(
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
                critic_loss,
                entropy,
                projected_entropy,
                awr_value_loss,
                awr_value_mean,
                awr_weight_mean,
                awr_weight_ess,
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
                    "entropy": entropy,
                    "target_entropy": projected_entropy,
                    "loss_coeff": jnp.mean(loss_weights),
                    "awr_value_loss": awr_value_loss,
                    "awr_value_mean": awr_value_mean,
                    "awr_weight_mean": awr_weight_mean,
                    "awr_weight_ess": awr_weight_ess,
                },
            )

        return update_fn

    def update(
        self,
        replay_iter: Iterator[dict],
        step: int,
        replay_buffer: ReplayBuffer = None,
    ) -> dict[str, np.ndarray]:
        """Pristine ``CQN.update`` plus the ``mc_return`` expectile target.

        Copied from ``robobase/method/cqn.py`` lines 689-763; the only change
        is the ``mc_returns`` batch read (``cqn_as_research.py:6507-6510``) and
        its position in the ``_update_impl`` argument list, immediately after
        ``demos`` -- exactly where the research monolith threads it.
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
            # Absent unless ``robobase.workspace._mc_return_anchor_enabled``
            # registered the replay element -- see the module docstring.
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
