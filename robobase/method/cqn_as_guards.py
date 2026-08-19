"""CQN-AS ``guards-schedules`` research line: run safety + schedule knobs.

This variant isolates four research-era additions that are about *running*
CQN-AS safely and instrumenting it, not about changing the learning rule:

``nonfinite_guard``
    The NaN guard.  In the research monolith this is FLAGLESS -- the update
    tail always computes finiteness flags over features / logits / targets /
    grads / updates / params / opt_state and commits the candidate state
    through ``jnp.where``, so a non-finite update is skipped and the last
    known-good state survives.  Here it is gated so that the flags-off path is
    bit-identical to the pristine official update.  When on, the update emits
    ``nan_diag/*`` metrics including ``nan_diag/update_committed``, which
    ``robobase/workspace.py::_guard_non_finite_update`` reads to abort training
    and dump forensics instead of silently training on poisoned parameters.

``bc_diagnostics``
    The BC-anchor diagnostics block (cqn-flow.md sec 64): ``bc_weight``,
    ``bc_agreement``, ``bc_binding_rate``, ``bc_margin_gap``,
    ``bc_sibling_q_span`` and ``bc_online_agreement``.  In the research
    monolith this block is unconditional; here it is gated, again so that
    flags-off is bit-identical to pristine.

``bc_lambda_schedule``
    Optional schedule string (e.g. ``"linear(1.0,0.2,10000)"``) replacing the
    constant ``bc_lambda`` for the demo FOSD / margin terms.  When set, the
    per-step weight is threaded into the jitted update as an extra argument;
    when null the update signature and the arithmetic are exactly pristine.

``demo_fosd``
    Canonical CQN-AS uses both FOSD and the expected-Q large-margin term.
    Setting this false yields the margin-only baseline used for
    objective-matched comparisons against scalar-critic variants.

Everything is a copy of the pristine method bodies from
``robobase/method/cqn.py`` (the update fn and ``update`` live in the CQN base;
Python MRO makes these overrides win for CQN-AS) with only this line's changes
applied.  Nothing here imports from the research monolith.
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

from robobase import utils
from robobase.method.cqn import project_categorical
from robobase.method.cqn_as import CQNAS, CQNASpec, cqn_as_spec_from_cfg
from robobase.method.rl_common import RLModelSpec
from robobase.replay_buffer.replay_buffer import ReplayBuffer


@dataclass(frozen=True)
class CQNASGuardedSpec(CQNASpec):
    """CQN-AS hyperparameters plus the guards/schedules knobs."""

    nonfinite_guard: bool
    bc_diagnostics: bool
    bc_lambda_schedule: str | None
    demo_fosd: bool


def cqn_as_guards_spec_from_cfg(cfg: DictConfig) -> CQNASGuardedSpec:
    method = cfg.method
    base = cqn_as_spec_from_cfg(cfg)
    base_values = {
        field.name: getattr(base, field.name) for field in fields(CQNASpec)
    }
    return CQNASGuardedSpec(
        **base_values,
        nonfinite_guard=bool(method.get("nonfinite_guard", False)),
        bc_diagnostics=bool(method.get("bc_diagnostics", False)),
        bc_lambda_schedule=(
            None
            if method.get("bc_lambda_schedule", None) is None
            else str(method.get("bc_lambda_schedule"))
        ),
        demo_fosd=bool(method.get("demo_fosd", True)),
    )


class CQNASGuarded(CQNAS):
    """CQN-AS with the optional non-finite guard, BC diagnostics and schedules.

    With every flag at its default (``nonfinite_guard=False``,
    ``bc_diagnostics=False``, ``bc_lambda_schedule=None``, ``demo_fosd=True``)
    the update graph, the jitted update signature and the emitted metrics are
    exactly the pristine official CQN-AS ones.
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
        nonfinite_guard: bool = False,
        bc_diagnostics: bool = False,
        bc_lambda_schedule: str | None = None,
        demo_fosd: bool = True,
    ):
        # These must be set before the base __init__ runs: it calls
        # ``_build_update_fn`` (overridden below), which closes over them.
        self.nonfinite_guard = bool(nonfinite_guard)
        self.bc_diagnostics = bool(bc_diagnostics)
        self.bc_lambda_schedule = (
            None if bc_lambda_schedule is None else str(bc_lambda_schedule)
        )
        self.demo_fosd = bool(demo_fosd)
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

    def _build_update_fn(self):
        optimizer = self.optimizer
        tau = self.critic_target_tau
        use_nonfinite_guard = bool(self.nonfinite_guard)
        use_bc_diagnostics = bool(self.bc_diagnostics)
        use_bc_schedule = self.bc_lambda_schedule is not None
        use_demo_fosd = bool(self.demo_fosd)
        base_bc_weight = float(self.bc_lambda)

        def array_all_finite(value):
            value = jnp.asarray(value)
            return jnp.all(jnp.isfinite(value))

        def array_max_abs_finite(value):
            value = jnp.asarray(value)
            finite_abs = jnp.where(jnp.isfinite(value), jnp.abs(value), 0.0)
            return jnp.max(finite_abs)

        def tree_all_finite(tree):
            return jnp.all(
                jnp.stack(
                    [array_all_finite(leaf) for leaf in jax.tree.leaves(tree)]
                )
            )

        def tree_max_abs_finite(tree):
            return jnp.max(
                jnp.stack(
                    [
                        array_max_abs_finite(leaf)
                        for leaf in jax.tree.leaves(tree)
                    ]
                )
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
            bc_weight,
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

                # ``bc_weight`` is the pristine ``self.bc_lambda`` float unless
                # a schedule is configured, so the arithmetic below is
                # bit-identical to pristine with the flags off.
                bc_fosd_term = jnp.asarray(0.0, dtype=jnp.float32)
                bc_margin_term = jnp.asarray(0.0, dtype=jnp.float32)
                if self.bc_lambda > 0.0 or use_bc_schedule:
                    demo_count = jnp.maximum(jnp.sum(demos), 1.0)
                    # CQN-AS historically uses both FOSD and an expected-Q
                    # margin.  Keep that behavior by default, while allowing
                    # a margin-only baseline for objective-matched flow
                    # experiments.
                    if use_demo_fosd:
                        chosen_cdf = jnp.cumsum(chosen_probabilities, axis=-1)
                        all_cdf = jnp.cumsum(all_probabilities, axis=-1)
                        fosd = jnp.maximum(
                            chosen_cdf[..., None, :] - all_cdf,
                            0.0,
                        ).sum(axis=-1).mean(axis=(1, 2, 3))
                        bc_fosd_term = bc_weight * (
                            jnp.sum(fosd * demos) / demo_count
                        )
                        critic_loss = critic_loss + bc_fosd_term
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
                        bc_margin_term = bc_weight * (
                            jnp.sum(margin * demos) / demo_count
                        )
                        critic_loss = critic_loss + bc_margin_term

                bc_diag_metrics = {}
                if use_bc_diagnostics:
                    # BC-anchor diagnostics (cqn-flow.md sec 64). The margin
                    # hinge implements the constraint
                    # Q(a_demo) >= max sibling + m, so what carries meaning
                    # across tasks is how often that constraint binds and
                    # whether it holds behaviorally -- not lambda's numeric
                    # value, which only sets a force whose counterpart (the TD
                    # force) is scaled by reward density, horizon and Q range.
                    # NOTE: in the research monolith this block is
                    # UNCONDITIONAL -- deliberately so, because gating it on
                    # ``bc_lambda > 0`` made lambda=0 arms silently
                    # instrument-blind.  Preserve that: once the flag is on the
                    # block runs regardless of bc_lambda.
                    diag_norm = jnp.maximum(jnp.sum(demos), 1.0)
                    diag_all_q = jnp.sum(
                        all_probabilities * self.support, axis=-1
                    )
                    diag_chosen_q = jnp.sum(
                        chosen_probabilities * self.support, axis=-1
                    )
                    diag_gap = diag_chosen_q - jnp.max(diag_all_q, axis=-1)
                    diag_sibling = (
                        jnp.abs(diag_chosen_q[..., None] - diag_all_q) > 1e-9
                    )
                    diag_violating = (
                        (
                            self.bc_margin
                            - (diag_chosen_q[..., None] - diag_all_q)
                        )
                        > 0.0
                    ) & diag_sibling
                    diag_binding = jnp.sum(
                        diag_violating.astype(jnp.float32), axis=(1, 2, 3)
                    ) / jnp.maximum(
                        jnp.sum(diag_sibling.astype(jnp.float32), axis=(1, 2, 3)),
                        1.0,
                    )
                    bc_diag_metrics = {
                        "bc_weight": jnp.asarray(bc_weight, dtype=jnp.float32),
                        "bc_agreement": jnp.sum(
                            (diag_gap >= -1e-6)
                            .astype(jnp.float32)
                            .mean(axis=(1, 2))
                            * demos
                        )
                        / diag_norm,
                        "bc_binding_rate": jnp.sum(diag_binding * demos)
                        / diag_norm,
                        "bc_margin_gap": jnp.sum(
                            diag_gap.mean(axis=(1, 2)) * demos
                        )
                        / diag_norm,
                        "bc_sibling_q_span": jnp.sum(
                            (
                                jnp.max(diag_all_q, axis=-1)
                                - jnp.min(diag_all_q, axis=-1)
                            ).mean(axis=(1, 2))
                            * demos
                        )
                        / diag_norm,
                        "bc_online_agreement": jnp.sum(
                            (diag_gap >= -1e-6)
                            .astype(jnp.float32)
                            .mean(axis=(1, 2))
                            * (1.0 - demos)
                        )
                        / jnp.maximum(jnp.sum(1.0 - demos), 1.0),
                    }

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
                nan_diag = {}
                if use_nonfinite_guard:
                    nan_diag = {
                        "features_all_finite": array_all_finite(features),
                        "next_features_all_finite": array_all_finite(
                            next_features
                        ),
                        "target_logits_all_finite": array_all_finite(
                            target_logits
                        ),
                        "target_probabilities_all_finite": array_all_finite(
                            target_probabilities
                        ),
                        "target_distribution_all_finite": array_all_finite(
                            target_distribution
                        ),
                        "chosen_logits_all_finite": array_all_finite(
                            chosen_logits
                        ),
                        "all_logits_all_finite": array_all_finite(all_logits),
                        "chosen_log_probabilities_all_finite": array_all_finite(
                            chosen_log_probabilities
                        ),
                        "canonical_per_sample_all_finite": array_all_finite(
                            per_sample
                        ),
                        "bc_fosd_term_all_finite": array_all_finite(
                            bc_fosd_term
                        ),
                        "bc_margin_term_all_finite": array_all_finite(
                            bc_margin_term
                        ),
                        "loss_all_finite": array_all_finite(critic_loss),
                        "features_max_abs_finite": array_max_abs_finite(
                            features
                        ),
                        "next_features_max_abs_finite": array_max_abs_finite(
                            next_features
                        ),
                        "target_logits_max_abs_finite": array_max_abs_finite(
                            target_logits
                        ),
                        "chosen_logits_max_abs_finite": array_max_abs_finite(
                            chosen_logits
                        ),
                    }
                return critic_loss, (
                    per_sample,
                    entropy,
                    target_entropy,
                    bc_diag_metrics,
                    nan_diag,
                )

            pre_params = params
            pre_target_critic_params = target_critic_params
            pre_opt_state = opt_state
            (critic_loss, aux), grads = jax.value_and_grad(
                loss_fn,
                has_aux=True,
            )(params)
            updates, candidate_opt_state = optimizer.update(
                grads, opt_state, params
            )
            candidate_params = self.optax.apply_updates(params, updates)
            candidate_target_critic_params = jax.tree.map(
                lambda target, online: (1.0 - tau) * target + tau * online,
                target_critic_params,
                candidate_params["critic"],
            )
            (
                per_sample,
                entropy,
                projected_entropy,
                bc_diag_metrics,
                nan_diag,
            ) = aux
            update_diag = {}
            update_all_finite = None
            if use_nonfinite_guard:
                update_diag = {
                    "pre_params_all_finite": tree_all_finite(pre_params),
                    "pre_target_all_finite": tree_all_finite(
                        pre_target_critic_params
                    ),
                    "pre_opt_state_all_finite": tree_all_finite(pre_opt_state),
                    "grads_all_finite": tree_all_finite(grads),
                    "updates_all_finite": tree_all_finite(updates),
                    "candidate_opt_state_all_finite": tree_all_finite(
                        candidate_opt_state
                    ),
                    "candidate_params_all_finite": tree_all_finite(
                        candidate_params
                    ),
                    "candidate_target_all_finite": tree_all_finite(
                        candidate_target_critic_params
                    ),
                    "grads_max_abs_finite": tree_max_abs_finite(grads),
                    "updates_max_abs_finite": tree_max_abs_finite(updates),
                }
                finite_flags = [
                    value
                    for key, value in {**nan_diag, **update_diag}.items()
                    if key.endswith("_all_finite")
                ]
                update_all_finite = jnp.all(jnp.stack(finite_flags))
                # Preserve the last known-good state on the first bad update.
                # For a finite update, selecting the candidate is bit-identical
                # to the pre-instrumentation return path.
                params = jax.tree.map(
                    lambda old, new: jnp.where(update_all_finite, new, old),
                    pre_params,
                    candidate_params,
                )
                target_critic_params = jax.tree.map(
                    lambda old, new: jnp.where(update_all_finite, new, old),
                    pre_target_critic_params,
                    candidate_target_critic_params,
                )
                opt_state = jax.tree.map(
                    lambda old, new: jnp.where(update_all_finite, new, old),
                    pre_opt_state,
                    candidate_opt_state,
                )
            else:
                params = candidate_params
                target_critic_params = candidate_target_critic_params
                opt_state = candidate_opt_state
            priority = jnp.sqrt(per_sample + 1e-10)
            priority = priority / jnp.maximum(jnp.max(priority), 1e-10)
            metrics = {
                "critic_loss": critic_loss,
                "entropy": entropy,
                "target_entropy": projected_entropy,
                "loss_coeff": jnp.mean(loss_weights),
            }
            if use_nonfinite_guard:
                priority = jnp.where(update_all_finite, priority, 0.0)
                metrics["nan_diag/update_committed"] = update_all_finite.astype(
                    jnp.float32
                )
                metrics.update(
                    {f"nan_diag/{key}": value for key, value in nan_diag.items()}
                )
                metrics.update(
                    {
                        f"nan_diag/{key}": value
                        for key, value in update_diag.items()
                    }
                )
            metrics.update(bc_diag_metrics)
            return (
                params,
                target_critic_params,
                opt_state,
                priority,
                metrics,
            )

        if use_bc_schedule:

            def update_fn(*args):
                (*core, bc_weight, action_key) = args
                return update_impl(*core, bc_weight, action_key)

        else:

            def update_fn(*args):
                # Keep the pristine jitted signature: the constant weight is a
                # closure value, not a traced argument.
                (*core, action_key) = args
                return update_impl(*core, base_bc_weight, action_key)

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
            # The scheduled BC weight is threaded as an extra jitted argument
            # only when a schedule is configured, so the null case keeps the
            # pristine call signature exactly.
            schedule_args = ()
            if self.bc_lambda_schedule is not None:
                schedule_args = (
                    float(utils.schedule(self.bc_lambda_schedule, step)),
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
                *schedule_args,
                self._next_action_key(),
            )
            uses_priorities = self._uses_replay_priorities(replay_buffer)
            if self._should_block_update(uses_priorities):
                self._block(jax_metrics["critic_loss"], priority)
            committed_metric = jax_metrics.get(
                "nan_diag/update_committed", None
            )
            update_committed = True
            if committed_metric is not None:
                update_committed = bool(
                    float(np.asarray(jax.device_get(committed_metric))) >= 0.5
                )
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
            elif not update_committed:
                # workspace._guard_non_finite_update aborts on this metric, so
                # a skipped update must surface even with logging disabled.
                metrics.update(
                    {
                        key: float(np.asarray(jax.device_get(value)))
                        for key, value in jax_metrics.items()
                    }
                )
        self._first_update_completed = True
        return metrics


__all__ = [
    "CQNASGuarded",
    "CQNASGuardedSpec",
    "cqn_as_guards_spec_from_cfg",
]
