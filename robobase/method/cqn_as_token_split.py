"""CQN-AS with per-token horizon-split TD targets (R-line wave 2).

SEAR-inspired variant of the pristine CQN-AS sequence critic: the action
chunk is partitioned along the token axis for *target* computation only.
Tokens whose 1-based chunk index is at or below ``token_split_boundary``
keep the exact legacy 1-step Bellman backup; the remaining tokens regress
to an auxiliary long-horizon (``replay.auxiliary_nstep``) backup taken from
the same start state.  Nothing about action selection, execution or the
online loss shape changes.

Flags (both OFF by default -> byte-identical graph to pristine ``CQNAS``):

* ``token_split_horizon_targets``: master switch.
* ``token_split_boundary``: 1-based token index in ``[1, action_sequence)``.

The auxiliary transition arrives in the replay batch under the ``_tp_aux``
suffix (``<obs>_tp_aux``, ``action_tp_aux``, ``reward_aux``,
``discount_aux``, ``terminal_aux``), which the uniform replay buffer emits
when ``replay.auxiliary_nstep > 1``.

This module follows the R2 extraction pattern: it subclasses the FROZEN
pristine classes in ``robobase/method/cqn_as.py`` / ``cqn.py`` and overrides
``_build_update_fn`` / ``update`` with copies of the pristine bodies plus the
token-split logic.  It never imports from the research monolith.
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

from robobase.method.cqn import project_categorical
from robobase.method.cqn_as import (
    CQNAS,
    CQNASpec,
    cqn_as_spec_from_cfg,
    random_shift_rgb,
)
from robobase.method.rl_common import RLModelSpec
from robobase.replay_buffer.replay_buffer import ReplayBuffer

# Fixed RNG fold-in offsets, kept identical to the research monolith so the
# auxiliary-horizon stream is reproducible across the extraction.
_AUX_RGB_SHIFT_FOLD = 4243
_AUX_ACTION_FOLD = 4245


@dataclass(frozen=True)
class CQNASTokenSplitSpec(CQNASpec):
    """Pristine CQN-AS hyperparameters plus the token-split flags."""

    token_split_horizon_targets: bool
    token_split_boundary: int | None


def cqn_as_token_split_spec_from_cfg(cfg: DictConfig) -> CQNASTokenSplitSpec:
    """Pristine CQN-AS spec + token-split keys, with the replay-chain check.

    The replay-side requirements are validated here (not in ``__init__``)
    because they live in ``cfg.replay``, which the agent never sees.
    """

    method = cfg.method
    base = cqn_as_spec_from_cfg(cfg)
    base_values = {
        field.name: getattr(base, field.name) for field in fields(CQNASpec)
    }
    token_split_horizon_targets = bool(
        method.get("token_split_horizon_targets", False)
    )
    if token_split_horizon_targets:
        token_split_nstep = cfg.replay.get("auxiliary_nstep", None)
        token_split_violations = []
        if int(cfg.replay.get("nstep", 1)) != 1:
            token_split_violations.append("replay.nstep=1")
        if token_split_nstep is None or int(token_split_nstep) <= 1:
            token_split_violations.append("replay.auxiliary_nstep > 1")
        if not bool(cfg.replay.get("include_tp1", True)):
            token_split_violations.append("replay.include_tp1=true")
        if token_split_violations:
            raise ValueError(
                "token_split_horizon_targets requires the auxiliary-horizon "
                "replay fields: " + "; ".join(token_split_violations)
            )
    return CQNASTokenSplitSpec(
        **base_values,
        token_split_horizon_targets=token_split_horizon_targets,
        token_split_boundary=(
            None
            if method.get("token_split_boundary", None) is None
            else int(method.get("token_split_boundary"))
        ),
    )


class CQNASTokenSplit(CQNAS):
    """Pristine CQN-AS with an optional per-token target horizon split."""

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
        token_split_horizon_targets: bool = False,
        token_split_boundary: int | None = None,
    ):
        # Set before super().__init__: the pristine constructor calls
        # ``_build_update_fn`` on its last lines, and the override below reads
        # these attributes at graph-build time.
        self.token_split_horizon_targets = bool(token_split_horizon_targets)
        self.token_split_boundary = (
            None if token_split_boundary is None else int(token_split_boundary)
        )
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
        # ``self.action_sequence`` only exists after the base constructor, so
        # the boundary-range check has to run here.  Pristine CQN-AS already
        # rejects action_sequence < 2 above.
        if self.token_split_horizon_targets:
            if self.token_split_boundary is None:
                raise ValueError(
                    "token_split_horizon_targets requires an explicit "
                    "token_split_boundary (1-based token index at or below "
                    "which the exact legacy 1-step backup is kept)."
                )
            if not 1 <= self.token_split_boundary < int(self.action_sequence):
                raise ValueError(
                    "token_split_boundary must lie in [1, action_sequence)."
                )

    def _auxiliary_rl_obs_inputs(self, batch: dict):
        """Prepare the auxiliary long-horizon bootstrap state (``_tp_aux``)."""

        suffix = "_tp_aux"
        observation_keys = self.observation_space.keys()
        if self._has_cached_pixel_features(batch):
            observation_keys = tuple(
                key for key in observation_keys if key not in self._rgb_batch_keys
            )
        auxiliary_batch = {}
        missing = []
        for key in observation_keys:
            auxiliary_key = f"{key}{suffix}"
            if auxiliary_key not in batch:
                missing.append(auxiliary_key)
            else:
                auxiliary_batch[key] = batch[auxiliary_key]
        if self._has_cached_pixel_features(batch):
            cached_auxiliary_key = f"{self._cached_pixel_feature_key}{suffix}"
            if cached_auxiliary_key not in batch:
                missing.append(cached_auxiliary_key)
            else:
                auxiliary_batch[self._cached_pixel_feature_key] = batch[
                    cached_auxiliary_key
                ]
        if missing:
            raise KeyError(
                "auxiliary TD replay batch is missing: "
                + ", ".join(sorted(missing))
            )
        return self._prepare_rl_obs_inputs(auxiliary_batch)

    def _build_update_fn(self):
        # Copy of the pristine ``CQN._build_update_fn`` body; the only edits
        # are the ``use_token_split`` blocks (auxiliary-horizon target and its
        # two wiring metrics) and the optional auxiliary tensor tail.
        optimizer = self.optimizer
        tau = self.critic_target_tau
        use_token_split = bool(
            getattr(self, "token_split_horizon_targets", False)
        )
        token_split_boundary = getattr(self, "token_split_boundary", None)

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
            action_key,
            aux_next_obs_inputs=None,
            aux_next_actions=None,
            aux_rewards=None,
            aux_discounts=None,
            aux_bootstrap=None,
        ):
            del aux_next_actions  # td_target_action_source is always "critic".
            obs_inputs, next_obs_inputs, action_key = (
                self._augment_update_obs_inputs(
                    obs_inputs,
                    next_obs_inputs,
                    action_key,
                )
            )
            if use_token_split and isinstance(aux_next_obs_inputs, dict):
                aux_next_obs_inputs = dict(aux_next_obs_inputs)
                if "rgb" in aux_next_obs_inputs:
                    aux_next_obs_inputs["rgb"] = random_shift_rgb(
                        aux_next_obs_inputs["rgb"],
                        jax.random.fold_in(action_key, _AUX_RGB_SHIFT_FOLD),
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
                token_split_aux_fraction = jnp.asarray(0.0, dtype=jnp.float32)
                token_split_aux_reward_mean = jnp.asarray(
                    0.0, dtype=jnp.float32
                )
                if use_token_split:
                    aux_features = self._rl_features(
                        encoder_params,
                        aux_next_obs_inputs,
                        stop_gradient=True,
                    )
                    aux_next_action, _ = self._greedy_action_for_update(
                        current_params["critic"],
                        aux_features,
                        jax.random.fold_in(action_key, _AUX_ACTION_FOLD),
                    )
                    aux_target_logits, _ = self._critic_logits_per_level(
                        target_critic_params,
                        aux_features,
                        aux_next_action,
                    )
                    aux_target_probabilities = jax.nn.softmax(
                        aux_target_logits,
                        axis=-1,
                    )
                    aux_target_distribution = project_categorical(
                        aux_target_probabilities,
                        aux_rewards,
                        aux_discounts,
                        aux_bootstrap,
                        self.support,
                    )
                    if self.centralized_critic:
                        aux_target_distribution = jnp.broadcast_to(
                            aux_target_distribution.mean(
                                axis=-2, keepdims=True
                            ),
                            aux_target_distribution.shape,
                        )
                    # The D axis is laid out [token 0 dims..., token 1 dims,
                    # ...]; tokens whose 1-based index exceeds the boundary
                    # regress to the long-horizon (auxiliary_nstep) backup,
                    # the rest keep the exact legacy 1-step backup.
                    token_index = (
                        jnp.arange(target_distribution.shape[2])
                        // self.action_dim
                    )
                    aux_token_mask = (
                        token_index + 1
                    ) > int(token_split_boundary)
                    target_distribution = jnp.where(
                        aux_token_mask[None, None, :, None],
                        aux_target_distribution,
                        target_distribution,
                    )
                    token_split_aux_fraction = jnp.mean(
                        aux_token_mask.astype(jnp.float32)
                    )
                    token_split_aux_reward_mean = jnp.mean(aux_rewards)
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
                    token_split_aux_fraction,
                    token_split_aux_reward_mean,
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
                token_split_aux_fraction,
                token_split_aux_reward_mean,
            ) = aux
            priority = jnp.sqrt(per_sample + 1e-10)
            priority = priority / jnp.maximum(jnp.max(priority), 1e-10)
            metrics = {
                "critic_loss": critic_loss,
                "entropy": entropy,
                "target_entropy": projected_entropy,
                "loss_coeff": jnp.mean(loss_weights),
            }
            if use_token_split:
                metrics["token_split_aux_fraction"] = token_split_aux_fraction
                metrics["token_split_aux_reward_mean"] = (
                    token_split_aux_reward_mean
                )
            return (
                params,
                target_critic_params,
                opt_state,
                priority,
                metrics,
            )

        # token-split forwards five auxiliary-horizon tensors after
        # action_key; with the flag off ``update`` passes none, so the traced
        # signature and graph are exactly the pristine ones.
        return update_impl

    def update(
        self,
        replay_iter: Iterator[dict],
        step: int,
        replay_buffer: ReplayBuffer = None,
    ) -> dict[str, np.ndarray]:
        # Copy of the pristine ``CQN.update`` body; the only edit is the
        # auxiliary-horizon batch extraction appended to the update call.
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
            auxiliary_args = ()
            if self.token_split_horizon_targets:
                required_auxiliary = (
                    "action_tp_aux",
                    "reward_aux",
                    "discount_aux",
                    "terminal_aux",
                )
                missing_auxiliary = [
                    name for name in required_auxiliary if name not in batch
                ]
                if missing_auxiliary:
                    raise KeyError(
                        "auxiliary-horizon targets require "
                        "replay.auxiliary_nstep; missing: "
                        + ", ".join(missing_auxiliary)
                    )
                auxiliary_next_obs_inputs = self._auxiliary_rl_obs_inputs(batch)
                auxiliary_action_values = batch["action_tp_aux"]
                auxiliary_next_actions = self._as_jax_array(
                    auxiliary_action_values,
                    self.jnp.float32,
                ).reshape((auxiliary_action_values.shape[0], -1))
                auxiliary_rewards = self._as_jax_array(
                    batch["reward_aux"], self.jnp.float32
                ).reshape(-1)
                auxiliary_discounts = self._as_jax_array(
                    batch["discount_aux"], self.jnp.float32
                ).reshape(-1)
                auxiliary_terminal = self._as_jax_array(
                    batch["terminal_aux"], self.jnp.float32
                ).reshape(-1)
                auxiliary_bootstrap = (
                    jnp.ones_like(auxiliary_terminal)
                    if self.always_bootstrap
                    else 1.0 - auxiliary_terminal
                )
                auxiliary_args = (
                    auxiliary_next_obs_inputs,
                    auxiliary_next_actions,
                    auxiliary_rewards,
                    auxiliary_discounts,
                    auxiliary_bootstrap,
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
                self._next_action_key(),
                *auxiliary_args,
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
    "CQNASTokenSplit",
    "CQNASTokenSplitSpec",
    "cqn_as_token_split_spec_from_cfg",
]
