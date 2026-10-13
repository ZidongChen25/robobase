"""Q-chunking with a coarse-to-fine factorized critic (QC x CQN-AS C2F).

Everything except the critic is inherited from ``QChunking``: flow-BC actor,
open-loop chunk execution, K-step chunk TD, feature trunk, augmentation and
low-dim masking.  The joint scalar critic is replaced by a CQN-style
coarse-to-fine critic: for every level ``l``, chunk step ``k`` and action
dimension ``d`` it scores the ``B`` sub-bins of the current interval, given the
interval midpoints of the whole chunk.  An ensemble of heads gives an
epistemic spread used for lower-confidence-bound argmax.

Value learning stays on the data/BC manifold: each (l, k, d, head) regresses
onto the same K-step target, bootstrapped at the Best-of-N flow action of the
next state (never at an off-manifold argmax).  Several action-extraction modes
probe whether the factorized critic can act on its own:

* ``bon``         flow candidates scored by the C2F critic (QC behaviour);
* ``argmax``      pure C2F greedy over all bins;
* ``argmax_lcb``  C2F greedy on mean - kappa * std over heads;
* ``support``     C2F greedy restricted to bins reached by flow candidates;
* ``support_lcb`` both restrictions.
"""

from __future__ import annotations

import time
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from robobase.method.q_chunking import (
    _OFFICIAL_DENSE_INIT,
    QChunking,
    q_chunking_td_target,
)
from robobase.method.rl_common import activation, random_shift_rgb

ACT_MODES = ("bon", "argmax", "argmax_lcb", "support", "support_lcb")


class C2FChunkCritic(nn.Module):
    """Ensemble of MLP heads: (features, level one-hot, midpoints) -> [N, K, D, B, E]."""

    hidden_dims: tuple[int, ...]
    action_sequence: int
    action_dim: int
    bins: int
    num_heads: int
    activation_name: str = "gelu"

    @nn.compact
    def __call__(self, features, level_one_hot, midpoints):
        x = jnp.concatenate(
            [
                features.astype(jnp.float32),
                level_one_hot.astype(jnp.float32),
                midpoints.astype(jnp.float32),
            ],
            axis=-1,
        )
        heads = []
        for head in range(self.num_heads):
            h = x
            for index, width in enumerate(self.hidden_dims):
                h = nn.Dense(width, kernel_init=_OFFICIAL_DENSE_INIT, name=f"h{head}_dense_{index}")(h)
                h = activation(h, self.activation_name)
                h = nn.LayerNorm(name=f"h{head}_norm_{index}")(h)
            q = nn.Dense(
                self.action_sequence * self.action_dim * self.bins,
                kernel_init=_OFFICIAL_DENSE_INIT,
                name=f"h{head}_out",
            )(h)
            heads.append(q.reshape((-1, self.action_sequence, self.action_dim, self.bins)))
        return jnp.stack(heads, axis=-1)


class QChunkingC2F(QChunking):
    """Q-chunking whose critic is a coarse-to-fine factorized ensemble."""

    def __init__(
        self,
        *args,
        c2f_levels: int = 3,
        c2f_bins: int = 5,
        c2f_heads: int = 5,
        c2f_cql_weight: float = 0.0,
        c2f_lcb_kappa: float = 1.0,
        c2f_train_act_mode: str = "bon",
        c2f_eval_act_mode: str = "bon",
        **kwargs,
    ):
        for mode in (c2f_train_act_mode, c2f_eval_act_mode):
            if mode not in ACT_MODES:
                raise ValueError(f"Unknown C2F act mode {mode!r}; expected one of {ACT_MODES}.")
        if c2f_levels < 1 or c2f_bins < 2 or c2f_heads < 1:
            raise ValueError("C2F critic needs levels >= 1, bins >= 2, heads >= 1.")
        self.c2f_levels = int(c2f_levels)
        self.c2f_bins = int(c2f_bins)
        self.c2f_heads = int(c2f_heads)
        self.c2f_cql_weight = float(c2f_cql_weight)
        self.c2f_lcb_kappa = float(c2f_lcb_kappa)
        self.c2f_train_act_mode = c2f_train_act_mode
        self.c2f_eval_act_mode = c2f_eval_act_mode
        self._policy_fns: dict[str, Any] = {}
        super().__init__(*args, **kwargs)

        # Replace the joint critic built by QChunking (update/policy functions
        # are traced lazily, so they pick up the new critic on first call).
        feature_dim = (
            self.trunk_model.output_dim if self.feature_projection else self._rl_feature_dim
        )
        self.critic_model = C2FChunkCritic(
            hidden_dims=tuple(self.model_spec.hidden_dims),
            action_sequence=self.action_sequence,
            action_dim=self.action_dim,
            bins=self.c2f_bins,
            num_heads=self.c2f_heads,
            activation_name=self.model_spec.activation,
        )
        self.rng_key, critic_key = jax.random.split(self.rng_key)
        self.params["critic"] = self.critic_model.init(
            critic_key,
            jnp.zeros((1, feature_dim), jnp.float32),
            jnp.zeros((1, self.c2f_levels), jnp.float32),
            jnp.zeros((1, self.action_sequence * self.action_dim), jnp.float32),
        )
        self.target_critic_params = self._critic_bundle(self.params)
        self.opt_state["critic"] = self.critic_optimizer.init(self._critic_bundle(self.params))

    # ------------------------------------------------------------------
    # Coarse-to-fine geometry
    # ------------------------------------------------------------------

    def _c2f_encode(self, unit_actions):
        """[N, K*D] in [-1, 1] -> interval midpoints and chosen bins, both [N, L, K, D]."""
        a = unit_actions.reshape((-1, self.action_sequence, self.action_dim))
        low = -jnp.ones_like(a)
        high = jnp.ones_like(a)
        mids, bins = [], []
        for _ in range(self.c2f_levels):
            mids.append(0.5 * (low + high))
            width = (high - low) / self.c2f_bins
            b = jnp.clip(jnp.floor((a - low) / width), 0, self.c2f_bins - 1).astype(jnp.int32)
            bins.append(b)
            low = low + b.astype(a.dtype) * width
            high = low + width
        return jnp.stack(mids, axis=1), jnp.stack(bins, axis=1)

    def _level_one_hot(self, n, level):
        return jnp.broadcast_to(jax.nn.one_hot(level, self.c2f_levels), (n, self.c2f_levels))

    def _c2f_level_q(self, critic_params, features, level, midpoints):
        """Scores of all bins at one level: midpoints [N, K, D] -> [N, K, D, B, E]."""
        n = features.shape[0]
        return self.critic_model.apply(
            critic_params,
            features,
            self._level_one_hot(n, level),
            midpoints.reshape((n, -1)),
        )

    def _c2f_all_levels(self, critic_params, features, unit_actions):
        """Q of every bin along an action's C2F path: [N, L, K, D, B, E] and bins [N, L, K, D]."""
        mids, bins = self._c2f_encode(unit_actions)
        n = features.shape[0]
        L = self.c2f_levels
        q = self.critic_model.apply(
            critic_params,
            jnp.repeat(features, L, axis=0),
            jnp.tile(jnp.eye(L, dtype=jnp.float32), (n, 1)),
            mids.reshape((n * L, -1)),
        )
        return q.reshape((n, L) + q.shape[1:]), bins

    def _c2f_chunk_score(self, critic_params, features, unit_actions):
        """Chunk score per head: deepest-level mean over (k, d) -> [N, E]."""
        mids, bins = self._c2f_encode(unit_actions)
        last = self.c2f_levels - 1
        q = self._c2f_level_q(critic_params, features, last, mids[:, last])
        taken = jnp.take_along_axis(q, bins[:, last][..., None, None], axis=3)[..., 0, :]
        return jnp.mean(taken, axis=(1, 2))

    # ------------------------------------------------------------------
    # Action extraction
    # ------------------------------------------------------------------

    def _best_of_n(self, actor_bundle, critic_bundle, actor_features, critic_features, key):
        candidates = self._flow_candidates(
            actor_bundle["actor"], actor_features, key, count=self.actor_num_samples
        )
        n, m, flat = candidates.shape
        head_scores = self._c2f_chunk_score(
            critic_bundle["critic"],
            jnp.repeat(critic_features, m, axis=0),
            candidates.reshape((n * m, flat)),
        ).reshape((n, m, self.c2f_heads))
        scores = self._aggregate_q(head_scores)
        best = jnp.argmax(scores, axis=1)
        selected = jnp.take_along_axis(candidates, best[:, None, None], axis=1)[:, 0]
        return selected, jnp.take_along_axis(scores, best[:, None], axis=1)[:, 0]

    def _c2f_greedy(self, critic_params, features, kappa, candidate_bins=None):
        """Factorized C2F greedy; optional LCB (kappa) and flow-support mask."""
        n = features.shape[0]
        shape = (n, self.action_sequence, self.action_dim)
        low = -jnp.ones(shape, jnp.float32)
        high = jnp.ones(shape, jnp.float32)
        match = None
        if candidate_bins is not None:  # [N, M, L, K, D]
            match = jnp.ones(candidate_bins.shape[:2] + shape[1:], jnp.bool_)
        last_score = None
        for level in range(self.c2f_levels):
            q = self._c2f_level_q(critic_params, features, level, 0.5 * (low + high))
            score = jnp.mean(q, axis=-1)
            if kappa:
                score = score - kappa * jnp.std(q, axis=-1)
            if match is not None:
                cand = candidate_bins[:, :, level]  # [N, M, K, D]
                hit = jax.nn.one_hot(cand, self.c2f_bins, dtype=jnp.bool_) & match[..., None]
                supported = jnp.any(hit, axis=1)  # [N, K, D, B]
                score = jnp.where(supported, score, -jnp.inf)
            b = jnp.argmax(score, axis=-1).astype(jnp.int32)
            last_score = jnp.take_along_axis(score, b[..., None], axis=-1)[..., 0]
            if match is not None:
                match = match & (candidate_bins[:, :, level] == b[:, None])
            width = (high - low) / self.c2f_bins
            low = low + b.astype(jnp.float32) * width
            high = low + width
        action = (0.5 * (low + high)).reshape((n, -1))
        return action, jnp.mean(last_score, axis=(1, 2))

    def _build_policy_fn(self, mode: str = "bon"):
        def policy_fn(params, obs_inputs, key):
            actor_bundle = self._actor_bundle(params)
            critic_bundle = self._critic_bundle(params)
            actor_features = self._actor_features(actor_bundle, obs_inputs, stop_gradient=True)
            critic_features = self._critic_features(critic_bundle, obs_inputs, stop_gradient=True)
            if mode == "bon":
                return self._best_of_n(actor_bundle, critic_bundle, actor_features, critic_features, key)
            kappa = self.c2f_lcb_kappa if mode.endswith("_lcb") else 0.0
            candidate_bins = None
            if mode.startswith("support"):
                candidates = self._flow_candidates(
                    actor_bundle["actor"], actor_features, key, count=self.actor_num_samples
                )
                n, m, flat = candidates.shape
                _, bins = self._c2f_encode(candidates.reshape((n * m, flat)))
                candidate_bins = bins.reshape((n, m) + bins.shape[1:])
            return self._c2f_greedy(critic_bundle["critic"], critic_features, kappa, candidate_bins)

        return policy_fn

    def _get_policy_fn(self, mode: str):
        if mode not in self._policy_fns:
            fn = self._build_policy_fn(mode)
            self._policy_fns[mode] = jax.jit(fn) if self._jit_enabled else fn
        return self._policy_fns[mode]

    def act(self, observations: dict, step: int, eval_mode: bool):
        mode = self.c2f_eval_act_mode if eval_mode else self.c2f_train_act_mode
        self._policy_action_impl = self._get_policy_fn(mode)
        return super().act(observations, step, eval_mode)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def _build_update_fn(self):
        actor_optimizer = self.actor_optimizer
        critic_optimizer = self.critic_optimizer
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
            action_valid,
            loss_weights,
            bc_weights,
            key,
        ):
            if self.rgb_shift_pad > 0 and isinstance(obs_inputs, dict) and "rgb" in obs_inputs:
                key, obs_shift_key, next_shift_key = jax.random.split(key, 3)
                obs_inputs = dict(obs_inputs)
                next_obs_inputs = dict(next_obs_inputs)
                obs_inputs["rgb"] = random_shift_rgb(obs_inputs["rgb"], obs_shift_key, self.rgb_shift_pad)
                next_obs_inputs["rgb"] = random_shift_rgb(
                    next_obs_inputs["rgb"], next_shift_key, self.rgb_shift_pad
                )
            if self.low_dim_mask_prob > 0.0 and isinstance(obs_inputs, dict) and "low_dim" in obs_inputs:
                key, mask_key, next_mask_key = jax.random.split(key, 3)
                obs_inputs = dict(obs_inputs)
                next_obs_inputs = dict(next_obs_inputs)
                obs_inputs["low_dim"] = self._mask_low_dim(obs_inputs["low_dim"], mask_key)
                next_obs_inputs["low_dim"] = self._mask_low_dim(next_obs_inputs["low_dim"], next_mask_key)
            key, next_action_key, source_key, time_key = jax.random.split(key, 4)
            full_chunk_valid = jnp.all(action_valid, axis=1).astype(jnp.float32)
            critic_weights = loss_weights * full_chunk_valid

            actor_bundle = self._actor_bundle(params)
            online_critic_bundle = self._critic_bundle(params)
            next_actor_features = self._actor_features(actor_bundle, next_obs_inputs, stop_gradient=True)
            next_online_critic_features = self._critic_features(
                online_critic_bundle, next_obs_inputs, stop_gradient=True
            )
            # Bootstrap only at an on-manifold action: Best-of-N over flow samples.
            next_actions, _ = self._best_of_n(
                actor_bundle,
                online_critic_bundle,
                next_actor_features,
                next_online_critic_features,
                next_action_key,
            )
            next_target_features = self._critic_features(
                target_critic_params, next_obs_inputs, stop_gradient=True
            )
            next_q_all, next_bins = self._c2f_all_levels(
                target_critic_params["critic"], next_target_features, next_actions
            )
            next_taken = jnp.take_along_axis(next_q_all, next_bins[..., None, None], axis=4)[..., 0, :]
            next_q = self._aggregate_q(next_taken)  # [N, L, K, D]: own head bootstrap per (l, k, d)
            target_q = jax.lax.stop_gradient(
                q_chunking_td_target(
                    rewards[:, None, None, None],
                    discounts[:, None, None, None],
                    bootstrap[:, None, None, None],
                    next_q,
                )
            )

            def critic_loss_fn(critic_bundle):
                features = self._critic_features(critic_bundle, obs_inputs)
                q_all, bins = self._c2f_all_levels(critic_bundle["critic"], features, actions)
                taken = jnp.take_along_axis(q_all, bins[..., None, None], axis=4)[..., 0, :]
                td = jnp.mean(jnp.square(taken - target_q[..., None]), axis=(1, 2, 3, 4))
                per_sample = td
                cql = jnp.zeros_like(td)
                if self.c2f_cql_weight > 0.0:
                    cql = jnp.mean(
                        jax.scipy.special.logsumexp(q_all, axis=4) - taken, axis=(1, 2, 3, 4)
                    )
                    per_sample = td + self.c2f_cql_weight * cql
                denominator = jnp.maximum(jnp.sum(critic_weights), 1.0)
                loss = jnp.sum(per_sample * critic_weights) / denominator
                gap = jnp.mean(jnp.max(jnp.mean(q_all, -1), axis=4) - jnp.mean(taken, -1))
                head_std = jnp.mean(jnp.std(taken, axis=-1))
                return loss, (td, taken, cql, gap, head_std)

            (critic_loss, (per_sample, taken, cql, gap, head_std)), critic_grads = jax.value_and_grad(
                critic_loss_fn, has_aux=True
            )(online_critic_bundle)
            critic_updates, critic_opt_state = critic_optimizer.update(
                critic_grads, opt_state["critic"], online_critic_bundle
            )
            updated_critic_bundle = self.optax.apply_updates(online_critic_bundle, critic_updates)

            unit_actions = actions.reshape((actions.shape[0], self.action_sequence, self.action_dim))

            def actor_loss_fn(current_actor_bundle):
                features = self._actor_features(current_actor_bundle, obs_inputs)
                source = jax.random.normal(source_key, actions.shape, dtype=jnp.float32)
                times = jax.random.uniform(time_key, (actions.shape[0], 1), dtype=jnp.float32)
                interpolated = (1.0 - times) * source + times * actions
                target_velocity = actions - source
                predicted_velocity = self.actor_model.apply(
                    current_actor_bundle["actor"], features, interpolated, times
                )
                squared_error = jnp.square(predicted_velocity - target_velocity).reshape(unit_actions.shape)
                token_error = jnp.mean(squared_error, axis=-1)
                valid = action_valid.astype(jnp.float32)
                per_sample_actor = jnp.sum(token_error * valid, axis=1) / jnp.maximum(
                    jnp.sum(valid, axis=1), 1.0
                )
                actor_denominator = jnp.maximum(jnp.sum(bc_weights), 1.0)
                loss = jnp.sum(per_sample_actor * bc_weights) / actor_denominator
                return loss, per_sample_actor

            current_actor_bundle = self._actor_bundle(params)
            (actor_loss, per_sample_actor), actor_grads = jax.value_and_grad(
                actor_loss_fn, has_aux=True
            )(current_actor_bundle)
            actor_updates, actor_opt_state = actor_optimizer.update(
                actor_grads, opt_state["actor"], current_actor_bundle
            )
            updated_actor_bundle = self.optax.apply_updates(current_actor_bundle, actor_updates)

            params = dict(params)
            params.update(updated_actor_bundle)
            params.update(updated_critic_bundle)
            target_critic_params = jax.tree.map(
                lambda target, online: (1.0 - tau) * target + tau * online,
                target_critic_params,
                updated_critic_bundle,
            )
            priority = jnp.sqrt(per_sample + 1e-10)
            priority = priority / jnp.maximum(jnp.max(priority), 1e-10)
            metrics = {
                "actor_loss": actor_loss,
                "bc_flow_loss": actor_loss,
                "critic_loss": critic_loss,
                "critic_q": jnp.mean(taken),
                "critic_target_q": jnp.mean(target_q),
                "c2f_action_gap": gap,
                "c2f_head_std": head_std,
                "c2f_cql": jnp.mean(cql),
                "chunk_valid_fraction": jnp.mean(full_chunk_valid),
                "action_valid_fraction": jnp.mean(action_valid),
                "actor_per_sample_loss": jnp.mean(per_sample_actor),
                "bc_weight_fraction": jnp.mean((bc_weights > 0).astype(jnp.float32)),
            }
            return (
                params,
                target_critic_params,
                {"actor": actor_opt_state, "critic": critic_opt_state},
                key,
                priority,
                metrics,
            )

        return update_fn


__all__ = ["ACT_MODES", "C2FChunkCritic", "QChunkingC2F"]
