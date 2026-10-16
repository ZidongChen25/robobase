"""RLPD -- Reinforcement Learning with Prior Data, implemented in JAX.

Ball, Smith, Kostrikov & Levine, "Efficient Online Reinforcement Learning with
Offline Data", ICML 2023.  This is a port of the official ``SACLearner`` with
``configs/rlpd_config.py`` (github.com/ikostrikov/rlpd):

* symmetric sampling: every gradient step uses ``online_batch_size`` online and
  ``offline_batch_size`` offline transitions (128 + 128), drawn uniformly with
  replacement;
* ``utd_ratio`` critic steps per environment step, each followed by a Polyak
  target update, then ONE actor step and ONE temperature step on the last
  critic minibatch (``SACLearner.update``);
* an ensemble of ``num_qs`` critics, ``Dense -> LayerNorm -> ReLU`` per hidden
  layer, Xavier-uniform kernels.  The TD target is the min over ``num_min_qs``
  members of the target ensemble drawn without replacement (2 = clipped double
  Q, 1 = a single random member).  The actor maximises the ensemble mean;
* tanh-Gaussian actor with a state-dependent log-std clipped to [-20, 2] and
  no LayerNorm; automatic temperature (``loss = T * (H - H_target)``, initial
  T = 1, ``H_target = -|A| / 2``) and an optional entropy backup.

Throughput design.  RoboBase's episode replay samples on the host, which would
cost ``utd_ratio`` host batches per environment step.  Here the offline dataset
and the online replay instead live on the accelerator as packed float32
matrices (one row per transition, see :func:`pack_transitions`), minibatch
indices are drawn inside the compiled step, and the transition insert, all
``utd_ratio`` critic steps and the actor/temperature step form one XLA program
whose train state and replay are donated (updated in place).  One environment
step therefore costs one dispatch plus one device-to-host copy (the action);
:class:`robobase.workspace_rlpd.RLPDWorkspace` overlaps the simulator step with
that program.  :meth:`RLPD.update` keeps RoboBase's ``replay_iter`` contract for
generic workspaces, sharing the same compiled update core.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import partial
from typing import Iterator

import jax
import jax.numpy as jnp
import numpy as np
import optax
from gymnasium import spaces
from omegaconf import DictConfig

from robobase.method.core import OffPolicyMethod
from robobase.method.jax_base import JaxMethodBase
from robobase.replay_buffer.replay_buffer import ReplayBuffer

LOG_TWO_PI = float(np.log(2.0 * np.pi))
LOG_TWO = float(np.log(2.0))
LAYER_NORM_EPS = 1e-6  # flax.linen.LayerNorm default
METRIC_NAMES = (
    "critic_loss",
    "q_mean",
    "target_q_mean",
    "actor_loss",
    "entropy",
    "temperature",
    "temperature_loss",
    "online_size",
)


@dataclass(frozen=True)
class RLPDSpec:
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temp_lr: float = 3e-4
    discount: float = 0.99
    tau: float = 0.005
    hidden_dims: tuple[int, ...] = (256, 256)
    num_qs: int = 10
    num_min_qs: int = 2
    critic_layer_norm: bool = True
    init_temperature: float = 1.0
    target_entropy: float | None = None
    backup_entropy: bool = True
    utd_ratio: int = 20
    online_batch_size: int = 128
    offline_batch_size: int = 128
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    offline_action_clip_eps: float = 1e-5
    num_explore_steps: int = 5000
    online_buffer_size: int = 1_000_000
    scan_unroll: int = 1


def rlpd_spec_from_cfg(cfg: DictConfig) -> RLPDSpec:
    method = cfg.method
    target_entropy = method.get("target_entropy", None)
    return RLPDSpec(
        actor_lr=float(method.get("actor_lr", 3e-4)),
        critic_lr=float(method.get("critic_lr", 3e-4)),
        temp_lr=float(method.get("temp_lr", 3e-4)),
        discount=float(method.get("discount", 0.99)),
        tau=float(method.get("tau", 0.005)),
        hidden_dims=tuple(int(v) for v in method.get("hidden_dims", (256, 256))),
        num_qs=int(method.get("num_qs", 10)),
        num_min_qs=int(method.get("num_min_qs", 2)),
        critic_layer_norm=bool(method.get("critic_layer_norm", True)),
        init_temperature=float(method.get("init_temperature", 1.0)),
        target_entropy=None if target_entropy is None else float(target_entropy),
        backup_entropy=bool(method.get("backup_entropy", True)),
        utd_ratio=int(method.get("utd_ratio", 20)),
        online_batch_size=int(method.get("online_batch_size", 128)),
        offline_batch_size=int(method.get("offline_batch_size", 128)),
        reward_scale=float(method.get("reward_scale", 1.0)),
        reward_bias=float(method.get("reward_bias", 0.0)),
        log_std_min=float(method.get("log_std_min", -20.0)),
        log_std_max=float(method.get("log_std_max", 2.0)),
        offline_action_clip_eps=float(method.get("offline_action_clip_eps", 1e-5)),
        num_explore_steps=int(
            method.get("num_explore_steps", cfg.get("num_explore_steps", 5000))
        ),
        online_buffer_size=int(
            method.get("online_buffer_size", None) or cfg.num_train_frames
        ),
        scan_unroll=int(method.get("scan_unroll", 1)),
    )


# ----------------------------------------------------------------------------
# Networks (pure functions over explicit parameter pytrees)
# ----------------------------------------------------------------------------


def _xavier_uniform(key, fan_in: int, fan_out: int, shape) -> jax.Array:
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, jnp.float32, -limit, limit)


def init_actor_params(key, obs_dim: int, action_dim: int, hidden_dims) -> dict:
    dims = (obs_dim,) + tuple(hidden_dims)
    keys = jax.random.split(key, len(hidden_dims) + 2)
    hidden = [
        {
            "w": _xavier_uniform(k, d_in, d_out, (d_in, d_out)),
            "b": jnp.zeros((d_out,), jnp.float32),
        }
        for k, d_in, d_out in zip(keys, dims[:-1], dims[1:])
    ]
    width = dims[-1]
    return {
        "hidden": hidden,
        "mean": {
            "w": _xavier_uniform(keys[-2], width, action_dim, (width, action_dim)),
            "b": jnp.zeros((action_dim,), jnp.float32),
        },
        "log_std": {
            "w": _xavier_uniform(keys[-1], width, action_dim, (width, action_dim)),
            "b": jnp.zeros((action_dim,), jnp.float32),
        },
    }


def init_critic_params(
    key, input_dim: int, hidden_dims, num_qs: int, layer_norm: bool
) -> dict:
    """Ensemble parameters with a leading ``num_qs`` axis on every leaf."""

    dims = (input_dim,) + tuple(hidden_dims)
    keys = jax.random.split(key, len(hidden_dims) + 1)
    hidden = []
    for k, d_in, d_out in zip(keys, dims[:-1], dims[1:]):
        layer = {
            "w": _xavier_uniform(k, d_in, d_out, (num_qs, d_in, d_out)),
            "b": jnp.zeros((num_qs, d_out), jnp.float32),
        }
        if layer_norm:
            layer["ln_scale"] = jnp.ones((num_qs, d_out), jnp.float32)
            layer["ln_bias"] = jnp.zeros((num_qs, d_out), jnp.float32)
        hidden.append(layer)
    width = dims[-1]
    return {
        "hidden": hidden,
        "out": {
            "w": _xavier_uniform(keys[-1], width, 1, (num_qs, width, 1)),
            "b": jnp.zeros((num_qs, 1), jnp.float32),
        },
    }


def actor_apply(params, obs, log_std_min: float, log_std_max: float):
    x = obs
    for layer in params["hidden"]:
        x = jax.nn.relu(x @ layer["w"] + layer["b"])
    mean = x @ params["mean"]["w"] + params["mean"]["b"]
    log_std = x @ params["log_std"]["w"] + params["log_std"]["b"]
    return mean, jnp.clip(log_std, log_std_min, log_std_max)


def _layer_norm(x, scale, bias):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + LAYER_NORM_EPS) * scale + bias


def critic_apply(params, obs, actions) -> jax.Array:
    """Q-values of every ensemble member, shape ``[members, batch]``."""

    x = jnp.concatenate([obs, actions], axis=-1)
    for index, layer in enumerate(params["hidden"]):
        if index == 0:
            # Shared input: one [B, in] x [E, in, out] contraction.
            x = jnp.einsum("bi,eio->ebo", x, layer["w"])
        else:
            x = jnp.einsum("ebi,eio->ebo", x, layer["w"])
        x = x + layer["b"][:, None, :]
        if "ln_scale" in layer:
            x = _layer_norm(
                x, layer["ln_scale"][:, None, :], layer["ln_bias"][:, None, :]
            )
        x = jax.nn.relu(x)
    out = params["out"]
    return jnp.einsum("ebi,eio->ebo", x, out["w"])[..., 0] + out["b"]


def tanh_normal_from_noise(noise, mean, log_std):
    """``tanh(mean + exp(log_std) * noise)`` and its log-density.

    The log-density is evaluated at the pre-tanh sample with TFP's stable Tanh
    log-det ``2 (log 2 - x - softplus(-2x))``.  This equals the reference
    ``TanhTransformedDistribution.log_prob`` of a cached sample, without the
    ``atanh`` round trip that overflows for saturated actions.
    """

    pre_tanh = mean + jnp.exp(log_std) * noise
    base = -0.5 * jnp.square(noise) - log_std - 0.5 * LOG_TWO_PI
    log_det = 2.0 * (LOG_TWO - pre_tanh - jax.nn.softplus(-2.0 * pre_tanh))
    return jnp.tanh(pre_tanh), jnp.sum(base - log_det, axis=-1)


def tanh_normal_sample(key, mean, log_std):
    noise = jax.random.normal(key, mean.shape, mean.dtype)
    return tanh_normal_from_noise(noise, mean, log_std)


def numpy_actor_mode(params, obs: np.ndarray) -> np.ndarray:
    """Deterministic ``tanh(mean)`` policy in NumPy, for out-of-process eval."""

    x = np.asarray(obs, dtype=np.float32)
    for layer in params["hidden"]:
        x = np.maximum(x @ layer["w"] + layer["b"], 0.0)
    return np.tanh(x @ params["mean"]["w"] + params["mean"]["b"])


# ----------------------------------------------------------------------------
# Packed transition layout: [obs | action | reward | mask | next_obs]
# ----------------------------------------------------------------------------


def transition_width(obs_dim: int, action_dim: int) -> int:
    return 2 * obs_dim + action_dim + 2


def pack_transitions(obs, actions, rewards, masks, next_obs) -> np.ndarray:
    obs = np.asarray(obs, np.float32)
    count = obs.shape[0]
    return np.concatenate(
        [
            obs.reshape(count, -1),
            np.asarray(actions, np.float32).reshape(count, -1),
            np.asarray(rewards, np.float32).reshape(count, 1),
            np.asarray(masks, np.float32).reshape(count, 1),
            np.asarray(next_obs, np.float32).reshape(count, -1),
        ],
        axis=1,
    )


def _unpack(rows, obs_dim: int, action_dim: int):
    a0 = obs_dim
    r0 = a0 + action_dim
    return (
        rows[..., :a0],
        rows[..., a0:r0],
        rows[..., r0],
        rows[..., r0 + 1],
        rows[..., r0 + 2 :],
    )


# ----------------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------------


class RLPD(JaxMethodBase, OffPolicyMethod):
    """RLPD for state-based observations (``low_dim_state``)."""

    def __init__(
        self,
        spec: RLPDSpec,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        num_train_envs: int,
        num_eval_envs: int,
        replay_alpha: float,
        replay_beta: float,
        frame_stack_on_channel: bool,
        intrinsic_reward_module=None,
        jit: bool = True,
        platform: str | None = None,
        seed: int = 0,
        update_block_every_steps: int = 1,
    ):
        super().__init__(
            lr=spec.actor_lr,
            adaptive_lr=False,
            num_train_steps=spec.online_buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            num_train_envs=num_train_envs,
            num_eval_envs=num_eval_envs,
            replay_alpha=replay_alpha,
            replay_beta=replay_beta,
            frame_stack_on_channel=frame_stack_on_channel,
            intrinsic_reward_module=intrinsic_reward_module,
            jit=jit,
            platform=platform,
            seed=seed,
            is_rl=True,
            update_block_every_steps=update_block_every_steps,
        )
        if self.use_pixels or set(observation_space.keys()) != {"low_dim_state"}:
            raise ValueError("RLPD here is state-based: only low_dim_state.")
        if self.action_sequence != 1:
            raise ValueError("RLPD requires action_sequence=1.")
        if not 1 <= spec.num_min_qs <= spec.num_qs:
            raise ValueError("num_min_qs must be in [1, num_qs].")
        if spec.utd_ratio < 1:
            raise ValueError("utd_ratio must be >= 1.")
        if spec.online_batch_size + spec.offline_batch_size <= 0:
            raise ValueError("RLPD needs a positive batch size.")
        if spec.scan_unroll < 1:
            raise ValueError("scan_unroll must be >= 1.")
        self.spec = spec
        self.num_explore_steps = int(spec.num_explore_steps)
        self.obs_dim = int(np.prod(observation_space["low_dim_state"].shape))
        self.width = transition_width(self.obs_dim, self.action_dim)
        self.target_entropy = (
            -0.5 * self.action_dim
            if spec.target_entropy is None
            else float(spec.target_entropy)
        )
        self.action_low, self.action_high = (
            jnp.asarray(np.asarray(b, np.float32).reshape(-1))
            for b in (action_space.low, action_space.high)
        )

        self.actor_tx = optax.adam(spec.actor_lr)
        self.critic_tx = optax.adam(spec.critic_lr)
        self.temp_tx = optax.adam(spec.temp_lr)
        actor_key, critic_key, rng, act_key = jax.random.split(self.rng_key, 4)
        actor = init_actor_params(
            actor_key, self.obs_dim, self.action_dim, spec.hidden_dims
        )
        critic = init_critic_params(
            critic_key,
            self.obs_dim + self.action_dim,
            spec.hidden_dims,
            spec.num_qs,
            spec.critic_layer_norm,
        )
        log_temp = jnp.asarray(np.log(spec.init_temperature), jnp.float32)
        self.state = {
            "actor": actor,
            "critic": critic,
            "target_critic": jax.tree.map(jnp.copy, critic),
            "log_temp": log_temp,
            "actor_opt": self.actor_tx.init(actor),
            "critic_opt": self.critic_tx.init(critic),
            "temp_opt": self.temp_tx.init(log_temp),
            "rng": rng,
        }
        self._act_key = act_key
        self.replay = None
        self.offline = None
        self._online_count = 0
        self._last_metrics = None
        self._np_rng = np.random.default_rng(int(seed))

        maybe_jit = jax.jit if self._jit_enabled else (lambda f, **_: f)
        self._sample_fn = maybe_jit(self._sample_action)
        self._mode_fn = maybe_jit(self._mode_action)
        self._insert_fn = maybe_jit(self._insert, donate_argnums=(0,))
        self._insert_update_fn = maybe_jit(
            self._insert_and_update, donate_argnums=(0, 1)
        )
        self._offline_update_fn = maybe_jit(
            self._offline_only_update, donate_argnums=(0,)
        )
        self._batch_update_fn = maybe_jit(self._update_core, donate_argnums=(0,))

    # ------------------------------------------------------------------
    # RoboBase parameter views
    # ------------------------------------------------------------------

    @property
    def params(self):
        return {
            key: self.state[key]
            for key in ("actor", "critic", "target_critic", "log_temp")
        }

    @params.setter
    def params(self, value):
        for key in ("actor", "critic", "target_critic", "log_temp"):
            if key in value:
                self.state[key] = value[key]

    @property
    def opt_state(self):
        return {k: self.state[k] for k in ("actor_opt", "critic_opt", "temp_opt")}

    @opt_state.setter
    def opt_state(self, value):
        self.state.update(value)

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def _flat_obs(self, observations) -> np.ndarray:
        obs = np.asarray(observations["low_dim_state"], dtype=np.float32)
        return obs.reshape(obs.shape[0], -1)

    def _scale(self, unit_action):
        return self.action_low + 0.5 * (unit_action + 1.0) * (
            self.action_high - self.action_low
        )

    def _sample_action(self, actor, obs, key):
        key, sample_key = jax.random.split(key)
        mean, log_std = actor_apply(
            actor, obs, self.spec.log_std_min, self.spec.log_std_max
        )
        action, _ = tanh_normal_sample(sample_key, mean, log_std)
        return self._scale(action), key

    def _mode_action(self, actor, obs):
        mean, _ = actor_apply(actor, obs, self.spec.log_std_min, self.spec.log_std_max)
        return self._scale(jnp.tanh(mean))

    def act(self, observations: dict, step: int, eval_mode: bool):
        num = int(np.asarray(observations["low_dim_state"]).shape[0])
        if not eval_mode and step < self.num_explore_steps:
            return self._np_rng.uniform(
                self.action_space.low,
                self.action_space.high,
                size=(num,) + self.action_space.shape,
            ).astype(np.float32)
        obs = self._flat_obs(observations)
        if eval_mode:
            action = self._mode_fn(self.state["actor"], obs)
        else:
            action, self._act_key = self._sample_fn(
                self.state["actor"], obs, self._act_key
            )
        return np.asarray(action, dtype=np.float32).reshape(num, 1, self.action_dim)

    # ------------------------------------------------------------------
    # Update core: utd critic steps, then one actor + temperature step
    # ------------------------------------------------------------------

    def _critic_step(self, temperature, carry, inputs):
        spec = self.spec
        critic, target, critic_opt = carry
        (
            obs,
            actions,
            rewards,
            masks,
            next_obs,
            next_actions,
            next_log_probs,
            member_mask,
        ) = inputs
        # All members are evaluated (one batched GEMM per layer, static
        # shapes); the min is taken over the sampled subset only.
        next_q = critic_apply(target, next_obs, next_actions)
        next_q = jnp.min(jnp.where(member_mask[:, None], next_q, jnp.inf), axis=0)
        target_q = rewards + spec.discount * masks * next_q
        if spec.backup_entropy:
            target_q = target_q - spec.discount * masks * temperature * next_log_probs
        target_q = jax.lax.stop_gradient(target_q)

        def loss_fn(params):
            q = critic_apply(params, obs, actions)
            return jnp.mean(jnp.square(q - target_q[None])), jnp.mean(q)

        (loss, q_mean), grads = jax.value_and_grad(loss_fn, has_aux=True)(critic)
        updates, critic_opt = self.critic_tx.update(grads, critic_opt, critic)
        critic = optax.apply_updates(critic, updates)
        target = optax.incremental_update(critic, target, spec.tau)
        return (critic, target, critic_opt), (loss, q_mean, target_q.mean())

    def _critic_inputs(self, actor, batches, key):
        """Per-step critic inputs with every random draw made up front.

        The actor is fixed during the ``utd_ratio`` critic steps, so the
        next-state actions of all minibatches come from one batched forward,
        and the target subsets are i.i.d. uniform without replacement per
        step -- the same distribution as sampling inside each step.
        """

        spec = self.spec
        utd, batch = batches.shape[:2]
        noise_key, member_key = jax.random.split(key)
        obs, actions, rewards, masks, next_obs = _unpack(
            batches, self.obs_dim, self.action_dim
        )
        rewards = rewards * spec.reward_scale + spec.reward_bias
        mean, log_std = actor_apply(
            actor,
            next_obs.reshape(utd * batch, self.obs_dim),
            spec.log_std_min,
            spec.log_std_max,
        )
        noise = jax.random.normal(noise_key, mean.shape, mean.dtype)
        next_actions, next_log_probs = tanh_normal_from_noise(noise, mean, log_std)
        if spec.num_min_qs < spec.num_qs:
            ranks = jnp.argsort(
                jnp.argsort(jax.random.uniform(member_key, (utd, spec.num_qs)), -1), -1
            )
            member_mask = ranks < spec.num_min_qs
        else:
            member_mask = jnp.ones((utd, spec.num_qs), bool)
        return (
            obs,
            actions,
            rewards,
            masks,
            next_obs,
            next_actions.reshape(utd, batch, self.action_dim),
            next_log_probs.reshape(utd, batch),
            member_mask,
        )

    def _update_core(self, state, batches):
        """``batches``: packed rows ``[utd_ratio, batch, width]``."""

        spec = self.spec
        actor = state["actor"]
        temperature = jnp.exp(state["log_temp"])
        rng, critic_key, actor_key = jax.random.split(state["rng"], 3)
        carry = (state["critic"], state["target_critic"], state["critic_opt"])
        carry, (losses, q_means, target_means) = jax.lax.scan(
            partial(self._critic_step, temperature),
            carry,
            self._critic_inputs(actor, batches, critic_key),
            unroll=spec.scan_unroll,
        )
        critic_info = (losses[-1], q_means[-1], target_means[-1])
        critic, target, critic_opt = carry

        obs = _unpack(batches[-1], self.obs_dim, self.action_dim)[0]

        def actor_loss_fn(params):
            mean, log_std = actor_apply(params, obs, spec.log_std_min, spec.log_std_max)
            actions, log_probs = tanh_normal_sample(actor_key, mean, log_std)
            q = critic_apply(critic, obs, actions).mean(axis=0)
            return jnp.mean(log_probs * temperature - q), -jnp.mean(log_probs)

        (actor_loss, entropy), actor_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(actor)
        actor_updates, actor_opt = self.actor_tx.update(
            actor_grads, state["actor_opt"], actor
        )
        actor = optax.apply_updates(actor, actor_updates)

        def temp_loss_fn(log_temp):
            return jnp.exp(log_temp) * (entropy - self.target_entropy)

        temp_loss, temp_grad = jax.value_and_grad(temp_loss_fn)(state["log_temp"])
        temp_updates, temp_opt = self.temp_tx.update(
            temp_grad, state["temp_opt"], state["log_temp"]
        )
        log_temp = optax.apply_updates(state["log_temp"], temp_updates)
        new_state = {
            "actor": actor,
            "critic": critic,
            "target_critic": target,
            "log_temp": log_temp,
            "actor_opt": actor_opt,
            "critic_opt": critic_opt,
            "temp_opt": temp_opt,
            "rng": rng,
        }
        metrics = jnp.stack(
            [
                critic_info[0],
                critic_info[1],
                critic_info[2],
                actor_loss,
                entropy,
                jnp.exp(log_temp),
                temp_loss,
                jnp.zeros((), jnp.float32),
            ]
        )
        return new_state, metrics

    # ------------------------------------------------------------------
    # Device replay path (RLPDWorkspace)
    # ------------------------------------------------------------------

    def _sample_rows(self, key, replay, offline):
        spec = self.spec
        utd = spec.utd_ratio
        offline_key, online_key = jax.random.split(key)
        parts = []
        if spec.offline_batch_size > 0:
            index = jax.random.randint(
                offline_key, (utd, spec.offline_batch_size), 0, offline.shape[0]
            )
            parts.append(offline[index])
        if spec.online_batch_size > 0:
            index = jax.random.randint(
                online_key,
                (utd, spec.online_batch_size),
                0,
                jnp.maximum(replay["size"], 1),
            )
            parts.append(replay["data"][index])
        return parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=1)

    def _insert(self, replay, row):
        capacity = replay["data"].shape[0]
        data = jax.lax.dynamic_update_slice(
            replay["data"], row[None, :], (replay["ptr"], jnp.int32(0))
        )
        return {
            "data": data,
            "ptr": (replay["ptr"] + 1) % capacity,
            "size": jnp.minimum(replay["size"] + 1, capacity),
        }

    def _insert_and_update(self, state, replay, row, offline):
        replay = self._insert(replay, row)
        rng, sample_key = jax.random.split(state["rng"])
        state = dict(state, rng=rng)
        batches = self._sample_rows(sample_key, replay, offline)
        state, metrics = self._update_core(state, batches)
        metrics = metrics.at[-1].set(replay["size"].astype(jnp.float32))
        return state, replay, metrics

    def _offline_only_update(self, state, offline):
        spec = self.spec
        rng, sample_key = jax.random.split(state["rng"])
        state = dict(state, rng=rng)
        batch = spec.online_batch_size + spec.offline_batch_size
        index = jax.random.randint(
            sample_key, (spec.utd_ratio, batch), 0, offline.shape[0]
        )
        return self._update_core(state, offline[index])

    def attach_device_replay(self, offline_rows: np.ndarray | None, capacity=None):
        """Move the offline dataset to the device and allocate online replay."""

        spec = self.spec
        if spec.offline_batch_size > 0:
            if offline_rows is None or len(offline_rows) == 0:
                raise ValueError("offline_batch_size > 0 requires offline data.")
            offline_rows = np.array(offline_rows, dtype=np.float32, copy=True)
            if offline_rows.shape[1] != self.width:
                raise ValueError(
                    f"offline rows have width {offline_rows.shape[1]}, "
                    f"expected {self.width}."
                )
            lim = 1.0 - spec.offline_action_clip_eps
            a0 = self.obs_dim
            unit = (
                2.0
                * (
                    offline_rows[:, a0 : a0 + self.action_dim]
                    - np.asarray(self.action_low)
                )
                / np.asarray(self.action_high - self.action_low)
                - 1.0
            )
            unit = np.clip(unit, -lim, lim)
            offline_rows[:, a0 : a0 + self.action_dim] = np.asarray(
                self.action_low
            ) + 0.5 * (unit + 1.0) * np.asarray(self.action_high - self.action_low)
            self.offline = jax.device_put(offline_rows)
        else:
            self.offline = jnp.zeros((1, self.width), jnp.float32)
        capacity = int(capacity or spec.online_buffer_size)
        self.replay = {
            "data": jnp.zeros((capacity, self.width), jnp.float32),
            "ptr": jnp.zeros((), jnp.int32),
            "size": jnp.zeros((), jnp.int32),
        }
        self._online_count = 0

    def observe(self, row: np.ndarray, update: bool) -> None:
        """Insert one packed online transition and optionally run one RLPD
        update (``utd_ratio`` critic steps + actor + temperature).

        Asynchronous: returns once the program is dispatched.
        """

        if update:
            self.state, self.replay, self._last_metrics = self._insert_update_fn(
                self.state, self.replay, row, self.offline
            )
            self._update_step_count += 1
        else:
            self.replay = self._insert_fn(self.replay, row)
        self._online_count += 1

    def pretrain_step(self) -> None:
        self.state, self._last_metrics = self._offline_update_fn(
            self.state, self.offline
        )
        self._update_step_count += 1

    def last_metrics(self) -> dict[str, float]:
        if self._last_metrics is None:
            return {}
        values = np.asarray(jax.device_get(self._last_metrics), dtype=np.float64)
        return dict(zip(METRIC_NAMES, values.tolist()))

    def actor_params_numpy(self) -> dict:
        return jax.tree.map(np.asarray, jax.device_get(self.state["actor"]))

    # ------------------------------------------------------------------
    # Generic RoboBase replay_iter path
    # ------------------------------------------------------------------

    def _rows_from_robobase_batch(self, batch: dict) -> np.ndarray:
        gamma = self.spec.discount
        terminal = np.asarray(batch["terminal"], np.float32).reshape(-1)
        discount = np.asarray(
            batch.get("discount", np.full_like(terminal, gamma)), np.float32
        ).reshape(-1)
        return pack_transitions(
            np.asarray(batch["low_dim_state"], np.float32),
            np.asarray(batch["action"], np.float32),
            np.asarray(batch["reward"], np.float32),
            (1.0 - terminal) * discount / gamma,
            np.asarray(batch["low_dim_state_tp1"], np.float32),
        )

    def update(
        self,
        replay_iter: Iterator[dict],
        step: int,
        replay_buffer: ReplayBuffer = None,
    ) -> dict[str, float]:
        """One RLPD update from ``utd_ratio`` RoboBase batches.

        Use with ``demo_batch_size`` so each batch already mixes online and
        offline halves.  :class:`RLPDWorkspace` bypasses this host path.
        """

        del step, replay_buffer
        batches = np.stack(
            [
                self._rows_from_robobase_batch(next(replay_iter))
                for _ in range(self.spec.utd_ratio)
            ]
        )
        start = time.perf_counter()
        self.state, self._last_metrics = self._batch_update_fn(self.state, batches)
        self._update_step_count += 1
        metrics = {}
        if self.logging:
            metrics = self.last_metrics()
            metrics.pop("online_size", None)
            metrics["backend/update_time_sec"] = time.perf_counter() - start
        self._first_update_completed = True
        return metrics

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {"params": self._tree_to_numpy(self.params)}

    def load_state_dict(self, state_dict: dict):
        self.params = self._tree_from_numpy(state_dict["params"])

    def checkpoint_state_dict(self) -> dict:
        state = {
            "opt_state": self._tree_to_numpy(self.opt_state),
            "rng_key": np.asarray(self.state["rng"]),
            "act_key": np.asarray(self._act_key),
            "update_step_count": int(self._update_step_count),
        }
        if self.replay is not None:
            size = int(self.replay["size"])
            state["online_replay"] = {
                "data": np.asarray(self.replay["data"][:size]),
                "ptr": int(self.replay["ptr"]),
            }
        return state

    def load_checkpoint_state_dict(self, state_dict: dict):
        if "opt_state" in state_dict:
            self.opt_state = self._tree_from_numpy(state_dict["opt_state"])
        if "rng_key" in state_dict:
            self.state["rng"] = jnp.asarray(state_dict["rng_key"])
        if "act_key" in state_dict:
            self._act_key = jnp.asarray(state_dict["act_key"])
        self._update_step_count = int(state_dict.get("update_step_count", 0))
        online = state_dict.get("online_replay")
        if online is not None and self.replay is not None:
            data = np.asarray(online["data"], np.float32)
            self.replay = {
                "data": self.replay["data"].at[: len(data)].set(data),
                "ptr": jnp.asarray(online["ptr"], jnp.int32),
                "size": jnp.asarray(len(data), jnp.int32),
            }


__all__ = [
    "METRIC_NAMES",
    "RLPD",
    "RLPDSpec",
    "actor_apply",
    "critic_apply",
    "init_actor_params",
    "init_critic_params",
    "numpy_actor_mode",
    "pack_transitions",
    "rlpd_spec_from_cfg",
    "tanh_normal_from_noise",
    "tanh_normal_sample",
    "transition_width",
]
