"""Base class for JAX-backed method implementations.

Extracts the shared initialisation, feature-extraction, and state-management
logic that was previously duplicated across JaxBC and JaxDiffusion.
"""

from __future__ import annotations

import time
import warnings
from abc import abstractmethod
from typing import Iterator, Optional

import numpy as np
from gymnasium import spaces

from robobase.method.bc_runtime import (
    bc_observation_layout,
    flatten_time_into_channel,
)
from robobase.method.core import Method
from robobase.method.jax_utils import maybe_numpy
from robobase.replay_buffer.replay_buffer import ReplayBuffer
from robobase.replay_buffer.vision_feature_cache import (
    cached_feature_observation_key,
)


class JaxMethodBase(Method):
    """Shared base for all JAX method implementations.

    Handles JAX module imports, observation layout parsing, encoder/fusion
    setup, feature extraction helpers, optimizer construction, and state
    serialisation.  Subclasses only need to implement :meth:`act`,
    :meth:`update`, and model-specific build logic.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(
        self,
        lr: float,
        adaptive_lr: bool,
        num_train_steps: int,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        num_train_envs: int,
        num_eval_envs: int,
        replay_alpha: float,
        replay_beta: float,
        frame_stack_on_channel: bool,
        intrinsic_reward_module=None,
        actor_grad_clip: Optional[float] = None,
        jit: bool = True,
        platform: str | None = None,
        seed: int = 0,
        is_rl: bool = False,
        use_ema: bool = False,
        update_block_every_steps: int = 1,
    ):
        if intrinsic_reward_module is not None:
            raise NotImplementedError(
                "JAX method implementations do not support intrinsic rewards yet."
            )
        if len(action_space.shape) != 2:
            raise ValueError(
                "JAX methods expect action_space.shape == (sequence_length, action_dim)."
            )

        # -- JAX imports (lazy so the module can be imported without JAX) --
        import jax
        import jax.numpy as jnp
        import optax

        if platform:
            requested_platform = str(platform)
            jax.config.update("jax_platform_name", requested_platform)
            try:
                # Force backend creation early so we can gracefully fallback in CPU-only envs.
                _ = jax.devices()
            except RuntimeError as exc:
                if requested_platform.lower() in {"cuda", "gpu"}:
                    jax.config.update("jax_platform_name", "cpu")
                    warnings.warn(
                        "Requested JAX platform '%s' is unavailable; falling back to 'cpu'."
                        % requested_platform,
                        RuntimeWarning,
                        stacklevel=2,
                    )
                else:
                    raise

        self.jax = jax
        self.jnp = jnp
        self.optax = optax

        # -- Hyperparameters --
        self.lr = lr
        self.adaptive_lr = adaptive_lr
        self.num_train_steps = max(1, num_train_steps)
        self.actor_grad_clip = actor_grad_clip

        # -- Spaces & env counts --
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_train_envs = num_train_envs
        self.num_eval_envs = num_eval_envs
        self.replay_alpha = replay_alpha
        self.replay_beta = replay_beta
        self.frame_stack_on_channel = frame_stack_on_channel

        # -- Runtime flags --
        self.training = True
        self.logging = False
        self.is_rl = is_rl
        self.use_ema = use_ema
        self._eval_env_running = False
        self.backend_name = "jax"
        self._jit_enabled = jit
        self._first_update_completed = False
        self._update_step_count = 0
        self._update_block_every_steps = max(1, int(update_block_every_steps))

        # -- Observation layout --
        self.obs_layout = bc_observation_layout(observation_space)
        self.time_dim = self.obs_layout.time_dim
        self.low_dim_size = self.obs_layout.low_dim_size
        self.use_pixels = self.obs_layout.use_pixels
        self.use_multicam_fusion = self.obs_layout.use_multicam_fusion
        self._rgb_batch_keys = tuple(self.obs_layout.rgb_keys)
        self.action_sequence = int(action_space.shape[0])
        self.action_dim = int(action_space.shape[1])

        # -- RNG --
        self.rng_key = jax.random.PRNGKey(int(seed))

    # Subclasses should call this after their own model build to set
    # ``self._cached_pixel_feature_key``.
    def _init_cached_pixel_feature_key(self, method_name: str) -> None:
        self._cached_pixel_feature_key = (
            cached_feature_observation_key(method_name, "jax")
            if self.use_pixels
            else None
        )

    # ------------------------------------------------------------------
    # Feature extraction helpers (shared between BC / Diffusion / ...)
    # ------------------------------------------------------------------

    def _extract_low_dim_batch(self, batch_or_obs: dict):
        if self.low_dim_size == 0 or "low_dim_state" not in batch_or_obs:
            return None
        low_dim_obs = self._as_jax_array(batch_or_obs["low_dim_state"], self.jnp.float32)
        low_dim_obs = flatten_time_into_channel(low_dim_obs)
        return low_dim_obs.reshape((low_dim_obs.shape[0], -1))

    def _extract_rgb_obs(self, batch_or_obs: dict):
        if not self.use_pixels:
            return None, {}
        rgb_obs_dict = {key: batch_or_obs[key] for key in self._rgb_batch_keys}
        metrics = {}
        if self.logging:
            metrics = {
                key: maybe_numpy(value)[0, -1]
                for key, value in rgb_obs_dict.items()
            }
        rgb_obs = flatten_time_into_channel(
            self.jnp.stack(
                [
                    self._as_jax_array(value, self.jnp.float32)
                    for value in rgb_obs_dict.values()
                ],
                axis=1,
            ),
            has_view_axis=True,
        )
        return rgb_obs, metrics

    def _has_cached_pixel_features(self, batch_or_obs: dict) -> bool:
        return (
            self._cached_pixel_feature_key is not None
            and self._cached_pixel_feature_key in batch_or_obs
        )

    def _extract_cached_pixel_features(self, batch_or_obs: dict):
        if not self._has_cached_pixel_features(batch_or_obs):
            return None
        cached = self._as_jax_array(
            batch_or_obs[self._cached_pixel_feature_key], self.jnp.float32
        )
        cached = flatten_time_into_channel(cached)
        return cached.reshape((cached.shape[0], -1))

    def _extract_action_pad_mask(self, batch: dict):
        if "action_pad_mask" not in batch:
            return None
        return self._as_jax_array(batch["action_pad_mask"], self.jnp.bool_)

    def _loss_weights(self, batch: dict):
        if "sampling_probabilities" in batch:
            probs = self._as_jax_array(
                batch["sampling_probabilities"], self.jnp.float32
            )
            weights = 1.0 / self.jnp.sqrt(probs + 1e-10)
            weights = (weights / self.jnp.max(weights)) ** self.replay_beta
            return weights.astype(self.jnp.float32)
        batch_size = int(batch["action"].shape[0])
        return self.jnp.ones((batch_size,), dtype=self.jnp.float32)

    def _encode_pixels(self, rgb_obs):
        if self.encoder is None:
            return None
        return self.encoder.encode(rgb_obs)

    def _fuse_multi_view(self, rgb_feats):
        if rgb_feats is None:
            return None
        rgb_feats = self.jnp.asarray(rgb_feats, dtype=self.jnp.float32)
        if self.view_fusion is not None:
            return self.view_fusion.apply(self.jnp, rgb_feats)
        return rgb_feats[:, 0]

    def _combine_features(self, low_dim_obs, fused_view_feats):
        features = []
        if low_dim_obs is not None:
            features.append(self.jnp.asarray(low_dim_obs, dtype=self.jnp.float32))
        if fused_view_feats is not None:
            features.append(self.jnp.asarray(fused_view_feats, dtype=self.jnp.float32))
        if not features:
            raise ValueError("Method requires at least one observation feature source.")
        if len(features) == 1:
            return features[0]
        return self.jnp.concatenate(features, axis=-1)

    def _prepare_obs_features(self, batch_or_obs: dict):
        """Extract low-dim + pixel features and combine into a single array.

        Returns ``(obs_features, metrics)`` where *metrics* is empty outside
        of logging mode.
        """
        low_dim_obs = self._extract_low_dim_batch(batch_or_obs)
        fused_view_feats = self._extract_cached_pixel_features(batch_or_obs)
        metrics = {}
        if self.use_pixels and fused_view_feats is None:
            rgb_obs, pixel_metrics = self._extract_rgb_obs(batch_or_obs)
            metrics.update(pixel_metrics)
            fused_view_feats = self._fuse_multi_view(self._encode_pixels(rgb_obs))
        obs_features = self._combine_features(low_dim_obs, fused_view_feats)
        return obs_features, metrics

    # ------------------------------------------------------------------
    # JAX helpers
    # ------------------------------------------------------------------

    def _block(self, *values):
        for value in values:
            self.jax.block_until_ready(value)

    def _as_jax_array(self, value, dtype=None):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return self.jnp.asarray(value, dtype=dtype)

    def prefetch_batch(self, batch: dict):
        return self.jax.tree_util.tree_map(self.jax.device_put, batch)

    def _tree_to_numpy(self, tree):
        return self.jax.tree_util.tree_map(
            lambda x: x if x is None else np.asarray(self.jax.device_get(x)),
            tree,
        )

    def _tree_from_numpy(self, tree):
        return self.jax.tree_util.tree_map(self.jnp.asarray, tree)

    # ------------------------------------------------------------------
    # Priority update (shared replay buffer interaction)
    # ------------------------------------------------------------------

    def _maybe_update_priorities(
        self, replay_buffer, batch, new_priority_np: np.ndarray
    ):
        if replay_buffer is not None and hasattr(replay_buffer, "set_priority"):
            replay_buffer.set_priority(
                indices=maybe_numpy(batch["indices"]),
                priorities=new_priority_np ** self.replay_alpha,
            )

    def _uses_replay_priorities(self, replay_buffer) -> bool:
        return replay_buffer is not None and hasattr(replay_buffer, "set_priority")

    def _should_block_update(self, uses_priorities: bool) -> bool:
        if uses_priorities or self.logging:
            return True
        return (self._update_step_count + 1) % self._update_block_every_steps == 0

    # ------------------------------------------------------------------
    # Logging metrics (shared)
    # ------------------------------------------------------------------

    def _maybe_log_update_metrics(
        self, metrics: dict, actor_loss, obs_features, elapsed: float
    ):
        if self.logging:
            metrics["actor_loss"] = float(
                np.asarray(self.jax.device_get(actor_loss))
            )
            metrics["backend/update_time_sec"] = elapsed
            metrics["backend/update_steps_per_second"] = obs_features.shape[0] / max(
                elapsed, 1e-12,
            )
            if not self._first_update_completed:
                metrics["backend/first_update_time_sec"] = elapsed

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {"params": self._tree_to_numpy(self.params)}

    def load_state_dict(self, state_dict: dict):
        self.params = self._tree_from_numpy(state_dict["params"])

    def checkpoint_state_dict(self) -> dict[str, dict]:
        return {
            "opt_state": self._tree_to_numpy(self.opt_state),
            "rng_key": np.asarray(self.rng_key),
            "update_step_count": int(self._update_step_count),
            "first_update_completed": bool(self._first_update_completed),
        }

    def load_checkpoint_state_dict(self, state_dict: dict[str, dict]):
        if "opt_state" in state_dict:
            self.opt_state = self._tree_from_numpy(state_dict["opt_state"])
        if "rng_key" in state_dict:
            self.rng_key = self.jnp.asarray(state_dict["rng_key"])
        self._update_step_count = int(state_dict.get("update_step_count", 0))
        self._first_update_completed = bool(
            state_dict.get("first_update_completed", False)
        )

    # ------------------------------------------------------------------
    # Training / eval helpers
    # ------------------------------------------------------------------

    @property
    def eval_env_running(self):
        return self._eval_env_running

    def set_eval_env_running(self, value: bool):
        self._eval_env_running = value

    def train(self, training: bool):
        self.training = bool(training)

    def reset(self, step: int, agents_to_reset: list[int]):
        del step, agents_to_reset

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def act(self, observations: dict, step: int, eval_mode: bool):
        pass

    @abstractmethod
    def update(
        self,
        replay_iter: Iterator[dict],
        step: int,
        replay_buffer: ReplayBuffer = None,
    ) -> dict[str, np.ndarray]:
        pass
