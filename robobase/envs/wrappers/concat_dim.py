"""Concatenates dictionary of observations that share same shape."""
import numpy as np

import gymnasium as gym
from gymnasium.spaces import Box, Dict


class ConcatDim(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Concatenates dictionary of observations that share same shape."""

    def __init__(
        self,
        env: gym.Env,
        shape_length: int,
        dim: int,
        new_name: str,
        norm_obs: bool = False,
        obs_stats: dict = None,
        obs_norm_type: str = "standardization",
        min_max_constant_value: float = 0.0,
        keys_to_ignore: list[str] = None,
    ):
        """Init.

        Args:
            env: The environment to apply the wrapper
            shape_length: The ndim we are interested in, e.g. images=3, low_dim=1.
            dim: The oberservations with this ...
            new_name: The name of the new observation.
            norm_obs: Whether to normalize observations.
            obs_stats: The obs statistics for normalizing observations.
            keys_to_ignore: A list of keys to not include in this combined observation,
                regardless if they meet shape_len.
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self._shape_length = shape_length + int(self.is_vector_env)
        self._dim = dim + int(self.is_vector_env)
        self._new_name = new_name
        self._keys_to_ignore = [] if keys_to_ignore is None else keys_to_ignore
        self._norm_obs = norm_obs
        self._obs_stats = obs_stats
        self._obs_norm_type = str(obs_norm_type).lower()
        self._min_max_constant_value = float(min_max_constant_value)
        new_obs_dict = {}
        combined = []
        for k, v in self.observation_space.items():
            if len(v.shape) == self._shape_length and k not in self._keys_to_ignore:
                combined.append(v)
            else:
                new_obs_dict[k] = v
        new_min = np.concatenate(list(map(lambda s: s.low, combined)), self._dim)
        new_max = np.concatenate(list(map(lambda s: s.high, combined)), self._dim)
        new_obs_dict[new_name] = Box(new_min, new_max, dtype=np.float32)
        self.observation_space = Dict(new_obs_dict)

    def _transform_timestep(self, observation, final: bool = False):
        shape_len = self._shape_length - int(final)
        dim = self._dim - int(final)
        new_obs = {}
        combined = []
        for k, v in observation.items():
            # We allow normalizing observations in the ConcatDim wrapper
            # because all obs stats are stored with original key names and
            # ConcatDim will rename them to new keys. Doing it here would
            # safer and cleaner.
            if len(v.shape) == shape_len and k not in self._keys_to_ignore:
                if (
                    self._norm_obs
                    and self._obs_stats is not None
                    and k in self._obs_stats["mean"]
                ):
                    if self._obs_norm_type in {"min_max", "minmax"}:
                        obs_min = self._obs_stats["min"][k]
                        obs_max = self._obs_stats["max"][k]
                        obs_range = obs_max - obs_min
                        mask = obs_range != 0
                        obs_range = np.where(obs_range == 0, 1.0, obs_range)
                        normalized = (v - obs_min) / obs_range * 2.0 - 1.0
                        v = np.where(
                            mask,
                            normalized,
                            self._min_max_constant_value,
                        ).astype(v.dtype, copy=False)
                    else:
                        v = (v - self._obs_stats["mean"][k]) / (
                            self._obs_stats["std"][k] + 1e-10
                        )
                combined.append(v)
            else:
                new_obs[k] = v
        new_obs[self._new_name] = np.concatenate(combined, dim)
        return new_obs

    def observation(self, observation):
        """Adds to the observation with the current time step.

        Args:
            observation: The observation to add the time step to

        Returns:
            The observation with the time step appended to
        """
        return self._transform_timestep(observation)

    def step(self, action):
        """Steps through the environment, incrementing the time step.

        Args:
            action: The action to take

        Returns:
            The environment's step using the action.
        """
        observations, *rest, info = super().step(action)
        if "final_observation" in info:
            for fidx in np.where(info["_final_observation"])[0]:
                info["final_observation"][fidx] = self._transform_timestep(
                    info["final_observation"][fidx], final=True
                )
        return self.observation(observations), *rest, info


class AppendKeysToLowDim(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Append raw low-dim keys to the tail of the combined low-dim state.

    Placing the appended keys at the end gives them a fixed, known position
    inside every stacked frame, which training-time low-dim masking relies
    on (mask everything except the last ``keep`` dims).
    """

    def __init__(
        self,
        env: gym.Env,
        keys: list[str],
        low_dim_key: str = "low_dim_state",
        norm_obs: bool = False,
        obs_stats: dict = None,
        obs_norm_type: str = "min_max",
    ):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self._keys = list(keys)
        self._low_dim_key = low_dim_key
        self._norm_obs = norm_obs
        self._obs_stats = obs_stats
        self._obs_norm_type = str(obs_norm_type).lower()
        spaces_dict = dict(self.observation_space.items())
        base = spaces_dict[low_dim_key]
        lows = [base.low] + [spaces_dict[k].low for k in self._keys]
        highs = [base.high] + [spaces_dict[k].high for k in self._keys]
        spaces_dict[low_dim_key] = Box(
            np.concatenate(lows, -1).astype(np.float32),
            np.concatenate(highs, -1).astype(np.float32),
            dtype=np.float32,
        )
        self.observation_space = Dict(spaces_dict)

    def _normalize(self, key: str, value: np.ndarray) -> np.ndarray:
        if (
            not self._norm_obs
            or self._obs_stats is None
            or key not in self._obs_stats["mean"]
        ):
            return value
        if self._obs_norm_type in {"min_max", "minmax"}:
            obs_min = self._obs_stats["min"][key]
            obs_max = self._obs_stats["max"][key]
            obs_range = np.where(obs_max - obs_min == 0, 1.0, obs_max - obs_min)
            return ((value - obs_min) / obs_range * 2.0 - 1.0).astype(
                np.float32, copy=False
            )
        return (
            (value - self._obs_stats["mean"][key])
            / (self._obs_stats["std"][key] + 1e-10)
        ).astype(np.float32, copy=False)

    def observation(self, observation):
        observation = dict(observation)
        parts = [np.asarray(observation[self._low_dim_key], dtype=np.float32)]
        for key in self._keys:
            parts.append(
                self._normalize(key, np.asarray(observation[key], np.float32))
            )
        observation[self._low_dim_key] = np.concatenate(parts, axis=-1)
        return observation

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "final_observation" in info:
            for fidx in np.where(info["_final_observation"])[0]:
                info["final_observation"][fidx] = self.observation(
                    info["final_observation"][fidx]
                )
        return self.observation(observation), reward, terminated, truncated, info
