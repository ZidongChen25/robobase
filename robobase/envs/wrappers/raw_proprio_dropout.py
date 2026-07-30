"""Raw-observation proprioception dropout applied before normalization."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class RawProprioDropout(gym.ObservationWrapper):
    """Replace selected raw proprioception fields with zeros."""

    def __init__(self, env: gym.Env, *, keys: tuple[str, ...], probability: float):
        super().__init__(env)
        self._keys = tuple(str(key) for key in keys)
        self._probability = float(probability)
        if not 0.0 <= self._probability <= 1.0:
            raise ValueError("Raw proprio dropout probability must be between 0 and 1.")
        missing = [
            key for key in self._keys if key not in self.observation_space.spaces
        ]
        if missing:
            raise ValueError(
                f"Raw proprio dropout keys are absent from the observation space: {missing}."
            )

    def observation(self, observation):
        obs = dict(observation)
        if self._probability <= 0.0:
            return obs
        should_drop = self._probability >= 1.0 or bool(
            self.np_random.random() < self._probability
        )
        if should_drop:
            for key in self._keys:
                obs[key] = np.zeros_like(obs[key])
        return obs
