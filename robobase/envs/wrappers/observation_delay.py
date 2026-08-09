"""Observation delay wrapper for delayed-policy training and evaluation.

Human demonstrations are not Markov in the observation the simulator reports at
step ``t``: the operator reacted to what they perceived some time earlier, so the
recorded action ``a_t`` is better explained by ``o_{t-h}`` than by ``o_t``.  A
"delayed policy" therefore conditions on ``o_{t-h}`` while still emitting the
action executed at ``t``.

This wrapper implements the acting half of that shift.  It buffers the last
``h + 1`` observations and returns the oldest one, so every consumer downstream
of it -- the agent at rollout/eval time, and the replay writer when demos are
imported through a demo env -- sees ``o_{t-h}``.  Actions, rewards, terminations
and infos keep their original timing.

Placement matters: insert it *inside* :class:`ActionSequence` /
:class:`RecedingHorizonControl` so the delay is counted in environment steps
rather than in policy decisions, and *outside* :class:`FrameStack` so a stack is
the history as of ``t - h`` rather than a stack straddling the delay.

Episode starts are padded by repeating the reset observation, which matches how
the replay buffers clamp negative observation indices to the episode start.
"""

from typing import Any

import numpy as np
import gymnasium as gym


def _copy_obs(observation: dict) -> dict:
    return {name: np.array(value, copy=True) for name, value in observation.items()}


class ObservationDelay(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Return the observation emitted ``delay`` environment steps ago.

    Args:
        env: environment to wrap. Its observation space must be a ``Dict``.
        delay: number of environment steps to look back. 0 is a no-op.
    """

    def __init__(self, env: gym.Env, delay: int):
        gym.utils.RecordConstructorArgs.__init__(self, delay=delay)
        gym.Wrapper.__init__(self, env)
        delay = int(delay)
        if delay < 0:
            raise ValueError(f"ObservationDelay requires delay >= 0, got {delay}.")
        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise ValueError(
                "ObservationDelay expects a Dict observation space, got "
                f"{type(self.observation_space)}."
            )
        self.delay = delay
        self.is_vector_env = getattr(env, "is_vector_env", False)
        # Oldest first: after a push at step t this holds [o_{t-delay}, ..., o_t].
        self._frames: list[dict] = []

    def _fill(self, observation: dict):
        self._frames = [_copy_obs(observation) for _ in range(self.delay + 1)]

    def _fill_at_idx(self, observation: dict, idx: int):
        for frame in self._frames:
            for name, value in observation.items():
                frame[name][idx] = value

    def _push(self, observation: dict):
        self._frames.append(_copy_obs(observation))
        self._frames.pop(0)

    def _push_at_idx(self, observation: dict, idx: int) -> dict:
        """Age row ``idx`` by one step and return its delayed observation."""
        for j in range(len(self._frames) - 1):
            for name in self._frames[j]:
                self._frames[j][name][idx] = self._frames[j + 1][name][idx]
        for name, value in observation.items():
            self._frames[-1][name][idx] = value
        return {
            name: np.array(value[idx], copy=True)
            for name, value in self._frames[0].items()
        }

    def _delayed(self) -> dict:
        return _copy_obs(self._frames[0])

    def reset(self, **kwargs) -> tuple[dict, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self._fill(observation)
        return self._delayed(), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self.is_vector_env and "final_observation" in info:
            # Autoreset already swapped in the next episode's first observation,
            # so the terminal observation only reaches us through ``info``. Age
            # it through the buffer, then restart the buffer from the new
            # episode's reset observation.
            for idx in np.where(info["_final_observation"])[0]:
                final_obs = info["final_observation"][idx]
                info["final_observation"][idx] = self._push_at_idx(final_obs, idx)
                self._fill_at_idx(
                    {name: value[idx] for name, value in observation.items()}, idx
                )
        self._push(observation)
        return self._delayed(), reward, terminated, truncated, info


def maybe_delay_observations(env: gym.Env, cfg) -> gym.Env:
    """Wrap ``env`` in :class:`ObservationDelay` when ``cfg.obs_delay > 0``.

    Reads the config defensively so checkpoints saved before ``obs_delay``
    existed keep loading.
    """
    delay = int(cfg.get("obs_delay", 0) or 0)
    if delay <= 0:
        return env
    return ObservationDelay(env, delay)
