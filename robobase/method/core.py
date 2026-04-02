"""Abstract base classes for learning methods.

This module defines the framework-agnostic ``Method`` ABC and its subclass
categories (imitation learning, off-policy, on-policy, model-based).
Concrete implementations (e.g. BC, Diffusion) should inherit from one of
these categories and from a backend-specific mixin such as ``JaxMethodBase``.
"""

from abc import ABC, abstractmethod
from typing import Iterator, TypeAlias

import numpy as np
from gymnasium import spaces

from robobase.replay_buffer.replay_buffer import ReplayBuffer


BatchedActionSequence: TypeAlias = np.ndarray
Metrics: TypeAlias = dict[str, np.ndarray]


class Method(ABC):
    """Framework-agnostic abstract base for all learning methods."""

    observation_space: spaces.Dict
    action_space: spaces.Box
    num_train_envs: int
    num_eval_envs: int

    @property
    def random_explore_action(self) -> np.ndarray:
        """Sample a random action in [-1, 1] for all train envs."""
        return np.random.uniform(
            low=-1.0,
            high=1.0,
            size=(self.num_train_envs,) + self.action_space.shape,
        ).astype(np.float32)

    @abstractmethod
    def act(
        self, observations: dict[str, np.ndarray], step: int, eval_mode: bool,
    ) -> BatchedActionSequence:
        pass

    @abstractmethod
    def update(
        self,
        replay_iter: Iterator[dict[str, np.ndarray]],
        step: int,
        replay_buffer: ReplayBuffer = None,
    ) -> Metrics:
        pass

    @abstractmethod
    def reset(self, step: int, agents_to_reset: list[int]):
        pass

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    @abstractmethod
    def state_dict(self) -> dict:
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: dict):
        pass

    @abstractmethod
    def checkpoint_state_dict(self) -> dict[str, dict]:
        pass

    @abstractmethod
    def load_checkpoint_state_dict(self, state_dict: dict[str, dict]):
        pass

    # ------------------------------------------------------------------
    # Eval-env helpers
    # ------------------------------------------------------------------

    @property
    def eval_env_running(self):
        return self._eval_env_running

    def set_eval_env_running(self, value: bool):
        self._eval_env_running = value


class ImitationLearningMethod(Method, ABC):
    pass


class OffPolicyMethod(Method, ABC):
    pass


class OnPolicyMethod(Method, ABC):
    pass


class ModelBasedMethod(Method, ABC):
    pass
