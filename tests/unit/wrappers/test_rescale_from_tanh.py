import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import SyncVectorEnv
from tests.unit.wrappers.utils import DummyEnv, DummyEEEnv, ACTION_SHAPE
from robobase.envs.wrappers import (
    RescaleFromTanh,
    RescaleFromTanhEEPose,
    RescaleFromTanhWithStandardization,
    RescaleFromTanhWithMinMax,
)

NUM_ENVS = 2


class TightActionEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(-1, 1, (1,), dtype=np.float32)
        self.action_space = spaces.Box(
            np.array([-0.34], dtype=np.float32),
            np.array([0.34], dtype=np.float32),
            dtype=np.float32,
        )


def _assert_rescale_from_tanh_to_2(env):
    assert np.all(env.action_space.low == -1)
    assert np.all(env.action_space.high == 1)
    rescaled_action = env.action(np.ones_like(env.action_space.sample()))
    assert np.all(rescaled_action == 2)


def _assert_rescale_from_tanh_ee_unit_quarternion(env):
    assert np.all(env.action_space.low == -1)
    assert np.all(env.action_space.high == 1)
    rescaled_action = env.action(env.action_space.sample())
    assert np.allclose(np.linalg.norm(rescaled_action[3:7]), 1)


def _assert_rescale_from_tanh_ee_unit_quarternion_vec_env(env):
    assert np.all(env.action_space.low == -1)
    assert np.all(env.action_space.high == 1)
    rescaled_action = env.action(env.action_space.sample())
    assert np.allclose(
        np.linalg.norm(rescaled_action[:, 3:7], axis=1),
        np.ones(rescaled_action.shape[0]),
    )


def _assert_rescale_from_tanh_with_standardization(env):
    assert np.all(env.action_space.low == -1)
    assert np.all(env.action_space.high == 1)
    rescaled_action = env.action(np.ones_like(env.action_space.sample()))
    assert np.allclose(rescaled_action, 3 * 0.1 + 0.5)
    rescaled_action = env.action(-np.ones_like(env.action_space.sample()))
    assert np.allclose(rescaled_action, -3 * 0.1 + 0.5)


def _assert_rescale_from_tanh_with_minmax(env):
    assert np.all(env.action_space.low == -1)
    assert np.all(env.action_space.high == 1)
    rescaled_action = env.action(np.ones_like(env.action_space.sample()))
    assert np.allclose(rescaled_action, 1 * 1.2 - 0.6 * 1.2)
    rescaled_action = env.action(-np.ones_like(env.action_space.sample()))
    assert np.allclose(rescaled_action, 0 * 1.2 - 0.6 * 1.2)


def test_rescale_single_env():
    env = RescaleFromTanh(DummyEnv())
    _assert_rescale_from_tanh_to_2(env)


def test_rescale_ee_single_env():
    env = RescaleFromTanhEEPose(DummyEEEnv())
    _assert_rescale_from_tanh_ee_unit_quarternion(env)


def test_rescale_with_standardization_single_env():
    action_stats = {
        "mean": np.ones(ACTION_SHAPE) * 0.5,
        "std": np.ones(ACTION_SHAPE) * 0.1,
    }
    env = RescaleFromTanhWithStandardization(DummyEnv(), action_stats)
    _assert_rescale_from_tanh_with_standardization(env)


def test_rescale_with_minmax_single_env():
    action_stats = {
        "min": -np.ones(ACTION_SHAPE) * 0.6,
        "max": np.ones(ACTION_SHAPE) * 0.4,
    }
    env = RescaleFromTanhWithMinMax(DummyEnv(), action_stats, min_max_margin=0.2)
    _assert_rescale_from_tanh_with_minmax(env)


def test_rescale_with_minmax_clips_to_raw_action_space():
    action_stats = {
        "min": np.array([-0.34], dtype=np.float32),
        "max": np.array([0.34000003], dtype=np.float32),
    }
    env = RescaleFromTanhWithMinMax(TightActionEnv(), action_stats)

    rescaled_action = env.action(np.ones_like(env.action_space.sample()))

    assert rescaled_action[0] == env.orig_action_space.high[0]


def test_rescale_wrapped_vec_env():
    env = RescaleFromTanh(SyncVectorEnv([lambda: DummyEnv() for _ in range(NUM_ENVS)]))
    _assert_rescale_from_tanh_to_2(env)


def test_rescale_ee_wrapped_vec_env():
    env = RescaleFromTanhEEPose(
        SyncVectorEnv([lambda: DummyEEEnv() for _ in range(NUM_ENVS)])
    )
    _assert_rescale_from_tanh_ee_unit_quarternion_vec_env(env)


def test_rescale_with_standardization_wrapped_vec_env():
    action_stats = {
        "mean": np.ones(ACTION_SHAPE) * 0.5,
        "std": np.ones(ACTION_SHAPE) * 0.1,
    }
    env = RescaleFromTanhWithStandardization(
        SyncVectorEnv([lambda: DummyEnv() for _ in range(NUM_ENVS)]), action_stats
    )
    _assert_rescale_from_tanh_with_standardization(env)


def test_rescale_with_minmax_wrapped_vec_env():
    action_stats = {
        "min": -np.ones(ACTION_SHAPE) * 0.6,
        "max": np.ones(ACTION_SHAPE) * 0.4,
    }
    env = RescaleFromTanhWithMinMax(
        SyncVectorEnv([lambda: DummyEnv() for _ in range(NUM_ENVS)]),
        action_stats,
        min_max_margin=0.2,
    )
    _assert_rescale_from_tanh_with_minmax(env)
