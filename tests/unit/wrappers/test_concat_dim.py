import numpy as np
from gymnasium.vector import SyncVectorEnv
from tests.unit.wrappers.utils import (
    DummyEnv,
    OBS_SIZE,
    OBS_NAME_FLAT1,
    OBS_NAME_IMG1,
    OBS_NAME_IMG2,
    IMG_SHAPE,
    OBS_NAME_FLAT2,
)
from robobase.envs.wrappers import ConcatDim

NUM_ENVS = 2
CAT_NAME = "combined"
NEW_FLAT_SIZE = OBS_SIZE + OBS_SIZE
NEW_IMG_SHAPE = (6,) + IMG_SHAPE[1:]


def test_concat_images_single_env():
    env = ConcatDim(DummyEnv(), 3, 0, CAT_NAME)
    assert OBS_NAME_IMG1 not in env.observation_space
    assert OBS_NAME_IMG2 not in env.observation_space
    assert CAT_NAME in dict(env.observation_space)
    assert env.observation_space[CAT_NAME].shape == NEW_IMG_SHAPE
    obs, _ = env.reset()
    assert obs[CAT_NAME].shape == NEW_IMG_SHAPE
    obs, *_ = env.step(env.action_space.sample())
    assert obs[CAT_NAME].shape == NEW_IMG_SHAPE


def test_concat_low_dim_single_env():
    env = ConcatDim(DummyEnv(), 1, 0, CAT_NAME)
    assert OBS_NAME_FLAT1 not in env.observation_space
    assert OBS_NAME_FLAT2 not in env.observation_space
    assert CAT_NAME in dict(env.observation_space)
    assert env.observation_space[CAT_NAME].shape == (NEW_FLAT_SIZE,)
    obs, _ = env.reset()
    assert obs[CAT_NAME].shape == (NEW_FLAT_SIZE,)
    obs, *_ = env.step(env.action_space.sample())
    assert obs[CAT_NAME].shape == (NEW_FLAT_SIZE,)


def test_concat_images_vec_wrapped_env():
    env = SyncVectorEnv(
        [lambda: ConcatDim(DummyEnv(), 1, 0, CAT_NAME) for _ in range(NUM_ENVS)]
    )
    assert OBS_NAME_FLAT1 not in env.observation_space
    assert OBS_NAME_FLAT2 not in env.observation_space
    assert CAT_NAME in dict(env.observation_space)
    assert env.observation_space[CAT_NAME].shape == (NUM_ENVS, NEW_FLAT_SIZE)
    obs, _ = env.reset()
    for _ in range(5):
        obs, *_, info = env.step(env.action_space.sample())
        assert obs[CAT_NAME].shape == (NUM_ENVS, NEW_FLAT_SIZE)
    assert info["final_observation"][0][CAT_NAME].shape == (NEW_FLAT_SIZE,)


def test_concat_images_wrapped_vec_env():
    env = ConcatDim(
        SyncVectorEnv([lambda: DummyEnv() for _ in range(NUM_ENVS)]), 1, 0, CAT_NAME
    )
    assert OBS_NAME_FLAT1 not in env.observation_space
    assert OBS_NAME_FLAT2 not in env.observation_space
    assert CAT_NAME in dict(env.observation_space)
    assert env.observation_space[CAT_NAME].shape == (NUM_ENVS, NEW_FLAT_SIZE)
    obs, _ = env.reset()
    for _ in range(5):
        obs, *_, info = env.step(env.action_space.sample())
        assert obs[CAT_NAME].shape == (NUM_ENVS, NEW_FLAT_SIZE)
    assert info["final_observation"][0][CAT_NAME].shape == (NEW_FLAT_SIZE,)


def test_min_max_constant_dimensions_use_configured_reference_value():
    base_env = DummyEnv()
    observation, _ = base_env.reset()
    flat_keys = (OBS_NAME_FLAT1, OBS_NAME_FLAT2)
    stats = {
        "mean": {key: observation[key].copy() for key in flat_keys},
        "std": {key: np.ones_like(observation[key]) for key in flat_keys},
        "min": {key: observation[key].copy() for key in flat_keys},
        "max": {key: observation[key].copy() for key in flat_keys},
    }
    env = ConcatDim(
        base_env,
        1,
        0,
        CAT_NAME,
        norm_obs=True,
        obs_stats=stats,
        obs_norm_type="min_max",
        min_max_constant_value=-1.0,
    )

    normalized = env.observation(observation)[CAT_NAME]

    np.testing.assert_array_equal(normalized, -np.ones_like(normalized))
