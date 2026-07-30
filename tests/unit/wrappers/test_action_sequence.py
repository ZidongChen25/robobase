import pytest
import numpy as np
from gymnasium import spaces
from gymnasium.vector import SyncVectorEnv
from tests.unit.wrappers.utils import DummyEnv, ACTION_SHAPE
from robobase.envs.wrappers import ActionSequence, RecedingHorizonControl

NUM_ENVS = 2
SEQ_LEN = 5
TIME_LIMIT = 100
EXE_LEN = 1


def _create_receding_horizon_env():
    env = RecedingHorizonControl(
        DummyEnv(),
        sequence_length=SEQ_LEN,
        time_limit=TIME_LIMIT,
        execution_length=EXE_LEN,
        temporal_ensemble=True,
    )
    return env


def test_action_sequence_has_correct_shape():
    env = ActionSequence(DummyEnv(), SEQ_LEN)
    assert env.action_space.shape == (SEQ_LEN,) + ACTION_SHAPE


def test_action_sequence_vec_has_correct_shape():
    env = SyncVectorEnv(
        [lambda: ActionSequence(DummyEnv(), SEQ_LEN) for _ in range(NUM_ENVS)]
    )
    assert env.action_space.shape == (NUM_ENVS, SEQ_LEN) + ACTION_SHAPE


def test_action_sequence_can_step():
    env = ActionSequence(DummyEnv(), SEQ_LEN)
    env.reset()
    for _ in range(5):
        obs, *_, info = env.step(env.action_space.sample())
        assert "action_sequence_mask" in info
        assert info["action_sequence_mask"].shape == (SEQ_LEN,)


def test_action_sequence_preserves_each_executed_feedback_state():
    class FeedbackEnv(DummyEnv):
        def __init__(self):
            super().__init__(episode_len=10)
            obs_spaces = dict(self.observation_space.spaces)
            obs_spaces["executed_action_feedback"] = spaces.Box(
                -np.inf, np.inf, shape=ACTION_SHAPE, dtype=np.float32
            )
            self.observation_space = spaces.Dict(obs_spaces)

        def _with_feedback(self, observation):
            observation["executed_action_feedback"] = np.full(
                ACTION_SHAPE, self._steps, dtype=np.float32
            )
            return observation

        def reset(self, *args, **kwargs):
            observation, info = super().reset(*args, **kwargs)
            return self._with_feedback(observation), info

        def step(self, action):
            observation, *rest = super().step(action)
            return self._with_feedback(observation), *rest

    env = ActionSequence(FeedbackEnv(), sequence_length=3)
    reset_observation, _ = env.reset()
    observation, *_ = env.step(env.action_space.sample())

    np.testing.assert_array_equal(
        reset_observation["executed_action_feedback"], np.zeros((3, 2))
    )
    np.testing.assert_array_equal(
        observation["executed_action_feedback"][:, 0], [1, 2, 3]
    )


def test_action_sequence_can_step_vec_wrapped_env():
    env = SyncVectorEnv(
        [lambda: ActionSequence(DummyEnv(), SEQ_LEN) for _ in range(NUM_ENVS)]
    )
    env.reset()
    for _ in range(5):
        obs, *_, info = env.step(env.action_space.sample())
        assert "action_sequence_mask" in info["final_info"][0]
        assert info["final_info"][0]["action_sequence_mask"].shape == (SEQ_LEN,)


def test_action_sequence_cant_be_used_with_vec_env():
    with pytest.raises(NotImplementedError):
        ActionSequence(
            SyncVectorEnv([lambda: DummyEnv() for _ in range(NUM_ENVS)]), SEQ_LEN
        )


def test_receding_horizon_has_correct_shape():
    env = _create_receding_horizon_env()
    assert env.action_space.shape == (SEQ_LEN,) + ACTION_SHAPE


def test_receding_horizon_vec_has_correct_shape():
    env = SyncVectorEnv(
        [lambda: _create_receding_horizon_env() for _ in range(NUM_ENVS)]
    )
    assert env.action_space.shape == (NUM_ENVS, SEQ_LEN) + ACTION_SHAPE


def test_receding_horizon_can_step():
    env = _create_receding_horizon_env()
    env.reset()
    for _ in range(5):
        obs, *_, info = env.step(env.action_space.sample())
        assert "action_sequence_mask" in info
        assert info["action_sequence_mask"].shape == (SEQ_LEN,)


def test_receding_horizon_can_step_vec_wrapped_env():
    env = SyncVectorEnv(
        [lambda: _create_receding_horizon_env() for _ in range(NUM_ENVS)]
    )
    env.reset()
    for _ in range(5):
        obs, *_, info = env.step(env.action_space.sample())
        if "final_info" in info:
            assert "action_sequence_mask" in info["final_info"][0]
            assert info["final_info"][0]["action_sequence_mask"].shape == (SEQ_LEN,)
        else:
            assert "action_sequence_mask" in info
            assert info["action_sequence_mask"][0].shape == (SEQ_LEN,)


def test_receding_horizon_cant_be_used_with_vec_env():
    with pytest.raises(NotImplementedError):
        RecedingHorizonControl(
            SyncVectorEnv([lambda: DummyEnv() for _ in range(NUM_ENVS)]),
            sequence_length=SEQ_LEN,
            time_limit=TIME_LIMIT,
            execution_length=EXE_LEN,
            temporal_ensemble=True,
        )
