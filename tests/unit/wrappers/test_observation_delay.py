import gymnasium as gym
import numpy as np
import pytest
from omegaconf import OmegaConf

from robobase.envs.wrappers import (
    ActionSequence,
    FrameStack,
    ObservationDelay,
    maybe_delay_observations,
)
from tests.unit.wrappers.utils import OBS_NAME_FLAT1, DummyEnv


def _value(observation) -> float:
    """DummyEnv fills every observation with its environment step index."""
    return float(np.asarray(observation[OBS_NAME_FLAT1]).reshape(-1)[0])


def test_observation_delay_returns_past_observation_and_pads_episode_start():
    env = ObservationDelay(DummyEnv(episode_len=6), delay=2)

    observation, _ = env.reset()
    seen = [_value(observation)]
    for _ in range(4):
        observation, _, _, _, _ = env.step(env.action_space.sample())
        seen.append(_value(observation))

    # Env emits o_0..o_4; the agent sees o_{t-2} with o_0 repeated at the start.
    assert seen == [0.0, 0.0, 0.0, 1.0, 2.0]


def test_observation_delay_zero_is_identity_stream():
    env = ObservationDelay(DummyEnv(episode_len=6), delay=0)

    observation, _ = env.reset()
    seen = [_value(observation)]
    for _ in range(3):
        observation, _, _, _, _ = env.step(env.action_space.sample())
        seen.append(_value(observation))

    assert seen == [0.0, 1.0, 2.0, 3.0]


def test_observation_delay_refills_buffer_on_reset():
    env = ObservationDelay(DummyEnv(episode_len=6), delay=2)

    env.reset()
    for _ in range(4):
        env.step(env.action_space.sample())

    observation, _ = env.reset()
    assert _value(observation) == 0.0
    observation, _, _, _, _ = env.step(env.action_space.sample())
    assert _value(observation) == 0.0


def test_observation_delay_does_not_mutate_returned_observations():
    env = ObservationDelay(DummyEnv(episode_len=6), delay=1)

    first, _ = env.reset()
    first[OBS_NAME_FLAT1][:] = -99.0
    second, _, _, _, _ = env.step(env.action_space.sample())

    assert _value(second) == 0.0


def test_observation_delay_rejects_negative_delay():
    with pytest.raises(ValueError, match="delay >= 0"):
        ObservationDelay(DummyEnv(), delay=-1)


def test_observation_delay_is_counted_in_env_steps_under_action_sequence():
    # The delay wrapper sits inside ActionSequence, so h counts environment
    # steps rather than policy decisions.
    env = ActionSequence(
        ObservationDelay(DummyEnv(episode_len=9), delay=1),
        sequence_length=3,
    )

    env.reset()
    observation, _, _, _, _ = env.step(env.action_space.sample())

    # Three env steps executed (o_1..o_3); the delayed stream ends at o_2.
    assert _value(observation) == 2.0


def test_observation_delay_outside_frame_stack_delays_whole_stack():
    env = ObservationDelay(FrameStack(DummyEnv(episode_len=9), 3), delay=2)

    env.reset()
    for _ in range(4):
        observation, _, _, _, _ = env.step(env.action_space.sample())

    # Env is at o_4; the stack is the history as of o_2.
    stack = np.asarray(observation[OBS_NAME_FLAT1])[:, 0]
    np.testing.assert_array_equal(stack, np.asarray([0.0, 1.0, 2.0]))


def test_observation_delay_handles_vector_env_autoreset():
    env = ObservationDelay(
        gym.vector.SyncVectorEnv([lambda: DummyEnv(episode_len=3)]),
        delay=1,
    )

    env.reset()
    seen, finals = [], []
    for _ in range(6):
        observation, _, terminated, _, info = env.step(env.action_space.sample())
        seen.append(_value({OBS_NAME_FLAT1: observation[OBS_NAME_FLAT1][0]}))
        if terminated[0]:
            finals.append(_value(info["final_observation"][0]))

    # Each episode runs o_1..o_3. The terminal observation reported in info is
    # aged by the delay too, and autoreset restarts the buffer from the next
    # episode's o_0 so the delay pads again instead of leaking across episodes.
    assert finals == [2.0, 2.0]
    assert seen == [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]


def test_maybe_delay_observations_is_a_noop_without_config():
    env = DummyEnv()

    assert maybe_delay_observations(env, OmegaConf.create({})) is env
    assert maybe_delay_observations(env, OmegaConf.create({"obs_delay": 0})) is env
    assert isinstance(
        maybe_delay_observations(env, OmegaConf.create({"obs_delay": 3})),
        ObservationDelay,
    )
