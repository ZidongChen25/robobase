import numpy as np
from gymnasium import spaces

from robobase.replay_buffer.uniform_replay_buffer import UniformReplayBuffer


def _make_buffer(tmp_path, *, transition_uniform_sampling, nstep=5):
    return UniformReplayBuffer(
        batch_size=2,
        replay_capacity=1024,
        nstep=nstep,
        gamma=0.99,
        action_shape=(nstep, 1),
        action_dtype=np.float32,
        reward_shape=(),
        reward_dtype=np.float32,
        observation_elements=spaces.Dict(
            {
                "low_dim_state": spaces.Box(
                    -np.inf, np.inf, shape=(1, 1), dtype=np.float32
                )
            }
        ),
        extra_replay_elements=spaces.Dict({}),
        save_dir=str(tmp_path),
        purge_replay_on_shutdown=False,
        save_snapshot=True,
        num_workers=0,
        transition_uniform_sampling=transition_uniform_sampling,
    )


def _add_episode(replay, length, offset):
    for index in range(length):
        replay.add(
            {"low_dim_state": np.asarray([offset + index], dtype=np.float32)},
            np.asarray([0.0], dtype=np.float32),
            np.float32(0.0),
            terminal=index == length - 1,
            truncated=False,
        )
    replay.add_final({"low_dim_state": np.asarray([offset + length], dtype=np.float32)})


def _count_short_episode_draws(replay, short_len, num_draws):
    np.random.seed(0)
    short_draws = 0
    for _ in range(num_draws):
        sample = replay.sample_single()
        if sample["indices"] < short_len:
            short_draws += 1
    return short_draws


def test_transition_uniform_sampling_weights_episodes_by_valid_starts(tmp_path):
    # Episode A: 6 steps -> 2 valid 5-step chunk starts.
    # Episode B: 54 steps -> 50 valid starts. Flat-buffer sampling should pick
    # A with probability ~2/52.
    replay = _make_buffer(tmp_path, transition_uniform_sampling=True)
    _add_episode(replay, 6, offset=0)
    _add_episode(replay, 54, offset=100)

    short_draws = _count_short_episode_draws(replay, short_len=6, num_draws=2000)
    # Expectation ~77; a generous band still rules out episode-uniform (~1000).
    assert 25 <= short_draws <= 160


def test_default_sampling_remains_episode_uniform(tmp_path):
    replay = _make_buffer(tmp_path, transition_uniform_sampling=False)
    _add_episode(replay, 6, offset=0)
    _add_episode(replay, 54, offset=100)

    short_draws = _count_short_episode_draws(replay, short_len=6, num_draws=2000)
    # Episode-uniform expectation ~1000.
    assert short_draws >= 800
