import numpy as np
import pytest
from gymnasium import spaces
import os

from robobase.replay_buffer.uniform_replay_buffer import (
    ACTION_TP_AUX,
    ACTION_TP1,
    DISCOUNT_AUX,
    REWARD_AUX,
    TERMINAL_AUX,
    UniformReplayBuffer,
    load_episode,
    save_episode,
)
from robobase.replay_buffer.shared_demo_cache import SharedDemoReplayCache


def test_episode_zip_compression_is_lossless_and_smaller(tmp_path):
    episode = {
        "rgb": np.zeros((32, 84, 84, 3), dtype=np.uint8),
        "action": np.arange(32 * 4, dtype=np.float32).reshape(32, 4),
    }
    plain = tmp_path / "plain.npz"
    compressed = tmp_path / "compressed.npz"

    save_episode(episode, plain)
    save_episode(episode, compressed, compression="zip")

    assert compressed.stat().st_size < plain.stat().st_size
    loaded = load_episode(compressed)
    assert set(loaded) == set(episode)
    for key, expected in episode.items():
        np.testing.assert_array_equal(loaded[key], expected)


def test_episode_compression_rejects_unknown_codec(tmp_path):
    with pytest.raises(ValueError, match="compression"):
        save_episode({"value": np.zeros(1)}, tmp_path / "bad.npz", "zstd")


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


def test_shared_demo_cache_hardlinks_but_run_cleanup_is_isolated(tmp_path):
    producer = _make_buffer(
        tmp_path / "producer", transition_uniform_sampling=True, nstep=1
    )
    _add_episode(producer, 6, offset=10)
    cache = SharedDemoReplayCache(tmp_path / "cache", "test-key")
    with cache.lock():
        cache.publish(
            {
                "all_demos": producer.replay_dir,
                "expert_demos": producer.replay_dir,
            }
        )

    consumer = _make_buffer(
        tmp_path / "consumer", transition_uniform_sampling=True, nstep=1
    )
    linked_count = consumer.seed_from_replay_directory(cache.source("all_demos"))
    assert linked_count == 1
    source = next(cache.source("all_demos").glob("*.npz"))
    linked = next(consumer.replay_dir.glob("*.npz"))
    assert os.stat(source).st_ino == os.stat(linked).st_ino
    assert len(consumer) == len(producer)

    assert consumer.clear_persisted_episodes() == 1
    assert source.is_file()


def test_next_action_sequence_starts_at_bootstrap_state(tmp_path):
    replay = UniformReplayBuffer(
        batch_size=1,
        replay_capacity=128,
        nstep=1,
        gamma=0.99,
        action_shape=(3, 1),
        action_dtype=np.float32,
        reward_shape=(),
        reward_dtype=np.float32,
        observation_elements=spaces.Dict(
            {
                "low_dim_state": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(1, 1),
                    dtype=np.float32,
                )
            }
        ),
        extra_replay_elements=spaces.Dict({}),
        save_dir=str(tmp_path / "next_action"),
        purge_replay_on_shutdown=False,
        save_snapshot=True,
        num_workers=0,
        action_padding="edge",
        include_tp1=True,
        include_next_action=True,
    )
    for index in range(5):
        replay.add(
            {
                "low_dim_state": np.asarray(
                    [float(index)],
                    dtype=np.float32,
                )
            },
            np.asarray([10.0 + index], dtype=np.float32),
            np.float32(index),
            terminal=index == 4,
            truncated=False,
        )
    replay.add_final(
        {
            "low_dim_state": np.asarray(
                [5.0],
                dtype=np.float32,
            )
        }
    )

    sample = replay.sample_single(global_index=1)
    np.testing.assert_array_equal(sample["low_dim_state"], [[1.0]])
    np.testing.assert_array_equal(sample["low_dim_state_tp1"], [[2.0]])
    np.testing.assert_array_equal(sample["action"][:, 0], [11.0, 12.0, 13.0])
    np.testing.assert_array_equal(
        sample[ACTION_TP1][:, 0],
        [12.0, 13.0, 14.0],
    )

    terminal_sample = replay.sample_single(global_index=4)
    np.testing.assert_array_equal(
        terminal_sample["action"][:, 0],
        [14.0, 14.0, 14.0],
    )
    np.testing.assert_array_equal(
        terminal_sample[ACTION_TP1][:, 0],
        [14.0, 14.0, 14.0],
    )


def test_auxiliary_nstep_is_terminal_truncated_from_same_start(tmp_path):
    replay = UniformReplayBuffer(
        batch_size=2,
        replay_capacity=128,
        nstep=1,
        auxiliary_nstep=4,
        gamma=0.5,
        action_shape=(3, 1),
        action_dtype=np.float32,
        reward_shape=(),
        reward_dtype=np.float32,
        observation_elements=spaces.Dict(
            {
                "low_dim_state": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(1, 1),
                    dtype=np.float32,
                )
            }
        ),
        extra_replay_elements=spaces.Dict({}),
        save_dir=str(tmp_path / "auxiliary_nstep"),
        purge_replay_on_shutdown=False,
        save_snapshot=True,
        num_workers=0,
        action_padding="edge",
        include_tp1=True,
        include_next_action=True,
    )
    for index in range(5):
        replay.add(
            {"low_dim_state": np.asarray([float(index)], dtype=np.float32)},
            np.asarray([10.0 + index], dtype=np.float32),
            np.float32(index + 1),
            terminal=index == 4,
            truncated=False,
        )
    replay.add_final(
        {"low_dim_state": np.asarray([5.0], dtype=np.float32)}
    )

    sample = replay.sample_single(global_index=1)
    np.testing.assert_array_equal(sample["low_dim_state"], [[1.0]])
    np.testing.assert_array_equal(sample["low_dim_state_tp1"], [[2.0]])
    np.testing.assert_array_equal(sample["low_dim_state_tp_aux"], [[5.0]])
    np.testing.assert_array_equal(
        sample[ACTION_TP_AUX][:, 0], [14.0, 14.0, 14.0]
    )
    np.testing.assert_allclose(sample[REWARD_AUX], 5.125)
    np.testing.assert_allclose(sample[DISCOUNT_AUX], 0.5**4)
    assert bool(sample[TERMINAL_AUX])

    terminal_sample = replay.sample_single(global_index=4)
    np.testing.assert_array_equal(
        terminal_sample["low_dim_state_tp_aux"], [[5.0]]
    )
    np.testing.assert_allclose(terminal_sample[REWARD_AUX], 5.0)
    np.testing.assert_allclose(terminal_sample[DISCOUNT_AUX], 0.5)
    assert bool(terminal_sample[TERMINAL_AUX])

    explicit = replay.sample_batch_indices([1, 4])
    scalar = replay._sample_batch_scalar(2, [1, 4])
    assert set(explicit) == set(scalar)
    for key in explicit:
        np.testing.assert_array_equal(explicit[key], scalar[key], err_msg=key)
