"""Bit-equality of the vectorized batch assembly against the per-sample path.

The vectorized ``UniformReplayBuffer.sample`` must reproduce the scalar
reference (``_sample_batch_scalar``) exactly: same RNG consumption, same
sampled transitions, same values, dtypes, and shapes, byte for byte.
"""

import numpy as np
import pytest
from gymnasium import spaces

from robobase.replay_buffer.uniform_replay_buffer import (
    INDICES,
    UniformReplayBuffer,
)


def _build_buffer(
    tmp_path,
    *,
    nstep,
    frame_stack,
    include_tp1,
    uniform,
    padding="edge",
    include_next_action=False,
    auxiliary_nstep=None,
    explore_truncate=False,
    batch_size=64,
    tag="",
):
    observation_elements = spaces.Dict(
        {
            "rgb": spaces.Box(
                0, 255, shape=(frame_stack, 3, 8, 8), dtype=np.uint8
            ),
            "low_dim_state": spaces.Box(
                -np.inf, np.inf, shape=(frame_stack, 5), dtype=np.float32
            ),
        }
    )
    extra_spaces = {"demo": spaces.Box(0, 1, shape=(), dtype=np.int8)}
    if explore_truncate:
        extra_spaces["explored"] = spaces.Box(0, 1, shape=(), dtype=np.uint8)
    extra_replay_elements = spaces.Dict(extra_spaces)
    return UniformReplayBuffer(
        batch_size=batch_size,
        replay_capacity=10_000,
        nstep=nstep,
        gamma=0.99,
        action_shape=(16, 4),
        action_dtype=np.float32,
        reward_shape=(),
        reward_dtype=np.float32,
        observation_elements=observation_elements,
        extra_replay_elements=extra_replay_elements,
        save_dir=str(tmp_path / f"buf{tag}"),
        purge_replay_on_shutdown=False,
        save_snapshot=True,
        num_workers=0,
        fetch_every=17,
        action_padding=padding,
        include_tp1=include_tp1,
        include_next_action=include_next_action,
        auxiliary_nstep=auxiliary_nstep,
        nstep_explore_truncate=explore_truncate,
        transition_uniform_sampling=uniform,
    )


def _fill(buffer, episode_lengths, rng, explored_flags=False):
    for episode_index, episode_length in enumerate(episode_lengths):
        for t in range(episode_length):
            extra = {"demo": np.int8((episode_index + t) % 2)}
            if explored_flags:
                extra["explored"] = np.uint8(t % 3 == 1)
            buffer.add(
                {
                    "rgb": rng.integers(
                        0, 256, size=(3, 8, 8), dtype=np.uint8
                    ),
                    "low_dim_state": rng.standard_normal(5).astype(np.float32),
                },
                rng.standard_normal(4).astype(np.float32),
                np.float32(rng.standard_normal()),
                terminal=t == episode_length - 1,
                truncated=False,
                **extra,
            )
        buffer.add_final(
            {
                "rgb": rng.integers(0, 256, size=(3, 8, 8), dtype=np.uint8),
                "low_dim_state": rng.standard_normal(5).astype(np.float32),
            }
        )


def _assert_batches_bit_equal(reference, vectorized):
    assert set(reference.keys()) == set(vectorized.keys())
    for key in reference:
        ref, vec = reference[key], vectorized[key]
        assert ref.dtype == vec.dtype, key
        assert ref.shape == vec.shape, key
        assert ref.tobytes() == vec.tobytes(), key


@pytest.mark.parametrize(
    "nstep,frame_stack,include_tp1,uniform,include_next_action,auxiliary_nstep",
    [
        (1, 1, False, False, False, None),
        (3, 2, True, False, False, None),
        (8, 4, True, False, True, None),
        (8, 4, True, True, True, None),
        (1, 4, True, False, True, 4),
        (1, 4, True, True, True, 4),
    ],
)
def test_vectorized_sample_bit_equal(
    tmp_path,
    nstep,
    frame_stack,
    include_tp1,
    uniform,
    include_next_action,
    auxiliary_nstep,
):
    rng = np.random.default_rng(7)
    episode_lengths = [nstep, nstep + 1, 13, 29]
    buffer = _build_buffer(
        tmp_path,
        nstep=nstep,
        frame_stack=frame_stack,
        include_tp1=include_tp1,
        uniform=uniform,
        include_next_action=include_next_action,
        auxiliary_nstep=auxiliary_nstep,
    )
    _fill(buffer, episode_lengths, rng)
    assert buffer._vectorized_sample_supported()

    buffer.sample(4)  # warm up fetch/cache state before the seeded draws

    np.random.seed(0xC0FFEE)
    reference = [buffer._sample_batch_scalar(64, None) for _ in range(3)]
    reference_state = np.random.get_state()

    np.random.seed(0xC0FFEE)
    vectorized = [buffer.sample(64) for _ in range(3)]
    vectorized_state = np.random.get_state()

    for ref, vec in zip(reference, vectorized):
        _assert_batches_bit_equal(ref, vec)
    for ref_part, vec_part in zip(reference_state, vectorized_state):
        np.testing.assert_array_equal(ref_part, vec_part)


def test_vectorized_sample_bit_equal_with_explicit_indices(tmp_path):
    rng = np.random.default_rng(11)
    buffer = _build_buffer(
        tmp_path, nstep=8, frame_stack=4, include_tp1=True, uniform=False
    )
    _fill(buffer, [8, 9, 13, 29], rng)

    np.random.seed(3)
    probe = buffer.sample(32)
    indices = [int(i) for i in probe[INDICES]]

    reference = buffer._sample_batch_scalar(len(indices), list(indices))
    vectorized = buffer.sample(len(indices), list(indices))
    _assert_batches_bit_equal(reference, vectorized)


def test_zero_padding_falls_back_to_scalar_path(tmp_path):
    rng = np.random.default_rng(13)
    buffer = _build_buffer(
        tmp_path,
        nstep=8,
        frame_stack=1,
        include_tp1=True,
        uniform=False,
        padding="zero",
    )
    _fill(buffer, [13, 29], rng)
    assert not buffer._vectorized_sample_supported()
    batch = buffer.sample(16)
    assert batch["action"].shape == (16, 16, 4)


def test_scalar_sample_env_kill_switch(tmp_path, monkeypatch):
    rng = np.random.default_rng(17)
    buffer = _build_buffer(
        tmp_path, nstep=1, frame_stack=1, include_tp1=False, uniform=False
    )
    _fill(buffer, [13], rng)
    assert buffer._vectorized_sample_supported()
    monkeypatch.setenv("ROBOBASE_SCALAR_SAMPLE", "1")
    assert not buffer._vectorized_sample_supported()


@pytest.mark.parametrize("nstep", [3, 8])
def test_vectorized_sample_bit_equal_with_explore_truncation(tmp_path, nstep):
    rng = np.random.default_rng(23)
    buffer = _build_buffer(
        tmp_path,
        nstep=nstep,
        frame_stack=2,
        include_tp1=True,
        uniform=False,
        include_next_action=True,
        explore_truncate=True,
    )
    _fill(buffer, [nstep, 13, 29], rng, explored_flags=True)
    assert buffer._vectorized_sample_supported()

    buffer.sample(4)
    np.random.seed(0xBEEF)
    reference = [buffer._sample_batch_scalar(64, None) for _ in range(3)]
    np.random.seed(0xBEEF)
    vectorized = [buffer.sample(64) for _ in range(3)]
    for ref, vec in zip(reference, vectorized):
        _assert_batches_bit_equal(ref, vec)


def test_explore_truncation_semantics(tmp_path):
    buffer = _build_buffer(
        tmp_path,
        nstep=8,
        frame_stack=1,
        include_tp1=True,
        uniform=False,
        explore_truncate=True,
    )
    rng = np.random.default_rng(29)
    # One 20-step episode with a single explored step at t=4.
    rewards = []
    for t in range(20):
        extra = {"demo": np.int8(0), "explored": np.uint8(t == 4)}
        reward = np.float32(rng.standard_normal())
        rewards.append(float(reward))
        buffer.add(
            {
                "rgb": rng.integers(0, 256, size=(3, 8, 8), dtype=np.uint8),
                "low_dim_state": rng.standard_normal(5).astype(np.float32),
            },
            rng.standard_normal(4).astype(np.float32),
            reward,
            terminal=t == 19,
            truncated=False,
            **extra,
        )
    buffer.add_final(
        {
            "rgb": rng.integers(0, 256, size=(3, 8, 8), dtype=np.uint8),
            "low_dim_state": rng.standard_normal(5).astype(np.float32),
        }
    )
    buffer.sample(2)  # trigger fetch

    # Window starting at idx=1 must truncate at the explored step t=4:
    # n_eff = 3, summing rewards[1:4], discount gamma**3.
    sample = buffer._sample_batch_scalar(1, [1])
    gamma = 0.99
    expected = np.float32(
        np.sum(
            np.asarray(rewards[1:4], dtype=np.float32)
            * np.asarray(
                [gamma**0, gamma**1, gamma**2], dtype=np.float32
            )
        )
    )
    assert abs(float(sample["reward"][0]) - float(expected)) < 1e-5
    assert abs(float(sample["discount"][0]) - gamma**3) < 1e-12

    # Window starting at idx=4 (explored first action itself): NOT truncated
    # by its own flag; full n unless a later flag appears (none) -> n_eff=8.
    sample = buffer._sample_batch_scalar(1, [4])
    assert abs(float(sample["discount"][0]) - gamma**8) < 1e-12

    # Vectorized path agrees on both.
    vec = buffer.sample(2, [1, 4])
    assert abs(float(vec["discount"][0]) - gamma**3) < 1e-12
    assert abs(float(vec["discount"][1]) - gamma**8) < 1e-12
