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
    padding="edge",
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
    extra_replay_elements = spaces.Dict(
        {"demo": spaces.Box(0, 1, shape=(), dtype=np.int8)}
    )
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
    )


def _fill(buffer, episode_lengths, rng):
    for episode_index, episode_length in enumerate(episode_lengths):
        for t in range(episode_length):
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
                demo=np.int8((episode_index + t) % 2),
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
    "nstep,frame_stack,include_tp1",
    [
        (1, 1, False),
        (3, 2, True),
        (8, 4, True),
    ],
)
def test_vectorized_sample_bit_equal(tmp_path, nstep, frame_stack, include_tp1):
    rng = np.random.default_rng(7)
    episode_lengths = [nstep, nstep + 1, 13, 29]
    buffer = _build_buffer(
        tmp_path,
        nstep=nstep,
        frame_stack=frame_stack,
        include_tp1=include_tp1,
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
        tmp_path, nstep=8, frame_stack=4, include_tp1=True
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
       
        padding="zero",
    )
    _fill(buffer, [13, 29], rng)
    assert not buffer._vectorized_sample_supported()
    batch = buffer.sample(16)
    assert batch["action"].shape == (16, 16, 4)


def test_scalar_sample_env_kill_switch(tmp_path, monkeypatch):
    rng = np.random.default_rng(17)
    buffer = _build_buffer(
        tmp_path, nstep=1, frame_stack=1, include_tp1=False
    )
    _fill(buffer, [13], rng)
    assert buffer._vectorized_sample_supported()
    monkeypatch.setenv("ROBOBASE_SCALAR_SAMPLE", "1")
    assert not buffer._vectorized_sample_supported()
