"""``chunk_tail_starts``: chunks may start in an episode's last ``nstep - 1`` steps.

The n-step window is truncated at the episode end, and the edge-repeated
actions that fill the chunk must be flagged in ``action_pad_mask`` so they are
never used as labels (BC) or as complete chunks (critic).
"""

import numpy as np
import pytest
from gymnasium import spaces

from robobase.replay_buffer.uniform_replay_buffer import (
    ACTION,
    ACTION_PAD_MASK,
    DISCOUNT,
    UniformReplayBuffer,
)

GAMMA = 0.99
NSTEP = 5


def _build_buffer(tmp_path):
    return UniformReplayBuffer(
        batch_size=64,
        replay_capacity=1_000,
        nstep=NSTEP,
        gamma=GAMMA,
        action_shape=(NSTEP, 2),
        action_dtype=np.float32,
        reward_shape=(),
        reward_dtype=np.float32,
        observation_elements=spaces.Dict(
            {
                "low_dim_state": spaces.Box(
                    -np.inf, np.inf, shape=(1, 3), dtype=np.float32
                )
            }
        ),
        extra_replay_elements=spaces.Dict(
            {"demo": spaces.Box(0, 1, shape=(), dtype=np.int8)}
        ),
        save_dir=str(tmp_path / "buf"),
        purge_replay_on_shutdown=False,
        save_snapshot=True,
        num_workers=0,
        fetch_every=17,
        action_padding="edge",
        include_tp1=True,
        chunk_tail_starts=True,
    )


def _fill(buffer, episode_lengths, rng):
    for length in episode_lengths:
        for t in range(length):
            buffer.add(
                {"low_dim_state": rng.standard_normal(3).astype(np.float32)},
                # Unique per (episode, step) so repeated padding is detectable.
                np.float32([100 * length + t, -(100 * length + t)]),
                np.float32(t == length - 1),
                terminal=t == length - 1,
                truncated=False,
                demo=np.int8(1),
            )
        buffer.add_final(
            {"low_dim_state": rng.standard_normal(3).astype(np.float32)}
        )


def test_tail_start_padding_is_masked(tmp_path):
    buffer = _build_buffer(tmp_path)
    _fill(buffer, [1, 3, 7, 13], np.random.default_rng(0))
    assert buffer._vectorized_sample_supported()

    np.random.seed(5)
    batch = buffer.sample(512)
    mask = batch[ACTION_PAD_MASK]
    n_eff = np.rint(np.log(batch[DISCOUNT]) / np.log(GAMMA)).astype(int)
    assert set(n_eff.tolist()) == set(range(1, NSTEP + 1))

    valid = ~mask
    # Valid steps form a prefix whose length is the truncated n-step window.
    np.testing.assert_array_equal(valid.sum(1), n_eff)
    np.testing.assert_array_equal(valid, np.arange(NSTEP)[None] < n_eff[:, None])
    # Every masked step is an edge repeat of the last real action.
    actions = batch[ACTION][..., 0]
    last_real = actions[np.arange(len(n_eff)), n_eff - 1]
    np.testing.assert_array_equal(
        np.where(mask, actions, last_real[:, None]),
        np.broadcast_to(last_real[:, None], actions.shape),
    )
    # Real actions are distinct consecutive steps, never repeats.
    steps = np.diff(actions, axis=1)
    assert np.all(steps[valid[:, 1:]] == 1)


@pytest.mark.parametrize("seed", [3, 11])
def test_tail_start_vectorized_matches_scalar(tmp_path, seed):
    buffer = _build_buffer(tmp_path)
    _fill(buffer, [1, 3, 7, 13], np.random.default_rng(seed))
    buffer.sample(4)  # warm up fetch/cache state before the seeded draws

    np.random.seed(seed)
    reference = [buffer._sample_batch_scalar(64, None) for _ in range(3)]
    np.random.seed(seed)
    vectorized = [buffer.sample(64) for _ in range(3)]
    for ref, vec in zip(reference, vectorized):
        assert set(ref) == set(vec)
        for key in ref:
            assert ref[key].dtype == vec[key].dtype, key
            assert ref[key].tobytes() == vec[key].tobytes(), key
