from pathlib import Path

import numpy as np
from gymnasium import spaces
from omegaconf import OmegaConf

from robobase.replay_buffer.bigym_lazy_replay import (
    LazyBiGymEpisode,
    LazyBiGymManifest,
    LazyBiGymReplayBuffer,
)


def _make_lazy_buffer(observation_timing: str):
    cfg = OmegaConf.create(
        {
            "frame_stack": 1,
            "norm_obs": False,
            "replay": {
                "nstep": 1,
                "gamma": 0.99,
                "action_sequence_start_offset": 0,
                "action_padding": "zero",
            },
            "lazy_replay": {
                "observation_timing": observation_timing,
                "include_tp1": False,
            },
            "env": {"cameras": []},
        }
    )
    episode = {
        "info_demo_action": np.arange(5, dtype=np.float32).reshape(5, 1),
        "obs_proprioception": np.arange(5, dtype=np.float32).reshape(5, 1),
        "obs_proprioception_grippers": np.zeros((5, 1), dtype=np.float32),
        "reward": np.zeros(5, dtype=np.float32),
        "termination": np.zeros(5, dtype=np.bool_),
        "truncation": np.zeros(5, dtype=np.bool_),
    }

    buffer = LazyBiGymReplayBuffer.__new__(LazyBiGymReplayBuffer)
    buffer.cfg = cfg
    buffer.observation_elements = {
        "low_dim_state": spaces.Box(-np.inf, np.inf, shape=(1, 2), dtype=np.float32)
    }
    buffer._batch_size = 1
    buffer._action_shape = (1,)
    buffer._action_seq_len = 3
    buffer._frame_stacks = 1
    buffer._nstep = 1
    buffer._gamma = 0.99
    buffer._action_sequence_start_offset = 0
    buffer._observation_timing = observation_timing
    buffer._action_index_offset = 1 if observation_timing == "post_action" else 0
    buffer._action_padding = "zero"
    buffer._manifest = LazyBiGymManifest(
        episodes=(),
        action_stats={},
        obs_stats={"mean": {}, "std": {}, "min": {}, "max": {}},
    )
    buffer._episodes = (
        LazyBiGymEpisode(
            uuid="episode",
            pixel_path=Path("episode.safetensors"),
            state_path=Path("episode.safetensors"),
            num_obs=5,
            num_actions=5,
            transition_len=4,
            global_start=0,
            successful=True,
        ),
    )
    buffer._starts = np.asarray([0], dtype=np.int64)
    buffer._ends = np.asarray([4], dtype=np.int64)
    buffer._size = 4
    buffer._include_tp1 = False
    buffer._lang_tokens = np.zeros((1, 77), dtype=np.int32)
    buffer._cached_episode = lambda episode_idx: episode
    buffer._transform_actions = lambda actions: actions.astype(np.float32, copy=False)
    return buffer


def test_lazy_bigym_pre_action_uses_same_action_index():
    buffer = _make_lazy_buffer("pre_action")

    batch = buffer.sample_batch_indices([0])

    np.testing.assert_array_equal(batch["action"][0, :, 0], np.asarray([0, 1, 2]))
    np.testing.assert_array_equal(batch["action_pad_mask"][0], np.asarray([0, 0, 0]))


def test_lazy_bigym_post_action_uses_next_action_index_and_pads_end():
    buffer = _make_lazy_buffer("post_action")

    first = buffer.sample_batch_indices([0])
    last = buffer.sample_batch_indices([3])

    np.testing.assert_array_equal(first["action"][0, :, 0], np.asarray([1, 2, 3]))
    np.testing.assert_array_equal(first["action_pad_mask"][0], np.asarray([0, 0, 0]))
    np.testing.assert_array_equal(last["action"][0, :, 0], np.asarray([4, 0, 0]))
    np.testing.assert_array_equal(last["action_pad_mask"][0], np.asarray([0, 1, 1]))
