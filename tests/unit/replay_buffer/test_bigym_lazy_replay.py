from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces
from omegaconf import OmegaConf

from robobase.replay_buffer.bigym_lazy_replay import (
    LazyBiGymEpisode,
    LazyBiGymManifest,
    LazyBiGymReplayBuffer,
    _metadata_dirs,
    lazy_replay_enabled,
)


def test_lazy_replay_explicit_false_disables_imitation_learning():
    cfg = OmegaConf.create(
        {
            "is_imitation_learning": True,
            "pixels": True,
            "demos": 1,
            "env": {"env_name": "bigym"},
            "lazy_replay": {"use": False},
        }
    )

    assert not lazy_replay_enabled(cfg)


def test_lazy_replay_disabled_for_non_bigym_imitation_learning():
    cfg = OmegaConf.create(
        {
            "is_imitation_learning": True,
            "pixels": True,
            "demos": 1,
            "env": {"env_name": "d4rl"},
            "lazy_replay": {"use": "auto"},
        }
    )

    assert not lazy_replay_enabled(cfg)


def test_lazy_replay_disabled_for_non_imitation_learning_even_with_use_flag():
    cfg = OmegaConf.create(
        {
            "is_imitation_learning": False,
            "pixels": True,
            "demos": 1,
            "env": {"env_name": "bigym"},
            "lazy_replay": {"use": True},
        }
    )

    assert not lazy_replay_enabled(cfg)


def test_lazy_replay_auto_enables_pixel_bigym_demos():
    cfg = OmegaConf.create(
        {
            "is_imitation_learning": True,
            "pixels": True,
            "demos": ".inf",
            "env": {"env_name": "bigym"},
            "lazy_replay": {"use": "auto"},
        }
    )

    assert lazy_replay_enabled(cfg)


@pytest.mark.parametrize(
    "pixels,demos",
    [(False, 1), (True, 0), (False, 0)],
)
def test_lazy_replay_auto_requires_pixels_and_demos(pixels, demos):
    cfg = OmegaConf.create(
        {
            "is_imitation_learning": True,
            "pixels": pixels,
            "demos": demos,
            "env": {"env_name": "bigym"},
            "lazy_replay": {"use": "auto"},
        }
    )

    assert not lazy_replay_enabled(cfg)


def test_lazy_replay_explicit_true_forces_supported_bigym_path():
    cfg = OmegaConf.create(
        {
            "is_imitation_learning": True,
            "pixels": False,
            "demos": 0,
            "env": {"env_name": "bigym"},
            "lazy_replay": {"use": True},
        }
    )

    assert lazy_replay_enabled(cfg)


def test_lazy_replay_metadata_dirs_honor_dataset_root(monkeypatch, tmp_path):
    cache_roots = []

    class FakeEnv:
        def close(self):
            pass

    class FakeDemoStore:
        def __init__(self, cache_root=None):
            cache_roots.append(cache_root)

        def _create_path(self, metadata, frequency):
            mode = str(metadata.observation_mode).split(".")[-1].lower()
            return tmp_path / mode / f"{frequency}.safetensors"

    metadata = SimpleNamespace(observation_mode=None)
    cfg = OmegaConf.create(
        {
            "env": {
                "dataset_root": str(tmp_path),
                "demo_down_sample_rate": 25,
            }
        }
    )
    monkeypatch.setattr(
        "robobase.replay_buffer.bigym_lazy_replay._make_metadata_env",
        lambda cfg: FakeEnv(),
    )
    monkeypatch.setattr(
        "robobase.replay_buffer.bigym_lazy_replay.Metadata.from_env",
        lambda env: metadata,
    )
    monkeypatch.setattr(
        "robobase.replay_buffer.bigym_lazy_replay.DemoStore",
        FakeDemoStore,
    )

    pixel_dir, state_dir = _metadata_dirs(cfg)

    assert cache_roots == [tmp_path]
    assert pixel_dir == tmp_path / "pixel"
    assert state_dir == tmp_path / "state"


def test_lazy_replay_metadata_dirs_allow_separate_pixel_and_state_roots(
    monkeypatch, tmp_path
):
    pixel_root = tmp_path / "pixels"
    state_root = tmp_path / "states"
    cache_roots = []

    class FakeEnv:
        def close(self):
            pass

    class FakeDemoStore:
        def __init__(self, cache_root=None):
            self.cache_root = cache_root
            cache_roots.append(cache_root)

        def _create_path(self, metadata, frequency):
            mode = str(metadata.observation_mode).split(".")[-1].lower()
            return self.cache_root / mode / f"{frequency}.safetensors"

    cfg = OmegaConf.create(
        {
            "env": {
                "dataset_root": "",
                "pixel_dataset_root": str(pixel_root),
                "state_dataset_root": str(state_root),
                "demo_down_sample_rate": 25,
            }
        }
    )
    monkeypatch.setattr(
        "robobase.replay_buffer.bigym_lazy_replay._make_metadata_env",
        lambda cfg: FakeEnv(),
    )
    monkeypatch.setattr(
        "robobase.replay_buffer.bigym_lazy_replay.Metadata.from_env",
        lambda env: SimpleNamespace(observation_mode=None),
    )
    monkeypatch.setattr(
        "robobase.replay_buffer.bigym_lazy_replay.DemoStore",
        FakeDemoStore,
    )

    pixel_dir, state_dir = _metadata_dirs(cfg)

    assert cache_roots == [pixel_root, state_root]
    assert pixel_dir == pixel_root / "pixel"
    assert state_dir == state_root / "state"


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
            "visual_observation_shape": [1, 1],
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
    buffer._lang_features = np.zeros((1, 512), dtype=np.float32)
    buffer._camera_param_cameras = ()
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


def test_lazy_bigym_returns_camera_params_for_model_side_plucker_generation():
    buffer = _make_lazy_buffer("pre_action")
    intrinsics = np.tile(np.eye(3, dtype=np.float32), (5, 1, 1))
    c2ws = np.tile(np.eye(4, dtype=np.float32), (5, 1, 1))
    c2ws[:, 0, 3] = np.arange(5, dtype=np.float32)

    buffer.cfg.env.cameras = ["head"]
    buffer.observation_elements["camera_intrinsic_head"] = spaces.Box(
        -np.inf,
        np.inf,
        shape=(1, 3, 3),
        dtype=np.float32,
    )
    buffer.observation_elements["camera_c2w_head"] = spaces.Box(
        -np.inf,
        np.inf,
        shape=(1, 4, 4),
        dtype=np.float32,
    )
    buffer._camera_param_cameras = ("head",)
    episode = buffer._cached_episode(0)
    episode["obs_camera_intrinsic_head"] = intrinsics
    episode["obs_camera_c2w_head"] = c2ws

    batch = buffer.sample_batch_indices([0, 2])

    assert batch["camera_intrinsic_head"].shape == (2, 1, 3, 3)
    assert batch["camera_c2w_head"].shape == (2, 1, 4, 4)
    np.testing.assert_allclose(batch["camera_intrinsic_head"][0, 0], intrinsics[0])
    np.testing.assert_allclose(batch["camera_c2w_head"][0, 0], c2ws[0])
    np.testing.assert_allclose(batch["camera_intrinsic_head"][1, 0], intrinsics[2])
    np.testing.assert_allclose(batch["camera_c2w_head"][1, 0], c2ws[2])


def test_lazy_bigym_returns_precomputed_lang_features():
    buffer = _make_lazy_buffer("pre_action")
    buffer.observation_elements["lang_features"] = spaces.Box(
        -np.inf,
        np.inf,
        shape=(1, 512),
        dtype=np.float32,
    )
    buffer._lang_features = np.full((1, 512), 0.125, dtype=np.float32)

    batch = buffer.sample_batch_indices([0, 2])

    assert batch["lang_features"].shape == (2, 1, 512)
    np.testing.assert_allclose(batch["lang_features"], 0.125)


def test_lazy_bigym_missing_language_source_preserves_legacy_clip_abi(monkeypatch):
    buffer = object.__new__(LazyBiGymReplayBuffer)
    buffer.cfg = OmegaConf.create(
        {"env": {"task_name": "flip_cutlery"}, "method": {}}
    )
    buffer.observation_elements = {
        "lang_tokens": spaces.Box(0, 100, shape=(1, 77), dtype=np.int32),
        "lang_features": spaces.Box(
            -np.inf,
            np.inf,
            shape=(1, 512),
            dtype=np.float32,
        ),
    }
    descriptions = []
    monkeypatch.setattr(
        "robobase.replay_buffer.bigym_lazy_replay.clip_tokenize_text",
        lambda description: (
            descriptions.append(description)
            or np.full((1, 77), 7, dtype=np.int32)
        ),
    )
    monkeypatch.setattr(
        "robobase.replay_buffer.bigym_lazy_replay.clip_text_feature_array",
        lambda description, device="cpu": (
            descriptions.append(description)
            or np.full((1, 512), 0.25, dtype=np.float32)
        ),
    )

    tokens = buffer._build_lang_tokens()
    buffer._lang_tokens = tokens
    features = buffer._build_lang_features()

    assert buffer._lang_feature_source() == "clip"
    assert descriptions == ["reach the target", "reach the target"]
    np.testing.assert_array_equal(tokens, np.full((1, 77), 7))
    np.testing.assert_allclose(features, 0.25)


def test_lazy_bigym_returns_time_onehot_with_frame_stack_and_tp1():
    buffer = _make_lazy_buffer("pre_action")
    buffer.cfg.frame_stack = 3
    buffer._frame_stacks = 3
    buffer._include_tp1 = True
    buffer.observation_elements["low_dim_state"] = spaces.Box(
        -np.inf,
        np.inf,
        shape=(3, 2),
        dtype=np.float32,
    )
    buffer.observation_elements["time"] = spaces.Box(
        0,
        1,
        shape=(3, 7),
        dtype=np.uint8,
    )

    batch = buffer.sample_batch_indices([0, 2])
    eye = np.eye(7, dtype=np.uint8)

    np.testing.assert_array_equal(batch["time"][0], eye[[0, 0, 0]])
    np.testing.assert_array_equal(batch["time_tp1"][0], eye[[0, 0, 1]])
    np.testing.assert_array_equal(batch["time"][1], eye[[0, 1, 2]])
    np.testing.assert_array_equal(batch["time_tp1"][1], eye[[1, 2, 3]])


def test_lazy_bigym_post_action_time_starts_after_first_action():
    buffer = _make_lazy_buffer("post_action")
    buffer.observation_elements["time"] = spaces.Box(
        0,
        1,
        shape=(1, 7),
        dtype=np.uint8,
    )

    batch = buffer.sample_batch_indices([0])

    np.testing.assert_array_equal(batch["time"][0, 0], np.eye(7, dtype=np.uint8)[1])


def test_lazy_bigym_rejects_unknown_observation_keys():
    cfg = OmegaConf.create({"env": {"cameras": []}})
    observation_space = spaces.Dict(
        {
            "unknown": spaces.Box(
                -np.inf,
                np.inf,
                shape=(1, 2),
                dtype=np.float32,
            )
        }
    )
    action_space = spaces.Box(-1.0, 1.0, shape=(1, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="does not know how to populate"):
        LazyBiGymReplayBuffer(
            cfg,
            observation_space,
            action_space,
            batch_size=1,
        )
