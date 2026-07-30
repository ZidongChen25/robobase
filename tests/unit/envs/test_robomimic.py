import json
import logging

import numpy as np
import pytest
from omegaconf import OmegaConf

h5py = pytest.importorskip("h5py")

from robobase.envs.robomimic import (
    RobomimicEnvFactory,
    _abs_action_to_raw_action,
    _raw_action_to_abs_action,
)
from robobase.envs.wrappers import RecedingHorizonControl


def _write_dataset(path):
    with h5py.File(path, "w") as dataset_file:
        data_group = dataset_file.create_group("data")
        data_group.attrs["env_args"] = json.dumps(
            {
                "env_name": "Lift",
                "env_version": "1.4.1",
                "type": 1,
                "env_kwargs": {
                    "robots": ["Panda"],
                    "controller_configs": {"type": "OSC_POSE"},
                    "reward_shaping": False,
                },
            }
        )

        for episode_idx, length in enumerate((3, 5)):
            episode = data_group.create_group(f"demo_{episode_idx}")
            episode.attrs["num_samples"] = length
            episode.create_dataset(
                "actions",
                data=np.full((length, 7), episode_idx, dtype=np.float32),
            )
            rewards = np.zeros(length, dtype=np.float32)
            rewards[-1] = 1.0
            episode.create_dataset("rewards", data=rewards)
            dones = np.zeros(length, dtype=np.uint8)
            dones[-1] = 1
            episode.create_dataset("dones", data=dones)

            obs = episode.create_group("obs")
            next_obs = episode.create_group("next_obs")
            base = np.arange(length * 3, dtype=np.float32).reshape(length, 3)
            obs.create_dataset("robot0_eef_pos", data=base)
            next_obs.create_dataset("robot0_eef_pos", data=base + 1)
            obj = np.arange(length * 10, dtype=np.float32).reshape(length, 10)
            obs.create_dataset("object", data=obj)
            next_obs.create_dataset("object", data=obj + 1)

        mask_group = dataset_file.create_group("mask")
        mask_group.create_dataset("train", data=np.asarray([b"demo_0"]))
        mask_group.create_dataset("valid", data=np.asarray([b"demo_1"]))


def _write_image_dataset(path, image_shape=(84, 84)):
    with h5py.File(path, "w") as dataset_file:
        data_group = dataset_file.create_group("data")
        data_group.attrs["env_args"] = json.dumps(
            {
                "env_name": "ToolHang",
                "env_version": "1.4.1",
                "type": 1,
                "env_kwargs": {
                    "robots": ["Panda"],
                    "controller_configs": {"type": "OSC_POSE"},
                    "reward_shaping": False,
                },
            }
        )

        episode = data_group.create_group("demo_0")
        length = 3
        episode.attrs["num_samples"] = length
        episode.create_dataset("actions", data=np.zeros((length, 7), dtype=np.float32))
        rewards = np.zeros(length, dtype=np.float32)
        rewards[-1] = 1.0
        episode.create_dataset("rewards", data=rewards)
        dones = np.zeros(length, dtype=np.uint8)
        dones[-1] = 1
        episode.create_dataset("dones", data=dones)

        obs = episode.create_group("obs")
        next_obs = episode.create_group("next_obs")
        image_height, image_width = image_shape
        obs.create_dataset(
            "robot0_eef_pos", data=np.arange(length * 3, dtype=np.float32).reshape(length, 3)
        )
        next_obs.create_dataset(
            "robot0_eef_pos",
            data=np.arange(length * 3, dtype=np.float32).reshape(length, 3) + 1,
        )
        obs.create_dataset(
            "sideview_image",
            data=np.full((length, image_height, image_width, 3), 64, dtype=np.uint8),
        )
        next_obs.create_dataset(
            "sideview_image",
            data=np.full((length, image_height, image_width, 3), 65, dtype=np.uint8),
        )
        obs.create_dataset(
            "robot0_eye_in_hand_image",
            data=np.full((length, image_height, image_width, 3), 128, dtype=np.uint8),
        )
        next_obs.create_dataset(
            "robot0_eye_in_hand_image",
            data=np.full((length, image_height, image_width, 3), 129, dtype=np.uint8),
        )

        mask_group = dataset_file.create_group("mask")
        mask_group.create_dataset("train", data=np.asarray([b"demo_0"]))


def _write_toolhang_abs_dataset(path):
    with h5py.File(path, "w") as dataset_file:
        data_group = dataset_file.create_group("data")
        data_group.attrs["env_args"] = json.dumps(
            {
                "env_name": "ToolHang",
                "env_version": "1.4.1",
                "type": 1,
                "env_kwargs": {
                    "robots": ["Panda"],
                    "controller_configs": {"type": "OSC_POSE", "control_delta": True},
                    "reward_shaping": False,
                },
            }
        )
        episode = data_group.create_group("demo_0")
        length = 4
        episode.attrs["num_samples"] = length
        actions = np.asarray(
            [
                [-0.1, 0.0, 1.0, 0.1, 0.2, 0.3, -1.0],
                [-0.05, 0.01, 1.02, 0.2, 0.1, 0.3, -1.0],
                [0.0, 0.02, 1.04, 0.3, 0.1, 0.2, 1.0],
                [0.05, 0.03, 1.06, 0.4, 0.2, 0.1, 1.0],
            ],
            dtype=np.float32,
        )
        episode.create_dataset("actions", data=actions)
        rewards = np.zeros(length, dtype=np.float32)
        rewards[-1] = 1.0
        episode.create_dataset("rewards", data=rewards)
        dones = np.zeros(length, dtype=np.uint8)
        dones[-1] = 1
        episode.create_dataset("dones", data=dones)

        obs = episode.create_group("obs")
        next_obs = episode.create_group("next_obs")
        obs.create_dataset(
            "object",
            data=np.arange(length * 44, dtype=np.float32).reshape(length, 44),
        )
        obs.create_dataset(
            "robot0_eef_pos",
            data=np.arange(length * 3, dtype=np.float32).reshape(length, 3) / 10.0,
        )
        obs.create_dataset(
            "robot0_eef_quat",
            data=np.tile(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (length, 1)),
        )
        obs.create_dataset(
            "robot0_gripper_qpos",
            data=np.arange(length * 2, dtype=np.float32).reshape(length, 2) / 10.0,
        )
        obs.create_dataset(
            "robot0_joint_pos",
            data=np.ones((length, 7), dtype=np.float32),
        )
        for key in obs.keys():
            next_obs.create_dataset(key, data=np.asarray(obs[key]) + 1.0)


def _make_cfg(dataset_path):
    return OmegaConf.create(
        {
            "pixels": False,
            "frame_stack": 2,
            "action_sequence": 3,
            "execution_length": 3,
            "demos": 0,
            "use_standardization": False,
            "use_min_max_normalization": False,
            "min_max_margin": 0.0,
            "norm_obs": False,
            "temporal_ensemble": True,
            "temporal_ensemble_gain": 0.01,
            "log_eval_video": False,
            "use_onehot_time_and_no_bootstrap": False,
            "visual_observation_shape": [84, 84],
            "num_train_envs": 1,
            "num_eval_envs": 1,
            "env": {
                "env_name": "robomimic",
                "task_name": "",
                "dataset_path": str(dataset_path),
                "episode_length": 0,
                "filter_key": "train",
                "random_traj": False,
                "use_live_env": False,
                "cameras": [],
                "render_gpu_device_id": -1,
            },
        }
    )


def _make_image_cfg(dataset_path):
    cfg = _make_cfg(dataset_path)
    cfg.pixels = True
    cfg.env.task_name = ""
    return cfg


def test_collect_demo_and_infer_spaces(tmp_path):
    dataset_path = tmp_path / "robomimic_test.hdf5"
    _write_dataset(dataset_path)
    cfg = _make_cfg(dataset_path)

    factory = RobomimicEnvFactory()
    factory.collect_or_fetch_demos(cfg, float("inf"))

    assert cfg.env.task_name == "Lift"
    assert cfg.env.episode_length == 3
    assert len(factory._raw_demos) == 1

    demo = factory._raw_demos[0]
    assert len(demo[0]) == 2
    assert len(demo[1]) == 5
    assert demo[0][1]["demo"] == 1
    assert "demo_action" in demo[1][-1]
    assert not hasattr(factory, "_obs_stats")
    assert not hasattr(factory, "_action_stats")


def test_collect_demo_only_computes_stats_when_requested(tmp_path):
    dataset_path = tmp_path / "robomimic_test.hdf5"
    _write_dataset(dataset_path)
    cfg = _make_cfg(dataset_path)
    cfg.use_standardization = True
    cfg.norm_obs = True

    factory = RobomimicEnvFactory()
    factory.collect_or_fetch_demos(cfg, float("inf"))

    assert hasattr(factory, "_action_stats")
    assert hasattr(factory, "_obs_stats")


def test_placeholder_eval_env_shapes(tmp_path):
    dataset_path = tmp_path / "robomimic_test.hdf5"
    _write_dataset(dataset_path)
    cfg = _make_cfg(dataset_path)

    factory = RobomimicEnvFactory()
    factory.collect_or_fetch_demos(cfg, 1)
    factory.post_collect_or_fetch_demos(cfg)
    env = factory.make_eval_env(cfg)

    obs, info = env.reset()
    assert info["demo"] == 0
    assert "low_dim_state" in obs
    assert obs["low_dim_state"].shape == (2, 13)
    assert env.action_space.shape == (3, 7)

    next_obs, reward, terminated, truncated, step_info = env.step(
        np.zeros(env.action_space.shape, dtype=np.float32)
    )
    assert reward == 0.0
    assert not terminated
    assert truncated
    assert next_obs["low_dim_state"].shape == (2, 13)
    assert step_info["placeholder_env"] == 1
    assert step_info["action_sequence_mask"].tolist() == [1, 0, 0]


class _FakeReplayBuffer:
    def __init__(self):
        self.sequential = False
        self.transitions = []
        self.final_observations = []

    def add(self, obs, action, rew, term, trunc, demo):
        self.transitions.append((obs, action, rew, term, trunc, demo))

    def add_final(self, obs):
        self.final_observations.append(obs)


def test_load_demos_into_replay_uses_single_step_storage(tmp_path):
    dataset_path = tmp_path / "robomimic_test.hdf5"
    _write_dataset(dataset_path)
    cfg = _make_cfg(dataset_path)

    factory = RobomimicEnvFactory()
    factory.collect_or_fetch_demos(cfg, 1)
    factory.post_collect_or_fetch_demos(cfg)
    buffer = _FakeReplayBuffer()
    factory.load_demos_into_replay(cfg, buffer, is_demo_buffer=True)

    assert len(buffer.transitions) == 3
    first_obs, first_action, *_ = buffer.transitions[0]
    assert first_obs["low_dim_state"].shape == (13,)
    assert first_action.shape == (7,)
    assert len(buffer.final_observations) == 1
    assert buffer.final_observations[0]["low_dim_state"].shape == (13,)


def test_image_dataset_infers_matching_cameras_for_demos(tmp_path):
    dataset_path = tmp_path / "robomimic_image_test.hdf5"
    _write_image_dataset(dataset_path)
    cfg = _make_image_cfg(dataset_path)

    factory = RobomimicEnvFactory()
    factory.collect_or_fetch_demos(cfg, 1)
    factory.post_collect_or_fetch_demos(cfg)

    assert cfg.env.task_name == "ToolHang"
    assert list(cfg.env.cameras) == ["robot0_eye_in_hand", "sideview"]

    buffer = _FakeReplayBuffer()
    factory.load_demos_into_replay(cfg, buffer, is_demo_buffer=True)

    assert len(buffer.transitions) == 3
    first_obs, *_ = buffer.transitions[0]
    assert "agentview_rgb" not in first_obs
    assert "robot0_eye_in_hand_rgb" in first_obs
    assert "sideview_rgb" in first_obs


def test_image_dataset_warns_for_explicit_camera_mismatch(tmp_path, caplog):
    dataset_path = tmp_path / "robomimic_image_test.hdf5"
    _write_image_dataset(dataset_path)
    cfg = _make_image_cfg(dataset_path)
    cfg.env.cameras = ["agentview"]

    factory = RobomimicEnvFactory()
    with caplog.at_level(logging.WARNING):
        factory.collect_or_fetch_demos(cfg, 1)

    assert list(cfg.env.cameras) == ["robot0_eye_in_hand", "sideview"]
    assert (
        "do not match cfg.env.cameras=['agentview']. Falling back to dataset cameras."
        in caplog.text
    )


def test_image_dataset_resizes_to_visual_observation_shape(tmp_path):
    dataset_path = tmp_path / "robomimic_image_test.hdf5"
    _write_image_dataset(dataset_path, image_shape=(240, 240))
    cfg = _make_image_cfg(dataset_path)

    factory = RobomimicEnvFactory()
    factory.collect_or_fetch_demos(cfg, 1)
    factory.post_collect_or_fetch_demos(cfg)

    buffer = _FakeReplayBuffer()
    factory.load_demos_into_replay(cfg, buffer, is_demo_buffer=True)
    first_obs, *_ = buffer.transitions[0]

    assert first_obs["robot0_eye_in_hand_rgb"].shape == (3, 84, 84)
    assert first_obs["sideview_rgb"].shape == (3, 84, 84)

    env = factory.make_eval_env(cfg)
    try:
        obs, _ = env.reset()
        assert obs["robot0_eye_in_hand_rgb"].shape == (2, 3, 84, 84)
        assert obs["sideview_rgb"].shape == (2, 3, 84, 84)
    finally:
        env.close()


def test_make_eval_envs_supports_vectorized_placeholder_eval(tmp_path):
    dataset_path = tmp_path / "robomimic_test.hdf5"
    _write_dataset(dataset_path)
    cfg = _make_cfg(dataset_path)
    cfg.num_eval_envs = 3

    factory = RobomimicEnvFactory()
    factory.collect_or_fetch_demos(cfg, 1)
    factory.post_collect_or_fetch_demos(cfg)
    envs = factory.make_eval_envs(cfg)

    try:
        assert envs.num_envs == 3
        obs, info = envs.reset()
        assert obs["low_dim_state"].shape == (3, 2, 13)
    finally:
        envs.close()


def test_postprocess_demos_supports_min_max_norm_and_rhc(tmp_path):
    dataset_path = tmp_path / "robomimic_test.hdf5"
    _write_dataset(dataset_path)
    cfg = _make_cfg(dataset_path)
    cfg.action_sequence = 4
    cfg.execution_length = 2
    cfg.use_min_max_normalization = True
    cfg.use_standardization = False
    cfg.min_max_margin = 0.0
    cfg.norm_obs = True
    cfg.demos = 1

    factory = RobomimicEnvFactory()
    factory.collect_or_fetch_demos(cfg, 1)

    raw_demo_action = factory._raw_demos[0][1][-1]["demo_action"].copy()
    factory.post_collect_or_fetch_demos(cfg)
    scaled_demo_action = factory._demos[0][1][-1]["demo_action"]

    assert np.all(scaled_demo_action <= 1.0)
    assert np.all(scaled_demo_action >= -1.0)
    assert not np.allclose(raw_demo_action, scaled_demo_action)

    env = factory.make_eval_env(cfg)
    try:
        current_env = env
        found_rhc = False
        while hasattr(current_env, "env"):
            if isinstance(current_env, RecedingHorizonControl):
                found_rhc = True
                break
            current_env = current_env.env
        assert found_rhc

        obs, _ = env.reset()
        assert obs["low_dim_state"].shape == (2, 13)
        _, _, _, truncated, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        assert truncated
        assert info["action_sequence_mask"].tolist() == [1, 0, 0, 0]
    finally:
        env.close()


def test_toolhang_clean_diffuser_lowdim_abs_parity(tmp_path):
    dataset_path = tmp_path / "toolhang_low_dim_abs.hdf5"
    _write_toolhang_abs_dataset(dataset_path)
    cfg = _make_cfg(dataset_path)
    cfg.env.filter_key = "all"
    cfg.env.obs_keys = [
        "object",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    ]
    cfg.env.abs_action = True
    cfg.env.episode_length = 700
    cfg.action_sequence = 3
    cfg.execution_length = 2
    cfg.action_execution_start = 1
    cfg.temporal_ensemble = False
    cfg.demos = 1
    cfg.use_min_max_normalization = True
    cfg.norm_obs = True
    cfg.obs_norm_type = "min_max"

    factory = RobomimicEnvFactory()
    factory.collect_or_fetch_demos(cfg, float("inf"))
    factory.post_collect_or_fetch_demos(cfg)

    raw_action = factory._raw_demos[0][1][-1]["demo_action"]
    assert raw_action.shape == (10,)
    raw7 = np.asarray([-0.1, 0.0, 1.0, 0.1, 0.2, 0.3, -1.0], dtype=np.float32)
    np.testing.assert_allclose(
        _abs_action_to_raw_action(_raw_action_to_abs_action(raw7)),
        raw7,
        atol=1e-6,
    )

    buffer = _FakeReplayBuffer()
    factory.load_demos_into_replay(cfg, buffer, is_demo_buffer=True)
    first_obs, first_action, *_ = buffer.transitions[0]
    assert first_obs["low_dim_state"].shape == (53,)
    assert first_action.shape == (10,)
    assert np.all(first_action <= 1.0)
    assert np.all(first_action >= -1.0)

    env = factory.make_eval_env(cfg)
    try:
        obs, _ = env.reset()
        assert obs["low_dim_state"].shape == (2, 53)
        assert env.action_space.shape == (3, 10)
        assert np.all(obs["low_dim_state"] <= 1.0)
        assert np.all(obs["low_dim_state"] >= -1.0)
    finally:
        env.close()
