import json

import numpy as np
import pytest
from omegaconf import OmegaConf

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from robobase.envs.pusht import PushTEnvFactory


def _write_lerobot_pusht_dataset(path, *, pixels=False):
    data_dir = path / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    meta_dir = path / "meta"
    meta_dir.mkdir(parents=True)

    episode_indices = [0, 0, 0, 0, 1, 1, 1]
    frame_indices = [0, 1, 2, 3, 0, 1, 2]
    states = [
        [10.0, 20.0],
        [11.0, 21.0],
        [12.0, 22.0],
        [13.0, 23.0],
        [30.0, 40.0],
        [31.0, 41.0],
        [32.0, 42.0],
    ]
    actions = [
        [100.0, 200.0],
        [101.0, 201.0],
        [102.0, 202.0],
        [103.0, 203.0],
        [300.0, 400.0],
        [301.0, 401.0],
        [302.0, 402.0],
    ]
    dones = [False, False, True, True, False, True, True]
    data = {
        "observation.state": states,
        "action": actions,
        "episode_index": episode_indices,
        "frame_index": frame_indices,
        "timestamp": [float(i) / 10.0 for i in frame_indices],
        "next.reward": [0.1 * (i + 1) for i in range(len(states))],
        "next.done": dones,
        "next.success": [False] * len(states),
        "index": list(range(len(states))),
        "task_index": [0] * len(states),
    }
    if pixels:
        images = []
        for index in range(len(states)):
            image = np.zeros((2, 3, 3), dtype=np.uint8)
            image[..., 0] = index
            image[..., 1] = index + 1
            image[..., 2] = index + 2
            images.append(image.tolist())
        data["observation.image"] = images

    pq.write_table(pa.table(data), data_dir / "file-000.parquet")
    features = {
        "observation.state": {"dtype": "float32", "shape": [2]},
        "action": {"dtype": "float32", "shape": [2]},
    }
    if pixels:
        features["observation.image"] = {
            "dtype": "image",
            "shape": [2, 3, 3],
        }
    (meta_dir / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "total_episodes": 2,
                "total_frames": len(states),
                "fps": 10,
                "splits": {"train": "0:2"},
                "features": features,
            }
        ),
        encoding="utf-8",
    )


def _make_cfg(dataset_path, *, pixels=False, use_live_env=False):
    return OmegaConf.create(
        {
            "pixels": pixels,
            "visual_observation_shape": [2, 3],
            "frame_stack": 2,
            "action_sequence": 3,
            "execution_length": 3,
            "demos": 0,
            "use_standardization": False,
            "use_min_max_normalization": False,
            "min_max_margin": 0.0,
            "norm_obs": False,
            "obs_norm_type": "standardization",
            "temporal_ensemble": True,
            "temporal_ensemble_gain": 0.01,
            "log_eval_video": False,
            "use_onehot_time_and_no_bootstrap": False,
            "num_train_envs": 1,
            "num_eval_envs": 1,
            "seed": 1,
            "env": {
                "env_name": "pusht",
                "task_name": "",
                "repo_id": "lerobot/pusht",
                "dataset_path": str(dataset_path),
                "cache_dir": None,
                "download": False,
                "split": "train",
                "random_traj": False,
                "episode_length": 0,
                "use_live_env": use_live_env,
                "obs_type": "",
                "image_key": "image_rgb",
                "render_mode": "rgb_array",
                "visualization_height": 32,
                "visualization_width": 32,
                "include_environment_state": False,
            },
        }
    )


class _FakeReplayBuffer:
    def __init__(self):
        self.sequential = False
        self.transitions = []
        self.final_observations = []

    def add(self, obs, action, rew, term, trunc, demo):
        self.transitions.append((obs, action, rew, term, trunc, demo))

    def add_final(self, obs):
        self.final_observations.append(obs)


def test_collect_demo_and_infer_spaces_without_early_done_truncation(tmp_path):
    _write_lerobot_pusht_dataset(tmp_path)
    cfg = _make_cfg(tmp_path)

    factory = PushTEnvFactory()
    factory.collect_or_fetch_demos(cfg, float("inf"))

    assert cfg.env.task_name == "PushT"
    assert cfg.env.episode_length == 300
    assert cfg.env.max_demo_episode_length == 4
    assert len(factory._raw_demos) == 2

    demo = factory._raw_demos[0]
    assert len(demo) == 5
    assert demo[0][1]["demo"] == 1
    assert demo[1][-1]["demo_action"].shape == (2,)
    assert demo[3][2] is False
    assert demo[4][2] is True

    obs_space, action_space = factory.get_spaces(cfg)
    assert obs_space["low_dim_state"].shape == (2, 2)
    assert action_space.shape == (3, 2)


def test_load_demos_into_replay_uses_single_step_storage(tmp_path):
    _write_lerobot_pusht_dataset(tmp_path)
    cfg = _make_cfg(tmp_path)
    cfg.use_min_max_normalization = True
    cfg.norm_obs = True
    cfg.demos = 1

    factory = PushTEnvFactory()
    factory.collect_or_fetch_demos(cfg, 1)
    factory.post_collect_or_fetch_demos(cfg)

    buffer = _FakeReplayBuffer()
    factory.load_demos_into_replay(cfg, buffer, is_demo_buffer=True)

    assert len(buffer.transitions) == 4
    first_obs, first_action, *_ = buffer.transitions[0]
    assert first_obs["low_dim_state"].shape == (2,)
    assert first_action.shape == (2,)
    assert np.all(first_action >= -1.0)
    assert np.all(first_action <= 1.0)
    assert len(buffer.final_observations) == 1
    assert buffer.final_observations[0]["low_dim_state"].shape == (2,)


def test_pixel_dataset_exposes_chw_rgb_observation(tmp_path):
    _write_lerobot_pusht_dataset(tmp_path, pixels=True)
    cfg = _make_cfg(tmp_path, pixels=True)

    factory = PushTEnvFactory()
    factory.collect_or_fetch_demos(cfg, 1)
    factory.post_collect_or_fetch_demos(cfg)

    first_obs = factory._raw_demos[0][0][0]
    assert first_obs["image_rgb"].shape == (3, 2, 3)

    obs_space, action_space = factory.get_spaces(cfg)
    assert obs_space["image_rgb"].shape == (2, 3, 2, 3)
    assert obs_space["low_dim_state"].shape == (2, 2)
    assert action_space.shape == (3, 2)


def test_live_eval_env_uses_gym_pusht(tmp_path):
    pytest.importorskip("gym_pusht")
    _write_lerobot_pusht_dataset(tmp_path)
    cfg = _make_cfg(tmp_path, use_live_env=True)
    cfg.action_sequence = 1
    cfg.execution_length = 1

    factory = PushTEnvFactory()
    env = factory.make_eval_env(cfg)
    try:
        obs, info = env.reset(seed=0)
        assert info["demo"] == 0
        assert obs["low_dim_state"].shape == (2, 2)
        next_obs, reward, terminated, truncated, step_info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        assert next_obs["low_dim_state"].shape == (2, 2)
        assert isinstance(float(reward), float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "task_success" in step_info
    finally:
        env.close()
