import json

import numpy as np
import pytest
from omegaconf import OmegaConf

h5py = pytest.importorskip("h5py")

from robobase.envs.robomimic import RobomimicEnvFactory


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


def _make_cfg(dataset_path):
    return OmegaConf.create(
        {
            "pixels": False,
            "frame_stack": 2,
            "action_sequence": 3,
            "use_onehot_time_and_no_bootstrap": False,
            "visual_observation_shape": [84, 84],
            "num_train_envs": 1,
            "env": {
                "env_name": "robomimic",
                "task_name": "",
                "dataset_path": str(dataset_path),
                "episode_length": 0,
                "filter_key": "train",
                "random_traj": False,
                "use_live_env": False,
                "cameras": ["agentview"],
            },
        }
    )


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
