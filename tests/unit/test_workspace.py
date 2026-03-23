import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from gymnasium import spaces

from robobase.envs.env import EnvFactory
from robobase.workspace import Workspace

h5py = pytest.importorskip("h5py")


def _write_robomimic_dataset(path: Path):
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


def test_robomimic_pretrain_eval_uses_vector_eval_envs(tmp_path):
    dataset_path = tmp_path / "robomimic_test.hdf5"
    _write_robomimic_dataset(dataset_path)

    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).resolve().parents[2] / "robobase/cfgs"),
        job_name="test_robomimic_parallel_eval",
    ):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "method=bc",
                "env=robomimic",
                "pixels=false",
                "demos=1",
                f"env.dataset_path={dataset_path}",
                "env.use_live_env=false",
                "env.filter_key=train",
                "num_train_envs=1",
                "num_eval_envs=4",
                "num_eval_episodes=5",
                "num_pretrain_steps=2",
                "num_train_frames=0",
                "replay_size_before_train=0",
                "num_gpus=0",
                "batch_size=2",
                "replay.size=128",
                "replay.nstep=1",
                "replay.num_workers=0",
                "replay.pin_memory=false",
                "log_pretrain_every=1",
                "eval_every_steps=1",
                "log_eval_video=false",
                "save_snapshot=false",
                "wandb.use=false",
            ],
        )

    workspace = Workspace(cfg, work_dir=tmp_path)
    assert workspace.eval_envs is not None
    assert workspace.eval_envs.num_envs == 4

    recorded_eval_batch_sizes = []
    original_act = workspace.agent.act

    def wrapped_act(observations, step, eval_mode):
        if eval_mode:
            recorded_eval_batch_sizes.append(
                int(next(iter(observations.values())).shape[0])
            )
        return original_act(observations, step, eval_mode)

    workspace.agent.act = wrapped_act
    train_completed = False
    try:
        workspace.train()
        train_completed = True
    finally:
        if not train_completed:
            workspace.shutdown()
        GlobalHydra.instance().clear()

    assert recorded_eval_batch_sizes
    assert set(recorded_eval_batch_sizes) == {4}


class _TinyEvalEnv(gym.Env):
    def __init__(self, episode_len: int = 2):
        super().__init__()
        self.observation_space = spaces.Dict(
            {
                "low_dim_state": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(1, 4),
                    dtype=np.float32,
                )
            }
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1, 2),
            dtype=np.float32,
        )
        self._episode_len = episode_len
        self._step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        return {"low_dim_state": np.zeros((1, 4), dtype=np.float32)}, {}

    def step(self, action):
        del action
        self._step += 1
        terminated = self._step >= self._episode_len
        obs = {"low_dim_state": np.full((1, 4), self._step, dtype=np.float32)}
        reward = float(terminated)
        info = {"task_success": terminated}
        return obs, reward, terminated, False, info


class _TinyTrainEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Dict(
            {
                "low_dim_state": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(1, 4),
                    dtype=np.float32,
                )
            }
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1, 2),
            dtype=np.float32,
        )
        self._step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        return {"low_dim_state": np.zeros((1, 4), dtype=np.float32)}, {}

    def step(self, action):
        del action
        self._step += 1
        obs = {"low_dim_state": np.full((1, 4), self._step, dtype=np.float32)}
        return obs, 0.0, False, False, {}


class _TinyVectorEvalFactory(EnvFactory):
    def make_train_env(self, cfg):
        raise AssertionError("train env should not be created for this test")

    def make_eval_env(self, cfg):
        del cfg
        return _TinyEvalEnv()

    def make_eval_envs(self, cfg):
        return gym.vector.SyncVectorEnv(
            [lambda: _TinyEvalEnv() for _ in range(cfg.num_eval_envs)]
        )


class _TinyTrainAndEvalFactory(EnvFactory):
    def make_train_env(self, cfg):
        return gym.vector.SyncVectorEnv(
            [lambda: _TinyTrainEnv() for _ in range(cfg.num_train_envs)]
        )

    def make_eval_env(self, cfg):
        del cfg
        return _TinyEvalEnv()

    def make_eval_envs(self, cfg):
        return gym.vector.SyncVectorEnv(
            [lambda: _TinyEvalEnv() for _ in range(cfg.num_eval_envs)]
        )


def test_generic_env_factory_can_use_vector_eval_envs(tmp_path):
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).resolve().parents[2] / "robobase/cfgs"),
        job_name="test_generic_parallel_eval",
    ):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "method=bc",
                "env=robomimic",
                "pixels=false",
                "demos=0",
                "num_train_envs=0",
                "num_eval_envs=3",
                "num_eval_episodes=3",
                "num_pretrain_steps=0",
                "num_train_frames=0",
                "replay_size_before_train=0",
                "num_gpus=0",
                "batch_size=2",
                "replay.size=16",
                "replay.nstep=1",
                "replay.num_workers=0",
                "replay.pin_memory=false",
                "action_sequence=1",
                "execution_length=1",
                "method.adaptive_lr=false",
                "log_eval_video=false",
                "save_snapshot=false",
                "wandb.use=false",
            ],
        )

    workspace = Workspace(
        cfg,
        env_factory=_TinyVectorEvalFactory(),
        work_dir=tmp_path,
    )
    assert workspace.eval_envs is not None
    assert workspace.eval_envs.num_envs == 3

    recorded_eval_batch_sizes = []
    original_act = workspace.agent.act

    def wrapped_act(observations, step, eval_mode):
        if eval_mode:
            recorded_eval_batch_sizes.append(
                int(next(iter(observations.values())).shape[0])
            )
        return original_act(observations, step, eval_mode)

    workspace.agent.act = wrapped_act
    try:
        metrics = workspace.eval()
    finally:
        workspace.shutdown()
        GlobalHydra.instance().clear()

    assert metrics["episode_success"] == 1.0
    assert recorded_eval_batch_sizes
    assert set(recorded_eval_batch_sizes) == {3}


def test_pretrain_steps_need_not_be_divisible_by_log_interval(tmp_path):
    dataset_path = tmp_path / "robomimic_test.hdf5"
    _write_robomimic_dataset(dataset_path)

    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).resolve().parents[2] / "robobase/cfgs"),
        job_name="test_pretrain_non_divisible_logging",
    ):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "method=bc",
                "env=robomimic",
                "pixels=false",
                "demos=1",
                f"env.dataset_path={dataset_path}",
                "env.use_live_env=false",
                "env.filter_key=train",
                "num_train_envs=1",
                "num_eval_envs=1",
                "num_eval_episodes=0",
                "num_pretrain_steps=101",
                "num_train_frames=0",
                "replay_size_before_train=0",
                "num_gpus=0",
                "batch_size=2",
                "replay.size=128",
                "replay.nstep=1",
                "replay.num_workers=0",
                "replay.pin_memory=false",
                "log_pretrain_every=100",
                "log_eval_video=false",
                "save_snapshot=false",
                "wandb.use=false",
            ],
        )

    workspace = Workspace(cfg, work_dir=tmp_path)
    train_completed = False
    try:
        workspace.train()
        train_completed = True
    finally:
        if not train_completed:
            workspace.shutdown()
        GlobalHydra.instance().clear()


def test_pretrain_final_step_triggers_eval_when_interval_divides_total_steps(tmp_path):
    dataset_path = tmp_path / "robomimic_pretrain_final_eval_test.hdf5"
    _write_robomimic_dataset(dataset_path)

    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).resolve().parents[2] / "robobase/cfgs"),
        job_name="test_pretrain_final_step_eval",
    ):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "method=bc",
                "env=robomimic",
                "pixels=false",
                "demos=1",
                f"env.dataset_path={dataset_path}",
                "env.use_live_env=false",
                "env.filter_key=train",
                "num_train_envs=1",
                "num_eval_envs=1",
                "num_eval_episodes=1",
                "num_pretrain_steps=4",
                "num_train_frames=0",
                "replay_size_before_train=0",
                "num_gpus=0",
                "batch_size=2",
                "replay.size=128",
                "replay.nstep=1",
                "replay.num_workers=0",
                "replay.pin_memory=false",
                "log_pretrain_every=1000",
                "eval_every_steps=2",
                "log_eval_video=false",
                "save_snapshot=false",
                "wandb.use=false",
            ],
        )

    workspace = Workspace(cfg, work_dir=tmp_path)
    eval_steps = []
    logged_eval_steps = []
    logged_env_steps = []

    def fake_eval(*args, **kwargs):
        del args, kwargs
        eval_steps.append(workspace.pretrain_steps + 1)
        return {"episode_reward": 0.0, "episode_length": 0.0}

    def fake_log_metrics(metrics, step, prefix):
        if prefix == "pretrain_eval":
            logged_eval_steps.append(step)
            logged_env_steps.append(metrics["env_steps"])

    workspace._eval = fake_eval
    workspace.logger.log_metrics = fake_log_metrics

    train_completed = False
    try:
        workspace.train()
        train_completed = True
    finally:
        if not train_completed:
            workspace.shutdown()
        GlobalHydra.instance().clear()

    assert eval_steps == [2, 4]
    assert logged_eval_steps == [2, 4]
    assert logged_env_steps == [2, 4]


def test_online_rl_final_step_triggers_eval_when_interval_divides_total_steps(tmp_path):
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).resolve().parents[2] / "robobase/cfgs"),
        job_name="test_online_final_step_eval",
    ):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "method=bc",
                "env=robomimic",
                "pixels=false",
                "demos=0",
                "num_train_envs=1",
                "num_eval_envs=1",
                "num_eval_episodes=1",
                "num_pretrain_steps=0",
                "num_train_frames=4",
                "replay_size_before_train=10",
                "num_gpus=0",
                "batch_size=2",
                "replay.size=16",
                "replay.nstep=1",
                "replay.num_workers=0",
                "replay.pin_memory=false",
                "method.adaptive_lr=false",
                "log_every=1000",
                "eval_every_steps=2",
                "log_eval_video=false",
                "save_snapshot=false",
                "wandb.use=false",
            ],
        )

    workspace = Workspace(
        cfg,
        env_factory=_TinyTrainAndEvalFactory(),
        work_dir=tmp_path,
    )
    eval_steps = []
    logged_eval_steps = []
    logged_env_steps = []

    def fake_eval(*args, **kwargs):
        del args, kwargs
        eval_steps.append(workspace.main_loop_iterations + 1)
        return {"episode_reward": 0.0, "episode_length": 0.0}

    def fake_log_metrics(metrics, step, prefix):
        if prefix == "eval":
            logged_eval_steps.append(step)
            logged_env_steps.append(metrics["env_steps"])

    workspace._eval = fake_eval
    workspace.logger.log_metrics = fake_log_metrics

    train_completed = False
    try:
        workspace.train()
        train_completed = True
    finally:
        if not train_completed:
            workspace.shutdown()
        GlobalHydra.instance().clear()

    assert eval_steps == [2, 4]
    assert logged_eval_steps == [2, 4]
    assert logged_env_steps == [2, 4]


def test_execution_length_cannot_exceed_available_diffusion_actions(tmp_path):
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).resolve().parents[2] / "robobase/cfgs"),
        job_name="test_diffusion_noise_mask_execution_length",
    ):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "method=diffusion",
                "env=robomimic",
                "pixels=false",
                "num_gpus=0",
                "action_sequence=16",
                "execution_length=9",
                "method.noise_mask_steps=8",
                "method.repeat_action=true",
            ],
        )

    with pytest.raises(ValueError, match="available actions"):
        Workspace(cfg, work_dir=tmp_path)
    GlobalHydra.instance().clear()


def _run_pretrain_steps(workspace: Workspace, num_steps: int):
    workspace._load_demos()
    for _ in range(num_steps):
        workspace._perform_updates()
        workspace._pretrain_step += 1


def test_snapshot_can_resume_pretraining_exactly(tmp_path):
    dataset_path = tmp_path / "robomimic_resume_test.hdf5"
    _write_robomimic_dataset(dataset_path)

    common_overrides = [
        "method=bc",
        "env=robomimic",
        "pixels=false",
        "demos=1",
        f"env.dataset_path={dataset_path}",
        "env.use_live_env=false",
        "env.filter_key=train",
        "num_train_envs=1",
        "num_eval_envs=1",
        "num_eval_episodes=0",
        "num_pretrain_steps=4",
        "num_train_frames=0",
        "replay_size_before_train=0",
        "num_gpus=0",
        "batch_size=2",
        "replay.size=128",
        "replay.nstep=1",
        "replay.num_workers=0",
        "replay.pin_memory=false",
        "save_snapshot=true",
        "wandb.use=false",
    ]
    config_dir = str(Path(__file__).resolve().parents[2] / "robobase/cfgs")

    baseline_dir = tmp_path / "baseline"
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=config_dir,
        job_name="test_snapshot_resume_baseline",
    ):
        baseline_cfg = compose(
            config_name="robobase_config",
            overrides=common_overrides,
        )
    baseline_workspace = Workspace(baseline_cfg, work_dir=baseline_dir)
    try:
        _run_pretrain_steps(baseline_workspace, num_steps=4)
        baseline_state = {
            k: v.detach().cpu().clone()
            for k, v in baseline_workspace.agent.state_dict().items()
        }
    finally:
        baseline_workspace.shutdown()
        GlobalHydra.instance().clear()

    resume_dir = tmp_path / "resume"
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=config_dir,
        job_name="test_snapshot_resume_partial",
    ):
        partial_cfg = compose(
            config_name="robobase_config",
            overrides=common_overrides,
        )
    partial_workspace = Workspace(partial_cfg, work_dir=resume_dir)
    try:
        _run_pretrain_steps(partial_workspace, num_steps=2)
        partial_workspace.save_snapshot()
    finally:
        partial_workspace.shutdown()
        GlobalHydra.instance().clear()

    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=config_dir,
        job_name="test_snapshot_resume_reload",
    ):
        resumed_cfg = compose(
            config_name="robobase_config",
            overrides=common_overrides,
        )
    resumed_workspace = Workspace(resumed_cfg, work_dir=resume_dir)
    try:
        resumed_workspace.load_snapshot()
        assert resumed_workspace.pretrain_steps == 2
        _run_pretrain_steps(resumed_workspace, num_steps=2)
        resumed_state = {
            k: v.detach().cpu().clone()
            for k, v in resumed_workspace.agent.state_dict().items()
        }
    finally:
        resumed_workspace.shutdown()
        GlobalHydra.instance().clear()

    assert baseline_state.keys() == resumed_state.keys()
    for key in baseline_state:
        assert torch.allclose(baseline_state[key], resumed_state[key])


def test_snapshot_can_load_without_replay_files_for_eval(tmp_path):
    dataset_path = tmp_path / "robomimic_eval_snapshot_test.hdf5"
    _write_robomimic_dataset(dataset_path)

    common_overrides = [
        "method=bc",
        "env=robomimic",
        "pixels=false",
        "demos=1",
        f"env.dataset_path={dataset_path}",
        "env.use_live_env=false",
        "env.filter_key=train",
        "num_train_envs=1",
        "num_eval_envs=1",
        "num_eval_episodes=0",
        "num_pretrain_steps=1",
        "num_train_frames=0",
        "replay_size_before_train=0",
        "num_gpus=0",
        "batch_size=2",
        "replay.size=128",
        "replay.nstep=1",
        "replay.num_workers=0",
        "replay.pin_memory=false",
        "save_snapshot=true",
        "wandb.use=false",
    ]
    config_dir = str(Path(__file__).resolve().parents[2] / "robobase/cfgs")

    source_dir = tmp_path / "source"
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=config_dir,
        job_name="test_snapshot_eval_source",
    ):
        source_cfg = compose(
            config_name="robobase_config",
            overrides=common_overrides,
        )
    source_workspace = Workspace(source_cfg, work_dir=source_dir)
    try:
        _run_pretrain_steps(source_workspace, num_steps=1)
        source_workspace.save_snapshot()
    finally:
        source_workspace.shutdown()
        GlobalHydra.instance().clear()

    replay_dir = source_dir / "replay"
    for replay_file in replay_dir.glob("*.npz"):
        replay_file.unlink()

    eval_dir = tmp_path / "eval"
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=config_dir,
        job_name="test_snapshot_eval_resume",
    ):
        eval_cfg = compose(
            config_name="robobase_config",
            overrides=common_overrides,
        )
    eval_workspace = Workspace(eval_cfg, work_dir=eval_dir)
    try:
        eval_workspace.load_snapshot(
            source_dir / "snapshots" / "latest_snapshot.pt",
            load_replay_buffer=False,
        )
        assert eval_workspace.pretrain_steps == 1
    finally:
        eval_workspace.shutdown()
        GlobalHydra.instance().clear()


def test_snapshot_resume_preserves_runtime_cfg_overrides(tmp_path):
    dataset_path = tmp_path / "robomimic_resume_override_test.hdf5"
    _write_robomimic_dataset(dataset_path)

    common_overrides = [
        "method=bc",
        "env=robomimic",
        "pixels=false",
        "demos=1",
        f"env.dataset_path={dataset_path}",
        "env.use_live_env=false",
        "env.filter_key=train",
        "num_train_envs=1",
        "num_eval_envs=1",
        "num_eval_episodes=1",
        "num_train_frames=0",
        "replay_size_before_train=0",
        "num_gpus=0",
        "batch_size=2",
        "replay.size=128",
        "replay.nstep=1",
        "replay.num_workers=0",
        "replay.pin_memory=false",
        "log_pretrain_every=1000",
        "eval_every_steps=1000",
        "save_snapshot=true",
        "snapshot_every_n=1000",
        "wandb.use=false",
    ]
    config_dir = str(Path(__file__).resolve().parents[2] / "robobase/cfgs")

    resume_dir = tmp_path / "resume_override"
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=config_dir,
        job_name="test_snapshot_resume_override_partial",
    ):
        partial_cfg = compose(
            config_name="robobase_config",
            overrides=common_overrides + ["num_pretrain_steps=4"],
        )
    partial_workspace = Workspace(partial_cfg, work_dir=resume_dir)
    try:
        _run_pretrain_steps(partial_workspace, num_steps=2)
        partial_workspace.save_snapshot()
    finally:
        partial_workspace.shutdown()
        GlobalHydra.instance().clear()

    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=config_dir,
        job_name="test_snapshot_resume_override_reload",
    ):
        resumed_cfg = compose(
            config_name="robobase_config",
            overrides=common_overrides + ["num_pretrain_steps=6"],
        )
    resumed_workspace = Workspace(resumed_cfg, work_dir=resume_dir)
    try:
        resumed_workspace.load_snapshot()
        assert resumed_workspace.pretrain_steps == 2
        assert resumed_workspace.cfg.num_pretrain_steps == 6
        assert resumed_workspace._snapshot_cfg.num_pretrain_steps == 4

        resumed_workspace.train()

        assert resumed_workspace.pretrain_steps == 6
    finally:
        resumed_workspace.shutdown()
        GlobalHydra.instance().clear()


def test_snapshot_saving_starts_after_configured_pretrain_step(tmp_path):
    dataset_path = tmp_path / "robomimic_snapshot_start_test.hdf5"
    _write_robomimic_dataset(dataset_path)

    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).resolve().parents[2] / "robobase/cfgs"),
        job_name="test_snapshot_save_start_step",
    ):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "method=bc",
                "env=robomimic",
                "pixels=false",
                "demos=1",
                f"env.dataset_path={dataset_path}",
                "env.use_live_env=false",
                "env.filter_key=train",
                "num_train_envs=1",
                "num_eval_envs=1",
                "num_eval_episodes=0",
                "num_pretrain_steps=6",
                "num_train_frames=0",
                "replay_size_before_train=0",
                "num_gpus=0",
                "batch_size=2",
                "replay.size=128",
                "replay.nstep=1",
                "replay.num_workers=0",
                "replay.pin_memory=false",
                "save_snapshot=true",
                "snapshot_every_n=2",
                "snapshot_save_start_step=4",
                "wandb.use=false",
            ],
        )

    workspace = Workspace(cfg, work_dir=tmp_path)
    train_completed = False
    try:
        workspace.train()
        train_completed = True
    finally:
        if not train_completed:
            workspace.shutdown()
        GlobalHydra.instance().clear()

    snapshot_names = sorted(
        path.name
        for path in (tmp_path / "snapshots").glob("*_snapshot.pt")
        if path.name != "latest_snapshot.pt"
    )
    assert snapshot_names == ["4_snapshot.pt", "6_snapshot.pt"]
