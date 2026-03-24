from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from robobase.method.bc import BCSpec, bc_spec_from_cfg
from robobase.backends.torch.method.bc import BC as TorchBackendBC
from robobase.envs.env import EnvFactory
from robobase.workspace import Workspace


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
        return obs, float(terminated), terminated, False, {"task_success": terminated}


class _TinyTrainEnv(_TinyEvalEnv):
    pass


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


def _compose_cfg(*overrides: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).resolve().parents[3] / "robobase/cfgs"),
        job_name="test_bc_backend_dispatch",
    ):
        return compose(
            config_name="robobase_config",
            overrides=[
                "method=bc",
                "env=dmc/cartpole_balance",
                "pixels=false",
                "demos=0",
                "num_train_envs=1",
                "num_eval_envs=1",
                "num_eval_episodes=0",
                "num_pretrain_steps=0",
                "num_train_frames=0",
                "replay_size_before_train=0",
                "num_gpus=0",
                "batch_size=1",
                "replay.size=16",
                "replay.nstep=1",
                "replay.num_workers=0",
                "replay.pin_memory=false",
                "action_repeat=1",
                "action_sequence=1",
                "execution_length=1",
                "env.episode_length=2",
                "method.adaptive_lr=false",
                "log_eval_video=false",
                "save_snapshot=false",
                "wandb.use=false",
                "tb.use=false",
                *overrides,
            ],
        )


def test_shared_bc_module_exposes_backend_neutral_spec():
    assert BCSpec.__module__ == "robobase.method.bc"


def test_bc_spec_is_shared_across_backends():
    cfg = _compose_cfg(
        "method.actor_model.hidden_dims=[32,64]",
        "method.adaptive_lr=false",
    )
    spec = bc_spec_from_cfg(cfg)
    GlobalHydra.instance().clear()

    assert spec.model.actor_model.hidden_dims == (32, 64)
    assert spec.adaptive_lr is False
    assert spec.lr == pytest.approx(1e-4)
    assert spec.model.actor_model.type == "mlp_bottleneck_sequence"
    assert spec.model.encoder_model.type == "resnet"
    assert spec.model.view_fusion_model.type == "multicam_feature"


def test_torch_bc_workspace_uses_torch_backend_module(tmp_path):
    cfg = _compose_cfg("backend=torch")

    workspace = Workspace(
        cfg,
        env_factory=_TinyTrainAndEvalFactory(),
        work_dir=tmp_path,
    )
    try:
        assert isinstance(workspace.agent, TorchBackendBC)
        assert workspace.agent.__class__.__module__ == "robobase.backends.torch.method.bc"
    finally:
        workspace.shutdown()
        GlobalHydra.instance().clear()


def test_jax_bc_workspace_uses_jax_backend_module(tmp_path):
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    from robobase.backends.jax.method.bc import JaxBC

    cfg = _compose_cfg("backend=jax")

    workspace = Workspace(
        cfg,
        env_factory=_TinyTrainAndEvalFactory(),
        work_dir=tmp_path,
    )
    try:
        assert isinstance(workspace.agent, JaxBC)
        assert workspace.agent.__class__.__module__ == "robobase.backends.jax.method.bc"
    finally:
        workspace.shutdown()
        GlobalHydra.instance().clear()


def test_robomimic_bc_launch_decouples_task_from_method_family():
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).resolve().parents[3] / "robobase/cfgs"),
        job_name="test_robomimic_bc_launch",
    ):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=bc_state_robomimic",
                "env=robomimic/tool_hang",
                "backend=jax",
            ],
        )
    try:
        assert cfg.env.task_name == "ToolHang"
        assert cfg.method.name == "bc"
        assert cfg.backend.name == "jax"
        assert cfg.is_imitation_learning is True
        assert cfg.demos == float("inf")
        assert cfg.num_pretrain_steps == 200000
        assert cfg.num_train_frames == 0
        assert cfg.replay.nstep == 1
    finally:
        GlobalHydra.instance().clear()
