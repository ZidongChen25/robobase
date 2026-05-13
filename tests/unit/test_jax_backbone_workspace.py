from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from robobase.envs.env import EnvFactory
from robobase.workspace import Workspace

pytest.importorskip("jax")
pytest.importorskip("optax")


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
            shape=(4, 2),
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


@pytest.mark.parametrize("method_name", ["diffusion", "flow_matching"])
@pytest.mark.parametrize("backbone_type", ["transformer", "dit"])
def test_transformer_and_dit_backbones_train_through_workspace(
    tmp_path,
    method_name,
    backbone_type,
):
    overrides = [
        "backend=jax",
        f"method={method_name}",
        "env=dmc/cartpole_balance",
        "pixels=false",
        "demos=0",
        "num_train_envs=1",
        "num_eval_envs=1",
        "num_eval_episodes=0",
        "num_pretrain_steps=0",
        "num_train_frames=4",
        "replay_size_before_train=2",
        "num_gpus=0",
        "batch_size=1",
        "replay.size=16",
        "replay.nstep=1",
        "replay.num_workers=0",
        "replay.pin_memory=false",
        "action_repeat=1",
        "action_sequence=4",
        "execution_length=1",
        "env.episode_length=2",
        "method.adaptive_lr=false",
        "method.backbone.d_model=32",
        "method.backbone.n_heads=4",
        "method.backbone.num_layers=1",
        "method.backbone.n_cond_layers=1",
        "method.backbone.depth=1",
        "method.backbone.diffusion_step_embed_dim=16",
        "method.backbone.dropout=0.0",
        f"method.backbone.type={backbone_type}",
        "log_every=1",
        "log_eval_video=false",
        "save_snapshot=false",
        "wandb.use=false",
    ]
    if method_name == "diffusion":
        overrides.extend(
            [
                "method.num_diffusion_iters=2",
                "method.use_ema=false",
            ]
        )
    else:
        overrides.extend(
            [
                "method.num_flow_steps=2",
            ]
        )

    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).resolve().parents[2] / "robobase/cfgs"),
        job_name=f"test_{method_name}_{backbone_type}_workspace",
    ):
        cfg = compose(config_name="robobase_config", overrides=overrides)

    workspace = Workspace(
        cfg,
        env_factory=_TinyTrainAndEvalFactory(),
        work_dir=tmp_path / method_name / backbone_type,
    )
    try:
        workspace.train()
    finally:
        GlobalHydra.instance().clear()
