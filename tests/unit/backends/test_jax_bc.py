from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import flax.linen as nn
from flax.core import freeze

from robobase.backends.jax.method.bc import JaxBC
from robobase.backends.jax.models.encoder import JaxResNetEncoder
from robobase.envs.env import EnvFactory
from robobase.method.bc import (
    BCActorModelSpec,
    BCEncoderModelSpec,
    BCModelSpec,
    BCViewFusionModelSpec,
)
from robobase.workspace import Workspace

jax = pytest.importorskip("jax")
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


class _TinyTrainEnv(gym.Env):
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
        obs = {"low_dim_state": np.full((1, 4), self._step, dtype=np.float32)}
        terminated = self._step >= self._episode_len
        return obs, float(terminated), terminated, False, {"task_success": terminated}


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


def _params_leaves(state_dict: dict):
    leaves, _ = jax.tree_util.tree_flatten(state_dict["params"])
    return [np.asarray(leaf) for leaf in leaves]


def _make_jax_bc(
    *,
    observation_space,
    action_space,
    hidden_dims=(16, 16),
    output_sequence_network_type="mlp",
):
    model = BCModelSpec(
        actor_model=BCActorModelSpec(
            type="mlp_bottleneck_sequence",
            hidden_dims=tuple(hidden_dims),
            num_rnn_layers=1,
            rnn_hidden_size=32,
            keys_to_bottleneck=(),
            bottleneck_size=16,
            norm_after_bottleneck=True,
            tanh_after_bottleneck=True,
            output_sequence_network_type=output_sequence_network_type,
            output_sequence_length=action_space.shape[0],
        ),
        encoder_model=BCEncoderModelSpec(type="resnet", model="resnet18")
        if any(key.startswith("rgb") for key in observation_space.keys())
        else None,
        view_fusion_model=BCViewFusionModelSpec(type="multicam_feature", mode="flatten")
        if len([key for key in observation_space.keys() if key.startswith("rgb")]) > 1
        else None,
    )
    return JaxBC(
        lr=1e-3,
        adaptive_lr=False,
        num_train_steps=4,
        model=model,
        observation_space=observation_space,
        action_space=action_space,
        num_train_envs=1,
        num_eval_envs=1,
        replay_alpha=0.6,
        replay_beta=0.4,
        frame_stack_on_channel=True,
        actor_grad_clip=None,
        jit=True,
        seed=0,
    )


def test_jax_bc_workspace_smoke_and_snapshot(tmp_path):
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).resolve().parents[3] / "robobase/cfgs"),
        job_name="test_jax_bc_workspace",
    ):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "backend=jax",
                "method=bc",
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
                "action_sequence=1",
                "execution_length=1",
                "env.episode_length=2",
                "method.adaptive_lr=false",
                "log_every=1",
                "log_eval_video=false",
                "save_snapshot=true",
                "snapshot_every_n=1",
                "wandb.use=false",
            ],
        )

    workspace = Workspace(
        cfg,
        env_factory=_TinyTrainAndEvalFactory(),
        work_dir=tmp_path,
    )
    train_completed = False
    try:
        workspace.train()
        train_completed = True
        saved_state = workspace.agent.state_dict()
    finally:
        if not train_completed:
            workspace.shutdown()

    snapshot_path = tmp_path / "snapshots" / "latest_snapshot.pt"
    assert snapshot_path.exists()

    restored = Workspace(
        cfg,
        env_factory=_TinyTrainAndEvalFactory(),
        work_dir=tmp_path,
    )
    try:
        restored.load_snapshot()
        restored_state = restored.agent.state_dict()
    finally:
        restored.shutdown()
        GlobalHydra.instance().clear()

    assert len(_params_leaves(saved_state)) == len(_params_leaves(restored_state))
    for before, after in zip(_params_leaves(saved_state), _params_leaves(restored_state)):
        assert np.allclose(before, after)


def test_jax_bc_supports_recurrent_sequence_output():
    observation_space = spaces.Dict(
        {
            "low_dim_state": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1, 4),
                dtype=np.float32,
            )
        }
    )
    action_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(3, 2),
        dtype=np.float32,
    )
    agent = _make_jax_bc(
        observation_space=observation_space,
        action_space=action_space,
        output_sequence_network_type="rnn",
    )

    obs = {"low_dim_state": np.zeros((2, 1, 4), dtype=np.float32)}
    actions = agent.act(obs, step=0, eval_mode=True)
    assert actions.shape == (2, 3, 2)

    batch = {
        "low_dim_state": np.zeros((2, 1, 4), dtype=np.float32),
        "action": np.zeros((2, 3, 2), dtype=np.float32),
        "indices": np.arange(2, dtype=np.int64),
    }
    metrics = agent.update(iter([batch]), step=0)
    assert isinstance(metrics, dict)


class _FakeResNetFeatureModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        return jax.numpy.zeros((x.shape[0], 512), dtype=jax.numpy.float32)


def _fake_pretrained_resnet_feature_model():
    return _FakeResNetFeatureModel(), freeze({}), 512


def test_jax_resnet_encoder_runs_native_forward(monkeypatch):
    monkeypatch.setattr(
        "robobase.backends.jax.models.encoder._load_pretrained_resnet_feature_model",
        lambda model_name: _fake_pretrained_resnet_feature_model(),
    )

    encoder = JaxResNetEncoder(
        input_shape=(2, 3, 8, 8),
        model="resnet18",
        jit=True,
    )
    rgb_obs = np.random.randint(0, 255, size=(3, 2, 3, 8, 8), dtype=np.uint8)
    feats = np.asarray(jax.device_get(encoder.encode(rgb_obs)))
    assert feats.shape == (3, 2, 512)
    assert np.all(np.isfinite(feats))


def test_jax_bc_supports_pixel_inputs_with_multicam_fusion(monkeypatch):
    monkeypatch.setattr(
        "robobase.backends.jax.models.encoder._load_pretrained_resnet_feature_model",
        lambda model_name: _fake_pretrained_resnet_feature_model(),
    )

    observation_space = spaces.Dict(
        {
            "low_dim_state": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1, 4),
                dtype=np.float32,
            ),
            "rgb_front": spaces.Box(
                low=0,
                high=255,
                shape=(1, 3, 8, 8),
                dtype=np.uint8,
            ),
            "rgb_wrist": spaces.Box(
                low=0,
                high=255,
                shape=(1, 3, 8, 8),
                dtype=np.uint8,
            ),
        }
    )
    action_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(2, 3),
        dtype=np.float32,
    )
    agent = _make_jax_bc(
        observation_space=observation_space,
        action_space=action_space,
        output_sequence_network_type="rnn",
    )

    obs = {
        "low_dim_state": np.zeros((2, 1, 4), dtype=np.float32),
        "rgb_front": np.full((2, 1, 3, 8, 8), 64, dtype=np.uint8),
        "rgb_wrist": np.full((2, 1, 3, 8, 8), 128, dtype=np.uint8),
    }
    act_out = agent.act(obs, step=0, eval_mode=True)
    assert act_out.shape == (2, 2, 3)

    batch = {
        **obs,
        "action": np.zeros((2, 2, 3), dtype=np.float32),
        "indices": np.arange(2, dtype=np.int64),
    }
    metrics = agent.update(iter([batch]), step=0)
    assert isinstance(metrics, dict)
