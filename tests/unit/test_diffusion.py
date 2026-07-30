from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import flax.linen as nn
from flax.core import freeze
from gymnasium import spaces
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from robobase.method.bc import BCEncoderModelSpec, BCViewFusionModelSpec
from robobase.method.diffusion import Diffusion as JaxDiffusion
from robobase.envs.env import EnvFactory
from robobase.method.diffusion import (
    DiffusionActorModelSpec,
    DiffusionModelSpec,
    diffusion_spec_from_cfg,
)
from robobase.models.backbone import DiffusionBackboneSpec, build_diffusion_backbone
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


def _params_leaves(state_dict: dict):
    leaves, _ = jax.tree_util.tree_flatten(state_dict["params"])
    return [np.asarray(leaf) for leaf in leaves]


class _FakeResNetFeatureModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        return jax.numpy.zeros((x.shape[0], 1, 1, 512), dtype=jax.numpy.float32)


def _fake_resnet_feature_model():
    return _FakeResNetFeatureModel(), freeze({}), 512


def _make_jax_diffusion(*, observation_space, action_space):
    model = DiffusionModelSpec(
        actor_model=DiffusionActorModelSpec(
            type="conditional_unet1d",
            sequence_length=action_space.shape[0],
            diffusion_step_embed_dim=32,
            down_dims=(32, 64),
            kernel_size=3,
            n_groups=4,
        ),
        encoder_model=None,
        view_fusion_model=None,
    )
    return JaxDiffusion(
        lr=1e-3,
        adaptive_lr=False,
        num_train_steps=4,
        num_diffusion_iters=4,
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
        use_ema=False,
    )


@pytest.mark.parametrize(
    "backbone_type",
    ["fully_connected", "unet1d", "transformer", "dit"],
)
def test_diffusion_backbone_registry_output_shape(backbone_type):
    spec = DiffusionBackboneSpec(
        type=backbone_type,
        sequence_length=4,
        diffusion_step_embed_dim=16,
        down_dims=(16, 32),
        hidden_dims=(32,),
        d_model=32,
        n_heads=4,
        num_layers=1,
        n_cond_layers=1 if backbone_type == "transformer" else 0,
        depth=1,
    )
    model = build_diffusion_backbone(
        spec,
        action_dim=2,
        sequence_length=4,
        condition_dim=5,
    )
    actions = jax.numpy.zeros((3, 4, 2), dtype=jax.numpy.float32)
    time = jax.numpy.arange(3, dtype=jax.numpy.float32)
    condition = jax.numpy.zeros((3, 5), dtype=jax.numpy.float32)

    params = model.init(jax.random.PRNGKey(0), actions, time, condition)
    output = model.apply(params, actions, time, condition)

    assert output.shape == actions.shape
    assert output.dtype == jax.numpy.float32


def test_diffusion_config_uses_objective_and_backbone():
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).resolve().parents[2] / "robobase/cfgs"),
        job_name="test_jax_diffusion_config",
    ):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "backend=jax",
                "method=diffusion",
                "env=dmc/cartpole_balance",
                "pixels=false",
                "action_sequence=4",
                "method.num_diffusion_iters=7",
                "method.backbone.type=fully_connected",
                "method.backbone.hidden_dims=[16]",
            ],
        )

    try:
        spec = diffusion_spec_from_cfg(cfg)
    finally:
        GlobalHydra.instance().clear()

    assert spec.objective_type == "ddpm"
    assert spec.num_diffusion_iters == 7
    assert spec.sampler == "ddim"
    assert spec.model.resolved_backbone.type == "fully_connected"
    assert spec.model.resolved_backbone.hidden_dims == (16,)


def test_jax_diffusion_fully_connected_backbone_update_and_act():
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
    model = DiffusionModelSpec(
        actor_model=DiffusionActorModelSpec(
            type="fully_connected",
            sequence_length=action_space.shape[0],
            diffusion_step_embed_dim=16,
            hidden_dims=(32,),
        ),
        encoder_model=None,
        view_fusion_model=None,
    )
    agent = JaxDiffusion(
        lr=1e-3,
        adaptive_lr=False,
        num_train_steps=4,
        num_diffusion_iters=4,
        model=model,
        observation_space=observation_space,
        action_space=action_space,
        num_train_envs=1,
        num_eval_envs=1,
        replay_alpha=0.6,
        replay_beta=0.4,
        frame_stack_on_channel=True,
        actor_grad_clip=None,
        jit=False,
        seed=0,
        use_ema=False,
    )
    batch = {
        "low_dim_state": np.zeros((2, 1, 4), dtype=np.float32),
        "action": np.zeros((2, 3, 2), dtype=np.float32),
    }

    before = _params_leaves(agent.state_dict())
    metrics = agent.update(iter([batch]), step=0)
    after = _params_leaves(agent.state_dict())
    actions = agent.act(batch, step=0, eval_mode=False)

    assert metrics == {}
    assert actions.shape == (2, 3, 2)
    assert agent.ema_params is None
    assert any(not np.allclose(a, b) for a, b in zip(before, after))


def test_jax_diffusion_all_valid_padding_mask_matches_unmasked_loss():
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
        shape=(4, 3),
        dtype=np.float32,
    )
    agent = JaxDiffusion(
        lr=1e-3,
        adaptive_lr=False,
        num_train_steps=4,
        num_diffusion_iters=4,
        model=DiffusionModelSpec(
            actor_model=DiffusionActorModelSpec(
                type="fully_connected",
                sequence_length=4,
                diffusion_step_embed_dim=16,
                hidden_dims=(32,),
            ),
            encoder_model=None,
            view_fusion_model=None,
        ),
        observation_space=observation_space,
        action_space=action_space,
        num_train_envs=1,
        num_eval_envs=1,
        replay_alpha=0.6,
        replay_beta=0.4,
        frame_stack_on_channel=True,
        jit=False,
        seed=0,
        use_ema=False,
    )
    batch = {
        "low_dim_state": np.zeros((2, 1, 4), dtype=np.float32),
        "action": np.ones((2, 4, 3), dtype=np.float32),
    }
    obs_features, _ = agent._prepare_obs_features(batch)
    common_args = (
        agent.params,
        agent.opt_state,
        agent.rng_key,
        obs_features,
        jax.numpy.asarray(batch["action"]),
        jax.numpy.ones((2,), dtype=jax.numpy.float32),
    )

    unmasked_loss = agent._update_impl(*common_args, None, None, 0)[3]
    all_valid_loss = agent._update_impl(
        *common_args,
        jax.numpy.zeros((2, 4), dtype=jax.numpy.bool_),
        None,
        0,
    )[3]

    np.testing.assert_allclose(all_valid_loss, unmasked_loss, rtol=1e-6)


def test_jax_diffusion_supports_plucker_camera_params_with_trainable_encoder(monkeypatch):
    monkeypatch.setattr(
        "robobase.models.encoder._load_resnet_feature_model",
        lambda model_name, pretrained=False: _fake_resnet_feature_model(),
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
            "camera_intrinsic_front": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1, 3, 3),
                dtype=np.float32,
            ),
            "camera_c2w_front": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1, 4, 4),
                dtype=np.float32,
            ),
            "rgb_wrist": spaces.Box(
                low=0,
                high=255,
                shape=(1, 3, 8, 8),
                dtype=np.uint8,
            ),
            "camera_intrinsic_wrist": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1, 3, 3),
                dtype=np.float32,
            ),
            "camera_c2w_wrist": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1, 4, 4),
                dtype=np.float32,
            ),
        }
    )
    action_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(3, 2),
        dtype=np.float32,
    )
    model = DiffusionModelSpec(
        actor_model=DiffusionActorModelSpec(
            type="fully_connected",
            sequence_length=action_space.shape[0],
            diffusion_step_embed_dim=16,
            hidden_dims=(32,),
        ),
        encoder_model=BCEncoderModelSpec(
            type="resnet",
            model="resnet18",
            trainable=True,
            use_plucker=True,
            plucker_hidden_channels=4,
        ),
        view_fusion_model=BCViewFusionModelSpec(
            type="multicam_feature",
            mode="flatten",
        ),
    )
    agent = JaxDiffusion(
        lr=1e-3,
        adaptive_lr=False,
        num_train_steps=4,
        num_diffusion_iters=4,
        model=model,
        observation_space=observation_space,
        action_space=action_space,
        num_train_envs=1,
        num_eval_envs=1,
        replay_alpha=0.6,
        replay_beta=0.4,
        frame_stack_on_channel=True,
        actor_grad_clip=None,
        jit=False,
        seed=0,
        use_ema=False,
    )

    intrinsic = np.eye(3, dtype=np.float32)
    c2w = np.eye(4, dtype=np.float32)
    batch = {
        "low_dim_state": np.zeros((2, 1, 4), dtype=np.float32),
        "rgb_front": np.full((2, 1, 3, 8, 8), 64, dtype=np.uint8),
        "camera_intrinsic_front": np.tile(intrinsic, (2, 1, 1, 1)),
        "camera_c2w_front": np.tile(c2w, (2, 1, 1, 1)),
        "rgb_wrist": np.full((2, 1, 3, 8, 8), 128, dtype=np.uint8),
        "camera_intrinsic_wrist": np.tile(intrinsic, (2, 1, 1, 1)),
        "camera_c2w_wrist": np.tile(c2w, (2, 1, 1, 1)),
        "action": np.zeros((2, 3, 2), dtype=np.float32),
    }

    metrics = agent.update(iter([batch]), step=0)
    actions = agent.act(batch, step=0, eval_mode=False)

    assert metrics == {}
    assert actions.shape == (2, 3, 2)


def test_jax_diffusion_update_many_runs_multiple_updates():
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
    model = DiffusionModelSpec(
        actor_model=DiffusionActorModelSpec(
            type="fully_connected",
            sequence_length=action_space.shape[0],
            diffusion_step_embed_dim=16,
            hidden_dims=(32,),
        ),
        encoder_model=None,
        view_fusion_model=None,
    )
    agent = JaxDiffusion(
        lr=1e-3,
        adaptive_lr=False,
        num_train_steps=4,
        num_diffusion_iters=4,
        model=model,
        observation_space=observation_space,
        action_space=action_space,
        num_train_envs=1,
        num_eval_envs=1,
        replay_alpha=0.6,
        replay_beta=0.4,
        frame_stack_on_channel=True,
        actor_grad_clip=None,
        jit=False,
        seed=0,
        use_ema=False,
        update_block_every_steps=2,
    )
    batches = [
        {
            "low_dim_state": np.zeros((2, 1, 4), dtype=np.float32),
            "action": np.zeros((2, 3, 2), dtype=np.float32),
        },
        {
            "low_dim_state": np.ones((2, 1, 4), dtype=np.float32),
            "action": np.ones((2, 3, 2), dtype=np.float32) * 0.1,
        },
    ]

    before = _params_leaves(agent.state_dict())
    metrics = agent.update_many(iter(batches), num_updates=2)
    after = _params_leaves(agent.state_dict())

    assert metrics == {}
    assert agent._update_step_count == 2
    assert agent.ema_params is None
    assert any(not np.allclose(a, b) for a, b in zip(before, after))


def test_jax_diffusion_workspace_smoke_and_snapshot(tmp_path):
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).resolve().parents[2] / "robobase/cfgs"),
        job_name="test_jax_diffusion_workspace",
    ):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "backend=jax",
                "method=diffusion",
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
                "method.num_diffusion_iters=4",
                "method.use_ema=false",
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

    snapshot_path = tmp_path / "snapshots" / "latest_snapshot.pkl"
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


def test_jax_diffusion_act_returns_action_sequence():
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
        shape=(4, 2),
        dtype=np.float32,
    )
    agent = _make_jax_diffusion(
        observation_space=observation_space,
        action_space=action_space,
    )

    obs = {"low_dim_state": np.zeros((2, 1, 4), dtype=np.float32)}
    actions = agent.act(obs, step=0, eval_mode=False)

    assert actions.shape == (2, 4, 2)
    assert actions.dtype == np.float32


def test_jax_diffusion_ema_decay_matches_diffusers_first_step():
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
        shape=(4, 2),
        dtype=np.float32,
    )
    agent = JaxDiffusion(
        lr=1e-3,
        adaptive_lr=False,
        num_train_steps=4,
        num_diffusion_iters=4,
        model=DiffusionModelSpec(
            actor_model=DiffusionActorModelSpec(
                type="conditional_unet1d",
                sequence_length=action_space.shape[0],
                diffusion_step_embed_dim=32,
                down_dims=(32, 64),
                kernel_size=3,
                n_groups=4,
            ),
            encoder_model=None,
            view_fusion_model=None,
        ),
        observation_space=observation_space,
        action_space=action_space,
        num_train_envs=1,
        num_eval_envs=1,
        replay_alpha=0.6,
        replay_beta=0.4,
        frame_stack_on_channel=True,
        actor_grad_clip=None,
        jit=False,
        seed=0,
        use_ema=True,
    )

    assert float(agent._ema_decay_value(1)) == pytest.approx(0.0)
    assert float(agent._ema_decay_value(2)) == pytest.approx(2.0 / 11.0)
