from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from robobase.envs.env import EnvFactory
from robobase.method.flow_matching import (
    FlowMatching,
    FlowMatchingBackboneSpec,
    FlowMatchingModelSpec,
    flow_matching_spec_from_cfg,
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


def _make_flow_matching(
    *,
    observation_space,
    action_space,
    jit=False,
    train_time_schedule="uniform",
    horizon_dropout_lengths=None,
    horizon_dropout_probs=None,
    horizon_loss_weights=None,
    use_lang_cond=False,
    lang_feature_dim=512,
    backbone_type="fully_connected",
):
    model = FlowMatchingModelSpec(
        backbone=FlowMatchingBackboneSpec(
            type=backbone_type,
            sequence_length=action_space.shape[0],
            diffusion_step_embed_dim=16,
            hidden_dims=(32,),
            d_model=32,
            n_heads=4,
            num_layers=1,
            n_cond_layers=1,
            depth=1,
            dropout=0.0,
        ),
        encoder_model=None,
        view_fusion_model=None,
        use_lang_cond=use_lang_cond,
        lang_feature_dim=lang_feature_dim,
    )
    return FlowMatching(
        lr=1e-3,
        adaptive_lr=False,
        num_train_steps=4,
        num_flow_steps=3,
        model=model,
        observation_space=observation_space,
        action_space=action_space,
        num_train_envs=1,
        num_eval_envs=1,
        replay_alpha=0.6,
        replay_beta=0.4,
        frame_stack_on_channel=True,
        actor_grad_clip=None,
        jit=jit,
        seed=0,
        use_ema=False,
        train_time_schedule=train_time_schedule,
        horizon_dropout_lengths=horizon_dropout_lengths,
        horizon_dropout_probs=horizon_dropout_probs,
        horizon_loss_weights=horizon_loss_weights,
    )


class _FakeFrozenEncoder:
    def __init__(self, value: float):
        self.value = jax.numpy.asarray([value], dtype=jax.numpy.float32)

    def frozen_state_dict(self):
        return {"batch_stats": {"value": self.value}}

    def load_frozen_state_dict(self, state_dict):
        self.value = jax.numpy.asarray(state_dict["batch_stats"]["value"])


def test_flow_matching_state_dict_roundtrips_encoder_frozen_state():
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
    saved_agent = _make_flow_matching(
        observation_space=observation_space,
        action_space=action_space,
        jit=False,
    )
    saved_agent.encoder = _FakeFrozenEncoder(3.5)
    state = saved_agent.state_dict()

    assert "_encoder_frozen_state" in state
    restored_agent = _make_flow_matching(
        observation_space=observation_space,
        action_space=action_space,
        jit=False,
    )
    restored_agent.encoder = _FakeFrozenEncoder(-1.0)
    restored_agent.load_state_dict(state)

    np.testing.assert_array_equal(
        np.asarray(restored_agent.encoder.value),
        np.asarray(saved_agent.encoder.value),
    )


def test_flow_matching_prefers_precomputed_lang_features(monkeypatch):
    observation_space = spaces.Dict(
        {
            "low_dim_state": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1, 4),
                dtype=np.float32,
            ),
            "lang_features": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1, 3),
                dtype=np.float32,
            ),
            "lang_tokens": spaces.Box(
                low=0,
                high=np.iinfo(np.int32).max,
                shape=(1, 77),
                dtype=np.int32,
            ),
        }
    )
    action_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(3, 2),
        dtype=np.float32,
    )
    agent = _make_flow_matching(
        observation_space=observation_space,
        action_space=action_space,
        jit=False,
        use_lang_cond=True,
        lang_feature_dim=3,
    )

    def fail_token_path(*args, **kwargs):
        del args, kwargs
        raise AssertionError("lang_features should bypass token feature hashing")

    monkeypatch.setattr("robobase.method.flow_matching.tokens_to_feature_jax", fail_token_path)
    features = np.array([[[0.25, -0.5, 1.5]]], dtype=np.float32)
    actual = agent._extract_lang_features(
        {
            "lang_features": features,
            "lang_tokens": np.ones((1, 1, 77), dtype=np.int32),
        }
    )

    np.testing.assert_allclose(np.asarray(jax.device_get(actual)), features[:, -1])


def test_flow_matching_config_uses_rectified_flow_and_backbone():
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).resolve().parents[2] / "robobase/cfgs"),
        job_name="test_jax_flow_matching_config",
    ):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "backend=jax",
                "method=flow_matching",
                "env=dmc/cartpole_balance",
                "pixels=false",
                "action_sequence=4",
                "method.num_flow_steps=3",
                "method.objective.train_time_schedule=beta_0p5_0p5",
                "method.horizon_dropout_lengths=[1,2,4]",
                "method.horizon_dropout_probs=[0.2,0.3,0.5]",
                "method.horizon_loss_weights=[0.5,0.25,0.15,0.1]",
                "method.backbone.type=transformer",
                "method.backbone.d_model=32",
                "method.backbone.n_heads=4",
                "method.backbone.num_layers=1",
            ],
        )

    try:
        spec = flow_matching_spec_from_cfg(cfg)
    finally:
        GlobalHydra.instance().clear()

    assert spec.objective_type == "rectified_flow"
    assert spec.num_flow_steps == 3
    assert spec.sampler == "euler"
    assert spec.train_time_schedule == "beta_0p5_0p5"
    assert spec.horizon_dropout_lengths == (1, 2, 4)
    assert spec.horizon_dropout_probs == (0.2, 0.3, 0.5)
    assert spec.horizon_loss_weights == (0.5, 0.25, 0.15, 0.1)
    assert spec.model.backbone.type == "transformer"
    assert spec.model.backbone.d_model == 32


def test_flow_matching_update_logs_loss_and_changes_params():
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
    agent = _make_flow_matching(
        observation_space=observation_space,
        action_space=action_space,
        jit=False,
    )
    agent.logging = True
    batch = {
        "low_dim_state": np.zeros((2, 1, 4), dtype=np.float32),
        "action": np.zeros((2, 3, 2), dtype=np.float32),
    }

    before = _params_leaves(agent.state_dict())
    metrics = agent.update(iter([batch]), step=0)
    after = _params_leaves(agent.state_dict())

    assert np.isfinite(metrics["actor_loss"])
    assert agent.ema_params is None
    assert any(not np.allclose(a, b) for a, b in zip(before, after))


def test_flow_matching_update_accepts_beta_train_time_schedule():
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
    agent = _make_flow_matching(
        observation_space=observation_space,
        action_space=action_space,
        jit=False,
        train_time_schedule="beta_0p5_0p5",
    )
    agent.logging = True
    batch = {
        "low_dim_state": np.zeros((2, 1, 4), dtype=np.float32),
        "action": np.zeros((2, 3, 2), dtype=np.float32),
    }

    metrics = agent.update(iter([batch]), step=0)

    assert agent.train_time_schedule == "beta_0p5_0p5"
    assert np.isfinite(metrics["actor_loss"])


def test_flow_matching_all_valid_padding_mask_matches_unmasked_loss():
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
    agent = _make_flow_matching(
        observation_space=observation_space,
        action_space=action_space,
        jit=False,
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


@pytest.mark.parametrize(
    ("backbone_type", "expected_shape", "expected_condition_dim"),
    [
        ("transformer", (2, 2, 4), 4),
        ("dit", (2, 8), 8),
    ],
)
def test_flow_matching_multiframe_state_condition_layout(
    backbone_type,
    expected_shape,
    expected_condition_dim,
):
    observation_space = spaces.Dict(
        {
            "low_dim_state": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2, 4),
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
    agent = _make_flow_matching(
        observation_space=observation_space,
        action_space=action_space,
        jit=False,
        backbone_type=backbone_type,
    )
    batch = {
        "low_dim_state": np.zeros((2, 2, 4), dtype=np.float32),
        "action": np.zeros((2, 4, 2), dtype=np.float32),
    }

    obs_features, _ = agent._prepare_obs_features(batch)
    metrics = agent.update(iter([batch]), step=0)

    assert obs_features.shape == expected_shape
    assert agent.actor_model.condition_dim == expected_condition_dim
    assert metrics == {}
    assert agent.ema_params is None


def test_flow_matching_multiframe_transformer_repeats_language_per_state_token():
    observation_space = spaces.Dict(
        {
            "low_dim_state": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2, 4),
                dtype=np.float32,
            ),
            "lang_features": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(2, 1, 3),
                dtype=np.float32,
            ),
        }
    )
    action_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(4, 2),
        dtype=np.float32,
    )
    agent = _make_flow_matching(
        observation_space=observation_space,
        action_space=action_space,
        jit=False,
        backbone_type="transformer",
        use_lang_cond=True,
        lang_feature_dim=3,
    )
    batch = {
        "low_dim_state": np.zeros((2, 2, 4), dtype=np.float32),
        "lang_features": np.ones((2, 2, 1, 3), dtype=np.float32),
        "action": np.zeros((2, 4, 2), dtype=np.float32),
    }

    obs_features, _ = agent._prepare_obs_features(batch)
    metrics = agent.update(iter([batch]), step=0)

    assert obs_features.shape == (2, 2, 7)
    assert agent.actor_model.condition_dim == 7
    assert metrics == {}


def test_flow_matching_update_accepts_horizon_dropout():
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
    agent = _make_flow_matching(
        observation_space=observation_space,
        action_space=action_space,
        jit=False,
        horizon_dropout_lengths=(1, 2, 4),
        horizon_dropout_probs=(0.25, 0.25, 0.5),
    )
    agent.logging = True
    batch = {
        "low_dim_state": np.zeros((2, 1, 4), dtype=np.float32),
        "action": np.zeros((2, 4, 2), dtype=np.float32),
    }

    metrics = agent.update(iter([batch]), step=0)

    assert np.isfinite(metrics["actor_loss"])


def test_flow_matching_update_accepts_horizon_loss_weights():
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
    agent = _make_flow_matching(
        observation_space=observation_space,
        action_space=action_space,
        jit=False,
        horizon_loss_weights=(0.5, 0.25, 0.15, 0.1),
    )
    agent.logging = True
    batch = {
        "low_dim_state": np.zeros((2, 1, 4), dtype=np.float32),
        "action": np.zeros((2, 4, 2), dtype=np.float32),
    }

    metrics = agent.update(iter([batch]), step=0)

    assert np.isfinite(metrics["actor_loss"])


def test_flow_matching_update_many_fuses_multiple_updates():
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
    agent = _make_flow_matching(
        observation_space=observation_space,
        action_space=action_space,
        jit=False,
    )
    agent.logging = True
    batch = {
        "low_dim_state": np.zeros((2, 1, 4), dtype=np.float32),
        "action": np.zeros((2, 3, 2), dtype=np.float32),
    }

    before = _params_leaves(agent.state_dict())
    metrics = agent.update_many(iter([batch, batch, batch]), num_updates=3)
    after = _params_leaves(agent.state_dict())

    assert agent._update_step_count == 3
    assert agent.ema_params is None
    assert np.isfinite(metrics["actor_loss"])
    assert any(not np.allclose(a, b) for a, b in zip(before, after))


def test_flow_matching_act_returns_action_sequence():
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
    agent = _make_flow_matching(
        observation_space=observation_space,
        action_space=action_space,
        jit=False,
    )

    obs = {"low_dim_state": np.zeros((2, 1, 4), dtype=np.float32)}
    actions = agent.act(obs, step=0, eval_mode=False)

    assert actions.shape == (2, 3, 2)
    assert actions.dtype == np.float32
    assert np.all(actions >= -1.0)
    assert np.all(actions <= 1.0)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_flow_matching_act_sanitizes_nonfinite_output(monkeypatch, bad_value):
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
    agent = _make_flow_matching(
        observation_space=observation_space,
        action_space=action_space,
        jit=False,
    )

    def _bad_apply_with_shape(*args, **kwargs):
        del kwargs
        current_sample = args[1]
        return agent.jnp.full_like(current_sample, bad_value)

    monkeypatch.setattr(agent.actor_model, "apply", _bad_apply_with_shape)

    obs = {"low_dim_state": np.zeros((2, 1, 4), dtype=np.float32)}
    actions = agent.act(obs, step=0, eval_mode=False)

    assert np.isfinite(actions).all()
    assert np.all(actions >= -1.0)
    assert np.all(actions <= 1.0)


def test_flow_matching_workspace_smoke_and_snapshot(tmp_path):
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).resolve().parents[2] / "robobase/cfgs"),
        job_name="test_jax_flow_matching_workspace",
    ):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "backend=jax",
                "method=flow_matching",
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
                "method.num_flow_steps=3",
                "method.backbone.hidden_dims=[32]",
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
