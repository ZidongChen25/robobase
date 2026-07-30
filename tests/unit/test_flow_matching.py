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
    _rectified_flow_training_pair,
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
    dropout=0.0,
    conditioning_mode="global",
    cond_predict_scale=False,
    global_condition_embed_dim=0,
    time_scale=1.0,
    num_flow_steps=3,
    ema_decay=0.9999,
    ema_decay_schedule="diffusers",
):
    model = FlowMatchingModelSpec(
        backbone=FlowMatchingBackboneSpec(
            type=backbone_type,
            sequence_length=action_space.shape[0],
            diffusion_step_embed_dim=16,
            down_dims=(16, 32),
            kernel_size=3,
            n_groups=4,
            hidden_dims=(32,),
            d_model=32,
            n_heads=4,
            num_layers=1,
            n_cond_layers=1,
            depth=1,
            dropout=dropout,
            conditioning_mode=conditioning_mode,
            cond_predict_scale=cond_predict_scale,
            global_condition_embed_dim=global_condition_embed_dim,
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
        num_flow_steps=num_flow_steps,
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
        ema_decay=ema_decay,
        ema_decay_schedule=ema_decay_schedule,
        train_time_schedule=train_time_schedule,
        time_scale=time_scale,
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

    @property
    def trainable_params(self):
        return {"kernel": self.value}


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


def test_flow_matching_load_rejects_incompatible_parameter_shapes():
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
    state = agent.state_dict()
    leaves, tree = jax.tree_util.tree_flatten(state["params"])
    incompatible = list(leaves)
    incompatible[0] = np.concatenate(
        [np.asarray(incompatible[0]), np.asarray(incompatible[0])],
        axis=0,
    )
    state["params"] = jax.tree_util.tree_unflatten(tree, incompatible)

    with pytest.raises(ValueError, match="parameter shape mismatch"):
        agent.load_state_dict(state)


def test_flow_matching_load_is_transactional_when_ema_is_incompatible():
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
    agent.use_ema = True
    agent.ema_params = agent.params
    original_params = jax.tree.map(lambda value: np.asarray(value).copy(), agent.params)
    state = agent.state_dict()
    state["params"] = jax.tree.map(
        lambda value: np.asarray(value) + 1.0,
        state["params"],
    )
    leaves, tree = jax.tree_util.tree_flatten(state["_ema_params"])
    incompatible = list(leaves)
    incompatible[0] = np.concatenate(
        [np.asarray(incompatible[0]), np.asarray(incompatible[0])],
        axis=0,
    )
    state["_ema_params"] = jax.tree_util.tree_unflatten(tree, incompatible)

    with pytest.raises(ValueError, match="EMA checkpoint parameter shape mismatch"):
        agent.load_state_dict(state)

    for original, current in zip(
        jax.tree_util.tree_leaves(original_params),
        jax.tree_util.tree_leaves(agent.params),
        strict=True,
    ):
        np.testing.assert_array_equal(original, np.asarray(current))


def test_flow_matching_load_rejects_incompatible_encoder_frozen_state():
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
    agent.encoder = _FakeFrozenEncoder(3.0)
    state = agent.state_dict()
    state["_encoder_frozen_state"]["batch_stats"]["value"] = np.zeros(
        (2,), dtype=np.float32
    )

    with pytest.raises(
        ValueError,
        match="Encoder frozen-state checkpoint parameter shape mismatch",
    ):
        agent.load_state_dict(state)

    np.testing.assert_array_equal(np.asarray(agent.encoder.value), np.asarray([3.0]))


@pytest.mark.parametrize("use_ema", [False, True])
def test_flow_matching_migrates_actor_only_checkpoint_for_trainable_encoder(use_ema):
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
    actor_only = agent._tree_to_numpy(agent.params)
    agent.encoder = _FakeFrozenEncoder(3.0)
    agent._trainable_encoder = True
    agent.use_ema = use_ema
    state = {"params": actor_only}
    if use_ema:
        state["_ema_params"] = actor_only

    agent.load_state_dict(state)

    assert set(agent.params) == {"actor", "encoder"}
    np.testing.assert_array_equal(
        np.asarray(agent.params["encoder"]["kernel"]), np.asarray([3.0])
    )
    if use_ema:
        assert set(agent.ema_params) == {"actor", "encoder"}


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

    monkeypatch.setattr(
        "robobase.method.flow_matching.tokens_to_feature_jax", fail_token_path
    )
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
    assert spec.adaptive_lr is False
    assert spec.train_time_schedule == "beta_0p5_0p5"
    assert spec.time_scale == pytest.approx(1000.0)
    assert spec.horizon_dropout_lengths == (1, 2, 4)
    assert spec.horizon_dropout_probs == (0.2, 0.3, 0.5)
    assert spec.horizon_loss_weights == (0.5, 0.25, 0.15, 0.1)
    assert spec.model.backbone.type == "transformer"
    assert spec.model.backbone.d_model == 32
    assert spec.model.backbone.conditioning_mode == "global"
    assert spec.model.backbone.cond_predict_scale is False
    assert spec.model.backbone.global_condition_embed_dim == 0
    assert spec.model.backbone.timestep_embedding_type == "campose"
    assert spec.model.encoder_model.type == "resnet"
    assert spec.model.encoder_model.plucker_fusion_mode is None
    assert spec.model.encoder_model.pretrained is True
    assert spec.model.encoder_model.trainable is False
    assert spec.image_augmentation_type == "none"
    assert spec.ema_decay_schedule == "diffusers"


def test_campose_fm_profile_composes_continuous_plucker_path():
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).resolve().parents[2] / "robobase/cfgs"),
        job_name="test_campose_fm_profile",
    ):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "backend=jax",
                "method=flow_matching",
                "profile=campose_fm",
            ],
        )

    try:
        spec = flow_matching_spec_from_cfg(cfg)
    finally:
        GlobalHydra.instance().clear()

    assert cfg.pixels is True
    assert cfg.visual_observation_shape == [256, 256]
    assert cfg.frame_stack == 1
    assert cfg.action_sequence == 32
    assert cfg.batch_size == 70
    assert cfg.env.cameras == ["head", "left_wrist"]
    assert cfg.replay.nstep == 1
    assert cfg.use_min_max_normalization is True
    assert cfg.obs_norm_type == "min_max"
    assert spec.objective_type == "rectified_flow"
    assert spec.sampler == "euler"
    assert spec.time_scale == pytest.approx(1.0)
    assert spec.image_augmentation_type == "campose_crop"
    assert spec.adaptive_lr is False
    assert spec.ema_decay_schedule == "constant"
    assert spec.model.backbone.conditioning_mode == "local"
    assert spec.model.backbone.operator_variant == "torch"
    assert spec.model.encoder_model.type == "dp_resnet"
    assert spec.model.encoder_model.use_plucker is True
    assert spec.model.encoder_model.plucker_fusion_mode == "dp_early"
    assert cfg.method.proprio_dropout_stage == "raw"
    assert cfg.method.proprio_dropout_prob == pytest.approx(1.0)


def test_clean_diffuser_rf_profile_composes_matched_core():
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).resolve().parents[2] / "robobase/cfgs"),
        job_name="test_clean_diffuser_rf_profile",
    ):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "backend=jax",
                "method=flow_matching",
                "profile=clean_diffuser_rf",
            ],
        )

    try:
        spec = flow_matching_spec_from_cfg(cfg)
    finally:
        GlobalHydra.instance().clear()

    assert cfg.pixels is False
    assert cfg.frame_stack == 2
    assert cfg.action_sequence == 16
    assert spec.num_flow_steps == 10
    assert spec.sampler == "euler"
    assert spec.sample_schedule == "uniform"
    assert spec.train_time_schedule == "uniform"
    assert spec.time_scale == pytest.approx(1.0)
    assert spec.use_ema is True
    assert spec.ema_decay == pytest.approx(0.995)
    assert spec.ema_decay_schedule == "constant"
    assert spec.model.backbone.compatibility_mode == "clean_diffuser"
    assert spec.model.encoder_model is None


def test_clean_diffuser_rf_robomimic_launch_composes_matched_protocol():
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).resolve().parents[2] / "robobase/cfgs"),
        job_name="test_clean_diffuser_rf_launch",
    ):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "backend=jax",
                "launch=clean_diffuser_rf_state_robomimic",
                "env=robomimic_clean/transport",
            ],
        )

    try:
        spec = flow_matching_spec_from_cfg(cfg)
    finally:
        GlobalHydra.instance().clear()

    assert cfg.frame_stack == 2
    assert cfg.action_execution_start == 1
    assert cfg.replay.action_sequence_start_offset == 1
    assert cfg.replay.action_padding == "edge"
    assert len(cfg.env.obs_keys) == 7
    assert spec.adaptive_lr is True
    assert spec.lr_schedule == "cosine"
    assert spec.num_flow_steps == 10
    assert spec.time_scale == pytest.approx(1.0)
    assert spec.ema_decay_schedule == "constant"
    assert spec.model.backbone.compatibility_mode == "clean_diffuser"


def test_clean_diffuser_rf_launch_preserves_task_episode_limit():
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).resolve().parents[2] / "robobase/cfgs"),
        job_name="test_clean_diffuser_rf_task_limit",
    ):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "backend=jax",
                "launch=clean_diffuser_rf_state_robomimic",
                "env=robomimic_clean/lift",
            ],
        )
    GlobalHydra.instance().clear()

    assert cfg.env.task_name == "Lift"
    assert cfg.env.episode_length == 400
    assert cfg.env.use_live_env is True


def test_flow_matching_constant_ema_uses_configured_decay_from_first_step():
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
    action_space = spaces.Box(-1.0, 1.0, shape=(3, 2), dtype=np.float32)
    agent = _make_flow_matching(
        observation_space=observation_space,
        action_space=action_space,
        jit=False,
        ema_decay=0.995,
        ema_decay_schedule="constant",
    )

    np.testing.assert_allclose(agent._ema_decay_value(1), 0.995)
    np.testing.assert_allclose(agent._ema_decay_value(100), 0.995)


def test_rectified_flow_interpolation_and_reverse_velocity_match_clean_diffuser():
    actions = jax.numpy.asarray([[[2.0]], [[2.0]], [[2.0]]], dtype=jax.numpy.float32)
    source_noise = jax.numpy.asarray(
        [[[-1.0]], [[-1.0]], [[-1.0]]], dtype=jax.numpy.float32
    )
    time = jax.numpy.asarray([0.0, 0.25, 1.0], dtype=jax.numpy.float32)

    sample, reverse_velocity = _rectified_flow_training_pair(
        actions, source_noise, time
    )

    np.testing.assert_allclose(sample[:, 0, 0], [2.0, 1.25, -1.0])
    np.testing.assert_allclose(reverse_velocity[:, 0, 0], [3.0, 3.0, 3.0])


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


def test_flow_matching_local_unet_update_and_sample_are_jittable():
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
        jit=True,
        backbone_type="unet1d",
        conditioning_mode="local",
        cond_predict_scale=True,
    )
    batch = {
        "low_dim_state": np.zeros((2, 1, 4), dtype=np.float32),
        "action": np.zeros((2, 4, 2), dtype=np.float32),
    }

    metrics = agent.update(iter([batch]), step=0)
    actions = agent.act(batch, step=0, eval_mode=False)

    assert metrics == {}
    assert actions.shape == (2, 4, 2)
    assert np.isfinite(actions).all()


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


def test_flow_matching_masked_action_values_cannot_leak_into_valid_loss():
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
        backbone_type="unet1d",
    )
    batch = {
        "low_dim_state": np.zeros((2, 1, 4), dtype=np.float32),
        "action": np.zeros((2, 4, 2), dtype=np.float32),
    }
    changed_actions = np.asarray(batch["action"]).copy()
    changed_actions[:, 2:] = 100.0
    action_pad_mask = jax.numpy.asarray(
        [[False, False, True, True], [False, False, True, True]]
    )
    obs_features, _ = agent._prepare_obs_features(batch)
    common_args = (
        agent.params,
        agent.opt_state,
        agent.rng_key,
        obs_features,
    )
    loss_coeff = jax.numpy.ones((2,), dtype=jax.numpy.float32)

    # This is an exact masking invariant. Avoid accelerator-default TF32
    # variation between two otherwise identical UNet executions.
    with jax.default_matmul_precision("highest"):
        base_loss = agent._update_impl(
            *common_args,
            jax.numpy.asarray(batch["action"]),
            loss_coeff,
            action_pad_mask,
            None,
            0,
        )[3]
        changed_loss = agent._update_impl(
            *common_args,
            jax.numpy.asarray(changed_actions),
            loss_coeff,
            action_pad_mask,
            None,
            0,
        )[3]

    np.testing.assert_allclose(changed_loss, base_loss, rtol=1e-6, atol=1e-6)


def test_flow_matching_euler_uses_raw_continuous_time_in_reverse_order(monkeypatch):
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
        num_flow_steps=2,
        time_scale=1.0,
    )

    def time_velocity(_params, sample, timesteps, _obs_features):
        return jax.numpy.broadcast_to(timesteps[:, None, None], sample.shape)

    monkeypatch.setattr(agent, "_apply_actor", time_velocity)
    sample = jax.numpy.zeros((1, 3, 2), dtype=jax.numpy.float32)
    obs_features = jax.numpy.zeros((1, 4), dtype=jax.numpy.float32)
    schedule = jax.numpy.asarray([0.0, 0.5, 1.0], dtype=jax.numpy.float32)

    output = agent._integrate_sample(
        agent.params,
        sample,
        obs_features,
        schedule,
        agent._time_scale,
    )

    # Backward denoising evaluates t=1 then t=0.5: 0.5*1 + 0.5*0.5.
    np.testing.assert_allclose(output, 0.75, rtol=1e-6, atol=1e-6)


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


def test_flow_matching_transformer_apply_honors_training_dropout():
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
        backbone_type="transformer",
        dropout=0.5,
    )
    actions = jax.numpy.zeros((2, 4, 2), dtype=jax.numpy.float32)
    timesteps = jax.numpy.full((2,), 0.5, dtype=jax.numpy.float32)
    obs_features = jax.numpy.zeros((2, 2, 4), dtype=jax.numpy.float32)

    train_a = agent._apply_actor(
        agent.params,
        actions,
        timesteps,
        obs_features,
        train=True,
        dropout_key=jax.random.PRNGKey(1),
    )
    train_b = agent._apply_actor(
        agent.params,
        actions,
        timesteps,
        obs_features,
        train=True,
        dropout_key=jax.random.PRNGKey(2),
    )
    eval_a = agent._apply_actor(agent.params, actions, timesteps, obs_features)
    eval_b = agent._apply_actor(agent.params, actions, timesteps, obs_features)

    assert not np.allclose(train_a, train_b)
    np.testing.assert_allclose(eval_a, eval_b, rtol=0.0, atol=0.0)


def test_flow_matching_local_unet_logs_pytree_observations():
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
        backbone_type="unet1d",
        conditioning_mode="local",
    )
    agent.logging = True
    batch = {
        "low_dim_state": np.zeros((2, 2, 4), dtype=np.float32),
        "action": np.zeros((2, 4, 2), dtype=np.float32),
    }

    metrics = agent.update(iter([batch]), step=0)

    assert np.isfinite(metrics["actor_loss"])
    assert metrics["backend/update_steps_per_second"] > 0.0


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
    for before, after in zip(
        _params_leaves(saved_state), _params_leaves(restored_state)
    ):
        assert np.allclose(before, after)
