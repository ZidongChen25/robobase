from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import flax.linen as nn
from flax.core import freeze, unfreeze
from gymnasium import spaces
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

import robobase.method.diffusion as diffusion_module
from robobase.method.bc import BCEncoderModelSpec, BCViewFusionModelSpec
from robobase.method.diffusion import Diffusion as JaxDiffusion
from robobase.envs.env import EnvFactory
from robobase.method.diffusion import (
    DiffusionActorModelSpec,
    DiffusionModelSpec,
    diffusion_spec_from_cfg,
)
from robobase.models.backbone import DiffusionBackboneSpec, build_diffusion_backbone
from robobase.models.backbones.common import (
    CleanDiffuserPosEmb,
    SinusoidalPosEmb,
)
from robobase.models.backbones.unet1d import Downsample1d, Upsample1d
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


def _make_jax_diffusion(*, observation_space, action_space, sampler="ddpm"):
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
        sampler=sampler,
    )


def test_diffusion_sampling_bounds_follow_action_normalization_space():
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
    standardized_actions = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(4, 2),
        dtype=np.float32,
    )
    min_max_actions = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(4, 2),
        dtype=np.float32,
    )

    standardized_agent = _make_jax_diffusion(
        observation_space=observation_space,
        action_space=standardized_actions,
    )
    bounded_agent = _make_jax_diffusion(
        observation_space=observation_space,
        action_space=min_max_actions,
    )

    assert standardized_agent._sample_clip_bounds is None
    assert bounded_agent._sample_clip_bounds is not None
    lower, upper = bounded_agent._sample_clip_bounds
    np.testing.assert_array_equal(lower, -np.ones((4, 2), dtype=np.float32))
    np.testing.assert_array_equal(upper, np.ones((4, 2), dtype=np.float32))


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


def test_unet_downsample_matches_torch_conv1d_padding_one():
    module = Downsample1d(channels=1, operator_variant="torch")
    inputs = jax.numpy.zeros((1, 4, 1), dtype=jax.numpy.float32)
    inputs = inputs.at[0, 1, 0].set(1.0)
    variables = unfreeze(module.init(jax.random.PRNGKey(0), inputs))
    variables["params"]["conv"]["kernel"] = jax.numpy.asarray(
        [[[1.0]], [[2.0]], [[3.0]]]
    )
    variables["params"]["conv"]["bias"] = jax.numpy.zeros((1,))

    output = module.apply(freeze(variables), inputs)

    np.testing.assert_allclose(output[0, :, 0], [3.0, 1.0])


def test_unet_upsample_matches_torch_conv_transpose1d_padding_one():
    module = Upsample1d(channels=1, operator_variant="torch")
    inputs = jax.numpy.zeros((1, 4, 1), dtype=jax.numpy.float32)
    inputs = inputs.at[0, 1, 0].set(1.0)
    variables = unfreeze(module.init(jax.random.PRNGKey(0), inputs))
    variables["params"]["conv_transpose"]["kernel"] = jax.numpy.asarray(
        [[[1.0]], [[2.0]], [[3.0]], [[4.0]]]
    )
    variables["params"]["conv_transpose"]["bias"] = jax.numpy.zeros((1,))

    output = module.apply(freeze(variables), inputs)

    np.testing.assert_allclose(
        output[0, :, 0], [0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0]
    )


def test_campose_timestep_embedding_matches_reference_formula():
    timesteps = jax.numpy.asarray([0.0, 1.0, 37.5], dtype=jax.numpy.float32)
    embedding = SinusoidalPosEmb(dim=8)

    actual = embedding.apply({}, timesteps)

    frequencies = np.exp(-np.arange(4) * np.log(10000.0) / 3.0)
    phase = np.asarray(timesteps)[:, None] * frequencies[None, :]
    expected = np.concatenate([np.sin(phase), np.cos(phase)], axis=-1)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_clean_diffuser_timestep_embedding_matches_reference_formula():
    timesteps = jax.numpy.asarray([0.0, 1.0, 37.5], dtype=jax.numpy.float32)
    embedding = CleanDiffuserPosEmb(dim=8)

    actual = embedding.apply({}, timesteps)

    frequencies = (1.0 / 10000.0) ** (np.arange(4) / 4.0)
    phase = np.asarray(timesteps)[:, None] * frequencies[None, :]
    expected = np.concatenate([np.cos(phase), np.sin(phase)], axis=-1)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_unet_supports_local_conditioning_film_scale_and_global_adapter():
    spec = DiffusionBackboneSpec(
        type="unet1d",
        sequence_length=8,
        diffusion_step_embed_dim=8,
        down_dims=(16, 32, 64),
        kernel_size=3,
        n_groups=4,
        operator_variant="torch",
        conditioning_mode="local",
        cond_predict_scale=True,
        global_condition_embed_dim=8,
    )
    model = build_diffusion_backbone(
        spec,
        action_dim=2,
        sequence_length=8,
        condition_dim=5,
        local_condition_dim=3,
    )
    actions = jax.numpy.zeros((2, 8, 2), dtype=jax.numpy.float32)
    timesteps = jax.numpy.asarray([0.25, 0.75], dtype=jax.numpy.float32)
    global_condition = jax.numpy.ones((2, 5), dtype=jax.numpy.float32)
    local_condition = jax.numpy.ones((2, 8, 3), dtype=jax.numpy.float32)

    variables = model.init(
        jax.random.PRNGKey(0),
        actions,
        timesteps,
        global_condition,
        local_condition,
    )
    output = model.apply(
        variables,
        actions,
        timesteps,
        global_condition,
        local_condition,
    )

    assert output.shape == actions.shape
    assert variables["params"]["global_cond_dense"]["kernel"].shape == (5, 8)
    assert variables["params"]["down_0_res1"]["cond_dense"]["kernel"].shape[-1] == 32
    assert "local_res1" in variables["params"]
    assert "local_res2" in variables["params"]


def test_unet_local_condition_changes_prediction():
    spec = DiffusionBackboneSpec(
        type="unet1d",
        sequence_length=4,
        diffusion_step_embed_dim=8,
        down_dims=(8, 16),
        kernel_size=3,
        n_groups=4,
        conditioning_mode="local",
    )
    model = build_diffusion_backbone(
        spec,
        action_dim=2,
        sequence_length=4,
        condition_dim=0,
        local_condition_dim=3,
    )
    actions = jax.numpy.zeros((1, 4, 2), dtype=jax.numpy.float32)
    timesteps = jax.numpy.zeros((1,), dtype=jax.numpy.float32)
    local_zeros = jax.numpy.zeros((1, 4, 3), dtype=jax.numpy.float32)
    local_ones = jax.numpy.ones((1, 4, 3), dtype=jax.numpy.float32)
    variables = model.init(jax.random.PRNGKey(0), actions, timesteps, None, local_zeros)

    output_zeros = model.apply(variables, actions, timesteps, None, local_zeros)
    output_ones = model.apply(variables, actions, timesteps, None, local_ones)

    assert not np.allclose(output_zeros, output_ones)


def test_unet_initializers_match_pytorch_default_bounds_and_nonzero_biases():
    spec = DiffusionBackboneSpec(
        type="unet1d",
        sequence_length=4,
        diffusion_step_embed_dim=8,
        down_dims=(8, 16),
        kernel_size=3,
        n_groups=4,
        operator_variant="torch",
    )
    model = build_diffusion_backbone(
        spec,
        action_dim=2,
        sequence_length=4,
        condition_dim=5,
    )
    variables = model.init(
        jax.random.PRNGKey(7),
        jax.numpy.zeros((1, 4, 2), dtype=jax.numpy.float32),
        jax.numpy.zeros((1,), dtype=jax.numpy.float32),
        jax.numpy.zeros((1, 5), dtype=jax.numpy.float32),
    )
    params = variables["params"]

    time_bias = np.asarray(params["time_dense1"]["bias"])
    down_bias = np.asarray(params["down_0_res1"]["block1"]["conv"]["bias"])
    up_bias = np.asarray(params["up_0_us"]["conv_transpose"]["bias"])
    down_kernel = np.asarray(params["down_0_res1"]["block1"]["conv"]["kernel"])

    assert np.any(time_bias != 0.0)
    assert np.any(down_bias != 0.0)
    assert np.any(up_bias != 0.0)
    assert np.max(np.abs(time_bias)) <= 1.0 / np.sqrt(8) + 1e-7
    assert np.max(np.abs(down_bias)) <= 1.0 / np.sqrt(3 * 2) + 1e-7
    assert np.max(np.abs(down_kernel)) <= 1.0 / np.sqrt(3 * 2) + 1e-7
    assert np.max(np.abs(up_bias)) <= 1.0 / np.sqrt(4 * 8) + 1e-7


def test_unet_validates_horizon_against_configured_downsampling_depth():
    spec = DiffusionBackboneSpec(
        type="unet1d",
        sequence_length=4,
        down_dims=(8, 16, 32, 64),
    )

    with pytest.raises(ValueError, match="downsampling factor 8"):
        build_diffusion_backbone(
            spec,
            action_dim=2,
            sequence_length=4,
            condition_dim=3,
        )


def test_unet_legacy_spec_defaults_to_global_bias_conditioning():
    spec = DiffusionBackboneSpec(type="unet1d", sequence_length=4)

    assert spec.conditioning_mode == "global"
    assert spec.cond_predict_scale is False
    assert spec.global_condition_embed_dim == 0
    assert spec.timestep_embedding_type == "campose"
    assert spec.operator_variant == "legacy"
    assert spec.compatibility_mode == "native"


def test_clean_diffuser_unet_compatibility_accepts_canonical_topology():
    spec = DiffusionBackboneSpec(
        type="unet1d",
        sequence_length=16,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        conditioning_mode="global",
        cond_predict_scale=True,
        global_condition_embed_dim=256,
        timestep_embedding_type="clean_diffuser",
        operator_variant="torch",
        compatibility_mode="clean_diffuser",
    )

    model = build_diffusion_backbone(
        spec,
        action_dim=10,
        sequence_length=16,
        condition_dim=46,
    )

    assert model.output_shape == (16, 10)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"sequence_length": 12}, "power of two"),
        ({"diffusion_step_embed_dim": 255}, "must be even"),
        ({"n_groups": 4}, "n_groups must be 8"),
        ({"down_dims": (8, 16, 32)}, "selects 2 CleanDiffuser groups"),
        ({"operator_variant": "legacy"}, "operator_variant must be torch"),
    ],
)
def test_clean_diffuser_unet_compatibility_rejects_silent_mismatches(
    overrides, message
):
    values = {
        "type": "unet1d",
        "sequence_length": 16,
        "diffusion_step_embed_dim": 256,
        "down_dims": (256, 512, 1024),
        "kernel_size": 5,
        "n_groups": 8,
        "conditioning_mode": "global",
        "cond_predict_scale": True,
        "global_condition_embed_dim": 256,
        "timestep_embedding_type": "clean_diffuser",
        "operator_variant": "torch",
        "compatibility_mode": "clean_diffuser",
    }
    values.update(overrides)
    spec = DiffusionBackboneSpec(**values)

    with pytest.raises(ValueError, match=message):
        build_diffusion_backbone(
            spec,
            action_dim=10,
            sequence_length=values["sequence_length"],
            condition_dim=46,
        )


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
    assert spec.adaptive_lr is False
    assert spec.model.resolved_backbone.conditioning_mode == "global"
    assert spec.model.resolved_backbone.cond_predict_scale is False
    assert spec.model.resolved_backbone.global_condition_embed_dim == 0
    assert spec.model.resolved_backbone.timestep_embedding_type == "campose"
    assert spec.model.encoder_model.type == "resnet"
    assert spec.model.encoder_model.plucker_fusion_mode is None
    assert spec.model.encoder_model.pretrained is True
    assert spec.model.encoder_model.trainable is False
    assert spec.image_augmentation_type == "none"
    assert spec.model.resolved_backbone.type == "fully_connected"
    assert spec.model.resolved_backbone.hidden_dims == (16,)


def test_campose_dp_profile_composes_complete_official_path():
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).resolve().parents[2] / "robobase/cfgs"),
        job_name="test_campose_dp_profile",
    ):
        cfg = compose(
            config_name="robobase_config",
            overrides=["backend=jax", "method=diffusion", "profile=campose_dp"],
        )

    try:
        spec = diffusion_spec_from_cfg(cfg)
    finally:
        GlobalHydra.instance().clear()

    assert cfg.pixels is True
    assert cfg.visual_observation_shape == [256, 256]
    assert cfg.frame_stack == 1
    assert cfg.action_sequence == 32
    assert cfg.batch_size == 70
    assert cfg.replay.nstep == 1
    assert cfg.use_min_max_normalization is True
    assert cfg.norm_obs is True
    assert cfg.obs_norm_type == "min_max"
    assert spec.lr == pytest.approx(2e-5)
    assert spec.adaptive_lr is False
    assert spec.actor_grad_clip == pytest.approx(1.0)
    assert spec.weight_decay == pytest.approx(1e-4)
    assert spec.image_augmentation_type == "campose_crop"
    assert spec.mask_padded_model_input is False
    assert spec.sampler == "ddpm"
    assert spec.model.resolved_backbone.down_dims == (512, 1024, 2048)
    assert spec.model.resolved_backbone.diffusion_step_embed_dim == 128
    assert spec.model.resolved_backbone.conditioning_mode == "local"
    assert spec.model.resolved_backbone.cond_predict_scale is True
    assert spec.model.resolved_backbone.operator_variant == "torch"
    assert cfg.env.cameras == ["head", "left_wrist"]
    assert cfg.method.proprio_dropout_stage == "raw"
    assert cfg.method.proprio_dropout_prob == pytest.approx(1.0)
    assert spec.model.encoder_model.type == "dp_resnet"
    assert spec.model.encoder_model.use_plucker is True
    assert spec.model.encoder_model.plucker_fusion_mode == "dp_early"


def test_clean_diffuser_profile_composes_matched_benchmark_path():
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).resolve().parents[2] / "robobase/cfgs"),
        job_name="test_clean_diffuser_profile",
    ):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "backend=jax",
                "method=diffusion",
                "profile=clean_diffuser_ddpm",
            ],
        )

    try:
        spec = diffusion_spec_from_cfg(cfg)
    finally:
        GlobalHydra.instance().clear()

    assert cfg.pixels is False
    assert cfg.frame_stack == 2
    assert cfg.action_sequence == 16
    assert cfg.replay.nstep == 1
    assert spec.adaptive_lr is False
    assert spec.use_ema is True
    assert spec.ema_decay == pytest.approx(0.995)
    assert spec.ema_decay_schedule == "constant"
    assert spec.model.resolved_backbone.operator_variant == "torch"
    assert spec.model.resolved_backbone.timestep_embedding_type == "clean_diffuser"
    assert spec.model.resolved_backbone.global_condition_embed_dim == 256
    assert spec.model.resolved_backbone.compatibility_mode == "clean_diffuser"
    assert spec.model.encoder_model is None


def test_clean_diffuser_dp_robomimic_launch_composes_training_recipe():
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).resolve().parents[2] / "robobase/cfgs"),
        job_name="test_clean_diffuser_dp_launch",
    ):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "backend=jax",
                "launch=clean_diffuser_dp_state_robomimic",
                "env=robomimic_clean/tool_hang",
            ],
        )

    try:
        spec = diffusion_spec_from_cfg(cfg)
    finally:
        GlobalHydra.instance().clear()

    assert cfg.seed == 0
    assert cfg.num_pretrain_steps == 1_000_000
    assert cfg.frame_stack == 2
    assert cfg.action_sequence == 16
    assert cfg.action_execution_start == 1
    assert cfg.execution_length == 8
    assert cfg.replay.action_sequence_start_offset == 1
    assert cfg.replay.action_padding == "edge"
    assert cfg.env.filter_key == "all"
    assert cfg.env.abs_action is True
    assert cfg.env.obs_keys == [
        "object",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    ]
    assert cfg.obs_min_max_constant_value == pytest.approx(-1.0)
    assert spec.adaptive_lr is True
    assert spec.lr_schedule == "cosine"
    assert spec.num_diffusion_iters == 50
    assert spec.sampler == "ddpm"
    assert spec.use_ema is True
    assert spec.ema_decay == pytest.approx(0.995)
    assert spec.ema_decay_schedule == "constant"
    assert spec.weight_decay == pytest.approx(1e-2)
    assert spec.model.resolved_backbone.compatibility_mode == "clean_diffuser"
    assert spec.model.encoder_model is None


@pytest.mark.parametrize(
    ("env_name", "task_name", "episode_length"),
    [
        ("lift", "Lift", 400),
        ("can", "Can", 400),
        ("square", "Square", 500),
        ("tool_hang", "ToolHang", 700),
        ("transport", "TwoArmTransport", 700),
    ],
)
def test_clean_diffuser_dp_official_robomimic_task_modules(
    env_name, task_name, episode_length
):
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).resolve().parents[2] / "robobase/cfgs"),
        job_name="test_clean_diffuser_task_module",
    ):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "backend=jax",
                "launch=clean_diffuser_dp_state_robomimic",
                f"env=robomimic_clean/{env_name}",
            ],
        )
    GlobalHydra.instance().clear()

    assert cfg.env.task_name == task_name
    assert cfg.env.episode_length == episode_length
    assert cfg.env.use_live_env is True
    assert cfg.env.abs_action is True
    assert cfg.env.filter_key == "all"
    assert cfg.env.obs_keys[0] == "object"
    assert len(cfg.env.obs_keys) == (7 if env_name == "transport" else 4)


@pytest.mark.parametrize(
    ("profile", "expected_embedding", "expected_adapter", "expected_dims"),
    [
        ("campose_dp_unet", "campose", 0, (512, 1024, 2048)),
        (
            "clean_diffuser_chi_unet",
            "clean_diffuser",
            256,
            (256, 512, 1024),
        ),
    ],
)
def test_diffusion_unet_profile_composes_expected_architecture(
    profile,
    expected_embedding,
    expected_adapter,
    expected_dims,
):
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).resolve().parents[2] / "robobase/cfgs"),
        job_name="test_diffusion_unet_profile",
    ):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "backend=jax",
                "method=diffusion",
                "env=dmc/cartpole_balance",
                "pixels=false",
                "action_sequence=8",
                f"backbone={profile}",
            ],
        )

    try:
        spec = diffusion_spec_from_cfg(cfg).model.resolved_backbone
    finally:
        GlobalHydra.instance().clear()

    assert spec.type == "unet1d"
    assert spec.timestep_embedding_type == expected_embedding
    assert spec.operator_variant == "torch"
    assert spec.global_condition_embed_dim == expected_adapter
    assert spec.down_dims == expected_dims


def test_diffusion_dp_early_plucker_preset_sets_official_training_flags():
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).resolve().parents[2] / "robobase/cfgs"),
        job_name="test_dp_early_plucker_preset",
    ):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "backend=jax",
                "method=diffusion",
                "env=dmc/cartpole_balance",
                "pixels=false",
                "action_sequence=4",
                "encoder=dp_early_plucker",
            ],
        )

    try:
        spec = diffusion_spec_from_cfg(cfg)
    finally:
        GlobalHydra.instance().clear()

    assert spec.model.encoder_model.use_plucker is True
    assert spec.model.encoder_model.type == "dp_resnet"
    assert spec.model.encoder_model.plucker_fusion_mode == "dp_early"
    assert spec.model.encoder_model.pretrained is False
    assert spec.model.encoder_model.trainable is True


def test_diffusion_dp_encoder_family_is_preserved_for_rgb_ablation(monkeypatch):
    captured = {}

    class _FakeDPEarlyEncoder:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.output_shape = (kwargs["input_shape"][0], 64)

    monkeypatch.setattr(
        diffusion_module, "JaxDPEarlyFusionEncoder", _FakeDPEarlyEncoder
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
        }
    )
    model_spec = DiffusionModelSpec(
        actor_model=DiffusionActorModelSpec(
            type="unet1d", sequence_length=4, down_dims=(8, 16)
        ),
        encoder_model=BCEncoderModelSpec(
            type="dp_resnet",
            model="resnet18",
            trainable=True,
            pretrained=False,
            use_plucker=False,
            plucker_fusion_mode="dp_early",
        ),
        view_fusion_model=None,
    )

    encoder, _, _ = diffusion_module._build_encoder_and_fusion(
        model_spec=model_spec,
        observation_space=observation_space,
        encoder_jit=False,
    )

    assert isinstance(encoder, _FakeDPEarlyEncoder)
    assert captured["use_plucker"] is False
    assert captured["plucker_fusion_mode"] == "none"


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


def test_jax_diffusion_local_unet_update_and_sample_are_jittable():
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
    model = DiffusionModelSpec(
        actor_model=DiffusionActorModelSpec(
            type="unet1d",
            sequence_length=4,
            diffusion_step_embed_dim=16,
            down_dims=(16, 32),
            kernel_size=3,
            n_groups=4,
            conditioning_mode="local",
            cond_predict_scale=True,
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
        jit=True,
        seed=0,
        use_ema=False,
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


def test_jax_diffusion_supports_plucker_camera_params_with_trainable_encoder(
    monkeypatch,
):
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
    for before, after in zip(
        _params_leaves(saved_state), _params_leaves(restored_state)
    ):
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


@pytest.mark.parametrize("sampler", ["ddpm", "ddim"])
def test_jax_diffusion_perfect_epsilon_predictor_recovers_clean_sample(
    monkeypatch, sampler
):
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
        sampler=sampler,
    )
    target = jax.numpy.full((2, 4, 2), 0.2, dtype=jax.numpy.float32)

    def perfect_epsilon(_params, current_sample, timesteps, _features):
        alpha = agent.alphas_cumprod[timesteps][:, None, None]
        return (current_sample - jax.numpy.sqrt(alpha) * target) / jax.numpy.sqrt(
            1.0 - alpha
        )

    monkeypatch.setattr(agent.actor_model, "apply", perfect_epsilon)
    obs = {"low_dim_state": np.zeros((2, 1, 4), dtype=np.float32)}
    obs_features, _ = agent._prepare_obs_features(obs)

    output = agent._sample_impl(agent.params, jax.random.PRNGKey(123), obs_features)

    np.testing.assert_allclose(output, target, rtol=1e-5, atol=1e-5)


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
