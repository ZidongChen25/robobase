from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import flax.linen as nn
from flax.core import freeze, unfreeze
from gymnasium import spaces
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from robobase.envs.env import EnvFactory
from robobase.method.bc import BCEncoderModelSpec, BCViewFusionModelSpec
from robobase.method.act import (
    ACT,
    ACTActorModelSpec,
    ACTModelSpec,
    _optimizer_labels,
    act_model_spec_from_cfg,
    act_spec_from_cfg,
)
from robobase.models.act import (
    ACTDetrTransformer,
    ACTImageProjection,
    JaxACTPolicy,
    _assemble_campose_conditioning_memory,
)
from robobase.workspace import Workspace

jax = pytest.importorskip("jax")
optax = pytest.importorskip("optax")


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


def _make_act(
    *,
    observation_space,
    action_space,
    jit=False,
    horizon_dropout_lengths=None,
    horizon_dropout_probs=None,
    gripper_dims=0,
    image_augmentation_type="legacy",
    data_augmentation=True,
    use_camera_extrinsics=False,
    num_camera_extrinsics=2,
    encoder_model=None,
):
    model = ACTModelSpec(
        actor_model=ACTActorModelSpec(
            type="transformer",
            hidden_dim=32,
            enc_layers=1,
            dec_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            nheads=4,
            num_queries=action_space.shape[0],
            pre_norm=False,
            gripper_dims=gripper_dims,
            image_augmentation_type=image_augmentation_type,
            data_augmentation=data_augmentation,
            use_camera_extrinsics=use_camera_extrinsics,
            num_camera_extrinsics=num_camera_extrinsics,
        ),
        encoder_model=encoder_model,
        view_fusion_model=None,
    )
    return ACT(
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
        horizon_dropout_lengths=horizon_dropout_lengths,
        horizon_dropout_probs=horizon_dropout_probs,
        jit=jit,
        seed=0,
    )


class _FakeResNetFeatureModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        return jax.numpy.zeros((x.shape[0], 1, 1, 512), dtype=jax.numpy.float32)


def _fake_resnet_feature_model():
    return _FakeResNetFeatureModel(), freeze({}), 512


def test_act_config_uses_jax_transformer_spec():
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).resolve().parents[2] / "robobase/cfgs"),
        job_name="test_jax_act_config",
    ):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "backend=jax",
                "method=act",
                "env=dmc/cartpole_balance",
                "pixels=false",
                "action_sequence=4",
                "method.actor_model.hidden_dim=32",
                "method.actor_model.nheads=4",
                "method.actor_model.dim_feedforward=64",
                "method.horizon_dropout_lengths=[1,2,4]",
                "method.horizon_dropout_probs=[0.2,0.3,0.5]",
            ],
        )

    try:
        spec = act_spec_from_cfg(cfg)
    finally:
        GlobalHydra.instance().clear()

    assert spec.model.actor_model.type == "transformer"
    assert spec.model.actor_model.num_queries == 4
    assert spec.model.actor_model.hidden_dim == 32
    assert spec.model.actor_model.gripper_dims == 0
    assert spec.model.actor_model.image_position_embedding_type == "sincos_legacy"
    assert spec.model.actor_model.image_augmentation_type == "legacy"
    assert spec.model.actor_model.use_camera_extrinsics is False
    assert spec.model.actor_model.num_camera_extrinsics == 2
    assert spec.model.actor_model.image_position_max_tokens is None
    assert spec.model.actor_model.proprio_dropout_prob == pytest.approx(0.0)
    assert spec.model.actor_model.proprio_projection_type == "legacy_mlp"
    assert spec.model.actor_model.style_cls_type == "learned"
    assert spec.model.actor_model.decoder_output_layer == "final"
    assert spec.model.actor_model.use_remat is True
    assert spec.model.encoder_model.plucker_fusion_mode is None
    assert spec.weight_decay == pytest.approx(1e-4)
    assert spec.horizon_dropout_lengths == (1, 2, 4)
    assert spec.horizon_dropout_probs == (0.2, 0.3, 0.5)


def test_act_model_config_parses_campose_conditioning_fields():
    cfg = OmegaConf.create(
        {
            "action_sequence": 4,
            "method": {
                "actor_model": {
                    "use_camera_extrinsics": True,
                    "num_camera_extrinsics": 3,
                    "image_augmentation_type": "campose_crop",
                    "image_position_max_tokens": 128,
                    "proprio_dropout_prob": 1.0,
                    "proprio_projection_type": "campose_single",
                    "style_cls_type": "zero",
                    "decoder_output_layer": "first",
                }
            },
        }
    )

    spec = act_model_spec_from_cfg(cfg).actor_model

    assert spec.use_camera_extrinsics is True
    assert spec.num_camera_extrinsics == 3
    assert spec.image_augmentation_type == "campose_crop"
    assert spec.image_position_max_tokens == 128
    assert spec.proprio_dropout_prob == pytest.approx(1.0)
    assert spec.proprio_projection_type == "campose_single"
    assert spec.style_cls_type == "zero"
    assert spec.decoder_output_layer == "first"
    # A frozen legacy config has no new key and retains its old memory policy.
    assert spec.use_remat is True


def test_act_remat_toggle_preserves_parameters_outputs_and_gradients():
    kwargs = dict(
        hidden_dim=8,
        nheads=2,
        enc_layers=1,
        dec_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        pre_norm=False,
    )
    remat = ACTDetrTransformer(**kwargs, use_remat=True)
    direct = ACTDetrTransformer(**kwargs, use_remat=False)
    src = jax.numpy.arange(24, dtype=jax.numpy.float32).reshape((1, 3, 8))
    query_embed = jax.numpy.arange(16, dtype=jax.numpy.float32).reshape((2, 8))
    pos_embed = jax.numpy.zeros_like(src)
    variables = remat.init(
        jax.random.PRNGKey(31),
        src,
        query_embed,
        pos_embed,
        deterministic=True,
    )

    def loss(module, params):
        output = module.apply(
            {"params": params},
            src,
            query_embed,
            pos_embed,
            deterministic=True,
        )
        return jax.numpy.sum(output**2)

    remat_output = remat.apply(
        variables, src, query_embed, pos_embed, deterministic=True
    )
    direct_output = direct.apply(
        variables, src, query_embed, pos_embed, deterministic=True
    )
    np.testing.assert_array_equal(remat_output, direct_output)
    remat_variables = remat.init(
        jax.random.PRNGKey(31), src, query_embed, pos_embed, deterministic=True
    )
    direct_variables = direct.init(
        jax.random.PRNGKey(31), src, query_embed, pos_embed, deterministic=True
    )
    assert jax.tree.structure(remat_variables) == jax.tree.structure(
        direct_variables
    )
    remat_grads = jax.grad(lambda p: loss(remat, p))(variables["params"])
    direct_grads = jax.grad(lambda p: loss(direct, p))(variables["params"])
    for remat_leaf, direct_leaf in zip(
        jax.tree.leaves(remat_grads),
        jax.tree.leaves(direct_grads),
        strict=True,
    ):
        np.testing.assert_allclose(remat_leaf, direct_leaf, rtol=1e-6, atol=1e-6)


def test_campose_act_profile_composes_official_conditioning_path():
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).resolve().parents[2] / "robobase/cfgs"),
        job_name="test_campose_act_profile",
    ):
        cfg = compose(
            config_name="robobase_config",
            overrides=["backend=jax", "method=act", "profile=campose_act"],
        )

    try:
        spec = act_spec_from_cfg(cfg)
    finally:
        GlobalHydra.instance().clear()

    actor = spec.model.actor_model
    assert cfg.pixels is True
    assert cfg.visual_observation_shape == [256, 256]
    assert cfg.frame_stack == 1
    assert cfg.action_sequence == 30
    assert cfg.batch_size == 70
    assert cfg.env.cameras == ["head", "left_wrist"]
    assert cfg.replay.nstep == 1
    assert cfg.use_standardization is True
    assert cfg.norm_obs is True
    assert actor.hidden_dim == 512
    assert actor.dec_layers == 7
    assert actor.pre_norm is True
    assert actor.image_augmentation_type == "campose_crop"
    assert actor.image_position_embedding_type == "campose_learned"
    assert actor.image_position_max_tokens == 128
    assert actor.use_camera_extrinsics is True
    assert actor.num_camera_extrinsics == 2
    assert actor.proprio_dropout_prob == pytest.approx(0.0)
    assert actor.proprio_projection_type == "campose_single"
    assert actor.style_cls_type == "zero"
    assert actor.decoder_output_layer == "first"
    assert spec.model.encoder_model.use_plucker is True
    assert spec.model.encoder_model.plucker_fusion_mode == "act_late"
    assert cfg.method.proprio_dropout_stage == "raw"
    assert cfg.method.proprio_dropout_prob == pytest.approx(1.0)


def test_act_optimizer_labels_only_backbone_subtrees_for_low_lr():
    params = {
        "actor": {"head": np.zeros((1,), dtype=np.float32)},
        "image_projection": {
            "input_proj": {"kernel": np.zeros((1,), dtype=np.float32)}
        },
        "encoder": {
            "layers_0": {
                "ConvBlock_0": {
                    "kernel": np.zeros((1,), dtype=np.float32),
                    "BatchNorm_0": {
                        "scale": np.ones((1,), dtype=np.float32),
                    },
                }
            },
            "resnet": {
                "layers_2": {
                    "kernel": np.zeros((1,), dtype=np.float32),
                    "ConvBlock_1": {
                        "BatchNorm_0": {
                            "bias": np.zeros((1,), dtype=np.float32),
                        }
                    },
                }
            },
            "film": {
                "text_proj": {"kernel": np.zeros((1,), dtype=np.float32)},
                "layer_1": {"kernel": np.zeros((1,), dtype=np.float32)},
            },
            "plucker": {"conv_0": {"kernel": np.zeros((1,), dtype=np.float32)}},
            "fusion": {"input_proj": {"kernel": np.zeros((1,), dtype=np.float32)}},
        },
    }

    labels = _optimizer_labels(params)

    assert labels["actor"]["head"] == "main"
    assert labels["image_projection"]["input_proj"]["kernel"] == "main"
    assert labels["encoder"]["layers_0"]["ConvBlock_0"]["kernel"] == "backbone"
    assert (
        labels["encoder"]["layers_0"]["ConvBlock_0"]["BatchNorm_0"]["scale"] == "frozen"
    )
    assert labels["encoder"]["resnet"]["layers_2"]["kernel"] == "backbone"
    assert (
        labels["encoder"]["resnet"]["layers_2"]["ConvBlock_1"]["BatchNorm_0"]["bias"]
        == "frozen"
    )
    assert labels["encoder"]["film"]["layer_1"]["kernel"] == "backbone"
    assert labels["encoder"]["film"]["text_proj"]["kernel"] == "main"
    assert labels["encoder"]["plucker"]["conv_0"]["kernel"] == "main"
    assert labels["encoder"]["fusion"]["input_proj"]["kernel"] == "main"


def test_act_optimizer_keeps_frozen_batch_norm_affine_values_exact():
    params = {
        "actor": {"kernel": np.ones((1,), dtype=np.float32)},
        "encoder": {
            "layers_0": {
                "ConvBlock_0": {
                    "kernel": np.ones((1,), dtype=np.float32),
                    "BatchNorm_0": {
                        "scale": np.ones((1,), dtype=np.float32),
                        "bias": np.zeros((1,), dtype=np.float32),
                    },
                }
            }
        },
    }
    optimizer = optax.multi_transform(
        {
            "main": optax.sgd(0.1),
            "backbone": optax.sgd(0.1),
            "frozen": optax.set_to_zero(),
        },
        _optimizer_labels,
    )
    opt_state = optimizer.init(params)
    grads = jax.tree.map(lambda value: np.ones_like(value), params)

    updates, _ = optimizer.update(grads, opt_state, params)
    updated = optax.apply_updates(params, updates)

    np.testing.assert_array_equal(
        updated["encoder"]["layers_0"]["ConvBlock_0"]["BatchNorm_0"]["scale"],
        params["encoder"]["layers_0"]["ConvBlock_0"]["BatchNorm_0"]["scale"],
    )
    np.testing.assert_array_equal(
        updated["encoder"]["layers_0"]["ConvBlock_0"]["BatchNorm_0"]["bias"],
        params["encoder"]["layers_0"]["ConvBlock_0"]["BatchNorm_0"]["bias"],
    )
    assert not np.array_equal(
        updated["encoder"]["layers_0"]["ConvBlock_0"]["kernel"],
        params["encoder"]["layers_0"]["ConvBlock_0"]["kernel"],
    )


def test_act_rgb_augmentation_preserves_shape_range_and_key_determinism():
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
    agent = _make_act(
        observation_space=observation_space,
        action_space=action_space,
        jit=False,
    )
    rgb = (np.arange(2 * 2 * 3 * 8 * 8) % 256).reshape(2, 2, 3, 8, 8)
    rgb = rgb.astype(np.float32)
    key = jax.random.PRNGKey(7)

    augmented = agent._augment_rgb(jax.numpy.asarray(rgb), key)
    augmented_again = agent._augment_rgb(jax.numpy.asarray(rgb), key)
    augmented_np = np.asarray(jax.device_get(augmented))

    assert augmented_np.shape == rgb.shape
    assert augmented_np.dtype == np.float32
    assert np.isfinite(augmented_np).all()
    assert augmented_np.min() >= 0.0
    assert augmented_np.max() <= 255.0
    np.testing.assert_allclose(
        augmented_np, np.asarray(jax.device_get(augmented_again))
    )


def test_act_campose_crop_uses_identical_rgb_and_explicit_raymap_geometry():
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
    agent = _make_act(
        observation_space=observation_space,
        action_space=action_space,
        image_augmentation_type="campose_crop",
    )
    pixel_indices = np.arange(10 * 10, dtype=np.float32).reshape(1, 1, 1, 10, 10)
    rgb = np.repeat(np.repeat(pixel_indices, 2, axis=0), 2, axis=1)
    rgb = np.repeat(rgb, 3, axis=2)
    raymap = np.repeat(pixel_indices, 2, axis=0)
    raymap = np.repeat(raymap, 2, axis=1)
    raymap = np.repeat(raymap, 6, axis=2)
    key = jax.random.PRNGKey(17)

    augmented = agent._augment_observation_images(
        {
            "rgb": jax.numpy.asarray(rgb),
            "raymap": jax.numpy.asarray(raymap),
        },
        key,
    )
    augmented_again = agent._augment_observation_images(
        {
            "rgb": jax.numpy.asarray(rgb),
            "raymap": jax.numpy.asarray(raymap),
        },
        key,
    )

    np.testing.assert_array_equal(
        np.asarray(augmented["rgb"][:, :, 0]),
        np.asarray(augmented["raymap"][:, :, 0]),
    )
    np.testing.assert_array_equal(
        np.asarray(augmented["rgb"]),
        np.asarray(augmented_again["rgb"]),
    )
    np.testing.assert_array_equal(
        np.asarray(augmented["raymap"]),
        np.asarray(augmented_again["raymap"]),
    )


def test_act_campose_crop_generates_raymap_before_shared_geometry(monkeypatch):
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
    agent = _make_act(
        observation_space=observation_space,
        action_space=action_space,
        image_augmentation_type="campose_crop",
        encoder_model=BCEncoderModelSpec(
            type="resnet",
            model="resnet18",
            trainable=True,
            use_plucker=True,
        ),
    )

    def fake_raymap(intrinsics, c2ws, height, width):
        del c2ws
        indices = jax.numpy.arange(height * width, dtype=jax.numpy.float32)
        indices = indices.reshape((1, height, width, 1))
        return jax.numpy.broadcast_to(
            indices,
            (intrinsics.shape[0], height, width, 6),
        )

    monkeypatch.setattr(
        "robobase.models.camera_augmentation._plucker_raymap_from_camera_params_jax",
        fake_raymap,
    )
    pixel_indices = np.arange(10 * 10, dtype=np.float32).reshape(1, 1, 1, 10, 10)
    rgb = np.repeat(pixel_indices, 3, axis=2)
    intrinsics = jax.numpy.eye(3)[None, None, None]
    c2ws = jax.numpy.eye(4)[None, None, None]

    augmented = agent._augment_observation_images(
        {
            "rgb": jax.numpy.asarray(rgb),
            "camera_intrinsic": intrinsics,
            "camera_c2w": c2ws,
        },
        jax.random.PRNGKey(23),
    )

    assert augmented["raymap"].shape == (1, 1, 6, 10, 10)
    np.testing.assert_array_equal(
        np.asarray(augmented["rgb"][:, :, 0]),
        np.asarray(augmented["raymap"][:, :, 0]),
    )


def test_act_pytorch_default_bias_initializers_are_not_zero():
    policy = JaxACTPolicy(
        hidden_dim=32,
        dropout=0.0,
        nheads=4,
        dim_feedforward=64,
        enc_layers=1,
        dec_layers=1,
        pre_norm=False,
        state_dim=4,
        action_dim=3,
        num_queries=2,
        latent_dim=8,
    )
    variables = policy.init(
        {
            "params": jax.random.PRNGKey(0),
            "dropout": jax.random.PRNGKey(1),
        },
        None,
        None,
        jax.numpy.zeros((2, 4), dtype=jax.numpy.float32),
        actions=jax.numpy.zeros((2, 2, 3), dtype=jax.numpy.float32),
        is_pad=jax.numpy.zeros((2, 2), dtype=jax.numpy.bool_),
        deterministic=False,
        latent_key=jax.random.PRNGKey(2),
    )
    params = variables["params"]

    for name in (
        "encoder_action_proj",
        "encoder_joint_proj",
        "latent_proj",
        "latent_out_proj",
        "input_proj_robot_state_0",
        "input_proj_robot_state_1",
        "action_head",
    ):
        assert not np.allclose(np.asarray(params[name]["bias"]), 0.0)

    image_projection = ACTImageProjection(hidden_dim=16)
    image_variables = image_projection.init(
        jax.random.PRNGKey(3),
        jax.numpy.zeros((2, 3, 1, 1, 512), dtype=jax.numpy.float32),
    )
    assert not np.allclose(
        np.asarray(image_variables["params"]["input_proj"]["bias"]),
        0.0,
    )


def test_act_campose_projection_uses_learned_camera_major_positions():
    projection = ACTImageProjection(
        hidden_dim=8,
        position_embedding_type="campose_learned",
    )
    features = jax.numpy.arange(2 * 2 * 2 * 3 * 4, dtype=jax.numpy.float32)
    features = features.reshape((2, 2, 2, 3, 4))
    variables = projection.init(jax.random.PRNGKey(3), features)
    mutable_variables = unfreeze(variables)
    kernel = np.zeros((1, 1, 4, 8), dtype=np.float32)
    kernel[0, 0, :, :4] = np.eye(4, dtype=np.float32)
    mutable_variables["params"]["input_proj"]["kernel"] = kernel
    mutable_variables["params"]["input_proj"]["bias"] = np.zeros((8,), dtype=np.float32)
    variables = freeze(mutable_variables)

    tokens, positions = projection.apply(variables, features)

    assert tokens.shape == (2, 1, 12, 8)
    assert positions.shape == tokens.shape
    np.testing.assert_array_equal(
        np.asarray(tokens[:, 0, :, :4]),
        np.asarray(features.reshape((2, 2, 6, 4)).reshape((2, 12, 4))),
    )
    np.testing.assert_array_equal(
        np.asarray(positions[0, 0]),
        np.asarray(variables["params"]["position_embedding"][:12]),
    )
    np.testing.assert_array_equal(np.asarray(positions[0]), np.asarray(positions[1]))


def test_act_campose_projection_rejects_image_tokens_over_configured_limit():
    projection = ACTImageProjection(
        hidden_dim=8,
        position_embedding_type="campose_learned",
        max_position_tokens=7,
    )
    features = jax.numpy.zeros((1, 2, 2, 2, 4), dtype=jax.numpy.float32)

    with pytest.raises(ValueError, match="at most 7 image tokens, got 8"):
        projection.init(jax.random.PRNGKey(3), features)


def test_act_campose_conditioning_memory_matches_official_token_order():
    image_tokens = jax.numpy.asarray(
        [[[[10.0, 10.0], [11.0, 11.0]]]], dtype=jax.numpy.float32
    )
    image_pos = image_tokens + 100.0
    camera_tokens = jax.numpy.asarray(
        [[[20.0, 20.0], [21.0, 21.0]]], dtype=jax.numpy.float32
    )
    proprio = jax.numpy.asarray([[30.0, 30.0]], dtype=jax.numpy.float32)
    latent = jax.numpy.asarray([[40.0, 40.0]], dtype=jax.numpy.float32)
    conditioning_pos = jax.numpy.asarray(
        [[200.0, 200.0], [201.0, 201.0], [202.0, 202.0], [203.0, 203.0]],
        dtype=jax.numpy.float32,
    )

    memory, positions = _assemble_campose_conditioning_memory(
        image_tokens,
        image_pos,
        camera_tokens,
        proprio,
        latent,
        conditioning_pos,
    )

    np.testing.assert_array_equal(
        np.asarray(memory[0, :, 0]),
        [10.0, 11.0, 20.0, 21.0, 30.0, 40.0],
    )
    np.testing.assert_array_equal(
        np.asarray(positions[0, :, 0]),
        [110.0, 111.0, 200.0, 201.0, 202.0, 203.0],
    )


def test_act_camera_extrinsics_create_shared_dense_tokens_and_affect_output():
    policy = JaxACTPolicy(
        hidden_dim=16,
        dropout=0.0,
        nheads=4,
        dim_feedforward=32,
        enc_layers=1,
        dec_layers=1,
        pre_norm=False,
        state_dim=4,
        action_dim=3,
        num_queries=2,
        latent_dim=8,
        use_camera_extrinsics=True,
        num_camera_extrinsics=2,
    )
    image_features = jax.numpy.zeros((2, 1, 2, 16), dtype=jax.numpy.float32)
    image_pos = jax.numpy.zeros_like(image_features)
    qpos = jax.numpy.zeros((2, 4), dtype=jax.numpy.float32)
    extrinsics = jax.numpy.zeros((2, 2, 4, 4), dtype=jax.numpy.float32)
    variables = policy.init(
        jax.random.PRNGKey(5),
        image_features,
        image_pos,
        qpos,
        camera_extrinsics=extrinsics,
        deterministic=True,
    )

    output_zero, _, _ = policy.apply(
        variables,
        image_features,
        image_pos,
        qpos,
        camera_extrinsics=extrinsics,
        deterministic=True,
    )
    output_one, _, _ = policy.apply(
        variables,
        image_features,
        image_pos,
        qpos,
        camera_extrinsics=jax.numpy.ones_like(extrinsics),
        deterministic=True,
    )
    params = variables["params"]

    assert params["input_proj_cam_extrinsics"]["kernel"].shape == (16, 16)
    assert params["input_embed"].shape == (4, 16)
    assert "additional_pos_embed" not in params
    assert not np.allclose(output_zero, output_one)


def test_act_default_modes_preserve_legacy_parameter_tree():
    kwargs = dict(
        hidden_dim=16,
        dropout=0.0,
        nheads=4,
        dim_feedforward=32,
        enc_layers=1,
        dec_layers=1,
        pre_norm=False,
        state_dim=4,
        action_dim=3,
        num_queries=2,
        latent_dim=8,
    )
    legacy = JaxACTPolicy(**kwargs)
    explicit_legacy = JaxACTPolicy(
        **kwargs,
        use_camera_extrinsics=False,
        num_camera_extrinsics=2,
        proprio_projection_type="legacy_mlp",
        style_cls_type="learned",
        decoder_output_layer="final",
    )
    init_args = (
        None,
        None,
        jax.numpy.zeros((2, 4), dtype=jax.numpy.float32),
    )
    init_kwargs = dict(
        actions=jax.numpy.zeros((2, 2, 3), dtype=jax.numpy.float32),
        is_pad=jax.numpy.zeros((2, 2), dtype=jax.numpy.bool_),
        deterministic=True,
    )
    legacy_variables = legacy.init(jax.random.PRNGKey(11), *init_args, **init_kwargs)
    explicit_variables = explicit_legacy.init(
        jax.random.PRNGKey(11), *init_args, **init_kwargs
    )

    assert jax.tree.structure(legacy_variables) == jax.tree.structure(
        explicit_variables
    )
    for legacy_leaf, explicit_leaf in zip(
        jax.tree.leaves(legacy_variables),
        jax.tree.leaves(explicit_variables),
        strict=True,
    ):
        np.testing.assert_array_equal(legacy_leaf, explicit_leaf)
    legacy_params = legacy_variables["params"]
    assert "style_encoder_norm" not in legacy_params
    assert "encoder_norm" not in legacy_params["transformer"]


def test_act_campose_modes_use_single_proprio_zero_cls_and_final_norms():
    policy = JaxACTPolicy(
        hidden_dim=16,
        dropout=0.0,
        nheads=4,
        dim_feedforward=32,
        enc_layers=1,
        dec_layers=1,
        pre_norm=True,
        state_dim=4,
        action_dim=3,
        num_queries=2,
        latent_dim=8,
        proprio_projection_type="campose_single",
        style_cls_type="zero",
        decoder_output_layer="first",
    )
    variables = policy.init(
        jax.random.PRNGKey(29),
        None,
        None,
        jax.numpy.zeros((2, 4), dtype=jax.numpy.float32),
        actions=jax.numpy.zeros((2, 2, 3), dtype=jax.numpy.float32),
        is_pad=jax.numpy.zeros((2, 2), dtype=jax.numpy.bool_),
        deterministic=True,
    )
    params = variables["params"]

    assert "cls_embed" not in params
    assert "input_proj_robot_state" in params
    assert "input_proj_robot_state_0" not in params
    assert "input_proj_robot_state_1" not in params
    assert "style_encoder_norm" in params
    assert "encoder_norm" in params["transformer"]


def test_act_first_decoder_output_is_independent_of_later_layers():
    kwargs = dict(
        hidden_dim=8,
        nheads=2,
        enc_layers=1,
        dec_layers=2,
        dim_feedforward=16,
        dropout=0.0,
        pre_norm=False,
    )
    first_decoder = ACTDetrTransformer(
        **kwargs,
        decoder_output_layer="first",
    )
    final_decoder = ACTDetrTransformer(
        **kwargs,
        decoder_output_layer="final",
    )
    src = jax.numpy.arange(24, dtype=jax.numpy.float32).reshape((1, 3, 8))
    query_embed = jax.numpy.arange(16, dtype=jax.numpy.float32).reshape((2, 8))
    pos_embed = jax.numpy.zeros_like(src)
    variables = first_decoder.init(
        jax.random.PRNGKey(31),
        src,
        query_embed,
        pos_embed,
        deterministic=True,
    )
    changed_variables = unfreeze(variables)
    changed_variables["params"]["decoder_1"]["mlp"]["Dense_1"]["bias"] += (
        jax.numpy.arange(8, dtype=jax.numpy.float32)
    )
    changed_variables = freeze(changed_variables)

    first_before = first_decoder.apply(
        variables,
        src,
        query_embed,
        pos_embed,
        deterministic=True,
    )
    first_after = first_decoder.apply(
        changed_variables,
        src,
        query_embed,
        pos_embed,
        deterministic=True,
    )
    final_before = final_decoder.apply(
        variables,
        src,
        query_embed,
        pos_embed,
        deterministic=True,
    )
    final_after = final_decoder.apply(
        changed_variables,
        src,
        query_embed,
        pos_embed,
        deterministic=True,
    )

    np.testing.assert_array_equal(first_before, first_after)
    assert not np.allclose(final_before, final_after)


def test_act_full_proprio_dropout_zeros_qpos_in_train_and_inference():
    policy = JaxACTPolicy(
        hidden_dim=16,
        dropout=0.0,
        nheads=4,
        dim_feedforward=32,
        enc_layers=1,
        dec_layers=1,
        pre_norm=False,
        state_dim=4,
        action_dim=3,
        num_queries=2,
        latent_dim=8,
        proprio_dropout_prob=1.0,
    )
    qpos_zero = jax.numpy.zeros((2, 4), dtype=jax.numpy.float32)
    qpos_one = jax.numpy.ones((2, 4), dtype=jax.numpy.float32)
    variables = policy.init(
        jax.random.PRNGKey(19),
        None,
        None,
        qpos_one,
        deterministic=True,
    )

    eval_zero = policy.apply(
        variables,
        None,
        None,
        qpos_zero,
        deterministic=True,
    )[0]
    eval_one = policy.apply(
        variables,
        None,
        None,
        qpos_one,
        deterministic=True,
    )[0]
    train_zero = policy.apply(
        variables,
        None,
        None,
        qpos_zero,
        deterministic=False,
        rngs={"dropout": jax.random.PRNGKey(20)},
    )[0]
    train_one = policy.apply(
        variables,
        None,
        None,
        qpos_one,
        deterministic=False,
        rngs={"dropout": jax.random.PRNGKey(20)},
    )[0]

    np.testing.assert_allclose(eval_zero, eval_one, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(train_zero, train_one, rtol=0.0, atol=0.0)


def test_act_method_forwards_and_zero_pads_camera_extrinsics():
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
        shape=(2, 3),
        dtype=np.float32,
    )
    agent = _make_act(
        observation_space=observation_space,
        action_space=action_space,
        use_camera_extrinsics=True,
        num_camera_extrinsics=2,
    )
    first_extrinsic = np.eye(4, dtype=np.float32)[None, None]
    first_extrinsic = np.repeat(first_extrinsic, 2, axis=0)
    obs = {
        "low_dim_state": np.zeros((2, 1, 4), dtype=np.float32),
        "cam_extrinsics": first_extrinsic,
    }

    prepared = agent._prepare_trainable_obs_inputs(obs)
    output = agent.act(obs, step=0, eval_mode=True)

    assert prepared["camera_extrinsics"].shape == (2, 2, 4, 4)
    np.testing.assert_array_equal(
        np.asarray(prepared["camera_extrinsics"][:, 0]),
        first_extrinsic[:, 0],
    )
    np.testing.assert_array_equal(
        np.asarray(prepared["camera_extrinsics"][:, 1]),
        np.zeros((2, 4, 4), dtype=np.float32),
    )
    assert output.shape == (2, 2, 3)


def test_act_update_uses_action_padding_mask_and_changes_params():
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
    agent = _make_act(
        observation_space=observation_space,
        action_space=action_space,
        jit=False,
    )
    agent.logging = True
    batch = {
        "low_dim_state": np.zeros((2, 1, 4), dtype=np.float32),
        "action": np.zeros((2, 3, 2), dtype=np.float32),
        "action_pad_mask": np.array(
            [[False, False, True], [False, True, True]],
            dtype=np.bool_,
        ),
    }

    before = _params_leaves(agent.state_dict())
    metrics = agent.update(iter([batch]), step=0)
    after = _params_leaves(agent.state_dict())

    assert np.isfinite(metrics["actor_loss"])
    assert any(not np.allclose(a, b) for a, b in zip(before, after))


def test_act_supports_single_action_chunk_for_delayed_policy():
    # The pure delayed policy (obs_delay=h, action_sequence=1) runs ACT with a
    # single decoder query, so the chunk axis degenerates to length 1.
    observation_space = spaces.Dict(
        {"low_dim_state": spaces.Box(-1.0, 1.0, shape=(1, 4), dtype=np.float32)}
    )
    action_space = spaces.Box(-1.0, 1.0, shape=(1, 2), dtype=np.float32)
    agent = _make_act(
        observation_space=observation_space,
        action_space=action_space,
        jit=False,
    )
    agent.logging = True
    batch = {
        "low_dim_state": np.zeros((2, 1, 4), dtype=np.float32),
        "action": np.zeros((2, 1, 2), dtype=np.float32),
        "action_pad_mask": np.zeros((2, 1), dtype=np.bool_),
    }

    before = _params_leaves(agent.state_dict())
    metrics = agent.update(iter([batch]), step=0)
    after = _params_leaves(agent.state_dict())

    assert np.isfinite(metrics["actor_loss"])
    assert any(not np.allclose(a, b) for a, b in zip(before, after))

    action = agent.act(
        {"low_dim_state": np.zeros((1, 1, 4), dtype=np.float32)},
        step=0,
        eval_mode=True,
    )
    assert np.asarray(action).shape == (1, 1, 2)


def test_act_treats_all_action_dims_as_continuous_by_default():
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
        shape=(3, 4),
        dtype=np.float32,
    )
    agent = _make_act(
        observation_space=observation_space,
        action_space=action_space,
        jit=False,
    )
    agent.params = jax.tree_util.tree_map(jax.numpy.zeros_like, agent.params)
    agent.logging = True
    batch = {
        "low_dim_state": np.zeros((2, 1, 4), dtype=np.float32),
        "action": np.ones((2, 3, 4), dtype=np.float32),
    }

    metrics = agent.update(iter([batch]), step=0)

    assert metrics["actor_l1_loss"] == pytest.approx(1.0)
    assert metrics["actor_gripper_loss"] == pytest.approx(0.0)


@pytest.mark.parametrize("gripper_dims", [-1, 5])
def test_act_rejects_invalid_gripper_dims(gripper_dims):
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
        shape=(3, 4),
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="gripper_dims"):
        _make_act(
            observation_space=observation_space,
            action_space=action_space,
            jit=False,
            gripper_dims=gripper_dims,
        )


def test_act_update_accepts_horizon_dropout():
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
    agent = _make_act(
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

    before = _params_leaves(agent.state_dict())
    metrics = agent.update(iter([batch]), step=0)
    after = _params_leaves(agent.state_dict())

    assert np.isfinite(metrics["actor_loss"])
    assert any(not np.allclose(a, b) for a, b in zip(before, after))


def test_act_update_many_fuses_multiple_updates():
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
    agent = _make_act(
        observation_space=observation_space,
        action_space=action_space,
        jit=False,
        horizon_dropout_lengths=(1, 2, 4),
        horizon_dropout_probs=(0.25, 0.25, 0.5),
    )
    batches = [
        {
            "low_dim_state": np.full((2, 1, 4), i, dtype=np.float32),
            "action": np.zeros((2, 4, 2), dtype=np.float32),
        }
        for i in range(3)
    ]

    before = _params_leaves(agent.state_dict())
    metrics = agent.update_many(iter(batches), num_updates=3)
    after = _params_leaves(agent.state_dict())

    assert metrics == {}
    assert agent._update_step_count == 3
    assert any(not np.allclose(a, b) for a, b in zip(before, after))


def test_act_padding_loss_matches_reference_full_denominator_mean():
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
        shape=(3, 1),
        dtype=np.float32,
    )
    agent = _make_act(
        observation_space=observation_space,
        action_space=action_space,
        jit=False,
    )
    agent.params = jax.tree_util.tree_map(jax.numpy.zeros_like, agent.params)
    agent.logging = True
    batch = {
        "low_dim_state": np.zeros((2, 1, 4), dtype=np.float32),
        "action": np.array(
            [
                [[1.0], [1.0], [1.0]],
                [[2.0], [2.0], [2.0]],
            ],
            dtype=np.float32,
        ),
        "action_pad_mask": np.array(
            [[False, True, True], [False, False, False]],
            dtype=np.bool_,
        ),
    }

    metrics = agent.update(iter([batch]), step=0)

    assert metrics["actor_l1_loss"] == pytest.approx(7.0 / 6.0)


def test_act_loads_legacy_encoder_extra_checkpoint_state():
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
    agent = _make_act(
        observation_space=observation_space,
        action_space=action_space,
        jit=False,
    )
    checkpoint_state = agent.checkpoint_state_dict()
    legacy_state = {
        **checkpoint_state,
        "opt_state": {
            "main": checkpoint_state["opt_state"],
            "encoder_extra": None,
        },
    }

    agent.load_checkpoint_state_dict(legacy_state)
    metrics = agent.update(
        iter(
            [
                {
                    "low_dim_state": np.zeros((2, 1, 4), dtype=np.float32),
                    "action": np.zeros((2, 3, 2), dtype=np.float32),
                }
            ]
        ),
        step=0,
    )

    assert isinstance(metrics, dict)


def test_act_reinitializes_incompatible_optimizer_checkpoint_state():
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
    agent = _make_act(
        observation_space=observation_space,
        action_space=action_space,
        jit=False,
    )
    expected_structure = jax.tree_util.tree_structure(agent.opt_state)

    with pytest.warns(RuntimeWarning, match="optimizer state is incompatible"):
        agent.load_checkpoint_state_dict(
            {
                "opt_state": {"unexpected": np.zeros((7,), dtype=np.float32)},
            }
        )

    assert jax.tree_util.tree_structure(agent.opt_state) == expected_structure


def test_act_returns_action_sequence():
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
    agent = _make_act(
        observation_space=observation_space,
        action_space=action_space,
        jit=False,
    )

    obs = {"low_dim_state": np.zeros((2, 1, 4), dtype=np.float32)}
    actions = agent.act(obs, step=0, eval_mode=True)

    assert actions.shape == (2, 3, 2)
    assert actions.dtype == np.float32


def test_act_requires_lang_tokens_when_language_conditioned():
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
    model = ACTModelSpec(
        actor_model=ACTActorModelSpec(
            type="transformer",
            hidden_dim=32,
            enc_layers=1,
            dec_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            nheads=4,
            num_queries=action_space.shape[0],
            pre_norm=False,
            use_lang_cond=True,
        ),
        encoder_model=None,
        view_fusion_model=None,
    )
    agent = ACT(
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
        jit=False,
        seed=0,
    )

    with pytest.raises(ValueError, match="lang_features.*lang_tokens"):
        agent.act(
            {"low_dim_state": np.zeros((2, 1, 4), dtype=np.float32)},
            step=0,
            eval_mode=True,
        )


def test_act_prefers_precomputed_lang_features(monkeypatch):
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
                shape=(1, 4),
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
    model = ACTModelSpec(
        actor_model=ACTActorModelSpec(
            type="transformer",
            hidden_dim=32,
            enc_layers=1,
            dec_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            nheads=4,
            num_queries=action_space.shape[0],
            pre_norm=False,
            use_lang_cond=True,
            lang_feature_dim=4,
        ),
        encoder_model=None,
        view_fusion_model=None,
    )
    agent = ACT(
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
        jit=False,
        seed=0,
    )

    def fail_token_path(*args, **kwargs):
        raise AssertionError("lang_features should bypass token feature hashing")

    monkeypatch.setattr("robobase.method.act.tokens_to_feature_jax", fail_token_path)
    features = np.arange(8, dtype=np.float32).reshape(2, 1, 4)

    actual = agent._extract_lang_features({"lang_features": features})

    np.testing.assert_allclose(np.asarray(jax.device_get(actual)), features[:, 0])


def test_act_supports_pixel_inputs_with_multicam_fusion(monkeypatch):
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
    model = ACTModelSpec(
        actor_model=ACTActorModelSpec(
            type="transformer",
            hidden_dim=32,
            enc_layers=1,
            dec_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            nheads=4,
            num_queries=action_space.shape[0],
            pre_norm=False,
        ),
        encoder_model=BCEncoderModelSpec(
            type="resnet",
            model="resnet18",
            trainable=True,
        ),
        view_fusion_model=BCViewFusionModelSpec(
            type="multicam_feature", mode="flatten"
        ),
    )
    agent = ACT(
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
        jit=False,
        seed=0,
    )
    assert not (
        isinstance(agent.opt_state, dict) and "encoder_extra" in agent.opt_state
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


def test_act_supports_plucker_camera_params_at_initialization(monkeypatch):
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
        shape=(2, 3),
        dtype=np.float32,
    )
    model = ACTModelSpec(
        actor_model=ACTActorModelSpec(
            type="transformer",
            hidden_dim=32,
            enc_layers=1,
            dec_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            nheads=4,
            num_queries=action_space.shape[0],
            pre_norm=False,
        ),
        encoder_model=BCEncoderModelSpec(
            type="resnet",
            model="resnet18",
            trainable=True,
            use_plucker=True,
            plucker_hidden_channels=4,
        ),
        view_fusion_model=BCViewFusionModelSpec(
            type="multicam_feature", mode="flatten"
        ),
    )
    agent = ACT(
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
        jit=False,
        seed=0,
    )

    intrinsic = np.eye(3, dtype=np.float32)
    c2w = np.eye(4, dtype=np.float32)
    obs = {
        "low_dim_state": np.zeros((2, 1, 4), dtype=np.float32),
        "rgb_front": np.full((2, 1, 3, 8, 8), 64, dtype=np.uint8),
        "camera_intrinsic_front": np.tile(intrinsic, (2, 1, 1, 1)),
        "camera_c2w_front": np.tile(c2w, (2, 1, 1, 1)),
        "rgb_wrist": np.full((2, 1, 3, 8, 8), 128, dtype=np.uint8),
        "camera_intrinsic_wrist": np.tile(intrinsic, (2, 1, 1, 1)),
        "camera_c2w_wrist": np.tile(c2w, (2, 1, 1, 1)),
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


def test_act_workspace_smoke_and_snapshot(tmp_path):
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).resolve().parents[2] / "robobase/cfgs"),
        job_name="test_jax_act_workspace",
    ):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "backend=jax",
                "method=act",
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
                "method.use_lang_cond=false",
                "method.actor_model.hidden_dim=32",
                "method.actor_model.enc_layers=1",
                "method.actor_model.dec_layers=1",
                "method.actor_model.dim_feedforward=64",
                "method.actor_model.nheads=4",
                "method.actor_model.dropout=0.0",
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
