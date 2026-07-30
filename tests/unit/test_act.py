from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import flax.linen as nn
from flax.core import freeze
from gymnasium import spaces
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from robobase.envs.env import EnvFactory
from robobase.method.bc import BCEncoderModelSpec, BCViewFusionModelSpec
from robobase.method.act import (
    ACT,
    ACTActorModelSpec,
    ACTModelSpec,
    _optimizer_labels,
    act_spec_from_cfg,
)
from robobase.models.act import ACTImageProjection, JaxACTPolicy
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


def _make_act(
    *,
    observation_space,
    action_space,
    jit=False,
    horizon_dropout_lengths=None,
    horizon_dropout_probs=None,
    gripper_dims=0,
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
        ),
        encoder_model=None,
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
    assert spec.weight_decay == pytest.approx(1e-4)
    assert spec.horizon_dropout_lengths == (1, 2, 4)
    assert spec.horizon_dropout_probs == (0.2, 0.3, 0.5)


def test_act_optimizer_labels_only_backbone_subtrees_for_low_lr():
    params = {
        "actor": {"head": np.zeros((1,), dtype=np.float32)},
        "image_projection": {"input_proj": {"kernel": np.zeros((1,), dtype=np.float32)}},
        "encoder": {
            "layers_0": {"ConvBlock_0": {"kernel": np.zeros((1,), dtype=np.float32)}},
            "resnet": {"layers_2": {"kernel": np.zeros((1,), dtype=np.float32)}},
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
    assert labels["encoder"]["resnet"]["layers_2"]["kernel"] == "backbone"
    assert labels["encoder"]["film"]["layer_1"]["kernel"] == "backbone"
    assert labels["encoder"]["film"]["text_proj"]["kernel"] == "main"
    assert labels["encoder"]["plucker"]["conv_0"]["kernel"] == "main"
    assert labels["encoder"]["fusion"]["input_proj"]["kernel"] == "main"


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
    np.testing.assert_allclose(augmented_np, np.asarray(jax.device_get(augmented_again)))


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
        view_fusion_model=BCViewFusionModelSpec(type="multicam_feature", mode="flatten"),
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
        view_fusion_model=BCViewFusionModelSpec(type="multicam_feature", mode="flatten"),
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
    for before, after in zip(_params_leaves(saved_state), _params_leaves(restored_state)):
        assert np.allclose(before, after)
