from pathlib import Path
import hashlib

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import flax.linen as nn
from flax.core import freeze

import robobase.models.encoder as encoder_module

from robobase.method.bc import BC as JaxBC
from robobase.models.encoder import (
    JaxDPEarlyFusionEncoder,
    JaxPluckerEncoder,
    JaxPluckerFusion,
    JaxResNetEncoder,
    _ensure_resnet18_safetensors,
    _load_pretrained_resnet_npz,
    _load_resnet_feature_model_cached,
    _resnet18_safetensors_to_variables,
    normalize_plucker_fusion_mode,
    resnet_weight_fingerprint,
)
from robobase.envs.env import EnvFactory
from robobase.method.bc import (
    BCActorModelSpec,
    BCEncoderModelSpec,
    BCModelSpec,
    BCViewFusionModelSpec,
)
from robobase.workspace import Workspace

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
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
    use_plucker=False,
    trainable_encoder=False,
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
    if model.encoder_model is not None:
        model = BCModelSpec(
            actor_model=model.actor_model,
            encoder_model=BCEncoderModelSpec(
                type=model.encoder_model.type,
                model=model.encoder_model.model,
                trainable=trainable_encoder,
                use_plucker=use_plucker,
                plucker_hidden_channels=4,
            ),
            view_fusion_model=model.view_fusion_model,
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
        config_dir=str(Path(__file__).resolve().parents[2] / "robobase/cfgs"),
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


def test_jax_bc_all_valid_padding_mask_matches_unmasked_loss():
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
    agent = _make_jax_bc(
        observation_space=observation_space,
        action_space=action_space,
    )
    batch = {
        "low_dim_state": np.zeros((2, 1, 4), dtype=np.float32),
        "action": np.ones((2, 2, 3), dtype=np.float32),
    }
    obs_features, _ = agent._prepare_obs_features(batch)
    common_args = (
        agent.params,
        agent.opt_state,
        obs_features,
        jnp.asarray(batch["action"]),
        jnp.ones((2,), dtype=jnp.float32),
    )

    unmasked_loss = agent._update_impl(*common_args, None)[2]
    all_valid_loss = agent._update_impl(
        *common_args,
        jnp.zeros((2, 2), dtype=jnp.bool_),
    )[2]

    np.testing.assert_allclose(all_valid_loss, unmasked_loss, rtol=1e-6)


class _FakeResNetFeatureModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        return jax.numpy.zeros((x.shape[0], 1, 1, 512), dtype=jax.numpy.float32)


def _fake_resnet_feature_model():
    return _FakeResNetFeatureModel(), freeze({}), 512


def _fake_resnet_feature_model_with_batch_stats(value: float):
    variables = freeze(
        {
            "batch_stats": {
                "fake_batch_norm": {
                    "mean": jnp.full((3,), value, dtype=jnp.float32),
                    "var": jnp.full((3,), value + 1.0, dtype=jnp.float32),
                }
            }
        }
    )
    return _FakeResNetFeatureModel(), variables, 512


def test_jax_resnet_encoder_frozen_state_roundtrip(monkeypatch):
    values = iter((2.0, 9.0))
    monkeypatch.setattr(
        "robobase.models.encoder._load_resnet_feature_model",
        lambda model_name, pretrained=False: (
            _fake_resnet_feature_model_with_batch_stats(next(values))
        ),
    )
    saved_encoder = JaxResNetEncoder(
        input_shape=(2, 3, 8, 8),
        model="resnet18",
        jit=True,
    )
    restored_encoder = JaxResNetEncoder(
        input_shape=(2, 3, 8, 8),
        model="resnet18",
        jit=True,
    )

    saved_state = jax.tree.map(np.asarray, saved_encoder.frozen_state_dict())
    restored_encoder.load_frozen_state_dict(saved_state)

    saved_leaves = jax.tree_util.tree_leaves(saved_encoder.batch_stats)
    restored_leaves = jax.tree_util.tree_leaves(restored_encoder.batch_stats)
    assert len(saved_leaves) == len(restored_leaves)
    for expected, actual in zip(saved_leaves, restored_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_jax_resnet_encoder_runs_native_forward(monkeypatch):
    monkeypatch.setattr(
        "robobase.models.encoder._load_resnet_feature_model",
        lambda model_name, pretrained=False: _fake_resnet_feature_model(),
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


def test_jax_resnet_encoder_accepts_jitted_device_inputs(monkeypatch):
    monkeypatch.setattr(
        "robobase.models.encoder._load_resnet_feature_model",
        lambda model_name, pretrained=False: _fake_resnet_feature_model(),
    )
    encoder = JaxResNetEncoder(
        input_shape=(2, 3, 8, 8),
        model="resnet18",
        jit=False,
    )
    rgb_obs = jnp.zeros((3, 2, 3, 8, 8), dtype=jnp.uint8)

    feats = jax.jit(encoder.encode)(rgb_obs)

    assert feats.shape == (3, 2, 512)


def test_pretrained_resnet34_uses_environment_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "resnet34.npz"
    expected = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.savez(checkpoint, **{"params/example": expected})
    monkeypatch.setenv("ROBOBASE_RESNET34_JAX_NPZ", str(checkpoint))

    variables = _load_pretrained_resnet_npz("resnet34")

    np.testing.assert_array_equal(np.asarray(variables["params"]["example"]), expected)


def _synthetic_torchvision_resnet18_tensors():
    tensors = {}

    def add_conv(name, value):
        tensors[name] = np.full((1, 1, 1, 1), value, dtype=np.float32)

    def add_batch_norm(name, value):
        for offset, suffix in enumerate(
            ("weight", "bias", "running_mean", "running_var")
        ):
            tensors[f"{name}.{suffix}"] = np.asarray([value + offset], dtype=np.float32)

    add_conv("conv1.weight", 1.0)
    add_batch_norm("bn1", 2.0)
    for stage in range(1, 5):
        for block in range(2):
            source = f"layer{stage}.{block}"
            add_conv(f"{source}.conv1.weight", 10.0 * stage + block)
            add_batch_norm(f"{source}.bn1", 20.0 * stage + block)
            add_conv(f"{source}.conv2.weight", 30.0 * stage + block)
            add_batch_norm(f"{source}.bn2", 40.0 * stage + block)
            if stage > 1 and block == 0:
                add_conv(f"{source}.downsample.0.weight", 50.0 * stage)
                add_batch_norm(f"{source}.downsample.1", 60.0 * stage)
    return tensors


def test_resnet18_safetensors_conversion_builds_expected_flax_tree():
    variables = _resnet18_safetensors_to_variables(
        _synthetic_torchvision_resnet18_tensors()
    )

    assert len(jax.tree_util.tree_leaves(variables)) == 100
    np.testing.assert_array_equal(
        np.asarray(variables["params"]["layers_0"]["ConvBlock_0"]["Conv_0"]["kernel"]),
        np.ones((1, 1, 1, 1), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(
            variables["batch_stats"]["layers_8"]["ResNetSkipConnection_0"][
                "ConvBlock_0"
            ]["BatchNorm_0"]["var"]
        ),
        np.asarray([243.0], dtype=np.float32),
    )


def test_resnet18_download_is_atomic_and_sha_verified(tmp_path, monkeypatch):
    source = tmp_path / "source.safetensors"
    source.write_bytes(b"pinned-resnet18-test-payload")
    destination = tmp_path / "cache" / "model.safetensors"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(
        encoder_module,
        "_DEFAULT_RESNET18_SAFETENSORS",
        destination,
    )
    monkeypatch.setattr(encoder_module, "_RESNET18_SAFETENSORS_URL", source.as_uri())
    monkeypatch.setattr(encoder_module, "_RESNET18_SAFETENSORS_SHA256", digest)

    assert _ensure_resnet18_safetensors() == destination
    assert destination.read_bytes() == source.read_bytes()
    assert list(destination.parent.glob("*.tmp")) == []


def test_resnet18_download_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(
        encoder_module,
        "_DEFAULT_RESNET18_SAFETENSORS",
        tmp_path / "missing.safetensors",
    )
    monkeypatch.setenv("ROBOBASE_DISABLE_PRETRAINED_DOWNLOAD", "true")

    with pytest.raises(FileNotFoundError, match="automatic download is disabled"):
        _ensure_resnet18_safetensors()


def test_pretrained_resnet_fingerprint_tracks_checkpoint_content(tmp_path, monkeypatch):
    checkpoint = tmp_path / "resnet34.npz"
    np.savez(checkpoint, **{"params/example": np.zeros((2, 3), dtype=np.float32)})
    monkeypatch.setenv("ROBOBASE_RESNET34_JAX_NPZ", str(checkpoint))

    first = resnet_weight_fingerprint("resnet34", pretrained=True)
    np.savez(checkpoint, **{"params/example": np.ones((3, 4), dtype=np.float32)})
    second = resnet_weight_fingerprint("resnet34", pretrained=True)

    assert first.startswith("sha256:")
    assert second.startswith("sha256:")
    assert first != second


def test_explicit_pretrained_resnet_path_overrides_default(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit.npz"
    fallback = tmp_path / "fallback.npz"
    np.savez(explicit, **{"params/example": np.full((2,), 3.0, dtype=np.float32)})
    np.savez(fallback, **{"params/example": np.full((2,), 7.0, dtype=np.float32)})
    monkeypatch.setenv("ROBOBASE_RESNET34_JAX_NPZ", str(fallback))

    variables = _load_pretrained_resnet_npz(
        "resnet34",
        pretrained_weights_path=explicit,
    )

    np.testing.assert_array_equal(
        np.asarray(variables["params"]["example"]),
        np.full((2,), 3.0, dtype=np.float32),
    )
    assert resnet_weight_fingerprint(
        "resnet34",
        pretrained=True,
        pretrained_weights_path=explicit,
    ).startswith("sha256:")


def test_missing_explicit_pretrained_resnet_path_fails_closed(tmp_path):
    missing = tmp_path / "missing.npz"

    with pytest.raises(FileNotFoundError, match="Explicit pretrained ResNet"):
        resnet_weight_fingerprint(
            "resnet18",
            pretrained=True,
            pretrained_weights_path=missing,
        )


def test_missing_pretrained_npz_fails_without_timm_or_torch_fallback(monkeypatch):
    monkeypatch.delenv("ROBOBASE_RESNET34_JAX_NPZ", raising=False)
    _load_resnet_feature_model_cached.cache_clear()

    assert resnet_weight_fingerprint("resnet34", pretrained=True) == (
        "missing-jax-npz:resnet34"
    )
    with pytest.raises(FileNotFoundError, match="converted JAX ResNet npz"):
        _load_resnet_feature_model_cached(
            "resnet34",
            True,
            "missing-jax-npz:resnet34",
        )
    _load_resnet_feature_model_cached.cache_clear()


def test_jax_resnet_encoder_late_fuses_plucker_features(monkeypatch):
    monkeypatch.setattr(
        "robobase.models.encoder._load_resnet_feature_model",
        lambda model_name, pretrained=False: _fake_resnet_feature_model(),
    )

    encoder = JaxResNetEncoder(
        input_shape=(2, 3, 8, 8),
        model="resnet18",
        jit=True,
        use_plucker=True,
        plucker_hidden_channels=4,
    )
    rgb_obs = np.random.randint(0, 255, size=(3, 2, 3, 8, 8), dtype=np.uint8)
    raymap_obs = np.ones((3, 2, 6, 8, 8), dtype=np.float32)

    feats = np.asarray(jax.device_get(encoder.encode(rgb_obs, raymap_obs=raymap_obs)))

    assert encoder.output_shape == (2, 512)
    assert feats.shape == (3, 2, 512)
    assert np.all(np.isfinite(feats))


def test_plucker_fusion_none_is_exact_parameter_free_identity():
    model = JaxPluckerFusion(mode="none")
    rgb = jax.random.normal(jax.random.PRNGKey(0), (2, 5, 7, 3))

    variables = model.init(jax.random.PRNGKey(1), rgb)
    actual = model.apply(variables, rgb)

    assert "params" not in variables
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(rgb))


@pytest.mark.parametrize(
    ("mode", "use_plucker", "expected"),
    [
        (None, None, "none"),
        ("act_late", None, "act_late"),
        (None, True, "projected_late"),
        ("dp_early", True, "dp_early"),
        ("dp_early", False, "none"),
    ],
)
def test_plucker_fusion_mode_enable_gate(mode, use_plucker, expected):
    assert normalize_plucker_fusion_mode(mode, use_plucker=use_plucker) == expected


def test_plucker_dp_early_fusion_preserves_rgb_then_ray_channel_order():
    model = JaxPluckerFusion(mode="dp_early")
    rgb = jnp.full((2, 5, 7, 3), 2.0, dtype=jnp.float32)
    raymap = jnp.arange(6, dtype=jnp.float32).reshape((1, 1, 1, 6))
    raymap = jnp.broadcast_to(raymap, (2, 5, 7, 6))

    variables = model.init(jax.random.PRNGKey(0), rgb, raymap)
    actual = np.asarray(model.apply(variables, rgb, raymap))

    assert "params" not in variables
    assert actual.shape == (2, 5, 7, 9)
    np.testing.assert_array_equal(actual[..., :3], np.full((2, 5, 7, 3), 2.0))
    np.testing.assert_array_equal(actual[0, 0, 0, 3:], np.arange(6, dtype=np.float32))


def test_plucker_act_late_module_matches_official_cnn_topology():
    model = JaxPluckerFusion(
        mode="act_late",
        plucker_out_channels=512,
        plucker_hidden_channels=64,
    )
    rgb_features = jnp.zeros((1, 1, 1, 512), dtype=jnp.float32)
    raymap = jnp.zeros((1, 32, 32, 6), dtype=jnp.float32)

    variables = model.init(jax.random.PRNGKey(0), rgb_features, raymap)
    actual = model.apply(variables, rgb_features, raymap)
    params = variables["params"]["plucker_encoder"]

    assert actual.shape == (1, 1, 1, 1024)
    assert [params[f"conv_{index}"]["kernel"].shape for index in range(5)] == [
        (7, 7, 6, 64),
        (3, 3, 64, 128),
        (3, 3, 128, 256),
        (3, 3, 256, 512),
        (3, 3, 512, 512),
    ]


def test_plucker_cnn_uses_torch_explicit_symmetric_padding():
    model = JaxPluckerEncoder(out_channels=1, hidden_channels=1)
    raymap = jnp.arange(32 * 32 * 6, dtype=jnp.float32).reshape((1, 32, 32, 6))
    raymap = raymap / float(32 * 32 * 6)
    variables = model.init(jax.random.PRNGKey(0), raymap)
    unit_variables = jax.tree.map(jnp.ones_like, variables)

    actual = model.apply(unit_variables, raymap)
    expected = raymap
    for index, kernel_size in enumerate((7, 3, 3, 3, 3)):
        kernel = unit_variables["params"][f"conv_{index}"]["kernel"]
        pad = kernel_size // 2
        expected = jax.lax.conv_general_dilated(
            expected,
            kernel,
            window_strides=(2, 2),
            padding=((pad, pad), (pad, pad)),
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        expected = jax.nn.relu(expected / np.sqrt(1.0 + 1e-5))

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-6)


def test_jax_resnet_act_late_returns_unprojected_concat(monkeypatch):
    monkeypatch.setattr(
        "robobase.models.encoder._load_resnet_feature_model",
        lambda model_name, pretrained=False: _fake_resnet_feature_model(),
    )
    encoder = JaxResNetEncoder(
        input_shape=(2, 3, 8, 8),
        model="resnet18",
        jit=False,
        use_plucker=True,
        plucker_fusion_mode="act_late",
        plucker_hidden_channels=4,
    )
    rgb = jnp.zeros((3, 2, 3, 8, 8), dtype=jnp.uint8)
    raymap = jnp.zeros((3, 2, 6, 8, 8), dtype=jnp.float32)

    actual = encoder.apply_trainable_spatial(
        encoder.trainable_params,
        rgb,
        raymap_obs=raymap,
    )

    assert encoder.output_shape == (2, 1024)
    assert set(encoder.trainable_params) == {"resnet", "plucker"}
    assert actual.shape == (3, 2, 1, 1, 1024)


def test_dp_early_fusion_encoder_runtime_surface_and_camera_params():
    encoder = JaxDPEarlyFusionEncoder(
        input_shape=(2, 3, 32, 32),
        jit=False,
        pretrained=False,
        plucker_fusion_mode="dp_early",
    )
    rgb = jnp.zeros((2, 2, 3, 32, 32), dtype=jnp.uint8)
    intrinsic = jnp.broadcast_to(jnp.eye(3), (2, 2, 1, 3, 3))
    c2w = jnp.broadcast_to(jnp.eye(4), (2, 2, 1, 4, 4))

    actual = encoder.apply_trainable(
        encoder.trainable_params,
        rgb,
        camera_intrinsic_obs=intrinsic,
        camera_c2w_obs=c2w,
    )

    assert encoder.output_shape == (2, 64)
    assert encoder.batch_stats is None
    assert encoder.frozen_state_dict() == {}
    assert actual.shape == (2, 2, 64)
    assert np.all(np.isfinite(np.asarray(actual)))


def test_dp_early_fusion_encoder_initialization_uses_experiment_seed():
    encoder_a = JaxDPEarlyFusionEncoder(
        input_shape=(1, 3, 8, 8),
        jit=False,
        pretrained=False,
        plucker_fusion_mode="dp_early",
        seed=3,
    )
    encoder_b = JaxDPEarlyFusionEncoder(
        input_shape=(1, 3, 8, 8),
        jit=False,
        pretrained=False,
        plucker_fusion_mode="dp_early",
        seed=4,
    )
    encoder_a_repeat = JaxDPEarlyFusionEncoder(
        input_shape=(1, 3, 8, 8),
        jit=False,
        pretrained=False,
        plucker_fusion_mode="dp_early",
        seed=3,
    )
    leaves_a = jax.tree_util.tree_leaves(encoder_a.trainable_params)
    leaves_b = jax.tree_util.tree_leaves(encoder_b.trainable_params)
    leaves_a_repeat = jax.tree_util.tree_leaves(encoder_a_repeat.trainable_params)

    assert len(leaves_a) == len(leaves_b)
    assert all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(leaves_a, leaves_a_repeat)
    )
    assert any(
        not np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(leaves_a, leaves_b)
    )


def test_dp_early_fusion_rgb_preprocessing_is_zero_one_without_imagenet_norm():
    encoder = JaxDPEarlyFusionEncoder(
        input_shape=(1, 3, 8, 8),
        jit=False,
        pretrained=False,
        plucker_fusion_mode="none",
    )
    rgb = jnp.zeros((1, 1, 3, 8, 8), dtype=jnp.uint8)
    rgb = rgb.at[:, :, 0].set(255)

    actual, *_ = encoder._preprocess_rgb(rgb)

    np.testing.assert_allclose(
        np.asarray(actual[..., 0]),
        np.ones((1, 8, 8)),
        atol=1e-7,
    )
    np.testing.assert_array_equal(np.asarray(actual[..., 1:]), np.zeros((1, 8, 8, 2)))


def test_jax_resnet_encoder_builds_plucker_from_camera_params(monkeypatch):
    monkeypatch.setattr(
        "robobase.models.encoder._load_resnet_feature_model",
        lambda model_name, pretrained=False: _fake_resnet_feature_model(),
    )

    encoder = JaxResNetEncoder(
        input_shape=(2, 3, 8, 8),
        model="resnet18",
        jit=True,
        use_plucker=True,
        plucker_hidden_channels=4,
    )
    rgb_obs = np.random.randint(0, 255, size=(3, 2, 3, 8, 8), dtype=np.uint8)
    intrinsics = np.tile(np.eye(3, dtype=np.float32), (3, 2, 1, 1, 1))
    c2ws = np.tile(np.eye(4, dtype=np.float32), (3, 2, 1, 1, 1))

    feats = np.asarray(
        jax.device_get(
            encoder.encode(
                rgb_obs,
                camera_intrinsic_obs=intrinsics,
                camera_c2w_obs=c2ws,
            )
        )
    )

    assert feats.shape == (3, 2, 512)
    assert np.all(np.isfinite(feats))


def test_jax_resnet_plucker_trainable_params_include_resnet_adapter_and_fusion(
    monkeypatch,
):
    monkeypatch.setattr(
        "robobase.models.encoder._load_resnet_feature_model",
        lambda model_name, pretrained=False: _fake_resnet_feature_model(),
    )

    encoder = JaxResNetEncoder(
        input_shape=(2, 3, 8, 8),
        model="resnet18",
        jit=False,
        use_plucker=True,
        plucker_hidden_channels=4,
    )
    rgb_obs = np.random.randint(0, 255, size=(3, 2, 3, 8, 8), dtype=np.uint8)
    raymap_obs = np.ones((3, 2, 6, 8, 8), dtype=np.float32)

    assert set(encoder.trainable_params.keys()) == {"resnet", "plucker", "fusion"}
    feats = np.asarray(
        jax.device_get(
            encoder.apply_trainable(
                encoder.trainable_params,
                jnp.asarray(rgb_obs),
                raymap_obs=jnp.asarray(raymap_obs),
            )
        )
    )

    assert feats.shape == (3, 2, 512)
    assert np.all(np.isfinite(feats))


def test_jax_resnet_plucker_identity_init_preserves_rgb_fusion(monkeypatch):
    monkeypatch.setattr(
        "robobase.models.encoder._load_resnet_feature_model",
        lambda model_name, pretrained=False: _fake_resnet_feature_model(),
    )

    encoder = JaxResNetEncoder(
        input_shape=(2, 3, 8, 8),
        model="resnet18",
        jit=False,
        use_plucker=True,
        plucker_hidden_channels=4,
        plucker_identity_init=True,
    )
    kernel = np.asarray(
        encoder.trainable_params["fusion"]["input_proj"]["kernel"][0, 0]
    )
    bias = np.asarray(encoder.trainable_params["fusion"]["input_proj"]["bias"])

    np.testing.assert_allclose(kernel[:512], np.eye(512, dtype=np.float32))
    np.testing.assert_allclose(kernel[512:], np.zeros((512, 512), dtype=np.float32))
    np.testing.assert_allclose(bias, np.zeros((512,), dtype=np.float32))


def test_jax_bc_supports_pixel_inputs_with_multicam_fusion(monkeypatch):
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


def test_jax_bc_supports_plucker_raymaps_with_trainable_encoder(monkeypatch):
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
            "raymap_front": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1, 6, 8, 8),
                dtype=np.float32,
            ),
            "rgb_wrist": spaces.Box(
                low=0,
                high=255,
                shape=(1, 3, 8, 8),
                dtype=np.uint8,
            ),
            "raymap_wrist": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1, 6, 8, 8),
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
    agent = _make_jax_bc(
        observation_space=observation_space,
        action_space=action_space,
        output_sequence_network_type="rnn",
        use_plucker=True,
        trainable_encoder=True,
    )

    obs = {
        "low_dim_state": np.zeros((2, 1, 4), dtype=np.float32),
        "rgb_front": np.full((2, 1, 3, 8, 8), 64, dtype=np.uint8),
        "raymap_front": np.ones((2, 1, 6, 8, 8), dtype=np.float32),
        "rgb_wrist": np.full((2, 1, 3, 8, 8), 128, dtype=np.uint8),
        "raymap_wrist": np.ones((2, 1, 6, 8, 8), dtype=np.float32),
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


def test_jax_bc_supports_plucker_camera_params_with_trainable_encoder(monkeypatch):
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
    agent = _make_jax_bc(
        observation_space=observation_space,
        action_space=action_space,
        output_sequence_network_type="rnn",
        use_plucker=True,
        trainable_encoder=True,
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


def test_jax_bc_rejects_plucker_with_frozen_encoder():
    observation_space = spaces.Dict(
        {
            "rgb_front": spaces.Box(
                low=0,
                high=255,
                shape=(1, 3, 8, 8),
                dtype=np.uint8,
            ),
            "raymap_front": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1, 6, 8, 8),
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

    with pytest.raises(ValueError, match="trainable=true"):
        _make_jax_bc(
            observation_space=observation_space,
            action_space=action_space,
            output_sequence_network_type="rnn",
            use_plucker=True,
            trainable_encoder=False,
        )
