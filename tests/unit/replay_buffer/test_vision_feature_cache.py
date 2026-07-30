from pathlib import Path

import numpy as np
import pytest
from gymnasium import spaces
from omegaconf import OmegaConf

from robobase.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from robobase.replay_buffer.vision_feature_cache import (
    JAX_ACT_FEATURE_KEY,
    JAX_CQN_AS_FEATURE_KEY,
    JAX_DIFFUSION_FEATURE_KEY,
    JAX_FLOW_MATCHING_FEATURE_KEY,
    JAX_Q_CHUNKING_FEATURE_KEY,
    VISION_CACHE_FINGERPRINT_KEY,
    build_vision_feature_cache_plan,
)


def _make_diffusion_cfg(cache_setting="auto"):
    return OmegaConf.create(
        {
            "action_sequence": 16,
            "method": {
                "name": "diffusion",
                "actor_model": {
                    "type": "conditional_unet1d",
                    "sequence_length": 16,
                },
                "encoder_model": {
                    "type": "resnet",
                    "model": "resnet18",
                    "trainable": False,
                    "pretrained": True,
                },
                "view_fusion_model": {
                    "type": "multicam_feature",
                    "mode": "flatten",
                },
            },
            "replay": {
                "cache_frozen_image_features": cache_setting,
                "cache_frozen_image_feature_backends": ["jax"],
            },
        }
    )


def _make_flow_matching_cfg(cache_setting="auto", *, seed=0):
    return OmegaConf.create(
        {
            "action_sequence": 16,
            "seed": seed,
            "method": {
                "name": "flow_matching",
                "backbone": {
                    "type": "fully_connected",
                    "sequence_length": 16,
                },
                "encoder_model": {
                    "type": "resnet",
                    "model": "resnet18",
                    "trainable": False,
                    "pretrained": False,
                },
                "view_fusion_model": {
                    "type": "multicam_feature",
                    "mode": "flatten",
                },
            },
            "replay": {
                "cache_frozen_image_features": cache_setting,
                "cache_frozen_image_feature_backends": ["jax"],
            },
        }
    )


def _make_act_cfg(cache_setting="auto"):
    return OmegaConf.create(
        {
            "action_sequence": 16,
            "method": {
                "name": "act",
                "actor_model": {
                    "type": "transformer",
                    "num_queries": 16,
                },
                "encoder_model": {
                    "type": "resnet",
                    "model": "resnet18",
                    "trainable": False,
                },
                "view_fusion_model": {
                    "type": "multicam_feature",
                    "mode": "flatten",
                },
            },
            "replay": {
                "cache_frozen_image_features": cache_setting,
                "cache_frozen_image_feature_backends": ["jax"],
            },
        }
    )


def _make_cqn_as_cfg(cache_setting="auto"):
    return OmegaConf.create(
        {
            "action_sequence": 4,
            "seed": 0,
            "method": {
                "name": "cqn_as",
                "encoder_model": {
                    "type": "resnet",
                    "model": "resnet18",
                    "trainable": False,
                    "pretrained": False,
                },
                "view_fusion_model": {
                    "type": "multicam_feature",
                    "mode": "flatten",
                },
            },
            "replay": {
                "cache_frozen_image_features": cache_setting,
                "cache_frozen_image_feature_backends": ["jax"],
            },
        }
    )


def _make_cqn_flow_cfg(cache_setting="auto"):
    cfg = _make_cqn_as_cfg(cache_setting)
    cfg.method.name = "cqn_flow"
    return cfg


def _make_q_chunking_cfg(cache_setting="auto"):
    cfg = _make_cqn_as_cfg(cache_setting)
    cfg.action_sequence = 5
    cfg.method.name = "q_chunking"
    return cfg


def _make_pixel_obs_space():
    return spaces.Dict(
        {
            "low_dim_state": spaces.Box(
                low=-1.0, high=1.0, shape=(1, 4), dtype=np.float32,
            ),
            "rgb0": spaces.Box(
                low=0, high=255, shape=(1, 3, 16, 16), dtype=np.uint8,
            ),
            "rgb1": spaces.Box(
                low=0, high=255, shape=(1, 3, 16, 16), dtype=np.uint8,
            ),
        }
    )


def test_build_vision_feature_cache_plan_replaces_raw_rgb_with_cached_features():
    plan = build_vision_feature_cache_plan(
        cfg=_make_diffusion_cfg(),
        observation_space=_make_pixel_obs_space(),
        save_dir=None,
        reuse_saved=False,
    )

    assert plan is not None
    assert "rgb0" not in plan.observation_space.spaces
    assert "rgb1" not in plan.observation_space.spaces
    assert JAX_DIFFUSION_FEATURE_KEY in plan.observation_space.spaces
    assert plan.observation_space[JAX_DIFFUSION_FEATURE_KEY].shape == (1, 1024)
    assert VISION_CACHE_FINGERPRINT_KEY in plan.observation_space.spaces


def test_cache_plan_supports_flow_matching_act_and_rl_value_feature_keys():
    flow_plan = build_vision_feature_cache_plan(
        cfg=_make_flow_matching_cfg(),
        observation_space=_make_pixel_obs_space(),
        save_dir=None,
        reuse_saved=False,
    )
    act_plan = build_vision_feature_cache_plan(
        cfg=_make_act_cfg(),
        observation_space=_make_pixel_obs_space(),
        save_dir=None,
        reuse_saved=False,
    )
    cqn_as_plan = build_vision_feature_cache_plan(
        cfg=_make_cqn_as_cfg(),
        observation_space=_make_pixel_obs_space(),
        save_dir=None,
        reuse_saved=False,
    )
    cqn_flow_plan = build_vision_feature_cache_plan(
        cfg=_make_cqn_flow_cfg(),
        observation_space=_make_pixel_obs_space(),
        save_dir=None,
        reuse_saved=False,
    )
    q_chunking_plan = build_vision_feature_cache_plan(
        cfg=_make_q_chunking_cfg(),
        observation_space=_make_pixel_obs_space(),
        save_dir=None,
        reuse_saved=False,
    )

    assert JAX_FLOW_MATCHING_FEATURE_KEY in flow_plan.observation_space.spaces
    assert flow_plan.observation_space[JAX_FLOW_MATCHING_FEATURE_KEY].shape == (1, 1024)
    assert JAX_ACT_FEATURE_KEY in act_plan.observation_space.spaces
    assert act_plan.observation_space[JAX_ACT_FEATURE_KEY].shape == (1, 1024)
    assert JAX_CQN_AS_FEATURE_KEY in cqn_as_plan.observation_space.spaces
    assert cqn_as_plan.observation_space[JAX_CQN_AS_FEATURE_KEY].shape == (1, 1024)
    assert JAX_CQN_AS_FEATURE_KEY in cqn_flow_plan.observation_space.spaces
    assert cqn_flow_plan.observation_space[JAX_CQN_AS_FEATURE_KEY].shape == (1, 1024)
    assert cqn_flow_plan.fingerprint == cqn_as_plan.fingerprint
    assert JAX_Q_CHUNKING_FEATURE_KEY in q_chunking_plan.observation_space.spaces
    assert q_chunking_plan.observation_space[JAX_Q_CHUNKING_FEATURE_KEY].shape == (
        1,
        1024,
    )


def test_feature_preprocessor_rewrites_raw_transition(monkeypatch):
    plan = build_vision_feature_cache_plan(
        cfg=_make_diffusion_cfg(),
        observation_space=_make_pixel_obs_space(),
        save_dir=None,
        reuse_saved=False,
    )
    preprocessor = plan.preprocessing_fn[0]

    monkeypatch.setattr(
        preprocessor,
        "_encode_jax",
        lambda rgb_batch: np.full((rgb_batch.shape[0], 1024), 2.0, dtype=np.float32),
    )

    transition = {
        "low_dim_state": np.zeros((4,), dtype=np.float32),
        "rgb0": np.zeros((3, 16, 16), dtype=np.uint8),
        "rgb1": np.ones((3, 16, 16), dtype=np.uint8),
    }
    processed = preprocessor([transition])[0]

    assert "rgb0" not in processed
    assert "rgb1" not in processed
    assert processed[JAX_DIFFUSION_FEATURE_KEY].shape == (1024,)
    assert np.all(processed[JAX_DIFFUSION_FEATURE_KEY] == 2.0)
    assert (
        processed[VISION_CACHE_FINGERPRINT_KEY].tobytes().hex()
        == plan.fingerprint
    )


def test_feature_preprocessor_forwards_pretrained_to_jax_encoder(monkeypatch):
    plan = build_vision_feature_cache_plan(
        cfg=_make_diffusion_cfg(),
        observation_space=_make_pixel_obs_space(),
        save_dir=None,
        reuse_saved=False,
    )
    preprocessor = plan.preprocessing_fn[0]
    captured = {}

    class FakeEncoder:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def encode(self, rgb_batch):
            return np.zeros(
                (rgb_batch.shape[0], rgb_batch.shape[1], 512),
                dtype=np.float32,
            )

    monkeypatch.setattr("robobase.models.encoder.JaxResNetEncoder", FakeEncoder)

    preprocessor._encode_jax(np.zeros((2, 2, 3, 16, 16), dtype=np.uint8))

    assert captured["pretrained"] is True
    assert captured["seed"] == 0


def test_random_encoder_seed_is_forwarded_and_changes_cache_fingerprint(monkeypatch):
    first = build_vision_feature_cache_plan(
        cfg=_make_flow_matching_cfg(seed=0),
        observation_space=_make_pixel_obs_space(),
        save_dir=None,
        reuse_saved=False,
    )
    second = build_vision_feature_cache_plan(
        cfg=_make_flow_matching_cfg(seed=7),
        observation_space=_make_pixel_obs_space(),
        save_dir=None,
        reuse_saved=False,
    )
    captured = {}

    class FakeEncoder:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def encode(self, rgb_batch):
            return np.zeros(
                (rgb_batch.shape[0], rgb_batch.shape[1], 512),
                dtype=np.float32,
            )

    monkeypatch.setattr("robobase.models.encoder.JaxResNetEncoder", FakeEncoder)
    second.preprocessing_fn[0]._encode_jax(
        np.zeros((1, 2, 3, 16, 16), dtype=np.uint8)
    )

    assert first.fingerprint != second.fingerprint
    assert captured["seed"] == 7


def test_feature_cache_v3_key_rejects_legacy_random_feature_cache(tmp_path: Path):
    np.savez(
        tmp_path / "episode_0_2_0.npz",
        vision_features_jax_diffusion=np.zeros((2, 1024), dtype=np.float32),
    )

    assert JAX_DIFFUSION_FEATURE_KEY == "vision_features_jax_diffusion_v3"
    with pytest.raises(ValueError, match="does not contain cached image features"):
        build_vision_feature_cache_plan(
            cfg=_make_diffusion_cfg(),
            observation_space=_make_pixel_obs_space(),
            save_dir=str(tmp_path),
            reuse_saved=True,
        )


def test_cache_plan_accepts_matching_saved_fingerprint(tmp_path: Path):
    initial_plan = build_vision_feature_cache_plan(
        cfg=_make_diffusion_cfg(),
        observation_space=_make_pixel_obs_space(),
        save_dir=None,
        reuse_saved=False,
    )
    encoded_fingerprint = np.frombuffer(
        bytes.fromhex(initial_plan.fingerprint),
        dtype=np.uint8,
    )
    np.savez(
        tmp_path / "episode_0_2_0.npz",
        **{
            JAX_DIFFUSION_FEATURE_KEY: np.zeros((2, 1024), dtype=np.float32),
            VISION_CACHE_FINGERPRINT_KEY: np.repeat(
                encoded_fingerprint[None],
                2,
                axis=0,
            ),
        },
    )

    reused_plan = build_vision_feature_cache_plan(
        cfg=_make_diffusion_cfg(),
        observation_space=_make_pixel_obs_space(),
        save_dir=str(tmp_path),
        reuse_saved=True,
    )

    assert reused_plan is not None
    assert reused_plan.fingerprint == initial_plan.fingerprint


def test_cache_plan_rejects_fingerprint_from_different_encoder(tmp_path: Path):
    initial_plan = build_vision_feature_cache_plan(
        cfg=_make_diffusion_cfg(),
        observation_space=_make_pixel_obs_space(),
        save_dir=None,
        reuse_saved=False,
    )
    encoded_fingerprint = np.frombuffer(
        bytes.fromhex(initial_plan.fingerprint),
        dtype=np.uint8,
    )
    np.savez(
        tmp_path / "episode_0_2_0.npz",
        **{
            JAX_DIFFUSION_FEATURE_KEY: np.zeros((2, 1024), dtype=np.float32),
            VISION_CACHE_FINGERPRINT_KEY: np.repeat(
                encoded_fingerprint[None],
                2,
                axis=0,
            ),
        },
    )
    changed_cfg = _make_diffusion_cfg()
    changed_cfg.method.encoder_model.model = "resnet34"

    with pytest.raises(ValueError, match="fingerprint"):
        build_vision_feature_cache_plan(
            cfg=changed_cfg,
            observation_space=_make_pixel_obs_space(),
            save_dir=str(tmp_path),
            reuse_saved=True,
        )


def test_cache_plan_falls_back_when_reusing_raw_saved_replay(tmp_path: Path):
    np.savez(
        tmp_path / "episode_0_2_0.npz",
        low_dim_state=np.zeros((2, 4), dtype=np.float32),
        rgb0=np.zeros((2, 3, 16, 16), dtype=np.uint8),
        rgb1=np.zeros((2, 3, 16, 16), dtype=np.uint8),
        action=np.zeros((2, 2), dtype=np.float32),
        reward=np.zeros((2,), dtype=np.float32),
        terminal=np.zeros((2,), dtype=np.int8),
        truncated=np.zeros((2,), dtype=np.int8),
    )

    plan = build_vision_feature_cache_plan(
        cfg=_make_diffusion_cfg(),
        observation_space=_make_pixel_obs_space(),
        save_dir=str(tmp_path),
        reuse_saved=True,
    )

    assert plan is None


def test_uniform_replay_buffer_preprocessing_runs_before_type_check():
    feature_key = JAX_DIFFUSION_FEATURE_KEY
    observation_space = spaces.Dict(
        {
            "low_dim_state": spaces.Box(
                low=-1.0, high=1.0, shape=(1, 4), dtype=np.float32,
            ),
            feature_key: spaces.Box(
                low=-np.inf, high=np.inf, shape=(1, 8), dtype=np.float32,
            ),
        }
    )

    def derive_feature(transitions):
        processed = []
        for transition in transitions:
            new_transition = dict(transition)
            new_transition.pop("rgb0")
            new_transition[feature_key] = np.full((8,), 3.0, dtype=np.float32)
            processed.append(new_transition)
        return processed

    buffer = UniformReplayBuffer(
        observation_elements=observation_space,
        replay_capacity=8,
        nstep=1,
        action_shape=(1, 2),
        batch_size=2,
        preprocessing_fn=[derive_feature],
    )

    obs = {
        "low_dim_state": np.zeros((4,), dtype=np.float32),
        "rgb0": np.zeros((3, 16, 16), dtype=np.uint8),
    }
    next_obs = {
        "low_dim_state": np.ones((4,), dtype=np.float32),
        "rgb0": np.ones((3, 16, 16), dtype=np.uint8),
    }

    buffer.add(obs, np.zeros((2,), dtype=np.float32), np.array(0.0, dtype=np.float32), True, False)
    buffer.add_final(next_obs)

    sample = buffer.sample(batch_size=1)
    assert feature_key in sample
    assert feature_key + "_tp1" in sample
    assert sample[feature_key].shape == (1, 1, 8)
    assert sample[feature_key + "_tp1"].shape == (1, 1, 8)
    assert np.all(sample[feature_key] == 3.0)
    assert np.all(sample[feature_key + "_tp1"] == 3.0)
