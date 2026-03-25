from pathlib import Path

import numpy as np
from gymnasium import spaces
from omegaconf import OmegaConf

from robobase.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from robobase.replay_buffer.vision_feature_cache import (
    JAX_DIFFUSION_FEATURE_KEY,
    TORCH_DIFFUSION_FEATURE_KEY,
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
                },
                "view_fusion_model": {
                    "type": "multicam_feature",
                    "mode": "flatten",
                },
            },
            "replay": {
                "cache_frozen_image_features": cache_setting,
                "cache_frozen_image_feature_backends": ["torch", "jax"],
            },
        }
    )


def _make_pixel_obs_space():
    return spaces.Dict(
        {
            "low_dim_state": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1, 4),
                dtype=np.float32,
            ),
            "rgb0": spaces.Box(
                low=0,
                high=255,
                shape=(1, 3, 16, 16),
                dtype=np.uint8,
            ),
            "rgb1": spaces.Box(
                low=0,
                high=255,
                shape=(1, 3, 16, 16),
                dtype=np.uint8,
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
    assert TORCH_DIFFUSION_FEATURE_KEY in plan.observation_space.spaces
    assert JAX_DIFFUSION_FEATURE_KEY in plan.observation_space.spaces
    assert plan.observation_space[TORCH_DIFFUSION_FEATURE_KEY].shape == (1, 1024)
    assert plan.observation_space[JAX_DIFFUSION_FEATURE_KEY].shape == (1, 1024)


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
        "_encode_torch",
        lambda rgb_batch: np.full((rgb_batch.shape[0], 1024), 1.0, dtype=np.float32),
    )
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
    assert processed[TORCH_DIFFUSION_FEATURE_KEY].shape == (1024,)
    assert processed[JAX_DIFFUSION_FEATURE_KEY].shape == (1024,)
    assert np.all(processed[TORCH_DIFFUSION_FEATURE_KEY] == 1.0)
    assert np.all(processed[JAX_DIFFUSION_FEATURE_KEY] == 2.0)


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
    feature_key = TORCH_DIFFUSION_FEATURE_KEY
    observation_space = spaces.Dict(
        {
            "low_dim_state": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1, 4),
                dtype=np.float32,
            ),
            feature_key: spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1, 8),
                dtype=np.float32,
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

    buffer.add(
        obs,
        np.zeros((2,), dtype=np.float32),
        np.array(0.0, dtype=np.float32),
        True,
        False,
    )
    buffer.add_final(next_obs)

    sample = buffer.sample(batch_size=1)
    assert feature_key in sample
    assert feature_key + "_tp1" in sample
    assert sample[feature_key].shape == (1, 1, 8)
    assert sample[feature_key + "_tp1"].shape == (1, 1, 8)
    assert np.all(sample[feature_key] == 3.0)
    assert np.all(sample[feature_key + "_tp1"] == 3.0)
