from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnasium import spaces
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from robobase.factory import create_agent, method_name_from_cfg
from robobase.method.drqv2 import (
    DrQV2,
    drqv2_sample_unit_action,
    drqv2_td_target,
)
from robobase.models.encoder import JaxDrQV2Encoder
from robobase.replay_buffer.vision_feature_cache import JAX_DRQV2_FEATURE_KEY


CONFIG_DIR = str((Path(__file__).parents[2] / "robobase" / "cfgs").resolve())


def _config(*overrides, jit=True):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(
            config_name="robobase_config",
            overrides=[
                "method=drqv2",
                "method.stddev_schedule=0.2",
                "method.model.hidden_dims=[32,32]",
                "method.model.feature_dim=16",
                "num_train_envs=1",
                "num_eval_envs=1",
                f"backend.jit={'true' if jit else 'false'}",
                "backend.platform=cpu",
                *overrides,
            ],
        )


def _spaces(*, pixels=False):
    observations = {
        "low_dim_state": spaces.Box(
            -np.inf,
            np.inf,
            shape=(1, 5),
            dtype=np.float32,
        )
    }
    if pixels:
        observations["rgb_front"] = spaces.Box(
            0,
            255,
            shape=(1, 3, 16, 16),
            dtype=np.uint8,
        )
    return spaces.Dict(observations), spaces.Box(
        -1.0,
        1.0,
        shape=(1, 2),
        dtype=np.float32,
    )


def _batch(batch_size=4, *, pixels=False, demos=False):
    rng = np.random.default_rng(7)
    batch = {
        "low_dim_state": rng.normal(size=(batch_size, 1, 5)).astype(np.float32),
        "low_dim_state_tp1": rng.normal(size=(batch_size, 1, 5)).astype(
            np.float32
        ),
        "action": rng.uniform(-1.0, 1.0, size=(batch_size, 1, 2)).astype(
            np.float32
        ),
        "reward": rng.normal(size=(batch_size,)).astype(np.float32),
        "discount": np.full((batch_size,), 0.99, dtype=np.float32),
        "terminal": np.zeros((batch_size,), dtype=bool),
        "truncated": np.zeros((batch_size,), dtype=bool),
        "demo": np.ones((batch_size,), dtype=np.uint8)
        if demos
        else np.zeros((batch_size,), dtype=np.uint8),
    }
    if pixels:
        batch["rgb_front"] = rng.integers(
            0,
            256,
            size=(batch_size, 1, 3, 16, 16),
            dtype=np.uint8,
        )
        batch["rgb_front_tp1"] = rng.integers(
            0,
            256,
            size=(batch_size, 1, 3, 16, 16),
            dtype=np.uint8,
        )
    return batch


def _tree_changed(before, after):
    before_leaves, before_tree = jax.tree.flatten(before)
    after_leaves, after_tree = jax.tree.flatten(after)
    assert before_tree == after_tree
    return any(
        not np.allclose(np.asarray(left), np.asarray(right))
        for left, right in zip(before_leaves, after_leaves, strict=True)
    )


def test_drqv2_launch_uses_reference_algorithm_and_modular_encoder():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=["launch=drqv2_pixel_dmc"],
        )

    assert method_name_from_cfg(cfg) == "drqv2"
    assert cfg.method._target_ == "robobase.method.drqv2.DrQV2"
    assert cfg.method.encoder_model.type == "drqv2"
    assert cfg.method.encoder_model.trainable
    assert cfg.method.encoder_model.num_downsample_convs == 1
    assert cfg.method.encoder_model.num_post_downsample_convs == 3
    assert cfg.method.encoder_model.channels == 32
    assert cfg.method.model.feature_dim == 50
    assert list(cfg.method.model.hidden_dims) == [1024, 1024]
    assert cfg.method.model.norm == "layer"
    assert cfg.method.num_critics == 2
    assert cfg.method.augmentation_pad == 4
    assert cfg.replay.nstep == 3
    assert cfg.update_every_steps == 2


def test_drqv2_reference_encoder_has_official_84px_representation_shape():
    encoder = JaxDrQV2Encoder(
        (1, 9, 84, 84),
        jit=False,
        seed=3,
    )

    # 84 -> 41 after the stride-2 conv, then 39 -> 37 -> 35.
    assert encoder.output_shape == (1, 32 * 35 * 35)


def test_drqv2_target_has_no_sac_entropy_term():
    target = drqv2_td_target(
        jnp.asarray([1.0, 2.0]),
        jnp.asarray([0.9, 0.9]),
        jnp.asarray([1.0, 0.0]),
        jnp.asarray([3.0, 10.0]),
    )

    np.testing.assert_allclose(np.asarray(target), [3.7, 2.0], rtol=1e-6)


def test_drqv2_target_policy_noise_is_clipped_before_action_clamp():
    mean = jnp.zeros((128, 3), dtype=jnp.float32)
    sampled = drqv2_sample_unit_action(
        mean,
        jax.random.PRNGKey(5),
        100.0,
        noise_clip=0.3,
    )

    assert np.max(np.abs(np.asarray(sampled))) <= 0.300001
    assert np.any(np.abs(np.asarray(sampled)) > 0.29)


def test_drqv2_jitted_update_changes_actor_and_twin_critic():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _config(jit=True),
        observation_space=observation_space,
        action_space=action_space,
    )
    assert isinstance(agent, DrQV2)
    agent.logging = True
    before_actor = jax.tree.map(np.asarray, agent.params["actor"])
    before_critic = jax.tree.map(np.asarray, agent.params["critic"])

    metrics = agent.update(iter([_batch()]), step=1)

    assert np.isfinite(metrics["actor_loss"])
    assert np.isfinite(metrics["critic_loss"])
    assert metrics["actor_bc_loss"] == pytest.approx(0.0)
    assert _tree_changed(before_actor, agent.params["actor"])
    assert _tree_changed(before_critic, agent.params["critic"])


def test_drqv2_pixel_update_uses_random_shift_and_trains_shared_encoder():
    observation_space, action_space = _spaces(pixels=True)
    agent = create_agent(
        _config(
            "pixels=true",
            "method.encoder_model.channels=16",
            jit=False,
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    assert isinstance(agent.encoder, JaxDrQV2Encoder)
    assert agent._cached_pixel_feature_key == JAX_DRQV2_FEATURE_KEY
    before_encoder = jax.tree.map(np.asarray, agent.params["encoder"])

    agent.update(iter([_batch(batch_size=8, pixels=True)]), step=1)

    assert _tree_changed(before_encoder, agent.params["encoder"])


def test_drqv2_demo_bc_is_an_explicit_additive_actor_objective():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _config("method.bc_lambda=1.0", jit=False),
        observation_space=observation_space,
        action_space=action_space,
    )
    agent.logging = True

    metrics = agent.update(iter([_batch(demos=True)]), step=1)

    assert metrics["ratio_of_demos"] == pytest.approx(1.0)
    assert metrics["actor_bc_loss"] > 0.0
    assert metrics["actor_loss"] == pytest.approx(
        metrics["actor_rl_loss"] + metrics["actor_bc_loss"],
        rel=1e-5,
    )


def test_drqv2_checkpoint_restores_target_critic_and_rng():
    observation_space, action_space = _spaces()
    cfg = _config(jit=False)
    agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    agent.update(iter([_batch()]), step=1)
    restored = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )

    restored.load_state_dict(agent.state_dict())
    restored.load_checkpoint_state_dict(agent.checkpoint_state_dict())

    for expected, actual in zip(
        jax.tree.leaves(agent.target_critic_params),
        jax.tree.leaves(restored.target_critic_params),
        strict=True,
    ):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))
    np.testing.assert_array_equal(
        np.asarray(restored.rng_key),
        np.asarray(agent.rng_key),
    )


def test_drqv2_distributional_legacy_launch_fails_explicitly():
    observation_space, action_space = _spaces()
    cfg = _config("method.distributional_critic=true", jit=False)

    with pytest.raises(NotImplementedError, match="canonical scalar twin-Q"):
        create_agent(
            cfg,
            observation_space=observation_space,
            action_space=action_space,
        )
