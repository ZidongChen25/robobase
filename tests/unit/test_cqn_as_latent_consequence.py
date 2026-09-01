from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax  # noqa: E402
import numpy as np  # noqa: E402
from gymnasium import spaces  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402

from robobase.factory import create_agent, method_name_from_cfg  # noqa: E402
from robobase.workspace import _validate_rl_action_sequence  # noqa: E402


CONFIG_DIR = str((Path(__file__).parents[2] / "robobase" / "cfgs").resolve())
K = 16
ACTION_DIM = 2
LOW_DIM = 5


def _compose(method: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(
            config_name="robobase_config",
            overrides=[
                f"method={method}",
                "pixels=true",
                "frame_stack=1",
                "visual_observation_shape=[16,16]",
                f"action_sequence={K}",
                "execution_length=1",
                "temporal_ensemble=false",
                "num_train_envs=1",
                "num_eval_envs=1",
                "num_explore_steps=0",
                "backend.jit=false",
                "backend.platform=cpu",
                "method.model.hidden_dims=[16,16]",
                "method.levels=2",
                "method.bins=5",
                "method.atoms=11",
                *(
                    [
                        "method.latent_consequence_hidden_dims=[16,16]",
                        "method.latent_consequence_ensemble_size=2",
                        "method.latent_consequence_minimum_model_updates=0",
                    ]
                    if method == "cqn_as_latent_consequence"
                    else []
                ),
            ],
        )


def _spaces():
    observation_space = spaces.Dict(
        {
            "low_dim_state": spaces.Box(
                -np.inf, np.inf, (1, LOW_DIM), dtype=np.float32
            ),
            "rgb_head": spaces.Box(
                0, 255, (1, 3, 16, 16), dtype=np.uint8
            ),
        }
    )
    action_space = spaces.Box(
        -1.0, 1.0, (K, ACTION_DIM), dtype=np.float32
    )
    return observation_space, action_space


def _agent(method: str):
    cfg = _compose(method)
    observation_space, action_space = _spaces()
    return cfg, create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )


def _batch(seed: int = 3, batch_size: int = 4):
    rng = np.random.default_rng(seed)
    return {
        "low_dim_state": rng.normal(
            size=(batch_size, 1, LOW_DIM)
        ).astype(np.float32),
        "low_dim_state_tp1": rng.normal(
            size=(batch_size, 1, LOW_DIM)
        ).astype(np.float32),
        "rgb_head": rng.integers(
            0, 255, size=(batch_size, 1, 3, 16, 16), dtype=np.uint8
        ),
        "rgb_head_tp1": rng.integers(
            0, 255, size=(batch_size, 1, 3, 16, 16), dtype=np.uint8
        ),
        "action": rng.uniform(
            -1.0, 1.0, size=(batch_size, K, ACTION_DIM)
        ).astype(np.float32),
        "reward": np.zeros((batch_size,), dtype=np.float32),
        "discount": np.full((batch_size,), 0.99, dtype=np.float32),
        "terminal": np.zeros((batch_size,), dtype=bool),
        "truncated": np.zeros((batch_size,), dtype=bool),
        "demo": np.ones((batch_size,), dtype=np.uint8),
    }


def _assert_tree_equal(left, right):
    for a, b in zip(
        jax.tree.leaves(left), jax.tree.leaves(right), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_factory_registers_rgb_k16_latent_consequence():
    cfg, agent = _agent("cqn_as_latent_consequence")
    assert method_name_from_cfg(cfg) == "cqn_as_latent_consequence"
    _validate_rl_action_sequence(cfg)
    assert agent.base.action_sequence == 16
    assert agent.base.use_pixels
    assert agent.rerank_train is False
    assert agent.rerank_eval is True


def test_bigym_launch_freezes_official_observation_contract():
    """Do not silently compare the floating-base variant to official CQN-AS."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_as_latent_consequence_pixel_bigym_demo_driven",
                "env=bigym/sandwich_remove",
            ],
        )

    assert cfg.env.append_floating_base_to_low_dim is False
    assert cfg.env.obs_std_floor_relative == 0.0


def test_model_uses_only_the_executed_first_action_and_keeps_base_update_exact():
    _, wrapped = _agent("cqn_as_latent_consequence")
    _, plain = _agent("cqn_as")
    batch = _batch()

    inputs = wrapped._model_batch_inputs(batch)
    np.testing.assert_array_equal(np.asarray(inputs[1]), batch["action"][:, 0])
    mutated = {key: np.asarray(value).copy() for key, value in batch.items()}
    mutated["action"][:, 1:] *= -1.0
    mutated_inputs = wrapped._model_batch_inputs(mutated)
    np.testing.assert_array_equal(np.asarray(inputs[1]), np.asarray(mutated_inputs[1]))

    wrapped.update(iter([batch]), step=1)
    plain.update(iter([batch]), step=1)
    _assert_tree_equal(wrapped.base.params, plain.params)
    _assert_tree_equal(
        wrapped.base.target_critic_params, plain.target_critic_params
    )
    assert wrapped.model_updates == 1


def test_eval_reranker_is_finite_and_state_roundtrips():
    _, agent = _agent("cqn_as_latent_consequence")
    rng = np.random.default_rng(7)
    observations = {
        "low_dim_state": rng.normal(size=(1, 1, LOW_DIM)).astype(np.float32),
        "rgb_head": rng.integers(
            0, 255, size=(1, 1, 3, 16, 16), dtype=np.uint8
        ),
    }
    action = agent.act(observations, step=1000, eval_mode=True)
    assert action.shape == (1, K, ACTION_DIM)
    assert np.all(np.isfinite(action))
    assert agent._eval_calls == 1

    state = agent.state_dict()
    checkpoint = agent.checkpoint_state_dict()
    agent.load_state_dict(state)
    agent.load_checkpoint_state_dict(checkpoint)
    assert agent.model_updates == 0
    assert agent._eval_calls == 1
