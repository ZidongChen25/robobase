from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnasium import spaces
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from robobase.factory import create_agent
from robobase.method.cqn import (
    decode_action,
    encode_action,
    project_categorical,
)
from robobase.method.ppo import generalized_advantage_estimate
from robobase.method.rl_common import squashed_normal_sample_and_log_prob


CONFIG_DIR = str((Path(__file__).parents[2] / "robobase" / "cfgs").resolve())


def _state_spaces():
    observation_space = spaces.Dict(
        {
            "low_dim_state": spaces.Box(
                -np.inf,
                np.inf,
                shape=(1, 5),
                dtype=np.float32,
            )
        }
    )
    action_space = spaces.Box(
        -1.0,
        1.0,
        shape=(1, 2),
        dtype=np.float32,
    )
    return observation_space, action_space


def _compose_method(method: str, overrides=()):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(
            config_name="robobase_config",
            overrides=[
                f"method={method}",
                "num_train_envs=1",
                "num_eval_envs=1",
                "backend.jit=false",
                *overrides,
            ],
        )


def _off_policy_batch(batch_size=8):
    rng = np.random.default_rng(5)
    return {
        "low_dim_state": rng.normal(size=(batch_size, 1, 5)).astype(np.float32),
        "low_dim_state_tp1": rng.normal(size=(batch_size, 1, 5)).astype(
            np.float32
        ),
        "action": rng.uniform(-1.0, 1.0, size=(batch_size, 2)).astype(np.float32),
        "reward": rng.normal(size=(batch_size,)).astype(np.float32),
        "discount": np.full((batch_size,), 0.99, dtype=np.float32),
        "terminal": np.zeros((batch_size,), dtype=bool),
        "truncated": np.zeros((batch_size,), dtype=bool),
    }


def _tree_changed(before, after):
    before_leaves, before_tree = jax.tree.flatten(before)
    after_leaves, after_tree = jax.tree.flatten(after)
    assert before_tree == after_tree
    return any(
        not np.allclose(np.asarray(left), np.asarray(right))
        for left, right in zip(before_leaves, after_leaves, strict=True)
    )


def test_squashed_normal_sample_has_finite_log_probability_and_bounds():
    mean = jnp.asarray([[0.0, 10.0, -10.0]], dtype=jnp.float32)
    log_std = jnp.asarray([[0.0, -2.0, 1.0]], dtype=jnp.float32)
    action, log_probability = squashed_normal_sample_and_log_prob(
        jax.random.PRNGKey(0),
        mean,
        log_std,
    )
    assert np.all(np.isfinite(np.asarray(log_probability)))
    assert np.all(np.asarray(action) >= -1.0)
    assert np.all(np.asarray(action) <= 1.0)


def test_cqn_action_codec_has_expected_final_bin_error():
    low = jnp.asarray([-2.0, 0.0], dtype=jnp.float32)
    high = jnp.asarray([2.0, 4.0], dtype=jnp.float32)
    actions = jnp.asarray(
        [[-2.0, 0.0], [-0.31, 1.27], [1.999, 3.999]],
        dtype=jnp.float32,
    )
    levels, bins = 3, 5
    encoded = encode_action(actions, low, high, levels, bins)
    decoded = decode_action(encoded, low, high, levels, bins)
    max_error = np.asarray((high - low) / (2.0 * bins**levels)) + 1e-6
    assert encoded.shape == (3, levels, 2)
    assert np.all(np.abs(np.asarray(decoded - actions)) <= max_error)
    assert np.all(np.asarray(decoded) >= np.asarray(low))
    assert np.all(np.asarray(decoded) <= np.asarray(high))


def test_cqn_categorical_projection_conserves_probability_mass():
    support = jnp.linspace(-2.0, 2.0, 9)
    probabilities = jnp.full((3, 2, 4, 9), 1.0 / 9.0)
    projected = project_categorical(
        probabilities,
        rewards=jnp.asarray([-1.0, 0.25, 1.0]),
        discounts=jnp.asarray([0.99, 0.99, 0.99]),
        bootstrap=jnp.asarray([1.0, 0.0, 1.0]),
        support=support,
    )
    np.testing.assert_allclose(
        np.asarray(projected.sum(axis=-1)),
        np.ones((3, 2, 4)),
        atol=1e-6,
    )
    assert np.all(np.asarray(projected) >= 0.0)


def test_cqn_critic_uses_paper_mlp_normalization_and_bias_settings():
    observation_space, action_space = _state_spaces()
    agent = create_agent(
        _compose_method("cqn"),
        observation_space=observation_space,
        action_space=action_space,
    )
    advantage = agent.params["critic"]["params"]["advantage"]
    assert "bias" not in advantage["dense_0"]
    assert "bias" not in advantage["dense_1"]
    assert set(advantage["norm_0"]) == {"bias", "scale"}
    assert set(advantage["norm_1"]) == {"bias", "scale"}


def test_ppo_gae_bootstraps_truncation_but_not_termination():
    rewards = np.asarray([[1.0, 1.0]], dtype=np.float32)
    values = np.zeros_like(rewards)
    next_values = np.asarray([[10.0, 10.0]], dtype=np.float32)
    terminated = np.asarray([[True, False]])
    truncated = np.asarray([[False, True]])
    advantages, returns = generalized_advantage_estimate(
        rewards,
        values,
        next_values,
        terminated,
        truncated,
        gamma=0.99,
        gae_lambda=0.95,
    )
    np.testing.assert_allclose(advantages[0, 0], 1.0)
    np.testing.assert_allclose(advantages[0, 1], 10.9, rtol=1e-6)
    np.testing.assert_allclose(returns, advantages)


@pytest.mark.parametrize("method", ["sac", "cqn"])
def test_jax_off_policy_agents_act_and_update(method):
    observation_space, action_space = _state_spaces()
    agent = create_agent(
        _compose_method(method),
        observation_space=observation_space,
        action_space=action_space,
    )
    observations = {"low_dim_state": np.zeros((1, 1, 5), dtype=np.float32)}
    action = agent.act(observations, step=3000, eval_mode=False)
    assert action.shape == (1, 1, 2)
    assert np.all(np.isfinite(action))
    before = jax.tree.map(np.asarray, agent.params)
    agent.update(iter([_off_policy_batch()]), step=1)
    assert _tree_changed(before, agent.params)


def test_jax_ppo_collects_on_policy_rollout_and_updates():
    observation_space, action_space = _state_spaces()
    agent = create_agent(
        _compose_method(
            "ppo",
            [
                "method.rollout_steps=4",
                "method.batch_size=4",
                "method.num_epochs=2",
            ],
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    before = jax.tree.map(np.asarray, agent.params)
    for step in range(4):
        observations = {
            "low_dim_state": np.full((1, 1, 5), step, dtype=np.float32)
        }
        action = agent.act(observations, step=step, eval_mode=False)
        assert action.shape == (1, 1, 2)
        agent.observe_transition(
            rewards=np.asarray([1.0], dtype=np.float32),
            terminations=np.asarray([False]),
            truncations=np.asarray([False]),
            next_observations={
                "low_dim_state": np.full(
                    (1, 1, 5), step + 1, dtype=np.float32
                )
            },
            next_info={},
        )
    assert agent.rollout_ready
    agent.update(None, step=4)
    assert not agent.rollout_ready
    assert _tree_changed(before, agent.params)


def test_jax_ppo_checkpoint_restores_partial_rollout():
    observation_space, action_space = _state_spaces()
    cfg = _compose_method(
        "ppo",
        [
            "method.rollout_steps=4",
            "method.batch_size=4",
            "method.num_epochs=1",
        ],
    )
    agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    observation = {"low_dim_state": np.zeros((1, 1, 5), dtype=np.float32)}
    agent.act(observation, step=0, eval_mode=False)
    agent.observe_transition(
        rewards=np.asarray([1.0], dtype=np.float32),
        terminations=np.asarray([False]),
        truncations=np.asarray([False]),
        next_observations=observation,
        next_info={},
    )
    model_state = agent.state_dict()
    checkpoint_state = agent.checkpoint_state_dict()

    restored = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    restored.load_state_dict(model_state)
    restored.load_checkpoint_state_dict(checkpoint_state)
    assert len(restored._rollout) == 1
    assert restored._pending_transition is None
    np.testing.assert_allclose(
        np.asarray(restored._rollout[0]["rewards"]),
        np.asarray([1.0], dtype=np.float32),
    )


@pytest.mark.parametrize("method", ["ppo", "sac", "cqn"])
def test_jax_rl_agents_accept_latest_onehot_time_feature(method):
    observation_space, action_space = _state_spaces()
    observation_space = spaces.Dict(
        {
            "low_dim_state": spaces.Box(
                -np.inf,
                np.inf,
                shape=(2, 5),
                dtype=np.float32,
            ),
            "time": spaces.Box(0, 1, shape=(2, 7), dtype=np.uint8),
        }
    )
    overrides = []
    if method == "ppo":
        overrides = ["method.rollout_steps=4", "method.batch_size=4"]
    agent = create_agent(
        _compose_method(method, overrides),
        observation_space=observation_space,
        action_space=action_space,
    )
    observations = {
        "low_dim_state": np.zeros((1, 2, 5), dtype=np.float32),
        "time": np.eye(7, dtype=np.uint8)[[1, 2]][None],
    }
    action = agent.act(observations, step=3000, eval_mode=True)
    assert action.shape == (1, 1, 2)
    assert np.all(np.isfinite(action))
