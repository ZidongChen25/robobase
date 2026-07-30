from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnasium import spaces
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from robobase.factory import create_agent
from robobase.method.cqn_direct_q import (
    CQNDirectQAS,
    action_centered_moment_loss,
)


CONFIG_DIR = str((Path(__file__).parents[2] / "robobase" / "cfgs").resolve())


def _config(*overrides):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(
            config_name="robobase_config",
            overrides=[
                "method=cqn_as",
                "action_sequence=3",
                "num_train_envs=1",
                "num_eval_envs=1",
                "backend.jit=false",
                "backend.platform=cpu",
                "method.model.hidden_dims=[32,32]",
                "method.atoms=11",
                "method.separate_bc_policy=true",
                "method.distinct_policy_encoder=true",
                "method.td_target_action_source=replay_next",
                "method.critic_sequence_mode=effective_k0",
                "method.demo_fosd=false",
                "method.mc_return_weight=0.1",
                "method._target_=robobase.method.cqn_direct_q.CQNDirectQAS",
                "+method.direct_scalar_q=true",
                "+method.direct_q_loss=mse",
                "+method.direct_q_huber_delta=1.0",
                *overrides,
            ],
        )


def _legacy_config(*overrides):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(
            config_name="robobase_config",
            overrides=[
                "method=cqn_as",
                "action_sequence=3",
                "num_train_envs=1",
                "num_eval_envs=1",
                "backend.jit=false",
                "backend.platform=cpu",
                "method.model.hidden_dims=[32,32]",
                "method.atoms=11",
                *overrides,
            ],
        )


def _spaces():
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
        shape=(3, 2),
        dtype=np.float32,
    )
    return observation_space, action_space


def _batch(batch_size=4):
    rng = np.random.default_rng(11)
    return {
        "low_dim_state": rng.normal(
            size=(batch_size, 1, 5)
        ).astype(np.float32),
        "low_dim_state_tp1": rng.normal(
            size=(batch_size, 1, 5)
        ).astype(np.float32),
        "action": rng.uniform(
            -1.0,
            1.0,
            size=(batch_size, 3, 2),
        ).astype(np.float32),
        "reward": rng.normal(size=(batch_size,)).astype(np.float32),
        "discount": np.full((batch_size,), 0.99, dtype=np.float32),
        "terminal": np.zeros((batch_size,), dtype=bool),
        "truncated": np.zeros((batch_size,), dtype=bool),
        "demo": np.ones((batch_size,), dtype=np.uint8),
        "mc_return": rng.normal(size=(batch_size,)).astype(np.float32),
    }


def _causal_batch():
    batch = _batch(batch_size=4)
    batch["demo"] = np.zeros((4,), dtype=np.uint8)
    batch["mc_return"] = np.asarray([1.0, 0.0, 1.0, 0.0], np.float32)
    batch["structured_explore"] = np.asarray([1, 0, 1, 0], np.uint8)
    batch["structured_explore_start"] = np.asarray(
        [1, 0, 1, 0], np.uint8
    )
    batch["structured_explore_dimension"] = np.asarray(
        [0, -1, 1, -1], np.int16
    )
    batch["structured_explore_delta"] = np.asarray(
        [0.08, 0.0, -0.08, 0.0], np.float32
    )
    batch["structured_explore_assignment_prob"] = np.asarray(
        [0.125, 0.5, 0.125, 0.5], np.float32
    )
    return batch


def _tree_changed(before, after):
    left, left_tree = jax.tree.flatten(before)
    right, right_tree = jax.tree.flatten(after)
    assert left_tree == right_tree
    return any(
        not np.allclose(np.asarray(x), np.asarray(y))
        for x, y in zip(left, right, strict=True)
    )


def _tree_exactly_equal(before, after):
    left, left_tree = jax.tree.flatten(before)
    right, right_tree = jax.tree.flatten(after)
    assert left_tree == right_tree
    return all(
        np.array_equal(np.asarray(x), np.asarray(y))
        for x, y in zip(left, right, strict=True)
    )


def test_direct_q_agent_updates_scalar_critic_and_independent_policy():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _config(),
        observation_space=observation_space,
        action_space=action_space,
    )

    assert isinstance(agent, CQNDirectQAS)
    assert agent.direct_scalar_q
    assert agent.direct_q_loss == "mse"
    assert agent.critic_sequence_mode == "effective_k0"
    features = jnp.zeros((2, 5), dtype=jnp.float32)
    actions = jnp.zeros((2, 3, 2), dtype=jnp.float32)
    chosen_q, all_q = agent._direct_q_per_level(
        agent.params["critic"],
        features,
        actions,
    )
    assert chosen_q.shape == (2, agent.levels, 6)
    assert all_q.shape == (2, agent.levels, 6, agent.bins)

    critic_before = jax.tree.map(np.asarray, agent.params["critic"])
    policy_before = jax.tree.map(np.asarray, agent.params["policy"])
    agent.logging = True
    metrics = agent.update(iter([_batch()]), step=1)

    assert _tree_changed(critic_before, agent.params["critic"])
    assert _tree_changed(policy_before, agent.params["policy"])
    assert np.isfinite(metrics["direct_q_loss"])
    assert metrics["direct_q_grad_nonfinite_fraction"] == pytest.approx(0.0)
    assert metrics["policy_bc_loss"] > 0.0


def test_direct_q_imports_and_freezes_exact_legacy_c51_policy(tmp_path):
    observation_space, action_space = _spaces()
    legacy = create_agent(
        _legacy_config(),
        observation_space=observation_space,
        action_space=action_space,
    )
    snapshot = tmp_path / "clean_cqn_as.pkl"
    with snapshot.open("wb") as stream:
        pickle.dump({"agent": legacy.state_dict()}, stream)

    agent = create_agent(
        _config(
            "method.freeze_bc_policy=true",
            "method.bc_policy_mode=legacy_c51",
            f"method.frozen_policy_snapshot={snapshot}",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )

    assert agent.freeze_bc_policy
    assert agent.bc_policy_mode == "legacy_c51"
    assert _tree_exactly_equal(
        legacy.target_critic_params,
        agent.params["policy"],
    )
    features = jnp.zeros((2, 5), dtype=jnp.float32)
    legacy_action, legacy_bins = legacy._greedy_action(
        legacy.target_critic_params,
        features,
        key=None,
    )
    frozen_action, frozen_bins = agent._policy_action(
        agent.params["policy"],
        features,
        key=None,
    )
    np.testing.assert_array_equal(frozen_action, legacy_action)
    np.testing.assert_array_equal(frozen_bins, legacy_bins)

    policy_before = jax.tree.map(np.asarray, agent.params["policy"])
    critic_before = jax.tree.map(np.asarray, agent.params["critic"])
    agent.logging = True
    metrics = agent.update(iter([_batch()]), step=1)

    assert _tree_exactly_equal(policy_before, agent.params["policy"])
    assert _tree_changed(critic_before, agent.params["critic"])
    assert metrics["policy_grad_norm"] == pytest.approx(0.0, abs=1e-8)
    assert metrics["policy_encoder_grad_norm"] == pytest.approx(
        0.0,
        abs=1e-8,
    )


def test_direct_q_policy_value_action_combines_q_and_bc_prior(monkeypatch):
    observation_space, action_space = _spaces()
    agent = create_agent(
        _config("method.policy_value_beta=0"),
        observation_space=observation_space,
        action_space=action_space,
    )

    class FakeCritic:
        def apply(self, _params, features, _level, _midpoint):
            q = jnp.zeros(
                (
                    features.shape[0],
                    agent.action_sequence,
                    agent.action_dim,
                    agent.bins,
                    1,
                ),
                dtype=jnp.float32,
            )
            return q.at[..., 4, 0].set(2.0)

    class FakePolicy:
        def apply(self, _params, features, _level, _midpoint):
            logits = jnp.zeros(
                (
                    features.shape[0],
                    agent.action_sequence,
                    agent.action_dim,
                    agent.bins,
                    1,
                ),
                dtype=jnp.float32,
            )
            return logits.at[..., 0, 0].set(4.0)

    monkeypatch.setattr(agent, "critic_model", FakeCritic())
    monkeypatch.setattr(agent, "policy_model", FakePolicy())
    features = jnp.zeros((1, 5), dtype=jnp.float32)

    _, q_bins = agent._direct_q_action(
        agent.params["critic"],
        features,
        key=None,
        policy_params=agent.params["policy"],
        policy_features=features,
    )
    agent.policy_value_beta = 4.0
    _, blended_bins = agent._direct_q_action(
        agent.params["critic"],
        features,
        key=None,
        policy_params=agent.params["policy"],
        policy_features=features,
    )

    np.testing.assert_array_equal(q_bins, 4)
    np.testing.assert_array_equal(blended_bins, 0)


def test_direct_q_jitted_update_smoke():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _config("backend.jit=true"),
        observation_space=observation_space,
        action_space=action_space,
    )
    agent.logging = True

    metrics = agent.update(iter([_batch()]), step=1)

    assert np.isfinite(metrics["critic_loss"])
    assert metrics["direct_q_grad_nonfinite_fraction"] == pytest.approx(0.0)


def test_action_centered_moment_removes_large_state_only_baseline():
    propensity = 0.2
    treated = np.asarray([1] * 200 + [0] * 800, np.float32)
    # Both arms have a large common baseline; the true treatment effect is 1.
    outcome = np.where(treated > 0, 101.0, 100.0).astype(np.float32)
    valid = np.ones_like(treated, dtype=bool)
    weights = np.ones_like(treated)

    def loss(tau):
        return action_centered_moment_loss(
            jnp.full(treated.shape, tau),
            outcome,
            treated,
            propensity,
            valid,
            weights,
        )

    assert float(loss(1.0)) < float(loss(0.0))
    assert float(jax.grad(loss)(1.0)) == pytest.approx(0.0, abs=2e-5)


def test_direct_q_causal_rct_update_uses_known_randomization():
    observation_space, action_space = _spaces()
    control = create_agent(
        _config(
            "method.structured_exploration_prob=0.5",
            "method.structured_exploration_horizon=1",
            "method.causal_rct_weight=0",
            "method.causal_rct_level=1",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    agent = create_agent(
        _config(
            "method.structured_exploration_prob=0.5",
            "method.structured_exploration_horizon=1",
            "method.causal_rct_weight=0.1",
            "method.causal_rct_level=1",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    control.update(iter([_causal_batch()]), step=1)
    agent.logging = True

    metrics = agent.update(iter([_causal_batch()]), step=1)

    assert np.isfinite(metrics["causal_rct_loss"])
    assert metrics["causal_rct_valid_fraction"] == pytest.approx(1.0)
    assert metrics["causal_rct_treated_fraction"] == pytest.approx(0.5)
    assert metrics["causal_rct_assignment_error_max"] == pytest.approx(0.0)
    assert metrics["causal_rct_tau_abs_mean"] >= 0.0
    assert _tree_changed(
        control.params["critic"],
        agent.params["critic"],
    )


def test_direct_q_causal_rct_keeps_clipped_zero_delta_assignment():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _config(
            "method.structured_exploration_prob=0.5",
            "method.structured_exploration_horizon=1",
            "method.causal_rct_weight=0.1",
            "method.causal_rct_level=1",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    batch = _causal_batch()
    # A direction sampled out of an action bound is still a genuine randomized
    # assignment, even though clipping makes its realized action effect zero.
    batch["structured_explore_delta"][0] = 0.0
    agent.logging = True

    metrics = agent.update(iter([batch]), step=1)

    assert np.isfinite(metrics["causal_rct_loss"])
    assert metrics["causal_rct_valid_fraction"] == pytest.approx(1.0)
    assert metrics["causal_rct_treated_fraction"] == pytest.approx(0.5)
    assert metrics["causal_rct_assignment_error_max"] == pytest.approx(0.0)


def test_direct_q_causal_rct_rejects_multistep_intervention():
    observation_space, action_space = _spaces()
    with pytest.raises(ValueError, match="one-step randomized"):
        create_agent(
            _config(
                "method.structured_exploration_prob=0.2",
                "method.structured_exploration_horizon=4",
                "method.causal_rct_weight=0.1",
            ),
            observation_space=observation_space,
            action_space=action_space,
        )


def test_direct_q_pixel_towers_materialize_scalar_bins():
    observation_space = spaces.Dict(
        {
            "low_dim_state": spaces.Box(
                -np.inf,
                np.inf,
                shape=(1, 5),
                dtype=np.float32,
            ),
            "rgb_front": spaces.Box(
                0,
                255,
                shape=(1, 3, 84, 84),
                dtype=np.uint8,
            ),
        }
    )
    _, action_space = _spaces()
    agent = create_agent(
        _config("pixels=true"),
        observation_space=observation_space,
        action_space=action_space,
    )
    observations = {
        "low_dim_state": np.zeros((1, 1, 5), dtype=np.float32),
        "rgb_front": np.zeros((1, 1, 3, 84, 84), dtype=np.uint8),
    }
    obs_inputs = agent._prepare_rl_obs_inputs(observations)
    features = agent._rl_features(
        agent.params["encoder"],
        obs_inputs,
        stop_gradient=True,
    )
    chosen_q, all_q = agent._direct_q_per_level(
        agent.params["critic"],
        features,
        jnp.zeros((1, 3, 2), dtype=jnp.float32),
    )

    assert chosen_q.shape == (1, agent.levels, 6)
    assert all_q.shape == (1, agent.levels, 6, agent.bins)
    assert np.all(np.isfinite(np.asarray(all_q)))


def test_direct_q_high_utd_launch_matches_floq_data_protocol():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_direct_q_two_tower_coherent_mc_high_utd4_gate",
                "env=bigym/move_plate",
            ],
        )

    assert cfg.method.name == "cqn_as"
    assert cfg.method.direct_scalar_q
    assert cfg.method.direct_q_loss == "mse"
    assert cfg.method.num_update_steps == 4
    assert cfg.method.separate_bc_policy
    assert cfg.method.distinct_policy_encoder
    assert cfg.method.td_target_action_source == "replay_next"
    assert cfg.method.critic_sequence_mode == "effective_k0"
    assert cfg.method.mc_return_weight == pytest.approx(0.1)
    assert cfg.method.structured_exploration_prob == pytest.approx(0.06)
    assert cfg.method.structured_exploration_horizon == 4
    assert not cfg.method.demo_fosd


def test_direct_q_h1_rct_launch_changes_only_identification_protocol():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        control = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_direct_q_h1_rct_control_two_tower_high_utd4_gate",
                "env=bigym/move_plate",
            ],
        )
        treatment = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_direct_q_h1_rct_two_tower_high_utd4_gate",
                "env=bigym/move_plate",
            ],
        )

    assert control.method.structured_exploration_prob == pytest.approx(0.2)
    assert control.method.structured_exploration_horizon == 1
    assert control.method.causal_rct_weight == pytest.approx(0.0)
    assert treatment.method.causal_rct_weight == pytest.approx(0.1)
    assert treatment.method.causal_rct_level == 1
    control_method = OmegaConf.to_container(control.method, resolve=True)
    treatment_method = OmegaConf.to_container(treatment.method, resolve=True)
    assert {
        key: (control_method.get(key), treatment_method.get(key))
        for key in sorted(set(control_method) | set(treatment_method))
        if control_method.get(key) != treatment_method.get(key)
    } == {"causal_rct_weight": (0.0, 0.1)}


def test_direct_q_frozen_clean_rct_launch_requires_exact_legacy_source():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_direct_q_h1_rct_frozen_clean_high_utd4_gate",
                "env=bigym/move_plate",
            ],
        )

    assert cfg.method.direct_scalar_q
    assert cfg.method.causal_rct_weight == pytest.approx(0.1)
    assert cfg.method.structured_exploration_horizon == 1
    assert cfg.method.freeze_bc_policy
    assert cfg.method.bc_policy_mode == "legacy_c51"
    assert cfg.method.frozen_policy_snapshot is None
    assert cfg.method.policy_value_beta is None


def test_direct_q_frozen_clean_control_changes_only_rct_weight():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        treatment = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_direct_q_h1_rct_frozen_clean_high_utd4_gate",
                "env=bigym/move_plate",
            ],
        )
        control = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_direct_q_h1_rct_frozen_clean_control_high_utd4_gate",
                "env=bigym/move_plate",
            ],
        )

    treatment_method = OmegaConf.to_container(
        treatment.method,
        resolve=True,
    )
    control_method = OmegaConf.to_container(control.method, resolve=True)
    assert {
        key: (control_method.get(key), treatment_method.get(key))
        for key in sorted(set(control_method) | set(treatment_method))
        if control_method.get(key) != treatment_method.get(key)
    } == {"causal_rct_weight": (0.0, 0.1)}
    assert control.method.freeze_bc_policy
    assert control.method.bc_policy_mode == "legacy_c51"
    assert control.method.policy_value_beta is None


def test_direct_q_policy_value_td_arm_keeps_rollout_exact_bc():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        parent = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_direct_q_two_tower_coherent_mc_high_utd4_gate",
                "env=bigym/move_plate",
            ],
        )
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_direct_q_td_policy_value_two_tower_coherent_mc_high_utd4_gate",
                "env=bigym/move_plate",
            ],
        )

    assert cfg.method.td_target_action_source == "policy_value"
    assert cfg.method.td_target_policy_value_beta == pytest.approx(1.0)
    assert cfg.method.policy_value_beta is None
    assert cfg.method.num_update_steps == 4
    parent_method = OmegaConf.to_container(parent.method, resolve=True)
    target_method = OmegaConf.to_container(cfg.method, resolve=True)
    assert {
        key: (parent_method.get(key), target_method.get(key))
        for key in sorted(set(parent_method) | set(target_method))
        if parent_method.get(key) != target_method.get(key)
    } == {
        "td_target_action_source": ("replay_next", "policy_value"),
        "td_target_policy_value_beta": (None, 1.0),
    }


def test_direct_q_policy_value_td_target_updates_with_bc_rollout():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _config(
            "method.td_target_action_source=policy_value",
            "method.td_target_policy_value_beta=1",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    assert agent.policy_value_beta is None
    assert agent.td_target_policy_value_beta == pytest.approx(1.0)
    agent.logging = True

    metrics = agent.update(iter([_batch()]), step=1)

    assert np.isfinite(metrics["direct_q_loss"])
    assert metrics["direct_q_grad_nonfinite_fraction"] == pytest.approx(0.0)


def test_direct_q_bc_policy_td_arm_changes_only_target_action_source():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        parent = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_direct_q_two_tower_coherent_mc_high_utd4_gate",
                "env=bigym/move_plate",
            ],
        )
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_direct_q_td_bc_policy_two_tower_coherent_mc_high_utd4_gate",
                "env=bigym/move_plate",
            ],
        )

    parent_method = OmegaConf.to_container(parent.method, resolve=True)
    target_method = OmegaConf.to_container(cfg.method, resolve=True)
    assert {
        key: (parent_method.get(key), target_method.get(key))
        for key in sorted(set(parent_method) | set(target_method))
        if parent_method.get(key) != target_method.get(key)
    } == {
        "td_target_action_source": ("replay_next", "bc_policy"),
    }
    assert cfg.method.policy_value_beta is None
    assert cfg.method.td_target_policy_value_beta is None


def test_direct_q_bc_policy_td_target_updates_with_exact_bc_rollout():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _config("method.td_target_action_source=bc_policy"),
        observation_space=observation_space,
        action_space=action_space,
    )
    assert agent.policy_value_beta is None
    assert agent.td_target_policy_value_beta is None
    agent.logging = True

    metrics = agent.update(iter([_batch()]), step=1)

    assert np.isfinite(metrics["direct_q_loss"])
    assert metrics["direct_q_grad_nonfinite_fraction"] == pytest.approx(0.0)
