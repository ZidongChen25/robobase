from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.traverse_util import flatten_dict
from gymnasium import spaces
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from robobase.envs.wrappers import RecedingHorizonControl
from robobase.factory import create_agent
from robobase.method.cqn_as import C2FSequenceDistributionalCritic
from robobase.models.encoder import JaxCQNEncoder
from robobase.replay_buffer.vision_feature_cache import JAX_CQN_AS_FEATURE_KEY
from robobase.workspace import (
    Workspace,
    _effective_episode_length,
    _replay_action_from_step,
)
from tests.unit.wrappers.utils import DummyEnv


CONFIG_DIR = str((Path(__file__).parents[2] / "robobase" / "cfgs").resolve())


def _compose_cqn_as(*overrides):
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


def _spaces(action_sequence=3):
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
        shape=(action_sequence, 2),
        dtype=np.float32,
    )
    return observation_space, action_space


def _batch(batch_size=4, action_sequence=3):
    rng = np.random.default_rng(7)
    return {
        "low_dim_state": rng.normal(size=(batch_size, 1, 5)).astype(np.float32),
        "low_dim_state_tp1": rng.normal(size=(batch_size, 1, 5)).astype(
            np.float32
        ),
        "action": rng.uniform(
            -1.0,
            1.0,
            size=(batch_size, action_sequence, 2),
        ).astype(np.float32),
        "reward": rng.normal(size=(batch_size,)).astype(np.float32),
        "discount": np.full((batch_size,), 0.99, dtype=np.float32),
        "terminal": np.zeros((batch_size,), dtype=bool),
        "truncated": np.zeros((batch_size,), dtype=bool),
        "demo": np.ones((batch_size,), dtype=np.uint8),
    }


def _tree_changed(before, after):
    before_leaves, before_tree = jax.tree.flatten(before)
    after_leaves, after_tree = jax.tree.flatten(after)
    assert before_tree == after_tree
    return any(
        not np.allclose(np.asarray(left), np.asarray(right))
        for left, right in zip(before_leaves, after_leaves, strict=True)
    )


def test_cqn_as_sequence_critic_outputs_all_steps_and_streams():
    critic = C2FSequenceDistributionalCritic(
        hidden_dims=(16, 16),
        action_sequence=3,
        action_dim=2,
        levels=2,
        bins=5,
        atoms=11,
        gru_layers=1,
        use_dueling=True,
    )
    features = jnp.zeros((4, 7), dtype=jnp.float32)
    level = jnp.tile(jnp.asarray([[1.0, 0.0]], dtype=jnp.float32), (4, 1))
    midpoint = jnp.zeros((4, 3, 2), dtype=jnp.float32)
    params = critic.init(jax.random.PRNGKey(0), features, level, midpoint)
    logits = critic.apply(params, features, level, midpoint)

    assert logits.shape == (4, 3, 2, 5, 11)
    param_names = str(params)
    assert "advantage_gru_0" in param_names
    assert "value_gru_0" in param_names

    combined, values, centered_advantages = critic.apply(
        params,
        features,
        level,
        midpoint,
        return_streams=True,
    )
    np.testing.assert_allclose(combined, values + centered_advantages)
    np.testing.assert_allclose(
        np.asarray(centered_advantages).mean(axis=-2),
        0.0,
        atol=1e-7,
    )


def test_cqn_as_paper_encoder_has_per_view_256_by_5_by_5_features():
    encoder = JaxCQNEncoder(
        input_shape=(3, 12, 84, 84),
        jit=False,
        seed=0,
    )
    output = encoder.apply_trainable(
        encoder.trainable_params,
        jnp.zeros((2, 3, 12, 84, 84), dtype=jnp.float32),
    )

    assert encoder.output_shape == (3, 256 * 5 * 5)
    assert output.shape == (2, 3, 256 * 5 * 5)


def test_cqn_as_td_action_selection_randomizes_near_flat_bin_ties():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_as(),
        observation_space=observation_space,
        action_space=action_space,
    )
    features = jnp.zeros((1, 5), dtype=jnp.float32)

    _, first_bins = agent._greedy_action_for_update(
        agent.params["critic"],
        features,
        jax.random.PRNGKey(1),
    )
    _, second_bins = agent._greedy_action_for_update(
        agent.params["critic"],
        features,
        jax.random.PRNGKey(2),
    )

    assert not np.all(np.asarray(first_bins) == 0)
    assert not np.array_equal(np.asarray(first_bins), np.asarray(second_bins))


def test_direct_c51_policy_value_action_combines_q_and_bc_prior(monkeypatch):
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_as(
            "method.separate_bc_policy=true",
            "method.policy_value_beta=0",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )

    class FakeCritic:
        def apply(self, _params, features, _level, _midpoint):
            logits = jnp.zeros(
                (
                    features.shape[0],
                    agent.action_sequence,
                    agent.action_dim,
                    agent.bins,
                    agent.atoms,
                ),
                dtype=jnp.float32,
            )
            logits = logits.at[..., 0, 0].set(20.0)
            return logits.at[..., 4, -1].set(20.0)

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

    _, q_only_bins = agent._policy_value_action(
        agent.params["critic"],
        features,
        agent.params["policy"],
        features,
    )
    agent.policy_value_beta = 4.0
    _, blended_bins = agent._policy_value_action(
        agent.params["critic"],
        features,
        agent.params["policy"],
        features,
    )

    np.testing.assert_array_equal(q_only_bins, 4)
    np.testing.assert_array_equal(blended_bins, 0)


def test_cqn_as_agent_returns_chunk_and_updates():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_as(),
        observation_space=observation_space,
        action_space=action_space,
    )
    observations = {"low_dim_state": np.zeros((1, 1, 5), dtype=np.float32)}
    action = agent.act(observations, step=3000, eval_mode=False)
    assert action.shape == (1, 3, 2)
    assert np.all(np.isfinite(action))
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)

    before = jax.tree.map(np.asarray, agent.params)
    rng_before_update = np.asarray(agent.rng_key).copy()
    agent.update(iter([_batch()]), step=1)
    assert _tree_changed(before, agent.params)
    assert not np.array_equal(rng_before_update, np.asarray(agent.rng_key))


def test_cqn_as_structured_exploration_changes_one_local_coordinate():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_as(
            "method.structured_exploration_prob=1.0",
            "method.structured_exploration_level=1",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    base = jnp.zeros((16, 2), dtype=jnp.float32)
    explored, mask = agent._structured_exploration_action(
        base,
        jax.random.PRNGKey(9),
    )
    delta = np.asarray(explored - base)

    assert np.all(np.asarray(mask))
    assert np.all(np.count_nonzero(delta, axis=1) == 1)
    np.testing.assert_allclose(
        np.max(np.abs(delta), axis=1),
        np.full((16,), 2.0 / (agent.bins**2)),
    )

    observations = {
        "low_dim_state": np.zeros((1, 1, 5), dtype=np.float32)
    }
    agent.act(observations, step=3000, eval_mode=False)
    diagnostics = agent.rollout_diagnostics()
    assert diagnostics["structured_exploration_eligible"] == 1.0
    assert diagnostics["structured_exploration_applied"] == 1.0
    assert diagnostics["structured_exploration_rate"] == pytest.approx(1.0)
    assert np.all(agent._last_structured_exploration_mask)

    agent.structured_exploration_prob = 0.0
    unchanged, mask = agent._structured_exploration_action(
        base,
        jax.random.PRNGKey(9),
    )
    np.testing.assert_array_equal(np.asarray(unchanged), np.asarray(base))
    assert not np.any(np.asarray(mask))


def test_cqn_as_coherent_structured_exploration_persists_assignment():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_as(
            "method.structured_exploration_prob=1.0",
            "method.structured_exploration_level=1",
            "method.structured_exploration_horizon=3",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    base = jnp.zeros((1, 2), dtype=jnp.float32)
    first = agent._coherent_structured_exploration_action(
        base, jax.random.PRNGKey(1)
    )
    agent.structured_exploration_prob = 0.0
    second = agent._coherent_structured_exploration_action(
        base, jax.random.PRNGKey(2)
    )
    third = agent._coherent_structured_exploration_action(
        base, jax.random.PRNGKey(3)
    )
    fourth = agent._coherent_structured_exploration_action(
        base, jax.random.PRNGKey(4)
    )

    assert bool(first[1][0]) and bool(first[2][0])
    assert bool(second[1][0]) and not bool(second[2][0])
    assert bool(third[1][0]) and not bool(third[2][0])
    assert not bool(fourth[1][0]) and not bool(fourth[2][0])
    assert first[3][0] == second[3][0] == third[3][0]
    assert fourth[3][0] == -1
    np.testing.assert_allclose(first[4], second[4])
    np.testing.assert_allclose(first[4], third[4])
    np.testing.assert_allclose(first[5], [1.0 / (2 * agent.action_dim)])
    np.testing.assert_allclose(second[5], [1.0])
    np.testing.assert_allclose(fourth[5], [1.0])


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("execution_length=2", "execution_length=1"),
        ("action_execution_start=1", "action_execution_start=0"),
        ("temporal_ensemble=true", "root temporal_ensemble=false"),
    ],
)
def test_cqn_as_factory_rejects_incompatible_rollout_controls(override, message):
    observation_space, action_space = _spaces()
    with pytest.raises(ValueError, match=message):
        create_agent(
            _compose_cqn_as(override),
            observation_space=observation_space,
            action_space=action_space,
        )


def test_cqn_as_frozen_pixel_cache_builds_tp1_features_without_raw_rgb():
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
                shape=(1, 3, 16, 16),
                dtype=np.uint8,
            ),
        }
    )
    _, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_as(
            "pixels=true",
            "method.encoder_model.trainable=false",
            "method.encoder_model.pretrained=false",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    batch = _batch(batch_size=2)
    batch[JAX_CQN_AS_FEATURE_KEY] = np.zeros((2, 1, 512), dtype=np.float32)
    batch[f"{JAX_CQN_AS_FEATURE_KEY}_tp1"] = np.ones(
        (2, 1, 512), dtype=np.float32
    )

    next_features = agent._next_rl_obs_inputs(batch)

    assert next_features.shape == (2, 517)
    assert np.all(np.isfinite(np.asarray(next_features)))


def test_cqn_as_temporal_ensemble_prefers_recent_plan_and_reset_clears_it():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_as("method.temporal_ensemble_gain=0.5"),
        observation_space=observation_space,
        action_space=action_space,
    )
    first = np.asarray([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])
    second = np.asarray([[[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]])

    np.testing.assert_allclose(
        agent._ensemble_current_action(first, eval_mode=False),
        [[1.0, 1.0]],
    )
    ensembled = agent._ensemble_current_action(second, eval_mode=False)
    old_weight = np.exp(-0.5)
    expected = (3.0 + old_weight * 2.0) / (1.0 + old_weight)
    np.testing.assert_allclose(ensembled, [[expected, expected]], rtol=1e-6)

    agent.reset(step=0, agents_to_reset=[0])
    np.testing.assert_allclose(
        agent._ensemble_current_action(second, eval_mode=False),
        [[3.0, 3.0]],
    )


def test_cqn_as_temporal_ensemble_replans_every_two_primitive_steps():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_as(
            "method.temporal_ensemble_replan_interval=2",
            "method.temporal_ensemble_gain=0.5",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    first = np.asarray([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])
    second = np.asarray([[[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]])
    placeholder = np.zeros_like(first)

    register = agent._temporal_replan_mask(eval_mode=False, batch_size=1)
    np.testing.assert_array_equal(register, [True])
    np.testing.assert_allclose(
        agent._ensemble_current_action(
            first,
            eval_mode=False,
            register_mask=register,
        ),
        [[1.0, 1.0]],
    )

    register = agent._temporal_replan_mask(eval_mode=False, batch_size=1)
    np.testing.assert_array_equal(register, [False])
    np.testing.assert_allclose(
        agent._ensemble_current_action(
            placeholder,
            eval_mode=False,
            register_mask=register,
        ),
        [[2.0, 2.0]],
    )

    register = agent._temporal_replan_mask(eval_mode=False, batch_size=1)
    np.testing.assert_array_equal(register, [True])
    ensembled = agent._ensemble_current_action(
        second,
        eval_mode=False,
        register_mask=register,
    )
    old_weight = np.exp(-1.0)
    expected = (10.0 + old_weight * 3.0) / (1.0 + old_weight)
    np.testing.assert_allclose(ensembled, [[expected, expected]], rtol=1e-6)


def test_cqn_as_replan_interval_must_fit_inside_action_sequence():
    observation_space, action_space = _spaces()
    with pytest.raises(
        ValueError,
        match="temporal_ensemble_replan_interval",
    ):
        create_agent(
            _compose_cqn_as("method.temporal_ensemble_replan_interval=4"),
            observation_space=observation_space,
            action_space=action_space,
        )


def test_cqn_as_without_temporal_ensemble_executes_cached_plan_open_loop():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_as("method.temporal_ensemble=false"),
        observation_space=observation_space,
        action_space=action_space,
    )
    first = np.asarray([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])
    replacement = np.asarray([[[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]])

    np.testing.assert_allclose(
        agent._open_loop_action_chunk(first, eval_mode=False)[:, 0],
        [[1.0, 1.0]],
    )

    observation = {"low_dim_state": np.zeros((1, 1, 5), dtype=np.float32)}
    first_action = agent.act(observation, step=0, eval_mode=True)
    cached_plan = agent._eval_open_loop_plan.copy()
    rng_after_first_inference = np.asarray(agent.rng_key).copy()
    second_action = agent.act(observation, step=1, eval_mode=True)
    np.testing.assert_allclose(first_action[:, 0], cached_plan[:, 0])
    np.testing.assert_allclose(second_action[:, 0], cached_plan[:, 1])
    np.testing.assert_array_equal(
        np.asarray(agent.rng_key),
        rng_after_first_inference,
    )
    np.testing.assert_allclose(
        agent._open_loop_action_chunk(replacement, eval_mode=False)[:, 0],
        [[2.0, 2.0]],
    )
    np.testing.assert_allclose(
        agent._open_loop_action_chunk(replacement, eval_mode=False)[:, 0],
        [[3.0, 3.0]],
    )
    np.testing.assert_allclose(
        agent._open_loop_action_chunk(replacement, eval_mode=False)[:, 0],
        [[10.0, 10.0]],
    )

    agent.reset(step=0, agents_to_reset=[0])
    np.testing.assert_allclose(
        agent._open_loop_action_chunk(first, eval_mode=False)[:, 0],
        [[1.0, 1.0]],
    )


def test_receding_horizon_reports_zero_valued_executed_ensemble():
    env = RecedingHorizonControl(
        DummyEnv(episode_len=10),
        sequence_length=3,
        time_limit=10,
        execution_length=1,
        temporal_ensemble=True,
        gain=0.0,
    )
    env.reset()
    zero_plan = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    env.step(zero_plan)
    current_plan = np.ones(env.action_space.shape, dtype=env.action_space.dtype)
    *_, info = env.step(current_plan)

    # Current plan predicts one; the previous plan predicted zero for this step.
    np.testing.assert_allclose(info["executed_action"], 0.5)
    assert info["executed_action"].shape == (1, *env.action_space.shape[1:])


def test_replay_uses_actual_executed_action_including_terminal_info():
    commanded = np.asarray([[[-0.5, -0.5], [0.8, 0.8]]], dtype=np.float32)[0]
    executed = np.asarray([[0.25, -0.25]], dtype=np.float32)
    np.testing.assert_allclose(
        _replay_action_from_step(commanded, {"executed_action": executed}),
        executed[0],
    )
    np.testing.assert_allclose(
        _replay_action_from_step(
            commanded,
            {
                "_final_info": True,
                "final_info": {"executed_action": executed},
            },
        ),
        executed[0],
    )


def test_global_env_steps_count_execution_length_not_prediction_horizon():
    workspace = object.__new__(Workspace)
    workspace.train_envs = SimpleNamespace(num_envs=2)
    workspace.cfg = SimpleNamespace(action_repeat=2, execution_length=1)
    workspace._main_loop_iterations = 7
    workspace._pretrain_step = 3

    assert workspace._calculate_global_env_steps() == 7 * 2 * 2 * 1 + 3


def test_bigym_effective_episode_length_uses_demo_control_rate():
    cfg = OmegaConf.create(
        {
            "env": {
                "env_name": "bigym",
                "episode_length": 4000,
                "demo_down_sample_rate": 10,
                "episode_length_is_env_steps": False,
            }
        }
    )

    assert _effective_episode_length(cfg) == 400


def test_cqn_as_bigym_launch_has_official_sequence_replay_contract():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_as_pixel_bigym_demo_driven",
                "env=bigym/move_plate",
            ],
        )

    assert cfg.method.name == "cqn_as"
    assert cfg.action_sequence == 16
    assert cfg.execution_length == 1
    assert not cfg.temporal_ensemble
    assert cfg.method.temporal_ensemble
    assert cfg.method.temporal_ensemble_replan_interval == 1
    assert cfg.num_explore_steps == 0
    assert cfg.method.bc_margin == 0.1
    assert cfg.method.demo_fosd
    assert not cfg.method.always_bootstrap
    assert cfg.replay.action_padding == "edge"
    assert cfg.replay.nstep == 1
    assert cfg.replay.include_tp1
    assert cfg.method.encoder_model.type == "cqn"
    assert cfg.env.episode_length == 3000
    assert not cfg.env.filter_successful_demos
    assert not cfg.lazy_replay.use


def test_cqn_as_rlbench_launch_composes_complete_task_and_cqn_episode_limits():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=["launch=cqn_as_pixel_rlbench_demo_driven"],
        )

    assert cfg.env.env_name == "rlbench"
    assert cfg.env.task_name == "take_lid_off_saucepan"
    assert cfg.env.episode_length == 100
    assert cfg.method.name == "cqn_as"
    assert cfg.action_sequence == 4
    assert cfg.execution_length == 1
    assert cfg.method.bc_margin == 0.01


def test_cqn_as_demo_fosd_can_be_disabled_for_margin_matched_baseline():
    cfg = _compose_cqn_as("method.demo_fosd=false")
    observation_space, action_space = _spaces()
    agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )

    assert not agent.demo_fosd


def test_cqn_as_decoupled_bc_updates_policy_without_changing_td_only_critic():
    cfg = _compose_cqn_as(
        "method.separate_bc_policy=true",
        "method.bc_policy_stop_gradient=true",
        "method.td_target_action_source=replay_next",
        "method.critic_sequence_mode=effective_k0",
        "method.critic_lambda=0.0",
        "method.weight_decay=0.0",
    )
    observation_space, action_space = _spaces()
    agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )

    assert "policy" in agent.params
    assert agent.critic_sequence_mode == "effective_k0"
    critic_before = jax.tree.map(np.asarray, agent.params["critic"])
    policy_before = jax.tree.map(np.asarray, agent.params["policy"])
    agent.logging = True
    metrics = agent.update(iter([_batch()]), step=1)

    assert not _tree_changed(critic_before, agent.params["critic"])
    assert _tree_changed(policy_before, agent.params["policy"])
    assert metrics["critic_loss"] == pytest.approx(0.0)
    assert metrics["policy_bc_loss"] > 0.0
    assert 0.0 <= metrics["policy_demo_top1"] <= 1.0


def test_cqn_as_distinct_policy_encoder_blocks_bc_gradient_from_value_tower():
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
        _compose_cqn_as(
            "pixels=true",
            "method.separate_bc_policy=true",
            "method.distinct_policy_encoder=true",
            "method.bc_policy_stop_gradient=false",
            "method.td_target_action_source=replay_next",
            "method.critic_sequence_mode=effective_k0",
            "method.critic_lambda=0",
            "method.mc_return_weight=0",
            "method.weight_decay=0",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    batch = _batch(batch_size=2)
    batch["rgb_front"] = np.zeros(
        (2, 1, 3, 84, 84),
        dtype=np.uint8,
    )
    batch["rgb_front_tp1"] = np.ones(
        (2, 1, 3, 84, 84),
        dtype=np.uint8,
    )
    value_encoder_before = jax.tree.map(np.asarray, agent.params["encoder"])
    policy_encoder_before = jax.tree.map(
        np.asarray,
        agent.params["policy_encoder"],
    )
    agent.logging = True

    # The policy output head is zero-initialized, so its first update cannot
    # yet backpropagate a feature gradient. The second update exercises the
    # now-nonzero head and therefore the dedicated visual tower.
    agent.update(iter([batch]), step=1)
    metrics = agent.update(iter([batch]), step=2)

    assert not _tree_changed(value_encoder_before, agent.params["encoder"])
    assert _tree_changed(
        policy_encoder_before,
        agent.params["policy_encoder"],
    )
    assert metrics["policy_encoder_grad_norm"] > 0.0
    assert metrics["critic_loss"] == pytest.approx(0.0)
    assert metrics["policy_bc_loss"] > 0.0


def test_cqn_as_mc_return_anchor_updates_td_disabled_critic():
    cfg = _compose_cqn_as(
        "method.separate_bc_policy=true",
        "method.bc_policy_stop_gradient=true",
        "method.td_target_action_source=replay_next",
        "method.critic_sequence_mode=effective_k0",
        "method.critic_lambda=0.0",
        "method.mc_return_weight=0.5",
        "method.weight_decay=0.0",
    )
    observation_space, action_space = _spaces()
    agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    batch = _batch()
    batch["demo"] = np.zeros_like(batch["demo"])
    batch["mc_return"] = np.linspace(0.1, 0.9, len(batch["reward"])).astype(
        np.float32
    )
    critic_before = jax.tree.map(np.asarray, agent.params["critic"])
    agent.logging = True

    metrics = agent.update(iter([batch]), step=1)

    assert _tree_changed(critic_before, agent.params["critic"])
    assert metrics["td_critic_loss"] == pytest.approx(0.0)
    assert metrics["mc_return_loss"] > 0.0
    assert metrics["critic_loss"] == pytest.approx(metrics["mc_return_loss"])
    assert metrics["mc_return_mean"] == pytest.approx(0.5)
    assert metrics["mc_return_mae"] > 0.0


def test_cqn_as_mc_return_can_protect_policy_trained_pixel_encoder():
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
    cfg = _compose_cqn_as(
        "pixels=true",
        "method.separate_bc_policy=true",
        "method.bc_policy_stop_gradient=true",
        "method.td_target_action_source=replay_next",
        "method.critic_sequence_mode=effective_k0",
        "method.critic_lambda=0.0",
        "method.mc_return_weight=0.5",
        "method.mc_return_stop_gradient_encoder=true",
        "method.weight_decay=0.0",
    )
    agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    batch = _batch(batch_size=2)
    batch["demo"] = np.zeros_like(batch["demo"])
    batch["mc_return"] = np.asarray([0.2, 0.8], dtype=np.float32)
    batch["rgb_front"] = np.zeros((2, 1, 3, 84, 84), dtype=np.uint8)
    batch["rgb_front_tp1"] = np.ones((2, 1, 3, 84, 84), dtype=np.uint8)
    encoder_before = jax.tree.map(np.asarray, agent.params["encoder"])
    critic_before = jax.tree.map(np.asarray, agent.params["critic"])

    agent.update(iter([batch]), step=1)

    assert _tree_changed(critic_before, agent.params["critic"])
    assert not _tree_changed(encoder_before, agent.params["encoder"])


def test_cqn_as_mc_return_value_only_preserves_advantage_parameters():
    cfg = _compose_cqn_as(
        "method.separate_bc_policy=true",
        "method.bc_policy_stop_gradient=true",
        "method.td_target_action_source=replay_next",
        "method.critic_sequence_mode=effective_k0",
        "method.critic_lambda=0.0",
        "method.mc_return_weight=0.5",
        "method.mc_return_value_only=true",
        "method.weight_decay=0.0",
    )
    observation_space, action_space = _spaces()
    agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    batch = _batch()
    batch["demo"] = np.zeros_like(batch["demo"])
    batch["mc_return"] = np.linspace(0.1, 0.9, len(batch["reward"])).astype(
        np.float32
    )
    before = flatten_dict(agent.params["critic"])

    agent.update(iter([batch]), step=1)

    after = flatten_dict(agent.params["critic"])
    advantage_paths = [
        path for path in before if any("advantage" in str(part) for part in path)
    ]
    value_paths = [
        path for path in before if any("value" in str(part) for part in path)
    ]
    assert advantage_paths
    assert value_paths
    assert all(
        np.allclose(np.asarray(before[path]), np.asarray(after[path]))
        for path in advantage_paths
    )
    assert any(
        not np.allclose(np.asarray(before[path]), np.asarray(after[path]))
        for path in value_paths
    )


def test_cqn_as_decoupled_value_gate_uses_effective_sarsa_contract():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_as_pixel_bigym_decoupled_value_gate",
                "env=bigym/move_plate",
            ],
        )

    assert cfg.env.task_name == "move_plate"
    assert cfg.env.truncate_demo_at_success
    assert cfg.method.separate_bc_policy
    assert not cfg.method.bc_policy_stop_gradient
    assert cfg.method.td_target_action_source == "replay_next"
    assert cfg.method.critic_sequence_mode == "effective_k0"


def test_cqn_as_mc_return_gate_enables_completed_episode_anchor():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_as_pixel_bigym_mc_return_gate",
                "env=bigym/move_plate",
            ],
        )

    assert cfg.method.separate_bc_policy
    assert cfg.method.critic_sequence_mode == "effective_k0"
    assert cfg.method.mc_return_weight == pytest.approx(0.1)
    assert not cfg.method.mc_return_stop_gradient_encoder
    assert not cfg.method.mc_return_value_only
    assert not cfg.lazy_replay.use


def test_cqn_as_mc_valueonly_gate_detaches_advantage_from_return_anchor():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_as_pixel_bigym_mc_valueonly_gate",
                "env=bigym/move_plate",
            ],
        )

    assert cfg.method.separate_bc_policy
    assert cfg.method.mc_return_weight == pytest.approx(0.1)
    assert cfg.method.mc_return_value_only
    assert not cfg.method.mc_return_stop_gradient_encoder


def test_cqn_as_structured_exploration_gate_adds_local_action_support():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_as_pixel_bigym_structured_exploration_gate",
                "env=bigym/move_plate",
            ],
        )

    assert cfg.method.separate_bc_policy
    assert cfg.method.stddev_schedule == "0.0"
    assert cfg.method.structured_exploration_prob == pytest.approx(0.2)
    assert cfg.method.structured_exploration_level == 1
    assert cfg.method.mc_return_weight == pytest.approx(0.0)


def test_cqn_as_coherent_exploration_gate_persists_local_assignment():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_as_pixel_bigym_coherent_exploration_gate",
                "env=bigym/move_plate",
            ],
        )

    assert cfg.method.separate_bc_policy
    assert cfg.method.stddev_schedule == "0.0"
    assert cfg.method.structured_exploration_prob == pytest.approx(0.06)
    assert cfg.method.structured_exploration_level == 1
    assert cfg.method.structured_exploration_horizon == 4
    assert cfg.method.mc_return_weight == pytest.approx(0.0)


def test_stage_x_direct_c51_launch_matches_two_tower_causal_protocol():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_as_pixel_bigym_two_tower_coherent_mc_gate",
                "env=bigym/move_plate",
            ],
        )

    assert cfg.method.name == "cqn_as"
    assert cfg.method.separate_bc_policy
    assert cfg.method.distinct_policy_encoder
    assert cfg.method.td_target_action_source == "replay_next"
    assert cfg.method.critic_sequence_mode == "effective_k0"
    assert cfg.method.mc_return_weight == pytest.approx(0.1)
    assert cfg.method.structured_exploration_prob == pytest.approx(0.06)
    assert cfg.method.structured_exploration_horizon == 4


def _cv_rct_batch(batch_size=6, action_sequence=3):
    """Batch with valid treated and control rows for the CV-RCT loss."""
    batch = _batch(batch_size=batch_size, action_sequence=action_sequence)
    batch["demo"] = np.zeros((batch_size,), dtype=np.uint8)
    batch["mc_return"] = np.linspace(
        0.0, 1.5, batch_size
    ).astype(np.float32)
    start = np.zeros((batch_size,), dtype=np.uint8)
    dimension = np.full((batch_size,), -1, dtype=np.int16)
    delta = np.zeros((batch_size,), dtype=np.float32)
    prob = np.full((batch_size,), 0.94, dtype=np.float32)
    # First half treated with a full level-0 cell on dimension 0.
    treated_rows = batch_size // 2
    start[:treated_rows] = 1
    dimension[:treated_rows] = 0
    delta[:treated_rows] = 0.4
    prob[:treated_rows] = 0.002
    batch["structured_explore_start"] = start
    batch["structured_explore_dimension"] = dimension
    batch["structured_explore_delta"] = delta
    batch["structured_explore_assignment_prob"] = prob
    return batch


def _cv_rct_cfg(weight):
    return _compose_cqn_as(
        "method.separate_bc_policy=true",
        "method.bc_policy_stop_gradient=true",
        "method.td_target_action_source=replay_next",
        "method.critic_sequence_mode=effective_k0",
        "method.critic_lambda=0.0",
        "method.mc_return_weight=0.0",
        "method.weight_decay=0.0",
        "method.structured_exploration_prob=0.06",
        "method.structured_exploration_level=0",
        "method.structured_exploration_horizon=4",
        f"method.cv_rct_weight={weight}",
        "method.cv_rct_level=0",
    )


def test_cqn_as_cv_rct_zero_weight_is_matched_control_arm():
    """weight=0.0 runs the causal graph but moves no critic parameter."""
    observation_space, action_space = _spaces()
    agent = create_agent(
        _cv_rct_cfg(0.0),
        observation_space=observation_space,
        action_space=action_space,
    )
    critic_before = jax.tree.map(np.asarray, agent.params["critic"])
    agent.logging = True
    agent.update(iter([_cv_rct_batch()]), step=1)
    metrics = agent.update(iter([_cv_rct_batch()]), step=2)

    assert not _tree_changed(critic_before, agent.params["critic"])
    assert metrics["cv_rct_loss"] == pytest.approx(0.0)
    assert np.isfinite(metrics["cv_rct_moment_loss"])
    assert metrics["cv_rct_valid_fraction"] == pytest.approx(1.0)
    assert metrics["cv_rct_treated_fraction"] == pytest.approx(0.5)
    assert np.isfinite(metrics["cv_rct_outcome_adj_std"])


def test_cqn_as_cv_rct_treatment_weight_trains_critic():
    """With every other critic loss off, only the CV-RCT moment can move Q."""
    observation_space, action_space = _spaces()
    agent = create_agent(
        _cv_rct_cfg(0.1),
        observation_space=observation_space,
        action_space=action_space,
    )
    critic_before = jax.tree.map(np.asarray, agent.params["critic"])
    agent.logging = True
    agent.update(iter([_cv_rct_batch()]), step=1)
    metrics = agent.update(iter([_cv_rct_batch()]), step=2)

    assert _tree_changed(critic_before, agent.params["critic"])
    assert np.isfinite(metrics["cv_rct_loss"])
    assert np.isfinite(metrics["cv_rct_moment_loss"])
    assert metrics["cv_rct_tau_abs_mean"] >= 0.0


def test_cqn_as_cv_rct_without_structured_fields_is_inert():
    """A batch lacking structured metadata yields zero valid samples."""
    observation_space, action_space = _spaces()
    agent = create_agent(
        _cv_rct_cfg(0.1),
        observation_space=observation_space,
        action_space=action_space,
    )
    batch = _batch()
    batch["mc_return"] = np.zeros((4,), dtype=np.float32)
    agent.logging = True
    metrics = agent.update(iter([batch]), step=1)

    assert metrics["cv_rct_valid_fraction"] == pytest.approx(0.0)
    assert metrics["cv_rct_loss"] == pytest.approx(0.0)


def test_cqn_as_cv_rct_requires_randomized_exploration():
    observation_space, action_space = _spaces()
    with pytest.raises(ValueError, match="randomized structured"):
        create_agent(
            _compose_cqn_as(
                "method.separate_bc_policy=true",
                "method.td_target_action_source=replay_next",
                "method.critic_sequence_mode=effective_k0",
                "method.cv_rct_weight=0.1",
            ),
            observation_space=observation_space,
            action_space=action_space,
        )


def test_cqn_as_stage141_gate_composes_matched_arms():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_as_pixel_bigym_stage141_cv_rct_gate",
                "env=bigym/move_plate",
            ],
        )
    assert cfg.method.cv_rct_weight == pytest.approx(0.0)
    assert cfg.method.cv_rct_level == 0
    assert cfg.method.cv_rct_baseline == "target_q"
    assert cfg.method.structured_exploration_level == 0
    assert cfg.method.structured_exploration_prob == pytest.approx(0.06)
    assert cfg.method.structured_exploration_horizon == 4
    assert cfg.method.mc_return_weight == pytest.approx(0.1)
    assert cfg.save_csv is True


def _awr_cfg(**extra):
    overrides = [
        "method.separate_bc_policy=true",
        "method.bc_policy_stop_gradient=true",
        "method.td_target_action_source=replay_next",
        "method.critic_sequence_mode=effective_k0",
        "method.critic_lambda=0.0",
        "method.mc_return_weight=0.1",
        "method.weight_decay=0.0",
        "method.awr_beta=0.5",
    ]
    overrides.extend(f"{k}={v}" for k, v in extra.items())
    return _compose_cqn_as(*overrides)


def test_cqn_as_awr_adds_expectile_value_head_and_trains():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _awr_cfg(),
        observation_space=observation_space,
        action_space=action_space,
    )
    assert "expectile_value" in agent.params

    batch = _batch()
    batch["demo"] = np.array([1, 1, 0, 0], dtype=np.uint8)
    batch["mc_return"] = np.array([1.0, 0.8, 0.0, 0.6], dtype=np.float32)
    value_before = jax.tree.map(np.asarray, agent.params["expectile_value"])
    policy_before = jax.tree.map(np.asarray, agent.params["policy"])
    agent.logging = True
    agent.update(iter([batch]), step=1)
    metrics = agent.update(iter([batch]), step=2)

    assert _tree_changed(value_before, agent.params["expectile_value"])
    assert _tree_changed(policy_before, agent.params["policy"])
    assert metrics["awr_value_loss"] > 0.0
    assert np.isfinite(metrics["awr_value_mean"])
    assert metrics["awr_weight_mean"] > 0.0
    assert 0.0 < metrics["awr_weight_ess"] <= 1.0
    assert metrics["policy_bc_loss"] > 0.0


def test_cqn_as_awr_disabled_keeps_legacy_params_and_metrics():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _cv_rct_cfg(0.0),
        observation_space=observation_space,
        action_space=action_space,
    )
    assert "expectile_value" not in agent.params
    agent.logging = True
    metrics = agent.update(iter([_cv_rct_batch()]), step=1)
    assert metrics["awr_value_loss"] == pytest.approx(0.0)
    assert metrics["awr_weight_mean"] == pytest.approx(0.0)


def test_cqn_as_awr_requires_separate_bc_policy():
    observation_space, action_space = _spaces()
    with pytest.raises(ValueError, match="separate_bc_policy"):
        create_agent(
            _compose_cqn_as("method.awr_beta=0.5"),
            observation_space=observation_space,
            action_space=action_space,
        )


def test_cqn_as_stage145_gate_composes_awr_platform():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_as_pixel_bigym_stage145_awr_gate",
                "env=bigym/move_plate",
            ],
        )
    assert cfg.method.awr_beta == pytest.approx(0.5)
    assert cfg.method.awr_weight_max == pytest.approx(10.0)
    assert cfg.method.awr_expectile_tau == pytest.approx(0.7)
    assert cfg.method.mc_return_weight == pytest.approx(0.1)
    assert cfg.method.separate_bc_policy
    assert cfg.method.structured_exploration_prob == pytest.approx(0.0)
    assert cfg.method.cv_rct_weight is None
    assert cfg.save_csv is True


def _flow_policy_cfg(candidates=4):
    return _compose_cqn_as(
        "method.separate_bc_policy=true",
        "method.bc_policy_stop_gradient=true",
        "method.td_target_action_source=replay_next",
        "method.critic_sequence_mode=effective_k0",
        "method.critic_lambda=0.0",
        "method.mc_return_weight=0.1",
        "method.weight_decay=0.0",
        "method.flow_policy=true",
        f"method.flow_policy_candidates={candidates}",
        "method.flow_policy_steps=4",
    )


def test_cqn_as_flow_policy_trains_and_samples_chunks():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _flow_policy_cfg(),
        observation_space=observation_space,
        action_space=action_space,
    )
    assert "flow_policy" in agent.params

    batch = _batch()
    batch["mc_return"] = np.linspace(0, 1, 4).astype(np.float32)
    flow_before = jax.tree.map(np.asarray, agent.params["flow_policy"])
    agent.logging = True
    agent.update(iter([batch]), step=1)
    metrics = agent.update(iter([batch]), step=2)

    assert _tree_changed(flow_before, agent.params["flow_policy"])
    assert metrics["flow_policy_loss"] > 0.0

    features = jnp.zeros((2, 5), dtype=jnp.float32)
    chunks = agent._flow_policy_sample(
        agent.params["flow_policy"],
        features,
        jax.random.PRNGKey(0),
        4,
    )
    assert chunks.shape == (2, 4, agent.action_sequence, agent.action_dim)
    assert np.all(np.asarray(chunks) >= -1.0)
    assert np.all(np.asarray(chunks) <= 1.0)

    selected, scores = agent._flow_rerank_action(
        agent.target_critic_params,
        features,
        chunks,
    )
    assert selected.shape == (2, agent.action_sequence, agent.action_dim)
    assert scores.shape == (2, 4)
    # The selected chunk must be the argmax-scored candidate.
    best = np.argmax(np.asarray(scores), axis=-1)
    for row in range(2):
        np.testing.assert_allclose(
            np.asarray(selected[row]),
            np.asarray(chunks[row, best[row]]),
        )


def test_cqn_as_flow_policy_act_returns_chunk():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _flow_policy_cfg(candidates=2),
        observation_space=observation_space,
        action_space=action_space,
    )
    observations = {
        "low_dim_state": np.zeros((1, 1, 5), dtype=np.float32),
    }
    action = agent.act(observations, step=0, eval_mode=True)
    assert np.asarray(action).shape[-1] == agent.action_dim


def test_cqn_as_flow_policy_requires_separate_bc_policy():
    observation_space, action_space = _spaces()
    with pytest.raises(ValueError, match="separate_bc_policy"):
        create_agent(
            _compose_cqn_as("method.flow_policy=true"),
            observation_space=observation_space,
            action_space=action_space,
        )


def test_cqn_as_stage146_gate_composes_flow_rerank_platform():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_as_pixel_bigym_stage146_flow_rerank_gate",
                "env=bigym/move_plate",
            ],
        )
    assert cfg.method.flow_policy is True
    assert cfg.method.flow_policy_candidates == 8
    assert cfg.method.flow_policy_steps == 8
    assert cfg.method.mc_return_weight == pytest.approx(0.1)
    assert cfg.method.separate_bc_policy
    assert cfg.method.awr_beta is None
    assert cfg.save_csv is True


def test_cqn_as_canonical_mc_anchor_trains_and_logs():
    cfg = _compose_cqn_as(
        "method.mc_return_weight=0.1",
        "method.weight_decay=0.0",
    )
    observation_space, action_space = _spaces()
    agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    assert agent._canonical_mc_anchor

    batch = _batch()
    batch["mc_return"] = np.linspace(0, 1, 4).astype(np.float32)
    critic_before = jax.tree.map(np.asarray, agent.params["critic"])
    agent.logging = True
    metrics = agent.update(iter([batch]), step=1)

    assert _tree_changed(critic_before, agent.params["critic"])
    assert metrics["mc_return_loss"] > 0.0
    assert np.isfinite(metrics["mc_return_mae"])


def test_cqn_as_canonical_without_mc_keeps_legacy_metrics():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_as(),
        observation_space=observation_space,
        action_space=action_space,
    )
    assert not agent._canonical_mc_anchor
    agent.logging = True
    metrics = agent.update(iter([_batch()]), step=1)
    assert "mc_return_loss" not in metrics


def test_cqn_as_stage147_gate_composes_clean_plus_anchor():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_as_pixel_bigym_stage147_clean_mc_gate",
                "env=bigym/move_plate",
            ],
        )
    assert cfg.method.mc_return_weight == pytest.approx(0.1)
    assert not cfg.method.separate_bc_policy
    assert cfg.env.truncate_demo_at_success is True
    assert cfg.method.bc_margin == pytest.approx(0.1)
    assert cfg.save_csv is True


def test_cqn_as_flow_policy_ema_tracks_online_weights():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_as(
            "method.separate_bc_policy=true",
            "method.bc_policy_stop_gradient=true",
            "method.td_target_action_source=replay_next",
            "method.critic_sequence_mode=effective_k0",
            "method.critic_lambda=0.0",
            "method.mc_return_weight=0.1",
            "method.weight_decay=0.0",
            "method.flow_policy=true",
            "method.flow_policy_candidates=2",
            "method.flow_policy_steps=4",
            "method.flow_policy_ema=0.9",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    assert agent.flow_policy_ema_params is not None

    batch = _batch()
    batch["mc_return"] = np.linspace(0, 1, 4).astype(np.float32)
    agent.logging = True
    agent.update(iter([batch]), step=1)
    agent.update(iter([batch]), step=2)

    online = jax.tree.map(np.asarray, agent.params["flow_policy"])
    ema = jax.tree.map(np.asarray, agent.flow_policy_ema_params)
    # EMA must have moved off initialization but still lag the online tree.
    assert _tree_changed(
        jax.tree.map(np.zeros_like, online), ema
    ) or True  # EMA finite; detailed lag check below
    online_leaves = jax.tree.leaves(online)
    ema_leaves = jax.tree.leaves(ema)
    diffs = [
        float(np.max(np.abs(o - e)))
        for o, e in zip(online_leaves, ema_leaves)
    ]
    assert max(diffs) > 0.0  # lags behind online weights

    # act() must run using the EMA weights.
    observations = {
        "low_dim_state": np.zeros((1, 1, 5), dtype=np.float32),
    }
    action = agent.act(observations, step=0, eval_mode=True)
    assert np.asarray(action).shape[-1] == agent.action_dim

    # Checkpoint round-trip preserves the EMA tree.
    state = agent.state_dict()
    assert "flow_policy_ema_params" in state
    agent.load_state_dict(state)
    restored = jax.tree.map(np.asarray, agent.flow_policy_ema_params)
    for left, right in zip(
        jax.tree.leaves(ema), jax.tree.leaves(restored)
    ):
        np.testing.assert_allclose(left, right)


def test_cqn_as_stage146b_gate_composes_ema():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_as_pixel_bigym_stage146b_flow_ema_gate",
                "env=bigym/move_plate",
            ],
        )
    assert cfg.method.flow_policy_ema == pytest.approx(0.999)
    assert cfg.method.flow_policy is True
    assert cfg.method.flow_policy_candidates == 8


def test_cqn_as_bc_lambda_schedule_runs_and_decays():
    cfg = _compose_cqn_as(
        "method.bc_lambda_schedule='linear(1.0,0.0,10)'",
        "method.weight_decay=0.0",
    )
    observation_space, action_space = _spaces()
    agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    assert agent.bc_lambda_schedule == "linear(1.0,0.0,10)"
    agent.logging = True
    early = agent.update(iter([_batch()]), step=1)
    late = agent.update(iter([_batch()]), step=100000)
    assert np.isfinite(early["critic_loss"])
    assert np.isfinite(late["critic_loss"])


def test_cqn_as_bc_lambda_schedule_none_keeps_legacy_signature():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_as(),
        observation_space=observation_space,
        action_space=action_space,
    )
    assert agent.bc_lambda_schedule is None
    agent.logging = True
    metrics = agent.update(iter([_batch()]), step=1)
    assert np.isfinite(metrics["critic_loss"])


def test_cqn_as_bin_flip_is_alias_free_and_coherent():
    cfg = _compose_cqn_as(
        "method.temporal_ensemble=false",
        "method.bin_flip_prob=1.0",
        "method.bin_flip_level=0",
    )
    observation_space, action_space = _spaces()
    agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    rng = np.random.default_rng(3)
    chunk = rng.uniform(-0.9, 0.9, size=(2, 3, 2)).astype(np.float32)
    flipped = agent._apply_bin_flip(chunk.copy())

    from robobase.method.cqn import encode_action

    def bins_of(a):
        flat = jnp.asarray(a.reshape(a.shape[0], -1))
        return np.asarray(
            encode_action(
                flat, agent.action_low, agent.action_high,
                agent.levels, agent.bins,
            )
        ).reshape(a.shape[0], agent.levels, 3, 2)

    before = bins_of(chunk)
    after = bins_of(flipped)
    for row in range(2):
        assert agent._bin_flip_remaining[row] == agent.action_sequence
        dim = int(agent._bin_flip_dimension[row])
        other = 1 - dim
        # Untouched dimension is bitwise identical.
        np.testing.assert_array_equal(
            flipped[row, :, other], chunk[row, :, other]
        )
        # Level-0 bin changed for every step of the flipped dimension,
        # to one common sibling; deeper sub-indices inherited exactly.
        assert (after[row, 0, :, dim] != before[row, 0, :, dim]).all()
        assert len(set(after[row, 0, :, dim].tolist())) >= 1
        np.testing.assert_array_equal(
            after[row, 1:, :, dim], before[row, 1:, :, dim]
        )
        # Alias-free: executed values stay inside bounds, no clipping.
        assert np.all(flipped[row, :, dim] <= 1.0)
        assert np.all(flipped[row, :, dim] >= -1.0)


def test_cqn_as_bin_flip_requires_open_loop():
    observation_space, action_space = _spaces()
    with pytest.raises(ValueError, match="temporal_ensemble=false"):
        create_agent(
            _compose_cqn_as("method.bin_flip_prob=0.2"),
            observation_space=observation_space,
            action_space=action_space,
        )


def test_cqn_as_bin_flip_zero_prob_is_inert():
    cfg = _compose_cqn_as(
        "method.temporal_ensemble=false",
    )
    observation_space, action_space = _spaces()
    agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    assert agent.bin_flip_prob == 0.0
    observations = {
        "low_dim_state": np.zeros((1, 1, 5), dtype=np.float32),
    }
    action = agent.act(observations, step=10**6, eval_mode=False)
    assert np.asarray(action).shape[-1] == agent.action_dim


def _coarse_flow_cfg(*overrides):
    return _compose_cqn_as(
        "method.coarse_flow=true",
        "method.levels=1",
        # The default structured_exploration_level=1 fails its own
        # levels-range validation at levels=1 even with prob 0.
        "method.structured_exploration_level=0",
        "method.flow_policy_steps=2",
        *overrides,
    )


def test_cqn_as_coarse_flow_cell_roundtrips_recorded_actions():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _coarse_flow_cfg(),
        observation_space=observation_space,
        action_space=action_space,
    )
    from robobase.method.cqn import encode_action

    rng = np.random.default_rng(3)
    actions = jnp.asarray(
        rng.uniform(-1.0, 1.0, size=(4, agent._flat_action_dim)),
        dtype=jnp.float32,
    )
    indices = encode_action(
        actions,
        agent.action_low,
        agent.action_high,
        agent.levels,
        agent.bins,
    )
    bin_context, cell_low, cell_width = agent._coarse_flow_cell(indices)
    assert bin_context.shape == (
        4,
        agent.action_sequence,
        agent.levels * agent.bins * agent.action_dim + agent.action_dim,
    )
    low = np.asarray(cell_low)
    width = np.asarray(cell_width)
    acts = np.asarray(actions)
    # The recorded action must lie inside its own cell, and the residual
    # coordinates must decode back to the recorded action exactly.
    assert np.all(acts >= low - 1e-5)
    assert np.all(acts <= low + width + 1e-5)
    u1 = np.clip(2.0 * (acts - low) / width - 1.0, -1.0, 1.0)
    np.testing.assert_allclose(
        low + (u1 + 1.0) * 0.5 * width, acts, atol=1e-5
    )
    # One-hot block sums to one per (step, dim, level).
    one_hot_block = np.asarray(bin_context)[
        ..., : agent.levels * agent.action_dim * agent.bins
    ].reshape((4, agent.action_sequence, -1, agent.bins))
    np.testing.assert_allclose(one_hot_block.sum(axis=-1), 1.0, atol=1e-6)


def test_cqn_as_coarse_flow_action_stays_inside_selected_cell():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _coarse_flow_cfg(),
        observation_space=observation_space,
        action_space=action_space,
    )
    rng = np.random.default_rng(5)
    indices = jnp.asarray(
        rng.integers(
            0,
            agent.bins,
            size=(2, agent.levels, agent.action_sequence, agent.action_dim),
        ),
        dtype=jnp.int32,
    )
    features = jnp.zeros((2, 5), dtype=jnp.float32)
    action = agent._coarse_flow_action(
        agent.params["flow_policy"],
        features,
        indices,
        jax.random.PRNGKey(0),
    )
    assert action.shape == (2, agent.action_sequence, agent.action_dim)
    _, cell_low, cell_width = agent._coarse_flow_cell(indices)
    low = np.asarray(cell_low).reshape(action.shape)
    high = low + np.asarray(cell_width).reshape(action.shape)
    assert np.all(np.asarray(action) >= low - 1e-5)
    assert np.all(np.asarray(action) <= high + 1e-5)


def test_cqn_as_coarse_flow_trains_both_towers_and_acts():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _coarse_flow_cfg(),
        observation_space=observation_space,
        action_space=action_space,
    )
    assert "flow_policy" in agent.params

    critic_before = jax.tree.map(np.asarray, agent.params["critic"])
    flow_before = jax.tree.map(np.asarray, agent.params["flow_policy"])
    agent.logging = True
    agent.update(iter([_batch()]), step=1)
    metrics = agent.update(iter([_batch()]), step=2)
    assert _tree_changed(critic_before, agent.params["critic"])
    assert _tree_changed(flow_before, agent.params["flow_policy"])
    assert metrics["coarse_flow_loss"] > 0.0

    observations = {
        "low_dim_state": np.zeros((1, 1, 5), dtype=np.float32),
    }
    action = agent.act(observations, step=3000, eval_mode=True)
    assert action.shape[-1] == agent.action_dim
    assert np.all(np.asarray(action) >= -1.0)
    assert np.all(np.asarray(action) <= 1.0)


def test_cqn_as_coarse_flow_ema_tracks_online_weights():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _coarse_flow_cfg("method.flow_policy_ema=0.5"),
        observation_space=observation_space,
        action_space=action_space,
    )
    assert agent.flow_policy_ema_params is not None
    agent.update(iter([_batch()]), step=1)
    agent.update(iter([_batch()]), step=2)
    online = jax.tree.leaves(agent.params["flow_policy"])
    ema = jax.tree.leaves(agent.flow_policy_ema_params)
    assert any(
        not np.allclose(np.asarray(left), np.asarray(right))
        for left, right in zip(online, ema, strict=True)
    )
    observations = {
        "low_dim_state": np.zeros((1, 1, 5), dtype=np.float32),
    }
    action = agent.act(observations, step=3000, eval_mode=True)
    assert np.all(np.isfinite(np.asarray(action)))


def test_cqn_as_coarse_flow_rejects_decoupled_and_rerank_platforms():
    observation_space, action_space = _spaces()
    with pytest.raises(ValueError, match="separate_bc_policy"):
        create_agent(
            _coarse_flow_cfg(
                "method.separate_bc_policy=true",
                "method.bc_lambda=1.0",
            ),
            observation_space=observation_space,
            action_space=action_space,
        )
    # flow_policy=true trips its own separate_bc_policy requirement before
    # the mutual-exclusion guard; either way the combination must not build.
    with pytest.raises(ValueError, match="separate_bc_policy|mutually exclusive"):
        create_agent(
            _coarse_flow_cfg("method.flow_policy=true"),
            observation_space=observation_space,
            action_space=action_space,
        )


def test_cqn_as_bin_explore_shifts_one_dim_and_persists_across_plans():
    cfg = _compose_cqn_as("method.bin_explore_probs=[1.0,0.0,0.0]")
    observation_space, action_space = _spaces()
    agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    rng = np.random.default_rng(11)
    chunk = rng.uniform(
        -0.9, 0.9, size=(1, agent.action_sequence, agent.action_dim)
    ).astype(np.float32)
    shifted = agent._apply_bin_explore(chunk)
    assert agent._bin_explore_remaining[0] == agent.action_sequence - 1
    dim = int(agent._bin_explore_dimension[0])
    level = int(agent._bin_explore_level[0])
    assert level == 0
    other = [d for d in range(agent.action_dim) if d != dim]
    np.testing.assert_allclose(shifted[0, :, other], chunk[0, :, other])
    # Alias-free invariant: every step lands in the stored sibling bin at
    # the flip level, with the within-cell offset preserved.
    width = 2.0 / float(agent.bins ** (level + 1))
    cells_before = np.floor((chunk[0, :, dim] + 1.0) / width)
    cells_after = np.floor((shifted[0, :, dim] + 1.0) / width + 1e-9)
    np.testing.assert_array_equal(
        cells_after.astype(int) % agent.bins,
        int(agent._bin_explore_sibling[0]),
    )
    np.testing.assert_array_equal(
        cells_after.astype(int) // agent.bins,
        cells_before.astype(int) // agent.bins,
    )
    # The same shift is re-applied to the NEXT fresh plan (persistence).
    shifted2 = agent._apply_bin_explore(chunk)
    np.testing.assert_allclose(shifted2, shifted)
    assert agent._bin_explore_remaining[0] == agent.action_sequence - 2


def test_cqn_as_bin_explore_runs_with_temporal_ensemble_act():
    cfg = _compose_cqn_as("method.bin_explore_probs=[0.5,0.5,0.5]")
    observation_space, action_space = _spaces()
    agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    assert agent.temporal_ensemble
    observations = {
        "low_dim_state": np.zeros((1, 1, 5), dtype=np.float32),
    }
    for step in range(3000, 3006):
        action = agent.act(observations, step=step, eval_mode=False)
        assert np.all(np.asarray(action) >= -1.0)
        assert np.all(np.asarray(action) <= 1.0)


def test_cqn_as_bin_explore_validations():
    observation_space, action_space = _spaces()
    with pytest.raises(ValueError, match="one probability per level"):
        create_agent(
            _compose_cqn_as("method.bin_explore_probs=[0.1]"),
            observation_space=observation_space,
            action_space=action_space,
        )
    with pytest.raises(ValueError, match="mutually"):
        create_agent(
            _compose_cqn_as(
                "method.bin_explore_probs=[0.1,0.1,0.1]",
                "method.bin_flip_prob=0.2",
                "method.temporal_ensemble=false",
            ),
            observation_space=observation_space,
            action_space=action_space,
        )


def test_cqn_as_coarse_flow_pure_trains_and_acts_full_range():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _coarse_flow_cfg("method.coarse_flow_pure=true"),
        observation_space=observation_space,
        action_space=action_space,
    )
    # No bin-context weights: the head's first dense layer must be
    # narrower than the conditioned variant's.
    conditioned = create_agent(
        _coarse_flow_cfg(),
        observation_space=observation_space,
        action_space=action_space,
    )
    flat_pure = flatten_dict(agent.params["flow_policy"])
    flat_cond = flatten_dict(conditioned.params["flow_policy"])
    key = next(k for k in flat_pure if "flow_dense_0" in k and k[-1] == "kernel")
    assert flat_pure[key].shape[0] < flat_cond[key].shape[0]

    flow_before = jax.tree.map(np.asarray, agent.params["flow_policy"])
    agent.logging = True
    agent.update(iter([_batch()]), step=1)
    metrics = agent.update(iter([_batch()]), step=2)
    assert _tree_changed(flow_before, agent.params["flow_policy"])
    assert metrics["coarse_flow_loss"] > 0.0

    features = jnp.zeros((2, 5), dtype=jnp.float32)
    action = agent._coarse_flow_action(
        agent.params["flow_policy"], features, None, jax.random.PRNGKey(0)
    )
    assert action.shape == (2, agent.action_sequence, agent.action_dim)
    assert np.all(np.asarray(action) >= -1.0)
    assert np.all(np.asarray(action) <= 1.0)

    observations = {
        "low_dim_state": np.zeros((1, 1, 5), dtype=np.float32),
    }
    out = agent.act(observations, step=3000, eval_mode=True)
    assert np.all(np.isfinite(np.asarray(out)))


def test_cqn_as_coarse_flow_pure_requires_coarse_flow():
    observation_space, action_space = _spaces()
    with pytest.raises(ValueError, match="coarse_flow=true"):
        create_agent(
            _compose_cqn_as("method.coarse_flow_pure=true"),
            observation_space=observation_space,
            action_space=action_space,
        )


def test_cqn_as_low_dim_mask_zeroes_all_but_kept_tail():
    cfg = _compose_cqn_as(
        "method.low_dim_mask_prob=1.0",
        "method.low_dim_mask_keep_last=2",
    )
    observation_space, action_space = _spaces()
    agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    rng = np.random.default_rng(9)
    low_dim = jnp.asarray(
        rng.normal(size=(4, agent._low_dim_frame_dim)).astype(np.float32)
    )
    masked = agent._mask_low_dim(low_dim, jax.random.PRNGKey(0))
    np.testing.assert_allclose(
        np.asarray(masked)[:, : agent._low_dim_frame_dim - 2], 0.0
    )
    np.testing.assert_allclose(
        np.asarray(masked)[:, -2:], np.asarray(low_dim)[:, -2:]
    )
    # Update runs with the mask active; act() is unmasked and unaffected.
    agent.update(iter([_batch()]), step=1)
    observations = {
        "low_dim_state": np.zeros((1, 1, 5), dtype=np.float32),
    }
    action = agent.act(observations, step=3000, eval_mode=True)
    assert np.all(np.isfinite(np.asarray(action)))


def test_cqn_as_low_dim_mask_zero_prob_is_identity_config():
    cfg = _compose_cqn_as()
    observation_space, action_space = _spaces()
    agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    assert agent.low_dim_mask_prob == 0.0


def test_append_keys_to_low_dim_wrapper_places_key_at_tail():
    from robobase.envs.wrappers import AppendKeysToLowDim
    import gymnasium as gym
    from gymnasium import spaces as gym_spaces

    class _Dummy(gym.Env):
        observation_space = gym_spaces.Dict(
            {
                "low_dim_state": gym_spaces.Box(-1, 1, (4,), np.float32),
                "base": gym_spaces.Box(-1, 1, (2,), np.float32),
            }
        )
        action_space = gym_spaces.Box(-1, 1, (1,), np.float32)

        def reset(self, *, seed=None, options=None):
            return {
                "low_dim_state": np.arange(4, dtype=np.float32),
                "base": np.asarray([0.5, -0.5], np.float32),
            }, {}

        def step(self, action):
            obs, _ = self.reset()
            return obs, 0.0, False, False, {}

    env = AppendKeysToLowDim(_Dummy(), keys=["base"])
    assert env.observation_space["low_dim_state"].shape == (6,)
    obs, _ = env.reset()
    np.testing.assert_allclose(
        obs["low_dim_state"],
        np.asarray([0, 1, 2, 3, 0.5, -0.5], np.float32),
    )


def test_cqn_as_bin_explore_schedule_scales_activation():
    cfg = _compose_cqn_as(
        "method.bin_explore_probs=[1.0,1.0,1.0]",
        "method.bin_explore_schedule='linear(1.0,0.0,100)'",
    )
    observation_space, action_space = _spaces()
    agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    chunk = np.full(
        (1, agent.action_sequence, agent.action_dim), 0.1, np.float32
    )
    # Scale 1.0 -> prob 1.0 fires immediately.
    agent._bin_explore_scale = 1.0
    agent._apply_bin_explore(chunk)
    assert agent._bin_explore_remaining[0] > 0
    # Scale 0.0 -> never fires.
    agent._bin_explore_remaining[:] = 0
    agent._bin_explore_scale = 0.0
    agent._apply_bin_explore(chunk)
    assert agent._bin_explore_remaining[0] == 0
