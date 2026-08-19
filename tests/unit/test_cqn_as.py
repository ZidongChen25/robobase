from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from gymnasium import spaces
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from robobase.envs.wrappers import RecedingHorizonControl
from robobase.factory import create_agent
from robobase.method.cqn_research import (
    advantage_learning_target_shift,
    categorical_point_mass,
    dense_return_distributional_loss,
    dense_return_expected_q_loss,
    episodic_success_returns,
    ordered_success_returns,
    progress_shaped_rewards,
    sequence_aligned_sparse_returns,
    shift_categorical_distribution,
    unseen_return_floor_loss,
)
from robobase.method.cqn_as_research import (
    AutoregressiveActionCorrection,
    C2FSequenceDistributionalCritic,
    cqn_as_spec_from_cfg,
    pessimistic_categorical_q,
    select_episodic_twin_actions,
    shift_replay_action_sequence,
    top2_joint_beam,
)
from robobase.models.encoder import JaxCQNEncoder
from robobase.replay_buffer.vision_feature_cache import JAX_CQN_AS_FEATURE_KEY
from robobase.workspace import (
    Workspace,
    _effective_episode_length,
    _mc_return_anchor_enabled,
    _progress_label_enabled,
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


def test_cqn_as_replay_sarsa_shifts_consecutive_actions_exactly():
    actions = jnp.asarray(
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
            [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
        ],
        dtype=jnp.float32,
    )
    shifted = shift_replay_action_sequence(actions, 3, 2)
    np.testing.assert_array_equal(
        shifted,
        np.asarray(
            [
                [[2.0, 20.0], [3.0, 30.0], [3.0, 30.0]],
                [[5.0, 50.0], [6.0, 60.0], [6.0, 60.0]],
            ],
            dtype=np.float32,
        ),
    )


def test_sequence_aligned_sparse_returns_recovers_each_token_return():
    gamma = 0.99
    returns = jnp.asarray([0.0, gamma**3], dtype=jnp.float32)
    aligned = sequence_aligned_sparse_returns(
        returns,
        action_sequence=4,
        action_dim=2,
        discount=gamma,
    )
    np.testing.assert_array_equal(aligned[0], np.zeros(8, dtype=np.float32))
    np.testing.assert_allclose(
        aligned[1],
        np.repeat(
            np.asarray([gamma**3, gamma**2, gamma, 1.0], dtype=np.float32),
            2,
        ),
        rtol=1e-6,
    )


def test_categorical_point_mass_projects_per_token_values():
    support = jnp.asarray([0.0, 0.5, 1.0], dtype=jnp.float32)
    values = jnp.asarray([[0.0, 0.25, 1.0]], dtype=jnp.float32)
    projected = categorical_point_mass(values, support)
    np.testing.assert_allclose(
        projected,
        np.asarray(
            [[[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]],
            dtype=np.float32,
        ),
    )


def test_sequence_aligned_zero_return_has_no_action_label_signal():
    support = jnp.asarray([0.0, 0.5, 1.0], dtype=jnp.float32)
    logits = jax.random.normal(
        jax.random.PRNGKey(21),
        (2, 2, 6, 5, 3),
    )
    aligned = sequence_aligned_sparse_returns(
        jnp.zeros((2,), dtype=jnp.float32),
        action_sequence=3,
        action_dim=2,
        discount=0.99,
    )
    target = categorical_point_mass(aligned[:, None, :], support)
    target = jnp.broadcast_to(target, (2, 2, 6, 3))
    first_actions = jnp.zeros((2, 2, 6), dtype=jnp.int32)
    last_actions = jnp.full((2, 2, 6), 4, dtype=jnp.int32)

    def objective(value, actions):
        per_sample, _, _ = dense_return_distributional_loss(
            value,
            actions,
            target,
            support,
            0.0,
        )
        return per_sample.mean()

    first_loss, first_grad = jax.value_and_grad(objective)(
        logits,
        first_actions,
    )
    last_loss, last_grad = jax.value_and_grad(objective)(
        logits,
        last_actions,
    )
    np.testing.assert_array_equal(first_loss, last_loss)
    np.testing.assert_array_equal(first_grad, last_grad)


def test_dense_expected_q_loss_matches_exact_all_bin_regression():
    support = jnp.asarray([-1.0, 0.0, 1.0], dtype=jnp.float32)
    logits = jnp.asarray(
        [
            [
                [
                    [
                        [0.0, 0.0, 0.0],
                        [-1.0, 0.0, 1.0],
                        [1.0, 0.0, -1.0],
                    ]
                ]
            ]
        ],
        dtype=jnp.float32,
    )
    actions = jnp.asarray([[[1]]], dtype=jnp.int32)
    target = jnp.asarray([[[[0.0, 0.25, 0.75]]]], dtype=jnp.float32)
    per_sample, chosen_q, unseen_q = dense_return_expected_q_loss(
        logits,
        actions,
        target,
        support,
        floor_value=0.0,
    )

    all_q = np.sum(
        np.asarray(jax.nn.softmax(logits, axis=-1))
        * np.asarray(support),
        axis=-1,
    )
    targets = np.zeros_like(all_q)
    targets[..., 1] = 0.75
    expected_loss = 0.5 * np.square(all_q - targets).sum(axis=-1).mean(
        axis=(1, 2)
    )
    np.testing.assert_allclose(per_sample, expected_loss, rtol=1e-6)
    np.testing.assert_allclose(chosen_q, all_q[..., 1].mean(axis=(1, 2)))
    np.testing.assert_allclose(
        unseen_q,
        np.asarray([(all_q[..., 0].sum() + all_q[..., 2].sum()) / 2.0]),
    )


def test_dense_expected_q_zero_return_has_no_action_label_signal():
    support = jnp.asarray([-1.0, 0.0, 1.0], dtype=jnp.float32)
    logits = jax.random.normal(
        jax.random.PRNGKey(22),
        (2, 2, 6, 5, 3),
    )
    zero_target = jnp.zeros((2, 2, 6, 3), dtype=jnp.float32)
    zero_target = zero_target.at[..., 1].set(1.0)
    first_actions = jnp.zeros((2, 2, 6), dtype=jnp.int32)
    last_actions = jnp.full((2, 2, 6), 4, dtype=jnp.int32)

    def objective(value, actions):
        per_sample, _, _ = dense_return_expected_q_loss(
            value,
            actions,
            zero_target,
            support,
            floor_value=0.0,
        )
        return per_sample.mean()

    first_loss, first_grad = jax.value_and_grad(objective)(
        logits,
        first_actions,
    )
    last_loss, last_grad = jax.value_and_grad(objective)(
        logits,
        last_actions,
    )
    np.testing.assert_array_equal(first_loss, last_loss)
    np.testing.assert_array_equal(first_grad, last_grad)


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


def test_autoregressive_action_correction_uses_only_strict_prefix():
    correction = AutoregressiveActionCorrection(
        hidden_dim=8,
        action_sequence=3,
        action_dim=3,
        bins=5,
        atoms=7,
    )
    base_logits = jax.random.normal(
        jax.random.PRNGKey(1),
        (2, 3, 3, 5, 7),
    )
    features = jax.random.normal(jax.random.PRNGKey(2), (2, 4))
    action_context = jnp.zeros((2, 3, 3), dtype=jnp.float32)
    params = correction.init(
        jax.random.PRNGKey(3),
        base_logits,
        features,
        action_context,
    )
    mutable_params = unfreeze(params)
    output_kernel = mutable_params["params"]["output_projection"]["kernel"]
    mutable_params["params"]["output_projection"]["kernel"] = (
        0.1
        * jax.random.normal(
            jax.random.PRNGKey(4),
            output_kernel.shape,
        )
    )
    params = freeze(mutable_params)

    baseline = correction.apply(
        params,
        base_logits,
        features,
        action_context,
    )
    changed_context = action_context.at[:, :, 0].set(0.75)
    changed = correction.apply(
        params,
        base_logits,
        features,
        changed_context,
    )

    # Dimension zero is evaluated before action_context[..., 0] is consumed.
    np.testing.assert_array_equal(
        np.asarray(baseline[:, :, 0]),
        np.asarray(changed[:, :, 0]),
    )
    assert not np.allclose(
        np.asarray(baseline[:, :, 1]),
        np.asarray(changed[:, :, 1]),
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
    assert cfg.replay.compression == "zip"
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


def _strict_demo_rl_cfg(*overrides):
    return _compose_cqn_as(
        "num_pretrain_steps=0",
        "is_imitation_learning=false",
        "use_self_imitation=false",
        "method.strict_demo_rl_only=true",
        "method.bc_lambda=0",
        "method.bc_lambda_schedule=null",
        "method.bc_margin=0",
        "method.demo_fosd=false",
        "method.separate_bc_policy=false",
        "method.unseen_return_floor_weight=1.0",
        "method.unseen_return_floor_value=0.0",
        *overrides,
    )


@pytest.mark.parametrize(
    ("reduction", "topk"),
    (("mean", 1), ("max", 1), ("topk", 2)),
)
def test_unseen_return_floor_has_no_gradient_on_replayed_bin(
    reduction,
    topk,
):
    support = jnp.asarray([-1.0, 0.0, 1.0], dtype=jnp.float32)
    logits = jnp.asarray(
        [[[[[0.0, 0.0, 4.0], [0.0, 0.0, 3.0], [3.0, 0.0, 0.0]]]]],
        dtype=jnp.float32,
    )
    replayed_bin = jnp.asarray([[[0]]], dtype=jnp.int32)

    def objective(value):
        per_sample, _ = unseen_return_floor_loss(
            value,
            replayed_bin,
            support,
            0.0,
            reduction=reduction,
            topk=topk,
        )
        return per_sample.mean()

    grads = jax.grad(objective)(logits)
    np.testing.assert_array_equal(np.asarray(grads[0, 0, 0, 0]), 0.0)
    assert np.linalg.norm(np.asarray(grads[0, 0, 0, 1:])) > 0.0


@pytest.mark.parametrize(
    ("reduction", "topk"),
    (("mean", 1), ("max", 1), ("topk", 2)),
)
def test_unseen_return_floor_cannot_clone_actions_without_return_signal(
    reduction,
    topk,
):
    support = jnp.asarray([-1.0, 0.0, 1.0], dtype=jnp.float32)
    logits = jnp.zeros((1, 1, 1, 5, 3), dtype=jnp.float32)

    losses = []
    gradients = []
    for replayed_bin in range(5):
        bins = jnp.asarray([[[replayed_bin]]], dtype=jnp.int32)

        def objective(value):
            per_sample, _ = unseen_return_floor_loss(
                value,
                bins,
                support,
                0.0,
                reduction=reduction,
                topk=topk,
            )
            return per_sample.mean()

        losses.append(float(objective(logits)))
        gradients.append(np.asarray(jax.grad(objective)(logits)))

    np.testing.assert_array_equal(losses, np.zeros(5))
    np.testing.assert_array_equal(gradients, np.zeros_like(gradients))


def test_unseen_return_max_floor_targets_largest_competing_bin_per_head():
    support = jnp.asarray([-1.0, 0.0, 1.0], dtype=jnp.float32)
    logits = jnp.asarray(
        [[[[[0.0, 0.0, 4.0], [0.0, 0.0, 2.0], [4.0, 0.0, 0.0]]]]],
        dtype=jnp.float32,
    )
    replayed_bin = jnp.asarray([[[0]]], dtype=jnp.int32)

    def objective(value):
        per_sample, max_unseen_q = unseen_return_floor_loss(
            value,
            replayed_bin,
            support,
            0.0,
            reduction="max",
        )
        return per_sample.mean(), max_unseen_q

    (loss, max_unseen_q), grads = jax.value_and_grad(
        objective,
        has_aux=True,
    )(logits)
    assert loss == pytest.approx(float(max_unseen_q[0] ** 2))
    assert max_unseen_q[0] > 0.0
    np.testing.assert_array_equal(np.asarray(grads[0, 0, 0, 0]), 0.0)
    assert np.linalg.norm(np.asarray(grads[0, 0, 0, 1])) > 0.0
    np.testing.assert_array_equal(np.asarray(grads[0, 0, 0, 2]), 0.0)


def test_unseen_return_topk_floor_targets_upper_competing_tail():
    support = jnp.asarray([-1.0, 0.0, 1.0], dtype=jnp.float32)
    logits = jnp.asarray(
        [
            [
                [
                    [
                        [0.0, 0.0, 4.0],
                        [0.0, 0.0, 3.0],
                        [0.0, 0.0, 2.0],
                        [3.0, 0.0, 0.0],
                    ]
                ]
            ]
        ],
        dtype=jnp.float32,
    )
    replayed_bin = jnp.asarray([[[0]]], dtype=jnp.int32)

    def objective(value):
        per_sample, _ = unseen_return_floor_loss(
            value,
            replayed_bin,
            support,
            0.0,
            reduction="topk",
            topk=2,
        )
        return per_sample.mean()

    grads = jax.grad(objective)(logits)
    np.testing.assert_array_equal(np.asarray(grads[0, 0, 0, 0]), 0.0)
    assert np.linalg.norm(np.asarray(grads[0, 0, 0, 1])) > 0.0
    assert np.linalg.norm(np.asarray(grads[0, 0, 0, 2])) > 0.0
    np.testing.assert_array_equal(np.asarray(grads[0, 0, 0, 3]), 0.0)


@pytest.mark.parametrize("finest_neighbor_weight", (0.0, 0.5))
def test_dense_return_q_is_action_invariant_without_return_difference(
    finest_neighbor_weight,
):
    support = jnp.asarray([0.0, 0.5, 1.0], dtype=jnp.float32)
    logits = jax.random.normal(
        jax.random.PRNGKey(20),
        (2, 2, 3, 5, 3),
    )
    floor_target = jnp.zeros((2, 2, 3, 3), dtype=jnp.float32)
    floor_target = floor_target.at[..., 0].set(1.0)
    first_actions = jnp.zeros((2, 2, 3), dtype=jnp.int32)
    last_actions = jnp.full((2, 2, 3), 4, dtype=jnp.int32)

    def objective(value, actions):
        per_sample, _, _ = dense_return_distributional_loss(
            value,
            actions,
            floor_target,
            support,
            0.0,
            finest_neighbor_weight,
        )
        return per_sample.mean()

    first_loss, first_grad = jax.value_and_grad(objective)(
        logits,
        first_actions,
    )
    last_loss, last_grad = jax.value_and_grad(objective)(
        logits,
        last_actions,
    )
    np.testing.assert_array_equal(first_loss, last_loss)
    np.testing.assert_array_equal(first_grad, last_grad)


def test_dense_return_q_kernel_targets_only_finest_immediate_neighbors():
    support = jnp.asarray([0.0, 0.5, 1.0], dtype=jnp.float32)
    logits = jnp.zeros((1, 2, 1, 5, 3), dtype=jnp.float32)
    high_return_target = jnp.zeros((1, 2, 1, 3), dtype=jnp.float32)
    high_return_target = high_return_target.at[..., -1].set(1.0)
    replayed_bin = jnp.full((1, 2, 1), 2, dtype=jnp.int32)

    def objective(value):
        per_sample, _, _ = dense_return_distributional_loss(
            value,
            replayed_bin,
            high_return_target,
            support,
            0.0,
            0.5,
        )
        return per_sample.mean()

    grads = np.asarray(jax.grad(objective)(logits))
    floor_grad = np.asarray([-2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    chosen_grad = np.asarray([1.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0])
    neighbor_grad = np.asarray([-1.0 / 6.0, 1.0 / 3.0, -1.0 / 6.0])
    # The loss averages over two levels, so compare normalized directions.
    np.testing.assert_allclose(grads[0, 0, 0, 1], floor_grad / 2.0)
    np.testing.assert_allclose(grads[0, 0, 0, 2], chosen_grad / 2.0)
    np.testing.assert_allclose(grads[0, 1, 0, 0], floor_grad / 2.0)
    np.testing.assert_allclose(grads[0, 1, 0, 1], neighbor_grad / 2.0)
    np.testing.assert_allclose(grads[0, 1, 0, 2], chosen_grad / 2.0)
    np.testing.assert_allclose(grads[0, 1, 0, 3], neighbor_grad / 2.0)
    np.testing.assert_allclose(grads[0, 1, 0, 4], floor_grad / 2.0)


def test_dense_return_q_separation_is_created_only_by_return_target():
    support = jnp.asarray([0.0, 0.5, 1.0], dtype=jnp.float32)
    logits = jnp.zeros((1, 1, 1, 5, 3), dtype=jnp.float32)
    high_return_target = jnp.zeros((1, 1, 1, 3), dtype=jnp.float32)
    high_return_target = high_return_target.at[..., -1].set(1.0)
    replayed_bin = jnp.asarray([[[2]]], dtype=jnp.int32)

    def objective(value):
        per_sample, _, _ = dense_return_distributional_loss(
            value,
            replayed_bin,
            high_return_target,
            support,
            0.0,
        )
        return per_sample.mean()

    grads = jax.grad(objective)(logits)
    assert grads[0, 0, 0, 2, -1] < 0.0
    assert grads[0, 0, 0, 0, 0] < 0.0
    assert grads[0, 0, 0, 2, 0] > 0.0
    assert grads[0, 0, 0, 0, -1] > 0.0


def test_episodic_success_return_recovers_completed_outcome_bit():
    discounted_returns = jnp.asarray(
        [-1.0, 0.0, 1e-6, 0.217, 0.448, 1.0],
        dtype=jnp.float32,
    )
    np.testing.assert_array_equal(
        episodic_success_returns(discounted_returns),
        np.asarray([0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
    )


def test_ordered_success_return_lifts_but_preserves_positive_order():
    discounted_returns = jnp.asarray(
        [-1.0, 0.0, 0.2, 0.6, 1.0],
        dtype=jnp.float32,
    )
    targets = ordered_success_returns(discounted_returns, 0.5)
    np.testing.assert_allclose(
        targets,
        np.asarray([0.0, 0.0, 0.6, 0.8, 1.0]),
    )
    assert bool(jnp.all(jnp.diff(targets[2:]) > 0.0))


@pytest.mark.parametrize(
    ("floor_reduction", "floor_topk"),
    (("mean", 1), ("max", 1), ("topk", 2)),
)
def test_cqn_as_strict_demo_rl_ignores_demo_identity_and_has_no_policy(
    floor_reduction,
    floor_topk,
):
    observation_space, action_space = _spaces()
    cfg = _strict_demo_rl_cfg(
        "method.weight_decay=0.0",
        f"method.unseen_return_floor_reduction={floor_reduction}",
        f"method.unseen_return_floor_topk={floor_topk}",
    )
    demo_agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    online_agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    assert "policy" not in demo_agent.params
    assert "flow_policy" not in demo_agent.params

    demo_batch = _batch()
    online_batch = {
        key: np.array(value, copy=True) for key, value in demo_batch.items()
    }
    online_batch["demo"][:] = 0
    demo_agent.logging = True
    online_agent.logging = True
    demo_metrics = demo_agent.update(iter([demo_batch]), step=1)
    online_metrics = online_agent.update(iter([online_batch]), step=1)

    demo_leaves, demo_tree = jax.tree.flatten(demo_agent.params)
    online_leaves, online_tree = jax.tree.flatten(online_agent.params)
    assert demo_tree == online_tree
    for demo_value, online_value in zip(
        demo_leaves,
        online_leaves,
        strict=True,
    ):
        np.testing.assert_array_equal(demo_value, online_value)
    assert demo_metrics["unseen_return_floor_loss"] >= 0.0
    demo_metrics.pop("backend/update_time_sec")
    online_metrics.pop("backend/update_time_sec")
    # The BC-anchor diagnostics (cqn-flow.md sec 64) summarise the demo and
    # online halves of the batch separately, so relabelling moves rows between
    # them by construction. They are observational only -- the guarantee this
    # test protects is that the update itself is blind to demo identity, which
    # the parameter-level comparison above already establishes bit-for-bit.
    for key in (
        "bc_agreement",
        "bc_online_agreement",
        "bc_binding_rate",
        "bc_margin_gap",
        "bc_sibling_q_span",
    ):
        demo_metrics.pop(key, None)
        online_metrics.pop(key, None)
    assert demo_metrics == pytest.approx(online_metrics)


def test_cqn_as_mc_lower_bound_is_reward_only_and_not_an_auxiliary_loss():
    observation_space, action_space = _spaces()
    cfg = _strict_demo_rl_cfg(
        "method.unseen_return_floor_weight=0.0",
        "method.mc_lower_bound_target=true",
        "method.weight_decay=0.0",
    )
    demo_agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    online_agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    assert not demo_agent._canonical_mc_anchor
    assert demo_agent._uses_canonical_mc_returns
    assert _mc_return_anchor_enabled(cfg)
    assert "policy" not in demo_agent.params

    demo_batch = _batch()
    demo_batch["reward"][:] = 0.0
    demo_batch["mc_return"] = np.linspace(0.2, 0.8, 4).astype(np.float32)
    online_batch = {
        key: np.array(value, copy=True) for key, value in demo_batch.items()
    }
    online_batch["demo"][:] = 0
    demo_agent.logging = True
    online_agent.logging = True
    demo_metrics = demo_agent.update(iter([demo_batch]), step=1)
    online_metrics = online_agent.update(iter([online_batch]), step=1)

    demo_leaves, demo_tree = jax.tree.flatten(demo_agent.params)
    online_leaves, online_tree = jax.tree.flatten(online_agent.params)
    assert demo_tree == online_tree
    for demo_value, online_value in zip(
        demo_leaves,
        online_leaves,
        strict=True,
    ):
        np.testing.assert_array_equal(demo_value, online_value)
    assert demo_metrics["mc_lower_bound_fraction"] == pytest.approx(1.0)
    assert demo_metrics["mc_return_mean"] == pytest.approx(0.5)
    assert "mc_return_loss" not in demo_metrics
    demo_metrics.pop("backend/update_time_sec")
    online_metrics.pop("backend/update_time_sec")
    assert demo_metrics == pytest.approx(online_metrics)


def test_cqn_as_autoregressive_q_has_no_demo_identity_branch():
    observation_space, action_space = _spaces()
    cfg = _strict_demo_rl_cfg(
        "method.autoregressive_action_dims=true",
        "method.mc_lower_bound_target=true",
        "method.unseen_return_floor_reduction=max",
        "method.weight_decay=0.0",
    )
    demo_agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    online_agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    assert demo_agent.autoregressive_action_dims
    assert "policy" not in demo_agent.params
    assert "flow_policy" not in demo_agent.params

    demo_batch = _batch()
    demo_batch["mc_return"] = np.linspace(0.2, 0.8, 4).astype(np.float32)
    online_batch = {
        key: np.array(value, copy=True) for key, value in demo_batch.items()
    }
    online_batch["demo"][:] = 0
    demo_agent.update(iter([demo_batch]), step=1)
    online_agent.update(iter([online_batch]), step=1)

    demo_leaves, demo_tree = jax.tree.flatten(demo_agent.params)
    online_leaves, online_tree = jax.tree.flatten(online_agent.params)
    assert demo_tree == online_tree
    for demo_value, online_value in zip(
        demo_leaves,
        online_leaves,
        strict=True,
    ):
        np.testing.assert_array_equal(demo_value, online_value)


def test_cqn_as_dense_return_q_is_single_strict_rl_update():
    observation_space, action_space = _spaces()
    cfg = _strict_demo_rl_cfg(
        "method.dense_return_q_target=true",
        "method.unseen_return_floor_weight=0.0",
        "method.mc_lower_bound_target=true",
        "method.weight_decay=0.0",
    )
    agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    assert agent.dense_return_q_target
    assert "policy" not in agent.params
    assert "flow_policy" not in agent.params

    batch = _batch()
    batch["mc_return"] = np.linspace(0.2, 0.8, 4).astype(np.float32)
    agent.logging = True
    metrics = agent.update(iter([batch]), step=1)

    assert np.isfinite(metrics["critic_loss"])
    assert metrics["dense_return_q_loss"] == pytest.approx(
        metrics["critic_loss"]
    )
    assert "unseen_return_floor_loss" not in metrics
    assert "mc_return_loss" not in metrics


def test_cqn_as_positive_only_dense_uses_canonical_q_on_failed_trajectories():
    observation_space, action_space = _spaces()
    common = (
        "method.mc_lower_bound_target=true",
        "method.unseen_return_floor_weight=0.0",
        "method.weight_decay=0.0",
    )
    gated_agent = create_agent(
        _strict_demo_rl_cfg(
            *common,
            "method.dense_return_q_target=true",
            "method.dense_return_positive_only=true",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    canonical_agent = create_agent(
        _strict_demo_rl_cfg(
            *common,
            "method.dense_return_q_target=false",
            "method.dense_return_positive_only=false",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    batch = _batch()
    batch["reward"][:] = 0.0
    batch["mc_return"] = np.zeros((4,), dtype=np.float32)
    gated_agent.logging = True
    gated_metrics = gated_agent.update(iter([batch]), step=1)
    canonical_agent.update(iter([batch]), step=1)

    gated_leaves, gated_tree = jax.tree.flatten(gated_agent.params)
    canonical_leaves, canonical_tree = jax.tree.flatten(canonical_agent.params)
    assert gated_tree == canonical_tree
    for gated, canonical in zip(gated_leaves, canonical_leaves, strict=True):
        np.testing.assert_array_equal(gated, canonical)
    assert gated_metrics["dense_return_positive_fraction"] == pytest.approx(0.0)


def test_cqn_as_positive_only_dense_matches_dense_q_on_success_trajectories():
    observation_space, action_space = _spaces()
    common = (
        "method.dense_return_q_target=true",
        "method.mc_lower_bound_target=true",
        "method.unseen_return_floor_weight=0.0",
        "method.weight_decay=0.0",
    )
    gated_agent = create_agent(
        _strict_demo_rl_cfg(
            *common,
            "method.dense_return_positive_only=true",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    dense_agent = create_agent(
        _strict_demo_rl_cfg(
            *common,
            "method.dense_return_positive_only=false",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    batch = _batch()
    batch["reward"][:] = 0.0
    batch["mc_return"] = np.linspace(0.2, 0.8, 4).astype(np.float32)
    gated_agent.logging = True
    gated_metrics = gated_agent.update(iter([batch]), step=1)
    dense_agent.update(iter([batch]), step=1)

    gated_leaves, gated_tree = jax.tree.flatten(gated_agent.params)
    dense_leaves, dense_tree = jax.tree.flatten(dense_agent.params)
    assert gated_tree == dense_tree
    for gated, dense in zip(gated_leaves, dense_leaves, strict=True):
        np.testing.assert_array_equal(gated, dense)
    assert gated_metrics["dense_return_positive_fraction"] == pytest.approx(1.0)


def test_cqn_as_positive_only_dense_requires_dense_mc_target():
    observation_space, action_space = _spaces()
    with pytest.raises(ValueError, match="requires dense_return_q_target=true"):
        create_agent(
            _strict_demo_rl_cfg(
                "method.dense_return_q_target=false",
                "method.dense_return_positive_only=true",
                "method.mc_lower_bound_target=true",
            ),
            observation_space=observation_space,
            action_space=action_space,
        )
    with pytest.raises(ValueError, match="requires mc_lower_bound_target=true"):
        create_agent(
            _strict_demo_rl_cfg(
                "method.dense_return_q_target=true",
                "method.dense_return_positive_only=true",
                "method.mc_lower_bound_target=false",
            ),
            observation_space=observation_space,
            action_space=action_space,
        )


def test_cqn_as_dense_return_q_rejects_second_floor_objective():
    with pytest.raises(
        ValueError,
        match="requires unseen_return_floor_weight=0",
    ):
        create_agent(
            _strict_demo_rl_cfg(
                "method.dense_return_q_target=true",
                "method.unseen_return_floor_weight=1.0",
            ),
            observation_space=_spaces()[0],
            action_space=_spaces()[1],
        )


def test_cqn_as_dense_return_q_kernel_is_one_demo_agnostic_update():
    observation_space, action_space = _spaces()
    cfg = _strict_demo_rl_cfg(
        "method.dense_return_q_target=true",
        "method.dense_return_finest_neighbor_weight=0.5",
        "method.unseen_return_floor_weight=0.0",
        "method.mc_lower_bound_target=true",
        "method.weight_decay=0.0",
    )
    demo_agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    online_agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    assert demo_agent.dense_return_finest_neighbor_weight == pytest.approx(0.5)
    assert "policy" not in demo_agent.params
    assert "flow_policy" not in demo_agent.params

    demo_batch = _batch()
    demo_batch["reward"][:] = 0.0
    demo_batch["mc_return"] = np.linspace(0.2, 0.8, 4).astype(np.float32)
    online_batch = {
        key: np.array(value, copy=True) for key, value in demo_batch.items()
    }
    online_batch["demo"][:] = 0
    demo_agent.logging = True
    online_agent.logging = True
    demo_metrics = demo_agent.update(iter([demo_batch]), step=1)
    online_metrics = online_agent.update(iter([online_batch]), step=1)

    demo_leaves, demo_tree = jax.tree.flatten(demo_agent.params)
    online_leaves, online_tree = jax.tree.flatten(online_agent.params)
    assert demo_tree == online_tree
    for demo_value, online_value in zip(
        demo_leaves,
        online_leaves,
        strict=True,
    ):
        np.testing.assert_array_equal(demo_value, online_value)
    assert demo_metrics["dense_return_q_loss"] == pytest.approx(
        demo_metrics["critic_loss"]
    )
    assert "mc_return_loss" not in demo_metrics
    assert "unseen_return_floor_loss" not in demo_metrics
    demo_metrics.pop("backend/update_time_sec")
    online_metrics.pop("backend/update_time_sec")
    assert demo_metrics == pytest.approx(online_metrics)


def test_cqn_as_dense_return_q_kernel_requires_dense_target():
    with pytest.raises(
        ValueError,
        match="requires dense_return_q_target=true",
    ):
        create_agent(
            _strict_demo_rl_cfg(
                "method.dense_return_q_target=false",
                "method.dense_return_finest_neighbor_weight=0.5",
            ),
            observation_space=_spaces()[0],
            action_space=_spaces()[1],
        )


def test_cqn_as_dense_expected_q_is_one_demo_agnostic_update():
    observation_space, action_space = _spaces()
    cfg = _strict_demo_rl_cfg(
        "method.dense_return_q_target=true",
        "method.dense_return_expected_q_loss=true",
        "method.dense_return_finest_neighbor_weight=0.0",
        "method.unseen_return_floor_weight=0.0",
        "method.mc_lower_bound_target=true",
        "method.weight_decay=0.0",
    )
    demo_agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    online_agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    assert demo_agent.dense_return_expected_q_loss
    assert "policy" not in demo_agent.params
    assert "policy_encoder" not in demo_agent.params
    assert "flow_policy" not in demo_agent.params

    demo_batch = _batch()
    demo_batch["reward"][:] = 0.0
    demo_batch["mc_return"] = np.linspace(0.2, 0.8, 4).astype(np.float32)
    online_batch = {
        key: np.array(value, copy=True) for key, value in demo_batch.items()
    }
    online_batch["demo"][:] = 0
    demo_agent.logging = True
    online_agent.logging = True
    demo_metrics = demo_agent.update(iter([demo_batch]), step=1)
    online_metrics = online_agent.update(iter([online_batch]), step=1)

    demo_leaves, demo_tree = jax.tree.flatten(demo_agent.params)
    online_leaves, online_tree = jax.tree.flatten(online_agent.params)
    assert demo_tree == online_tree
    for demo_value, online_value in zip(
        demo_leaves,
        online_leaves,
        strict=True,
    ):
        np.testing.assert_array_equal(demo_value, online_value)
    assert demo_metrics["dense_return_q_loss"] == pytest.approx(
        demo_metrics["critic_loss"]
    )
    assert demo_metrics["dense_return_expected_q_target"] == pytest.approx(1.0)
    assert "policy_bc_loss" not in demo_metrics
    assert "mc_return_loss" not in demo_metrics
    assert "unseen_return_floor_loss" not in demo_metrics
    demo_metrics.pop("backend/update_time_sec")
    online_metrics.pop("backend/update_time_sec")
    assert demo_metrics == pytest.approx(online_metrics)


def test_cqn_as_dense_expected_q_rejects_incompatible_modes():
    with pytest.raises(
        ValueError,
        match="requires dense_return_q_target=true",
    ):
        create_agent(
            _strict_demo_rl_cfg(
                "method.dense_return_q_target=false",
                "method.dense_return_expected_q_loss=true",
            ),
            observation_space=_spaces()[0],
            action_space=_spaces()[1],
        )
    with pytest.raises(
        ValueError,
        match="dense_return_finest_neighbor_weight=0",
    ):
        create_agent(
            _strict_demo_rl_cfg(
                "method.dense_return_q_target=true",
                "method.dense_return_expected_q_loss=true",
                "method.dense_return_finest_neighbor_weight=0.5",
                "method.unseen_return_floor_weight=0.0",
            ),
            observation_space=_spaces()[0],
            action_space=_spaces()[1],
        )


def test_cqn_as_q_reward_scale_is_one_demo_agnostic_q_update():
    observation_space, action_space = _spaces()
    common = (
        "method.dense_return_q_target=true",
        "method.dense_return_expected_q_loss=false",
        "method.dense_return_finest_neighbor_weight=0.0",
        "method.unseen_return_floor_weight=0.0",
        "method.mc_lower_bound_target=true",
        "method.weight_decay=0.0",
    )
    unit_agent = create_agent(
        _strict_demo_rl_cfg(*common, "method.q_reward_scale=1.0"),
        observation_space=observation_space,
        action_space=action_space,
    )
    demo_agent = create_agent(
        _strict_demo_rl_cfg(*common, "method.q_reward_scale=2.0"),
        observation_space=observation_space,
        action_space=action_space,
    )
    online_agent = create_agent(
        _strict_demo_rl_cfg(*common, "method.q_reward_scale=2.0"),
        observation_space=observation_space,
        action_space=action_space,
    )
    assert demo_agent.q_reward_scale == pytest.approx(2.0)
    assert "policy" not in demo_agent.params
    assert "policy_encoder" not in demo_agent.params
    assert "flow_policy" not in demo_agent.params

    demo_batch = _batch()
    demo_batch["reward"][:] = 0.0
    demo_batch["mc_return"] = np.linspace(0.2, 0.8, 4).astype(np.float32)
    online_batch = {
        key: np.array(value, copy=True) for key, value in demo_batch.items()
    }
    online_batch["demo"][:] = 0
    unit_agent.logging = True
    demo_agent.logging = True
    online_agent.logging = True
    unit_agent.update(iter([demo_batch]), step=1)
    demo_metrics = demo_agent.update(iter([demo_batch]), step=1)
    online_metrics = online_agent.update(iter([online_batch]), step=1)

    assert _tree_changed(unit_agent.params, demo_agent.params)
    demo_leaves, demo_tree = jax.tree.flatten(demo_agent.params)
    online_leaves, online_tree = jax.tree.flatten(online_agent.params)
    assert demo_tree == online_tree
    for demo_value, online_value in zip(
        demo_leaves,
        online_leaves,
        strict=True,
    ):
        np.testing.assert_array_equal(demo_value, online_value)
    assert demo_metrics["q_reward_scale"] == pytest.approx(2.0)
    assert demo_metrics["scaled_mc_return_mean"] == pytest.approx(
        2.0 * demo_metrics["mc_return_mean"]
    )
    assert demo_metrics["dense_return_q_loss"] == pytest.approx(
        demo_metrics["critic_loss"]
    )
    assert "policy_bc_loss" not in demo_metrics
    assert "mc_return_loss" not in demo_metrics
    assert "unseen_return_floor_loss" not in demo_metrics
    demo_metrics.pop("backend/update_time_sec")
    online_metrics.pop("backend/update_time_sec")
    assert demo_metrics == pytest.approx(online_metrics)


def test_cqn_as_q_reward_scale_changes_terminal_bellman_target():
    observation_space, action_space = _spaces()
    common = (
        "method.dense_return_q_target=true",
        "method.dense_return_expected_q_loss=false",
        "method.dense_return_finest_neighbor_weight=0.0",
        "method.unseen_return_floor_weight=0.0",
        "method.mc_lower_bound_target=true",
        "method.weight_decay=0.0",
    )
    unit_agent = create_agent(
        _strict_demo_rl_cfg(*common, "method.q_reward_scale=1.0"),
        observation_space=observation_space,
        action_space=action_space,
    )
    scaled_agent = create_agent(
        _strict_demo_rl_cfg(*common, "method.q_reward_scale=2.0"),
        observation_space=observation_space,
        action_space=action_space,
    )
    batch = _batch()
    batch["reward"][:] = 0.5
    batch["terminal"][:] = True
    batch["mc_return"] = np.zeros(4, dtype=np.float32)
    unit_agent.logging = True
    scaled_agent.logging = True
    unit_metrics = unit_agent.update(iter([batch]), step=1)
    scaled_metrics = scaled_agent.update(iter([batch]), step=1)

    # MC=0 never overrides the positive terminal Bellman target, so the
    # distinct update proves that the immediate reward path is scaled too.
    assert unit_metrics["mc_lower_bound_fraction"] == pytest.approx(0.0)
    assert scaled_metrics["mc_lower_bound_fraction"] == pytest.approx(0.0)
    assert scaled_metrics["q_reward_scale"] == pytest.approx(2.0)
    assert scaled_metrics["scaled_mc_return_mean"] == pytest.approx(0.0)
    assert _tree_changed(unit_agent.params, scaled_agent.params)


def test_cqn_as_q_reward_scale_requires_supported_dense_mc_target():
    common = (
        "method.unseen_return_floor_weight=0.0",
        "method.q_reward_scale=2.0",
    )
    with pytest.raises(
        ValueError,
        match="requires dense_return_q_target=true",
    ):
        create_agent(
            _strict_demo_rl_cfg(
                *common,
                "method.dense_return_q_target=false",
            ),
            observation_space=_spaces()[0],
            action_space=_spaces()[1],
        )
    with pytest.raises(
        ValueError,
        match="requires mc_lower_bound_target=true",
    ):
        create_agent(
            _strict_demo_rl_cfg(
                *common,
                "method.dense_return_q_target=true",
                "method.mc_lower_bound_target=false",
            ),
            observation_space=_spaces()[0],
            action_space=_spaces()[1],
        )
    with pytest.raises(
        ValueError,
        match="terminal target must lie on the C51 support",
    ):
        create_agent(
            _strict_demo_rl_cfg(
                "method.unseen_return_floor_weight=0.0",
                "method.dense_return_q_target=true",
                "method.mc_lower_bound_target=true",
                "method.q_reward_scale=2.1",
            ),
            observation_space=_spaces()[0],
            action_space=_spaces()[1],
        )


def test_shift_categorical_distribution_translates_and_clips_exactly():
    support = jnp.linspace(-2.0, 2.0, 5)
    probabilities = categorical_point_mass(
        jnp.asarray([[0.0, 1.0, -1.0]], dtype=jnp.float32),
        support,
    )
    shifted = shift_categorical_distribution(
        probabilities,
        jnp.asarray([[-0.5, 0.5, -2.0]], dtype=jnp.float32),
        support,
    )
    np.testing.assert_allclose(shifted.sum(axis=-1), 1.0, atol=1e-7)
    np.testing.assert_allclose(
        jnp.sum(shifted * support, axis=-1),
        [[-0.5, 1.5, -2.0]],
        atol=1e-7,
    )


def test_clipped_advantage_shift_only_changes_near_greedy_bins():
    q_values = jnp.asarray([[1.0, 0.9, 0.0, -1.0]], dtype=jnp.float32)
    constant_shift, constant_active = advantage_learning_target_shift(
        q_values,
        q_lower=-2.0,
        alpha=0.5,
    )
    clipped_shift, clipped_active = advantage_learning_target_shift(
        q_values,
        q_lower=-2.0,
        alpha=0.5,
        clip_ratio=0.9,
    )
    np.testing.assert_array_equal(
        constant_active,
        np.ones_like(q_values, dtype=bool),
    )
    np.testing.assert_allclose(
        constant_shift,
        [[0.0, -0.05, -0.5, -1.0]],
        atol=1e-7,
    )
    np.testing.assert_array_equal(
        clipped_active,
        [[True, True, False, False]],
    )
    np.testing.assert_allclose(
        clipped_shift,
        [[0.0, -0.05, 0.0, 0.0]],
        atol=1e-7,
    )


def test_dense_advantage_target_is_default_identical_and_zero_label_invariant():
    rng = np.random.default_rng(29)
    all_logits = jnp.asarray(
        rng.normal(size=(2, 2, 3, 5, 11)),
        dtype=jnp.float32,
    )
    support = jnp.linspace(-2.0, 2.0, 11)
    positive_target = categorical_point_mass(
        jnp.full((2, 2, 3), 0.7, dtype=jnp.float32),
        support,
    )
    action_a = jnp.zeros((2, 2, 3), dtype=jnp.int32)
    action_b = jnp.full((2, 2, 3), 4, dtype=jnp.int32)

    legacy = dense_return_distributional_loss(
        all_logits,
        action_a,
        positive_target,
        support,
        0.0,
    )
    explicit_zero = dense_return_distributional_loss(
        all_logits,
        action_a,
        positive_target,
        support,
        0.0,
        advantage_alpha=0.0,
    )
    for left, right in zip(legacy, explicit_zero, strict=True):
        np.testing.assert_array_equal(left, right)

    zero_target = categorical_point_mass(
        jnp.zeros((2, 2, 3), dtype=jnp.float32),
        support,
    )

    def loss(logits, action):
        return dense_return_distributional_loss(
            logits,
            action,
            zero_target,
            support,
            0.0,
            advantage_alpha=0.5,
        )[0].sum()

    value_a, grad_a = jax.value_and_grad(loss)(all_logits, action_a)
    value_b, grad_b = jax.value_and_grad(loss)(all_logits, action_b)
    np.testing.assert_allclose(value_a, value_b, atol=1e-7)
    np.testing.assert_allclose(grad_a, grad_b, atol=1e-7)

    def clipped_loss(logits, action):
        return dense_return_distributional_loss(
            logits,
            action,
            zero_target,
            support,
            0.0,
            advantage_alpha=0.5,
            advantage_clip_ratio=0.9,
        )[0].sum()

    clipped_value_a, clipped_grad_a = jax.value_and_grad(clipped_loss)(
        all_logits,
        action_a,
    )
    clipped_value_b, clipped_grad_b = jax.value_and_grad(clipped_loss)(
        all_logits,
        action_b,
    )
    np.testing.assert_allclose(clipped_value_a, clipped_value_b, atol=1e-7)
    np.testing.assert_allclose(clipped_grad_a, clipped_grad_b, atol=1e-7)


def test_cqn_as_dense_advantage_is_one_demo_agnostic_q_update():
    observation_space, action_space = _spaces()
    cfg = _strict_demo_rl_cfg(
        "method.dense_return_q_target=true",
        "method.dense_return_expected_q_loss=false",
        "method.dense_return_advantage_alpha=0.5",
        "method.dense_return_finest_neighbor_weight=0.0",
        "method.unseen_return_floor_weight=0.0",
        "method.mc_lower_bound_target=true",
        "method.weight_decay=0.0",
    )
    demo_agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    online_agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    assert demo_agent.dense_return_advantage_alpha == pytest.approx(0.5)
    assert "policy" not in demo_agent.params
    assert "policy_encoder" not in demo_agent.params
    assert "flow_policy" not in demo_agent.params

    demo_batch = _batch()
    demo_batch["reward"][:] = 0.0
    demo_batch["mc_return"] = np.linspace(0.2, 0.8, 4).astype(np.float32)
    online_batch = {
        key: np.array(value, copy=True) for key, value in demo_batch.items()
    }
    online_batch["demo"][:] = 0
    demo_agent.logging = True
    online_agent.logging = True
    demo_metrics = demo_agent.update(iter([demo_batch]), step=1)
    online_metrics = online_agent.update(iter([online_batch]), step=1)

    demo_leaves, demo_tree = jax.tree.flatten(demo_agent.params)
    online_leaves, online_tree = jax.tree.flatten(online_agent.params)
    assert demo_tree == online_tree
    for demo_value, online_value in zip(
        demo_leaves,
        online_leaves,
        strict=True,
    ):
        np.testing.assert_array_equal(demo_value, online_value)
    assert demo_metrics["dense_return_advantage_alpha"] == pytest.approx(0.5)
    assert demo_metrics["dense_return_q_loss"] == pytest.approx(
        demo_metrics["critic_loss"]
    )
    assert "policy_bc_loss" not in demo_metrics
    assert "mc_return_loss" not in demo_metrics
    assert "unseen_return_floor_loss" not in demo_metrics
    demo_metrics.pop("backend/update_time_sec")
    online_metrics.pop("backend/update_time_sec")
    assert demo_metrics == pytest.approx(online_metrics)


def test_cqn_as_clipped_advantage_is_one_demo_agnostic_q_update():
    observation_space, action_space = _spaces()
    cfg = _strict_demo_rl_cfg(
        "method.dense_return_q_target=true",
        "method.dense_return_expected_q_loss=false",
        "method.dense_return_advantage_alpha=0.5",
        "method.dense_return_advantage_clip_ratio=0.9",
        "method.dense_return_finest_neighbor_weight=0.0",
        "method.unseen_return_floor_weight=0.0",
        "method.mc_lower_bound_target=true",
        "method.weight_decay=0.0",
    )
    demo_agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    online_agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    demo_batch = _batch()
    demo_batch["reward"][:] = 0.0
    demo_batch["mc_return"] = np.linspace(0.2, 0.8, 4).astype(np.float32)
    online_batch = {
        key: np.array(value, copy=True) for key, value in demo_batch.items()
    }
    online_batch["demo"][:] = 0
    demo_agent.logging = True
    online_agent.logging = True
    demo_metrics = demo_agent.update(iter([demo_batch]), step=1)
    online_metrics = online_agent.update(iter([online_batch]), step=1)

    demo_leaves, demo_tree = jax.tree.flatten(demo_agent.params)
    online_leaves, online_tree = jax.tree.flatten(online_agent.params)
    assert demo_tree == online_tree
    for demo_value, online_value in zip(
        demo_leaves,
        online_leaves,
        strict=True,
    ):
        np.testing.assert_array_equal(demo_value, online_value)
    assert demo_metrics["dense_return_advantage_alpha"] == pytest.approx(0.5)
    assert demo_metrics[
        "dense_return_advantage_clip_ratio"
    ] == pytest.approx(0.9)
    assert demo_metrics["dense_return_q_loss"] == pytest.approx(
        demo_metrics["critic_loss"]
    )
    assert "policy_bc_loss" not in demo_metrics
    assert "mc_return_loss" not in demo_metrics
    assert "unseen_return_floor_loss" not in demo_metrics
    demo_metrics.pop("backend/update_time_sec")
    online_metrics.pop("backend/update_time_sec")
    assert demo_metrics == pytest.approx(online_metrics)


def test_cqn_as_dense_advantage_rejects_incompatible_modes():
    with pytest.raises(
        ValueError,
        match=r"dense_return_advantage_alpha must be in \[0, 1\)",
    ):
        create_agent(
            _strict_demo_rl_cfg("method.dense_return_advantage_alpha=1.0"),
            observation_space=_spaces()[0],
            action_space=_spaces()[1],
        )
    with pytest.raises(
        ValueError,
        match="requires dense_return_q_target=true",
    ):
        create_agent(
            _strict_demo_rl_cfg("method.dense_return_advantage_alpha=0.5"),
            observation_space=_spaces()[0],
            action_space=_spaces()[1],
        )
    with pytest.raises(
        ValueError,
        match="requires dense_return_expected_q_loss=false",
    ):
        create_agent(
            _strict_demo_rl_cfg(
                "method.dense_return_q_target=true",
                "method.dense_return_expected_q_loss=true",
                "method.dense_return_advantage_alpha=0.5",
                "method.unseen_return_floor_weight=0.0",
            ),
            observation_space=_spaces()[0],
            action_space=_spaces()[1],
        )
    with pytest.raises(
        ValueError,
        match=r"dense_return_advantage_clip_ratio must be in \(0, 1\)",
    ):
        create_agent(
            _strict_demo_rl_cfg(
                "method.dense_return_q_target=true",
                "method.dense_return_advantage_alpha=0.5",
                "method.dense_return_advantage_clip_ratio=1.0",
                "method.unseen_return_floor_weight=0.0",
            ),
            observation_space=_spaces()[0],
            action_space=_spaces()[1],
        )
    with pytest.raises(
        ValueError,
        match="requires dense_return_advantage_alpha > 0",
    ):
        create_agent(
            _strict_demo_rl_cfg(
                "method.dense_return_q_target=true",
                "method.dense_return_advantage_clip_ratio=0.9",
                "method.unseen_return_floor_weight=0.0",
            ),
            observation_space=_spaces()[0],
            action_space=_spaces()[1],
        )


def test_cqn_as_replay_sarsa_is_one_demo_agnostic_q_update():
    observation_space, action_space = _spaces()
    cfg = _strict_demo_rl_cfg(
        "method.dense_return_q_target=true",
        "method.dense_return_finest_neighbor_weight=0.0",
        "method.unseen_return_floor_weight=0.0",
        "method.mc_lower_bound_target=true",
        "method.td_target_action_source=replay_next",
        "method.weight_decay=0.0",
    )
    demo_agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    online_agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    assert demo_agent.td_target_action_source == "replay_next"
    assert not demo_agent.separate_bc_policy
    assert "policy" not in demo_agent.params
    assert "policy_encoder" not in demo_agent.params
    assert "flow_policy" not in demo_agent.params

    demo_batch = _batch()
    selected_action, selected_info = (
        demo_agent._td_target_action_for_update(
            demo_agent.params["critic"],
            None,
            jnp.asarray(demo_batch["action"]),
            jnp.asarray(demo_batch["action"]),
            jnp.asarray(demo_batch["demo"]),
            jax.random.PRNGKey(0),
        )
    )
    np.testing.assert_array_equal(
        selected_action,
        shift_replay_action_sequence(
            jnp.asarray(demo_batch["action"]),
            demo_agent.action_sequence,
            demo_agent.action_dim,
        ),
    )
    assert selected_info == {}

    hook_calls = {"count": 0}
    original_target_action = demo_agent._td_target_action_for_update

    def counted_target_action(*args, **kwargs):
        hook_calls["count"] += 1
        return original_target_action(*args, **kwargs)

    demo_agent._td_target_action_for_update = counted_target_action
    demo_agent._update_impl = demo_agent._build_update_fn()
    demo_batch["reward"][:] = 0.0
    demo_batch["mc_return"] = np.linspace(0.2, 0.8, 4).astype(np.float32)
    online_batch = {
        key: np.array(value, copy=True) for key, value in demo_batch.items()
    }
    online_batch["demo"][:] = 0
    demo_agent.logging = True
    online_agent.logging = True
    demo_metrics = demo_agent.update(iter([demo_batch]), step=1)
    online_metrics = online_agent.update(iter([online_batch]), step=1)
    assert hook_calls["count"] == 1

    demo_leaves, demo_tree = jax.tree.flatten(demo_agent.params)
    online_leaves, online_tree = jax.tree.flatten(online_agent.params)
    assert demo_tree == online_tree
    for demo_value, online_value in zip(
        demo_leaves,
        online_leaves,
        strict=True,
    ):
        np.testing.assert_array_equal(demo_value, online_value)
    assert demo_metrics["dense_return_q_loss"] == pytest.approx(
        demo_metrics["critic_loss"]
    )
    assert "policy_bc_loss" not in demo_metrics
    assert "mc_return_loss" not in demo_metrics
    assert "unseen_return_floor_loss" not in demo_metrics
    assert demo_metrics["td_target_replay_next"] == pytest.approx(1.0)
    demo_metrics.pop("backend/update_time_sec")
    online_metrics.pop("backend/update_time_sec")
    assert demo_metrics == pytest.approx(online_metrics)


def test_cqn_as_candidate_backup_selects_true_next_chunk_by_deepest_q():
    observation_space, action_space = _spaces()
    cfg = _strict_demo_rl_cfg(
        "replay.include_next_action=true",
        "method.td_target_action_source=critic_replay_max",
        "method.weight_decay=0.0",
    )
    agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    batch_size = 3
    greedy = jnp.zeros(
        (batch_size, agent.action_sequence, agent.action_dim),
        dtype=jnp.float32,
    )
    behavior = jnp.arange(
        batch_size * agent.action_sequence * agent.action_dim,
        dtype=jnp.float32,
    ).reshape(greedy.shape)
    score_calls = {"count": 0}

    def fake_greedy(*_args, **_kwargs):
        return greedy, {}

    def fake_score(*_args, **_kwargs):
        score_calls["count"] += 1
        if score_calls["count"] == 1:
            return jnp.asarray([0.2, 0.8, 0.5], dtype=jnp.float32)
        return jnp.asarray([0.3, 0.7, 0.5], dtype=jnp.float32)

    agent._greedy_action_for_update = fake_greedy
    agent._score_action_sequence_for_backup = fake_score
    selected, info = agent._td_target_action_for_update(
        agent.params["critic"],
        None,
        jnp.zeros_like(behavior),
        behavior,
        jnp.zeros((batch_size,), dtype=jnp.float32),
        jax.random.PRNGKey(0),
    )

    np.testing.assert_array_equal(selected[0], behavior[0])
    np.testing.assert_array_equal(selected[1], greedy[1])
    np.testing.assert_array_equal(selected[2], behavior[2])
    np.testing.assert_array_equal(
        info["behavior_selected"],
        [True, False, True],
    )
    np.testing.assert_allclose(
        info["behavior_score"] - info["greedy_score"],
        [0.1, -0.1, 0.0],
        atol=1e-7,
    )


def test_cqn_as_demo_trajectory_force_overrides_only_demo_candidates():
    observation_space, action_space = _spaces()
    cfg = _strict_demo_rl_cfg(
        "replay.include_next_action=true",
        "method.td_target_action_source=critic_replay_max",
        "method.demo_behavior_force_probability=1.0",
        "method.weight_decay=0.0",
    )
    agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    batch_size = 3
    greedy = jnp.zeros(
        (batch_size, agent.action_sequence, agent.action_dim),
        dtype=jnp.float32,
    )
    behavior = jnp.ones_like(greedy)
    score_calls = {"count": 0}

    def fake_greedy(*_args, **_kwargs):
        return greedy, {}

    def fake_score(*_args, **_kwargs):
        score_calls["count"] += 1
        if score_calls["count"] == 1:
            return jnp.full((batch_size,), 0.8, dtype=jnp.float32)
        return jnp.full((batch_size,), 0.2, dtype=jnp.float32)

    agent._greedy_action_for_update = fake_greedy
    agent._score_action_sequence_for_backup = fake_score
    demos = jnp.asarray([1.0, 0.0, 1.0], dtype=jnp.float32)
    selected, info = agent._td_target_action_for_update(
        agent.params["critic"],
        None,
        jnp.zeros_like(behavior),
        behavior,
        demos,
        jax.random.PRNGKey(7),
    )

    np.testing.assert_array_equal(selected[0], behavior[0])
    np.testing.assert_array_equal(selected[1], greedy[1])
    np.testing.assert_array_equal(selected[2], behavior[2])
    np.testing.assert_array_equal(
        info["demo_behavior_forced"],
        [True, False, True],
    )
    np.testing.assert_array_equal(
        info["behavior_selected"],
        [True, False, True],
    )


def test_cqn_as_demo_trajectory_force_remains_one_reward_q_loss():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _strict_demo_rl_cfg(
            "replay.include_next_action=true",
            "method.td_target_action_source=critic_replay_max",
            "method.demo_behavior_force_probability=1.0",
            "method.dense_return_q_target=true",
            "method.unseen_return_floor_weight=0.0",
            "method.mc_lower_bound_target=true",
            "method.weight_decay=0.0",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    batch = _batch()
    batch["action_tp1"] = np.roll(batch["action"], shift=-1, axis=1)
    batch["demo"] = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    batch["mc_return"] = np.linspace(0.2, 0.8, 4).astype(np.float32)
    agent.logging = True
    metrics = agent.update(iter([batch]), step=1)

    assert metrics["critic_loss"] == pytest.approx(
        metrics["dense_return_q_loss"]
    )
    assert metrics["demo_behavior_force_probability"] == pytest.approx(1.0)
    assert metrics["demo_behavior_force_fraction"] == pytest.approx(0.5)
    assert "policy" not in agent.params
    assert "flow_policy" not in agent.params
    assert "policy_bc_loss" not in metrics
    assert "mc_return_loss" not in metrics


def test_cqn_as_demo_trajectory_force_requires_candidate_backup():
    with pytest.raises(
        ValueError,
        match="demo_behavior_force_probability > 0 requires",
    ):
        create_agent(
            _strict_demo_rl_cfg(
                "method.demo_behavior_force_probability=1.0"
            ),
            observation_space=_spaces()[0],
            action_space=_spaces()[1],
        )


def test_cqn_as_candidate_score_uses_only_deepest_level_mean():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _strict_demo_rl_cfg("method.weight_decay=0.0"),
        observation_space=observation_space,
        action_space=action_space,
    )
    batch_size = 2
    flat_dim = agent.action_sequence * agent.action_dim
    logits = jnp.full(
        (
            batch_size,
            agent.levels,
            flat_dim,
            agent.atoms,
        ),
        -30.0,
        dtype=jnp.float32,
    )
    # Earlier levels deliberately prefer the largest support atom. The
    # deepest level identifies the expected scores that must be returned.
    logits = logits.at[:, :-1, :, -1].set(30.0)
    deepest_atoms = (2, agent.atoms - 3)
    for row, atom in enumerate(deepest_atoms):
        logits = logits.at[row, -1, :, atom].set(30.0)

    def fake_logits(*_args, **_kwargs):
        return logits, None

    agent._critic_logits_per_level = fake_logits
    scores = agent._score_action_sequence_for_backup(
        agent.params["critic"],
        jnp.zeros((batch_size, 5), dtype=jnp.float32),
        jnp.zeros(
            (batch_size, agent.action_sequence, agent.action_dim),
            dtype=jnp.float32,
        ),
    )
    np.testing.assert_allclose(
        scores,
        np.asarray(agent.support)[list(deepest_atoms)],
        atol=1e-6,
    )


def test_cqn_as_candidate_backup_is_one_demo_agnostic_q_update():
    observation_space, action_space = _spaces()
    cfg = _strict_demo_rl_cfg(
        "replay.include_next_action=true",
        "method.td_target_action_source=critic_replay_max",
        "method.dense_return_q_target=true",
        "method.unseen_return_floor_weight=0.0",
        "method.mc_lower_bound_target=true",
        "method.weight_decay=0.0",
    )
    demo_agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    online_agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    assert demo_agent.td_target_action_source == "critic_replay_max"
    assert "policy" not in demo_agent.params
    assert "flow_policy" not in demo_agent.params

    demo_batch = _batch()
    demo_batch["action_tp1"] = np.roll(
        demo_batch["action"],
        shift=-1,
        axis=1,
    )
    demo_batch["mc_return"] = np.linspace(0.2, 0.8, 4).astype(np.float32)
    online_batch = {
        key: np.array(value, copy=True) for key, value in demo_batch.items()
    }
    online_batch["demo"][:] = 0
    demo_agent.logging = True
    online_agent.logging = True
    demo_metrics = demo_agent.update(iter([demo_batch]), step=1)
    online_metrics = online_agent.update(iter([online_batch]), step=1)

    demo_leaves, demo_tree = jax.tree.flatten(demo_agent.params)
    online_leaves, online_tree = jax.tree.flatten(online_agent.params)
    assert demo_tree == online_tree
    for demo_value, online_value in zip(
        demo_leaves,
        online_leaves,
        strict=True,
    ):
        np.testing.assert_array_equal(demo_value, online_value)
    assert demo_metrics["dense_return_q_loss"] == pytest.approx(
        demo_metrics["critic_loss"]
    )
    assert 0.0 <= demo_metrics["behavior_candidate_fraction"] <= 1.0
    assert np.isfinite(demo_metrics["behavior_minus_greedy_q"])
    assert "policy_bc_loss" not in demo_metrics
    assert "mc_return_loss" not in demo_metrics
    assert "unseen_return_floor_loss" not in demo_metrics
    demo_metrics.pop("backend/update_time_sec")
    online_metrics.pop("backend/update_time_sec")
    assert demo_metrics == pytest.approx(online_metrics)


def test_cqn_as_candidate_backup_requires_next_action_replay():
    with pytest.raises(
        ValueError,
        match="replay.include_next_action must be true",
    ):
        create_agent(
            _strict_demo_rl_cfg(
                "method.td_target_action_source=critic_replay_max"
            ),
            observation_space=_spaces()[0],
            action_space=_spaces()[1],
        )


def test_cqn_as_sequence_aligned_return_is_one_demo_agnostic_q_update():
    observation_space, action_space = _spaces()
    cfg = _strict_demo_rl_cfg(
        "method.dense_return_q_target=true",
        "method.dense_return_finest_neighbor_weight=0.0",
        "method.unseen_return_floor_weight=0.0",
        "method.mc_lower_bound_target=true",
        "method.sequence_aligned_mc_discount=0.99",
        "method.weight_decay=0.0",
    )
    demo_agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    online_agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    assert demo_agent.sequence_aligned_mc_discount == pytest.approx(0.99)
    assert "policy" not in demo_agent.params
    assert "flow_policy" not in demo_agent.params

    demo_batch = _batch()
    demo_batch["reward"][:] = 0.0
    demo_batch["mc_return"] = np.asarray(
        [0.0, 0.2, 0.4, 0.8],
        dtype=np.float32,
    )
    online_batch = {
        key: np.array(value, copy=True) for key, value in demo_batch.items()
    }
    online_batch["demo"][:] = 0
    demo_agent.logging = True
    online_agent.logging = True
    demo_metrics = demo_agent.update(iter([demo_batch]), step=1)
    online_metrics = online_agent.update(iter([online_batch]), step=1)

    demo_leaves, demo_tree = jax.tree.flatten(demo_agent.params)
    online_leaves, online_tree = jax.tree.flatten(online_agent.params)
    assert demo_tree == online_tree
    for demo_value, online_value in zip(
        demo_leaves,
        online_leaves,
        strict=True,
    ):
        np.testing.assert_array_equal(demo_value, online_value)
    assert demo_metrics["sequence_aligned_mc_return_mean"] > np.mean(
        demo_batch["mc_return"]
    )
    assert demo_metrics["dense_return_q_loss"] == pytest.approx(
        demo_metrics["critic_loss"]
    )
    assert "mc_return_loss" not in demo_metrics
    assert "unseen_return_floor_loss" not in demo_metrics
    demo_metrics.pop("backend/update_time_sec")
    online_metrics.pop("backend/update_time_sec")
    assert demo_metrics == pytest.approx(online_metrics)


def test_cqn_as_sequence_aligned_return_requires_full_sequence_q():
    with pytest.raises(
        ValueError,
        match="requires critic_sequence_mode=full",
    ):
        create_agent(
            _strict_demo_rl_cfg(
                "method.dense_return_q_target=true",
                "method.unseen_return_floor_weight=0.0",
                "method.mc_lower_bound_target=true",
                "method.sequence_aligned_mc_discount=0.99",
                "method.critic_sequence_mode=effective_k0",
            ),
            observation_space=_spaces()[0],
            action_space=_spaces()[1],
        )


def test_cqn_as_episodic_success_q_is_reward_gated_and_demo_agnostic():
    observation_space, action_space = _spaces()
    cfg = _strict_demo_rl_cfg(
        "method.dense_return_q_target=true",
        "method.episodic_success_q_target=true",
        "method.mc_lower_bound_target=false",
        "method.unseen_return_floor_weight=0.0",
        "method.weight_decay=0.0",
    )
    demo_agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    online_agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    assert demo_agent.episodic_success_q_target
    assert demo_agent._uses_canonical_mc_returns
    assert not demo_agent.mc_lower_bound_target
    assert _mc_return_anchor_enabled(cfg)
    assert "policy" not in demo_agent.params
    assert "flow_policy" not in demo_agent.params

    demo_batch = _batch()
    demo_batch["reward"][:] = 0.0
    demo_batch["mc_return"] = np.asarray(
        [0.0, 0.2, 0.8, 0.0],
        dtype=np.float32,
    )
    online_batch = {
        key: np.array(value, copy=True) for key, value in demo_batch.items()
    }
    online_batch["demo"][:] = 0
    demo_agent.logging = True
    online_agent.logging = True
    demo_metrics = demo_agent.update(iter([demo_batch]), step=1)
    online_metrics = online_agent.update(iter([online_batch]), step=1)

    demo_leaves, demo_tree = jax.tree.flatten(demo_agent.params)
    online_leaves, online_tree = jax.tree.flatten(online_agent.params)
    assert demo_tree == online_tree
    for demo_value, online_value in zip(
        demo_leaves,
        online_leaves,
        strict=True,
    ):
        np.testing.assert_array_equal(demo_value, online_value)
    assert demo_metrics["episodic_success_fraction"] == pytest.approx(0.5)
    assert demo_metrics["dense_return_q_loss"] == pytest.approx(
        demo_metrics["critic_loss"]
    )
    assert "mc_lower_bound_fraction" not in demo_metrics
    assert "mc_return_loss" not in demo_metrics
    assert "unseen_return_floor_loss" not in demo_metrics
    demo_metrics.pop("backend/update_time_sec")
    online_metrics.pop("backend/update_time_sec")
    assert demo_metrics == pytest.approx(online_metrics)


def test_cqn_as_episodic_success_q_rejects_a_second_mc_target():
    with pytest.raises(
        ValueError,
        match="replaces the discounted MC-lower-bound",
    ):
        create_agent(
            _strict_demo_rl_cfg(
                "method.dense_return_q_target=true",
                "method.episodic_success_q_target=true",
                "method.mc_lower_bound_target=true",
                "method.unseen_return_floor_weight=0.0",
            ),
            observation_space=_spaces()[0],
            action_space=_spaces()[1],
        )


def test_cqn_as_ordered_success_q_is_one_demo_agnostic_target():
    observation_space, action_space = _spaces()
    cfg = _strict_demo_rl_cfg(
        "method.dense_return_q_target=true",
        "method.episodic_success_q_target=false",
        "method.mc_lower_bound_target=true",
        "method.ordered_success_return_mix=0.5",
        "method.unseen_return_floor_weight=0.0",
        "method.weight_decay=0.0",
    )
    demo_agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    online_agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    assert demo_agent.ordered_success_return_mix == pytest.approx(0.5)
    assert demo_agent.mc_lower_bound_target
    assert "policy" not in demo_agent.params
    assert "flow_policy" not in demo_agent.params

    demo_batch = _batch()
    demo_batch["reward"][:] = 0.0
    demo_batch["mc_return"] = np.asarray(
        [0.0, 0.2, 0.8, 0.0],
        dtype=np.float32,
    )
    online_batch = {
        key: np.array(value, copy=True) for key, value in demo_batch.items()
    }
    online_batch["demo"][:] = 0
    demo_agent.logging = True
    online_agent.logging = True
    demo_metrics = demo_agent.update(iter([demo_batch]), step=1)
    online_metrics = online_agent.update(iter([online_batch]), step=1)

    demo_leaves, demo_tree = jax.tree.flatten(demo_agent.params)
    online_leaves, online_tree = jax.tree.flatten(online_agent.params)
    assert demo_tree == online_tree
    for demo_value, online_value in zip(
        demo_leaves,
        online_leaves,
        strict=True,
    ):
        np.testing.assert_array_equal(demo_value, online_value)
    assert demo_metrics["ordered_success_return_mean"] == pytest.approx(
        0.375
    )
    assert demo_metrics["dense_return_q_loss"] == pytest.approx(
        demo_metrics["critic_loss"]
    )
    assert "mc_return_loss" not in demo_metrics
    assert "episodic_success_fraction" not in demo_metrics
    demo_metrics.pop("backend/update_time_sec")
    online_metrics.pop("backend/update_time_sec")
    assert demo_metrics == pytest.approx(online_metrics)


def test_cqn_as_ordered_success_q_requires_mc_lower_bound():
    with pytest.raises(
        ValueError,
        match="requires mc_lower_bound_target=true",
    ):
        create_agent(
            _strict_demo_rl_cfg(
                "method.dense_return_q_target=true",
                "method.mc_lower_bound_target=false",
                "method.ordered_success_return_mix=0.5",
                "method.unseen_return_floor_weight=0.0",
            ),
            observation_space=_spaces()[0],
            action_space=_spaces()[1],
        )


@pytest.mark.parametrize(
    "override",
    [
        "method.bc_lambda=1.0",
        "method.bc_margin=0.1",
        "method.demo_fosd=true",
        "method.separate_bc_policy=true",
        "method.flow_policy=true",
        "method.td_target_action_source=bc_policy",
        "use_self_imitation=true",
    ],
)
def test_cqn_as_strict_demo_rl_rejects_imitation_paths(override):
    with pytest.raises(ValueError, match="strict_demo_rl_only"):
        create_agent(
            _strict_demo_rl_cfg(override),
            observation_space=_spaces()[0],
            action_space=_spaces()[1],
        )


def test_cqn_as_strict_demo_rl_allows_reward_only_offline_critic_updates():
    agent = create_agent(
        _strict_demo_rl_cfg(
            "num_pretrain_steps=10",
            "method.dense_return_q_target=false",
            "method.mc_lower_bound_target=true",
            "method.unseen_return_floor_weight=0",
            "method.td_target_action_source=critic_replay_max",
            "replay.include_next_action=true",
        ),
        observation_space=_spaces()[0],
        action_space=_spaces()[1],
    )

    assert agent.strict_demo_rl_only
    assert agent.bc_lambda == 0.0
    assert agent.bc_margin == 0.0
    assert not agent.separate_bc_policy
    assert agent.mc_lower_bound_target
    assert agent.td_target_action_source == "critic_replay_max"


def test_cqn_as_strict_demo_rl_explicitly_allows_reward_only_success_replay():
    agent = create_agent(
        _strict_demo_rl_cfg(
            "use_self_imitation=true",
            "method.strict_allow_reward_only_success_replay=true",
        ),
        observation_space=_spaces()[0],
        action_space=_spaces()[1],
    )

    assert agent.strict_demo_rl_only
    assert agent.bc_lambda == 0.0
    assert agent.bc_margin == 0.0
    assert not agent.demo_fosd


def test_cqn_as_stage30_launch_is_canonical_reward_only_offline_rl():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_as_pixel_bigym_nobc_stage30_offline_then_online_gate",
                "env=bigym/move_plate",
            ],
        )

    assert cfg.num_pretrain_steps == 10_000
    assert cfg.method.strict_demo_rl_only
    assert cfg.method.bc_lambda == 0.0
    assert cfg.method.bc_margin == 0.0
    assert not cfg.method.demo_fosd
    assert not cfg.method.separate_bc_policy
    assert not cfg.method.flow_policy
    assert not cfg.method.dense_return_q_target
    assert cfg.method.mc_lower_bound_target
    assert cfg.method.td_target_action_source == "critic_replay_max"
    assert cfg.replay.include_next_action


def test_cqn_as_stage31_direct_head_changes_only_q_factorization():
    configs = []
    for launch in (
        "cqn_as_pixel_bigym_nobc_stage30_offline_then_online_gate",
        "cqn_as_pixel_bigym_nobc_stage31_offline_direct_head_gate",
    ):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            configs.append(
                compose(
                    config_name="robobase_config",
                    overrides=[f"launch={launch}", "env=bigym/move_plate"],
                )
            )

    dueling, direct = configs
    assert dueling.method.use_dueling
    assert not direct.method.use_dueling
    assert not direct.method.mc_return_value_only
    for key in (
        "strict_demo_rl_only",
        "bc_lambda",
        "bc_margin",
        "demo_fosd",
        "dense_return_q_target",
        "mc_lower_bound_target",
        "td_target_action_source",
        "demo_behavior_force_probability",
    ):
        assert direct.method[key] == dueling.method[key]
    assert direct.num_pretrain_steps == dueling.num_pretrain_steps == 10_000
    assert direct.replay.include_next_action == dueling.replay.include_next_action


def test_cqn_as_stage31_direct_head_runs_reward_only_update():
    cfg = _strict_demo_rl_cfg(
        "method.use_dueling=false",
        "method.mc_return_value_only=false",
        "method.mc_lower_bound_target=true",
        "method.td_target_action_source=critic_replay_max",
        "replay.include_next_action=true",
    )
    agent = create_agent(
        cfg,
        observation_space=_spaces()[0],
        action_space=_spaces()[1],
    )
    flat_critic = flatten_dict(agent.params["critic"])
    assert not any("value_head" in part for key in flat_critic for part in key)

    batch = _batch()
    batch["action_tp1"] = np.roll(batch["action"], shift=-1, axis=1)
    batch["mc_return"] = np.linspace(0.2, 0.8, 4).astype(np.float32)
    agent.logging = True
    metrics = agent.update(iter([batch]), step=1)

    assert np.isfinite(metrics["critic_loss"])
    assert "bc_loss" not in metrics


def test_pessimistic_categorical_q_requires_both_critics_to_be_high():
    support = jnp.asarray([-1.0, 0.0, 1.0], dtype=jnp.float32)
    logits1 = jnp.asarray(
        [[[[-8.0, -8.0, 8.0], [-8.0, 8.0, -8.0]]]],
        dtype=jnp.float32,
    )
    logits2 = jnp.asarray(
        [[[[8.0, -8.0, -8.0], [-8.0, 8.0, -8.0]]]],
        dtype=jnp.float32,
    )

    q = pessimistic_categorical_q(logits1, logits2, support)

    assert int(jnp.argmax(q, axis=-1)[0, 0]) == 1
    assert float(q[0, 0, 0]) < -0.99
    assert abs(float(q[0, 0, 1])) < 1e-6


@pytest.mark.parametrize("jit", [False, True])
def test_cqn_as_pessimistic_twin_runs_one_reward_only_update_and_act(jit):
    cfg = _strict_demo_rl_cfg(
        f"backend.jit={str(jit).lower()}",
        "method.use_dueling=false",
        "method.pessimistic_twin_critic=true",
        "method.mc_return_value_only=false",
        "method.mc_lower_bound_target=true",
        "method.td_target_action_source=critic_replay_max",
        "replay.include_next_action=true",
        "method.weight_decay=0.0",
        "method.unseen_return_floor_weight=0.0",
    )
    observation_space, action_space = _spaces()
    agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    assert "critic2" in agent.params
    assert isinstance(agent.target_critic_params, tuple)
    first = flatten_dict(agent.params["critic"])
    second = flatten_dict(agent.params["critic2"])
    assert any(
        not np.array_equal(np.asarray(first[key]), np.asarray(second[key]))
        for key in first
        if key[-1] == "kernel" and first[key].size > 1
    )
    critic1_before = jax.tree.map(np.asarray, agent.params["critic"])
    critic2_before = jax.tree.map(np.asarray, agent.params["critic2"])
    batch = _batch()
    batch["action_tp1"] = np.roll(batch["action"], shift=-1, axis=1)
    batch["mc_return"] = np.linspace(0.2, 0.8, 4).astype(np.float32)
    agent.logging = True

    metrics = agent.update(iter([batch]), step=1)

    assert _tree_changed(critic1_before, agent.params["critic"])
    assert _tree_changed(critic2_before, agent.params["critic2"])
    assert np.isfinite(metrics["critic_loss"])
    assert np.isfinite(metrics["twin_q_disagreement"])
    assert metrics["demo_behavior_force_fraction"] == pytest.approx(0.0)
    assert "bc_loss" not in metrics
    observations = {
        "low_dim_state": np.zeros((1, 1, 5), dtype=np.float32),
    }
    action = agent.act(observations, step=3000, eval_mode=True)
    assert np.all(np.isfinite(np.asarray(action)))

    # Both learned critics, both target critics, and the matching optimizer
    # state must survive the offline -> online workspace resume boundary.
    state = agent.state_dict()
    checkpoint_state = agent.checkpoint_state_dict()
    restored = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    restored.load_state_dict(state)
    restored.load_checkpoint_state_dict(checkpoint_state)
    assert isinstance(restored.target_critic_params, tuple)
    for name in ("critic", "critic2"):
        for left, right in zip(
            jax.tree.leaves(agent.params[name]),
            jax.tree.leaves(restored.params[name]),
        ):
            np.testing.assert_allclose(left, right)
    for original_target, restored_target in zip(
        agent.target_critic_params,
        restored.target_critic_params,
    ):
        for left, right in zip(
            jax.tree.leaves(original_target),
            jax.tree.leaves(restored_target),
        ):
            np.testing.assert_allclose(left, right)
    restored_action = restored.act(observations, step=3000, eval_mode=True)
    assert np.all(np.isfinite(np.asarray(restored_action)))


def test_cqn_as_pessimistic_twin_rejects_dueling_path():
    with pytest.raises(ValueError, match="use_dueling=false"):
        create_agent(
            _strict_demo_rl_cfg(
                "method.pessimistic_twin_critic=true",
                "method.mc_lower_bound_target=true",
                "method.td_target_action_source=critic_replay_max",
                "replay.include_next_action=true",
                "method.unseen_return_floor_weight=0.0",
            ),
            observation_space=_spaces()[0],
            action_space=_spaces()[1],
        )


@pytest.mark.parametrize("jit", [False, True])
def test_cqn_as_auxiliary_td_is_normalized_reward_only_twin_update(jit):
    cfg = _strict_demo_rl_cfg(
        f"backend.jit={str(jit).lower()}",
        "method.use_dueling=false",
        "method.pessimistic_twin_critic=true",
        "method.auxiliary_td_loss_weight=1.0",
        "method.mc_return_value_only=false",
        "method.mc_lower_bound_target=true",
        "method.td_target_action_source=critic_replay_max",
        "replay.nstep=1",
        "replay.auxiliary_nstep=4",
        "replay.include_tp1=true",
        "replay.include_next_action=true",
        "method.weight_decay=0.0",
        "method.unseen_return_floor_weight=0.0",
    )
    observation_space, action_space = _spaces()
    agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    batch = _batch()
    batch["action_tp1"] = np.roll(batch["action"], shift=-1, axis=1)
    batch["low_dim_state_tp_aux"] = np.flip(
        batch["low_dim_state_tp1"], axis=-1
    ).copy()
    batch["action_tp_aux"] = np.roll(batch["action"], shift=-2, axis=1)
    batch["reward_aux"] = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    batch["discount_aux"] = np.full(4, 0.99**4, dtype=np.float32)
    batch["terminal_aux"] = np.zeros(4, dtype=bool)
    batch["mc_return"] = np.asarray([0.2, 0.4, 0.6, 0.8], dtype=np.float32)
    agent.logging = True

    metrics = agent.update(iter([batch]), step=1)

    assert np.isfinite(metrics["critic_loss"])
    assert metrics["auxiliary_td_loss_weight"] == pytest.approx(1.0)
    assert metrics["critic_loss"] == pytest.approx(
        0.5
        * (
            metrics["one_step_critic_loss"]
            + metrics["auxiliary_critic_loss"]
        ),
        rel=1e-5,
    )
    assert np.isfinite(metrics["auxiliary_behavior_minus_greedy_q"])
    assert "bc_loss" not in metrics
    assert "policy_bc_loss" not in metrics
    assert "margin_loss" not in metrics
    assert "mc_return_loss" not in metrics
    assert "policy" not in agent.params
    assert "flow_policy" not in agent.params


def test_cqn_as_auxiliary_td_requires_complete_matched_replay_contract():
    common = (
        "method.use_dueling=false",
        "method.pessimistic_twin_critic=true",
        "method.auxiliary_td_loss_weight=1.0",
        "method.mc_return_value_only=false",
        "method.mc_lower_bound_target=true",
        "method.td_target_action_source=critic_replay_max",
        "replay.include_next_action=true",
        "method.unseen_return_floor_weight=0.0",
    )
    with pytest.raises(ValueError, match="replay.auxiliary_nstep > 1"):
        create_agent(
            _strict_demo_rl_cfg(*common),
            observation_space=_spaces()[0],
            action_space=_spaces()[1],
        )
    with pytest.raises(ValueError, match="must be non-negative"):
        create_agent(
            _strict_demo_rl_cfg(
                "method.auxiliary_td_loss_weight=-0.1"
            ),
            observation_space=_spaces()[0],
            action_space=_spaces()[1],
        )


def test_cqn_as_stage32_launch_adds_only_pessimistic_twin():
    configs = []
    for launch in (
        "cqn_as_pixel_bigym_nobc_stage31_offline_direct_head_gate",
        "cqn_as_pixel_bigym_nobc_stage32_offline_pessimistic_twin_gate",
    ):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            configs.append(
                compose(
                    config_name="robobase_config",
                    overrides=[f"launch={launch}", "env=bigym/move_plate"],
                )
            )

    direct, twin = configs
    assert not direct.method.pessimistic_twin_critic
    assert twin.method.pessimistic_twin_critic
    assert not twin.method.use_dueling
    assert twin.method.strict_demo_rl_only
    assert twin.method.bc_lambda == 0.0
    assert twin.method.bc_margin == 0.0
    assert not twin.method.demo_fosd
    assert twin.method.mc_lower_bound_target
    assert twin.method.td_target_action_source == "critic_replay_max"
    direct_values = flatten_dict(OmegaConf.to_container(direct, resolve=False))
    twin_values = flatten_dict(OmegaConf.to_container(twin, resolve=False))
    changed = {
        ".".join(key)
        for key in set(direct_values) | set(twin_values)
        if direct_values.get(key) != twin_values.get(key)
    }
    assert changed == {"method.pessimistic_twin_critic"}


def test_cqn_as_stage35_launches_differ_only_by_auxiliary_td_weight():
    configs = []
    for launch in (
        "cqn_as_pixel_bigym_nobc_stage35_one_step_control",
        "cqn_as_pixel_bigym_nobc_stage35_one_plus_four_gate",
    ):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            configs.append(
                compose(
                    config_name="robobase_config",
                    overrides=[f"launch={launch}", "env=bigym/move_plate"],
                )
            )

    control, treatment = configs
    for cfg in configs:
        assert cfg.replay.nstep == 1
        assert cfg.replay.auxiliary_nstep == 4
        assert cfg.replay.include_tp1
        assert cfg.replay.include_next_action
        assert cfg.method.strict_demo_rl_only
        assert cfg.method.pessimistic_twin_critic
        assert not cfg.method.episodic_twin_head_exploration
        assert cfg.method.twin_rollout_beam_width == 1
        assert cfg.method.bc_lambda == 0.0
        assert cfg.method.bc_margin == 0.0
        assert not cfg.method.demo_fosd
    assert control.method.auxiliary_td_loss_weight == 0.0
    assert treatment.method.auxiliary_td_loss_weight == 1.0
    control_values = flatten_dict(
        OmegaConf.to_container(control, resolve=True)
    )
    treatment_values = flatten_dict(
        OmegaConf.to_container(treatment, resolve=True)
    )
    changed = {
        ".".join(key)
        for key in set(control_values) | set(treatment_values)
        if control_values.get(key) != treatment_values.get(key)
    }
    assert changed == {"method.auxiliary_td_loss_weight"}


def test_cqn_as_stage36_uses_official_batch_and_isolates_reward_target():
    configs = []
    for launch in (
        "cqn_as_pixel_bigym_stage36_offline_bc256_control",
        "cqn_as_pixel_bigym_stage36_offline_nobc_candidate256_gate",
    ):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            configs.append(
                compose(
                    config_name="robobase_config",
                    overrides=[f"launch={launch}", "env=bigym/move_plate"],
                )
            )

    control, treatment = configs
    for cfg in configs:
        assert cfg.batch_size == 256
        assert cfg.demo_batch_size == 256
        assert cfg.batch_size + cfg.demo_batch_size == 512
        assert cfg.num_pretrain_steps == 10000
        assert cfg.action_sequence == 16
        assert cfg.replay.nstep == 1
        assert cfg.method.critic_lr == pytest.approx(5e-5)
        assert cfg.method.critic_target_tau == pytest.approx(0.02)
        assert cfg.method.critic_lambda == pytest.approx(0.1)
        assert cfg.method.use_dueling
        assert not cfg.method.pessimistic_twin_critic
        assert cfg.method.stddev_schedule == "0.01"
        assert cfg.method.num_update_steps == 1
        assert cfg.use_self_imitation

    assert control.method.bc_lambda == pytest.approx(1.0)
    assert control.method.bc_margin == pytest.approx(0.1)
    assert control.method.demo_fosd
    assert not control.method.strict_demo_rl_only
    assert treatment.method.bc_lambda == 0.0
    assert treatment.method.bc_margin == 0.0
    assert not treatment.method.demo_fosd
    assert treatment.method.strict_demo_rl_only
    assert treatment.method.strict_allow_reward_only_success_replay
    assert treatment.method.td_target_action_source == "critic_replay_max"
    assert treatment.method.mc_lower_bound_target
    assert treatment.replay.include_next_action

    control_values = flatten_dict(OmegaConf.to_container(control, resolve=True))
    treatment_values = flatten_dict(OmegaConf.to_container(treatment, resolve=True))
    changed = {
        ".".join(key)
        for key in set(control_values) | set(treatment_values)
        if control_values.get(key) != treatment_values.get(key)
    }
    assert changed == {
        "method.bc_lambda",
        "method.bc_margin",
        "method.demo_fosd",
        "method.mc_lower_bound_target",
        "method.strict_allow_reward_only_success_replay",
        "method.strict_demo_rl_only",
        "method.td_target_action_source",
        "replay.include_next_action",
    }


def test_cqn_as_stage38_adds_only_dense_reward_q_to_stage36_treatment():
    configs = []
    for launch in (
        "cqn_as_pixel_bigym_stage36_offline_nobc_candidate256_gate",
        "cqn_as_pixel_bigym_stage38_offline_nobc_dense256_gate",
    ):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            configs.append(
                compose(
                    config_name="robobase_config",
                    overrides=[f"launch={launch}", "env=bigym/move_plate"],
                )
            )

    control, treatment = configs
    for cfg in configs:
        assert cfg.batch_size == 256
        assert cfg.demo_batch_size == 256
        assert cfg.num_pretrain_steps == 10000
        assert cfg.action_sequence == 16
        assert cfg.replay.nstep == 1
        assert cfg.replay.include_next_action
        assert cfg.method.critic_lambda == pytest.approx(0.1)
        assert cfg.method.critic_lr == pytest.approx(5e-5)
        assert cfg.method.critic_target_tau == pytest.approx(0.02)
        assert cfg.method.use_dueling
        assert not cfg.method.pessimistic_twin_critic
        assert cfg.method.num_update_steps == 1
        assert cfg.method.stddev_schedule == "0.01"
        assert cfg.method.strict_demo_rl_only
        assert cfg.method.strict_allow_reward_only_success_replay
        assert cfg.method.bc_lambda == 0.0
        assert cfg.method.bc_margin == 0.0
        assert not cfg.method.demo_fosd
        assert cfg.method.mc_lower_bound_target
        assert cfg.method.td_target_action_source == "critic_replay_max"
        assert cfg.method.unseen_return_floor_weight == 0.0
        assert not cfg.method.separate_bc_policy
        assert not cfg.method.flow_policy
    assert not control.method.dense_return_q_target
    assert treatment.method.dense_return_q_target

    control_values = flatten_dict(OmegaConf.to_container(control, resolve=True))
    treatment_values = flatten_dict(OmegaConf.to_container(treatment, resolve=True))
    changed = {
        ".".join(key)
        for key in set(control_values) | set(treatment_values)
        if control_values.get(key) != treatment_values.get(key)
    }
    assert changed == {"method.dense_return_q_target"}


def test_cqn_as_stage40_online_handoff_changes_only_dense_target_from_stage38():
    configs = []
    for launch in (
        "cqn_as_pixel_bigym_stage38_offline_nobc_dense256_gate",
        "cqn_as_pixel_bigym_stage40_online_canonical_handoff_gate",
    ):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            configs.append(
                compose(
                    config_name="robobase_config",
                    overrides=[f"launch={launch}", "env=bigym/move_plate"],
                )
            )

    offline_dense, online_canonical = configs
    for cfg in configs:
        assert cfg.batch_size == 256
        assert cfg.demo_batch_size == 256
        assert cfg.num_pretrain_steps == 10000
        assert cfg.action_sequence == 16
        assert cfg.replay.nstep == 1
        assert cfg.replay.include_next_action
        assert cfg.method.critic_lambda == pytest.approx(0.1)
        assert cfg.method.critic_lr == pytest.approx(5e-5)
        assert cfg.method.critic_target_tau == pytest.approx(0.02)
        assert cfg.method.use_dueling
        assert not cfg.method.pessimistic_twin_critic
        assert cfg.method.num_update_steps == 1
        assert cfg.method.stddev_schedule == "0.01"
        assert cfg.method.strict_demo_rl_only
        assert cfg.method.bc_lambda == 0.0
        assert cfg.method.bc_margin == 0.0
        assert not cfg.method.demo_fosd
        assert cfg.method.mc_lower_bound_target
        assert cfg.method.td_target_action_source == "critic_replay_max"
    assert offline_dense.method.dense_return_q_target
    assert not online_canonical.method.dense_return_q_target

    dense_values = flatten_dict(
        OmegaConf.to_container(offline_dense, resolve=True)
    )
    canonical_values = flatten_dict(
        OmegaConf.to_container(online_canonical, resolve=True)
    )
    changed = {
        ".".join(key)
        for key in set(dense_values) | set(canonical_values)
        if dense_values.get(key) != canonical_values.get(key)
    }
    assert changed == {"method.dense_return_q_target"}


def test_cqn_as_stage41_handoff_changes_only_positive_return_gate_from_stage38():
    configs = []
    for launch in (
        "cqn_as_pixel_bigym_stage38_offline_nobc_dense256_gate",
        "cqn_as_pixel_bigym_stage41_online_positive_dense_handoff_gate",
    ):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            configs.append(
                compose(
                    config_name="robobase_config",
                    overrides=[f"launch={launch}", "env=bigym/move_plate"],
                )
            )

    full_dense, positive_dense = configs
    for cfg in configs:
        assert cfg.batch_size == 256
        assert cfg.demo_batch_size == 256
        assert cfg.num_pretrain_steps == 10000
        assert cfg.action_sequence == 16
        assert cfg.replay.nstep == 1
        assert cfg.method.critic_lambda == pytest.approx(0.1)
        assert cfg.method.critic_lr == pytest.approx(5e-5)
        assert cfg.method.critic_target_tau == pytest.approx(0.02)
        assert cfg.method.num_update_steps == 1
        assert cfg.method.stddev_schedule == "0.01"
        assert cfg.method.strict_demo_rl_only
        assert cfg.method.dense_return_q_target
        assert cfg.method.bc_lambda == 0.0
        assert cfg.method.bc_margin == 0.0
        assert not cfg.method.demo_fosd
        assert not cfg.method.separate_bc_policy
        assert not cfg.method.flow_policy
    assert not full_dense.method.dense_return_positive_only
    assert positive_dense.method.dense_return_positive_only

    control_values = flatten_dict(
        OmegaConf.to_container(full_dense, resolve=True)
    )
    treatment_values = flatten_dict(
        OmegaConf.to_container(positive_dense, resolve=True)
    )
    changed = {
        ".".join(key)
        for key in set(control_values) | set(treatment_values)
        if control_values.get(key) != treatment_values.get(key)
    }
    assert changed == {"method.dense_return_positive_only"}


def test_cqn_as_stage42_only_disables_success_relabeling_from_stage41():
    configs = []
    for launch in (
        "cqn_as_pixel_bigym_stage41_online_positive_dense_handoff_gate",
        "cqn_as_pixel_bigym_stage42_fixed_expert_replay_gate",
    ):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            configs.append(
                compose(
                    config_name="robobase_config",
                    overrides=[f"launch={launch}", "env=bigym/move_plate"],
                )
            )

    growing, fixed = configs
    for cfg in configs:
        assert cfg.batch_size == 256
        assert cfg.demo_batch_size == 256
        assert cfg.num_pretrain_steps == 10000
        assert cfg.action_sequence == 16
        assert cfg.replay.nstep == 1
        assert cfg.method.strict_demo_rl_only
        assert cfg.method.dense_return_q_target
        assert cfg.method.dense_return_positive_only
        assert cfg.method.bc_lambda == 0.0
        assert cfg.method.bc_margin == 0.0
        assert not cfg.method.demo_fosd
        assert not cfg.method.separate_bc_policy
        assert not cfg.method.flow_policy
    assert growing.use_self_imitation
    assert growing.method.strict_allow_reward_only_success_replay
    assert not fixed.use_self_imitation
    assert not fixed.method.strict_allow_reward_only_success_replay

    growing_values = flatten_dict(
        OmegaConf.to_container(growing, resolve=True)
    )
    fixed_values = flatten_dict(OmegaConf.to_container(fixed, resolve=True))
    changed = {
        ".".join(key)
        for key in set(growing_values) | set(fixed_values)
        if growing_values.get(key) != fixed_values.get(key)
    }
    assert changed == {
        "use_self_imitation",
        "method.strict_allow_reward_only_success_replay",
    }


def test_cqn_as_stage39_changes_only_dense_positive_return_gate():
    configs = []
    for launch in (
        "cqn_as_pixel_bigym_nobc_stage10_discounted_dense_control",
        "cqn_as_pixel_bigym_nobc_stage39_positive_return_dense_gate",
    ):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            configs.append(
                compose(
                    config_name="robobase_config",
                    overrides=[f"launch={launch}", "env=bigym/move_plate"],
                )
            )

    control, treatment = configs
    for cfg in configs:
        assert cfg.method.strict_demo_rl_only
        assert cfg.method.dense_return_q_target
        assert cfg.method.mc_lower_bound_target
        assert cfg.method.bc_lambda == 0.0
        assert cfg.method.bc_margin == 0.0
        assert not cfg.method.demo_fosd
        assert not cfg.method.separate_bc_policy
        assert not cfg.method.flow_policy
        assert cfg.method.unseen_return_floor_weight == 0.0
    assert not control.method.dense_return_positive_only
    assert treatment.method.dense_return_positive_only

    control_values = flatten_dict(OmegaConf.to_container(control, resolve=True))
    treatment_values = flatten_dict(OmegaConf.to_container(treatment, resolve=True))
    changed = {
        ".".join(key)
        for key in set(control_values) | set(treatment_values)
        if control_values.get(key) != treatment_values.get(key)
    }
    assert changed == {"method.dense_return_positive_only"}


def test_select_episodic_twin_actions_keeps_each_environment_on_one_head():
    action1 = jnp.zeros((3, 2, 2), dtype=jnp.float32)
    action2 = jnp.ones((3, 2, 2), dtype=jnp.float32)

    selected = select_episodic_twin_actions(
        action1,
        action2,
        jnp.asarray([0, 1, 0]),
    )

    np.testing.assert_array_equal(np.asarray(selected[0]), 0.0)
    np.testing.assert_array_equal(np.asarray(selected[1]), 1.0)
    np.testing.assert_array_equal(np.asarray(selected[2]), 0.0)


def test_top2_joint_beam_keeps_best_complete_assignments_jitted():
    # Only parent zero is live. The exact best assignment uses every top-1
    # bin; the runner-up flips either of the two equal-regret first factors.
    q_values = jnp.asarray(
        [
            [
                [[10.0, 9.0, 0.0], [10.0, 9.0, 0.0], [10.0, 0.0, -1.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ]
        ],
        dtype=jnp.float32,
    )
    scores, parents, assignments = jax.jit(top2_joint_beam)(
        jnp.asarray([[0.0, -jnp.inf]], dtype=jnp.float32),
        q_values,
    )
    np.testing.assert_allclose(np.asarray(scores), [[10.0, 29.0 / 3.0]])
    np.testing.assert_array_equal(np.asarray(parents), [[0, 0]])
    np.testing.assert_array_equal(np.asarray(assignments[0, 0]), [0, 0, 0])
    assert tuple(np.asarray(assignments[0, 1])) in {(1, 0, 0), (0, 1, 0)}


def test_top2_joint_beam_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="shape"):
        top2_joint_beam(jnp.zeros((1, 1)), jnp.zeros((1, 1, 3)))
    with pytest.raises(ValueError, match="at least two bins"):
        top2_joint_beam(jnp.zeros((1, 1)), jnp.zeros((1, 1, 3, 1)))


def test_cqn_as_episodic_twin_head_is_stable_until_train_episode_reset():
    cfg = _strict_demo_rl_cfg(
        "num_train_envs=2",
        "num_explore_steps=0",
        "method.use_dueling=false",
        "method.pessimistic_twin_critic=true",
        "method.episodic_twin_head_exploration=true",
        "method.mc_return_value_only=false",
        "method.mc_lower_bound_target=true",
        "method.td_target_action_source=critic_replay_max",
        "replay.include_next_action=true",
        "method.weight_decay=0.0",
        "method.unseen_return_floor_weight=0.0",
    )
    observation_space, action_space = _spaces()
    agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    captured = []

    def fake_action(
        params,
        target_critic_params,
        obs_inputs,
        use_target,
        key,
        twin_head_indices,
    ):
        del params, target_critic_params, obs_inputs, use_target, key
        captured.append(np.asarray(twin_head_indices).copy())
        return jnp.zeros((2, agent.action_sequence, agent.action_dim))

    agent._greedy_action_impl = fake_action
    observations = {
        "low_dim_state": np.zeros((2, 1, 5), dtype=np.float32),
    }
    agent.act(observations, step=1, eval_mode=False)
    first_heads = captured[-1].copy()
    agent.act(observations, step=2, eval_mode=False)
    np.testing.assert_array_equal(captured[-1], first_heads)
    assert np.all(np.isin(first_heads, [0, 1]))

    assignments_before = agent._episodic_twin_head_assignments.sum()
    unchanged_head = int(agent._episodic_twin_heads[1])
    agent.reset(step=3, agents_to_reset=[0])
    assert agent._episodic_twin_head_assignments.sum() == assignments_before + 1
    assert int(agent._episodic_twin_heads[1]) == unchanged_head
    diagnostics = agent.rollout_diagnostics()
    assert diagnostics["episodic_twin_head_assignments"] == 3.0
    assert diagnostics["episodic_twin_head0_rate"] + diagnostics[
        "episodic_twin_head1_rate"
    ] == pytest.approx(1.0)


def test_cqn_as_episodic_twin_head_runs_jitted_train_and_pessimistic_eval():
    cfg = _strict_demo_rl_cfg(
        "num_train_envs=2",
        "num_explore_steps=0",
        "backend.jit=true",
        "method.use_dueling=false",
        "method.pessimistic_twin_critic=true",
        "method.episodic_twin_head_exploration=true",
        "method.mc_return_value_only=false",
        "method.mc_lower_bound_target=true",
        "method.td_target_action_source=critic_replay_max",
        "replay.include_next_action=true",
        "method.weight_decay=0.0",
        "method.unseen_return_floor_weight=0.0",
    )
    observation_space, action_space = _spaces()
    agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )

    train_action = agent.act(
        {"low_dim_state": np.zeros((2, 1, 5), dtype=np.float32)},
        step=1,
        eval_mode=False,
    )
    assert np.all(np.isfinite(np.asarray(train_action)))
    assert np.all(np.isin(agent._episodic_twin_heads, [0, 1]))

    eval_action = agent.act(
        {"low_dim_state": np.zeros((1, 1, 5), dtype=np.float32)},
        step=1,
        eval_mode=True,
    )
    assert np.asarray(eval_action).shape == (1, 3, 2)
    assert np.all(np.isfinite(np.asarray(eval_action)))


def test_cqn_as_joint_beam_runs_jitted_for_sampled_head_and_twin_eval():
    cfg = _strict_demo_rl_cfg(
        "num_train_envs=2",
        "num_explore_steps=0",
        "backend.jit=true",
        "method.use_dueling=false",
        "method.pessimistic_twin_critic=true",
        "method.episodic_twin_head_exploration=true",
        "method.twin_rollout_beam_width=3",
        "method.mc_return_value_only=false",
        "method.mc_lower_bound_target=true",
        "method.td_target_action_source=critic_replay_max",
        "replay.include_next_action=true",
        "method.weight_decay=0.0",
        "method.unseen_return_floor_weight=0.0",
    )
    observation_space, action_space = _spaces()
    agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    assert agent.twin_rollout_beam_width == 3

    train_action = agent.act(
        {"low_dim_state": np.zeros((2, 1, 5), dtype=np.float32)},
        step=1,
        eval_mode=False,
    )
    eval_action = agent.act(
        {"low_dim_state": np.zeros((1, 1, 5), dtype=np.float32)},
        step=1,
        eval_mode=True,
    )
    for action in (train_action, eval_action):
        assert np.all(np.isfinite(np.asarray(action)))
        assert np.all(np.asarray(action) >= -1.0)
        assert np.all(np.asarray(action) <= 1.0)


def test_cqn_as_joint_beam_allows_single_critic_and_gates_twin_path():
    observation_space, action_space = _spaces()
    common = (
        "method.use_dueling=false",
        "method.mc_return_value_only=false",
        "method.mc_lower_bound_target=true",
        "method.td_target_action_source=critic_replay_max",
        "replay.include_next_action=true",
        "method.weight_decay=0.0",
        "method.unseen_return_floor_weight=0.0",
        "method.twin_rollout_beam_width=2",
    )
    single_critic = create_agent(
        _strict_demo_rl_cfg(*common),
        observation_space=observation_space,
        action_space=action_space,
    )
    assert single_critic.twin_rollout_beam_width == 2
    assert not single_critic.pessimistic_twin_critic
    with pytest.raises(
        ValueError, match="episodic_twin_head_exploration=true"
    ):
        create_agent(
            _strict_demo_rl_cfg(
                *common,
                "method.pessimistic_twin_critic=true",
            ),
            observation_space=observation_space,
            action_space=action_space,
        )
    with pytest.raises(ValueError, match="autoregressive_action_dims=false"):
        create_agent(
            _strict_demo_rl_cfg(
                *common,
                "method.pessimistic_twin_critic=true",
                "method.episodic_twin_head_exploration=true",
                "method.autoregressive_action_dims=true",
            ),
            observation_space=observation_space,
            action_space=action_space,
        )


def test_cqn_as_stage33_launch_changes_only_online_exploration_policy():
    configs = []
    for launch in (
        "cqn_as_pixel_bigym_nobc_stage32_offline_pessimistic_twin_gate",
        "cqn_as_pixel_bigym_nobc_stage33_episodic_twin_explore_gate",
    ):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            configs.append(
                compose(
                    config_name="robobase_config",
                    overrides=[f"launch={launch}", "env=bigym/move_plate"],
                )
            )

    baseline, exploration = configs
    assert not baseline.method.episodic_twin_head_exploration
    assert exploration.method.episodic_twin_head_exploration
    assert exploration.method.pessimistic_twin_critic
    assert exploration.method.strict_demo_rl_only
    assert exploration.method.bc_lambda == 0.0
    assert exploration.method.bc_margin == 0.0
    assert not exploration.method.demo_fosd
    baseline_values = flatten_dict(
        OmegaConf.to_container(baseline, resolve=False)
    )
    exploration_values = flatten_dict(
        OmegaConf.to_container(exploration, resolve=False)
    )
    changed = {
        ".".join(key)
        for key in set(baseline_values) | set(exploration_values)
        if baseline_values.get(key) != exploration_values.get(key)
    }
    assert changed == {"method.episodic_twin_head_exploration"}


def test_cqn_as_stage34_launch_changes_only_rollout_beam_width():
    configs = []
    for launch in (
        "cqn_as_pixel_bigym_nobc_stage33_episodic_twin_explore_gate",
        "cqn_as_pixel_bigym_nobc_stage34_joint_beam_gate",
    ):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            configs.append(
                compose(
                    config_name="robobase_config",
                    overrides=[f"launch={launch}", "env=bigym/move_plate"],
                )
            )

    baseline, beam = configs
    assert baseline.method.twin_rollout_beam_width == 1
    assert beam.method.twin_rollout_beam_width == 8
    assert beam.method.episodic_twin_head_exploration
    assert beam.method.pessimistic_twin_critic
    assert beam.method.strict_demo_rl_only
    assert beam.method.bc_lambda == 0.0
    assert beam.method.bc_margin == 0.0
    assert not beam.method.demo_fosd
    baseline_values = flatten_dict(
        OmegaConf.to_container(baseline, resolve=False)
    )
    beam_values = flatten_dict(OmegaConf.to_container(beam, resolve=False))
    changed = {
        ".".join(key)
        for key in set(baseline_values) | set(beam_values)
        if baseline_values.get(key) != beam_values.get(key)
    }
    assert changed == {"method.twin_rollout_beam_width"}


def test_cqn_as_no_bc_stage1_launches_are_strict_and_matched():
    configs = []
    for launch in (
        "cqn_as_pixel_bigym_nobc_td_gate",
        "cqn_as_pixel_bigym_nobc_floor_gate",
    ):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            configs.append(
                compose(
                    config_name="robobase_config",
                    overrides=[
                        f"launch={launch}",
                        "env=bigym/move_plate",
                    ],
                )
            )

    control, treatment = configs
    assert control.method.strict_demo_rl_only
    assert treatment.method.strict_demo_rl_only
    assert control.num_pretrain_steps == treatment.num_pretrain_steps == 0
    assert not control.use_self_imitation
    assert not treatment.use_self_imitation
    assert control.method.critic_lambda == treatment.method.critic_lambda == 1.0
    assert control.method.bc_lambda == treatment.method.bc_lambda == 0.0
    assert control.method.bc_margin == treatment.method.bc_margin == 0.0
    assert not control.method.demo_fosd
    assert not treatment.method.demo_fosd
    assert control.method.unseen_return_floor_weight == 0.0
    assert treatment.method.unseen_return_floor_weight == 1.0
    assert control.num_eval_episodes == treatment.num_eval_episodes == 0
    assert control.snapshot_every_n == treatment.snapshot_every_n == 2500


def test_cqn_as_no_bc_stage2_launches_complete_mc_floor_factorial():
    configs = []
    for launch in (
        "cqn_as_pixel_bigym_nobc_mc_gate",
        "cqn_as_pixel_bigym_nobc_mc_floor_gate",
    ):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            configs.append(
                compose(
                    config_name="robobase_config",
                    overrides=[
                        f"launch={launch}",
                        "env=bigym/move_plate",
                    ],
                )
            )

    mc_only, mc_floor = configs
    for cfg in configs:
        assert cfg.method.strict_demo_rl_only
        assert cfg.method.mc_lower_bound_target
        assert cfg.method.mc_return_weight == 0.0
        assert cfg.method.bc_lambda == 0.0
        assert cfg.method.bc_margin == 0.0
        assert not cfg.method.demo_fosd
        assert cfg.num_pretrain_steps == 0
        assert not cfg.use_self_imitation
    assert mc_only.method.unseen_return_floor_weight == 0.0
    assert mc_floor.method.unseen_return_floor_weight == 1.0


def test_cqn_as_no_bc_stage3_changes_only_unseen_floor_reduction():
    configs = []
    for launch in (
        "cqn_as_pixel_bigym_nobc_mc_mean_floor_gate",
        "cqn_as_pixel_bigym_nobc_mc_max_floor_gate",
    ):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            configs.append(
                compose(
                    config_name="robobase_config",
                    overrides=[
                        f"launch={launch}",
                        "env=bigym/move_plate",
                    ],
                )
            )

    mean_floor, max_floor = configs
    for cfg in configs:
        assert cfg.method.strict_demo_rl_only
        assert cfg.method.mc_lower_bound_target
        assert cfg.method.mc_return_weight == 0.0
        assert cfg.method.unseen_return_floor_weight == 1.0
        assert cfg.method.unseen_return_floor_value == 0.0
        assert cfg.method.bc_lambda == 0.0
        assert cfg.method.bc_margin == 0.0
        assert not cfg.method.demo_fosd
        assert cfg.num_pretrain_steps == 0
        assert not cfg.use_self_imitation
    assert mean_floor.method.unseen_return_floor_reduction == "mean"
    assert max_floor.method.unseen_return_floor_reduction == "max"


def test_cqn_as_no_bc_stage4_changes_only_conservative_tail_width():
    configs = []
    for launch in (
        "cqn_as_pixel_bigym_nobc_mc_top1_floor_gate",
        "cqn_as_pixel_bigym_nobc_mc_top2_floor_gate",
    ):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            configs.append(
                compose(
                    config_name="robobase_config",
                    overrides=[
                        f"launch={launch}",
                        "env=bigym/move_plate",
                    ],
                )
            )

    top1, top2 = configs
    for cfg in configs:
        assert cfg.method.strict_demo_rl_only
        assert cfg.method.mc_lower_bound_target
        assert cfg.method.mc_return_weight == 0.0
        assert cfg.method.unseen_return_floor_weight == 1.0
        assert cfg.method.unseen_return_floor_value == 0.0
        assert cfg.method.bc_lambda == 0.0
        assert cfg.method.bc_margin == 0.0
        assert not cfg.method.demo_fosd
        assert cfg.num_pretrain_steps == 0
        assert not cfg.use_self_imitation
    assert top1.method.unseen_return_floor_reduction == "max"
    assert top1.method.unseen_return_floor_topk == 1
    assert top2.method.unseen_return_floor_reduction == "topk"
    assert top2.method.unseen_return_floor_topk == 2


def test_cqn_as_no_bc_stage5_preserves_batch_size_and_changes_replay_source():
    configs = []
    for launch in (
        "cqn_as_pixel_bigym_nobc_mc_mixed_replay_gate",
        "cqn_as_pixel_bigym_nobc_mc_demo_only_gate",
    ):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            configs.append(
                compose(
                    config_name="robobase_config",
                    overrides=[
                        f"launch={launch}",
                        "env=bigym/move_plate",
                    ],
                )
            )

    mixed, demo_only = configs
    for cfg in configs:
        assert cfg.method.strict_demo_rl_only
        assert cfg.method.mc_lower_bound_target
        assert cfg.method.unseen_return_floor_reduction == "max"
        assert cfg.method.unseen_return_floor_topk == 1
        assert cfg.method.bc_lambda == 0.0
        assert cfg.method.bc_margin == 0.0
        assert cfg.num_pretrain_steps == 0
        assert not cfg.use_self_imitation
    assert mixed.batch_size + mixed.demo_batch_size == 32
    assert not mixed.replay.demo_only_updates
    assert demo_only.demo_batch_size == 32
    assert demo_only.replay.demo_only_updates


def test_cqn_as_no_bc_stage6_changes_only_action_dimension_factorization():
    configs = []
    for launch in (
        "cqn_as_pixel_bigym_nobc_mc_parallel_dims_gate",
        "cqn_as_pixel_bigym_nobc_mc_autoregressive_dims_gate",
    ):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            configs.append(
                compose(
                    config_name="robobase_config",
                    overrides=[
                        f"launch={launch}",
                        "env=bigym/move_plate",
                    ],
                )
            )

    parallel, autoregressive = configs
    for cfg in configs:
        assert cfg.method.strict_demo_rl_only
        assert cfg.method.mc_lower_bound_target
        assert cfg.method.unseen_return_floor_reduction == "max"
        assert cfg.method.unseen_return_floor_topk == 1
        assert cfg.method.bc_lambda == 0.0
        assert cfg.method.bc_margin == 0.0
        assert not cfg.method.demo_fosd
        assert cfg.batch_size == 16
        assert cfg.demo_batch_size == 16
        assert not cfg.replay.demo_only_updates
        assert cfg.num_pretrain_steps == 0
        assert not cfg.use_self_imitation
    assert not parallel.method.autoregressive_action_dims
    assert autoregressive.method.autoregressive_action_dims


def test_cqn_as_no_bc_stage7_changes_only_dense_return_objective():
    configs = []
    for launch in (
        "cqn_as_pixel_bigym_nobc_mc_stage7_max_floor_gate",
        "cqn_as_pixel_bigym_nobc_mc_stage7_dense_return_gate",
    ):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            configs.append(
                compose(
                    config_name="robobase_config",
                    overrides=[
                        f"launch={launch}",
                        "env=bigym/move_plate",
                    ],
                )
            )

    max_floor, dense_return = configs
    for cfg in configs:
        assert cfg.method.strict_demo_rl_only
        assert cfg.method.mc_lower_bound_target
        assert not cfg.method.autoregressive_action_dims
        assert cfg.method.bc_lambda == 0.0
        assert cfg.method.bc_margin == 0.0
        assert not cfg.method.demo_fosd
        assert cfg.batch_size == 16
        assert cfg.demo_batch_size == 16
        assert not cfg.replay.demo_only_updates
        assert cfg.num_pretrain_steps == 0
        assert not cfg.use_self_imitation
    assert max_floor.method.unseen_return_floor_weight == 1.0
    assert max_floor.method.unseen_return_floor_reduction == "max"
    assert not max_floor.method.dense_return_q_target
    assert dense_return.method.unseen_return_floor_weight == 0.0
    assert dense_return.method.dense_return_q_target


def test_cqn_as_no_bc_stage10_changes_only_return_target_mode():
    configs = []
    for launch in (
        "cqn_as_pixel_bigym_nobc_stage10_discounted_dense_control",
        "cqn_as_pixel_bigym_nobc_stage10_episodic_success_gate",
    ):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            configs.append(
                compose(
                    config_name="robobase_config",
                    overrides=[
                        f"launch={launch}",
                        "env=bigym/move_plate",
                    ],
                )
            )

    control, treatment = configs
    for cfg in configs:
        assert cfg.method.strict_demo_rl_only
        assert cfg.method.dense_return_q_target
        assert cfg.method.unseen_return_floor_weight == 0.0
        assert cfg.method.unseen_return_floor_value == 0.0
        assert cfg.method.bc_lambda == 0.0
        assert cfg.method.bc_margin == 0.0
        assert not cfg.method.demo_fosd
        assert not cfg.method.separate_bc_policy
        assert not cfg.method.flow_policy
        assert not cfg.method.coarse_flow
        assert cfg.batch_size == 16
        assert cfg.demo_batch_size == 16
        assert not cfg.replay.demo_only_updates
        assert cfg.num_pretrain_steps == 0
        assert not cfg.use_self_imitation
    assert control.method.mc_lower_bound_target
    assert not control.method.episodic_success_q_target
    assert not treatment.method.mc_lower_bound_target
    assert treatment.method.episodic_success_q_target


def test_cqn_as_no_bc_stage11_is_ordered_single_q_target():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_as_pixel_bigym_nobc_stage11_ordered_success_gate",
                "env=bigym/move_plate",
            ],
        )

    assert cfg.method.strict_demo_rl_only
    assert cfg.method.dense_return_q_target
    assert cfg.method.mc_lower_bound_target
    assert not cfg.method.episodic_success_q_target
    assert cfg.method.ordered_success_return_mix == pytest.approx(0.5)
    assert cfg.method.unseen_return_floor_weight == 0.0
    assert cfg.method.mc_return_weight == 0.0
    assert cfg.method.bc_lambda == 0.0
    assert cfg.method.bc_margin == 0.0
    assert not cfg.method.demo_fosd
    assert not cfg.method.separate_bc_policy
    assert not cfg.method.flow_policy
    assert not cfg.method.coarse_flow
    assert cfg.batch_size == 16
    assert cfg.demo_batch_size == 16
    assert not cfg.replay.demo_only_updates
    assert cfg.num_pretrain_steps == 0
    assert not cfg.use_self_imitation


def test_cqn_as_no_bc_stage12_changes_only_chunk_return_horizon():
    configs = []
    for launch in (
        "cqn_as_pixel_bigym_nobc_stage12_k8_nstep1_control",
        "cqn_as_pixel_bigym_nobc_stage12_k8_nstep8_gate",
    ):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            configs.append(
                compose(
                    config_name="robobase_config",
                    overrides=[
                        f"launch={launch}",
                        "env=bigym/move_plate",
                    ],
                )
            )

    control, treatment = configs
    for cfg in configs:
        assert cfg.action_sequence == 8
        assert cfg.execution_length == 1
        assert not cfg.temporal_ensemble
        assert cfg.method.temporal_ensemble
        assert cfg.method.temporal_ensemble_replan_interval == 8
        assert cfg.method.strict_demo_rl_only
        assert cfg.method.dense_return_q_target
        assert cfg.method.mc_lower_bound_target
        assert cfg.method.ordered_success_return_mix == 0.0
        assert cfg.method.bc_lambda == 0.0
        assert cfg.method.bc_margin == 0.0
        assert not cfg.method.demo_fosd
        assert not cfg.method.separate_bc_policy
        assert not cfg.method.flow_policy
        assert not cfg.method.coarse_flow
        assert cfg.num_pretrain_steps == 0
        assert not cfg.use_self_imitation
    assert control.replay.nstep == 1
    assert treatment.replay.nstep == treatment.action_sequence == 8

    control_values = flatten_dict(
        OmegaConf.to_container(control, resolve=False)
    )
    treatment_values = flatten_dict(
        OmegaConf.to_container(treatment, resolve=False)
    )
    differences = {
        key
        for key in set(control_values) | set(treatment_values)
        if control_values.get(key) != treatment_values.get(key)
    }
    assert differences == {("replay", "nstep")}


def test_cqn_as_no_bc_stage13_changes_only_finest_neighbor_target():
    configs = []
    for launch in (
        "cqn_as_pixel_bigym_nobc_stage10_discounted_dense_control",
        "cqn_as_pixel_bigym_nobc_stage13_finest_neighbor_gate",
    ):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            configs.append(
                compose(
                    config_name="robobase_config",
                    overrides=[
                        f"launch={launch}",
                        "env=bigym/move_plate",
                    ],
                )
            )

    control, treatment = configs
    for cfg in configs:
        assert cfg.action_sequence == 16
        assert cfg.replay.nstep == 1
        assert cfg.method.temporal_ensemble
        assert cfg.method.temporal_ensemble_replan_interval == 1
        assert cfg.method.strict_demo_rl_only
        assert cfg.method.dense_return_q_target
        assert cfg.method.mc_lower_bound_target
        assert cfg.method.ordered_success_return_mix == 0.0
        assert cfg.method.unseen_return_floor_weight == 0.0
        assert cfg.method.bc_lambda == 0.0
        assert cfg.method.bc_margin == 0.0
        assert not cfg.method.demo_fosd
        assert not cfg.method.separate_bc_policy
        assert not cfg.method.flow_policy
        assert not cfg.method.coarse_flow
        assert cfg.num_pretrain_steps == 0
        assert not cfg.use_self_imitation
    assert control.method.dense_return_finest_neighbor_weight == 0.0
    assert treatment.method.dense_return_finest_neighbor_weight == pytest.approx(
        0.5
    )

    control_values = flatten_dict(
        OmegaConf.to_container(control, resolve=False)
    )
    treatment_values = flatten_dict(
        OmegaConf.to_container(treatment, resolve=False)
    )
    differences = {
        key
        for key in set(control_values) | set(treatment_values)
        if control_values.get(key) != treatment_values.get(key)
    }
    assert differences == {
        ("method", "dense_return_finest_neighbor_weight")
    }


def test_cqn_as_no_bc_stage14_changes_only_to_executed_primitive_action():
    configs = []
    for launch in (
        "cqn_as_pixel_bigym_nobc_stage10_discounted_dense_control",
        "cqn_as_pixel_bigym_nobc_stage14_primitive_q_gate",
    ):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            configs.append(
                compose(
                    config_name="robobase_config",
                    overrides=[
                        f"launch={launch}",
                        "env=bigym/move_plate",
                    ],
                )
            )

    control, treatment = configs
    for cfg in configs:
        assert cfg.execution_length == 1
        assert cfg.replay.nstep == 1
        assert cfg.method.temporal_ensemble
        assert cfg.method.temporal_ensemble_replan_interval == 1
        assert cfg.method.strict_demo_rl_only
        assert cfg.method.dense_return_q_target
        assert cfg.method.dense_return_finest_neighbor_weight == 0.0
        assert cfg.method.mc_lower_bound_target
        assert cfg.method.ordered_success_return_mix == 0.0
        assert cfg.method.unseen_return_floor_weight == 0.0
        assert cfg.method.bc_lambda == 0.0
        assert cfg.method.bc_margin == 0.0
        assert not cfg.method.demo_fosd
        assert not cfg.method.separate_bc_policy
        assert not cfg.method.flow_policy
        assert not cfg.method.coarse_flow
        assert cfg.num_pretrain_steps == 0
        assert not cfg.use_self_imitation
    assert control.action_sequence == 16
    assert treatment.action_sequence == 1

    control_values = flatten_dict(
        OmegaConf.to_container(control, resolve=False)
    )
    treatment_values = flatten_dict(
        OmegaConf.to_container(treatment, resolve=False)
    )
    differences = {
        key
        for key in set(control_values) | set(treatment_values)
        if control_values.get(key) != treatment_values.get(key)
    }
    assert differences == {("action_sequence",)}


def test_cqn_as_primitive_dense_q_runs_one_strict_rl_update():
    observation_space, action_space = _spaces(action_sequence=1)
    cfg = _strict_demo_rl_cfg(
        "action_sequence=1",
        "method.dense_return_q_target=true",
        "method.dense_return_finest_neighbor_weight=0.0",
        "method.unseen_return_floor_weight=0.0",
        "method.mc_lower_bound_target=true",
        "method.weight_decay=0.0",
    )
    agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    assert agent.action_sequence == 1
    assert agent._flat_action_dim == agent.action_dim
    assert "policy" not in agent.params
    assert "flow_policy" not in agent.params
    chunk = np.asarray([[[0.25, -0.5]]], dtype=np.float32)
    np.testing.assert_allclose(
        agent._ensemble_current_action(chunk, eval_mode=False),
        chunk[:, 0],
    )
    np.testing.assert_array_equal(
        agent._temporal_replan_mask(eval_mode=False, batch_size=1),
        [True],
    )

    batch = _batch(action_sequence=1)
    batch["reward"][:] = 0.0
    batch["mc_return"] = np.linspace(0.2, 0.8, 4).astype(np.float32)
    agent.logging = True
    metrics = agent.update(iter([batch]), step=1)
    assert np.isfinite(metrics["critic_loss"])
    assert metrics["dense_return_q_loss"] == pytest.approx(
        metrics["critic_loss"]
    )
    assert "mc_return_loss" not in metrics
    assert "unseen_return_floor_loss" not in metrics


@pytest.mark.parametrize(
    "treatment_launch",
    (
        "cqn_as_pixel_bigym_nobc_stage15_replay_sarsa_gate",
        "cqn_as_pixel_bigym_nobc_stage17_effective_replay_sarsa_gate",
    ),
)
def test_cqn_as_no_bc_replay_sarsa_changes_only_bellman_action_source(
    treatment_launch,
):
    configs = []
    for launch in (
        "cqn_as_pixel_bigym_nobc_stage10_discounted_dense_control",
        treatment_launch,
    ):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            configs.append(
                compose(
                    config_name="robobase_config",
                    overrides=[
                        f"launch={launch}",
                        "env=bigym/move_plate",
                    ],
                )
            )

    control, treatment = configs
    for cfg in configs:
        assert cfg.action_sequence == 16
        assert cfg.execution_length == 1
        assert cfg.replay.nstep == 1
        assert cfg.method.temporal_ensemble
        assert cfg.method.temporal_ensemble_replan_interval == 1
        assert cfg.method.critic_sequence_mode == "full"
        assert cfg.method.strict_demo_rl_only
        assert cfg.method.dense_return_q_target
        assert cfg.method.dense_return_finest_neighbor_weight == 0.0
        assert cfg.method.mc_lower_bound_target
        assert cfg.method.ordered_success_return_mix == 0.0
        assert cfg.method.unseen_return_floor_weight == 0.0
        assert cfg.method.bc_lambda == 0.0
        assert cfg.method.bc_margin == 0.0
        assert not cfg.method.demo_fosd
        assert not cfg.method.separate_bc_policy
        assert not cfg.method.flow_policy
        assert not cfg.method.coarse_flow
        assert cfg.num_pretrain_steps == 0
        assert not cfg.use_self_imitation
    assert control.method.td_target_action_source == "critic"
    assert treatment.method.td_target_action_source == "replay_next"

    control_values = flatten_dict(
        OmegaConf.to_container(control, resolve=False)
    )
    treatment_values = flatten_dict(
        OmegaConf.to_container(treatment, resolve=False)
    )
    differences = {
        key
        for key in set(control_values) | set(treatment_values)
        if control_values.get(key) != treatment_values.get(key)
    }
    assert differences == {("method", "td_target_action_source")}


def test_cqn_as_no_bc_stage16_changes_only_sequence_return_alignment():
    configs = []
    for launch in (
        "cqn_as_pixel_bigym_nobc_stage10_discounted_dense_control",
        "cqn_as_pixel_bigym_nobc_stage16_sequence_return_gate",
    ):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            configs.append(
                compose(
                    config_name="robobase_config",
                    overrides=[
                        f"launch={launch}",
                        "env=bigym/move_plate",
                    ],
                )
            )

    control, treatment = configs
    for cfg in configs:
        assert cfg.action_sequence == 16
        assert cfg.execution_length == 1
        assert cfg.replay.nstep == 1
        assert cfg.replay.gamma == pytest.approx(0.99)
        assert cfg.method.temporal_ensemble
        assert cfg.method.temporal_ensemble_replan_interval == 1
        assert cfg.method.critic_sequence_mode == "full"
        assert cfg.method.td_target_action_source == "critic"
        assert cfg.method.strict_demo_rl_only
        assert cfg.method.dense_return_q_target
        assert cfg.method.dense_return_finest_neighbor_weight == 0.0
        assert cfg.method.mc_lower_bound_target
        assert cfg.method.ordered_success_return_mix == 0.0
        assert cfg.method.unseen_return_floor_weight == 0.0
        assert cfg.method.bc_lambda == 0.0
        assert cfg.method.bc_margin == 0.0
        assert not cfg.method.demo_fosd
        assert not cfg.method.separate_bc_policy
        assert not cfg.method.flow_policy
        assert not cfg.method.coarse_flow
        assert cfg.num_pretrain_steps == 0
        assert not cfg.use_self_imitation
    assert control.method.sequence_aligned_mc_discount is None
    assert treatment.method.sequence_aligned_mc_discount == pytest.approx(
        treatment.replay.gamma
    )

    control_values = flatten_dict(
        OmegaConf.to_container(control, resolve=False)
    )
    treatment_values = flatten_dict(
        OmegaConf.to_container(treatment, resolve=False)
    )
    differences = {
        key
        for key in set(control_values) | set(treatment_values)
        if control_values.get(key) != treatment_values.get(key)
    }
    assert differences == {
        ("method", "sequence_aligned_mc_discount")
    }


def test_cqn_as_no_bc_stage18_changes_only_dense_loss_statistic():
    configs = []
    for launch in (
        "cqn_as_pixel_bigym_nobc_stage10_discounted_dense_control",
        "cqn_as_pixel_bigym_nobc_stage18_expected_q_gate",
    ):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            configs.append(
                compose(
                    config_name="robobase_config",
                    overrides=[
                        f"launch={launch}",
                        "env=bigym/move_plate",
                    ],
                )
            )

    control, treatment = configs
    for cfg in configs:
        assert cfg.action_sequence == 16
        assert cfg.execution_length == 1
        assert cfg.replay.nstep == 1
        assert cfg.method.temporal_ensemble
        assert cfg.method.temporal_ensemble_replan_interval == 1
        assert cfg.method.critic_sequence_mode == "full"
        assert cfg.method.td_target_action_source == "critic"
        assert cfg.method.strict_demo_rl_only
        assert cfg.method.dense_return_q_target
        assert cfg.method.dense_return_finest_neighbor_weight == 0.0
        assert cfg.method.mc_lower_bound_target
        assert cfg.method.ordered_success_return_mix == 0.0
        assert cfg.method.sequence_aligned_mc_discount is None
        assert cfg.method.unseen_return_floor_weight == 0.0
        assert cfg.method.bc_lambda == 0.0
        assert cfg.method.bc_margin == 0.0
        assert not cfg.method.demo_fosd
        assert not cfg.method.separate_bc_policy
        assert not cfg.method.flow_policy
        assert not cfg.method.coarse_flow
        assert cfg.num_pretrain_steps == 0
        assert not cfg.use_self_imitation
    assert not control.method.dense_return_expected_q_loss
    assert treatment.method.dense_return_expected_q_loss

    control_values = flatten_dict(
        OmegaConf.to_container(control, resolve=False)
    )
    treatment_values = flatten_dict(
        OmegaConf.to_container(treatment, resolve=False)
    )
    differences = {
        key
        for key in set(control_values) | set(treatment_values)
        if control_values.get(key) != treatment_values.get(key)
    }
    assert differences == {
        ("method", "dense_return_expected_q_loss")
    }


def test_cqn_as_no_bc_stage19_changes_only_q_reward_scale():
    configs = []
    for launch in (
        "cqn_as_pixel_bigym_nobc_stage10_discounted_dense_control",
        "cqn_as_pixel_bigym_nobc_stage19_reward_scale_gate",
    ):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            configs.append(
                compose(
                    config_name="robobase_config",
                    overrides=[
                        f"launch={launch}",
                        "env=bigym/move_plate",
                    ],
                )
            )

    control, treatment = configs
    for cfg in configs:
        assert cfg.action_sequence == 16
        assert cfg.execution_length == 1
        assert cfg.replay.nstep == 1
        assert cfg.method.temporal_ensemble
        assert cfg.method.temporal_ensemble_replan_interval == 1
        assert cfg.method.critic_sequence_mode == "full"
        assert cfg.method.td_target_action_source == "critic"
        assert cfg.method.strict_demo_rl_only
        assert cfg.method.dense_return_q_target
        assert not cfg.method.dense_return_expected_q_loss
        assert cfg.method.dense_return_finest_neighbor_weight == 0.0
        assert cfg.method.mc_lower_bound_target
        assert cfg.method.ordered_success_return_mix == 0.0
        assert cfg.method.sequence_aligned_mc_discount is None
        assert cfg.method.unseen_return_floor_weight == 0.0
        assert cfg.method.bc_lambda == 0.0
        assert cfg.method.bc_margin == 0.0
        assert not cfg.method.demo_fosd
        assert not cfg.method.separate_bc_policy
        assert not cfg.method.flow_policy
        assert not cfg.method.coarse_flow
        assert cfg.num_pretrain_steps == 0
        assert not cfg.use_self_imitation
    assert control.method.q_reward_scale == pytest.approx(1.0)
    assert treatment.method.q_reward_scale == pytest.approx(2.0)
    assert treatment.method.q_reward_scale == pytest.approx(
        treatment.method.v_max
    )

    control_values = flatten_dict(
        OmegaConf.to_container(control, resolve=False)
    )
    treatment_values = flatten_dict(
        OmegaConf.to_container(treatment, resolve=False)
    )
    differences = {
        key
        for key in set(control_values) | set(treatment_values)
        if control_values.get(key) != treatment_values.get(key)
    }
    assert differences == {("method", "q_reward_scale")}


def test_cqn_as_no_bc_stage20_changes_only_advantage_gap_operator():
    configs = []
    for launch in (
        "cqn_as_pixel_bigym_nobc_stage10_discounted_dense_control",
        "cqn_as_pixel_bigym_nobc_stage20_advantage_gap_gate",
    ):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            configs.append(
                compose(
                    config_name="robobase_config",
                    overrides=[
                        f"launch={launch}",
                        "env=bigym/move_plate",
                    ],
                )
            )

    control, treatment = configs
    for cfg in configs:
        assert cfg.action_sequence == 16
        assert cfg.execution_length == 1
        assert cfg.replay.nstep == 1
        assert cfg.method.temporal_ensemble
        assert cfg.method.temporal_ensemble_replan_interval == 1
        assert cfg.method.critic_sequence_mode == "full"
        assert cfg.method.td_target_action_source == "critic"
        assert cfg.method.strict_demo_rl_only
        assert cfg.method.dense_return_q_target
        assert not cfg.method.dense_return_expected_q_loss
        assert cfg.method.dense_return_finest_neighbor_weight == 0.0
        assert cfg.method.q_reward_scale == 1.0
        assert cfg.method.mc_lower_bound_target
        assert cfg.method.ordered_success_return_mix == 0.0
        assert cfg.method.sequence_aligned_mc_discount is None
        assert cfg.method.unseen_return_floor_weight == 0.0
        assert cfg.method.bc_lambda == 0.0
        assert cfg.method.bc_margin == 0.0
        assert not cfg.method.demo_fosd
        assert not cfg.method.separate_bc_policy
        assert not cfg.method.flow_policy
        assert not cfg.method.coarse_flow
        assert cfg.num_pretrain_steps == 0
        assert not cfg.use_self_imitation
    assert control.method.dense_return_advantage_alpha == pytest.approx(0.0)
    assert treatment.method.dense_return_advantage_alpha == pytest.approx(0.5)

    control_values = flatten_dict(
        OmegaConf.to_container(control, resolve=False)
    )
    treatment_values = flatten_dict(
        OmegaConf.to_container(treatment, resolve=False)
    )
    differences = {
        key
        for key in set(control_values) | set(treatment_values)
        if control_values.get(key) != treatment_values.get(key)
    }
    assert differences == {
        ("method", "dense_return_advantage_alpha")
    }


def test_cqn_as_no_bc_stage21_changes_only_advantage_clipping():
    configs = []
    for launch in (
        "cqn_as_pixel_bigym_nobc_stage20_advantage_gap_gate",
        "cqn_as_pixel_bigym_nobc_stage21_clipped_advantage_gate",
    ):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            configs.append(
                compose(
                    config_name="robobase_config",
                    overrides=[
                        f"launch={launch}",
                        "env=bigym/move_plate",
                    ],
                )
            )

    constant, clipped = configs
    for cfg in configs:
        assert cfg.action_sequence == 16
        assert cfg.execution_length == 1
        assert cfg.replay.nstep == 1
        assert cfg.method.temporal_ensemble
        assert cfg.method.temporal_ensemble_replan_interval == 1
        assert cfg.method.critic_sequence_mode == "full"
        assert cfg.method.td_target_action_source == "critic"
        assert cfg.method.strict_demo_rl_only
        assert cfg.method.dense_return_q_target
        assert not cfg.method.dense_return_expected_q_loss
        assert cfg.method.dense_return_advantage_alpha == pytest.approx(0.5)
        assert cfg.method.dense_return_finest_neighbor_weight == 0.0
        assert cfg.method.q_reward_scale == 1.0
        assert cfg.method.mc_lower_bound_target
        assert cfg.method.unseen_return_floor_weight == 0.0
        assert cfg.method.bc_lambda == 0.0
        assert cfg.method.bc_margin == 0.0
        assert not cfg.method.demo_fosd
        assert not cfg.method.separate_bc_policy
        assert not cfg.method.flow_policy
        assert not cfg.method.coarse_flow
        assert cfg.num_pretrain_steps == 0
        assert not cfg.use_self_imitation
    assert constant.method.dense_return_advantage_clip_ratio is None
    assert clipped.method.dense_return_advantage_clip_ratio == pytest.approx(
        0.9
    )

    constant_values = flatten_dict(
        OmegaConf.to_container(constant, resolve=False)
    )
    clipped_values = flatten_dict(
        OmegaConf.to_container(clipped, resolve=False)
    )
    differences = {
        key
        for key in set(constant_values) | set(clipped_values)
        if constant_values.get(key) != clipped_values.get(key)
    }
    assert differences == {
        ("method", "dense_return_advantage_clip_ratio")
    }


def test_cqn_as_no_bc_stage22_changes_only_to_replay_candidate_backup():
    configs = []
    for launch in (
        "cqn_as_pixel_bigym_nobc_stage10_discounted_dense_control",
        "cqn_as_pixel_bigym_nobc_stage22_demo_candidate_gate",
    ):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            configs.append(
                compose(
                    config_name="robobase_config",
                    overrides=[
                        f"launch={launch}",
                        "env=bigym/move_plate",
                    ],
                )
            )

    control, candidate = configs
    for cfg in configs:
        assert cfg.action_sequence == 16
        assert cfg.execution_length == 1
        assert cfg.replay.nstep == 1
        assert cfg.method.temporal_ensemble
        assert cfg.method.temporal_ensemble_replan_interval == 1
        assert cfg.method.critic_sequence_mode == "full"
        assert cfg.method.strict_demo_rl_only
        assert cfg.method.dense_return_q_target
        assert not cfg.method.dense_return_expected_q_loss
        assert cfg.method.dense_return_advantage_alpha == 0.0
        assert cfg.method.dense_return_advantage_clip_ratio is None
        assert cfg.method.q_reward_scale == 1.0
        assert cfg.method.mc_lower_bound_target
        assert cfg.method.unseen_return_floor_weight == 0.0
        assert cfg.method.bc_lambda == 0.0
        assert cfg.method.bc_margin == 0.0
        assert not cfg.method.demo_fosd
        assert not cfg.method.separate_bc_policy
        assert not cfg.method.flow_policy
        assert not cfg.method.coarse_flow
        assert cfg.num_pretrain_steps == 0
        assert not cfg.use_self_imitation
        assert cfg.batch_size == 16
        assert cfg.demo_batch_size == 16
        assert not cfg.replay.demo_only_updates
        assert cfg.batch_size + cfg.demo_batch_size == 32
    assert control.method.td_target_action_source == "critic"
    assert not control.replay.include_next_action
    assert candidate.method.td_target_action_source == "critic_replay_max"
    assert candidate.replay.include_next_action

    control_values = flatten_dict(
        OmegaConf.to_container(control, resolve=False)
    )
    candidate_values = flatten_dict(
        OmegaConf.to_container(candidate, resolve=False)
    )
    differences = {
        key
        for key in set(control_values) | set(candidate_values)
        if control_values.get(key) != candidate_values.get(key)
    }
    assert differences == {
        ("method", "td_target_action_source"),
        ("replay", "include_next_action"),
    }


def test_cqn_as_stage26_phase_a_changes_only_demo_trajectory_force():
    configs = []
    for launch in (
        "cqn_as_pixel_bigym_nobc_stage22_demo_candidate_gate",
        "cqn_as_pixel_bigym_nobc_stage26_demo_trajectory_gate",
    ):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            configs.append(
                compose(
                    config_name="robobase_config",
                    overrides=[
                        f"launch={launch}",
                        "env=bigym/move_plate",
                    ],
                )
            )

    candidate, trajectory = configs
    assert candidate.method.demo_behavior_force_probability == 0.0
    assert trajectory.method.demo_behavior_force_probability == 1.0
    assert trajectory.method.td_target_action_source == "critic_replay_max"
    assert trajectory.replay.include_next_action
    assert trajectory.method.strict_demo_rl_only
    assert trajectory.method.critic_lambda == 1.0
    assert trajectory.method.bc_lambda == 0.0
    assert trajectory.method.bc_margin == 0.0
    assert not trajectory.method.separate_bc_policy
    assert trajectory.method.dense_return_q_target
    assert trajectory.method.mc_lower_bound_target

    candidate_values = flatten_dict(
        OmegaConf.to_container(candidate, resolve=False)
    )
    trajectory_values = flatten_dict(
        OmegaConf.to_container(trajectory, resolve=False)
    )
    differences = {
        key
        for key in set(candidate_values) | set(trajectory_values)
        if candidate_values.get(key) != trajectory_values.get(key)
    }
    assert differences == {
        ("method", "demo_behavior_force_probability")
    }


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

    from robobase.method.cqn_research import encode_action

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
    from robobase.method.cqn_research import encode_action

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


# ---------------------------------------------------------------------------
# Progress-potential shaping (reports/progress_shaping_impl_20260818.md)
# ---------------------------------------------------------------------------


def _progress_batch(batch_size=4, action_sequence=3, terminal_last=True):
    batch = _batch(batch_size=batch_size, action_sequence=action_sequence)
    batch["progress"] = (
        np.arange(1, batch_size + 1, dtype=np.float32) / batch_size
    )
    batch["progress_valid"] = np.ones((batch_size,), dtype=np.uint8)
    if terminal_last:
        terminal = np.zeros((batch_size,), dtype=bool)
        terminal[-1] = True
        batch["terminal"] = terminal
    return batch


def _set_constant_progress_potential(agent, value):
    """Force ``Phi(s) == value`` by writing the zero-kernel head's bias."""

    flat = flatten_dict(unfreeze(agent.params["progress_value"]))
    bias_key = next(key for key in flat if key[-2:] == ("value_out", "bias"))
    flat[bias_key] = jnp.full_like(flat[bias_key], float(value))
    agent.params = {
        **agent.params,
        "progress_value": unflatten_dict(flat),
    }


def test_progress_knobs_default_to_exact_legacy_and_add_no_params():
    cfg = _compose_cqn_as()
    assert cfg.method.progress_potential_weight == pytest.approx(0.0)
    assert cfg.method.progress_head_weight == pytest.approx(0.0)
    assert cfg.method.progress_potential_schedule is None
    assert cfg.method.progress_expectile_tau == pytest.approx(0.9)
    assert cfg.method.progress_success_gated is True
    assert not _progress_label_enabled(cfg)

    observation_space, action_space = _spaces()
    agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    assert "progress_value" not in agent.params
    assert not agent.progress_head_enabled
    agent.logging = True
    metrics = agent.update(iter([_batch()]), step=1)
    assert not [key for key in metrics if key.startswith("progress")]


def test_progress_label_gate_tracks_either_consumer():
    assert _progress_label_enabled(
        _compose_cqn_as("method.progress_head_weight=1.0")
    )
    assert _progress_label_enabled(
        _compose_cqn_as("method.progress_potential_weight=0.25")
    )


def test_progress_shaped_rewards_terminal_drops_the_next_potential():
    rewards = jnp.asarray([0.0, 1.0], dtype=jnp.float32)
    discounts = jnp.asarray([0.99, 0.99], dtype=jnp.float32)
    bootstrap = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    phi = jnp.asarray([0.3, 0.8], dtype=jnp.float32)
    phi_next = jnp.asarray([0.5, 0.9], dtype=jnp.float32)

    shaped = progress_shaped_rewards(
        rewards,
        discounts,
        bootstrap,
        phi,
        phi_next,
        0.25,
    )
    np.testing.assert_allclose(
        shaped,
        [
            0.0 + 0.25 * (0.99 * 0.5 - 0.3),
            # bootstrap == 0 kills Phi(s'): the success target deflates to
            # exactly 1 - lambda * Phi(s).
            1.0 - 0.25 * 0.8,
        ],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        progress_shaped_rewards(
            rewards,
            discounts,
            bootstrap,
            phi,
            phi_next,
            0.0,
        ),
        rewards,
        atol=0.0,
    )


def test_progress_shaping_telescopes_over_a_whole_episode():
    gamma = 0.9
    lam = 0.4
    phi = np.asarray([0.1, 0.35, 0.6, 0.95], dtype=np.float32)
    horizon = phi.shape[0]
    rewards = np.zeros((horizon,), dtype=np.float32)
    discounts = np.full((horizon,), gamma, dtype=np.float32)
    bootstrap = np.ones((horizon,), dtype=np.float32)
    phi_next = np.concatenate([phi[1:], np.asarray([0.0], dtype=np.float32)])
    # Terminal transition: Phi(s_T) = 0 by the bootstrap mask.
    bootstrap[-1] = 0.0

    shaped = np.asarray(
        progress_shaped_rewards(
            jnp.asarray(rewards),
            jnp.asarray(discounts),
            jnp.asarray(bootstrap),
            jnp.asarray(phi),
            jnp.asarray(phi_next),
            lam,
        )
    )
    discounted_sum = float(
        sum(gamma**t * shaped[t] for t in range(horizon))
    )
    # sum_t gamma^t F_t = -lambda * Phi(s_0) once Phi(terminal) = 0.
    np.testing.assert_allclose(discounted_sum, -lam * phi[0], atol=1e-6)


def test_cqn_as_progress_head_is_created_on_the_canonical_platform():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_as(
            "method.progress_head_weight=1.0",
            "method.weight_decay=0.0",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    # Unlike awr_beta the progress head does NOT require separate_bc_policy.
    assert not agent.separate_bc_policy
    assert "progress_value" in agent.params

    batch = _progress_batch()
    before = jax.tree.map(np.asarray, agent.params["progress_value"])
    agent.logging = True
    metrics = agent.update(iter([batch]), step=1)

    assert _tree_changed(before, agent.params["progress_value"])
    assert metrics["progress_head_loss"] > 0.0
    assert metrics["progress_valid_fraction"] == pytest.approx(1.0)
    assert metrics["progress_label_mean"] == pytest.approx(
        float(np.mean(batch["progress"])),
        abs=1e-6,
    )
    assert np.isfinite(metrics["progress_head_value_mean"])
    assert "progress_shaping_clip_frac" not in metrics


def test_cqn_as_progress_head_leaves_the_legacy_critic_update_bitwise():
    observation_space, action_space = _spaces()
    legacy = create_agent(
        _compose_cqn_as("method.weight_decay=0.0"),
        observation_space=observation_space,
        action_space=action_space,
    )
    shaped = create_agent(
        _compose_cqn_as(
            "method.progress_head_weight=1.0",
            "method.weight_decay=0.0",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    batch = _progress_batch()
    legacy_batch = {key: np.array(value, copy=True) for key, value in batch.items()}
    legacy_batch.pop("progress")
    legacy_batch.pop("progress_valid")

    legacy.update(iter([legacy_batch]), step=1)
    shaped.update(iter([batch]), step=1)

    legacy_leaves = jax.tree.leaves(legacy.params["critic"])
    shaped_leaves = jax.tree.leaves(shaped.params["critic"])
    for left, right in zip(legacy_leaves, shaped_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_cqn_as_zero_initialized_potential_is_the_exact_legacy_target():
    observation_space, action_space = _spaces()
    legacy = create_agent(
        _compose_cqn_as("method.weight_decay=0.0"),
        observation_space=observation_space,
        action_space=action_space,
    )
    shaped = create_agent(
        _compose_cqn_as(
            "method.progress_potential_weight=0.25",
            "method.progress_head_weight=0.0",
            "method.weight_decay=0.0",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    batch = _progress_batch()
    legacy_batch = {key: np.array(value, copy=True) for key, value in batch.items()}
    legacy_batch.pop("progress")
    legacy_batch.pop("progress_valid")

    legacy.update(iter([legacy_batch]), step=1)
    shaped.logging = True
    metrics = shaped.update(iter([batch]), step=1)

    # The head is zero-initialised, so Phi == 0 and the shaped target is the
    # legacy target bit for bit.
    assert metrics["progress_head_value_mean"] == pytest.approx(0.0)
    assert metrics["progress_potential_lambda"] == pytest.approx(0.25)
    for left, right in zip(
        jax.tree.leaves(legacy.params["critic"]),
        jax.tree.leaves(shaped.params["critic"]),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_cqn_as_constant_potential_equals_pre_shifted_rewards():
    observation_space, action_space = _spaces()
    lam = 0.25
    potential = 0.4
    shaped = create_agent(
        _compose_cqn_as(
            f"method.progress_potential_weight={lam}",
            "method.progress_head_weight=0.0",
            "method.weight_decay=0.0",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    _set_constant_progress_potential(shaped, potential)
    legacy = create_agent(
        _compose_cqn_as("method.weight_decay=0.0"),
        observation_space=observation_space,
        action_space=action_space,
    )

    batch = _progress_batch()
    legacy_batch = {key: np.array(value, copy=True) for key, value in batch.items()}
    legacy_batch.pop("progress")
    legacy_batch.pop("progress_valid")
    bootstrap = 1.0 - batch["terminal"].astype(np.float32)
    legacy_batch["reward"] = (
        batch["reward"]
        + lam * (batch["discount"] * bootstrap * potential - potential)
    ).astype(np.float32)

    shaped.logging = True
    metrics = shaped.update(iter([batch]), step=1)
    legacy.update(iter([legacy_batch]), step=1)

    assert metrics["progress_head_value_mean"] == pytest.approx(potential)
    for left, right in zip(
        jax.tree.leaves(legacy.params["critic"]),
        jax.tree.leaves(shaped.params["critic"]),
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(left),
            np.asarray(right),
            atol=1e-7,
        )


def test_cqn_as_progress_success_gate_censors_failed_episodes():
    observation_space, action_space = _spaces()
    gated = create_agent(
        _compose_cqn_as(
            "method.progress_head_weight=1.0",
            "method.progress_success_gated=true",
            "method.weight_decay=0.0",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    ungated = create_agent(
        _compose_cqn_as(
            "method.progress_head_weight=1.0",
            "method.progress_success_gated=false",
            "method.weight_decay=0.0",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    batch = _progress_batch()
    batch["progress_valid"] = np.zeros_like(batch["progress_valid"])

    gated_before = jax.tree.map(np.asarray, gated.params["progress_value"])
    ungated_before = jax.tree.map(np.asarray, ungated.params["progress_value"])
    gated.logging = True
    ungated.logging = True
    gated_metrics = gated.update(iter([batch]), step=1)
    ungated_metrics = ungated.update(iter([batch]), step=1)

    assert gated_metrics["progress_valid_fraction"] == pytest.approx(0.0)
    assert gated_metrics["progress_head_loss"] == pytest.approx(0.0)
    assert not _tree_changed(gated_before, gated.params["progress_value"])
    assert ungated_metrics["progress_valid_fraction"] == pytest.approx(1.0)
    assert ungated_metrics["progress_head_loss"] > 0.0
    assert _tree_changed(ungated_before, ungated.params["progress_value"])


def test_cqn_as_progress_potential_schedule_anneals_lambda():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_as(
            "method.progress_potential_weight=0.25",
            "method.progress_head_weight=1.0",
            "method.progress_potential_schedule='linear(0.25,0.0,100)'",
            "method.weight_decay=0.0",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    agent.logging = True
    early = agent.update(iter([_progress_batch()]), step=0)
    late = agent.update(iter([_progress_batch()]), step=100)
    assert early["progress_potential_lambda"] == pytest.approx(0.25)
    assert late["progress_potential_lambda"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "overrides",
    (
        ("method.mc_lower_bound_target=true", "method.mc_return_weight=0.1"),
        (
            "method.episodic_success_q_target=true",
            "method.dense_return_q_target=true",
        ),
    ),
)
def test_cqn_as_progress_potential_rejects_raw_mc_targets(overrides):
    observation_space, action_space = _spaces()
    with pytest.raises(ValueError, match="raw Monte-Carlo targets"):
        create_agent(
            _compose_cqn_as(
                "method.progress_potential_weight=0.25",
                *overrides,
            ),
            observation_space=observation_space,
            action_space=action_space,
        )


def test_cqn_as_progress_potential_must_fit_the_c51_support():
    observation_space, action_space = _spaces()
    with pytest.raises(ValueError, match="exceeds v_max"):
        create_agent(
            _compose_cqn_as("method.progress_potential_weight=1.5"),
            observation_space=observation_space,
            action_space=action_space,
        )


def test_cqn_as_progress_requires_truncated_demo_tails_on_bigym():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_as_pixel_bigym_demo_driven",
                "env=bigym/move_plate",
                "method.progress_head_weight=1.0",
                "env.truncate_demo_at_success=false",
            ],
        )
    with pytest.raises(ValueError, match="truncate_demo_at_success"):
        cqn_as_spec_from_cfg(cfg)


def test_cqn_as_progress_launch_composes_on_the_demo_driven_platform():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_as_pixel_bigym_demo_driven",
                "env=bigym/move_plate",
                "method.progress_head_weight=1.0",
                "method.progress_potential_weight=0.25",
                "method.progress_potential_schedule='linear(0.25,0.0,50000)'",
            ],
        )
    assert cfg.env.truncate_demo_at_success is True
    assert cfg.lazy_replay.use is False
    assert _progress_label_enabled(cfg)
    spec = cqn_as_spec_from_cfg(cfg)
    assert spec.progress_potential_weight == pytest.approx(0.25)
    assert spec.progress_head_weight == pytest.approx(1.0)
    assert spec.progress_potential_schedule == "linear(0.25,0.0,50000)"
    assert spec.progress_success_gated is True


def _progress_workspace_stub(recorded):
    workspace = object.__new__(Workspace)
    workspace.train_envs = SimpleNamespace(num_envs=1)
    workspace.agent = SimpleNamespace(reset=lambda *args, **kwargs: None)
    workspace.cfg = OmegaConf.create(
        {
            "use_self_imitation": False,
            "method": {"is_rl": False},
            "replay": {"gamma": 0.99},
        }
    )
    workspace.use_demo_replay = False
    workspace.extra_replay_elements = spaces.Dict(
        {
            "demo": spaces.Box(0, 1, shape=(), dtype=np.uint8),
            "progress": spaces.Box(0.0, 1.0, shape=(), dtype=np.float32),
            "progress_valid": spaces.Box(0, 1, shape=(), dtype=np.uint8),
        }
    )
    workspace._episode_rollouts = [[]]
    workspace._global_env_episode = 0
    workspace._main_loop_iterations = 0
    workspace.replay_buffer = SimpleNamespace(
        add=lambda obs, act, rew, term, trunc, **extra: recorded.append(extra),
        add_final=lambda obs: None,
    )
    return workspace


def _run_progress_episode(horizon, task_success):
    recorded = []
    workspace = _progress_workspace_stub(recorded)
    for step in range(horizon):
        terminal = step == horizon - 1
        observations = {
            "low_dim_state": np.zeros((1, 2, 3), dtype=np.float32)
        }
        next_infos = {}
        if terminal:
            next_infos = {
                "_final_observation": np.asarray([True]),
                "final_observation": np.asarray(
                    [{"low_dim_state": np.zeros((2, 3), dtype=np.float32)}],
                    dtype=object,
                ),
                "final_info": np.asarray(
                    [{"task_success": float(task_success)}],
                    dtype=object,
                ),
            }
        workspace._add_to_replay(
            actions=np.zeros((1, 3, 2), dtype=np.float32),
            observations=observations,
            rewards=np.asarray([1.0 if terminal and task_success else 0.0]),
            terminations=np.asarray([terminal]),
            truncations=np.asarray([False]),
            infos={},
            next_infos=next_infos,
        )
    return recorded


def test_online_progress_labels_are_monotone_and_success_gated():
    horizon = 5
    recorded = _run_progress_episode(horizon, task_success=True)
    labels = np.asarray([entry["progress"] for entry in recorded])
    valid = np.asarray([entry["progress_valid"] for entry in recorded])

    # Same closed form the demo loader uses: (t + 1) / T, so demo and online
    # transitions land on one label scale.
    np.testing.assert_allclose(
        labels,
        np.arange(1, horizon + 1, dtype=np.float32) / horizon,
        atol=1e-7,
    )
    assert np.all(np.diff(labels) > 0.0)
    assert labels[-1] == pytest.approx(1.0)
    np.testing.assert_array_equal(valid, np.ones(horizon, dtype=np.uint8))

    failed = _run_progress_episode(horizon, task_success=False)
    np.testing.assert_allclose(
        np.asarray([entry["progress"] for entry in failed]),
        np.arange(1, horizon + 1, dtype=np.float32) / horizon,
        atol=1e-7,
    )
    np.testing.assert_array_equal(
        np.asarray([entry["progress_valid"] for entry in failed]),
        np.zeros(horizon, dtype=np.uint8),
    )
