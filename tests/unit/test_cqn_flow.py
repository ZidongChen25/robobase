from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnasium import spaces
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from robobase.factory import create_agent
from robobase.method.cqn_flow import (
    C2FSequenceFlowCritic,
    aggregate_return_samples,
    categorical_cross_entropy,
    centered_log_probabilities,
    cqn_flow_spec_from_cfg,
    demo_fosd_per_sample,
    demo_margin_per_sample,
    expected_q,
    evor_velocity_td_pair,
    flow_logits_to_probabilities,
    hl_gauss_encode,
    integrate_value_flow,
    integrate_value_flow_trajectory,
    integrate_value_flow_with_source_jvp,
    quantile_couple_return_samples,
    quantile_huber_endpoint_loss,
    scalar_flow_trajectory_diagnostics,
    scalar_to_categorical,
    select_single_supported_lcb_plan,
    sibling_bin_candidate_plans,
    source_bin_flip_rate_per_sample,
    supported_lcb_action_indices,
)
from robobase.replay_buffer.vision_feature_cache import JAX_CQN_AS_FEATURE_KEY
from robobase.workspace import (
    _mc_return_anchor_enabled,
    _structured_exploration_enabled,
    _validate_rl_action_sequence,
)


CONFIG_DIR = str((Path(__file__).parents[2] / "robobase" / "cfgs").resolve())


def _compose_cqn_flow(*overrides):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(
            config_name="robobase_config",
            overrides=[
                "method=cqn_flow",
                "action_sequence=2",
                "num_train_envs=1",
                "num_eval_envs=1",
                "backend.jit=false",
                "backend.platform=cpu",
                "method.model.hidden_dims=[16]",
                "method.query_hidden_dim=8",
                "method.time_embed_dim=4",
                "method.atoms=5",
                "method.levels=1",
                "method.num_flow_steps=1",
                "method.num_flow_samples=1",
                *overrides,
            ],
        )


def _spaces(action_sequence=2):
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


def _batch(batch_size=2, action_sequence=2):
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


def _tree_exactly_equal(before, after):
    before_leaves, before_tree = jax.tree.flatten(before)
    after_leaves, after_tree = jax.tree.flatten(after)
    assert before_tree == after_tree
    return all(
        np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(before_leaves, after_leaves, strict=True)
    )


def _tree_all_finite(tree):
    return all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree.leaves(tree))


def _flow_critic_inputs(*, value_dim, query_bins=5):
    batch, sources, sequence, action_dim = 2, 3, 4, 2
    features = jnp.zeros((batch, 7), dtype=jnp.float32)
    level = jnp.zeros((batch, 3), dtype=jnp.float32)
    midpoint = jnp.zeros((batch, sequence, action_dim), dtype=jnp.float32)
    half_width = jnp.ones_like(midpoint)
    candidate_bins = jnp.broadcast_to(
        jnp.arange(query_bins, dtype=jnp.int32)[None, None, None, :],
        (batch, sequence, action_dim, query_bins),
    )
    candidate_centers = candidate_bins.astype(jnp.float32)
    flow_values = jnp.zeros(
        (batch, sources, sequence, action_dim, query_bins, value_dim),
        dtype=jnp.float32,
    )
    tau = jnp.ones((batch, sources), dtype=jnp.float32)
    return (
        features,
        level,
        midpoint,
        half_width,
        candidate_bins,
        candidate_centers,
        flow_values,
        tau,
    )


def test_value_flow_trajectory_endpoint_matches_endpoint_integrator():
    source = jnp.linspace(-0.8, 0.8, 24, dtype=jnp.float32).reshape(
        (2, 3, 2, 2, 1)
    )

    def velocity(value, tau):
        tau = jnp.asarray(tau, dtype=value.dtype).reshape(
            (*jnp.shape(tau), *((1,) * (value.ndim - jnp.ndim(tau))))
        )
        return 0.2 * value + 0.3 * tau

    endpoint = integrate_value_flow(
        velocity,
        source,
        num_flow_steps=4,
        end_tau=0.1,
        clip_min=-1.0,
        clip_max=1.0,
    )
    trajectory = integrate_value_flow_trajectory(
        velocity,
        source,
        num_flow_steps=4,
        end_tau=0.1,
        clip_min=-1.0,
        clip_max=1.0,
    )

    np.testing.assert_array_equal(trajectory[0], source)
    np.testing.assert_array_equal(trajectory[-1], endpoint)
    assert trajectory.shape == (5, *source.shape)


def test_evor_velocity_td_pair_matches_equations_and_detaches_target():
    source = jnp.asarray(
        [
            [[[[[1.0]]]], [[[[2.0]]]]],
            [[[[[-1.0]]]], [[[[0.5]]]]],
        ],
        dtype=jnp.float32,
    )
    endpoint = source + 2.0
    next_velocity = jnp.full_like(source, 0.4)
    reward = jnp.asarray([0.2, 0.7], dtype=jnp.float32)
    discount = jnp.asarray([0.9, 0.0], dtype=jnp.float32)
    tau = jnp.asarray([[0.25, 0.75], [0.5, 0.1]], dtype=jnp.float32)

    pair = evor_velocity_td_pair(
        source,
        endpoint,
        reward,
        discount,
        next_velocity,
        tau,
    )

    tau_broadcast = tau.reshape((2, 2, 1, 1, 1, 1))
    np.testing.assert_allclose(
        pair.current_sample,
        tau_broadcast * source + (1.0 - tau_broadcast) * endpoint,
        atol=1e-7,
    )
    expected_target = reward[:, None, None, None, None, None] + (
        discount[:, None, None, None, None, None] * next_velocity
    )
    np.testing.assert_allclose(pair.target_velocity, expected_target)
    target_grad = jax.grad(
        lambda velocity: evor_velocity_td_pair(
            source,
            endpoint,
            reward,
            discount,
            velocity,
            tau,
        ).target_velocity.sum()
    )(next_velocity)
    np.testing.assert_array_equal(target_grad, jnp.zeros_like(next_velocity))


def test_evor_loss_uses_offline_return_endpoint_and_shared_interpolant(
    monkeypatch,
):
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(
            "method.value_mode=return_sample",
            "method.atom_ce_lambda=0",
            "method.demo_fosd=false",
            "method.bcfm_lambda=0",
            "method.evor_td_lambda=1",
            "method.separate_bc_policy=true",
            "method.td_target_action_source=bc_policy",
            "method.num_flow_samples=1",
            "method.num_target_flow_samples=1",
            "method.flow_source_type=gaussian",
            "method.mc_return_weight=0",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    batch_size = 2
    source = jnp.arange(
        batch_size
        * agent.num_flow_samples
        * agent.action_sequence
        * agent.action_dim,
        dtype=jnp.float32,
    ).reshape(
        (
            batch_size,
            agent.num_flow_samples,
            agent.action_sequence,
            agent.action_dim,
            1,
            1,
        )
    ) / 10.0
    tau = jnp.asarray([[0.25], [0.75]], dtype=jnp.float32)
    calls = []

    def fake_flow_source(_key, batch_size_arg, query_bins, num_samples):
        assert batch_size_arg == batch_size
        assert query_bins == 1
        assert num_samples == 1
        return source

    class GoldenCritic:
        def apply(
            self,
            params,
            features,
            level_one_hot,
            midpoint,
            half_width,
            candidate_bins,
            centers,
            values,
            time,
            source_quantiles,
        ):
            del (
                features,
                level_one_hot,
                midpoint,
                half_width,
                candidate_bins,
                centers,
                source_quantiles,
            )
            calls.append((params["bias"], values, time))
            return 0.25 * values + params["bias"]

    monkeypatch.setattr(agent, "_flow_source", fake_flow_source)
    monkeypatch.setattr(agent, "critic_model", GoldenCritic())
    monkeypatch.setattr(
        jax.random,
        "uniform",
        lambda *_args, **_kwargs: tau,
    )
    features = jnp.zeros((batch_size, 5), dtype=jnp.float32)
    actions = jnp.zeros(
        (batch_size, agent.action_sequence, agent.action_dim),
        dtype=jnp.float32,
    )
    rewards = jnp.asarray([0.2, 0.7], dtype=jnp.float32)
    discounts = jnp.asarray([0.9, 0.9], dtype=jnp.float32)
    bootstrap = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    mc_returns = jnp.asarray([0.8, -0.2], dtype=jnp.float32)
    current_params = {"bias": jnp.asarray(0.3, dtype=jnp.float32)}
    target_params = {"bias": jnp.asarray(-0.1, dtype=jnp.float32)}

    loss = agent._evor_td_loss(
        current_params,
        target_params,
        features,
        features,
        actions,
        actions,
        rewards,
        discounts,
        bootstrap,
        mc_returns,
        jax.random.PRNGKey(3),
        jax.random.PRNGKey(4),
    )

    endpoint = jnp.broadcast_to(
        mc_returns[:, None, None, None, None, None],
        source.shape,
    )
    expected_sample = (
        tau[..., None, None, None, None] * source
        + (1.0 - tau[..., None, None, None, None]) * endpoint
    )
    expected_next = 0.25 * expected_sample + target_params["bias"]
    expected_target = rewards[:, None, None, None, None, None] + (
        (discounts * bootstrap)[:, None, None, None, None, None]
        * expected_next
    )
    expected_prediction = 0.25 * expected_sample + current_params["bias"]
    expected_loss = jnp.square(
        expected_prediction - expected_target
    ).mean(axis=(1, 2, 3, 4, 5))

    np.testing.assert_allclose(loss, expected_loss, atol=1e-6)
    assert len(calls) == 2 * agent.levels
    for _, values, time in calls:
        np.testing.assert_allclose(values, expected_sample, atol=1e-7)
        np.testing.assert_array_equal(time, tau)


def test_scalar_flow_trajectory_diagnostics_are_zero_for_straight_path():
    source = jnp.asarray(
        [[[-1.0], [1.0]], [[-0.5], [0.5]]],
        dtype=jnp.float32,
    )[:, :, None, :]
    endpoint = source * 0.25 + 0.7
    trajectory = jnp.stack(
        [
            source,
            source + 0.5 * (endpoint - source),
            endpoint,
        ],
        axis=0,
    )

    metrics = scalar_flow_trajectory_diagnostics(trajectory)

    np.testing.assert_allclose(metrics["curvature_rms"], 0.0, atol=1e-7)
    np.testing.assert_allclose(
        metrics["normalized_curvature_rms"], 0.0, atol=1e-7
    )
    np.testing.assert_allclose(
        metrics["normalized_increment_variation"], 0.0, atol=1e-7
    )
    np.testing.assert_allclose(
        metrics["source_contraction_ratio"], 0.25, atol=2e-6
    )


def test_scalar_flow_trajectory_diagnostics_detect_curved_path():
    source = jnp.asarray(
        [[[-1.0], [1.0]], [[-0.5], [0.5]]],
        dtype=jnp.float32,
    )[:, :, None, :]
    endpoint = source * 0.25 + 0.7
    curved_midpoint = source + 0.5 * (endpoint - source) + 0.2
    trajectory = jnp.stack([source, curved_midpoint, endpoint], axis=0)

    metrics = scalar_flow_trajectory_diagnostics(trajectory)

    assert float(metrics["curvature_rms"]) > 0.19
    assert float(metrics["normalized_curvature_rms"]) > 0.0
    assert float(metrics["normalized_increment_variation"]) > 0.0


def test_flow_utilization_probe_is_read_only_and_detects_zero_velocity():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(
            "method.value_mode=scalar",
            "method.atom_ce_lambda=0",
            "method.demo_fosd=false",
            "method.num_flow_steps=2",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    observations = {
        "low_dim_state": np.zeros((2, 1, 5), dtype=np.float32),
    }
    actions = np.zeros((2, 2, 2), dtype=np.float32)
    rng_before = np.asarray(agent.rng_key).copy()

    metrics = agent.flow_utilization_probe(
        observations,
        actions,
        num_source_samples=2,
        step_counts=(1, 2, 4),
        seed=41,
        use_target_network=False,
    )

    assert _tree_all_finite(metrics)
    np.testing.assert_array_equal(agent.rng_key, rng_before)
    np.testing.assert_array_equal(metrics["step_counts"], [1, 2, 4])
    np.testing.assert_allclose(
        metrics["per_level_normalized_curvature_rms"], [0.0], atol=1e-7
    )
    np.testing.assert_allclose(
        metrics["per_level_source_contraction_ratio"], [1.0], atol=2e-5
    )
    np.testing.assert_allclose(
        metrics["per_level_step_ranking_agreement"],
        [[1.0, 1.0, 1.0]],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        metrics["per_level_step_q_rmse"], 0.0, atol=1e-7
    )


def test_flow_utilization_probe_requires_explicit_action_without_bc_policy():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(
            "method.value_mode=scalar",
            "method.atom_ce_lambda=0",
            "method.demo_fosd=false",
            "method.num_flow_steps=2",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )

    with pytest.raises(ValueError, match="independent BC policy"):
        agent.flow_utilization_probe(
            {"low_dim_state": np.zeros((1, 1, 5), dtype=np.float32)},
            num_source_samples=2,
        )


def test_evor_flowtd_update_isolated_from_bcfm_and_updates_critic():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(
            "method.value_mode=return_sample",
            "method.atom_ce_lambda=0",
            "method.demo_fosd=false",
            "method.bcfm_lambda=0",
            "method.dcfm_lambda=0",
            "method.evor_td_lambda=1",
            "method.separate_bc_policy=true",
            "method.td_target_action_source=bc_policy",
            "method.num_flow_samples=1",
            "method.num_target_flow_samples=1",
            "method.num_action_flow_samples=2",
            "method.flow_source_type=gaussian",
            "method.antithetic_flow_sources=false",
            "method.mc_return_weight=0",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    critic_before = jax.tree.map(np.asarray, agent.params["critic"])
    agent.logging = True

    batch = _batch(batch_size=2)
    batch["mc_return"] = np.asarray([0.8, 0.2], dtype=np.float32)
    metrics = agent.update(iter([batch]), step=1)

    assert _tree_all_finite(metrics)
    assert float(metrics["evor_td_loss"]) > 0.0
    np.testing.assert_allclose(metrics["bcfm_loss"], 0.0, atol=1e-8)
    np.testing.assert_allclose(metrics["dcfm_loss"], 0.0, atol=1e-8)
    assert metrics["mc_return_mean"] == pytest.approx(0.5)
    assert _tree_changed(critic_before, agent.params["critic"])


def test_evor_flowtd_rejects_non_bc_bellman_action_source():
    observation_space, action_space = _spaces()

    with pytest.raises(ValueError, match="td_target_action_source=bc_policy"):
        create_agent(
            _compose_cqn_flow(
                "method.value_mode=return_sample",
                "method.atom_ce_lambda=0",
                "method.demo_fosd=false",
                "method.bcfm_lambda=0",
                "method.evor_td_lambda=1",
                "method.separate_bc_policy=true",
                "method.td_target_action_source=replay_next",
                "method.num_flow_samples=1",
                "method.num_target_flow_samples=1",
                "method.flow_source_type=gaussian",
            ),
            observation_space=observation_space,
            action_space=action_space,
        )


def test_supported_lcb_gate_overrides_only_with_agreement_and_bc_support():
    policy_logits = jnp.asarray([[2.0, 1.7, -2.0]], dtype=jnp.float32)
    advantages = jnp.asarray(
        [
            [[0.0, 1.2, 10.0]],
            [[0.0, 1.0, 10.0]],
            [[0.0, 0.8, 10.0]],
        ],
        dtype=jnp.float32,
    )

    result = supported_lcb_action_indices(
        policy_logits,
        advantages,
        lcb_scale=1.0,
        min_lcb_margin=0.5,
        max_bc_logprob_drop=0.5,
    )

    # Bin 2 has a much larger predicted value but lies outside BC support.
    np.testing.assert_array_equal(result.support_mask, [[True, True, False]])
    np.testing.assert_array_equal(result.bc_indices, [0])
    np.testing.assert_array_equal(result.indices, [1])
    np.testing.assert_array_equal(result.override_mask, [True])


def test_supported_lcb_gate_falls_back_when_critics_disagree():
    policy_logits = jnp.asarray([[2.0, 1.9]], dtype=jnp.float32)
    advantages = jnp.asarray(
        [
            [[0.0, 2.0]],
            [[0.0, 0.2]],
            [[0.0, -1.0]],
        ],
        dtype=jnp.float32,
    )

    result = supported_lcb_action_indices(
        policy_logits,
        advantages,
        lcb_scale=1.0,
        min_lcb_margin=0.1,
        max_bc_logprob_drop=1.0,
    )

    assert float(result.lcb_delta[0, 1]) < 0.1
    np.testing.assert_array_equal(result.indices, result.bc_indices)
    np.testing.assert_array_equal(result.override_mask, [False])


def test_sibling_candidates_repeat_exact_delta_over_requested_horizon():
    baseline = jnp.asarray(
        [[[0.23, -0.4], [0.4, -0.2], [0.6, 0.0]]],
        dtype=jnp.float32,
    )

    candidates, deltas = sibling_bin_candidate_plans(
        baseline,
        jnp.asarray([-1.0, -1.0]),
        jnp.asarray([1.0, 1.0]),
        bins=5,
        force_level=1,
        intervention_horizon=2,
    )

    assert candidates.shape == (1, 2, 5, 3, 2)
    np.testing.assert_allclose(
        candidates[0, 0, :, :2, 0] - baseline[0, None, :2, 0],
        np.broadcast_to(np.asarray(deltas[0, 0, :, None]), (5, 2)),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        candidates[0, 0, :, 2],
        np.broadcast_to(np.asarray(baseline[0, 2]), (5, 2)),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        candidates[0, 0, :, :, 1],
        np.broadcast_to(np.asarray(baseline[0, :, 1]), (5, 3)),
        atol=1e-7,
    )


def test_supported_lcb_plan_applies_only_highest_lcb_dimension():
    baseline = jnp.zeros((1, 2, 2), dtype=jnp.float32)
    candidates = jnp.broadcast_to(
        baseline[:, None, None],
        (1, 2, 3, 2, 2),
    )
    candidates = candidates.at[0, 0, 1, :, 0].set(0.25)
    candidates = candidates.at[0, 1, 2, :, 1].set(0.75)
    policy_scores = jnp.asarray(
        [[[2.0, 1.9, -2.0], [2.0, 1.9, 1.8]]],
        dtype=jnp.float32,
    )
    advantages = jnp.asarray(
        [
            [[[0.0, 0.6, 10.0], [0.0, 0.5, 1.0]]],
            [[[0.0, 0.7, 10.0], [0.0, 0.6, 1.1]]],
            [[[0.0, 0.8, 10.0], [0.0, 0.7, 1.2]]],
        ],
        dtype=jnp.float32,
    )

    result = select_single_supported_lcb_plan(
        baseline,
        candidates,
        policy_scores,
        advantages,
        lcb_scale=1.0,
        min_lcb_margin=0.2,
        max_bc_logprob_drop=0.5,
    )

    # Both dimensions have an eligible override, but dimension 1 has the
    # larger pessimistic improvement and is the only intervention applied.
    np.testing.assert_array_equal(
        result.eligible_override_mask,
        [[True, True]],
    )
    np.testing.assert_array_equal(result.applied_override, [True])
    np.testing.assert_array_equal(result.selected_dimension, [1])
    np.testing.assert_allclose(result.action[0, :, 0], 0.0)
    np.testing.assert_allclose(result.action[0, :, 1], 0.75)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"lcb_scale": -1.0}, "lcb_scale"),
        ({"min_lcb_margin": -1.0}, "min_lcb_margin"),
        ({"max_bc_logprob_drop": -1.0}, "max_bc_logprob_drop"),
    ],
)
def test_supported_lcb_gate_rejects_negative_thresholds(kwargs, match):
    settings = {
        "lcb_scale": 1.0,
        "min_lcb_margin": 0.0,
        "max_bc_logprob_drop": 1.0,
    }
    settings.update(kwargs)
    with pytest.raises(ValueError, match=match):
        supported_lcb_action_indices(
            jnp.zeros((1, 2)),
            jnp.zeros((2, 1, 2)),
            **settings,
        )


@pytest.mark.parametrize("value_dim", [11, 1])
def test_sequence_flow_critic_vectorizes_sources_steps_dims_and_bins(value_dim):
    critic = C2FSequenceFlowCritic(
        hidden_dims=(16,),
        query_hidden_dim=8,
        time_embed_dim=4,
        action_sequence=4,
        action_dim=2,
        levels=3,
        bins=5,
        value_dim=value_dim,
        gru_layers=1,
        use_dueling=False,
    )
    inputs = _flow_critic_inputs(value_dim=value_dim)

    params = critic.init(jax.random.PRNGKey(0), *inputs)
    velocity = critic.apply(params, *inputs)

    assert velocity.shape == (2, 3, 4, 2, 5, value_dim)
    np.testing.assert_allclose(velocity, 0.0, atol=1e-7)
    param_names = str(params)
    assert "context_gru_0" in param_names
    assert "velocity_head" in param_names
    assert "distill" not in param_names.lower()


def test_sequence_flow_critic_can_evaluate_selected_bin_without_all_bin_axis():
    critic = C2FSequenceFlowCritic(
        hidden_dims=(16,),
        query_hidden_dim=8,
        time_embed_dim=4,
        action_sequence=4,
        action_dim=2,
        levels=3,
        bins=5,
        value_dim=11,
        gru_layers=1,
        use_dueling=False,
    )
    inputs = _flow_critic_inputs(value_dim=11, query_bins=1)

    params = critic.init(jax.random.PRNGKey(0), *inputs)
    velocity = critic.apply(params, *inputs)

    assert velocity.shape == (2, 3, 4, 2, 1, 11)


def test_sequence_flow_critic_conditions_on_low_dim_and_pixel_features_once():
    critic = C2FSequenceFlowCritic(
        hidden_dims=(16,),
        query_hidden_dim=8,
        time_embed_dim=4,
        action_sequence=4,
        action_dim=2,
        levels=3,
        bins=5,
        value_dim=1,
        low_dim_size=5,
        gru_layers=1,
        use_dueling=False,
    )
    inputs = list(_flow_critic_inputs(value_dim=1))
    inputs[0] = jnp.zeros((2, 17), dtype=jnp.float32)

    params = critic.init(jax.random.PRNGKey(0), *inputs)
    velocity = critic.apply(params, *inputs)

    assert velocity.shape == (2, 3, 4, 2, 5, 1)
    param_names = str(params)
    assert "context_rgb_dense_0" in param_names
    assert "context_low_dim_projection" in param_names


def test_centered_log_probabilities_roundtrip_through_softmax():
    probabilities = jnp.asarray(
        [[0.05, 0.15, 0.30, 0.50], [0.40, 0.10, 0.20, 0.30]],
        dtype=jnp.float32,
    )

    logits = centered_log_probabilities(probabilities)
    reconstructed = flow_logits_to_probabilities(logits)

    np.testing.assert_allclose(logits.mean(axis=-1), 0.0, atol=1e-6)
    np.testing.assert_allclose(reconstructed, probabilities, atol=1e-6)


def test_centered_log_probabilities_remain_finite_for_zero_mass_atoms():
    probabilities = jnp.asarray([[1.0, 0.0, 0.0]], dtype=jnp.float32)

    logits = centered_log_probabilities(probabilities)
    reconstructed = flow_logits_to_probabilities(logits)

    assert np.all(np.isfinite(np.asarray(logits)))
    assert np.all(np.asarray(reconstructed) > 0.0)
    np.testing.assert_allclose(reconstructed.sum(axis=-1), 1.0, atol=1e-6)


def test_scalar_to_categorical_two_hot_preserves_arbitrary_leading_shape():
    support = jnp.linspace(-2.0, 2.0, 5)
    values = jnp.asarray(
        [[[-3.0, -1.5], [0.0, 0.75]], [[1.5, 3.0], [-0.5, 1.0]]],
        dtype=jnp.float32,
    )

    projected = scalar_to_categorical(values, support)

    assert projected.shape == (*values.shape, 5)
    np.testing.assert_allclose(projected.sum(axis=-1), 1.0, atol=1e-7)
    reconstructed = jnp.sum(projected * support, axis=-1)
    np.testing.assert_allclose(
        reconstructed,
        jnp.clip(values, support[0], support[-1]),
        atol=1e-6,
    )


def test_value_only_flow_critic_has_no_candidate_conditioned_parameters():
    critic = C2FSequenceFlowCritic(
        hidden_dims=(16,),
        query_hidden_dim=8,
        time_embed_dim=4,
        action_sequence=4,
        action_dim=2,
        levels=3,
        bins=5,
        value_dim=1,
        gru_layers=1,
        use_dueling=True,
        value_only=True,
    )
    inputs = _flow_critic_inputs(value_dim=1)

    params = critic.init(jax.random.PRNGKey(0), *inputs)
    velocity = critic.apply(params, *inputs)
    param_names = str(params)

    assert velocity.shape == (2, 3, 4, 2, 5, 1)
    assert "value_velocity_head" in param_names
    assert "candidate_projection" not in param_names
    assert "'velocity_head'" not in param_names


def test_hl_gauss_encode_preserves_leading_shape_and_normalizes_bins():
    values = jnp.asarray(
        [
            [[-20.0], [-0.5], [0.0]],
            [[0.5], [1.5], [20.0]],
        ],
        dtype=jnp.float32,
    )

    encoded = hl_gauss_encode(
        values,
        v_min=-2.0,
        v_max=2.0,
        bins=9,
        sigma=1.5,
    )

    assert encoded.shape == (2, 3, 8)
    assert np.all(np.isfinite(np.asarray(encoded)))
    assert np.all(np.asarray(encoded) >= 0.0)
    np.testing.assert_allclose(encoded.sum(axis=-1), 1.0, atol=1e-6)


def test_flow_categorical_expected_q_uses_full_atom_distribution():
    support = jnp.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=jnp.float32)
    probabilities = jnp.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.25, 0.0, 0.50, 0.0, 0.25],
        ],
        dtype=jnp.float32,
    )

    values = expected_q(probabilities, support)

    np.testing.assert_allclose(values, [2.0, 0.0], atol=1e-6)


def test_flow_categorical_expected_q_vectorizes_sequence_dims_and_bins():
    support = jnp.linspace(-1.0, 1.0, 11, dtype=jnp.float32)
    probabilities = jnp.broadcast_to(
        jax.nn.one_hot(10, 11, dtype=jnp.float32),
        (2, 3, 4, 2, 5, 11),
    )

    values = expected_q(probabilities, support)

    assert values.shape == (2, 3, 4, 2, 5)
    np.testing.assert_allclose(values, 1.0, atol=1e-6)


def test_atom_cross_entropy_measures_endpoint_distribution_not_demo_action():
    target = jnp.asarray([[0.75, 0.25, 0.0]], dtype=jnp.float32)
    predicted_logits = jnp.log(
        jnp.asarray([[0.50, 0.25, 0.25]], dtype=jnp.float32)
    )

    loss = categorical_cross_entropy(target, predicted_logits)
    expected = -(0.75 * np.log(0.50) + 0.25 * np.log(0.25))

    np.testing.assert_allclose(loss, [expected], rtol=1e-6, atol=1e-6)


def test_atom_ce_lambda_zero_has_no_effect_on_loss_or_gradient():
    target = jnp.asarray([[0.1, 0.2, 0.7]], dtype=jnp.float32)
    initial_logits = jnp.asarray([[0.4, -0.2, 0.1]], dtype=jnp.float32)

    def total_loss(logits, atom_ce_lambda):
        flow_loss = jnp.square(logits).mean()
        ce_loss = categorical_cross_entropy(target, logits).mean()
        return flow_loss + atom_ce_lambda * ce_loss

    base_value, base_grad = jax.value_and_grad(total_loss)(initial_logits, 0.0)
    expected_value, expected_grad = jax.value_and_grad(
        lambda logits: jnp.square(logits).mean()
    )(initial_logits)

    np.testing.assert_allclose(base_value, expected_value, atol=1e-7)
    np.testing.assert_allclose(base_grad, expected_grad, atol=1e-7)


def test_demo_margin_returns_per_sample_losses_for_external_demo_masking():
    # [batch, level, sequence, action_dim, action_bin]
    q_values = jnp.asarray(
        [
            [[[[0.0, 1.0, 0.2]]]],  # expert bin 1 clears a 0.5 margin
            [[[[2.0, 0.0, 1.0]]]],  # non-demo row must be ignored
            [[[[0.9, 1.0, 0.8]]]],  # expert bin 1 violates the margin
        ],
        dtype=jnp.float32,
    )
    chosen_q = jnp.asarray([[[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype=jnp.float32)
    demos = jnp.asarray([1.0, 0.0, 1.0], dtype=jnp.float32)

    per_sample = demo_margin_per_sample(
        q_values,
        chosen_q,
        margin=0.5,
    )
    loss = jnp.sum(per_sample * demos) / demos.sum()

    # This deliberately matches CQN-AS: the chosen bin remains in all_q. Its
    # constant self-margin has zero gradient, while the non-demo row is masked.
    expected_demo_0 = (0.0 + 0.5 + 0.0) / 3.0
    expected_demo_2 = (0.4 + 0.5 + 0.3) / 3.0
    np.testing.assert_allclose(
        loss,
        (expected_demo_0 + expected_demo_2) / 2.0,
        atol=1e-6,
    )


def test_demo_fosd_penalizes_chosen_distribution_with_lower_returns():
    # [batch, level, sequence, action_dim, atom]
    chosen_probabilities = jnp.asarray(
        [
            [[[[0.0, 1.0]]]],  # all mass at the high-return atom
            [[[[1.0, 0.0]]]],  # all mass at the low-return atom
        ],
        dtype=jnp.float32,
    )
    # [batch, level, sequence, action_dim, action_bin, atom]
    all_probabilities = jnp.asarray(
        [
            [[[[[0.0, 1.0], [1.0, 0.0]]]]],
            [[[[[1.0, 0.0], [0.0, 1.0]]]]],
        ],
        dtype=jnp.float32,
    )

    per_sample = demo_fosd_per_sample(
        chosen_probabilities,
        all_probabilities,
    )

    # The first chosen distribution first-order dominates both bins. For the
    # second row, one of two bins dominates the chosen distribution by one CDF
    # unit, hence the mean over bins is 0.5.
    np.testing.assert_allclose(per_sample, [0.0, 0.5], atol=1e-6)


def test_scalar_expected_q_is_mean_over_initial_noise_samples():
    # [batch, noise, level, sequence, action_dim, action_bin]
    endpoints = jnp.asarray(
        [
            [
                [[[[1.0, 3.0]]]],
                [[[[3.0, 5.0]]]],
                [[[[5.0, 7.0]]]],
            ]
        ],
        dtype=jnp.float32,
    )

    scalar_q = endpoints.mean(axis=1)

    assert scalar_q.shape == (1, 1, 1, 1, 2)
    np.testing.assert_allclose(scalar_q, [[[[[3.0, 5.0]]]]], atol=1e-6)


def test_value_flow_euler_preserves_vectorized_shape_and_runs_source_to_target():
    # [batch, noise, sequence, action_dim, action_bin, value_dim]
    source = jnp.zeros((2, 3, 4, 2, 5, 1), dtype=jnp.float32)

    def time_velocity(sample, tau):
        return jnp.broadcast_to(tau, sample.shape)

    integrate = jax.jit(
        lambda value: integrate_value_flow(
            time_velocity,
            value,
            num_flow_steps=2,
        )
    )
    endpoint = integrate(source)

    # The repo convention evaluates tau=1 then tau=0.5 while integrating from
    # source to target: 0.5 * 1 + 0.5 * 0.5 = 0.75.
    assert endpoint.shape == source.shape
    np.testing.assert_allclose(endpoint, 0.75, atol=1e-6)


def test_value_flow_calls_velocity_once_per_configured_flow_step():
    seen_tau = []

    def zero_velocity(sample, tau):
        seen_tau.append(float(tau))
        return jnp.zeros_like(sample)

    integrate_value_flow(
        zero_velocity,
        jnp.zeros((2, 3, 4, 2, 5, 1), dtype=jnp.float32),
        num_flow_steps=4,
    )

    np.testing.assert_allclose(seen_tau, [1.0, 0.75, 0.5, 0.25])


def test_value_flow_end_tau_integrates_only_requested_partial_path():
    source = jnp.zeros((2, 3, 4, 1), dtype=jnp.float32)
    end_tau = jnp.asarray(
        [[0.0, 0.25, 0.5], [0.125, 0.75, 1.0]],
        dtype=jnp.float32,
    )

    endpoint = jax.jit(
        lambda value, stop: integrate_value_flow(
            lambda sample, _tau: jnp.ones_like(sample),
            value,
            num_flow_steps=4,
            end_tau=stop,
        )
    )(source, end_tau)

    expected = jnp.broadcast_to(
        (1.0 - end_tau)[..., None, None],
        source.shape,
    )
    assert endpoint.shape == source.shape
    np.testing.assert_allclose(endpoint, expected, atol=1e-6)


def test_value_flow_can_clip_each_euler_step_to_return_support():
    endpoint = integrate_value_flow(
        lambda sample, _tau: jnp.full_like(sample, 10.0),
        jnp.zeros((2, 3, 1), dtype=jnp.float32),
        num_flow_steps=2,
        clip_min=-0.5,
        clip_max=0.5,
    )

    np.testing.assert_allclose(endpoint, 0.5, atol=1e-6)


def test_value_flow_source_jvp_matches_discrete_flow_derivative():
    source = jnp.asarray([[[2.0]], [[-1.0]]], dtype=jnp.float32)

    endpoint, source_jvp = integrate_value_flow_with_source_jvp(
        lambda value, _tau: 2.0 * value,
        source,
        num_flow_steps=2,
    )

    # Each Euler step multiplies both value and its source derivative by
    # 1 + (1 / 2) * 2 = 2.
    np.testing.assert_allclose(endpoint, 4.0 * source, atol=1e-6)
    np.testing.assert_allclose(source_jvp, 4.0, atol=1e-6)


def test_flowiqn_quantile_coupling_sorts_only_within_each_condition():
    source = jnp.asarray(
        [
            [0.8, 0.1, 0.4],
            [0.2, 0.9, 0.5],
        ],
        dtype=jnp.float32,
    )[:, :, None, None, None, None]
    target = jnp.asarray(
        [
            [8.0, 1.0, 4.0],
            [-2.0, -9.0, -5.0],
        ],
        dtype=jnp.float32,
    )[:, :, None, None, None, None]

    coupled = quantile_couple_return_samples(
        source,
        target,
        source_min=0.0,
        source_max=1.0,
    )

    np.testing.assert_allclose(
        np.asarray(coupled.source[:, :, 0, 0, 0, 0]),
        [[0.1, 0.4, 0.8], [0.2, 0.5, 0.9]],
    )
    np.testing.assert_allclose(
        np.asarray(coupled.target[:, :, 0, 0, 0, 0]),
        [[1.0, 4.0, 8.0], [-9.0, -5.0, -2.0]],
    )
    np.testing.assert_allclose(coupled.source_quantile, coupled.source)


def test_quantile_huber_endpoint_loss_compares_all_particle_pairs():
    predictions = jnp.asarray([[0.0, 1.0]], dtype=jnp.float32)
    targets = jnp.asarray([[0.0, 1.0]], dtype=jnp.float32)
    quantiles = jnp.asarray([[0.25, 0.75]], dtype=jnp.float32)

    loss = quantile_huber_endpoint_loss(
        predictions,
        targets,
        quantiles,
        kappa=1.0,
    )

    # Two of four pairs have |error|=1. Their quantile weight is .25 and
    # Huber(1)=.5, so the all-pairs mean is (2 * .25 * .5) / 4.
    np.testing.assert_allclose(loss, [0.0625], atol=1e-7)


def test_quantile_huber_endpoint_loss_rejects_invalid_shapes_and_kappa():
    with pytest.raises(ValueError, match="same shape"):
        quantile_huber_endpoint_loss(
            jnp.zeros((1, 2)),
            jnp.zeros((1, 2)),
            jnp.zeros((1, 1)),
        )
    with pytest.raises(ValueError, match="positive"):
        quantile_huber_endpoint_loss(
            jnp.zeros((1, 2)),
            jnp.zeros((1, 2)),
            jnp.zeros((1, 2)),
            kappa=0.0,
        )


def test_cqn_flow_config_parses_categorical_defaults():
    spec = cqn_flow_spec_from_cfg(_compose_cqn_flow())

    assert spec.value_mode == "categorical"
    assert spec.num_flow_steps == 1
    assert spec.num_flow_samples == 1
    assert spec.atom_ce_lambda == 1.0
    assert spec.pcbf_loss_coeff == 0.0
    assert spec.pcbf_lambda == 0.0
    assert spec.confidence_weight_temp is None
    assert not spec.flow_iqn_quantile_coupling
    assert spec.quantile_endpoint_lambda == 0.0
    assert spec.quantile_huber_kappa == 1.0
    assert spec.demo_fosd
    assert spec.time_scale == 1000.0


@pytest.mark.parametrize(
    (
        "launch",
        "mode",
        "steps",
        "time_embedding",
        "dcfm_lambda",
        "pcbf_loss_coeff",
    ),
    [
        (
            "cqn_flow_floq_pixel_bigym_demo_driven",
            "scalar",
            8,
            "fourier",
            0.0,
            0.0,
        ),
        (
            "cqn_value_flows_pixel_bigym_demo_driven",
            "return_sample",
            10,
            "raw",
            1.0,
            0.0,
        ),
        (
            "cqn_pcbf_pixel_bigym_demo_driven",
            "return_sample",
            10,
            "raw",
            0.0,
            1.0,
        ),
    ],
)
def test_cqn_flow_research_launches_apply_profile_after_method_defaults(
    launch,
    mode,
    steps,
    time_embedding,
    dcfm_lambda,
    pcbf_loss_coeff,
):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[f"launch={launch}", "env=bigym/move_plate"],
        )

    assert cfg.method.value_mode == mode
    assert cfg.method.num_flow_steps == steps
    assert cfg.method.time_embedding_type == time_embedding
    assert cfg.method.dcfm_lambda == dcfm_lambda
    assert cfg.method.pcbf_loss_coeff == pcbf_loss_coeff
    assert cfg.method.critic_target_tau == pytest.approx(0.005)


@pytest.mark.parametrize(
    ("launch", "confidence_temp"),
    [
        ("cqn_value_flows_bc_target_two_tower_high_utd4_gate", None),
        (
            "cqn_value_flows_confidence_bc_target_two_tower_high_utd4_gate",
            0.3,
        ),
    ],
)
def test_value_flows_high_utd_launches_are_matched_except_confidence(
    launch,
    confidence_temp,
):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[f"launch={launch}", "env=bigym/move_plate"],
        )

    assert cfg.method.value_mode == "return_sample"
    assert cfg.method.bcfm_lambda == pytest.approx(1.0)
    assert cfg.method.dcfm_lambda == pytest.approx(1.0)
    assert cfg.method.flow_source_type == "gaussian"
    assert cfg.method.num_flow_steps == 10
    assert cfg.method.num_flow_samples == 4
    assert cfg.method.num_target_flow_samples == 4
    assert cfg.method.num_action_flow_samples == 4
    assert cfg.method.time_embedding_type == "raw"
    assert cfg.method.td_target_action_source == "bc_policy"
    assert cfg.method.separate_bc_policy
    assert cfg.method.distinct_policy_encoder
    assert cfg.method.num_update_steps == 4
    assert cfg.method.confidence_weight_temp == confidence_temp


def test_evor_flowtd_high_utd_launch_is_isolated_and_bc_targeted():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_evor_flowtd_bc_target_two_tower_high_utd4_gate",
                "env=bigym/move_plate",
            ],
        )

    assert cfg.method.value_mode == "return_sample"
    assert cfg.method.evor_td_lambda == pytest.approx(1.0)
    assert cfg.method.bcfm_lambda == pytest.approx(0.0)
    assert cfg.method.dcfm_lambda == pytest.approx(0.0)
    assert cfg.method.pcbf_loss_coeff == pytest.approx(0.0)
    assert cfg.method.flow_distill_lambda == pytest.approx(0.0)
    assert cfg.method.mc_return_weight == pytest.approx(0.0)
    assert cfg.method.num_flow_steps == 10
    assert cfg.method.num_flow_samples == 1
    assert cfg.method.num_target_flow_samples == 1
    assert cfg.method.num_action_flow_samples == 16
    assert cfg.method.flow_source_type == "gaussian"
    assert not cfg.method.antithetic_flow_sources
    assert cfg.method.return_sample_aggregation == "entropic"
    assert cfg.method.return_sample_temperature == pytest.approx(1.0)
    assert cfg.method.time_embedding_type == "raw"
    assert not cfg.method.clip_flow_trajectory
    assert cfg.method.td_target_action_source == "bc_policy"
    assert cfg.method.separate_bc_policy
    assert cfg.method.distinct_policy_encoder
    assert cfg.method.num_update_steps == 4
    assert cfg.method.policy_value_beta is None
    assert _mc_return_anchor_enabled(cfg)


@pytest.mark.parametrize(
    ("launch", "method_name", "value_mode"),
    [
        ("cqn_as_pixel_bigym_stage1_gate", "cqn_as", None),
        ("cqn_flow_floq_pixel_bigym_stage1_gate", "cqn_flow", "scalar"),
        (
            "cqn_flow_floq_anchored_pixel_bigym_stage1_gate",
            "cqn_flow",
            "scalar",
        ),
        (
            "cqn_value_flows_pixel_bigym_stage1_gate",
            "cqn_flow",
            "return_sample",
        ),
        ("cqn_pcbf_pixel_bigym_stage1_gate", "cqn_flow", "return_sample"),
    ],
)
def test_stage1_gate_launches_are_frozen_demo_and_compute_matched(
    launch,
    method_name,
    value_mode,
):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[f"launch={launch}", "env=bigym/move_plate"],
        )

    assert cfg.method.name == method_name
    assert cfg.num_train_frames == 10500
    assert cfg.batch_size == 16
    assert cfg.demo_batch_size == 16
    assert not cfg.use_self_imitation
    assert cfg.method.critic_target_tau == pytest.approx(0.005)
    assert not cfg.method.demo_fosd
    if value_mode is not None:
        assert cfg.method.value_mode == value_mode
        assert cfg.method.num_flow_steps == 4
        assert cfg.method.num_flow_samples == 4
        assert cfg.method.num_target_flow_samples == 4
        assert cfg.method.num_action_flow_samples == 4
    if launch == "cqn_pcbf_pixel_bigym_stage1_gate":
        assert cfg.method.pcbf_loss_coeff == pytest.approx(1.0)
        assert cfg.method.pcbf_lambda == pytest.approx(cfg.replay.gamma)
        assert cfg.method.critic_grad_clip == pytest.approx(1.0)
        assert cfg.method.demo_flow_steps is None
    if launch == "cqn_flow_floq_anchored_pixel_bigym_stage1_gate":
        assert cfg.method.endpoint_q_lambda == pytest.approx(1.0)
        assert cfg.method.source_consistency_lambda == pytest.approx(0.1)


def test_pcbf_nan_diagnostic_is_short_fine_grained_and_snapshot_free():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_pcbf_pixel_bigym_nan_diagnostic",
                "env=bigym/move_plate",
            ],
        )

    assert cfg.num_train_frames == 1100
    assert cfg.num_eval_episodes == 0
    assert cfg.log_every == 25
    assert not cfg.save_snapshot
    assert cfg.method.critic_grad_clip == pytest.approx(1.0)
    assert cfg.method.pcbf_loss_coeff == pytest.approx(1.0)


def test_flowiqn_stage1_launch_is_sorted_uniform_return_flow():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_flowiqn_pixel_bigym_stage1_gate",
                "env=bigym/move_plate",
            ],
        )

    assert cfg.num_train_frames == 10500
    assert cfg.batch_size == cfg.demo_batch_size == 8
    assert not cfg.use_self_imitation
    assert cfg.method.value_mode == "return_sample"
    assert cfg.method.flow_iqn_quantile_coupling
    assert cfg.method.flow_source_type == "uniform"
    assert cfg.method.flow_source_min == pytest.approx(0.0)
    assert cfg.method.flow_source_max == pytest.approx(1.0)
    assert not cfg.method.antithetic_flow_sources
    assert not cfg.method.action_flow_quantile_grid
    assert cfg.method.num_flow_samples == 8
    assert cfg.method.num_target_flow_samples == 8
    assert cfg.method.num_action_flow_samples == 4
    assert cfg.method.bcfm_lambda == pytest.approx(1.0)
    assert cfg.method.dcfm_lambda == pytest.approx(0.0)
    assert cfg.method.pcbf_loss_coeff == pytest.approx(0.0)


def test_flowiqn_corrected_launch_freezes_moveplate_mechanism_settings():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_flowiqn_pixel_bigym_corrected_gate",
                "env=bigym/move_plate",
            ],
        )

    assert cfg.num_train_frames == 10500
    assert cfg.batch_size == cfg.demo_batch_size == 8
    assert not cfg.use_self_imitation
    assert cfg.method.value_mode == "return_sample"
    assert cfg.method.flow_iqn_quantile_coupling
    assert cfg.method.flow_source_type == "uniform"
    assert cfg.method.flow_source_min == pytest.approx(0.9)
    assert cfg.method.flow_source_max == pytest.approx(1.0)
    assert not cfg.method.antithetic_flow_sources
    assert cfg.method.fixed_action_flow_sources
    assert cfg.method.action_flow_quantile_grid
    assert cfg.method.num_flow_steps == 8
    assert cfg.method.num_flow_samples == 8
    assert cfg.method.num_target_flow_samples == 8
    assert cfg.method.num_action_flow_samples == 4
    assert cfg.method.scalar_value_embedding == "hl_gauss"
    assert cfg.method.scalar_embed_bins == 51
    assert cfg.method.scalar_embed_sigma == pytest.approx(16.0)
    assert cfg.method.time_embedding_type == "fourier"
    assert cfg.method.time_embed_dim == 64


@pytest.mark.parametrize(
    ("launch", "bcfm_lambda", "quantile_endpoint_lambda"),
    [
        (
            "cqn_flowiqn_bc_target_two_tower_high_utd4_gate",
            1.0,
            0.0,
        ),
        (
            "cqn_qr_flowiqn_equal_bc_target_two_tower_high_utd4_gate",
            1.0,
            1.0,
        ),
        (
            "cqn_qr_flowiqn_dbc_ratio_bc_target_two_tower_high_utd4_gate",
            0.01,
            1.0,
        ),
    ],
)
def test_qr_flowiqn_factorial_launches_change_only_registered_objectives(
    launch,
    bcfm_lambda,
    quantile_endpoint_lambda,
):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[f"launch={launch}", "env=bigym/move_plate"],
        )

    assert cfg.batch_size == cfg.demo_batch_size == 8
    assert cfg.method.value_mode == "return_sample"
    assert cfg.method.flow_iqn_quantile_coupling
    assert cfg.method.bcfm_lambda == pytest.approx(bcfm_lambda)
    assert cfg.method.quantile_endpoint_lambda == pytest.approx(
        quantile_endpoint_lambda
    )
    assert cfg.method.quantile_huber_kappa == pytest.approx(1.0)
    assert cfg.method.num_flow_steps == 8
    assert cfg.method.num_flow_samples == 8
    assert cfg.method.num_target_flow_samples == 8
    assert cfg.method.num_action_flow_samples == 4
    assert cfg.method.separate_bc_policy
    assert cfg.method.distinct_policy_encoder
    assert cfg.method.td_target_action_source == "bc_policy"
    assert cfg.method.td_target_policy_value_beta is None
    assert cfg.method.policy_value_beta is None
    assert cfg.method.mc_return_weight == pytest.approx(0.0)


def test_flowiqn_action_source_uses_fixed_midpoint_quantile_grid_only_for_ranking():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(
            "method.value_mode=return_sample",
            "method.atom_ce_lambda=0",
            "method.demo_fosd=false",
            "method.flow_source_type=uniform",
            "method.flow_source_min=0.9",
            "method.flow_source_max=1",
            "method.antithetic_flow_sources=false",
            "method.flow_iqn_quantile_coupling=true",
            "method.action_flow_quantile_grid=true",
            "method.num_flow_samples=4",
            "method.num_target_flow_samples=4",
            "method.num_action_flow_samples=4",
            "method.bcfm_lambda=1",
            "method.dcfm_lambda=0",
            "method.pcbf_loss_coeff=0",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )

    first = agent._action_flow_source(
        jax.random.PRNGKey(1), 2, agent.bins
    )
    second = agent._action_flow_source(
        jax.random.PRNGKey(2), 2, agent.bins
    )
    expected = np.array([0.9125, 0.9375, 0.9625, 0.9875], np.float32)

    np.testing.assert_allclose(first, second, atol=0.0)
    np.testing.assert_allclose(
        np.asarray(first[0, :, 0, 0, 0, 0]),
        expected,
        atol=1e-7,
    )
    random_first = agent._flow_source(
        jax.random.PRNGKey(1), 2, agent.bins, num_samples=4
    )
    random_second = agent._flow_source(
        jax.random.PRNGKey(2), 2, agent.bins, num_samples=4
    )
    assert not np.array_equal(random_first, random_second)


def test_action_quantile_grid_requires_flowiqn_objective():
    observation_space, action_space = _spaces()
    with pytest.raises(
        ValueError,
        match="requires flow_iqn_quantile_coupling=true",
    ):
        create_agent(
            _compose_cqn_flow("method.action_flow_quantile_grid=true"),
            observation_space=observation_space,
            action_space=action_space,
        )


def test_floq_profile_runs_fourier_hl_gauss_scalar_action_path():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(
            "profile=cqn_flow_floq",
            "method.num_flow_steps=1",
            "method.num_flow_samples=1",
            "method.num_target_flow_samples=1",
            "method.num_action_flow_samples=1",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    observations = {"low_dim_state": np.zeros((1, 1, 5), dtype=np.float32)}

    action = agent.act(observations, step=3000, eval_mode=True)

    assert agent.value_mode == "scalar"
    assert agent.flow_source_type == "uniform"
    assert agent.scalar_value_embedding == "hl_gauss"
    assert agent.time_embedding_type == "fourier"
    assert action.shape == (1, 2, 2)
    assert np.all(np.isfinite(action))


def test_flowiqn_agent_updates_with_quantile_conditioned_uniform_flow():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(
            "method.value_mode=return_sample",
            "method.atom_ce_lambda=0",
            "method.demo_fosd=false",
            "method.flow_source_type=uniform",
            "method.flow_source_min=0",
            "method.flow_source_max=1",
            "method.antithetic_flow_sources=false",
            "method.flow_iqn_quantile_coupling=true",
            "method.num_flow_samples=3",
            "method.num_target_flow_samples=3",
            "method.num_action_flow_samples=3",
            "method.bcfm_lambda=1",
            "method.dcfm_lambda=0",
            "method.pcbf_loss_coeff=0",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    agent.logging = True

    metrics = agent.update(iter([_batch()]), step=1)

    assert agent.flow_iqn_quantile_coupling
    assert "quantile_projection" in agent.params["critic"]["params"]
    assert np.isfinite(metrics["flow_loss"])
    assert metrics["flow_loss"] > 0.0


def test_flowiqn_all_pairs_endpoint_quantile_loss_updates_velocity_flow():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(
            "method.value_mode=return_sample",
            "method.atom_ce_lambda=0",
            "method.demo_fosd=false",
            "method.flow_source_type=uniform",
            "method.flow_source_min=0",
            "method.flow_source_max=1",
            "method.antithetic_flow_sources=false",
            "method.flow_iqn_quantile_coupling=true",
            "method.num_flow_samples=3",
            "method.num_target_flow_samples=3",
            "method.num_action_flow_samples=3",
            "method.bcfm_lambda=0.01",
            "method.quantile_endpoint_lambda=1",
            "method.quantile_huber_kappa=1",
            "method.dcfm_lambda=0",
            "method.pcbf_loss_coeff=0",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    agent.logging = True

    metrics = agent.update(iter([_batch()]), step=1)

    assert np.isfinite(metrics["quantile_endpoint_loss"])
    assert metrics["quantile_endpoint_loss"] > 0.0
    assert metrics["velocity_head_grad_norm"] > 0.0


def test_quantile_endpoint_loss_requires_flowiqn_conditioning():
    observation_space, action_space = _spaces()
    with pytest.raises(
        ValueError,
        match="requires flow_iqn_quantile_coupling=true",
    ):
        create_agent(
            _compose_cqn_flow("method.quantile_endpoint_lambda=1"),
            observation_space=observation_space,
            action_space=action_space,
        )


@pytest.mark.parametrize(
    "override",
    [
        "method.flow_source_type=gaussian",
        "method.antithetic_flow_sources=true",
        "method.num_target_flow_samples=2",
        "method.value_mode=scalar",
    ],
)
def test_flowiqn_rejects_incompatible_objectives(override):
    observation_space, action_space = _spaces()
    base = [
        "method.value_mode=return_sample",
        "method.atom_ce_lambda=0",
        "method.demo_fosd=false",
        "method.flow_source_type=uniform",
        "method.flow_source_min=0",
        "method.flow_source_max=1",
        "method.antithetic_flow_sources=false",
        "method.flow_iqn_quantile_coupling=true",
        "method.num_flow_samples=3",
        "method.num_target_flow_samples=3",
        "method.num_action_flow_samples=3",
        "method.bcfm_lambda=1",
        "method.dcfm_lambda=0",
        "method.pcbf_loss_coeff=0",
        override,
    ]

    with pytest.raises(ValueError, match="FlowIQN"):
        create_agent(
            _compose_cqn_flow(*base),
            observation_space=observation_space,
            action_space=action_space,
        )


def test_decoupled_floq_updates_policy_flow_and_mc_endpoint_separately():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(
            "method.value_mode=scalar",
            "method.atom_ce_lambda=0",
            "method.demo_fosd=false",
            "method.separate_bc_policy=true",
            "method.td_target_action_source=replay_next",
            "method.critic_sequence_mode=effective_k0",
            "method.mc_return_weight=0.1",
            "method.structured_exploration_level=0",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    batch = _batch()
    batch["mc_return"] = np.asarray([0.8, -0.2], np.float32)
    agent.logging = True
    critic_before = jax.tree.map(np.asarray, agent.params["critic"])
    policy_before = jax.tree.map(np.asarray, agent.params["policy"])

    metrics = agent.update(iter([batch]), step=1)

    assert _tree_changed(critic_before, agent.params["critic"])
    assert _tree_changed(policy_before, agent.params["policy"])
    assert metrics["flow_loss"] >= 0.0
    assert metrics["endpoint_q_loss"] >= 0.0
    assert metrics["mc_return_loss"] > 0.0
    assert metrics["mc_return_mae"] > 0.0
    assert metrics["policy_ce"] > 0.0
    assert metrics["policy_bc_loss"] > 0.0
    assert metrics["demo_margin_loss"] == pytest.approx(0.0, abs=1e-8)
    assert metrics["demo_fosd_loss"] == pytest.approx(0.0, abs=1e-8)
    assert np.isfinite(metrics["total_loss"])


def test_floq_policy_value_td_target_updates_with_exact_bc_rollout():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(
            "method.value_mode=scalar",
            "method.atom_ce_lambda=0",
            "method.demo_fosd=false",
            "method.separate_bc_policy=true",
            "method.td_target_action_source=policy_value",
            "method.td_target_policy_value_beta=1",
            "method.critic_sequence_mode=effective_k0",
            "method.mc_return_weight=0.1",
            "method.structured_exploration_level=0",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    assert agent.policy_value_beta is None
    assert agent.td_target_policy_value_beta == pytest.approx(1.0)
    agent.logging = True

    metrics = agent.update(iter([_batch()]), step=1)

    assert np.isfinite(metrics["flow_loss"])
    assert np.isfinite(metrics["endpoint_q_loss"])
    assert np.isfinite(metrics["total_loss"])


def test_floq_bc_policy_td_target_updates_with_exact_bc_rollout():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(
            "method.value_mode=scalar",
            "method.atom_ce_lambda=0",
            "method.demo_fosd=false",
            "method.separate_bc_policy=true",
            "method.td_target_action_source=bc_policy",
            "method.critic_sequence_mode=effective_k0",
            "method.mc_return_weight=0.1",
            "method.structured_exploration_level=0",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    assert agent.policy_value_beta is None
    assert agent.td_target_policy_value_beta is None
    agent.logging = True

    metrics = agent.update(iter([_batch()]), step=1)

    assert np.isfinite(metrics["flow_loss"])
    assert np.isfinite(metrics["endpoint_q_loss"])
    assert np.isfinite(metrics["total_loss"])


def test_floq_distilled_readout_uses_detached_online_endpoint(monkeypatch):
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(
            "method.value_mode=scalar",
            "method.atom_ce_lambda=0",
            "method.demo_fosd=false",
            "method.bc_lambda=0",
            "method.bcfm_lambda=0",
            "method.endpoint_q_lambda=0",
            "method.source_consistency_lambda=0",
            "method.mc_return_weight=0",
            "method.flow_distill_lambda=1",
            "method.flow_distill_action_readout=false",
            "method.weight_decay=0",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )

    def fixed_endpoint(
        _critic_params,
        features,
        _action,
        _key,
        *,
        num_samples=None,
    ):
        samples = 1 if num_samples is None else int(num_samples)
        return jnp.full(
            (
                features.shape[0],
                samples,
                agent.levels,
                agent.action_sequence,
                agent.action_dim,
                1,
            ),
            0.75,
            dtype=jnp.float32,
        )

    monkeypatch.setattr(
        agent,
        "_selected_endpoints_per_level",
        fixed_endpoint,
    )
    batch = _batch()
    agent.logging = True
    flow_before = jax.tree.map(np.asarray, agent.params["critic"])
    readout_before = jax.tree.map(
        np.asarray,
        agent.params["flow_distill_readout"],
    )
    obs_inputs = agent._prepare_rl_obs_inputs(batch)
    features = agent._rl_features(
        agent.params.get("encoder"),
        obs_inputs,
    )
    initial_chosen_q, _ = agent._flow_distill_outputs_per_level(
        agent.params["flow_distill_readout"],
        jax.lax.stop_gradient(features),
        jnp.asarray(batch["action"], dtype=jnp.float32),
    )
    initial_chosen_q = agent._sequence_training_slice(
        initial_chosen_q,
        sequence_axis=2,
    )
    expected_error = np.asarray(initial_chosen_q) - 0.75

    metrics = agent.update(iter([batch]), step=1)

    # Only the scalar readout sees this auxiliary objective. Its endpoint
    # target and shared value features are both detached.
    assert _tree_exactly_equal(flow_before, agent.params["critic"])
    assert _tree_changed(
        readout_before,
        agent.params["flow_distill_readout"],
    )
    assert metrics["flow_distill_loss"] == pytest.approx(
        float(np.mean(np.square(expected_error))),
        rel=1e-5,
    )
    assert metrics["flow_distill_mae"] == pytest.approx(
        float(np.mean(np.abs(expected_error))),
        rel=1e-5,
    )
    assert metrics["flow_distill_readout_grad_norm"] > 0.0
    assert metrics["flow_critic_grad_norm"] == pytest.approx(0.0, abs=1e-8)


def test_floq_distilled_action_readout_combines_q_and_bc_prior(monkeypatch):
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(
            "method.value_mode=scalar",
            "method.atom_ce_lambda=0",
            "method.demo_fosd=false",
            "method.separate_bc_policy=true",
            "method.flow_distill_lambda=1",
            "method.flow_distill_action_readout=true",
            "method.policy_value_beta=0",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    q_values = jnp.asarray(
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        dtype=jnp.float32,
    )

    def fake_readout_level(_params, features, _level, _low, _high):
        return jnp.broadcast_to(
            q_values,
            (
                features.shape[0],
                agent.action_sequence,
                agent.action_dim,
                agent.bins,
            ),
        )

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

    monkeypatch.setattr(agent, "_flow_distill_level", fake_readout_level)
    monkeypatch.setattr(agent, "policy_model", FakePolicy())
    features = jnp.zeros((1, 5), dtype=jnp.float32)

    _, q_only_bins = agent._flow_distill_policy_value_action(
        agent.params["flow_distill_readout"],
        features,
        agent.params["policy"],
        features,
    )
    agent.policy_value_beta = 4.0
    _, blended_bins = agent._flow_distill_policy_value_action(
        agent.params["flow_distill_readout"],
        features,
        agent.params["policy"],
        features,
    )

    np.testing.assert_array_equal(q_only_bins, 4)
    np.testing.assert_array_equal(blended_bins, 0)


def test_evor_entropic_return_aggregation_is_stable_log_mean_exp():
    samples = jnp.asarray([[0.0, 1.0], [0.3, 0.3]], dtype=jnp.float32)
    mean = aggregate_return_samples(
        samples,
        aggregation="mean",
        temperature=1.0,
        sample_axis=1,
    )
    entropic = aggregate_return_samples(
        samples,
        aggregation="entropic",
        temperature=1.0,
        sample_axis=1,
    )

    np.testing.assert_allclose(mean, [0.5, 0.3], rtol=1e-6)
    np.testing.assert_allclose(
        entropic[0],
        np.log((1.0 + np.e) / 2.0),
        rtol=1e-6,
    )
    np.testing.assert_allclose(entropic[1], 0.3, rtol=1e-6)


def test_flowcritic_truncated_mean_discards_largest_return_samples():
    samples = jnp.asarray(
        [[-2.0, 1.0, 3.0, 0.0], [0.1, 0.2, 0.3, 0.4]],
        dtype=jnp.float32,
    )

    truncated = aggregate_return_samples(
        samples,
        aggregation="truncated_mean",
        temperature=1.0,
        truncate_top=1,
        sample_axis=1,
    )

    np.testing.assert_allclose(
        truncated,
        [(-2.0 + 0.0 + 1.0) / 3.0, 0.2],
        rtol=1e-6,
    )
    with pytest.raises(ValueError, match="smaller than"):
        aggregate_return_samples(
            samples,
            aggregation="truncated_mean",
            temperature=1.0,
            truncate_top=4,
            sample_axis=1,
        )


def test_flowcritic_truncated_readout_config_validates_sample_count():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(
            "method.value_mode=return_sample",
            "method.atom_ce_lambda=0",
            "method.demo_fosd=false",
            "method.num_flow_samples=4",
            "method.num_target_flow_samples=4",
            "method.num_action_flow_samples=10",
            "method.return_sample_aggregation=truncated_mean",
            "method.return_sample_truncate_top=1",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )

    assert agent.return_sample_aggregation == "truncated_mean"
    assert agent.return_sample_truncate_top == 1

    with pytest.raises(ValueError, match="smaller than every"):
        create_agent(
            _compose_cqn_flow(
                "method.value_mode=return_sample",
                "method.atom_ce_lambda=0",
                "method.demo_fosd=false",
                "method.num_flow_samples=4",
                "method.num_target_flow_samples=4",
                "method.num_action_flow_samples=10",
                "method.return_sample_aggregation=truncated_mean",
                "method.return_sample_truncate_top=4",
            ),
            observation_space=observation_space,
            action_space=action_space,
        )


def test_evor_entropic_readout_rejects_deterministic_scalar_flow():
    observation_space, action_space = _spaces()
    with pytest.raises(ValueError, match="value_mode=return_sample"):
        create_agent(
            _compose_cqn_flow(
                "method.value_mode=scalar",
                "method.atom_ce_lambda=0",
                "method.demo_fosd=false",
                "method.return_sample_aggregation=entropic",
            ),
            observation_space=observation_space,
            action_space=action_space,
        )


def test_raw_return_flow_action_supports_mean_and_evor_readouts(monkeypatch):
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(
            "method.value_mode=return_sample",
            "method.atom_ce_lambda=0",
            "method.demo_fosd=false",
            "method.separate_bc_policy=true",
            "method.flow_q_action_readout=true",
            "method.policy_value_beta=0",
            "method.return_sample_aggregation=mean",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )

    def fake_integrate(_params, features, _condition, _key, **_kwargs):
        endpoints = jnp.zeros(
            (
                features.shape[0],
                2,
                agent.action_sequence,
                agent.action_dim,
                agent.bins,
                1,
            ),
            dtype=jnp.float32,
        )
        # Mean prefers bin 4 (0.6 > 0.5), while eta=0.1 entropic value
        # prefers bin 0 because its [1, 0] upper tail is meaningful.
        endpoints = endpoints.at[:, 0, ..., 0, 0].set(1.0)
        endpoints = endpoints.at[:, :, ..., 4, 0].set(0.6)
        return endpoints

    def uniform_policy(_params, features, _level, _midpoint):
        return jnp.zeros(
            (
                features.shape[0],
                agent.action_sequence,
                agent.action_dim,
                agent.bins,
            ),
            dtype=jnp.float32,
        )

    monkeypatch.setattr(agent, "_integrate_level", fake_integrate)
    monkeypatch.setattr(agent, "_policy_bin_scores", uniform_policy)
    features = jnp.zeros((1, 5), dtype=jnp.float32)

    _, mean_bins = agent._flow_q_policy_value_action(
        agent.params["critic"],
        features,
        agent.params["policy"],
        features,
    )
    agent.return_sample_aggregation = "entropic"
    agent.return_sample_temperature = 0.1
    _, entropic_bins = agent._flow_q_policy_value_action(
        agent.params["critic"],
        features,
        agent.params["policy"],
        features,
    )

    np.testing.assert_array_equal(mean_bins, 4)
    np.testing.assert_array_equal(entropic_bins, 0)


def test_legacy_c51_policy_mode_selects_bins_by_distributional_expectation(
    monkeypatch,
):
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(
            "method.value_mode=scalar",
            "method.atom_ce_lambda=0",
            "method.demo_fosd=false",
            "method.separate_bc_policy=true",
            "method.distinct_policy_encoder=true",
            "method.freeze_bc_policy=true",
            "method.bc_policy_mode=legacy_c51",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    features = jnp.zeros((1, 5), dtype=jnp.float32)
    initialized_outputs = agent.policy_model.apply(
        agent.params["policy"],
        features,
        jnp.ones((1, agent.levels), dtype=jnp.float32),
        jnp.zeros(
            (1, agent.action_sequence, agent.action_dim),
            dtype=jnp.float32,
        ),
    )
    assert initialized_outputs.shape[-1] == agent.atoms

    class FakeLegacyPolicy:
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
            # Bin 0 puts mass on v_min, whereas bin 4 puts mass on v_max.
            logits = logits.at[..., 0, 0].set(20.0)
            return logits.at[..., 4, -1].set(20.0)

    monkeypatch.setattr(agent, "policy_model", FakeLegacyPolicy())

    _, selected_bins = agent._policy_action(
        agent.params["policy"],
        features,
    )

    assert agent.bc_policy_mode == "legacy_c51"
    np.testing.assert_array_equal(selected_bins, 4)


def test_flow_v_direct_a_updates_each_role_and_centers_action_advantage():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(
            "method.value_mode=scalar",
            "method.atom_ce_lambda=0",
            "method.demo_fosd=false",
            "method.separate_bc_policy=true",
            "method.td_target_action_source=replay_next",
            "method.critic_sequence_mode=effective_k0",
            "method.mc_return_weight=0.1",
            "method.critic_architecture=flow_v_direct_a",
            "method.structured_exploration_level=0",
            "method.weight_decay=0",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    batch = _batch()
    batch["mc_return"] = np.asarray([0.8, -0.2], np.float32)
    agent.logging = True
    flow_before = jax.tree.map(np.asarray, agent.params["critic"])
    advantage_before = jax.tree.map(np.asarray, agent.params["advantage"])
    policy_before = jax.tree.map(np.asarray, agent.params["policy"])

    features = jnp.asarray(batch["low_dim_state"][:, -1])
    _, _, _, all_advantage = agent._advantage_outputs_per_level(
        agent.params["advantage"],
        features,
        batch["action"],
    )
    np.testing.assert_allclose(
        all_advantage.mean(axis=-1),
        0.0,
        atol=1e-6,
    )

    metrics = agent.update(iter([batch]), step=1)

    assert agent.critic_model.value_only
    assert set(agent.target_critic_params) == {"critic", "advantage"}
    assert _tree_changed(flow_before, agent.params["critic"])
    assert _tree_changed(advantage_before, agent.params["advantage"])
    assert _tree_changed(policy_before, agent.params["policy"])
    assert metrics["bcfm_loss"] > 0.0
    assert metrics["advantage_c51_loss"] > 0.0
    assert metrics["advantage_q_loss"] > 0.0
    assert metrics["mc_return_loss"] > 0.0
    assert metrics["policy_ce"] > 0.0
    assert np.isfinite(metrics["total_loss"])


def test_flow_v_direct_a_can_freeze_bc_policy_and_distinct_encoder_exactly():
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
        _compose_cqn_flow(
            "pixels=true",
            "method.value_mode=scalar",
            "method.atom_ce_lambda=0",
            "method.demo_fosd=false",
            "method.separate_bc_policy=true",
            "method.distinct_policy_encoder=true",
            "method.freeze_bc_policy=true",
            "method.td_target_action_source=replay_next",
            "method.critic_sequence_mode=effective_k0",
            "method.mc_return_weight=0.1",
            "method.critic_architecture=flow_v_direct_a",
            "method.structured_exploration_level=0",
            # Exercise the AdamW path: zero gradients alone are not enough
            # to freeze parameters when decoupled weight decay is active.
            "method.weight_decay=0.1",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    batch = _batch()
    batch["rgb_front"] = np.zeros((2, 1, 3, 84, 84), dtype=np.uint8)
    batch["rgb_front_tp1"] = np.ones(
        (2, 1, 3, 84, 84),
        dtype=np.uint8,
    )
    batch["mc_return"] = np.asarray([0.8, -0.2], np.float32)
    agent.logging = True
    critic_before = jax.tree.map(np.asarray, agent.params["critic"])
    advantage_before = jax.tree.map(np.asarray, agent.params["advantage"])
    policy_before = jax.tree.map(np.asarray, agent.params["policy"])
    policy_encoder_before = jax.tree.map(
        np.asarray,
        agent.params["policy_encoder"],
    )

    metrics = agent.update(iter([batch]), step=1)

    assert _tree_changed(critic_before, agent.params["critic"])
    assert _tree_changed(advantage_before, agent.params["advantage"])
    assert _tree_exactly_equal(policy_before, agent.params["policy"])
    assert _tree_exactly_equal(
        policy_encoder_before,
        agent.params["policy_encoder"],
    )
    assert metrics["policy_ce"] > 0.0
    assert metrics["policy_bc_loss"] > 0.0
    assert metrics["policy_encoder_grad_norm"] == pytest.approx(0.0, abs=1e-8)


def test_flow_v_direct_a_trains_from_static_causal_branch_cache(tmp_path):
    observation_space, action_space = _spaces()
    rng = np.random.default_rng(19)
    cache_path = tmp_path / "branches.npz"
    branch_returns = np.zeros((6, 5), dtype=np.float32)
    branch_returns[:4] = np.asarray(
        [-0.2, -0.1, 0.0, 0.1, 0.2],
        dtype=np.float32,
    )
    np.savez_compressed(
        cache_path,
        train_features=rng.normal(size=(6, 5)).astype(np.float32),
        train_actions=rng.uniform(
            -1.0,
            1.0,
            size=(6, 5, 2, 2),
        ).astype(np.float32),
        train_returns=branch_returns,
        train_action_dimensions=np.asarray([0, 1, 0, 1, 0, 1], np.int32),
    )
    agent = create_agent(
        _compose_cqn_flow(
            "method.value_mode=scalar",
            "method.atom_ce_lambda=0",
            "method.demo_fosd=false",
            "method.separate_bc_policy=true",
            "method.td_target_action_source=replay_next",
            "method.critic_sequence_mode=effective_k0",
            "method.mc_return_weight=0.1",
            "method.critic_architecture=flow_v_direct_a",
            "method.structured_exploration_level=0",
            "method.weight_decay=0",
            f"method.causal_branch_cache={cache_path}",
            "method.causal_branch_weight=0.1",
            "method.causal_branch_delta_weight=10",
            "method.causal_branch_batch_size=4",
            "method.causal_branch_level=0",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    batch = _batch()
    batch["mc_return"] = np.asarray([0.8, -0.2], np.float32)
    agent.logging = True
    advantage_before = jax.tree.map(np.asarray, agent.params["advantage"])

    metrics = agent.update(iter([batch]), step=1)

    assert _tree_changed(advantage_before, agent.params["advantage"])
    assert metrics["causal_branch_loss"] > 0.0
    assert metrics["causal_branch_ranking_loss"] > 0.0
    assert metrics["causal_branch_delta_loss"] >= 0.0
    assert 0.0 <= metrics["causal_branch_pairwise_accuracy"] <= 1.0
    assert metrics["causal_branch_q_span"] >= 0.0
    assert np.isfinite(metrics["total_loss"])


def test_hybrid_policy_value_action_combines_advantage_and_bc_prior(monkeypatch):
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(
            "method.value_mode=scalar",
            "method.atom_ce_lambda=0",
            "method.demo_fosd=false",
            "method.separate_bc_policy=true",
            "method.td_target_action_source=replay_next",
            "method.critic_sequence_mode=effective_k0",
            "method.mc_return_weight=0.1",
            "method.critic_architecture=flow_v_direct_a",
            "method.structured_exploration_level=0",
            "method.policy_value_beta=0",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    advantage = jnp.asarray(
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        dtype=jnp.float32,
    )

    def fake_advantage_level(_params, features, _level, _low, _high):
        values = jnp.broadcast_to(
            advantage,
            (
                features.shape[0],
                agent.action_sequence,
                agent.action_dim,
                agent.bins,
            ),
        )
        return jnp.zeros((*values.shape, agent.atoms)), values

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

    monkeypatch.setattr(agent, "_advantage_level", fake_advantage_level)
    monkeypatch.setattr(agent, "policy_model", FakePolicy())
    features = jnp.zeros((1, 5), dtype=jnp.float32)

    _, a_only_bins = agent._hybrid_policy_value_action(
        agent.params["advantage"],
        features,
        agent.params["policy"],
        features,
    )
    agent.policy_value_beta = 4.0
    _, blended_bins = agent._hybrid_policy_value_action(
        agent.params["advantage"],
        features,
        agent.params["policy"],
        features,
    )

    np.testing.assert_array_equal(a_only_bins, 4)
    np.testing.assert_array_equal(blended_bins, 0)


def test_decoupled_categorical_flow_uses_completed_return_ce():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(
            "method.value_mode=categorical",
            "method.atom_ce_lambda=1",
            "method.demo_fosd=false",
            "method.separate_bc_policy=true",
            "method.td_target_action_source=replay_next",
            "method.critic_sequence_mode=effective_k0",
            "method.mc_return_weight=0.1",
            "method.structured_exploration_level=0",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    batch = _batch()
    batch["mc_return"] = np.asarray([0.8, -0.2], np.float32)
    agent.logging = True

    metrics = agent.update(iter([batch]), step=1)

    assert metrics["atom_ce_loss"] > 0.0
    assert metrics["mc_return_loss"] > 0.0
    assert metrics["mc_return_mae"] > 0.0
    assert np.isfinite(metrics["total_loss"])


def test_floq_coherent_value_gate_matches_direct_causal_protocol():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_flow_floq_coherent_value_gate",
                "env=bigym/move_plate",
            ],
        )

    assert cfg.method.name == "cqn_flow"
    assert cfg.method.value_mode == "scalar"
    assert cfg.method.separate_bc_policy
    assert cfg.method.td_target_action_source == "replay_next"
    assert cfg.method.critic_sequence_mode == "effective_k0"
    assert cfg.method.mc_return_weight == pytest.approx(0.1)
    assert cfg.method.structured_exploration_prob == pytest.approx(0.06)
    assert cfg.method.structured_exploration_level == 1
    assert cfg.method.structured_exploration_horizon == 4
    assert cfg.method.endpoint_q_lambda == pytest.approx(1.0)
    assert cfg.method.source_consistency_lambda == pytest.approx(0.1)
    assert cfg.env.truncate_demo_at_success
    assert not cfg.use_self_imitation
    assert _mc_return_anchor_enabled(cfg)
    assert _structured_exploration_enabled(cfg)


def test_stage_x_hybrid_launch_uses_compute8_two_tower_causal_protocol():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_flow_v_direct_a_coherent_value_gate",
                "env=bigym/move_plate",
            ],
        )

    assert cfg.method.critic_architecture == "flow_v_direct_a"
    assert cfg.method.value_mode == "scalar"
    assert cfg.method.num_flow_steps == 8
    assert cfg.method.num_flow_samples == 8
    assert cfg.method.separate_bc_policy
    assert cfg.method.distinct_policy_encoder
    assert cfg.method.td_target_action_source == "replay_next"
    assert cfg.method.mc_return_weight == pytest.approx(0.1)
    assert cfg.method.structured_exploration_horizon == 4
    assert _mc_return_anchor_enabled(cfg)
    assert _structured_exploration_enabled(cfg)


def test_frozen_bc_value_gate_preserves_bc_tower_and_trains_value_side():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_flow_causal_a_frozen_bc_value_gate",
                "env=bigym/move_plate",
            ],
        )

    assert cfg.method.critic_architecture == "flow_v_direct_a"
    assert cfg.method.separate_bc_policy
    assert cfg.method.distinct_policy_encoder
    assert cfg.method.freeze_bc_policy
    assert cfg.method.policy_value_beta is None
    assert cfg.method.causal_branch_weight == pytest.approx(0.1)
    assert cfg.method.causal_branch_delta_weight == pytest.approx(10.0)


def test_cqn_flow_replay_contract_disables_optional_fields_at_zero_weight():
    cfg = _compose_cqn_flow(
        "method.mc_return_weight=0",
        "method.structured_exploration_prob=0",
    )

    assert not _mc_return_anchor_enabled(cfg)
    assert not _structured_exploration_enabled(cfg)


@pytest.mark.parametrize(
    ("launch", "value_mode", "steps", "samples"),
    [
        ("cqn_flow_c51_coherent_value_gate", "categorical", 4, 4),
        ("cqn_flow_floq_compute8_coherent_value_gate", "scalar", 8, 8),
    ],
)
def test_next_flow_value_gates_preserve_causal_protocol(
    launch, value_mode, steps, samples
):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[f"launch={launch}", "env=bigym/move_plate"],
        )

    assert cfg.method.value_mode == value_mode
    assert cfg.method.num_flow_steps == steps
    assert cfg.method.num_flow_samples == samples
    assert cfg.method.separate_bc_policy
    assert cfg.method.td_target_action_source == "replay_next"
    assert cfg.method.critic_sequence_mode == "effective_k0"
    assert cfg.method.mc_return_weight == pytest.approx(0.1)
    assert cfg.method.structured_exploration_horizon == 4
    assert _mc_return_anchor_enabled(cfg)
    assert _structured_exploration_enabled(cfg)


def test_floq_distill_gate_trains_readout_without_changing_bc_collection():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_flow_floq_distill_two_tower_high_utd4_gate",
                "env=bigym/move_plate",
            ],
        )

    assert cfg.method.value_mode == "scalar"
    assert cfg.method.num_update_steps == 4
    assert cfg.method.distinct_policy_encoder
    assert cfg.method.flow_distill_lambda == pytest.approx(1.0)
    assert not cfg.method.flow_distill_action_readout
    assert cfg.method.policy_value_beta is None
    assert cfg.method.td_target_action_source == "replay_next"


def test_floq_policy_value_td_arm_changes_only_target_policy():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        parent = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_flow_floq_distill_two_tower_high_utd4_gate",
                "env=bigym/move_plate",
            ],
        )
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_flow_floq_td_policy_value_two_tower_high_utd4_gate",
                "env=bigym/move_plate",
            ],
        )

    assert cfg.method.value_mode == "scalar"
    assert cfg.method.num_flow_steps == 8
    assert cfg.method.num_flow_samples == 8
    assert cfg.method.num_update_steps == 4
    assert cfg.method.td_target_action_source == "policy_value"
    assert cfg.method.td_target_policy_value_beta == pytest.approx(1.0)
    assert cfg.method.policy_value_beta is None
    assert cfg.method.separate_bc_policy
    assert cfg.method.distinct_policy_encoder
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


def test_floq_bc_policy_td_arm_changes_only_target_action_source():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        parent = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_flow_floq_distill_two_tower_high_utd4_gate",
                "env=bigym/move_plate",
            ],
        )
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_flow_floq_td_bc_policy_two_tower_high_utd4_gate",
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


@pytest.mark.parametrize(
    ("launch", "source_min", "source_max", "bcfm_lambda"),
    [
        (
            "cqn_flow_floq_source01_distill_two_tower_high_utd4_gate",
            0.0,
            0.1,
            1.0,
        ),
        (
            "cqn_flow_floq_bcfm8_distill_two_tower_high_utd4_gate",
            None,
            None,
            8.0,
        ),
        (
            "cqn_flow_floq_source01_bcfm8_distill_two_tower_high_utd4_gate",
            0.0,
            0.1,
            8.0,
        ),
    ],
)
def test_floq_fidelity_arms_materialize_declared_variables(
    launch,
    source_min,
    source_max,
    bcfm_lambda,
):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[f"launch={launch}", "env=bigym/move_plate"],
        )

    assert cfg.method.value_mode == "scalar"
    assert cfg.method.num_flow_steps == 8
    assert cfg.method.num_flow_samples == 8
    assert cfg.method.num_target_flow_samples == 8
    assert cfg.method.num_action_flow_samples == 8
    assert cfg.method.flow_source_type == "uniform"
    assert cfg.method.flow_source_min == source_min
    assert cfg.method.flow_source_max == source_max
    assert cfg.method.bcfm_lambda == pytest.approx(bcfm_lambda)
    assert cfg.method.flow_distill_lambda == pytest.approx(1.0)
    assert cfg.method.num_update_steps == 4
    assert cfg.method.separate_bc_policy
    assert cfg.method.distinct_policy_encoder
    assert cfg.method.td_target_action_source == "replay_next"
    assert cfg.method.critic_sequence_mode == "effective_k0"
    assert cfg.method.mc_return_weight == pytest.approx(0.1)
    assert cfg.method.policy_value_beta is None


def test_clean_cqn_as_repro_gate_materializes_exact_7500_checkpoint():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=cqn_as_pixel_bigym_value_fidelity_repro_gate",
                "env=bigym/move_plate",
            ],
        )

    assert cfg.method.name == "cqn_as"
    assert cfg.env.truncate_demo_at_success
    assert cfg.eval_every_steps == 2500
    assert cfg.num_eval_episodes == 25
    assert cfg.snapshot_every_n == 500
    assert cfg.save_snapshot
    assert cfg.save_csv


def test_workspace_allows_cqn_flow_action_sequence():
    cfg = _compose_cqn_flow()

    _validate_rl_action_sequence(cfg)


def test_cqn_flow_reuses_cqn_as_frozen_pixel_feature_namespace():
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
        _compose_cqn_flow(
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

    assert agent._cached_pixel_feature_key == JAX_CQN_AS_FEATURE_KEY
    assert next_features.shape == (2, 517)
    assert np.all(np.isfinite(np.asarray(next_features)))


def test_cqn_flow_source_is_shared_across_bins_for_stable_ranking():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow("method.num_flow_samples=3"),
        observation_space=observation_space,
        action_space=action_space,
    )

    source = agent._flow_source(
        jax.random.PRNGKey(5),
        batch_size=2,
        query_bins=agent.bins,
    )

    assert source.shape == (2, 3, 2, 2, 5, 5)
    np.testing.assert_allclose(
        source,
        jnp.broadcast_to(source[..., :1, :], source.shape),
        atol=0.0,
    )
    np.testing.assert_allclose(source.mean(axis=-1), 0.0, atol=1e-6)


def test_uniform_antithetic_flow_sources_respect_bounds_and_pair_exactly():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(
            "method.value_mode=return_sample",
            "method.atom_ce_lambda=0",
            "method.demo_fosd=false",
            "method.num_flow_samples=4",
            "method.num_target_flow_samples=4",
            "method.flow_source_type=uniform",
            "method.flow_source_min=-0.2",
            "method.flow_source_max=0.4",
            "method.antithetic_flow_sources=true",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )

    source = agent._flow_source(
        jax.random.PRNGKey(11),
        batch_size=2,
        query_bins=agent.bins,
        num_samples=4,
    )

    assert source.shape == (2, 4, 2, 2, 5, 1)
    assert np.all(np.asarray(source) >= -0.2)
    assert np.all(np.asarray(source) <= 0.4)
    np.testing.assert_allclose(
        source,
        jnp.broadcast_to(source[..., :1, :], source.shape),
        atol=0.0,
    )
    # Four samples contain two draws followed by their reflections about the
    # uniform interval midpoint: x + x_antithetic == min + max.
    np.testing.assert_allclose(
        source[:, :2] + source[:, 2:],
        0.2,
        atol=1e-7,
    )


def test_demo_top1_splits_credit_across_equal_q_ties():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(
            "method.value_mode=scalar",
            "method.atom_ce_lambda=0",
            "method.demo_fosd=false",
            "method.num_flow_samples=2",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    all_endpoints = jnp.zeros((1, 2, 1, 2, 2, 5, 1), dtype=jnp.float32)
    chosen_endpoints = jnp.zeros((1, 2, 1, 2, 2, 1), dtype=jnp.float32)

    losses = agent._demo_losses(all_endpoints, chosen_endpoints)

    expert_top1 = losses[2]
    np.testing.assert_allclose(expert_top1, 1.0 / agent.bins, atol=1e-7)


def test_source_bin_flip_rate_compares_each_source_to_mean_q_choice():
    all_q_samples = jnp.asarray(
        [
            [[[[[3.0, 1.0, 0.0]]]], [[[[0.0, 3.0, 1.0]]]], [[[[4.0, 0.0, 1.0]]]]],
            [[[[[0.0, 1.0, 3.0]]]], [[[[0.0, 2.0, 4.0]]]], [[[[1.0, 0.0, 2.0]]]]],
        ],
        dtype=jnp.float32,
    )

    rates = source_bin_flip_rate_per_sample(all_q_samples)

    np.testing.assert_allclose(rates, [1.0 / 3.0, 0.0], atol=1e-7)


def test_source_resampling_probe_forces_independent_full_ranking_paths(monkeypatch):
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(
            "method.value_mode=scalar",
            "method.atom_ce_lambda=0",
            "method.demo_fosd=false",
            "method.fixed_action_flow_sources=true",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    calls = []

    def alternating_sources(key, batch_size, query_bins, num_samples):
        del key
        calls.append((batch_size, query_bins, num_samples))
        signs = jnp.where(
            jnp.arange(batch_size) % 2 == 0,
            -1.0,
            1.0,
        ).reshape((batch_size, 1, 1, 1, 1, 1))
        return jnp.broadcast_to(
            signs,
            (
                batch_size,
                1,
                agent.action_sequence,
                agent.action_dim,
                query_bins,
                1,
            ),
        )

    def source_ranked_endpoints(
        critic_params,
        features,
        condition,
        key,
        *,
        source,
        num_samples,
    ):
        del critic_params, features, key
        assert num_samples == 1
        candidate_bins = condition[3]
        return source * candidate_bins[:, None, ..., None].astype(jnp.float32)

    monkeypatch.setattr(agent, "_flow_source", alternating_sources)
    monkeypatch.setattr(agent, "_integrate_level", source_ranked_endpoints)
    observations = {
        "low_dim_state": np.zeros((2, 1, 5), dtype=np.float32),
    }
    rng_before = np.asarray(agent.rng_key).copy()

    metrics = agent.source_resampling_ranking_probe(
        observations,
        num_source_draws=4,
        seed=31,
        use_target_network=False,
    )

    # The configured fixed rollout bank is bypassed: one independently indexed
    # source is explicitly generated for every observation/draw row.
    assert agent.fixed_action_flow_sources
    assert calls == [(8, agent.bins, 1)]
    np.testing.assert_array_equal(agent.rng_key, rng_before)
    np.testing.assert_allclose(metrics["per_level_bin_agreement"], [0.5])
    np.testing.assert_allclose(metrics["per_level_bin_flip_rate"], [0.5])
    np.testing.assert_allclose(metrics["per_level_q_span"], [4.0])
    np.testing.assert_allclose(metrics["per_level_top2_gap"], [1.0])
    np.testing.assert_allclose(metrics["per_level_source_q_std"], [2.0])
    np.testing.assert_allclose(metrics["per_level_rank_snr"], [0.5], atol=1e-6)
    np.testing.assert_allclose(metrics["action_mean"], 0.0, atol=1e-7)
    np.testing.assert_allclose(metrics["action_source_std"], 0.8, atol=1e-6)
    np.testing.assert_allclose(metrics["action_source_std_mean"], 0.8, atol=1e-6)
    np.testing.assert_allclose(metrics["action_source_std_max"], 0.8, atol=1e-6)
    assert metrics["selected_bins"].shape == (2, 4, 1, 2, 2)


def test_source_resampling_probe_rejects_single_draw():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(),
        observation_space=observation_space,
        action_space=action_space,
    )

    with pytest.raises(ValueError, match="at least 2"):
        agent.source_resampling_ranking_probe(
            {"low_dim_state": np.zeros((1, 1, 5), dtype=np.float32)},
            num_source_draws=1,
        )


def test_source_resampling_probe_averages_action_flow_samples(monkeypatch):
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(
            "method.value_mode=scalar",
            "method.atom_ce_lambda=0",
            "method.demo_fosd=false",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    calls = []

    def balanced_sources(key, batch_size, query_bins, num_samples):
        del key
        calls.append((batch_size, query_bins, num_samples))
        assert num_samples == 2
        signs = jnp.asarray([-1.0, 1.0]).reshape((1, 2, 1, 1, 1, 1))
        return jnp.broadcast_to(
            signs,
            (
                batch_size,
                num_samples,
                agent.action_sequence,
                agent.action_dim,
                query_bins,
                1,
            ),
        )

    def source_ranked_endpoints(
        critic_params,
        features,
        condition,
        key,
        *,
        source,
        num_samples,
    ):
        del critic_params, features, key
        assert num_samples == 2
        candidate_bins = condition[3]
        return source * candidate_bins[:, None, ..., None].astype(jnp.float32)

    monkeypatch.setattr(agent, "_flow_source", balanced_sources)
    monkeypatch.setattr(agent, "_integrate_level", source_ranked_endpoints)
    metrics = agent.source_resampling_ranking_probe(
        {"low_dim_state": np.zeros((1, 1, 5), dtype=np.float32)},
        num_source_draws=4,
        num_action_flow_samples=2,
        seed=37,
        use_target_network=False,
    )

    # Every group contains a balanced negative/positive pair.  Averaging the
    # two endpoints makes all group rankings identical; reading only the first
    # endpoint would instead select the opposite edge bin.
    assert calls == [(4, agent.bins, 2)]
    np.testing.assert_allclose(metrics["per_level_bin_agreement"], [1.0])
    np.testing.assert_allclose(metrics["per_level_bin_flip_rate"], [0.0])
    np.testing.assert_allclose(metrics["per_level_source_q_std"], [0.0])
    assert metrics["selected_bins"].shape == (1, 4, 1, 2, 2)


def test_selected_endpoint_query_matches_gather_from_parallel_all_bins():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(),
        observation_space=observation_space,
        action_space=action_space,
    )
    batch = _batch(batch_size=2)
    agent.update(iter([batch]), step=1)
    features = jnp.zeros((2, 5), dtype=jnp.float32)
    actions = jnp.asarray(batch["action"])
    key = jax.random.PRNGKey(17)

    selected_only = agent._selected_endpoints_per_level(
        agent.params["critic"], features, actions, key
    )
    selected_from_all, _ = agent._endpoints_per_level(
        agent.params["critic"], features, actions, key
    )

    np.testing.assert_allclose(selected_only, selected_from_all, atol=1e-6)


def test_one_step_demo_readout_is_source_plus_initial_velocity():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(
            "method.value_mode=return_sample",
            "method.atom_ce_lambda=0",
            "method.demo_fosd=false",
            "method.num_flow_steps=3",
            "method.demo_flow_steps=1",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    batch_size = 2
    features = jnp.zeros((batch_size, 5), dtype=jnp.float32)
    actions = jnp.zeros((batch_size, 2, 2), dtype=jnp.float32)
    key = jax.random.PRNGKey(23)

    _, all_readouts = agent._endpoints_per_level(
        agent.params["critic"],
        features,
        actions,
        key,
        num_flow_steps=1,
    )

    level_key = jax.random.split(key, agent.levels)[0]
    low = jnp.broadcast_to(agent.action_low, (batch_size, agent._flat_action_dim))
    high = jnp.broadcast_to(agent.action_high, (batch_size, agent._flat_action_dim))
    condition = agent._level_condition(low, high, level=0)
    source = agent._flow_source(
        level_key,
        batch_size,
        agent.bins,
        num_samples=agent.num_action_flow_samples,
    )
    velocity = agent.critic_model.apply(
        agent.params["critic"],
        features,
        *condition,
        source,
        jnp.ones((batch_size, agent.num_action_flow_samples), dtype=jnp.float32),
    )
    expected = source + velocity
    if agent.clip_flow_trajectory:
        expected = jnp.clip(expected, agent.v_min, agent.v_max)

    np.testing.assert_allclose(all_readouts[:, :, 0], expected, atol=1e-6)


def test_demo_diagnostics_are_detached_at_zero_source_variance():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(
            "method.value_mode=return_sample",
            "method.atom_ce_lambda=0",
            "method.demo_fosd=false",
            "method.num_flow_samples=2",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    all_endpoints = jnp.zeros((1, 2, 1, 2, 2, 5, 1), dtype=jnp.float32)
    chosen_endpoints = jnp.zeros((1, 2, 1, 2, 2, 1), dtype=jnp.float32)

    def diagnostic_sum(all_values, chosen_values):
        outputs = agent._demo_losses(all_values, chosen_values)
        return sum(jnp.sum(value) for value in outputs[2:])

    all_grads, chosen_grads = jax.grad(
        diagnostic_sum,
        argnums=(0, 1),
    )(all_endpoints, chosen_endpoints)

    assert np.all(np.isfinite(all_grads))
    assert np.all(np.isfinite(chosen_grads))
    np.testing.assert_array_equal(all_grads, 0.0)
    np.testing.assert_array_equal(chosen_grads, 0.0)


def test_v2_endpoint_ce_backpropagates_through_full_ode_endpoint():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow("method.num_flow_steps=2"),
        observation_space=observation_space,
        action_space=action_space,
    )
    batch_size = 1
    features = jnp.zeros((batch_size, 5), dtype=jnp.float32)
    actions = jnp.zeros((batch_size, 2, 2), dtype=jnp.float32)
    target = jnp.broadcast_to(
        jax.nn.one_hot(4, 5, dtype=jnp.float32),
        (batch_size, agent.levels, 2, 2, 5),
    )

    def endpoint_ce(critic_params):
        chosen_endpoints = agent._selected_endpoints_per_level(
            critic_params,
            features,
            actions,
            jax.random.PRNGKey(9),
        )
        ce = categorical_cross_entropy(target[:, None], chosen_endpoints)
        return ce.mean(), chosen_endpoints

    (loss, endpoints), grads = jax.value_and_grad(
        endpoint_ce,
        has_aux=True,
    )(agent.params["critic"])
    grad_norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree.leaves(grads))
    )

    assert endpoints.shape == (1, 1, 1, 2, 2, 5)
    assert np.isfinite(float(loss))
    assert float(grad_norm) > 0.0


def test_cqn_flow_level_condition_tracks_zoom_interval_and_bin_centers():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow("method.levels=2"),
        observation_space=observation_space,
        action_space=action_space,
    )
    low = jnp.broadcast_to(agent.action_low, (1, agent._flat_action_dim))
    high = jnp.broadcast_to(agent.action_high, (1, agent._flat_action_dim))

    _, midpoint, half_width, candidate_bins, centers = agent._level_condition(
        low,
        high,
        level=0,
    )

    np.testing.assert_allclose(midpoint, 0.0, atol=1e-6)
    np.testing.assert_allclose(half_width, 1.0, atol=1e-6)
    np.testing.assert_array_equal(candidate_bins[0, 0, 0], np.arange(5))
    np.testing.assert_allclose(
        centers[0, 0, 0],
        [-0.8, -0.4, 0.0, 0.4, 0.8],
        atol=1e-6,
    )

    zoomed_low = jnp.full_like(low, 0.6)
    zoomed_high = jnp.full_like(high, 1.0)
    _, midpoint, half_width, _, centers = agent._level_condition(
        zoomed_low,
        zoomed_high,
        level=1,
    )

    np.testing.assert_allclose(midpoint, 0.8, atol=1e-6)
    np.testing.assert_allclose(half_width, 0.2, atol=1e-6)
    np.testing.assert_allclose(
        centers[0, 0, 0],
        [0.64, 0.72, 0.8, 0.88, 0.96],
        atol=1e-6,
    )


def test_scalar_terminal_target_is_reward_without_flow_bootstrap(monkeypatch):
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(
            "method.value_mode=scalar",
            "method.atom_ce_lambda=0",
            "method.demo_fosd=false",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    batch_size = 2
    endpoints = jnp.full(
        (
            batch_size,
            agent.num_flow_samples,
            agent.levels,
            agent.action_sequence,
            agent.action_dim,
            1,
        ),
        100.0,
        dtype=jnp.float32,
    )
    monkeypatch.setattr(
        agent,
        "_selected_endpoints_per_level",
        lambda *_args, **_kwargs: endpoints,
    )
    rewards = jnp.asarray([0.25, -0.5], dtype=jnp.float32)

    targets = agent._target_values(
        agent.target_critic_params,
        jnp.zeros((batch_size, 5), dtype=jnp.float32),
        jnp.zeros((batch_size, 2, 2), dtype=jnp.float32),
        rewards,
        jnp.full((batch_size,), 0.99, dtype=jnp.float32),
        jnp.zeros((batch_size,), dtype=jnp.float32),
        jax.random.PRNGKey(0),
    )

    expected = jnp.broadcast_to(rewards[:, None, None, None], targets.shape)
    np.testing.assert_allclose(targets, expected, atol=1e-6)


def test_return_sample_terminal_target_preserves_source_sample_axis(monkeypatch):
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(
            "method.value_mode=return_sample",
            "method.atom_ce_lambda=0",
            "method.demo_fosd=false",
            "method.num_flow_samples=3",
            "method.num_target_flow_samples=3",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    batch_size = 2
    endpoints = jnp.full(
        (
            batch_size,
            agent.num_target_flow_samples,
            agent.levels,
            agent.action_sequence,
            agent.action_dim,
            1,
        ),
        100.0,
        dtype=jnp.float32,
    )
    monkeypatch.setattr(
        agent,
        "_selected_endpoints_per_level",
        lambda *_args, **_kwargs: endpoints,
    )
    rewards = jnp.asarray([0.25, -0.5], dtype=jnp.float32)

    targets = agent._target_values(
        agent.target_critic_params,
        jnp.zeros((batch_size, 5), dtype=jnp.float32),
        jnp.zeros((batch_size, 2, 2), dtype=jnp.float32),
        rewards,
        jnp.full((batch_size,), 0.99, dtype=jnp.float32),
        jnp.zeros((batch_size,), dtype=jnp.float32),
        jax.random.PRNGKey(0),
    )

    assert targets.shape == (2, 3, 1, 2, 2)
    expected = jnp.broadcast_to(
        rewards[:, None, None, None, None],
        targets.shape,
    )
    np.testing.assert_allclose(targets, expected, atol=1e-6)


def test_return_sample_dcfm_matches_bellman_partial_flow_golden(monkeypatch):
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(
            "method.value_mode=return_sample",
            "method.atom_ce_lambda=0",
            "method.demo_fosd=false",
            "method.num_flow_samples=2",
            "method.num_target_flow_samples=2",
            "method.dcfm_lambda=1",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    batch_size = 2
    source = jnp.arange(
        batch_size
        * agent.num_flow_samples
        * agent.action_sequence
        * agent.action_dim,
        dtype=jnp.float32,
    ).reshape(
        (
            batch_size,
            agent.num_flow_samples,
            agent.action_sequence,
            agent.action_dim,
            1,
            1,
        )
    ) / 10.0
    tau = jnp.asarray([[0.25, 0.75], [0.4, 0.6]], dtype=jnp.float32)
    tau_values = tau[..., None, None, None, None]
    partial_next_return = source + 2.0 * tau_values
    calls = {"source": [], "integrate": [], "apply": []}
    record_calls = True

    def fake_flow_source(key, batch_size_arg, query_bins, num_samples):
        assert batch_size_arg == batch_size
        assert query_bins == 1
        assert num_samples == agent.num_flow_samples
        if record_calls:
            calls["source"].append(key)
        return source

    def fake_integrate_level(
        params,
        features,
        condition,
        key,
        *,
        source,
        end_tau,
    ):
        del params, features, condition
        if record_calls:
            calls["integrate"].append((key, source, end_tau))
        return source + 2.0 * end_tau[..., None, None, None, None]

    class GoldenCritic:
        def apply(
            self,
            params,
            features,
            level_one_hot,
            midpoint,
            half_width,
            candidate_bins,
            centers,
            values,
            time,
        ):
            del (
                features,
                level_one_hot,
                midpoint,
                half_width,
                candidate_bins,
                centers,
            )
            if record_calls:
                calls["apply"].append((params["bias"], values, time))
            return values + params["bias"] + (
                3.0 * time[..., None, None, None, None]
            )

    monkeypatch.setattr(agent, "_flow_source", fake_flow_source)
    monkeypatch.setattr(agent, "_integrate_level", fake_integrate_level)
    monkeypatch.setattr(agent, "critic_model", GoldenCritic())
    monkeypatch.setattr(jax.random, "uniform", lambda *_args, **_kwargs: tau)

    current_params = {"bias": jnp.asarray(2.0, dtype=jnp.float32)}
    target_params = {"bias": jnp.asarray(5.0, dtype=jnp.float32)}
    features = jnp.zeros((batch_size, 5), dtype=jnp.float32)
    next_features = jnp.ones_like(features)
    actions = jnp.zeros(
        (batch_size, agent.action_sequence, agent.action_dim),
        dtype=jnp.float32,
    )
    rewards = jnp.asarray([1.0, -3.0], dtype=jnp.float32)
    discounts = jnp.asarray([0.5, 0.75], dtype=jnp.float32)
    bootstrap = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    source_key = jax.random.PRNGKey(11)

    actual = agent._dcfm_loss(
        current_params,
        target_params,
        features,
        next_features,
        actions,
        actions,
        rewards,
        discounts,
        bootstrap,
        source_key,
        jax.random.PRNGKey(12),
    )

    current_input = rewards[:, None, None, None, None, None] + (
        discounts * bootstrap
    )[:, None, None, None, None, None] * partial_next_return
    current_velocity = current_input + current_params["bias"] + 3.0 * tau_values
    # DCFM compares directly to the next-state velocity. In particular, there
    # is intentionally no second multiplication by the Bellman discount here.
    target_velocity = (
        partial_next_return + target_params["bias"] + 3.0 * tau_values
    )
    expected = jnp.square(current_velocity - target_velocity)
    expected = expected * bootstrap[:, None, None, None, None, None]
    expected = expected.mean(axis=(1, 2, 3, 4, 5))
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert actual[1] == pytest.approx(0.0, abs=1e-8)

    assert len(calls["source"]) == len(calls["integrate"]) == 1
    integrate_key, integrated_source, integrated_tau = calls["integrate"][0]
    np.testing.assert_array_equal(calls["source"][0], integrate_key)
    np.testing.assert_allclose(integrated_source, source, atol=0.0)
    np.testing.assert_allclose(integrated_tau, tau, atol=0.0)
    assert len(calls["apply"]) == 2
    _, observed_current_input, current_tau = calls["apply"][0]
    _, observed_target_input, target_tau = calls["apply"][1]
    np.testing.assert_allclose(observed_current_input, current_input, atol=1e-6)
    np.testing.assert_allclose(
        observed_target_input,
        partial_next_return,
        atol=1e-6,
    )
    np.testing.assert_allclose(current_tau, tau, atol=0.0)
    np.testing.assert_allclose(target_tau, tau, atol=0.0)

    # The EMA target velocity is an explicit stop-gradient target, while the
    # online field remains trainable. Keep integration independent of the
    # target bias here so the derivative isolates that velocity-side boundary.
    record_calls = False

    def loss_with_biases(current_bias, target_bias):
        return agent._dcfm_loss(
            {"bias": current_bias},
            {"bias": target_bias},
            features,
            next_features,
            actions,
            actions,
            rewards,
            discounts,
            bootstrap,
            source_key,
            jax.random.PRNGKey(12),
        ).sum()

    current_grad, target_grad = jax.grad(loss_with_biases, argnums=(0, 1))(
        current_params["bias"],
        target_params["bias"],
    )
    assert abs(float(current_grad)) > 1e-6
    assert target_grad == pytest.approx(0.0, abs=1e-8)


def test_return_sample_pcbf_matches_source_consistent_path_golden(monkeypatch):
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(
            "method.value_mode=return_sample",
            "method.atom_ce_lambda=0",
            "method.demo_fosd=false",
            "method.num_flow_samples=2",
            "method.num_target_flow_samples=2",
            "method.bcfm_lambda=0",
            "method.pcbf_loss_coeff=1",
            "method.pcbf_lambda=0.4",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    batch_size = 2
    source = jnp.arange(
        batch_size
        * agent.num_flow_samples
        * agent.action_sequence
        * agent.action_dim,
        dtype=jnp.float32,
    ).reshape(
        (
            batch_size,
            agent.num_flow_samples,
            agent.action_sequence,
            agent.action_dim,
            1,
            1,
        )
    ) / 10.0
    # Repository tau runs endpoint <- source. Include both exact boundaries;
    # PCBF's paper time is t = 1 - tau.
    repo_tau = jnp.asarray([[1.0, 0.0], [0.25, 0.75]], dtype=jnp.float32)
    forward_time = (1.0 - repo_tau)[..., None, None, None, None]
    calls = {"source": [], "integrate": [], "apply": []}
    record_calls = True

    def fake_flow_source(key, batch_size_arg, query_bins, num_samples):
        assert batch_size_arg == batch_size
        assert query_bins == 1
        assert num_samples == agent.num_flow_samples
        if record_calls:
            calls["source"].append(key)
        return source

    def fake_integrate_level(
        params,
        features,
        condition,
        key,
        *,
        source,
        end_tau,
    ):
        del features, condition
        if record_calls:
            calls["integrate"].append((key, source, end_tau))
        return source + params["endpoint"]

    class GoldenCritic:
        def apply(
            self,
            params,
            features,
            level_one_hot,
            midpoint,
            half_width,
            candidate_bins,
            centers,
            values,
            time,
        ):
            del (
                features,
                level_one_hot,
                midpoint,
                half_width,
                candidate_bins,
                centers,
            )
            if record_calls:
                calls["apply"].append((params["velocity"], values, time))
            model_forward_time = (1.0 - time)[..., None, None, None, None]
            return 0.25 * values + params["velocity"] + 3.0 * model_forward_time

    monkeypatch.setattr(agent, "_flow_source", fake_flow_source)
    monkeypatch.setattr(agent, "_integrate_level", fake_integrate_level)
    monkeypatch.setattr(agent, "critic_model", GoldenCritic())
    monkeypatch.setattr(
        jax.random,
        "uniform",
        lambda *_args, **_kwargs: repo_tau,
    )

    current_params = {
        "velocity": jnp.asarray(2.0, dtype=jnp.float32),
        "endpoint": jnp.asarray(0.0, dtype=jnp.float32),
    }
    target_params = {
        "velocity": jnp.asarray(5.0, dtype=jnp.float32),
        "endpoint": jnp.asarray(2.5, dtype=jnp.float32),
    }
    features = jnp.zeros((batch_size, 5), dtype=jnp.float32)
    next_features = jnp.ones_like(features)
    actions = jnp.zeros(
        (batch_size, agent.action_sequence, agent.action_dim),
        dtype=jnp.float32,
    )
    rewards = jnp.asarray([1.0, -3.0], dtype=jnp.float32)
    discounts = jnp.asarray([0.5, 0.75], dtype=jnp.float32)
    bootstrap = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    source_key = jax.random.PRNGKey(21)

    actual = agent._pcbf_loss(
        current_params,
        target_params,
        features,
        next_features,
        actions,
        actions,
        rewards,
        discounts,
        bootstrap,
        source_key,
        jax.random.PRNGKey(22),
    )

    next_endpoint = source + target_params["endpoint"]
    successor_sample = (
        (1.0 - forward_time) * source + forward_time * next_endpoint
    )
    reward_values = rewards[:, None, None, None, None, None]
    effective_discount = (discounts * bootstrap)[
        :, None, None, None, None, None
    ]
    bootstrap_values = bootstrap[:, None, None, None, None, None]
    current_sample = (
        forward_time * reward_values
        + effective_discount * successor_sample
        + (1.0 - forward_time) * (1.0 - effective_discount) * source
    )
    next_velocity = (
        0.25 * successor_sample
        + target_params["velocity"]
        + 3.0 * forward_time
    )
    sample_target = reward_values + effective_discount * next_endpoint - source
    correction = next_velocity - (next_endpoint - source)
    target_velocity = sample_target + (
        agent.pcbf_lambda * bootstrap_values * correction
    )
    current_velocity = (
        0.25 * current_sample
        + current_params["velocity"]
        + 3.0 * forward_time
    )
    expected = jnp.square(current_velocity - target_velocity).mean(
        axis=(1, 2, 3, 4, 5)
    )
    np.testing.assert_allclose(actual, expected, atol=1e-6)

    # t=0 uses the exact common source; t=1 reaches the Bellman endpoint.
    np.testing.assert_allclose(successor_sample[0, 0], source[0, 0], atol=0.0)
    np.testing.assert_allclose(current_sample[0, 0], source[0, 0], atol=0.0)
    np.testing.assert_allclose(
        successor_sample[0, 1], next_endpoint[0, 1], atol=0.0
    )
    np.testing.assert_allclose(
        current_sample[0, 1],
        reward_values[0, 0]
        + effective_discount[0, 0] * next_endpoint[0, 1],
        atol=1e-6,
    )
    # Terminal transitions are the ordinary source-to-reward path and their
    # velocity target is exactly r - epsilon, independent of lambda.
    terminal_current = (
        forward_time[1] * reward_values[1]
        + (1.0 - forward_time[1]) * source[1]
    )
    np.testing.assert_allclose(current_sample[1], terminal_current, atol=1e-6)
    np.testing.assert_allclose(
        target_velocity[1], reward_values[1] - source[1], atol=1e-6
    )

    assert len(calls["source"]) == len(calls["integrate"]) == 1
    integrate_key, integrated_source, integrated_tau = calls["integrate"][0]
    np.testing.assert_array_equal(calls["source"][0], integrate_key)
    np.testing.assert_allclose(integrated_source, source, atol=0.0)
    assert integrated_tau == pytest.approx(0.0)
    assert len(calls["apply"]) == 2
    _, observed_successor, successor_tau = calls["apply"][0]
    _, observed_current, current_tau = calls["apply"][1]
    np.testing.assert_allclose(observed_successor, successor_sample, atol=1e-6)
    np.testing.assert_allclose(observed_current, current_sample, atol=1e-6)
    np.testing.assert_allclose(successor_tau, repo_tau, atol=0.0)
    np.testing.assert_allclose(current_tau, repo_tau, atol=0.0)

    # Both the integrated target endpoint and target-field correction are
    # stop-gradient; only the online velocity field receives this loss.
    record_calls = False

    def loss_with_parameters(current_velocity_bias, target_velocity_bias, endpoint):
        current = {"velocity": current_velocity_bias, "endpoint": 0.0}
        target = {"velocity": target_velocity_bias, "endpoint": endpoint}
        return agent._pcbf_loss(
            current,
            target,
            features,
            next_features,
            actions,
            actions,
            rewards,
            discounts,
            bootstrap,
            source_key,
            jax.random.PRNGKey(22),
        ).sum()

    current_grad, target_velocity_grad, endpoint_grad = jax.grad(
        loss_with_parameters, argnums=(0, 1, 2)
    )(
        current_params["velocity"],
        target_params["velocity"],
        target_params["endpoint"],
    )
    assert abs(float(current_grad)) > 1e-6
    assert target_velocity_grad == pytest.approx(0.0, abs=1e-8)
    assert endpoint_grad == pytest.approx(0.0, abs=1e-8)


def test_return_sample_bcfm_and_dcfm_agent_act_and_update_with_finite_losses():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(
            "method.value_mode=return_sample",
            "method.atom_ce_lambda=0",
            "method.demo_fosd=false",
            "method.bc_lambda=0",
            "method.num_flow_samples=2",
            "method.num_target_flow_samples=2",
            "method.num_action_flow_samples=2",
            "method.bcfm_lambda=1",
            "method.dcfm_lambda=1",
            "method.scalar_value_embedding=hl_gauss",
            "method.scalar_embed_bins=9",
            "method.scalar_embed_sigma=1.5",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    agent.logging = True
    observations = {"low_dim_state": np.zeros((1, 1, 5), dtype=np.float32)}

    action = agent.act(observations, step=3000, eval_mode=False)

    assert action.shape == (1, 2, 2)
    assert np.all(np.isfinite(action))
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)

    # The zero-initialized online and target velocity heads make the very first
    # DCFM residual exactly zero. One BCFM update separates them, so the second
    # update exercises a genuinely non-zero DCFM loss.
    agent.update(iter([_batch()]), step=1)
    metrics = agent.update(iter([_batch()]), step=2)

    for name in (
        "critic_loss",
        "flow_loss",
        "bcfm_loss",
        "dcfm_loss",
        "endpoint_q_loss",
        "source_q_std",
        "target_return_std",
        "demo_source_bin_flip_rate",
    ):
        assert np.isfinite(metrics[name]), name
    assert metrics["critic_loss"] > 0.0
    assert metrics["bcfm_loss"] > 0.0
    assert metrics["dcfm_loss"] > 0.0

    terminal_batch = _batch()
    terminal_batch["terminal"] = np.ones_like(terminal_batch["terminal"])
    terminal_metrics = agent.update(iter([terminal_batch]), step=3)
    assert terminal_metrics["dcfm_loss"] == pytest.approx(0.0, abs=1e-8)


def test_value_flow_confidence_weight_uses_finite_stopped_source_jvp():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(
            "method.value_mode=return_sample",
            "method.atom_ce_lambda=0",
            "method.demo_fosd=false",
            "method.bc_lambda=0",
            "method.num_flow_samples=2",
            "method.num_target_flow_samples=2",
            "method.num_action_flow_samples=2",
            "method.bcfm_lambda=1",
            "method.dcfm_lambda=1",
            "method.confidence_weight_temp=0.3",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    agent.logging = True

    metrics = agent.update(iter([_batch()]), step=1)

    for name in (
        "confidence_weight_mean",
        "confidence_weight_min",
        "confidence_weight_max",
        "confidence_weight_std",
        "confidence_return_std_mean",
        "confidence_return_std_min",
        "confidence_return_std_max",
    ):
        assert np.isfinite(metrics[name]), name
    assert 0.5 < metrics["confidence_weight_min"] <= 1.0
    assert 0.5 < metrics["confidence_weight_max"] <= 1.0
    assert metrics["confidence_return_std_min"] > 0.0
    assert agent.confidence_weight_temp == pytest.approx(0.3)


def test_return_sample_pcbf_agent_updates_and_logs_independent_loss():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(
            "method.value_mode=return_sample",
            "method.atom_ce_lambda=0",
            "method.demo_fosd=false",
            "method.bc_lambda=0",
            "method.num_flow_samples=2",
            "method.num_target_flow_samples=2",
            "method.num_action_flow_samples=2",
            "method.bcfm_lambda=0",
            "method.dcfm_lambda=0",
            "method.pcbf_loss_coeff=1",
            "method.pcbf_lambda=0.4",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    agent.logging = True

    metrics = agent.update(iter([_batch()]), step=1)

    assert all(np.isscalar(value) for value in metrics.values())
    assert np.isfinite(metrics["critic_loss"])
    assert np.isfinite(metrics["pcbf_loss"])
    assert metrics["critic_loss"] > 0.0
    assert metrics["pcbf_loss"] > 0.0
    # The loss is independent: a zero BCFM coefficient skips that extra field
    # evaluation and reports an exact zero.
    assert agent.bcfm_lambda == 0.0
    assert agent.pcbf_loss_coeff == 1.0
    assert metrics["bcfm_loss"] == pytest.approx(0.0, abs=1e-8)

    terminal_batch = _batch()
    terminal_batch["terminal"] = np.ones_like(terminal_batch["terminal"])
    terminal_metrics = agent.update(iter([terminal_batch]), step=2)
    assert terminal_metrics["pcbf_loss"] > 0.0


def test_pcbf_clipped_optimizer_remains_finite_across_repeated_demo_updates():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(
            "method.value_mode=return_sample",
            "method.atom_ce_lambda=0",
            "method.demo_fosd=false",
            "method.num_flow_samples=2",
            "method.num_target_flow_samples=2",
            "method.num_action_flow_samples=2",
            "method.bcfm_lambda=0",
            "method.dcfm_lambda=0",
            "method.pcbf_loss_coeff=1",
            "method.pcbf_lambda=0.99",
            "method.critic_grad_clip=1",
            "method.num_flow_steps=2",
            "method.demo_flow_steps=1",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    agent.logging = True
    assert agent.demo_flow_steps == 1

    for step in range(1, 6):
        metrics = agent.update(iter([_batch()]), step=step)
        assert np.isfinite(metrics["critic_loss"])
        assert np.isfinite(metrics["pcbf_loss"])
        assert np.isfinite(metrics["critic_grad_norm"])
        assert np.isfinite(metrics["flow_critic_grad_norm"])
        assert np.isfinite(metrics["encoder_grad_norm"])
        assert np.isfinite(metrics["velocity_head_grad_norm"])
        assert metrics["flow_critic_grad_nonfinite_fraction"] == 0.0
        assert metrics["encoder_grad_nonfinite_fraction"] == 0.0
        assert np.isfinite(metrics["critic_update_norm"])
        assert _tree_all_finite(agent.params)
        assert _tree_all_finite(agent.target_critic_params)


@pytest.mark.parametrize(
    ("mode", "atom_ce_lambda", "demo_fosd", "expected_value_dim"),
    [
        pytest.param("categorical", 1.0, True, 5, id="v2_logit_flow_ce"),
        pytest.param("categorical", 0.0, True, 5, id="v3_logit_flow_only"),
        pytest.param("scalar", 0.0, False, 1, id="v4_scalar_expected_q"),
    ],
)
def test_cqn_flow_variants_act_and_update(
    mode,
    atom_ce_lambda,
    demo_fosd,
    expected_value_dim,
):
    observation_space, action_space = _spaces()
    agent = create_agent(
        _compose_cqn_flow(
            f"method.value_mode={mode}",
            f"method.atom_ce_lambda={atom_ce_lambda}",
            f"method.demo_fosd={str(demo_fosd).lower()}",
        ),
        observation_space=observation_space,
        action_space=action_space,
    )
    agent.logging = True
    observations = {"low_dim_state": np.zeros((1, 1, 5), dtype=np.float32)}

    action = agent.act(observations, step=3000, eval_mode=False)

    assert agent.value_dim == expected_value_dim
    assert action.shape == (1, 2, 2)
    assert np.all(np.isfinite(action))
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)

    params_before = jax.tree.map(np.asarray, agent.params)
    target_before = jax.tree.map(np.asarray, agent.target_critic_params)
    metrics = agent.update(iter([_batch()]), step=1)

    assert _tree_changed(params_before, agent.params)
    assert _tree_changed(target_before, agent.target_critic_params)
    assert np.isfinite(metrics["critic_loss"])
    assert metrics["demo_margin_loss"] > 0.0
    if mode == "categorical" and atom_ce_lambda > 0.0:
        assert metrics["atom_ce_loss"] > 0.0
    else:
        assert metrics["atom_ce_loss"] == pytest.approx(0.0, abs=1e-8)
    if mode == "categorical":
        assert metrics["endpoint_ce"] > 0.0
    else:
        assert metrics["endpoint_ce"] == pytest.approx(0.0, abs=1e-8)
    if mode == "scalar":
        assert metrics["demo_fosd_loss"] == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            ("method.value_mode=scalar", "method.atom_ce_lambda=0.1"),
            "atom_ce_lambda=0",
        ),
        (
            (
                "method.value_mode=scalar",
                "method.atom_ce_lambda=0",
                "method.demo_fosd=true",
            ),
            "demo_fosd=false",
        ),
    ],
)
def test_scalar_cqn_flow_rejects_distribution_only_losses(overrides, message):
    observation_space, action_space = _spaces()

    with pytest.raises(ValueError, match=message):
        create_agent(
            _compose_cqn_flow(*overrides),
            observation_space=observation_space,
            action_space=action_space,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param(("method.dcfm_lambda=1",), id="categorical"),
        pytest.param(
            (
                "method.value_mode=scalar",
                "method.atom_ce_lambda=0",
                "method.demo_fosd=false",
                "method.dcfm_lambda=1",
            ),
            id="expected_value_scalar",
        ),
    ],
)
def test_dcfm_rejected_outside_return_sample_mode(overrides):
    observation_space, action_space = _spaces()

    with pytest.raises(
        ValueError,
        match="DCFM is only defined for value_mode=return_sample",
    ):
        create_agent(
            _compose_cqn_flow(*overrides),
            observation_space=observation_space,
            action_space=action_space,
        )


@pytest.mark.parametrize(
    "override",
    ["method.pcbf_loss_coeff=1", "method.pcbf_lambda=0.4"],
)
def test_pcbf_rejected_outside_return_sample_mode(override):
    observation_space, action_space = _spaces()

    with pytest.raises(
        ValueError,
        match="PCBF is only defined for value_mode=return_sample",
    ):
        create_agent(
            _compose_cqn_flow(override),
            observation_space=observation_space,
            action_space=action_space,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            ("method.confidence_weight_temp=0",),
            "confidence_weight_temp must be positive",
        ),
        (
            ("method.confidence_weight_temp=0.3",),
            "requires value_mode=return_sample",
        ),
        (
            (
                "method.value_mode=return_sample",
                "method.atom_ce_lambda=0",
                "method.demo_fosd=false",
                "method.flow_source_type=uniform",
                "method.flow_source_min=0",
                "method.flow_source_max=1",
                "method.confidence_weight_temp=0.3",
            ),
            "requires a Gaussian source",
        ),
    ],
)
def test_value_flow_confidence_weight_rejects_invalid_modes(overrides, message):
    observation_space, action_space = _spaces()

    with pytest.raises(ValueError, match=message):
        create_agent(
            _compose_cqn_flow(*overrides),
            observation_space=observation_space,
            action_space=action_space,
        )


def test_cqn_flow_rejects_too_small_time_embedding():
    observation_space, action_space = _spaces()

    with pytest.raises(ValueError, match="time_embed_dim must be >= 2"):
        create_agent(
            _compose_cqn_flow("method.time_embed_dim=1"),
            observation_space=observation_space,
            action_space=action_space,
        )
