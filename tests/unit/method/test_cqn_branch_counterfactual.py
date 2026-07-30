import numpy as np

from scripts.analyze_cqn_branch_counterfactual import (
    _action_nearness_scores,
    _coherent_sibling_plans,
    _effective_policy_plan,
    _pairwise_sign_stats,
    _ranking_metrics,
    _policy_readout_label,
    _policy_value_beta,
    _resolve_value_readout,
    _select_action_dimension,
    _selected_path_log_probability,
)


class _TemporalAgent:
    temporal_ensemble = True

    def __init__(self, *, register=True):
        self.register = register
        self.key_uses = 0
        self.registered = None

    def _temporal_replan_mask(self, *, eval_mode, batch_size):
        assert eval_mode
        assert batch_size == 1
        return np.asarray([self.register], dtype=np.bool_)

    def _next_action_key(self):
        self.key_uses += 1

    def _ensemble_current_action(
        self, action_chunk, *, eval_mode, register_mask
    ):
        assert eval_mode
        self.registered = action_chunk.copy()
        if register_mask[0]:
            return action_chunk[:, 0] * 0.25
        return np.full_like(action_chunk[:, 0], 7.0)


def test_effective_policy_plan_registers_candidate_and_updates_first_action():
    agent = _TemporalAgent(register=True)
    candidate = np.arange(6, dtype=np.float32).reshape(3, 2)

    effective, registered = _effective_policy_plan(agent, candidate)

    assert registered
    assert agent.key_uses == 1
    np.testing.assert_array_equal(agent.registered[0], candidate)
    np.testing.assert_array_equal(effective[0], candidate[0] * 0.25)
    np.testing.assert_array_equal(effective[1:], candidate[1:])


def test_effective_policy_plan_matches_non_replanning_history_action():
    agent = _TemporalAgent(register=False)
    candidate = np.arange(6, dtype=np.float32).reshape(3, 2)

    effective, registered = _effective_policy_plan(agent, candidate)

    assert not registered
    assert agent.key_uses == 0
    np.testing.assert_array_equal(agent.registered, np.zeros((1, 3, 2)))
    np.testing.assert_array_equal(effective[0], [7.0, 7.0])


def test_pairwise_sign_stats_ignores_tied_realized_returns():
    predicted = np.asarray([3.0, 2.0, 1.0])
    realized = np.asarray([1.0, 1.0, 2.0])

    accuracy, count = _pairwise_sign_stats(predicted, realized)

    assert count == 2
    assert accuracy == 0.0


def test_ranking_metrics_support_anti_cheat_proxy_scores():
    metrics = _ranking_metrics(
        np.asarray([0.0, 2.0, 1.0]),
        np.asarray([0.0, 1.0, 0.0]),
    )

    assert metrics["pairwise_sign_accuracy"] == 1.0
    assert metrics["num_informative_pairs"] == 2
    assert metrics["top1_match"]
    assert metrics["realized_regret"] == 0.0


def test_action_nearness_uses_independent_policy_preferred_bin():
    scores = _action_nearness_scores(
        np.asarray([-3.0, 4.0, 1.0, 0.0, -2.0])
    )

    np.testing.assert_array_equal(scores, [-1.0, 0.0, -1.0, -2.0, -3.0])


def test_round_robin_dimension_selection_is_independent_of_q_span():
    q_values = np.asarray(
        [
            [0.0, 100.0, 0.0],
            [0.0, 0.1, 0.0],
            [-1.0, 1.0, 0.0],
        ]
    )

    assert (
        _select_action_dimension(
            q_values, selection="q_span", state_index=1
        )
        == 0
    )
    assert (
        _select_action_dimension(
            q_values, selection="round_robin", state_index=1
        )
        == 1
    )
    assert (
        _select_action_dimension(
            q_values, selection="round_robin", state_index=5
        )
        == 2
    )


def test_selected_path_log_probability_scores_executed_coordinate():
    scores = np.asarray(
        [
            [[[3.0, 0.0]], [[0.0, 2.0]]],
            [[[1.0, 1.0]], [[4.0, -1.0]]],
        ],
        dtype=np.float32,
    )
    selected = np.asarray(
        [
            [[0], [1]],
            [[1], [0]],
        ],
        dtype=np.int32,
    )

    actual = np.asarray(
        _selected_path_log_probability(
            scores,
            selected,
            sequence_index=0,
            action_dimension=0,
        )
    )
    expected = []
    for candidate_scores, candidate_bins in zip(scores, selected):
        logits = candidate_scores[0, 0]
        index = candidate_bins[0, 0]
        expected.append(
            float(logits[index] - np.log(np.exp(logits).sum()))
        )
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_coherent_sibling_plans_repeat_first_action_delta_with_clipping():
    agent = type(
        "Agent",
        (),
        {
            "action_sequence": 5,
            "action_dim": 2,
            "_step_action_low": np.asarray([-1.0, -1.0], np.float32),
            "_step_action_high": np.asarray([1.0, 1.0], np.float32),
        },
    )()
    baseline = np.zeros((5, 2), np.float32)
    baseline[:4, 1] = np.asarray([0.97, 0.0, -0.97, 0.5])
    siblings = np.repeat(baseline[None], 3, axis=0)
    siblings[:, 0, 1] = np.asarray([0.89, 0.97, 1.0])

    candidates, deltas = _coherent_sibling_plans(
        agent,
        baseline,
        siblings,
        action_dimension=1,
        intervention_horizon=4,
    )

    np.testing.assert_allclose(deltas, [-0.08, 0.0, 0.03], atol=1e-6)
    np.testing.assert_allclose(
        candidates[:, 0, 1], [0.89, 0.97, 1.0], atol=1e-6
    )
    np.testing.assert_allclose(
        candidates[:, 1, 1], [-0.08, 0.0, 0.03], atol=1e-6
    )
    np.testing.assert_allclose(
        candidates[:, 2, 1], [-1.0, -0.97, -0.94], atol=1e-6
    )
    np.testing.assert_allclose(
        candidates[:, 3, 1], [0.42, 0.5, 0.53], atol=1e-6
    )
    np.testing.assert_allclose(
        candidates[:, 4], np.repeat(baseline[None, 4], 3, axis=0)
    )


def test_resolve_value_readout_prefers_available_distill_for_auto():
    assert (
        _resolve_value_readout(
            "cqn_flow",
            "auto",
            has_flow_distill=True,
        )
        == "distill"
    )
    assert (
        _resolve_value_readout(
            "cqn_flow",
            "auto",
            has_flow_distill=False,
        )
        == "integrated"
    )
    assert (
        _resolve_value_readout(
            "cqn_as",
            "auto",
            has_flow_distill=False,
            direct_scalar_q=True,
        )
        == "direct_scalar_q"
    )


def test_resolve_value_readout_rejects_invalid_explicit_modes():
    import pytest

    with pytest.raises(ValueError, match="requires flow_distill_lambda"):
        _resolve_value_readout(
            "cqn_flow",
            "distill",
            has_flow_distill=False,
        )
    with pytest.raises(ValueError, match="applies only"):
        _resolve_value_readout(
            "cqn_as",
            "integrated",
            has_flow_distill=False,
        )


def test_policy_value_beta_parses_config_bc_and_numeric():
    import argparse
    import pytest

    assert _policy_value_beta("config") == "config"
    assert _policy_value_beta("bc") is None
    assert _policy_value_beta("1.5") == 1.5
    with pytest.raises(argparse.ArgumentTypeError):
        _policy_value_beta("-0.1")


def test_policy_readout_label_does_not_mislabel_native_cqn_as_as_bc():
    assert (
        _policy_readout_label(
            "categorical_c51",
            None,
            separate_bc_policy=False,
        )
        == "categorical_c51"
    )
    assert (
        _policy_readout_label(
            "distill",
            1.0,
            separate_bc_policy=True,
        )
        == "distill_plus_bc"
    )
