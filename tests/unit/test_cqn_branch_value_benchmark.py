import json

import numpy as np
import pytest

from scripts.benchmark_cqn_branch_value_models import (
    _build_conditioners,
    _categorical_targets,
    _make_flow_sources,
    _policy_prior_rank_targets,
    _ranking_metrics,
    _training_phase,
    _within_state_best_targets,
    _within_state_return_rank_targets,
    _within_state_softmax_targets,
)


def test_categorical_targets_interpolate_on_uniform_c51_support():
    probabilities = _categorical_targets(
        np.asarray([0.0, 0.125, 1.0], np.float32),
        atoms=5,
        v_min=0.0,
        v_max=1.0,
    )

    np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0)
    np.testing.assert_allclose(probabilities[0], [1, 0, 0, 0, 0])
    np.testing.assert_allclose(probabilities[1], [0.5, 0.5, 0, 0, 0])
    np.testing.assert_allclose(probabilities[2], [0, 0, 0, 0, 1])


def test_ranking_metrics_ignore_tied_pairs_and_accept_tied_optima():
    targets = np.asarray(
        [[0.0, 0.5, 0.5], [0.2, 0.2, 0.2]],
        np.float32,
    )
    predictions = np.asarray(
        [[0.0, 0.4, 0.6], [3.0, -1.0, 2.0]],
        np.float32,
    )

    metrics = _ranking_metrics(
        predictions,
        targets,
        return_atol=1e-12,
    )

    assert metrics["num_informative_states"] == 1
    assert metrics["pairwise_accuracy"] == 1.0
    assert metrics["top1_accuracy"] == 1.0
    assert metrics["regret"] == 0.0


def test_conditioner_reserves_validation_seed_and_includes_bin_condition(
    tmp_path,
):
    candidates = 3
    action_sequence = 2
    action_dims = 2
    train_records = [
        {"eval_seed": 10},
        {"eval_seed": 10},
        {"eval_seed": 11},
    ]
    heldout_records = [{"eval_seed": 20}, {"eval_seed": 21}]
    cache_path = tmp_path / "cache.npz"
    rng = np.random.default_rng(3)
    np.savez_compressed(
        cache_path,
        train_features=rng.normal(size=(3, 6)).astype(np.float32),
        train_actions=rng.normal(
            size=(3, candidates, action_sequence, action_dims)
        ).astype(np.float32),
        train_returns=rng.random(size=(3, candidates)).astype(np.float32),
        train_action_dimensions=np.asarray([0, 1, 0], np.int32),
        train_metadata=np.asarray(json.dumps(train_records)),
        heldout_features=rng.normal(size=(2, 6)).astype(np.float32),
        heldout_actions=rng.normal(
            size=(2, candidates, action_sequence, action_dims)
        ).astype(np.float32),
        heldout_returns=rng.random(size=(2, candidates)).astype(np.float32),
        heldout_action_dimensions=np.asarray([0, 1], np.int32),
        heldout_metadata=np.asarray(json.dumps(heldout_records)),
    )

    data = _build_conditioners(
        cache_path,
        pca_components=2,
        validation_seed=11,
    )

    assert data.validation_seed == 11
    assert data.fit_x.shape[0] == 2 * candidates
    assert data.validation_x.shape[0] == candidates
    assert data.heldout_x.shape[0] == 2 * candidates
    assert data.condition_dim == (
        data.state_components + action_sequence * action_dims + 2 + 3
    )
    assert data.state_mean.shape == (1, 6)
    assert data.state_basis.shape == (data.state_components, 6)
    assert data.state_scale.shape == (1, data.state_components)
    assert data.action_mean.shape == (1, action_sequence * action_dims)
    assert data.action_scale.shape == (1, action_sequence * action_dims)
    assert data.action_dim_count == action_dims
    # The last three coordinates are the explicit sibling-bin condition.
    np.testing.assert_allclose(
        data.validation_x[:, -candidates:],
        np.eye(candidates),
    )


def test_uniform_flow_sources_are_bounded_and_antithetic():
    sources = _make_flow_sources(
        seed=3,
        samples=8,
        count=5,
        source_type="uniform",
        source_min=0.0,
        source_max=0.1,
    )

    assert sources.shape == (8, 5, 1)
    assert np.all(sources >= 0.0)
    assert np.all(sources <= 0.1)
    np.testing.assert_allclose(
        sources[:4] + sources[4:],
        0.1,
        atol=1e-7,
    )


def test_policy_prior_rank_targets_preserve_bc_order_and_ties():
    targets = _policy_prior_rank_targets(
        [
            {"policy_log_probability": [-3.0, -1.0, -2.0]},
            {"policy_log_probability": [0.0, 0.0, -1.0]},
        ],
        candidates=3,
    )

    np.testing.assert_allclose(targets[0], [0.0, 1.0, 0.5])
    np.testing.assert_allclose(targets[1], [0.75, 0.75, 0.0])


def test_return_rank_targets_preserve_order_and_average_ties():
    targets = _within_state_return_rank_targets(
        np.asarray(
            [
                [0.4, 0.1, 0.7],
                [0.5, 0.5, 0.0],
            ],
            np.float32,
        )
    )

    np.testing.assert_allclose(targets[0], [0.5, 0.0, 1.0])
    np.testing.assert_allclose(targets[1], [0.75, 0.75, 0.0])


def test_best_targets_mark_every_tied_counterfactual_optimum():
    targets = _within_state_best_targets(
        np.asarray(
            [
                [0.4, 0.1, 0.7],
                [0.5, 0.5, 0.0],
            ],
            np.float32,
        ),
        return_atol=1e-6,
    )

    np.testing.assert_array_equal(targets[0], [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(targets[1], [1.0, 1.0, 0.0])


def test_softmax_targets_preserve_return_order_gaps_and_ties():
    targets = _within_state_softmax_targets(
        np.asarray(
            [
                [0.4, 0.1, 0.7],
                [0.5, 0.5, 0.0],
            ],
            np.float32,
        ),
        temperature=0.1,
    )

    np.testing.assert_allclose(targets.sum(axis=1), 1.0, atol=1e-6)
    assert targets[0, 2] > targets[0, 0] > targets[0, 1]
    assert targets[1, 0] == pytest.approx(targets[1, 1])
    assert targets[1, 0] > targets[1, 2]


def test_target_shift_phase_switches_after_exact_warmup_budget():
    assert (
        _training_phase(
            4,
            warmup_target="policy_prior",
            warmup_updates=5,
        )
        == "policy_prior"
    )
    assert (
        _training_phase(
            5,
            warmup_target="policy_prior",
            warmup_updates=5,
        )
        == "return"
    )
    assert (
        _training_phase(
            0,
            warmup_target="none",
            warmup_updates=0,
        )
        == "return"
    )


def test_conditioner_rejects_all_tie_validation_seed(tmp_path):
    cache_path = tmp_path / "cache.npz"
    records = [
        {"eval_seed": 10},
        {"eval_seed": 10},
        {"eval_seed": 11},
        {"eval_seed": 11},
    ]
    np.savez_compressed(
        cache_path,
        train_features=np.arange(12, dtype=np.float32).reshape(4, 3),
        train_actions=np.zeros((4, 3, 1, 1), np.float32),
        train_returns=np.zeros((4, 3), np.float32),
        train_action_dimensions=np.zeros((4,), np.int32),
        train_metadata=np.asarray(json.dumps(records)),
        heldout_features=np.zeros((1, 3), np.float32),
        heldout_actions=np.zeros((1, 3, 1, 1), np.float32),
        heldout_returns=np.zeros((1, 3), np.float32),
        heldout_action_dimensions=np.zeros((1,), np.int32),
        heldout_metadata=np.asarray(json.dumps([{"eval_seed": 20}])),
    )

    with pytest.raises(ValueError, match="no informative"):
        _build_conditioners(
            cache_path,
            pca_components=1,
            validation_seed=11,
        )
