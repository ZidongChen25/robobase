import json

import numpy as np

from scripts.run_cqn_structured_delta_gate import (
    BranchData,
    _candidate_scores,
    _load_branch_data,
    _seed_bootstrap,
    fit_structured_delta_model,
    predict_optimal_delta,
)


def _synthetic_data() -> BranchData:
    features = []
    returns = []
    dimensions = []
    metadata = []
    deltas = np.asarray([-1.0, 0.0, 1.0], np.float32)
    for seed in range(8):
        state = -1.0 + 2.0 * seed / 7.0
        for dimension in range(2):
            optimum = state if dimension == 0 else -state
            candidate_returns = -np.abs(deltas - optimum)
            features.append([state, state * state, 1.0])
            returns.append(candidate_returns)
            dimensions.append(dimension)
            metadata.append(
                {
                    "eval_seed": seed,
                    "anchor_step": 30,
                    "action_dimension": dimension,
                    "actual_first_delta": deltas.tolist(),
                    "policy_log_probability": (-np.abs(deltas)).tolist(),
                }
            )
    return BranchData(
        features=np.asarray(features, np.float32),
        returns=np.asarray(returns, np.float32),
        action_dimensions=np.asarray(dimensions, np.int32),
        metadata=tuple(metadata),
    )


def test_structured_delta_model_learns_dimension_conditioned_optimum():
    data = _synthetic_data()
    model = fit_structured_delta_model(
        data,
        pca_components=2,
        ridge_alpha=1e-4,
        return_atol=1e-12,
    )

    prediction = predict_optimal_delta(model, data)

    expected = np.asarray(
        [
            (-1.0 + 2.0 * seed / 7.0) * (1.0 if dimension == 0 else -1.0)
            for seed in range(8)
            for dimension in range(2)
        ]
    )
    np.testing.assert_allclose(prediction, expected, atol=2e-3)
    scores = _candidate_scores(prediction, data.metadata)
    assert scores.shape == data.returns.shape


def test_seed_bootstrap_reports_positive_model_effect():
    data = _synthetic_data()
    model = fit_structured_delta_model(
        data,
        pca_components=2,
        ridge_alpha=1e-4,
        return_atol=1e-12,
    )
    model_scores = _candidate_scores(
        predict_optimal_delta(model, data),
        data.metadata,
    )
    behavior = np.asarray(
        [-np.abs(record["actual_first_delta"]) for record in data.metadata]
    )

    result = _seed_bootstrap(
        data,
        {
            "model": model_scores,
            "behavior": behavior,
            "policy": behavior,
        },
        return_atol=1e-12,
        replicates=200,
        seed=5,
    )

    assert result["num_seeds"] == 8
    assert (
        result["paired_deltas"]["model_minus_behavior"][
            "pairwise_accuracy"
        ][0]
        > 0.0
    )


def test_load_branch_data_combines_requested_splits(tmp_path):
    data = _synthetic_data()
    cache = tmp_path / "branches.npz"
    midpoint = len(data.metadata) // 2
    np.savez_compressed(
        cache,
        train_features=data.features[:midpoint],
        train_returns=data.returns[:midpoint],
        train_action_dimensions=data.action_dimensions[:midpoint],
        train_metadata=np.asarray(json.dumps(data.metadata[:midpoint])),
        heldout_features=data.features[midpoint:],
        heldout_returns=data.returns[midpoint:],
        heldout_action_dimensions=data.action_dimensions[midpoint:],
        heldout_metadata=np.asarray(json.dumps(data.metadata[midpoint:])),
    )

    loaded = _load_branch_data(cache, ("train", "heldout"))

    np.testing.assert_allclose(loaded.features, data.features)
    np.testing.assert_allclose(loaded.returns, data.returns)
    assert loaded.metadata == data.metadata
