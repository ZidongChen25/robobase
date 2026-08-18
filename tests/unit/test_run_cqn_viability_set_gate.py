import json

import numpy as np

from scripts.run_cqn_viability_set_gate import (
    BranchData,
    _load_branch_data,
    _success_labels,
    fit_viability_ensemble,
    predict_viability,
    select_complete_candidates,
)


def _synthetic_data() -> BranchData:
    features = []
    actions = []
    returns = []
    dimensions = []
    metadata = []
    candidate_values = np.asarray([-1.0, 0.0, 1.0], np.float32)
    for seed in range(20):
        state_sign = -1.0 if seed % 2 == 0 else 1.0
        for dimension in range(2):
            desired_sign = state_sign * (1.0 if dimension == 0 else -1.0)
            candidate_actions = np.zeros((3, 2, 2), np.float32)
            candidate_actions[:, 0, dimension] = candidate_values
            successes = (candidate_values * desired_sign) >= 0.0
            candidate_returns = successes.astype(np.float32)
            features.append([state_sign, float(dimension), state_sign * dimension])
            actions.append(candidate_actions)
            returns.append(candidate_returns)
            dimensions.append(dimension)
            metadata.append(
                {
                    "eval_seed": seed,
                    "anchor_step": 5,
                    "action_dimension": dimension,
                    "actual_first_delta": candidate_values.tolist(),
                    "policy_log_probability": (-np.abs(candidate_values)).tolist(),
                    "outcomes": [
                        {"success": bool(success), "discounted_return": float(success)}
                        for success in successes
                    ],
                }
            )
    return BranchData(
        features=np.asarray(features, np.float32),
        actions=np.asarray(actions, np.float32),
        returns=np.asarray(returns, np.float32),
        action_dimensions=np.asarray(dimensions, np.int32),
        metadata=tuple(metadata),
    )


def test_success_labels_keep_every_successful_candidate_positive():
    labels = _success_labels(_synthetic_data())

    assert labels.shape == (40, 3)
    np.testing.assert_array_equal(np.sum(labels, axis=1), 2.0)


def test_joint_viability_model_learns_set_valued_success():
    data = _synthetic_data()
    model = fit_viability_ensemble(
        data,
        state_components=2,
        action_components=2,
        ridge_strength=0.01,
        ensemble_size=8,
        bootstrap_seed=7,
        optimizer_maxiter=300,
    )

    mean, uncertainty = predict_viability(model, data)
    choices = select_complete_candidates(mean - 0.5 * uncertainty)
    labels = _success_labels(data)

    assert np.mean(labels[np.arange(labels.shape[0]), choices]) > 0.95
    assert mean.shape == labels.shape
    assert uncertainty.shape == labels.shape


def test_selector_returns_one_complete_candidate_without_recombination():
    data = _synthetic_data()
    scores = np.tile(np.asarray([0.1, 0.9, 0.2]), (len(data.metadata), 1))

    choices = select_complete_candidates(scores)
    selected_chunks = data.actions[np.arange(len(choices)), choices]

    np.testing.assert_array_equal(choices, 1)
    np.testing.assert_allclose(selected_chunks, data.actions[:, 1])


def test_load_branch_data_requires_and_preserves_complete_actions(tmp_path):
    data = _synthetic_data()
    cache = tmp_path / "branches.npz"
    midpoint = len(data.metadata) // 2
    np.savez_compressed(
        cache,
        train_features=data.features[:midpoint],
        train_actions=data.actions[:midpoint],
        train_returns=data.returns[:midpoint],
        train_action_dimensions=data.action_dimensions[:midpoint],
        train_metadata=np.asarray(json.dumps(data.metadata[:midpoint])),
        heldout_features=data.features[midpoint:],
        heldout_actions=data.actions[midpoint:],
        heldout_returns=data.returns[midpoint:],
        heldout_action_dimensions=data.action_dimensions[midpoint:],
        heldout_metadata=np.asarray(json.dumps(data.metadata[midpoint:])),
    )

    loaded = _load_branch_data(cache, ("train", "heldout"))

    np.testing.assert_allclose(loaded.actions, data.actions)
    np.testing.assert_allclose(loaded.features, data.features)
    assert loaded.metadata == data.metadata
