import numpy as np

from scripts.run_cqn_prediction_blend_discovery import (
    blend_scores,
    row_standardize,
    select_validation_blend,
)


def test_row_standardize_is_per_state_and_handles_constant_scores():
    scores = row_standardize(
        np.asarray([[1.0, 2.0, 3.0], [4.0, 4.0, 4.0]])
    )

    np.testing.assert_allclose(scores.mean(axis=1), 0.0, atol=1e-8)
    np.testing.assert_allclose(scores[1], 0.0)


def test_blend_retains_positive_model_weight():
    model = np.asarray([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
    behavior = np.asarray([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
    policy = behavior.copy()
    blended = blend_scores(
        model,
        behavior,
        policy,
        proxy="behavior",
        model_weight=0.5,
    )

    assert blended[0, 2] > blended[0, 0]
    assert abs(blended[1, 0] - blended[1, 2]) < 3e-6


def test_validation_selection_uses_targets_without_external_data():
    targets = np.asarray([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
    model = targets.copy()
    behavior = np.asarray([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    policy = behavior.copy()

    selected, rows = select_validation_blend(
        model,
        behavior,
        policy,
        targets,
        proxies=["behavior"],
        weights=[0.25, 1.0],
        return_atol=1e-12,
    )

    assert len(rows) == 2
    assert selected["model_weight"] == 1.0
    assert selected["metrics"]["pairwise_accuracy"] == 1.0
