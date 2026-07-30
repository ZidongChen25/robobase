import numpy as np
import pytest

from scripts.run_cqn_prediction_ensemble_gate import ensemble_predictions


def _summary():
    return {
        "status": "complete",
        "results": [
            {
                "method": "direct",
                "seed": 1,
                "selected_heldout_predictions": [[1.0, 3.0]],
            },
            {
                "method": "direct",
                "seed": 2,
                "selected_heldout_predictions": [[3.0, 1.0]],
            },
            {
                "method": "direct",
                "seed": 3,
                "selected_heldout_predictions": [[2.0, 2.0]],
            },
        ],
    }


def test_prediction_ensemble_uses_frozen_arithmetic_mean():
    predictions = ensemble_predictions(
        _summary(),
        method="direct",
        model_seeds=[1, 2, 3],
    )

    np.testing.assert_allclose(predictions, [[2.0, 2.0]])


def test_prediction_ensemble_rejects_seed_subselection():
    with pytest.raises(ValueError, match="extra"):
        ensemble_predictions(
            _summary(),
            method="direct",
            model_seeds=[1, 2],
        )
