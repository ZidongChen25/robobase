import numpy as np

from scripts.run_cqn_structured_delta_gate import BranchData
from scripts.run_cqn_structured_delta_reliability_gate import (
    _reliable_group,
    derive_crossfit_support,
)


def _synthetic_reliable_data() -> BranchData:
    features = []
    returns = []
    dimensions = []
    metadata = []
    deltas = np.asarray([-1.0, 0.0, 1.0], np.float32)
    for seed in range(8):
        state = -0.9 + 1.8 * seed / 7.0
        for dimension in range(2):
            optimum = state if dimension == 0 else -state
            features.append([state, state * state, 1.0])
            returns.append(-np.abs(deltas - optimum))
            dimensions.append(dimension)
            metadata.append(
                {
                    "eval_seed": seed,
                    "anchor_step": 120,
                    "action_dimension": dimension,
                    "actual_first_delta": deltas.tolist(),
                    "policy_log_probability": (
                        -np.abs(deltas)
                    ).tolist(),
                }
            )
    return BranchData(
        features=np.asarray(features, np.float32),
        returns=np.asarray(returns, np.float32),
        action_dimensions=np.asarray(dimensions, np.int32),
        metadata=tuple(metadata),
    )


def test_reliable_group_requires_both_anti_imitation_proxies():
    metrics = {
        "model": {
            "num_informative_states": 30,
            "pairwise_accuracy": 0.65,
            "regret": 0.08,
            "mean_spearman": 0.2,
        },
        "behavior": {"pairwise_accuracy": 0.60, "regret": 0.10},
        "policy": {"pairwise_accuracy": 0.66, "regret": 0.11},
    }

    supported, checks = _reliable_group(
        metrics,
        min_informative=20,
    )

    assert not supported
    assert checks["pairwise_above_behavior"]
    assert not checks["pairwise_above_policy"]


def test_crossfit_support_is_derived_without_same_seed_fit_rows():
    support = derive_crossfit_support(
        _synthetic_reliable_data(),
        pca_components=2,
        ridge_alpha=1e-4,
        return_atol=1e-12,
        min_informative=4,
    )

    assert support["crossfit_seed_ids"] == list(range(8))
    assert support["supported_anchor_steps"] == [120]
    assert support["supported_action_dimensions"] == [0, 1]
    assert support["supported_intersection_num_states"] == 16
