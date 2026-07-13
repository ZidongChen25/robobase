import csv

import numpy as np

from scripts.analyze_linear_predictive_observability import (
    action_observability_matrix,
    action_observability_metrics,
    main,
    predictive_covariances,
)


def test_action_observability_matrix_matches_closed_loop_rollout_actions():
    A_closed = np.asarray([[0.8, 0.2], [-0.1, 0.6]])
    K = np.asarray([[1.2, -0.4]])
    x0 = np.asarray([0.7, -1.1])
    horizon = 5

    observability = action_observability_matrix(A_closed, K, horizon)
    predicted_actions = observability @ x0

    rollout_actions = []
    state = x0.copy()
    for _ in range(horizon):
        rollout_actions.append((K @ state).item())
        state = A_closed @ state

    np.testing.assert_allclose(predicted_actions, rollout_actions, atol=1e-12)


def test_action_observability_rank_grows_with_horizon():
    A_closed = np.diag([0.9, 0.4])
    K = np.asarray([[1.0, 1.0]])

    metrics_h1 = action_observability_metrics(
        action_observability_matrix(A_closed, K, horizon=1)
    )
    metrics_h2 = action_observability_metrics(
        action_observability_matrix(A_closed, K, horizon=2)
    )
    metrics_h4 = action_observability_metrics(
        action_observability_matrix(A_closed, K, horizon=4)
    )

    assert [metrics_h1["rank"], metrics_h2["rank"], metrics_h4["rank"]] == [
        1,
        2,
        2,
    ]
    assert metrics_h1["fully_observable"] is False
    assert np.isinf(metrics_h1["condition"])
    assert metrics_h2["fully_observable"] is True
    assert np.isfinite(metrics_h2["condition"])


def test_predictive_covariance_matches_manual_recursion():
    A_closed = np.asarray([[0.8, 0.1], [0.0, 0.6]])
    K = np.asarray([[1.0, -0.5]])
    P0 = np.asarray([[1.0, 0.2], [0.2, 0.5]])
    Q = np.diag([0.03, 0.02])
    R = np.asarray([[0.04]])

    result = predictive_covariances(A_closed, K, P0, Q, R, horizon=2)
    expected_action0 = K @ P0 @ K.T + R
    expected_state1 = A_closed @ P0 @ A_closed.T + Q
    expected_action1 = K @ expected_state1 @ K.T + R
    expected_state2 = A_closed @ expected_state1 @ A_closed.T + Q

    assert result.state.shape == (3, 2, 2)
    assert result.action.shape == (2, 1, 1)
    np.testing.assert_allclose(result.state[0], P0, atol=1e-12)
    np.testing.assert_allclose(result.action[0], expected_action0, atol=1e-12)
    np.testing.assert_allclose(result.state[1], expected_state1, atol=1e-12)
    np.testing.assert_allclose(result.action[1], expected_action1, atol=1e-12)
    np.testing.assert_allclose(result.state[2], expected_state2, atol=1e-12)


def test_cli_writes_cartesian_sweep_csv(tmp_path):
    output_csv = tmp_path / "sweep.csv"

    exit_code = main(
        [
            "--output-csv",
            str(output_csv),
            "--horizons",
            "1",
            "2",
            "--spectral-radii",
            "0.5",
            "1.1",
            "--action-observability-scales",
            "0.0",
            "1.0",
            "--noise-scales",
            "0.0",
            "2.0",
        ]
    )

    assert exit_code == 0
    with output_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * 2 * 2 * 2
    assert {float(row["target_spectral_radius"]) for row in rows} == {0.5, 1.1}
    assert {float(row["action_observability_scale"]) for row in rows} == {
        0.0,
        1.0,
    }
    assert {float(row["noise_scale"]) for row in rows} == {0.0, 2.0}
    for row in rows:
        np.testing.assert_allclose(
            float(row["closed_loop_spectral_radius"]),
            float(row["target_spectral_radius"]),
            atol=1e-12,
        )
