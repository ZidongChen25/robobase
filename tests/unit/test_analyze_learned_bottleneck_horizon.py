import csv

import numpy as np

from scripts.analyze_learned_bottleneck_horizon import (
    action_readout,
    main,
    observability_matrix,
    observability_metrics,
    predictive_noise_variances,
    rotational_dynamics,
    run_sweep,
)


def _summary_by_condition(rows):
    return {
        (
            row["observability_scale"],
            row["predictive_noise_std"],
            row["train_horizon"],
        ): row
        for row in rows
        if row["eval_offset"] == 0
    }


def test_observability_controls_do_not_change_predictive_noise_variance():
    state_dim = 6
    dynamics = rotational_dynamics(state_dim, spectral_radius=0.95)
    low_rank_readout = action_readout(
        state_dim,
        observability_rank=2,
        observability_scale=0.1,
    )
    weak_readout = action_readout(
        state_dim,
        observability_rank=6,
        observability_scale=0.1,
    )
    strong_readout = action_readout(
        state_dim,
        observability_rank=6,
        observability_scale=1.0,
    )

    np.testing.assert_allclose(
        dynamics @ dynamics.T,
        0.95**2 * np.eye(state_dim),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(np.linalg.norm(low_rank_readout), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(weak_readout), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(strong_readout), 1.0, atol=1e-12)

    low_rank_observability = observability_metrics(
        observability_matrix(dynamics, low_rank_readout, horizon=20)
    )
    weak_observability = observability_metrics(
        observability_matrix(dynamics, weak_readout, horizon=20)
    )
    strong_observability = observability_metrics(
        observability_matrix(dynamics, strong_readout, horizon=20)
    )
    assert low_rank_observability["rank"] == 2
    assert weak_observability["rank"] == 6
    assert strong_observability["rank"] == 6
    assert (
        strong_observability["sigma_min_nonzero"]
        > weak_observability["sigma_min_nonzero"]
    )

    low_rank_noise = predictive_noise_variances(
        dynamics,
        low_rank_readout,
        predictive_noise_std=0.7,
        horizon=20,
    )
    weak_noise = predictive_noise_variances(
        dynamics,
        weak_readout,
        predictive_noise_std=0.7,
        horizon=20,
    )
    strong_noise = predictive_noise_variances(
        dynamics,
        strong_readout,
        predictive_noise_std=0.7,
        horizon=20,
    )
    np.testing.assert_allclose(
        low_rank_noise,
        strong_noise,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(weak_noise, strong_noise, rtol=1e-12, atol=1e-12)

    lower_radius_noise = predictive_noise_variances(
        rotational_dynamics(state_dim, spectral_radius=0.7),
        strong_readout,
        predictive_noise_std=0.7,
        horizon=20,
    )
    assert strong_noise[-1] > lower_radius_noise[-1]


def test_factorial_sweep_exposes_opposite_observability_and_uncertainty_effects():
    rows = run_sweep(
        horizons=(1, 20),
        spectral_radii=(0.7,),
        observability_ranks=(6,),
        observability_scales=(0.1, 1.0),
        predictive_noise_stds=(0.25, 1.0),
        seeds=(0,),
        n_train=256,
        n_test=512,
        state_dim=6,
        nuisance_dim=18,
        bottleneck_width=3,
        observation_noise_std=0.1,
    )
    summary = _summary_by_condition(rows)

    weak_observability = summary[(0.1, 0.25, 20)]
    strong_observability = summary[(1.0, 0.25, 20)]
    strong_observability_high_noise = summary[(1.0, 1.0, 20)]
    strong_observability_k1_low_noise = summary[(1.0, 0.25, 1)]
    strong_observability_k1_high_noise = summary[(1.0, 1.0, 1)]

    # Stronger deterministic observability improves the held-out state probe.
    assert (
        strong_observability["state_probe_r2"]
        > weak_observability["state_probe_r2"] + 0.05
    )
    # More future uncertainty has the opposite effect at fixed observability.
    assert (
        strong_observability_high_noise["state_probe_r2"]
        < strong_observability["state_probe_r2"] - 0.05
    )
    # Process noise starts after a_0, so K=1 is an exact negative control.
    np.testing.assert_allclose(
        strong_observability_k1_low_noise["first_action_mse"],
        strong_observability_k1_high_noise["first_action_mse"],
        rtol=0.0,
        atol=1e-14,
    )
    # With K=20, unpredictable future heads compete for the shared bottleneck.
    high_noise_degradation = (
        strong_observability_high_noise["first_action_mse"]
        - strong_observability_k1_high_noise["first_action_mse"]
    )
    low_noise_degradation = (
        strong_observability["first_action_mse"]
        - strong_observability_k1_low_noise["first_action_mse"]
    )
    assert high_noise_degradation > low_noise_degradation + 0.02


def test_cli_writes_tidy_per_offset_csv(tmp_path):
    output_csv = tmp_path / "learned_bottleneck.csv"

    exit_code = main(
        [
            "--output-csv",
            str(output_csv),
            "--horizons",
            "1",
            "2",
            "--spectral-radii",
            "0.7",
            "--observability-ranks",
            "2",
            "--observability-scales",
            "0.1",
            "--predictive-noise-stds",
            "0.25",
            "--seeds",
            "0",
            "--n-train",
            "64",
            "--n-test",
            "128",
            "--nuisance-dim",
            "6",
        ]
    )

    assert exit_code == 0
    with output_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3  # one K=1 offset and two K=2 offsets
    assert {int(row["train_horizon"]) for row in rows} == {1, 2}
    assert {int(row["eval_offset"]) for row in rows} == {0, 1}
    required_metrics = {
        "state_probe_r2",
        "offset_mse",
        "first_action_mse",
        "observability_rank_realized",
        "offset_predictive_noise_variance",
    }
    assert required_metrics <= set(rows[0])
