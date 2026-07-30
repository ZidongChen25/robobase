import json

import pytest

from scripts.summarize_cqn_multiseed_paired import summarize


def _write(path, snapshot, values, *, seed_start=49000):
    path.write_text(
        json.dumps(
            {
                "status": "ok",
                "snapshot": snapshot,
                "episode_results": [
                    {
                        "seed": seed_start + index,
                        "episode_success": value,
                    }
                    for index, value in enumerate(values)
                ],
            }
        )
    )


def test_multiseed_paired_summary_crosses_model_and_environment_seeds(
    tmp_path,
):
    pairs = []
    for seed, baseline_values, candidate_values in (
        (1, [0, 0, 0, 1], [1, 1, 0, 1]),
        (2, [0, 0, 1, 0], [1, 0, 1, 1]),
        (3, [0, 1, 0, 0], [1, 1, 1, 0]),
    ):
        baseline = tmp_path / f"baseline{seed}.json"
        candidate = tmp_path / f"candidate{seed}.json"
        _write(baseline, f"baseline{seed}.pkl", baseline_values)
        _write(candidate, f"candidate{seed}.pkl", candidate_values)
        pairs.append((f"seed{seed}", baseline, candidate))

    payload = summarize(
        pairs,
        bootstrap_replicates=2_000,
        bootstrap_seed=7,
        min_mean_delta=0.0,
        min_ci_lower=0.0,
    )

    assert payload["mean_baseline_success"] == pytest.approx(0.25)
    assert payload["mean_candidate_success"] == pytest.approx(0.75)
    assert payload["mean_paired_delta"] == pytest.approx(0.5)
    assert payload["aggregate_paired_wins"] == 6
    assert payload["aggregate_paired_losses"] == 0
    assert payload["gate"] == "pass"
    assert payload["crossed_bootstrap_ci95"][0] >= 0.0


def test_multiseed_paired_summary_requires_common_eval_seeds(tmp_path):
    baseline1 = tmp_path / "baseline1.json"
    candidate1 = tmp_path / "candidate1.json"
    baseline2 = tmp_path / "baseline2.json"
    candidate2 = tmp_path / "candidate2.json"
    _write(baseline1, "b1.pkl", [0, 1])
    _write(candidate1, "c1.pkl", [1, 1])
    _write(baseline2, "b2.pkl", [0, 1], seed_start=50000)
    _write(candidate2, "c2.pkl", [1, 1], seed_start=50000)

    with pytest.raises(ValueError, match="common eval seeds"):
        summarize(
            [
                ("seed1", baseline1, candidate1),
                ("seed2", baseline2, candidate2),
            ],
            bootstrap_replicates=100,
            bootstrap_seed=7,
            min_mean_delta=0.0,
            min_ci_lower=0.0,
        )
