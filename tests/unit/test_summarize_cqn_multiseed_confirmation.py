import json

import pytest

from scripts.summarize_cqn_multiseed_confirmation import summarize


def _write(path, snapshot, values):
    path.write_text(
        json.dumps(
            {
                "status": "ok",
                "snapshot": snapshot,
                "episode_results": [
                    {"seed": 43000 + index, "episode_success": value}
                    for index, value in enumerate(values)
                ],
            }
        )
    )


def test_multiseed_confirmation_reports_selected_checkpoint_distribution(
    tmp_path,
):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write(first, "5000_snapshot.pkl", [1, 0, 1, 0])
    _write(second, "2500_snapshot.pkl", [1, 1, 1, 0])

    payload = summarize(
        [("seed1", first), ("seed2", second)],
        bootstrap_replicates=1000,
        bootstrap_seed=7,
    )

    assert payload["per_training_seed_success"] == {
        "seed1": 0.5,
        "seed2": 0.75,
    }
    assert payload["mean_success_across_training_seeds"] == pytest.approx(
        0.625
    )
    assert payload["sample_std_across_training_seeds"] > 0.0
    assert payload["crossed_bootstrap_ci95"][0] <= 0.625
    assert payload["crossed_bootstrap_ci95"][1] >= 0.625
