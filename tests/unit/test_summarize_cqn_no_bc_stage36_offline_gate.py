import csv
from pathlib import Path

import pytest

from scripts.summarize_cqn_no_bc_stage36_offline_gate import (
    EXPECTED_STEPS,
    summarize,
)


def _write_curve(
    path: Path,
    values: list[float],
    *,
    episodes: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "env_steps",
                "episode_success",
                "eval_episodes",
                "eval_seed_start",
            ),
        )
        writer.writeheader()
        for step, value in zip(EXPECTED_STEPS, values, strict=True):
            writer.writerow(
                {
                    "env_steps": step,
                    "episode_success": value,
                    "eval_episodes": episodes,
                    "eval_seed_start": 400,
                }
            )


def _treatment(base: Path) -> Path:
    return base / "treatment" / "offline_then_online_seed1"


def _control(base: Path) -> Path:
    return base / "control" / "offline_then_online_seed1"


def test_all_zero_coarse_curve_stops_without_matched_validation(tmp_path):
    _write_curve(
        _treatment(tmp_path) / "val20_seeds400_coarse.csv",
        [0.0] * 5,
        episodes=20,
    )

    result = summarize(tmp_path)

    assert result["coarse_nonzero"] is False
    assert result["matched_validation_complete"] is False
    assert result["eligible_for_20k_online_extension"] is False
    assert result["next_decision"] == "stop_all_zero_initial_scaling_curve"


def test_nonzero_coarse_curve_unlocks_matched_validation_only(tmp_path):
    _write_curve(
        _treatment(tmp_path) / "val20_seeds400_coarse.csv",
        [0.0, 0.05, 0.0, 0.10, 0.10],
        episodes=20,
    )

    result = summarize(tmp_path)

    assert result["coarse_nonzero"] is True
    assert result["matched_validation_complete"] is False
    assert result["next_decision"] == "run_matched_50_episode_selection_curve"


def test_matched_best_uses_earliest_tie_and_late_gate(tmp_path):
    _write_curve(
        _treatment(tmp_path) / "val20_seeds400_coarse.csv",
        [0.0, 0.05, 0.10, 0.20, 0.15],
        episodes=20,
    )
    _write_curve(
        _control(tmp_path) / "val50_seeds400_selection.csv",
        [0.10, 0.30, 0.30, 0.20, 0.20],
        episodes=50,
    )
    _write_curve(
        _treatment(tmp_path) / "val50_seeds400_selection.csv",
        [0.0, 0.20, 0.40, 0.40, 0.30],
        episodes=50,
    )

    result = summarize(tmp_path)

    assert result["matched_control"]["best_step"] == 12500
    assert result["matched_treatment"]["best_step"] == 15000
    assert result["treatment_minus_control_best"] == pytest.approx(0.10)
    assert result["eligible_for_20k_online_extension"] is True
    assert result["next_decision"] == (
        "eligible_for_separately_launched_20k_online_extension"
    )


def test_rejects_wrong_episode_count(tmp_path):
    _write_curve(
        _treatment(tmp_path) / "val20_seeds400_coarse.csv",
        [0.0] * 5,
        episodes=50,
    )

    with pytest.raises(ValueError, match="does not use 20 episodes"):
        summarize(tmp_path)
