import csv
from pathlib import Path

import pytest

from scripts.summarize_cqn_no_bc_stage38 import EXPECTED_STEPS, summarize


def _write(path: Path, values: list[float], episodes: int = 20) -> None:
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


def _candidate(base: Path, seed: int, name: str) -> Path:
    return base / f"dense_seed{seed}" / "offline_then_online" / name


def _baseline(base: Path, arm: str) -> Path:
    return base / arm / "offline_then_online_seed1" / "val50_seeds400_stage38.csv"


def test_two_all_zero_curves_stop(tmp_path):
    baseline = tmp_path / "baseline"
    for seed in (1, 2):
        _write(_candidate(tmp_path, seed, "val20_seeds400_coarse.csv"), [0.0] * 5)

    result = summarize(tmp_path, baseline)

    assert result["coarse_qualification_pass"] is False
    assert result["eligible_for_20k_online_extension"] is False
    assert result["next_decision"] == "stop_dense_offline_gate_without_full_sweep"


def test_requires_both_seeds_nonzero_and_a_late_signal(tmp_path):
    baseline = tmp_path / "baseline"
    _write(
        _candidate(tmp_path, 1, "val20_seeds400_coarse.csv"),
        [0.0, 0.20, 0.30, 0.20, 0.20],
    )
    _write(_candidate(tmp_path, 2, "val20_seeds400_coarse.csv"), [0.0] * 5)

    result = summarize(tmp_path, baseline)

    assert result["coarse_both_seeds_nonzero"] is False
    assert result["coarse_any_late_at_least_20pct"] is True
    assert result["coarse_qualification_pass"] is False


def test_qualified_coarse_curves_request_matched_selection(tmp_path):
    baseline = tmp_path / "baseline"
    _write(
        _candidate(tmp_path, 1, "val20_seeds400_coarse.csv"),
        [0.0, 0.10, 0.20, 0.25, 0.25],
    )
    _write(
        _candidate(tmp_path, 2, "val20_seeds400_coarse.csv"),
        [0.0, 0.05, 0.10, 0.10, 0.15],
    )

    result = summarize(tmp_path, baseline)

    assert result["coarse_qualification_pass"] is True
    assert result["matched_validation_complete"] is False
    assert result["next_decision"] == "run_matched_50_episode_selection_curves"


def test_matched_gate_uses_two_seed_best_and_late_curves(tmp_path):
    baseline = tmp_path / "baseline"
    coarse = {
        1: [0.0, 0.10, 0.20, 0.25, 0.25],
        2: [0.0, 0.05, 0.10, 0.20, 0.20],
    }
    matched = {
        1: [0.10, 0.40, 0.50, 0.30, 0.25],
        2: [0.10, 0.30, 0.40, 0.25, 0.20],
    }
    for seed in (1, 2):
        _write(_candidate(tmp_path, seed, "val20_seeds400_coarse.csv"), coarse[seed])
        _write(
            _candidate(tmp_path, seed, "val50_seeds400_selection.csv"),
            matched[seed],
            episodes=50,
        )
    _write(_baseline(baseline, "treatment"), [0.0] * 5, episodes=50)
    _write(
        _baseline(baseline, "control"),
        [0.20, 0.30, 0.40, 0.35, 0.30],
        episodes=50,
    )

    result = summarize(tmp_path, baseline)

    assert result["matched"]["dense_seed1"]["best_step"] == 15000
    assert result["dense_two_seed_mean_best"] == pytest.approx(0.45)
    assert result["dense_seed1_minus_stage36_nobc_best"] == pytest.approx(0.50)
    assert result["dense_seed1_minus_stage36_bc_best"] == pytest.approx(0.10)
    assert result["eligible_for_20k_online_extension"] is True


def test_rejects_wrong_coarse_episode_count(tmp_path):
    baseline = tmp_path / "baseline"
    _write(
        _candidate(tmp_path, 1, "val20_seeds400_coarse.csv"),
        [0.0] * 5,
        episodes=50,
    )
    _write(_candidate(tmp_path, 2, "val20_seeds400_coarse.csv"), [0.0] * 5)

    with pytest.raises(ValueError, match="does not use 20 episodes"):
        summarize(tmp_path, baseline)
