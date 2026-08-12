import csv
from pathlib import Path

import pytest

from scripts.summarize_cqn_no_bc_stage40 import STEPS, summarize


def _write_curve(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "env_steps",
                "eval_episodes",
                "eval_seed_start",
                "episode_success",
            ),
        )
        writer.writeheader()
        for step, value in zip(STEPS, values, strict=True):
            writer.writerow(
                {
                    "env_steps": step,
                    "eval_episodes": 50,
                    "eval_seed_start": 400,
                    "episode_success": value,
                }
            )


def _paths(tmp_path: Path, treatment, control):
    stage = tmp_path / "stage40"
    stage38 = tmp_path / "stage38"
    for seed in (1, 2):
        _write_curve(
            stage
            / f"seed{seed}"
            / "offline_dense_online_canonical"
            / "val50_seeds400_stage40.csv",
            treatment[seed - 1],
        )
        _write_curve(
            stage38
            / f"dense_seed{seed}"
            / "offline_then_online"
            / "val50_seeds400_selection.csv",
            control[seed - 1],
        )
    return stage, stage38


def test_stage40_gate_passes_only_on_paired_gain_and_endpoint(tmp_path: Path):
    stage, stage38 = _paths(
        tmp_path,
        treatment=[
            [0.52, 0.60, 0.62, 0.58, 0.54],
            [0.50, 0.54, 0.56, 0.52, 0.50],
        ],
        control=[
            [0.52, 0.52, 0.56, 0.58, 0.50],
            [0.50, 0.40, 0.50, 0.38, 0.42],
        ],
    )
    result = summarize(stage, stage38)
    assert result["shared_raw10_branch_integrity"]
    assert result["control_post_handoff_mean_best"] == pytest.approx(0.54)
    assert result["treatment_post_handoff_mean_best"] == pytest.approx(0.59)
    assert result["mean_best_gain"] == pytest.approx(0.05)
    assert result["treatment_raw20_endpoint_mean"] == pytest.approx(0.52)
    assert result["mechanism_pass"]
    assert result["eligible_for_bounded_raw30_extension"]
    assert not result["eligible_for_full_run"]


def test_stage40_gate_fails_without_positive_scaling(tmp_path: Path):
    stage, stage38 = _paths(
        tmp_path,
        treatment=[
            [0.52, 0.54, 0.56, 0.50, 0.48],
            [0.50, 0.44, 0.48, 0.42, 0.40],
        ],
        control=[
            [0.52, 0.52, 0.56, 0.58, 0.50],
            [0.50, 0.40, 0.50, 0.38, 0.42],
        ],
    )
    result = summarize(stage, stage38)
    assert not result["mechanism_pass"]
    assert result["next_decision"] == "stop_stage40_after_raw20_gate"


def test_stage40_gate_fails_if_shared_checkpoint_eval_changes(tmp_path: Path):
    stage, stage38 = _paths(
        tmp_path,
        treatment=[
            [0.50, 0.62, 0.64, 0.60, 0.56],
            [0.50, 0.56, 0.58, 0.54, 0.52],
        ],
        control=[
            [0.52, 0.52, 0.56, 0.58, 0.50],
            [0.50, 0.40, 0.50, 0.38, 0.42],
        ],
    )
    result = summarize(stage, stage38)
    assert not result["shared_raw10_branch_integrity"]
    assert not result["mechanism_pass"]
