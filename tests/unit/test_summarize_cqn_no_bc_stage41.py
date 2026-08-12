import csv
from pathlib import Path

import pytest

from scripts.summarize_cqn_no_bc_stage41 import POST_STEPS, summarize


def _write(path: Path, values: dict[int, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "env_steps",
                "iteration",
                "episode_success",
                "episode_reward",
                "eval_episodes",
                "eval_seed_start",
            ),
        )
        writer.writeheader()
        for step, value in values.items():
            writer.writerow(
                {
                    "env_steps": step,
                    "iteration": step,
                    "episode_success": value,
                    "episode_reward": value,
                    "eval_episodes": 50,
                    "eval_seed_start": 400,
                }
            )


def _setup(tmp_path: Path, treatment, control):
    stage41 = tmp_path / "stage41"
    stage38 = tmp_path / "stage38"
    stage40 = tmp_path / "stage40"
    for seed in (1, 2):
        _write(
            stage41
            / f"seed{seed}"
            / "offline_dense_online_positive_dense"
            / "val50_seeds400_stage41.csv",
            dict(zip(POST_STEPS, treatment[seed - 1], strict=True)),
        )
        _write(
            stage38
            / f"dense_seed{seed}"
            / "offline_then_online"
            / "val50_seeds400_selection.csv",
            {
                10000: (0.52, 0.50)[seed - 1],
                **dict(zip(POST_STEPS, control[seed - 1], strict=True)),
            },
        )
    return stage41, stage38, stage40


def test_stage41_passes_noninferior_best_and_better_endpoint(tmp_path: Path):
    stage41, stage38, stage40 = _setup(
        tmp_path,
        treatment=[[0.58, 0.60, 0.58, 0.52], [0.50, 0.52, 0.50, 0.48]],
        control=[[0.52, 0.56, 0.58, 0.50], [0.40, 0.50, 0.38, 0.42]],
    )
    result = summarize(stage41, stage38, stage40)
    assert result["treatment_post_mean_best"] == pytest.approx(0.56)
    assert result["stage38_full_dense_post_mean_best"] == pytest.approx(0.54)
    assert result["treatment_raw20_endpoint_mean"] == pytest.approx(0.50)
    assert result["mechanism_pass"]
    assert result["eligible_for_bounded_raw30_extension"]
    assert not result["eligible_for_full_run"]


def test_stage41_fails_if_one_seed_collapses(tmp_path: Path):
    stage41, stage38, stage40 = _setup(
        tmp_path,
        treatment=[[0.58, 0.60, 0.58, 0.52], [0.0, 0.0, 0.0, 0.0]],
        control=[[0.52, 0.56, 0.58, 0.50], [0.40, 0.50, 0.38, 0.42]],
    )
    result = summarize(stage41, stage38, stage40)
    assert not result["mechanism_pass"]
    assert result["next_decision"] == "stop_stage41_after_raw20_gate"
