import csv
import json
from pathlib import Path

import pytest

from scripts.summarize_cqn_no_bc_stage41_extension import (
    EXTENSION_STEPS,
    summarize,
)


def _curve(path: Path, values: list[float]) -> None:
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
        for step, value in zip(EXTENSION_STEPS, values, strict=True):
            writer.writerow(
                {
                    "env_steps": step,
                    "episode_success": value,
                    "eval_episodes": 50,
                    "eval_seed_start": 400,
                }
            )


def _setup(tmp_path: Path, curves):
    stage = tmp_path / "stage41"
    stage38 = tmp_path / "stage38"
    (stage / "stage41_summary.json").parent.mkdir(parents=True, exist_ok=True)
    (stage / "stage41_summary.json").write_text(
        json.dumps(
            {
                "eligible_for_bounded_raw30_extension": True,
                "treatment_post_mean_best": 0.60,
                "per_seed": {
                    "seed1": {"treatment_post_best": {"best_success": 0.62}},
                    "seed2": {"treatment_post_best": {"best_success": 0.58}},
                },
            }
        )
    )
    stage38.mkdir(parents=True, exist_ok=True)
    (stage38 / "stage38_extension_summary.json").write_text(
        json.dumps(
            {
                "per_seed": {
                    "seed1": {"extension": {"best_success": 0.52}},
                    "seed2": {"extension": {"best_success": 0.44}},
                }
            }
        )
    )
    for seed, values in enumerate(curves, start=1):
        _curve(
            stage
            / f"seed{seed}"
            / "offline_dense_online_positive_dense"
            / "val50_seeds400_stage41_extension.csv",
            values,
        )
    return stage, stage38


def test_stage41_extension_passes_stable_scaling(tmp_path: Path):
    stage, stage38 = _setup(
        tmp_path,
        [[0.60, 0.62, 0.58, 0.56], [0.56, 0.58, 0.56, 0.50]],
    )
    result = summarize(stage, stage38)
    assert result["extension_mean_best"] == pytest.approx(0.60)
    assert result["raw30_endpoint_mean"] == pytest.approx(0.53)
    assert result["any_extension_best_reaches_initial"]
    assert result["eligible_for_separately_designed_50k_sentinel"]
    assert not result["eligible_for_full_run"]


def test_stage41_extension_fails_endpoint_erosion(tmp_path: Path):
    stage, stage38 = _setup(
        tmp_path,
        [[0.62, 0.58, 0.50, 0.44], [0.58, 0.54, 0.46, 0.42]],
    )
    result = summarize(stage, stage38)
    assert result["extension_mean_best"] == pytest.approx(0.60)
    assert not result["eligible_for_separately_designed_50k_sentinel"]
    assert result["next_decision"] == "stop_stage41_scaling_after_raw30k"
