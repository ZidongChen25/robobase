import csv
import json
from pathlib import Path

import pytest

from scripts.summarize_cqn_no_bc_stage41_raw50_sentinel import (
    SENTINEL_STEPS,
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
        for step, value in zip(SENTINEL_STEPS, values, strict=True):
            writer.writerow(
                {
                    "env_steps": step,
                    "episode_success": value,
                    "eval_episodes": 50,
                    "eval_seed_start": 400,
                }
            )


def _setup(tmp_path: Path, curves: list[list[float]]) -> Path:
    stage = tmp_path / "stage41"
    stage.mkdir()
    (stage / "stage41_extension_summary.json").write_text(
        json.dumps(
            {
                "eligible_for_separately_designed_50k_sentinel": True,
                "per_seed": {
                    "seed1": {"extension_best": {"best_success": 0.66}},
                    "seed2": {"extension_best": {"best_success": 0.52}},
                },
            }
        )
    )
    for seed, values in enumerate(curves, start=1):
        _curve(
            stage
            / f"seed{seed}"
            / "offline_dense_online_positive_dense"
            / "val50_seeds400_stage41_raw50_sentinel.csv",
            values,
        )
    return stage


def test_raw50_sentinel_passes_sustained_curve(tmp_path: Path):
    stage = _setup(
        tmp_path,
        [
            [0.62, 0.60, 0.64, 0.58, 0.62, 0.60, 0.64, 0.58],
            [0.50, 0.52, 0.48, 0.54, 0.50, 0.52, 0.50, 0.48],
        ],
    )
    result = summarize(stage)
    assert result["sentinel_mean_best"] == pytest.approx(0.59)
    assert result["late_window_mean_best"] == pytest.approx(0.58)
    assert result["raw50_endpoint_mean"] == pytest.approx(0.53)
    assert result["eligible_for_full_run_protocol"]
    assert not result["heldout_opened"]


def test_raw50_sentinel_rejects_late_erosion_despite_early_peak(tmp_path: Path):
    stage = _setup(
        tmp_path,
        [
            [0.68, 0.66, 0.62, 0.58, 0.52, 0.46, 0.42, 0.40],
            [0.56, 0.54, 0.52, 0.50, 0.48, 0.44, 0.42, 0.40],
        ],
    )
    result = summarize(stage)
    assert result["sentinel_mean_best"] == pytest.approx(0.62)
    assert not result["eligible_for_full_run_protocol"]
    assert result["next_decision"] == "stop_stage41_scaling_after_raw50k"


def test_raw50_sentinel_rejects_checkpoint_selection_only_signal(
    tmp_path: Path,
):
    stage = _setup(
        tmp_path,
        [
            [0.66, 0.48, 0.48, 0.48, 0.48, 0.60, 0.48, 0.46],
            [0.52, 0.48, 0.48, 0.48, 0.48, 0.52, 0.48, 0.46],
        ],
    )
    result = summarize(stage)
    assert result["sentinel_mean_best"] == pytest.approx(0.59)
    assert result["all_checkpoint_mean"] < 0.52
    assert not result["eligible_for_full_run_protocol"]


def test_raw50_sentinel_requires_registered_raw30_gate(tmp_path: Path):
    stage = _setup(
        tmp_path,
        [[0.60] * 8, [0.60] * 8],
    )
    summary_path = stage / "stage41_extension_summary.json"
    payload = json.loads(summary_path.read_text())
    payload["eligible_for_separately_designed_50k_sentinel"] = False
    summary_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="did not authorize"):
        summarize(stage)
