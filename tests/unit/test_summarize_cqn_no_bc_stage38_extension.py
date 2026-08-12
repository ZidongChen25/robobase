import csv
import json
from pathlib import Path

import pytest

from scripts.summarize_cqn_no_bc_stage38_extension import (
    EXTENSION_STEPS,
    INITIAL_STEPS,
    summarize,
)


def _write(path: Path, steps: tuple[int, ...], values: list[float]) -> None:
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
        for step, value in zip(steps, values, strict=True):
            writer.writerow(
                {
                    "env_steps": step,
                    "episode_success": value,
                    "eval_episodes": 50,
                    "eval_seed_start": 400,
                }
            )


def _prepare(tmp_path: Path, extension: dict[int, list[float]]) -> None:
    initial = {
        1: [0.52, 0.52, 0.56, 0.58, 0.50],
        2: [0.50, 0.40, 0.50, 0.38, 0.42],
    }
    summary = {
        "matched_validation_complete": True,
        "eligible_for_20k_online_extension": True,
        "matched": {"stage36_bc_seed1": {"best_success": 0.66}},
    }
    (tmp_path / "stage38_summary.json").write_text(json.dumps(summary))
    for seed in (1, 2):
        run = tmp_path / f"dense_seed{seed}" / "offline_then_online"
        _write(run / "val50_seeds400_selection.csv", INITIAL_STEPS, initial[seed])
        _write(
            run / "val50_seeds400_extension.csv",
            EXTENSION_STEPS,
            extension[seed],
        )


def test_supported_extension_can_only_request_50k_sentinel(tmp_path):
    _prepare(
        tmp_path,
        {
            1: [0.48, 0.55, 0.52, 0.50],
            2: [0.42, 0.45, 0.50, 0.48],
        },
    )

    result = summarize(tmp_path)

    assert result["initial_two_seed_mean_best"] == pytest.approx(0.54)
    assert result["extension_two_seed_mean_best"] == pytest.approx(0.525)
    assert result["combined_two_seed_mean_best"] == pytest.approx(0.54)
    assert result["eligible_for_separately_designed_50k_scaling_sentinel"] is True
    assert result["eligible_for_full_run"] is False
    assert result["protocol"]["heldout_seeds_800_999"] == "sealed"


def test_collapsed_extension_stops_without_scale_or_full(tmp_path):
    _prepare(tmp_path, {1: [0.0] * 4, 2: [0.0] * 4})

    result = summarize(tmp_path)

    assert result["both_extension_best_at_least_40pct"] is False
    assert result["eligible_for_separately_designed_50k_scaling_sentinel"] is False
    assert result["eligible_for_full_run"] is False
    assert result["next_decision"] == "stop_stage38_scaling_after_raw30k"


def test_extension_requires_completed_matched_gate(tmp_path):
    (tmp_path / "stage38_summary.json").write_text(
        json.dumps(
            {
                "matched_validation_complete": False,
                "eligible_for_20k_online_extension": True,
            }
        )
    )

    with pytest.raises(ValueError, match="matched validation is not complete"):
        summarize(tmp_path)
