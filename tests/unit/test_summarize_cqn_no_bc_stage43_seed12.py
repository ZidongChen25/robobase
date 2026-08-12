import csv
import json
from pathlib import Path

import pytest

from scripts.summarize_cqn_no_bc_stage43_seed12 import (
    INITIAL_STEPS,
    LATER_STEPS,
    REPLICATION_STEPS,
    summarize,
)


def _curve(path: Path, steps: tuple[int, ...], values: list[float]) -> None:
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


def _setup(tmp_path: Path, later_curves: list[list[float]], *, grow=False):
    stage42 = tmp_path / "stage42"
    stage43 = tmp_path / "stage43"
    stage42.mkdir()
    stage43.mkdir()
    (stage42 / "stage42_raw50_summary.json").write_text(
        json.dumps({"eligible_for_matched_raw101k_full_protocol": True})
    )
    initial = ([0.62, 0.56, 0.62, 0.66, 0.56, 0.74, 0.56, 0.54],
               [0.46, 0.52, 0.48, 0.66, 0.54, 0.50, 0.56, 0.62])
    replication = ([0.64, 0.62, 0.66, 0.64, 0.68, 0.66, 0.62, 0.62],
                   [0.58, 0.62, 0.70, 0.70, 0.80, 0.72, 0.74, 0.68])
    for seed in (1, 2):
        old = stage42 / f"seed{seed}" / "offline_dense_online_positive_fixed_expert"
        _curve(old / "val50_seeds400_stage42.csv", INITIAL_STEPS, list(initial[seed - 1]))
        _curve(
            old / "val50_seeds400_stage42_raw50.csv",
            REPLICATION_STEPS,
            list(replication[seed - 1]),
        )
        run = stage43 / f"seed{seed}" / "fixed_expert_101k_online"
        _curve(
            run / "val50_seeds400_stage43_seed12.csv",
            LATER_STEPS,
            later_curves[seed - 1],
        )
        (run / "stage43_branch_manifest.json").write_text(
            json.dumps({"replay": {"demo_replay": {"num_transitions": 9253}}})
        )
        with (run / "train.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("demo_buffer_size",))
            writer.writeheader()
            writer.writerow({"demo_buffer_size": 9900 if grow else 9253})
    return stage43, stage42


def test_stage43_seed12_passes_persistent_full_curves(tmp_path: Path):
    dirs = _setup(
        tmp_path,
        [[0.66, 0.64, 0.62, 0.60, 0.58, 0.56, 0.58],
         [0.72, 0.70, 0.68, 0.66, 0.64, 0.62, 0.60]],
    )
    result = summarize(*dirs)
    assert result["validation_selected_mean_best"] == pytest.approx(0.77)
    assert result["raw111k_fixed_endpoint_mean"] == pytest.approx(0.59)
    assert result["eligible_for_fresh_seed34_full_runs"]
    assert not result["heldout_opened"]


def test_stage43_seed12_rejects_full_endpoint_collapse(tmp_path: Path):
    dirs = _setup(
        tmp_path,
        [[0.62, 0.58, 0.52, 0.48, 0.42, 0.38, 0.30],
         [0.66, 0.60, 0.54, 0.48, 0.42, 0.36, 0.30]],
    )
    result = summarize(*dirs)
    assert not result["eligible_for_fresh_seed34_full_runs"]
    assert result["next_decision"].startswith("stop_stage43")


def test_stage43_seed12_rejects_demo_buffer_growth(tmp_path: Path):
    dirs = _setup(tmp_path, [[0.65] * 7, [0.70] * 7], grow=True)
    result = summarize(*dirs)
    assert not result["all_demo_buffers_fixed"]
    assert not result["eligible_for_fresh_seed34_full_runs"]
