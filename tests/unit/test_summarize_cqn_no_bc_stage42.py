import csv
import json
from pathlib import Path

import pytest

from scripts.summarize_cqn_no_bc_stage42 import STEPS, summarize


def _write_curve(
    path: Path,
    values: list[float],
    steps: tuple[int, ...] = STEPS,
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
        for step, value in zip(steps, values, strict=True):
            writer.writerow(
                {
                    "env_steps": step,
                    "episode_success": value,
                    "eval_episodes": 50,
                    "eval_seed_start": 400,
                }
            )

def _setup(tmp_path: Path, treatment: list[list[float]], *, grow_buffer=False):
    stage42 = tmp_path / "stage42"
    stage41 = tmp_path / "stage41"
    stage38 = tmp_path / "stage38"
    stage41_values = ([0.56, 0.50, 0.62, 0.62, 0.64, 0.64, 0.54, 0.66],
                      [0.50, 0.56, 0.58, 0.48, 0.46, 0.46, 0.52, 0.46])
    stage38_values = ([0.52, 0.56, 0.58, 0.50, 0.50, 0.50, 0.52, 0.46],
                      [0.40, 0.50, 0.38, 0.42, 0.44, 0.38, 0.36, 0.38])
    for seed in (1, 2):
        run = stage42 / f"seed{seed}" / "offline_dense_online_positive_fixed_expert"
        _write_curve(run / "val50_seeds400_stage42.csv", treatment[seed - 1])
        run.mkdir(parents=True, exist_ok=True)
        (run / "stage42_branch_manifest.json").write_text(
            json.dumps({"replay": {"demo_replay": {"num_transitions": 9253}}})
        )
        with (run / "train.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("env_steps", "demo_buffer_size"))
            writer.writeheader()
            writer.writerow({"env_steps": 10000, "demo_buffer_size": 9253})
            writer.writerow(
                {
                    "env_steps": 29000,
                    "demo_buffer_size": 10000 if grow_buffer else 9253,
                }
            )

        s41 = stage41 / f"seed{seed}" / "offline_dense_online_positive_dense"
        _write_curve(
            s41 / "val50_seeds400_stage41.csv",
            stage41_values[seed - 1][:4],
            STEPS[:4],
        )
        _write_curve(
            s41 / "val50_seeds400_stage41_extension.csv",
            stage41_values[seed - 1][4:],
            STEPS[4:],
        )
        s38 = stage38 / f"dense_seed{seed}" / "offline_then_online"
        _write_curve(
            s38 / "val50_seeds400_selection.csv",
            stage38_values[seed - 1][:4],
            STEPS[:4],
        )
        _write_curve(
            s38 / "val50_seeds400_extension.csv",
            stage38_values[seed - 1][4:],
            STEPS[4:],
        )
    return stage42, stage41, stage38


def test_stage42_passes_robust_fixed_expert_curve(tmp_path: Path):
    dirs = _setup(
        tmp_path,
        [
            [0.62, 0.64, 0.60, 0.62, 0.58, 0.64, 0.60, 0.58],
            [0.56, 0.58, 0.56, 0.54, 0.58, 0.56, 0.54, 0.52],
        ],
    )
    result = summarize(*dirs)
    assert result["fixed_expert_mean_best"] == pytest.approx(0.61)
    assert result["raw30_endpoint_mean"] == pytest.approx(0.55)
    assert result["all_demo_buffers_fixed"]
    assert result["eligible_for_separately_designed_raw50_replication"]
    assert not result["eligible_for_full_run"]


def test_stage42_rejects_seed_collapse(tmp_path: Path):
    dirs = _setup(
        tmp_path,
        [[0.64] * 8, [0.54, 0.52, 0.46, 0.44, 0.42, 0.40, 0.38, 0.36]],
    )
    result = summarize(*dirs)
    assert not result["eligible_for_separately_designed_raw50_replication"]
    assert result["next_decision"] == "stop_stage42_after_raw30_gate"


def test_stage42_rejects_any_demo_buffer_growth(tmp_path: Path):
    dirs = _setup(tmp_path, [[0.62] * 8, [0.58] * 8], grow_buffer=True)
    result = summarize(*dirs)
    assert not result["all_demo_buffers_fixed"]
    assert not result["eligible_for_separately_designed_raw50_replication"]
