import csv
import json
from pathlib import Path

import pytest

from scripts.summarize_cqn_no_bc_stage42_raw50 import STEPS, summarize


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
        for step, value in zip(STEPS, values, strict=True):
            writer.writerow(
                {
                    "env_steps": step,
                    "episode_success": value,
                    "eval_episodes": 50,
                    "eval_seed_start": 400,
                }
            )


def _setup(tmp_path: Path, curves: list[list[float]], *, grow=False):
    stage42 = tmp_path / "stage42"
    stage41 = tmp_path / "stage41"
    stage42.mkdir()
    stage41.mkdir()
    (stage42 / "stage42_summary.json").write_text(
        json.dumps(
            {
                "eligible_for_separately_designed_raw50_replication": True,
                "per_seed": {
                    "seed1": {"fixed_expert_best": {"best_success": 0.74}},
                    "seed2": {"fixed_expert_best": {"best_success": 0.66}},
                },
            }
        )
    )
    (stage41 / "stage41_raw50_sentinel_summary.json").write_text(
        json.dumps(
            {
                "per_seed": {
                    "seed1": {"sentinel_best": {"best_success": 0.76}},
                    "seed2": {"sentinel_best": {"best_success": 0.44}},
                }
            }
        )
    )
    for seed in (1, 2):
        run = stage42 / f"seed{seed}" / "offline_dense_online_positive_fixed_expert"
        _curve(run / "val50_seeds400_stage42_raw50.csv", curves[seed - 1])
        (run / "stage42_branch_manifest.json").write_text(
            json.dumps({"replay": {"demo_replay": {"num_transitions": 9253}}})
        )
        with (run / "train.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("env_steps", "demo_buffer_size"))
            writer.writeheader()
            writer.writerow({"env_steps": 10000, "demo_buffer_size": 9253})
            writer.writerow(
                {"env_steps": 49000, "demo_buffer_size": 9900 if grow else 9253}
            )
    return stage42, stage41


def test_stage42_raw50_passes_robust_replication(tmp_path: Path):
    dirs = _setup(
        tmp_path,
        [
            [0.68, 0.70, 0.66, 0.62, 0.68, 0.64, 0.66, 0.62],
            [0.60, 0.62, 0.58, 0.60, 0.62, 0.58, 0.60, 0.56],
        ],
    )
    result = summarize(*dirs)
    assert result["replication_mean_best"] == pytest.approx(0.66)
    assert result["raw50_endpoint_mean"] == pytest.approx(0.59)
    assert result["eligible_for_matched_raw101k_full_protocol"]
    assert not result["heldout_opened"]


def test_stage42_raw50_rejects_late_seed_erosion(tmp_path: Path):
    dirs = _setup(
        tmp_path,
        [
            [0.70, 0.68, 0.66, 0.64, 0.62, 0.58, 0.56, 0.54],
            [0.62, 0.60, 0.58, 0.54, 0.50, 0.46, 0.42, 0.40],
        ],
    )
    result = summarize(*dirs)
    assert not result["eligible_for_matched_raw101k_full_protocol"]
    assert result["next_decision"] == "stop_stage42_after_raw50_replication"


def test_stage42_raw50_rejects_demo_buffer_growth(tmp_path: Path):
    dirs = _setup(tmp_path, [[0.66] * 8, [0.62] * 8], grow=True)
    result = summarize(*dirs)
    assert not result["all_demo_buffers_fixed"]
    assert not result["eligible_for_matched_raw101k_full_protocol"]
