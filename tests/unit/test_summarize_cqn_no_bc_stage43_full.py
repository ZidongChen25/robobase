import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.summarize_cqn_no_bc_stage43_full import summarize
from scripts.summarize_cqn_no_bc_stage43_seed12 import ALL_STEPS


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
        for step, value in zip(ALL_STEPS, values, strict=True):
            writer.writerow(
                {
                    "env_steps": step,
                    "episode_success": value,
                    "eval_episodes": 50,
                    "eval_seed_start": 400,
                }
            )


def _setup(tmp_path: Path, curves: list[list[float]], *, grow=False) -> Path:
    base = tmp_path / "stage43"
    base.mkdir()
    per_seed = {}
    for seed in (1, 2):
        per_seed[f"seed{seed}"] = {
            "curve": {
                str(step): value
                for step, value in zip(ALL_STEPS, curves[seed - 1], strict=True)
            },
            "demo_buffer": {
                "expected_transitions": 9253,
                "observed_sizes": [9253],
                "fixed": True,
            },
        }
    (base / "stage43_seed12_summary.json").write_text(
        json.dumps(
            {
                "eligible_for_fresh_seed34_full_runs": True,
                "per_seed": per_seed,
            }
        )
    )
    for seed in (3, 4):
        run = base / f"seed{seed}" / "fixed_expert_101k_online"
        _curve(run / "val50_seeds400_stage43_seed34.csv", curves[seed - 1])
        with (run / "train.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("demo_buffer_size",))
            writer.writeheader()
            writer.writerow({"demo_buffer_size": 9900 if grow else 9253})
    return base


def test_stage43_full_passes_four_seed_heldout_gate(tmp_path: Path):
    curves = [
        [0.70] * 22 + [0.62],
        [0.72] * 22 + [0.64],
        [0.68] * 22 + [0.60],
        [0.74] * 22 + [0.58],
    ]
    result = summarize(_setup(tmp_path, curves))
    assert result["validation_selected_mean_best"] == pytest.approx(0.71)
    assert result["raw111k_fixed_endpoint_mean"] == pytest.approx(0.61)
    assert result["eligible_for_sealed_heldout"]
    assert not result["heldout_opened"]


def test_stage43_full_rejects_weak_fixed_endpoints(tmp_path: Path):
    curves = [
        [0.72] * 22 + [0.50],
        [0.72] * 22 + [0.50],
        [0.72] * 22 + [0.50],
        [0.72] * 22 + [0.44],
    ]
    result = summarize(_setup(tmp_path, curves))
    assert not result["eligible_for_sealed_heldout"]
    assert result["next_decision"].startswith("stop_stage43")


def test_stage43_full_rejects_fresh_seed_buffer_growth(tmp_path: Path):
    curves = [[0.70] * 23 for _ in range(4)]
    result = summarize(_setup(tmp_path, curves, grow=True))
    assert not result["all_demo_buffers_fixed"]
    assert not result["eligible_for_sealed_heldout"]


def test_stage43_full_cli_runs_from_repository_root(tmp_path: Path):
    curves = [[0.70] * 23 for _ in range(4)]
    stage43 = _setup(tmp_path, curves)
    output = tmp_path / "summary.json"
    repository_root = Path(__file__).resolve().parents[2]

    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_cqn_no_bc_stage43_full.py",
            "--stage43-dir",
            str(stage43),
            "--output",
            str(output),
        ],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(output.read_text())
    assert result["eligible_for_sealed_heldout"]
