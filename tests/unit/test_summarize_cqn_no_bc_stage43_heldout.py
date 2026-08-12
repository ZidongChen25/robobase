import csv
import json
from pathlib import Path

import pytest

from scripts.summarize_cqn_no_bc_stage43_heldout import (
    OFFICIAL_RUN_NAMES,
    summarize,
)


def _endpoint(path: Path, step: int, value: float) -> None:
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
        writer.writerow(
            {
                "env_steps": step,
                "episode_success": value,
                "eval_episodes": 200,
                "eval_seed_start": 800,
            }
        )


def _setup(tmp_path: Path, nobc: list[float]) -> tuple[Path, Path]:
    stage43 = tmp_path / "stage43"
    official = tmp_path / "official"
    stage43.mkdir()
    (stage43 / "stage43_full_summary.json").write_text(
        json.dumps({"eligible_for_sealed_heldout": True})
    )
    reference = [0.62, 0.605, 0.62, 0.74]
    for seed, name in enumerate(OFFICIAL_RUN_NAMES, start=1):
        _endpoint(
            stage43
            / f"seed{seed}"
            / "fixed_expert_101k_online"
            / "heldout200_seeds800_stage43.csv",
            111000,
            nobc[seed - 1],
        )
        _endpoint(official / name / "ep200_seeds800.csv", 101000, reference[seed - 1])
    return stage43, official


def test_stage43_heldout_meets_empirical_parity(tmp_path: Path):
    result = summarize(*_setup(tmp_path, [0.65, 0.64, 0.65, 0.65]))
    assert result["official_fixed_endpoint_mean"] == pytest.approx(0.64625)
    assert result["no_bc_fixed_endpoint_mean"] == pytest.approx(0.6475)
    assert result["empirical_parity_or_better"]
    assert result["goal_criterion_met"]


def test_stage43_heldout_rejects_mean_below_official(tmp_path: Path):
    result = summarize(*_setup(tmp_path, [0.62, 0.62, 0.62, 0.62]))
    assert not result["empirical_parity_or_better"]
    assert result["next_decision"] == "continue_reward_q_research"


def test_stage43_heldout_rejects_wrong_episode_count(tmp_path: Path):
    stage43, official = _setup(tmp_path, [0.65] * 4)
    path = (
        stage43
        / "seed1"
        / "fixed_expert_101k_online"
        / "heldout200_seeds800_stage43.csv"
    )
    text = path.read_text().replace(",200,800", ",50,800")
    path.write_text(text)
    with pytest.raises(ValueError, match="not 200 episodes"):
        summarize(stage43, official)
