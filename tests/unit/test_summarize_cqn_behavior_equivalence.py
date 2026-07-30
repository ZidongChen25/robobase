import json
from pathlib import Path

import pytest

from scripts.summarize_cqn_behavior_equivalence import summarize


def _write_eval(path: Path, rewards: list[float]) -> Path:
    payload = {
        "status": "ok",
        "task": "move_plate",
        "episode_results": [
            {
                "seed": 40_000 + index,
                "episode_success": float(reward > 0.0),
                "episode_reward": reward,
                "episode_length": 100.0 + index,
            }
            for index, reward in enumerate(rewards)
        ],
    }
    path.write_text(json.dumps(payload))
    return path


def test_exact_paired_equivalence_passes(tmp_path):
    reference = _write_eval(tmp_path / "reference.json", [1.0, 0.0])
    candidate = _write_eval(tmp_path / "candidate.json", [1.0, 0.0])

    result = summarize(reference, candidate)

    assert result["gate"] == "pass"
    assert result["exact_closed_loop_equivalence"]
    assert all(
        metric["num_mismatched_seeds"] == 0
        for metric in result["metrics"].values()
    )


def test_outcome_mismatch_fails(tmp_path):
    reference = _write_eval(tmp_path / "reference.json", [1.0, 0.0])
    candidate = _write_eval(tmp_path / "candidate.json", [0.0, 0.0])

    result = summarize(reference, candidate)

    assert result["gate"] == "fail"
    assert not result["exact_closed_loop_equivalence"]
    assert result["metrics"]["episode_success"]["num_mismatched_seeds"] == 1


def test_seed_mismatch_is_rejected(tmp_path):
    reference = _write_eval(tmp_path / "reference.json", [1.0, 0.0])
    candidate = _write_eval(tmp_path / "candidate.json", [1.0])

    with pytest.raises(ValueError, match="seed sets differ"):
        summarize(reference, candidate)
