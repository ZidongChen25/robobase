from pathlib import Path

from scripts.summarize_cqn_paired_eval import summarize


def _payload(successes):
    return {
        "status": "ok",
        "episode_results": [
            {"seed": 500 + index, "episode_success": success}
            for index, success in enumerate(successes)
        ],
    }


def test_paired_superiority_gate_passes_positive_matched_result():
    result = summarize(
        _payload([0.0, 1.0, 0.0, 1.0]),
        _payload([1.0, 1.0, 0.0, 1.0]),
        baseline_path=Path("/tmp/baseline.json"),
        candidate_path=Path("/tmp/candidate.json"),
        bootstrap_samples=2_000,
        bootstrap_seed=5,
        min_delta=0.0,
        min_ci_lower=0.0,
    )
    assert result["paired_delta"] == 0.25
    assert result["paired_wins"] == 1
    assert result["paired_losses"] == 0
    assert result["gate"] == "pass"


def test_paired_superiority_gate_rejects_negative_direction():
    result = summarize(
        _payload([1.0, 1.0, 0.0, 1.0]),
        _payload([0.0, 1.0, 0.0, 1.0]),
        baseline_path=Path("/tmp/baseline.json"),
        candidate_path=Path("/tmp/candidate.json"),
        bootstrap_samples=2_000,
        bootstrap_seed=5,
        min_delta=0.0,
        min_ci_lower=0.0,
    )
    assert result["gate"] == "fail"
