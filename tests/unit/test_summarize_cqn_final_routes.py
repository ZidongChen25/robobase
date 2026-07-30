import pytest

from scripts.summarize_cqn_final_routes import summarize_final_routes


def _task(gate):
    return {
        "_path": "/tmp/task.json",
        "status": "ok",
        "gate": gate,
        "mean_baseline_success": 0.65,
        "mean_candidate_success": 0.70,
        "mean_paired_delta": 0.05,
        "crossed_bootstrap_ci95": [0.01, 0.09],
        "gate_checks": {"strict": gate == "pass"},
    }


def _causal(gate):
    return {
        "_path": "/tmp/causal.json",
        "status": "ok",
        "gate": gate,
        "aggregate_pairwise_sign_accuracy": 0.62,
        "aggregate_pairwise_sign_accuracy_ci": [0.53, 0.70],
        "aggregate_mean_spearman": 0.20,
        "aggregate_mean_spearman_ci": [0.04, 0.34],
        "gate_checks": {"strict": gate == "pass"},
    }


def test_final_routes_pass_only_when_task_and_causal_gates_pass():
    payload = summarize_final_routes(
        _task("pass"),
        _causal("pass"),
        _task("pass"),
        _causal("pass"),
        b_source="stage78_fidelity",
    )

    assert payload["research_goal_gate"] == "pass"
    assert payload["route_a"]["overall_gate"] == "pass"
    assert payload["route_b"]["overall_gate"] == "pass"
    assert "promote" in payload["route_b"]["recommendation"]


def test_final_routes_separate_task_failure_from_causal_success():
    payload = summarize_final_routes(
        _task("fail"),
        _causal("pass"),
        None,
        None,
        b_source="none",
    )

    assert payload["research_goal_gate"] == "fail"
    assert payload["route_a"]["unmet_gates"] == ["task_requirement"]
    assert payload["route_b"]["unmet_gates"] == [
        "task_requirement",
        "causal_value_requirement",
    ]
    assert "audit-only" in payload["route_a"]["recommendation"]


def test_task_qualified_candidate_requires_causal_artifact():
    with pytest.raises(ValueError, match="without causal audit"):
        summarize_final_routes(
            _task("pass"),
            _causal("pass"),
            _task("pass"),
            None,
            b_source="stage64_distill",
        )
