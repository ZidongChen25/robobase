import json
import sys

from scripts.summarize_cqn_autoresearch_routes import (
    _base_candidates,
    main,
    summarize,
    summarize_multi,
)


def _task(gate: str, delta: float, lower: float):
    return {
        "_path": "/tmp/task.json",
        "status": "ok",
        "gate": gate,
        "num_training_seeds": 3,
        "num_eval_seeds": 200,
        "mean_baseline_success": 0.60,
        "mean_candidate_success": 0.60 + delta,
        "mean_paired_delta": delta,
        "crossed_bootstrap_ci95": [lower, delta + 0.05],
        "thresholds": {
            "min_mean_delta": 0.0,
            "min_ci_lower": 0.0,
        },
        "gate_checks": {
            "mean_delta_strictly_above_threshold": gate == "pass",
            "crossed_ci_lower_at_least_threshold": gate == "pass",
            "aggregate_wins_above_losses": gate == "pass",
            "positive_training_seed_majority": gate == "pass",
        },
    }


def _causal(gate: str):
    return {
        "_path": "/tmp/causal.json",
        "status": "ok",
        "gate": gate,
        "aggregate_pairwise_sign_accuracy": 0.61,
        "policy_value_beta": None,
        "intervention_horizon": 1,
        "anti_cheat_proxies_required": True,
        "num_training_seeds": 3,
        "num_eval_seeds": 32,
        "dimension_selection": "round_robin",
        "gate_checks": {
            "dimension_selection_is_value_independent": True,
            "informative_dimension_coverage_per_training_seed": True,
            "anti_cheat_proxy_coverage_per_training_seed": True,
            "q_pairwise_above_policy_prior_proxy_ci": True,
            "q_pairwise_above_policy_path_proxy_ci": True,
            "q_pairwise_above_action_nearness_proxy_ci": True,
        },
    }


def test_b_candidate_requires_task_and_causal_pass():
    payload = summarize(
        _task("pass", 0.02, 0.0),
        _causal("pass"),
        [
            ("task_only", _task("pass", 0.08, 0.02), _causal("fail")),
            ("causal_only", _task("fail", -0.01, -0.06), _causal("pass")),
        ],
    )

    assert payload["route_a"]["overall_gate"] == "pass"
    assert payload["route_b"]["overall_gate"] == "fail"
    assert payload["research_goal_gate"] == "fail"


def test_b_selection_prefers_stronger_ci_lower_bound():
    payload = summarize(
        _task("pass", 0.02, 0.0),
        _causal("pass"),
        [
            ("large_mean", _task("pass", 0.10, 0.001), _causal("pass")),
            ("robust", _task("pass", 0.06, 0.02), _causal("pass")),
        ],
    )

    assert payload["route_b"]["selected_candidate"] == "robust"
    assert payload["research_goal_gate"] == "pass"


def test_a_task_and_causal_failures_remain_separate():
    payload = summarize(
        _task("fail", -0.01, -0.05),
        _causal("pass"),
        [("flow", _task("pass", 0.04, 0.01), _causal("pass"))],
    )

    assert payload["route_a"]["unmet_gates"] == [
        "no non-Flow candidate passed both task and causal gates"
    ]
    assert payload["route_b"]["overall_gate"] == "pass"
    assert payload["research_goal_gate"] == "fail"


def test_q_span_selected_causal_probe_cannot_finish_research_goal():
    causal = _causal("pass")
    causal["dimension_selection"] = "q_span"
    causal["gate_checks"] = {}

    payload = summarize(
        _task("pass", 0.02, 0.0),
        causal,
        [("flow", _task("pass", 0.04, 0.01), causal)],
    )

    assert payload["route_a"]["overall_gate"] == "fail"
    assert payload["route_b"]["overall_gate"] == "fail"
    assert payload["research_goal_gate"] == "fail"


def test_numeric_beta_causal_probe_cannot_finish_research_goal():
    causal = _causal("pass")
    causal["policy_value_beta"] = 1.0

    payload = summarize(
        _task("pass", 0.02, 0.0),
        causal,
        [("flow", _task("pass", 0.04, 0.01), causal)],
    )

    assert payload["route_a"]["overall_gate"] == "fail"
    assert payload["route_b"]["overall_gate"] == "fail"
    assert not payload["route_a"]["candidates"][0]["checks"][
        "independent_bc_causal_protocol"
    ]


def test_route_b_requires_strictly_positive_task_ci_lower():
    payload = summarize(
        _task("pass", 0.02, 0.0),
        _causal("pass"),
        [("flow", _task("pass", 0.04, 0.0), _causal("pass"))],
    )

    assert payload["route_a"]["overall_gate"] == "pass"
    assert payload["route_b"]["overall_gate"] == "fail"
    assert not payload["route_b"]["candidates"][0]["checks"][
        "sealed_multiseed_task_protocol"
    ]


def test_two_training_seed_task_cannot_finish_research_goal():
    task = _task("pass", 0.04, 0.01)
    task["num_training_seeds"] = 2

    payload = summarize(
        task,
        _causal("pass"),
        [("flow", task, _causal("pass"))],
    )

    assert payload["research_goal_gate"] == "fail"


def test_route_a_can_select_policy_value_td_variant():
    payload = summarize_multi(
        [
            (
                "replay_next",
                _task("fail", -0.01, -0.04),
                _causal("pass"),
            ),
            (
                "policy_value_td",
                _task("pass", 0.04, 0.01),
                _causal("pass"),
            ),
        ],
        [("flow", _task("pass", 0.05, 0.01), _causal("pass"))],
    )

    assert payload["route_a"]["selected_candidate"] == "policy_value_td"
    assert payload["research_goal_gate"] == "pass"


def test_base_summary_candidates_can_be_restored_and_extended():
    base = summarize_multi(
        [("replay_next", _task("fail", -0.01, -0.04), _causal("pass"))],
        [("legacy_flow", _task("pass", 0.05, 0.01), _causal("pass"))],
    )

    restored_a = _base_candidates(base, "route_a")
    restored_b = _base_candidates(base, "route_b")
    payload = summarize_multi(
        [
            *restored_a,
            ("bc_policy", _task("pass", 0.03, 0.0), _causal("pass")),
        ],
        restored_b,
    )

    assert payload["route_a"]["selected_candidate"] == "bc_policy"
    assert payload["route_b"]["selected_candidate"] == "legacy_flow"
    assert payload["research_goal_gate"] == "pass"


def test_cli_appends_candidate_to_base_summary(monkeypatch, tmp_path):
    base = summarize_multi(
        [("replay_next", _task("fail", -0.01, -0.04), _causal("pass"))],
        [("legacy_flow", _task("pass", 0.05, 0.01), _causal("pass"))],
    )
    base_path = tmp_path / "base.json"
    task_path = tmp_path / "task.json"
    causal_path = tmp_path / "causal.json"
    output_path = tmp_path / "output.json"
    base_path.write_text(json.dumps(base))
    task = _task("pass", 0.03, 0.0)
    causal = _causal("pass")
    task.pop("_path")
    causal.pop("_path")
    task_path.write_text(json.dumps(task))
    causal_path.write_text(json.dumps(causal))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_cqn_autoresearch_routes.py",
            "--base-summary",
            str(base_path),
            "--a-candidate",
            f"bc_policy={task_path},{causal_path}",
            "--output",
            str(output_path),
        ],
    )

    assert main() == 0
    payload = json.loads(output_path.read_text())
    assert payload["route_a"]["selected_candidate"] == "bc_policy"
    assert payload["route_b"]["selected_candidate"] == "legacy_flow"
    assert payload["research_goal_gate"] == "pass"
