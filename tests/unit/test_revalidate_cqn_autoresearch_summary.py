import json
import sys

from scripts.revalidate_cqn_autoresearch_summary import main
from scripts.summarize_cqn_autoresearch_routes import summarize_multi


def _task(path):
    return {
        "_path": str(path),
        "status": "ok",
        "gate": "pass",
        "num_training_seeds": 3,
        "num_eval_seeds": 200,
        "mean_paired_delta": 0.05,
        "crossed_bootstrap_ci95": [0.01, 0.09],
        "thresholds": {
            "min_mean_delta": 0.0,
            "min_ci_lower": 0.0,
        },
        "gate_checks": {
            "mean_delta_strictly_above_threshold": True,
            "crossed_ci_lower_at_least_threshold": True,
            "aggregate_wins_above_losses": True,
            "positive_training_seed_majority": True,
        },
    }


def _causal(path, *, dimension_selection):
    unbiased = dimension_selection == "round_robin"
    return {
        "_path": str(path),
        "status": "ok",
        "gate": "pass",
        "policy_value_beta": None,
        "intervention_horizon": 1,
        "anti_cheat_proxies_required": True,
        "num_training_seeds": 3,
        "num_eval_seeds": 32,
        "dimension_selection": dimension_selection,
        "gate_checks": {
            "dimension_selection_is_value_independent": unbiased,
            "informative_dimension_coverage_per_training_seed": unbiased,
            "anti_cheat_proxy_coverage_per_training_seed": True,
            "q_pairwise_above_policy_prior_proxy_ci": True,
            "q_pairwise_above_policy_path_proxy_ci": True,
            "q_pairwise_above_action_nearness_proxy_ci": True,
        },
        "aggregate_pairwise_sign_accuracy": 0.61,
    }


def test_revalidation_replaces_discovery_causal_evidence(
    monkeypatch, tmp_path
):
    base = summarize_multi(
        [
            (
                "route_a_candidate",
                _task(tmp_path / "a_task.json"),
                _causal(
                    tmp_path / "a_discovery.json",
                    dimension_selection="q_span",
                ),
            )
        ],
        [
            (
                "route_b_candidate",
                _task(tmp_path / "b_task.json"),
                _causal(
                    tmp_path / "b_unbiased.json",
                    dimension_selection="round_robin",
                ),
            )
        ],
    )
    assert base["route_a"]["overall_gate"] == "fail"
    assert base["route_b"]["overall_gate"] == "pass"
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base))
    override_path = tmp_path / "a_unbiased.json"
    override = _causal(
        override_path,
        dimension_selection="round_robin",
    )
    override.pop("_path")
    override_path.write_text(json.dumps(override))
    output = tmp_path / "strict.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "revalidate_cqn_autoresearch_summary.py",
            "--base-summary",
            str(base_path),
            "--causal-override",
            f"route_a_candidate={override_path}",
            "--output",
            str(output),
        ],
    )

    assert main() == 0
    payload = json.loads(output.read_text())
    assert payload["route_a"]["overall_gate"] == "pass"
    assert payload["route_b"]["overall_gate"] == "pass"
    assert payload["research_goal_gate"] == "pass"
    assert payload["strict_revalidation"]["selection_use_forbidden"]
