from scripts.summarize_cqn_route_a import summarize_route_a


def test_route_a_separates_safe_audit_from_action_facing_claim():
    structured = {
        "gate": "pass",
        "checks": {"one": True, "two": True},
        "metrics": {"model": {"pairwise_accuracy": 0.63}},
    }
    policy = {
        "bc": {"episode_success": 0.56},
        "gate_passed": False,
        "variants": {
            "medium": {
                "episode_success": 0.56,
                "success_delta_vs_bc": 0.0,
                "paired_wins": 0,
                "paired_losses": 0,
                "total_applied_overrides": 0,
            }
        },
    }
    direct = {"gate": "fail"}
    blend = {"gate": "fail"}

    payload = summarize_route_a(structured, policy, direct, blend)

    assert payload["safe_audit_gate"] == "pass"
    assert payload["action_facing_gate"] == "fail"
    assert payload["task_noninferiority"]["exact_fallback_variants"] == [
        "medium"
    ]


def test_route_a_requires_exact_matched_clean_fallback():
    structured = {"gate": "pass", "checks": {"one": True}}
    policy = {
        "bc": {"episode_success": 0.56},
        "gate_passed": False,
        "variants": {
            "changed": {
                "episode_success": 0.58,
                "success_delta_vs_bc": 0.02,
                "paired_wins": 2,
                "paired_losses": 1,
                "total_applied_overrides": 0,
            }
        },
    }

    payload = summarize_route_a(
        structured,
        policy,
        {"gate": "fail"},
        {"gate": "fail"},
    )

    assert payload["safe_audit_gate"] == "fail"
