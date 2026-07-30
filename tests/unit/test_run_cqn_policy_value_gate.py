from scripts.run_cqn_policy_value_gate import (
    confirmation_gate,
    select_candidate,
    validation_gate,
)


def _row(beta, success, delta, wins, losses, lower=-0.04):
    return {
        "policy_value_beta": beta,
        "success": success,
        "paired_delta_vs_bc": delta,
        "paired_delta_ci95": [lower, 0.2],
        "paired_wins": wins,
        "paired_losses": losses,
    }


def test_select_candidate_prefers_success_then_conservative_beta():
    summary = {
        "results": [
            _row(None, 0.6, 0.0, 0, 0),
            _row(0.1, 0.7, 0.1, 5, 0),
            _row(1.0, 0.7, 0.1, 5, 0),
            _row(3.0, 0.65, 0.05, 3, 1),
        ]
    }
    assert select_candidate(summary)["policy_value_beta"] == 1.0


def test_validation_gate_requires_effect_size_and_positive_pairs():
    assert validation_gate(_row(0.3, 0.7, 0.04, 4, 2), 0.02)[0]
    assert not validation_gate(_row(0.3, 0.7, 0.01, 4, 2), 0.02)[0]
    assert not validation_gate(_row(0.3, 0.7, 0.04, 2, 2), 0.02)[0]


def test_confirmation_gate_is_positive_and_five_pp_noninferior():
    assert confirmation_gate(_row(0.3, 0.7, 0.04, 6, 2, -0.04))[0]
    assert not confirmation_gate(_row(0.3, 0.7, 0.00, 2, 2, -0.04))[0]
    assert not confirmation_gate(_row(0.3, 0.7, 0.04, 6, 2, -0.06))[0]
