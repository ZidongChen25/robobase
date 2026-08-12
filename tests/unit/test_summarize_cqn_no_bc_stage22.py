from scripts.summarize_cqn_no_bc_stage22 import _development_decision


def _arm(best_step, at_7500, at_10000):
    return {
        "best_step": best_step,
        "curve": {
            "2500": 0.0,
            "5000": 0.0,
            "7500": at_7500,
            "10000": at_10000,
        },
    }


def test_stage22_stable_relative_gain_requests_seed3_confirmation():
    treatments = {
        "seed1": _arm(10000, 0.50, 0.56),
        "seed2": _arm(7500, 0.52, 0.48),
    }
    decision, flags = _development_decision(
        treatments,
        {"seed1": 0.08, "seed2": 0.04},
    )
    assert decision == "run_seed3_and_independent_dev_confirmation"
    assert flags["immediate_replication"]


def test_stage22_near_control_rising_boundary_requests_20k():
    treatments = {
        "seed1": _arm(10000, 0.42, 0.46),
        "seed2": _arm(7500, 0.46, 0.44),
    }
    decision, flags = _development_decision(
        treatments,
        {"seed1": -0.02, "seed2": -0.02},
    )
    assert decision == "continue_treatments_to_20k"
    assert flags["rising_at_10k"]
    assert flags["scale_continuation"]


def test_stage22_bad_short_result_is_not_named_full_budget_failure():
    treatments = {
        "seed1": _arm(7500, 0.30, 0.24),
        "seed2": _arm(7500, 0.28, 0.20),
    }
    decision, flags = _development_decision(
        treatments,
        {"seed1": -0.18, "seed2": -0.20},
    )
    assert decision == "stop_development_candidate_without_full_budget_claim"
    assert not any(flags.values())
