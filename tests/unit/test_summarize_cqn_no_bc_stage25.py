import pytest

from scripts.summarize_cqn_no_bc_stage25 import _development_decision


def test_stage25_gate_passes_with_mean_five_and_two_nonnegative():
    decision, flags = _development_decision(
        {"seed1": 0.08, "seed2": 0.10, "seed3": -0.03}
    )
    assert decision == "run_independent100_confirmation"
    assert flags["development_gate_pass"]
    assert flags["nonnegative_seed_count"] == 2
    assert flags["mean_improvement"] == pytest.approx(0.05)


def test_stage25_gate_fails_without_reproducible_signs():
    decision, flags = _development_decision(
        {"seed1": 0.20, "seed2": -0.01, "seed3": -0.01}
    )
    assert decision == (
        "stop_exact_candidate_only_variant_without_full_budget_claim"
    )
    assert not flags["development_gate_pass"]
