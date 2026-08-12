from scripts.summarize_cqn_no_bc_stage26 import _decision


def test_stage26_stable_relative_gain_requests_seed3():
    decision, flags = _decision({"seed1": 0.06, "seed2": 0.04})
    assert decision == "run_seed3_then_independent100_if_replicated"
    assert flags["strong_pass"]
    assert flags["both_nonnegative"]


def test_stage26_mixed_positive_gain_requests_seed3_only():
    decision, flags = _decision({"seed1": 0.10, "seed2": -0.02})
    assert decision == "run_seed3_to_resolve_mixed_signs"
    assert not flags["strong_pass"]
    assert flags["mixed_positive"]


def test_stage26_bad_20k_result_is_not_named_full_budget_failure():
    decision, flags = _decision({"seed1": -0.02, "seed2": -0.04})
    assert decision == (
        "stop_exact_force_then_candidate_variant_without_full_budget_claim"
    )
    assert not flags["strong_pass"]
    assert not flags["mixed_positive"]


def test_stage26_scale_boundary_can_extend_without_relative_quality_pass():
    decision, flags = _decision(
        {"seed1": -0.02, "seed2": -0.04},
        scale_continuation=True,
    )
    assert decision == "extend_seeds1_2_to50k_before_rejection"
    assert not flags["strong_pass"]
    assert not flags["mixed_positive"]
    assert flags["scale_continuation"]
