from scripts.summarize_cqn_no_bc_stage32_seed3 import confirmation_decision


def test_seed3_confirmation_cannot_override_failed_primary_gate():
    decision, flags = confirmation_decision(
        primary_mechanism_pass=False,
        direct_success=0.0,
        twin_success=0.8,
    )

    assert decision == "supplemental_only_primary_gate_failed"
    assert not flags["confirmation_pass"]


def test_seed3_confirmation_requires_noninferiority_and_task_signal():
    low_decision, low_flags = confirmation_decision(
        primary_mechanism_pass=True,
        direct_success=0.0,
        twin_success=0.18,
    )
    inferior_decision, inferior_flags = confirmation_decision(
        primary_mechanism_pass=True,
        direct_success=0.30,
        twin_success=0.20,
    )

    assert low_decision == "pessimistic_twin_seed3_confirmation_failed"
    assert not low_flags["confirmation_pass"]
    assert inferior_decision == "pessimistic_twin_seed3_confirmation_failed"
    assert not inferior_flags["confirmation_pass"]


def test_seed3_confirmation_advances_only_after_both_gates_pass():
    decision, flags = confirmation_decision(
        primary_mechanism_pass=True,
        direct_success=0.12,
        twin_success=0.24,
    )

    assert decision == "advance_pessimistic_twin_to_full_101k_protocol"
    assert flags["confirmation_pass"]
    assert flags["seed3_noninferior_to_direct"]
    assert flags["seed3_twin_at_least_20pct"]
