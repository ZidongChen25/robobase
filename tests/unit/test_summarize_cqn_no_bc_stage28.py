from scripts.summarize_cqn_no_bc_stage28 import _decision


def test_stage28_replication_and_scale_runs_both_confirmations():
    decision, flags = _decision(
        {"seed1": 0.06, "seed2": 0.04, "seed3": 0.05},
        stage27_scale_continuation=True,
    )
    assert decision == "run_independent100_and_extend_seeds1_2_3_to50k"
    assert flags["replication_pass"]
    assert flags["nonnegative_seed_count"] == 3


def test_stage28_replication_alone_earns_50k_budget():
    decision, flags = _decision(
        {"seed1": 0.06, "seed2": 0.04, "seed3": 0.05},
        stage27_scale_continuation=False,
    )
    assert decision == "run_independent100_and_extend_seeds1_2_3_to50k"
    assert flags["replication_pass"]


def test_stage28_scale_can_continue_without_replication_claim():
    decision, flags = _decision(
        {"seed1": 0.06, "seed2": 0.04, "seed3": -0.10},
        stage27_scale_continuation=True,
    )
    assert decision == "extend_seeds1_2_to50k_without_replication_claim"
    assert not flags["replication_pass"]


def test_stage28_failed_replication_is_not_full_budget_failure():
    decision, flags = _decision(
        {"seed1": 0.06, "seed2": 0.04, "seed3": -0.10},
        stage27_scale_continuation=False,
    )
    assert decision == (
        "stop_reward_scale_variant_without_full_budget_claim"
    )
    assert not flags["replication_pass"]
