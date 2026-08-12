from scripts.summarize_cqn_no_bc_stage30 import ONLINE_STEPS
from scripts.summarize_cqn_no_bc_stage31 import _decision


def _arms(seed1_best=0.24, seed2_best=0.16):
    arms = {}
    for seed, best in (("seed1", seed1_best), ("seed2", seed2_best)):
        curve = {str(step): max(0.0, best - 0.04) for step in ONLINE_STEPS}
        curve["17500"] = max(0.0, best - 0.02)
        curve["20000"] = best
        arms[seed] = {"best_success": best, "online_curve": curve}
    return arms


def test_stage31_direct_head_mechanism_pass_requires_seed3():
    decision, flags = _decision(
        {"seed1": 0.24, "seed2": 0.16},
        _arms(),
    )

    assert decision == "run_direct_head_seed3_then_update_matched_confirmation"
    assert flags["mechanism_pass"]


def test_stage31_rising_boundary_extends_before_rejection():
    arms = _arms(seed1_best=0.20, seed2_best=0.0)
    decision, flags = _decision(
        {"seed1": 0.08, "seed2": 0.02},
        arms,
    )

    assert decision == "extend_direct_head_to50k_before_rejection"
    assert flags["scale_continuation"]
