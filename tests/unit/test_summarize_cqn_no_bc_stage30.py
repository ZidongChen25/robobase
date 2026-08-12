from scripts.summarize_cqn_no_bc_stage30 import (
    OFFLINE_UPDATES,
    ONLINE_STEPS,
    TREATMENT_RAW_STEPS,
    _decision,
)


def _treatments(seed1_endpoint=0.54, seed2_endpoint=0.48):
    return {
        "seed1": {
            "online_curve": {
                str(step): (
                    seed1_endpoint if step == 20_000 else seed1_endpoint - 0.02
                )
                for step in ONLINE_STEPS
            }
        },
        "seed2": {
            "online_curve": {
                str(step): (
                    seed2_endpoint if step == 20_000 else seed2_endpoint - 0.02
                )
                for step in ONLINE_STEPS
            }
        },
    }


def test_stage30_stable_gain_requires_seed3_and_compute_matched_control():
    decision, flags = _decision(
        {"seed1": 0.06, "seed2": 0.04},
        _treatments(),
    )

    assert decision == (
        "run_seed3_then_update_count_matched_control_and_independent100"
    )
    assert flags["strong_pass"]


def test_stage30_good_boundary_can_extend_before_rejection():
    decision, flags = _decision(
        {"seed1": -0.02, "seed2": -0.04},
        _treatments(),
    )

    assert decision == "extend_offline_then_online_to50k_before_rejection"
    assert flags["scale_continuation"]


def test_stage30_raw_snapshot_clock_maps_offline_plus_online_steps():
    assert TREATMENT_RAW_STEPS[0] == OFFLINE_UPDATES
    assert TREATMENT_RAW_STEPS[1:] == tuple(
        OFFLINE_UPDATES + step for step in ONLINE_STEPS
    )
    assert TREATMENT_RAW_STEPS[-1] == 30_000
