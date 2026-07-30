from __future__ import annotations

from scripts.summarize_cqn_lcb_calibration import summarize


def _payload(successes, overrides=0):
    return {
        "status": "ok",
        "thresholds": {"margin": 0.1},
        "episode_results": [
            {
                "seed": 100 + index,
                "episode_success": success,
                "episode_reward": success,
                "inference_count": 10,
                "applied_override_count": overrides,
            }
            for index, success in enumerate(successes)
        ],
    }


def test_selects_highest_success_gate_pass_then_lower_override():
    bc = _payload([0, 1, 0, 1])
    variants = {
        "low_override": _payload([1, 1, 0, 1], overrides=1),
        "high_override": _payload([1, 1, 0, 1], overrides=2),
        "worse": _payload([0, 0, 0, 1], overrides=1),
    }

    result = summarize(
        bc,
        variants,
        bootstrap_replicates=100,
        seed=4,
    )

    assert result["selected_variant"] == "low_override"
    assert result["gate_passed"]
    assert result["variants"]["low_override"]["paired_wins"] == 1
    assert not result["variants"]["worse"]["gate_passed"]


def test_falls_back_to_bc_when_no_variant_passes():
    bc = _payload([1, 1, 0, 1])
    variants = {"worse": _payload([0, 1, 0, 1], overrides=1)}

    result = summarize(
        bc,
        variants,
        bootstrap_replicates=0,
        seed=4,
    )

    assert result["selected_variant"] is None
    assert not result["gate_passed"]


def test_confirmation_requires_positive_override_episode_effect():
    bc = _payload([0, 1, 0, 1])
    improving = _payload([1, 1, 0, 1], overrides=1)
    neutral = _payload([0, 1, 0, 1], overrides=1)

    passed = summarize(
        bc,
        {"selected": improving},
        bootstrap_replicates=100,
        seed=4,
        stage="confirmation",
    )
    failed = summarize(
        bc,
        {"selected": neutral},
        bootstrap_replicates=100,
        seed=4,
        stage="confirmation",
    )

    assert passed["stage"] == "confirmation"
    assert passed["gate_passed"]
    assert (
        passed["variants"]["selected"]["override_episode_success_delta"]
        > 0.0
    )
    assert not failed["gate_passed"]
    assert not failed["variants"]["selected"]["gate_checks"][
        "override_episode_success_delta_positive"
    ]
