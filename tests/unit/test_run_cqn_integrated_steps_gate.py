from scripts.run_cqn_integrated_steps_gate import (
    confirmation_gate,
    paired_stats,
    validation_gate,
)


def _payload(successes):
    return {
        "status": "ok",
        "episode_success": sum(successes) / len(successes),
        "episode_results": [
            {"seed": 300 + index, "episode_success": success}
            for index, success in enumerate(successes)
        ],
    }


def test_paired_stats_can_use_a_pilot_seed_prefix():
    candidate = _payload([1.0, 1.0, 0.0, 1.0])
    reference = _payload([0.0, 1.0, 0.0, 0.0])
    row = paired_stats(
        candidate,
        reference,
        limit=2,
        bootstrap_samples=2_000,
        bootstrap_seed=11,
    )
    assert row["episodes"] == 2
    assert row["paired_delta"] == 0.5
    assert row["paired_wins"] == 1


def test_validation_requires_improvement_over_both_references():
    versus_bc = {
        "paired_delta": 0.04,
        "paired_wins": 4,
        "paired_losses": 2,
    }
    versus_distill = {
        "paired_delta": 0.06,
        "paired_wins": 5,
        "paired_losses": 2,
    }
    assert validation_gate(versus_bc, versus_distill, 0.02)[0]
    assert not validation_gate(
        {**versus_bc, "paired_delta": 0.01},
        versus_distill,
        0.02,
    )[0]


def test_confirmation_requires_positive_dual_reference_direction():
    versus_bc = {
        "paired_delta": 0.03,
        "paired_delta_ci95": [-0.04, 0.1],
        "paired_wins": 6,
        "paired_losses": 3,
    }
    versus_distill = {
        "paired_delta": 0.08,
        "paired_delta_ci95": [-0.02, 0.15],
        "paired_wins": 8,
        "paired_losses": 2,
    }
    assert confirmation_gate(versus_bc, versus_distill)[0]
    assert not confirmation_gate(
        {**versus_bc, "paired_delta_ci95": [-0.06, 0.1]},
        versus_distill,
    )[0]
