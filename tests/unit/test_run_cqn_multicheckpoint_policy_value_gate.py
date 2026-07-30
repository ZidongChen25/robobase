from argparse import Namespace
from pathlib import Path

from scripts.run_cqn_multicheckpoint_policy_value_gate import (
    build_variants,
    confirmation_gate,
    paired_row,
    select_variant,
    validation_gate,
)


def _args():
    return Namespace(
        snapshot=[
            ("step2500", Path("/tmp/2500.pkl")),
            ("step10000", Path("/tmp/10000.pkl")),
        ],
        betas=[0.0, 0.3, 1.0],
    )


def _payload(successes):
    return {
        "episode_success": sum(successes) / len(successes),
        "episode_results": [
            {"seed": 200 + index, "episode_success": success}
            for index, success in enumerate(successes)
        ],
    }


def test_build_variants_crosses_checkpoints_and_betas():
    variants = build_variants(_args())
    assert len(variants) == 8
    assert "step2500_bc" in {variant.label for variant in variants}
    assert "step10000_beta_0p3" in {
        variant.label for variant in variants
    }


def test_selection_is_independent_and_uses_conservative_ties():
    variants = build_variants(_args())
    payloads = {
        variant.label: _payload([1.0, 0.0]) for variant in variants
    }
    baseline = select_variant(
        variants,
        payloads,
        kind="baseline",
    )
    candidate = select_variant(
        variants,
        payloads,
        kind="candidate",
    )
    assert baseline.label == "step2500_bc"
    assert candidate.label == "step2500_beta_1"


def test_multicheckpoint_paired_gate_uses_selected_common_seeds():
    variants = build_variants(_args())
    baseline = next(
        variant for variant in variants if variant.label == "step2500_bc"
    )
    candidate = next(
        variant
        for variant in variants
        if variant.label == "step10000_beta_0p3"
    )
    payloads = {
        baseline.label: _payload([0.0, 1.0, 0.0, 1.0]),
        candidate.label: _payload([1.0, 1.0, 0.0, 1.0]),
    }
    row = paired_row(
        candidate,
        baseline,
        payloads,
        bootstrap_samples=2_000,
        bootstrap_seed=9,
    )
    assert row["paired_delta"] == 0.25
    assert row["paired_wins"] == 1
    assert row["paired_losses"] == 0
    assert validation_gate(row, 0.02)[0]
    assert confirmation_gate(row)[0]
