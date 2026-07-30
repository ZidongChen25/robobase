from argparse import Namespace
from pathlib import Path

from scripts.run_cqn_flow_return_readout_gate import (
    build_variants,
    confirmation_gate,
    paired_row,
    select_variant,
    validation_gate,
)


def _args():
    return Namespace(
        snapshot=[
            ("step7000", Path("/tmp/7000.pkl")),
            ("step8000", Path("/tmp/8000.pkl")),
        ],
        baseline_action_flow_samples=4,
        candidate_action_flow_samples=16,
        temperatures=[0.3, 1.0],
    )


def _payload(successes):
    return {
        "episode_success": sum(successes) / len(successes),
        "episode_results": [
            {"seed": 100 + index, "episode_success": success}
            for index, success in enumerate(successes)
        ],
    }


def test_build_variants_covers_native_mean_and_entropic_candidates():
    variants = build_variants(_args())
    assert len(variants) == 8
    assert {variant.kind for variant in variants} == {
        "baseline",
        "candidate",
    }
    assert "step8000_r16_entropic_eta0.3" in {
        variant.label for variant in variants
    }


def test_select_variant_prefers_mean_then_earlier_snapshot_on_ties():
    variants = build_variants(_args())
    payloads = {
        variant.label: _payload([1.0, 0.0]) for variant in variants
    }
    selected = select_variant(
        variants,
        payloads,
        kind="candidate",
    )
    assert selected.label == "step7000_r16_mean"


def test_paired_statistics_and_gates_use_common_episode_seeds():
    variants = build_variants(_args())
    baseline = next(
        variant
        for variant in variants
        if variant.label == "step7000_r4_mean"
    )
    candidate = next(
        variant
        for variant in variants
        if variant.label == "step7000_r16_mean"
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
        bootstrap_seed=7,
    )
    assert row["paired_delta"] == 0.25
    assert row["paired_wins"] == 1
    assert row["paired_losses"] == 0
    assert validation_gate(row, 0.02)[0]
    assert confirmation_gate(row)[0]
