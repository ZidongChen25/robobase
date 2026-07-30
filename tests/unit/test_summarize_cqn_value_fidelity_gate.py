from __future__ import annotations

import copy

import pytest

from scripts.summarize_cqn_value_fidelity_gate import summarize


def _record(
    seed: int,
    dimension: int,
    *,
    pairwise: float,
    spearman: float,
    top1: bool,
    regret: float,
) -> dict:
    return {
        "eval_seed": seed,
        "anchor_step": 30,
        "action_dimension": dimension,
        "realized_return": [0.0, 1.0],
        "num_informative_pairs": 1,
        "pairwise_sign_accuracy": pairwise,
        "spearman": spearman,
        "top1_match": top1,
        "realized_regret": regret,
    }


def _payload() -> dict:
    before = [
        _record(
            10,
            0,
            pairwise=0.0,
            spearman=-1.0,
            top1=False,
            regret=1.0,
        ),
        _record(
            11,
            0,
            pairwise=1.0,
            spearman=0.0,
            top1=False,
            regret=0.5,
        ),
    ]
    after = [
        _record(
            10,
            0,
            pairwise=1.0,
            spearman=1.0,
            top1=True,
            regret=0.0,
        ),
        _record(
            11,
            0,
            pairwise=1.0,
            spearman=0.5,
            top1=True,
            regret=0.25,
        ),
    ]
    return {
        "status": "ok",
        "coverage_only": True,
        "source_snapshot": "before.pkl",
        "comparison_snapshot": "after.pkl",
        "frozen_component_bitwise_equal": True,
        "results": {
            "train_before": {"records": before[:1]},
            "train_after": {"records": after[:1]},
            "heldout_before": {"records": before[1:]},
            "heldout_after": {"records": after[1:]},
        },
    }


def test_combines_partitions_and_passes_directional_gate():
    result = summarize(_payload(), bootstrap_replicates=100, seed=3)

    assert result["before"]["pairwise_sign_accuracy"] == pytest.approx(0.5)
    assert result["after"]["pairwise_sign_accuracy"] == pytest.approx(1.0)
    assert result["delta_after_minus_before"]["mean_realized_regret"] < 0
    assert result["gate"]["passed"]
    assert result["paired_seed_bootstrap"]["num_seeds"] == 2
    assert set(result["per_seed"]) == {"10", "11"}


def test_rejects_changed_counterfactual_outcomes():
    payload = _payload()
    payload = copy.deepcopy(payload)
    payload["results"]["heldout_after"]["records"][0][
        "realized_return"
    ] = [1.0, 0.0]

    with pytest.raises(ValueError, match="counterfactual outcomes changed"):
        summarize(payload, bootstrap_replicates=0, seed=3)


def test_rejects_non_coverage_comparison():
    payload = _payload()
    payload["coverage_only"] = False

    with pytest.raises(ValueError, match="coverage-only"):
        summarize(payload, bootstrap_replicates=0, seed=3)
