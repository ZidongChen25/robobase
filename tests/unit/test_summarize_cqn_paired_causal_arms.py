import json
from pathlib import Path

import pytest

from scripts.summarize_cqn_paired_causal_arms import (
    ArmPair,
    _arm_pair,
    summarize,
)


def _write_probe(
    path: Path,
    *,
    q_correct: bool,
    return_offset: float = 0.0,
) -> None:
    eval_seeds = list(range(214_000, 214_008))
    realised = [-2.0, -1.0, 0.0, 1.0, 2.0]
    predicted = realised if q_correct else list(reversed(realised))
    proxy = list(reversed(realised))
    records = []
    for index, seed in enumerate(eval_seeds):
        outcomes = []
        for bin_index, (q_value, return_value) in enumerate(
            zip(predicted, realised, strict=True)
        ):
            outcomes.append(
                {
                    "bin": bin_index,
                    "predicted_q": q_value,
                    "discounted_return": return_value + return_offset,
                    "rollout_length": 20 + bin_index,
                    "success": return_value > 0,
                    "terminated": False,
                    "truncated": True,
                    "raw_forced_action": float(bin_index),
                    "effective_forced_action": float(bin_index),
                    "intervention_delta": 0.1 * bin_index,
                    "intervention_horizon": 1,
                }
            )
        correct = 1.0 if q_correct else 0.0
        records.append(
            {
                "eval_seed": seed,
                "anchor_step": 30,
                "action_dimension": index % 16,
                "realized_return_span": 4.0,
                "num_informative_pairs": 10,
                "pairwise_sign_accuracy": correct,
                "spearman": 1.0 if q_correct else -1.0,
                "outcomes": outcomes,
                **{
                    f"{name}_proxy": {
                        "num_informative_pairs": 10,
                        "pairwise_sign_accuracy": 0.0,
                        "scores": proxy,
                    }
                    for name in (
                        "policy_prior",
                        "policy_path",
                        "action_nearness",
                    )
                },
            }
        )
    path.write_text(
        json.dumps(
            {
                "status": "ok",
                "eval_seeds": eval_seeds,
                "intervention_horizon": 1,
                "dimension_selection": "round_robin",
                "policy_value_beta": None,
                "value_readout": "direct_scalar_q",
                "policy_rng_protocol": (
                    "common_prngkey_probe_seed_plus_eval_seed"
                ),
                "num_action_dimensions": 16,
                "records": records,
            }
        )
    )


def _pair(tmp_path: Path, index: int = 1) -> ArmPair:
    control = tmp_path / f"control{index}.json"
    treatment = tmp_path / f"treatment{index}.json"
    _write_probe(control, q_correct=False)
    _write_probe(treatment, q_correct=True)
    return ArmPair(f"seed{index}", control, treatment)


def test_pair_parser():
    assert _arm_pair("seed1=control.json,treatment.json") == ArmPair(
        "seed1",
        Path("control.json"),
        Path("treatment.json"),
    )
    with pytest.raises(Exception, match="pair must be"):
        _arm_pair("seed1=control.json")


def test_discovery_gate_requires_paired_rct_improvement(tmp_path):
    result = summarize(
        [_pair(tmp_path)],
        bootstrap_replicates=100,
        bootstrap_seed=3,
        min_training_seeds=1,
        min_eval_seeds=8,
        min_informative_states=8,
        required_positive_training_seeds=1,
        strict_ci=False,
    )

    assert result["gate"] == "pass"
    assert result["claim_scope"] == "seed1_discovery_only"
    assert result["point"]["treatment_pairwise"] == pytest.approx(1.0)
    assert result["point"]["control_pairwise"] == pytest.approx(0.0)
    assert result["point"][
        "treatment_minus_control_pairwise"
    ] == pytest.approx(1.0)


def test_formal_gate_uses_matched_training_seed_and_environment_ci(tmp_path):
    result = summarize(
        [_pair(tmp_path, index) for index in range(1, 4)],
        bootstrap_replicates=200,
        bootstrap_seed=5,
        min_training_seeds=3,
        min_eval_seeds=8,
        min_informative_states=8,
        required_positive_training_seeds=2,
        strict_ci=True,
    )

    assert result["gate"] == "pass"
    assert result["claim_scope"] == "formal_matched_frozen_policy_rct_effect"
    assert result["crossed_bootstrap_ci95"][
        "treatment_minus_control_pairwise"
    ] == pytest.approx([1.0, 1.0])


def test_different_counterfactual_outcomes_are_rejected(tmp_path):
    pair = _pair(tmp_path)
    _write_probe(pair.treatment, q_correct=True, return_offset=1.0)

    with pytest.raises(ValueError, match="counterfactual outcomes differ"):
        summarize(
            [pair],
            bootstrap_replicates=10,
            bootstrap_seed=1,
            min_training_seeds=1,
            min_eval_seeds=8,
            min_informative_states=8,
            required_positive_training_seeds=1,
            strict_ci=False,
        )


def test_no_treatment_improvement_fails(tmp_path):
    control = tmp_path / "control.json"
    treatment = tmp_path / "treatment.json"
    _write_probe(control, q_correct=True)
    _write_probe(treatment, q_correct=True)

    result = summarize(
        [ArmPair("seed1", control, treatment)],
        bootstrap_replicates=20,
        bootstrap_seed=2,
        min_training_seeds=1,
        min_eval_seeds=8,
        min_informative_states=8,
        required_positive_training_seeds=1,
        strict_ci=False,
    )

    assert result["gate"] == "fail"
    assert not result["gate_checks"][
        "treatment_point_improves_control_pairwise"
    ]
