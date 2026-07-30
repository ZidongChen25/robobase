import json
from argparse import Namespace

from scripts.summarize_cqn_counterfactual_audits import compare


def _record(seed, dimension, predicted):
    return {
        "eval_seed": seed,
        "anchor_step": 30,
        "action_dimension": dimension,
        "predicted_q": predicted,
        "realized_return": [0.0, 1.0],
        "num_informative_pairs": 1,
        "pairwise_sign_accuracy": float(predicted[1] > predicted[0]),
        "spearman": 1.0 if predicted[1] > predicted[0] else -1.0,
        "top1_match": predicted[1] > predicted[0],
        "realized_regret": 0.0 if predicted[1] > predicted[0] else 1.0,
    }


def _tie_record(seed, dimension, predicted):
    return {
        "eval_seed": seed,
        "anchor_step": 30,
        "action_dimension": dimension,
        "predicted_q": predicted,
        "realized_return": [0.0, 0.0],
        "num_informative_pairs": 0,
        "pairwise_sign_accuracy": float("nan"),
        "spearman": float("nan"),
        "top1_match": True,
        "realized_regret": 0.0,
    }


def _audit(records):
    summary = {
        "records": records,
        "dimension_q_return_span_spearman": 1.0,
    }
    return {
        "status": "ok",
        "train_seeds": [1, 2],
        "heldout_seeds": [3, 4],
        "anchor_steps": [30],
        "action_dimensions": [0],
        "candidate_mode": "sibling_bins",
        "force_level": 1,
        "intervention_horizon": 4,
        "score_level": 1,
        "max_continuation_steps": 300,
        "results": {
            "train_before": summary,
            "heldout_before": summary,
        },
    }


def test_compare_reports_paired_value_improvement(tmp_path):
    baseline_records = [
        _record(1, 0, [1.0, 0.0]),
        _record(2, 0, [1.0, 0.0]),
    ]
    trained_records = [
        _record(1, 0, [0.0, 1.0]),
        _record(2, 0, [0.0, 1.0]),
    ]
    baseline_path = tmp_path / "baseline.json"
    trained_path = tmp_path / "trained.json"
    baseline_path.write_text(json.dumps(_audit(baseline_records)))
    trained_path.write_text(json.dumps(_audit(trained_records)))

    result = compare(
        Namespace(
            baseline=baseline_path,
            trained=trained_path,
            bootstrap_samples=100,
            bootstrap_seed=7,
            return_atol=1e-8,
        )
    )

    assert result["train"]["matched_realized_returns"]
    assert (
        result["train"]["delta_positive_is_improvement"][
            "pairwise_sign_accuracy"
        ]
        == 1.0
    )
    assert result["heldout"]["strict_pairwise_improvement"]
    assert result["value_authenticity_gate_passed"]


def test_compare_bootstrap_ignores_all_tie_seed_cluster(tmp_path):
    baseline_records = [
        _record(1, 0, [1.0, 0.0]),
        _tie_record(2, 0, [1.0, 0.0]),
    ]
    trained_records = [
        _record(1, 0, [0.0, 1.0]),
        _tie_record(2, 0, [0.0, 1.0]),
    ]
    baseline_path = tmp_path / "baseline.json"
    trained_path = tmp_path / "trained.json"
    baseline_path.write_text(json.dumps(_audit(baseline_records)))
    trained_path.write_text(json.dumps(_audit(trained_records)))

    result = compare(
        Namespace(
            baseline=baseline_path,
            trained=trained_path,
            bootstrap_samples=100,
            bootstrap_seed=11,
            return_atol=1e-8,
        )
    )

    assert result["train"]["strict_pairwise_improvement"]
    assert result["value_authenticity_gate_passed"]


def test_compare_reports_metrics_on_exactly_matched_record_subset(tmp_path):
    baseline_records = [
        _record(1, 0, [1.0, 0.0]),
        _record(2, 0, [1.0, 0.0]),
    ]
    trained_records = [
        _record(1, 0, [0.0, 1.0]),
        _record(2, 0, [0.0, 1.0]),
    ]
    trained_records[1]["realized_return"] = [0.0, 2.0]
    baseline_path = tmp_path / "baseline.json"
    trained_path = tmp_path / "trained.json"
    baseline_path.write_text(json.dumps(_audit(baseline_records)))
    trained_path.write_text(json.dumps(_audit(trained_records)))

    result = compare(
        Namespace(
            baseline=baseline_path,
            trained=trained_path,
            bootstrap_samples=100,
            bootstrap_seed=13,
            return_atol=1e-8,
        )
    )

    train = result["train"]
    assert not train["matched_realized_returns"]
    assert train["num_mismatched_return_records"] == 1
    assert train["matched_record_subset"]["num_records"] == 1
    assert (
        train["matched_record_subset"]["delta_positive_is_improvement"][
            "pairwise_sign_accuracy"
        ]
        == 1.0
    )
