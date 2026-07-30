#!/usr/bin/env python3
"""Compare two matched CQN counterfactual audit JSON artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--trained", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    parser.add_argument("--return-atol", type=float, default=1e-8)
    return parser.parse_args()


def _record_key(record: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(record["eval_seed"]),
        int(record["anchor_step"]),
        int(record["action_dimension"]),
    )


def _records_by_key(payload: dict[str, Any]) -> dict[tuple[int, int, int], dict]:
    records = payload["records"]
    result = {_record_key(record): record for record in records}
    if len(result) != len(records):
        raise ValueError("audit contains duplicate seed/anchor/dimension records")
    return result


def _aggregate(records: list[dict[str, Any]]) -> dict[str, float]:
    informative = [
        record for record in records if int(record["num_informative_pairs"]) > 0
    ]
    if not informative:
        raise ValueError("audit split contains no informative records")
    pair_count = np.asarray(
        [record["num_informative_pairs"] for record in informative],
        np.float64,
    )
    pair_correct = np.asarray(
        [
            record["pairwise_sign_accuracy"]
            * record["num_informative_pairs"]
            for record in informative
        ],
        np.float64,
    )
    spearman = np.asarray(
        [record["spearman"] for record in informative],
        np.float64,
    )
    return {
        "pairwise_sign_accuracy": float(pair_correct.sum() / pair_count.sum()),
        "mean_spearman": float(np.nanmean(spearman)),
        "top1_match_rate": float(
            np.mean([record["top1_match"] for record in informative])
        ),
        "mean_realized_regret": float(
            np.mean([record["realized_regret"] for record in informative])
        ),
    }


def _paired_seed_bootstrap(
    baseline_records: dict[tuple[int, int, int], dict],
    trained_records: dict[tuple[int, int, int], dict],
    *,
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    all_seed_ids = sorted({key[0] for key in baseline_records})
    grouped_keys = {
        seed_id: [key for key in baseline_records if key[0] == seed_id]
        for seed_id in all_seed_ids
    }
    # A seed whose every candidate return is tied contributes no ranking
    # information. Sampling only that seed used to make an otherwise valid
    # cluster-bootstrap replicate raise in _aggregate(). Exclude such clusters
    # from the bootstrap population; they have zero weight in the point
    # estimate as well.
    seed_ids = [
        seed_id
        for seed_id in all_seed_ids
        if any(
            int(baseline_records[key]["num_informative_pairs"]) > 0
            for key in grouped_keys[seed_id]
        )
    ]
    if not seed_ids:
        raise ValueError("audit split contains no informative simulator seeds")
    rng = np.random.default_rng(seed)
    deltas = {
        "pairwise_sign_accuracy": np.empty(samples, np.float64),
        "mean_spearman": np.empty(samples, np.float64),
        "top1_match_rate": np.empty(samples, np.float64),
        # Positive means the trained critic has lower regret.
        "regret_reduction": np.empty(samples, np.float64),
    }
    for sample_index in range(samples):
        selected = rng.integers(0, len(seed_ids), size=len(seed_ids))
        keys = [
            key
            for index in selected
            for key in grouped_keys[seed_ids[int(index)]]
        ]
        baseline = _aggregate([baseline_records[key] for key in keys])
        trained = _aggregate([trained_records[key] for key in keys])
        deltas["pairwise_sign_accuracy"][sample_index] = (
            trained["pairwise_sign_accuracy"]
            - baseline["pairwise_sign_accuracy"]
        )
        deltas["mean_spearman"][sample_index] = (
            trained["mean_spearman"] - baseline["mean_spearman"]
        )
        deltas["top1_match_rate"][sample_index] = (
            trained["top1_match_rate"] - baseline["top1_match_rate"]
        )
        deltas["regret_reduction"][sample_index] = (
            baseline["mean_realized_regret"]
            - trained["mean_realized_regret"]
        )
    return {
        name: [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]
        for name, values in deltas.items()
    }


def _protocol(payload: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "train_seeds",
        "heldout_seeds",
        "anchor_steps",
        "action_dimensions",
        "candidate_mode",
        "force_level",
        "intervention_horizon",
        "score_level",
        "max_continuation_steps",
    )
    return {key: payload[key] for key in keys}


def _split_comparison(
    baseline: dict[str, Any],
    trained: dict[str, Any],
    split: str,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    return_atol: float,
) -> dict[str, Any]:
    baseline_summary = baseline["results"][f"{split}_before"]
    trained_summary = trained["results"][f"{split}_before"]
    baseline_records = _records_by_key(baseline_summary)
    trained_records = _records_by_key(trained_summary)
    if set(baseline_records) != set(trained_records):
        raise ValueError(f"{split} audit record keys do not match")

    return_differences = []
    action_differences = []
    matched_keys = []
    for key in sorted(baseline_records):
        baseline_record = baseline_records[key]
        trained_record = trained_records[key]
        return_difference = float(
            np.max(
                np.abs(
                    np.asarray(
                        baseline_record["realized_return"],
                        np.float64,
                    )
                    - np.asarray(
                        trained_record["realized_return"],
                        np.float64,
                    )
                )
            )
        )
        return_differences.append(return_difference)
        if return_difference <= return_atol:
            matched_keys.append(key)
        action_differences.append(
            float(
                np.max(
                    np.abs(
                        np.asarray(
                            baseline_record["predicted_q"],
                            np.float64,
                        )
                        - np.asarray(
                            trained_record["predicted_q"],
                            np.float64,
                        )
                    )
                )
            )
        )

    baseline_metrics = _aggregate(list(baseline_records.values()))
    trained_metrics = _aggregate(list(trained_records.values()))
    delta = {
        "pairwise_sign_accuracy": (
            trained_metrics["pairwise_sign_accuracy"]
            - baseline_metrics["pairwise_sign_accuracy"]
        ),
        "mean_spearman": (
            trained_metrics["mean_spearman"]
            - baseline_metrics["mean_spearman"]
        ),
        "top1_match_rate": (
            trained_metrics["top1_match_rate"]
            - baseline_metrics["top1_match_rate"]
        ),
        "regret_reduction": (
            baseline_metrics["mean_realized_regret"]
            - trained_metrics["mean_realized_regret"]
        ),
        "dimension_q_return_span_spearman": (
            trained_summary["dimension_q_return_span_spearman"]
            - baseline_summary["dimension_q_return_span_spearman"]
        ),
    }
    confidence_intervals = _paired_seed_bootstrap(
        baseline_records,
        trained_records,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    matched_returns = max(return_differences, default=0.0) <= return_atol
    matched_subset = None
    if any(
        int(baseline_records[key]["num_informative_pairs"]) > 0
        for key in matched_keys
    ):
        matched_baseline = _aggregate(
            [baseline_records[key] for key in matched_keys]
        )
        matched_trained = _aggregate(
            [trained_records[key] for key in matched_keys]
        )
        matched_subset = {
            "num_records": len(matched_keys),
            "baseline": matched_baseline,
            "trained": matched_trained,
            "delta_positive_is_improvement": {
                "pairwise_sign_accuracy": (
                    matched_trained["pairwise_sign_accuracy"]
                    - matched_baseline["pairwise_sign_accuracy"]
                ),
                "mean_spearman": (
                    matched_trained["mean_spearman"]
                    - matched_baseline["mean_spearman"]
                ),
                "top1_match_rate": (
                    matched_trained["top1_match_rate"]
                    - matched_baseline["top1_match_rate"]
                ),
                "regret_reduction": (
                    matched_baseline["mean_realized_regret"]
                    - matched_trained["mean_realized_regret"]
                ),
            },
        }
    return {
        "num_records": len(baseline_records),
        "num_mismatched_return_records": (
            len(baseline_records) - len(matched_keys)
        ),
        "matched_realized_returns": matched_returns,
        "max_realized_return_abs_difference": max(
            return_differences,
            default=0.0,
        ),
        "max_predicted_q_abs_difference": max(
            action_differences,
            default=0.0,
        ),
        "baseline": {
            **baseline_metrics,
            "dimension_q_return_span_spearman": baseline_summary[
                "dimension_q_return_span_spearman"
            ],
        },
        "trained": {
            **trained_metrics,
            "dimension_q_return_span_spearman": trained_summary[
                "dimension_q_return_span_spearman"
            ],
        },
        "delta_positive_is_improvement": delta,
        "paired_seed_bootstrap_ci95": confidence_intervals,
        "matched_record_subset": matched_subset,
        "strict_pairwise_improvement": bool(
            matched_returns
            and delta["pairwise_sign_accuracy"] > 0.0
            and confidence_intervals["pairwise_sign_accuracy"][0] > 0.0
        ),
    }


def compare(args: argparse.Namespace) -> dict[str, Any]:
    baseline = json.loads(args.baseline.read_text())
    trained = json.loads(args.trained.read_text())
    if baseline.get("status") != "ok" or trained.get("status") != "ok":
        raise ValueError("both audits must have status=ok")
    baseline_protocol = _protocol(baseline)
    trained_protocol = _protocol(trained)
    if baseline_protocol != trained_protocol:
        raise ValueError("baseline and trained audit protocols do not match")
    train = _split_comparison(
        baseline,
        trained,
        "train",
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        return_atol=args.return_atol,
    )
    heldout = _split_comparison(
        baseline,
        trained,
        "heldout",
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed + 1,
        return_atol=args.return_atol,
    )
    return {
        "status": "ok",
        "baseline": str(args.baseline.resolve()),
        "trained": str(args.trained.resolve()),
        "protocol": baseline_protocol,
        "bootstrap_samples": args.bootstrap_samples,
        "train": train,
        "heldout": heldout,
        "value_authenticity_gate_passed": bool(
            train["strict_pairwise_improvement"]
            and heldout["strict_pairwise_improvement"]
        ),
    }


def main() -> int:
    args = parse_args()
    payload = compare(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for split in ("train", "heldout"):
        result = payload[split]
        delta = result["delta_positive_is_improvement"]
        intervals = result["paired_seed_bootstrap_ci95"]
        print(
            f"{split}: pairwise={delta['pairwise_sign_accuracy']:+.4f} "
            f"CI={intervals['pairwise_sign_accuracy']} "
            f"spearman={delta['mean_spearman']:+.4f} "
            f"regret_reduction={delta['regret_reduction']:+.4f} "
            f"matched_returns={result['matched_realized_returns']}"
        )
    print(
        "value_authenticity_gate_passed="
        f"{payload['value_authenticity_gate_passed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
