#!/usr/bin/env python3
"""Summarize a matched before/after CQN branch-oracle test.

The input is produced by ``finetune_cqn_branch_oracle.py`` in
``--coverage-only --comparison-snapshot`` mode.  Its historical ``train`` and
``heldout`` result names describe cache partitions; no fitting occurs during
that comparison.  This script combines both partitions, verifies that the
counterfactual outcomes are identical before and after, and bootstraps paired
deltas by simulator seed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


METRIC_DIRECTIONS = {
    "pairwise_sign_accuracy": 1.0,
    "mean_spearman": 1.0,
    "top1_match_rate": 1.0,
    "mean_realized_regret": -1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=28_005)
    return parser.parse_args()


def _record_key(record: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(record["eval_seed"]),
        int(record["anchor_step"]),
        int(record["action_dimension"]),
    )


def _index_records(
    records: list[dict[str, Any]],
) -> dict[tuple[int, int, int], dict[str, Any]]:
    indexed: dict[tuple[int, int, int], dict[str, Any]] = {}
    for record in records:
        key = _record_key(record)
        if key in indexed:
            raise ValueError(f"duplicate branch record: {key}")
        indexed[key] = record
    return indexed


def _matched_records(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    before_by_key = _index_records(before)
    after_by_key = _index_records(after)
    if before_by_key.keys() != after_by_key.keys():
        missing_after = sorted(before_by_key.keys() - after_by_key.keys())
        missing_before = sorted(after_by_key.keys() - before_by_key.keys())
        raise ValueError(
            "before/after branch keys differ: "
            f"missing_after={missing_after[:5]}, "
            f"missing_before={missing_before[:5]}"
        )

    matched = []
    for key in sorted(before_by_key):
        before_record = before_by_key[key]
        after_record = after_by_key[key]
        before_return = np.asarray(
            before_record["realized_return"], dtype=np.float64
        )
        after_return = np.asarray(
            after_record["realized_return"], dtype=np.float64
        )
        if not np.array_equal(before_return, after_return):
            raise ValueError(
                f"counterfactual outcomes changed between models at {key}"
            )
        if (
            int(before_record["num_informative_pairs"])
            != int(after_record["num_informative_pairs"])
        ):
            raise ValueError(f"informative-pair count changed at {key}")
        matched.append((before_record, after_record))
    return matched


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    informative = [
        record
        for record in records
        if int(record["num_informative_pairs"]) > 0
    ]
    pair_count = sum(
        int(record["num_informative_pairs"]) for record in informative
    )
    pair_correct = sum(
        float(record["pairwise_sign_accuracy"])
        * int(record["num_informative_pairs"])
        for record in informative
    )
    spearman = np.asarray(
        [float(record["spearman"]) for record in informative],
        dtype=np.float64,
    )
    finite_spearman = spearman[np.isfinite(spearman)]
    return {
        "num_states": len(records),
        "num_informative_states": len(informative),
        "num_informative_pairs": pair_count,
        "pairwise_sign_accuracy": (
            float(pair_correct / pair_count) if pair_count else None
        ),
        "mean_spearman": (
            float(finite_spearman.mean()) if finite_spearman.size else None
        ),
        "top1_match_rate": (
            float(
                np.mean(
                    [float(record["top1_match"]) for record in informative]
                )
            )
            if informative
            else None
        ),
        "mean_realized_regret": (
            float(
                np.mean(
                    [
                        float(record["realized_regret"])
                        for record in informative
                    ]
                )
            )
            if informative
            else None
        ),
    }


def _deltas(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for metric in METRIC_DIRECTIONS:
        before_value = before[metric]
        after_value = after[metric]
        result[metric] = (
            float(after_value - before_value)
            if before_value is not None and after_value is not None
            else None
        )
    return result


def _paired_seed_bootstrap(
    matched: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[
        int, list[tuple[dict[str, Any], dict[str, Any]]]
    ] = {}
    for pair in matched:
        if int(pair[0]["num_informative_pairs"]) > 0:
            grouped.setdefault(int(pair[0]["eval_seed"]), []).append(pair)
    seed_ids = sorted(grouped)
    payload: dict[str, Any] = {
        "unit": "informative_eval_seed",
        "confidence": 0.95,
        "num_replicates": int(replicates),
        "num_seeds": len(seed_ids),
        "delta_ci": {metric: [None, None] for metric in METRIC_DIRECTIONS},
    }
    if not seed_ids or replicates <= 0:
        return payload

    rng = np.random.default_rng(seed)
    samples = {
        metric: np.full(replicates, np.nan, dtype=np.float64)
        for metric in METRIC_DIRECTIONS
    }
    for bootstrap_index in range(replicates):
        selected = rng.integers(0, len(seed_ids), size=len(seed_ids))
        sampled = [
            pair
            for selected_index in selected
            for pair in grouped[seed_ids[int(selected_index)]]
        ]
        before_summary = _summary([pair[0] for pair in sampled])
        after_summary = _summary([pair[1] for pair in sampled])
        delta = _deltas(before_summary, after_summary)
        for metric, value in delta.items():
            if value is not None:
                samples[metric][bootstrap_index] = value

    payload["delta_ci"] = {
        metric: [
            float(value)
            for value in np.nanpercentile(sample, [2.5, 97.5])
        ]
        for metric, sample in samples.items()
    }
    return payload


def summarize(
    payload: dict[str, Any],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    if payload.get("status") != "ok":
        raise ValueError("input comparison did not complete successfully")
    if not payload.get("coverage_only"):
        raise ValueError("matched gate requires a coverage-only comparison")
    results = payload.get("results", {})
    required = {
        "train_before",
        "train_after",
        "heldout_before",
        "heldout_after",
    }
    if not required.issubset(results):
        raise ValueError(
            f"input is missing result groups: {sorted(required - results.keys())}"
        )

    before = (
        list(results["train_before"]["records"])
        + list(results["heldout_before"]["records"])
    )
    after = (
        list(results["train_after"]["records"])
        + list(results["heldout_after"]["records"])
    )
    matched = _matched_records(before, after)
    before_summary = _summary(before)
    after_summary = _summary(after)
    delta = _deltas(before_summary, after_summary)

    seed_ids = sorted({int(pair[0]["eval_seed"]) for pair in matched})
    per_seed = {}
    for eval_seed in seed_ids:
        seed_pairs = [
            pair
            for pair in matched
            if int(pair[0]["eval_seed"]) == eval_seed
        ]
        seed_before = _summary([pair[0] for pair in seed_pairs])
        seed_after = _summary([pair[1] for pair in seed_pairs])
        per_seed[str(eval_seed)] = {
            "before": seed_before,
            "after": seed_after,
            "delta": _deltas(seed_before, seed_after),
        }

    checks = {
        metric: (
            value is not None
            and math.isfinite(value)
            and direction * value > 0.0
        )
        for (metric, direction), value in zip(
            METRIC_DIRECTIONS.items(),
            (delta[metric] for metric in METRIC_DIRECTIONS),
        )
    }
    informative_seed_ids = [
        eval_seed
        for eval_seed in seed_ids
        if per_seed[str(eval_seed)]["before"]["num_informative_states"] > 0
    ]
    all_direction_count = sum(
        all(
            per_seed[str(eval_seed)]["delta"][metric] is not None
            and direction
            * per_seed[str(eval_seed)]["delta"][metric]
            > 0.0
            for metric, direction in METRIC_DIRECTIONS.items()
        )
        for eval_seed in informative_seed_ids
    )
    return {
        "status": "ok",
        "protocol": "matched_coverage_only_combined_partitions",
        "source_snapshot": payload.get("source_snapshot"),
        "comparison_snapshot": payload.get("comparison_snapshot"),
        "frozen_component_bitwise_equal": payload.get(
            "frozen_component_bitwise_equal"
        ),
        "eval_seeds": seed_ids,
        "num_informative_seeds": len(informative_seed_ids),
        "before": before_summary,
        "after": after_summary,
        "delta_after_minus_before": delta,
        "per_seed": per_seed,
        "paired_seed_bootstrap": _paired_seed_bootstrap(
            matched,
            replicates=bootstrap_replicates,
            seed=seed,
        ),
        "directional_stability": {
            "all_four_metrics_improve_count": all_direction_count,
            "num_informative_seeds": len(informative_seed_ids),
        },
        "gate": {
            "criterion": (
                "combined mean pairwise, Spearman, and top1 increase; "
                "combined mean regret decreases"
            ),
            "checks": checks,
            "passed": all(checks.values()),
        },
    }


def main() -> int:
    args = parse_args()
    if args.bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates must be non-negative")
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    payload = summarize(
        json.loads(input_path.read_text()),
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
