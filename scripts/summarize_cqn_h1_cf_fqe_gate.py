#!/usr/bin/env python3
"""Gate one-step simulator-branch MC fine-tuning of a direct CQN-AS critic.

The positive and negative-control fits must use the same frozen branch cache.
The negative control permutes candidate returns within each simulator state,
so it preserves state difficulty and label marginals while destroying the
causal action/return association.  Held-out comparisons bootstrap whole
simulator seeds and never treat sibling actions from one restored state as
independent samples.

Despite the historical ``cf_fqe`` filename, this is not conventional
replay-only FQE: its labels are Monte-Carlo returns from restored simulator
branches.  The gate records and enforces that distinction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive", required=True, type=Path)
    parser.add_argument("--shuffle-control", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-informative-states", type=int, default=24)
    parser.add_argument("--min-eval-seeds", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=152_000)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text())
    if payload.get("status") != "ok":
        raise ValueError(f"incomplete oracle artifact: {resolved}")
    payload["_path"] = str(resolved)
    return payload


def _record_key(record: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(record["eval_seed"]),
        int(record["anchor_step"]),
        int(record["action_dimension"]),
    )


def _matched_records(
    positive: dict[str, Any],
    shuffle: dict[str, Any],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    positive_records = positive["results"]["heldout_after"]["records"]
    shuffle_records = shuffle["results"]["heldout_after"]["records"]
    positive_by_key = {_record_key(record): record for record in positive_records}
    shuffle_by_key = {_record_key(record): record for record in shuffle_records}
    if len(positive_by_key) != len(positive_records):
        raise ValueError("positive held-out records contain duplicate keys")
    if len(shuffle_by_key) != len(shuffle_records):
        raise ValueError("shuffle held-out records contain duplicate keys")
    if positive_by_key.keys() != shuffle_by_key.keys():
        raise ValueError("positive and shuffle held-out record keys differ")

    matched = []
    for key in sorted(positive_by_key):
        left = positive_by_key[key]
        right = shuffle_by_key[key]
        if not np.array_equal(
            np.asarray(left["realized_return"], np.float64),
            np.asarray(right["realized_return"], np.float64),
        ):
            raise ValueError(f"realized returns differ for held-out state {key}")
        if int(left["num_informative_pairs"]) != int(
            right["num_informative_pairs"]
        ):
            raise ValueError(f"informative pair counts differ for state {key}")
        matched.append((left, right))
    return matched


def _weighted_accuracy(
    records: list[dict[str, Any]],
    accuracy_key: str,
    count_key: str,
) -> float:
    count = np.asarray([record[count_key] for record in records], np.float64)
    correct = np.asarray(
        [record[accuracy_key] * record[count_key] for record in records],
        np.float64,
    )
    denominator = float(count.sum())
    return float(correct.sum() / denominator) if denominator else float("nan")


def _metrics(
    matched: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, float]:
    positive = [item[0] for item in matched]
    shuffle = [item[1] for item in matched]
    informative = [
        (left, right)
        for left, right in zip(positive, shuffle, strict=True)
        if int(left["num_informative_pairs"]) > 0
    ]
    if not informative:
        return {
            "pairwise_sign_accuracy": float("nan"),
            "mean_spearman": float("nan"),
            "q_minus_behavior_pairwise": float("nan"),
            "q_minus_policy_pairwise": float("nan"),
            "q_minus_shuffle_pairwise": float("nan"),
        }
    positive = [item[0] for item in informative]
    shuffle = [item[1] for item in informative]
    q_accuracy = _weighted_accuracy(
        positive,
        "pairwise_sign_accuracy",
        "num_informative_pairs",
    )
    behavior_accuracy = _weighted_accuracy(
        positive,
        "behavior_proxy_pairwise_sign_accuracy",
        "behavior_proxy_num_informative_pairs",
    )
    policy_accuracy = _weighted_accuracy(
        positive,
        "policy_prior_pairwise_sign_accuracy",
        "policy_prior_num_informative_pairs",
    )
    shuffle_accuracy = _weighted_accuracy(
        shuffle,
        "pairwise_sign_accuracy",
        "num_informative_pairs",
    )
    spearman = np.asarray(
        [record["spearman"] for record in positive],
        np.float64,
    )
    spearman = spearman[np.isfinite(spearman)]
    return {
        "pairwise_sign_accuracy": q_accuracy,
        "mean_spearman": (
            float(spearman.mean()) if spearman.size else float("nan")
        ),
        "q_minus_behavior_pairwise": q_accuracy - behavior_accuracy,
        "q_minus_policy_pairwise": q_accuracy - policy_accuracy,
        "q_minus_shuffle_pairwise": q_accuracy - shuffle_accuracy,
    }


def _interval(values: np.ndarray) -> list[float]:
    finite = np.asarray(values, np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return [float("nan"), float("nan")]
    return [
        float(np.quantile(finite, 0.025)),
        float(np.quantile(finite, 0.975)),
    ]


def _seed_bootstrap(
    matched: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[
        int,
        list[tuple[dict[str, Any], dict[str, Any]]],
    ] = {}
    for pair in matched:
        grouped.setdefault(int(pair[0]["eval_seed"]), []).append(pair)
    seed_ids = sorted(grouped)
    point = _metrics(matched)
    samples = {
        key: np.full(max(replicates, 0), np.nan, np.float64)
        for key in point
    }
    if replicates > 0 and seed_ids:
        rng = np.random.default_rng(int(seed))
        for index in range(replicates):
            selected = rng.integers(0, len(seed_ids), size=len(seed_ids))
            sampled = [
                pair
                for selected_index in selected
                for pair in grouped[seed_ids[int(selected_index)]]
            ]
            values = _metrics(sampled)
            for key, value in values.items():
                samples[key][index] = value
    return {
        "unit": "heldout_simulator_seed",
        "num_seeds": len(seed_ids),
        "num_states": len(matched),
        "num_replicates": int(replicates),
        "point": point,
        "ci95": {key: _interval(values) for key, values in samples.items()},
    }


def summarize(
    positive: dict[str, Any],
    shuffle: dict[str, Any],
    *,
    min_informative_states: int,
    min_eval_seeds: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    matched = _matched_records(positive, shuffle)
    bootstrap = _seed_bootstrap(
        matched,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    heldout = positive["results"]["heldout_after"]
    positive_ci = bootstrap["ci95"]
    positive_point = bootstrap["point"]
    source_match = (
        positive.get("source_snapshot") == shuffle.get("source_snapshot")
    )
    cache_match = (
        positive.get("dataset_cache") == shuffle.get("dataset_cache")
    )
    checks = {
        "simulator_branch_mc_target": (
            positive.get("target_estimator")
            == "simulator_branch_monte_carlo"
            and shuffle.get("target_estimator")
            == "simulator_branch_monte_carlo"
        ),
        "fixed_independent_bc_continuation": (
            positive.get("continuation_policy")
            == "frozen_independent_bc"
            and shuffle.get("continuation_policy")
            == "frozen_independent_bc"
            and positive.get("continuation_policy_value_beta") is None
            and shuffle.get("continuation_policy_value_beta") is None
        ),
        "one_step_intervention": (
            int(positive.get("intervention_horizon", -1)) == 1
            and int(shuffle.get("intervention_horizon", -1)) == 1
        ),
        "sibling_bin_intervention": (
            positive.get("candidate_mode") == "sibling_bins"
            and shuffle.get("candidate_mode") == "sibling_bins"
        ),
        "direct_scalar_q": (
            positive.get("critic_parameterization") == "direct_scalar_q"
            and shuffle.get("critic_parameterization") == "direct_scalar_q"
        ),
        "positive_labels_unshuffled": (
            positive.get("train_return_shuffle") == "none"
        ),
        "negative_control_within_state_shuffled": (
            shuffle.get("train_return_shuffle") == "within_state"
        ),
        "matched_source_snapshot": source_match,
        "matched_branch_cache": cache_match,
        "frozen_policy_preserved": (
            positive.get("frozen_policy_bitwise_equal_after_fit") is True
            and shuffle.get("frozen_policy_bitwise_equal_after_fit") is True
        ),
        "enough_train_information": (
            int(positive.get("num_informative_train_states", 0))
            >= min_informative_states
        ),
        "enough_heldout_information": (
            int(heldout.get("num_informative_states", 0))
            >= min_informative_states
        ),
        "enough_heldout_simulator_seeds": (
            int(bootstrap["num_seeds"]) >= min_eval_seeds
        ),
        "pairwise_above_chance": (
            positive_ci["pairwise_sign_accuracy"][0] > 0.5
        ),
        "spearman_positive": (
            positive_ci["mean_spearman"][0] > 0.0
        ),
        "beats_action_nearness_proxy": (
            positive_ci["q_minus_behavior_pairwise"][0] > 0.0
        ),
        "beats_independent_bc_proxy": (
            positive_ci["q_minus_policy_pairwise"][0] > 0.0
        ),
        "beats_within_state_shuffle_control": (
            positive_ci["q_minus_shuffle_pairwise"][0] > 0.0
        ),
    }
    return {
        "status": "ok",
        "gate": "pass" if all(checks.values()) else "fail",
        "hypothesis": (
            "one-step same-state simulator-branch MC supervision teaches "
            "direct scalar-Q action effects beyond imitation and state-only "
            "shortcuts"
        ),
        "positive_artifact": positive["_path"],
        "shuffle_control_artifact": shuffle["_path"],
        "source_snapshot": positive.get("source_snapshot"),
        "dataset_cache": positive.get("dataset_cache"),
        "intervention_horizon": positive.get("intervention_horizon"),
        "num_informative_train_states": positive.get(
            "num_informative_train_states"
        ),
        "num_informative_heldout_states": heldout.get(
            "num_informative_states"
        ),
        "heldout_pairwise_sign_accuracy": positive_point[
            "pairwise_sign_accuracy"
        ],
        "heldout_mean_spearman": positive_point["mean_spearman"],
        "heldout_q_minus_behavior_pairwise": positive_point[
            "q_minus_behavior_pairwise"
        ],
        "heldout_q_minus_policy_pairwise": positive_point[
            "q_minus_policy_pairwise"
        ],
        "heldout_q_minus_shuffle_pairwise": positive_point[
            "q_minus_shuffle_pairwise"
        ],
        "seed_bootstrap": bootstrap,
        "gate_checks": checks,
    }


def main() -> int:
    args = parse_args()
    if args.min_informative_states < 1:
        raise ValueError("--min-informative-states must be positive")
    if args.min_eval_seeds < 2:
        raise ValueError("--min-eval-seeds must be at least two")
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be positive")
    payload = summarize(
        _load(args.positive),
        _load(args.shuffle_control),
        min_informative_states=int(args.min_informative_states),
        min_eval_seeds=int(args.min_eval_seeds),
        bootstrap_replicates=int(args.bootstrap_replicates),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
