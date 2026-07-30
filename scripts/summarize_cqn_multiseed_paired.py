#!/usr/bin/env python3
"""Crossed-bootstrap paired comparison over model and environment seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _labeled_pair(value: str) -> tuple[str, Path, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "pair must be LABEL=BASELINE_PATH,CANDIDATE_PATH"
        )
    label, paths = value.split("=", 1)
    pieces = paths.split(",", 1)
    if not label or len(pieces) != 2 or not all(pieces):
        raise argparse.ArgumentTypeError(
            "pair must be LABEL=BASELINE_PATH,CANDIDATE_PATH"
        )
    return label, Path(pieces[0]), Path(pieces[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair",
        required=True,
        action="append",
        type=_labeled_pair,
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=49_100)
    parser.add_argument("--min-mean-delta", type=float, default=0.0)
    parser.add_argument("--min-ci-lower", type=float, default=0.0)
    return parser.parse_args()


def _outcomes(path: Path) -> tuple[dict, list[int], np.ndarray]:
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text())
    if payload.get("status") != "ok":
        raise ValueError(f"incomplete eval artifact: {path}")
    rows = payload.get("episode_results", [])
    seeds = [int(row["seed"]) for row in rows]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"invalid episode seeds: {path}")
    success = np.asarray(
        [float(row["episode_success"]) for row in rows],
        dtype=np.float64,
    )
    if not np.all(np.isin(success, (0.0, 1.0))):
        raise ValueError(f"episode success must be binary: {path}")
    return payload, seeds, success


def summarize(
    pairs: list[tuple[str, Path, Path]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    min_mean_delta: float,
    min_ci_lower: float,
) -> dict:
    if len(pairs) < 2:
        raise ValueError("at least two training-seed pairs are required")
    labels = [label for label, _, _ in pairs]
    if len(set(labels)) != len(labels):
        raise ValueError("training-seed labels must be unique")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap replicates must be positive")

    reference_seeds = None
    baseline_rows = []
    candidate_rows = []
    per_training_seed = {}
    sources = {}
    for label, baseline_path, candidate_path in pairs:
        baseline_payload, baseline_seeds, baseline = _outcomes(baseline_path)
        candidate_payload, candidate_seeds, candidate = _outcomes(
            candidate_path
        )
        if baseline_seeds != candidate_seeds:
            raise ValueError(f"{label} pair does not use matched eval seeds")
        if reference_seeds is None:
            reference_seeds = baseline_seeds
        elif baseline_seeds != reference_seeds:
            raise ValueError("training seeds do not share common eval seeds")

        delta = candidate - baseline
        wins = int(np.sum(delta > 0))
        losses = int(np.sum(delta < 0))
        baseline_rows.append(baseline)
        candidate_rows.append(candidate)
        per_training_seed[label] = {
            "baseline_success": float(baseline.mean()),
            "candidate_success": float(candidate.mean()),
            "paired_delta": float(delta.mean()),
            "paired_wins": wins,
            "paired_losses": losses,
            "paired_ties": int(delta.size - wins - losses),
        }
        sources[label] = {
            "baseline": str(baseline_path.expanduser().resolve()),
            "candidate": str(candidate_path.expanduser().resolve()),
            "baseline_snapshot": str(baseline_payload["snapshot"]),
            "candidate_snapshot": str(candidate_payload["snapshot"]),
        }

    baseline = np.stack(baseline_rows)
    candidate = np.stack(candidate_rows)
    delta = candidate - baseline
    rng = np.random.default_rng(int(bootstrap_seed))
    model_indices = rng.integers(
        0,
        delta.shape[0],
        size=(bootstrap_replicates, delta.shape[0]),
    )
    environment_indices = rng.integers(
        0,
        delta.shape[1],
        size=(bootstrap_replicates, delta.shape[1]),
    )
    sampled_delta = delta[
        model_indices[:, :, None],
        environment_indices[:, None, :],
    ].mean(axis=(1, 2))
    ci = [
        float(np.quantile(sampled_delta, 0.025)),
        float(np.quantile(sampled_delta, 0.975)),
    ]
    per_seed_delta = delta.mean(axis=1)
    wins = int(np.sum(delta > 0))
    losses = int(np.sum(delta < 0))
    checks = {
        "mean_delta_strictly_above_threshold": (
            float(delta.mean()) > float(min_mean_delta)
        ),
        "crossed_ci_lower_at_least_threshold": (
            ci[0] >= float(min_ci_lower)
        ),
        "aggregate_wins_above_losses": wins > losses,
        "positive_training_seed_majority": (
            int(np.sum(per_seed_delta > 0)) > len(per_seed_delta) / 2
        ),
    }
    return {
        "status": "ok",
        "labels": labels,
        "sources": sources,
        "num_training_seeds": int(delta.shape[0]),
        "num_eval_seeds": int(delta.shape[1]),
        "eval_seed_start": int(reference_seeds[0]),
        "eval_seed_end": int(reference_seeds[-1]),
        "per_training_seed": per_training_seed,
        "mean_baseline_success": float(baseline.mean()),
        "mean_candidate_success": float(candidate.mean()),
        "mean_paired_delta": float(delta.mean()),
        "sample_std_training_seed_delta": float(
            per_seed_delta.std(ddof=1)
        ),
        "aggregate_paired_wins": wins,
        "aggregate_paired_losses": losses,
        "aggregate_paired_ties": int(delta.size - wins - losses),
        "crossed_bootstrap_ci95": ci,
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_seed": int(bootstrap_seed),
        "thresholds": {
            "min_mean_delta": float(min_mean_delta),
            "min_ci_lower": float(min_ci_lower),
        },
        "gate_checks": checks,
        "gate": "pass" if all(checks.values()) else "fail",
    }


def main() -> int:
    args = parse_args()
    payload = summarize(
        args.pair,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        min_mean_delta=args.min_mean_delta,
        min_ci_lower=args.min_ci_lower,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
