#!/usr/bin/env python3
"""Summarize held-out outcomes for validation-selected training seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("run must be LABEL=PATH")
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        required=True,
        action="append",
        type=_labeled_path,
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=43_100)
    return parser.parse_args()


def summarize(
    runs: list[tuple[str, Path]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict:
    if len(runs) < 2:
        raise ValueError("at least two training-seed runs are required")
    if len({label for label, _ in runs}) != len(runs):
        raise ValueError("run labels must be unique")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap-replicates must be positive")

    labels = []
    sources = []
    snapshots = []
    seed_reference = None
    success_rows = []
    for label, path in runs:
        path = path.expanduser().resolve()
        payload = json.loads(path.read_text())
        if payload.get("status") != "ok":
            raise ValueError(f"{label} did not complete successfully")
        episode_results = payload.get("episode_results", [])
        seeds = [int(row["seed"]) for row in episode_results]
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError(f"{label} has invalid episode seeds")
        if seed_reference is None:
            seed_reference = seeds
        elif seeds != seed_reference:
            raise ValueError(f"{label} does not use the common held-out seeds")
        labels.append(label)
        sources.append(str(path))
        snapshots.append(str(payload["snapshot"]))
        success_rows.append(
            [float(row["episode_success"]) for row in episode_results]
        )

    success = np.asarray(success_rows, dtype=np.float64)
    run_means = success.mean(axis=1)
    rng = np.random.default_rng(bootstrap_seed)
    model_indices = rng.integers(
        0,
        success.shape[0],
        size=(bootstrap_replicates, success.shape[0]),
    )
    environment_indices = rng.integers(
        0,
        success.shape[1],
        size=(bootstrap_replicates, success.shape[1]),
    )
    sampled = success[
        model_indices[:, :, None],
        environment_indices[:, None, :],
    ].mean(axis=(1, 2))
    return {
        "status": "ok",
        "labels": labels,
        "sources": sources,
        "snapshots": snapshots,
        "num_training_seeds": int(success.shape[0]),
        "num_eval_seeds": int(success.shape[1]),
        "eval_seed_start": int(seed_reference[0]),
        "eval_seed_end": int(seed_reference[-1]),
        "per_training_seed_success": {
            label: float(value) for label, value in zip(labels, run_means)
        },
        "mean_success_across_training_seeds": float(run_means.mean()),
        "sample_std_across_training_seeds": float(
            run_means.std(ddof=1)
        ),
        "crossed_bootstrap_ci95": [
            float(np.quantile(sampled, 0.025)),
            float(np.quantile(sampled, 0.975)),
        ],
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_seed": int(bootstrap_seed),
    }


def main() -> int:
    args = parse_args()
    payload = summarize(
        args.run,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
