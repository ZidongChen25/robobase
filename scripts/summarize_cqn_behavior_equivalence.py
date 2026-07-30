#!/usr/bin/env python3
"""Verify paired closed-loop equivalence for two CQN policy artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


_OUTCOME_KEYS = ("episode_success", "episode_reward", "episode_length")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--atol", type=float, default=0.0)
    return parser.parse_args()


def _load(path: Path) -> dict:
    payload = json.loads(path.expanduser().resolve().read_text())
    if payload.get("status") != "ok":
        raise ValueError(f"evaluation did not complete successfully: {path}")
    if not payload.get("episode_results"):
        raise ValueError(f"evaluation has no episode results: {path}")
    return payload


def summarize(
    reference_path: Path,
    candidate_path: Path,
    *,
    atol: float = 0.0,
) -> dict:
    if atol < 0.0:
        raise ValueError("atol must be non-negative")
    reference = _load(reference_path)
    candidate = _load(candidate_path)
    if reference.get("task") != candidate.get("task"):
        raise ValueError("reference and candidate tasks differ")

    def by_seed(payload: dict) -> dict[int, dict]:
        rows = {int(row["seed"]): row for row in payload["episode_results"]}
        if len(rows) != len(payload["episode_results"]):
            raise ValueError("evaluation contains duplicate seeds")
        return rows

    reference_by_seed = by_seed(reference)
    candidate_by_seed = by_seed(candidate)
    if set(reference_by_seed) != set(candidate_by_seed):
        raise ValueError("reference and candidate seed sets differ")

    seeds = sorted(reference_by_seed)
    per_metric = {}
    all_equal = True
    for key in _OUTCOME_KEYS:
        reference_values = np.asarray(
            [reference_by_seed[seed][key] for seed in seeds],
            dtype=np.float64,
        )
        candidate_values = np.asarray(
            [candidate_by_seed[seed][key] for seed in seeds],
            dtype=np.float64,
        )
        deltas = candidate_values - reference_values
        equal = np.isclose(
            candidate_values,
            reference_values,
            atol=atol,
            rtol=0.0,
        )
        per_metric[key] = {
            "reference_mean": float(reference_values.mean()),
            "candidate_mean": float(candidate_values.mean()),
            "mean_delta": float(deltas.mean()),
            "max_abs_delta": float(np.max(np.abs(deltas))),
            "num_mismatched_seeds": int(np.sum(~equal)),
        }
        all_equal = all_equal and bool(np.all(equal))

    return {
        "status": "ok",
        "gate": "pass" if all_equal else "fail",
        "exact_closed_loop_equivalence": bool(all_equal and atol == 0.0),
        "within_tolerance_closed_loop_equivalence": bool(all_equal),
        "atol": float(atol),
        "task": reference.get("task"),
        "num_paired_episodes": len(seeds),
        "seed_start": int(seeds[0]),
        "seed_end": int(seeds[-1]),
        "reference": str(reference_path.expanduser().resolve()),
        "candidate": str(candidate_path.expanduser().resolve()),
        "metrics": per_metric,
    }


def main() -> int:
    args = parse_args()
    payload = summarize(
        args.reference,
        args.candidate,
        atol=args.atol,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["gate"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
