#!/usr/bin/env python3
"""Compare two completed CQN evaluations on exactly matched episode seeds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--min-ci-lower", type=float, default=0.0)
    return parser.parse_args()


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("status") != "ok":
        raise ValueError(f"evaluation is not complete: {path}")
    return payload


def _exact_mcnemar_p(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(wins, losses) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def summarize(
    baseline: dict,
    candidate: dict,
    *,
    baseline_path: Path,
    candidate_path: Path,
    bootstrap_samples: int,
    bootstrap_seed: int,
    min_delta: float,
    min_ci_lower: float,
) -> dict:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap-samples must be positive")
    if not math.isfinite(min_delta) or not math.isfinite(min_ci_lower):
        raise ValueError("gate thresholds must be finite")
    baseline_by_seed = {
        int(row["seed"]): float(row["episode_success"])
        for row in baseline["episode_results"]
    }
    candidate_by_seed = {
        int(row["seed"]): float(row["episode_success"])
        for row in candidate["episode_results"]
    }
    if not baseline_by_seed or set(candidate_by_seed) != set(baseline_by_seed):
        raise ValueError("candidate and baseline must share a nonempty seed set")
    seeds = np.asarray(sorted(baseline_by_seed), dtype=np.int64)
    baseline_success = np.asarray(
        [baseline_by_seed[int(seed)] for seed in seeds],
        dtype=np.float64,
    )
    candidate_success = np.asarray(
        [candidate_by_seed[int(seed)] for seed in seeds],
        dtype=np.float64,
    )
    delta = candidate_success - baseline_success
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(
        0,
        len(seeds),
        size=(bootstrap_samples, len(seeds)),
    )
    boot = delta[indices].mean(axis=1)
    ci = [
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    ]
    wins = int(np.sum(delta > 0))
    losses = int(np.sum(delta < 0))
    mean_delta = float(delta.mean())
    passed = (
        mean_delta > min_delta
        and wins > losses
        and ci[0] >= min_ci_lower
    )
    reasons = []
    if mean_delta <= min_delta:
        reasons.append(
            f"delta {mean_delta:+.4f} is not above {min_delta:+.4f}"
        )
    if wins <= losses:
        reasons.append(f"wins/losses are not positive ({wins}/{losses})")
    if ci[0] < min_ci_lower:
        reasons.append(
            f"CI lower {ci[0]:+.4f} is below {min_ci_lower:+.4f}"
        )
    return {
        "status": "ok",
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "episodes": int(len(seeds)),
        "seed_start": int(seeds.min()),
        "seed_end": int(seeds.max()),
        "baseline_success": float(baseline_success.mean()),
        "candidate_success": float(candidate_success.mean()),
        "paired_delta": mean_delta,
        "paired_delta_ci95": ci,
        "paired_wins": wins,
        "paired_losses": losses,
        "paired_ties": int(len(seeds) - wins - losses),
        "mcnemar_exact_p": _exact_mcnemar_p(wins, losses),
        "min_delta": float(min_delta),
        "min_ci_lower": float(min_ci_lower),
        "gate": "pass" if passed else "fail",
        "gate_reason": (
            "paired superiority gate passed"
            if passed
            else "; ".join(reasons)
        ),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
    }


def main() -> int:
    args = parse_args()
    baseline_path = args.baseline.expanduser().resolve()
    candidate_path = args.candidate.expanduser().resolve()
    payload = summarize(
        _load(baseline_path),
        _load(candidate_path),
        baseline_path=baseline_path,
        candidate_path=candidate_path,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        min_delta=args.min_delta,
        min_ci_lower=args.min_ci_lower,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
