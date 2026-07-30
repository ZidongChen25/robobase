#!/usr/bin/env python3
"""Summarize a CQN-Flow policy/value sweep with paired statistics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    return parser.parse_args()


def _label(payload: dict) -> str:
    beta = payload["policy_value_beta"]
    return "bc" if beta is None else f"beta_{float(beta):g}"


def _exact_mcnemar_p(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(wins, losses) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _load_payloads(input_dir: Path) -> list[dict]:
    payloads = []
    for path in sorted(input_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") != "ok":
            continue
        if "policy_value_beta" not in payload or "episode_results" not in payload:
            continue
        payload["_path"] = str(path.resolve())
        payloads.append(payload)
    if not payloads:
        raise ValueError(f"no completed policy-value eval JSON files in {input_dir}")
    return payloads


def summarize(args: argparse.Namespace) -> dict:
    input_dir = args.input_dir.expanduser().resolve()
    payloads = _load_payloads(input_dir)
    baselines = [
        payload for payload in payloads if payload["policy_value_beta"] is None
    ]
    if len(baselines) != 1:
        raise ValueError(
            f"expected exactly one BC-only baseline, found {len(baselines)}"
        )
    baseline = baselines[0]
    baseline_by_seed = {
        int(row["seed"]): float(row["episode_success"])
        for row in baseline["episode_results"]
    }
    seeds = np.asarray(sorted(baseline_by_seed), dtype=np.int64)
    baseline_success = np.asarray(
        [baseline_by_seed[int(seed)] for seed in seeds],
        dtype=np.float64,
    )
    if len(seeds) < 1:
        raise ValueError("BC-only result contains no episodes")

    rng = np.random.default_rng(args.bootstrap_seed)
    bootstrap_indices = rng.integers(
        0,
        len(seeds),
        size=(args.bootstrap_samples, len(seeds)),
    )
    rows = []
    for payload in sorted(
        payloads,
        key=lambda value: (
            value["policy_value_beta"] is not None,
            -1.0
            if value["policy_value_beta"] is None
            else float(value["policy_value_beta"]),
        ),
    ):
        by_seed = {
            int(row["seed"]): float(row["episode_success"])
            for row in payload["episode_results"]
        }
        if set(by_seed) != set(baseline_by_seed):
            raise ValueError(
                f"{_label(payload)} does not contain the BC seed set"
            )
        success = np.asarray(
            [by_seed[int(seed)] for seed in seeds],
            dtype=np.float64,
        )
        paired_delta = success - baseline_success
        bootstrap_delta = paired_delta[bootstrap_indices].mean(axis=1)
        wins = int(np.sum(paired_delta > 0))
        losses = int(np.sum(paired_delta < 0))
        rows.append(
            {
                "label": _label(payload),
                "policy_value_beta": payload["policy_value_beta"],
                "episodes": int(len(seeds)),
                "success": float(success.mean()),
                "paired_delta_vs_bc": float(paired_delta.mean()),
                "paired_delta_ci95": [
                    float(np.quantile(bootstrap_delta, 0.025)),
                    float(np.quantile(bootstrap_delta, 0.975)),
                ],
                "paired_wins": wins,
                "paired_losses": losses,
                "paired_ties": int(len(seeds) - wins - losses),
                "mcnemar_exact_p": _exact_mcnemar_p(wins, losses),
                "source": payload["_path"],
            }
        )

    candidates = [row for row in rows if row["policy_value_beta"] is not None]
    best_candidate = max(
        candidates,
        key=lambda row: (row["success"], row["policy_value_beta"]),
        default=None,
    )
    return {
        "status": "ok",
        "input_dir": str(input_dir),
        "baseline_success": float(baseline_success.mean()),
        "seed_start": int(seeds.min()),
        "seed_end": int(seeds.max()),
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "best_non_bc_candidate": (
            None if best_candidate is None else best_candidate["label"]
        ),
        "results": rows,
    }


def main() -> int:
    args = parse_args()
    payload = summarize(args)
    output = args.output or args.input_dir / "summary.json"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for row in payload["results"]:
        low, high = row["paired_delta_ci95"]
        print(
            f"{row['label']:>10} success={row['success']:.3f} "
            f"delta={row['paired_delta_vs_bc']:+.3f} "
            f"CI=[{low:+.3f},{high:+.3f}] "
            f"W/L/T={row['paired_wins']}/{row['paired_losses']}/"
            f"{row['paired_ties']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
