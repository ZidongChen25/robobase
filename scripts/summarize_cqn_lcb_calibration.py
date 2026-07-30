#!/usr/bin/env python3
"""Select one LCB sidecar threshold from paired episode outcomes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _variant(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("variant must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("variant must be LABEL=PATH")
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc", required=True, type=Path)
    parser.add_argument(
        "--variant",
        required=True,
        action="append",
        type=_variant,
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=33_050)
    parser.add_argument(
        "--noninferiority-margin",
        type=float,
        default=0.05,
        help=(
            "Confirmation requires the paired-success CI lower bound to be "
            "at least the negative of this margin."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=("auto", "calibration", "confirmation"),
        default="auto",
    )
    return parser.parse_args()


def _episode_map(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    if payload.get("status") != "ok":
        raise ValueError("evaluation payload did not complete successfully")
    records = payload.get("episode_results", [])
    indexed = {}
    for record in records:
        seed = int(record["seed"])
        if seed in indexed:
            raise ValueError(f"duplicate episode seed: {seed}")
        indexed[seed] = record
    if not indexed:
        raise ValueError("evaluation payload has no episode results")
    return indexed


def _paired_bootstrap(
    deltas: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> list[float | None]:
    if replicates <= 0 or not deltas.size:
        return [None, None]
    rng = np.random.default_rng(seed)
    selected = rng.integers(
        0,
        deltas.size,
        size=(replicates, deltas.size),
    )
    samples = deltas[selected].mean(axis=1)
    return [
        float(value)
        for value in np.percentile(samples, [2.5, 97.5])
    ]


def _exact_mcnemar_p(wins: int, losses: int) -> float:
    discordant = int(wins + losses)
    if discordant == 0:
        return 1.0
    tail = min(wins, losses)
    lower_tail = sum(
        math.comb(discordant, index) for index in range(tail + 1)
    ) / float(2**discordant)
    return float(min(1.0, 2.0 * lower_tail))


def summarize(
    bc: dict[str, Any],
    variants: dict[str, dict[str, Any]],
    *,
    bootstrap_replicates: int,
    seed: int,
    stage: str = "calibration",
    noninferiority_margin: float = 0.05,
) -> dict[str, Any]:
    if stage not in {"calibration", "confirmation"}:
        raise ValueError("stage must be calibration or confirmation")
    if (
        not math.isfinite(noninferiority_margin)
        or noninferiority_margin < 0.0
    ):
        raise ValueError("noninferiority_margin must be finite and non-negative")
    bc_records = _episode_map(bc)
    bc_seeds = sorted(bc_records)
    bc_success = np.asarray(
        [float(bc_records[item]["episode_success"]) for item in bc_seeds],
        dtype=np.float64,
    )
    summaries = {}
    for variant_index, (label, payload) in enumerate(variants.items()):
        records = _episode_map(payload)
        if sorted(records) != bc_seeds:
            raise ValueError(f"{label} episode seeds do not match BC")
        success = np.asarray(
            [float(records[item]["episode_success"]) for item in bc_seeds],
            dtype=np.float64,
        )
        delta = success - bc_success
        wins = int(np.sum(delta > 0))
        losses = int(np.sum(delta < 0))
        ties = int(np.sum(delta == 0))
        total_inferences = sum(
            int(records[item].get("inference_count", 0))
            for item in bc_seeds
        )
        total_overrides = sum(
            int(records[item].get("applied_override_count", 0))
            for item in bc_seeds
        )
        override_episode_mask = np.asarray(
            [
                int(records[item].get("applied_override_count", 0)) > 0
                for item in bc_seeds
            ],
            dtype=bool,
        )
        bc_reward = np.asarray(
            [
                float(
                    bc_records[item].get(
                        "episode_reward",
                        bc_records[item]["episode_success"],
                    )
                )
                for item in bc_seeds
            ],
            dtype=np.float64,
        )
        reward = np.asarray(
            [
                float(
                    records[item].get(
                        "episode_reward",
                        records[item]["episode_success"],
                    )
                )
                for item in bc_seeds
            ],
            dtype=np.float64,
        )
        override_success_delta = delta[override_episode_mask]
        override_reward_delta = (
            reward - bc_reward
        )[override_episode_mask]
        success_rate = float(success.mean())
        bc_success_rate = float(bc_success.mean())
        paired_success_delta_ci = _paired_bootstrap(
            delta,
            replicates=bootstrap_replicates,
            seed=seed + variant_index,
        )
        if stage == "confirmation":
            gate_checks = {
                "success_not_below_bc": success_rate >= bc_success_rate,
                "paired_success_ci_lower_at_least_minus_margin": (
                    paired_success_delta_ci[0] is not None
                    and paired_success_delta_ci[0]
                    >= -float(noninferiority_margin)
                ),
                "produced_override": total_overrides > 0,
                "override_episode_success_delta_positive": (
                    override_success_delta.size > 0
                    and float(override_success_delta.mean()) > 0.0
                ),
                "override_episode_reward_delta_positive": (
                    override_reward_delta.size > 0
                    and float(override_reward_delta.mean()) > 0.0
                ),
            }
        else:
            gate_checks = {
                "success_not_below_bc": success_rate >= bc_success_rate,
                "produced_override": total_overrides > 0,
                "paired_wins_not_below_losses": wins >= losses,
            }
        summaries[label] = {
            "path": payload.get("_source_path"),
            "thresholds": payload.get("thresholds"),
            "episode_success": success_rate,
            "success_delta_vs_bc": float(delta.mean()),
            "paired_success_delta_ci": paired_success_delta_ci,
            "paired_wins": wins,
            "paired_losses": losses,
            "paired_ties": ties,
            "exact_mcnemar_p": _exact_mcnemar_p(wins, losses),
            "total_inferences": total_inferences,
            "total_applied_overrides": total_overrides,
            "override_rate": (
                float(total_overrides / total_inferences)
                if total_inferences
                else 0.0
            ),
            "override_episode_count": int(override_episode_mask.sum()),
            "override_episode_success_delta": (
                float(override_success_delta.mean())
                if override_success_delta.size
                else None
            ),
            "override_episode_success_delta_ci": _paired_bootstrap(
                override_success_delta,
                replicates=bootstrap_replicates,
                seed=seed + 10_000 + variant_index,
            ),
            "override_episode_reward_delta": (
                float(override_reward_delta.mean())
                if override_reward_delta.size
                else None
            ),
            "override_episode_reward_delta_ci": _paired_bootstrap(
                override_reward_delta,
                replicates=bootstrap_replicates,
                seed=seed + 20_000 + variant_index,
            ),
            "gate_checks": gate_checks,
            "gate_passed": all(gate_checks.values()),
        }

    eligible = [
        (label, result)
        for label, result in summaries.items()
        if result["gate_passed"]
    ]
    eligible.sort(
        key=lambda item: (
            -item[1]["episode_success"],
            item[1]["override_rate"],
            item[0],
        )
    )
    selected = eligible[0][0] if eligible else None
    return {
        "status": "ok",
        "stage": stage,
        "selection_protocol": (
            (
                "heldout_noninferiority_and_positive_override_subset"
                if stage == "confirmation"
                else (
                    "highest_calibration_success_among_gate_passes_then_"
                    "lower_override_rate"
                )
            )
        ),
        "eval_seeds": bc_seeds,
        "num_eval_episodes": len(bc_seeds),
        "noninferiority_margin": (
            float(noninferiority_margin)
            if stage == "confirmation"
            else None
        ),
        "bc": {
            "path": bc.get("_source_path"),
            "episode_success": float(bc_success.mean()),
        },
        "variants": summaries,
        "selected_variant": selected,
        "gate_passed": selected is not None,
        "next_gate": (
            (
                "confirmation passed; advance the caller's pre-registered "
                "next stage"
                if selected is not None and stage == "confirmation"
                else (
                    "run paired held-out confirmation on the caller-specified "
                    "disjoint seed range"
                    if selected is not None
                    else (
                        "exact BC fallback; increase branch coverage before "
                        "deployment"
                    )
                )
            )
        ),
    }


def main() -> int:
    args = parse_args()
    if args.bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates must be non-negative")
    if (
        not math.isfinite(args.noninferiority_margin)
        or args.noninferiority_margin < 0.0
    ):
        raise ValueError(
            "noninferiority-margin must be finite and non-negative"
        )
    bc_path = args.bc.expanduser().resolve()
    bc = json.loads(bc_path.read_text())
    bc["_source_path"] = str(bc_path)
    variants = {}
    for label, path in args.variant:
        if label in variants:
            raise ValueError(f"duplicate variant label: {label}")
        resolved = path.expanduser().resolve()
        payload = json.loads(resolved.read_text())
        payload["_source_path"] = str(resolved)
        variants[label] = payload
    stage = args.stage
    if stage == "auto":
        # The held-out protocol uses 100 never-selected seeds and writes under
        # a confirm directory. Either signal is sufficient so an already
        # running durable launcher can pick up confirmation semantics.
        episode_count = len(_episode_map(bc))
        stage = (
            "confirmation"
            if episode_count >= 100 or "confirm" in str(args.output).lower()
            else "calibration"
        )
    result = summarize(
        bc,
        variants,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        stage=stage,
        noninferiority_margin=args.noninferiority_margin,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
