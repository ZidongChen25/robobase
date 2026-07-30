#!/usr/bin/env python3
"""Cost-aware gate for iterative scalar-flow action readouts.

The first split is a small pilot used only to discard catastrophically bad
Euler-step choices.  The selected step is then rerun on the full validation
seed set and must improve both the checkpoint's own BC tower and its distilled
readout.  Only that winner is evaluated on the disjoint confirmation split.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def _finite_nonnegative(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise argparse.ArgumentTypeError(
            "value must be finite and non-negative"
        )
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--gpu-id", required=True, type=int)
    parser.add_argument(
        "--steps",
        nargs="+",
        type=_positive_int,
        default=[2, 4, 8],
    )
    parser.add_argument(
        "--policy-value-beta",
        type=_finite_nonnegative,
        default=3.0,
    )
    parser.add_argument("--pilot-episodes", type=int, default=20)
    parser.add_argument("--validation-episodes", type=int, default=50)
    parser.add_argument("--validation-seed-start", type=int, default=36000)
    parser.add_argument("--confirmation-episodes", type=int, default=100)
    parser.add_argument("--confirmation-seed-start", type=int, default=37000)
    parser.add_argument("--validation-bc", required=True, type=Path)
    parser.add_argument("--validation-distill", required=True, type=Path)
    parser.add_argument("--confirmation-bc", required=True, type=Path)
    parser.add_argument("--confirmation-distill", required=True, type=Path)
    parser.add_argument(
        "--pilot-noninferiority-delta",
        type=float,
        default=-0.05,
    )
    parser.add_argument("--min-validation-delta", type=float, default=0.02)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    return parser.parse_args()


def _completed_payload(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text()).get("status") == "ok"
    except (json.JSONDecodeError, OSError):
        return False


def _run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as stream:
        stream.write("$ " + " ".join(command) + "\n")
        stream.flush()
        subprocess.run(
            command,
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )


def _run_integrated(
    *,
    args: argparse.Namespace,
    split: str,
    steps: int,
    episodes: int,
    seed_start: int,
) -> Path:
    result_path = args.output_dir / split / f"steps{steps}.json"
    if _completed_payload(result_path):
        return result_path
    command = [
        sys.executable,
        str(Path(__file__).with_name("eval_cqn_flow_policy_value.py")),
        "--run-dir",
        str(args.run_dir),
        "--snapshot",
        str(args.snapshot),
        "--output",
        str(result_path),
        "--work-dir",
        str(args.work_root / split / f"steps{steps}"),
        "--gpu-id",
        str(args.gpu_id),
        "--num-eval-episodes",
        str(episodes),
        "--eval-seed-start",
        str(seed_start),
        "--policy-value-beta",
        f"{args.policy_value_beta:g}",
        "--flow-readout",
        "integrated",
        "--num-flow-steps",
        str(steps),
    ]
    _run_logged(
        command,
        args.output_dir / split / f"steps{steps}.log",
    )
    if not _completed_payload(result_path):
        raise RuntimeError(f"integrated eval did not complete: {result_path}")
    return result_path


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("status") != "ok":
        raise ValueError(f"reference eval is not complete: {path}")
    return payload


def _by_seed(payload: dict, limit: int | None = None) -> dict[int, float]:
    rows = payload["episode_results"]
    if limit is not None:
        rows = rows[:limit]
    result = {
        int(row["seed"]): float(row["episode_success"]) for row in rows
    }
    if not result or len(result) != len(rows):
        raise ValueError("invalid or duplicate episode seeds")
    return result


def paired_stats(
    candidate: dict,
    reference: dict,
    *,
    limit: int | None,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    candidate_by_seed = _by_seed(candidate, limit)
    reference_by_seed = _by_seed(reference, limit)
    if set(candidate_by_seed) != set(reference_by_seed):
        raise ValueError("candidate/reference seed sets do not match")
    seeds = np.asarray(sorted(reference_by_seed), dtype=np.int64)
    candidate_success = np.asarray(
        [candidate_by_seed[int(seed)] for seed in seeds],
        dtype=np.float64,
    )
    reference_success = np.asarray(
        [reference_by_seed[int(seed)] for seed in seeds],
        dtype=np.float64,
    )
    delta = candidate_success - reference_success
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(
        0,
        len(seeds),
        size=(bootstrap_samples, len(seeds)),
    )
    boot = delta[indices].mean(axis=1)
    wins = int(np.sum(delta > 0))
    losses = int(np.sum(delta < 0))
    return {
        "episodes": int(len(seeds)),
        "candidate_success": float(candidate_success.mean()),
        "reference_success": float(reference_success.mean()),
        "paired_delta": float(delta.mean()),
        "paired_delta_ci95": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "paired_wins": wins,
        "paired_losses": losses,
        "paired_ties": int(len(seeds) - wins - losses),
    }


def validation_gate(
    versus_bc: dict,
    versus_distill: dict,
    min_delta: float,
) -> tuple[bool, str]:
    for label, row in (
        ("BC", versus_bc),
        ("distill", versus_distill),
    ):
        delta = float(row["paired_delta"])
        wins = int(row["paired_wins"])
        losses = int(row["paired_losses"])
        if delta + 1e-12 < min_delta:
            return (
                False,
                f"delta vs {label} {delta:+.4f} is below "
                f"{min_delta:+.4f}",
            )
        if wins <= losses:
            return (
                False,
                f"wins/losses vs {label} are not positive "
                f"({wins}/{losses})",
            )
    return True, "full validation improvement gate passed"


def confirmation_gate(
    versus_bc: dict,
    versus_distill: dict,
) -> tuple[bool, str]:
    for label, row in (
        ("BC", versus_bc),
        ("distill", versus_distill),
    ):
        delta = float(row["paired_delta"])
        lower = float(row["paired_delta_ci95"][0])
        wins = int(row["paired_wins"])
        losses = int(row["paired_losses"])
        if delta <= 0.0 or wins <= losses:
            return (
                False,
                f"held-out direction vs {label} is not positive "
                f"(delta={delta:+.4f}, W/L={wins}/{losses})",
            )
        if lower < -0.05:
            return (
                False,
                f"held-out CI lower bound vs {label} "
                f"{lower:+.4f} is below -0.0500",
            )
    return True, "held-out dual-reference gate passed"


def run_gate(args: argparse.Namespace) -> dict:
    if len(set(args.steps)) != len(args.steps):
        raise ValueError("steps must be unique")
    if args.pilot_episodes < 1:
        raise ValueError("pilot-episodes must be positive")
    if args.validation_episodes < args.pilot_episodes:
        raise ValueError("validation must contain the pilot seed prefix")
    if args.confirmation_episodes < 1:
        raise ValueError("confirmation-episodes must be positive")
    if args.bootstrap_samples < 1:
        raise ValueError("bootstrap-samples must be positive")
    for name in (
        "pilot_noninferiority_delta",
        "min_validation_delta",
    ):
        if not math.isfinite(float(getattr(args, name))):
            raise ValueError(f"{name} must be finite")

    for name in (
        "run_dir",
        "snapshot",
        "output_dir",
        "work_root",
        "validation_bc",
        "validation_distill",
        "confirmation_bc",
        "confirmation_distill",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    for path in (
        args.snapshot,
        args.validation_bc,
        args.validation_distill,
        args.confirmation_bc,
        args.confirmation_distill,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_bc = _load(args.validation_bc)
    validation_distill = _load(args.validation_distill)
    confirmation_bc = _load(args.confirmation_bc)
    confirmation_distill = _load(args.confirmation_distill)
    started = time.time()

    pilot_payloads = {}
    pilot_rows = []
    for steps in args.steps:
        path = _run_integrated(
            args=args,
            split="pilot",
            steps=steps,
            episodes=args.pilot_episodes,
            seed_start=args.validation_seed_start,
        )
        candidate = _load(path)
        pilot_payloads[steps] = candidate
        versus_bc = paired_stats(
            candidate,
            validation_bc,
            limit=args.pilot_episodes,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        versus_distill = paired_stats(
            candidate,
            validation_distill,
            limit=args.pilot_episodes,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        pilot_rows.append(
            {
                "steps": int(steps),
                "versus_bc": versus_bc,
                "versus_distill": versus_distill,
            }
        )
    selected_pilot = max(
        pilot_rows,
        key=lambda row: (
            row["versus_bc"]["candidate_success"],
            -row["steps"],
        ),
    )
    selected_steps = int(selected_pilot["steps"])
    pilot_passed = (
        float(selected_pilot["versus_bc"]["paired_delta"])
        + 1e-12
        >= args.pilot_noninferiority_delta
    )

    payload = {
        "status": "ok",
        "run_dir": str(args.run_dir),
        "snapshot": str(args.snapshot),
        "policy_value_beta": float(args.policy_value_beta),
        "steps": [int(value) for value in args.steps],
        "pilot_episodes": int(args.pilot_episodes),
        "pilot_rows": pilot_rows,
        "selected_steps": selected_steps,
        "pilot_gate": "pass" if pilot_passed else "fail",
        "pilot_gate_reason": (
            "selected step is within pilot BC non-inferiority threshold"
            if pilot_passed
            else (
                "selected step delta vs BC "
                f"{selected_pilot['versus_bc']['paired_delta']:+.4f} "
                f"is below {args.pilot_noninferiority_delta:+.4f}"
            )
        ),
        "full_validation": None,
        "validation_gate": "not_run",
        "validation_gate_reason": (
            None if pilot_passed else "pilot gate failed"
        ),
        "confirmation": None,
        "confirmation_gate": "not_run",
        "confirmation_gate_reason": (
            None if pilot_passed else "pilot gate failed"
        ),
    }

    if pilot_passed:
        validation_path = _run_integrated(
            args=args,
            split="validation",
            steps=selected_steps,
            episodes=args.validation_episodes,
            seed_start=args.validation_seed_start,
        )
        validation_candidate = _load(validation_path)
        validation_vs_bc = paired_stats(
            validation_candidate,
            validation_bc,
            limit=None,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed + 1,
        )
        validation_vs_distill = paired_stats(
            validation_candidate,
            validation_distill,
            limit=None,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed + 1,
        )
        passed_validation, validation_reason = validation_gate(
            validation_vs_bc,
            validation_vs_distill,
            args.min_validation_delta,
        )
        payload.update(
            {
                "full_validation": {
                    "steps": selected_steps,
                    "versus_bc": validation_vs_bc,
                    "versus_distill": validation_vs_distill,
                },
                "validation_gate": (
                    "pass" if passed_validation else "fail"
                ),
                "validation_gate_reason": validation_reason,
                "confirmation_gate_reason": (
                    None
                    if passed_validation
                    else "full validation gate failed"
                ),
            }
        )
        if passed_validation:
            confirmation_path = _run_integrated(
                args=args,
                split="confirmation",
                steps=selected_steps,
                episodes=args.confirmation_episodes,
                seed_start=args.confirmation_seed_start,
            )
            confirmation_candidate = _load(confirmation_path)
            confirmation_vs_bc = paired_stats(
                confirmation_candidate,
                confirmation_bc,
                limit=None,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed + 2,
            )
            confirmation_vs_distill = paired_stats(
                confirmation_candidate,
                confirmation_distill,
                limit=None,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed + 2,
            )
            passed_confirmation, confirmation_reason = confirmation_gate(
                confirmation_vs_bc,
                confirmation_vs_distill,
            )
            payload.update(
                {
                    "confirmation": {
                        "steps": selected_steps,
                        "versus_bc": confirmation_vs_bc,
                        "versus_distill": confirmation_vs_distill,
                    },
                    "confirmation_gate": (
                        "pass" if passed_confirmation else "fail"
                    ),
                    "confirmation_gate_reason": confirmation_reason,
                }
            )

    payload["elapsed_seconds"] = time.time() - started
    return payload


def main() -> int:
    args = parse_args()
    output = args.output_dir.expanduser().resolve() / "gate_summary.json"
    try:
        payload = run_gate(args)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except BaseException as exc:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
