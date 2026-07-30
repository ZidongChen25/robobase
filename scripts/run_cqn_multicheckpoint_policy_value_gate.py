#!/usr/bin/env python3
"""Select CQN policy/value readouts jointly across checkpoints and betas."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("snapshot must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("snapshot must be LABEL=PATH")
    return label, Path(path)


def _finite_nonnegative(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise argparse.ArgumentTypeError(
            "beta must be finite and non-negative"
        )
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--snapshot",
        required=True,
        action="append",
        type=_labeled_path,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--gpu-id", required=True, type=int)
    parser.add_argument(
        "--betas",
        nargs="+",
        type=_finite_nonnegative,
        default=[0.0, 0.03, 0.1, 0.3, 1.0, 3.0],
    )
    parser.add_argument(
        "--flow-readout",
        choices=("auto", "distill", "integrated"),
        default="auto",
    )
    parser.add_argument("--num-flow-steps", type=int)
    parser.add_argument("--validation-episodes", type=int, default=50)
    parser.add_argument("--validation-seed-start", type=int, default=46000)
    parser.add_argument("--confirmation-episodes", type=int, default=100)
    parser.add_argument("--confirmation-seed-start", type=int, default=47000)
    parser.add_argument("--min-validation-delta", type=float, default=0.02)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    return parser.parse_args()


def _number_label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class Variant:
    label: str
    snapshot_label: str
    snapshot: Path
    beta: float | None
    kind: str
    snapshot_order: int


def build_variants(args: argparse.Namespace) -> list[Variant]:
    variants = []
    for order, (snapshot_label, snapshot) in enumerate(args.snapshot):
        variants.append(
            Variant(
                label=f"{snapshot_label}_bc",
                snapshot_label=snapshot_label,
                snapshot=snapshot,
                beta=None,
                kind="baseline",
                snapshot_order=order,
            )
        )
        variants.extend(
            Variant(
                label=f"{snapshot_label}_beta_{_number_label(beta)}",
                snapshot_label=snapshot_label,
                snapshot=snapshot,
                beta=float(beta),
                kind="candidate",
                snapshot_order=order,
            )
            for beta in args.betas
        )
    labels = [variant.label for variant in variants]
    if len(labels) != len(set(labels)):
        raise ValueError("variant labels are not unique")
    return variants


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


def _run_variant(
    *,
    args: argparse.Namespace,
    variant: Variant,
    split: str,
    episodes: int,
    seed_start: int,
) -> Path:
    split_dir = args.output_dir / split
    result_path = split_dir / f"{variant.label}.json"
    if _completed_payload(result_path):
        return result_path
    command = [
        sys.executable,
        str(Path(__file__).with_name("eval_cqn_flow_policy_value.py")),
        "--run-dir",
        str(args.run_dir),
        "--snapshot",
        str(variant.snapshot),
        "--output",
        str(result_path),
        "--work-dir",
        str(args.work_root / split / variant.label),
        "--gpu-id",
        str(args.gpu_id),
        "--num-eval-episodes",
        str(episodes),
        "--eval-seed-start",
        str(seed_start),
        "--policy-value-beta",
        "bc" if variant.beta is None else f"{variant.beta:g}",
        "--flow-readout",
        args.flow_readout,
    ]
    if args.num_flow_steps is not None:
        command.extend(["--num-flow-steps", str(args.num_flow_steps)])
    _run_logged(command, split_dir / f"{variant.label}.log")
    if not _completed_payload(result_path):
        raise RuntimeError(f"variant did not produce a valid result: {result_path}")
    return result_path


def _success(payload: dict) -> float:
    return float(payload["episode_success"])


def select_variant(
    variants: list[Variant],
    payloads: dict[str, dict],
    *,
    kind: str,
) -> Variant:
    eligible = [variant for variant in variants if variant.kind == kind]
    if not eligible:
        raise ValueError(f"no {kind} variants")

    def key(variant: Variant) -> tuple[float, float, int]:
        # Candidate ties prefer stronger BC regularization; all ties then
        # prefer the earlier validation-best checkpoint.
        beta_priority = -1.0 if variant.beta is None else variant.beta
        return (
            _success(payloads[variant.label]),
            beta_priority,
            -variant.snapshot_order,
        )

    return max(eligible, key=key)


def _exact_mcnemar_p(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(wins, losses) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def paired_row(
    candidate: Variant,
    baseline: Variant,
    payloads: dict[str, dict],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    baseline_by_seed = {
        int(row["seed"]): float(row["episode_success"])
        for row in payloads[baseline.label]["episode_results"]
    }
    candidate_by_seed = {
        int(row["seed"]): float(row["episode_success"])
        for row in payloads[candidate.label]["episode_results"]
    }
    if set(candidate_by_seed) != set(baseline_by_seed):
        raise ValueError("candidate and baseline do not share the same seeds")
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
    wins = int(np.sum(delta > 0))
    losses = int(np.sum(delta < 0))
    return {
        "label": candidate.label,
        "snapshot_label": candidate.snapshot_label,
        "snapshot": str(candidate.snapshot),
        "policy_value_beta": candidate.beta,
        "episodes": int(len(seeds)),
        "success": float(candidate_success.mean()),
        "baseline_label": baseline.label,
        "baseline_snapshot_label": baseline.snapshot_label,
        "baseline_success": float(baseline_success.mean()),
        "paired_delta": float(delta.mean()),
        "paired_delta_ci95": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "paired_wins": wins,
        "paired_losses": losses,
        "paired_ties": int(len(seeds) - wins - losses),
        "mcnemar_exact_p": _exact_mcnemar_p(wins, losses),
    }


def validation_gate(row: dict, min_delta: float) -> tuple[bool, str]:
    delta = float(row["paired_delta"])
    wins = int(row["paired_wins"])
    losses = int(row["paired_losses"])
    if delta + 1e-12 < min_delta:
        return False, f"paired delta {delta:+.4f} is below {min_delta:+.4f}"
    if wins <= losses:
        return False, f"paired wins/losses are not positive ({wins}/{losses})"
    return True, "validation improvement gate passed"


def confirmation_gate(row: dict) -> tuple[bool, str]:
    delta = float(row["paired_delta"])
    lower = float(row["paired_delta_ci95"][0])
    wins = int(row["paired_wins"])
    losses = int(row["paired_losses"])
    if delta <= 0.0 or wins <= losses:
        return (
            False,
            "held-out paired direction is not positive "
            f"(delta={delta:+.4f}, W/L={wins}/{losses})",
        )
    if lower < -0.05:
        return (
            False,
            f"held-out CI lower bound {lower:+.4f} is below -0.0500",
        )
    return True, "held-out positive-direction/non-inferiority gate passed"


def _variant_dict(variant: Variant) -> dict:
    return {
        "label": variant.label,
        "snapshot_label": variant.snapshot_label,
        "snapshot": str(variant.snapshot),
        "policy_value_beta": variant.beta,
        "kind": variant.kind,
    }


def _load_payloads(
    variants: list[Variant],
    output_dir: Path,
    split: str,
) -> dict[str, dict]:
    return {
        variant.label: json.loads(
            (output_dir / split / f"{variant.label}.json").read_text()
        )
        for variant in variants
        if (output_dir / split / f"{variant.label}.json").is_file()
    }


def run_gate(args: argparse.Namespace) -> dict:
    if args.validation_episodes < 1 or args.confirmation_episodes < 1:
        raise ValueError("episode counts must be positive")
    if args.num_flow_steps is not None and args.num_flow_steps < 1:
        raise ValueError("num-flow-steps must be positive")
    if len(args.snapshot) < 1:
        raise ValueError("at least one snapshot is required")
    if len({label for label, _ in args.snapshot}) != len(args.snapshot):
        raise ValueError("snapshot labels must be unique")
    if not args.betas or len(set(args.betas)) != len(args.betas):
        raise ValueError("betas must be nonempty and unique")
    if args.bootstrap_samples < 1:
        raise ValueError("bootstrap-samples must be positive")
    if not math.isfinite(args.min_validation_delta):
        raise ValueError("min-validation-delta must be finite")

    args.run_dir = args.run_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.work_root = args.work_root.expanduser().resolve()
    args.snapshot = [
        (label, path.expanduser().resolve()) for label, path in args.snapshot
    ]
    for _, snapshot in args.snapshot:
        if not snapshot.is_file():
            raise FileNotFoundError(snapshot)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = build_variants(args)
    started = time.time()

    for variant in variants:
        _run_variant(
            args=args,
            variant=variant,
            split="validation",
            episodes=args.validation_episodes,
            seed_start=args.validation_seed_start,
        )
    validation_payloads = _load_payloads(
        variants,
        args.output_dir,
        "validation",
    )
    baseline = select_variant(
        variants,
        validation_payloads,
        kind="baseline",
    )
    candidate = select_variant(
        variants,
        validation_payloads,
        kind="candidate",
    )
    rows = [
        paired_row(
            variant,
            baseline,
            validation_payloads,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        for variant in variants
    ]
    selected_validation = next(
        row for row in rows if row["label"] == candidate.label
    )
    passed_validation, validation_reason = validation_gate(
        selected_validation,
        args.min_validation_delta,
    )
    (args.output_dir / "validation" / "summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "selected_baseline": _variant_dict(baseline),
                "selected_candidate": _variant_dict(candidate),
                "results": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    payload = {
        "status": "ok",
        "run_dir": str(args.run_dir),
        "flow_readout": args.flow_readout,
        "num_flow_steps": args.num_flow_steps,
        "variants": [_variant_dict(variant) for variant in variants],
        "selection_rule": (
            "select baseline and candidate independently by validation "
            "success; candidate ties choose larger beta, then earlier snapshot"
        ),
        "selected_baseline": _variant_dict(baseline),
        "selected_candidate": _variant_dict(candidate),
        "selected_validation_row": selected_validation,
        "validation_seed_start": int(args.validation_seed_start),
        "validation_episodes": int(args.validation_episodes),
        "validation_gate": "pass" if passed_validation else "fail",
        "validation_gate_reason": validation_reason,
        "confirmation_seed_start": int(args.confirmation_seed_start),
        "confirmation_episodes": int(args.confirmation_episodes),
        "selected_confirmation_row": None,
        "confirmation_gate": "not_run",
        "confirmation_gate_reason": (
            None if passed_validation else "validation gate failed"
        ),
    }

    if passed_validation:
        confirmation_variants = [baseline, candidate]
        for variant in confirmation_variants:
            _run_variant(
                args=args,
                variant=variant,
                split="confirmation",
                episodes=args.confirmation_episodes,
                seed_start=args.confirmation_seed_start,
            )
        confirmation_payloads = _load_payloads(
            confirmation_variants,
            args.output_dir,
            "confirmation",
        )
        selected_confirmation = paired_row(
            candidate,
            baseline,
            confirmation_payloads,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed + 1,
        )
        passed_confirmation, confirmation_reason = confirmation_gate(
            selected_confirmation
        )
        (args.output_dir / "confirmation" / "summary.json").write_text(
            json.dumps(
                {
                    "status": "ok",
                    "baseline": _variant_dict(baseline),
                    "candidate": _variant_dict(candidate),
                    "result": selected_confirmation,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        payload.update(
            {
                "selected_confirmation_row": selected_confirmation,
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
