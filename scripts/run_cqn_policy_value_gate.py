#!/usr/bin/env python3
"""Run a validation-selected, held-out CQN policy/value control gate.

Every policy mode is evaluated in a fresh subprocess because
``policy_value_beta`` is captured when the JAX action function is built.  The
validation split selects one numeric beta against the checkpoint's own BC
tower.  A disjoint confirmation split is run only when that candidate clears
the predeclared validation improvement threshold.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path


def _finite_nonnegative(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise argparse.ArgumentTypeError("beta must be finite and non-negative")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--gpu-id", required=True, type=int)
    parser.add_argument(
        "--betas",
        nargs="+",
        type=_finite_nonnegative,
        default=[0.0, 0.03, 0.1, 0.3, 1.0, 3.0],
    )
    parser.add_argument("--validation-episodes", type=int, default=50)
    parser.add_argument("--validation-seed-start", type=int, default=36000)
    parser.add_argument("--confirmation-episodes", type=int, default=100)
    parser.add_argument("--confirmation-seed-start", type=int, default=37000)
    parser.add_argument("--min-validation-delta", type=float, default=0.02)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument(
        "--flow-readout",
        choices=("auto", "distill", "integrated"),
        default="auto",
    )
    parser.add_argument("--num-flow-steps", type=int)
    return parser.parse_args()


def _label(beta: float | None) -> str:
    return "bc" if beta is None else f"beta_{beta:g}"


def select_candidate(summary: dict) -> dict:
    """Return the validation-selected numeric beta row.

    Success is primary.  Ties choose larger beta because it places more weight
    on the independent BC prior and is therefore the conservative deployment
    choice.
    """

    candidates = [
        row
        for row in summary["results"]
        if row.get("policy_value_beta") is not None
    ]
    if not candidates:
        raise ValueError("summary contains no numeric policy/value candidate")
    return max(
        candidates,
        key=lambda row: (
            float(row["success"]),
            float(row["policy_value_beta"]),
        ),
    )


def validation_gate(candidate: dict, min_delta: float) -> tuple[bool, str]:
    delta = float(candidate["paired_delta_vs_bc"])
    wins = int(candidate["paired_wins"])
    losses = int(candidate["paired_losses"])
    if delta + 1e-12 < min_delta:
        return (
            False,
            f"paired delta {delta:+.4f} is below {min_delta:+.4f}",
        )
    if wins <= losses:
        return False, f"paired wins/losses are not positive ({wins}/{losses})"
    return True, "validation improvement gate passed"


def confirmation_gate(candidate: dict) -> tuple[bool, str]:
    """Require positive held-out direction and a 5pp non-inferiority bound."""

    delta = float(candidate["paired_delta_vs_bc"])
    lower = float(candidate["paired_delta_ci95"][0])
    wins = int(candidate["paired_wins"])
    losses = int(candidate["paired_losses"])
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
    split: str,
    beta: float | None,
    episodes: int,
    seed_start: int,
) -> Path:
    label = _label(beta)
    split_dir = args.output_dir / split
    result_path = split_dir / f"{label}.json"
    if _completed_payload(result_path):
        return result_path
    work_dir = args.work_root / split / label
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
        str(work_dir),
        "--gpu-id",
        str(args.gpu_id),
        "--num-eval-episodes",
        str(episodes),
        "--eval-seed-start",
        str(seed_start),
        "--policy-value-beta",
        "bc" if beta is None else f"{beta:g}",
        "--flow-readout",
        args.flow_readout,
    ]
    if args.num_flow_steps is not None:
        command.extend(["--num-flow-steps", str(args.num_flow_steps)])
    _run_logged(command, split_dir / f"{label}.log")
    if not _completed_payload(result_path):
        raise RuntimeError(f"variant did not produce a valid result: {result_path}")
    return result_path


def _summarize(args: argparse.Namespace, split: str) -> dict:
    split_dir = args.output_dir / split
    summary_path = split_dir / "summary.json"
    command = [
        sys.executable,
        str(Path(__file__).with_name("summarize_cqn_flow_policy_value.py")),
        "--input-dir",
        str(split_dir),
        "--output",
        str(summary_path),
        "--bootstrap-samples",
        str(args.bootstrap_samples),
    ]
    _run_logged(command, split_dir / "summary.log")
    return json.loads(summary_path.read_text())


def run_gate(args: argparse.Namespace) -> dict:
    if args.validation_episodes < 1 or args.confirmation_episodes < 1:
        raise ValueError("episode counts must be positive")
    if args.num_flow_steps is not None and args.num_flow_steps < 1:
        raise ValueError("num-flow-steps must be positive")
    if not args.betas:
        raise ValueError("at least one numeric beta is required")
    if len(set(args.betas)) != len(args.betas):
        raise ValueError("betas must be unique")
    if not math.isfinite(args.min_validation_delta):
        raise ValueError("min-validation-delta must be finite")

    args.output_dir = args.output_dir.expanduser().resolve()
    args.work_root = args.work_root.expanduser().resolve()
    args.run_dir = args.run_dir.expanduser().resolve()
    args.snapshot = args.snapshot.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.snapshot.is_file():
        raise FileNotFoundError(args.snapshot)

    started = time.time()
    for beta in [None, *args.betas]:
        _run_variant(
            args=args,
            split="validation",
            beta=beta,
            episodes=args.validation_episodes,
            seed_start=args.validation_seed_start,
        )
    validation = _summarize(args, "validation")
    selected = select_candidate(validation)
    passed_validation, validation_reason = validation_gate(
        selected,
        args.min_validation_delta,
    )

    payload = {
        "status": "ok",
        "run_dir": str(args.run_dir),
        "snapshot": str(args.snapshot),
        "betas": [float(beta) for beta in args.betas],
        "flow_readout": args.flow_readout,
        "num_flow_steps": args.num_flow_steps,
        "selection_rule": (
            "maximum validation success; ties choose the largest beta"
        ),
        "min_validation_delta": float(args.min_validation_delta),
        "validation_seed_start": int(args.validation_seed_start),
        "validation_episodes": int(args.validation_episodes),
        "validation_summary": str(
            (args.output_dir / "validation" / "summary.json").resolve()
        ),
        "selected_validation_row": selected,
        "validation_gate": (
            "pass" if passed_validation else "fail"
        ),
        "validation_gate_reason": validation_reason,
        "confirmation_seed_start": int(args.confirmation_seed_start),
        "confirmation_episodes": int(args.confirmation_episodes),
        "confirmation_summary": None,
        "selected_confirmation_row": None,
        "confirmation_gate": "not_run",
        "confirmation_gate_reason": (
            "validation gate failed" if not passed_validation else None
        ),
    }

    if passed_validation:
        selected_beta = float(selected["policy_value_beta"])
        for beta in (None, selected_beta):
            _run_variant(
                args=args,
                split="confirmation",
                beta=beta,
                episodes=args.confirmation_episodes,
                seed_start=args.confirmation_seed_start,
            )
        confirmation = _summarize(args, "confirmation")
        confirmed = select_candidate(confirmation)
        passed_confirmation, confirmation_reason = confirmation_gate(confirmed)
        payload.update(
            {
                "confirmation_summary": str(
                    (
                        args.output_dir
                        / "confirmation"
                        / "summary.json"
                    ).resolve()
                ),
                "selected_confirmation_row": confirmed,
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
        payload = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
