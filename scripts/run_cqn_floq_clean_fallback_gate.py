#!/usr/bin/env python3
"""Calibrate and confirm clean-CQN-AS fallback with FLOQ interventions."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from summarize_cqn_lcb_calibration import summarize
except ImportError:
    from scripts.summarize_cqn_lcb_calibration import summarize


@dataclass(frozen=True)
class Variant:
    label: str
    min_value_margin: float
    max_bc_logprob_drop: float
    max_best_bc_logprob_drop: float
    min_source_win_fraction: float
    min_source_mean_delta: float


def _variant(value: str) -> Variant:
    parts = value.split(":")
    if len(parts) != 6 or not parts[0]:
        raise argparse.ArgumentTypeError(
            "variant must be "
            "LABEL:MARGIN:BC_DROP:BEST_BC_DROP:SOURCE_WIN:SOURCE_DELTA"
        )
    numbers = [float(item) for item in parts[1:]]
    if any(not math.isfinite(item) or item < 0.0 for item in numbers[:3]):
        raise argparse.ArgumentTypeError(
            "margin and log-probability thresholds must be non-negative"
        )
    if not 0.0 <= numbers[3] <= 1.0:
        raise argparse.ArgumentTypeError("SOURCE_WIN must be in [0, 1]")
    if not math.isfinite(numbers[4]):
        raise argparse.ArgumentTypeError("SOURCE_DELTA must be finite")
    return Variant(parts[0], *numbers)


def _default_variants() -> list[Variant]:
    return [
        Variant("conservative", 1.25, 0.25, 0.25, 0.75, 0.0),
        Variant("medium", 1.00, 0.50, 0.50, 0.625, 0.0),
        Variant("wide", 0.75, 0.75, 0.75, 0.50, 0.0),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-run-dir", required=True, type=Path)
    parser.add_argument("--clean-snapshot", required=True, type=Path)
    parser.add_argument("--flow-run-dir", required=True, type=Path)
    parser.add_argument("--flow-snapshot", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--gpu-id", required=True, type=int)
    parser.add_argument("--variant", action="append", type=_variant)
    parser.add_argument("--force-level", type=int, default=1)
    parser.add_argument("--intervention-horizon", type=int, default=4)
    parser.add_argument("--policy-value-beta", type=float, default=1.0)
    parser.add_argument("--calibration-episodes", type=int, default=50)
    parser.add_argument("--calibration-seed-start", type=int, default=60_000)
    parser.add_argument("--confirmation-episodes", type=int, default=200)
    parser.add_argument("--confirmation-seed-start", type=int, default=61_000)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument(
        "--confirmation-noninferiority-margin",
        type=float,
        default=0.05,
    )
    return parser.parse_args()


def _completed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text()).get("status") == "ok"
    except (OSError, json.JSONDecodeError):
        return False


def _run_logged(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        stream.write("$ " + " ".join(command) + "\n")
        stream.flush()
        subprocess.run(
            command,
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )


def _run_eval(
    args: argparse.Namespace,
    *,
    split: str,
    episodes: int,
    seed_start: int,
    variant: Variant | None,
) -> Path:
    label = "clean" if variant is None else variant.label
    split_dir = args.output_dir / split
    output = split_dir / f"{label}.json"
    if _completed(output):
        return output
    command = [
        sys.executable,
        str(Path(__file__).with_name("eval_cqn_floq_clean_fallback.py")),
        "--clean-run-dir",
        str(args.clean_run_dir),
        "--clean-snapshot",
        str(args.clean_snapshot),
        "--output",
        str(output),
        "--work-dir",
        str(args.work_root / split / label),
        "--gpu-id",
        str(args.gpu_id),
        "--num-eval-episodes",
        str(episodes),
        "--eval-seed-start",
        str(seed_start),
        "--force-level",
        str(args.force_level),
        "--intervention-horizon",
        str(args.intervention_horizon),
    ]
    if variant is None:
        command.append("--clean-only")
    else:
        command.extend(
            [
                "--flow-run-dir",
                str(args.flow_run_dir),
                "--flow-snapshot",
                str(args.flow_snapshot),
                "--policy-value-beta",
                f"{args.policy_value_beta:g}",
                "--min-value-margin",
                f"{variant.min_value_margin:g}",
                "--max-bc-logprob-drop",
                f"{variant.max_bc_logprob_drop:g}",
                "--max-best-bc-logprob-drop",
                f"{variant.max_best_bc_logprob_drop:g}",
                "--min-source-win-fraction",
                f"{variant.min_source_win_fraction:g}",
                "--min-source-mean-delta",
                f"{variant.min_source_mean_delta:g}",
            ]
        )
    _run_logged(command, split_dir / f"{label}.log")
    if not _completed(output):
        raise RuntimeError(f"fallback eval did not complete: {output}")
    return output


def _load_payload(path: Path) -> dict:
    payload = json.loads(path.read_text())
    payload["_source_path"] = str(path.resolve())
    return payload


def _summarize(
    args: argparse.Namespace,
    *,
    split: str,
    stage: str,
    variants: list[Variant],
) -> dict:
    split_dir = args.output_dir / split
    payload = summarize(
        _load_payload(split_dir / "clean.json"),
        {
            variant.label: _load_payload(
                split_dir / f"{variant.label}.json"
            )
            for variant in variants
        },
        bootstrap_replicates=args.bootstrap_replicates,
        seed=(
            args.calibration_seed_start + args.calibration_episodes
            if stage == "calibration"
            else args.confirmation_seed_start + args.confirmation_episodes
        ),
        stage=stage,
        noninferiority_margin=args.confirmation_noninferiority_margin,
    )
    (split_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


def run_gate(args: argparse.Namespace) -> dict:
    variants = list(args.variant or _default_variants())
    if not variants or len({item.label for item in variants}) != len(variants):
        raise ValueError("variant labels must be non-empty and unique")
    if args.force_level < 0 or args.intervention_horizon < 1:
        raise ValueError("invalid intervention configuration")
    if args.calibration_episodes < 1 or args.confirmation_episodes < 1:
        raise ValueError("episode counts must be positive")
    if args.bootstrap_replicates < 0:
        raise ValueError("bootstrap-replicates must be non-negative")
    if (
        not math.isfinite(args.policy_value_beta)
        or args.policy_value_beta < 0.0
    ):
        raise ValueError("policy-value-beta must be finite and non-negative")
    if (
        not math.isfinite(args.confirmation_noninferiority_margin)
        or args.confirmation_noninferiority_margin < 0.0
    ):
        raise ValueError("invalid confirmation non-inferiority margin")

    for name in (
        "clean_run_dir",
        "clean_snapshot",
        "flow_run_dir",
        "flow_snapshot",
        "output_dir",
        "work_root",
    ):
        value = getattr(args, name)
        setattr(args, name, value.expanduser().resolve())
    for path in (args.clean_snapshot, args.flow_snapshot):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    _run_eval(
        args,
        split="calibration",
        episodes=args.calibration_episodes,
        seed_start=args.calibration_seed_start,
        variant=None,
    )
    for variant in variants:
        _run_eval(
            args,
            split="calibration",
            episodes=args.calibration_episodes,
            seed_start=args.calibration_seed_start,
            variant=variant,
        )
    calibration = _summarize(
        args,
        split="calibration",
        stage="calibration",
        variants=variants,
    )
    selected_label = calibration["selected_variant"]
    payload = {
        "status": "ok",
        "clean_run_dir": str(args.clean_run_dir),
        "clean_snapshot": str(args.clean_snapshot),
        "flow_run_dir": str(args.flow_run_dir),
        "flow_snapshot": str(args.flow_snapshot),
        "policy_value_beta": float(args.policy_value_beta),
        "variants": [
            {
                "label": item.label,
                "min_value_margin": item.min_value_margin,
                "max_bc_logprob_drop": item.max_bc_logprob_drop,
                "max_best_bc_logprob_drop": item.max_best_bc_logprob_drop,
                "min_source_win_fraction": item.min_source_win_fraction,
                "min_source_mean_delta": item.min_source_mean_delta,
            }
            for item in variants
        ],
        "calibration_summary": str(
            (args.output_dir / "calibration" / "summary.json").resolve()
        ),
        "selected_variant": selected_label,
        "calibration_gate": "pass" if selected_label else "fail",
        "confirmation_summary": None,
        "confirmation_gate": "not_run",
    }
    if selected_label is not None:
        selected = next(
            item for item in variants if item.label == selected_label
        )
        _run_eval(
            args,
            split="confirmation",
            episodes=args.confirmation_episodes,
            seed_start=args.confirmation_seed_start,
            variant=None,
        )
        _run_eval(
            args,
            split="confirmation",
            episodes=args.confirmation_episodes,
            seed_start=args.confirmation_seed_start,
            variant=selected,
        )
        confirmation = _summarize(
            args,
            split="confirmation",
            stage="confirmation",
            variants=[selected],
        )
        payload.update(
            {
                "confirmation_summary": str(
                    (
                        args.output_dir
                        / "confirmation"
                        / "summary.json"
                    ).resolve()
                ),
                "confirmation_gate": (
                    "pass" if confirmation["gate_passed"] else "fail"
                ),
            }
        )
    payload["elapsed_seconds"] = time.time() - started
    return payload


def main() -> int:
    args = parse_args()
    output = args.output_dir.expanduser().resolve() / "gate_summary.json"
    try:
        payload = run_gate(args)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
