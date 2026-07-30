#!/usr/bin/env python3
"""Calibrate and confirm a structured-delta CQN value sidecar policy."""

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
    max_state_rms: float
    max_bc_logprob_drop: float


def _variant(value: str) -> Variant:
    parts = value.split(":")
    if len(parts) != 4 or not parts[0]:
        raise argparse.ArgumentTypeError(
            "variant must be LABEL:MARGIN:MAX_STATE_RMS:MAX_BC_LOGPROB_DROP"
        )
    numbers = []
    for item in parts[1:]:
        number = float(item)
        if not math.isfinite(number) or number < 0.0:
            raise argparse.ArgumentTypeError(
                "variant thresholds must be finite and non-negative"
            )
        numbers.append(number)
    return Variant(parts[0], *numbers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--model-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--gpu-id", required=True, type=int)
    parser.add_argument(
        "--variant",
        action="append",
        type=_variant,
        help="LABEL:MARGIN:MAX_STATE_RMS:MAX_BC_LOGPROB_DROP",
    )
    parser.add_argument("--force-level", type=int, default=1)
    parser.add_argument("--intervention-horizon", type=int, default=4)
    parser.add_argument("--calibration-episodes", type=int, default=50)
    parser.add_argument("--calibration-seed-start", type=int, default=58_000)
    parser.add_argument("--confirmation-episodes", type=int, default=200)
    parser.add_argument("--confirmation-seed-start", type=int, default=59_000)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument(
        "--confirmation-noninferiority-margin",
        type=float,
        default=0.05,
    )
    return parser.parse_args()


def _default_variants() -> list[Variant]:
    return [
        # Frozen from an exact-BC diagnostic rollout: observed value margins
        # were 0.16--0.24 and state RMS was 0.21--0.97.  These thresholds
        # create materially distinct support/coverage regimes before any
        # action-facing outcome is inspected.
        Variant("conservative", 0.235, 0.6, 0.25),
        Variant("medium", 0.235, 0.7, 0.25),
        Variant("wide", 0.235, 0.8, 0.25),
    ]


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
    model: Path,
) -> Path:
    label = "bc" if variant is None else variant.label
    split_dir = args.output_dir / split
    output = split_dir / f"{label}.json"
    if _completed(output):
        return output
    command = [
        sys.executable,
        str(Path(__file__).with_name("eval_cqn_structured_delta_sidecar.py")),
        "--run-dir",
        str(args.run_dir),
        "--snapshot",
        str(args.snapshot),
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
        command.append("--bc-only")
    else:
        command.extend(
            [
                "--model",
                str(model),
                "--model-summary",
                str(args.model_summary),
                "--min-value-margin",
                f"{variant.min_value_margin:g}",
                "--max-state-rms",
                f"{variant.max_state_rms:g}",
                "--max-bc-logprob-drop",
                f"{variant.max_bc_logprob_drop:g}",
            ]
        )
    _run_logged(command, split_dir / f"{label}.log")
    if not _completed(output):
        raise RuntimeError(f"structured sidecar eval did not complete: {output}")
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
        _load_payload(split_dir / "bc.json"),
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
    if args.calibration_episodes < 1 or args.confirmation_episodes < 1:
        raise ValueError("episode counts must be positive")
    if args.bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates must be non-negative")
    if (
        not math.isfinite(args.confirmation_noninferiority_margin)
        or args.confirmation_noninferiority_margin < 0.0
    ):
        raise ValueError("invalid confirmation noninferiority margin")

    args.run_dir = args.run_dir.expanduser().resolve()
    args.snapshot = args.snapshot.expanduser().resolve()
    args.model_summary = args.model_summary.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.work_root = args.work_root.expanduser().resolve()
    for path in (args.snapshot, args.model_summary):
        if not path.is_file():
            raise FileNotFoundError(path)
    model_summary = json.loads(args.model_summary.read_text())
    if model_summary.get("status") != "ok":
        raise ValueError("model summary status is not ok")
    if model_summary.get("gate") != "pass":
        raise ValueError(
            "structured value fidelity gate did not pass; deployment forbidden"
        )
    model = Path(model_summary["model"]["path"]).expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    _run_eval(
        args,
        split="calibration",
        episodes=args.calibration_episodes,
        seed_start=args.calibration_seed_start,
        variant=None,
        model=model,
    )
    for variant in variants:
        _run_eval(
            args,
            split="calibration",
            episodes=args.calibration_episodes,
            seed_start=args.calibration_seed_start,
            variant=variant,
            model=model,
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
        "run_dir": str(args.run_dir),
        "snapshot": str(args.snapshot),
        "model_summary": str(args.model_summary),
        "model": str(model),
        "variants": [
            {
                "label": item.label,
                "min_value_margin": item.min_value_margin,
                "max_state_rms": item.max_state_rms,
                "max_bc_logprob_drop": item.max_bc_logprob_drop,
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
            model=model,
        )
        _run_eval(
            args,
            split="confirmation",
            episodes=args.confirmation_episodes,
            seed_start=args.confirmation_seed_start,
            variant=selected,
            model=model,
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
                        args.output_dir / "confirmation" / "summary.json"
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
