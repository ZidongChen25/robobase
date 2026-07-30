#!/usr/bin/env python3
"""Run calibration and held-out confirmation for an LCB sidecar ensemble."""

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
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument(
        "--sidecar-snapshot",
        required=True,
        action="append",
        type=Path,
    )
    parser.add_argument(
        "--sidecar-seed",
        required=True,
        action="append",
        type=int,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--gpu-id", required=True, type=int)
    parser.add_argument(
        "--margins",
        nargs="+",
        type=_finite_nonnegative,
        default=[0.01, 0.02, 0.04, 0.08],
    )
    parser.add_argument("--lcb-scale", type=_finite_nonnegative, default=1.0)
    parser.add_argument(
        "--max-bc-logprob-drop",
        type=_finite_nonnegative,
        default=0.5,
    )
    parser.add_argument("--force-level", type=int, default=1)
    parser.add_argument("--intervention-horizon", type=int, default=4)
    parser.add_argument("--calibration-episodes", type=int, default=50)
    parser.add_argument("--calibration-seed-start", type=int, default=41000)
    parser.add_argument("--confirmation-episodes", type=int, default=100)
    parser.add_argument("--confirmation-seed-start", type=int, default=42000)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument(
        "--confirmation-noninferiority-margin",
        type=_finite_nonnegative,
        default=0.05,
    )
    return parser.parse_args()


def margin_label(margin: float) -> str:
    text = f"{margin:g}".replace("-", "m").replace(".", "p")
    return f"margin_{text}"


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


def _run_eval(
    args: argparse.Namespace,
    *,
    split: str,
    episodes: int,
    seed_start: int,
    margin: float | None,
) -> Path:
    label = "bc" if margin is None else margin_label(margin)
    split_dir = args.output_dir / split
    output = split_dir / f"{label}.json"
    if _completed_payload(output):
        return output
    command = [
        sys.executable,
        str(Path(__file__).with_name("eval_cqn_lcb_sidecar.py")),
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
    if margin is None:
        command.append("--bc-only")
    else:
        for path in args.sidecar_snapshot:
            command.extend(["--sidecar-snapshot", str(path)])
        for seed in args.sidecar_seed:
            command.extend(["--sidecar-seed", str(seed)])
        command.extend(
            [
                "--lcb-scale",
                f"{args.lcb_scale:g}",
                "--min-lcb-margin",
                f"{margin:g}",
                "--max-bc-logprob-drop",
                f"{args.max_bc_logprob_drop:g}",
            ]
        )
    _run_logged(command, split_dir / f"{label}.log")
    if not _completed_payload(output):
        raise RuntimeError(f"LCB eval did not complete: {output}")
    return output


def _summarize(
    args: argparse.Namespace,
    *,
    split: str,
    stage: str,
    margins: list[float],
) -> dict:
    split_dir = args.output_dir / split
    output = split_dir / "summary.json"
    command = [
        sys.executable,
        str(Path(__file__).with_name("summarize_cqn_lcb_calibration.py")),
        "--bc",
        str(split_dir / "bc.json"),
        "--output",
        str(output),
        "--bootstrap-replicates",
        str(args.bootstrap_replicates),
        "--stage",
        stage,
        "--noninferiority-margin",
        f"{args.confirmation_noninferiority_margin:g}",
    ]
    for margin in margins:
        label = margin_label(margin)
        command.extend(
            ["--variant", f"{label}={split_dir / f'{label}.json'}"]
        )
    _run_logged(command, split_dir / "summary.log")
    return json.loads(output.read_text())


def run_gate(args: argparse.Namespace) -> dict:
    if len(args.sidecar_snapshot) < 2:
        raise ValueError("at least two sidecar snapshots are required")
    if len(args.sidecar_seed) != len(args.sidecar_snapshot):
        raise ValueError("sidecar seed/path counts must match")
    if len(set(args.sidecar_seed)) != len(args.sidecar_seed):
        raise ValueError("sidecar seeds must be unique")
    if not args.margins or len(set(args.margins)) != len(args.margins):
        raise ValueError("margins must be non-empty and unique")
    if args.force_level < 0 or args.intervention_horizon < 1:
        raise ValueError("invalid force level/intervention horizon")
    if args.calibration_episodes < 1 or args.confirmation_episodes < 1:
        raise ValueError("episode counts must be positive")

    args.run_dir = args.run_dir.expanduser().resolve()
    args.snapshot = args.snapshot.expanduser().resolve()
    args.sidecar_snapshot = [
        path.expanduser().resolve() for path in args.sidecar_snapshot
    ]
    args.output_dir = args.output_dir.expanduser().resolve()
    args.work_root = args.work_root.expanduser().resolve()
    for path in [args.snapshot, *args.sidecar_snapshot]:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    _run_eval(
        args,
        split="calibration",
        episodes=args.calibration_episodes,
        seed_start=args.calibration_seed_start,
        margin=None,
    )
    for margin in args.margins:
        _run_eval(
            args,
            split="calibration",
            episodes=args.calibration_episodes,
            seed_start=args.calibration_seed_start,
            margin=margin,
        )
    calibration = _summarize(
        args,
        split="calibration",
        stage="calibration",
        margins=args.margins,
    )
    selected_label = calibration["selected_variant"]
    payload = {
        "status": "ok",
        "run_dir": str(args.run_dir),
        "snapshot": str(args.snapshot),
        "sidecar_snapshots": [str(path) for path in args.sidecar_snapshot],
        "sidecar_seeds": list(args.sidecar_seed),
        "margins": list(args.margins),
        "confirmation_noninferiority_margin": float(
            args.confirmation_noninferiority_margin
        ),
        "calibration_summary": str(
            (args.output_dir / "calibration" / "summary.json").resolve()
        ),
        "selected_variant": selected_label,
        "calibration_gate": (
            "pass" if selected_label is not None else "fail"
        ),
        "confirmation_summary": None,
        "confirmation_gate": "not_run",
    }

    if selected_label is not None:
        selected_row = calibration["variants"][selected_label]
        selected_margin = float(
            selected_row["thresholds"]["min_lcb_margin"]
        )
        _run_eval(
            args,
            split="confirmation",
            episodes=args.confirmation_episodes,
            seed_start=args.confirmation_seed_start,
            margin=None,
        )
        _run_eval(
            args,
            split="confirmation",
            episodes=args.confirmation_episodes,
            seed_start=args.confirmation_seed_start,
            margin=selected_margin,
        )
        confirmation = _summarize(
            args,
            split="confirmation",
            stage="confirmation",
            margins=[selected_margin],
        )
        payload.update(
            {
                "selected_margin": selected_margin,
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
