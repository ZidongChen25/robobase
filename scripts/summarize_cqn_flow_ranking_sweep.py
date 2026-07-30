#!/usr/bin/env python3
"""Summarize source-resampling ranking probes across checkpoints and R_action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--required-snapshots", nargs="+")
    parser.add_argument("--gate-action-flow-samples", type=int, default=16)
    parser.add_argument("--max-flip-rate", type=float, default=0.10)
    return parser.parse_args()


def _snapshot_step(payload: dict) -> int:
    stem = Path(payload["snapshot"]).stem
    prefix = stem.split("_", 1)[0]
    try:
        return int(prefix)
    except ValueError as exc:
        raise ValueError(f"cannot infer snapshot step from {stem!r}") from exc


def summarize(
    input_dir: Path,
    *,
    required_snapshots: list[int] | None,
    gate_action_flow_samples: int,
    max_flip_rate: float,
) -> dict:
    rows = []
    seen = set()
    for path in sorted(input_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") != "ok" or "metrics" not in payload:
            continue
        step = _snapshot_step(payload)
        samples = int(payload["probe_action_flow_samples"])
        key = (step, samples)
        if key in seen:
            raise ValueError(f"duplicate probe for snapshot/R_action {key}")
        seen.add(key)
        flip = np.asarray(
            payload["metrics"]["per_level_bin_flip_rate"],
            dtype=np.float64,
        )
        snr = np.asarray(
            payload["metrics"]["per_level_rank_snr"],
            dtype=np.float64,
        )
        if (
            flip.ndim != 1
            or snr.shape != flip.shape
            or not np.all(np.isfinite(flip))
            or not np.all(np.isfinite(snr))
        ):
            raise ValueError(f"invalid ranking metrics in {path}")
        rows.append(
            {
                "snapshot_step": step,
                "action_flow_samples": samples,
                "per_level_flip_rate": flip.tolist(),
                "mean_flip_rate": float(flip.mean()),
                "max_flip_rate": float(flip.max()),
                "per_level_rank_snr": snr.tolist(),
                "mean_rank_snr": float(snr.mean()),
                "source": str(path.resolve()),
            }
        )

    if not rows:
        raise ValueError(f"no completed ranking probes in {input_dir}")
    rows.sort(key=lambda row: (row["snapshot_step"], row["action_flow_samples"]))
    required = (
        sorted({row["snapshot_step"] for row in rows})
        if required_snapshots is None
        else sorted(set(required_snapshots))
    )

    gate_rows = {
        row["snapshot_step"]: row
        for row in rows
        if row["action_flow_samples"] == gate_action_flow_samples
    }
    missing = [step for step in required if step not in gate_rows]
    if missing:
        gate = "incomplete"
        reason = f"missing R_action={gate_action_flow_samples} for {missing}"
    else:
        failed = [
            step
            for step in required
            if gate_rows[step]["max_flip_rate"] > max_flip_rate + 1e-12
        ]
        if failed:
            gate = "fail"
            reason = (
                f"max per-level flip exceeds {max_flip_rate:.3f} "
                f"at snapshots {failed}"
            )
        else:
            gate = "pass"
            reason = (
                f"all required neighbor snapshots have max per-level flip "
                f"<= {max_flip_rate:.3f}"
            )

    return {
        "status": "ok",
        "input_dir": str(input_dir.resolve()),
        "required_snapshots": required,
        "gate_action_flow_samples": int(gate_action_flow_samples),
        "max_flip_rate_threshold": float(max_flip_rate),
        "gate": gate,
        "gate_reason": reason,
        "results": rows,
    }


def main() -> int:
    args = parse_args()
    required = (
        None
        if args.required_snapshots is None
        else [int(step) for step in args.required_snapshots]
    )
    payload = summarize(
        args.input_dir.expanduser().resolve(),
        required_snapshots=required,
        gate_action_flow_samples=args.gate_action_flow_samples,
        max_flip_rate=args.max_flip_rate,
    )
    output = args.output or args.input_dir / "summary.json"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for row in payload["results"]:
        print(
            f"step={row['snapshot_step']:>5} "
            f"R={row['action_flow_samples']:>2} "
            f"flip(mean/max)={row['mean_flip_rate']:.3f}/"
            f"{row['max_flip_rate']:.3f} "
            f"rank_snr={row['mean_rank_snr']:.3f}"
        )
    print(f"gate={payload['gate']}: {payload['gate_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
