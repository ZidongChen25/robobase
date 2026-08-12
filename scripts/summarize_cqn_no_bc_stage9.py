#!/usr/bin/env python3
"""Select dense-return seed-2/3 checkpoints over the combined 2.5k--20k curve."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


OLD_STEPS = (2500, 5000, 7500, 10000)
NEW_STEPS = (12500, 15000, 17500, 20000)


def _read_curve(path: Path, expected_steps: tuple[int, ...]) -> dict[int, float]:
    curve = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                step = int(float(row["env_steps"]))
            except (TypeError, ValueError):
                continue
            if step in expected_steps:
                curve[step] = float(row["episode_success"])
    missing = sorted(set(expected_steps) - set(curve))
    if missing:
        raise ValueError(f"{path} is missing validation steps {missing}")
    return curve


def _arm(run_dir: Path) -> dict:
    old_curve = _read_curve(
        run_dir / "val50_seeds400.csv",
        OLD_STEPS,
    )
    new_curve = _read_curve(
        run_dir / "val50_ext20k_seeds400.csv",
        NEW_STEPS,
    )
    old_best_step, old_best = max(
        old_curve.items(),
        key=lambda item: (item[1], -item[0]),
    )
    combined = old_curve | new_curve
    best_step, best_success = max(
        combined.items(),
        key=lambda item: (item[1], -item[0]),
    )
    return {
        "run_dir": str(run_dir.resolve()),
        "old_curve": {
            str(step): value for step, value in old_curve.items()
        },
        "new_curve": {
            str(step): value for step, value in new_curve.items()
        },
        "old_best_step": old_best_step,
        "old_best_success": old_best,
        "extended_best_step": best_step,
        "extended_best_success": best_success,
        "improvement": best_success - old_best,
        "selected_snapshot": str(
            (run_dir / "snapshots" / f"{best_step}_snapshot.pkl").resolve()
        ),
    }


def summarize(seed2_dir: Path, seed3_dir: Path) -> dict:
    arms = {
        "seed2": _arm(seed2_dir),
        "seed3": _arm(seed3_dir),
    }
    mean_best = sum(
        arm["extended_best_success"] for arm in arms.values()
    ) / len(arms)
    each_improves = all(
        arm["improvement"] >= 0.15 for arm in arms.values()
    )
    gate_pass = each_improves and mean_best >= 0.68
    return {
        "protocol": {
            "selection_seeds": [400, 449],
            "episodes_per_checkpoint": 50,
            "old_steps": list(OLD_STEPS),
            "new_steps": list(NEW_STEPS),
            "checkpoint_tie_break": "earliest checkpoint",
            "heldout_status": "sealed",
            "gate": (
                "both seeds improve >=15pp and extended-best mean >=68%"
            ),
        },
        "runs": arms,
        "mean_extended_best": mean_best,
        "each_improves_15pp": each_improves,
        "compute_gate": "pass" if gate_pass else "fail",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed2-run", type=Path, required=True)
    parser.add_argument("--seed3-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.seed2_run, args.seed3_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
