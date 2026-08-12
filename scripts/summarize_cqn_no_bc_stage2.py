"""Select Stage-2 reward-propagation arms on the fixed validation split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED_STEPS = (2500, 5000, 7500, 10000)


def _curve(run_dir: Path) -> dict[int, float]:
    path = run_dir / "val50_seeds400.csv"
    rows = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            step = int(float(row["env_steps"]))
            if step in EXPECTED_STEPS:
                rows[step] = float(row["episode_success"])
    missing = sorted(set(EXPECTED_STEPS) - set(rows))
    if missing:
        raise ValueError(f"{path} is missing validation steps {missing}")
    return rows


def _arm(run_dir: Path) -> dict:
    curve = _curve(run_dir)
    best_step, best_success = max(
        curve.items(),
        key=lambda item: (item[1], -item[0]),
    )
    return {
        "run_dir": str(run_dir.resolve()),
        "curve": {str(k): v for k, v in curve.items()},
        "best_step": best_step,
        "best_success": best_success,
    }


def summarize(
    td_dir: Path,
    floor_dir: Path,
    mc_dir: Path,
    mc_floor_dir: Path,
) -> dict:
    arms = {
        "td": _arm(td_dir),
        "floor": _arm(floor_dir),
        "mc": _arm(mc_dir),
        "mc_floor": _arm(mc_floor_dir),
    }
    mc_delta = arms["mc"]["best_success"] - arms["td"]["best_success"]
    mc_floor_delta = (
        arms["mc_floor"]["best_success"] - arms["floor"]["best_success"]
    )
    # Prefer the simpler MC-only arm on exact validation ties.
    selected_name = max(
        ("mc", "mc_floor"),
        key=lambda name: (
            arms[name]["best_success"],
            name == "mc",
        ),
    )
    selected = arms[selected_name]
    matched_delta = mc_delta if selected_name == "mc" else mc_floor_delta
    propagation_pass = (
        selected["best_success"] >= 0.20 and matched_delta >= 0.10
    )
    return {
        "protocol": {
            "selection_seeds": [400, 449],
            "episodes_per_checkpoint": 50,
            "expected_steps": list(EXPECTED_STEPS),
            "tie_break": "earliest checkpoint, then MC-only method",
            "propagation_pass": (
                "selected MC best >= 20% and matched no-MC delta >= 10pp"
            ),
        },
        "arms": arms,
        "contrasts": {
            "mc_minus_td": mc_delta,
            "mc_floor_minus_floor": mc_floor_delta,
        },
        "selected_variant": selected_name,
        "selected_step": selected["best_step"],
        "selected_success": selected["best_success"],
        "propagation_gate": "pass" if propagation_pass else "fail",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--td-run", type=Path, required=True)
    parser.add_argument("--floor-run", type=Path, required=True)
    parser.add_argument("--mc-run", type=Path, required=True)
    parser.add_argument("--mc-floor-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.td_run,
        args.floor_run,
        args.mc_run,
        args.mc_floor_run,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
