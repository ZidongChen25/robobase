#!/usr/bin/env python3
"""Summarize validation-selected CQN online curves without using final-step bias.

The training CSV logger may rewrite a file and leave repeated header rows.
This utility ignores those rows, requires every predeclared evaluation step,
selects the earliest checkpoint among best-success ties, and records both the
best-checkpoint comparison and all same-step deltas.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be LABEL=RUN_DIR")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("run label must not be empty")
    return label, Path(raw_path).expanduser()


def _integer_list(value: str) -> list[int]:
    try:
        steps = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected steps must be comma-separated integers"
        ) from exc
    if not steps or any(step < 0 for step in steps):
        raise argparse.ArgumentTypeError(
            "expected steps must contain non-negative integers"
        )
    if len(set(steps)) != len(steps):
        raise argparse.ArgumentTypeError("expected steps must be unique")
    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=_parse_run,
        metavar="LABEL=RUN_DIR",
    )
    parser.add_argument("--baseline")
    parser.add_argument("--challenger")
    parser.add_argument(
        "--expected-steps",
        type=_integer_list,
        default=_integer_list("2500,5000,7500,10000"),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Report missing evaluation steps instead of raising an error.",
    )
    return parser.parse_args()


def _finite_float(value: str, field: str, path: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: invalid {field} value {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{path}: non-finite {field} value {value!r}")
    return result


def read_eval_curve(path: Path) -> dict[int, float]:
    """Read the last success value for each integer evaluation step."""

    if not path.is_file():
        raise FileNotFoundError(path)
    curve: dict[int, float] = {}
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        required = {"iteration", "episode_success"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{path}: expected CSV fields {sorted(required)}, got "
                f"{reader.fieldnames}"
            )
        for row in reader:
            # The logger can append another literal header when it rewrites.
            if row.get("iteration") == "iteration":
                continue
            iteration = _finite_float(row.get("iteration", ""), "iteration", path)
            success = _finite_float(
                row.get("episode_success", ""),
                "episode_success",
                path,
            )
            rounded_iteration = int(round(iteration))
            if not math.isclose(iteration, rounded_iteration, abs_tol=1e-6):
                raise ValueError(
                    f"{path}: non-integral evaluation iteration {iteration}"
                )
            if not 0.0 <= success <= 1.0:
                raise ValueError(
                    f"{path}: episode_success outside [0,1]: {success}"
                )
            curve[rounded_iteration] = success
    if not curve:
        raise ValueError(f"{path}: no evaluation rows")
    return curve


def summarize_run(
    label: str,
    run_dir: Path,
    expected_steps: list[int],
    *,
    allow_incomplete: bool,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    curve = read_eval_curve(run_dir / "eval.csv")
    missing_steps = [step for step in expected_steps if step not in curve]
    if missing_steps and not allow_incomplete:
        raise ValueError(
            f"{label}: missing predeclared eval steps {missing_steps}"
        )
    available = {
        step: curve[step] for step in expected_steps if step in curve
    }
    if not available:
        raise ValueError(f"{label}: no predeclared eval steps are available")
    # max() over (success, -step) makes a tie choose the earlier checkpoint.
    best_step = max(available, key=lambda step: (available[step], -step))
    last_step = max(available)
    return {
        "label": label,
        "run_dir": str(run_dir),
        "complete": not missing_steps,
        "missing_steps": missing_steps,
        "curve": {str(step): value for step, value in available.items()},
        "best_step": int(best_step),
        "best_success": float(available[best_step]),
        "last_step": int(last_step),
        "last_success": float(available[last_step]),
        "mean_synchronized_success": float(
            sum(available.values()) / len(available)
        ),
    }


def compare_runs(
    baseline: dict[str, Any],
    challenger: dict[str, Any],
) -> dict[str, Any]:
    baseline_curve = {
        int(step): float(value) for step, value in baseline["curve"].items()
    }
    challenger_curve = {
        int(step): float(value) for step, value in challenger["curve"].items()
    }
    common_steps = sorted(set(baseline_curve) & set(challenger_curve))
    if not common_steps:
        raise ValueError("baseline and challenger have no common eval steps")
    deltas = {
        str(step): challenger_curve[step] - baseline_curve[step]
        for step in common_steps
    }
    best_delta = (
        float(challenger["best_success"]) - float(baseline["best_success"])
    )
    return {
        "baseline": baseline["label"],
        "challenger": challenger["label"],
        "selection_rule": (
            "maximum predeclared episode_success; earliest step breaks ties"
        ),
        "best_success_delta": best_delta,
        "same_step_success_delta": deltas,
        "mean_synchronized_success_delta": float(
            sum(deltas.values()) / len(deltas)
        ),
        # Stage-XX predeclared a strict best-success screen.  The same-step
        # curve is retained as evidence and is not silently promoted into a
        # post-hoc second pass criterion.
        "gate_rule": "challenger_best_success > baseline_best_success",
        "gate_passed": bool(best_delta > 0.0),
    }


def build_summary(
    runs: list[tuple[str, Path]],
    expected_steps: list[int],
    *,
    baseline_label: str | None,
    challenger_label: str | None,
    allow_incomplete: bool,
) -> dict[str, Any]:
    labels = [label for label, _ in runs]
    if len(set(labels)) != len(labels):
        raise ValueError("run labels must be unique")
    summaries = {
        label: summarize_run(
            label,
            path,
            expected_steps,
            allow_incomplete=allow_incomplete,
        )
        for label, path in runs
    }
    comparison = None
    if (baseline_label is None) != (challenger_label is None):
        raise ValueError("--baseline and --challenger must be provided together")
    if baseline_label is not None:
        if baseline_label not in summaries:
            raise ValueError(f"unknown baseline label {baseline_label!r}")
        if challenger_label not in summaries:
            raise ValueError(f"unknown challenger label {challenger_label!r}")
        comparison = compare_runs(
            summaries[baseline_label],
            summaries[challenger_label],
        )
    return {
        "status": "ok",
        "expected_steps": expected_steps,
        "runs": summaries,
        "comparison": comparison,
    }


def main() -> int:
    args = parse_args()
    payload = build_summary(
        args.run,
        args.expected_steps,
        baseline_label=args.baseline,
        challenger_label=args.challenger,
        allow_incomplete=bool(args.allow_incomplete),
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
