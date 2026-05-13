#!/usr/bin/env python3
"""Export the latest RoboMimic pretrain evaluation row as a one-row CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _numeric_or_none(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _latest_row(rows: list[dict[str, str]]) -> dict[str, str]:
    if not rows:
        raise ValueError("evaluation CSV has no data rows")

    if "iteration" not in rows[0]:
        return rows[-1]

    best_row = rows[-1]
    best_iteration = _numeric_or_none(best_row.get("iteration", ""))
    for row in rows:
        iteration = _numeric_or_none(row.get("iteration", ""))
        if iteration is None:
            continue
        if best_iteration is None or iteration > best_iteration:
            best_row = row
            best_iteration = iteration
    return best_row


def export_final_eval(run_dir: Path, output: Path) -> Path:
    source = run_dir / "pretrain_eval.csv"
    if not source.is_file():
        raise FileNotFoundError(f"missing evaluation CSV: {source}")

    with source.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    row = dict(_latest_row(rows))
    if "source_run_dir" not in fieldnames:
        fieldnames.append("source_run_dir")
    row["source_run_dir"] = str(run_dir)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerow(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    output = export_final_eval(args.run_dir, args.output)
    print(output)


if __name__ == "__main__":
    main()
