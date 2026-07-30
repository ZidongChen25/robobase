#!/usr/bin/env python3
"""Resolve the pre-registered Stage-149 primary snapshot for a run.

Rule (cqn-flow.md 30.3, frozen before evaluation): the saved snapshot whose
step is nearest to the internal validation-best eval step (best success in
eval.csv; ties on success -> earlier eval step; ties on distance -> earlier
snapshot).  Prints "<primary_snapshot> <final_snapshot>".
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


def main() -> None:
    run_dir = Path(sys.argv[1])
    eval_rows = list(csv.DictReader(open(run_dir / "eval.csv")))
    rows = [
        (float(r["env_steps"]), float(r["episode_success"]))
        for r in eval_rows
        if r.get("env_steps") not in (None, "", "env_steps")
    ]
    if not rows:
        raise SystemExit(f"no eval rows in {run_dir}")
    best_success = max(s for _, s in rows)
    best_step = min(step for step, s in rows if s == best_success)

    snapshots = []
    for path in (run_dir / "snapshots").glob("*_snapshot.pkl"):
        match = re.match(r"(\d+)_snapshot\.pkl", path.name)
        if match:
            snapshots.append((int(match.group(1)), path))
    if not snapshots:
        raise SystemExit(f"no numbered snapshots in {run_dir}")
    snapshots.sort()
    primary = min(
        snapshots,
        key=lambda item: (abs(item[0] - best_step), item[0]),
    )[1]
    final = snapshots[-1][1]
    print(primary, final)


if __name__ == "__main__":
    main()
