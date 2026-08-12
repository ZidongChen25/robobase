#!/usr/bin/env python3
"""Drop ``agent_checkpoint_state`` from archived training snapshots.

A snapshot stores two large trees: ``agent`` (parameters, needed for
evaluation and checkpoint selection) and ``agent_checkpoint_state`` (Adam
moments plus RNG counters, needed only to resume training bit-exactly).  On
CQN-AS runs the optimizer tree is ~51% of every file, and ``snapshot_every_n``
writes one every few thousand steps, so archived intermediates carry hundreds
of gigabytes of resume state that nothing will ever resume from.

Stripping is safe for the retained files because
``JaxAgent.load_checkpoint_state_dict`` already tolerates a missing
``opt_state`` and falls back to a freshly initialized optimizer.

Protected from stripping:

* runs with a live training process (``hydra.run.dir`` / ``--run-dir``);
* anything modified inside ``--min-age-hours``;
* the highest-step snapshot of every run and the ``latest_snapshot.pkl``
  target, which are the resume points;
* every snapshot named as a ``source_run``/``snapshot_step`` pair by a
  ``stage40_branch_manifest.json``, which continues training from it.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prune_experiment_storage import active_run_dirs, is_under  # noqa: E402


STRIPPED_KEY = "agent_checkpoint_state_stripped"


def branch_source_snapshots(root: Path) -> set[Path]:
    """Snapshots a Stage-40 branch was cut from, which may be re-cut."""

    protected: set[Path] = set()
    for manifest in root.glob("**/stage40_branch_manifest.json"):
        try:
            payload = json.loads(manifest.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        source = payload.get("source_run")
        step = payload.get("snapshot_step")
        if not source or step is None:
            continue
        candidate = Path(source)
        if not candidate.is_absolute():
            candidate = root / candidate
        protected.add(
            (candidate / "snapshots" / f"{step}_snapshot.pkl").resolve()
        )
    return protected


def resume_points(snapshot_dir: Path) -> set[Path]:
    """The highest-step snapshot plus whatever ``latest`` points at."""

    keep: set[Path] = set()
    latest = snapshot_dir / "latest_snapshot.pkl"
    if latest.exists():
        try:
            keep.add(latest.resolve())
        except OSError:
            pass
    numbered = []
    for path in snapshot_dir.glob("*_snapshot.pkl"):
        if path.is_symlink():
            continue
        stem = path.name.removesuffix("_snapshot.pkl")
        if stem.isdigit():
            numbered.append((int(stem), path))
    if numbered:
        keep.add(max(numbered)[1].resolve())
    return keep


def candidates(root: Path, group: str, min_age_hours: float) -> list[Path]:
    active = active_run_dirs(root)
    protected = branch_source_snapshots(root / group)
    cutoff = time.time() - min_age_hours * 3600.0
    found: list[Path] = []
    for snapshot_dir in (root / group).glob("**/snapshots"):
        if not snapshot_dir.is_dir():
            continue
        run_dir = snapshot_dir.parent.resolve()
        if any(
            is_under(run_dir, item) or is_under(item, run_dir) for item in active
        ):
            continue
        keep = resume_points(snapshot_dir) | protected
        for path in sorted(snapshot_dir.glob("*_snapshot.pkl")):
            if path.is_symlink():
                continue
            resolved = path.resolve()
            if resolved in keep:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_mtime > cutoff:
                continue
            found.append(path)
    return found


def strip_one(path: Path) -> tuple[int, int]:
    """Rewrite ``path`` without its optimizer tree; return (before, after)."""

    before = path.stat().st_size
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"unexpected snapshot payload type: {type(payload)}")
    if "agent" not in payload:
        raise KeyError("snapshot has no 'agent' tree; refusing to rewrite")
    if STRIPPED_KEY in payload and "agent_checkpoint_state" not in payload:
        return before, before
    payload.pop("agent_checkpoint_state", None)
    payload[STRIPPED_KEY] = True

    temporary = path.with_name(f".{path.name}.strip.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        # Verify the rewrite before it replaces the original.
        with temporary.open("rb") as handle:
            verify = pickle.load(handle)
        if set(verify["agent"]) != set(payload["agent"]):
            raise RuntimeError(f"agent tree changed while rewriting {path}")
        os.utime(temporary, (path.stat().st_atime, path.stat().st_mtime))
        after = temporary.stat().st_size
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return before, after


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--group", default="exp_local")
    parser.add_argument("--min-age-hours", type=float, default=48.0)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    targets = candidates(root, args.group, args.min_age_hours)
    total = sum(path.stat().st_size for path in targets)
    print(f"candidates: {len(targets)} snapshots, {total / 1e9:.1f} GB on disk")
    if not args.apply:
        print("dry run; pass --apply to rewrite")
        for path in targets[:10]:
            print(f"  {path.relative_to(root)}")
        return 0

    reclaimed = 0
    failures: list[tuple[str, str]] = []
    for index, path in enumerate(targets, start=1):
        try:
            before, after = strip_one(path)
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            failures.append((str(path.relative_to(root)), repr(exc)))
            continue
        reclaimed += before - after
        if index % 50 == 0 or index == len(targets):
            print(
                f"  {index}/{len(targets)}  reclaimed {reclaimed / 1e9:.1f} GB",
                flush=True,
            )

    result = {
        "root": str(root),
        "group": args.group,
        "min_age_hours": args.min_age_hours,
        "snapshots_considered": len(targets),
        "snapshots_failed": len(failures),
        "reclaimed_bytes": reclaimed,
        "failures": failures[:50],
        "completed_unix_ns": time.time_ns(),
    }
    print(json.dumps({k: v for k, v in result.items() if k != "failures"}, indent=2))
    if failures:
        print(f"{len(failures)} failures; first: {failures[0]}")
    if args.report:
        Path(args.report).write_text(json.dumps(result, indent=2) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
