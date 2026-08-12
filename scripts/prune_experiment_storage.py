#!/usr/bin/env python3
"""Plan and apply conservative experiment-storage pruning.

The planner only admits completed runs with a multi-checkpoint validation
curve.  It preserves the validation-selected best snapshot, the final
endpoint, and raw step 100000 when present.  Applying a plan revalidates
process activity and every target's filesystem metadata before unlinking.

Replay removal is a separate operation because it preserves evaluation but
removes exact-resume capability.  The manifest records and revalidates the
complete replay directory inventory before removal.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from omegaconf import OmegaConf


SNAPSHOT_RE = re.compile(r"^(\d+)_snapshot\.pkl$")
SEALED_NAME_PARTS = ("heldout", "sealed", "test", "ep200")
NON_SELECTION_NAME_PARTS = ("train", "pretrain")
REPLAY_DIR_NAMES = ("replay", "demo_replay")


@dataclass(frozen=True)
class FileRecord:
    path: str
    size: int
    blocks: int
    device: int
    inode: int
    mtime_ns: int


def file_record(path: Path, root: Path) -> FileRecord:
    stat = path.stat()
    return FileRecord(
        path=str(path.relative_to(root)),
        size=stat.st_size,
        blocks=stat.st_blocks * 512,
        device=stat.st_dev,
        inode=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )


def replay_inventory(path: Path, root: Path) -> dict:
    records = []
    for item in sorted(path.rglob("*")):
        if item.is_file():
            records.append(file_record(item, root))
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            (
                f"{record.path}\0{record.size}\0{record.blocks}\0"
                f"{record.device}\0{record.inode}\0{record.mtime_ns}\n"
            ).encode()
        )
    return {
        "path": str(path.relative_to(root)),
        "file_count": len(records),
        "logical_bytes": sum(record.size for record in records),
        "allocated_bytes": sum(record.blocks for record in records),
        "inventory_sha256": digest.hexdigest(),
    }


def validation_rows(path: Path) -> list[tuple[int, float]]:
    try:
        rows = list(csv.DictReader(path.open(newline="")))
    except (OSError, csv.Error, UnicodeDecodeError):
        return []
    parsed = []
    for row in rows:
        try:
            step = int(float(row.get("env_steps") or row.get("iteration") or ""))
            success = float(row["episode_success"])
        except (KeyError, TypeError, ValueError):
            continue
        try:
            seed_start = row.get("eval_seed_start")
            if seed_start not in (None, "") and int(float(seed_start)) >= 800:
                continue
        except ValueError:
            continue
        parsed.append((step, success))
    return parsed


def validation_candidate(path: Path) -> tuple | None:
    name = path.name.lower()
    if any(part in name for part in SEALED_NAME_PARTS):
        return None
    if any(part in name for part in NON_SELECTION_NAME_PARTS):
        return None
    rows = validation_rows(path)
    if len({step for step, _ in rows}) < 2:
        return None
    if name.startswith("val"):
        priority = 3
    elif "sweep_eval" in name:
        priority = 2
    elif name == "eval.csv":
        priority = 1
    else:
        priority = 0
    episodes = 0
    try:
        raw_rows = list(csv.DictReader(path.open(newline="")))
        episodes = max(
            (int(float(row.get("eval_episodes") or 0)) for row in raw_rows),
            default=0,
        )
    except (OSError, csv.Error, TypeError, ValueError):
        pass
    return priority, episodes, path.stat().st_mtime_ns, path, rows


def sweep_eval_candidate(run_dir: Path) -> tuple | None:
    """Validation curve from ``sweep_evals/eval_<step>.json``.

    The async-eval protocol scores snapshots from a separate process, so those
    runs have no in-run eval CSV and would otherwise be unprunable.  Sealed
    seeds are excluded with the same ``eval_seed_start >= 800`` rule the CSV
    path applies, so held-out scores can never drive checkpoint selection.
    """

    sweep_dir = run_dir / "sweep_evals"
    if not sweep_dir.is_dir():
        return None
    rows: list[tuple[int, float]] = []
    episodes = 0
    newest_mtime_ns = 0
    for path in sorted(sweep_dir.glob("eval_*.json")):
        try:
            record = json.loads(path.read_text())
            step = int(float(record.get("env_steps") or record.get("iteration")))
            success = float(record["episode_success"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        seed_start = record.get("eval_seed_start")
        try:
            if seed_start is not None and int(float(seed_start)) >= 800:
                continue
        except (TypeError, ValueError):
            continue
        try:
            episodes = max(episodes, int(float(record.get("eval_episodes") or 0)))
        except (TypeError, ValueError):
            pass
        newest_mtime_ns = max(newest_mtime_ns, path.stat().st_mtime_ns)
        rows.append((step, success))
    if len({step for step, _ in rows}) < 2:
        return None
    return 2, episodes, newest_mtime_ns, sweep_dir, rows


def select_validation(run_dir: Path) -> tuple[Path, list[tuple[int, float]]] | None:
    candidates = []
    for path in run_dir.glob("*.csv"):
        candidate = validation_candidate(path)
        if candidate is not None:
            candidates.append(candidate)
    sweep = sweep_eval_candidate(run_dir)
    if sweep is not None:
        candidates.append(sweep)
    if not candidates:
        return None
    _, _, _, path, rows = max(candidates, key=lambda item: item[:3])
    return path, rows


def configured_endpoint(run_dir: Path) -> int | None:
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.is_file():
        return None
    try:
        value = OmegaConf.load(config_path).get("num_train_frames")
        return int(value)
    except (OSError, TypeError, ValueError):
        return None


def active_run_dirs(root: Path) -> set[Path]:
    active = set()
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
            cwd = Path(os.readlink(entry / "cwd"))
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        tokens = [token.decode(errors="replace") for token in raw.split(b"\0") if token]
        for index, token in enumerate(tokens):
            value = None
            if token == "--run-dir" and index + 1 < len(tokens):
                value = tokens[index + 1]
            elif token.startswith("--run-dir="):
                value = token.split("=", 1)[1]
            elif token.startswith("hydra.run.dir="):
                value = token.split("=", 1)[1]
            if value:
                candidate = Path(value)
                if not candidate.is_absolute():
                    candidate = cwd / candidate
                try:
                    candidate = candidate.resolve()
                    candidate.relative_to(root)
                except (OSError, ValueError):
                    continue
                active.add(candidate)
    return active


def is_under(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def plan(args: argparse.Namespace) -> dict:
    root = Path(args.root).resolve()
    group_roots = [(root / group).resolve() for group in args.group]
    missing = [str(path) for path in group_roots if not path.is_dir()]
    if missing:
        raise SystemExit(f"missing group directories: {missing}")
    active = active_run_dirs(root)
    cutoff_ns = time.time_ns() - int(args.min_age_hours * 3600 * 1e9)
    runs = []
    skipped = {
        "active": 0,
        "recent": 0,
        "no_validation_curve": 0,
        "incomplete_endpoint": 0,
    }

    for snapshot_dir in sorted(
        path for group in group_roots for path in group.rglob("snapshots")
    ):
        if not snapshot_dir.is_dir():
            continue
        run_dir = snapshot_dir.parent.resolve()
        if any(is_under(run_dir, item) or is_under(item, run_dir) for item in active):
            skipped["active"] += 1
            continue
        snapshots = []
        for path in snapshot_dir.iterdir():
            match = SNAPSHOT_RE.match(path.name)
            if match and path.is_file():
                snapshots.append((int(match.group(1)), path))
        if len(snapshots) < 2:
            continue
        if max(path.stat().st_mtime_ns for _, path in snapshots) >= cutoff_ns:
            skipped["recent"] += 1
            continue
        selection = select_validation(run_dir)
        if selection is None:
            skipped["no_validation_curve"] += 1
            continue
        selection_path, rows = selection
        endpoint = max(step for step, _ in snapshots)
        expected_endpoint = configured_endpoint(run_dir)
        if expected_endpoint is None or endpoint < expected_endpoint:
            skipped["incomplete_endpoint"] += 1
            continue

        best_success = max(success for _, success in rows)
        best_eval_step = min(step for step, success in rows if success == best_success)
        best_snapshot_step = min(
            snapshots,
            key=lambda item: (abs(item[0] - best_eval_step), item[0]),
        )[0]
        keep_steps = {best_snapshot_step, endpoint}
        if any(step == 100000 for step, _ in snapshots):
            keep_steps.add(100000)
        delete = [
            file_record(path, root)
            for step, path in snapshots
            if step not in keep_steps
        ]
        if not delete:
            continue
        replay_dirs = []
        for name in REPLAY_DIR_NAMES:
            path = run_dir / name
            if path.is_dir():
                replay_dirs.append(replay_inventory(path, root))
        runs.append(
            {
                "run_dir": str(run_dir.relative_to(root)),
                "selection_csv": str(selection_path.relative_to(root)),
                "best_eval_step": best_eval_step,
                "best_snapshot_step": best_snapshot_step,
                "best_success": best_success,
                "configured_endpoint": expected_endpoint,
                "endpoint_step": endpoint,
                "keep_steps": sorted(keep_steps),
                "delete_snapshots": [asdict(record) for record in delete],
                "replay_dirs": replay_dirs,
            }
        )

    selected_records = [
        record for run in runs for record in run["delete_snapshots"]
    ]
    inode_records: dict[tuple[int, int], list[FileRecord]] = {}
    for group in group_roots:
        for path in group.rglob("*_snapshot.pkl"):
            if path.is_file() and SNAPSHOT_RE.match(path.name):
                record = file_record(path, root)
                inode_records.setdefault((record.device, record.inode), []).append(record)
    selected_paths = {record["path"] for record in selected_records}
    reclaim = 0
    for records in inode_records.values():
        stat = (root / records[0].path).stat()
        if stat.st_nlink == len(records) and all(
            record.path in selected_paths for record in records
        ):
            reclaim += records[0].blocks

    manifest = {
        "version": 1,
        "created_unix_ns": time.time_ns(),
        "root": str(root),
        "groups": [str(path.relative_to(root)) for path in group_roots],
        "min_age_hours": args.min_age_hours,
        "active_run_dirs_at_plan": sorted(
            str(path.relative_to(root)) for path in active
        ),
        "skipped": skipped,
        "summary": {
            "runs": len(runs),
            "snapshot_paths": len(selected_records),
            "snapshot_logical_bytes": sum(r["size"] for r in selected_records),
            "snapshot_inode_aware_reclaim_bytes": reclaim,
            "replay_dirs": sum(len(run["replay_dirs"]) for run in runs),
            "replay_allocated_bytes": sum(
                item["allocated_bytes"]
                for run in runs
                for item in run["replay_dirs"]
            ),
        },
        "runs": runs,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def validate_file(root: Path, raw: dict) -> Path:
    path = (root / raw["path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"target escapes root: {path}") from exc
    if not path.is_file() or not SNAPSHOT_RE.match(path.name):
        raise RuntimeError(f"snapshot target missing or invalid: {path}")
    current = file_record(path, root)
    for field in ("size", "blocks", "device", "inode", "mtime_ns"):
        if getattr(current, field) != raw[field]:
            raise RuntimeError(f"snapshot changed since plan ({field}): {path}")
    return path


def validate_replay(root: Path, raw: dict, run_dir: Path) -> Path:
    path = (root / raw["path"]).resolve()
    if path.parent != run_dir or path.name not in REPLAY_DIR_NAMES:
        raise RuntimeError(f"invalid replay target: {path}")
    if not path.is_dir():
        raise RuntimeError(f"replay target missing: {path}")
    current = replay_inventory(path, root)
    for field in (
        "file_count",
        "logical_bytes",
        "allocated_bytes",
        "inventory_sha256",
    ):
        if current[field] != raw[field]:
            raise RuntimeError(f"replay changed since plan ({field}): {path}")
    return path


def apply(args: argparse.Namespace) -> dict:
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    root = Path(manifest["root"]).resolve()
    if args.confirm != "DELETE-VALIDATED-EXPERIMENT-STORAGE":
        raise SystemExit("refusing apply: exact --confirm phrase is required")
    active = active_run_dirs(root)
    actions = []
    for run in manifest["runs"]:
        run_dir = (root / run["run_dir"]).resolve()
        if any(is_under(run_dir, item) or is_under(item, run_dir) for item in active):
            raise RuntimeError(f"run became active after planning: {run_dir}")
        for step in run["keep_steps"]:
            keep = run_dir / "snapshots" / f"{step}_snapshot.pkl"
            if not keep.is_file():
                raise RuntimeError(f"required retained snapshot missing: {keep}")
        if args.kind in ("snapshots", "all"):
            targets = [validate_file(root, raw) for raw in run["delete_snapshots"]]
            actions.extend(("snapshot", path) for path in targets)
        if args.kind in ("replay", "all"):
            targets = [
                validate_replay(root, raw, run_dir) for raw in run["replay_dirs"]
            ]
            actions.extend(("replay", path) for path in targets)

    # All targets are validated before the first destructive operation.
    removed = {"snapshot_files": 0, "replay_dirs": 0}
    for kind, path in actions:
        if kind == "snapshot":
            path.unlink()
            removed["snapshot_files"] += 1
        else:
            shutil.rmtree(path)
            removed["replay_dirs"] += 1
    result = {
        "manifest": str(manifest_path.resolve()),
        "kind": args.kind,
        "completed_unix_ns": time.time_ns(),
        **removed,
    }
    result_path = manifest_path.with_suffix(manifest_path.suffix + f".{args.kind}.done")
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    planner = subparsers.add_parser("plan")
    planner.add_argument("--root", default="exp_local")
    planner.add_argument("--group", action="append", required=True)
    planner.add_argument("--min-age-hours", type=float, default=36.0)
    planner.add_argument("--output", required=True)
    applier = subparsers.add_parser("apply")
    applier.add_argument("--manifest", required=True)
    applier.add_argument("--kind", choices=("snapshots", "replay", "all"), required=True)
    applier.add_argument("--confirm", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = plan(args) if args.command == "plan" else apply(args)
    print(json.dumps(result.get("summary", result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
