#!/usr/bin/env python3
"""Prune numbered experiment snapshots while retaining reproducibility artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path


SNAPSHOT_RE = re.compile(r"^(?P<step>\d+)_snapshot\.pkl$")
REFERENCE_RE = re.compile(
    r"(?P<path>(?:/home/[A-Za-z0-9_.-]+/)?"
    r"(?:[A-Za-z0-9_.$-]+/)*snapshots/(?P<step>\d+)_snapshot\.pkl)"
)
REFERENCE_SUFFIXES = {
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
}
IGNORED_REFERENCE_PARTS = {
    ".git",
    ".venv",
    "checkpoints",
    "demo_replay",
    "replay",
    "snapshots",
    "wandb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("exp_local"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--delete", action="store_true")
    return parser.parse_args()


def numbered_snapshots(snapshot_dir: Path) -> dict[int, Path]:
    snapshots: dict[int, Path] = {}
    for path in snapshot_dir.iterdir():
        match = SNAPSHOT_RE.match(path.name)
        if match and path.is_file():
            snapshots[int(match.group("step"))] = path
    return snapshots


def active_command_lines() -> list[str]:
    commands = []
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            commands.append(cmdline.read_bytes().replace(b"\0", b" ").decode(errors="replace"))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return commands


def is_active_run(run_dir: Path, workspace: Path, commands: list[str]) -> bool:
    absolute = str(run_dir.resolve())
    try:
        relative = str(run_dir.resolve().relative_to(workspace))
    except ValueError:
        relative = str(run_dir)
    return any(absolute in command or relative in command for command in commands)


def best_snapshot_step(run_dir: Path, available_steps: set[int]) -> tuple[int | None, str | None]:
    metrics_path = run_dir / "pretrain_eval.csv"
    if not metrics_path.is_file():
        return None, None

    try:
        with metrics_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, csv.Error, UnicodeDecodeError):
        return None, None
    if not rows:
        return None, None

    columns = rows[0].keys()
    metric = next((name for name in columns if name.endswith("episode_success")), None)
    if metric is None:
        metric = next((name for name in columns if name.endswith("episode_reward")), None)
    step_column = next(
        (name for name in ("iteration", "env_steps", "step", "global_step") if name in columns),
        None,
    )
    if metric is None or step_column is None:
        return None, None

    candidates: list[tuple[float, int]] = []
    for row in rows:
        try:
            step = int(float(row[step_column]))
            score = float(row[metric])
        except (KeyError, TypeError, ValueError):
            continue
        if step in available_steps:
            candidates.append((score, step))
    if not candidates:
        return None, metric
    _, step = max(candidates, key=lambda item: (item[0], item[1]))
    return step, metric


def referenced_snapshots(workspace: Path) -> set[Path]:
    exact_paths: set[Path] = set()
    roots = [
        workspace / "scripts",
        workspace / "benchmarks",
        workspace,
    ]

    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        iterator = root.rglob("*") if root != workspace else root.iterdir()
        for path in iterator:
            if not path.is_file() or path in seen or path.suffix not in REFERENCE_SUFFIXES:
                continue
            seen.add(path)
            if any(part in IGNORED_REFERENCE_PARTS for part in path.parts):
                continue
            try:
                if path.stat().st_size > 16 * 1024 * 1024:
                    continue
                text = path.read_text(errors="replace")
            except OSError:
                continue
            for match in REFERENCE_RE.finditer(text):
                raw_path = Path(match.group("path"))
                candidates = [raw_path]
                if not raw_path.is_absolute():
                    candidates.extend((workspace / raw_path, path.parent / raw_path))
                for candidate in candidates:
                    if candidate.is_file():
                        exact_paths.add(candidate.resolve())
    return exact_paths


def main() -> None:
    args = parse_args()
    workspace = Path.cwd().resolve()
    root = args.root.resolve()
    commands = active_command_lines()
    exact_references = referenced_snapshots(workspace)

    records = []
    delete_paths: list[Path] = []
    for snapshot_dir in sorted(root.rglob("snapshots")):
        if not snapshot_dir.is_dir():
            continue
        snapshots = numbered_snapshots(snapshot_dir)
        if not snapshots:
            continue
        run_dir = snapshot_dir.parent
        active = is_active_run(run_dir, workspace, commands)
        final_step = max(snapshots)
        best_step, best_metric = best_snapshot_step(run_dir, set(snapshots))

        keep_reasons: dict[int, list[str]] = {final_step: ["final"]}
        if best_step is not None:
            keep_reasons.setdefault(best_step, []).append(f"best:{best_metric}")
        for step, path in snapshots.items():
            if path.resolve() in exact_references:
                keep_reasons.setdefault(step, []).append("explicit_reference")
        if active:
            for step in snapshots:
                keep_reasons.setdefault(step, []).append("active_run")

        for step, path in sorted(snapshots.items()):
            reasons = keep_reasons.get(step, [])
            action = "keep" if reasons else "delete"
            record = {
                "action": action,
                "bytes": path.stat().st_size,
                "path": str(path),
                "reasons": reasons,
                "run_active": active,
                "step": step,
            }
            records.append(record)
            if action == "delete":
                delete_paths.append(path)

    summary = {
        "delete_bytes": sum(record["bytes"] for record in records if record["action"] == "delete"),
        "delete_count": sum(record["action"] == "delete" for record in records),
        "keep_bytes": sum(record["bytes"] for record in records if record["action"] == "keep"),
        "keep_count": sum(record["action"] == "keep" for record in records),
        "root": str(root),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({"summary": summary, "records": records}, indent=2) + "\n")

    if args.delete:
        for path in delete_paths:
            path.unlink()

    print(json.dumps(summary, indent=2))
    print(f"manifest={args.manifest}")
    print(f"deleted={args.delete}")


if __name__ == "__main__":
    main()
