#!/usr/bin/env python3
"""Create an exact, future-replay-free branch from a workspace snapshot."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any


def _contains_mapping_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(
            _contains_mapping_key(child, key) for child in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_mapping_key(child, key) for child in value)
    return False


def _episode_record(path: Path) -> dict[str, Any]:
    try:
        _, episode_index, length, global_index = path.stem.rsplit("_", 3)
    except ValueError as exc:
        raise ValueError(f"Unrecognised replay filename: {path.name}") from exc
    episode_index_i = int(episode_index)
    length_i = int(length)
    global_index_i = int(global_index)
    return {
        "path": path,
        "name": path.name,
        "episode_index": episode_index_i,
        "length": length_i,
        "global_index": global_index_i,
        "end_index": global_index_i + length_i,
    }


def select_snapshot_replay_files(
    replay_dir: Path, state: dict[str, Any]
) -> list[dict[str, Any]]:
    """Select exactly the immutable episodes visible at snapshot creation."""

    add_count = int(state["add_count"])
    num_episodes = int(state["num_episodes"])
    num_transitions = int(state["num_transitions"])
    records = [_episode_record(path) for path in replay_dir.glob("*.npz")]
    selected = [
        record
        for record in records
        if record["episode_index"] < num_episodes
        and record["end_index"] <= add_count
    ]
    selected.sort(key=lambda record: (record["global_index"], record["episode_index"]))

    episode_indices = [record["episode_index"] for record in selected]
    expected_episode_indices = list(range(num_episodes))
    if sorted(episode_indices) != expected_episode_indices:
        raise ValueError(
            f"Replay snapshot expects episode indices 0..{num_episodes - 1}, "
            f"found {sorted(episode_indices)}"
        )
    cursor = 0
    for record in selected:
        if record["global_index"] != cursor:
            raise ValueError(
                f"Replay files are not contiguous at {record['name']}: "
                f"expected {cursor}, found {record['global_index']}"
            )
        cursor = record["end_index"]
    if cursor != add_count:
        raise ValueError(
            f"Replay files end at {cursor}, snapshot add_count is {add_count}"
        )
    if sum(record["length"] for record in selected) != num_transitions:
        raise ValueError(
            "Selected replay transition count does not match snapshot "
            f"num_transitions={num_transitions}"
        )
    return selected


def _link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def prepare_branch(
    *,
    source_run: Path,
    destination_run: Path,
    snapshot_step: int,
    expected_pretrain_step: int | None = None,
    expected_main_loop_iterations: int = 0,
    manifest_name: str = "stage40_branch_manifest.json",
) -> dict[str, Any]:
    source_run = source_run.resolve()
    destination_run = destination_run.resolve()
    snapshot = source_run / "snapshots" / f"{snapshot_step}_snapshot.pkl"
    if not snapshot.is_file():
        raise ValueError(f"Missing source snapshot: {snapshot}")
    if destination_run.exists() and any(destination_run.iterdir()):
        raise ValueError(f"Destination must be absent or empty: {destination_run}")
    if Path(manifest_name).name != manifest_name or not manifest_name.endswith(
        ".json"
    ):
        raise ValueError("manifest_name must be a JSON basename")

    with snapshot.open("rb") as handle:
        payload = pickle.load(handle)
    if expected_pretrain_step is None:
        expected_pretrain_step = snapshot_step
    if int(payload.get("_pretrain_step", -1)) != expected_pretrain_step:
        raise ValueError(
            f"Expected offline pretrain step {expected_pretrain_step}, found "
            f"{payload.get('_pretrain_step')}"
        )
    if (
        int(payload.get("_main_loop_iterations", -1))
        != expected_main_loop_iterations
    ):
        raise ValueError(
            "Expected main-loop iterations "
            f"{expected_main_loop_iterations}, found "
            f"{payload.get('_main_loop_iterations')}"
        )
    for state_key in ("agent", "agent_checkpoint_state"):
        if _contains_mapping_key(
            payload.get(state_key, {}), "dense_return_q_target"
        ):
            raise ValueError(
                "Snapshot agent state unexpectedly stores dense_return_q_target; "
                "the online objective could not be changed safely"
            )

    destination_run.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "source_run": str(source_run),
        "destination_run": str(destination_run),
        "snapshot_step": snapshot_step,
        "pretrain_step": int(payload["_pretrain_step"]),
        "main_loop_iterations": int(payload["_main_loop_iterations"]),
        "expected_pretrain_step": expected_pretrain_step,
        "expected_main_loop_iterations": expected_main_loop_iterations,
        "snapshot_agent_state_keys": sorted(payload["agent"]),
        "snapshot_checkpoint_state_keys": sorted(
            payload.get("agent_checkpoint_state", {})
        ),
        "objective_flags_restored_from_snapshot": False,
        "replay": {},
    }

    for state_key, directory_name in (
        ("replay_buffer", "replay"),
        ("demo_replay_buffer", "demo_replay"),
    ):
        state = payload.get(state_key)
        if state is None:
            raise ValueError(f"Snapshot is missing {state_key}")
        source_replay = source_run / directory_name
        selected = select_snapshot_replay_files(source_replay, state)
        modes: set[str] = set()
        for record in selected:
            modes.add(
                _link_or_copy(
                    record["path"],
                    destination_run / directory_name / record["name"],
                )
            )
        manifest["replay"][directory_name] = {
            "add_count": int(state["add_count"]),
            "num_episodes": int(state["num_episodes"]),
            "num_transitions": int(state["num_transitions"]),
            "file_count": len(selected),
            "first_global_index": selected[0]["global_index"],
            "last_end_index": selected[-1]["end_index"],
            "transfer_modes": sorted(modes),
            "files": [record["name"] for record in selected],
        }

    destination_snapshot = (
        destination_run / "snapshots" / f"{snapshot_step}_snapshot.pkl"
    )
    manifest["snapshot_transfer_mode"] = _link_or_copy(
        snapshot, destination_snapshot
    )
    latest = destination_snapshot.parent / "latest_snapshot.pkl"
    latest.symlink_to(destination_snapshot.name)
    manifest_path = destination_run / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--destination-run", type=Path, required=True)
    parser.add_argument("--snapshot-step", type=int, default=10000)
    parser.add_argument("--expected-pretrain-step", type=int)
    parser.add_argument("--expected-main-loop-iterations", type=int, default=0)
    parser.add_argument(
        "--manifest-name", default="stage40_branch_manifest.json"
    )
    args = parser.parse_args()
    manifest = prepare_branch(
        source_run=args.source_run,
        destination_run=args.destination_run,
        snapshot_step=args.snapshot_step,
        expected_pretrain_step=args.expected_pretrain_step,
        expected_main_loop_iterations=args.expected_main_loop_iterations,
        manifest_name=args.manifest_name,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
