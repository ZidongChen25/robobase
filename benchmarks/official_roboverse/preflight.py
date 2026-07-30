#!/usr/bin/env python3
"""Validate official source identity and an A2A RoboVerse Zarr dataset."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from benchmarks.official_bigym.a2a_upstream import (
    file_sha256,
    validate_official_checkout,
)
from benchmarks.official_roboverse.protocol import (
    DEFAULT_PAPER_CHECKOUT,
    GLOBAL_EXACT_PROTOCOL_BLOCKERS,
    PAPER_ACTION_DIM,
    PAPER_DEMONSTRATIONS,
    PAPER_IMAGE_SIZE,
    PAPER_SOURCE_COMMIT,
    get_task,
)


@dataclass(frozen=True)
class DatasetAudit:
    path: str
    episodes: int
    frames: int
    image_shape: tuple[int, ...]
    state_shape: tuple[int, ...]
    action_shape: tuple[int, ...]
    state_action_finite: bool


def validate_paper_checkout(
    checkout: str | Path,
    *,
    expected_commit: str = PAPER_SOURCE_COMMIT,
) -> tuple[Path, str]:
    """Reject source drift, including tracked edits on top of the pinned commit."""

    checkout, commit = validate_official_checkout(
        checkout, expected_commit=expected_commit
    )
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(
            "Pinned A2A checkout has tracked modifications; use a clean detached "
            f"worktree before benchmarking:\n{status}"
        )
    return checkout, commit


def audit_zarr_dataset(
    dataset: str | Path,
    *,
    expected_episodes: int = PAPER_DEMONSTRATIONS,
    action_dim: int = PAPER_ACTION_DIM,
    image_size: int = PAPER_IMAGE_SIZE,
) -> DatasetAudit:
    """Fully inspect the arrays instead of trusting converter metadata.

    The upstream converter silently falls back to fewer available demos. Reading
    ``meta/episode_ends`` catches that case and also verifies that every data
    array ends at the final episode boundary.
    """

    if expected_episodes < 1:
        raise ValueError("expected_episodes must be positive.")
    dataset = Path(dataset).expanduser().resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(dataset)
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover - official env dependency path
        raise RuntimeError(
            "Strict dataset preflight requires zarr in the isolated official environment."
        ) from exc

    root = zarr.open_group(str(dataset), mode="r")
    required = ("meta/episode_ends", "data/head_camera", "data/state", "data/action")
    missing = [key for key in required if key not in root]
    if missing:
        raise ValueError(f"Dataset is missing official arrays: {missing}.")

    episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    if episode_ends.ndim != 1 or episode_ends.size == 0:
        raise ValueError("meta/episode_ends must be a non-empty rank-1 array.")
    if np.any(episode_ends <= 0) or np.any(np.diff(episode_ends) <= 0):
        raise ValueError("meta/episode_ends must be positive and strictly increasing.")
    if episode_ends.size != expected_episodes:
        raise ValueError(
            f"Dataset has {episode_ends.size} episodes; expected exactly "
            f"{expected_episodes}. The upstream converter may have silently used "
            "fewer available demonstrations."
        )

    image_shape = tuple(int(value) for value in root["data/head_camera"].shape)
    state_shape = tuple(int(value) for value in root["data/state"].shape)
    action_shape = tuple(int(value) for value in root["data/action"].shape)
    frames = int(episode_ends[-1])
    if any(shape[0] != frames for shape in (image_shape, state_shape, action_shape)):
        raise ValueError(
            "Data-array lengths do not match the final episode boundary: "
            f"frames={frames}, image={image_shape}, state={state_shape}, action={action_shape}."
        )
    if image_shape[1:] != (3, image_size, image_size):
        raise ValueError(
            f"Expected NCHW RGB shape (*, 3, {image_size}, {image_size}), got {image_shape}."
        )
    if state_shape[1:] != (action_dim,) or action_shape[1:] != (action_dim,):
        raise ValueError(
            f"Expected state/action dimension {action_dim}, got "
            f"state={state_shape}, action={action_shape}."
        )
    for key in ("data/state", "data/action"):
        array = root[key]
        chunk_rows = int(array.chunks[0]) if array.chunks else min(frames, 4096)
        for start in range(0, frames, chunk_rows):
            values = np.asarray(array[start : start + chunk_rows])
            if not np.isfinite(values).all():
                raise ValueError(f"{key} contains non-finite values near row {start}.")
    return DatasetAudit(
        path=str(dataset),
        episodes=int(episode_ends.size),
        frames=frames,
        image_shape=image_shape,
        state_shape=state_shape,
        action_shape=action_shape,
        state_action_finite=True,
    )


def run_preflight(
    *,
    task_key: str,
    dataset: str | Path,
    checkout: str | Path = DEFAULT_PAPER_CHECKOUT,
    expected_episodes: int = PAPER_DEMONSTRATIONS,
    simulator: str | None = None,
    allow_proxy: bool = False,
) -> dict[str, object]:
    task = get_task(task_key)
    if not task.is_exact and not allow_proxy:
        raise RuntimeError(
            f"{task.paper_name} has no exact public task mapping: {task.mapping_note} "
            "Pass --allow-proxy only to run the declared non-paper proxy."
        )
    effective_simulator = simulator or task.simulator
    if effective_simulator not in ("isaacsim", "mujoco"):
        raise ValueError("simulator must be 'isaacsim' or 'mujoco'.")
    simulator_matches_paper = effective_simulator == task.simulator
    if not simulator_matches_paper and not allow_proxy:
        raise RuntimeError(
            f"{task.paper_name} uses {task.simulator} in the paper protocol, but "
            f"{effective_simulator} was requested. Pass --allow-proxy to record an "
            "explicit simulator proxy."
        )
    checkout, commit = validate_paper_checkout(checkout)
    trajectory_path = checkout / task.trajectory_relpath
    if not trajectory_path.is_file():
        raise FileNotFoundError(
            f"Pinned source trajectory is missing: {trajectory_path}. "
            "Run the official asset download before preflight."
        )
    trajectory_hash = file_sha256(trajectory_path)
    if trajectory_hash != task.trajectory_sha256:
        raise RuntimeError(
            f"Source trajectory hash mismatch for {task.paper_name}: "
            f"got {trajectory_hash}, expected {task.trajectory_sha256}."
        )
    dataset_audit = audit_zarr_dataset(
        dataset, expected_episodes=expected_episodes
    )
    declared_task_data_controls_match = (
        task.is_exact
        and expected_episodes == PAPER_DEMONSTRATIONS
        and simulator_matches_paper
    )
    return {
        "status": "pass",
        "declared_task_data_controls_match": declared_task_data_controls_match,
        "exact_paper_protocol": False,
        "exact_protocol_blockers": list(GLOBAL_EXACT_PROTOCOL_BLOCKERS),
        "source_checkout": str(checkout),
        "source_commit": commit,
        "simulator": effective_simulator,
        "simulator_matches_paper": simulator_matches_paper,
        "source_trajectory": {
            "path": str(trajectory_path),
            "sha256": trajectory_hash,
        },
        "task": asdict(task),
        "dataset": asdict(dataset_audit),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--official-checkout", type=Path, default=Path(DEFAULT_PAPER_CHECKOUT)
    )
    parser.add_argument("--expected-episodes", type=int, default=PAPER_DEMONSTRATIONS)
    parser.add_argument("--simulator", choices=("isaacsim", "mujoco"))
    parser.add_argument("--allow-proxy", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run_preflight(
            task_key=args.task,
            dataset=args.dataset,
            checkout=args.official_checkout,
            expected_episodes=args.expected_episodes,
            simulator=args.simulator,
            allow_proxy=args.allow_proxy,
        )
    except Exception as exc:
        print(
            json.dumps(
                {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)},
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DatasetAudit",
    "audit_zarr_dataset",
    "run_preflight",
    "validate_paper_checkout",
]
