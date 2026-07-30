"""Build and audit fixed RoboVerse initial-state randomization cohorts."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import pickle
from pathlib import Path
import random
from typing import Any

import numpy as np

from benchmarks.official_bigym.a2a_upstream import file_sha256
from benchmarks.official_roboverse.audit_proxy_data import (
    HASH_SPEC as PROVENANCE_HASH_SPEC,
    SCHEMA as PROVENANCE_SCHEMA,
)


SCHEMA = "roboverse_random_initialization_v1"


def _load_v2(path: Path) -> dict[str, list[dict[str, Any]]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as file:
        payload = pickle.load(file)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Invalid RoboVerse v2 trajectory payload: {path}")
    return payload


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _quat_multiply_wxyz(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = first
    w2, x2, y2, z2 = second
    result = np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )
    return result / np.linalg.norm(result)


def _quat_angle(first: np.ndarray, second: np.ndarray) -> float:
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    return 2.0 * math.acos(float(np.clip(abs(np.dot(first, second)), 0.0, 1.0)))


def _pose_distances(
    position: np.ndarray,
    rotation: np.ndarray,
    reference_positions: np.ndarray,
    reference_rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    position_distance = np.linalg.norm(reference_positions - position, axis=1)
    rotation_distance = np.asarray(
        [_quat_angle(rotation, reference) for reference in reference_rotations]
    )
    return position_distance, rotation_distance


def generate_close_box_random_initializations(
    *,
    source_trajectory: str | Path,
    output_trajectory: str | Path,
    output_manifest: str | Path,
    training_source_indices: list[int],
    count: int = 50,
    seed: int = 20260721,
    position_jitter_m: float = 0.02,
    yaw_jitter_degrees: float = 5.0,
    near_duplicate_position_threshold_m: float = 0.005,
    near_duplicate_rotation_threshold_degrees: float = 1.0,
) -> dict[str, object]:
    """Generate deterministic, source-disjoint Level-0 pose perturbations.

    The perturbation follows RoboVerse ``PoseRandomCfg`` semantics: uniform
    additive XYZ offsets and a local-Z quaternion update. Candidates remain
    inside the position bounds of the audited training source states.
    """

    source_trajectory = Path(source_trajectory).expanduser().resolve()
    output_trajectory = Path(output_trajectory).expanduser().resolve()
    output_manifest = Path(output_manifest).expanduser().resolve()
    if count < 1:
        raise ValueError("count must be positive")
    if position_jitter_m <= 0 or yaw_jitter_degrees <= 0:
        raise ValueError("randomization ranges must be positive")
    if output_trajectory.exists() or output_manifest.exists():
        raise FileExistsError("Refusing to overwrite a random-initialization artifact")
    if "v2" not in output_trajectory.name:
        raise ValueError("output trajectory filename must contain 'v2'")

    payload = _load_v2(source_trajectory)
    if "franka" not in payload:
        raise ValueError("Close Box trajectory does not contain the Franka arm")
    trajectories = payload["franka"]
    if len(set(training_source_indices)) != len(training_source_indices):
        raise ValueError("training_source_indices must be unique")
    if not training_source_indices or min(training_source_indices) < 0:
        raise ValueError("training_source_indices must be non-empty and non-negative")
    if max(training_source_indices) >= len(trajectories):
        raise ValueError("training source index exceeds the trajectory bank")

    training_states = [trajectories[index]["init_state"] for index in training_source_indices]
    training_hashes = {_canonical_hash(state) for state in training_states}
    positions = np.asarray(
        [state["box_base"]["pos"] for state in training_states], dtype=np.float64
    )
    rotations = np.asarray(
        [state["box_base"]["rot"] for state in training_states], dtype=np.float64
    )
    position_min = positions.min(axis=0)
    position_max = positions.max(axis=0)
    if near_duplicate_position_threshold_m <= 0:
        raise ValueError("near-duplicate position threshold must be positive")
    if near_duplicate_rotation_threshold_degrees <= 0:
        raise ValueError("near-duplicate rotation threshold must be positive")
    near_duplicate_rotation_threshold = math.radians(
        near_duplicate_rotation_threshold_degrees
    )
    rng = random.Random(seed)
    generated: list[dict[str, Any]] = []
    generated_hashes: set[str] = set()
    records: list[dict[str, object]] = []

    attempts = 0
    maximum_attempts = count * 1000
    while len(generated) < count and attempts < maximum_attempts:
        attempts += 1
        anchor_source_index = training_source_indices[len(generated) % len(training_source_indices)]
        state = copy.deepcopy(trajectories[anchor_source_index]["init_state"])
        box = state["box_base"]
        anchor_position = np.asarray(box["pos"], dtype=np.float64)
        offset = np.asarray(
            [
                rng.uniform(-position_jitter_m, position_jitter_m),
                rng.uniform(-position_jitter_m, position_jitter_m),
                0.0,
            ]
        )
        candidate_position = anchor_position + offset
        if np.any(candidate_position < position_min) or np.any(candidate_position > position_max):
            continue
        yaw_degrees = rng.uniform(-yaw_jitter_degrees, yaw_jitter_degrees)
        yaw = math.radians(yaw_degrees)
        yaw_quaternion = np.asarray([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
        candidate_rotation = _quat_multiply_wxyz(
            np.asarray(box["rot"], dtype=np.float64), yaw_quaternion
        )
        position_distance, rotation_distance = _pose_distances(
            candidate_position, candidate_rotation, positions, rotations
        )
        normalized_pose_distance = np.sqrt(
            (position_distance / near_duplicate_position_threshold_m) ** 2
            + (rotation_distance / near_duplicate_rotation_threshold) ** 2
        )
        nearest = int(np.argmin(normalized_pose_distance))
        if normalized_pose_distance[nearest] < 1.0:
            continue

        box["pos"] = candidate_position.astype(np.float32).tolist()
        box["rot"] = candidate_rotation.astype(np.float32).tolist()
        state_hash = _canonical_hash(state)
        if state_hash in training_hashes or state_hash in generated_hashes:
            continue
        generated_hashes.add(state_hash)
        generated.append({"init_state": state})
        records.append(
            {
                "index": len(generated) - 1,
                "anchor_source_index": anchor_source_index,
                "position_offset_m": offset.tolist(),
                "yaw_offset_degrees": yaw_degrees,
                "nearest_training_source_index": training_source_indices[nearest],
                "nearest_training_position_distance_m": float(position_distance[nearest]),
                "nearest_training_rotation_distance_degrees": math.degrees(
                    float(rotation_distance[nearest])
                ),
                "nearest_training_normalized_pose_distance": float(
                    normalized_pose_distance[nearest]
                ),
                "initial_state_sha256": state_hash,
            }
        )

    if len(generated) != count:
        raise RuntimeError(
            f"Generated only {len(generated)} of {count} states after {attempts} attempts"
        )

    output_trajectory.parent.mkdir(parents=True, exist_ok=True)
    with output_trajectory.open("xb") as raw_file:
        with gzip.GzipFile(fileobj=raw_file, mode="wb", mtime=0) as compressed:
            pickle.dump({"franka": generated}, compressed, protocol=4)

    manifest: dict[str, object] = {
        "schema": SCHEMA,
        "task": "close_box",
        "simulator": "isaacsim",
        "generation": "roboverse_pose_randomizer_uniform_additive_proxy",
        "source_trajectory": str(source_trajectory),
        "source_trajectory_sha256": file_sha256(source_trajectory),
        "output_trajectory": str(output_trajectory),
        "output_trajectory_sha256": file_sha256(output_trajectory),
        "seed": seed,
        "count": count,
        "training_source_indices": training_source_indices,
        "training_initial_state_hashes": sorted(training_hashes),
        "position_jitter_m": position_jitter_m,
        "yaw_jitter_degrees": yaw_jitter_degrees,
        "position_bounds_m": [position_min.tolist(), position_max.tolist()],
        "near_duplicate_position_threshold_m": near_duplicate_position_threshold_m,
        "near_duplicate_rotation_threshold_degrees": (
            near_duplicate_rotation_threshold_degrees
        ),
        "minimum_normalized_pose_distance": min(
            float(row["nearest_training_normalized_pose_distance"])
            for row in records
        ),
        "exact_training_overlap_count": 0,
        "unique_generated_state_count": len(generated_hashes),
        "states": records,
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_manifest.with_name(f".{output_manifest.name}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(output_manifest)
    return manifest


def validate_random_initialization_manifest(
    manifest_path: str | Path,
    *,
    expected_count: int = 50,
) -> dict[str, object]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"Invalid random-initialization schema: {manifest_path}")
    if manifest.get("count") != expected_count:
        raise ValueError(f"Expected {expected_count} generated states")
    if manifest.get("exact_training_overlap_count") != 0:
        raise ValueError("Generated initial states overlap training states")
    if manifest.get("unique_generated_state_count") != expected_count:
        raise ValueError("Generated initial states are not unique")
    source = Path(str(manifest.get("source_trajectory", ""))).resolve()
    if file_sha256(source) != manifest.get("source_trajectory_sha256"):
        raise ValueError("Random-initialization source trajectory hash mismatch")
    trajectory = Path(str(manifest.get("output_trajectory", ""))).resolve()
    if file_sha256(trajectory) != manifest.get("output_trajectory_sha256"):
        raise ValueError("Random-initialization trajectory hash mismatch")
    source_payload = _load_v2(source)
    generated_payload = _load_v2(trajectory)
    source_indices = manifest.get("training_source_indices")
    if not isinstance(source_indices, list) or any(
        not isinstance(index, int) or isinstance(index, bool) or index < 0
        for index in source_indices
    ):
        raise ValueError("Invalid training source indices in manifest")
    source_trajectories = source_payload.get("franka", [])
    if not source_indices or max(source_indices) >= len(source_trajectories):
        raise ValueError("Training source indices exceed source trajectory")
    actual_training_hashes = {
        _canonical_hash(source_trajectories[index]["init_state"])
        for index in source_indices
    }
    if sorted(actual_training_hashes) != manifest.get("training_initial_state_hashes"):
        raise ValueError("Training initial-state hashes do not match source trajectory")
    generated_trajectories = generated_payload.get("franka", [])
    if len(generated_trajectories) != expected_count:
        raise ValueError("Generated trajectory count does not match manifest")
    generated_hashes = [
        _canonical_hash(trajectory["init_state"])
        for trajectory in generated_trajectories
    ]
    if len(set(generated_hashes)) != expected_count:
        raise ValueError("Generated trajectory contains duplicate initial states")
    if actual_training_hashes.intersection(generated_hashes):
        raise ValueError("Generated trajectory contains a training initial state")
    return manifest


def validate_random_initialization_dataset_binding(
    manifest_path: str | Path,
    provenance_path: str | Path,
    *,
    dataset: str | Path,
    expected_episodes: int,
    expected_count: int = 50,
) -> tuple[dict[str, object], dict[str, object]]:
    """Bind a generated cohort to the audited source IDs of a training Zarr."""

    manifest = validate_random_initialization_manifest(
        manifest_path, expected_count=expected_count
    )
    provenance_path = Path(provenance_path).expanduser().resolve()
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected = {
        "schema": PROVENANCE_SCHEMA,
        "status": "pass",
        "hash_spec": PROVENANCE_HASH_SPEC,
        "episodes": expected_episodes,
        "errors": [],
        "raw_exact_match": True,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(
                f"Dataset provenance field {key!r}={provenance.get(key)!r}; "
                f"expected {value!r}"
            )
    dataset = Path(dataset).expanduser().resolve()
    if Path(str(provenance.get("dataset", ""))).resolve() != dataset:
        raise ValueError("Dataset provenance does not describe the requested dataset")
    if provenance.get("selected_source_indices") != manifest.get(
        "training_source_indices"
    ):
        raise ValueError(
            "Random-initialization training source IDs do not match dataset provenance"
        )
    logical_hash = provenance.get("logical_content_sha256")
    if not isinstance(logical_hash, str) or len(logical_hash) != 64:
        raise ValueError("Dataset provenance has an invalid logical content hash")
    binding = {
        "path": str(provenance_path),
        "file_sha256": file_sha256(provenance_path),
        "dataset": str(dataset),
        "episodes": expected_episodes,
        "logical_content_sha256": logical_hash,
        "selected_source_indices": provenance["selected_source_indices"],
        "random_initialization_source_ids_match": True,
    }
    return manifest, binding


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-trajectory", type=Path, required=True)
    parser.add_argument("--dataset-provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--position-jitter-m", type=float, default=0.02)
    parser.add_argument("--yaw-jitter-degrees", type=float, default=5.0)
    parser.add_argument(
        "--near-duplicate-position-threshold-m", type=float, default=0.005
    )
    parser.add_argument(
        "--near-duplicate-rotation-threshold-degrees", type=float, default=1.0
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    provenance = json.loads(args.dataset_provenance.read_text(encoding="utf-8"))
    if (
        provenance.get("schema") != PROVENANCE_SCHEMA
        or provenance.get("status") != "pass"
        or provenance.get("errors") != []
        or provenance.get("raw_exact_match") is not True
    ):
        raise ValueError("Dataset provenance must be a passing exact raw-data audit")
    source_indices = provenance.get("selected_source_indices")
    if not isinstance(source_indices, list):
        raise ValueError("Dataset provenance has no selected source indices")
    output_dir = args.output_dir.expanduser().resolve()
    manifest = generate_close_box_random_initializations(
        source_trajectory=args.source_trajectory,
        output_trajectory=output_dir / "close_box_random_init_v2.pkl.gz",
        output_manifest=output_dir / "manifest.json",
        training_source_indices=source_indices,
        count=args.count,
        seed=args.seed,
        position_jitter_m=args.position_jitter_m,
        yaw_jitter_degrees=args.yaw_jitter_degrees,
        near_duplicate_position_threshold_m=(
            args.near_duplicate_position_threshold_m
        ),
        near_duplicate_rotation_threshold_degrees=(
            args.near_duplicate_rotation_threshold_degrees
        ),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA",
    "generate_close_box_random_initializations",
    "main",
    "validate_random_initialization_dataset_binding",
    "validate_random_initialization_manifest",
]
