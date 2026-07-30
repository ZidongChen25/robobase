#!/usr/bin/env python3
"""Audit a RoboVerse proxy Zarr against every raw successful demonstration.

The ordinary preflight checks schema, dimensions, and finite values. This tool
adds content provenance: it reads every logical Zarr element in bounded row
chunks, decodes every source RGB video, compares every episode bit-for-bit, and
computes a chunk-layout-independent SHA-256 digest.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


SCHEMA = "official_a2a_roboverse_proxy_provenance_v1"
HASH_SPEC = "a2a_proxy_logical_content_v1"
DATA_ARRAYS = (
    ("state", "data/state"),
    ("action", "data/action"),
    ("head_camera", "data/head_camera"),
)


def _canonical_dtype(dtype: np.dtype[Any]) -> np.dtype[Any]:
    dtype = np.dtype(dtype)
    return dtype if dtype.itemsize == 1 else dtype.newbyteorder("<")


def _array_header(name: str, dtype: np.dtype[Any], shape: Sequence[int]) -> bytes:
    """Return the domain-separated header used by the v1 logical hash."""

    canonical_dtype = _canonical_dtype(np.dtype(dtype))
    return (
        name.encode("utf-8")
        + b"\0"
        + str(canonical_dtype).encode("ascii")
        + b"\0"
        + json.dumps(tuple(int(value) for value in shape)).encode("ascii")
        + b"\0"
    )


def _canonical_bytes(values: np.ndarray, dtype: np.dtype[Any]) -> memoryview:
    contiguous = np.ascontiguousarray(values, dtype=_canonical_dtype(dtype))
    return memoryview(contiguous).cast("B")


def _begin_array_hash(
    name: str, dtype: np.dtype[Any], shape: Sequence[int]
) -> Any:
    digest = hashlib.sha256()
    digest.update(_array_header(name, dtype, shape))
    return digest


def _update_array_hash(
    digest: Any, values: np.ndarray, dtype: np.dtype[Any]
) -> None:
    digest.update(_canonical_bytes(values, dtype))


def _hash_numpy_array(name: str, values: np.ndarray) -> str:
    values = np.asarray(values)
    digest = _begin_array_hash(name, values.dtype, values.shape)
    _update_array_hash(digest, values, values.dtype)
    return digest.hexdigest()


def _iter_row_slices(start: int, stop: int, chunk_rows: int) -> Iterable[slice]:
    for row in range(start, stop, chunk_rows):
        yield slice(row, min(row + chunk_rows, stop))


def _duplicate_groups(values: Sequence[str]) -> list[list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(values):
        groups[value].append(index)
    return [indices for indices in groups.values() if len(indices) > 1]


def _file_sha256(path: Path, *, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _decode_rgb_video(path: Path) -> np.ndarray:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:  # pragma: no cover - official env dependency
        raise RuntimeError(
            "Raw provenance audit requires imageio and an FFmpeg backend."
        ) from exc
    return np.asarray(imageio.mimread(path))


def _open_zarr(dataset: Path):
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover - official env dependency
        raise RuntimeError("Proxy provenance audit requires zarr.") from exc
    return zarr.open_group(str(dataset), mode="r")


def _numeric_demo_dirs(raw_success_dir: Path) -> list[tuple[int, Path]]:
    result: list[tuple[int, Path]] = []
    for path in raw_success_dir.glob("demo_*"):
        if not path.is_dir():
            continue
        try:
            index = int(path.name.removeprefix("demo_"))
        except ValueError:
            continue
        if (path / "metadata.json").is_file() and (path / "rgb.mp4").is_file():
            result.append((index, path))
    return sorted(result)


def _load_dataset_metadata(dataset: Path, root: Any) -> tuple[dict[str, Any], bool]:
    metadata_path = dataset / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing converter metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Converter metadata must be a JSON object: {metadata_path}")
    return metadata, metadata == dict(root["meta"].attrs)


def _validate_zarr_structure(root: Any) -> np.ndarray:
    required = ("meta/episode_ends", *(path for _, path in DATA_ARRAYS))
    missing = [path for path in required if path not in root]
    if missing:
        raise ValueError(f"Dataset is missing required arrays: {missing}.")
    episode_ends = np.asarray(root["meta/episode_ends"][:])
    if episode_ends.ndim != 1 or episode_ends.size == 0:
        raise ValueError("meta/episode_ends must be a non-empty rank-1 array.")
    if not np.issubdtype(episode_ends.dtype, np.integer):
        raise ValueError("meta/episode_ends must have an integer dtype.")
    if np.any(episode_ends <= 0) or np.any(np.diff(episode_ends) <= 0):
        raise ValueError("meta/episode_ends must be positive and strictly increasing.")
    if episode_ends.dtype != np.dtype(np.int64):
        raise ValueError(
            f"meta/episode_ends must have converter dtype int64, got {episode_ends.dtype}."
        )
    frames = int(episode_ends[-1])
    for _, path in DATA_ARRAYS:
        if int(root[path].shape[0]) != frames:
            raise ValueError(
                f"{path} has {root[path].shape[0]} rows but final episode end is {frames}."
            )
    state = root["data/state"]
    action = root["data/action"]
    image = root["data/head_camera"]
    if state.dtype != np.dtype(np.float32) or action.dtype != np.dtype(np.float32):
        raise ValueError(
            "data/state and data/action must have converter dtype float32; got "
            f"state={state.dtype}, action={action.dtype}."
        )
    if image.dtype != np.dtype(np.uint8):
        raise ValueError(
            f"data/head_camera must have converter dtype uint8, got {image.dtype}."
        )
    if len(state.shape) != 2 or action.shape != state.shape:
        raise ValueError(
            f"state/action must be matching rank-2 arrays, got {state.shape}/{action.shape}."
        )
    if len(image.shape) != 4 or image.shape[1] != 3:
        raise ValueError(
            f"data/head_camera must have NCHW RGB shape, got {image.shape}."
        )
    return episode_ends


def _logical_digest(
    metadata: Mapping[str, Any],
    episode_ends: np.ndarray,
    source_indices: Sequence[int],
    episode_hashes: Sequence[str],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(_array_header("episode_ends", episode_ends.dtype, episode_ends.shape))
    digest.update(_canonical_bytes(episode_ends, episode_ends.dtype))
    for source_index, episode_hash in zip(
        source_indices, episode_hashes, strict=True
    ):
        digest.update(f"{source_index}:{episode_hash}\n".encode("ascii"))
    return digest.hexdigest()


def _scan_zarr_content(
    root: Any,
    *,
    metadata: Mapping[str, Any],
    source_indices: Sequence[int],
    chunk_rows: int,
    expected_episode: Callable[[int, int, int], Mapping[str, np.ndarray] | None]
    | None = None,
) -> dict[str, Any]:
    """Read and hash all logical arrays, optionally comparing expected episodes."""

    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive.")
    episode_ends = _validate_zarr_structure(root)
    if len(source_indices) != len(episode_ends):
        raise ValueError(
            f"Got {len(source_indices)} source indices for {len(episode_ends)} episodes."
        )
    starts = np.concatenate(
        (np.asarray([0], dtype=episode_ends.dtype), episode_ends[:-1])
    )

    array_hashers = {
        name: _begin_array_hash(name, root[path].dtype, root[path].shape)
        for name, path in DATA_ARRAYS
    }
    episode_hashes: list[str] = []
    state_action_hashes: list[str] = []
    image_hashes: list[str] = []
    errors: list[str] = []
    nonfinite = {"state": 0, "action": 0}
    extrema: dict[str, list[np.ndarray] | None] = {
        "state": None,
        "action": None,
    }
    image_min = 255
    image_max = 0
    constant_image_frames = 0

    for episode_index, (start_value, end_value) in enumerate(
        zip(starts, episode_ends, strict=True)
    ):
        start, end = int(start_value), int(end_value)
        length = end - start
        expected = (
            expected_episode(episode_index, start, end)
            if expected_episode is not None
            else None
        )
        episode_digest = hashlib.sha256()
        state_action_digest = hashlib.sha256()
        image_digest = hashlib.sha256()

        for name, path in DATA_ARRAYS:
            array = root[path]
            episode_shape = (length, *tuple(int(value) for value in array.shape[1:]))
            header = _array_header(name, array.dtype, episode_shape)
            episode_digest.update(header)
            if name == "head_camera":
                image_digest.update(header)
            else:
                state_action_digest.update(header)

            expected_values = None if expected is None else np.asarray(expected[name])
            comparable = expected_values is None or expected_values.shape == episode_shape
            if expected_values is not None and not comparable:
                errors.append(
                    f"episode {episode_index} {name}: raw shape "
                    f"{expected_values.shape} != Zarr shape {episode_shape}"
                )
            differs = False
            max_abs_error = 0.0
            for row_slice in _iter_row_slices(start, end, chunk_rows):
                values = np.asarray(array[row_slice])
                _update_array_hash(array_hashers[name], values, array.dtype)
                payload = _canonical_bytes(values, array.dtype)
                episode_digest.update(payload)
                if name == "head_camera":
                    image_digest.update(payload)
                    flat = values.reshape((values.shape[0], -1))
                    image_min = min(image_min, int(flat.min()))
                    image_max = max(image_max, int(flat.max()))
                    constant_image_frames += int(
                        np.count_nonzero(flat.min(axis=1) == flat.max(axis=1))
                    )
                else:
                    state_action_digest.update(payload)
                    nonfinite[name] += int(np.count_nonzero(~np.isfinite(values)))
                    chunk_min = values.min(axis=0).astype(np.float64, copy=False)
                    chunk_max = values.max(axis=0).astype(np.float64, copy=False)
                    if extrema[name] is None:
                        extrema[name] = [chunk_min.copy(), chunk_max.copy()]
                    else:
                        extrema[name][0] = np.minimum(extrema[name][0], chunk_min)
                        extrema[name][1] = np.maximum(extrema[name][1], chunk_max)

                if expected_values is not None and comparable:
                    local = slice(row_slice.start - start, row_slice.stop - start)
                    expected_chunk = expected_values[local]
                    if not np.array_equal(values, expected_chunk):
                        differs = True
                        if values.shape == expected_chunk.shape:
                            delta = np.abs(
                                values.astype(np.float64)
                                - expected_chunk.astype(np.float64)
                            )
                            max_abs_error = max(max_abs_error, float(delta.max()))
            if differs:
                errors.append(
                    f"episode {episode_index} {name}: Zarr differs from raw "
                    f"(max_abs_error={max_abs_error:g})"
                )

        episode_hashes.append(episode_digest.hexdigest())
        state_action_hashes.append(state_action_digest.hexdigest())
        image_hashes.append(image_digest.hexdigest())

    episode_ends_hash = _hash_numpy_array("episode_ends", episode_ends)
    array_hashes = {
        name: digest.hexdigest() for name, digest in array_hashers.items()
    }
    array_hashes["episode_ends"] = episode_ends_hash
    return {
        "logical_content_sha256": _logical_digest(
            metadata, episode_ends, source_indices, episode_hashes
        ),
        "array_sha256": array_hashes,
        "episode_hashes": episode_hashes,
        "state_action_hashes": state_action_hashes,
        "image_hashes": image_hashes,
        "episode_ends": episode_ends,
        "errors": errors,
        "statistics": {
            "state_nonfinite_count": nonfinite["state"],
            "action_nonfinite_count": nonfinite["action"],
            "state_min_per_dim": extrema["state"][0].tolist(),
            "state_max_per_dim": extrema["state"][1].tolist(),
            "action_min_per_dim": extrema["action"][0].tolist(),
            "action_max_per_dim": extrema["action"][1].tolist(),
            "image_value_range": [image_min, image_max],
            "constant_image_frames": constant_image_frames,
        },
    }


def hash_zarr_logical_content(
    dataset: str | Path,
    *,
    source_indices: Sequence[int],
    chunk_rows: int = 32,
) -> dict[str, Any]:
    """Hash all core arrays without requiring the raw source directory."""

    dataset = Path(dataset).expanduser().resolve()
    root = _open_zarr(dataset)
    metadata, _ = _load_dataset_metadata(dataset, root)
    scan = _scan_zarr_content(
        root,
        metadata=metadata,
        source_indices=source_indices,
        chunk_rows=chunk_rows,
    )
    return {
        "hash_spec": HASH_SPEC,
        "logical_content_sha256": scan["logical_content_sha256"],
        "array_sha256": scan["array_sha256"],
        "episode_sha256": scan["episode_hashes"],
    }


def _pad_joint_positions(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 0:
        return values
    if values.ndim != 2 or values.shape[1] > width:
        raise ValueError(
            f"Cannot pad joint-position array with shape {values.shape} to width {width}."
        )
    return np.pad(values, ((0, 0), (0, width - values.shape[1])))


def audit_proxy_dataset(
    dataset: str | Path,
    raw_success_dir: str | Path,
    *,
    expected_episodes: int | None = None,
    chunk_rows: int = 32,
    expected_logical_sha256: str | None = None,
    require_unique_episodes: bool = True,
) -> dict[str, Any]:
    """Return a complete raw-to-Zarr provenance audit.

    The raw episode order exactly follows the official converter: numeric demo
    index order, truncated to the requested number of episodes.
    """

    started = time.monotonic()
    dataset = Path(dataset).expanduser().resolve()
    raw_success_dir = Path(raw_success_dir).expanduser().resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(dataset)
    if not raw_success_dir.is_dir():
        raise FileNotFoundError(raw_success_dir)
    root = _open_zarr(dataset)
    episode_ends = _validate_zarr_structure(root)
    episodes = int(episode_ends.size)
    if expected_episodes is not None and episodes != expected_episodes:
        raise ValueError(
            f"Dataset has {episodes} episodes; expected exactly {expected_episodes}."
        )
    metadata, metadata_matches_attrs = _load_dataset_metadata(dataset, root)
    raw_candidates = _numeric_demo_dirs(raw_success_dir)
    if len(raw_candidates) < episodes:
        raise ValueError(
            f"Raw directory has {len(raw_candidates)} complete demos for {episodes} "
            "Zarr episodes."
        )
    selected = raw_candidates[:episodes]
    source_indices = [index for index, _ in selected]
    errors: list[str] = []
    if not metadata_matches_attrs:
        errors.append("metadata.json does not equal the meta group attributes")
    if metadata.get("num_episodes") != episodes:
        errors.append(
            f"metadata num_episodes={metadata.get('num_episodes')!r} != {episodes}"
        )
    if metadata.get("observation_space") != "joint_pos":
        raise ValueError("Raw audit currently supports joint_pos observations only.")
    if metadata.get("action_space") != "joint_pos":
        raise ValueError("Raw audit currently supports joint_pos actions only.")
    downsample_ratio = int(metadata.get("downsample_ratio", 1))
    joint_pos_padding = int(metadata.get("joint_pos_padding", 0))
    if downsample_ratio < 1:
        raise ValueError("downsample_ratio must be positive.")

    raw_records: list[dict[str, Any]] = []
    raw_episode_lengths: list[int] = []

    def expected_episode(
        episode_index: int, start: int, end: int
    ) -> Mapping[str, np.ndarray]:
        source_index, raw_dir = selected[episode_index]
        metadata_path = raw_dir / "metadata.json"
        video_path = raw_dir / "rgb.mp4"
        raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(raw_metadata, dict):
            raise ValueError(f"Raw metadata must be an object: {metadata_path}")
        list_lengths = {
            key: len(value)
            for key, value in raw_metadata.items()
            if isinstance(value, list)
        }
        if len(set(list_lengths.values())) != 1:
            errors.append(
                f"episode {episode_index}: inconsistent raw metadata lengths "
                f"{sorted(set(list_lengths.values()))}"
            )
        status_path = raw_dir / "status.txt"
        status = (
            status_path.read_text(encoding="utf-8").strip()
            if status_path.is_file()
            else None
        )
        if status != "success":
            errors.append(
                f"episode {episode_index}: raw status is {status!r}, expected 'success'"
            )

        decoded = _decode_rgb_video(video_path)
        if decoded.ndim != 4 or decoded.shape[-1] != 3:
            raise ValueError(
                f"episode {episode_index}: decoded RGB has invalid shape {decoded.shape}."
            )
        if len(decoded) == 0:
            raise ValueError(f"episode {episode_index}: decoded RGB is empty.")
        raw_state_all = np.asarray(raw_metadata["joint_qpos"], dtype=np.float32)
        raw_action_all = np.asarray(
            raw_metadata["joint_qpos_target"], dtype=np.float32
        )
        if not (
            len(decoded) == len(raw_state_all) == len(raw_action_all)
        ):
            errors.append(
                f"episode {episode_index}: pre-downsample lengths differ: "
                f"rgb={len(decoded)}, state={len(raw_state_all)}, "
                f"action={len(raw_action_all)}"
            )
        indices = np.arange(0, len(decoded), downsample_ratio, dtype=np.int64)
        if len(raw_state_all) <= int(indices[-1]) or len(raw_action_all) <= int(
            indices[-1]
        ):
            raise ValueError(
                f"episode {episode_index}: metadata is shorter than decoded RGB."
            )
        state = _pad_joint_positions(raw_state_all[indices], joint_pos_padding)
        action = _pad_joint_positions(raw_action_all[indices], joint_pos_padding)
        image = np.moveaxis(decoded[indices], -1, 1)
        raw_episode_lengths.append(len(indices))
        raw_records.append(
            {
                "episode": episode_index,
                "source_index": source_index,
                "source_directory": raw_dir.name,
                "frames": int(len(indices)),
                "metadata_json_sha256": _file_sha256(metadata_path),
                "rgb_mp4_sha256": _file_sha256(video_path),
            }
        )
        return {"state": state, "action": action, "head_camera": image}

    scan = _scan_zarr_content(
        root,
        metadata=metadata,
        source_indices=source_indices,
        chunk_rows=chunk_rows,
        expected_episode=expected_episode,
    )
    errors.extend(scan["errors"])
    for name in ("state", "action"):
        count = int(scan["statistics"][f"{name}_nonfinite_count"])
        if count:
            errors.append(f"data/{name} contains {count} non-finite values")
    cumulative_raw_lengths = np.cumsum(raw_episode_lengths, dtype=np.int64)
    if not np.array_equal(cumulative_raw_lengths, scan["episode_ends"]):
        errors.append("episode_ends do not equal cumulative raw episode lengths")

    combined_duplicates = _duplicate_groups(scan["episode_hashes"])
    state_action_duplicates = _duplicate_groups(scan["state_action_hashes"])
    image_duplicates = _duplicate_groups(scan["image_hashes"])
    if require_unique_episodes and combined_duplicates:
        errors.append(
            f"combined logical episode hashes contain duplicates: {combined_duplicates}"
        )
    if (
        expected_logical_sha256 is not None
        and scan["logical_content_sha256"] != expected_logical_sha256
    ):
        errors.append(
            "logical SHA-256 mismatch: expected "
            f"{expected_logical_sha256}, got {scan['logical_content_sha256']}"
        )

    for record, combined, state_action, image in zip(
        raw_records,
        scan["episode_hashes"],
        scan["state_action_hashes"],
        scan["image_hashes"],
        strict=True,
    ):
        record.update(
            {
                "logical_episode_sha256": combined,
                "state_action_sha256": state_action,
                "head_camera_sha256": image,
            }
        )

    lengths = np.diff(np.concatenate(([0], scan["episode_ends"])))
    return {
        "schema": SCHEMA,
        "status": "pass" if not errors else "failed",
        "hash_spec": HASH_SPEC,
        "dataset": str(dataset),
        "raw_success_dir": str(raw_success_dir),
        "logical_content_sha256": scan["logical_content_sha256"],
        "expected_logical_sha256": expected_logical_sha256,
        "array_sha256": scan["array_sha256"],
        "episodes": episodes,
        "frames": int(scan["episode_ends"][-1]),
        "episode_lengths": {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "sum": int(lengths.sum()),
        },
        "chunk_rows": chunk_rows,
        "fully_read_arrays": ["episode_ends", *(name for name, _ in DATA_ARRAYS)],
        "metadata": metadata,
        "metadata_file_equals_zarr_attrs": metadata_matches_attrs,
        "raw_candidate_count": len(raw_candidates),
        "selected_source_indices": source_indices,
        "selected_source_indices_unique": len(set(source_indices)) == episodes,
        "raw_exact_match": not scan["errors"]
        and np.array_equal(cumulative_raw_lengths, scan["episode_ends"]),
        "duplicates": {
            "combined_episode_groups": combined_duplicates,
            "state_action_groups": state_action_duplicates,
            "head_camera_groups": image_duplicates,
            "combined_unique_count": len(set(scan["episode_hashes"])),
        },
        "statistics": scan["statistics"],
        "episodes_provenance": raw_records,
        "errors": errors,
        "elapsed_seconds": time.monotonic() - started,
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--raw-success-dir", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--chunk-rows", type=int, default=32)
    parser.add_argument("--expected-logical-sha256")
    parser.add_argument("--allow-duplicate-episodes", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = audit_proxy_dataset(
            args.dataset,
            args.raw_success_dir,
            expected_episodes=args.expected_episodes,
            chunk_rows=args.chunk_rows,
            expected_logical_sha256=args.expected_logical_sha256,
            require_unique_episodes=not args.allow_duplicate_episodes,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    if args.output is not None:
        _atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HASH_SPEC",
    "SCHEMA",
    "audit_proxy_dataset",
    "hash_zarr_logical_content",
]
