from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import benchmarks.official_roboverse.audit_proxy_data as audit_module
from benchmarks.official_roboverse.audit_proxy_data import (
    HASH_SPEC,
    audit_proxy_dataset,
    hash_zarr_logical_content,
)


def _make_fixture(tmp_path: Path, *, zero_zarr: bool = False):
    zarr = pytest.importorskip("zarr")
    dataset = tmp_path / "proxy.zarr"
    raw_root = tmp_path / "success"
    root = zarr.group(str(dataset))
    data = root.create_group("data")
    meta_group = root.create_group("meta")
    lengths = [2, 3]
    ends = np.cumsum(lengths, dtype=np.int64)
    states = np.arange(45, dtype=np.float32).reshape(5, 9) / 10
    actions = states + 1
    images = np.arange(5 * 3 * 4 * 4, dtype=np.uint8).reshape(5, 3, 4, 4)
    data.create_dataset(
        "state",
        data=np.zeros_like(states) if zero_zarr else states,
        chunks=(2, 9),
    )
    data.create_dataset(
        "action",
        data=np.zeros_like(actions) if zero_zarr else actions,
        chunks=(2, 9),
    )
    data.create_dataset(
        "head_camera",
        data=np.zeros_like(images) if zero_zarr else images,
        chunks=(2, 3, 4, 4),
    )
    meta_group.create_dataset("episode_ends", data=ends, chunks=(2,))
    converter_metadata = {
        "observation_space": "joint_pos",
        "action_space": "joint_pos",
        "delta_ee": 0,
        "joint_pos_padding": 0,
        "task_name": "unit_test",
        "num_episodes": 2,
        "downsample_ratio": 1,
    }
    meta_group.attrs.update(converter_metadata)
    (dataset / "metadata.json").write_text(
        json.dumps(converter_metadata), encoding="utf-8"
    )

    videos: dict[Path, np.ndarray] = {}
    start = 0
    for episode, (source_index, length) in enumerate(zip((3, 7), lengths)):
        raw_dir = raw_root / f"demo_{source_index:04d}"
        raw_dir.mkdir(parents=True)
        stop = start + length
        raw_metadata = {
            "joint_qpos": states[start:stop].tolist(),
            "joint_qpos_target": actions[start:stop].tolist(),
        }
        (raw_dir / "metadata.json").write_text(
            json.dumps(raw_metadata), encoding="utf-8"
        )
        (raw_dir / "rgb.mp4").write_bytes(f"video-{episode}".encode())
        (raw_dir / "status.txt").write_text("success\n", encoding="utf-8")
        videos[(raw_dir / "rgb.mp4").resolve()] = np.moveaxis(
            images[start:stop], 1, -1
        )
        start = stop
    return dataset, raw_root, videos


def test_logical_hash_is_chunk_invariant_and_content_sensitive(tmp_path):
    dataset, _, _ = _make_fixture(tmp_path)
    small = hash_zarr_logical_content(
        dataset, source_indices=[3, 7], chunk_rows=1
    )
    large = hash_zarr_logical_content(
        dataset, source_indices=[3, 7], chunk_rows=100
    )
    assert small == large
    assert small["hash_spec"] == HASH_SPEC

    zarr = pytest.importorskip("zarr")
    root = zarr.open_group(str(dataset), mode="a")
    root["data/head_camera"][4, 2, 3, 3] ^= np.uint8(1)
    changed = hash_zarr_logical_content(
        dataset, source_indices=[3, 7], chunk_rows=2
    )
    assert changed["logical_content_sha256"] != small["logical_content_sha256"]
    assert changed["array_sha256"]["state"] == small["array_sha256"]["state"]
    assert (
        changed["array_sha256"]["head_camera"]
        != small["array_sha256"]["head_camera"]
    )


def test_proxy_audit_matches_every_raw_episode(tmp_path, monkeypatch):
    dataset, raw_root, videos = _make_fixture(tmp_path)
    monkeypatch.setattr(
        audit_module,
        "_decode_rgb_video",
        lambda path: videos[path.resolve()],
    )

    result = audit_proxy_dataset(
        dataset, raw_root, expected_episodes=2, chunk_rows=1
    )

    assert result["status"] == "pass"
    assert result["raw_exact_match"] is True
    assert result["selected_source_indices"] == [3, 7]
    assert result["duplicates"]["combined_unique_count"] == 2
    assert len(result["episodes_provenance"]) == 2
    assert set(result["array_sha256"]) == {
        "episode_ends",
        "state",
        "action",
        "head_camera",
    }


def test_proxy_audit_rejects_zero_zarr_with_valid_shapes(tmp_path, monkeypatch):
    dataset, raw_root, videos = _make_fixture(tmp_path, zero_zarr=True)
    monkeypatch.setattr(
        audit_module,
        "_decode_rgb_video",
        lambda path: videos[path.resolve()],
    )

    result = audit_proxy_dataset(dataset, raw_root, expected_episodes=2)

    assert result["status"] == "failed"
    assert result["raw_exact_match"] is False
    assert any("state: Zarr differs from raw" in error for error in result["errors"])
    assert any(
        "head_camera: Zarr differs from raw" in error for error in result["errors"]
    )


def test_proxy_audit_enforces_expected_logical_hash(tmp_path, monkeypatch):
    dataset, raw_root, videos = _make_fixture(tmp_path)
    monkeypatch.setattr(
        audit_module,
        "_decode_rgb_video",
        lambda path: videos[path.resolve()],
    )

    result = audit_proxy_dataset(
        dataset,
        raw_root,
        expected_logical_sha256="0" * 64,
    )

    assert result["status"] == "failed"
    assert any("logical SHA-256 mismatch" in error for error in result["errors"])
