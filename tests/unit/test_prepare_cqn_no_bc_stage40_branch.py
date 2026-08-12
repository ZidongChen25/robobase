import os
import pickle
from pathlib import Path

import pytest

from scripts.prepare_cqn_no_bc_stage40_branch import (
    prepare_branch,
    select_snapshot_replay_files,
)


def _write_episode(directory: Path, episode: int, length: int, start: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"20260801T000000_{episode}_{length}_{start}.npz"
    path.write_bytes(f"episode-{episode}".encode())
    return path


def _state(directory: Path, *, count: int, episodes: int) -> dict[str, object]:
    return {
        "add_count": count,
        "num_episodes": episodes,
        "num_transitions": count,
        "is_first": False,
        "save_snapshot": True,
        "replay_dir": str(directory),
    }


def test_prepare_branch_excludes_post_snapshot_replay(tmp_path: Path):
    source = tmp_path / "source"
    for name in ("replay", "demo_replay"):
        directory = source / name
        _write_episode(directory, 0, 3, 0)
        _write_episode(directory, 1, 2, 3)
        _write_episode(directory, 2, 4, 5)  # appended after the snapshot

    payload = {
        "_pretrain_step": 10000,
        "_main_loop_iterations": 0,
        "agent": {"params": {"critic": {}}},
        "agent_checkpoint_state": {"update_step_count": 10000},
        "replay_buffer": _state(source / "replay", count=5, episodes=2),
        "demo_replay_buffer": _state(
            source / "demo_replay", count=5, episodes=2
        ),
    }
    snapshot = source / "snapshots" / "10000_snapshot.pkl"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(pickle.dumps(payload))

    destination = tmp_path / "destination"
    manifest = prepare_branch(
        source_run=source, destination_run=destination, snapshot_step=10000
    )

    for name in ("replay", "demo_replay"):
        assert sorted(path.name for path in (destination / name).glob("*.npz")) == [
            "20260801T000000_0_3_0.npz",
            "20260801T000000_1_2_3.npz",
        ]
        assert manifest["replay"][name]["file_count"] == 2
        assert manifest["replay"][name]["last_end_index"] == 5
    assert (destination / "snapshots" / "latest_snapshot.pkl").is_symlink()
    assert os.path.samefile(
        destination / "snapshots" / "10000_snapshot.pkl", snapshot
    )
    assert not manifest["objective_flags_restored_from_snapshot"]


def test_select_snapshot_replay_rejects_gap(tmp_path: Path):
    replay = tmp_path / "replay"
    _write_episode(replay, 0, 2, 0)
    _write_episode(replay, 1, 2, 3)
    with pytest.raises(ValueError, match="not contiguous"):
        select_snapshot_replay_files(
            replay, _state(replay, count=5, episodes=2)
        )


def test_prepare_branch_rejects_objective_flag_in_agent_state(tmp_path: Path):
    source = tmp_path / "source"
    for name in ("replay", "demo_replay"):
        _write_episode(source / name, 0, 1, 0)
    payload = {
        "_pretrain_step": 10000,
        "_main_loop_iterations": 0,
        "agent": {"dense_return_q_target": True},
        "agent_checkpoint_state": {},
        "replay_buffer": _state(source / "replay", count=1, episodes=1),
        "demo_replay_buffer": _state(
            source / "demo_replay", count=1, episodes=1
        ),
    }
    snapshot = source / "snapshots" / "10000_snapshot.pkl"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError, match="stores dense_return_q_target"):
        prepare_branch(
            source_run=source,
            destination_run=tmp_path / "destination",
            snapshot_step=10000,
        )


def test_prepare_branch_accepts_stage_specific_manifest_name(tmp_path: Path):
    source = tmp_path / "source"
    for name in ("replay", "demo_replay"):
        _write_episode(source / name, 0, 1, 0)
    payload = {
        "_pretrain_step": 10000,
        "_main_loop_iterations": 0,
        "agent": {"params": {}},
        "agent_checkpoint_state": {},
        "replay_buffer": _state(source / "replay", count=1, episodes=1),
        "demo_replay_buffer": _state(
            source / "demo_replay", count=1, episodes=1
        ),
    }
    snapshot = source / "snapshots" / "10000_snapshot.pkl"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(pickle.dumps(payload))
    destination = tmp_path / "destination"
    prepare_branch(
        source_run=source,
        destination_run=destination,
        snapshot_step=10000,
        manifest_name="stage41_branch_manifest.json",
    )
    assert (destination / "stage41_branch_manifest.json").is_file()
    assert not (destination / "stage40_branch_manifest.json").exists()


def test_prepare_branch_accepts_exact_online_snapshot(tmp_path: Path):
    source = tmp_path / "source"
    for name in ("replay", "demo_replay"):
        _write_episode(source / name, 0, 3, 0)
        _write_episode(source / name, 1, 2, 3)
        _write_episode(source / name, 2, 4, 5)  # future replay is excluded
    payload = {
        "_pretrain_step": 10000,
        "_main_loop_iterations": 40000,
        "agent": {"params": {}},
        "agent_checkpoint_state": {},
        "replay_buffer": _state(source / "replay", count=5, episodes=2),
        "demo_replay_buffer": _state(
            source / "demo_replay", count=5, episodes=2
        ),
    }
    snapshot = source / "snapshots" / "50000_snapshot.pkl"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(pickle.dumps(payload))

    destination = tmp_path / "destination"
    manifest = prepare_branch(
        source_run=source,
        destination_run=destination,
        snapshot_step=50000,
        expected_pretrain_step=10000,
        expected_main_loop_iterations=40000,
        manifest_name="stage43_branch_manifest.json",
    )

    assert manifest["pretrain_step"] == 10000
    assert manifest["main_loop_iterations"] == 40000
    assert manifest["expected_pretrain_step"] == 10000
    assert manifest["expected_main_loop_iterations"] == 40000
    assert (destination / "snapshots" / "50000_snapshot.pkl").is_file()
    assert len(list((destination / "replay").glob("*.npz"))) == 2


def test_prepare_branch_rejects_wrong_online_iteration(tmp_path: Path):
    source = tmp_path / "source"
    for name in ("replay", "demo_replay"):
        _write_episode(source / name, 0, 1, 0)
    payload = {
        "_pretrain_step": 10000,
        "_main_loop_iterations": 39999,
        "agent": {"params": {}},
        "agent_checkpoint_state": {},
        "replay_buffer": _state(source / "replay", count=1, episodes=1),
        "demo_replay_buffer": _state(
            source / "demo_replay", count=1, episodes=1
        ),
    }
    snapshot = source / "snapshots" / "50000_snapshot.pkl"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(pickle.dumps(payload))

    with pytest.raises(ValueError, match="Expected main-loop iterations 40000"):
        prepare_branch(
            source_run=source,
            destination_run=tmp_path / "destination",
            snapshot_step=50000,
            expected_pretrain_step=10000,
            expected_main_loop_iterations=40000,
        )
