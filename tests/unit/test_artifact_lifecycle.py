import csv
import pickle
from types import SimpleNamespace

from omegaconf import OmegaConf

from robobase.workspace import Workspace, _Timer
from scripts.eval_cqn_as_snapshot_sweep import finalize_run_artifacts


class _Agent:
    def __init__(self):
        self.loaded = None
        self.loaded_checkpoint = None

    def state_dict(self):
        return {"params": {"weight": 3.0}}

    def load_state_dict(self, state):
        self.loaded = state

    def checkpoint_state_dict(self):
        return {"optimizer": {"count": 7}}

    def load_checkpoint_state_dict(self, state):
        self.loaded_checkpoint = state


class _Replay:
    def state_dict(self):
        return {"add_count": 12}


def _workspace(tmp_path):
    workspace = Workspace.__new__(Workspace)
    workspace.work_dir = tmp_path
    workspace.cfg = OmegaConf.create(
        {
            "action_repeat": 1,
            "execution_length": 1,
            "artifacts": {
                "resume_keep_last": 2,
                "save_eval_checkpoints": True,
                "delete_replay_on_train_complete": False,
                "delete_resume_on_train_complete": False,
            },
        }
    )
    workspace.train_envs = SimpleNamespace(num_envs=1)
    workspace._pretrain_step = 0
    workspace._main_loop_iterations = 0
    workspace._global_env_episode = 0
    workspace.agent = _Agent()
    workspace.replay_buffer = _Replay()
    workspace.use_demo_replay = False
    workspace.logger = SimpleNamespace(wandb_run_id=None)
    workspace._timer = _Timer()
    workspace._replay_iter = None
    workspace._snapshot_loaded = False
    workspace._snapshot_cfg = None
    return workspace


def test_snapshot_rotation_keeps_two_resume_and_all_params_only(tmp_path):
    workspace = _workspace(tmp_path)
    for step in (5, 10, 15):
        workspace._main_loop_iterations = step
        workspace.save_snapshot()

    resumes = sorted(
        path.name
        for path in (tmp_path / "snapshots").glob("*_snapshot.pkl")
        if path.name != "latest_snapshot.pkl"
    )
    assert resumes == ["10_snapshot.pkl", "15_snapshot.pkl"]
    assert (tmp_path / "snapshots" / "latest_snapshot.pkl").resolve() == (
        tmp_path / "snapshots" / "15_snapshot.pkl"
    )
    evals = sorted(path.name for path in (tmp_path / "eval_checkpoints").glob("*"))
    assert evals == [
        "10_checkpoint.pkl",
        "15_checkpoint.pkl",
        "5_checkpoint.pkl",
    ]
    with (tmp_path / "eval_checkpoints" / "15_checkpoint.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    assert "agent_checkpoint_state" not in payload
    assert "replay_buffer" not in payload
    assert payload["agent"] == {"params": {"weight": 3.0}}

    workspace.load_snapshot(
        tmp_path / "eval_checkpoints" / "15_checkpoint.pkl",
        load_replay_buffer=False,
    )
    assert workspace.agent.loaded == {"params": {"weight": 3.0}}
    assert workspace.agent.loaded_checkpoint == {}


def test_optimizer_state_lives_in_one_sidecar_not_every_snapshot(tmp_path):
    workspace = _workspace(tmp_path)
    for step in (5, 10, 15):
        workspace._main_loop_iterations = step
        workspace.save_snapshot()

    snapshots = tmp_path / "snapshots"
    with (snapshots / "15_snapshot.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    assert "agent_checkpoint_state" not in payload
    assert payload["agent"] == {"params": {"weight": 3.0}}

    with (snapshots / "resume_state.pkl").open("rb") as handle:
        sidecar = pickle.load(handle)
    assert sidecar["snapshot_step"] == 15
    assert sidecar["agent_checkpoint_state"] == {"optimizer": {"count": 7}}

    # Resuming from the newest snapshot still restores the optimizer.
    workspace.load_snapshot(snapshots / "latest_snapshot.pkl", load_replay_buffer=False)
    assert workspace.agent.loaded_checkpoint == {"optimizer": {"count": 7}}

    # An earlier snapshot is evaluation-only: the sidecar tracks step 15, so
    # the optimizer is reinitialized rather than silently mismatched.
    workspace.agent.loaded_checkpoint = None
    workspace.load_snapshot(snapshots / "10_snapshot.pkl", load_replay_buffer=False)
    assert workspace.agent.loaded_checkpoint == {}


def test_legacy_embedded_optimizer_state_still_loads(tmp_path):
    workspace = _workspace(tmp_path)
    legacy = tmp_path / "legacy_snapshot.pkl"
    with legacy.open("wb") as handle:
        pickle.dump(
            {
                "snapshot_version": 2,
                "_pretrain_step": 0,
                "_main_loop_iterations": 10,
                "_global_env_episode": 0,
                "cfg": None,
                "agent": {"params": {"weight": 3.0}},
                "agent_checkpoint_state": {"optimizer": {"count": 99}},
            },
            handle,
        )
    workspace.load_snapshot(legacy, load_replay_buffer=False)
    assert workspace.agent.loaded_checkpoint == {"optimizer": {"count": 99}}


def _write_selection_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("env_steps", "episode_success", "eval_seed_start"),
        )
        writer.writeheader()
        writer.writerows(rows)


def test_eval_finalization_keeps_validation_best_and_final(tmp_path):
    checkpoint_dir = tmp_path / "eval_checkpoints"
    checkpoint_dir.mkdir()
    checkpoints = []
    for step in (5, 10, 15):
        path = checkpoint_dir / f"{step}_checkpoint.pkl"
        path.write_bytes(b"params")
        checkpoints.append((step, path))
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    (snapshot_dir / "15_snapshot.pkl").write_bytes(b"optimizer")
    (tmp_path / "replay").mkdir()
    (tmp_path / "replay" / "episode.npz").write_bytes(b"replay")
    selection = tmp_path / "val.csv"
    _write_selection_csv(
        selection,
        [
            {"env_steps": 5, "episode_success": 0.5, "eval_seed_start": 400},
            {"env_steps": 10, "episode_success": 0.8, "eval_seed_start": 400},
            {"env_steps": 15, "episode_success": 0.6, "eval_seed_start": 400},
        ],
    )

    record = finalize_run_artifacts(tmp_path, selection, checkpoints)

    assert record["retained_steps"] == [10, 15]
    assert sorted(path.name for path in checkpoint_dir.glob("*")) == [
        "10_checkpoint.pkl",
        "15_checkpoint.pkl",
    ]
    assert not snapshot_dir.joinpath("15_snapshot.pkl").exists()
    assert not (tmp_path / "replay").exists()


def test_eval_finalization_rejects_heldout_selection(tmp_path):
    checkpoint_dir = tmp_path / "eval_checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "5_checkpoint.pkl"
    checkpoint.write_bytes(b"params")
    selection = tmp_path / "heldout.csv"
    _write_selection_csv(
        selection,
        [{"env_steps": 5, "episode_success": 1.0, "eval_seed_start": 800}],
    )

    try:
        finalize_run_artifacts(tmp_path, selection, [(5, checkpoint)])
    except ValueError as exc:
        assert "held-out" in str(exc)
    else:
        raise AssertionError("held-out data must never select retained checkpoints")
