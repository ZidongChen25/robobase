import os

from omegaconf import OmegaConf

from scripts.eval_execution_length_sweep import (
    _configure_process,
    _parse_args,
    _restore_eval_state,
    _set_if_present,
)


def test_configure_process_overrides_jax_gpu_selection(monkeypatch):
    monkeypatch.setenv("JAX_CUDA_VISIBLE_DEVICES", "7")

    _configure_process(2)

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
    assert os.environ["JAX_CUDA_VISIBLE_DEVICES"] == "2"


def test_set_if_present_does_not_create_missing_path():
    cfg = OmegaConf.create({"method": {"objective": None}})

    _set_if_present(cfg, "method.objective.num_flow_steps", 2)

    assert cfg.method.objective is None


def test_parse_args_can_disable_temporal_ensemble(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "eval_execution_length_sweep.py",
            "--run-dir",
            str(tmp_path),
            "--snapshot",
            str(tmp_path / "snapshot.pkl"),
            "--execution-lengths",
            "1",
            "--output-dir",
            str(tmp_path / "out"),
            "--work-dir",
            str(tmp_path / "work"),
            "--no-temporal-ensemble",
        ],
    )

    args = _parse_args()

    assert args.temporal_ensemble is False


def test_restore_eval_state_reloads_rng_and_resets_aligned_noise(tmp_path):
    class Agent:
        def __init__(self):
            self.reset_calls = 0

        def reset_aligned_eval_noise(self):
            self.reset_calls += 1

    class Workspace:
        def __init__(self):
            self.agent = Agent()
            self.load_calls = []

        def load_snapshot(self, path, *, load_replay_buffer):
            self.load_calls.append((path, load_replay_buffer))

    workspace = Workspace()
    snapshot = tmp_path / "snapshot.pkl"

    _restore_eval_state(workspace, snapshot)
    _restore_eval_state(workspace, snapshot)

    assert workspace.load_calls == [(snapshot, False), (snapshot, False)]
    assert workspace.agent.reset_calls == 2
