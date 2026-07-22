import sys
from argparse import Namespace
from types import SimpleNamespace

from omegaconf import OmegaConf

from scripts import eval_cqn_dmc_checkpoint
from scripts import eval_cqn_dmc_paper_runs


def test_eval_cqn_checkpoint_loads_without_replay_and_compares_paper(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "run"
    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir(parents=True)
    snapshot = run_dir / "snapshots" / "latest_snapshot.pkl"
    snapshot.parent.mkdir()
    snapshot.write_bytes(b"snapshot")
    OmegaConf.save(
        OmegaConf.create(
            {
                "create_train_env": True,
                "num_train_envs": 1,
                "num_train_frames": 500000,
                "num_eval_envs": 1,
                "num_eval_episodes": 10,
                "log_train_video": True,
                "log_eval_video": True,
                "save_snapshot": True,
                "save_csv": True,
                "gpu_id": 0,
                "wandb": {"use": True},
                "tb": {"use": True},
                "replay": {"num_workers": 4},
                "lazy_replay": {"num_workers": 1, "persistent_workers": True},
                "backend": {
                    "replay_prefetch_size": 4,
                    "replay_device_prefetch": True,
                    "fused_update_steps": 8,
                    "update_block_every_steps": 1,
                },
                "env": {"task_name": "cartpole_swingup_sparse"},
                "method": {"name": "cqn"},
            }
        ),
        hydra_dir / "config.yaml",
    )

    class FakeWorkspace:
        cfg = None

        def __init__(self, cfg, work_dir):
            del work_dir
            FakeWorkspace.cfg = cfg

        def load_snapshot(self, path, load_replay_buffer):
            assert path == snapshot.resolve()
            assert load_replay_buffer is False

        def eval(self):
            return {"episode_reward": 750.0, "episode_length": 1000.0}

        def shutdown(self):
            pass

    monkeypatch.setitem(
        sys.modules,
        "robobase.workspace",
        SimpleNamespace(Workspace=FakeWorkspace),
    )
    args = Namespace(
        run_dir=run_dir,
        snapshot=None,
        output=None,
        work_dir=tmp_path / "eval",
        gpu_id=1,
        num_eval_episodes=50,
        eval_seed_start=20000,
        paper_reference_return=780.0,
        paper_tolerance=100.0,
    )

    payload = eval_cqn_dmc_checkpoint._run_eval(args)

    assert FakeWorkspace.cfg.create_train_env is False
    assert FakeWorkspace.cfg.num_train_envs == 0
    assert FakeWorkspace.cfg.env.eval_seed_start == 20000
    assert payload["metrics"]["episode_reward"] == 750.0
    assert payload["paper_comparison"]["alignment"] == "within_reference_band"
    assert payload["paper_comparison"]["meets_reference_lower_band"] is True


def test_paper_eval_aggregates_exactly_four_training_seeds(tmp_path, monkeypatch):
    returns = iter([700.0, 760.0, 800.0, 820.0])
    monkeypatch.setattr(
        eval_cqn_dmc_paper_runs.checkpoint_eval,
        "_configure_process",
        lambda _gpu_id: None,
    )
    monkeypatch.setattr(
        eval_cqn_dmc_paper_runs.checkpoint_eval,
        "_run_eval",
        lambda _args: {"metrics": {"episode_reward": next(returns)}},
    )
    args = Namespace(
        run_dirs=[tmp_path / f"seed{seed}" for seed in range(1, 5)],
        output=tmp_path / "aggregate.json",
        gpu_id=0,
        num_eval_episodes=50,
        eval_seed_start=20000,
        paper_reference_return=780.0,
        paper_tolerance=100.0,
    )

    payload = eval_cqn_dmc_paper_runs._run(args)

    assert payload["num_training_seeds"] == 4
    assert payload["mean_return"] == 770.0
    assert payload["paper_comparison"]["alignment"] == "within_reference_band"
