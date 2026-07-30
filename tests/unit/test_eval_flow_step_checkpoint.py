import os
import sys
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from scripts import eval_flow_step_checkpoint


def test_eval_checkpoint_uses_precomputed_language_and_keeps_lazy_replay(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "run"
    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir(parents=True)
    snapshot = run_dir / "snapshot.pkl"
    snapshot.write_bytes(b"snapshot")
    feature_path = tmp_path / "language.npy"
    np.save(feature_path, np.ones((1, 8), dtype=np.float32))
    encoder_weights_path = tmp_path / "resnet18.npz"
    np.savez(
        encoder_weights_path,
        **{"params/example": np.ones((1,), dtype=np.float32)},
    )
    OmegaConf.save(
        OmegaConf.create(
            {
                "create_train_env": True,
                "num_train_envs": 1,
                "num_train_frames": 1,
                "num_eval_envs": 1,
                "num_eval_episodes": 50,
                "log_train_video": True,
                "log_eval_video": True,
                "save_snapshot": True,
                "save_csv": True,
                "gpu_id": 0,
                "action_sequence": 20,
                "execution_length": 20,
                "wandb": {"use": True},
                "tb": {"use": True},
                "replay": {"num_workers": 8},
                "lazy_replay": {
                    "use": "auto",
                    "num_workers": 8,
                    "persistent_workers": True,
                },
                "backend": {
                    "replay_prefetch_size": 8,
                    "replay_device_prefetch": True,
                    "fused_update_steps": 8,
                    "update_block_every_steps": 8,
                },
                "env": {"task_name": "flip_cutlery"},
                "method": {
                    "num_flow_steps": 4,
                    "lang_feature_source": "clip",
                    "objective": {
                        "num_flow_steps": 4,
                        "train_time_schedule": "uniform",
                    },
                    "backbone": {"sequence_length": 20},
                    "encoder_model": {
                        "pretrained": True,
                        "pretrained_weights_path": None,
                    },
                },
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
            assert path == snapshot
            assert load_replay_buffer is False

        def eval(self):
            return {"episode_success": 0.5}

        def shutdown(self):
            pass

    monkeypatch.setitem(
        sys.modules,
        "robobase.workspace",
        SimpleNamespace(Workspace=FakeWorkspace),
    )
    args = Namespace(
        run_dir=run_dir,
        snapshot=snapshot,
        flow_steps=10,
        output=tmp_path / "result.json",
        work_dir=tmp_path / "work",
        gpu_id=1,
        num_eval_episodes=50,
        num_eval_envs=1,
        action_sequence=20,
        backbone_sequence_length=20,
        execution_length=20,
        flow_schedule=None,
        lang_feature_path=feature_path,
        encoder_weights_path=encoder_weights_path,
        xla_fusion_cache_dir=None,
    )

    payload = eval_flow_step_checkpoint._run_eval(args)

    assert FakeWorkspace.cfg.lazy_replay.use == "auto"
    assert FakeWorkspace.cfg.method.lang_feature_source == "precomputed"
    assert FakeWorkspace.cfg.method.lang_feature_path == str(feature_path.resolve())
    assert FakeWorkspace.cfg.method.num_flow_steps == 10
    assert FakeWorkspace.cfg.method.encoder_model.pretrained_weights_path == str(
        encoder_weights_path.resolve()
    )
    assert payload["lang_feature_source"] == "precomputed"
    assert payload["lang_feature_path"] == str(feature_path.resolve())
    assert payload["encoder_weights_path"] == str(encoder_weights_path.resolve())
    assert payload["metrics"]["episode_success"] == 0.5


def test_configure_process_adds_reproducible_gpu_flags(tmp_path, monkeypatch):
    cache_dir = tmp_path / "xla_cache"
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("JAX_CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)

    eval_flow_step_checkpoint._configure_process(3, cache_dir)

    flags = os.environ["XLA_FLAGS"].split()
    assert "--xla_gpu_enable_command_buffer=" in flags
    assert (
        f"--xla_gpu_per_fusion_autotune_cache_dir={cache_dir.resolve()}" in flags
    )
    assert "--xla_gpu_deterministic_ops=true" in flags
    assert "--xla_gpu_exclude_nondeterministic_ops=true" in flags
    assert cache_dir.is_dir()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
    assert os.environ["JAX_CUDA_VISIBLE_DEVICES"] == "3"
    assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "3"


def test_configure_process_rejects_conflicting_xla_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_deterministic_ops=false")

    try:
        eval_flow_step_checkpoint._configure_process(0, tmp_path / "xla_cache")
    except ValueError as exc:
        assert "Conflicting --xla_gpu_deterministic_ops" in str(exc)
    else:
        raise AssertionError("Expected a conflicting XLA flag to fail closed.")
