import sys
import types

import numpy as np
from omegaconf import OmegaConf

from robobase.logger import Logger


class _GradTensorLike:
    def __init__(self, value, *, detached=False, on_cpu=False):
        self._value = np.asarray(value)
        self._detached = detached
        self._on_cpu = on_cpu

    def detach(self):
        return _GradTensorLike(self._value, detached=True, on_cpu=self._on_cpu)

    def cpu(self):
        assert self._detached
        return _GradTensorLike(self._value, detached=True, on_cpu=True)

    def numpy(self):
        assert self._detached and self._on_cpu
        return self._value

    def __array__(self):
        raise RuntimeError("requires grad")


def test_logger_passes_wandb_resume_arguments(monkeypatch, tmp_path):
    captured_kwargs = {}

    class DummyRun:
        id = "ma9u6uis"

    def fake_wandb_init(**kwargs):
        captured_kwargs.update(kwargs)
        return DummyRun()

    monkeypatch.setattr("robobase.logger.wandb.init", fake_wandb_init)

    cfg = OmegaConf.create(
        {
            "save_csv": False,
            "wandb": {
                "use": True,
                "project": "robomimic_dp",
                "entity": "younggyo",
                "name": "dp_tool_hang_h8",
                "id": "ma9u6uis",
                "resume": "must",
            },
            "tb": {
                "use": False,
                "log_dir": str(tmp_path),
                "name": None,
            },
        }
    )

    logger = Logger(tmp_path, cfg=cfg)

    assert captured_kwargs["project"] == "robomimic_dp"
    assert captured_kwargs["entity"] == "younggyo"
    assert captured_kwargs["name"] == "dp_tool_hang_h8"
    assert captured_kwargs["id"] == "ma9u6uis"
    assert captured_kwargs["resume"] == "must"
    assert logger.wandb_run_id == "ma9u6uis"


def test_logger_does_not_pass_wandb_entity_when_null(monkeypatch, tmp_path):
    captured_kwargs = {}

    class DummyRun:
        id = "run123"

    def fake_wandb_init(**kwargs):
        captured_kwargs.update(kwargs)
        return DummyRun()

    monkeypatch.setattr("robobase.logger.wandb.init", fake_wandb_init)

    cfg = OmegaConf.create(
        {
            "save_csv": False,
            "wandb": {
                "use": True,
                "project": "robomimic_dp",
                "entity": None,
                "name": "dp_tool_hang_h8",
                "id": None,
                "resume": None,
            },
            "tb": {
                "use": False,
                "log_dir": str(tmp_path),
                "name": None,
            },
        }
    )

    Logger(tmp_path, cfg=cfg)

    assert "entity" not in captured_kwargs


def test_logger_converts_grad_tensor_without_importing_torch(tmp_path):
    cfg = OmegaConf.create(
        {
            "save_csv": False,
            "wandb": {"use": False},
            "tb": {"use": False},
        }
    )
    logger = Logger(tmp_path, cfg=cfg)

    value = logger._to_numpy_value(_GradTensorLike([1.0, 2.0]))

    np.testing.assert_array_equal(value, np.array([1.0, 2.0]))


def test_tensorboard_logger_uses_tensorboardx(monkeypatch, tmp_path):
    captured_logdirs = []

    class DummySummaryWriter:
        def __init__(self, logdir):
            captured_logdirs.append(logdir)

    fake_tensorboardx = types.SimpleNamespace(SummaryWriter=DummySummaryWriter)
    monkeypatch.setitem(sys.modules, "tensorboardX", fake_tensorboardx)

    cfg = OmegaConf.create(
        {
            "save_csv": False,
            "wandb": {"use": False},
            "tb": {
                "use": True,
                "log_dir": str(tmp_path),
                "name": "tb_run",
            },
        }
    )

    Logger(tmp_path, cfg=cfg)

    assert captured_logdirs == [str(tmp_path / "tb_run")]


def test_pretrain_console_log_uses_step_label(capsys, tmp_path):
    cfg = OmegaConf.create(
        {
            "save_csv": False,
            "wandb": {"use": False},
            "tb": {"use": False},
        }
    )
    logger = Logger(tmp_path, cfg=cfg)

    logger.log_metrics(
        {
            "iteration": 123,
            "total_time": 1.0,
            "buffer_size": 10,
            "agent_batched_updates_per_second": 2.0,
        },
        step=123,
        prefix="pretrain",
    )

    output = capsys.readouterr().out
    assert "Step: 123" in output
    assert "Iter:" not in output
