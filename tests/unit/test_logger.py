import csv
import sys
import types

import numpy as np
from omegaconf import OmegaConf

from robobase.logger import Logger, MetersGroup


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


class _ArrayLike:
    def __init__(self, value):
        self._value = np.asarray(value)

    def __array__(self):
        return self._value


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


def test_logger_keeps_executed_action_matrix_as_grayscale_image(
    monkeypatch,
    tmp_path,
):
    cfg = OmegaConf.create(
        {
            "save_csv": False,
            "wandb": {"use": False},
            "tb": {"use": False},
        }
    )
    logger = Logger(tmp_path, cfg=cfg)
    logger._use_wandb = True
    captured = []
    monkeypatch.setattr(
        "robobase.logger.wandb.Image",
        lambda value: captured.append(np.asarray(value)) or "image",
    )
    executed_action = np.arange(16 * 15, dtype=np.float32).reshape(16, 15)

    logger._log(
        "train/env_info/executed_action",
        executed_action,
        step=1,
    )

    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0], executed_action)


def test_logger_expands_one_dimensional_array_like_diagnostic(monkeypatch, tmp_path):
    cfg = OmegaConf.create(
        {
            "save_csv": False,
            "wandb": {"use": False},
            "tb": {"use": False},
        }
    )
    logger = Logger(tmp_path, cfg=cfg)
    logger._use_wandb = True
    monkeypatch.setattr(logger, "_dump", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "robobase.logger.wandb.Image",
        lambda _value: (_ for _ in ()).throw(
            AssertionError("1-D diagnostics must not be logged as images")
        ),
    )

    logger.log_metrics(
        {"per_action": _ArrayLike(np.arange(15, dtype=np.float32))},
        step=1,
        prefix="train",
    )

    assert set(logger._wandb_logs) == {
        f"train/per_action{index}" for index in range(15)
    }


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


def test_csv_schema_expands_when_online_metrics_appear_later(tmp_path):
    path = tmp_path / "train.csv"
    meters = MetersGroup(path, [], save_csv=True)
    meters.log("train_env_steps", 0)
    meters.dump(0, "train")
    meters.log("train_env_steps", 1000)
    meters.log("train_episode_reward", 750.0)
    meters.dump(1000, "train")

    with path.open() as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["episode_reward"] == "0.0"
    assert rows[1]["episode_reward"] == "750.0"
