from omegaconf import OmegaConf

from robobase.logger import Logger


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
