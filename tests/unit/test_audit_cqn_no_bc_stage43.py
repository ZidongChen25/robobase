from pathlib import Path

import pytest
import yaml

from scripts.audit_cqn_no_bc_stage43 import audit_phase_config


def _config(*, phase: str, self_imitation: bool = False) -> dict:
    offline = phase == "offline"
    return {
        "batch_size": 256,
        "demo_batch_size": 256,
        "num_pretrain_steps": 10000,
        "is_imitation_learning": False,
        "use_self_imitation": self_imitation,
        "replay": {"demo_only_updates": offline},
        "method": {
            "is_rl": True,
            "strict_demo_rl_only": True,
            "strict_allow_reward_only_success_replay": self_imitation,
            "bc_lambda": 0.0,
            "bc_margin": 0.0,
            "demo_fosd": False,
            "dense_return_q_target": True,
            "dense_return_positive_only": not offline,
            "mc_lower_bound_target": True,
            "td_target_action_source": "critic_replay_max",
            "demo_behavior_force_probability": 1.0 if offline else 0.0,
        },
    }


def _write(path: Path, cfg: dict) -> Path:
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_audit_phase_config_accepts_strict_reward_q_phases(tmp_path: Path):
    offline = audit_phase_config(
        _write(tmp_path / "offline.yaml", _config(phase="offline")),
        phase="offline",
    )
    online = audit_phase_config(
        _write(tmp_path / "online.yaml", _config(phase="online")),
        phase="online",
    )
    assert offline["demo_only_updates"]
    assert online["dense_return_positive_only"]


def test_audit_phase_config_rejects_imitation_objective(tmp_path: Path):
    cfg = _config(phase="online")
    cfg["method"]["bc_lambda"] = 1.0
    with pytest.raises(ValueError, match="bc_lambda is nonzero"):
        audit_phase_config(
            _write(tmp_path / "bad.yaml", cfg),
            phase="online",
        )


def test_offline_self_imitation_flag_requires_explicit_inert_proof(
    tmp_path: Path,
):
    path = _write(
        tmp_path / "historical.yaml",
        _config(phase="offline", self_imitation=True),
    )
    with pytest.raises(ValueError, match="self-imitation can be active"):
        audit_phase_config(path, phase="offline")
    result = audit_phase_config(
        path,
        phase="offline",
        allow_inert_self_imitation_flag=True,
    )
    assert result["self_imitation_configured"]
    assert not result["self_imitation_operationally_active"]
