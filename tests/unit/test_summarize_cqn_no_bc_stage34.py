from omegaconf import OmegaConf
import pytest

from scripts.summarize_cqn_no_bc_stage34 import (
    _beam_contract,
    _map_decision,
)


def _write_phases(
    run_dir,
    *,
    width,
    twin=True,
    episodic=True,
    autoregressive=False,
):
    phase_dir = run_dir.parent / "phase_configs"
    phase_dir.mkdir(parents=True)
    for phase in ("offline", "online"):
        OmegaConf.save(
            OmegaConf.create(
                {
                    "method": {
                        "twin_rollout_beam_width": width,
                        "pessimistic_twin_critic": twin,
                        "episodic_twin_head_exploration": episodic,
                        "autoregressive_action_dims": autoregressive,
                        "use_dueling": False,
                    }
                }
            ),
            phase_dir / f"{phase}_seed1.yaml",
        )


def test_stage34_contract_accepts_exact_width_and_legacy_width_one(tmp_path):
    beam_dir = tmp_path / "beam_seed1"
    _write_phases(beam_dir, width=8)
    assert _beam_contract(beam_dir, 1, 8)["online"][
        "twin_rollout_beam_width"
    ] == 8

    baseline_root = tmp_path / "baseline"
    baseline_dir = baseline_root / "seed1"
    phase_dir = baseline_root / "phase_configs"
    phase_dir.mkdir(parents=True)
    for phase in ("offline", "online"):
        OmegaConf.save(
            OmegaConf.create(
                {
                    "method": {
                        "pessimistic_twin_critic": True,
                        "episodic_twin_head_exploration": True,
                        "autoregressive_action_dims": False,
                        "use_dueling": False,
                    }
                }
            ),
            phase_dir / f"{phase}_seed1.yaml",
        )
    assert _beam_contract(baseline_dir, 1, 1)["offline"][
        "twin_rollout_beam_width"
    ] == 1


def test_stage34_contract_rejects_wrong_width_and_platform(tmp_path):
    run_dir = tmp_path / "beam_seed1"
    _write_phases(run_dir, width=4)
    with pytest.raises(ValueError, match="expected 8"):
        _beam_contract(run_dir, 1, 8)

    run_dir = tmp_path / "bad" / "beam_seed1"
    _write_phases(run_dir, width=8, autoregressive=True)
    with pytest.raises(ValueError, match="autoregressive_action_dims"):
        _beam_contract(run_dir, 1, 8)


def test_stage34_decision_names_joint_beam_followup():
    assert _map_decision(
        "run_direct_head_seed3_then_update_matched_confirmation"
    ) == "run_joint_beam_seed3_matched_confirmation"
    assert _map_decision(
        "extend_direct_head_to50k_before_rejection"
    ) == "extend_joint_beam_to50k_before_rejection"
    assert _map_decision(
        "stop_direct_head_and_test_pessimistic_double_q"
    ) == "stop_joint_beam_primary_and_continue_registered_scale_sentinel"
