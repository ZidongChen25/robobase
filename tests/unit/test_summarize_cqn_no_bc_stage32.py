from omegaconf import OmegaConf

from scripts.summarize_cqn_no_bc_stage32 import (
    _map_decision,
    _twin_contract,
)


def _write_phases(run_dir, *, twin_field):
    phase_dir = run_dir.parent / "phase_configs"
    phase_dir.mkdir(parents=True)
    for phase in ("offline", "online"):
        method = {"use_dueling": False}
        if twin_field is not None:
            method["pessimistic_twin_critic"] = twin_field
        OmegaConf.save(
            OmegaConf.create({"method": method}),
            phase_dir / f"{phase}_seed1.yaml",
        )


def test_stage32_contract_accepts_historical_missing_false_field(tmp_path):
    run_dir = tmp_path / "direct_seed1"
    _write_phases(run_dir, twin_field=None)

    result = _twin_contract(run_dir, 1, False)

    assert not result["offline"]["pessimistic_twin_critic"]
    assert not result["online"]["pessimistic_twin_critic"]


def test_stage32_decision_names_pessimistic_twin_followup():
    assert _map_decision(
        "run_direct_head_seed3_then_update_matched_confirmation"
    ) == "run_pessimistic_twin_seed3_then_update_matched_confirmation"
    assert _map_decision(
        "extend_direct_head_to50k_before_rejection"
    ) == "extend_pessimistic_twin_to50k_before_rejection"
