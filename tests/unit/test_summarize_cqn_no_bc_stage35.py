import csv

from omegaconf import OmegaConf
import pytest

from scripts.summarize_cqn_no_bc_stage35 import (
    OFFLINE_UPDATES,
    ONLINE_STEPS,
    _METRICS,
    _arm,
    _decision,
    _phase_contract,
)


def _write_phases(base, seed, arm, weight, *, bc_lambda=0.0):
    phase_dir = base / f"seed{seed}" / arm / "phase_configs"
    phase_dir.mkdir(parents=True)
    for phase, frames, demo_batch, demo_only, force in (
        ("offline", 10_000, 32, True, 1.0),
        ("online", 111_000, 16, False, 0.0),
    ):
        cfg = OmegaConf.create(
            {
                "num_pretrain_steps": 10_000,
                "num_train_frames": frames,
                "demo_batch_size": demo_batch,
                "is_imitation_learning": False,
                "use_self_imitation": False,
                "replay": {
                    "demo_only_updates": demo_only,
                    "nstep": 1,
                    "auxiliary_nstep": 4,
                    "include_tp1": True,
                    "include_next_action": True,
                },
                "method": {
                    "strict_demo_rl_only": True,
                    "bc_lambda": bc_lambda,
                    "bc_margin": 0.0,
                    "demo_fosd": False,
                    "separate_bc_policy": False,
                    "flow_policy": False,
                    "critic_lambda": 1.0,
                    "use_dueling": False,
                    "pessimistic_twin_critic": True,
                    "episodic_twin_head_exploration": False,
                    "twin_rollout_beam_width": 1,
                    "mc_lower_bound_target": True,
                    "td_target_action_source": "critic_replay_max",
                    "demo_behavior_force_probability": force,
                    "auxiliary_td_loss_weight": weight,
                },
            }
        )
        OmegaConf.save(cfg, phase_dir / f"{phase}_seed{seed}.yaml")


def test_stage35_phase_contract_accepts_exact_control_and_treatment(tmp_path):
    _write_phases(tmp_path, 1, "control", 0.0)
    _write_phases(tmp_path, 1, "treatment", 1.0)

    assert _phase_contract(tmp_path, 1, "control", 0.0)["verified"]
    assert _phase_contract(tmp_path, 1, "treatment", 1.0)["verified"]


def test_stage35_phase_contract_rejects_imitation_weight(tmp_path):
    _write_phases(tmp_path, 1, "treatment", 1.0, bc_lambda=0.1)

    with pytest.raises(ValueError, match="bc_lambda"):
        _phase_contract(tmp_path, 1, "treatment", 1.0)


def test_stage35_arm_maps_raw_clock_and_selects_earliest_peak(tmp_path):
    run_dir = tmp_path / "offline_twin_seed1"
    run_dir.mkdir()
    with (run_dir / "val50_seeds400_full_raw_steps.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("env_steps", "episode_success")
        )
        writer.writeheader()
        writer.writerow(
            {"env_steps": OFFLINE_UPDATES, "episode_success": 0.0}
        )
        for step in ONLINE_STEPS:
            success = 0.4 if step in {50_000, 101_000} else 0.1
            writer.writerow(
                {
                    "env_steps": OFFLINE_UPDATES + step,
                    "episode_success": success,
                }
            )
    with (run_dir / "train.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("env_steps", *_METRICS)
        )
        writer.writeheader()
        for step in (60_000, 111_000):
            writer.writerow(
                {
                    "env_steps": step,
                    **{metric: 0.5 for metric in _METRICS},
                }
            )

    arm = _arm(run_dir)

    assert arm["offline_endpoint_success"] == 0.0
    assert arm["best_online_step"] == 50_000
    assert arm["best_success"] == pytest.approx(0.4)
    assert arm["final_101k_success"] == pytest.approx(0.4)


def test_stage35_decision_applies_frozen_strong_and_weak_gates():
    controls = {
        "seed1": {"best_success": 0.2},
        "seed2": {"best_success": 0.2},
    }
    strong, flags = _decision(
        controls,
        {
            "seed1": {"best_success": 0.45},
            "seed2": {"best_success": 0.4},
        },
    )
    assert strong.startswith("add_seeds3_4")
    assert flags["strong_gate"]

    weak, flags = _decision(
        controls,
        {
            "seed1": {"best_success": 0.25},
            "seed2": {"best_success": 0.2},
        },
    )
    assert weak.startswith("run_third_matched")
    assert flags["weak_gate"]

    rejected, flags = _decision(
        controls,
        {
            "seed1": {"best_success": 0.0},
            "seed2": {"best_success": 0.0},
        },
    )
    assert rejected == "reject_simultaneous_one_plus_four_mechanism"
    assert not flags["weak_gate"]
