import csv

from omegaconf import OmegaConf
import pytest

from scripts.summarize_cqn_no_bc_stage33_scale_sentinel import (
    OFFLINE_UPDATES,
    ONLINE_STEPS,
    _scale_decision,
    _scale_arm,
    _scale_phase_contract,
)


def _write_scale_phases(run_dir, *, exploration, bc_lambda=0.0):
    phase_dir = run_dir.parent / "phase_configs"
    phase_dir.mkdir(parents=True)
    common = {
        "batch_size": 16,
        "is_imitation_learning": False,
        "use_self_imitation": False,
        "replay": {
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
            "episodic_twin_head_exploration": exploration,
            "mc_lower_bound_target": True,
            "td_target_action_source": "critic_replay_max",
        },
    }
    for phase, frames, demo_batch, demo_only, force in (
        ("offline", 10_000, 32, True, 1.0),
        ("online", 111_000, 16, False, 0.0),
    ):
        cfg = OmegaConf.create(
            {
                **common,
                "num_pretrain_steps": 10_000,
                "num_train_frames": frames,
                "demo_batch_size": demo_batch,
                "replay": {
                    **common["replay"],
                    "demo_only_updates": demo_only,
                },
                "method": {
                    **common["method"],
                    "demo_behavior_force_probability": force,
                },
            }
        )
        OmegaConf.save(cfg, phase_dir / f"{phase}_seed4.yaml")


def test_stage33_scale_decision_requires_large_paired_gain_for_full_protocol():
    decision, flags = _scale_decision(0.10, 0.42)
    assert decision == "launch_stage33_full_multiseed_101k_protocol"
    assert flags["strong_scale_signal"]

    decision, flags = _scale_decision(0.22, 0.24)
    assert decision == "run_second_matched_scale_seed_before_decision"
    assert flags["weak_scale_signal"]

    decision, flags = _scale_decision(0.10, 0.18)
    assert decision == "scale_sentinel_does_not_support_episodic_twin"
    assert not flags["weak_scale_signal"]


@pytest.mark.parametrize("exploration", [False, True])
def test_stage33_scale_phase_contract_accepts_exact_reward_only_pair(
    tmp_path,
    exploration,
):
    run_dir = tmp_path / "arm" / "offline_twin_seed4"
    _write_scale_phases(run_dir, exploration=exploration)

    result = _scale_phase_contract(
        run_dir,
        expected_exploration=exploration,
    )

    assert result["verified"]
    assert result["phases"]["online"]["num_train_frames_global_clock"] == 111_000


def test_stage33_scale_phase_contract_rejects_imitation_weight(tmp_path):
    run_dir = tmp_path / "arm" / "offline_twin_seed4"
    _write_scale_phases(run_dir, exploration=True, bc_lambda=0.1)

    with pytest.raises(ValueError, match="bc_lambda"):
        _scale_phase_contract(run_dir, expected_exploration=True)


def test_stage33_scale_arm_maps_raw_clock_and_selects_earliest_peak(tmp_path):
    run_dir = tmp_path / "offline_twin_seed4"
    run_dir.mkdir()
    with (run_dir / "val50_seeds400_scale_raw_steps.csv").open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("env_steps", "episode_success"),
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
            handle,
            fieldnames=(
                "env_steps",
                "critic_loss",
                "episodic_twin_head_assignments",
                "episodic_twin_head0_rate",
                "episodic_twin_head1_rate",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "env_steps": 110_000,
                "critic_loss": 1.2,
                "episodic_twin_head_assignments": 20,
                "episodic_twin_head0_rate": 0.45,
                "episodic_twin_head1_rate": 0.55,
            }
        )

    arm = _scale_arm(run_dir)

    assert arm["offline_endpoint_success"] == 0.0
    assert arm["best_online_step"] == 50_000
    assert arm["best_success"] == pytest.approx(0.4)
    assert arm["final_101k_success"] == pytest.approx(0.4)
    assert arm["final_metrics"]["episodic_twin_head_assignments"] == 20.0
