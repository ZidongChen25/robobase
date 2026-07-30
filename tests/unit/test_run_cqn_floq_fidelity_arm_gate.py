from pathlib import Path

import pytest

from scripts.run_cqn_floq_fidelity_arm_gate import (
    Candidate,
    EvalJob,
    build_eval_command,
    confirmation_gate,
    paired_result,
    select_validation_winner,
    validation_gate,
)


def _payload(successes, seed_start=100):
    return {
        "status": "ok",
        "episode_results": [
            {
                "seed": seed_start + index,
                "episode_success": success,
            }
            for index, success in enumerate(successes)
        ],
    }


def test_candidate_eval_command_freezes_arm_step_and_beta(tmp_path):
    candidate = Candidate("source01", Path("source-run"), 0)
    command = build_eval_command(
        EvalJob("candidate", candidate, 5_000, 1.0),
        clean_run_dir=Path("clean-run"),
        clean_snapshot=Path("clean.pkl"),
        output=tmp_path / "candidate.json",
        work_dir=tmp_path / "work",
        gpu_id=5,
        episodes=50,
        seed_start=115_000,
        flow_readout="distill",
        num_flow_steps=None,
    )

    assert command[command.index("--run-dir") + 1] == "source-run"
    assert command[command.index("--snapshot") + 1] == (
        "source-run/snapshots/5000_snapshot.pkl"
    )
    assert command[command.index("--policy-value-beta") + 1] == "1"
    assert command[command.index("--flow-readout") + 1] == "distill"


def test_candidate_eval_command_supports_flowcritic_truncated_readout(
    tmp_path,
):
    candidate = Candidate(
        "truncated",
        Path("return-run"),
        0,
        return_sample_aggregation="truncated_mean",
        num_action_flow_samples=10,
        return_sample_truncate_top=1,
    )
    command = build_eval_command(
        EvalJob("candidate", candidate, 4_000, 0.3),
        clean_run_dir=Path("clean-run"),
        clean_snapshot=Path("clean.pkl"),
        output=tmp_path / "candidate.json",
        work_dir=tmp_path / "work",
        gpu_id=1,
        episodes=10,
        seed_start=187_000,
        flow_readout="integrated",
        num_flow_steps=10,
    )

    assert command[command.index("--return-sample-aggregation") + 1] == (
        "truncated_mean"
    )
    assert command[command.index("--num-action-flow-samples") + 1] == "10"
    assert command[command.index("--return-sample-truncate-top") + 1] == "1"


def test_clean_eval_command_uses_frozen_clean_snapshot(tmp_path):
    command = build_eval_command(
        EvalJob("clean"),
        clean_run_dir=Path("clean-run"),
        clean_snapshot=Path("clean.pkl"),
        output=tmp_path / "clean.json",
        work_dir=tmp_path / "work",
        gpu_id=1,
        episodes=50,
        seed_start=115_000,
        flow_readout="distill",
        num_flow_steps=None,
    )

    assert command[command.index("--run-dir") + 1] == "clean-run"
    assert command[command.index("--snapshot") + 1] == "clean.pkl"
    assert command[command.index("--policy-value-beta") + 1] == "bc"
    assert command[command.index("--flow-readout") + 1] == "auto"


def test_joint_selection_ties_prefer_bc_then_early_step_then_arm_order():
    source = Candidate("source01", Path("source"), 0)
    bcfm = Candidate("bcfm8", Path("bcfm"), 1)
    winner = select_validation_winner(
        [
            (bcfm, 2_000, 1.0, 0.6),
            (source, 2_000, 1.0, 0.6),
            (source, 1_000, 1.0, 0.6),
            (source, 1_000, 3.0, 0.6),
        ]
    )

    assert winner[:3] == (source, 1_000, 3.0)


def test_paired_promotion_gates_are_separate():
    result = paired_result(
        _payload([0, 0, 1, 0]),
        _payload([1, 0, 1, 0]),
        bootstrap_replicates=2_000,
        bootstrap_seed=7,
    )

    assert result["paired_delta"] == pytest.approx(0.25)
    assert validation_gate(result, 0.02)[0]
    assert confirmation_gate(result, -0.05)[0]


def test_confirmation_requires_positive_direction():
    result = paired_result(
        _payload([1, 0, 1, 0]),
        _payload([0, 0, 1, 0]),
        bootstrap_replicates=2_000,
        bootstrap_seed=8,
    )

    assert not confirmation_gate(result, -0.05)[0]
