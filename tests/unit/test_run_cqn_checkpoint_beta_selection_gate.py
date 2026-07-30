from pathlib import Path

import pytest

from scripts.run_cqn_checkpoint_beta_selection_gate import (
    EvalJob,
    TrainingSeed,
    build_eval_command,
    select_global_beta_and_steps,
)


def test_eval_command_freezes_direct_scalar_checkpoint_and_beta(tmp_path):
    item = TrainingSeed(
        "seed1",
        Path("clean"),
        Path("clean.pkl"),
        Path("direct-q"),
    )
    command = build_eval_command(
        EvalJob(item, "candidate", 5_000, 1.0),
        output=tmp_path / "candidate.json",
        work_dir=tmp_path / "work",
        gpu_id=5,
        episodes=50,
        seed_start=112_000,
        candidate_readout="auto",
        num_flow_steps=None,
    )

    assert command[command.index("--run-dir") + 1] == "direct-q"
    assert command[command.index("--snapshot") + 1] == (
        "direct-q/snapshots/5000_snapshot.pkl"
    )
    assert command[command.index("--policy-value-beta") + 1] == "1"
    assert command[command.index("--flow-readout") + 1] == "auto"


def test_global_beta_selection_uses_mean_best_seed_success():
    validation = {
        "seed1": {
            0.3: {1_000: 0.4, 2_000: 0.5},
            1.0: {1_000: 0.7, 2_000: 0.6},
            3.0: {1_000: 0.6, 2_000: 0.5},
        },
        "seed2": {
            0.3: {1_000: 0.5, 2_000: 0.4},
            1.0: {1_000: 0.5, 2_000: 0.8},
            3.0: {1_000: 0.6, 2_000: 0.5},
        },
    }

    beta, steps, means = select_global_beta_and_steps(validation)

    assert beta == pytest.approx(1.0)
    assert steps == {"seed1": 1_000, "seed2": 2_000}
    assert means[1.0] == pytest.approx(0.75)


def test_global_beta_tie_prefers_stronger_bc_prior():
    validation = {
        "seed1": {0.3: {1_000: 0.5}, 3.0: {1_000: 0.5}},
        "seed2": {0.3: {1_000: 0.5}, 3.0: {1_000: 0.5}},
    }

    beta, _, _ = select_global_beta_and_steps(validation)

    assert beta == pytest.approx(3.0)
