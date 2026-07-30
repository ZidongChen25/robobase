from pathlib import Path

import pytest

from scripts.run_cqn_flow_readout_multiseed_gate import (
    Checkpoint,
    Readout,
    _checkpoint,
    build_eval_command,
    selection_checks,
)


def test_readout_command_distinguishes_distill_and_integrated(tmp_path):
    checkpoint = _checkpoint("seed1=run,snapshot.pkl")
    assert checkpoint == Checkpoint(
        "seed1", Path("run"), Path("snapshot.pkl")
    )
    with pytest.raises(Exception, match="checkpoint must be"):
        _checkpoint("seed1=run")

    shared = {
        "checkpoint": checkpoint,
        "output": tmp_path / "result.json",
        "work_dir": tmp_path / "work",
        "gpu_id": 1,
        "episodes": 50,
        "seed_start": 96_000,
        "policy_value_beta": 1.0,
    }
    distill = build_eval_command(
        readout=Readout("distill", "distill", None),
        **shared,
    )
    integrated = build_eval_command(
        readout=Readout("integrated_steps8", "integrated", 8),
        **shared,
    )

    assert distill[distill.index("--flow-readout") + 1] == "distill"
    assert "--num-flow-steps" not in distill
    assert integrated[integrated.index("--flow-readout") + 1] == "integrated"
    assert integrated[integrated.index("--num-flow-steps") + 1] == "8"
    assert integrated[integrated.index("--policy-value-beta") + 1] == "1"


def test_selection_requires_point_wins_and_training_seed_majority():
    passing = {
        "mean_paired_delta": 0.02,
        "aggregate_paired_wins": 8,
        "aggregate_paired_losses": 5,
        "gate_checks": {"positive_training_seed_majority": True},
    }
    assert all(selection_checks(passing).values())
    assert not all(
        selection_checks(
            {**passing, "aggregate_paired_wins": 5}
        ).values()
    )
    assert not all(
        selection_checks(
            {
                **passing,
                "gate_checks": {
                    "positive_training_seed_majority": False
                },
            }
        ).values()
    )
