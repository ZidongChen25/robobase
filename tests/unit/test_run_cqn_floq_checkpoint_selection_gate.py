from pathlib import Path

import pytest

from scripts.run_cqn_floq_checkpoint_selection_gate import (
    EvalJob,
    TrainingSeed,
    _training_seed,
    build_eval_command,
    select_top_steps,
    select_winner,
)


def test_training_seed_parser_and_eval_command(tmp_path):
    item = _training_seed("seed1=clean,clean.pkl,flow")
    assert item == TrainingSeed(
        "seed1",
        Path("clean"),
        Path("clean.pkl"),
        Path("flow"),
    )
    with pytest.raises(Exception, match="training-seed must be"):
        _training_seed("seed1=clean,clean.pkl")

    command = build_eval_command(
        EvalJob(item, "flow", 5_000),
        output=tmp_path / "flow.json",
        work_dir=tmp_path / "work",
        gpu_id=5,
        episodes=10,
        seed_start=101_000,
        flow_readout="distill",
        num_flow_steps=None,
        policy_value_beta=1.0,
    )
    assert command[command.index("--snapshot") + 1] == (
        "flow/snapshots/5000_snapshot.pkl"
    )
    assert command[command.index("--policy-value-beta") + 1] == "1"
    assert command[command.index("--flow-readout") + 1] == "distill"

    direct = build_eval_command(
        EvalJob(item, "flow", 5_000),
        output=tmp_path / "direct.json",
        work_dir=tmp_path / "direct-work",
        gpu_id=1,
        episodes=10,
        seed_start=106_000,
        flow_readout="auto",
        num_flow_steps=None,
        policy_value_beta=3.0,
    )
    assert direct[direct.index("--flow-readout") + 1] == "auto"
    assert direct[direct.index("--policy-value-beta") + 1] == "3"
    assert "--num-flow-steps" not in direct

    truncated = build_eval_command(
        EvalJob(item, "flow", 4_000),
        output=tmp_path / "truncated.json",
        work_dir=tmp_path / "truncated-work",
        gpu_id=5,
        episodes=10,
        seed_start=190_000,
        flow_readout="integrated",
        num_flow_steps=10,
        policy_value_beta=0.3,
        return_sample_aggregation="truncated_mean",
        num_action_flow_samples=10,
        return_sample_truncate_top=1,
    )
    assert truncated[
        truncated.index("--return-sample-aggregation") + 1
    ] == "truncated_mean"
    assert truncated[
        truncated.index("--num-action-flow-samples") + 1
    ] == "10"
    assert truncated[
        truncated.index("--return-sample-truncate-top") + 1
    ] == "1"


def test_checkpoint_selection_uses_success_then_earlier_tie_break():
    rows = {1_000: 0.4, 2_000: 0.6, 3_000: 0.6, 4_000: 0.2}

    assert select_top_steps(rows, top_k=2) == [2_000, 3_000]
    assert select_winner(rows) == 2_000

    with pytest.raises(ValueError):
        select_top_steps(rows, top_k=5)
