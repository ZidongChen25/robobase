import json
from pathlib import Path

import pytest

from scripts.run_cqn_pcbf_lambda_gate import (
    FlowCheckpoint,
    confirmation_gate,
    discover_flow_checkpoints,
    paired_statistics,
    select_checkpoint,
)


def _payload(values, seed_start=100):
    return {
        "status": "ok",
        "episode_success": sum(values) / len(values),
        "episode_results": [
            {"seed": seed_start + index, "episode_success": value}
            for index, value in enumerate(values)
        ],
    }


def test_discover_flow_checkpoints_ignores_latest(tmp_path: Path):
    run = tmp_path / "run"
    snapshots = run / "snapshots"
    snapshots.mkdir(parents=True)
    for name in ("2000_snapshot.pkl", "1000_snapshot.pkl", "latest_snapshot.pkl"):
        (snapshots / name).touch()

    found = discover_flow_checkpoints([("lambda0", run)])

    assert [checkpoint.step for checkpoint in found] == [1000, 2000]
    assert [checkpoint.label for checkpoint in found] == [
        "lambda0_step1000",
        "lambda0_step2000",
    ]


def test_discover_flow_checkpoints_rejects_duplicate_labels(tmp_path: Path):
    with pytest.raises(ValueError, match="unique"):
        discover_flow_checkpoints(
            [("lambda0", tmp_path / "a"), ("lambda0", tmp_path / "b")]
        )


def test_select_checkpoint_uses_success_then_earlier_step(tmp_path: Path):
    checkpoints = [
        FlowCheckpoint("lambda0", tmp_path, tmp_path / "a", 1000, 0),
        FlowCheckpoint("lambda0", tmp_path, tmp_path / "b", 2000, 0),
    ]
    payloads = {
        checkpoint.label: {"episode_success": 0.4}
        for checkpoint in checkpoints
    }

    selected = select_checkpoint(
        checkpoints,
        payloads,
        run_label="lambda0",
    )

    assert selected.step == 1000


def test_paired_statistics_preserves_common_seed_pairing():
    candidate = _payload([1.0, 1.0, 0.0, 1.0])
    baseline = _payload([0.0, 1.0, 0.0, 0.0])

    row = paired_statistics(
        candidate,
        baseline,
        bootstrap_samples=200,
        bootstrap_seed=7,
    )

    assert row["paired_delta"] == pytest.approx(0.5)
    assert row["paired_wins"] == 2
    assert row["paired_losses"] == 0
    assert row["paired_ties"] == 2


def test_paired_statistics_rejects_mismatched_seeds():
    with pytest.raises(ValueError, match="same seeds"):
        paired_statistics(
            _payload([1.0], seed_start=10),
            _payload([0.0], seed_start=20),
            bootstrap_samples=10,
            bootstrap_seed=1,
        )


def test_confirmation_gate_requires_ci_and_more_wins():
    passing = {
        "paired_delta": 0.1,
        "paired_delta_ci95": [0.0, 0.2],
        "paired_wins": 5,
        "paired_losses": 2,
    }
    assert confirmation_gate(passing)[0] == "pass"

    crossing = dict(passing, paired_delta_ci95=[-0.01, 0.2])
    assert confirmation_gate(crossing)[0] == "fail"

    tied_wins = dict(passing, paired_wins=2, paired_losses=2)
    assert confirmation_gate(tied_wins)[0] == "fail"
