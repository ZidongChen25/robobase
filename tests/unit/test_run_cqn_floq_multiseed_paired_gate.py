from pathlib import Path

import pytest

from scripts.run_cqn_floq_multiseed_paired_gate import (
    Pair,
    _pair,
    build_eval_jobs,
    build_eval_command,
)


def test_pair_parser_requires_four_paths():
    pair = _pair("seed1=clean,clean.pkl,flow,flow.pkl")

    assert pair.label == "seed1"
    assert pair.clean_snapshot == Path("clean.pkl")
    assert pair.flow_snapshot == Path("flow.pkl")

    with pytest.raises(Exception, match="pair must be"):
        _pair("seed1=clean,clean.pkl,flow")


def test_eval_commands_freeze_clean_and_flow_readouts(tmp_path):
    pair = Pair(
        "seed1",
        Path("clean-run"),
        Path("clean.pkl"),
        Path("flow-run"),
        Path("flow.pkl"),
    )
    shared = {
        "pair": pair,
        "output": tmp_path / "result.json",
        "work_dir": tmp_path / "work",
        "gpu_id": 5,
        "episodes": 200,
        "seed_start": 92_000,
        "policy_value_beta": 1.0,
        "flow_readout": "distill",
        "num_flow_steps": None,
    }

    clean = build_eval_command(candidate=False, **shared)
    flow = build_eval_command(candidate=True, **shared)

    assert clean[clean.index("--run-dir") + 1] == "clean-run"
    assert clean[clean.index("--snapshot") + 1] == "clean.pkl"
    assert clean[clean.index("--policy-value-beta") + 1] == "bc"
    assert clean[clean.index("--flow-readout") + 1] == "auto"
    assert flow[flow.index("--run-dir") + 1] == "flow-run"
    assert flow[flow.index("--snapshot") + 1] == "flow.pkl"
    assert flow[flow.index("--policy-value-beta") + 1] == "1"
    assert flow[flow.index("--flow-readout") + 1] == "distill"
    assert flow[flow.index("--num-eval-episodes") + 1] == "200"
    assert flow[flow.index("--eval-seed-start") + 1] == "92000"


def test_eval_command_supports_direct_c51_auto_readout(tmp_path):
    pair = Pair(
        "seed1",
        Path("clean-run"),
        Path("clean.pkl"),
        Path("direct-run"),
        Path("direct.pkl"),
    )

    command = build_eval_command(
        pair,
        candidate=True,
        output=tmp_path / "direct.json",
        work_dir=tmp_path / "work",
        gpu_id=1,
        episodes=20,
        seed_start=106_000,
        policy_value_beta=3.0,
        flow_readout="auto",
        num_flow_steps=None,
    )

    assert command[command.index("--run-dir") + 1] == "direct-run"
    assert command[command.index("--policy-value-beta") + 1] == "3"
    assert command[command.index("--flow-readout") + 1] == "auto"
    assert "--num-flow-steps" not in command


def test_clean_and_candidate_are_independent_gpu_jobs():
    pairs = [
        Pair(
            f"seed{seed}",
            Path(f"clean-{seed}"),
            Path(f"clean-{seed}.pkl"),
            Path(f"flow-{seed}"),
            Path(f"flow-{seed}.pkl"),
        )
        for seed in (1, 2, 3)
    ]

    jobs = build_eval_jobs(pairs)

    assert len(jobs) == 6
    assert [
        (job.pair.label, job.candidate)
        for job in jobs
    ] == [
        ("seed1", False),
        ("seed1", True),
        ("seed2", False),
        ("seed2", True),
        ("seed3", False),
        ("seed3", True),
    ]
