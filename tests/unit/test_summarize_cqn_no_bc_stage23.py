import csv

import pytest

from scripts.summarize_cqn_no_bc_stage23 import (
    CANDIDATE_FIELDS,
    EXTENDED_STEPS,
    SHORT_STEPS,
    _decisions,
    summarize,
)


def _write_eval(path, steps, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("env_steps", "episode_success"),
        )
        writer.writeheader()
        for step, value in zip(steps, values, strict=True):
            writer.writerow(
                {"env_steps": step, "episode_success": value}
            )


def _write_train(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ("env_steps",) + CANDIDATE_FIELDS
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for step in (*SHORT_STEPS, *EXTENDED_STEPS):
            writer.writerow(
                {
                    "env_steps": step,
                    **{field: 0.1 for field in CANDIDATE_FIELDS},
                }
            )


def _run(tmp_path, name, short, extended=None, candidate=False):
    run = tmp_path / name
    _write_eval(run / "val50_seeds400.csv", SHORT_STEPS, short)
    if extended is not None:
        _write_eval(
            run / "val50_ext20k_seeds400.csv",
            EXTENDED_STEPS[: len(extended)],
            extended,
        )
    if candidate:
        _write_train(run / "train.csv")
    return run


def test_stage23_keeps_replication_and_scale_gates_independent(tmp_path):
    baseline1 = _run(tmp_path, "b1", (0.1, 0.2, 0.3, 0.48))
    baseline2 = _run(
        tmp_path,
        "b2",
        (0.1, 0.2, 0.3, 0.46),
        (0.46, 0.40, 0.44, 0.38),
    )
    baseline3 = _run(tmp_path, "b3", (0.1, 0.3, 0.36, 0.46))
    treatment1 = _run(
        tmp_path,
        "t1",
        (0.1, 0.2, 0.4, 0.56),
        candidate=True,
    )
    treatment2 = _run(
        tmp_path,
        "t2",
        (0.1, 0.2, 0.3, 0.44),
        (0.48, 0.52, 0.50, 0.46),
        candidate=True,
    )
    treatment3 = _run(
        tmp_path,
        "t3",
        (0.1, 0.3, 0.4, 0.56),
        candidate=True,
    )

    result = summarize(
        baseline1,
        baseline2,
        baseline3,
        treatment1,
        treatment2,
        treatment3,
    )

    assert result["decision_flags"]["replication_pass"]
    assert result["decision_flags"]["scale_pass"]
    assert result["decision_flags"]["nonnegative_seed_count"] == 2
    assert result["next_decision"] == (
        "run_independent100_and_extend_candidate_seeds1_3"
    )


def test_stage23_missing_scale_checkpoint_fails_closed(tmp_path):
    run = _run(
        tmp_path,
        "incomplete",
        (0.1, 0.2, 0.3, 0.4),
        (0.4, 0.5, 0.6),
        candidate=True,
    )
    with pytest.raises(ValueError, match="missing validation steps"):
        summarize(run, run, run, run, run, run)


def test_stage23_failure_is_not_named_full_budget_failure():
    decision, flags = _decisions(
        {"seed1": 0.01, "seed2": -0.02, "seed3": 0.0},
        scale_delta=0.02,
        scale_best_step=15000,
    )
    assert decision == (
        "stop_exact_candidate_only_variant_without_full_budget_claim"
    )
    assert not flags["replication_pass"]
    assert not flags["scale_pass"]
