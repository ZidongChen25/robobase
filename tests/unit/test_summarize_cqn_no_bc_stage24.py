import csv

import pytest

from scripts.summarize_cqn_no_bc_stage23 import (
    CANDIDATE_FIELDS,
    EXTENDED_STEPS,
    SHORT_STEPS,
)
from scripts.summarize_cqn_no_bc_stage24 import (
    _scale_replication,
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


def _run(tmp_path, name, short, extended=None, candidate=False):
    run = tmp_path / name
    _write_eval(run / "val50_seeds400.csv", SHORT_STEPS, short)
    if extended is not None:
        _write_eval(
            run / "val50_ext20k_seeds400.csv",
            EXTENDED_STEPS,
            extended,
        )
    if candidate:
        with (run / "train.csv").open("w", newline="") as handle:
            fields = ("env_steps",) + CANDIDATE_FIELDS
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for step in (*SHORT_STEPS, *EXTENDED_STEPS):
                writer.writerow(
                    {
                        "env_steps": step,
                        **{field: 0.1 for field in CANDIDATE_FIELDS},
                    }
                )
    return run


def test_stage24_reports_matched_seed3_and_defers_seed1_delta(tmp_path):
    baseline1 = _run(tmp_path, "b1", (0.1, 0.2, 0.3, 0.48))
    baseline3 = _run(
        tmp_path,
        "b3",
        (0.1, 0.3, 0.36, 0.46),
        (0.50, 0.56, 0.50, 0.40),
    )
    treatment1 = _run(
        tmp_path,
        "t1",
        (0.1, 0.2, 0.4, 0.56),
        (0.58, 0.60, 0.54, 0.50),
        candidate=True,
    )
    treatment3 = _run(
        tmp_path,
        "t3",
        (0.0, 0.32, 0.32, 0.50),
        (0.54, 0.62, 0.58, 0.52),
        candidate=True,
    )

    result = summarize(
        baseline1,
        baseline3,
        treatment1,
        treatment3,
    )

    assert result["seed3_matched_scale_replication"]["improvement"] == (
        pytest.approx(0.06)
    )
    assert (
        result["seed3_matched_scale_replication"]["replication"]
        == "strong_replication"
    )
    assert result["seed1_candidate_extension"]["post10k_method_delta"] == (
        "pending matched control extension"
    )
    assert result["next_decision"] == (
        "extend_seed1_no_bc_control_then_apply_three_seed_gate"
    )


def test_stage24_scale_replication_requires_post10k_gain():
    assert _scale_replication(0.06, 10000) == (
        "partial_nonnegative_replication"
    )
    assert _scale_replication(-0.01, 15000) == (
        "not_replicated_on_seed3"
    )
