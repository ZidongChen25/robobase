from pathlib import Path

import pytest

from scripts.summarize_cqn_online_gate import (
    build_summary,
    read_eval_curve,
    summarize_run,
)


HEADER = "iteration,episode_success,episode_reward\n"


def _write_eval(run_dir: Path, rows: list[str]) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "eval.csv").write_text(HEADER + "".join(rows))


def test_read_eval_curve_skips_repeated_header_and_keeps_last(tmp_path):
    path = tmp_path / "eval.csv"
    path.write_text(
        HEADER
        + "2500,0.4,0.4\n"
        + HEADER
        + "2500,0.48,0.48\n"
        + "5000,0.6,0.6\n"
    )

    assert read_eval_curve(path) == {2500: 0.48, 5000: 0.6}


def test_summarize_uses_earliest_best_checkpoint(tmp_path):
    run_dir = tmp_path / "run"
    _write_eval(
        run_dir,
        [
            "2500,0.4,0.4\n",
            "5000,0.8,0.8\n",
            "7500,0.8,0.8\n",
            "10000,0.6,0.6\n",
        ],
    )

    result = summarize_run(
        "method",
        run_dir,
        [2500, 5000, 7500, 10000],
        allow_incomplete=False,
    )

    assert result["best_step"] == 5000
    assert result["best_success"] == 0.8
    assert result["last_success"] == 0.6


def test_build_summary_reports_best_gate_and_same_step_curve(tmp_path):
    direct = tmp_path / "direct"
    flow = tmp_path / "flow"
    _write_eval(direct, ["2500,0.4,0\n", "5000,0.6,0\n"])
    _write_eval(flow, ["2500,0.5,0\n", "5000,0.7,0\n"])

    result = build_summary(
        [("direct", direct), ("flow", flow)],
        [2500, 5000],
        baseline_label="direct",
        challenger_label="flow",
        allow_incomplete=False,
    )

    comparison = result["comparison"]
    assert comparison["gate_passed"] is True
    assert comparison["best_success_delta"] == pytest.approx(0.1)
    assert comparison["same_step_success_delta"] == pytest.approx(
        {"2500": 0.1, "5000": 0.1}
    )


def test_incomplete_curve_fails_unless_explicitly_allowed(tmp_path):
    run_dir = tmp_path / "run"
    _write_eval(run_dir, ["2500,0.4,0\n"])

    with pytest.raises(ValueError, match="missing predeclared"):
        summarize_run(
            "method",
            run_dir,
            [2500, 5000],
            allow_incomplete=False,
        )

    result = summarize_run(
        "method",
        run_dir,
        [2500, 5000],
        allow_incomplete=True,
    )
    assert result["complete"] is False
    assert result["missing_steps"] == [5000]
