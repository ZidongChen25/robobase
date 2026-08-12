import pytest

from scripts.analyze_cqn_value_fidelity import GROUPS, parse_groups, summarize
from scripts.run_cqn_no_bc_stage30_diagnostics import (
    _diagnostic_metrics,
    _selected_jobs,
)


def _record(group, *, predicted_q, discounted_return, distance):
    return {
        "group": group,
        "predicted_q": predicted_q,
        "greedy_predicted_q": predicted_q + 0.1,
        "replay_minus_greedy_q": -0.1,
        "discounted_return": discounted_return,
        "support_clipped_return": min(discounted_return, 0.7),
        "first_success_return": discounted_return,
        "distance_to_terminal": distance,
        "replay_bin_top1_rate": 0.25,
        "replay_bin_top1_rate_current_action": 0.5,
        "replay_bin_top1_rate_by_sequence": [0.5, 0.0],
        "replay_bin_top2_rate": 0.75,
        "replay_bin_top2_rate_current_action": 1.0,
        "replay_bin_top2_rate_by_sequence": [1.0, 0.5],
        "greedy_bin_agreement": 0.25,
        "greedy_bin_agreement_current_action": 0.5,
        "behavior_bin_agreement": 0.25,
        "behavior_bin_agreement_current_action": 0.5,
        "critic_behavior_disagreement": 0.0,
        "critic_behavior_disagreement_current_action": 0.0,
        "twin_head_action_diagnostics_available": 1.0,
        "twin_head_bin_disagreement": 0.3,
        "twin_head_bin_disagreement_current_action": 0.4,
        "twin_head_chunk_disagreement": 1.0,
        "twin_head_current_action_disagreement": 0.75,
        "twin_head_normalized_action_l1": 0.2,
        "twin_head_normalized_current_action_l1": 0.25,
        "normalized_replay_bin_rank": 0.5,
        "candidate_q_span": 0.4,
        "candidate_top2_gap": 0.1,
        "max_minus_replay_q": 0.2,
        "max_minus_replay_q_by_sequence": [0.1, 0.3],
        "candidate_q_span_by_sequence": [0.2, 0.6],
    }


def test_value_fidelity_summary_reports_top2_and_calibration_by_distance():
    records = [
        _record(
            group,
            predicted_q=0.6,
            discounted_return=0.8,
            distance=8 if group == "demo_success" else 40,
        )
        for group in GROUPS
    ]

    result = summarize(records, bins=5)
    demo = result["demo_success"]

    assert demo["imitation"]["replay_bin_top2_rate"] == pytest.approx(0.75)
    assert demo["imitation"]["replay_bin_top2_rate_by_sequence"] == [
        1.0,
        0.5,
    ]
    assert demo["value"]["q_raw_return_mae"] == pytest.approx(0.2)
    assert demo["value"]["q_support_clipped_return_mae"] == pytest.approx(0.1)
    assert demo["value"]["replay_minus_greedy_q_mean"] == pytest.approx(-0.1)
    assert demo["value"]["q_raw_return_mae_by_terminal_distance"]["0-15"][
        "num_samples"
    ] == 1
    assert result["interpretation"]["random_top2_reference"] == pytest.approx(
        0.4
    )
    assert demo["exploration"]["twin_head_bin_disagreement"] == pytest.approx(
        0.3
    )
    assert demo["exploration"][
        "twin_head_current_action_disagreement_rate"
    ] == pytest.approx(0.75)


def test_value_fidelity_group_parser_allows_demo_only_audit():
    assert parse_groups("demo_success,demo_failure") == (
        "demo_success",
        "demo_failure",
    )
    with pytest.raises(ValueError, match="invalid replay groups"):
        parse_groups("demo_success,unknown")
    with pytest.raises(ValueError, match="duplicates"):
        parse_groups("demo_success,demo_success")


def test_stage30_diagnostics_selects_online_and_offline_checkpoints(tmp_path):
    arm = {
        "run_dir": str(tmp_path / "run"),
        "selected_snapshot": str(tmp_path / "selected.pkl"),
        "offline_endpoint_snapshot": str(tmp_path / "offline.pkl"),
    }
    summary = {
        "matched_online_only_controls": {
            "seed1": arm,
            "seed2": arm,
        },
        "offline_then_online_treatments": {
            "seed1": arm,
            "seed2": arm,
        },
    }

    jobs = _selected_jobs(tmp_path, summary)

    assert len(jobs) == 6
    assert {job["name"] for job in jobs} == {
        "online_only_seed1_selected",
        "online_only_seed2_selected",
        "offline_then_online_seed1_selected",
        "offline_then_online_seed2_selected",
        "offline_endpoint_seed1",
        "offline_endpoint_seed2",
    }


def test_stage30_diagnostic_metric_projection_uses_demo_success():
    payload = {
        "snapshot": "/tmp/checkpoint.pkl",
        "summary": {
            "demo_success": {
                "num_samples": 8,
                "imitation": {
                    "replay_bin_top1_rate": 0.5,
                    "replay_bin_top1_rate_current_action": 0.6,
                    "replay_bin_top2_rate": 0.7,
                    "replay_bin_top2_rate_current_action": 0.8,
                },
                "value": {
                    "predicted_q_mean": 0.4,
                    "greedy_predicted_q_mean": 0.45,
                    "replay_minus_greedy_q_mean": -0.05,
                    "q_raw_return_mae": 0.2,
                    "q_raw_return_mae_by_terminal_distance": {},
                    "q_raw_return_pearson": 0.3,
                },
                "collapse": {
                    "max_minus_replay_q": 0.1,
                    "candidate_q_span": 0.5,
                },
            }
        },
    }

    result = _diagnostic_metrics(payload)

    assert result["expert_bin_top2"] == pytest.approx(0.7)
    assert result["expert_minus_greedy_q"] == pytest.approx(-0.05)
    assert result["rtg_mae"] == pytest.approx(0.2)
