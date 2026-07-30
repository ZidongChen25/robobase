import copy

from scripts.summarize_cqn_flow_utilization import summarize


def _payload():
    return {
        "status": "ok",
        "run_dir": "run",
        "snapshot": "run/snapshots/4000_snapshot.pkl",
        "critic": "target",
        "eval_seeds": [10, 11],
        "metrics": {
            "configured_num_flow_steps": 8,
            "step_counts": [1, 2, 4, 8],
            "per_level_configured_q_span": [0.2, 0.3],
            "mean_source_contraction_ratio": 0.4,
            "mean_normalized_curvature_rms": 0.02,
            "mean_step_ranking_agreement": [0.8, 0.9, 0.95, 1.0],
            "mean_step_normalized_q_rmse": [0.1, 0.05, 0.02, 0.0],
        },
    }


def _summarize(payload):
    return summarize(
        payload,
        min_q_span=1e-3,
        max_source_contraction=0.95,
        min_normalized_curvature=0.01,
        max_one_step_agreement=0.98,
        min_one_step_normalized_rmse=0.02,
    )


def test_utilization_summary_passes_nontrivial_flow():
    summary = _summarize(_payload())

    assert summary["gate"] == "pass"
    assert all(summary["checks"].values())
    assert summary["diagnostic_only"]
    assert summary["selection_use_forbidden"]


def test_utilization_summary_rejects_identity_collapse():
    payload = copy.deepcopy(_payload())
    payload["metrics"].update(
        {
            "mean_source_contraction_ratio": 1.0,
            "mean_normalized_curvature_rms": 0.0,
            "mean_step_ranking_agreement": [1.0, 1.0, 1.0, 1.0],
            "mean_step_normalized_q_rmse": [0.0, 0.0, 0.0, 0.0],
        }
    )

    summary = _summarize(payload)

    assert summary["gate"] == "fail"
    assert not summary["checks"]["source_noise_contracted"]
    assert not summary["checks"]["nonlinear_or_depth_sensitive"]
