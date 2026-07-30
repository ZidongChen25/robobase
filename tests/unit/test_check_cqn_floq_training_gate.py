import csv
from pathlib import Path

from omegaconf import OmegaConf

from scripts.check_cqn_floq_training_gate import (
    REQUIRED_FINITE_METRICS,
    check_run,
)


def _run(
    tmp_path: Path,
    *,
    bcfm_lambda: float,
    source_min: float | None,
    source_max: float | None,
    td_target_action_source: str = "replay_next",
    td_target_policy_value_beta: float | None = None,
) -> Path:
    run = tmp_path / "run"
    (run / ".hydra").mkdir(parents=True)
    (run / "snapshots").mkdir()
    OmegaConf.save(
        {
            "method": {
                "value_mode": "scalar",
                "num_flow_steps": 8,
                "num_flow_samples": 8,
                "num_target_flow_samples": 8,
                "num_action_flow_samples": 8,
                "flow_source_type": "uniform",
                "flow_source_min": source_min,
                "flow_source_max": source_max,
                "flow_distill_lambda": 1.0,
                "bcfm_lambda": bcfm_lambda,
                "num_update_steps": 4,
                "separate_bc_policy": True,
                "distinct_policy_encoder": True,
                "td_target_action_source": td_target_action_source,
                "td_target_policy_value_beta": td_target_policy_value_beta,
                "policy_value_beta": None,
                "critic_sequence_mode": "effective_k0",
            }
        },
        run / ".hydra" / "config.yaml",
    )
    (run / "snapshots" / "1000_snapshot.pkl").write_bytes(b"snapshot")
    fields = ["iteration", *REQUIRED_FINITE_METRICS]
    with (run / "train.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for iteration in (0, 1_000):
            row = {name: 0.1 for name in REQUIRED_FINITE_METRICS}
            row["flow_critic_grad_nonfinite_fraction"] = 0.0
            row["encoder_grad_nonfinite_fraction"] = 0.0
            writer.writerow({"iteration": iteration, **row})
    return run


def _check(
    run: Path,
    *,
    bcfm_lambda: float,
    source_min: float | None,
    source_max: float | None,
    td_target_action_source: str = "replay_next",
    td_target_policy_value_beta: float | None = None,
) -> dict:
    return check_run(
        run,
        required_snapshot_step=1_000,
        min_log_rows=2,
        min_max_flow_grad=1e-8,
        min_max_distill_grad=1e-8,
        expected_bcfm_lambda=bcfm_lambda,
        expected_source_min=source_min,
        expected_source_max=source_max,
        expected_td_target_action_source=td_target_action_source,
        expected_td_target_policy_value_beta=td_target_policy_value_beta,
    )


def test_flow_training_gate_accepts_source_corrected_arm(tmp_path):
    payload = _check(
        _run(
            tmp_path,
            bcfm_lambda=1.0,
            source_min=0.0,
            source_max=0.1,
        ),
        bcfm_lambda=1.0,
        source_min=0.0,
        source_max=0.1,
    )

    assert payload["gate"] == "pass"
    assert all(payload["checks"].values())


def test_flow_training_gate_accepts_bcfm_sum_equivalent_arm(tmp_path):
    payload = _check(
        _run(
            tmp_path,
            bcfm_lambda=8.0,
            source_min=None,
            source_max=None,
        ),
        bcfm_lambda=8.0,
        source_min=None,
        source_max=None,
    )

    assert payload["gate"] == "pass"


def test_flow_training_gate_accepts_policy_value_td_target_arm(tmp_path):
    payload = _check(
        _run(
            tmp_path,
            bcfm_lambda=1.0,
            source_min=None,
            source_max=None,
            td_target_action_source="policy_value",
            td_target_policy_value_beta=1.0,
        ),
        bcfm_lambda=1.0,
        source_min=None,
        source_max=None,
        td_target_action_source="policy_value",
        td_target_policy_value_beta=1.0,
    )

    assert payload["gate"] == "pass"
    assert payload["checks"]["rollout_remains_exact_bc"]


def test_flow_training_gate_rejects_nonfinite_gradient(tmp_path):
    run = _run(
        tmp_path,
        bcfm_lambda=1.0,
        source_min=0.0,
        source_max=0.1,
    )
    text = (run / "train.csv").read_text().replace(",0.0,0.0,", ",nan,0.0,", 1)
    (run / "train.csv").write_text(text)

    payload = _check(
        run,
        bcfm_lambda=1.0,
        source_min=0.0,
        source_max=0.1,
    )

    assert payload["gate"] == "fail"
    assert not payload["checks"]["all_required_metrics_finite"]
