import csv
from pathlib import Path

from omegaconf import OmegaConf

from scripts.check_cqn_value_flows_training_gate import (
    REQUIRED_FINITE_METRICS,
    check_run,
)


def _write_run(
    root: Path,
    *,
    confidence_temp: float | None,
    weighted: bool,
) -> Path:
    run = root / ("weighted" if weighted else "control")
    (run / ".hydra").mkdir(parents=True)
    (run / "snapshots").mkdir()
    OmegaConf.save(
        OmegaConf.create(
            {
                "method": {
                    "value_mode": "return_sample",
                    "bcfm_lambda": 1.0,
                    "dcfm_lambda": 1.0,
                    "num_flow_steps": 10,
                    "num_flow_samples": 4,
                    "num_target_flow_samples": 4,
                    "flow_source_type": "gaussian",
                    "time_embedding_type": "raw",
                    "clip_flow_trajectory": True,
                    "num_update_steps": 4,
                    "separate_bc_policy": True,
                    "distinct_policy_encoder": True,
                    "td_target_action_source": "bc_policy",
                    "policy_value_beta": None,
                    "confidence_weight_temp": confidence_temp,
                }
            }
        ),
        run / ".hydra" / "config.yaml",
    )
    (run / "snapshots" / "1000_snapshot.pkl").touch()
    fieldnames = ["iteration", *REQUIRED_FINITE_METRICS]
    rows = []
    for iteration in (0, 1000):
        row = {name: 0.1 for name in REQUIRED_FINITE_METRICS}
        row.update(
            {
                "iteration": iteration,
                "flow_critic_grad_norm": 0.2,
                "velocity_head_grad_norm": 0.1,
                "flow_critic_grad_nonfinite_fraction": 0.0,
                "encoder_grad_nonfinite_fraction": 0.0,
                "dcfm_loss": 0.01 if iteration else 0.0,
            }
        )
        if weighted:
            row.update(
                {
                    "confidence_weight_mean": 0.8,
                    "confidence_weight_min": 0.7,
                    "confidence_weight_max": 0.9,
                    "confidence_weight_std": 0.05,
                    "confidence_return_std_mean": 1.0,
                    "confidence_return_std_min": 0.5,
                    "confidence_return_std_max": 1.5,
                }
            )
        else:
            row.update(
                {
                    "confidence_weight_mean": 1.0,
                    "confidence_weight_min": 1.0,
                    "confidence_weight_max": 1.0,
                    "confidence_weight_std": 0.0,
                    "confidence_return_std_mean": 0.0,
                    "confidence_return_std_min": 0.0,
                    "confidence_return_std_max": 0.0,
                }
            )
        rows.append(row)
    with (run / "train.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return run


def _check(run: Path, expected_temp: float | None):
    return check_run(
        run,
        required_snapshot_step=1000,
        min_log_rows=2,
        expected_flow_steps=10,
        expected_flow_samples=4,
        expected_confidence_temp=expected_temp,
        min_weight_range=1e-6,
    )


def test_value_flows_gate_accepts_exact_control_and_weighted_treatment(tmp_path):
    control = _write_run(
        tmp_path,
        confidence_temp=None,
        weighted=False,
    )
    weighted = _write_run(
        tmp_path,
        confidence_temp=0.3,
        weighted=True,
    )

    assert _check(control, None)["gate"] == "pass"
    assert _check(weighted, 0.3)["gate"] == "pass"


def test_value_flows_gate_rejects_collapsed_confidence_weights(tmp_path):
    weighted = _write_run(
        tmp_path,
        confidence_temp=0.3,
        weighted=True,
    )
    path = weighted / "train.csv"
    rows = list(csv.DictReader(path.open(newline="")))
    for row in rows:
        row["confidence_weight_min"] = "0.8"
        row["confidence_weight_max"] = "0.8"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    result = _check(weighted, 0.3)

    assert result["gate"] == "fail"
    assert not result["checks"]["confidence_weights_nonconstant"]
