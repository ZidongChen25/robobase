import csv
from pathlib import Path

from omegaconf import OmegaConf

from scripts.check_cqn_evor_training_gate import (
    REQUIRED_FINITE_METRICS,
    check_run,
)


def _write_run(root: Path) -> Path:
    run = root / "run"
    (run / ".hydra").mkdir(parents=True)
    (run / "snapshots").mkdir()
    OmegaConf.save(
        OmegaConf.create(
            {
                "method": {
                    "value_mode": "return_sample",
                    "evor_td_lambda": 1.0,
                    "bcfm_lambda": 0.0,
                    "dcfm_lambda": 0.0,
                    "pcbf_loss_coeff": 0.0,
                    "pcbf_lambda": 0.0,
                    "flow_distill_lambda": 0.0,
                    "endpoint_q_lambda": 0.0,
                    "source_consistency_lambda": 0.0,
                    "flow_iqn_quantile_coupling": False,
                    "confidence_weight_temp": None,
                    "mc_return_weight": 0.0,
                    "num_flow_steps": 10,
                    "num_flow_samples": 1,
                    "num_target_flow_samples": 1,
                    "num_action_flow_samples": 16,
                    "flow_source_type": "gaussian",
                    "antithetic_flow_sources": False,
                    "fixed_action_flow_sources": True,
                    "return_sample_aggregation": "entropic",
                    "return_sample_temperature": 1.0,
                    "time_embedding_type": "raw",
                    "clip_flow_trajectory": False,
                    "flow_q_action_readout": False,
                    "num_update_steps": 4,
                    "separate_bc_policy": True,
                    "distinct_policy_encoder": True,
                    "td_target_action_source": "bc_policy",
                    "td_target_policy_value_beta": None,
                    "policy_value_beta": None,
                }
            }
        ),
        run / ".hydra" / "config.yaml",
    )
    (run / "snapshots" / "1000_snapshot.pkl").touch()
    fieldnames = ["iteration", *REQUIRED_FINITE_METRICS]
    with (run / "train.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, iteration in enumerate((0, 1000)):
            row = {name: 0.1 for name in REQUIRED_FINITE_METRICS}
            row.update(
                {
                    "iteration": iteration,
                    "evor_td_loss": 0.2 + 0.1 * index,
                    "bcfm_loss": 0.0,
                    "dcfm_loss": 0.0,
                    "pcbf_loss": 0.0,
                    "flow_critic_grad_norm": 0.2,
                    "velocity_head_grad_norm": 0.1,
                    "critic_update_norm": 0.01,
                    "flow_critic_grad_nonfinite_fraction": 0.0,
                    "encoder_grad_nonfinite_fraction": 0.0,
                    "policy_demo_top1": 0.75,
                }
            )
            writer.writerow(row)
    return run


def _check(run: Path) -> dict:
    return check_run(
        run,
        required_snapshot_step=1000,
        min_log_rows=2,
        expected_flow_steps=10,
        expected_action_flow_samples=16,
        min_loss_range=1e-10,
    )


def test_evor_training_gate_accepts_isolated_active_flowtd(tmp_path):
    result = _check(_write_run(tmp_path))

    assert result["gate"] == "pass"
    assert all(result["checks"].values())


def test_evor_training_gate_rejects_auxiliary_bcfm_activity(tmp_path):
    run = _write_run(tmp_path)
    path = run / "train.csv"
    rows = list(csv.DictReader(path.open(newline="")))
    rows[-1]["bcfm_loss"] = "0.01"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    result = _check(run)

    assert result["gate"] == "fail"
    assert not result["checks"]["auxiliary_flow_objectives_exactly_zero"]


def test_evor_training_gate_rejects_constant_flowtd_loss(tmp_path):
    run = _write_run(tmp_path)
    path = run / "train.csv"
    rows = list(csv.DictReader(path.open(newline="")))
    for row in rows:
        row["evor_td_loss"] = "0.2"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    result = _check(run)

    assert result["gate"] == "fail"
    assert not result["checks"]["evor_loss_nonconstant"]


def test_evor_training_gate_rejects_missing_offline_return_endpoint(tmp_path):
    run = _write_run(tmp_path)
    path = run / "train.csv"
    rows = list(csv.DictReader(path.open(newline="")))
    for row in rows:
        row["mc_return_mean"] = "0"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    result = _check(run)

    assert result["gate"] == "fail"
    assert not result["checks"]["offline_return_endpoint_present"]
