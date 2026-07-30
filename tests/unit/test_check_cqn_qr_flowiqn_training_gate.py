import csv
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scripts.check_cqn_qr_flowiqn_training_gate import (
    ARM_COEFFICIENTS,
    REQUIRED_FINITE_METRICS,
    check_run,
)


def _write_run(root: Path, arm: str) -> Path:
    bcfm_lambda, quantile_lambda = ARM_COEFFICIENTS[arm]
    run = root / arm
    (run / ".hydra").mkdir(parents=True)
    (run / "snapshots").mkdir()
    OmegaConf.save(
        OmegaConf.create(
            {
                "method": {
                    "value_mode": "return_sample",
                    "flow_iqn_quantile_coupling": True,
                    "bcfm_lambda": bcfm_lambda,
                    "quantile_endpoint_lambda": quantile_lambda,
                    "quantile_huber_kappa": 1.0,
                    "flow_source_type": "uniform",
                    "flow_source_min": 0.9,
                    "flow_source_max": 1.0,
                    "antithetic_flow_sources": False,
                    "fixed_action_flow_sources": True,
                    "action_flow_quantile_grid": True,
                    "num_flow_steps": 8,
                    "num_flow_samples": 8,
                    "num_target_flow_samples": 8,
                    "num_action_flow_samples": 4,
                    "num_update_steps": 4,
                    "separate_bc_policy": True,
                    "distinct_policy_encoder": True,
                    "td_target_action_source": "bc_policy",
                    "td_target_policy_value_beta": None,
                    "policy_value_beta": None,
                    "mc_return_weight": 0.0,
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
                    "bcfm_loss": 0.2 + 0.1 * index,
                    "quantile_endpoint_loss": (
                        0.0
                        if quantile_lambda == 0.0
                        else 0.3 + 0.1 * index
                    ),
                    "dcfm_loss": 0.0,
                    "evor_td_loss": 0.0,
                    "pcbf_loss": 0.0,
                    "mc_return_loss": 0.0,
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


def _check(run: Path, arm: str) -> dict:
    return check_run(
        run,
        expected_arm=arm,
        required_snapshot_step=1000,
        min_log_rows=2,
        min_loss_range=1e-10,
    )


@pytest.mark.parametrize("arm", tuple(ARM_COEFFICIENTS))
def test_qr_flowiqn_gate_accepts_registered_healthy_arms(tmp_path, arm):
    result = _check(_write_run(tmp_path, arm), arm)

    assert result["gate"] == "pass"
    assert all(result["checks"].values())


def test_qr_flowiqn_gate_rejects_unregistered_objective_drift(tmp_path):
    run = _write_run(tmp_path, "dbc_ratio")
    cfg_path = run / ".hydra" / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    cfg.method.bcfm_lambda = 0.02
    OmegaConf.save(cfg, cfg_path)

    result = _check(run, "dbc_ratio")

    assert result["gate"] == "fail"
    assert not result["checks"]["declared_registered_arm"]


def test_qr_flowiqn_gate_rejects_inactive_primary_quantile_loss(tmp_path):
    run = _write_run(tmp_path, "joint_equal")
    path = run / "train.csv"
    rows = list(csv.DictReader(path.open(newline="")))
    for row in rows:
        row["quantile_endpoint_loss"] = "0.0"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    result = _check(run, "joint_equal")

    assert result["gate"] == "fail"
    assert not result["checks"]["quantile_loss_matches_arm"]
