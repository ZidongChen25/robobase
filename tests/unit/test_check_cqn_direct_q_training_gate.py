import csv
from pathlib import Path

from omegaconf import OmegaConf

from scripts.check_cqn_direct_q_training_gate import (
    REQUIRED_FINITE_METRICS,
    check_run,
)


def _run(
    tmp_path: Path,
    *,
    td_target_action_source: str | None = None,
    td_target_policy_value_beta: float | None = None,
) -> Path:
    run = tmp_path / "run"
    (run / ".hydra").mkdir(parents=True)
    (run / "snapshots").mkdir()
    OmegaConf.save(
        {
            "method": {
                "direct_scalar_q": True,
                "direct_q_loss": "mse",
                "td_target_action_source": td_target_action_source,
                "td_target_policy_value_beta": td_target_policy_value_beta,
                "policy_value_beta": None,
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
            row["direct_q_grad_nonfinite_fraction"] = 0.0
            writer.writerow({"iteration": iteration, **row})
    return run


def test_training_gate_passes_finite_mse_run(tmp_path):
    payload = check_run(
        _run(tmp_path),
        required_snapshot_step=1_000,
        min_log_rows=2,
        min_max_direct_q_grad=1e-8,
    )

    assert payload["gate"] == "pass"
    assert all(payload["checks"].values())


def test_training_gate_rejects_nonfinite_gradients(tmp_path):
    run = _run(tmp_path)
    text = (run / "train.csv").read_text().replace(",0.0,", ",nan,", 1)
    (run / "train.csv").write_text(text)

    payload = check_run(
        run,
        required_snapshot_step=1_000,
        min_log_rows=2,
        min_max_direct_q_grad=1e-8,
    )

    assert payload["gate"] == "fail"
    assert not payload["checks"]["all_required_metrics_finite"]


def test_training_gate_validates_policy_value_target_and_bc_rollout(tmp_path):
    payload = check_run(
        _run(
            tmp_path,
            td_target_action_source="policy_value",
            td_target_policy_value_beta=1.0,
        ),
        required_snapshot_step=1_000,
        min_log_rows=2,
        min_max_direct_q_grad=1e-8,
        expected_td_target_action_source="policy_value",
        expected_td_target_policy_value_beta=1.0,
    )

    assert payload["gate"] == "pass"
    assert payload["checks"]["rollout_remains_exact_bc"]
