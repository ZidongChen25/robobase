from pathlib import Path
import pickle

import numpy as np
from omegaconf import OmegaConf

from scripts.check_cqn_direct_q_rct_training_gate import check_run


def _run(tmp_path: Path, weight: float) -> Path:
    run = tmp_path / f"run_{weight}"
    (run / ".hydra").mkdir(parents=True)
    (run / "snapshots").mkdir()
    (run / "replay").mkdir()
    (run / ".hydra" / "config.yaml").write_text(
        f"""
num_explore_steps: 0
replay:
  prioritization: false
  nstep: 1
method:
  direct_scalar_q: true
  separate_bc_policy: true
  distinct_policy_encoder: true
  policy_value_beta: null
  mc_return_weight: 0.1
  num_explore_steps: 0
  stddev_schedule: 0.0
  critic_sequence_mode: effective_k0
  td_target_action_source: replay_next
  structured_exploration_prob: 0.2
  structured_exploration_horizon: 1
  causal_rct_weight: {weight}
  causal_rct_level: 1
"""
    )
    (run / "snapshots" / "1000_snapshot.pkl").write_bytes(b"snapshot")
    causal = (
        [0.01, -0.02, 0.5, 0.2, 0.1, 0.0]
        if weight > 0
        else [0.0] * 6
    )
    (run / "train.csv").write_text(
        "iteration,critic_loss,td_critic_loss,mc_return_loss,"
        "direct_q_grad_norm,direct_q_grad_nonfinite_fraction,"
        "causal_rct_loss,causal_rct_moment_loss,"
        "causal_rct_valid_fraction,causal_rct_treated_fraction,"
        "causal_rct_tau_abs_mean,causal_rct_assignment_error_max\n"
        "1000,0.2,0.1,0.1,0.3,0,"
        + ",".join(str(value) for value in causal)
        + "\n"
    )
    starts = np.zeros((100,), np.uint8)
    starts[:20] = 1
    dimensions = np.full((100,), -1, np.int16)
    dimensions[:10] = 0
    dimensions[10:20] = 1
    delta = np.zeros((100,), np.float32)
    delta[:10] = 0.08
    delta[10:20] = -0.08
    assignment = np.full((100,), 0.8, np.float32)
    assignment[:20] = 0.05
    np.savez(
        run / "replay" / "20260724T000000_1_100_100.npz",
        action=np.zeros((100, 2), np.float32),
        demo=np.zeros((100,), np.uint8),
        structured_explore=starts,
        structured_explore_start=starts,
        structured_explore_dimension=dimensions,
        structured_explore_delta=delta,
        structured_explore_assignment_prob=assignment,
    )
    return run


def _check(run: Path, weight: float):
    return check_run(
        run,
        expected_causal_rct_weight=weight,
        expected_exploration_prob=0.2,
        expected_level=1,
        required_snapshot_step=1000,
        min_log_rows=1,
        min_online_starts=20,
        min_starts_per_dimension=10,
    )


def _enable_frozen_policy(run: Path, source: Path) -> None:
    cfg_path = run / ".hydra" / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    cfg.method.freeze_bc_policy = True
    cfg.method.bc_policy_mode = "legacy_c51"
    cfg.method.frozen_policy_snapshot = str(source.resolve())
    OmegaConf.save(cfg, cfg_path)

    policy = {
        "Dense_0": {
            "kernel": np.arange(12, dtype=np.float32).reshape(3, 4),
            "bias": np.arange(4, dtype=np.float32),
        }
    }
    encoder = {
        "Conv_0": {
            "kernel": np.arange(8, dtype=np.float32).reshape(2, 2, 1, 2),
        }
    }
    with source.open("wb") as stream:
        pickle.dump(
            {
                "agent": {
                    "params": {
                        "critic": {"unused_online": np.zeros((1,))},
                        "encoder": encoder,
                    },
                    "target_critic_params": policy,
                }
            },
            stream,
        )
    with (run / "snapshots" / "1000_snapshot.pkl").open("wb") as stream:
        pickle.dump(
            {
                "agent": {
                    "params": {
                        "critic": {"value": np.ones((1,))},
                        "policy": policy,
                        "policy_encoder": encoder,
                    }
                }
            },
            stream,
        )
    train_csv = run / "train.csv"
    lines = train_csv.read_text().splitlines()
    lines[0] += ",policy_grad_norm,policy_encoder_grad_norm"
    lines[1] += ",0,0"
    train_csv.write_text("\n".join(lines) + "\n")


def test_rct_training_gate_accepts_treatment_with_exact_propensity(tmp_path):
    payload = _check(_run(tmp_path, 0.1), 0.1)

    assert payload["gate"] == "pass"
    assert payload["replay"]["start_rate"] == 0.2
    assert payload["replay"]["starts_per_dimension"] == [10, 10]


def test_rct_training_gate_accepts_matched_noop_control(tmp_path):
    payload = _check(_run(tmp_path, 0.0), 0.0)

    assert payload["gate"] == "pass"
    assert payload["checks"]["causal_metrics_exact_noop"]


def test_rct_training_gate_accepts_clipped_zero_delta_assignment(tmp_path):
    run = _run(tmp_path, 0.1)
    archive = next((run / "replay").glob("*.npz"))
    with np.load(archive, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    arrays["structured_explore_delta"][0] = 0.0
    np.savez(archive, **arrays)

    payload = _check(run, 0.1)

    assert payload["gate"] == "pass"
    assert payload["replay"]["starts"] == 20


def test_rct_training_gate_proves_frozen_legacy_policy_bitwise(tmp_path):
    run = _run(tmp_path, 0.1)
    source = tmp_path / "clean_cqn_as.pkl"
    _enable_frozen_policy(run, source)

    payload = check_run(
        run,
        expected_causal_rct_weight=0.1,
        expected_exploration_prob=0.2,
        expected_level=1,
        required_snapshot_step=1000,
        min_log_rows=1,
        min_online_starts=20,
        min_starts_per_dimension=10,
        expected_frozen_policy_snapshot=source,
    )

    assert payload["gate"] == "pass"
    assert payload["checks"]["frozen_policy_bitwise_equal"]
    assert payload["checks"]["frozen_policy_encoder_bitwise_equal"]
    assert payload["checks"]["frozen_policy_gradient_exact_zero"]
    assert payload["checks"]["frozen_policy_encoder_gradient_exact_zero"]


def test_rct_training_gate_rejects_frozen_policy_drift(tmp_path):
    run = _run(tmp_path, 0.1)
    source = tmp_path / "clean_cqn_as.pkl"
    _enable_frozen_policy(run, source)
    trained_snapshot = run / "snapshots" / "1000_snapshot.pkl"
    with trained_snapshot.open("rb") as stream:
        trained = pickle.load(stream)
    trained["agent"]["params"]["policy"]["Dense_0"]["kernel"][0, 0] += 1
    with trained_snapshot.open("wb") as stream:
        pickle.dump(trained, stream)

    payload = check_run(
        run,
        expected_causal_rct_weight=0.1,
        expected_exploration_prob=0.2,
        expected_level=1,
        required_snapshot_step=1000,
        min_log_rows=1,
        min_online_starts=20,
        min_starts_per_dimension=10,
        expected_frozen_policy_snapshot=source,
    )

    assert payload["gate"] == "fail"
    assert not payload["checks"]["frozen_policy_bitwise_equal"]


def test_rct_training_gate_rejects_wrong_logged_propensity(tmp_path):
    run = _run(tmp_path, 0.1)
    archive = next((run / "replay").glob("*.npz"))
    with np.load(archive, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    arrays["structured_explore_assignment_prob"][:20] = 0.1
    np.savez(archive, **arrays)

    payload = _check(run, 0.1)

    assert payload["gate"] == "fail"
    assert not payload["checks"]["replay_assignment_probability_exact"]


def test_rct_training_gate_rejects_unlogged_exploration_noise(tmp_path):
    run = _run(tmp_path, 0.1)
    config_path = run / ".hydra" / "config.yaml"
    cfg = OmegaConf.load(config_path)
    cfg.method.stddev_schedule = 0.01
    OmegaConf.save(cfg, config_path)

    payload = _check(run, 0.1)

    assert payload["gate"] == "fail"
    assert not payload["checks"]["no_unlogged_gaussian_exploration"]
