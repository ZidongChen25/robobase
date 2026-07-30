import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from scripts import run_cqn_qr_flowiqn_family_gate as family


def _write_flow_run(root: Path, label: str, quantile_lambda: float) -> Path:
    run = root / label
    (run / ".hydra").mkdir(parents=True)
    (run / "snapshots").mkdir()
    OmegaConf.save(
        OmegaConf.create(
            {
                "method": {
                    "value_mode": "return_sample",
                    "flow_iqn_quantile_coupling": True,
                    "quantile_endpoint_lambda": quantile_lambda,
                    "separate_bc_policy": True,
                    "distinct_policy_encoder": True,
                    "td_target_action_source": "bc_policy",
                    "policy_value_beta": None,
                }
            }
        ),
        run / ".hydra" / "config.yaml",
    )
    for step in (1000, 2000):
        (run / "snapshots" / f"{step}_snapshot.pkl").touch()
    return run


def _write_clean_run(root: Path) -> tuple[Path, Path]:
    run = root / "clean"
    (run / ".hydra").mkdir(parents=True)
    OmegaConf.save(
        OmegaConf.create({"method": {"name": "cqn_as"}}),
        run / ".hydra" / "config.yaml",
    )
    snapshot = run / "snapshot.pkl"
    snapshot.touch()
    return run, snapshot


def _args(tmp_path: Path) -> argparse.Namespace:
    anchor = _write_flow_run(tmp_path, "anchor", 0.0)
    joint = _write_flow_run(tmp_path, "joint", 1.0)
    ratio = _write_flow_run(tmp_path, "ratio", 1.0)
    clean, clean_snapshot = _write_clean_run(tmp_path)
    return argparse.Namespace(
        anchor=("anchor_only", anchor),
        treatment=[("joint_equal", joint), ("dbc_ratio", ratio)],
        clean_run_dir=clean,
        clean_snapshot=clean_snapshot,
        gpu_id=[1, 5],
        output_dir=tmp_path / "output",
        work_root=tmp_path / "work",
        checkpoint_step=[1000, 2000],
        screen_top_k=1,
        beta=[0.3, 1.0],
        screen_beta=1.0,
        num_flow_steps=8,
        num_action_flow_samples=8,
        screen_episodes=4,
        screen_seed_start=210000,
        validation_episodes=4,
        validation_seed_start=211000,
        confirmation_episodes=8,
        confirmation_seed_start=212000,
        bootstrap_replicates=2000,
        bootstrap_seed=212200,
        clean_min_ci_lower=-0.05,
    )


def test_family_eval_command_freezes_integrated_mean_readout(tmp_path):
    arm = family.Arm("joint", tmp_path / "joint", "treatment")
    candidate = family.build_eval_command(
        family.EvalJob("joint", 3000, 0.3),
        arms={"joint": arm},
        clean_run_dir=tmp_path / "clean",
        clean_snapshot=tmp_path / "clean.pkl",
        output=tmp_path / "out.json",
        work_dir=tmp_path / "work",
        gpu_id=5,
        episodes=50,
        seed_start=211000,
        num_flow_steps=8,
        num_action_flow_samples=8,
    )
    clean = family.build_eval_command(
        family.EvalJob("clean", None, None),
        arms={"joint": arm},
        clean_run_dir=tmp_path / "clean",
        clean_snapshot=tmp_path / "clean.pkl",
        output=tmp_path / "clean.json",
        work_dir=tmp_path / "clean-work",
        gpu_id=1,
        episodes=50,
        seed_start=211000,
        num_flow_steps=8,
        num_action_flow_samples=8,
    )

    assert candidate[candidate.index("--policy-value-beta") + 1] == "0.3"
    assert candidate[candidate.index("--flow-readout") + 1] == "integrated"
    assert candidate[candidate.index("--num-flow-steps") + 1] == "8"
    assert (
        candidate[candidate.index("--num-action-flow-samples") + 1] == "8"
    )
    assert (
        candidate[candidate.index("--return-sample-aggregation") + 1]
        == "mean"
    )
    assert clean[clean.index("--policy-value-beta") + 1] == "bc"
    assert clean[clean.index("--flow-readout") + 1] == "auto"
    assert "--num-flow-steps" not in clean


def test_family_selection_ties_are_preregistered():
    beta, step, success = family.select_arm_winner(
        {
            0.3: {1000: 0.7, 2000: 0.7},
            1.0: {1000: 0.7, 2000: 0.7},
        }
    )
    winner = family.select_treatment(
        ["joint_equal", "dbc_ratio"],
        {
            "joint_equal": {"validation_success": 0.8},
            "dbc_ratio": {"validation_success": 0.8},
        },
    )

    assert (beta, step, success) == (1.0, 1000, 0.7)
    assert winner == "joint_equal"


def test_family_gate_selects_on_validation_then_confirms_once(
    tmp_path,
    monkeypatch,
):
    args = _args(tmp_path)

    def fake_run_jobs(*, jobs, split, args, arms, episodes, seed_start):
        del arms
        for job in jobs:
            path = family._result_path(args.output_dir, split, job)
            path.parent.mkdir(parents=True, exist_ok=True)
            if split == "screen":
                rate = 0.75 if job.step == 2000 else 0.5
            elif split == "validation":
                by_label = {
                    "anchor_only": 0.5,
                    "joint_equal": 0.75,
                    "dbc_ratio": 1.0,
                }
                rate = by_label[job.label]
            elif job.label == "clean":
                rate = 0.5
            elif job.label == "anchor_only":
                rate = 0.5
            else:
                rate = 1.0
            successes = [
                1.0 if index < round(rate * episodes) else 0.0
                for index in range(episodes)
            ]
            payload = {
                "status": "ok",
                "episode_success": sum(successes) / len(successes),
                "episode_results": [
                    {
                        "seed": seed_start + index,
                        "episode_success": value,
                    }
                    for index, value in enumerate(successes)
                ],
            }
            path.write_text(json.dumps(payload) + "\n")

    monkeypatch.setattr(family, "_run_jobs", fake_run_jobs)

    result = family.run_gate(args)

    assert result["selection"]["selected_treatment"] == "dbc_ratio"
    assert result["confirmation"]["treatment_vs_anchor"]["gate"] == "pass"
    assert result["gate"] == "pass"
    assert result["route_b_claim_forbidden"]
