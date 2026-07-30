import argparse
import json
from pathlib import Path

from scripts.run_cqn_unbiased_causal_from_task import build_command


def test_build_command_freezes_task_selected_checkpoints_and_unbiased_gate(
    tmp_path,
):
    training_seeds = []
    selected_steps = {}
    for index in (1, 2, 3):
        label = f"seed{index}"
        run_dir = tmp_path / label
        (run_dir / ".hydra").mkdir(parents=True)
        (run_dir / ".hydra" / "config.yaml").write_text("method: {}\n")
        (run_dir / "snapshots").mkdir()
        step = index * 1000
        (run_dir / "snapshots" / f"{step}_snapshot.pkl").write_bytes(b"x")
        selected_steps[label] = step
        training_seeds.append(
            {
                "label": label,
                "flow_run_dir": str(run_dir),
                "clean_run_dir": str(tmp_path / "unused"),
                "clean_snapshot": str(tmp_path / "unused.pkl"),
            }
        )

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "training_seeds": training_seeds,
                "flow_readout": "integrated",
                "num_flow_steps": 10,
                "policy_value_beta": 0.3,
                "return_sample_aggregation": "entropic",
                "num_action_flow_samples": 16,
                "return_sample_truncate_top": None,
            }
        )
    )
    task_path = tmp_path / "summary.json"
    task_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "gate": "pass",
                "manifest": str(manifest_path),
                "selected_steps": selected_steps,
                "candidate_readout": "integrated",
                "num_flow_steps": 10,
                "policy_value_beta": 0.3,
                "return_sample_aggregation": "entropic",
                "num_action_flow_samples": 16,
                "return_sample_truncate_top": None,
            }
        )
    )
    args = argparse.Namespace(
        task_summary=task_path,
        output_dir=tmp_path / "output",
        gpu_id=[1, 5],
        eval_seed_start=209_000,
        num_eval_seeds=32,
        anchor_steps="30,75,120",
        force_level=1,
        max_continuation_steps=300,
        bootstrap_replicates=20_000,
        bootstrap_seed=209_100,
        min_informative_states=24,
        min_informative_dimensions=8,
        min_informative_states_per_dimension=2,
        required_positive_training_seeds=2,
    )

    command, preregistration = build_command(args)

    assert command.count("--checkpoint") == 3
    assert command[command.index("--dimension-selection") + 1] == (
        "round_robin"
    )
    assert command[command.index("--return-sample-aggregation") + 1] == (
        "entropic"
    )
    assert command[command.index("--num-flow-steps") + 1] == "10"
    assert command[command.index("--policy-value-beta") + 1] == "0.3"
    assert command[command.index("--continuation-policy") + 1] == "bc"
    assert preregistration["selection_use_forbidden"]
    assert preregistration["deployment_policy_value_beta"] == 0.3
    assert preregistration["continuation_policy"] == "bc"
    assert preregistration["continuation_policy_value_beta"] is None
    assert preregistration["dimension_selection_forbidden_inputs"] == [
        "Q",
        "BC",
        "realized_return",
    ]
    assert {
        Path(item["snapshot"]).name
        for item in preregistration["checkpoints"]
    } == {
        "1000_snapshot.pkl",
        "2000_snapshot.pkl",
        "3000_snapshot.pkl",
    }
