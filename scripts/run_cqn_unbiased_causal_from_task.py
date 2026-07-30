#!/usr/bin/env python3
"""Re-run a task-selected candidate with an unbiased causal dimension audit."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpu-id", required=True, action="append", type=int)
    parser.add_argument("--eval-seed-start", required=True, type=int)
    parser.add_argument("--num-eval-seeds", type=int, default=32)
    parser.add_argument("--anchor-steps", default="30,75,120")
    parser.add_argument("--force-level", type=int, default=1)
    parser.add_argument("--max-continuation-steps", type=int, default=300)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", required=True, type=int)
    parser.add_argument("--min-informative-states", type=int, default=24)
    parser.add_argument("--min-informative-dimensions", type=int, default=8)
    parser.add_argument(
        "--min-informative-states-per-dimension",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--required-positive-training-seeds",
        type=int,
        default=2,
    )
    return parser.parse_args()


def _load(path: Path) -> tuple[Path, dict]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text())
    if payload.get("status") != "ok":
        raise ValueError(f"incomplete artifact: {resolved}")
    return resolved, payload


def _finite_nonnegative(value, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def build_command(args: argparse.Namespace) -> tuple[list[str], dict]:
    task_path, task = _load(args.task_summary)
    if task.get("gate") != "pass":
        raise ValueError("unbiased confirmation requires a task-pass artifact")
    manifest_value = task.get("manifest")
    if not isinstance(manifest_value, str) or not manifest_value:
        raise ValueError("task summary does not identify its manifest")
    manifest_path, manifest = _load(Path(manifest_value))

    selected_steps = task.get("selected_steps")
    training_seeds = manifest.get("training_seeds")
    if not isinstance(selected_steps, dict) or not isinstance(
        training_seeds, list
    ):
        raise ValueError("task artifacts do not contain selected checkpoints")
    if len(training_seeds) < 2:
        raise ValueError("at least two training seeds are required")

    readout = (
        task.get("candidate_readout")
        or manifest.get("flow_readout")
        or "auto"
    )
    if readout not in {"auto", "distill", "integrated"}:
        raise ValueError(f"unsupported task readout: {readout}")
    policy_value_beta = task.get(
        "policy_value_beta", manifest.get("policy_value_beta")
    )
    if policy_value_beta is None:
        raise ValueError(
            "a value-authenticity audit requires a task-selected value weight"
        )
    policy_value_beta = _finite_nonnegative(
        policy_value_beta, "policy_value_beta"
    )

    command = [
        sys.executable,
        str(
            Path(__file__).with_name(
                "run_cqn_flow_branch_multiseed_gate.py"
            )
        ),
    ]
    checkpoints = []
    for item in training_seeds:
        label = str(item["label"])
        run_dir = Path(item["flow_run_dir"]).expanduser().resolve()
        step = int(selected_steps[label])
        snapshot = run_dir / "snapshots" / f"{step}_snapshot.pkl"
        if not (run_dir / ".hydra" / "config.yaml").is_file():
            raise FileNotFoundError(run_dir)
        if not snapshot.is_file():
            raise FileNotFoundError(snapshot)
        command.extend(
            [
                "--checkpoint",
                f"{label}={run_dir},{snapshot}",
            ]
        )
        checkpoints.append(
            {
                "label": label,
                "run_dir": str(run_dir),
                "snapshot": str(snapshot),
                "selected_step": step,
            }
        )
    for gpu_id in args.gpu_id:
        command.extend(["--gpu-id", str(gpu_id)])

    command.extend(
        [
            "--output-dir",
            str(args.output_dir.expanduser().resolve()),
            "--eval-seed-start",
            str(args.eval_seed_start),
            "--num-eval-seeds",
            str(args.num_eval_seeds),
            "--anchor-steps",
            args.anchor_steps,
            "--force-level",
            str(args.force_level),
            "--dimension-selection",
            "round_robin",
            "--intervention-mode",
            "sibling_horizon",
            "--intervention-horizon",
            "1",
            "--max-continuation-steps",
            str(args.max_continuation_steps),
            "--flow-readout",
            str(readout),
            "--policy-value-beta",
            f"{policy_value_beta:g}",
            "--continuation-policy",
            "bc",
            "--bootstrap-replicates",
            str(args.bootstrap_replicates),
            "--bootstrap-seed",
            str(args.bootstrap_seed),
            "--min-informative-states",
            str(args.min_informative_states),
            "--min-informative-dimensions",
            str(args.min_informative_dimensions),
            "--min-informative-states-per-dimension",
            str(args.min_informative_states_per_dimension),
            "--required-positive-training-seeds",
            str(args.required_positive_training_seeds),
            "--require-anti-cheat-proxies",
        ]
    )

    num_flow_steps = task.get(
        "num_flow_steps", manifest.get("num_flow_steps")
    )
    if readout == "integrated" and num_flow_steps is not None:
        command.extend(["--num-flow-steps", str(int(num_flow_steps))])
    aggregation = task.get(
        "return_sample_aggregation",
        manifest.get("return_sample_aggregation", "config"),
    )
    if aggregation is not None:
        command.extend(["--return-sample-aggregation", str(aggregation)])
    num_samples = task.get(
        "num_action_flow_samples",
        manifest.get("num_action_flow_samples"),
    )
    if num_samples is not None:
        command.extend(
            ["--num-action-flow-samples", str(int(num_samples))]
        )
    truncate_top = task.get(
        "return_sample_truncate_top",
        manifest.get("return_sample_truncate_top"),
    )
    if truncate_top is not None:
        command.extend(
            ["--return-sample-truncate-top", str(int(truncate_top))]
        )

    preregistration = {
        "status": "preregistered",
        "selection_use_forbidden": True,
        "task_summary": str(task_path),
        "task_manifest": str(manifest_path),
        "checkpoints": checkpoints,
        "dimension_selection": "round_robin",
        "dimension_selection_depends_on": [
            "eval_seed_position",
            "anchor_position",
        ],
        "dimension_selection_forbidden_inputs": [
            "Q",
            "BC",
            "realized_return",
        ],
        "deployment_policy_value_beta": policy_value_beta,
        "continuation_policy": "bc",
        "continuation_policy_value_beta": None,
        "eval_seed_start": args.eval_seed_start,
        "num_eval_seeds": args.num_eval_seeds,
        "anchor_steps": args.anchor_steps,
        "min_informative_dimensions": args.min_informative_dimensions,
        "min_informative_states_per_dimension": (
            args.min_informative_states_per_dimension
        ),
        "command": command,
    }
    return command, preregistration


def main() -> int:
    args = parse_args()
    if len(set(args.gpu_id)) != len(args.gpu_id):
        raise ValueError("GPU workers must be unique")
    if (
        args.num_eval_seeds < 1
        or args.bootstrap_replicates < 1
        or args.min_informative_dimensions < 1
        or args.min_informative_states_per_dimension < 1
    ):
        raise ValueError("audit counts must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command, preregistration = build_command(args)
    preregistration_path = output_dir / "unbiased_preregistration.json"
    preregistration_path.write_text(
        json.dumps(preregistration, indent=2, sort_keys=True) + "\n"
    )
    subprocess.run(command, check=True)
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    preregistration["status"] = "complete"
    preregistration["summary"] = str(summary_path)
    preregistration["gate"] = summary.get("gate")
    preregistration_path.write_text(
        json.dumps(preregistration, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
