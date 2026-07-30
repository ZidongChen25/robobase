#!/usr/bin/env python3
"""Evaluate pinned official A2A/FM-UNet checkpoints on 50 task states."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Literal

from benchmarks.official_bigym.a2a_upstream import file_sha256
from benchmarks.official_roboverse.heldout import (
    validate_disjoint_dataset_provenance,
)
from benchmarks.official_roboverse.preflight import run_preflight
from benchmarks.official_roboverse.random_initialization import (
    validate_random_initialization_dataset_binding,
)
from benchmarks.official_roboverse.protocol import (
    DEFAULT_PAPER_CHECKOUT,
    GLOBAL_EXACT_PROTOCOL_BLOCKERS,
    PAPER_ACTION_DIM,
    PAPER_ACTION_STEPS,
    PAPER_DEMONSTRATIONS,
    PAPER_EVAL_EPISODES,
    PAPER_HORIZON,
    PAPER_IMAGE_SIZE,
    PAPER_MAX_EVAL_STEPS,
    PAPER_OBSERVATION_STEPS,
    PAPER_SOURCE_COMMIT,
    get_task,
)
from benchmarks.official_roboverse.results import validate_evaluation_outputs
from benchmarks.official_roboverse.train import (
    CURRENT_A2A_MATCHER,
    GAUSSIAN_LATENT_POLICY,
    METHOD_FLOW_STEPS,
    UPSTREAM_POLICY_NAMES,
    _repo_root,
    _subprocess_environment,
)


MethodName = Literal["a2a", "a2a_current", "a2a_gaussian_latent", "fm_unet"]


def validate_checkpoint_ready(
    checkpoint: Path,
    *,
    stable_polls: int = 3,
    poll_interval_seconds: float = 2.0,
) -> None:
    """Reject a checkpoint that is still being written or cannot be loaded."""
    if stable_polls < 1 or poll_interval_seconds < 0:
        raise ValueError("Checkpoint stability settings must be non-negative.")
    previous: tuple[int, int] | None = None
    consecutive = 0
    while consecutive < stable_polls:
        stat = checkpoint.stat()
        current = (stat.st_size, stat.st_mtime_ns)
        if stat.st_size <= 0:
            raise ValueError(f"Checkpoint is empty: {checkpoint}")
        consecutive = consecutive + 1 if current == previous else 0
        previous = current
        if consecutive < stable_polls:
            time.sleep(poll_interval_seconds)

    # Imported only in the isolated official benchmark environment. A complete
    # load is the reliable boundary for the upstream asynchronous torch.save.
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping):
        raise ValueError(f"Checkpoint payload must be a mapping: {checkpoint}")
    required_mappings = ("cfg", "state_dicts", "pickles")
    missing = [key for key in required_mappings if key not in state]
    if missing:
        raise ValueError(
            f"Checkpoint payload is missing required fields {missing}: {checkpoint}"
        )
    invalid = [key for key in required_mappings if not isinstance(state[key], Mapping)]
    if invalid:
        raise ValueError(
            f"Checkpoint fields must be mappings, got invalid fields {invalid}: "
            f"{checkpoint}"
        )
    del state


def prepare_empty_output_directory(output: str | Path) -> Path:
    """Create an evaluation output directory or reject any existing contents."""

    output = Path(output).expanduser().resolve()
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"Evaluation output is not a directory: {output}")
        if next(output.iterdir(), None) is not None:
            raise FileExistsError(
                f"Evaluation output must be empty before launch: {output}"
            )
    else:
        output.mkdir(parents=True)
    return output


def write_manifest_atomic(path: str | Path, manifest: dict[str, object]) -> None:
    """Atomically publish a manifest only after evaluation evidence is complete."""

    path = Path(path).expanduser().resolve()
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation manifest {path}.")
    try:
        with temporary.open("x", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_eval_command(
    *,
    task_key: str,
    dataset: str | Path,
    checkpoint: str | Path,
    output: str | Path,
    method: MethodName,
    checkpoint_epoch: int,
    checkout: str | Path = DEFAULT_PAPER_CHECKOUT,
    python: str | Path = sys.executable,
    device: str = "cuda:0",
    gpu_id: int = 0,
    expected_episodes: int = PAPER_DEMONSTRATIONS,
    simulator: str | None = None,
    eval_start_index: int = 0,
    dataset_provenance: str | Path | None = None,
    random_initialization_manifest: str | Path | None = None,
    flow_steps: int | None = None,
    fm_solver: Literal["midpoint", "euler"] = "midpoint",
) -> tuple[list[str], dict[str, object]]:
    """Build the native upstream evaluator command for 50 fixed init states."""

    task = get_task(task_key)
    if method not in METHOD_FLOW_STEPS:
        raise ValueError(f"Unknown method {method!r}.")
    if method != "fm_unet" and fm_solver != "midpoint":
        raise ValueError("fm_solver only applies to fm_unet")
    if checkpoint_epoch not in (30, 200):
        raise ValueError("checkpoint_epoch must be 30 or 200 for this comparison.")
    if gpu_id < 0:
        raise ValueError("gpu_id must be non-negative.")
    if expected_episodes < 1:
        raise ValueError("expected_episodes must be positive.")
    if eval_start_index < 0:
        raise ValueError("eval_start_index must be non-negative.")
    eval_stop_index = eval_start_index + PAPER_EVAL_EPISODES
    if eval_stop_index > task.public_unique_trajectories:
        raise ValueError(
            f"Evaluation range [{eval_start_index}, {eval_stop_index}) exceeds "
            f"the {task.public_unique_trajectories} public task states."
        )
    effective_simulator = simulator or task.simulator
    if effective_simulator not in ("isaacsim", "mujoco"):
        raise ValueError("simulator must be 'isaacsim' or 'mujoco'.")
    checkout = Path(checkout).expanduser().resolve()
    dataset = Path(dataset).expanduser().resolve()
    checkpoint = Path(checkpoint).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    provenance = None
    random_initialization = None
    if random_initialization_manifest is not None:
        if task_key != "close_box":
            raise ValueError("Random-initialization evaluation currently supports close_box")
        if eval_start_index != 0:
            raise ValueError("Random-initialization trajectory indices must start at zero")
        if dataset_provenance is None:
            raise ValueError(
                "Random-initialization evaluation requires --dataset-provenance"
            )
        random_manifest, provenance = validate_random_initialization_dataset_binding(
            random_initialization_manifest,
            dataset_provenance,
            dataset=dataset,
            expected_episodes=expected_episodes,
            expected_count=PAPER_EVAL_EPISODES,
        )
        random_manifest_path = Path(random_initialization_manifest).expanduser().resolve()
        random_initialization = {
            "manifest_path": str(random_manifest_path),
            "manifest_sha256": file_sha256(random_manifest_path),
            "trajectory": random_manifest["output_trajectory"],
            "trajectory_sha256": random_manifest["output_trajectory_sha256"],
            "generation": random_manifest["generation"],
            "seed": random_manifest["seed"],
            "count": random_manifest["count"],
            "exact_training_overlap_count": random_manifest[
                "exact_training_overlap_count"
            ],
            "unique_generated_state_count": random_manifest[
                "unique_generated_state_count"
            ],
            "near_duplicate_position_threshold_m": random_manifest[
                "near_duplicate_position_threshold_m"
            ],
            "near_duplicate_rotation_threshold_degrees": random_manifest[
                "near_duplicate_rotation_threshold_degrees"
            ],
            "minimum_normalized_pose_distance": random_manifest[
                "minimum_normalized_pose_distance"
            ],
        }
    elif dataset_provenance is not None:
        provenance = validate_disjoint_dataset_provenance(
            dataset_provenance,
            dataset=dataset,
            expected_episodes=expected_episodes,
            eval_start_index=eval_start_index,
        )
    elif eval_start_index != 0:
        raise ValueError(
            "A nonzero eval_start_index requires --dataset-provenance so the "
            "held-out claim can be verified."
        )
    entrypoint = _repo_root() / "benchmarks/official_bigym/a2a_official_entrypoint.py"
    train_script = checkout / "roboverse_learn/il/train.py"
    effective_flow_steps = METHOD_FLOW_STEPS[method] if flow_steps is None else flow_steps
    if effective_flow_steps < 1:
        raise ValueError("flow_steps must be positive")
    command = [
        # Do not resolve the venv's python symlink into the system interpreter.
        str(Path(python).expanduser().absolute()),
        str(entrypoint),
        str(train_script),
        "--config-name=default_runner.yaml",
        f"task_name={task.official_task_name}",
        f"dataset_config.zarr_path={dataset}",
        f"shape_meta.obs.head_cam.shape=[3,{PAPER_IMAGE_SIZE},{PAPER_IMAGE_SIZE}]",
        f"shape_meta.obs.agent_pos.shape=[{PAPER_ACTION_DIM}]",
        f"shape_meta.action.shape=[{PAPER_ACTION_DIM}]",
        f"horizon={PAPER_HORIZON}",
        f"n_obs_steps={PAPER_OBSERVATION_STEPS}",
        f"n_action_steps={PAPER_ACTION_STEPS}",
        f"train_config.training_params.device={device}",
        "eval_config.policy_runner.obs.obs_type=joint_pos",
        "eval_config.policy_runner.action.action_type=joint_pos",
        "eval_config.policy_runner.action.delta=0",
        f"eval_config.eval_args.task={task.official_task_name}",
        f"eval_config.eval_args.sim={effective_simulator}",
        f"eval_config.eval_args.max_step={PAPER_MAX_EVAL_STEPS}",
        "eval_config.eval_args.num_envs=1",
        f"+eval_config.eval_args.gpu_id={gpu_id}",
        f"+eval_config.eval_args.task_id_range_low={eval_start_index}",
        f"+eval_config.eval_args.task_id_range_high={eval_stop_index}",
        f"+eval_config.eval_args.max_demo={PAPER_EVAL_EPISODES}",
        "train_enable=false",
        "eval_enable=true",
        f"eval_path={checkpoint}",
        "logging.mode=disabled",
        f"checkpoint.save_root_dir={output}",
        f"hydra.run.dir={output}",
    ]
    upstream_policy_name = UPSTREAM_POLICY_NAMES[method]
    if method in ("a2a", "a2a_current", "a2a_gaussian_latent"):
        command.append(
            f"policy_config.flow_matcher.num_sampling_steps={effective_flow_steps}"
        )
        if method == "a2a_current":
            command.append(
                f"policy_config.flow_matcher._target_={CURRENT_A2A_MATCHER}"
            )
        elif method == "a2a_gaussian_latent":
            command.extend(
                [
                    f"policy_name={upstream_policy_name}",
                    f"policy_config._target_={GAUSSIAN_LATENT_POLICY}",
                    f"policy_config.flow_matcher._target_={CURRENT_A2A_MATCHER}",
                ]
            )
    else:
        command.append(f"policy_config.num_inference_steps={effective_flow_steps}")
        if fm_solver == "euler":
            command.append(
                "policy_config._target_="
                "benchmarks.official_bigym.fm_unet_euler_policy."
                "EulerFlowMatchingUnetImagePolicy"
            )

    declared_paper_controls_match = (
        task.is_exact
        and expected_episodes == PAPER_DEMONSTRATIONS
        and effective_simulator == task.simulator
        and method not in ("a2a_current", "a2a_gaussian_latent")
        and eval_start_index == 0
        and provenance is None
        and random_initialization is None
    )
    if random_initialization is not None:
        evaluation_split = "heldout_random_initialization"
        evaluation_set_id = (
            f"{evaluation_split}:"
            f"{str(random_initialization['trajectory_sha256'])[:16]}"
        )
    else:
        evaluation_split = (
            "heldout_source_disjoint" if provenance is not None else "official_fixed"
        )
        evaluation_set_id = (
            f"{evaluation_split}:{eval_start_index}-{eval_stop_index - 1}"
        )
    manifest: dict[str, object] = {
        "schema": "official_a2a_roboverse_eval_v1",
        "source_commit": PAPER_SOURCE_COMMIT,
        "source_checkout": str(checkout),
        "task_key": task_key,
        "task": asdict(task),
        "method": method,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": checkpoint_epoch,
        "dataset": str(dataset),
        "demonstrations_expected": expected_episodes,
        "exact_demo_budget": expected_episodes == PAPER_DEMONSTRATIONS,
        "simulator": effective_simulator,
        "simulator_matches_paper": effective_simulator == task.simulator,
        "declared_paper_controls_match": declared_paper_controls_match,
        "exact_paper_protocol": False,
        "exact_protocol_blockers": list(GLOBAL_EXACT_PROTOCOL_BLOCKERS),
        "output": str(output),
        "device": device,
        "gpu_id": gpu_id,
        "flow_steps": effective_flow_steps,
        "solver": (
            "euler"
            if method in ("a2a", "a2a_current", "a2a_gaussian_latent")
            else fm_solver
        ),
        "model_calls_per_replan": effective_flow_steps * (
            2 if method == "fm_unet" and fm_solver == "midpoint" else 1
        ),
        "eval_episodes": PAPER_EVAL_EPISODES,
        "eval_start_index": eval_start_index,
        "eval_trajectory_indices": [eval_start_index, eval_stop_index - 1],
        "evaluation_split": evaluation_split,
        "evaluation_set_id": evaluation_set_id,
        "dataset_provenance": provenance,
        "random_initialization": random_initialization,
        "max_eval_steps": PAPER_MAX_EVAL_STEPS,
        "observation_steps": PAPER_OBSERVATION_STEPS,
        "prediction_steps": PAPER_ACTION_STEPS,
        "execution_steps": PAPER_ACTION_STEPS,
        "command": command,
        "upstream_policy_name": upstream_policy_name,
        "source_variant": (
            "initial_release_ot"
            if method == "a2a"
            else "current_main_conditional"
            if method == "a2a_current"
            else "gaussian_latent_source_ablation"
            if method == "a2a_gaussian_latent"
            else "pinned_fm_unet"
        ),
        "environment_overrides": {
            "policy_name": upstream_policy_name,
            **(
                {"ROBOBASE_OFFICIAL_EVAL_FORCE_JOINT_POS": "1"}
                if method == "a2a_gaussian_latent"
                else {}
            ),
            **(
                {
                    "ROBOBASE_OFFICIAL_EVAL_TASK": task.official_task_name,
                    "ROBOBASE_OFFICIAL_EVAL_TRAJECTORY": str(
                        random_initialization["trajectory"]
                    ),
                }
                if random_initialization is not None
                else {}
            ),
        },
    }
    return command, manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-epoch", type=int, choices=(30, 200), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=tuple(METHOD_FLOW_STEPS), required=True)
    parser.add_argument(
        "--official-checkout", type=Path, default=Path(DEFAULT_PAPER_CHECKOUT)
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--expected-episodes", type=int, default=PAPER_DEMONSTRATIONS)
    parser.add_argument("--simulator", choices=("isaacsim", "mujoco"))
    parser.add_argument("--eval-start-index", type=int, default=0)
    parser.add_argument("--dataset-provenance", type=Path)
    parser.add_argument("--random-initialization-manifest", type=Path)
    parser.add_argument("--flow-steps", type=int)
    parser.add_argument(
        "--fm-solver", choices=("midpoint", "euler"), default="midpoint"
    )
    parser.add_argument("--allow-proxy", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Validate a completed upstream output and publish its manifest without rerunning.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    preflight = run_preflight(
        task_key=args.task,
        dataset=args.dataset,
        checkout=args.official_checkout,
        expected_episodes=args.expected_episodes,
        simulator=args.simulator,
        allow_proxy=args.allow_proxy,
    )
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if checkpoint.name != f"{args.checkpoint_epoch}.ckpt":
        raise ValueError(
            f"Checkpoint filename {checkpoint.name!r} does not match declared "
            f"epoch {args.checkpoint_epoch}; expected {args.checkpoint_epoch}.ckpt."
        )
    validate_checkpoint_ready(checkpoint)
    command, manifest = build_eval_command(
        task_key=args.task,
        dataset=args.dataset,
        checkpoint=checkpoint,
        output=args.output,
        method=args.method,
        checkpoint_epoch=args.checkpoint_epoch,
        checkout=args.official_checkout,
        python=args.python,
        device=args.device,
        gpu_id=args.gpu_id,
        expected_episodes=args.expected_episodes,
        simulator=args.simulator,
        eval_start_index=args.eval_start_index,
        dataset_provenance=args.dataset_provenance,
        random_initialization_manifest=args.random_initialization_manifest,
        flow_steps=args.flow_steps,
        fm_solver=args.fm_solver,
    )
    manifest["checkpoint_sha256"] = file_sha256(checkpoint)
    manifest["preflight"] = preflight
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if args.print_command:
        return 0

    if args.finalize_existing:
        output = args.output.expanduser().resolve()
        if not output.is_dir():
            raise FileNotFoundError(output)
    else:
        output = prepare_empty_output_directory(args.output)
    manifest_path = output / "eval_manifest.json"
    checkout = args.official_checkout.expanduser().resolve()
    if not args.finalize_existing:
        environment = _subprocess_environment(checkout, args.method)
        if manifest["random_initialization"] is not None:
            environment.update(
                {
                    "ROBOBASE_OFFICIAL_EVAL_TASK": str(
                        manifest["task"]["official_task_name"]
                    ),
                    "ROBOBASE_OFFICIAL_EVAL_TRAJECTORY": str(
                        manifest["random_initialization"]["trajectory"]
                    ),
                }
            )
        if args.method == "a2a_gaussian_latent":
            environment["ROBOBASE_OFFICIAL_EVAL_FORCE_JOINT_POS"] = "1"
        subprocess.run(
            command,
            cwd=checkout,
            env=environment,
            check=True,
        )
    validate_evaluation_outputs(
        output,
        task=get_task(args.task),
        upstream_policy_name=UPSTREAM_POLICY_NAMES[args.method],
        checkpoint_epoch=args.checkpoint_epoch,
        episode_index_start=args.eval_start_index,
    )
    if args.random_initialization_manifest is not None:
        _, reverified_provenance = validate_random_initialization_dataset_binding(
            args.random_initialization_manifest,
            args.dataset_provenance,
            dataset=args.dataset,
            expected_episodes=args.expected_episodes,
            expected_count=PAPER_EVAL_EPISODES,
        )
        if manifest["dataset_provenance"] != reverified_provenance:
            raise RuntimeError(
                "Random-initialization or dataset provenance changed while "
                "evaluation was running; refusing to publish the manifest."
            )
    elif args.dataset_provenance is not None:
        reverified_provenance = validate_disjoint_dataset_provenance(
            args.dataset_provenance,
            dataset=args.dataset,
            expected_episodes=args.expected_episodes,
            eval_start_index=args.eval_start_index,
        )
        if manifest["dataset_provenance"] != reverified_provenance:
            raise RuntimeError(
                "Dataset provenance changed while evaluation was running; "
                "refusing to publish the manifest."
            )
    current_checkpoint_sha256 = file_sha256(checkpoint)
    if manifest["checkpoint_sha256"] != current_checkpoint_sha256:
        raise RuntimeError(
            "Checkpoint changed while evaluation was running; refusing to publish "
            "the manifest."
        )
    write_manifest_atomic(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_eval_command",
    "prepare_empty_output_directory",
    "validate_disjoint_dataset_provenance",
    "validate_checkpoint_ready",
    "write_manifest_atomic",
]
