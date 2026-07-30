#!/usr/bin/env python3
"""Launch pinned official A2A/FM-UNet RoboVerse training runs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Literal

from benchmarks.official_roboverse.preflight import run_preflight
from benchmarks.official_roboverse.protocol import (
    DEFAULT_PAPER_CHECKOUT,
    GLOBAL_EXACT_PROTOCOL_BLOCKERS,
    PAPER_ACTION_DIM,
    PAPER_ACTION_STEPS,
    PAPER_BATCH_SIZE,
    PAPER_DEMONSTRATIONS,
    PAPER_EPOCHS,
    PAPER_HORIZON,
    PAPER_IMAGE_SIZE,
    PAPER_MAX_EVAL_STEPS,
    PAPER_MAX_TRAIN_STEPS_PER_EPOCH,
    PAPER_OBSERVATION_STEPS,
    PAPER_SEED,
    PAPER_SOURCE_COMMIT,
    get_task,
)


MethodName = Literal["a2a", "a2a_current", "a2a_gaussian_latent", "fm_unet"]
ArmName = Literal["fresh30", "long200"]


@dataclass(frozen=True)
class TrainArm:
    name: ArmName
    epochs: int
    checkpoint_every: int
    saved_checkpoints: tuple[int, ...]
    comparison_checkpoints: tuple[int, ...]
    interpretation: str


TRAIN_ARMS: dict[str, TrainArm] = {
    "fresh30": TrainArm(
        name="fresh30",
        epochs=PAPER_EPOCHS,
        checkpoint_every=PAPER_EPOCHS,
        saved_checkpoints=(30,),
        comparison_checkpoints=(30,),
        interpretation=(
            "Independent paper-budget run. Its cosine LR horizon is 30 epochs."
        ),
    ),
    "long200": TrainArm(
        name="long200",
        epochs=200,
        checkpoint_every=30,
        saved_checkpoints=(30, 60, 90, 120, 150, 180, 200),
        comparison_checkpoints=(30, 200),
        interpretation=(
            "Uninterrupted 200-epoch run. E30 and E200 share one trajectory; "
            "E30 is not optimizer-schedule-equivalent to fresh30 because its cosine "
            "LR horizon is 200 epochs."
        ),
    ),
}

METHOD_FLOW_STEPS: dict[str, int] = {
    "a2a": 6,
    "a2a_current": 6,
    "a2a_gaussian_latent": 6,
    "fm_unet": 10,
}
UPSTREAM_POLICY_NAMES: dict[str, str] = {
    "a2a": "a2a",
    "a2a_current": "a2a",
    "a2a_gaussian_latent": "a2a_gaussian_latent",
    "fm_unet": "fm_unet",
}
UPSTREAM_CONFIG_NAMES: dict[str, str] = {
    "a2a": "a2a",
    "a2a_current": "a2a",
    "a2a_gaussian_latent": "a2a",
    "fm_unet": "fm_unet",
}
CURRENT_A2A_MATCHER = (
    "roboverse_learn.il.utils.flow.flow_matchers.ConditionalFlowMatcher"
)
GAUSSIAN_LATENT_POLICY = (
    "benchmarks.official_bigym.a2a_gaussian_policy."
    "GaussianLatentA2AImagePolicy"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_arm(name: str) -> TrainArm:
    try:
        return TRAIN_ARMS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown arm {name!r}; choose one of {', '.join(TRAIN_ARMS)}.") from exc


def build_train_command(
    *,
    task_key: str,
    dataset: str | Path,
    output: str | Path,
    method: MethodName,
    arm: ArmName,
    checkout: str | Path = DEFAULT_PAPER_CHECKOUT,
    python: str | Path = sys.executable,
    device: str = "cuda:0",
    expected_episodes: int = PAPER_DEMONSTRATIONS,
    simulator: str | None = None,
    skip_latent_visualization: bool = False,
    full_epochs: bool = False,
) -> tuple[list[str], dict[str, object]]:
    """Build a command without touching the filesystem; launch performs preflight."""

    task = get_task(task_key)
    if method not in METHOD_FLOW_STEPS:
        raise ValueError(f"Unknown method {method!r}.")
    if expected_episodes < 1:
        raise ValueError("expected_episodes must be positive.")
    arm_config = get_arm(arm)
    effective_simulator = simulator or task.simulator
    if effective_simulator not in ("isaacsim", "mujoco"):
        raise ValueError("simulator must be 'isaacsim' or 'mujoco'.")
    checkout = Path(checkout).expanduser().resolve()
    dataset = Path(dataset).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    entrypoint = _repo_root() / "benchmarks/official_bigym/a2a_official_entrypoint.py"
    train_script = checkout / "roboverse_learn/il/train.py"
    command = [
        # Do not resolve the venv's python symlink into the system interpreter.
        str(Path(python).expanduser().absolute()),
        str(entrypoint),
        str(train_script),
        "--config-name=default_runner.yaml",
        f"task_name={task.official_task_name}",
        f"dataset_config.zarr_path={dataset}",
        f"dataset_config.batch_size={PAPER_BATCH_SIZE}",
        "dataset_config.val_ratio=0.02",
        f"train_config.dataloader.batch_size={PAPER_BATCH_SIZE}",
        f"train_config.val_dataloader.batch_size={PAPER_BATCH_SIZE}",
        f"train_config.training_params.seed={PAPER_SEED}",
        f"train_config.training_params.num_epochs={arm_config.epochs}",
        f"train_config.training_params.checkpoint_every={arm_config.checkpoint_every}",
        f"train_config.training_params.device={device}",
        (
            "train_config.training_params.max_train_steps="
            f"{'null' if full_epochs else PAPER_MAX_TRAIN_STEPS_PER_EPOCH}"
        ),
        (
            "train_config.training_params.max_val_steps="
            f"{PAPER_MAX_TRAIN_STEPS_PER_EPOCH}"
        ),
        f"shape_meta.obs.head_cam.shape=[3,{PAPER_IMAGE_SIZE},{PAPER_IMAGE_SIZE}]",
        f"shape_meta.obs.agent_pos.shape=[{PAPER_ACTION_DIM}]",
        f"shape_meta.action.shape=[{PAPER_ACTION_DIM}]",
        f"horizon={PAPER_HORIZON}",
        f"n_obs_steps={PAPER_OBSERVATION_STEPS}",
        f"n_action_steps={PAPER_ACTION_STEPS}",
        "eval_config.policy_runner.obs.obs_type=joint_pos",
        "eval_config.policy_runner.action.action_type=joint_pos",
        "eval_config.policy_runner.action.delta=0",
        f"eval_config.eval_args.task={task.official_task_name}",
        f"eval_config.eval_args.sim={effective_simulator}",
        f"eval_config.eval_args.max_step={PAPER_MAX_EVAL_STEPS}",
        "eval_config.eval_args.num_envs=1",
        "train_enable=true",
        "eval_enable=false",
        "eval_path=null",
        "logging.mode=disabled",
        f"checkpoint.save_root_dir={output}",
        f"hydra.run.dir={output}",
    ]
    flow_steps = METHOD_FLOW_STEPS[method]
    upstream_policy_name = UPSTREAM_POLICY_NAMES[method]
    if method in ("a2a", "a2a_current", "a2a_gaussian_latent"):
        command.append(f"policy_config.flow_matcher.num_sampling_steps={flow_steps}")
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
        command.append(f"policy_config.num_inference_steps={flow_steps}")

    declared_paper_controls_match = (
        task.is_exact
        and expected_episodes == PAPER_DEMONSTRATIONS
        and effective_simulator == task.simulator
        and method not in ("a2a_current", "a2a_gaussian_latent")
        and not full_epochs
    )
    manifest: dict[str, object] = {
        "schema": "official_a2a_roboverse_train_v1",
        "source_commit": PAPER_SOURCE_COMMIT,
        "source_checkout": str(checkout),
        "task_key": task_key,
        "task": asdict(task),
        "method": method,
        "arm": asdict(arm_config),
        "dataset": str(dataset),
        "output": str(output),
        "device": device,
        "seed": PAPER_SEED,
        "batch_size": PAPER_BATCH_SIZE,
        "demonstrations_expected": expected_episodes,
        "exact_demo_budget": expected_episodes == PAPER_DEMONSTRATIONS,
        "simulator": effective_simulator,
        "simulator_matches_paper": effective_simulator == task.simulator,
        "declared_paper_controls_match": declared_paper_controls_match,
        "exact_paper_protocol": False,
        "exact_protocol_blockers": list(GLOBAL_EXACT_PROTOCOL_BLOCKERS),
        "horizon": PAPER_HORIZON,
        "observation_steps": PAPER_OBSERVATION_STEPS,
        "action_steps": PAPER_ACTION_STEPS,
        "execution_steps": PAPER_ACTION_STEPS,
        "action_dim": PAPER_ACTION_DIM,
        "image_size": PAPER_IMAGE_SIZE,
        "flow_steps": flow_steps,
        "max_train_steps_per_epoch": (
            None if full_epochs else PAPER_MAX_TRAIN_STEPS_PER_EPOCH
        ),
        "full_epochs": full_epochs,
        "lr_schedule_epoch_horizon": arm_config.epochs,
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
            "policy_name": UPSTREAM_CONFIG_NAMES[method],
        },
        "skip_latent_visualization": skip_latent_visualization,
        "latent_visualization_mode": (
            "rng_preserving_no_plot" if skip_latent_visualization else "upstream"
        ),
    }
    if skip_latent_visualization:
        manifest["environment_overrides"]["ROBOBASE_OFFICIAL_SKIP_LATENT_VIZ"] = "1"
    return command, manifest


def _subprocess_environment(
    checkout: Path,
    method: str,
    *,
    skip_latent_visualization: bool = False,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment["policy_name"] = UPSTREAM_CONFIG_NAMES[method]
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (
            str(checkout),
            str(_repo_root()),
            environment.get("PYTHONPATH", ""),
        )
        if value
    )
    if skip_latent_visualization:
        environment["ROBOBASE_OFFICIAL_SKIP_LATENT_VIZ"] = "1"
    return environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=tuple(METHOD_FLOW_STEPS), required=True)
    parser.add_argument("--arm", choices=tuple(TRAIN_ARMS), required=True)
    parser.add_argument(
        "--official-checkout", type=Path, default=Path(DEFAULT_PAPER_CHECKOUT)
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-episodes", type=int, default=PAPER_DEMONSTRATIONS)
    parser.add_argument("--simulator", choices=("isaacsim", "mujoco"))
    parser.add_argument("--allow-proxy", action="store_true")
    parser.add_argument(
        "--skip-latent-visualization",
        action="store_true",
        help=(
            "Skip only deterministic t-SNE/PNG generation while retaining the "
            "upstream diagnostic data/model calls and RNG path."
        ),
    )
    parser.add_argument(
        "--full-epochs",
        action="store_true",
        help="Consume the complete training dataloader instead of stopping at 250 batches.",
    )
    parser.add_argument("--print-command", action="store_true")
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
    command, manifest = build_train_command(
        task_key=args.task,
        dataset=args.dataset,
        output=args.output,
        method=args.method,
        arm=args.arm,
        checkout=args.official_checkout,
        python=args.python,
        device=args.device,
        expected_episodes=args.expected_episodes,
        simulator=args.simulator,
        skip_latent_visualization=args.skip_latent_visualization,
        full_epochs=args.full_epochs,
    )
    manifest["preflight"] = preflight
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if args.print_command:
        return 0

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "train_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing run manifest {manifest_path}."
        )
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
    checkout = args.official_checkout.expanduser().resolve()
    subprocess.run(
        command,
        cwd=checkout,
        env=_subprocess_environment(
            checkout,
            args.method,
            skip_latent_visualization=args.skip_latent_visualization,
        ),
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "METHOD_FLOW_STEPS",
    "GAUSSIAN_LATENT_POLICY",
    "UPSTREAM_CONFIG_NAMES",
    "UPSTREAM_POLICY_NAMES",
    "TRAIN_ARMS",
    "TrainArm",
    "build_train_command",
    "get_arm",
]
