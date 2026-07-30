#!/usr/bin/env python3
"""Launch the unmodified official A2A trainer on an exported BiGym Zarr."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys

from benchmarks.official_bigym.a2a_upstream import (
    OFFICIAL_A2A_COMMIT,
    validate_official_checkout,
)


@dataclass(frozen=True)
class TrainPreset:
    epochs: int
    checkpoint_every: int
    batch_size: int
    history_steps: int
    action_steps: int
    flow_steps: int
    seed: int


PRESETS = {
    # Paper Table 1: 100 demonstrations, 30 epochs, n=m=8, batch size 32.
    "paper": TrainPreset(30, 30, 32, 8, 8, 6, 42),
    # User-requested long run; all other A2A architecture settings stay official.
    "bigym-200": TrainPreset(200, 20, 32, 8, 8, 6, 0),
}

METHOD_FLOW_STEPS = {
    "a2a": 6,
    "fm_unet": 10,
}


def _positive(value: int, name: str) -> int:
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return value


def resolve_settings(args: argparse.Namespace) -> dict[str, int]:
    preset = PRESETS[args.preset]
    settings = {
        "epochs": preset.epochs if args.epochs is None else args.epochs,
        "checkpoint_every": (
            preset.checkpoint_every
            if args.checkpoint_every is None
            else args.checkpoint_every
        ),
        "batch_size": preset.batch_size if args.batch_size is None else args.batch_size,
        "history_steps": (
            preset.history_steps if args.history_steps is None else args.history_steps
        ),
        "action_steps": (
            preset.action_steps if args.action_steps is None else args.action_steps
        ),
        "flow_steps": (
            METHOD_FLOW_STEPS[args.method]
            if args.flow_steps is None
            else args.flow_steps
        ),
        "seed": preset.seed if args.seed is None else args.seed,
    }
    for name, value in settings.items():
        if name != "seed":
            _positive(value, name)
    # The upstream Conv1D encoder computes its flattened width with horizon//8.
    # Non-multiples of eight do not match PyTorch's SAME-like convolution output.
    for name in ("history_steps", "action_steps"):
        if settings[name] % 8:
            raise ValueError(
                f"Official A2A {name} must be a multiple of 8; got {settings[name]}."
            )
    return settings


def build_train_command(args: argparse.Namespace) -> tuple[list[str], dict[str, object]]:
    expected = None if args.allow_unpinned_upstream else OFFICIAL_A2A_COMMIT
    checkout, commit = validate_official_checkout(
        args.official_checkout, expected_commit=expected
    )
    dataset = Path(args.dataset).expanduser().resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(dataset)
    output = Path(args.output).expanduser().resolve()
    settings = resolve_settings(args)
    horizon = settings["history_steps"] + settings["action_steps"]
    train_script = checkout / "roboverse_learn/il/train.py"
    entrypoint = Path(__file__).resolve().with_name("a2a_official_entrypoint.py")
    command = [
        str(Path(args.python).expanduser().absolute()),
        str(entrypoint),
        str(train_script),
        "--config-name=default_runner.yaml",
        f"task_name={args.task_name}",
        f"dataset_config.zarr_path={dataset}",
        f"dataset_config.batch_size={settings['batch_size']}",
        f"dataset_config.val_ratio={args.val_ratio}",
        f"train_config.dataloader.batch_size={settings['batch_size']}",
        f"train_config.val_dataloader.batch_size={settings['batch_size']}",
        f"train_config.training_params.seed={settings['seed']}",
        f"train_config.training_params.num_epochs={settings['epochs']}",
        f"train_config.training_params.checkpoint_every={settings['checkpoint_every']}",
        f"train_config.training_params.device={args.device}",
        f"train_config.training_params.max_train_steps={args.max_train_steps}",
        f"shape_meta.obs.head_cam.shape=[3,{args.image_size},{args.image_size}]",
        f"shape_meta.obs.agent_pos.shape=[{args.action_dim}]",
        f"shape_meta.action.shape=[{args.action_dim}]",
        f"horizon={horizon}",
        f"n_obs_steps={settings['history_steps']}",
        f"n_action_steps={settings['action_steps']}",
        "train_enable=true",
        "eval_enable=false",
        "eval_path=null",
        "logging.mode=disabled",
        f"checkpoint.save_root_dir={output}",
        f"hydra.run.dir={output}",
    ]
    if args.method == "a2a":
        command.append(
            f"policy_config.flow_matcher.num_sampling_steps={settings['flow_steps']}"
        )
    else:
        command.append(f"policy_config.num_inference_steps={settings['flow_steps']}")
    manifest: dict[str, object] = {
        "official_commit": commit,
        "official_checkout": str(checkout),
        "dataset": str(dataset),
        "output": str(output),
        "task_name": args.task_name,
        "device": args.device,
        "image_size": args.image_size,
        "action_dim": args.action_dim,
        "max_train_steps_per_epoch": args.max_train_steps,
        "validation_ratio": args.val_ratio,
        "preset": args.preset,
        "method": args.method,
        **settings,
        "horizon": horizon,
        "command": command,
    }
    return command, manifest


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--official-checkout",
        type=Path,
        default=root / "third_party/A2A_Flow_Matching_official",
    )
    parser.add_argument("--python", default=Path(sys.prefix) / "bin/python")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="paper")
    parser.add_argument("--method", choices=tuple(METHOD_FLOW_STEPS), default="a2a")
    parser.add_argument("--task-name", default="flip_cutlery_bigym")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--action-dim", type=int, default=16)
    parser.add_argument("--max-train-steps", type=int, default=250)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--history-steps", type=int, default=None)
    parser.add_argument("--action-steps", type=int, default=None)
    parser.add_argument("--flow-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--allow-unpinned-upstream", action="store_true")
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Validate and print the resolved command without starting training.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0, 1).")
    command, manifest = build_train_command(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if args.print_command:
        return 0
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"official_{args.method}_train_manifest.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
    environment = os.environ.copy()
    environment["policy_name"] = args.method
    checkout = str(Path(args.official_checkout).expanduser().resolve())
    root = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (checkout, root, environment.get("PYTHONPATH", ""))
        if value
    )
    subprocess.run(command, cwd=checkout, env=environment, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
