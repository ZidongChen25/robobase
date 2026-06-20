#!/usr/bin/env python3
import argparse
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from bigym.action_modes import JointPositionActionMode, PelvisDof
from bigym.bigym_env import CONTROL_FREQUENCY_MAX
from bigym.utils.observation_config import CameraConfig, ObservationConfig
from demonstrations.demo_store import DemoStore
from demonstrations.utils import Metadata

from robobase.envs.utils.bigym_utils import TASK_MAP


TASK_DEFAULTS = {
    "flip_cup": 10,
    "flip_cutlery": 25,
    "dishwasher_load_cups": 10,
    "put_cups": 20,
    "sandwich_remove": 25,
    "dishwasher_open": 20,
}


def _parse_resolution(value: str) -> list[int]:
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("resolution must be WIDTHxHEIGHT")
    return [int(parts[0]), int(parts[1])]


def _make_env(
    task_name: str,
    downsample_rate: int,
    cameras: list[str],
    resolution: list[int],
):
    task_class = TASK_MAP[task_name]
    camera_configs = [
        CameraConfig(
            name=camera_name,
            rgb=True,
            depth=False,
            resolution=resolution,
        )
        for camera_name in cameras
    ]
    action_mode = JointPositionActionMode(
        absolute=True,
        floating_base=True,
        floating_dofs=[
            PelvisDof.X,
            PelvisDof.Y,
            PelvisDof.Z,
            PelvisDof.RZ,
        ],
    )
    return task_class(
        render_mode="rgb_array",
        action_mode=action_mode,
        observation_config=ObservationConfig(
            cameras=camera_configs,
            proprioception=True,
            privileged_information=False,
        ),
        control_frequency=CONTROL_FREQUENCY_MAX // downsample_rate,
    )


def _demo_success(demo) -> bool:
    return sum(float(step.reward) for step in demo.timesteps) > 0.25


def cache_task(args: argparse.Namespace) -> dict:
    downsample_rate = int(
        args.downsample_rate
        if args.downsample_rate is not None
        else TASK_DEFAULTS[args.task]
    )
    frequency = CONTROL_FREQUENCY_MAX // downsample_rate
    env = _make_env(args.task, downsample_rate, args.cameras, args.resolution)
    try:
        metadata = Metadata.from_env(env)
        logging.info(
            "Caching task=%s class=%s frequency=%dhz cameras=%s resolution=%s",
            args.task,
            metadata.env_name,
            frequency,
            ",".join(args.cameras),
            "x".join(map(str, args.resolution)),
        )
        demos = DemoStore().get_demos(metadata, amount=args.amount, frequency=frequency)
    finally:
        env.close()

    success_lengths = [len(demo.timesteps) for demo in demos if _demo_success(demo)]
    all_lengths = [len(demo.timesteps) for demo in demos]
    summary = {
        "task": args.task,
        "frequency": frequency,
        "downsample_rate": downsample_rate,
        "demo_count": len(demos),
        "successful_count": len(success_lengths),
        "max_success_len": max(success_lengths) if success_lengths else None,
        "min_success_len": min(success_lengths) if success_lengths else None,
        "max_len": max(all_lengths) if all_lengths else None,
        "cache_root": str(DemoStore()._cache_path),
    }
    logging.info("summary=%s", json.dumps(summary, sort_keys=True))
    if args.summary_file:
        summary_path = Path(args.summary_file)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASK_DEFAULTS), required=True)
    parser.add_argument("--amount", type=int, default=-1)
    parser.add_argument("--downsample-rate", type=int, default=None)
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["head", "left_wrist", "right_wrist"],
    )
    parser.add_argument("--resolution", type=_parse_resolution, default=[256, 256])
    parser.add_argument("--summary-file", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    np.random.seed(0)
    cache_task(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
