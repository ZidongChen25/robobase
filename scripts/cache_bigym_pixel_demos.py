#!/usr/bin/env python3
import argparse
import json
import logging
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from bigym.action_modes import JointPositionActionMode, PelvisDof
from bigym.bigym_env import CONTROL_FREQUENCY_MAX
from bigym.utils.observation_config import CameraConfig, ObservationConfig
from demonstrations.demo import Demo
from demonstrations.demo_converter import DemoConverter
from demonstrations.demo_store import DemoStore
from demonstrations.utils import Metadata, ObservationMode
from safetensors import safe_open

from robobase.envs.utils.bigym_utils import TASK_MAP


TASK_DEFAULTS = {
    "move_plate": 10,
    "flip_cup": 10,
    "flip_cutlery": 25,
    "dishwasher_load_cups": 10,
    "put_cups": 20,
    "sandwich_remove": 25,
    "dishwasher_open": 20,
}


TASK_ENABLE_ALL_FLOATING_DOF = {
    "move_plate": False,
}


def _parse_resolution(value: str) -> list[int]:
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("resolution must be WIDTHxHEIGHT")
    width, height = (int(part) for part in parts)
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("resolution dimensions must be positive")
    return [height, width]


def _validate_amount(amount: int, *, force_recache: bool) -> None:
    if amount < -1:
        raise ValueError("amount must be -1 (all) or a non-negative integer")
    if force_recache and amount == 0:
        raise ValueError("--amount 0 cannot be combined with --force-recache")


def _normalize_observation_timing(value: str) -> str:
    timing = str(value).strip().lower().replace("-", "_")
    aliases = {"pre": "pre_action", "post": "post_action"}
    timing = aliases.get(timing, timing)
    if timing not in {"pre_action", "post_action"}:
        raise ValueError(
            "observation timing must be either 'pre_action' or 'post_action'"
        )
    return timing


@contextmanager
def _demo_conversion_environment(
    *, observation_timing: str, include_camera_params: bool
):
    """Configure the BiGym converter without leaking process-global state."""

    timing = _normalize_observation_timing(observation_timing)
    updates = {
        "BIGYM_DEMO_RESET_ALIGNED": "1" if timing == "pre_action" else "0",
        "BIGYM_DEMO_CAMERA_PARAMS": "1" if include_camera_params else "0",
    }
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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
    if TASK_ENABLE_ALL_FLOATING_DOF.get(task_name, True):
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
    else:
        action_mode = JointPositionActionMode(
            absolute=True,
            floating_base=True,
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


def _camera_params(env, camera_name: str, image_size) -> tuple[np.ndarray, np.ndarray]:
    physics = env.mojo.physics
    try:
        camera_id = int(physics.model.name2id(camera_name, "camera"))
    except Exception:
        camera_id = -1
        for candidate in range(int(physics.model.ncam)):
            name = physics.model.id2name(candidate, "camera")
            if name == camera_name or str(name).endswith(f"/{camera_name}"):
                camera_id = candidate
                break
        if camera_id < 0:
            raise ValueError(f"Camera {camera_name!r} not found in BiGym physics.")
    height, width = (int(dim) for dim in image_size)
    fovy = float(physics.model.cam_fovy[camera_id])
    focal = height / (2.0 * np.tan(np.deg2rad(fovy) / 2.0))
    intrinsic = np.asarray(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 3] = np.asarray(physics.data.cam_xpos[camera_id], dtype=np.float32)
    c2w[:3, :3] = np.asarray(
        physics.data.cam_xmat[camera_id], dtype=np.float32
    ).reshape(3, 3)
    return intrinsic, c2w


def _with_camera_params(observation: dict, env) -> dict:
    observation = dict(observation)
    for camera in env.observation_config.cameras:
        intrinsic, c2w = _camera_params(env, camera.name, camera.resolution)
        observation[f"camera_intrinsic_{camera.name}"] = intrinsic
        observation[f"camera_c2w_{camera.name}"] = c2w
    return observation


def _create_replayed_demo(
    demo: Demo,
    env,
    *,
    observation_timing: str,
    include_camera_params: bool,
) -> Demo:
    """Replay a lightweight demo without relying on a patched BiGym submodule."""

    observation, _ = env.reset(seed=demo.seed)
    metadata = Metadata.from_env(env)
    metadata.uuid = demo.metadata.uuid
    converted = Demo(metadata)
    for timestep in demo.timesteps:
        action = timestep.executed_action
        if observation_timing == "pre_action":
            stored_observation = observation
            if include_camera_params:
                stored_observation = _with_camera_params(stored_observation, env)
            next_observation, reward, terminated, truncated, info = env.step(action)
        else:
            next_observation, reward, terminated, truncated, info = env.step(action)
            stored_observation = next_observation
            if include_camera_params:
                stored_observation = _with_camera_params(stored_observation, env)
        converted.add_timestep(
            stored_observation,
            reward,
            terminated,
            truncated,
            info,
            action,
        )
        observation = next_observation
    return converted


def _demo_success(demo) -> bool:
    return sum(float(step.reward) for step in demo.timesteps) > 0.25


def _camera_param_keys_present(path: Path, cameras: list[str]) -> bool:
    with safe_open(path, framework="np", device="cpu") as handle:
        keys = set(handle.keys())
    return all(
        f"obs_camera_intrinsic_{camera}" in keys and f"obs_camera_c2w_{camera}" in keys
        for camera in cameras
    )


def _replace_cache_dir(staging_dir: Path, cache_dir: Path) -> int:
    old_count = len(list(cache_dir.glob("*.safetensors"))) if cache_dir.exists() else 0
    backup_dir = cache_dir.with_name(
        f".{cache_dir.name}.old-{time.strftime('%Y%m%d%H%M%S')}-{os.getpid()}"
    )
    try:
        if cache_dir.exists():
            cache_dir.rename(backup_dir)
        staging_dir.rename(cache_dir)
    except Exception:
        if cache_dir.exists() and not staging_dir.exists():
            cache_dir.rename(staging_dir)
        if backup_dir.exists() and not cache_dir.exists():
            backup_dir.rename(cache_dir)
        raise
    shutil.rmtree(backup_dir, ignore_errors=True)
    return old_count


def _force_recache_to_staging(
    source_store: DemoStore,
    metadata: Metadata,
    frequency: int,
    cache_dir: Path,
    args: argparse.Namespace,
) -> list[Path]:
    light_metadata = Metadata(
        observation_mode=ObservationMode.Lightweight,
        environment_data=metadata.environment_data,
        seed=metadata.seed,
        package_versions=metadata.package_versions,
        date=metadata.date,
        uuid=metadata.uuid,
    )
    light_dir = source_store._create_path(light_metadata).parent
    light_files = sorted(light_dir.glob("*.safetensors"))
    if args.amount > 0:
        light_files = light_files[: args.amount]
    if not light_files:
        raise FileNotFoundError(
            f"No lightweight BiGym demos found at {light_dir}; cannot force-recache "
            "pixel demos without a replay source."
        )

    staging_dir = cache_dir.with_name(f".{cache_dir.name}.tmp-{os.getpid()}")
    shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=False)
    saved_paths = []
    try:
        logging.info(
            "Force-recaching %d demos from %s into staging dir %s",
            len(light_files),
            light_dir,
            staging_dir,
        )
        robot = metadata.get_robot()
        replay_env = metadata.get_env(frequency)
        try:
            for source_path in light_files:
                demo = Demo.from_safetensors(source_path)
                demo = DemoConverter.decimate(
                    demo,
                    frequency,
                    CONTROL_FREQUENCY_MAX,
                    robot=robot,
                )
                demo = _create_replayed_demo(
                    demo,
                    replay_env,
                    observation_timing=_normalize_observation_timing(
                        args.observation_timing
                    ),
                    include_camera_params=bool(args.include_camera_params),
                )
                saved_paths.append(demo.save(staging_dir / demo.metadata.filename))
        finally:
            replay_env.close()

        if not saved_paths:
            raise RuntimeError("Force recache produced no pixel demo files.")
        if args.include_camera_params and not _camera_param_keys_present(
            saved_paths[0], args.cameras
        ):
            raise RuntimeError(
                "Force recache completed but staged demos are missing camera "
                "parameter tensors."
            )
        old_count = _replace_cache_dir(staging_dir, cache_dir)
        logging.info(
            "Replaced pixel cache %s old_files=%d new_files=%d",
            cache_dir,
            old_count,
            len(saved_paths),
        )
        return sorted(cache_dir.glob("*.safetensors"))
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def cache_task(args: argparse.Namespace) -> dict:
    _validate_amount(int(args.amount), force_recache=bool(args.force_recache))
    observation_timing = _normalize_observation_timing(
        getattr(args, "observation_timing", "pre_action")
    )
    downsample_rate = int(
        args.downsample_rate
        if args.downsample_rate is not None
        else TASK_DEFAULTS[args.task]
    )
    frequency = CONTROL_FREQUENCY_MAX // downsample_rate
    env = _make_env(args.task, downsample_rate, args.cameras, args.resolution)
    store = DemoStore(cache_root=Path(args.cache_root) if args.cache_root else None)
    source_store = (
        DemoStore(cache_root=Path(args.source_cache_root))
        if args.source_cache_root
        else store
    )
    with _demo_conversion_environment(
        observation_timing=observation_timing,
        include_camera_params=bool(args.include_camera_params),
    ):
        try:
            metadata = Metadata.from_env(env)
            cache_dir = store._create_path(metadata, frequency).parent
            if args.force_recache:
                pixel_files = _force_recache_to_staging(
                    source_store,
                    metadata,
                    frequency,
                    cache_dir,
                    args,
                )
                demos = [Demo.from_safetensors(path) for path in pixel_files]
            else:
                demos = store.get_demos(
                    metadata, amount=args.amount, frequency=frequency
                )
            logging.info(
                "Caching task=%s class=%s frequency=%dhz cameras=%s resolution=%s "
                "observation_timing=%s include_camera_params=%s force_recache=%s "
                "demos=%d",
                args.task,
                metadata.env_name,
                frequency,
                ",".join(args.cameras),
                f"{args.resolution[1]}x{args.resolution[0]}",
                observation_timing,
                args.include_camera_params,
                args.force_recache,
                len(demos),
            )
        finally:
            env.close()

    success_lengths = [len(demo.timesteps) for demo in demos if _demo_success(demo)]
    all_lengths = [len(demo.timesteps) for demo in demos]
    camera_param_keys_present = False
    if demos and demos[0].timesteps:
        observation = demos[0].timesteps[0].observation
        camera_param_keys_present = all(
            f"camera_intrinsic_{camera}" in observation
            and f"camera_c2w_{camera}" in observation
            for camera in args.cameras
        )
    if args.include_camera_params and not camera_param_keys_present:
        logging.warning(
            "Cached demos do not contain camera parameter tensors. Re-run with "
            "--force-recache if this reused an older pixel cache."
        )
    summary = {
        "task": args.task,
        "frequency": frequency,
        "downsample_rate": downsample_rate,
        "demo_count": len(demos),
        "successful_count": len(success_lengths),
        "max_success_len": max(success_lengths) if success_lengths else None,
        "min_success_len": min(success_lengths) if success_lengths else None,
        "max_len": max(all_lengths) if all_lengths else None,
        "include_camera_params": bool(args.include_camera_params),
        "camera_param_keys_present": bool(camera_param_keys_present),
        "observation_timing": observation_timing,
        "reset_aligned": observation_timing == "pre_action",
        "force_recache": bool(args.force_recache),
        "cache_root": str(store._cache_path),
        "cache_dir": str(cache_dir),
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
    parser.add_argument(
        "--observation-timing",
        choices=("pre_action", "post_action"),
        default="pre_action",
        help=(
            "Observation/action alignment used while regenerating demos. "
            "pre_action stores the reset observation as frame 0 and pairs "
            "obs[i] with action[i]."
        ),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Optional target BiGym cache root; defaults to ~/.bigym.",
    )
    parser.add_argument(
        "--source-cache-root",
        type=Path,
        default=None,
        help="Optional source cache root for lightweight demos during --force-recache.",
    )
    parser.add_argument(
        "--include-camera-params",
        dest="include_camera_params",
        action="store_true",
        default=True,
        help="Store per-frame camera intrinsics and c2w poses for Plucker replay.",
    )
    parser.add_argument(
        "--no-camera-params",
        dest="include_camera_params",
        action="store_false",
        help="Do not add camera parameter tensors to regenerated pixel demos.",
    )
    parser.add_argument(
        "--force-recache",
        action="store_true",
        help="Delete existing matching pixel safetensors before caching.",
    )
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
