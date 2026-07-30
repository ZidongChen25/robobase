"""BiGym cache readers shared by isolated official-code benchmarks.

This module deliberately has no JAX or PyTorch imports. Heavy optional
dependencies are imported only by the functions that need them.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Iterable, Mapping, Sequence

import numpy as np


# H1FineManipulation stores all 30 limb joint positions in `proprioception`.
# These are the ten joints controlled by JointPositionActionMode, in actuator
# order. Floating-base and gripper states are stored in dedicated observations.
H1_FINE_MANIPULATION_LIMB_QPOS_INDICES = (
    0,
    1,
    2,
    3,
    12,
    13,
    14,
    15,
    16,
    25,
)


@dataclass(frozen=True)
class BigymEpisode:
    """One pre-action-aligned BiGym demonstration."""

    source_path: Path
    rgb: Mapping[str, np.ndarray]
    state: np.ndarray
    action: np.ndarray
    success: bool
    seed: int | None
    environment_data: Mapping[str, object]

    def __post_init__(self) -> None:
        length = int(self.action.shape[0])
        if self.state.shape != self.action.shape:
            raise ValueError(
                "Official A2A requires state and action to share shape; got "
                f"{self.state.shape} and {self.action.shape}."
            )
        for camera, frames in self.rgb.items():
            if frames.ndim != 4 or frames.shape[0] != length:
                raise ValueError(
                    f"Camera {camera!r} must have shape (T,C,H,W); got "
                    f"{frames.shape} for T={length}."
                )


def discover_demo_files(cache_dir: str | Path) -> list[Path]:
    """Return deterministic safetensors episode order for a BiGym cache."""

    cache_dir = Path(cache_dir).expanduser().resolve()
    files = sorted(cache_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors demonstrations found in {cache_dir}.")
    return files


def _decode_json_metadata(metadata: Mapping[str, str] | None, key: str, default):
    if not metadata or key not in metadata:
        return default
    value = metadata[key]
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def build_actuated_qpos_state(
    proprioception: np.ndarray,
    floating_base_qpos: np.ndarray,
    gripper_qpos: np.ndarray,
    *,
    limb_qpos_indices: Sequence[int] = H1_FINE_MANIPULATION_LIMB_QPOS_INDICES,
) -> np.ndarray:
    """Reconstruct `robot.qpos_actuated` from cached H1 observations.

    BiGym stores `[all_qpos, all_qvel]` in `proprioception`. The official A2A
    implementation consumes measured joint positions as its history source, so
    this reconstruction intentionally does not use the previous command.
    """

    proprioception = np.asarray(proprioception)
    floating_base_qpos = np.asarray(floating_base_qpos)
    gripper_qpos = np.asarray(gripper_qpos)
    if proprioception.ndim != 2 or proprioception.shape[1] % 2:
        raise ValueError(
            "proprioception must have shape (T, 2*num_joints); got "
            f"{proprioception.shape}."
        )
    if floating_base_qpos.ndim != 2 or gripper_qpos.ndim != 2:
        raise ValueError("floating-base and gripper observations must be rank two.")
    if not (
        proprioception.shape[0]
        == floating_base_qpos.shape[0]
        == gripper_qpos.shape[0]
    ):
        raise ValueError("All proprioceptive observations must have the same length.")

    qpos = proprioception[:, : proprioception.shape[1] // 2]
    indices = np.asarray(tuple(limb_qpos_indices), dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("limb_qpos_indices must contain at least one index.")
    if indices.min() < 0 or indices.max() >= qpos.shape[1]:
        raise ValueError(
            f"Limb qpos indices {indices.tolist()} are invalid for {qpos.shape[1]} joints."
        )
    return np.concatenate(
        [floating_base_qpos, qpos[:, indices], gripper_qpos], axis=-1
    ).astype(np.float32, copy=False)


def load_bigym_episode(
    path: str | Path,
    *,
    cameras: Sequence[str] = ("head",),
    limb_qpos_indices: Sequence[int] = H1_FINE_MANIPULATION_LIMB_QPOS_INDICES,
) -> BigymEpisode:
    """Load one reset-aligned BiGym episode for the official A2A dataset."""

    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("Install safetensors to read the BiGym cache.") from exc

    source_path = Path(path).expanduser().resolve()
    with safe_open(str(source_path), framework="np", device="cpu") as handle:
        keys = set(handle.keys())
        required = {
            "info_demo_action",
            "obs_proprioception",
            "obs_proprioception_floating_base",
            "obs_proprioception_grippers",
        }
        required.update(f"obs_rgb_{camera}" for camera in cameras)
        missing = sorted(required - keys)
        if missing:
            raise KeyError(f"{source_path.name} is missing tensors: {missing}.")
        action = handle.get_tensor("info_demo_action").astype(np.float32)
        state = build_actuated_qpos_state(
            handle.get_tensor("obs_proprioception"),
            handle.get_tensor("obs_proprioception_floating_base"),
            handle.get_tensor("obs_proprioception_grippers"),
            limb_qpos_indices=limb_qpos_indices,
        )
        rgb = {
            camera: handle.get_tensor(f"obs_rgb_{camera}") for camera in cameras
        }
        reward = (
            handle.get_tensor("reward")
            if "reward" in keys
            else np.zeros(action.shape[0], dtype=np.float32)
        )
        metadata = handle.metadata()

    environment_data = _decode_json_metadata(metadata, "environment_data", {})
    seed_value = _decode_json_metadata(metadata, "seed", None)
    seed = None if seed_value is None else int(seed_value)
    return BigymEpisode(
        source_path=source_path,
        rgb=rgb,
        state=state,
        action=action,
        success=bool(np.asarray(reward, dtype=np.float64).sum() > 0.25),
        seed=seed,
        environment_data=environment_data,
    )


def stack_recent_history(values: Iterable[np.ndarray], length: int) -> np.ndarray:
    """Match the official runner's edge-padded observation history."""

    if length < 1:
        raise ValueError("History length must be positive.")
    recent_values = list(values)
    if not recent_values:
        raise ValueError("At least one observation is required.")
    recent = [np.asarray(value) for value in recent_values[-length:]]
    first_shape = recent[0].shape
    if any(value.shape != first_shape for value in recent):
        raise ValueError("All history observations must have the same shape.")
    if len(recent) < length:
        recent = [recent[0]] * (length - len(recent)) + recent
    return np.stack(recent, axis=0)


def _create_appendable_array(group, name: str, sample: np.ndarray, chunk_frames: int):
    shape = (0, *sample.shape[1:])
    chunks = (min(chunk_frames, max(1, sample.shape[0])), *sample.shape[1:])
    return group.create_dataset(
        name,
        shape=shape,
        chunks=chunks,
        dtype=sample.dtype,
        overwrite=False,
    )


def _append(array, values: np.ndarray) -> None:
    start = int(array.shape[0])
    array.resize((start + values.shape[0], *array.shape[1:]))
    array[start:] = values


def export_official_zarr(
    episode_files: Iterable[str | Path],
    output_path: str | Path,
    *,
    camera: str = "head",
    cameras: Sequence[str] | None = None,
    successful_only: bool = True,
    max_episodes: int | None = None,
    overwrite: bool = False,
    observation_timing: str = "pre_action",
    control_frequency_hz: int = 20,
) -> dict[str, object]:
    """Stream BiGym episodes into the Zarr schema used by official A2A."""

    if observation_timing != "pre_action":
        raise ValueError(
            "Official A2A export requires pre_action observations paired with "
            "the action at the same index."
        )
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("Install zarr to export the official A2A dataset.") from exc

    camera_names = tuple(cameras) if cameras is not None else (camera,)
    if not camera_names or len(set(camera_names)) != len(camera_names):
        raise ValueError("cameras must contain unique camera names.")
    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Refusing to replace existing dataset {output_path}; pass --overwrite."
            )
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.group(str(output_path))
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")

    arrays = None
    episode_ends: list[int] = []
    used_files: list[str] = []
    skipped_failed = 0
    total_frames = 0
    environment_data: Mapping[str, object] = {}
    for path in episode_files:
        episode = load_bigym_episode(path, cameras=camera_names)
        if successful_only and not episode.success:
            skipped_failed += 1
            continue
        if arrays is None:
            arrays = {
                "state": _create_appendable_array(
                    data_group, "state", episode.state, chunk_frames=1024
                ),
                "action": _create_appendable_array(
                    data_group, "action", episode.action, chunk_frames=1024
                ),
            }
            for camera_name in camera_names:
                key = f"{camera_name}_camera"
                arrays[key] = _create_appendable_array(
                    data_group,
                    key,
                    np.asarray(episode.rgb[camera_name]),
                    chunk_frames=64,
                )
            environment_data = episode.environment_data
        if episode.state.shape[1:] != arrays["state"].shape[1:]:
            raise ValueError("State dimensions differ between BiGym episodes.")
        if episode.action.shape[1:] != arrays["action"].shape[1:]:
            raise ValueError("Action dimensions differ between BiGym episodes.")
        for camera_name in camera_names:
            _append(
                arrays[f"{camera_name}_camera"],
                np.asarray(episode.rgb[camera_name]),
            )
        _append(arrays["state"], episode.state)
        _append(arrays["action"], episode.action)
        total_frames += episode.action.shape[0]
        episode_ends.append(total_frames)
        used_files.append(episode.source_path.name)
        if max_episodes is not None and len(episode_ends) >= max_episodes:
            break

    if arrays is None or not episode_ends:
        raise ValueError("No eligible BiGym episodes were exported.")
    meta_group.create_dataset(
        "episode_ends",
        data=np.asarray(episode_ends, dtype=np.int64),
        shape=(len(episode_ends),),
        dtype=np.int64,
    )
    manifest: dict[str, object] = {
        "schema": "official_a2a_robot_image_dataset_v1",
        "camera": camera_names[0],
        "cameras": list(camera_names),
        "observation_timing": observation_timing,
        "state": "robot.qpos_actuated reconstructed from proprioceptive feedback",
        "control_frequency_hz": int(control_frequency_hz),
        "successful_only": bool(successful_only),
        "num_episodes": len(episode_ends),
        "num_frames": total_frames,
        "action_dim": int(arrays["action"].shape[-1]),
        "skipped_failed": skipped_failed,
        "source_files": used_files,
        "environment_data": dict(environment_data),
    }
    for key, value in manifest.items():
        if key != "source_files":
            meta_group.attrs[key] = value
    with (output_path / "adapter_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
    return manifest


__all__ = [
    "BigymEpisode",
    "H1_FINE_MANIPULATION_LIMB_QPOS_INDICES",
    "build_actuated_qpos_state",
    "discover_demo_files",
    "export_official_zarr",
    "load_bigym_episode",
    "stack_recent_history",
]
