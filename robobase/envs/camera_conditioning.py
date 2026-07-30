from __future__ import annotations

from functools import lru_cache

import numpy as np


def normalize_camera_conditioning(mode) -> str:
    if mode is None:
        return "none"
    if isinstance(mode, bool):
        return "plucker" if mode else "none"
    normalized = str(mode).strip().lower()
    if normalized in {"", "none", "false", "0", "off", "no"}:
        return "none"
    if normalized in {"plucker", "raymap", "camera", "camera_pose"}:
        return "plucker"
    raise ValueError(
        "camera_conditioning must be one of none/plucker; "
        f"got {mode!r}."
    )


def camera_conditioning_enabled(mode) -> bool:
    return normalize_camera_conditioning(mode) == "plucker"


def intrinsic_from_fovy(fovy: float, height: int, width: int) -> np.ndarray:
    focal = float(height) / (2.0 * np.tan(np.deg2rad(float(fovy)) / 2.0))
    return np.asarray(
        [
            [focal, 0.0, float(width) / 2.0],
            [0.0, focal, float(height) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


@lru_cache(maxsize=128)
def _camera_space_rays(
    intrinsic_bytes: bytes,
    height: int,
    width: int,
    pixel_center_offset: float,
    principal_point_offset: float,
    camera_convention: str,
) -> np.ndarray:
    intrinsic = np.frombuffer(intrinsic_bytes, dtype=np.float32).reshape(3, 3)
    v, u = np.meshgrid(
        np.arange(height, dtype=np.float32) + pixel_center_offset,
        np.arange(width, dtype=np.float32) + pixel_center_offset,
        indexing="ij",
    )
    x = (u.reshape(-1) - intrinsic[0, 2] + principal_point_offset) / intrinsic[0, 0]
    y = (v.reshape(-1) - intrinsic[1, 2] + principal_point_offset) / intrinsic[1, 1]

    convention = str(camera_convention).strip().lower()
    if convention in {"opengl", "mujoco", "dm_control"}:
        ray_cam = np.stack([x, -y, -np.ones_like(x)], axis=-1)
    elif convention in {"opencv", "cv"}:
        ray_cam = np.stack([x, y, np.ones_like(x)], axis=-1)
    else:
        raise ValueError(
            "camera_convention must be opengl/mujoco/dm_control or opencv; "
            f"got {camera_convention!r}."
        )
    ray_cam /= np.maximum(np.linalg.norm(ray_cam, axis=-1, keepdims=True), 1e-9)
    return np.ascontiguousarray(ray_cam, dtype=np.float32)


def plucker_raymap_from_c2w(
    intrinsic: np.ndarray,
    c2w: np.ndarray,
    height: int,
    width: int,
    *,
    channels_first: bool = True,
    pixel_center_offset: float = 0.5,
    principal_point_offset: float = 0.0,
    camera_convention: str = "opengl",
    channel_order: str = "direction_moment",
) -> np.ndarray:
    """Build a per-pixel Plucker ray map from a camera-to-world pose.

    The default layout follows the official controller implementation: pixel
    centers use ``+0.5`` with no extra principal-point shift, and the output
    channels are ``[direction, moment]``.

    The default ``opengl`` convention matches MuJoCo/dm_control cameras:
    +X points right, +Y points up, and cameras look along -Z. Pixel
    coordinates still use image origin at top-left with v increasing down.
    """

    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    c2w = np.asarray(c2w, dtype=np.float32)
    if intrinsic.shape != (3, 3):
        raise ValueError(f"Expected intrinsic shape (3, 3), got {intrinsic.shape}.")
    if c2w.shape != (4, 4):
        raise ValueError(f"Expected c2w shape (4, 4), got {c2w.shape}.")

    ray_cam = _camera_space_rays(
        intrinsic.tobytes(),
        int(height),
        int(width),
        float(pixel_center_offset),
        float(principal_point_offset),
        str(camera_convention).strip().lower(),
    )
    direction = ray_cam @ c2w[:3, :3].T
    camera_center = c2w[:3, 3]
    moment = np.cross(np.broadcast_to(camera_center, direction.shape), direction)
    order = str(channel_order).strip().lower()
    if order in {"direction_moment", "official", "readme", "snippet"}:
        values = (direction, moment)
    elif order in {"moment_direction", "cam_pose", "campose"}:
        values = (moment, direction)
    else:
        raise ValueError(
            "channel_order must be moment_direction or direction_moment; "
            f"got {channel_order!r}."
        )
    raymap = np.concatenate(values, axis=-1).reshape(height, width, 6)
    if channels_first:
        raymap = np.moveaxis(raymap, -1, 0)
    return np.ascontiguousarray(raymap, dtype=np.float32)
