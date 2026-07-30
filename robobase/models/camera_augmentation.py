"""Shared JAX augmentations for camera-conditioned policies."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from robobase.models.encoder import _plucker_raymap_from_camera_params_jax


def raymap_from_camera_params(
    intrinsics: jnp.ndarray,
    c2ws: jnp.ndarray,
    *,
    height: int,
    width: int,
) -> jnp.ndarray:
    if intrinsics.ndim != 5 or c2ws.ndim != 5:
        raise ValueError(
            "Camera-parameter crop augmentation expects (batch, views, time, "
            f"3, 3)/(4, 4), got {intrinsics.shape}/{c2ws.shape}."
        )
    if intrinsics.shape[:3] != c2ws.shape[:3]:
        raise ValueError("Camera intrinsic and c2w dimensions must match.")
    batch_size, num_views, frame_stack = intrinsics.shape[:3]
    flat_raymap = _plucker_raymap_from_camera_params_jax(
        intrinsics.reshape((-1, 3, 3)),
        c2ws.reshape((-1, 4, 4)),
        int(height),
        int(width),
    )
    raymap = flat_raymap.reshape((batch_size, num_views, frame_stack, height, width, 6))
    raymap = jnp.transpose(raymap, (0, 1, 2, 5, 3, 4))
    return raymap.reshape((batch_size, num_views, frame_stack * 6, height, width))


def _square_crop_resize(
    images: jnp.ndarray,
    crop_sizes: jnp.ndarray,
    crop_tops: jnp.ndarray,
    crop_lefts: jnp.ndarray,
) -> jnp.ndarray:
    batch_size, num_views, channels, height, width = images.shape
    output_y = jnp.arange(height, dtype=jnp.int32)
    output_x = jnp.arange(width, dtype=jnp.int32)
    source_y = crop_tops[..., None] + (
        output_y[None, None, :] * crop_sizes[..., None] // height
    )
    source_x = crop_lefts[..., None] + (
        output_x[None, None, :] * crop_sizes[..., None] // width
    )

    def crop_one(image, y_indices, x_indices):
        image = jnp.take(image, y_indices, axis=1)
        return jnp.take(image, x_indices, axis=2)

    flat = images.astype(jnp.float32).reshape(
        (batch_size * num_views, channels, height, width)
    )
    cropped = jax.vmap(crop_one)(
        flat,
        source_y.reshape((batch_size * num_views, height)),
        source_x.reshape((batch_size * num_views, width)),
    )
    return cropped.reshape((batch_size, num_views, channels, height, width))


def campose_crop_rgb_and_raymap(
    rgb: jnp.ndarray,
    raymap: jnp.ndarray | None,
    rng_key,
    *,
    min_scale: float = 0.8,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Apply CamPose's shared per-camera nearest crop to RGB and rays."""

    if rgb.ndim != 5:
        raise ValueError(f"Expected RGB shape (B, V, C, H, W), got {rgb.shape}.")
    batch_size, num_views, _, height, width = rgb.shape
    if raymap is not None and (
        raymap.ndim != 5
        or raymap.shape[:2] != rgb.shape[:2]
        or raymap.shape[-2:] != rgb.shape[-2:]
    ):
        raise ValueError(
            "RGB and raymap crop inputs must share batch/view/spatial shapes; "
            f"got rgb={rgb.shape}, raymap={raymap.shape}."
        )
    max_side = min(int(height), int(width))
    min_side = max(1, int(max_side * float(min_scale)))
    size_key, top_key, left_key = jax.random.split(rng_key, 3)
    crop_sizes = jax.random.randint(
        size_key,
        (batch_size, num_views),
        min_side,
        max_side + 1,
    )
    top_room = int(height) - crop_sizes + 1
    left_room = int(width) - crop_sizes + 1
    crop_tops = jnp.floor(
        jax.random.uniform(top_key, (batch_size, num_views)) * top_room
    ).astype(jnp.int32)
    crop_lefts = jnp.floor(
        jax.random.uniform(left_key, (batch_size, num_views)) * left_room
    ).astype(jnp.int32)
    rgb = _square_crop_resize(rgb, crop_sizes, crop_tops, crop_lefts)
    if raymap is not None:
        raymap = _square_crop_resize(
            raymap,
            crop_sizes,
            crop_tops,
            crop_lefts,
        )
    return rgb, raymap


def augment_campose_observation(
    obs_inputs: dict,
    rng_key,
    *,
    require_raymap: bool,
) -> dict:
    """Generate rays when needed and jointly crop an observation dictionary."""

    if "rgb" not in obs_inputs:
        return obs_inputs
    rgb = obs_inputs["rgb"]
    raymap = obs_inputs.get("raymap", None)
    if raymap is None and require_raymap:
        intrinsics = obs_inputs.get("camera_intrinsic", None)
        c2ws = obs_inputs.get("camera_c2w", None)
        if intrinsics is None or c2ws is None:
            raise ValueError(
                "CamPose crop with Plucker conditioning requires an explicit "
                "raymap or camera_intrinsic/camera_c2w observations."
            )
        raymap = raymap_from_camera_params(
            intrinsics,
            c2ws,
            height=int(rgb.shape[-2]),
            width=int(rgb.shape[-1]),
        )
    rgb, raymap = campose_crop_rgb_and_raymap(rgb, raymap, rng_key)
    augmented = {**obs_inputs, "rgb": rgb}
    if raymap is not None:
        augmented["raymap"] = raymap
    return augmented


__all__ = [
    "augment_campose_observation",
    "campose_crop_rgb_and_raymap",
    "raymap_from_camera_params",
]
