from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from robobase.method.utils import extract_from_spec, extract_many_from_spec


@dataclass(frozen=True)
class BCObservationLayout:
    time_dim: int
    low_dim_size: int
    use_pixels: bool
    use_multicam_fusion: bool
    rgb_keys: tuple[str, ...]
    rgb_input_shape: tuple[int, int, int, int] | None
    raymap_keys: tuple[str, ...] = ()
    raymap_input_shape: tuple[int, int, int, int] | None = None
    camera_intrinsic_keys: tuple[str, ...] = ()
    camera_c2w_keys: tuple[str, ...] = ()

    @property
    def has_camera_conditioning(self) -> bool:
        return bool(
            self.raymap_input_shape is not None
            or (self.camera_intrinsic_keys and self.camera_c2w_keys)
        )


def raymap_key_for_rgb_key(rgb_key: str, available_keys) -> str | None:
    """Return the camera-conditioning key paired with an RGB key, if present."""

    key_set = set(str(key) for key in available_keys)
    rgb_key = str(rgb_key)
    candidates = []
    if rgb_key.startswith("rgb_"):
        candidates.append(f"raymap_{rgb_key[len('rgb_') :]}")
    if rgb_key.endswith("_rgb"):
        candidates.append(f"{rgb_key[: -len('_rgb')]}_raymap")
    candidates.extend((f"{rgb_key}_raymap", f"raymap_{rgb_key}"))
    for candidate in candidates:
        if candidate in key_set:
            return candidate
    return None


def camera_param_keys_for_rgb_key(
    rgb_key: str, available_keys
) -> tuple[str, str] | None:
    """Return camera intrinsic/c2w keys paired with an RGB key, if present."""

    key_set = set(str(key) for key in available_keys)
    rgb_key = str(rgb_key)
    camera_names = []
    if rgb_key.startswith("rgb_"):
        camera_names.append(rgb_key[len("rgb_") :])
    if rgb_key.endswith("_rgb"):
        camera_names.append(rgb_key[: -len("_rgb")])
    camera_names.append(rgb_key)

    for camera_name in camera_names:
        intrinsic_candidates = (
            f"camera_intrinsic_{camera_name}",
            f"{camera_name}_camera_intrinsic",
            f"intrinsic_{camera_name}",
        )
        c2w_candidates = (
            f"camera_c2w_{camera_name}",
            f"{camera_name}_camera_c2w",
            f"c2w_{camera_name}",
        )
        for intrinsic_key in intrinsic_candidates:
            if intrinsic_key not in key_set:
                continue
            for c2w_key in c2w_candidates:
                if c2w_key in key_set:
                    return intrinsic_key, c2w_key
            ambiguous_extrinsics = (
                f"camera_extrinsic_{camera_name}",
                f"{camera_name}_camera_extrinsic",
            )
            if any(key in key_set for key in ambiguous_extrinsics):
                raise ValueError(
                    "Camera keys named 'extrinsic' are ambiguous between w2c and c2w. "
                    "Expose an explicit camera_c2w_* observation, inverting w2c "
                    "matrices in the environment adapter when necessary."
                )
    return None


def bc_observation_layout(observation_space: spaces.Dict) -> BCObservationLayout:
    rgb_spaces = extract_many_from_spec(
        observation_space, r"(^rgb|_rgb$)", missing_ok=True
    )
    time_dim = int(list(observation_space.values())[0].shape[0])

    low_dim_state_spec = extract_from_spec(
        observation_space, "low_dim_state", missing_ok=True
    )
    low_dim_size = 0
    if low_dim_state_spec is not None:
        low_dim_size = int(np.prod(low_dim_state_spec.shape))

    rgb_input_shape = None
    if rgb_spaces:
        rgb_shapes = [space.shape for space in rgb_spaces.values()]
        if not np.all([len(shape) == 4 for shape in rgb_shapes]):
            raise ValueError(
                "RGB observations must have shape (time, channels, height, width); "
                "add an explicit time axis for single-frame environments."
            )
        if not np.all([shape == rgb_shapes[0] for shape in rgb_shapes]):
            raise ValueError("Expected all RGB observations to have the same shape.")
        obs_shape = (int(np.prod(rgb_shapes[0][:2])), *rgb_shapes[0][2:])
        rgb_input_shape = (len(rgb_shapes), *obs_shape)

    raymap_keys = []
    raymap_input_shape = None
    camera_intrinsic_keys = []
    camera_c2w_keys = []
    if rgb_spaces:
        available_keys = tuple(observation_space.keys())
        for rgb_key in rgb_spaces:
            raymap_key = raymap_key_for_rgb_key(rgb_key, available_keys)
            if raymap_key is not None:
                raymap_keys.append(raymap_key)
        if raymap_keys:
            if len(raymap_keys) != len(rgb_spaces):
                raise ValueError(
                    "Camera conditioning requires one raymap observation per RGB "
                    f"view. Found rgb_keys={tuple(rgb_spaces.keys())}, "
                    f"raymap_keys={tuple(raymap_keys)}."
                )
            raymap_shapes = [
                observation_space.spaces[raymap_key].shape for raymap_key in raymap_keys
            ]
            if not np.all([len(shape) == 4 for shape in raymap_shapes]):
                raise ValueError(
                    "Raymap observations must have shape (time, 6, height, width)."
                )
            if not np.all([shape == raymap_shapes[0] for shape in raymap_shapes]):
                raise ValueError(
                    "Expected all raymap observations to have the same shape."
                )
            raymap_obs_shape = (
                int(np.prod(raymap_shapes[0][:2])),
                *raymap_shapes[0][2:],
            )
            raymap_input_shape = (len(raymap_shapes), *raymap_obs_shape)
        else:
            for rgb_key in rgb_spaces:
                camera_param_keys = camera_param_keys_for_rgb_key(
                    rgb_key, available_keys
                )
                if camera_param_keys is not None:
                    intrinsic_key, c2w_key = camera_param_keys
                    camera_intrinsic_keys.append(intrinsic_key)
                    camera_c2w_keys.append(c2w_key)
            if camera_intrinsic_keys or camera_c2w_keys:
                if len(camera_intrinsic_keys) != len(rgb_spaces) or len(
                    camera_c2w_keys
                ) != len(rgb_spaces):
                    raise ValueError(
                        "Camera conditioning requires one intrinsic and one c2w "
                        "observation per RGB view. Found "
                        f"rgb_keys={tuple(rgb_spaces.keys())}, "
                        f"intrinsic_keys={tuple(camera_intrinsic_keys)}, "
                        f"c2w_keys={tuple(camera_c2w_keys)}."
                    )
                intrinsic_shapes = [
                    observation_space.spaces[key].shape for key in camera_intrinsic_keys
                ]
                c2w_shapes = [
                    observation_space.spaces[key].shape for key in camera_c2w_keys
                ]
                if not np.all([shape[-2:] == (3, 3) for shape in intrinsic_shapes]):
                    raise ValueError(
                        "Expected camera intrinsic observations to end with shape "
                        "(3, 3)."
                    )
                if not np.all([len(shape) == 3 for shape in intrinsic_shapes]):
                    raise ValueError(
                        "Camera intrinsic observations must have shape (time, 3, 3)."
                    )
                if not np.all([shape[-2:] == (4, 4) for shape in c2w_shapes]):
                    raise ValueError(
                        "Expected camera c2w observations to end with shape (4, 4)."
                    )
                if not np.all([len(shape) == 3 for shape in c2w_shapes]):
                    raise ValueError(
                        "Camera c2w observations must have shape (time, 4, 4)."
                    )
                rgb_time = rgb_shapes[0][0]
                if not np.all(
                    [shape[0] == rgb_time for shape in intrinsic_shapes + c2w_shapes]
                ):
                    raise ValueError(
                        "Camera parameter and RGB observations must share the same "
                        "time dimension."
                    )

    return BCObservationLayout(
        time_dim=time_dim,
        low_dim_size=low_dim_size,
        use_pixels=bool(rgb_spaces),
        use_multicam_fusion=len(rgb_spaces) > 1,
        rgb_keys=tuple(rgb_spaces.keys()),
        rgb_input_shape=rgb_input_shape,
        raymap_keys=tuple(raymap_keys),
        raymap_input_shape=raymap_input_shape,
        camera_intrinsic_keys=tuple(camera_intrinsic_keys),
        camera_c2w_keys=tuple(camera_c2w_keys),
    )


def bc_actor_input_shapes(
    *,
    low_dim_size: int,
    rgb_latent_size: int,
    frame_stack_on_channel: bool,
    time_dim: int,
) -> dict[str, tuple[int, ...]]:
    obs_features_size = int(low_dim_size + rgb_latent_size)
    input_shape: tuple[int, ...] = (obs_features_size,)
    if not frame_stack_on_channel and time_dim > 0:
        input_shape = (time_dim, *input_shape)
    return {"features": input_shape}


def flatten_time_into_channel(value, *, has_view_axis: bool = False):
    if has_view_axis:
        bs, v, t, ch = value.shape[:4]
        return value.reshape(bs, v, t * ch, *value.shape[4:])
    bs, t, ch = value.shape[:3]
    return value.reshape(bs, t * ch, *value.shape[3:])
