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


def bc_observation_layout(observation_space: spaces.Dict) -> BCObservationLayout:
    rgb_spaces = extract_many_from_spec(observation_space, r"rgb.*", missing_ok=True)
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
        if not np.all([shape == rgb_shapes[0] for shape in rgb_shapes]):
            raise ValueError("Expected all RGB observations to have the same shape.")
        obs_shape = (int(np.prod(rgb_shapes[0][:2])), *rgb_shapes[0][2:])
        rgb_input_shape = (len(rgb_shapes), *obs_shape)

    return BCObservationLayout(
        time_dim=time_dim,
        low_dim_size=low_dim_size,
        use_pixels=bool(rgb_spaces),
        use_multicam_fusion=len(rgb_spaces) > 1,
        rgb_keys=tuple(rgb_spaces.keys()),
        rgb_input_shape=rgb_input_shape,
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

