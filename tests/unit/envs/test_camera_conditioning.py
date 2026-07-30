import numpy as np

from robobase.envs.camera_conditioning import plucker_raymap_from_c2w
from robobase.models.encoder import _plucker_raymap_from_camera_params_jax


def test_plucker_raymap_defaults_match_official_channel_order():
    intrinsic = np.asarray(
        [[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 3] = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)

    raymap = plucker_raymap_from_c2w(
        intrinsic,
        c2w,
        height=1,
        width=1,
        channels_first=True,
        camera_convention="opengl",
    )

    np.testing.assert_allclose(raymap[:3, 0, 0], [0.0, 0.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(raymap[3:, 0, 0], [-2.0, 1.0, 0.0], atol=1e-6)


def test_plucker_raymap_defaults_match_official_pixel_offset():
    intrinsic = np.eye(3, dtype=np.float32)
    c2w = np.eye(4, dtype=np.float32)

    raymap = plucker_raymap_from_c2w(
        intrinsic,
        c2w,
        height=1,
        width=1,
        channels_first=False,
        camera_convention="opengl",
    )

    expected_dir = np.asarray([0.5, -0.5, -1.0], dtype=np.float32)
    expected_dir = expected_dir / np.linalg.norm(expected_dir)
    np.testing.assert_allclose(raymap[0, 0, :3], expected_dir, atol=1e-6)


def test_plucker_raymap_supports_moment_first_layout():
    intrinsic = np.asarray(
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    c2w = np.eye(4, dtype=np.float32)

    raymap = plucker_raymap_from_c2w(
        intrinsic,
        c2w,
        height=1,
        width=1,
        channels_first=True,
        principal_point_offset=0.0,
        camera_convention="opengl",
        channel_order="moment_direction",
    )

    np.testing.assert_allclose(raymap[:3, 0, 0], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(raymap[3:, 0, 0], [0.0, 0.0, -1.0], atol=1e-6)


def test_plucker_raymap_invariants_hold_for_translated_camera():
    intrinsic = np.asarray(
        [[4.0, 0.0, 2.5], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 3] = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)

    raymap = plucker_raymap_from_c2w(intrinsic, c2w, height=4, width=5)
    direction = np.moveaxis(raymap[:3], 0, -1).reshape(-1, 3)
    moment = np.moveaxis(raymap[3:], 0, -1).reshape(-1, 3)

    np.testing.assert_allclose(np.linalg.norm(direction, axis=-1), 1.0, atol=1e-6)
    np.testing.assert_allclose(
        np.sum(direction * moment, axis=-1),
        np.zeros(direction.shape[0], dtype=np.float32),
        atol=1e-6,
    )


def test_jax_camera_param_raymap_matches_numpy_helper():
    intrinsic = np.asarray(
        [[4.0, 0.0, 2.5], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 3] = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)

    expected = plucker_raymap_from_c2w(
        intrinsic,
        c2w,
        height=4,
        width=5,
        channels_first=False,
    )
    actual = np.asarray(
        _plucker_raymap_from_camera_params_jax(
            intrinsic[None],
            c2w[None],
            height=4,
            width=5,
        )[0]
    )

    np.testing.assert_allclose(actual, expected, atol=1e-6)
