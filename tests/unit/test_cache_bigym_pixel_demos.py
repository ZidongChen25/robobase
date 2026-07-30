import argparse
from types import SimpleNamespace

import numpy as np

import pytest

from scripts.cache_bigym_pixel_demos import (
    _demo_conversion_environment,
    _create_replayed_demo,
    _parse_resolution,
    _validate_amount,
)


def test_parse_resolution_converts_width_height_to_bigym_shape():
    assert _parse_resolution("320x192") == [192, 320]


def test_parse_resolution_rejects_non_positive_dimensions():
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        _parse_resolution("0x192")


def test_force_recache_rejects_zero_amount():
    with pytest.raises(ValueError, match="amount 0"):
        _validate_amount(0, force_recache=True)


def test_zero_amount_without_force_is_a_noop_request():
    _validate_amount(0, force_recache=False)


def test_pre_action_conversion_stores_reset_observation_as_frame_zero(monkeypatch):
    reset_observation = {"rgb_head": np.asarray([0], dtype=np.uint8)}
    post_action_observation = {"rgb_head": np.asarray([1], dtype=np.uint8)}

    class FakeEnv:
        def reset(self, *, seed):
            assert seed == 7
            return reset_observation, {}

        def step(self, action):
            assert action == pytest.approx(0.25)
            return post_action_observation, 0.0, False, False, {}

    source_metadata = SimpleNamespace(seed=7, uuid="episode")
    source_demo = SimpleNamespace(
        seed=7,
        metadata=source_metadata,
        timesteps=[SimpleNamespace(executed_action=0.25)],
    )
    converted_metadata = SimpleNamespace(uuid=None)
    monkeypatch.setattr(
        "scripts.cache_bigym_pixel_demos.Metadata.from_env",
        lambda env: converted_metadata,
    )

    with _demo_conversion_environment(
        observation_timing="pre_action",
        include_camera_params=False,
    ):
        converted = _create_replayed_demo(
            source_demo,
            FakeEnv(),
            observation_timing="pre_action",
            include_camera_params=False,
        )

    np.testing.assert_array_equal(
        converted.timesteps[0].observation["rgb_head"],
        reset_observation["rgb_head"],
    )
    assert not np.array_equal(
        converted.timesteps[0].observation["rgb_head"],
        post_action_observation["rgb_head"],
    )
