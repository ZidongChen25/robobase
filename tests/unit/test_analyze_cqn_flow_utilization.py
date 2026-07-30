import argparse

import numpy as np
import pytest
from omegaconf import OmegaConf

from scripts.analyze_cqn_flow_utilization import (
    _eval_seeds,
    _jsonable,
    _parse_step_counts,
    _stack_observations,
)


def test_parse_step_counts_accepts_positive_unique_counts():
    assert _parse_step_counts("1,2,4,8") == (1, 2, 4, 8)

    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        _parse_step_counts("1,0")
    with pytest.raises(argparse.ArgumentTypeError, match="duplicates"):
        _parse_step_counts("2,2")


def test_eval_seeds_and_observation_stack_are_deterministic():
    cfg = OmegaConf.create({"env": {"eval_seeds": [11, 13]}})
    assert _eval_seeds(cfg, count=3, seed_start=None) == [11, 13, 11]
    assert _eval_seeds(cfg, count=2, seed_start=20) == [20, 21]

    stacked = _stack_observations(
        [
            {"state": np.asarray([1.0, 2.0])},
            {"state": np.asarray([3.0, 4.0])},
        ]
    )
    np.testing.assert_array_equal(
        stacked["state"],
        [[1.0, 2.0], [3.0, 4.0]],
    )


def test_jsonable_converts_nested_jax_style_arrays():
    assert _jsonable(
        {"scalar": np.asarray(2), "vector": np.asarray([1.0, 3.0])}
    ) == {"scalar": 2, "vector": [1.0, 3.0]}
