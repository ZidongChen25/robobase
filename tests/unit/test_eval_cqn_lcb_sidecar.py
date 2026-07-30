from __future__ import annotations

import argparse

import numpy as np
import pytest

from scripts.eval_cqn_lcb_sidecar import (
    _non_negative_finite,
    _trees_bitwise_equal,
)


def test_non_negative_threshold_parser():
    assert _non_negative_finite("0") == 0.0
    assert _non_negative_finite("1.25") == 1.25
    with pytest.raises(argparse.ArgumentTypeError):
        _non_negative_finite("-0.1")
    with pytest.raises(argparse.ArgumentTypeError):
        _non_negative_finite("nan")


def test_tree_bitwise_equality_checks_values_and_structure():
    left = {"a": np.asarray([1.0, 2.0]), "b": (np.asarray([3]),)}
    same = {"a": np.asarray([1.0, 2.0]), "b": (np.asarray([3]),)}
    changed = {"a": np.asarray([1.0, 2.1]), "b": (np.asarray([3]),)}
    restructured = {"a": np.asarray([1.0, 2.0]), "b": [np.asarray([3])]}

    assert _trees_bitwise_equal(left, same)
    assert not _trees_bitwise_equal(left, changed)
    assert not _trees_bitwise_equal(left, restructured)
