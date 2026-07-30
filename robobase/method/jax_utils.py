"""Shared JAX/numpy utility functions for method implementations."""

from __future__ import annotations

import numpy as np


def maybe_numpy(value) -> np.ndarray:
    """Convert array-like values to numpy arrays."""
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def stack_tensor_dictionary(tensor_dict: dict, axis: int) -> np.ndarray:
    """Stack dictionary values into a single numpy array along *axis*."""
    return np.stack([maybe_numpy(value) for value in tensor_dict.values()], axis=axis)
