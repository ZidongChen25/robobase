"""Memory-efficient max pooling for the ResNet stems.

``flax.linen.max_pool`` lowers to ``reduce_window`` whose gradient XLA:GPU
implements as ``select_and_scatter`` -> a generic scatter. For a batch of
ImageNet-shaped stems that gradient materialises a padded copy of the input,
an ``s32[N*H*W*C, 4]`` index tensor and a scatter output at once: measured on
(192, 128, 128, 64) it is 1.56 GiB of temporaries and 15 ms per forward +
backward, i.e. 60% of the whole Diffusion/Flow-Matching update arena and the
largest single buffer in the ACT update.

``max_pool_3x3_stride2_pad1`` keeps the forward as the very same
``reduce_window`` and replaces only the backward with nine strided-window
masks combined through dilated ``pad`` ops, which XLA fuses into one loop
fusion. Ties resolve to the first window element in row-major order, exactly
like ``select_and_scatter`` with ``ge``, so parameter gradients are
bit-identical (verified in ``tests/unit/test_pixel_memory_paths.py``). Same input:
0.42 GiB and 4.9 ms.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

_WINDOW = 3
_STRIDE = 2
_PAD = 1


def _reference_max_pool(x: jnp.ndarray) -> jnp.ndarray:
    return nn.max_pool(
        x,
        window_shape=(_WINDOW, _WINDOW),
        strides=(_STRIDE, _STRIDE),
        padding=((_PAD, _PAD), (_PAD, _PAD)),
    )


@jax.custom_vjp
def max_pool_3x3_stride2_pad1(x: jnp.ndarray) -> jnp.ndarray:
    """3x3 / stride 2 / pad 1 max pool over NHWC with a scatter-free backward."""

    return _reference_max_pool(x)


def _forward(x: jnp.ndarray):
    out = _reference_max_pool(x)
    return out, (x, out)


def _backward(residuals, cotangent):
    x, out = residuals
    height, width = int(x.shape[1]), int(x.shape[2])
    out_height, out_width = int(out.shape[1]), int(out.shape[2])
    padded_height = height + 2 * _PAD
    padded_width = width + 2 * _PAD
    padded = jnp.pad(
        x,
        ((0, 0), (_PAD, _PAD), (_PAD, _PAD), (0, 0)),
        constant_values=-jnp.inf,
    )
    span_h = _STRIDE * (out_height - 1) + 1
    span_w = _STRIDE * (out_width - 1) + 1
    zero = jnp.zeros((), cotangent.dtype)
    assigned = jnp.zeros(out.shape, dtype=bool)
    contributions = {}
    for dy in range(_WINDOW):
        for dx in range(_WINDOW):
            window = padded[:, dy : dy + span_h : _STRIDE, dx : dx + span_w : _STRIDE, :]
            # First maximal element in row-major window order takes the
            # gradient, matching XLA's select_and_scatter(ge) tie-break.
            hit = jnp.logical_and(window == out, jnp.logical_not(assigned))
            assigned = jnp.logical_or(assigned, hit)
            contributions[(dy, dx)] = jnp.where(hit, cotangent, zero)
    # An input element that is the maximum of several overlapping windows
    # receives their cotangents summed in output-window (row-major) order in
    # select_and_scatter. Window (oy, ox) reaches input row 2*oy + dy - 1, so
    # ascending oy is descending dy; accumulate offsets in descending order
    # to reproduce that summation order and keep the result bit-identical.
    grad = None
    for dy in reversed(range(_WINDOW)):
        for dx in reversed(range(_WINDOW)):
            scattered = jax.lax.pad(
                contributions[(dy, dx)],
                zero,
                (
                    (0, 0, 0),
                    (dy, padded_height - dy - span_h, _STRIDE - 1),
                    (dx, padded_width - dx - span_w, _STRIDE - 1),
                    (0, 0, 0),
                ),
            )
            grad = scattered if grad is None else grad + scattered
    return (grad[:, _PAD : _PAD + height, _PAD : _PAD + width, :],)


max_pool_3x3_stride2_pad1.defvjp(_forward, _backward)


__all__ = ["max_pool_3x3_stride2_pad1"]
