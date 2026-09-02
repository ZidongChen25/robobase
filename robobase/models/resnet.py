"""Pure-Flax ResNet feature backbones with jax-resnet-compatible variables."""

from __future__ import annotations

from collections.abc import Callable

import flax.linen as nn

from robobase.models.pooling import max_pool_3x3_stride2_pad1


class _ConvBlock(nn.Module):
    features: int
    kernel_size: tuple[int, int] = (3, 3)
    strides: tuple[int, int] = (1, 1)
    padding: object = ((0, 0), (0, 0))
    activate: bool = True
    zero_scale: bool = False
    # Convolution / stored-activation dtype. ``None`` keeps float32. The
    # normalisation itself always promotes to float32 (its parameters and
    # statistics are float32), so only the conv inputs/outputs are narrowed.
    dtype: object = None

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            self.features,
            self.kernel_size,
            self.strides,
            padding=self.padding,
            use_bias=False,
            kernel_init=nn.initializers.kaiming_normal(),
            dtype=self.dtype,
            name="Conv_0",
        )(x)
        x = nn.BatchNorm(
            momentum=0.9,
            use_running_average=not self.is_mutable_collection("batch_stats"),
            scale_init=(
                nn.initializers.zeros if self.zero_scale else nn.initializers.ones
            ),
            name="BatchNorm_0",
        )(x)
        x = nn.relu(x) if self.activate else x
        return x if self.dtype is None else x.astype(self.dtype)


class _ResNetStem(nn.Module):
    dtype: object = None

    @nn.compact
    def __call__(self, x):
        if self.dtype is not None:
            x = x.astype(self.dtype)
        return _ConvBlock(
            64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding=((3, 3), (3, 3)),
            dtype=self.dtype,
            name="ConvBlock_0",
        )(x)


class _ResNetSkipConnection(nn.Module):
    strides: tuple[int, int]
    dtype: object = None

    @nn.compact
    def __call__(self, x, output_shape):
        if x.shape != output_shape:
            x = _ConvBlock(
                int(output_shape[-1]),
                kernel_size=(1, 1),
                strides=self.strides,
                activate=False,
                dtype=self.dtype,
                name="ConvBlock_0",
            )(x)
        return x


class _ResNetBlock(nn.Module):
    features: int
    strides: tuple[int, int] = (1, 1)
    dtype: object = None

    @nn.compact
    def __call__(self, x):
        residual = _ResNetSkipConnection(
            self.strides,
            dtype=self.dtype,
            name="ResNetSkipConnection_0",
        )(x, self._output_shape(x))
        y = _ConvBlock(
            self.features,
            strides=self.strides,
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
            name="ConvBlock_0",
        )(x)
        y = _ConvBlock(
            self.features,
            padding=((1, 1), (1, 1)),
            activate=False,
            zero_scale=True,
            dtype=self.dtype,
            name="ConvBlock_1",
        )(y)
        return nn.relu(y + residual)

    def _output_shape(self, x) -> tuple[int, ...]:
        height = (int(x.shape[1]) + self.strides[0] - 1) // self.strides[0]
        width = (int(x.shape[2]) + self.strides[1] - 1) // self.strides[1]
        return (int(x.shape[0]), height, width, int(self.features))


def resnet_feature_model(depth: int, dtype=None) -> nn.Sequential:
    """Return the spatial ResNet18/34 trunk used by existing JAX checkpoints.

    ``dtype`` narrows the convolution inputs/outputs (e.g. ``jnp.bfloat16``);
    the variable tree is identical for every dtype, so checkpoints are shared.
    """

    stage_sizes = {
        18: (2, 2, 2, 2),
        34: (3, 4, 6, 3),
    }
    if int(depth) not in stage_sizes:
        raise ValueError(f"Unsupported ResNet depth {depth}; expected 18 or 34.")

    layers: list[Callable] = [
        _ResNetStem(dtype=dtype),
        max_pool_3x3_stride2_pad1,
    ]
    for stage, (features, blocks) in enumerate(
        zip((64, 128, 256, 512), stage_sizes[int(depth)], strict=True)
    ):
        for block in range(blocks):
            stride = 2 if stage > 0 and block == 0 else 1
            layers.append(
                _ResNetBlock(features, strides=(stride, stride), dtype=dtype)
            )
    return nn.Sequential(layers)


__all__ = ["resnet_feature_model"]
