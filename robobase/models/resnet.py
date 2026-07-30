"""Pure-Flax ResNet feature backbones with jax-resnet-compatible variables."""

from __future__ import annotations

from collections.abc import Callable

import flax.linen as nn


class _ConvBlock(nn.Module):
    features: int
    kernel_size: tuple[int, int] = (3, 3)
    strides: tuple[int, int] = (1, 1)
    padding: object = ((0, 0), (0, 0))
    activate: bool = True
    zero_scale: bool = False

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            self.features,
            self.kernel_size,
            self.strides,
            padding=self.padding,
            use_bias=False,
            kernel_init=nn.initializers.kaiming_normal(),
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
        return nn.relu(x) if self.activate else x


class _ResNetStem(nn.Module):
    @nn.compact
    def __call__(self, x):
        return _ConvBlock(
            64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding=((3, 3), (3, 3)),
            name="ConvBlock_0",
        )(x)


class _ResNetSkipConnection(nn.Module):
    strides: tuple[int, int]

    @nn.compact
    def __call__(self, x, output_shape):
        if x.shape != output_shape:
            x = _ConvBlock(
                int(output_shape[-1]),
                kernel_size=(1, 1),
                strides=self.strides,
                activate=False,
                name="ConvBlock_0",
            )(x)
        return x


class _ResNetBlock(nn.Module):
    features: int
    strides: tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = _ResNetSkipConnection(
            self.strides,
            name="ResNetSkipConnection_0",
        )(x, self._output_shape(x))
        y = _ConvBlock(
            self.features,
            strides=self.strides,
            padding=((1, 1), (1, 1)),
            name="ConvBlock_0",
        )(x)
        y = _ConvBlock(
            self.features,
            padding=((1, 1), (1, 1)),
            activate=False,
            zero_scale=True,
            name="ConvBlock_1",
        )(y)
        return nn.relu(y + residual)

    def _output_shape(self, x) -> tuple[int, ...]:
        height = (int(x.shape[1]) + self.strides[0] - 1) // self.strides[0]
        width = (int(x.shape[2]) + self.strides[1] - 1) // self.strides[1]
        return (int(x.shape[0]), height, width, int(self.features))


def resnet_feature_model(depth: int) -> nn.Sequential:
    """Return the spatial ResNet18/34 trunk used by existing JAX checkpoints."""

    stage_sizes = {
        18: (2, 2, 2, 2),
        34: (3, 4, 6, 3),
    }
    if int(depth) not in stage_sizes:
        raise ValueError(f"Unsupported ResNet depth {depth}; expected 18 or 34.")

    layers: list[Callable] = [
        _ResNetStem(),
        lambda x: nn.max_pool(
            x,
            window_shape=(3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),
        ),
    ]
    for stage, (features, blocks) in enumerate(
        zip((64, 128, 256, 512), stage_sizes[int(depth)], strict=True)
    ):
        for block in range(blocks):
            stride = 2 if stage > 0 and block == 0 else 1
            layers.append(_ResNetBlock(features, strides=(stride, stride)))
    return nn.Sequential(layers)


__all__ = ["resnet_feature_model"]
