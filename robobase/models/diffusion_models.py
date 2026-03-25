"""Compatibility re-exports for legacy torch diffusion model imports."""

from robobase.backends.torch.models.diffusion import (
    ConditionalResidualBlock1D,
    ConditionalUnet1D,
    Conv1dBlock,
    Downsample1d,
    MLPWithBottleneckFeaturesForDiffusion,
    SinusoidalPosEmb,
    TransformerForDiffusion,
    Upsample1d,
    replace_bn_with_gn,
    replace_submodules,
)

__all__ = [
    "ConditionalResidualBlock1D",
    "ConditionalUnet1D",
    "Conv1dBlock",
    "Downsample1d",
    "MLPWithBottleneckFeaturesForDiffusion",
    "SinusoidalPosEmb",
    "TransformerForDiffusion",
    "Upsample1d",
    "replace_bn_with_gn",
    "replace_submodules",
]
