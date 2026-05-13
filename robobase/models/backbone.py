"""Shared backbone registry for JAX imitation-learning objectives."""

from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
from omegaconf import DictConfig

from robobase.models.backbones import (
    JaxChiTransformerBackbone,
    JaxConditionalUnet1D,
    JaxDiT1DBackbone,
    JaxFullyConnectedBackbone,
)


@dataclass(frozen=True)
class DiffusionBackboneSpec:
    type: str
    sequence_length: int
    diffusion_step_embed_dim: int = 256
    down_dims: tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    hidden_dims: tuple[int, ...] = (256, 256)
    d_model: int = 256
    n_heads: int = 4
    num_layers: int = 8
    n_cond_layers: int = 0
    depth: int = 12
    dropout: float = 0.0


def canonical_backbone_type(backbone_type: str) -> str:
    aliases = {
        "conditional_unet1d": "unet1d",
        "jaxconditionalunet1d": "unet1d",
        "mlp": "fully_connected",
        "fc": "fully_connected",
        "fullyconnected": "fully_connected",
        "chitransformer": "transformer",
        "chi_transformer": "transformer",
        "dit1d": "dit",
        "dit_1d": "dit",
    }
    normalized = str(backbone_type).lower()
    return aliases.get(normalized, normalized)


def backbone_spec_from_cfg(
    cfg: DictConfig,
    *,
    default_type: str = "unet1d",
    default_sequence_length: int | None = None,
) -> DiffusionBackboneSpec:
    backbone_type = str(cfg.get("type", default_type)).lower()
    sequence_length = cfg.get("sequence_length", default_sequence_length)
    if sequence_length is None:
        raise ValueError("Backbone config requires sequence_length.")
    return DiffusionBackboneSpec(
        type=canonical_backbone_type(backbone_type),
        sequence_length=int(sequence_length),
        diffusion_step_embed_dim=int(cfg.get("diffusion_step_embed_dim", 256)),
        down_dims=tuple(int(v) for v in cfg.get("down_dims", (256, 512, 1024))),
        kernel_size=int(cfg.get("kernel_size", 5)),
        n_groups=int(cfg.get("n_groups", 8)),
        hidden_dims=tuple(int(v) for v in cfg.get("hidden_dims", (256, 256))),
        d_model=int(cfg.get("d_model", cfg.get("model_dim", 256))),
        n_heads=int(cfg.get("n_heads", cfg.get("nhead", 4))),
        num_layers=int(cfg.get("num_layers", 8)),
        n_cond_layers=int(cfg.get("n_cond_layers", 0)),
        depth=int(cfg.get("depth", cfg.get("num_layers", 12))),
        dropout=float(cfg.get("dropout", cfg.get("p_drop_attn", 0.0))),
    )


def build_diffusion_backbone(
    spec: DiffusionBackboneSpec,
    *,
    action_dim: int,
    sequence_length: int,
    condition_dim: int,
) -> nn.Module:
    backbone_type = canonical_backbone_type(spec.type)
    if spec.sequence_length != sequence_length:
        raise ValueError(
            "Backbone sequence_length does not match the action space: "
            f"{spec.sequence_length} != {sequence_length}."
        )

    if backbone_type == "fully_connected":
        return JaxFullyConnectedBackbone(
            action_dim=action_dim,
            sequence_length=sequence_length,
            condition_dim=condition_dim,
            time_embed_dim=spec.diffusion_step_embed_dim,
            hidden_dims=spec.hidden_dims,
        )
    if backbone_type == "unet1d":
        return JaxConditionalUnet1D(
            action_dim=action_dim,
            sequence_length=sequence_length,
            feature_dim=condition_dim,
            diffusion_step_embed_dim=spec.diffusion_step_embed_dim,
            down_dims=spec.down_dims,
            kernel_size=spec.kernel_size,
            n_groups=spec.n_groups,
        )
    if backbone_type == "transformer":
        return JaxChiTransformerBackbone(
            action_dim=action_dim,
            sequence_length=sequence_length,
            condition_dim=condition_dim,
            d_model=spec.d_model,
            n_heads=spec.n_heads,
            num_layers=spec.num_layers,
            n_cond_layers=spec.n_cond_layers,
            dropout=spec.dropout,
        )
    if backbone_type == "dit":
        return JaxDiT1DBackbone(
            action_dim=action_dim,
            sequence_length=sequence_length,
            condition_dim=condition_dim,
            time_embed_dim=spec.diffusion_step_embed_dim,
            d_model=spec.d_model,
            n_heads=spec.n_heads,
            depth=spec.depth,
            dropout=spec.dropout,
        )
    raise NotImplementedError(f"Unsupported backbone type '{spec.type}'.")


__all__ = [
    "DiffusionBackboneSpec",
    "backbone_spec_from_cfg",
    "build_diffusion_backbone",
    "canonical_backbone_type",
]
