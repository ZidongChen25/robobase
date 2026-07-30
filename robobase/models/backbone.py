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
    full_memory_attention: bool = False
    depth: int = 12
    dropout: float = 0.0
    embedding_dropout: float | None = None
    conditioning_mode: str = "global"
    cond_predict_scale: bool = False
    global_condition_embed_dim: int = 0
    timestep_embedding_type: str = "campose"
    operator_variant: str = "legacy"
    compatibility_mode: str = "native"
    condition_adapter: str = "linear"
    condition_hidden_dims: tuple[int, ...] = ()
    condition_dropout: float = 0.0
    fourier_scale: float = 16.0


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
        full_memory_attention=bool(cfg.get("full_memory_attention", False)),
        depth=int(cfg.get("depth", cfg.get("num_layers", 12))),
        dropout=float(cfg.get("dropout", cfg.get("p_drop_attn", 0.0))),
        embedding_dropout=(
            float(cfg.get("embedding_dropout", cfg.get("p_drop_emb")))
            if cfg.get("embedding_dropout", cfg.get("p_drop_emb")) is not None
            else None
        ),
        conditioning_mode=str(cfg.get("conditioning_mode", "global")).lower(),
        cond_predict_scale=bool(cfg.get("cond_predict_scale", False)),
        global_condition_embed_dim=int(cfg.get("global_condition_embed_dim", 0)),
        timestep_embedding_type=str(
            cfg.get("timestep_embedding_type", "campose")
        ).lower(),
        operator_variant=str(cfg.get("operator_variant", "legacy")).lower(),
        compatibility_mode=str(cfg.get("compatibility_mode", "native")).lower(),
        condition_adapter=str(cfg.get("condition_adapter", "linear")).lower(),
        condition_hidden_dims=tuple(
            int(v) for v in cfg.get("condition_hidden_dims", ())
        ),
        condition_dropout=float(cfg.get("condition_dropout", 0.0)),
        fourier_scale=float(cfg.get("fourier_scale", 16.0)),
    )


def build_diffusion_backbone(
    spec: DiffusionBackboneSpec,
    *,
    action_dim: int,
    sequence_length: int,
    condition_dim: int,
    local_condition_dim: int = 0,
    input_action_dim: int | None = None,
) -> nn.Module:
    backbone_type = canonical_backbone_type(spec.type)
    input_action_dim = (
        int(action_dim) if input_action_dim is None else int(input_action_dim)
    )
    if input_action_dim <= 0:
        raise ValueError("input_action_dim must be positive.")
    if spec.sequence_length != sequence_length:
        raise ValueError(
            "Backbone sequence_length does not match the action space: "
            f"{spec.sequence_length} != {sequence_length}."
        )

    conditioning_mode = str(spec.conditioning_mode).lower()
    if conditioning_mode not in {"global", "local"}:
        raise ValueError(
            "Backbone conditioning_mode must be 'global' or 'local', got "
            f"'{spec.conditioning_mode}'."
        )

    compatibility_mode = str(spec.compatibility_mode).lower()
    if compatibility_mode not in {"native", "clean_diffuser"}:
        raise ValueError(
            "Backbone compatibility_mode must be 'native' or 'clean_diffuser', "
            f"got '{spec.compatibility_mode}'."
        )

    if backbone_type not in {"unet1d", "dit", "transformer"}:
        unsupported = []
        if conditioning_mode != "global":
            unsupported.append("conditioning_mode")
        if spec.cond_predict_scale:
            unsupported.append("cond_predict_scale")
        if spec.global_condition_embed_dim != 0:
            unsupported.append("global_condition_embed_dim")
        if spec.timestep_embedding_type != "campose":
            unsupported.append("timestep_embedding_type")
        if spec.operator_variant != "legacy":
            unsupported.append("operator_variant")
        if compatibility_mode != "native":
            unsupported.append("compatibility_mode")
        if unsupported:
            raise ValueError(
                f"Backbone type '{backbone_type}' does not support UNet-only "
                f"options: {', '.join(unsupported)}."
            )

    if backbone_type == "transformer":
        unsupported = []
        if conditioning_mode != "global":
            unsupported.append("conditioning_mode")
        if spec.cond_predict_scale:
            unsupported.append("cond_predict_scale")
        if spec.global_condition_embed_dim != 0:
            unsupported.append("global_condition_embed_dim")
        if spec.timestep_embedding_type != "campose":
            unsupported.append("timestep_embedding_type")
        if compatibility_mode != "native":
            unsupported.append("compatibility_mode")
        if unsupported:
            raise ValueError(
                "Transformer does not support these options: "
                + ", ".join(unsupported)
                + "."
            )
        if spec.operator_variant not in {"legacy", "torch"}:
            raise ValueError(
                "Transformer operator_variant must be 'legacy' or 'torch', got "
                f"'{spec.operator_variant}'."
            )
        if spec.d_model <= 0:
            raise ValueError("Transformer d_model must be positive.")
        if spec.n_heads <= 0 or spec.d_model % spec.n_heads:
            raise ValueError(
                "Transformer n_heads must be positive and divide d_model exactly."
            )
        if spec.num_layers < 0 or spec.n_cond_layers < 0:
            raise ValueError("Transformer layer counts must be non-negative.")
        if not 0.0 <= spec.dropout < 1.0:
            raise ValueError("Transformer dropout must lie in [0, 1).")
        if (
            spec.embedding_dropout is not None
            and not 0.0 <= spec.embedding_dropout < 1.0
        ):
            raise ValueError("Transformer embedding_dropout must lie in [0, 1).")

    if backbone_type != "dit":
        unsupported = []
        if spec.condition_adapter != "linear":
            unsupported.append("condition_adapter")
        if spec.condition_hidden_dims:
            unsupported.append("condition_hidden_dims")
        if spec.condition_dropout != 0.0:
            unsupported.append("condition_dropout")
        if spec.fourier_scale != 16.0:
            unsupported.append("fourier_scale")
        if unsupported:
            raise ValueError(
                f"Backbone type '{backbone_type}' does not support DiT-only "
                f"options: {', '.join(unsupported)}."
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
        if not spec.down_dims or any(int(dim) <= 0 for dim in spec.down_dims):
            raise ValueError("UNet down_dims must contain positive channel sizes.")
        if spec.diffusion_step_embed_dim <= 0:
            raise ValueError("UNet diffusion_step_embed_dim must be positive.")
        if spec.kernel_size <= 0 or spec.kernel_size % 2 == 0:
            raise ValueError("UNet kernel_size must be a positive odd integer.")
        if spec.n_groups <= 0:
            raise ValueError("UNet n_groups must be positive.")
        if spec.global_condition_embed_dim < 0:
            raise ValueError("UNet global_condition_embed_dim must be non-negative.")
        if spec.timestep_embedding_type not in {"campose", "clean_diffuser"}:
            raise ValueError(
                "UNet timestep_embedding_type must be 'campose' or "
                f"'clean_diffuser', got '{spec.timestep_embedding_type}'."
            )
        if spec.operator_variant not in {"legacy", "torch"}:
            raise ValueError(
                "UNet operator_variant must be 'legacy' or 'torch', got "
                f"'{spec.operator_variant}'."
            )

        if compatibility_mode == "clean_diffuser":
            mismatches = []
            if sequence_length & (sequence_length - 1):
                mismatches.append("sequence_length must be a power of two")
            if spec.diffusion_step_embed_dim % 2:
                mismatches.append("diffusion_step_embed_dim must be even")
            if spec.n_groups != 8:
                mismatches.append("n_groups must be 8")
            if conditioning_mode != "global":
                mismatches.append("conditioning_mode must be global")
            if not spec.cond_predict_scale:
                mismatches.append("cond_predict_scale must be true")
            if spec.global_condition_embed_dim != spec.diffusion_step_embed_dim:
                mismatches.append(
                    "global_condition_embed_dim must equal diffusion_step_embed_dim"
                )
            if spec.timestep_embedding_type != "clean_diffuser":
                mismatches.append("timestep_embedding_type must be clean_diffuser")
            if spec.operator_variant != "torch":
                mismatches.append("operator_variant must be torch")
            if condition_dim <= 0:
                mismatches.append("global condition_dim must be positive")
            for previous, channels in zip(spec.down_dims, spec.down_dims[1:]):
                if channels < previous or channels % previous:
                    mismatches.append(
                        "down_dims must be non-decreasing cumulative integer multiples"
                    )
                    break
            for channels in spec.down_dims:
                clean_groups = min(8, channels // 4)
                native_groups = min(8, channels)
                while channels % native_groups and native_groups > 1:
                    native_groups -= 1
                if clean_groups < 1 or channels % clean_groups:
                    mismatches.append(
                        f"down_dims contains unsupported CleanDiffuser width {channels}"
                    )
                    break
                if clean_groups != native_groups:
                    mismatches.append(
                        f"width {channels} selects {clean_groups} CleanDiffuser groups "
                        f"but {native_groups} native groups"
                    )
                    break
            if mismatches:
                raise ValueError(
                    "CleanDiffuser ChiUNet compatibility mismatch: "
                    + "; ".join(mismatches)
                    + "."
                )
        horizon_divisor = 2 ** max(len(spec.down_dims) - 1, 0)
        if sequence_length % horizon_divisor != 0:
            raise ValueError(
                "UNet sequence_length must be divisible by its downsampling "
                f"factor {horizon_divisor}, got {sequence_length}."
            )
        if conditioning_mode == "local" and local_condition_dim <= 0:
            raise ValueError(
                "UNet local conditioning requires local_condition_dim > 0."
            )
        if (
            conditioning_mode == "local"
            and spec.timestep_embedding_type == "clean_diffuser"
        ):
            raise NotImplementedError(
                "CleanDiffuser local-condition UNet topology is not implemented; "
                "use global conditioning or the CamPose local profile."
            )
        return JaxConditionalUnet1D(
            action_dim=action_dim,
            sequence_length=sequence_length,
            feature_dim=condition_dim,
            diffusion_step_embed_dim=spec.diffusion_step_embed_dim,
            down_dims=spec.down_dims,
            input_action_dim=input_action_dim,
            kernel_size=spec.kernel_size,
            n_groups=spec.n_groups,
            local_feature_dim=int(local_condition_dim),
            cond_predict_scale=bool(spec.cond_predict_scale),
            global_condition_embed_dim=int(spec.global_condition_embed_dim),
            timestep_embedding_type=spec.timestep_embedding_type,
            operator_variant=spec.operator_variant,
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
            full_memory_attention=spec.full_memory_attention,
            dropout=spec.dropout,
            embedding_dropout=spec.embedding_dropout,
            operator_variant=spec.operator_variant,
        )
    if backbone_type == "dit":
        if conditioning_mode != "global":
            raise ValueError("DiT supports global conditioning only.")
        if spec.cond_predict_scale or spec.global_condition_embed_dim != 0:
            raise ValueError(
                "DiT does not support UNet cond_predict_scale or "
                "global_condition_embed_dim options."
            )
        if spec.d_model <= 0 or spec.d_model % 2:
            raise ValueError("DiT d_model must be a positive even integer.")
        if spec.n_heads <= 0 or spec.d_model % spec.n_heads:
            raise ValueError("DiT d_model must be divisible by n_heads.")
        if spec.depth <= 0:
            raise ValueError("DiT depth must be positive.")
        if not 0.0 <= spec.dropout < 1.0:
            raise ValueError("DiT dropout must be in [0, 1).")
        if spec.operator_variant not in {"legacy", "torch"}:
            raise ValueError(
                "DiT operator_variant must be 'legacy' or 'torch', got "
                f"'{spec.operator_variant}'."
            )
        if spec.timestep_embedding_type not in {
            "campose",
            "clean_diffuser",
            "fourier",
        }:
            raise ValueError(
                "DiT timestep_embedding_type must be 'campose', "
                f"'clean_diffuser', or 'fourier', got "
                f"'{spec.timestep_embedding_type}'."
            )
        if spec.condition_adapter not in {"linear", "direct", "clean_mlp"}:
            raise ValueError(
                "DiT condition_adapter must be 'linear', 'direct', or "
                f"'clean_mlp', got '{spec.condition_adapter}'."
            )
        if any(dim <= 0 for dim in spec.condition_hidden_dims):
            raise ValueError("DiT condition_hidden_dims must contain positive widths.")
        if not 0.0 <= spec.condition_dropout < 1.0:
            raise ValueError("DiT condition_dropout must be in [0, 1).")
        if spec.fourier_scale <= 0.0:
            raise ValueError("DiT fourier_scale must be positive.")
        if spec.condition_adapter == "direct" and condition_dim not in {
            0,
            spec.diffusion_step_embed_dim,
        }:
            raise ValueError(
                "DiT direct conditioning requires condition_dim to equal "
                "diffusion_step_embed_dim."
            )
        if spec.condition_adapter == "clean_mlp":
            if condition_dim <= 0:
                raise ValueError(
                    "DiT clean_mlp conditioning requires condition_dim > 0."
                )
            if not spec.condition_hidden_dims:
                raise ValueError(
                    "DiT clean_mlp conditioning requires condition_hidden_dims."
                )
        if spec.timestep_embedding_type == "fourier" and (
            spec.diffusion_step_embed_dim <= 0 or spec.diffusion_step_embed_dim % 8
        ):
            raise ValueError(
                "DiT Fourier embedding requires diffusion_step_embed_dim to be "
                "a positive multiple of 8."
            )
        if spec.timestep_embedding_type != "fourier" and (
            spec.diffusion_step_embed_dim <= 0 or spec.diffusion_step_embed_dim % 2
        ):
            raise ValueError(
                "DiT positional embedding requires diffusion_step_embed_dim to "
                "be a positive even integer."
            )
        if spec.operator_variant == "legacy" and spec.dropout != 0.0:
            raise NotImplementedError(
                "Legacy DiT dropout is deterministic; use operator_variant=torch "
                "or set dropout=0."
            )
        if compatibility_mode == "clean_diffuser":
            mismatches = []
            if spec.operator_variant != "torch":
                mismatches.append("operator_variant must be torch")
            if spec.timestep_embedding_type not in {"clean_diffuser", "fourier"}:
                mismatches.append(
                    "timestep_embedding_type must be clean_diffuser or fourier"
                )
            if spec.condition_adapter not in {"direct", "clean_mlp"}:
                mismatches.append("condition_adapter must be direct or clean_mlp")
            if mismatches:
                raise ValueError(
                    "CleanDiffuser DiT compatibility mismatch: "
                    + "; ".join(mismatches)
                    + "."
                )
        return JaxDiT1DBackbone(
            action_dim=action_dim,
            sequence_length=sequence_length,
            condition_dim=condition_dim,
            time_embed_dim=spec.diffusion_step_embed_dim,
            d_model=spec.d_model,
            n_heads=spec.n_heads,
            depth=spec.depth,
            dropout=spec.dropout,
            timestep_embedding_type=spec.timestep_embedding_type,
            operator_variant=spec.operator_variant,
            condition_adapter=spec.condition_adapter,
            condition_hidden_dims=spec.condition_hidden_dims,
            condition_dropout=spec.condition_dropout,
            fourier_scale=spec.fourier_scale,
        )
    raise NotImplementedError(f"Unsupported backbone type '{spec.type}'.")


__all__ = [
    "DiffusionBackboneSpec",
    "backbone_spec_from_cfg",
    "build_diffusion_backbone",
    "canonical_backbone_type",
]
