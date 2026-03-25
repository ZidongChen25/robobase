from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from omegaconf import DictConfig

from robobase.method.bc import BCEncoderModelSpec, BCViewFusionModelSpec


@dataclass(frozen=True)
class DiffusionActorModelSpec:
    type: str
    sequence_length: int
    diffusion_step_embed_dim: int
    down_dims: tuple[int, ...]
    kernel_size: int
    n_groups: int


@dataclass(frozen=True)
class DiffusionModelSpec:
    actor_model: DiffusionActorModelSpec
    encoder_model: BCEncoderModelSpec | None
    view_fusion_model: BCViewFusionModelSpec | None


@dataclass(frozen=True)
class DiffusionSpec:
    lr: float
    adaptive_lr: bool
    num_train_steps: int
    actor_grad_clip: float | None
    num_diffusion_iters: int
    use_ema: bool
    model: DiffusionModelSpec


def _config_type(
    cfg: DictConfig | None,
    *,
    default: str,
    target_to_type: dict[str, str] | None = None,
) -> str:
    if cfg is None:
        return default

    config_type = cfg.get("type", None)
    if config_type is not None:
        return str(config_type).lower()

    if target_to_type is not None:
        target = str(cfg.get("_target_", "")).strip()
        if target in target_to_type:
            return target_to_type[target]

    return default


def diffusion_model_spec_from_cfg(cfg: DictConfig) -> DiffusionModelSpec:
    method_cfg = cfg.method
    actor_model_cfg = method_cfg.get("actor_model", None)
    if actor_model_cfg is None:
        raise ValueError("Diffusion requires an actor_model config.")

    actor_model_type = _config_type(
        actor_model_cfg,
        default="conditional_unet1d",
        target_to_type={
            "robobase.models.diffusion_models.ConditionalUnet1D": "conditional_unet1d",
            "robobase.backends.torch.models.diffusion.ConditionalUnet1D": "conditional_unet1d",
            "robobase.backends.jax.models.diffusion.JaxConditionalUnet1D": "conditional_unet1d",
        },
    )
    actor_model_spec = DiffusionActorModelSpec(
        type=actor_model_type,
        sequence_length=int(actor_model_cfg.get("sequence_length", cfg.action_sequence)),
        diffusion_step_embed_dim=int(
            actor_model_cfg.get("diffusion_step_embed_dim", 256)
        ),
        down_dims=tuple(int(v) for v in actor_model_cfg.get("down_dims", (256, 512, 1024))),
        kernel_size=int(actor_model_cfg.get("kernel_size", 5)),
        n_groups=int(actor_model_cfg.get("n_groups", 8)),
    )

    encoder_model_cfg = method_cfg.get("encoder_model", None)
    encoder_model_spec = None
    if encoder_model_cfg is not None:
        encoder_model_type = _config_type(
            encoder_model_cfg,
            default="resnet",
            target_to_type={
                "robobase.models.ResNetEncoder": "resnet",
                "robobase.backends.torch.models.encoder.ResNetEncoder": "resnet",
            },
        )
        encoder_model_spec = BCEncoderModelSpec(
            type=encoder_model_type,
            model=str(encoder_model_cfg.get("model", "resnet18")),
        )

    view_fusion_model_cfg = method_cfg.get("view_fusion_model", None)
    view_fusion_model_spec = None
    if view_fusion_model_cfg is not None:
        view_fusion_model_type = _config_type(
            view_fusion_model_cfg,
            default="multicam_feature",
            target_to_type={
                "robobase.models.FusionMultiCamFeature": "multicam_feature",
                "robobase.backends.torch.models.fusion.FusionMultiCamFeature": "multicam_feature",
            },
        )
        view_fusion_model_spec = BCViewFusionModelSpec(
            type=view_fusion_model_type,
            mode=str(view_fusion_model_cfg.get("mode", "flatten")).lower(),
        )

    return DiffusionModelSpec(
        actor_model=actor_model_spec,
        encoder_model=encoder_model_spec,
        view_fusion_model=view_fusion_model_spec,
    )


def diffusion_spec_from_cfg(cfg: DictConfig) -> DiffusionSpec:
    return DiffusionSpec(
        lr=float(cfg.method.lr),
        adaptive_lr=bool(cfg.method.adaptive_lr),
        num_train_steps=int(cfg.method.num_train_steps),
        actor_grad_clip=(
            None
            if cfg.method.actor_grad_clip is None
            else float(cfg.method.actor_grad_clip)
        ),
        num_diffusion_iters=int(cfg.method.num_diffusion_iters),
        use_ema=bool(cfg.method.get("use_ema", False)),
        model=diffusion_model_spec_from_cfg(cfg),
    )


if TYPE_CHECKING:
    from robobase.backends.torch.method.diffusion import Diffusion


def __getattr__(name: str):
    if name == "Diffusion":
        from robobase.backends.torch.method.diffusion import Diffusion

        return Diffusion
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DiffusionActorModelSpec",
    "DiffusionModelSpec",
    "DiffusionSpec",
    "Diffusion",
    "diffusion_model_spec_from_cfg",
    "diffusion_spec_from_cfg",
]
