from __future__ import annotations

from dataclasses import dataclass

from omegaconf import DictConfig


@dataclass(frozen=True)
class BCActorModelSpec:
    type: str
    hidden_dims: tuple[int, ...]
    num_rnn_layers: int
    rnn_hidden_size: int
    keys_to_bottleneck: tuple[str, ...]
    bottleneck_size: int
    norm_after_bottleneck: bool
    tanh_after_bottleneck: bool
    output_sequence_network_type: str
    output_sequence_length: int


@dataclass(frozen=True)
class BCEncoderModelSpec:
    type: str
    model: str


@dataclass(frozen=True)
class BCViewFusionModelSpec:
    type: str
    mode: str


@dataclass(frozen=True)
class BCModelSpec:
    actor_model: BCActorModelSpec
    encoder_model: BCEncoderModelSpec | None
    view_fusion_model: BCViewFusionModelSpec | None


@dataclass(frozen=True)
class BCSpec:
    lr: float
    adaptive_lr: bool
    num_train_steps: int
    actor_grad_clip: float | None
    model: BCModelSpec


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


def bc_model_spec_from_cfg(cfg: DictConfig) -> BCModelSpec:
    method_cfg = cfg.method
    actor_model_cfg = method_cfg.get("actor_model", None)
    if actor_model_cfg is None:
        raise ValueError("BC requires an actor_model config.")

    actor_model_type = _config_type(
        actor_model_cfg,
        default="mlp_bottleneck_sequence",
        target_to_type={
            "robobase.models.MLPWithBottleneckFeaturesAndSequenceOutput": "mlp_bottleneck_sequence",
            "robobase.backends.torch.models.fully_connected.MLPWithBottleneckFeaturesAndSequenceOutput": "mlp_bottleneck_sequence",
        },
    )
    actor_model_spec = BCActorModelSpec(
        type=actor_model_type,
        hidden_dims=tuple(
            int(v)
            for v in actor_model_cfg.get(
                "hidden_dims",
                actor_model_cfg.get("mlp_nodes", (256, 256)),
            )
        ),
        num_rnn_layers=int(actor_model_cfg.get("num_rnn_layers", 1)),
        rnn_hidden_size=int(actor_model_cfg.get("rnn_hidden_size", 128)),
        keys_to_bottleneck=tuple(str(v) for v in actor_model_cfg.get("keys_to_bottleneck", ())),
        bottleneck_size=int(actor_model_cfg.get("bottleneck_size", 50)),
        norm_after_bottleneck=bool(actor_model_cfg.get("norm_after_bottleneck", True)),
        tanh_after_bottleneck=bool(actor_model_cfg.get("tanh_after_bottleneck", True)),
        output_sequence_network_type=str(
            actor_model_cfg.get("output_sequence_network_type", "rnn")
        ).lower(),
        output_sequence_length=int(
            actor_model_cfg.get("output_sequence_length", cfg.action_sequence)
        ),
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

    return BCModelSpec(
        actor_model=actor_model_spec,
        encoder_model=encoder_model_spec,
        view_fusion_model=view_fusion_model_spec,
    )


def bc_spec_from_cfg(cfg: DictConfig) -> BCSpec:
    return BCSpec(
        lr=float(cfg.method.lr),
        adaptive_lr=bool(cfg.method.adaptive_lr),
        num_train_steps=int(cfg.method.num_train_steps),
        actor_grad_clip=(
            None
            if cfg.method.actor_grad_clip is None
            else float(cfg.method.actor_grad_clip)
        ),
        model=bc_model_spec_from_cfg(cfg),
    )


__all__ = [
    "BCActorModelSpec",
    "BCEncoderModelSpec",
    "BCModelSpec",
    "BCSpec",
    "BCViewFusionModelSpec",
    "bc_model_spec_from_cfg",
    "bc_spec_from_cfg",
]
