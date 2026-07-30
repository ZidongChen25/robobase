"""Rectified Flow / Flow Matching imitation-learning method in JAX."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Iterator, Literal, Optional

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig

from robobase.language import lang_feature_rows, lang_token_rows, tokens_to_feature_jax
from robobase.method.bc import BCEncoderModelSpec, BCViewFusionModelSpec
from robobase.method.bc_runtime import bc_observation_layout
from robobase.method.flow_sources import (
    A2AFlowSource,
    GaussianFlowSource,
    LegatoFlowSource,
    legato_inference_source,
    legato_schedule,
)
from robobase.method.jax_base import JaxMethodBase
from robobase.models.a2a import ActionChunkDecoder, TemporalActionEncoder
from robobase.models.backbone import (
    DiffusionBackboneSpec,
    backbone_spec_from_cfg,
    build_diffusion_backbone,
    canonical_backbone_type,
)
from robobase.models.camera_augmentation import augment_campose_observation
from robobase.models.encoder import (
    JaxDPEarlyFusionEncoder,
    JaxResNetEncoder,
    normalize_plucker_fusion_mode,
)
from robobase.models.fusion import JaxFusionMultiCamFeature
from robobase.replay_buffer.replay_buffer import ReplayBuffer


FlowMatchingBackboneSpec = DiffusionBackboneSpec


@dataclass(frozen=True)
class FlowSourceSpec:
    """Configuration for the replaceable source/path part of a flow policy."""

    type: Literal["gaussian", "a2a", "a2a_noise", "legato"] = "gaussian"

    # A2A. The public implementation uses two temporal encoders and one decoder.
    history_horizon: int = 8
    history_padding: str = "zero"
    history_source: Literal[
        "commanded_action", "executed_action_feedback"
    ] = "commanded_action"
    latent_dim: int = 512
    hidden_dim: int = 512
    encoder_layers: int = 3
    decoder_layers: int = 4
    kernel_size: int = 5
    decoder_dropout: float = 0.0
    train_history_noise_std: float = 0.0
    eval_history_noise_std: float = 0.0
    noise_exclude_last_n: int = 0
    consistency_steps: int = 1
    consistency_loss_type: Literal["mse", "l1"] = "mse"
    consistency_stop_gradient_target: bool = False
    fm_loss_weight: float = 1.0
    autoencoder_loss_weight: float = 0.5
    consistency_loss_weight: float = 1.0
    consistency_action_loss_weight: float = 0.5

    # Legato. Omega follows the paper convention: one is fully guided.
    delay_min_steps: int = 1
    delay_max_steps: int = 2
    ramp_min_steps: int = 1
    ramp_max_steps: int = 2
    schedule_profiles: tuple[str, ...] = ("linear",)
    strength_min: float = 1.0
    strength_max: float = 1.0
    no_guidance_probability: float = 0.0
    eval_delay_steps: int = 1
    eval_ramp_steps: int = 2
    eval_schedule_profile: str = "linear"
    eval_strength: float = 1.0
    target_mode: Literal["paper_minus", "public_kinetix_plus"] = "paper_minus"
    reference_padding: Literal["last", "zero"] = "last"


@dataclass(frozen=True)
class FlowMatchingModelSpec:
    backbone: FlowMatchingBackboneSpec
    encoder_model: BCEncoderModelSpec | None
    view_fusion_model: BCViewFusionModelSpec | None
    use_lang_cond: bool = False
    lang_feature_dim: int = 512


@dataclass(frozen=True)
class FlowMatchingSpec:
    lr: float
    adaptive_lr: bool
    lr_schedule: str
    num_train_steps: int
    actor_grad_clip: float | None
    objective_type: str
    num_flow_steps: int
    sampler: str
    sample_schedule: str
    train_time_schedule: str
    time_scale: float
    horizon_dropout_lengths: tuple[int, ...] | None
    horizon_dropout_probs: tuple[float, ...] | None
    horizon_loss_weights: tuple[float, ...] | None
    use_ema: bool
    ema_decay: float
    ema_decay_schedule: str
    weight_decay: float
    image_augmentation_type: str
    flow_source: FlowSourceSpec
    model: FlowMatchingModelSpec


def validate_legato_overlap(
    flow_source: FlowSourceSpec,
    *,
    action_sequence: int,
    execution_length: int,
    action_execution_start: int,
) -> None:
    """Reject continuation schedules that reach past the retained old chunk."""
    available_reference_steps = int(action_sequence) - int(execution_length)
    checks = (
        (
            flow_source.delay_max_steps,
            flow_source.ramp_max_steps,
            "delay_max_steps",
            "ramp_max_steps",
        ),
        (
            flow_source.eval_delay_steps,
            flow_source.eval_ramp_steps,
            "eval_delay_steps",
            "eval_ramp_steps",
        ),
    )
    for delay, ramp, delay_name, ramp_name in checks:
        schedule_end = int(action_execution_start) + int(delay) + int(ramp)
        if schedule_end > available_reference_steps:
            raise ValueError(
                f"Legato action_execution_start + {delay_name} + {ramp_name} "
                "cannot exceed the previous chunk overlap "
                "(action_sequence - execution_length)."
            )


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


def _sample_schedule_jump_point(sample_schedule: str) -> float | None:
    if sample_schedule in {"uniform", "linear"}:
        return None
    for prefix in ("front_loaded_", "nonuniform_"):
        if sample_schedule.startswith(prefix):
            value = sample_schedule[len(prefix) :].replace("p", ".")
            try:
                jump_point = float(value)
            except ValueError:
                break
            if 0.0 < jump_point < 1.0:
                return jump_point
            break
    raise NotImplementedError(
        f"Unsupported Flow Matching sample_schedule '{sample_schedule}'."
    )


def _parse_schedule_float(value: str) -> float:
    return float(value.replace("p", "."))


def _train_time_beta_params(train_time_schedule: str) -> tuple[float, float] | None:
    if train_time_schedule in {"uniform", "linear"}:
        return None
    if train_time_schedule in {"beta", "u_shaped", "u_shaped_beta"}:
        return 0.5, 0.5
    for prefix in ("beta_", "u_shaped_beta_"):
        if not train_time_schedule.startswith(prefix):
            continue
        parts = train_time_schedule[len(prefix) :].split("_")
        try:
            if len(parts) == 1:
                alpha = beta = _parse_schedule_float(parts[0])
            elif len(parts) == 2:
                alpha = _parse_schedule_float(parts[0])
                beta = _parse_schedule_float(parts[1])
            else:
                break
        except ValueError:
            break
        if alpha > 0.0 and beta > 0.0:
            return alpha, beta
        break
    raise NotImplementedError(
        f"Unsupported Flow Matching train_time_schedule '{train_time_schedule}'."
    )


def _rectified_flow_training_pair(actions, source_noise, time):
    """Interpolate noise-to-data and return CleanDiffuser's reverse velocity."""
    time = time[:, None, None]
    sample = time * source_noise + (1.0 - time) * actions
    return sample, actions - source_noise


def flow_matching_model_spec_from_cfg(cfg: DictConfig) -> FlowMatchingModelSpec:
    method_cfg = cfg.method
    backbone_cfg = method_cfg.get("backbone", None)
    if backbone_cfg is None:
        raise ValueError("Flow Matching requires a backbone config.")

    backbone_spec = backbone_spec_from_cfg(
        backbone_cfg,
        default_type="fully_connected",
        default_sequence_length=int(cfg.action_sequence),
    )

    encoder_model_cfg = method_cfg.get("encoder_model", None)
    encoder_model_spec = None
    if encoder_model_cfg is not None:
        encoder_model_type = _config_type(
            encoder_model_cfg,
            default="resnet",
            target_to_type={
                "robobase.models.encoder.JaxResNetEncoder": "resnet",
            },
        )
        encoder_model_spec = BCEncoderModelSpec(
            type=encoder_model_type,
            model=str(encoder_model_cfg.get("model", "resnet18")),
            trainable=bool(encoder_model_cfg.get("trainable", False)),
            pretrained=bool(encoder_model_cfg.get("pretrained", True)),
            pretrained_weights_path=encoder_model_cfg.get(
                "pretrained_weights_path",
                None,
            ),
            use_plucker=bool(encoder_model_cfg.get("use_plucker", False)),
            plucker_hidden_channels=int(
                encoder_model_cfg.get("plucker_hidden_channels", 64)
            ),
            plucker_identity_init=bool(
                encoder_model_cfg.get("plucker_identity_init", False)
            ),
            plucker_fusion_mode=encoder_model_cfg.get("plucker_fusion_mode", None),
        )

    view_fusion_model_cfg = method_cfg.get("view_fusion_model", None)
    view_fusion_model_spec = None
    if view_fusion_model_cfg is not None:
        view_fusion_model_type = _config_type(
            view_fusion_model_cfg,
            default="multicam_feature",
            target_to_type={
                "robobase.models.fusion.JaxFusionMultiCamFeature": "multicam_feature",
            },
        )
        view_fusion_model_spec = BCViewFusionModelSpec(
            type=view_fusion_model_type,
            mode=str(view_fusion_model_cfg.get("mode", "flatten")).lower(),
        )

    return FlowMatchingModelSpec(
        backbone=backbone_spec,
        encoder_model=encoder_model_spec,
        view_fusion_model=view_fusion_model_spec,
        use_lang_cond=bool(method_cfg.get("use_lang_cond", False)),
        lang_feature_dim=int(method_cfg.get("lang_feature_dim", 512)),
    )


def flow_matching_spec_from_cfg(cfg: DictConfig) -> FlowMatchingSpec:
    objective_cfg = cfg.method.get("objective", None)
    objective_type = (
        str(objective_cfg.get("type", "rectified_flow")).lower()
        if objective_cfg is not None
        else "rectified_flow"
    )
    num_flow_steps = (
        int(objective_cfg.get("num_flow_steps"))
        if objective_cfg is not None and objective_cfg.get("num_flow_steps") is not None
        else int(cfg.method.get("num_flow_steps", 5))
    )
    sampler = (
        str(objective_cfg.get("sampler", "euler")).lower()
        if objective_cfg is not None
        else str(cfg.method.get("sampler", "euler")).lower()
    )
    sample_schedule = (
        str(
            objective_cfg.get(
                "sample_schedule",
                cfg.method.get("sample_schedule", "uniform"),
            )
        ).lower()
        if objective_cfg is not None
        else str(cfg.method.get("sample_schedule", "uniform")).lower()
    )
    train_time_schedule = (
        str(
            objective_cfg.get(
                "train_time_schedule",
                cfg.method.get("train_time_schedule", "uniform"),
            )
        ).lower()
        if objective_cfg is not None
        else str(cfg.method.get("train_time_schedule", "uniform")).lower()
    )
    horizon_dropout_lengths = cfg.method.get("horizon_dropout_lengths", None)
    horizon_dropout_probs = cfg.method.get("horizon_dropout_probs", None)
    if horizon_dropout_lengths is not None:
        horizon_dropout_lengths = tuple(int(v) for v in horizon_dropout_lengths)
        if len(horizon_dropout_lengths) == 0:
            horizon_dropout_lengths = None
    if horizon_dropout_probs is not None:
        horizon_dropout_probs = tuple(float(v) for v in horizon_dropout_probs)
        if len(horizon_dropout_probs) == 0:
            horizon_dropout_probs = None
    horizon_loss_weights = cfg.method.get("horizon_loss_weights", None)
    if horizon_loss_weights is not None:
        horizon_loss_weights = tuple(float(v) for v in horizon_loss_weights)
        if len(horizon_loss_weights) == 0:
            horizon_loss_weights = None
    source_cfg = cfg.method.get("flow_source", {})
    method_name = cfg.method.get("name", None)
    if method_name is None:
        method_target = str(cfg.method.get("_target_", "")).strip()
        method_name = {
            "robobase.method.a2a.A2A": "a2a",
            "robobase.method.legato.Legato": "legato",
        }.get(method_target, "flow_matching")
    method_name = str(method_name).lower()
    default_source_type = (
        method_name if method_name in {"a2a", "a2a_noise", "legato"} else "gaussian"
    )
    source_type = str(source_cfg.get("type", default_source_type)).lower()
    if source_type not in {"gaussian", "a2a", "a2a_noise", "legato"}:
        raise ValueError(
            "method.flow_source.type must be gaussian, a2a, a2a_noise, or "
            f"legato; got {source_type!r}."
        )
    schedule_profiles = tuple(
        str(value).lower() for value in source_cfg.get("schedule_profiles", ["linear"])
    )
    return FlowMatchingSpec(
        lr=float(cfg.method.lr),
        adaptive_lr=bool(cfg.method.adaptive_lr),
        lr_schedule=str(cfg.method.get("lr_schedule", "cosine")),
        num_train_steps=int(cfg.method.num_train_steps),
        actor_grad_clip=(
            None
            if cfg.method.actor_grad_clip is None
            else float(cfg.method.actor_grad_clip)
        ),
        objective_type=objective_type,
        num_flow_steps=num_flow_steps,
        sampler=sampler,
        sample_schedule=sample_schedule,
        train_time_schedule=train_time_schedule,
        time_scale=float(
            objective_cfg.get("time_scale", cfg.method.get("time_scale", 1000.0))
            if objective_cfg is not None
            else cfg.method.get("time_scale", 1000.0)
        ),
        horizon_dropout_lengths=horizon_dropout_lengths,
        horizon_dropout_probs=horizon_dropout_probs,
        horizon_loss_weights=horizon_loss_weights,
        use_ema=bool(cfg.method.get("use_ema", False)),
        ema_decay=float(cfg.method.get("ema_decay", 0.9999)),
        ema_decay_schedule=str(
            cfg.method.get("ema_decay_schedule", "diffusers")
        ).lower(),
        weight_decay=float(cfg.method.get("weight_decay", 1e-6)),
        image_augmentation_type=str(
            cfg.method.get("image_augmentation_type", "none")
        ).lower(),
        flow_source=FlowSourceSpec(
            type=source_type,
            history_horizon=int(source_cfg.get("history_horizon", 8)),
            history_padding=str(source_cfg.get("history_padding", "zero")).lower(),
            history_source=str(
                source_cfg.get("history_source", "commanded_action")
            ).lower(),
            latent_dim=int(source_cfg.get("latent_dim", 512)),
            hidden_dim=int(source_cfg.get("hidden_dim", 512)),
            encoder_layers=int(source_cfg.get("encoder_layers", 3)),
            decoder_layers=int(source_cfg.get("decoder_layers", 4)),
            kernel_size=int(source_cfg.get("kernel_size", 5)),
            decoder_dropout=float(source_cfg.get("decoder_dropout", 0.0)),
            train_history_noise_std=float(
                source_cfg.get(
                    "train_history_noise_std",
                    source_cfg.get("history_noise_std", 0.0),
                )
            ),
            eval_history_noise_std=float(
                source_cfg.get(
                    "eval_history_noise_std",
                    source_cfg.get("history_noise_std", 0.0),
                )
            ),
            noise_exclude_last_n=int(source_cfg.get("noise_exclude_last_n", 0)),
            consistency_steps=int(source_cfg.get("consistency_steps", 1)),
            consistency_loss_type=str(
                source_cfg.get("consistency_loss_type", "mse")
            ).lower(),
            consistency_stop_gradient_target=bool(
                source_cfg.get("consistency_stop_gradient_target", False)
            ),
            fm_loss_weight=float(source_cfg.get("fm_loss_weight", 1.0)),
            autoencoder_loss_weight=float(
                source_cfg.get("autoencoder_loss_weight", 0.5)
            ),
            consistency_loss_weight=float(
                source_cfg.get("consistency_loss_weight", 1.0)
            ),
            consistency_action_loss_weight=float(
                source_cfg.get("consistency_action_loss_weight", 0.5)
            ),
            delay_min_steps=int(source_cfg.get("delay_min_steps", 1)),
            delay_max_steps=int(source_cfg.get("delay_max_steps", 2)),
            ramp_min_steps=int(source_cfg.get("ramp_min_steps", 1)),
            ramp_max_steps=int(source_cfg.get("ramp_max_steps", 2)),
            schedule_profiles=schedule_profiles,
            strength_min=float(source_cfg.get("strength_min", 1.0)),
            strength_max=float(source_cfg.get("strength_max", 1.0)),
            no_guidance_probability=float(
                source_cfg.get("no_guidance_probability", 0.0)
            ),
            eval_delay_steps=int(source_cfg.get("eval_delay_steps", 1)),
            eval_ramp_steps=int(source_cfg.get("eval_ramp_steps", 2)),
            eval_schedule_profile=str(
                source_cfg.get("eval_schedule_profile", "linear")
            ).lower(),
            eval_strength=float(source_cfg.get("eval_strength", 1.0)),
            target_mode=str(source_cfg.get("target_mode", "paper_minus")).lower(),
            reference_padding=str(source_cfg.get("reference_padding", "last")).lower(),
        ),
        model=flow_matching_model_spec_from_cfg(cfg),
    )


@dataclass(frozen=True)
class _BuiltFlowMatchingModel:
    backbone_model: object
    encoder_model: JaxResNetEncoder | JaxDPEarlyFusionEncoder | None
    view_fusion_model: JaxFusionMultiCamFeature | None


def _build_encoder_and_fusion(
    *,
    model_spec: FlowMatchingModelSpec,
    observation_space: spaces.Dict,
    encoder_jit: bool,
    encoder_seed: int = 0,
) -> tuple[
    JaxResNetEncoder | JaxDPEarlyFusionEncoder | None,
    JaxFusionMultiCamFeature | None,
    int,
]:
    obs_layout = bc_observation_layout(observation_space)
    encoder_model = None
    view_fusion_model = None

    if obs_layout.use_pixels:
        if model_spec.encoder_model is None:
            raise ValueError(
                "Pixel Flow Matching requires encoder_model in the model spec."
            )
        if model_spec.encoder_model.type not in {"resnet", "dp_resnet"}:
            raise NotImplementedError(
                "Unsupported Flow Matching encoder model type "
                f"'{model_spec.encoder_model.type}'."
            )
        if obs_layout.rgb_input_shape is None:
            raise ValueError("Pixel Flow Matching expected a valid RGB input shape.")
        if model_spec.encoder_model.type == "dp_resnet":
            if model_spec.encoder_model.pretrained:
                raise ValueError(
                    "encoder type 'dp_resnet' requires pretrained=false for "
                    "CamPose DP parity."
                )
            if not model_spec.encoder_model.trainable:
                raise ValueError(
                    "encoder type 'dp_resnet' requires trainable=true so RGB "
                    "and Plucker ablations use the same learned encoder family."
                )
        if model_spec.encoder_model.use_plucker:
            if not model_spec.encoder_model.trainable:
                raise ValueError(
                    "Flow Matching encoder_model.use_plucker=true requires "
                    "encoder_model.trainable=true so the Plucker adapter is trained."
                )
            if not obs_layout.has_camera_conditioning:
                raise ValueError(
                    "Flow Matching encoder_model.use_plucker=true requires raymap or "
                    "camera parameter observations paired with every RGB observation."
                )
        if model_spec.encoder_model.type == "dp_resnet":
            encoder_cls = JaxDPEarlyFusionEncoder
            expected_fusion_mode = (
                "dp_early" if model_spec.encoder_model.use_plucker else "none"
            )
            plucker_fusion_mode = normalize_plucker_fusion_mode(
                model_spec.encoder_model.plucker_fusion_mode,
                use_plucker=model_spec.encoder_model.use_plucker,
            )
            if plucker_fusion_mode != expected_fusion_mode:
                raise ValueError(
                    "encoder type 'dp_resnet' supports only plucker_fusion_mode="
                    f"'{expected_fusion_mode}', got '{plucker_fusion_mode}'."
                )
        else:
            encoder_cls = JaxResNetEncoder
            plucker_fusion_mode = normalize_plucker_fusion_mode(
                model_spec.encoder_model.plucker_fusion_mode,
                use_plucker=model_spec.encoder_model.use_plucker,
            )
            if plucker_fusion_mode == "dp_early":
                raise ValueError(
                    "plucker_fusion_mode='dp_early' requires encoder type "
                    "'dp_resnet'; type='resnet' is the legacy spatial encoder."
                )
        encoder_kwargs = dict(
            input_shape=obs_layout.rgb_input_shape,
            model=model_spec.encoder_model.model,
            jit=encoder_jit,
            pretrained=model_spec.encoder_model.pretrained,
            use_plucker=model_spec.encoder_model.use_plucker,
            plucker_fusion_mode=plucker_fusion_mode,
            seed=encoder_seed,
        )
        if encoder_cls is JaxResNetEncoder:
            encoder_kwargs.update(
                pretrained_weights_path=(
                    model_spec.encoder_model.pretrained_weights_path
                ),
                plucker_hidden_channels=(
                    model_spec.encoder_model.plucker_hidden_channels
                ),
                plucker_identity_init=(model_spec.encoder_model.plucker_identity_init),
            )
        encoder_model = encoder_cls(**encoder_kwargs)
        if obs_layout.use_multicam_fusion:
            if model_spec.view_fusion_model is None:
                raise ValueError(
                    "Multi-camera pixel Flow Matching requires view_fusion_model."
                )
            if model_spec.view_fusion_model.type != "multicam_feature":
                raise NotImplementedError(
                    "Unsupported Flow Matching view fusion model type "
                    f"'{model_spec.view_fusion_model.type}'."
                )
            view_fusion_model = JaxFusionMultiCamFeature.from_input_shape(
                input_shape=encoder_model.output_shape,
                mode=model_spec.view_fusion_model.mode,
            )
    elif model_spec.encoder_model is not None and model_spec.encoder_model.type not in {
        "resnet",
        "dp_resnet",
    }:
        raise NotImplementedError(
            "Unsupported Flow Matching encoder model type "
            f"'{model_spec.encoder_model.type}'."
        )

    rgb_latent_size = 0
    if encoder_model is not None:
        if view_fusion_model is not None:
            rgb_latent_size = int(view_fusion_model.output_shape[-1])
        else:
            rgb_latent_size = int(encoder_model.output_shape[-1])
    return encoder_model, view_fusion_model, rgb_latent_size


def _build_model(
    model_spec: FlowMatchingModelSpec,
    *,
    observation_space: spaces.Dict,
    action_space: spaces.Box,
    encoder_jit: bool,
    encoder_seed: int = 0,
    actor_sequence_length: int | None = None,
    actor_action_dim: int | None = None,
    actor_input_action_dim: int | None = None,
) -> tuple[_BuiltFlowMatchingModel, int]:
    obs_layout = bc_observation_layout(observation_space)
    backbone_type = canonical_backbone_type(model_spec.backbone.type)
    encoder_model, view_fusion_model, rgb_latent_size = _build_encoder_and_fusion(
        model_spec=model_spec,
        observation_space=observation_space,
        encoder_jit=encoder_jit,
        encoder_seed=encoder_seed,
    )
    lang_feature_dim = (
        int(model_spec.lang_feature_dim) if model_spec.use_lang_cond else 0
    )
    feature_dim = int(obs_layout.low_dim_size + rgb_latent_size + lang_feature_dim)
    local_condition_dim = 0
    actor_condition_dim = feature_dim
    if (
        backbone_type == "unet1d"
        and str(model_spec.backbone.conditioning_mode).lower() == "local"
    ):
        local_condition_dim = int(obs_layout.low_dim_size)
        actor_condition_dim = int(rgb_latent_size + lang_feature_dim)
    if backbone_type == "transformer" and not obs_layout.use_pixels:
        low_dim_state_spec = observation_space.spaces.get("low_dim_state")
        if low_dim_state_spec is not None:
            feature_dim = int(low_dim_state_spec.shape[-1])
            if model_spec.use_lang_cond:
                feature_dim += int(model_spec.lang_feature_dim)
            actor_condition_dim = feature_dim
    actor_sequence_length = (
        int(action_space.shape[0])
        if actor_sequence_length is None
        else int(actor_sequence_length)
    )
    actor_action_dim = (
        int(action_space.shape[1])
        if actor_action_dim is None
        else int(actor_action_dim)
    )
    actor_input_action_dim = (
        actor_action_dim
        if actor_input_action_dim is None
        else int(actor_input_action_dim)
    )
    return (
        _BuiltFlowMatchingModel(
            backbone_model=build_diffusion_backbone(
                model_spec.backbone,
                action_dim=actor_action_dim,
                sequence_length=actor_sequence_length,
                condition_dim=actor_condition_dim,
                local_condition_dim=local_condition_dim,
                input_action_dim=actor_input_action_dim,
            ),
            encoder_model=encoder_model,
            view_fusion_model=view_fusion_model,
        ),
        feature_dim,
    )


class FlowMatching(JaxMethodBase):
    """Rectified Flow method using JAX/Flax backbones."""

    def __init__(
        self,
        lr: float,
        adaptive_lr: bool,
        num_train_steps: int,
        num_flow_steps: int,
        model: FlowMatchingModelSpec,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        num_train_envs: int,
        num_eval_envs: int,
        replay_alpha: float,
        replay_beta: float,
        frame_stack_on_channel: bool,
        intrinsic_reward_module=None,
        actor_grad_clip: Optional[float] = None,
        lr_schedule: str = "cosine",
        jit: bool = True,
        platform: str | None = None,
        seed: int = 0,
        is_rl: bool = False,
        use_ema: bool = False,
        ema_decay: float = 0.9999,
        ema_decay_schedule: str = "diffusers",
        weight_decay: float = 1e-6,
        objective_type: str = "rectified_flow",
        sampler: str = "euler",
        sample_schedule: str = "uniform",
        train_time_schedule: str = "uniform",
        time_scale: float = 1000.0,
        image_augmentation_type: str = "none",
        horizon_dropout_lengths: tuple[int, ...] | None = None,
        horizon_dropout_probs: tuple[float, ...] | None = None,
        horizon_loss_weights: tuple[float, ...] | None = None,
        update_block_every_steps: int = 1,
        flow_source: FlowSourceSpec | None = None,
        execution_length: int | None = None,
        action_execution_start: int = 0,
    ):
        if not frame_stack_on_channel:
            raise NotImplementedError(
                "frame_stack_on_channel must be true for Flow Matching policies."
            )
        if str(objective_type).lower() not in {"rectified_flow", "flow_matching"}:
            raise NotImplementedError(
                "FlowMatching currently supports the Rectified Flow objective."
            )
        if str(sampler).lower() != "euler":
            raise NotImplementedError("FlowMatching currently supports Euler sampling.")
        sample_schedule = str(sample_schedule).lower()
        sample_schedule_jump_point = _sample_schedule_jump_point(sample_schedule)
        train_time_schedule = str(train_time_schedule).lower()
        train_time_beta_params = _train_time_beta_params(train_time_schedule)
        if int(num_flow_steps) < 1:
            raise ValueError("num_flow_steps must be >= 1.")
        if not np.isfinite(time_scale) or float(time_scale) <= 0.0:
            raise ValueError("time_scale must be finite and > 0.")

        super().__init__(
            lr=lr,
            adaptive_lr=adaptive_lr,
            num_train_steps=num_train_steps,
            observation_space=observation_space,
            action_space=action_space,
            num_train_envs=num_train_envs,
            num_eval_envs=num_eval_envs,
            replay_alpha=replay_alpha,
            replay_beta=replay_beta,
            frame_stack_on_channel=frame_stack_on_channel,
            intrinsic_reward_module=intrinsic_reward_module,
            actor_grad_clip=actor_grad_clip,
            jit=jit,
            platform=platform,
            seed=seed,
            is_rl=is_rl,
            use_ema=use_ema,
            update_block_every_steps=update_block_every_steps,
        )

        self.flow_source = FlowSourceSpec() if flow_source is None else flow_source
        self._flow_source_type = str(self.flow_source.type).lower()
        if self._flow_source_type not in {
            "gaussian",
            "a2a",
            "a2a_noise",
            "legato",
        }:
            raise ValueError(f"Unsupported flow source {self.flow_source.type!r}.")
        if self._flow_source_type == "gaussian":
            self.flow_source_module = GaussianFlowSource()
        elif self._flow_source_type in {"a2a", "a2a_noise"}:
            self.flow_source_module = A2AFlowSource()
        else:
            self.flow_source_module = LegatoFlowSource(
                target_mode=self.flow_source.target_mode
            )
        self.execution_length = (
            self.action_sequence if execution_length is None else int(execution_length)
        )
        self.action_execution_start = int(action_execution_start)
        if self.execution_length < 1:
            raise ValueError("execution_length must be >= 1.")
        if (
            self.action_execution_start < 0
            or self.action_execution_start + self.execution_length
            > self.action_sequence
        ):
            raise ValueError(
                "action_execution_start + execution_length must fit in the "
                "action sequence."
            )
        if self._flow_source_type in {"a2a", "a2a_noise"}:
            if self.flow_source.history_source not in {
                "commanded_action",
                "executed_action_feedback",
            }:
                raise ValueError(
                    "A2A history_source must be commanded_action or "
                    "executed_action_feedback."
                )
            if self.flow_source.history_horizon < 1:
                raise ValueError("A2A history_horizon must be >= 1.")
            if self.flow_source.latent_dim < 1 or self.flow_source.hidden_dim < 1:
                raise ValueError("A2A latent_dim and hidden_dim must be positive.")
            if (
                self.flow_source.encoder_layers < 1
                or self.flow_source.decoder_layers < 1
            ):
                raise ValueError("A2A encoder_layers and decoder_layers must be >= 1.")
            if self.flow_source.consistency_steps < 1:
                raise ValueError("A2A consistency_steps must be >= 1.")
            if self.flow_source.consistency_loss_type not in {"mse", "l1"}:
                raise ValueError("A2A consistency_loss_type must be mse or l1.")
            if not 0 <= self.flow_source.noise_exclude_last_n <= self.action_dim:
                raise ValueError(
                    "A2A noise_exclude_last_n must be between zero and action_dim."
                )
            if (
                self.flow_source.train_history_noise_std < 0.0
                or self.flow_source.eval_history_noise_std < 0.0
            ):
                raise ValueError("A2A history noise standard deviations must be >= 0.")
            if not 0.0 <= self.flow_source.decoder_dropout < 1.0:
                raise ValueError("A2A decoder_dropout must lie in [0, 1).")
            if sample_schedule not in {"uniform", "linear"}:
                raise ValueError(
                    "A2A latent integration currently requires a uniform Euler "
                    "sample schedule."
                )
        if self._flow_source_type == "legato":
            if self.flow_source.target_mode not in {
                "paper_minus",
                "public_kinetix_plus",
            }:
                raise ValueError(
                    "Legato target_mode must be paper_minus or public_kinetix_plus."
                )
            profiles = set(self.flow_source.schedule_profiles)
            if not profiles or not profiles.issubset({"hard", "linear", "cosine"}):
                raise ValueError(
                    "Legato schedule_profiles must contain hard, linear, or cosine."
                )
            if self.flow_source.eval_schedule_profile not in {
                "hard",
                "linear",
                "cosine",
            }:
                raise ValueError("Unsupported Legato eval_schedule_profile.")
            if self.flow_source.reference_padding not in {"last", "zero"}:
                raise ValueError("Legato reference_padding must be last or zero.")
            if not (
                0.0
                <= self.flow_source.strength_min
                <= self.flow_source.strength_max
                <= 1.0
            ):
                raise ValueError("Legato training strengths must lie in [0, 1].")
            if not 0.0 <= self.flow_source.eval_strength <= 1.0:
                raise ValueError("Legato eval_strength must lie in [0, 1].")
            if not 0.0 <= self.flow_source.no_guidance_probability <= 1.0:
                raise ValueError(
                    "Legato no_guidance_probability must lie in [0, 1]."
                )
            for minimum, maximum, label in (
                (
                    self.flow_source.delay_min_steps,
                    self.flow_source.delay_max_steps,
                    "delay",
                ),
                (
                    self.flow_source.ramp_min_steps,
                    self.flow_source.ramp_max_steps,
                    "ramp",
                ),
            ):
                if minimum < 0 or maximum < minimum:
                    raise ValueError(
                        f"Legato {label} bounds must satisfy 0 <= min <= max."
                    )
                if maximum > self.action_sequence:
                    raise ValueError(
                        f"Legato {label} maximum cannot exceed action_sequence."
                    )
            if not 0 <= self.flow_source.eval_delay_steps <= self.execution_length:
                raise ValueError(
                    "Legato eval_delay_steps must be between zero and execution_length."
                )
            if not 0 <= self.flow_source.eval_ramp_steps <= self.action_sequence:
                raise ValueError(
                    "Legato eval_ramp_steps must be between zero and action_sequence."
                )
            validate_legato_overlap(
                self.flow_source,
                action_sequence=self.action_sequence,
                execution_length=self.execution_length,
                action_execution_start=self.action_execution_start,
            )
            if sample_schedule not in {"uniform", "linear"}:
                raise ValueError(
                    "Legato closed-form training currently requires a uniform "
                    "Euler sample schedule."
                )
        if self._flow_source_type != "gaussian" and (
            horizon_dropout_lengths is not None or horizon_loss_weights is not None
        ):
            raise ValueError(
                "Horizon dropout/weighting is currently defined only for the "
                "Gaussian FM source."
            )

        self.objective_type = str(objective_type).lower()
        self.sampler = str(sampler).lower()
        self.sample_schedule = sample_schedule
        self._sample_schedule_jump_point = sample_schedule_jump_point
        self.train_time_schedule = train_time_schedule
        self._train_time_beta_params = train_time_beta_params
        self._horizon_dropout_lengths = None
        self._horizon_dropout_probs = None
        self._horizon_loss_weights = None
        if horizon_dropout_lengths is not None:
            lengths = tuple(int(v) for v in horizon_dropout_lengths)
            if any(v < 1 or v > self.action_sequence for v in lengths):
                raise ValueError(
                    "horizon_dropout_lengths must be between 1 and action_sequence."
                )
            if horizon_dropout_probs is None:
                probs = tuple([1.0 / len(lengths)] * len(lengths))
            else:
                probs = tuple(float(v) for v in horizon_dropout_probs)
                if len(probs) != len(lengths):
                    raise ValueError(
                        "horizon_dropout_probs must match horizon_dropout_lengths."
                    )
                if any(v < 0.0 for v in probs) or sum(probs) <= 0.0:
                    raise ValueError(
                        "horizon_dropout_probs must be non-negative and sum > 0."
                    )
                prob_sum = sum(probs)
                probs = tuple(v / prob_sum for v in probs)
            self._horizon_dropout_lengths = jnp.asarray(lengths, dtype=jnp.int32)
            self._horizon_dropout_probs = jnp.asarray(probs, dtype=jnp.float32)
        if horizon_loss_weights is not None:
            weights = tuple(float(v) for v in horizon_loss_weights)
            if len(weights) != self.action_sequence:
                raise ValueError(
                    "horizon_loss_weights must have length equal to action_sequence."
                )
            if any(v < 0.0 for v in weights) or sum(weights) <= 0.0:
                raise ValueError(
                    "horizon_loss_weights must be non-negative and sum > 0."
                )
            weight_sum = sum(weights)
            weights = tuple(v / weight_sum for v in weights)
            self._horizon_loss_weights = jnp.asarray(weights, dtype=jnp.float32)
        self.num_flow_steps = int(num_flow_steps)
        # ContinuousRectifiedFlow feeds t in [0, 1] directly. Keep an explicit
        # scale for experimentation without changing the interpolation clock.
        self._time_scale = float(time_scale)
        self.image_augmentation_type = str(image_augmentation_type).lower()
        if self.image_augmentation_type not in {"none", "campose_crop"}:
            raise ValueError(
                "image_augmentation_type must be 'none' or 'campose_crop'; "
                f"got {image_augmentation_type!r}."
            )
        self.model_spec = model
        self.use_lang_cond = bool(model.use_lang_cond)
        self.lang_feature_dim = int(model.lang_feature_dim) if self.use_lang_cond else 0
        self._backbone_type = canonical_backbone_type(self.model_spec.backbone.type)
        if (
            self._flow_source_type in {"a2a", "a2a_noise"}
            and self._backbone_type == "transformer"
            and not self.model_spec.backbone.full_memory_attention
        ):
            self.model_spec = replace(
                self.model_spec,
                backbone=replace(
                    self.model_spec.backbone,
                    full_memory_attention=True,
                ),
            )
        self._condition_as_sequence = self._backbone_type == "transformer"
        self._condition_as_local = (
            self._backbone_type == "unet1d"
            and str(self.model_spec.backbone.conditioning_mode).lower() == "local"
        )
        if self._flow_source_type in {"a2a", "a2a_noise"}:
            if self._condition_as_local:
                raise NotImplementedError(
                    "A2A latent flow currently requires global conditioning."
                )
            if (
                self._backbone_type == "unet1d"
                and len(self.model_spec.backbone.down_dims) > 1
            ):
                raise NotImplementedError(
                    "A2A transports one latent token and therefore cannot use a "
                    "multi-scale UNet. Use Transformer, DiT, MLP, or configure a "
                    "single UNet down_dim."
                )
            self._actor_sequence_length = 1
            self._actor_action_dim = int(self.flow_source.latent_dim)
            self._actor_input_action_dim = int(self.flow_source.latent_dim)
        else:
            self._actor_sequence_length = self.action_sequence
            self._actor_action_dim = self.action_dim
            self._actor_input_action_dim = self.action_dim + (
                1 if self._flow_source_type == "legato" else 0
            )
        self._init_cached_pixel_feature_key("flow_matching")

        built_model, feature_dim = _build_model(
            self.model_spec,
            observation_space=observation_space,
            action_space=action_space,
            encoder_jit=jit,
            encoder_seed=seed,
            actor_sequence_length=self._actor_sequence_length,
            actor_action_dim=self._actor_action_dim,
            actor_input_action_dim=self._actor_input_action_dim,
        )
        self.actor_model = built_model.backbone_model
        self.encoder = built_model.encoder_model
        self.view_fusion = built_model.view_fusion_model
        self._trainable_encoder = (
            self.encoder is not None
            and self.model_spec.encoder_model is not None
            and bool(self.model_spec.encoder_model.trainable)
        )
        self._uses_plucker = bool(
            self.model_spec.encoder_model is not None
            and self.model_spec.encoder_model.use_plucker
        )
        if self.image_augmentation_type != "none" and not self._trainable_encoder:
            raise ValueError(
                "Image augmentation requires a trainable encoder and raw RGB inputs."
            )

        def augmentation_impl(obs_inputs, rng_key):
            return augment_campose_observation(
                obs_inputs,
                rng_key,
                require_raymap=self._uses_plucker,
            )

        self._augment_trainable_impl = (
            jax.jit(augmentation_impl) if self._jit_enabled else augmentation_impl
        )
        self.rgb_latent_size = max(
            0, feature_dim - self.low_dim_size - self.lang_feature_dim
        )

        self.rng_key, init_key = jax.random.split(self.rng_key)
        dummy_actions = jnp.zeros(
            (1, self._actor_sequence_length, self._actor_input_action_dim),
            dtype=jnp.float32,
        )
        dummy_timesteps = jnp.zeros((1,), dtype=jnp.float32)
        actor_feature_dim = int(getattr(self.actor_model, "feature_dim", feature_dim))
        if feature_dim > 0 and self._condition_as_sequence and not self.use_pixels:
            dummy_features = jnp.zeros(
                (1, self.time_dim, feature_dim), dtype=jnp.float32
            )
        else:
            dummy_features = (
                jnp.zeros((1, actor_feature_dim), dtype=jnp.float32)
                if actor_feature_dim > 0
                else None
            )
        if self._condition_as_local:
            dummy_local_features = jnp.zeros(
                (1, self._actor_sequence_length, self.low_dim_size),
                dtype=jnp.float32,
            )
            actor_params = self.actor_model.init(
                init_key,
                dummy_actions,
                dummy_timesteps,
                dummy_features,
                dummy_local_features,
            )
        else:
            actor_params = self.actor_model.init(
                init_key, dummy_actions, dummy_timesteps, dummy_features
            )
        if self._flow_source_type in {"a2a", "a2a_noise"}:
            self.history_encoder = TemporalActionEncoder(
                latent_dim=self.flow_source.latent_dim,
                hidden_dim=self.flow_source.hidden_dim,
                num_layers=self.flow_source.encoder_layers,
                kernel_size=self.flow_source.kernel_size,
            )
            self.future_encoder = TemporalActionEncoder(
                latent_dim=self.flow_source.latent_dim,
                hidden_dim=self.flow_source.hidden_dim,
                num_layers=self.flow_source.encoder_layers,
                kernel_size=self.flow_source.kernel_size,
            )
            self.action_decoder = ActionChunkDecoder(
                horizon=self.action_sequence,
                action_dim=self.action_dim,
                latent_dim=self.flow_source.latent_dim,
                hidden_dim=self.flow_source.hidden_dim,
                num_layers=self.flow_source.decoder_layers,
                dropout=self.flow_source.decoder_dropout,
            )
            self.rng_key, history_key, future_key, decoder_key = jax.random.split(
                self.rng_key, 4
            )
            source_params = {
                "history_encoder": self.history_encoder.init(
                    history_key,
                    jnp.zeros(
                        (
                            1,
                            self.flow_source.history_horizon,
                            self.action_dim,
                        ),
                        dtype=jnp.float32,
                    ),
                ),
                "future_encoder": self.future_encoder.init(
                    future_key,
                    jnp.zeros(
                        (1, self.action_sequence, self.action_dim),
                        dtype=jnp.float32,
                    ),
                ),
                "decoder": self.action_decoder.init(
                    decoder_key,
                    jnp.zeros((1, self.flow_source.latent_dim), dtype=jnp.float32),
                    train=False,
                ),
            }
            self.params = {"actor": actor_params, "source": source_params}
            if self._trainable_encoder:
                self.params["encoder"] = self.encoder.trainable_params
        else:
            self.params = (
                {"actor": actor_params, "encoder": self.encoder.trainable_params}
                if self._trainable_encoder
                else actor_params
            )
        self.ema_params = self.params if self.use_ema else None

        self._ema_decay = ema_decay
        self._ema_decay_schedule = str(ema_decay_schedule).lower()
        if self._ema_decay_schedule not in {"constant", "diffusers"}:
            raise ValueError(
                "ema_decay_schedule must be 'constant' or 'diffusers'; "
                f"got {ema_decay_schedule!r}."
            )
        self._ema_min_decay = 0.0
        self._ema_update_after_step = 0
        self._ema_use_warmup = False
        self._ema_inv_gamma = 1.0
        self._ema_power = 0.75
        self._ema_optimization_step = 0

        learning_rate = lr
        if adaptive_lr:
            lr_schedule = str(lr_schedule).lower()
            if int(num_train_steps) <= 0:
                raise ValueError(
                    "Flow Matching adaptive_lr requires method.num_train_steps > 0. "
                    "Set the optimizer horizon explicitly or disable adaptive_lr."
                )
            if lr_schedule == "cosine":
                learning_rate = self.optax.cosine_decay_schedule(
                    init_value=lr,
                    decay_steps=self.num_train_steps,
                    alpha=0.0,
                )
            elif lr_schedule in {"warmup_cosine", "warmup-cosine"}:
                if int(num_train_steps) <= 100:
                    raise ValueError(
                        "Flow Matching warmup_cosine requires method.num_train_steps "
                        "> 100 (the fixed warmup length)."
                    )
                learning_rate = self.optax.warmup_cosine_decay_schedule(
                    init_value=0.0,
                    peak_value=lr,
                    warmup_steps=100,
                    decay_steps=self.num_train_steps,
                    end_value=0.0,
                )
            else:
                raise ValueError(f"Unknown flow matching lr_schedule '{lr_schedule}'.")
        transforms = []
        if actor_grad_clip is not None:
            transforms.append(self.optax.clip_by_global_norm(float(actor_grad_clip)))
        transforms.append(self.optax.adamw(learning_rate, weight_decay=weight_decay))
        if (
            self._backbone_type == "dit"
            and self.actor_model.timestep_embedding_type == "fourier"
        ):
            frozen_frequency_mask = jax.tree_util.tree_map_with_path(
                lambda path, _: any(
                    isinstance(key, jax.tree_util.DictKey)
                    and key.key == "fourier_frequencies"
                    for key in path
                ),
                self.params,
            )
            transforms.append(
                self.optax.masked(self.optax.set_to_zero(), frozen_frequency_mask)
            )
        self.optimizer = self.optax.chain(*transforms)
        self.opt_state = self.optimizer.init(self.params)

        if self._flow_source_type == "gaussian":
            update_fn = self._build_update_fn()
            update_many_fn = self._build_update_many_fn(update_fn)
            sample_fn = self._build_sample_fn()
            sample_from_noise_fn = self._build_sample_from_noise_fn()
        elif self._flow_source_type in {"a2a", "a2a_noise"}:
            update_fn = self._build_a2a_update_fn()
            update_many_fn = self._build_a2a_update_many_fn(update_fn)
            sample_fn = self._build_a2a_sample_fn()
            sample_from_noise_fn = None
        else:
            update_fn = self._build_legato_update_fn()
            update_many_fn = self._build_legato_update_many_fn(update_fn)
            sample_fn = self._build_legato_sample_fn()
            sample_from_noise_fn = self._build_legato_sample_from_noise_fn()
        if self._jit_enabled:
            update_fn = jax.jit(update_fn)
            sample_fn = jax.jit(sample_fn)
            if update_many_fn is not None:
                update_many_fn = jax.jit(update_many_fn)
            if sample_from_noise_fn is not None:
                sample_from_noise_fn = jax.jit(sample_from_noise_fn)
        self._update_impl = update_fn
        self._update_many_impl = update_many_fn
        self._sample_impl = sample_fn
        self._sample_from_noise_impl = sample_from_noise_fn

        # Optional seed-aligned eval noise. When `_active_eval_seeds` is set
        # (list of env seeds in observation-batch-row order), `act` draws the
        # flow-matching initial noise deterministically from
        # (env_seed, per-seed episode step) instead of advancing the shared
        # rng_key batch-wise. This makes vectorized and serial evaluation
        # reproduce each other per seed. Disabled by default (None) so training
        # and ordinary eval behaviour is unchanged.
        self._eval_noise_base_key = jax.random.PRNGKey(0)
        self._eval_step_by_seed = {}
        self._active_eval_seeds = None

        self._history_storage_horizon = (
            self.flow_source.history_horizon + self.action_execution_start
        )
        self._train_action_history = jnp.zeros(
            (
                self.num_train_envs,
                self._history_storage_horizon,
                self.action_dim,
            ),
            dtype=jnp.float32,
        )
        self._train_action_history_valid = jnp.zeros(
            (self.num_train_envs, self._history_storage_horizon), dtype=jnp.bool_
        )
        self._eval_action_history = jnp.zeros(
            (
                self.num_eval_envs,
                self._history_storage_horizon,
                self.action_dim,
            ),
            dtype=jnp.float32,
        )
        self._eval_action_history_valid = jnp.zeros(
            (self.num_eval_envs, self._history_storage_horizon), dtype=jnp.bool_
        )
        self._train_previous_chunk = jnp.zeros(
            (self.num_train_envs, self.action_sequence, self.action_dim),
            dtype=jnp.float32,
        )
        self._train_previous_chunk_valid = jnp.zeros(
            (self.num_train_envs, self.action_sequence), dtype=jnp.bool_
        )
        self._eval_previous_chunk = jnp.zeros(
            (self.num_eval_envs, self.action_sequence, self.action_dim),
            dtype=jnp.float32,
        )
        self._eval_previous_chunk_valid = jnp.zeros(
            (self.num_eval_envs, self.action_sequence), dtype=jnp.bool_
        )
        self._train_last_issued_action = jnp.zeros(
            (self.num_train_envs, self.action_dim), dtype=jnp.float32
        )
        self._train_last_issued_valid = jnp.zeros(
            (self.num_train_envs,), dtype=jnp.bool_
        )
        self._eval_last_issued_action = jnp.zeros(
            (self.num_eval_envs, self.action_dim), dtype=jnp.float32
        )
        self._eval_last_issued_valid = jnp.zeros((self.num_eval_envs,), dtype=jnp.bool_)
        self._eval_slot_by_global_id: dict[int, int] = {}
        self._rollout_metric_sums: dict[str, float] = {}
        self._rollout_metric_counts: dict[str, int] = {}

    def _ema_decay_value(self, optimization_step):
        if self._ema_decay_schedule == "constant":
            return jnp.asarray(self._ema_decay, dtype=jnp.float32)
        step = jnp.maximum(0, optimization_step - self._ema_update_after_step - 1)
        if self._ema_use_warmup:
            decay = 1.0 - (1.0 + step / self._ema_inv_gamma) ** (-self._ema_power)
        else:
            decay = (1.0 + step) / (10.0 + step)
        decay = jnp.where(step <= 0, 0.0, decay)
        decay = jnp.minimum(decay, self._ema_decay)
        return jnp.maximum(decay, self._ema_min_decay)

    # ------------------------------------------------------------------
    # Language Conditioning
    # ------------------------------------------------------------------

    def _lang_token_rows(self, batch_or_obs: dict):
        return lang_token_rows(
            batch_or_obs,
            context="Language-conditioned flow matching",
        )

    def _encode_lang_tokens(self, tokens):
        return tokens_to_feature_jax(tokens, feature_dim=self.lang_feature_dim)

    def _extract_lang_features(self, batch_or_obs: dict):
        if not self.use_lang_cond:
            return None
        if "lang_features" in batch_or_obs:
            features = self._as_jax_array(
                lang_feature_rows(
                    batch_or_obs,
                    context="Language-conditioned flow matching",
                ),
                self.jnp.float32,
            )
            if features.shape[-1] != self.lang_feature_dim:
                raise ValueError(
                    "Language-conditioned flow matching expected lang_features "
                    f"with final dimension {self.lang_feature_dim}, got "
                    f"{features.shape[-1]}."
                )
            return features
        tokens = self._as_jax_array(
            self._lang_token_rows(batch_or_obs),
            self.jnp.float32,
        )
        return self._as_jax_array(
            self._encode_lang_tokens(tokens),
            self.jnp.float32,
        )

    def _prepare_obs_features(self, batch_or_obs: dict):
        if self._condition_as_local:
            local_features = self._extract_low_dim_batch(batch_or_obs)
            if local_features is None:
                raise ValueError(
                    "Local UNet conditioning requires low_dim_state observations."
                )
            local_features = self.jnp.repeat(
                local_features[:, None, :], self.action_sequence, axis=1
            )

            global_parts = []
            fused_view_feats = self._extract_cached_pixel_features(batch_or_obs)
            metrics = {}
            if self.use_pixels and fused_view_feats is None:
                rgb_obs, pixel_metrics = self._extract_rgb_obs(batch_or_obs)
                raymap_obs = self._extract_raymap_obs(batch_or_obs)
                camera_intrinsic_obs, camera_c2w_obs = self._extract_camera_param_obs(
                    batch_or_obs
                )
                metrics.update(pixel_metrics)
                fused_view_feats = self._fuse_multi_view(
                    self._encode_pixels(
                        rgb_obs,
                        raymap_obs=raymap_obs,
                        camera_intrinsic_obs=camera_intrinsic_obs,
                        camera_c2w_obs=camera_c2w_obs,
                    )
                )
            if fused_view_feats is not None:
                global_parts.append(fused_view_feats)
            if self.use_lang_cond:
                global_parts.append(self._extract_lang_features(batch_or_obs))
            global_features = (
                self.jnp.concatenate(global_parts, axis=-1)
                if len(global_parts) > 1
                else (global_parts[0] if global_parts else None)
            )
            return (global_features, local_features), metrics
        if self._condition_as_sequence and not self.use_pixels:
            if "low_dim_state" not in batch_or_obs:
                raise ValueError(
                    "Transformer flow matching expects low_dim_state observations."
                )
            obs_features = self._as_jax_array(
                batch_or_obs["low_dim_state"], self.jnp.float32
            )
            if self.use_lang_cond:
                lang_features = self._extract_lang_features(batch_or_obs)
                lang_features = self.jnp.repeat(
                    lang_features[:, None, :],
                    obs_features.shape[1],
                    axis=1,
                )
                obs_features = self.jnp.concatenate(
                    [obs_features, lang_features],
                    axis=-1,
                )
            metrics = {}
            if self.logging:
                metrics["low_dim_state"] = np.asarray(
                    self.jax.device_get(obs_features[0, -1])
                )
            return obs_features, metrics
        obs_features, metrics = super()._prepare_obs_features(batch_or_obs)
        if self.use_lang_cond:
            obs_features = self.jnp.concatenate(
                [obs_features, self._extract_lang_features(batch_or_obs)],
                axis=-1,
            )
        return obs_features, metrics

    def _actor_params(self, params):
        if self._trainable_encoder or self._flow_source_type in {"a2a", "a2a_noise"}:
            return params["actor"]
        return params

    def _apply_actor(
        self,
        params,
        actions,
        timesteps,
        obs_features,
        *,
        train: bool = False,
        dropout_key=None,
    ):
        actor_params = self._actor_params(params)
        if self._condition_as_local:
            global_features, local_features = obs_features
            return self.actor_model.apply(
                actor_params,
                actions,
                timesteps,
                global_features,
                local_features,
            )
        if self._condition_as_sequence:
            kwargs = {"train": train}
            if dropout_key is not None:
                kwargs["rngs"] = {"dropout": dropout_key}
            return self.actor_model.apply(
                actor_params,
                actions,
                timesteps,
                obs_features,
                **kwargs,
            )
        if self._backbone_type == "dit":
            kwargs = {"train": train}
            if dropout_key is not None:
                kwargs["rngs"] = {"dropout": dropout_key}
            return self.actor_model.apply(
                actor_params,
                actions,
                timesteps,
                obs_features,
                **kwargs,
            )
        return self.actor_model.apply(actor_params, actions, timesteps, obs_features)

    def _prepare_trainable_obs_inputs(self, batch_or_obs: dict):
        inputs = {}
        low_dim_obs = self._extract_low_dim_batch(batch_or_obs)
        if low_dim_obs is not None:
            inputs["low_dim"] = low_dim_obs
        if self.use_pixels:
            if self._has_cached_pixel_features(batch_or_obs):
                raise ValueError(
                    "Trainable JAX flow-matching encoder requires raw RGB "
                    "observations; disable replay.cache_frozen_image_features."
                )
            rgb_obs, _ = self._extract_rgb_obs(batch_or_obs)
            inputs["rgb"] = rgb_obs
            raymap_obs = self._extract_raymap_obs(batch_or_obs)
            if raymap_obs is not None:
                inputs["raymap"] = raymap_obs
            camera_intrinsic_obs, camera_c2w_obs = self._extract_camera_param_obs(
                batch_or_obs
            )
            if camera_intrinsic_obs is not None:
                inputs["camera_intrinsic"] = camera_intrinsic_obs
                inputs["camera_c2w"] = camera_c2w_obs
        if self.use_lang_cond:
            inputs["lang"] = self._extract_lang_features(batch_or_obs)
        return inputs

    def _augment_trainable_obs_inputs(self, obs_inputs: dict, rng_key) -> dict:
        if self.image_augmentation_type == "none":
            return obs_inputs
        return self._augment_trainable_impl(obs_inputs, rng_key)

    def _features_from_inputs(self, params, obs_inputs):
        if not self._trainable_encoder:
            return obs_inputs
        if self._condition_as_local:
            if "low_dim" not in obs_inputs:
                raise ValueError(
                    "Local UNet conditioning requires low_dim_state observations."
                )
            local_features = self.jnp.repeat(
                obs_inputs["low_dim"][:, None, :],
                self.action_sequence,
                axis=1,
            )
            global_parts = []
            if "rgb" in obs_inputs:
                rgb_feats = self.encoder.apply_trainable(
                    params["encoder"],
                    obs_inputs["rgb"],
                    raymap_obs=obs_inputs.get("raymap", None),
                    camera_intrinsic_obs=obs_inputs.get("camera_intrinsic", None),
                    camera_c2w_obs=obs_inputs.get("camera_c2w", None),
                )
                global_parts.append(self._fuse_multi_view(rgb_feats))
            if "lang" in obs_inputs:
                global_parts.append(obs_inputs["lang"])
            global_features = (
                self.jnp.concatenate(global_parts, axis=-1)
                if len(global_parts) > 1
                else (global_parts[0] if global_parts else None)
            )
            return global_features, local_features
        features = []
        if "low_dim" in obs_inputs:
            features.append(obs_inputs["low_dim"])
        if "rgb" in obs_inputs:
            rgb_feats = self.encoder.apply_trainable(
                params["encoder"],
                obs_inputs["rgb"],
                raymap_obs=obs_inputs.get("raymap", None),
                camera_intrinsic_obs=obs_inputs.get("camera_intrinsic", None),
                camera_c2w_obs=obs_inputs.get("camera_c2w", None),
            )
            features.append(self._fuse_multi_view(rgb_feats))
        if "lang" in obs_inputs:
            features.append(obs_inputs["lang"])
        if not features:
            raise ValueError("Flow matching requires at least one observation feature.")
        if len(features) == 1:
            return features[0]
        return jnp.concatenate(features, axis=-1)

    def _build_update_fn(self):
        optimizer = self.optimizer
        optax = self.optax
        use_ema = self.use_ema
        time_scale = self._time_scale
        train_time_beta_params = self._train_time_beta_params
        horizon_dropout_lengths = self._horizon_dropout_lengths
        horizon_dropout_probs = self._horizon_dropout_probs
        horizon_loss_weights = self._horizon_loss_weights

        def update_fn(
            params,
            opt_state,
            rng_key,
            obs_inputs,
            actions,
            loss_coeff,
            action_pad_mask,
            ema_params,
            ema_optimization_step,
        ):
            source_key, time_key, dropout_key, next_key = jax.random.split(rng_key, 4)
            actor_dropout_key = jax.random.fold_in(dropout_key, 1)
            x1 = jax.random.normal(source_key, shape=actions.shape)
            if train_time_beta_params is None:
                t = jax.random.uniform(
                    time_key,
                    shape=(actions.shape[0],),
                    minval=0.0,
                    maxval=1.0,
                )
            else:
                alpha, beta = train_time_beta_params
                t = jax.random.beta(
                    time_key,
                    alpha,
                    beta,
                    shape=(actions.shape[0],),
                )
                t = jnp.clip(t, 1e-5, 1.0 - 1e-5)
            xt, target_velocity = _rectified_flow_training_pair(actions, x1, t)

            effective_action_pad_mask = action_pad_mask
            if horizon_dropout_lengths is not None:
                length_indices = jax.random.choice(
                    dropout_key,
                    horizon_dropout_lengths.shape[0],
                    shape=(actions.shape[0],),
                    p=horizon_dropout_probs,
                )
                sampled_lengths = horizon_dropout_lengths[length_indices]
                token_positions = jnp.arange(actions.shape[1], dtype=jnp.int32)
                dropout_mask = token_positions[None, :] >= sampled_lengths[:, None]
                if effective_action_pad_mask is None:
                    effective_action_pad_mask = dropout_mask
                else:
                    effective_action_pad_mask = jnp.logical_or(
                        effective_action_pad_mask, dropout_mask
                    )

            if effective_action_pad_mask is not None:
                valid_mask = jnp.logical_not(effective_action_pad_mask)
                # Mask the model input as well as the target. Feeding the true
                # padded/dropped action here leaks future context through UNet
                # convolutions and transformer attention.
                xt = jnp.where(valid_mask[..., None], xt, 0.0)
                target_velocity = jnp.where(
                    valid_mask[..., None],
                    target_velocity,
                    0.0,
                )

            def loss_fn(current_params):
                obs_features = self._features_from_inputs(current_params, obs_inputs)
                velocity_pred = self._apply_actor(
                    current_params,
                    xt,
                    t * time_scale,
                    obs_features,
                    train=True,
                    dropout_key=actor_dropout_key,
                )
                per_token_mse = jnp.square(velocity_pred - target_velocity)
                reduce_dims = tuple(range(1, per_token_mse.ndim))
                if horizon_loss_weights is not None:
                    action_reduce_dims = tuple(range(2, per_token_mse.ndim))
                    per_step_mse = per_token_mse.mean(axis=action_reduce_dims)
                    weights = horizon_loss_weights
                    if effective_action_pad_mask is None:
                        mse_loss = (per_step_mse * weights[None, :]).sum(axis=1)
                    else:
                        valid_mask = jnp.logical_not(effective_action_pad_mask).astype(
                            per_step_mse.dtype
                        )
                        weighted_mask = valid_mask * weights[None, :]
                        denom = jnp.clip(weighted_mask.sum(axis=1), min=1e-8)
                        mse_loss = (per_step_mse * weighted_mask).sum(axis=1) / denom
                elif effective_action_pad_mask is None:
                    mse_loss = per_token_mse.mean(axis=reduce_dims)
                else:
                    valid_mask = jnp.logical_not(effective_action_pad_mask).astype(
                        per_token_mse.dtype
                    )
                    while valid_mask.ndim < per_token_mse.ndim:
                        valid_mask = valid_mask[..., None]
                    valid_mask = jnp.broadcast_to(valid_mask, per_token_mse.shape)
                    masked_loss = per_token_mse * valid_mask
                    denom = jnp.clip(valid_mask.sum(axis=reduce_dims), min=1.0)
                    mse_loss = masked_loss.sum(axis=reduce_dims) / denom
                total_loss = (mse_loss * loss_coeff).mean()
                return total_loss, mse_loss

            (loss, mse_loss), grads = jax.value_and_grad(
                loss_fn,
                has_aux=True,
            )(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            new_ema_params = ema_params
            new_ema_step = ema_optimization_step
            if use_ema:
                new_ema_step = ema_optimization_step + 1
                decay = self._ema_decay_value(new_ema_step)
                one_minus_decay = 1.0 - decay
                new_ema_params = jax.tree.map(
                    lambda ema, param: ema - one_minus_decay * (ema - param),
                    ema_params,
                    new_params,
                )

            new_pri = jnp.sqrt(mse_loss + 1e-10)
            max_pri = jnp.max(new_pri)
            normalized_pri = new_pri / jnp.where(max_pri > 0, max_pri, 1.0)
            return (
                new_params,
                new_opt_state,
                next_key,
                loss,
                normalized_pri,
                new_ema_params,
                new_ema_step,
            )

        return update_fn

    def _masked_action_error(self, error, pad_mask):
        """Reduce an action-shaped error to one value per batch row."""
        reduce_dims = tuple(range(1, error.ndim))
        if pad_mask is None:
            return error.mean(axis=reduce_dims)
        valid = jnp.logical_not(pad_mask).astype(error.dtype)
        while valid.ndim < error.ndim:
            valid = valid[..., None]
        valid = jnp.broadcast_to(valid, error.shape)
        return (error * valid).sum(axis=reduce_dims) / jnp.clip(
            valid.sum(axis=reduce_dims), min=1.0
        )

    def _masked_actions(self, actions, pad_mask):
        if pad_mask is None:
            return actions
        return jnp.where(jnp.logical_not(pad_mask)[..., None], actions, 0.0)

    def _a2a_encode_history(self, params, history_actions, history_pad_mask=None):
        return self.history_encoder.apply(
            params["source"]["history_encoder"],
            history_actions,
            history_pad_mask,
        )

    def _a2a_encode_future(self, params, future_actions, action_pad_mask=None):
        return self.future_encoder.apply(
            params["source"]["future_encoder"],
            future_actions,
            action_pad_mask,
        )

    def _a2a_decode(self, params, latent, *, train: bool = False, dropout_key=None):
        kwargs = {"train": train}
        if dropout_key is not None:
            kwargs["rngs"] = {"dropout": dropout_key}
        return self.action_decoder.apply(params["source"]["decoder"], latent, **kwargs)

    def _perturb_a2a_history(self, history, history_pad_mask, key, std):
        history = self._masked_actions(history, history_pad_mask)
        if std <= 0.0:
            return history
        continuous_dims = self.action_dim - self.flow_source.noise_exclude_last_n
        dim_mask = jnp.arange(self.action_dim) < continuous_dims
        valid = (
            jnp.ones(history.shape[:-1], dtype=jnp.bool_)
            if history_pad_mask is None
            else jnp.logical_not(history_pad_mask)
        )
        noise_mask = valid[..., None] & dim_mask
        if key.ndim == 2:
            noise = jax.vmap(
                lambda row_key: jax.random.normal(
                    row_key, history.shape[1:], dtype=history.dtype
                )
            )(key)
        else:
            noise = jax.random.normal(key, history.shape, dtype=history.dtype)
        return history + jnp.where(noise_mask, std * noise, 0.0)

    def _integrate_a2a_latent(self, params, latent, obs_features, num_steps):
        batch_size = latent.shape[0]
        delta_t = jnp.asarray(1.0 / int(num_steps), dtype=latent.dtype)

        def body_fn(index, current_latent):
            tau = 1.0 - index.astype(jnp.float32) / float(num_steps)
            t_batch = jnp.full((batch_size,), tau, dtype=jnp.float32)
            velocity = self._apply_actor(
                params,
                current_latent,
                t_batch * self._time_scale,
                obs_features,
            )
            velocity = jnp.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)
            return jnp.nan_to_num(current_latent + delta_t * velocity, nan=0.0)

        return jax.lax.fori_loop(0, int(num_steps), body_fn, latent)

    def _build_a2a_update_fn(self):
        optimizer = self.optimizer
        optax = self.optax
        use_ema = self.use_ema
        source_spec = self.flow_source
        train_time_beta_params = self._train_time_beta_params

        def update_fn(
            params,
            opt_state,
            rng_key,
            obs_inputs,
            history_actions,
            history_pad_mask,
            actions,
            loss_coeff,
            action_pad_mask,
            ema_params,
            ema_optimization_step,
        ):
            (
                history_noise_key,
                time_key,
                dropout_key,
                next_key,
            ) = jax.random.split(rng_key, 4)
            history_actions_noisy = self._perturb_a2a_history(
                history_actions,
                history_pad_mask,
                history_noise_key,
                source_spec.train_history_noise_std,
            )
            future_actions = self._masked_actions(actions, action_pad_mask)
            if train_time_beta_params is None:
                tau = jax.random.uniform(
                    time_key, shape=(actions.shape[0],), minval=0.0, maxval=1.0
                )
            else:
                alpha, beta = train_time_beta_params
                tau = jax.random.beta(time_key, alpha, beta, shape=(actions.shape[0],))
                tau = jnp.clip(tau, 1e-5, 1.0 - 1e-5)
            actor_dropout_key = jax.random.fold_in(dropout_key, 1)
            reconstruction_dropout_key = jax.random.fold_in(dropout_key, 2)
            consistency_dropout_key = jax.random.fold_in(dropout_key, 3)

            def loss_fn(current_params):
                obs_features = self._features_from_inputs(current_params, obs_inputs)
                z_history = self._a2a_encode_history(
                    current_params,
                    history_actions_noisy,
                    history_pad_mask,
                )[:, None, :]
                z_future = self._a2a_encode_future(
                    current_params,
                    future_actions,
                    action_pad_mask,
                )[:, None, :]
                pair = self.flow_source_module.build_training_pair(
                    history_noise_key,
                    z_future,
                    tau,
                    source=z_history,
                )
                velocity_pred = self._apply_actor(
                    current_params,
                    pair.sample,
                    tau * self._time_scale,
                    obs_features,
                    train=True,
                    dropout_key=actor_dropout_key,
                )
                fm_per_example = jnp.square(velocity_pred - pair.target_velocity).mean(
                    axis=(1, 2)
                )

                future_reconstruction = self._a2a_decode(
                    current_params,
                    z_future,
                    train=True,
                    dropout_key=reconstruction_dropout_key,
                )
                autoencoder_per_example = self._masked_action_error(
                    jnp.abs(future_reconstruction - actions), action_pad_mask
                )

                generated_latent = self._integrate_a2a_latent(
                    current_params,
                    z_history,
                    obs_features,
                    source_spec.consistency_steps,
                )
                consistency_target = (
                    jax.lax.stop_gradient(z_future)
                    if source_spec.consistency_stop_gradient_target
                    else z_future
                )
                latent_error = generated_latent - consistency_target
                if source_spec.consistency_loss_type == "mse":
                    latent_error = jnp.square(latent_error)
                else:
                    latent_error = jnp.abs(latent_error)
                latent_consistency_per_example = latent_error.mean(axis=(1, 2))
                generated_actions = self._a2a_decode(
                    current_params,
                    generated_latent,
                    train=True,
                    dropout_key=consistency_dropout_key,
                )
                action_consistency_per_example = self._masked_action_error(
                    jnp.abs(generated_actions - actions), action_pad_mask
                )
                consistency_per_example = (
                    latent_consistency_per_example
                    + source_spec.consistency_action_loss_weight
                    * action_consistency_per_example
                )
                total_per_example = (
                    source_spec.fm_loss_weight * fm_per_example
                    + source_spec.autoencoder_loss_weight * autoencoder_per_example
                    + source_spec.consistency_loss_weight * consistency_per_example
                )
                if action_pad_mask is not None:
                    example_valid = jnp.logical_not(action_pad_mask).any(axis=1)
                    total_per_example = jnp.where(example_valid, total_per_example, 0.0)
                total_loss = (total_per_example * loss_coeff).mean()
                aux = (
                    total_per_example,
                    fm_per_example.mean(),
                    autoencoder_per_example.mean(),
                    latent_consistency_per_example.mean(),
                    action_consistency_per_example.mean(),
                )
                return total_loss, aux

            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            (
                total_per_example,
                fm_loss,
                autoencoder_loss,
                latent_consistency_loss,
                action_consistency_loss,
            ) = aux
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            new_ema_params = ema_params
            new_ema_step = ema_optimization_step
            if use_ema:
                new_ema_step = ema_optimization_step + 1
                decay = self._ema_decay_value(new_ema_step)
                new_ema_params = jax.tree.map(
                    lambda ema, param: ema - (1.0 - decay) * (ema - param),
                    ema_params,
                    new_params,
                )
            priorities = jnp.sqrt(jnp.maximum(total_per_example, 0.0) + 1e-10)
            priorities = priorities / jnp.maximum(jnp.max(priorities), 1e-8)
            diagnostics = {
                "fm_loss": fm_loss,
                "a2a_autoencoder_loss": autoencoder_loss,
                "a2a_consistency_latent_loss": latent_consistency_loss,
                "a2a_consistency_action_loss": action_consistency_loss,
            }
            return (
                new_params,
                new_opt_state,
                next_key,
                loss,
                priorities,
                new_ema_params,
                new_ema_step,
                diagnostics,
            )

        return update_fn

    def _sample_legato_schedule(
        self,
        delay_key,
        ramp_key,
        profile_key,
        strength_key,
        no_guidance_key,
        batch_size,
    ):
        source_spec = self.flow_source
        delay = jax.random.randint(
            delay_key,
            (batch_size,),
            minval=source_spec.delay_min_steps,
            maxval=source_spec.delay_max_steps + 1,
        )
        ramp = jax.random.randint(
            ramp_key,
            (batch_size,),
            minval=source_spec.ramp_min_steps,
            maxval=source_spec.ramp_max_steps + 1,
        )
        candidates = jnp.stack(
            [
                legato_schedule(
                    self.action_sequence,
                    delay,
                    ramp,
                    start=self.action_execution_start,
                    kind=profile,
                    dtype=jnp.float32,
                )
                for profile in source_spec.schedule_profiles
            ],
            axis=0,
        )
        if len(source_spec.schedule_profiles) == 1:
            omega = candidates[0]
        else:
            profile_index = jax.random.randint(
                profile_key,
                (batch_size,),
                minval=0,
                maxval=len(source_spec.schedule_profiles),
            )
            selector = jax.nn.one_hot(
                profile_index, len(source_spec.schedule_profiles)
            ).T[:, :, None, None]
            omega = (candidates * selector).sum(axis=0)
        strength = jax.random.uniform(
            strength_key,
            (batch_size, 1, 1),
            minval=source_spec.strength_min,
            maxval=source_spec.strength_max,
        )
        omega = omega * strength
        no_guidance = jax.random.bernoulli(
            no_guidance_key,
            p=source_spec.no_guidance_probability,
            shape=(batch_size, 1, 1),
        )
        return jnp.where(no_guidance, 0.0, omega)

    def _build_legato_update_fn(self):
        optimizer = self.optimizer
        optax = self.optax
        use_ema = self.use_ema
        train_time_beta_params = self._train_time_beta_params
        time_scale = self._time_scale
        dt = 1.0 / self.num_flow_steps

        def update_fn(
            params,
            opt_state,
            rng_key,
            obs_inputs,
            actions,
            loss_coeff,
            action_pad_mask,
            ema_params,
            ema_optimization_step,
        ):
            keys = jax.random.split(rng_key, 9)
            (
                source_key,
                time_key,
                delay_key,
                ramp_key,
                profile_key,
                strength_key,
                no_guidance_key,
                dropout_key,
                next_key,
            ) = keys
            if train_time_beta_params is None:
                tau = jax.random.uniform(
                    time_key, shape=(actions.shape[0],), minval=0.0, maxval=1.0
                )
            else:
                alpha, beta = train_time_beta_params
                tau = jax.random.beta(time_key, alpha, beta, shape=(actions.shape[0],))
                tau = jnp.clip(tau, 1e-5, 1.0 - 1e-5)
            omega = self._sample_legato_schedule(
                delay_key,
                ramp_key,
                profile_key,
                strength_key,
                no_guidance_key,
                actions.shape[0],
            )
            if action_pad_mask is not None:
                omega = jnp.where(
                    jnp.logical_not(action_pad_mask)[..., None], omega, 0.0
                )
            pair = self.flow_source_module.build_training_pair(
                source_key,
                actions,
                tau,
                omega=omega,
                dt=dt,
            )
            target_velocity = pair.target_velocity
            model_input = jnp.concatenate([pair.sample, pair.schedule_channel], axis=-1)
            if action_pad_mask is not None:
                valid = jnp.logical_not(action_pad_mask)[..., None]
                model_input = jnp.where(valid, model_input, 0.0)
                target_velocity = jnp.where(valid, target_velocity, 0.0)
            actor_dropout_key = jax.random.fold_in(dropout_key, 1)

            def loss_fn(current_params):
                obs_features = self._features_from_inputs(current_params, obs_inputs)
                velocity_pred = self._apply_actor(
                    current_params,
                    model_input,
                    tau * time_scale,
                    obs_features,
                    train=True,
                    dropout_key=actor_dropout_key,
                )
                mse_per_example = self._masked_action_error(
                    jnp.square(velocity_pred - target_velocity), action_pad_mask
                )
                loss = (mse_per_example * loss_coeff).mean()
                return loss, mse_per_example

            (loss, mse_per_example), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params
            )
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            new_ema_params = ema_params
            new_ema_step = ema_optimization_step
            if use_ema:
                new_ema_step = ema_optimization_step + 1
                decay = self._ema_decay_value(new_ema_step)
                new_ema_params = jax.tree.map(
                    lambda ema, param: ema - (1.0 - decay) * (ema - param),
                    ema_params,
                    new_params,
                )
            priorities = jnp.sqrt(mse_per_example + 1e-10)
            priorities = priorities / jnp.maximum(jnp.max(priorities), 1e-8)
            diagnostics = {
                "fm_loss": mse_per_example.mean(),
                "legato_guidance_mean": omega.mean(),
            }
            return (
                new_params,
                new_opt_state,
                next_key,
                loss,
                priorities,
                new_ema_params,
                new_ema_step,
                diagnostics,
            )

        return update_fn

    def _build_update_many_fn(self, update_fn):
        def update_many_fn(
            params,
            opt_state,
            rng_key,
            obs_inputs,
            actions,
            loss_coeff,
            action_pad_mask,
            ema_params,
            ema_optimization_step,
        ):
            def body_fn(carry, xs):
                (
                    current_params,
                    current_opt_state,
                    current_key,
                    current_ema,
                    current_ema_step,
                ) = carry
                if action_pad_mask is None:
                    step_obs_inputs, step_actions, step_loss_coeff = xs
                    step_action_pad_mask = None
                else:
                    (
                        step_obs_inputs,
                        step_actions,
                        step_loss_coeff,
                        step_action_pad_mask,
                    ) = xs
                (
                    next_params,
                    next_opt_state,
                    next_key,
                    loss,
                    priority,
                    next_ema,
                    next_ema_step,
                ) = update_fn(
                    current_params,
                    current_opt_state,
                    current_key,
                    step_obs_inputs,
                    step_actions,
                    step_loss_coeff,
                    step_action_pad_mask,
                    current_ema,
                    current_ema_step,
                )
                next_carry = (
                    next_params,
                    next_opt_state,
                    next_key,
                    next_ema,
                    next_ema_step,
                )
                return next_carry, (loss, priority)

            xs = (obs_inputs, actions, loss_coeff)
            if action_pad_mask is not None:
                xs = (*xs, action_pad_mask)
            (
                (
                    new_params,
                    new_opt_state,
                    new_key,
                    new_ema,
                    new_ema_step,
                ),
                (losses, priorities),
            ) = jax.lax.scan(
                body_fn,
                (params, opt_state, rng_key, ema_params, ema_optimization_step),
                xs,
            )
            return (
                new_params,
                new_opt_state,
                new_key,
                losses[-1],
                priorities[-1],
                new_ema,
                new_ema_step,
            )

        return update_many_fn

    def _build_a2a_update_many_fn(self, update_fn):
        def update_many_fn(
            params,
            opt_state,
            rng_key,
            obs_inputs,
            history_actions,
            history_pad_mask,
            actions,
            loss_coeff,
            action_pad_mask,
            ema_params,
            ema_optimization_step,
        ):
            def body_fn(carry, xs):
                current_params, current_opt, current_key, current_ema, ema_step = carry
                if action_pad_mask is None:
                    (
                        step_obs,
                        step_history,
                        step_history_mask,
                        step_actions,
                        step_coeff,
                    ) = xs
                    step_action_mask = None
                else:
                    (
                        step_obs,
                        step_history,
                        step_history_mask,
                        step_actions,
                        step_coeff,
                        step_action_mask,
                    ) = xs
                result = update_fn(
                    current_params,
                    current_opt,
                    current_key,
                    step_obs,
                    step_history,
                    step_history_mask,
                    step_actions,
                    step_coeff,
                    step_action_mask,
                    current_ema,
                    ema_step,
                )
                (
                    next_params,
                    next_opt,
                    next_key,
                    loss,
                    priority,
                    next_ema,
                    next_ema_step,
                    diagnostics,
                ) = result
                return (
                    next_params,
                    next_opt,
                    next_key,
                    next_ema,
                    next_ema_step,
                ), (loss, priority, diagnostics)

            xs = (
                obs_inputs,
                history_actions,
                history_pad_mask,
                actions,
                loss_coeff,
            )
            if action_pad_mask is not None:
                xs = (*xs, action_pad_mask)
            carry, outputs = jax.lax.scan(
                body_fn,
                (params, opt_state, rng_key, ema_params, ema_optimization_step),
                xs,
            )
            new_params, new_opt, new_key, new_ema, new_ema_step = carry
            losses, priorities, diagnostics = outputs
            return (
                new_params,
                new_opt,
                new_key,
                losses[-1],
                priorities[-1],
                new_ema,
                new_ema_step,
                jax.tree.map(lambda value: value[-1], diagnostics),
            )

        return update_many_fn

    def _build_legato_update_many_fn(self, update_fn):
        def update_many_fn(
            params,
            opt_state,
            rng_key,
            obs_inputs,
            actions,
            loss_coeff,
            action_pad_mask,
            ema_params,
            ema_optimization_step,
        ):
            def body_fn(carry, xs):
                current_params, current_opt, current_key, current_ema, ema_step = carry
                if action_pad_mask is None:
                    step_obs, step_actions, step_coeff = xs
                    step_action_mask = None
                else:
                    step_obs, step_actions, step_coeff, step_action_mask = xs
                result = update_fn(
                    current_params,
                    current_opt,
                    current_key,
                    step_obs,
                    step_actions,
                    step_coeff,
                    step_action_mask,
                    current_ema,
                    ema_step,
                )
                (
                    next_params,
                    next_opt,
                    next_key,
                    loss,
                    priority,
                    next_ema,
                    next_ema_step,
                    diagnostics,
                ) = result
                return (
                    next_params,
                    next_opt,
                    next_key,
                    next_ema,
                    next_ema_step,
                ), (loss, priority, diagnostics)

            xs = (obs_inputs, actions, loss_coeff)
            if action_pad_mask is not None:
                xs = (*xs, action_pad_mask)
            carry, outputs = jax.lax.scan(
                body_fn,
                (params, opt_state, rng_key, ema_params, ema_optimization_step),
                xs,
            )
            new_params, new_opt, new_key, new_ema, new_ema_step = carry
            losses, priorities, diagnostics = outputs
            return (
                new_params,
                new_opt,
                new_key,
                losses[-1],
                priorities[-1],
                new_ema,
                new_ema_step,
                jax.tree.map(lambda value: value[-1], diagnostics),
            )

        return update_many_fn

    def _clip_physical_actions(self, actions):
        actions = jnp.nan_to_num(actions, nan=0.0)
        if self._sample_clip_bounds is None:
            return actions
        sample_min, sample_max = self._sample_clip_bounds
        return jnp.clip(actions, sample_min, sample_max)

    def _build_a2a_sample_fn(self):
        source_spec = self.flow_source

        def sample_fn(params, rng_key, obs_inputs, history, history_pad_mask):
            obs_features = self._features_from_inputs(params, obs_inputs)
            history = self._perturb_a2a_history(
                history,
                history_pad_mask,
                rng_key,
                source_spec.eval_history_noise_std,
            )
            latent = self._a2a_encode_history(
                params,
                history,
                history_pad_mask,
            )[:, None, :]
            latent = self._integrate_a2a_latent(
                params, latent, obs_features, self.num_flow_steps
            )
            return self._clip_physical_actions(
                self._a2a_decode(params, latent, train=False)
            )

        return sample_fn

    def _eval_legato_omega(self, reference_valid):
        omega = legato_schedule(
            self.action_sequence,
            self.flow_source.eval_delay_steps,
            self.flow_source.eval_ramp_steps,
            start=self.action_execution_start,
            kind=self.flow_source.eval_schedule_profile,
            dtype=jnp.float32,
        )
        omega = omega * self.flow_source.eval_strength
        omega = jnp.broadcast_to(
            omega,
            (*reference_valid.shape, 1),
        )
        return jnp.where(reference_valid[..., None], omega, 0.0)

    def _integrate_legato_sample(self, params, sample, obs_features, reference, omega):
        sample_schedule = self._build_sample_schedule()
        batch_size = sample.shape[0]

        def body_fn(index, current_sample):
            step = self.num_flow_steps - index
            tau = sample_schedule[step]
            previous_tau = sample_schedule[step - 1]
            delta_t = tau - previous_tau
            guided = legato_inference_source(current_sample, reference, omega)
            model_input = jnp.concatenate(
                [guided.sample, guided.schedule_channel], axis=-1
            )
            t_batch = jnp.full((batch_size,), tau, dtype=jnp.float32)
            velocity = self._apply_actor(
                params,
                model_input,
                t_batch * self._time_scale,
                obs_features,
            )
            velocity = jnp.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)
            return jnp.nan_to_num(guided.sample + delta_t * velocity, nan=0.0)

        sample = jax.lax.fori_loop(0, self.num_flow_steps, body_fn, sample)
        return self._clip_physical_actions(sample)

    def _build_legato_sample_fn(self):
        def sample_fn(params, rng_key, obs_inputs, reference, reference_valid):
            obs_features = self._features_from_inputs(params, obs_inputs)
            sample = jax.random.normal(rng_key, reference.shape, dtype=reference.dtype)
            omega = self._eval_legato_omega(reference_valid)
            return self._integrate_legato_sample(
                params, sample, obs_features, reference, omega
            )

        return sample_fn

    def _build_legato_sample_from_noise_fn(self):
        def sample_from_noise_fn(params, noise, obs_inputs, reference, reference_valid):
            obs_features = self._features_from_inputs(params, obs_inputs)
            omega = self._eval_legato_omega(reference_valid)
            return self._integrate_legato_sample(
                params, noise, obs_features, reference, omega
            )

        return sample_from_noise_fn

    def _build_sample_fn(self):
        time_scale = self._time_scale
        sample_schedule = self._build_sample_schedule()

        def sample_fn(params, rng_key, obs_inputs):
            obs_features = self._features_from_inputs(params, obs_inputs)
            batch_size = (
                obs_features[1].shape[0]
                if self._condition_as_local
                else obs_features.shape[0]
            )
            sample = jax.random.normal(
                rng_key,
                shape=(batch_size, self.action_sequence, self.action_dim),
            )
            return self._integrate_sample(
                params, sample, obs_features, sample_schedule, time_scale
            )

        return sample_fn

    def _build_sample_from_noise_fn(self):
        """Like `sample_fn` but integrates an externally supplied initial noise
        of shape (batch, action_sequence, action_dim). Used by seed-aligned
        eval so the noise can be keyed on (env_seed, episode_step)."""
        time_scale = self._time_scale
        sample_schedule = self._build_sample_schedule()

        def sample_from_noise_fn(params, noise, obs_inputs):
            obs_features = self._features_from_inputs(params, obs_inputs)
            return self._integrate_sample(
                params, noise, obs_features, sample_schedule, time_scale
            )

        return sample_from_noise_fn

    def _build_sample_schedule(self):
        if self._sample_schedule_jump_point is None:
            return jnp.linspace(
                0.0,
                1.0,
                self.num_flow_steps + 1,
                dtype=jnp.float32,
            )
        jump_point = float(self._sample_schedule_jump_point)
        return jnp.concatenate(
            [
                jnp.linspace(
                    0.0,
                    jump_point,
                    self.num_flow_steps,
                    dtype=jnp.float32,
                ),
                jnp.asarray([1.0], dtype=jnp.float32),
            ],
            axis=0,
        )

    def _integrate_sample(
        self, params, sample, obs_features, sample_schedule, time_scale
    ):
        sample_clip_bounds = self._sample_clip_bounds
        batch_size = (
            obs_features[1].shape[0]
            if self._condition_as_local
            else obs_features.shape[0]
        )

        def body_fn(index, current_sample):
            step = self.num_flow_steps - index
            t_value = sample_schedule[step]
            prev_t = sample_schedule[step - 1]
            delta_t = t_value - prev_t
            t_batch = jnp.full((batch_size,), t_value, dtype=jnp.float32)
            velocity = self._apply_actor(
                params,
                current_sample,
                t_batch * time_scale,
                obs_features,
            )
            velocity = jnp.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)
            next_sample = current_sample + delta_t * velocity
            next_sample = jnp.nan_to_num(next_sample, nan=0.0)
            return next_sample

        sample = jax.lax.fori_loop(0, self.num_flow_steps, body_fn, sample)
        if sample_clip_bounds is None:
            return jnp.nan_to_num(sample, nan=0.0)
        sample_min, sample_max = sample_clip_bounds
        sample = jnp.nan_to_num(sample, nan=0.0)
        return jnp.clip(sample, sample_min, sample_max)

    def _fuse_multi_view(self, rgb_feats):
        if rgb_feats is None:
            return None
        rgb_feats = jnp.asarray(rgb_feats, dtype=jnp.float32)
        if self.view_fusion is not None:
            return self.view_fusion.apply({}, rgb_feats)
        return rgb_feats[:, 0]

    def _rollout_history_storage(self, eval_mode: bool, batch_size: int):
        history = self._eval_action_history if eval_mode else self._train_action_history
        valid = (
            self._eval_action_history_valid
            if eval_mode
            else self._train_action_history_valid
        )
        if batch_size > history.shape[0]:
            raise ValueError(
                f"Action batch has {batch_size} rows but only {history.shape[0]} "
                "rollout slots are configured."
            )
        return history[:batch_size], valid[:batch_size]

    def _rollout_history(self, eval_mode: bool, batch_size: int):
        history, valid = self._rollout_history_storage(eval_mode, batch_size)
        history_horizon = self.flow_source.history_horizon
        return history[:, :history_horizon], valid[:, :history_horizon]

    def _append_rollout_history(self, actions, eval_mode: bool):
        batch_size = actions.shape[0]
        history, valid = self._rollout_history_storage(eval_mode, batch_size)
        executed = actions[
            :,
            self.action_execution_start : self.action_execution_start
            + self.execution_length,
        ]
        executed_valid = jnp.ones(executed.shape[:2], dtype=jnp.bool_)
        history = jnp.concatenate([history, executed], axis=1)[
            :, -self._history_storage_horizon :
        ]
        valid = jnp.concatenate([valid, executed_valid], axis=1)[
            :, -self._history_storage_horizon :
        ]
        if eval_mode:
            self._eval_action_history = self._eval_action_history.at[:batch_size].set(
                history
            )
            self._eval_action_history_valid = self._eval_action_history_valid.at[
                :batch_size
            ].set(valid)
        else:
            self._train_action_history = self._train_action_history.at[:batch_size].set(
                history
            )
            self._train_action_history_valid = self._train_action_history_valid.at[
                :batch_size
            ].set(valid)

    def _append_feedback_history(self, observations: dict, eval_mode: bool):
        key = "executed_action_feedback"
        if key not in observations:
            raise ValueError(
                "A2A history_source=executed_action_feedback requires the "
                f"{key!r} observation."
            )
        feedback = self._as_jax_array(observations[key], self.jnp.float32)
        if feedback.ndim == 2:
            feedback = feedback[:, None, :]
        elif feedback.ndim != 3:
            raise ValueError(
                f"{key} must have shape [B,A] or [B,T,A], got {feedback.shape}."
            )
        if feedback.shape[-1] != self.action_dim:
            raise ValueError(
                f"{key} has action dimension {feedback.shape[-1]}, expected "
                f"{self.action_dim}."
            )
        batch_size = feedback.shape[0]
        history, valid = self._rollout_history_storage(eval_mode, batch_size)
        cold = self.jnp.logical_not(self.jnp.any(valid, axis=1))
        if self.flow_source.history_padding in {"edge", "repeat"}:
            repeated = self.jnp.repeat(
                feedback[:, -1:], self._history_storage_horizon, axis=1
            )
            history = self.jnp.where(cold[:, None, None], repeated, history)
            valid = self.jnp.where(cold[:, None], True, valid)
        history = self.jnp.concatenate([history, feedback], axis=1)[
            :, -self._history_storage_horizon :
        ]
        valid = self.jnp.concatenate(
            [
                valid,
                self.jnp.ones(feedback.shape[:2], dtype=self.jnp.bool_),
            ],
            axis=1,
        )[:, -self._history_storage_horizon :]
        if eval_mode:
            self._eval_action_history = self._eval_action_history.at[:batch_size].set(
                history
            )
            self._eval_action_history_valid = self._eval_action_history_valid.at[
                :batch_size
            ].set(valid)
        else:
            self._train_action_history = self._train_action_history.at[:batch_size].set(
                history
            )
            self._train_action_history_valid = self._train_action_history_valid.at[
                :batch_size
            ].set(valid)

    def _legato_reference(self, eval_mode: bool, batch_size: int):
        previous = (
            self._eval_previous_chunk if eval_mode else self._train_previous_chunk
        )[:batch_size]
        previous_valid = (
            self._eval_previous_chunk_valid
            if eval_mode
            else self._train_previous_chunk_valid
        )[:batch_size]
        shift = self.execution_length
        if shift >= self.action_sequence:
            return jnp.zeros_like(previous), jnp.zeros_like(previous_valid)
        tail = previous[:, shift:]
        tail_valid = previous_valid[:, shift:]
        pad_len = shift
        if self.flow_source.reference_padding == "last":
            pad = jnp.repeat(previous[:, -1:], pad_len, axis=1)
            pad_valid = jnp.repeat(previous_valid[:, -1:], pad_len, axis=1)
        else:
            pad = jnp.zeros(
                (batch_size, pad_len, self.action_dim), dtype=previous.dtype
            )
            pad_valid = jnp.zeros((batch_size, pad_len), dtype=jnp.bool_)
        return (
            jnp.concatenate([tail, pad], axis=1),
            jnp.concatenate([tail_valid, pad_valid], axis=1),
        )

    def _store_previous_chunk(self, actions, eval_mode: bool):
        batch_size = actions.shape[0]
        valid = jnp.ones(actions.shape[:2], dtype=jnp.bool_)
        if eval_mode:
            self._eval_previous_chunk = self._eval_previous_chunk.at[:batch_size].set(
                actions
            )
            self._eval_previous_chunk_valid = self._eval_previous_chunk_valid.at[
                :batch_size
            ].set(valid)
        else:
            self._train_previous_chunk = self._train_previous_chunk.at[:batch_size].set(
                actions
            )
            self._train_previous_chunk_valid = self._train_previous_chunk_valid.at[
                :batch_size
            ].set(valid)

    def _legato_issued_chunk(self, generated, reference, reference_valid):
        execution_index = jnp.arange(self.action_sequence)[None, :]
        delay_mask = (execution_index >= self.action_execution_start) & (
            execution_index
            < self.action_execution_start + self.flow_source.eval_delay_steps
        )
        use_reference = delay_mask & reference_valid
        return jnp.where(use_reference[..., None], reference, generated)

    def _record_legato_metrics(
        self,
        generated,
        reference,
        reference_valid,
        *,
        eval_mode: bool,
    ):
        if not eval_mode:
            return
        generated = np.asarray(jax.device_get(generated), dtype=np.float32)
        reference = np.asarray(jax.device_get(reference), dtype=np.float32)
        reference_valid = np.asarray(
            jax.device_get(reference_valid),
            dtype=np.bool_,
        )
        omega = np.asarray(
            jax.device_get(self._eval_legato_omega(reference_valid)),
            dtype=np.float32,
        )[..., 0]
        position = np.arange(self.action_sequence)[None, :]
        prefix = (
            (position >= self.action_execution_start)
            & (
                position
                < self.action_execution_start + self.flow_source.eval_delay_steps
            )
            & reference_valid
        )
        overlap = (omega > 0.0) & reference_valid
        squared_error = np.mean(np.square(generated - reference), axis=-1)
        for name, mask in (
            ("legato_prefix_rmse", prefix),
            ("legato_overlap_rmse", overlap),
        ):
            count = mask.sum(axis=1)
            values = np.sqrt(
                (squared_error * mask).sum(axis=1) / np.maximum(count, 1)
            )
            finite = count > 0
            if np.any(finite):
                self._rollout_metric_sums[name] = self._rollout_metric_sums.get(
                    name, 0.0
                ) + float(values[finite].sum())
                self._rollout_metric_counts[name] = self._rollout_metric_counts.get(
                    name, 0
                ) + int(finite.sum())

    def _record_rollout_metrics(self, actions: np.ndarray, eval_mode: bool):
        batch_size = actions.shape[0]
        slot_capacity = (
            self._eval_last_issued_action.shape[0]
            if eval_mode
            else self._train_last_issued_action.shape[0]
        )
        tracked_rows = min(batch_size, slot_capacity)
        executed = actions[
            :,
            self.action_execution_start : self.action_execution_start
            + self.execution_length,
        ]
        first = executed[:, 0]
        last = executed[:, -1]
        previous = (
            self._eval_last_issued_action
            if eval_mode
            else self._train_last_issued_action
        )[:tracked_rows]
        previous_valid = (
            self._eval_last_issued_valid if eval_mode else self._train_last_issued_valid
        )[:tracked_rows]
        previous = np.asarray(jax.device_get(previous), dtype=np.float32)
        previous_valid = np.asarray(jax.device_get(previous_valid), dtype=np.bool_)

        per_row = {}
        if executed.shape[1] > 1:
            first_difference = np.diff(executed, axis=1)
            per_row["action_first_difference"] = np.linalg.norm(
                first_difference, axis=-1
            ).mean(axis=1)
        if executed.shape[1] > 2:
            second_difference = np.diff(executed, n=2, axis=1)
            per_row["action_second_difference"] = np.linalg.norm(
                second_difference, axis=-1
            ).mean(axis=1)
        if executed.shape[1] > 3:
            third_difference = np.diff(executed, n=3, axis=1)
            per_row["action_jerk"] = np.linalg.norm(third_difference, axis=-1).mean(
                axis=1
            )
        if np.any(previous_valid):
            boundary = np.linalg.norm(first[:tracked_rows] - previous, axis=-1)
            per_row["action_boundary_jump"] = np.where(previous_valid, boundary, np.nan)

        if eval_mode:
            for name, values in per_row.items():
                finite = np.isfinite(values)
                if np.any(finite):
                    self._rollout_metric_sums[name] = self._rollout_metric_sums.get(
                        name, 0.0
                    ) + float(np.asarray(values)[finite].sum())
                    self._rollout_metric_counts[name] = self._rollout_metric_counts.get(
                        name, 0
                    ) + int(finite.sum())

        if tracked_rows == 0:
            return
        last_device = jnp.asarray(last[:tracked_rows], dtype=jnp.float32)
        if eval_mode:
            self._eval_last_issued_action = self._eval_last_issued_action.at[
                :tracked_rows
            ].set(last_device)
            self._eval_last_issued_valid = self._eval_last_issued_valid.at[
                :tracked_rows
            ].set(True)
        else:
            self._train_last_issued_action = self._train_last_issued_action.at[
                :tracked_rows
            ].set(last_device)
            self._train_last_issued_valid = self._train_last_issued_valid.at[
                :tracked_rows
            ].set(True)

    def rollout_diagnostics(self) -> dict[str, float]:
        return {
            name: total / max(self._rollout_metric_counts.get(name, 0), 1)
            for name, total in self._rollout_metric_sums.items()
        }

    def act(self, observations: dict, step: int, eval_mode: bool):
        del step
        if self._trainable_encoder:
            obs_inputs = self._prepare_trainable_obs_inputs(observations)
        else:
            obs_inputs, _ = self._prepare_obs_features(observations)
        sample_params = self.ema_params if (eval_mode and self.use_ema) else self.params
        batch_size = int(next(iter(observations.values())).shape[0])
        if self._flow_source_type == "gaussian":
            if eval_mode and self._active_eval_seeds is not None:
                noise = self._aligned_eval_noise(self._active_eval_seeds)
                actions = self._sample_from_noise_impl(sample_params, noise, obs_inputs)
            else:
                self.rng_key, sample_key = jax.random.split(self.rng_key)
                actions = self._sample_impl(sample_params, sample_key, obs_inputs)
        elif self._flow_source_type in {"a2a", "a2a_noise"}:
            if eval_mode and self._active_eval_seeds is not None:
                sample_key = self._aligned_eval_keys(self._active_eval_seeds)
            else:
                self.rng_key, sample_key = jax.random.split(self.rng_key)
            if self.flow_source.history_source == "executed_action_feedback":
                self._append_feedback_history(observations, eval_mode)
            history, history_valid = self._rollout_history(eval_mode, batch_size)
            actions = self._sample_impl(
                sample_params,
                sample_key,
                obs_inputs,
                history,
                jnp.logical_not(history_valid),
            )
            if self.flow_source.history_source == "commanded_action":
                self._append_rollout_history(actions, eval_mode)
        else:
            reference, reference_valid = self._legato_reference(eval_mode, batch_size)
            if eval_mode and self._active_eval_seeds is not None:
                noise = self._aligned_eval_noise(self._active_eval_seeds)
                generated_actions = self._sample_from_noise_impl(
                    sample_params,
                    noise,
                    obs_inputs,
                    reference,
                    reference_valid,
                )
            else:
                self.rng_key, sample_key = jax.random.split(self.rng_key)
                generated_actions = self._sample_impl(
                    sample_params,
                    sample_key,
                    obs_inputs,
                    reference,
                    reference_valid,
                )
            self._record_legato_metrics(
                generated_actions,
                reference,
                reference_valid,
                eval_mode=eval_mode,
            )
            self._store_previous_chunk(generated_actions, eval_mode)
            actions = self._legato_issued_chunk(
                generated_actions, reference, reference_valid
            )
        self._block(actions)
        actions_np = np.asarray(jax.device_get(actions), dtype=np.float32)
        self._record_rollout_metrics(actions_np, eval_mode)
        return actions_np

    def set_eval_env_running(self, value: bool):
        super().set_eval_env_running(value)
        if value:
            self._rollout_metric_sums = {}
            self._rollout_metric_counts = {}

    def reset(self, step: int, agents_to_reset: list[int]):
        del step
        if not agents_to_reset:
            return
        if self.eval_env_running:
            if len(agents_to_reset) == self.num_eval_envs or any(
                int(agent_id) not in self._eval_slot_by_global_id
                for agent_id in agents_to_reset
            ):
                for row, agent_id in enumerate(agents_to_reset):
                    if row >= self.num_eval_envs:
                        break
                    self._eval_slot_by_global_id[int(agent_id)] = row
            rows = [
                self._eval_slot_by_global_id[int(agent_id)]
                for agent_id in agents_to_reset
                if int(agent_id) in self._eval_slot_by_global_id
            ]
            for row in rows:
                self._eval_action_history = self._eval_action_history.at[row].set(0.0)
                self._eval_action_history_valid = self._eval_action_history_valid.at[
                    row
                ].set(False)
                self._eval_previous_chunk = self._eval_previous_chunk.at[row].set(0.0)
                self._eval_previous_chunk_valid = self._eval_previous_chunk_valid.at[
                    row
                ].set(False)
                self._eval_last_issued_action = self._eval_last_issued_action.at[
                    row
                ].set(0.0)
                self._eval_last_issued_valid = self._eval_last_issued_valid.at[row].set(
                    False
                )
            return
        for row in agents_to_reset:
            row = int(row)
            if not 0 <= row < self.num_train_envs:
                continue
            self._train_action_history = self._train_action_history.at[row].set(0.0)
            self._train_action_history_valid = self._train_action_history_valid.at[
                row
            ].set(False)
            self._train_previous_chunk = self._train_previous_chunk.at[row].set(0.0)
            self._train_previous_chunk_valid = self._train_previous_chunk_valid.at[
                row
            ].set(False)
            self._train_last_issued_action = self._train_last_issued_action.at[row].set(
                0.0
            )
            self._train_last_issued_valid = self._train_last_issued_valid.at[row].set(
                False
            )

    def set_active_eval_seeds(self, seeds_in_batch_order):
        """Enable seed-aligned eval noise. `seeds_in_batch_order` is the env
        seed for each observation-batch row, in row order. Pass None to
        disable and restore the default shared-rng_key sampling."""
        if seeds_in_batch_order is None:
            self._active_eval_seeds = None
        else:
            self._active_eval_seeds = [int(s) for s in seeds_in_batch_order]

    def reset_aligned_eval_noise(self):
        """Clear per-seed episode-step counters (call once before an eval run)."""
        self._eval_step_by_seed = {}

    def _aligned_eval_keys(self, seeds):
        rows = []
        for s in seeds:
            s = int(s)
            t = self._eval_step_by_seed.get(s, 0)
            self._eval_step_by_seed[s] = t + 1
            row_key = jax.random.fold_in(
                jax.random.fold_in(self._eval_noise_base_key, s), t
            )
            rows.append(row_key)
        return jnp.stack(rows, axis=0)

    def _aligned_eval_noise(self, seeds):
        keys = self._aligned_eval_keys(seeds)
        return jax.vmap(
            lambda key: jax.random.normal(
                key, shape=(self.action_sequence, self.action_dim)
            )
        )(keys)

    def _last_obs_inputs(self, obs_inputs):
        if isinstance(obs_inputs, dict):
            return self.jax.tree_util.tree_map(lambda value: value[-1], obs_inputs)
        return obs_inputs[-1]

    def _extract_a2a_history(self, batch):
        missing_fields = {
            "action_history",
            "action_history_pad_mask",
        }.difference(batch)
        if missing_fields:
            raise ValueError(
                "A2A replay batches are missing required fields: "
                + ", ".join(sorted(missing_fields))
                + "."
            )
        return (
            self._as_jax_array(batch["action_history"], self.jnp.float32),
            self._as_jax_array(
                batch["action_history_pad_mask"],
                self.jnp.bool_,
            ),
        )

    def update(
        self,
        replay_iter: Iterator[dict],
        step: int,
        replay_buffer: ReplayBuffer = None,
    ) -> dict[str, np.ndarray]:
        del step
        batch = next(replay_iter)
        actions = self._as_jax_array(batch["action"], self.jnp.float32)
        action_pad_mask = self._extract_action_pad_mask(batch)
        loss_coeff = self._loss_weights(batch)
        if self._trainable_encoder:
            obs_inputs = self._prepare_trainable_obs_inputs(batch)
            if self.image_augmentation_type != "none":
                self.rng_key, aug_key = jax.random.split(self.rng_key)
                obs_inputs = self._augment_trainable_obs_inputs(obs_inputs, aug_key)
            metrics = {}
        else:
            obs_inputs, metrics = self._prepare_obs_features(batch)

        start_time = time.perf_counter()
        source_diagnostics = None
        if self._flow_source_type in {"a2a", "a2a_noise"}:
            history_actions, history_pad_mask = self._extract_a2a_history(batch)
            (
                self.params,
                self.opt_state,
                self.rng_key,
                actor_loss,
                new_priority,
                self.ema_params,
                self._ema_optimization_step,
                source_diagnostics,
            ) = self._update_impl(
                self.params,
                self.opt_state,
                self.rng_key,
                obs_inputs,
                history_actions,
                history_pad_mask,
                actions,
                loss_coeff,
                action_pad_mask,
                self.ema_params,
                self._ema_optimization_step,
            )
        elif self._flow_source_type == "legato":
            (
                self.params,
                self.opt_state,
                self.rng_key,
                actor_loss,
                new_priority,
                self.ema_params,
                self._ema_optimization_step,
                source_diagnostics,
            ) = self._update_impl(
                self.params,
                self.opt_state,
                self.rng_key,
                obs_inputs,
                actions,
                loss_coeff,
                action_pad_mask,
                self.ema_params,
                self._ema_optimization_step,
            )
        else:
            (
                self.params,
                self.opt_state,
                self.rng_key,
                actor_loss,
                new_priority,
                self.ema_params,
                self._ema_optimization_step,
            ) = self._update_impl(
                self.params,
                self.opt_state,
                self.rng_key,
                obs_inputs,
                actions,
                loss_coeff,
                action_pad_mask,
                self.ema_params,
                self._ema_optimization_step,
            )
        uses_priorities = self._uses_replay_priorities(replay_buffer)
        if self._should_block_update(uses_priorities):
            if uses_priorities:
                self._block(actor_loss, new_priority)
            else:
                self._block(actor_loss)
        elapsed = time.perf_counter() - start_time
        self._update_step_count += 1

        if uses_priorities:
            new_priority_np = np.asarray(
                jax.device_get(new_priority),
                dtype=np.float32,
            )
            self._maybe_update_priorities(replay_buffer, batch, new_priority_np)
        self._maybe_log_update_metrics(metrics, actor_loss, obs_inputs, elapsed)
        if self.logging and source_diagnostics is not None:
            metrics.update(
                {
                    key: float(np.asarray(jax.device_get(value), dtype=np.float32))
                    for key, value in source_diagnostics.items()
                }
            )
        self._first_update_completed = True
        return metrics

    def update_many(
        self,
        replay_iter: Iterator[dict],
        num_updates: int,
        replay_buffer: ReplayBuffer = None,
    ) -> dict[str, np.ndarray]:
        num_updates = int(num_updates)
        if num_updates <= 1 or self._uses_replay_priorities(replay_buffer):
            metrics = {}
            for _ in range(max(num_updates, 1)):
                metrics.update(self.update(replay_iter, 0, replay_buffer))
            return metrics

        obs_inputs_list = []
        actions = []
        loss_coeffs = []
        action_pad_masks = []
        action_histories = []
        action_history_pad_masks = []
        has_action_pad_mask = None
        augmentation_keys = None
        if self._trainable_encoder and self.image_augmentation_type != "none":
            split_keys = jax.random.split(self.rng_key, num_updates + 1)
            self.rng_key = split_keys[0]
            augmentation_keys = split_keys[1:]
        for update_idx in range(num_updates):
            batch = next(replay_iter)
            actions.append(self._as_jax_array(batch["action"], self.jnp.float32))
            loss_coeffs.append(self._loss_weights(batch))
            if self._flow_source_type in {"a2a", "a2a_noise"}:
                history, history_pad_mask = self._extract_a2a_history(batch)
                action_histories.append(history)
                action_history_pad_masks.append(history_pad_mask)
            if self._trainable_encoder:
                obs = self._prepare_trainable_obs_inputs(batch)
                if augmentation_keys is not None:
                    obs = self._augment_trainable_obs_inputs(
                        obs,
                        augmentation_keys[update_idx],
                    )
            else:
                obs, _ = self._prepare_obs_features(batch)
            obs_inputs_list.append(obs)
            action_pad_mask = self._extract_action_pad_mask(batch)
            current_has_mask = action_pad_mask is not None
            if has_action_pad_mask is None:
                has_action_pad_mask = current_has_mask
            elif has_action_pad_mask != current_has_mask:
                raise ValueError(
                    "Cannot fuse updates with mixed action_pad_mask presence."
                )
            if current_has_mask:
                action_pad_masks.append(action_pad_mask)

        obs_inputs = self.jax.tree_util.tree_map(
            lambda *values: self.jnp.stack(values, axis=0),
            *obs_inputs_list,
        )
        actions = self.jnp.stack(actions, axis=0)
        loss_coeffs = self.jnp.stack(loss_coeffs, axis=0)
        action_pad_mask = (
            self.jnp.stack(action_pad_masks, axis=0) if has_action_pad_mask else None
        )
        if action_histories:
            action_history = self.jnp.stack(action_histories, axis=0)
            action_history_pad_mask = self.jnp.stack(action_history_pad_masks, axis=0)

        start_time = time.perf_counter()
        source_diagnostics = None
        if self._flow_source_type in {"a2a", "a2a_noise"}:
            result = self._update_many_impl(
                self.params,
                self.opt_state,
                self.rng_key,
                obs_inputs,
                action_history,
                action_history_pad_mask,
                actions,
                loss_coeffs,
                action_pad_mask,
                self.ema_params,
                self._ema_optimization_step,
            )
        else:
            result = self._update_many_impl(
                self.params,
                self.opt_state,
                self.rng_key,
                obs_inputs,
                actions,
                loss_coeffs,
                action_pad_mask,
                self.ema_params,
                self._ema_optimization_step,
            )
        if self._flow_source_type == "gaussian":
            (
                self.params,
                self.opt_state,
                self.rng_key,
                actor_loss,
                _new_priority,
                self.ema_params,
                self._ema_optimization_step,
            ) = result
        else:
            (
                self.params,
                self.opt_state,
                self.rng_key,
                actor_loss,
                _new_priority,
                self.ema_params,
                self._ema_optimization_step,
                source_diagnostics,
            ) = result
        if (
            self.logging
            or (self._update_step_count + num_updates) % self._update_block_every_steps
            == 0
        ):
            self._block(actor_loss)
        elapsed = time.perf_counter() - start_time
        self._update_step_count += num_updates
        self._first_update_completed = True

        if not self.logging:
            return {}
        metrics = {}
        self._maybe_log_update_metrics(
            metrics,
            actor_loss,
            self._last_obs_inputs(obs_inputs),
            elapsed / max(num_updates, 1),
        )
        if source_diagnostics is not None:
            metrics.update(
                {
                    key: float(np.asarray(jax.device_get(value), dtype=np.float32))
                    for key, value in source_diagnostics.items()
                }
            )
        return metrics

    def state_dict(self) -> dict:
        state = {"params": self._tree_to_numpy(self.params)}
        if self.encoder is not None:
            encoder_frozen_state = self.encoder.frozen_state_dict()
            if encoder_frozen_state:
                state["_encoder_frozen_state"] = self._tree_to_numpy(
                    encoder_frozen_state
                )
        if self.use_ema and self.ema_params is not None:
            state["_ema_params"] = self._tree_to_numpy(self.ema_params)
            state["_ema_state"] = {
                "decay": float(self._ema_decay),
                "decay_schedule": self._ema_decay_schedule,
                "min_decay": float(self._ema_min_decay),
                "optimization_step": int(self._ema_optimization_step),
                "update_after_step": int(self._ema_update_after_step),
                "use_ema_warmup": bool(self._ema_use_warmup),
                "inv_gamma": float(self._ema_inv_gamma),
                "power": float(self._ema_power),
            }
        return state

    @staticmethod
    def _validate_checkpoint_param_shapes(expected, loaded, *, label: str) -> None:
        expected_with_path, expected_tree = jax.tree_util.tree_flatten_with_path(
            expected
        )
        loaded_with_path, loaded_tree = jax.tree_util.tree_flatten_with_path(loaded)
        if expected_tree != loaded_tree:
            raise ValueError(
                f"{label} parameter tree is incompatible with the current model. "
                "This checkpoint was likely created by a different architecture."
            )
        for (path, expected_leaf), (_, loaded_leaf) in zip(
            expected_with_path,
            loaded_with_path,
            strict=True,
        ):
            expected_shape = np.shape(expected_leaf)
            loaded_shape = np.shape(loaded_leaf)
            if expected_shape != loaded_shape:
                raise ValueError(
                    f"{label} parameter shape mismatch at "
                    f"{jax.tree_util.keystr(path)}: checkpoint has "
                    f"{loaded_shape}, current model expects {expected_shape}."
                )

    def load_state_dict(self, state_dict: dict):
        expected_params = self.params
        params = self._tree_from_numpy(state_dict["params"])
        if self._trainable_encoder and not (
            isinstance(params, dict) and "actor" in params
        ):
            params = {
                "actor": params,
                "encoder": self.encoder.trainable_params,
            }
            if not (
                isinstance(expected_params, dict) and "actor" in expected_params
            ):
                expected_params = {
                    "actor": expected_params,
                    "encoder": self.encoder.trainable_params,
                }
        self._validate_checkpoint_param_shapes(
            expected_params,
            params,
            label="Checkpoint",
        )
        encoder_frozen_state = state_dict.get("_encoder_frozen_state")
        loaded_encoder_frozen_state = None
        if self.encoder is not None and encoder_frozen_state is not None:
            loaded_encoder_frozen_state = self._tree_from_numpy(
                encoder_frozen_state
            )
            expected_encoder_frozen_state = self.encoder.frozen_state_dict()
            if expected_encoder_frozen_state:
                self._validate_checkpoint_param_shapes(
                    expected_encoder_frozen_state,
                    loaded_encoder_frozen_state,
                    label="Encoder frozen-state checkpoint",
                )

        new_ema_params = None
        new_ema_decay = self._ema_decay
        new_ema_decay_schedule = self._ema_decay_schedule
        new_ema_min_decay = self._ema_min_decay
        new_ema_optimization_step = self._ema_optimization_step
        new_ema_update_after_step = self._ema_update_after_step
        new_ema_use_warmup = self._ema_use_warmup
        new_ema_inv_gamma = self._ema_inv_gamma
        new_ema_power = self._ema_power
        ema_params = state_dict.get("_ema_params")
        ema_state = state_dict.get("_ema_state")
        if self.use_ema:
            if ema_params is not None:
                new_ema_params = self._tree_from_numpy(ema_params)
                if self._trainable_encoder and not (
                    isinstance(new_ema_params, dict) and "actor" in new_ema_params
                ):
                    new_ema_params = {
                        "actor": new_ema_params,
                        "encoder": self.encoder.trainable_params,
                    }
                self._validate_checkpoint_param_shapes(
                    params,
                    new_ema_params,
                    label="EMA checkpoint",
                )
            else:
                new_ema_params = params
            if ema_state is None:
                new_ema_optimization_step = 0
            else:
                new_ema_decay = float(ema_state.get("decay", new_ema_decay))
                new_ema_decay_schedule = str(
                    ema_state.get("decay_schedule", new_ema_decay_schedule)
                )
                new_ema_min_decay = float(
                    ema_state.get("min_decay", new_ema_min_decay)
                )
                new_ema_optimization_step = int(
                    ema_state.get("optimization_step", new_ema_optimization_step)
                )
                new_ema_update_after_step = int(
                    ema_state.get("update_after_step", new_ema_update_after_step)
                )
                new_ema_use_warmup = bool(
                    ema_state.get("use_ema_warmup", new_ema_use_warmup)
                )
                new_ema_inv_gamma = float(
                    ema_state.get("inv_gamma", new_ema_inv_gamma)
                )
                new_ema_power = float(ema_state.get("power", new_ema_power))

        if self.encoder is not None and loaded_encoder_frozen_state is not None:
            self.encoder.load_frozen_state_dict(loaded_encoder_frozen_state)
        self.params = params
        if not self.use_ema:
            return
        self.ema_params = new_ema_params
        self._ema_decay = new_ema_decay
        self._ema_decay_schedule = new_ema_decay_schedule
        self._ema_min_decay = new_ema_min_decay
        self._ema_optimization_step = new_ema_optimization_step
        self._ema_update_after_step = new_ema_update_after_step
        self._ema_use_warmup = new_ema_use_warmup
        self._ema_inv_gamma = new_ema_inv_gamma
        self._ema_power = new_ema_power


__all__ = [
    "FlowMatching",
    "FlowMatchingBackboneSpec",
    "FlowMatchingModelSpec",
    "FlowMatchingSpec",
    "FlowSourceSpec",
    "flow_matching_model_spec_from_cfg",
    "flow_matching_spec_from_cfg",
]
