"""ACT imitation-learning method backed by JAX/Flax."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Optional

import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jndi
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig

from robobase.language import lang_feature_rows, lang_token_rows, tokens_to_feature_jax
from robobase.method.bc import (
    BCEncoderModelSpec,
    BCViewFusionModelSpec,
    legacy_resnet_encoder_impl,
)
from robobase.method.bc_runtime import bc_observation_layout
from robobase.method.jax_base import JaxMethodBase
from robobase.models.act import ACTImageProjection, JaxACTPolicy
from robobase.models.camera_augmentation import augment_campose_observation
from robobase.models.encoder import JaxResNetEncoder
from robobase.replay_buffer.replay_buffer import ReplayBuffer


@dataclass(frozen=True)
class ACTActorModelSpec:
    type: str
    hidden_dim: int
    enc_layers: int
    dec_layers: int
    dim_feedforward: int
    dropout: float
    nheads: int
    num_queries: int
    pre_norm: bool
    latent_dim: int = 32
    kl_weight: float = 10.0
    gripper_loss_weight: float = 0.05
    gripper_dims: int = 0
    data_augmentation: bool = True
    use_lang_cond: bool = False
    lang_feature_dim: int = 512
    image_position_embedding_type: str = "sincos_legacy"
    image_augmentation_type: str = "legacy"
    use_camera_extrinsics: bool = False
    num_camera_extrinsics: int = 2
    image_position_max_tokens: int | None = None
    proprio_dropout_prob: float = 0.0
    proprio_projection_type: str = "legacy_mlp"
    style_cls_type: str = "learned"
    decoder_output_layer: str = "final"


@dataclass(frozen=True)
class ACTModelSpec:
    actor_model: ACTActorModelSpec
    encoder_model: BCEncoderModelSpec | None
    view_fusion_model: BCViewFusionModelSpec | None


@dataclass(frozen=True)
class ACTSpec:
    lr: float
    lr_backbone: float
    adaptive_lr: bool
    num_train_steps: int
    actor_grad_clip: float | None
    weight_decay: float
    horizon_dropout_lengths: tuple[int, ...] | None
    horizon_dropout_probs: tuple[float, ...] | None
    model: ACTModelSpec


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


def act_model_spec_from_cfg(cfg: DictConfig) -> ACTModelSpec:
    method_cfg = cfg.method
    actor_model_cfg = method_cfg.get("actor_model", None)
    if actor_model_cfg is None:
        raise ValueError("ACT requires an actor_model config.")

    actor_model_type = _config_type(
        actor_model_cfg,
        default="transformer",
        target_to_type={
            "robobase.models.multi_view_transformer.MultiViewTransformerEncoderDecoderACT": (
                "transformer"
            ),
            "robobase.models.act.JaxACTTransformer": "transformer",
            "robobase.models.act.JaxACTPolicy": "transformer",
            "robobase.models.act.MultiViewTransformerEncoderDecoderACT": "transformer",
        },
    )
    actor_model_spec = ACTActorModelSpec(
        type=actor_model_type,
        hidden_dim=int(actor_model_cfg.get("hidden_dim", 512)),
        enc_layers=int(actor_model_cfg.get("enc_layers", 4)),
        dec_layers=int(actor_model_cfg.get("dec_layers", 1)),
        dim_feedforward=int(actor_model_cfg.get("dim_feedforward", 3200)),
        dropout=float(actor_model_cfg.get("dropout", 0.1)),
        nheads=int(actor_model_cfg.get("nheads", 8)),
        num_queries=int(actor_model_cfg.get("num_queries", cfg.action_sequence)),
        pre_norm=bool(actor_model_cfg.get("pre_norm", False)),
        latent_dim=int(actor_model_cfg.get("latent_dim", 32)),
        kl_weight=float(actor_model_cfg.get("kl_weight", 10.0)),
        gripper_loss_weight=float(actor_model_cfg.get("gripper_loss_weight", 0.05)),
        gripper_dims=int(actor_model_cfg.get("gripper_dims", 0)),
        data_augmentation=bool(actor_model_cfg.get("data_augmentation", True)),
        use_lang_cond=bool(
            actor_model_cfg.get("use_lang_cond", method_cfg.get("use_lang_cond", False))
        ),
        lang_feature_dim=int(
            actor_model_cfg.get(
                "lang_feature_dim",
                method_cfg.get("lang_feature_dim", 512),
            )
        ),
        image_position_embedding_type=str(
            actor_model_cfg.get(
                "image_position_embedding_type",
                "sincos_legacy",
            )
        ).lower(),
        image_augmentation_type=str(
            actor_model_cfg.get("image_augmentation_type", "legacy")
        ).lower(),
        use_camera_extrinsics=bool(actor_model_cfg.get("use_camera_extrinsics", False)),
        num_camera_extrinsics=int(actor_model_cfg.get("num_camera_extrinsics", 2)),
        image_position_max_tokens=(
            None
            if actor_model_cfg.get("image_position_max_tokens", None) is None
            else int(actor_model_cfg.get("image_position_max_tokens"))
        ),
        proprio_dropout_prob=float(actor_model_cfg.get("proprio_dropout_prob", 0.0)),
        proprio_projection_type=str(
            actor_model_cfg.get("proprio_projection_type", "legacy_mlp")
        ).lower(),
        style_cls_type=str(actor_model_cfg.get("style_cls_type", "learned")).lower(),
        decoder_output_layer=str(
            actor_model_cfg.get("decoder_output_layer", "final")
        ).lower(),
    )

    encoder_model_cfg = method_cfg.get("encoder_model", None)
    encoder_model_spec = None
    if encoder_model_cfg is not None:
        encoder_model_type = _config_type(
            encoder_model_cfg,
            default="resnet",
            target_to_type={
                "robobase.method.act.ImageEncoderACT": "resnet",
                "robobase.models.encoder.JaxResNetEncoder": "resnet",
            },
        )
        encoder_model_spec = BCEncoderModelSpec(
            legacy_impl=legacy_resnet_encoder_impl(cfg),
            type=encoder_model_type,
            model=str(
                encoder_model_cfg.get(
                    "model",
                    encoder_model_cfg.get("backbone", "resnet18"),
                )
            ),
            trainable=bool(encoder_model_cfg.get("trainable", True)),
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

    return ACTModelSpec(
        actor_model=actor_model_spec,
        encoder_model=encoder_model_spec,
        view_fusion_model=view_fusion_model_spec,
    )


def act_spec_from_cfg(cfg: DictConfig) -> ACTSpec:
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
    return ACTSpec(
        lr=float(cfg.method.lr),
        lr_backbone=float(cfg.method.get("lr_backbone", cfg.method.lr)),
        adaptive_lr=bool(cfg.method.adaptive_lr),
        num_train_steps=int(cfg.method.num_train_steps),
        actor_grad_clip=(
            None
            if cfg.method.actor_grad_clip is None
            else float(cfg.method.actor_grad_clip)
        ),
        weight_decay=float(cfg.method.get("weight_decay", 1e-4)),
        horizon_dropout_lengths=horizon_dropout_lengths,
        horizon_dropout_probs=horizon_dropout_probs,
        model=act_model_spec_from_cfg(cfg),
    )


@dataclass(frozen=True)
class _BuiltACTModel:
    actor_model: JaxACTPolicy
    encoder_model: JaxResNetEncoder | None
    image_projection_model: ACTImageProjection | None


def _build_model(
    model_spec: ACTModelSpec,
    *,
    observation_space: spaces.Dict,
    action_space: spaces.Box,
    encoder_jit: bool,
    encoder_seed: int = 0,
) -> _BuiltACTModel:
    obs_layout = bc_observation_layout(observation_space)
    actor_spec = model_spec.actor_model
    if actor_spec.type != "transformer":
        raise NotImplementedError(
            f"Unsupported ACT actor model type '{actor_spec.type}'."
        )
    if actor_spec.num_queries != int(action_space.shape[0]):
        raise ValueError(
            "ACT actor_model.num_queries must match action_space sequence length."
        )

    encoder_model = None
    image_projection_model = None

    if obs_layout.use_pixels:
        if model_spec.encoder_model is None:
            raise ValueError("Pixel ACT requires encoder_model in the model spec.")
        if model_spec.encoder_model.type != "resnet":
            raise NotImplementedError(
                f"Unsupported ACT encoder model type '{model_spec.encoder_model.type}'."
            )
        if not model_spec.encoder_model.trainable:
            raise ValueError(
                "Reference-style JAX ACT requires encoder_model.trainable=true for "
                "pixel inputs because it consumes spatial ResNet feature maps."
            )
        if obs_layout.rgb_input_shape is None:
            raise ValueError("Pixel ACT expected a valid RGB input shape.")
        if (
            model_spec.encoder_model.use_plucker
            and not obs_layout.has_camera_conditioning
        ):
            raise ValueError(
                "ACT encoder_model.use_plucker=true requires raymap or camera "
                "parameter observations paired with every RGB observation."
            )
        act_encoder_cls = JaxResNetEncoder
        if getattr(model_spec.encoder_model, "legacy_impl", False):
            from robobase.models.encoder import LegacyJaxResNetEncoder

            act_encoder_cls = LegacyJaxResNetEncoder
        encoder_model = act_encoder_cls(
            input_shape=obs_layout.rgb_input_shape,
            model=model_spec.encoder_model.model,
            jit=encoder_jit,
            pretrained=model_spec.encoder_model.pretrained,
            pretrained_weights_path=(
                model_spec.encoder_model.pretrained_weights_path
            ),
            resize_to_224=False,
            use_plucker=model_spec.encoder_model.use_plucker,
            plucker_fusion_mode=model_spec.encoder_model.plucker_fusion_mode,
            plucker_hidden_channels=model_spec.encoder_model.plucker_hidden_channels,
            plucker_identity_init=model_spec.encoder_model.plucker_identity_init,
            use_film=actor_spec.use_lang_cond,
            film_task_input_dim=actor_spec.lang_feature_dim,
            film_task_hidden_dim=actor_spec.hidden_dim,
            seed=encoder_seed,
        )
        image_projection_model = ACTImageProjection(
            hidden_dim=actor_spec.hidden_dim,
            position_embedding_type=actor_spec.image_position_embedding_type,
            max_position_tokens=actor_spec.image_position_max_tokens,
        )
    elif (
        model_spec.encoder_model is not None
        and model_spec.encoder_model.type != "resnet"
    ):
        raise NotImplementedError(
            f"Unsupported ACT encoder model type '{model_spec.encoder_model.type}'."
        )

    return _BuiltACTModel(
        actor_model=JaxACTPolicy(
            hidden_dim=actor_spec.hidden_dim,
            dropout=actor_spec.dropout,
            nheads=actor_spec.nheads,
            dim_feedforward=actor_spec.dim_feedforward,
            enc_layers=actor_spec.enc_layers,
            dec_layers=actor_spec.dec_layers,
            pre_norm=actor_spec.pre_norm,
            state_dim=int(obs_layout.low_dim_size),
            action_dim=int(action_space.shape[1]),
            num_queries=actor_spec.num_queries,
            latent_dim=actor_spec.latent_dim,
            use_lang_cond=actor_spec.use_lang_cond,
            use_camera_extrinsics=actor_spec.use_camera_extrinsics,
            num_camera_extrinsics=actor_spec.num_camera_extrinsics,
            proprio_dropout_prob=actor_spec.proprio_dropout_prob,
            proprio_projection_type=actor_spec.proprio_projection_type,
            style_cls_type=actor_spec.style_cls_type,
            decoder_output_layer=actor_spec.decoder_output_layer,
        ),
        encoder_model=encoder_model,
        image_projection_model=image_projection_model,
    )


def _optimizer_labels(params):
    def label(path, _):
        keys = [getattr(item, "key", item) for item in path]
        if not keys or keys[0] != "encoder":
            return "main"
        # CamPose ACT builds its ResNet with FrozenBatchNorm2d. Flax stores the
        # equivalent affine values in the params collection, so keep those
        # leaves in the checkpoint tree but exclude them from AdamW updates.
        if any(str(key) == "BatchNorm_0" for key in keys):
            return "frozen"
        if len(keys) == 1:
            return "backbone"
        second = str(keys[1])
        if second == "resnet" or second.startswith("layers_"):
            return "backbone"
        if second == "film":
            return (
                "backbone"
                if len(keys) > 2 and str(keys[2]).startswith("layer_")
                else "main"
            )
        return "main"

    return jax.tree_util.tree_map_with_path(label, params)


def _identity_camera_param_batch(
    observation_space: spaces.Dict,
    keys: tuple[str, ...],
    *,
    size: int,
) -> jnp.ndarray:
    values = []
    eye = jnp.eye(size, dtype=jnp.float32)
    for key in keys:
        shape = tuple(int(dim) for dim in observation_space.spaces[key].shape)
        prefix = shape[:-2] or (1,)
        values.append(jnp.broadcast_to(eye, (1, *prefix, size, size)))
    return jnp.stack(values, axis=1)


def _normalize_camera_extrinsics(
    camera_extrinsics: jnp.ndarray,
    *,
    num_camera_extrinsics: int,
) -> jnp.ndarray:
    camera_extrinsics = camera_extrinsics.astype(jnp.float32)
    if camera_extrinsics.ndim == 5:
        # Direct observations may retain a leading frame-stack dimension.
        camera_extrinsics = camera_extrinsics[:, -1]
    if camera_extrinsics.ndim != 4 or camera_extrinsics.shape[-2:] != (4, 4):
        raise ValueError(
            "camera_extrinsics must have shape (batch, cameras, 4, 4) or "
            "(batch, time, cameras, 4, 4); got "
            f"{camera_extrinsics.shape}."
        )
    camera_extrinsics = camera_extrinsics[:, :num_camera_extrinsics]
    missing = int(num_camera_extrinsics) - int(camera_extrinsics.shape[1])
    if missing > 0:
        camera_extrinsics = jnp.pad(
            camera_extrinsics,
            ((0, 0), (0, missing), (0, 0), (0, 0)),
        )
    return camera_extrinsics


def _gaussian_kernel1d(sigma: float, radius: int) -> jnp.ndarray:
    offsets = jnp.arange(-int(radius), int(radius) + 1, dtype=jnp.float32)
    kernel = jnp.exp(-0.5 * jnp.square(offsets / float(sigma)))
    return kernel / jnp.sum(kernel)


def _depthwise_blur_2d(field: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    channels = int(field.shape[-1])
    kernel_y = jnp.broadcast_to(
        kernel[:, None, None, None],
        (kernel.shape[0], 1, 1, channels),
    )
    field = jax.lax.conv_general_dilated(
        field,
        kernel_y,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        feature_group_count=channels,
    )
    kernel_x = jnp.broadcast_to(
        kernel[None, :, None, None],
        (1, kernel.shape[0], 1, channels),
    )
    return jax.lax.conv_general_dilated(
        field,
        kernel_x,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        feature_group_count=channels,
    )


def _elastic_transform_chw(
    image: jnp.ndarray,
    displacement: jnp.ndarray,
) -> jnp.ndarray:
    channels, height, width = image.shape
    y, x = jnp.meshgrid(
        jnp.arange(height, dtype=jnp.float32),
        jnp.arange(width, dtype=jnp.float32),
        indexing="ij",
    )
    y = jnp.clip(y + displacement[..., 0], 0.0, float(height - 1))
    x = jnp.clip(x + displacement[..., 1], 0.0, float(width - 1))
    c = jnp.broadcast_to(
        jnp.arange(channels, dtype=jnp.float32)[:, None, None],
        image.shape,
    )
    return jndi.map_coordinates(
        image,
        (
            c,
            jnp.broadcast_to(y[None], image.shape),
            jnp.broadcast_to(x[None], image.shape),
        ),
        order=1,
        mode="nearest",
    )


def _apply_elastic_transform(
    flat: jnp.ndarray,
    rng_key,
    *,
    alpha: float = 80.0,
    sigma: float = 10.0,
) -> jnp.ndarray:
    apply_key, displacement_key = jax.random.split(rng_key)
    _, _, height, width = flat.shape
    apply_mask = jax.random.bernoulli(apply_key, 0.5)
    displacement = jax.random.uniform(
        displacement_key,
        (height, width, 2),
        minval=-1.0,
        maxval=1.0,
    )
    displacement = _depthwise_blur_2d(
        displacement[None],
        _gaussian_kernel1d(sigma=sigma, radius=int(round(3.0 * sigma))),
    )[0]
    displacement = displacement * float(alpha)
    warped = jax.vmap(_elastic_transform_chw, in_axes=(0, None))(flat, displacement)
    return jnp.where(apply_mask, warped, flat)


def _rgb_to_hsv(rgb: jnp.ndarray) -> jnp.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = jnp.max(rgb, axis=-1)
    minc = jnp.min(rgb, axis=-1)
    delta = maxc - minc
    safe_delta = jnp.where(delta > 1e-6, delta, 1.0)
    h_r = ((g - b) / safe_delta) % 6.0
    h_g = ((b - r) / safe_delta) + 2.0
    h_b = ((r - g) / safe_delta) + 4.0
    h = jnp.where(
        delta <= 1e-6,
        0.0,
        jnp.where(maxc == r, h_r, jnp.where(maxc == g, h_g, h_b)) / 6.0,
    )
    s = jnp.where(maxc <= 1e-6, 0.0, delta / maxc)
    return jnp.stack([h % 1.0, s, maxc], axis=-1)


def _hsv_to_rgb(hsv: jnp.ndarray) -> jnp.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    h6 = h * 6.0
    i = jnp.floor(h6).astype(jnp.int32)
    f = h6 - jnp.floor(h6)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i_mod = i % 6
    r = jnp.select(
        [i_mod == 0, i_mod == 1, i_mod == 2, i_mod == 3, i_mod == 4],
        [v, q, p, p, t],
        default=v,
    )
    g = jnp.select(
        [i_mod == 0, i_mod == 1, i_mod == 2, i_mod == 3, i_mod == 4],
        [t, v, v, q, p],
        default=p,
    )
    b = jnp.select(
        [i_mod == 0, i_mod == 1, i_mod == 2, i_mod == 3, i_mod == 4],
        [p, p, t, v, v],
        default=q,
    )
    return jnp.stack([r, g, b], axis=-1)


def _adjust_hue_chw(flat: jnp.ndarray, hue_delta: jnp.ndarray) -> jnp.ndarray:
    rgb = jnp.transpose(flat, (0, 2, 3, 1)) / 255.0
    hsv = _rgb_to_hsv(jnp.clip(rgb, 0.0, 1.0))
    h = (hsv[..., 0] + hue_delta.reshape((-1, 1, 1))) % 1.0
    rgb = _hsv_to_rgb(jnp.concatenate([h[..., None], hsv[..., 1:]], axis=-1))
    return jnp.transpose(rgb * 255.0, (0, 3, 1, 2))


class ACT(JaxMethodBase):
    """Chunked-action ACT method implemented entirely with JAX/Flax."""

    def __init__(
        self,
        lr: float,
        adaptive_lr: bool,
        num_train_steps: int,
        model: ACTModelSpec,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        num_train_envs: int,
        num_eval_envs: int,
        replay_alpha: float,
        replay_beta: float,
        frame_stack_on_channel: bool,
        intrinsic_reward_module=None,
        actor_grad_clip: Optional[float] = None,
        weight_decay: float = 1e-4,
        lr_backbone: float | None = None,
        horizon_dropout_lengths: tuple[int, ...] | None = None,
        horizon_dropout_probs: tuple[float, ...] | None = None,
        jit: bool = True,
        platform: str | None = None,
        seed: int = 0,
        is_rl: bool = False,
        use_ema: bool = False,
        update_block_every_steps: int = 1,
    ):
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

        self.model_spec = model
        self.actor_spec = model.actor_model
        self.lr_backbone = float(lr if lr_backbone is None else lr_backbone)
        self.data_augmentation = bool(self.actor_spec.data_augmentation)
        self.image_augmentation_type = str(
            self.actor_spec.image_augmentation_type
        ).lower()
        if self.image_augmentation_type not in {"legacy", "campose_crop"}:
            raise ValueError(
                "actor_model.image_augmentation_type must be 'legacy' or "
                f"'campose_crop', got {self.actor_spec.image_augmentation_type!r}."
            )
        self.use_camera_extrinsics = bool(self.actor_spec.use_camera_extrinsics)
        self.num_camera_extrinsics = int(self.actor_spec.num_camera_extrinsics)
        if self.num_camera_extrinsics < 1:
            raise ValueError("actor_model.num_camera_extrinsics must be >= 1.")
        if (
            self.actor_spec.image_position_max_tokens is not None
            and int(self.actor_spec.image_position_max_tokens) < 1
        ):
            raise ValueError("actor_model.image_position_max_tokens must be >= 1.")
        if not 0.0 <= float(self.actor_spec.proprio_dropout_prob) <= 1.0:
            raise ValueError(
                "actor_model.proprio_dropout_prob must be between 0 and 1."
            )
        if self.actor_spec.proprio_projection_type not in {
            "legacy_mlp",
            "campose_single",
        }:
            raise ValueError(
                "actor_model.proprio_projection_type must be 'legacy_mlp' or "
                f"'campose_single', got {self.actor_spec.proprio_projection_type!r}."
            )
        if self.actor_spec.style_cls_type not in {"learned", "zero"}:
            raise ValueError(
                "actor_model.style_cls_type must be 'learned' or 'zero', got "
                f"{self.actor_spec.style_cls_type!r}."
            )
        if self.actor_spec.decoder_output_layer not in {"final", "first"}:
            raise ValueError(
                "actor_model.decoder_output_layer must be 'final' or 'first', got "
                f"{self.actor_spec.decoder_output_layer!r}."
            )
        self._uses_plucker = bool(
            model.encoder_model is not None and model.encoder_model.use_plucker
        )

        def campose_augmentation(obs_inputs, rng_key):
            return augment_campose_observation(
                obs_inputs,
                rng_key,
                require_raymap=self._uses_plucker,
            )

        self._campose_augmentation = (
            jax.jit(campose_augmentation) if self._jit_enabled else campose_augmentation
        )
        # The legacy augmentation is a chain of dozens of elementwise ops; run
        # eagerly each dispatches its own kernel (measured 321 ms per
        # 128-sample batch on a 5090). Jitting fuses it like the campose path
        # above; the traced ops and RNG usage are unchanged, so outputs are
        # identical.
        self._legacy_rgb_augmentation = (
            jax.jit(self._augment_rgb_impl) if self._jit_enabled
            else self._augment_rgb_impl
        )
        self.use_lang_cond = bool(self.actor_spec.use_lang_cond)
        self.gripper_dims = int(self.actor_spec.gripper_dims)
        if self.gripper_dims < 0 or self.gripper_dims > self.action_dim:
            raise ValueError(
                "actor_model.gripper_dims must be between 0 and action_dim; "
                f"got {self.gripper_dims} for action_dim={self.action_dim}."
            )
        self.lang_feature_dim = (
            int(self.actor_spec.lang_feature_dim) if self.use_lang_cond else 0
        )
        self._horizon_dropout_lengths = None
        self._horizon_dropout_probs = None
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
        self._init_cached_pixel_feature_key("act")

        built_model = _build_model(
            self.model_spec,
            observation_space=observation_space,
            action_space=action_space,
            encoder_jit=jit,
            encoder_seed=seed,
        )
        self.actor_model = built_model.actor_model
        self.encoder = built_model.encoder_model
        self.image_projection = built_model.image_projection_model
        self._trainable_encoder = self.encoder is not None

        (
            actor_key,
            image_projection_key,
            dropout_key,
            latent_key,
            self.rng_key,
        ) = jax.random.split(self.rng_key, 5)

        params = {}
        image_features = None
        image_pos = None
        dummy_task_emb = (
            jnp.zeros((1, self.lang_feature_dim), dtype=jnp.float32)
            if self.use_lang_cond
            else None
        )
        if self.encoder is not None:
            dummy_rgb = jnp.zeros(
                (1, *self.obs_layout.rgb_input_shape),
                dtype=jnp.float32,
            )
            dummy_encoder_inputs = {}
            if self.model_spec.encoder_model.use_plucker:
                if self.obs_layout.raymap_input_shape is not None:
                    dummy_encoder_inputs["raymap_obs"] = jnp.zeros(
                        (1, *self.obs_layout.raymap_input_shape),
                        dtype=jnp.float32,
                    )
                elif (
                    self.obs_layout.camera_intrinsic_keys
                    and self.obs_layout.camera_c2w_keys
                ):
                    dummy_encoder_inputs["camera_intrinsic_obs"] = (
                        _identity_camera_param_batch(
                            observation_space,
                            self.obs_layout.camera_intrinsic_keys,
                            size=3,
                        )
                    )
                    dummy_encoder_inputs["camera_c2w_obs"] = (
                        _identity_camera_param_batch(
                            observation_space,
                            self.obs_layout.camera_c2w_keys,
                            size=4,
                        )
                    )
            if self.use_lang_cond:
                spatial_features, dummy_task_emb = self.encoder.apply_trainable_spatial(
                    self.encoder.trainable_params,
                    dummy_rgb,
                    **dummy_encoder_inputs,
                    task_emb=dummy_task_emb,
                    return_task_emb=True,
                )
            else:
                spatial_features = self.encoder.apply_trainable_spatial(
                    self.encoder.trainable_params,
                    dummy_rgb,
                    **dummy_encoder_inputs,
                )
            image_projection_params = self.image_projection.init(
                image_projection_key,
                spatial_features,
            )
            image_features, image_pos = self.image_projection.apply(
                image_projection_params,
                spatial_features,
            )
            params["encoder"] = self.encoder.trainable_params
            params["image_projection"] = image_projection_params

        dummy_qpos = jnp.zeros((1, self.low_dim_size), dtype=jnp.float32)
        dummy_actions = jnp.zeros(
            (1, self.action_sequence, self.action_dim),
            dtype=jnp.float32,
        )
        dummy_is_pad = jnp.zeros((1, self.action_sequence), dtype=jnp.bool_)
        dummy_camera_extrinsics = (
            jnp.zeros(
                (1, self.num_camera_extrinsics, 4, 4),
                dtype=jnp.float32,
            )
            if self.use_camera_extrinsics
            else None
        )
        params["actor"] = self.actor_model.init(
            {"params": actor_key, "dropout": dropout_key},
            image_features,
            image_pos,
            dummy_qpos,
            actions=dummy_actions,
            is_pad=dummy_is_pad,
            task_emb=dummy_task_emb,
            camera_extrinsics=dummy_camera_extrinsics,
            deterministic=False,
            latent_key=latent_key,
        )
        self.params = params

        learning_rate = lr
        if adaptive_lr:
            learning_rate = self.optax.cosine_decay_schedule(
                init_value=lr,
                decay_steps=self.num_train_steps,
            )
        transforms = []
        if actor_grad_clip is not None:
            transforms.append(self.optax.clip_by_global_norm(float(actor_grad_clip)))
        if self._trainable_encoder:
            backbone_learning_rate = (
                learning_rate if self.lr_backbone == float(lr) else self.lr_backbone
            )
            transforms.append(
                self.optax.multi_transform(
                    {
                        "main": self.optax.adamw(
                            learning_rate,
                            weight_decay=float(weight_decay),
                        ),
                        "backbone": self.optax.adamw(
                            backbone_learning_rate,
                            weight_decay=float(weight_decay),
                        ),
                        "frozen": self.optax.set_to_zero(),
                    },
                    _optimizer_labels,
                )
            )
        else:
            transforms.append(
                self.optax.adamw(learning_rate, weight_decay=float(weight_decay))
            )
        self.optimizer = self.optax.chain(*transforms)
        self.opt_state = self.optimizer.init(self.params)

        update_fn = self._build_update_fn()
        update_many_fn = self._build_update_many_fn(update_fn)
        predict_fn = self._predict_impl
        if self._jit_enabled:
            update_fn = jax.jit(update_fn)
            update_many_fn = jax.jit(update_many_fn)
            predict_fn = jax.jit(predict_fn)
        self._update_impl = update_fn
        self._update_many_impl = update_many_fn
        self._predict = predict_fn

    def _extract_lang_features(self, batch_or_obs: dict):
        if not self.use_lang_cond:
            return None
        if "lang_features" in batch_or_obs:
            features = self._as_jax_array(
                lang_feature_rows(batch_or_obs, context="Language-conditioned ACT"),
                self.jnp.float32,
            )
            if int(features.shape[-1]) != int(self.lang_feature_dim):
                raise ValueError(
                    "Language-conditioned ACT expected lang_features with final "
                    f"dimension {self.lang_feature_dim}, got {features.shape}."
                )
            return features
        if "lang_tokens" not in batch_or_obs:
            raise ValueError(
                "Language-conditioned ACT requires 'lang_features' or "
                "'lang_tokens' observations."
            )
        features = tokens_to_feature_jax(
            self._as_jax_array(
                lang_token_rows(batch_or_obs, context="Language-conditioned ACT"),
                self.jnp.float32,
            ),
            feature_dim=self.lang_feature_dim,
        )
        return self.jnp.asarray(features, dtype=self.jnp.float32)

    def _prepare_trainable_obs_inputs(self, batch_or_obs: dict):
        inputs = {}
        low_dim_obs = self._extract_low_dim_batch(batch_or_obs)
        if self.low_dim_size > 0 and low_dim_obs is None:
            raise ValueError("ACT requires low_dim_state observations.")
        if low_dim_obs is not None:
            inputs["low_dim"] = low_dim_obs
        if self.use_pixels:
            if self._has_cached_pixel_features(batch_or_obs):
                raise ValueError(
                    "Reference-style JAX ACT requires raw RGB observations; "
                    "disable replay.cache_frozen_image_features."
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
        if self.use_camera_extrinsics:
            direct_extrinsics = None
            for key in ("camera_extrinsics", "cam_extrinsics"):
                if key in batch_or_obs:
                    direct_extrinsics = self._as_jax_array(
                        batch_or_obs[key], self.jnp.float32
                    )
                    break
            if direct_extrinsics is None and "camera_c2w" in inputs:
                direct_extrinsics = inputs["camera_c2w"][:, :, -1]
            if direct_extrinsics is None:
                batch_size = self._batch_size_from_inputs(inputs)
                direct_extrinsics = self.jnp.zeros(
                    (batch_size, 0, 4, 4), dtype=self.jnp.float32
                )
            inputs["camera_extrinsics"] = _normalize_camera_extrinsics(
                direct_extrinsics,
                num_camera_extrinsics=self.num_camera_extrinsics,
            )
        return inputs

    def _batch_size_from_inputs(self, obs_inputs: dict) -> int:
        if not obs_inputs:
            raise ValueError("ACT requires at least one observation feature.")
        return int(self.jax.tree_util.tree_leaves(obs_inputs)[0].shape[0])

    def _policy_inputs_from_obs(self, params, obs_inputs: dict):
        batch_size = self._batch_size_from_inputs(obs_inputs)
        qpos = obs_inputs.get("low_dim", None)
        if qpos is None:
            qpos = self.jnp.zeros(
                (batch_size, self.low_dim_size), dtype=self.jnp.float32
            )

        image_features = None
        image_pos = None
        task_emb = obs_inputs.get("lang", None)
        if "rgb" in obs_inputs:
            if self.use_lang_cond:
                spatial_features, task_emb = self.encoder.apply_trainable_spatial(
                    params["encoder"],
                    obs_inputs["rgb"],
                    raymap_obs=obs_inputs.get("raymap", None),
                    camera_intrinsic_obs=obs_inputs.get("camera_intrinsic", None),
                    camera_c2w_obs=obs_inputs.get("camera_c2w", None),
                    task_emb=task_emb,
                    return_task_emb=True,
                )
            else:
                spatial_features = self.encoder.apply_trainable_spatial(
                    params["encoder"],
                    obs_inputs["rgb"],
                    raymap_obs=obs_inputs.get("raymap", None),
                    camera_intrinsic_obs=obs_inputs.get("camera_intrinsic", None),
                    camera_c2w_obs=obs_inputs.get("camera_c2w", None),
                )
            image_features, image_pos = self.image_projection.apply(
                params["image_projection"],
                spatial_features,
            )
        return (
            image_features,
            image_pos,
            qpos,
            task_emb,
            obs_inputs.get("camera_extrinsics", None),
        )

    def _augment_rgb(self, rgb: jnp.ndarray, rng_key) -> jnp.ndarray:
        if not self.data_augmentation or rgb.shape[2] != 3:
            return rgb
        return self._legacy_rgb_augmentation(rgb, rng_key)

    def _augment_rgb_impl(self, rgb: jnp.ndarray, rng_key) -> jnp.ndarray:
        batch_size, num_views, channels, height, width = rgb.shape
        elastic_key, color_key, crop_key, noise_key = jax.random.split(rng_key, 4)
        pad = 4

        flat = rgb.reshape((batch_size * num_views, channels, height, width)).astype(
            jnp.float32
        )
        flat = _apply_elastic_transform(flat, elastic_key)

        (
            color_apply_key,
            color_order_key,
            brightness_key,
            contrast_key,
            saturation_key,
            hue_key,
        ) = jax.random.split(color_key, 6)
        color_mask = jax.random.bernoulli(color_apply_key, 0.5)
        brightness = jax.random.uniform(
            brightness_key,
            (),
            minval=0.8,
            maxval=1.2,
        )
        contrast = jax.random.uniform(
            contrast_key,
            (),
            minval=0.8,
            maxval=1.2,
        )
        saturation = jax.random.uniform(
            saturation_key,
            (),
            minval=0.9,
            maxval=1.1,
        )
        hue_delta = jax.random.uniform(
            hue_key,
            (),
            minval=-0.05,
            maxval=0.05,
        )
        grayscale_weights = jnp.asarray(
            [0.2989, 0.5870, 0.1140],
            dtype=flat.dtype,
        ).reshape((1, 3, 1, 1))

        def adjust_brightness(image):
            return image * brightness

        def adjust_contrast(image):
            gray_mean = (image * grayscale_weights).sum(axis=1, keepdims=True)
            gray_mean = gray_mean.mean(axis=(2, 3), keepdims=True)
            return (image - gray_mean) * contrast + gray_mean

        def adjust_saturation(image):
            gray = (image * grayscale_weights).sum(axis=1, keepdims=True)
            return (image - gray) * saturation + gray

        def adjust_hue(image):
            return _adjust_hue_chw(
                jnp.clip(image, 0.0, 255.0),
                jnp.full((image.shape[0], 1), hue_delta, dtype=image.dtype),
            )

        order = jax.random.permutation(color_order_key, jnp.arange(4))
        color = flat
        for order_idx in range(4):
            color = jax.lax.switch(
                order[order_idx],
                (
                    adjust_brightness,
                    adjust_contrast,
                    adjust_saturation,
                    adjust_hue,
                ),
                color,
            )
        flat = jnp.where(color_mask, color, flat)

        padded = jnp.pad(
            flat,
            ((0, 0), (0, 0), (pad, pad), (pad, pad)),
            mode="constant",
        )
        crop_apply_key, crop_y_key, crop_x_key = jax.random.split(crop_key, 3)
        crop_apply = jax.random.bernoulli(crop_apply_key, 0.5)
        crop_y = jax.random.randint(crop_y_key, (), 0, 2 * pad + 1)
        crop_x = jax.random.randint(crop_x_key, (), 0, 2 * pad + 1)

        def crop_one(image, y_offset, x_offset, should_crop):
            cropped = jax.lax.dynamic_slice(
                image,
                (0, y_offset, x_offset),
                (channels, height, width),
            )
            center = image[:, pad : pad + height, pad : pad + width]
            return jnp.where(should_crop, cropped, center)

        flat = jax.vmap(crop_one, in_axes=(0, None, None, None))(
            padded,
            crop_y,
            crop_x,
            crop_apply,
        )
        flat = flat + jax.random.normal(noise_key, flat.shape) * 5.0
        flat = jnp.clip(flat, 0.0, 255.0)
        return flat.reshape((batch_size, num_views, channels, height, width))

    def _augment_observation_images(self, obs_inputs: dict, rng_key) -> dict:
        if not self.data_augmentation or "rgb" not in obs_inputs:
            return obs_inputs
        if self.image_augmentation_type == "legacy":
            return {
                **obs_inputs,
                "rgb": self._augment_rgb(obs_inputs["rgb"], rng_key),
            }

        return self._campose_augmentation(obs_inputs, rng_key)

    def _predict_impl(self, params, obs_inputs):
        (
            image_features,
            image_pos,
            qpos,
            task_emb,
            camera_extrinsics,
        ) = self._policy_inputs_from_obs(params, obs_inputs)
        action_pred, _, _ = self.actor_model.apply(
            params["actor"],
            image_features,
            image_pos,
            qpos,
            task_emb=task_emb,
            camera_extrinsics=camera_extrinsics,
            deterministic=True,
        )
        return action_pred

    def _build_update_fn(self):
        optimizer = self.optimizer
        optax = self.optax
        kl_weight = float(self.actor_spec.kl_weight)
        gripper_loss_weight = float(self.actor_spec.gripper_loss_weight)
        gripper_dims = self.gripper_dims
        horizon_dropout_lengths = self._horizon_dropout_lengths
        horizon_dropout_probs = self._horizon_dropout_probs

        def masked_mean_per_sample(values, valid_mask):
            valid = valid_mask.astype(values.dtype)
            while valid.ndim < values.ndim:
                valid = valid[..., None]
            return (values * valid).mean(axis=tuple(range(1, values.ndim)))

        def update_fn(
            params,
            opt_state,
            obs_inputs,
            actions,
            loss_coeff,
            action_pad_mask,
            dropout_key,
            latent_key,
            horizon_key,
        ):
            effective_action_pad_mask = action_pad_mask
            if horizon_dropout_lengths is not None:
                length_indices = jax.random.choice(
                    horizon_key,
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

            def loss_fn(current_params):
                (
                    image_features,
                    image_pos,
                    qpos,
                    task_emb,
                    camera_extrinsics,
                ) = self._policy_inputs_from_obs(current_params, obs_inputs)
                action_pred, mu, logvar = self.actor_model.apply(
                    current_params["actor"],
                    image_features,
                    image_pos,
                    qpos,
                    actions=actions,
                    is_pad=effective_action_pad_mask,
                    task_emb=task_emb,
                    camera_extrinsics=camera_extrinsics,
                    deterministic=False,
                    latent_key=latent_key,
                    rngs={"dropout": dropout_key},
                )

                valid_mask = (
                    jnp.ones(actions.shape[:2], dtype=jnp.bool_)
                    if effective_action_pad_mask is None
                    else jnp.logical_not(effective_action_pad_mask)
                )
                continuous_dim = actions.shape[-1] - gripper_dims
                l1_per_sample = jnp.zeros((actions.shape[0],), dtype=actions.dtype)
                if continuous_dim > 0:
                    l1_values = jnp.abs(
                        action_pred[..., :continuous_dim]
                        - actions[..., :continuous_dim]
                    )
                    l1_per_sample = masked_mean_per_sample(l1_values, valid_mask)

                gripper_per_sample = jnp.zeros((actions.shape[0],), dtype=actions.dtype)
                if continuous_dim < actions.shape[-1]:
                    gripper_loss = optax.sigmoid_binary_cross_entropy(
                        action_pred[..., continuous_dim:],
                        actions[..., continuous_dim:],
                    )
                    gripper_per_sample = gripper_loss_weight * masked_mean_per_sample(
                        gripper_loss, valid_mask
                    )

                behavior_per_sample = l1_per_sample + gripper_per_sample
                behavior_loss = (behavior_per_sample * loss_coeff).mean()

                if mu is None or logvar is None:
                    kl_loss = jnp.asarray(0.0, dtype=actions.dtype)
                else:
                    kl_loss = -0.5 * (1.0 + logvar - jnp.square(mu) - jnp.exp(logvar))
                    kl_loss = kl_loss.sum(axis=-1).mean()

                total_loss = behavior_loss + kl_weight * kl_loss
                l1_loss = (l1_per_sample * loss_coeff).mean()
                gripper_loss = (gripper_per_sample * loss_coeff).mean()
                return total_loss, (
                    behavior_per_sample,
                    l1_loss,
                    gripper_loss,
                    kl_loss,
                )

            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            behavior_per_sample, l1_loss, gripper_loss, kl_loss = aux
            updates, new_opt_state = optimizer.update(
                grads,
                opt_state,
                params,
            )
            new_params = optax.apply_updates(params, updates)
            new_pri = behavior_per_sample + 1e-6
            max_pri = jnp.max(new_pri)
            return (
                new_params,
                new_opt_state,
                loss,
                new_pri / jnp.where(max_pri > 0, max_pri, 1.0),
                l1_loss,
                gripper_loss,
                kl_loss,
            )

        return update_fn

    def _build_update_many_fn(self, update_fn):
        def update_many_fn(
            params,
            opt_state,
            obs_inputs,
            actions,
            loss_coeff,
            action_pad_mask,
            dropout_keys,
            latent_keys,
            horizon_keys,
        ):
            def body_fn(carry, xs):
                current_params, current_opt_state = carry
                if action_pad_mask is None:
                    (
                        step_obs_inputs,
                        step_actions,
                        step_loss_coeff,
                        step_dropout_key,
                        step_latent_key,
                        step_horizon_key,
                    ) = xs
                    step_action_pad_mask = None
                else:
                    (
                        step_obs_inputs,
                        step_actions,
                        step_loss_coeff,
                        step_action_pad_mask,
                        step_dropout_key,
                        step_latent_key,
                        step_horizon_key,
                    ) = xs
                (
                    next_params,
                    next_opt_state,
                    loss,
                    priority,
                    l1_loss,
                    gripper_loss,
                    kl_loss,
                ) = update_fn(
                    current_params,
                    current_opt_state,
                    step_obs_inputs,
                    step_actions,
                    step_loss_coeff,
                    step_action_pad_mask,
                    step_dropout_key,
                    step_latent_key,
                    step_horizon_key,
                )
                return (next_params, next_opt_state), (
                    loss,
                    priority,
                    l1_loss,
                    gripper_loss,
                    kl_loss,
                )

            xs = (
                obs_inputs,
                actions,
                loss_coeff,
                dropout_keys,
                latent_keys,
                horizon_keys,
            )
            if action_pad_mask is not None:
                xs = (
                    obs_inputs,
                    actions,
                    loss_coeff,
                    action_pad_mask,
                    dropout_keys,
                    latent_keys,
                    horizon_keys,
                )
            (
                (
                    new_params,
                    new_opt_state,
                ),
                (
                    losses,
                    priorities,
                    l1_losses,
                    gripper_losses,
                    kl_losses,
                ),
            ) = jax.lax.scan(
                body_fn,
                (params, opt_state),
                xs,
            )
            return (
                new_params,
                new_opt_state,
                losses[-1],
                priorities[-1],
                l1_losses[-1],
                gripper_losses[-1],
                kl_losses[-1],
            )

        return update_many_fn

    def act(self, observations: dict, step: int, eval_mode: bool):
        del step, eval_mode
        obs_inputs = self._prepare_trainable_obs_inputs(observations)
        actions = self._predict(self.params, obs_inputs)
        self._block(actions)
        return np.asarray(jax.device_get(actions), dtype=np.float32)

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
        obs_inputs = self._prepare_trainable_obs_inputs(batch)
        metrics = {}

        self.rng_key, dropout_key, latent_key, aug_key, horizon_key = jax.random.split(
            self.rng_key,
            5,
        )
        obs_inputs = self._augment_observation_images(obs_inputs, aug_key)
        start_time = time.perf_counter()
        (
            self.params,
            self.opt_state,
            actor_loss,
            new_priority,
            l1_loss,
            gripper_loss,
            kl_loss,
        ) = self._update_impl(
            self.params,
            self.opt_state,
            obs_inputs,
            actions,
            loss_coeff,
            action_pad_mask,
            dropout_key,
            latent_key,
            horizon_key,
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
        if self.logging:
            metrics["actor_l1_loss"] = float(np.asarray(jax.device_get(l1_loss)))
            metrics["actor_gripper_loss"] = float(
                np.asarray(jax.device_get(gripper_loss))
            )
            metrics["actor_kl_loss"] = float(np.asarray(jax.device_get(kl_loss)))
        self._first_update_completed = True
        return metrics

    def update_many(
        self,
        replay_iter: Iterator[dict],
        num_updates: int,
        replay_buffer: ReplayBuffer = None,
    ) -> dict[str, np.ndarray]:
        num_updates = int(num_updates)
        uses_priorities = self._uses_replay_priorities(replay_buffer)
        if num_updates <= 1 or uses_priorities:
            metrics = {}
            for _ in range(max(num_updates, 1)):
                metrics.update(self.update(replay_iter, 0, replay_buffer))
            return metrics

        key_count_per_update = 4
        split_keys = jax.random.split(
            self.rng_key,
            1 + num_updates * key_count_per_update,
        )
        self.rng_key = split_keys[0]
        update_keys = split_keys[1:].reshape(
            (num_updates, key_count_per_update, *split_keys.shape[1:])
        )
        dropout_keys = update_keys[:, 0]
        latent_keys = update_keys[:, 1]
        horizon_keys = update_keys[:, 2]
        aug_keys = update_keys[:, 3]

        obs_inputs_list = []
        actions = []
        loss_coeffs = []
        action_pad_masks = []
        has_action_pad_mask = None
        for update_idx in range(num_updates):
            batch = next(replay_iter)
            actions.append(self._as_jax_array(batch["action"], self.jnp.float32))
            loss_coeffs.append(self._loss_weights(batch))
            obs = self._prepare_trainable_obs_inputs(batch)
            obs = self._augment_observation_images(obs, aug_keys[update_idx])
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

        start_time = time.perf_counter()
        (
            self.params,
            self.opt_state,
            actor_loss,
            new_priority,
            l1_loss,
            gripper_loss,
            kl_loss,
        ) = self._update_many_impl(
            self.params,
            self.opt_state,
            obs_inputs,
            actions,
            loss_coeffs,
            action_pad_mask,
            dropout_keys,
            latent_keys,
            horizon_keys,
        )
        if (
            self.logging
            or (self._update_step_count + num_updates) % self._update_block_every_steps
            == 0
        ):
            self._block(actor_loss, new_priority)
        elapsed = time.perf_counter() - start_time
        self._update_step_count += num_updates

        metrics = {}
        self._maybe_log_update_metrics(metrics, actor_loss, obs_inputs, elapsed)
        if self.logging:
            metrics["actor_l1_loss"] = float(np.asarray(jax.device_get(l1_loss)))
            metrics["actor_gripper_loss"] = float(
                np.asarray(jax.device_get(gripper_loss))
            )
            metrics["actor_kl_loss"] = float(np.asarray(jax.device_get(kl_loss)))
        self._first_update_completed = True
        return metrics

    def load_checkpoint_state_dict(self, state_dict: dict[str, dict]):
        opt_state = state_dict.get("opt_state", None)
        if isinstance(opt_state, dict) and "main" in opt_state:
            state_dict = {**state_dict, "opt_state": opt_state["main"]}
        super().load_checkpoint_state_dict(state_dict)


__all__ = [
    "ACTActorModelSpec",
    "ACTModelSpec",
    "ACTSpec",
    "ACT",
    "act_model_spec_from_cfg",
    "act_spec_from_cfg",
]
