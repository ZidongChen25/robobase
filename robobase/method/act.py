"""PyTorch ACT imitation-learning method matched to mobile-genima-bigym."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from omegaconf import DictConfig

from robobase.method.bc import BCEncoderModelSpec, BCViewFusionModelSpec
from robobase.method.bc_runtime import bc_observation_layout
from robobase.method.core import Method
from robobase.replay_buffer.replay_buffer import ReplayBuffer


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


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
    kl_weight: float = 10.0
    latent_dim: int = 32
    data_augmentation: bool = True
    use_lang_cond: bool = False


@dataclass(frozen=True)
class ACTModelSpec:
    actor_model: ACTActorModelSpec
    encoder_model: BCEncoderModelSpec | None
    view_fusion_model: BCViewFusionModelSpec | None


@dataclass(frozen=True)
class ACTSpec:
    lr: float
    adaptive_lr: bool
    num_train_steps: int
    actor_grad_clip: float | None
    weight_decay: float
    lr_backbone: float
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
            "robobase.models.multi_view_transformer.MultiViewTransformerEncoderDecoderACT": "transformer",
            "robobase.models.act.JaxACTTransformer": "transformer",
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
        kl_weight=float(actor_model_cfg.get("kl_weight", method_cfg.get("kl_weight", 10.0))),
        latent_dim=int(actor_model_cfg.get("latent_dim", 32)),
        data_augmentation=bool(actor_model_cfg.get("data_augmentation", True)),
        use_lang_cond=bool(
            actor_model_cfg.get("use_lang_cond", method_cfg.get("use_lang_cond", False))
        ),
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
            type=encoder_model_type,
            model=str(encoder_model_cfg.get("model", encoder_model_cfg.get("backbone", "resnet18"))),
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
    return ACTSpec(
        lr=float(cfg.method.lr),
        adaptive_lr=bool(cfg.method.adaptive_lr),
        num_train_steps=int(cfg.method.num_train_steps),
        actor_grad_clip=(
            None
            if cfg.method.actor_grad_clip is None
            else float(cfg.method.actor_grad_clip)
        ),
        weight_decay=float(cfg.method.get("weight_decay", 1e-4)),
        lr_backbone=float(cfg.method.get("lr_backbone", 1e-5)),
        model=act_model_spec_from_cfg(cfg),
    )


def _kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return klds.sum(1).mean(0, keepdim=True)[0]


def _get_sinusoid_encoding_table(n_position: int, d_hid: int) -> torch.Tensor:
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def _build_2d_sincos_pos_embed(
    hidden_dim: int, height: int, width: int, *, device: torch.device
) -> torch.Tensor:
    if hidden_dim % 2 != 0:
        return torch.zeros((1, hidden_dim, height, width), device=device)
    not_mask = torch.ones((1, height, width), dtype=torch.float32, device=device)
    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    x_embed = not_mask.cumsum(2, dtype=torch.float32)
    eps = 1e-6
    y_embed = y_embed / (y_embed[:, -1:, :] + eps) * (2 * math.pi)
    x_embed = x_embed / (x_embed[:, :, -1:] + eps) * (2 * math.pi)

    num_pos_feats = hidden_dim // 2
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=device)
    dim_t = 10000 ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack(
        (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
    ).flatten(3)
    pos_y = torch.stack(
        (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
    ).flatten(3)
    return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class AddGaussianNoise:
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = float(mean)
        self.std = float(std)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn_like(tensor, dtype=torch.float32) * self.std + self.mean


def _conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, dilation: int = 1
) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        bias=False,
        dilation=dilation,
    )


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class _FilmBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        norm_layer: type[nn.Module] | None = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(
        self, x: torch.Tensor, film_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if film_features is not None:
            gamma = film_features[:, 0].reshape(x.shape[0], -1, 1, 1)
            beta = film_features[:, 1].reshape(x.shape[0], -1, 1, 1)
            out = (1.0 + gamma) * out + beta
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class _FilmResNet18Backbone(nn.Module):
    num_channels = 512
    layers = (2, 2, 2, 2)
    film_planes = (64, 128, 256, 512)

    def __init__(self, *, task_embedding_dim: int, pretrained: bool = True):
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights
            from torchvision.ops.misc import FrozenBatchNorm2d
        except ImportError as exc:
            raise ImportError("Language-conditioned ACT requires torchvision.") from exc

        norm_layer = FrozenBatchNorm2d
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, self.layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(128, self.layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(256, self.layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(512, self.layers[3], stride=2, norm_layer=norm_layer)
        # Match the reference ResNet-FiLM constructor: build the classifier head
        # and run the ResNet init pass before loading ImageNet weights. The head is
        # replaced afterwards, but its initialization advances the RNG before FiLM
        # and ACT projection layers are initialized.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1000)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            state_dict = weights.get_state_dict(progress=True)
            self.load_state_dict(state_dict, strict=False)
        self.avgpool = nn.Identity()
        self.fc = nn.Identity()
        self.film_models = nn.ModuleDict(
            {
                str(layer_idx): nn.Linear(
                    task_embedding_dim,
                    self.layers[layer_idx] * 2 * self.film_planes[layer_idx],
                )
                for layer_idx in (1, 2, 3)
            }
        )

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int = 1,
        norm_layer: type[nn.Module] | None = None,
    ) -> nn.ModuleList:
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes, stride),
                norm_layer(planes),
            )
        layers = [_FilmBasicBlock(self.inplanes, planes, stride, downsample, norm_layer)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(_FilmBasicBlock(self.inplanes, planes, norm_layer=norm_layer))
        return nn.ModuleList(layers)

    def _film_for_layer(
        self, task_emb: torch.Tensor | None, layer_idx: int
    ) -> list[torch.Tensor | None]:
        layer_key = str(layer_idx)
        film_model = self.film_models[layer_key] if layer_key in self.film_models else None
        if task_emb is None or film_model is None:
            return [None] * self.layers[layer_idx]
        planes = self.film_planes[layer_idx]
        film = film_model(task_emb).view(task_emb.shape[0], 2, self.layers[layer_idx], planes)
        return [film[:, :, block_idx] for block_idx in range(self.layers[layer_idx])]

    def forward(self, x: torch.Tensor, task_emb: torch.Tensor | None) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        for layer_idx, layer in enumerate(
            [self.layer1, self.layer2, self.layer3, self.layer4]
        ):
            film_features = self._film_for_layer(task_emb, layer_idx)
            for block, block_film in zip(layer, film_features):
                x = block(x, film_features=block_film)
        return x


class ImageEncoderACT(nn.Module):
    """Trainable ResNet image encoder used by the reference PyTorch ACT."""

    def __init__(
        self,
        input_shape: tuple[int, int, int, int],
        hidden_dim: int = 512,
        backbone: str = "resnet18",
        pretrained: bool = True,
        use_lang_cond: bool = False,
    ):
        super().__init__()
        if len(input_shape) != 4:
            raise ValueError(f"Expected image input shape (V,C,H,W), got {input_shape}.")
        if backbone != "resnet18":
            raise NotImplementedError("ACT currently supports only resnet18.")
        self.input_shape = tuple(int(v) for v in input_shape)
        self.hidden_dim = int(hidden_dim)
        self.use_lang_cond = bool(use_lang_cond)
        self._output_shape = None
        try:
            from torchvision.models import ResNet18_Weights, resnet18
            from torchvision.ops.misc import FrozenBatchNorm2d
        except ImportError as exc:
            raise ImportError("PyTorch ACT requires torchvision.") from exc

        if self.use_lang_cond:
            self.backbone = _FilmResNet18Backbone(
                task_embedding_dim=self.hidden_dim,
                pretrained=pretrained,
            )
        else:
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = resnet18(weights=weights, norm_layer=FrozenBatchNorm2d)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.input_proj = nn.Conv2d(512, self.hidden_dim, kernel_size=1)
        if self.use_lang_cond:
            self.proj_text_emb = nn.Linear(self.backbone.num_channels, self.hidden_dim)
        mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    @property
    def output_shape(self):
        if self._output_shape is None:
            batch_size = 1
            device = next(self.backbone.parameters()).device
            x = torch.randn((batch_size, *self.input_shape)).to(device)
            task_emb = None
            if self.use_lang_cond:
                task_emb = torch.randn((batch_size, self.backbone.num_channels)).to(
                    device
                )
            with torch.no_grad():
                encoded = self.forward(x, task_emb=task_emb)
            if self.use_lang_cond:
                feat, pos, task_emb = encoded
                self._output_shape = (feat[0].shape, pos[0].shape, task_emb[0].shape)
            else:
                feat, pos = encoded
                self._output_shape = (feat[0].shape, pos[0].shape, None)
        return self._output_shape

    def forward(
        self, image: torch.Tensor, task_emb: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_views, channels, height, width = image.shape
        if channels % 3 != 0:
            raise ValueError(f"Expected image channels divisible by 3, got {channels}.")
        frame_stack = channels // 3
        if self.use_lang_cond:
            if task_emb is None:
                raise ValueError("Language-conditioned ACT requires lang_tokens/task_emb.")
            task_emb = self.proj_text_emb(task_emb)
        x = image.reshape(batch_size, num_views * frame_stack, 3, height, width)
        x = x.float() / 255.0
        x = (x - self.mean) / self.std
        x = x.reshape(batch_size * num_views * frame_stack, 3, height, width)
        if self.use_lang_cond:
            expanded_task = task_emb.repeat_interleave(num_views * frame_stack, dim=0)
            feat = self.backbone(x, expanded_task)
        else:
            feat = self.backbone(x)
        feat = self.input_proj(feat)
        _, hidden_dim, feat_h, feat_w = feat.shape
        feat = feat.reshape(batch_size, num_views, frame_stack, hidden_dim, feat_h, feat_w)
        feat = feat.mean(dim=2)
        feat = torch.cat([feat[:, view_idx] for view_idx in range(num_views)], dim=3)
        pos_single = _build_2d_sincos_pos_embed(
            hidden_dim, feat_h, feat_w, device=feat.device
        )
        pos = torch.cat([pos_single for _ in range(num_views)], dim=3)
        pos = pos.expand(batch_size, -1, -1, -1)
        if self.use_lang_cond:
            return feat, pos, task_emb
        return feat, pos


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def _with_pos_embed(tensor: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
    return tensor if pos is None else tensor + pos


class ACTTransformerEncoderLayer(nn.Module):
    """DETR-style encoder layer used by Mobile-GENIMA ACT."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        norm_first: bool,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm_first = bool(norm_first)

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        pos: torch.Tensor | None,
    ) -> torch.Tensor:
        q = k = _with_pos_embed(x, pos)
        x = self.self_attn(
            q,
            k,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.dropout2(x)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, pos
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, src_mask, src_key_padding_mask, pos)
            )
            x = self.norm2(x + self._ff_block(x))
        return x


class ACTTransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer: ACTTransformerEncoderLayer,
        num_layers: int,
        norm: nn.Module | None = None,
    ):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm = norm

    def forward(
        self,
        src: torch.Tensor,
        mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class ACTTransformerDecoderLayer(nn.Module):
    """DETR-style decoder layer that injects query/pos every layer."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        norm_first: bool,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm_first = bool(norm_first)

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        query_pos: torch.Tensor | None,
    ) -> torch.Tensor:
        q = k = _with_pos_embed(x, query_pos)
        x = self.self_attn(
            q,
            k,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    def _mha_block(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        pos: torch.Tensor | None,
        query_pos: torch.Tensor | None,
    ) -> torch.Tensor:
        x = self.multihead_attn(
            _with_pos_embed(x, query_pos),
            _with_pos_embed(memory, pos),
            memory,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.dropout3(x)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), tgt_mask, tgt_key_padding_mask, query_pos
            )
            x = x + self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, query_pos)
            )
            x = self.norm2(
                x
                + self._mha_block(
                    x,
                    memory,
                    memory_mask,
                    memory_key_padding_mask,
                    pos,
                    query_pos,
                )
            )
            x = self.norm3(x + self._ff_block(x))
        return x


class ACTTransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer: ACTTransformerDecoderLayer,
        num_layers: int,
        norm: nn.Module | None = None,
        return_intermediate: bool = True,
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output) if self.norm is not None else output)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output.unsqueeze(0)


class ACTDetrTransformer(nn.Module):
    """Mobile-GENIMA/DETR transformer block for ACT decoding."""

    def __init__(
        self,
        hidden_dim: int,
        nheads: int,
        enc_layers: int,
        dec_layers: int,
        dim_feedforward: int,
        dropout: float,
        pre_norm: bool,
    ):
        super().__init__()
        encoder_layer = ACTTransformerEncoderLayer(
            hidden_dim, nheads, dim_feedforward, dropout, pre_norm
        )
        encoder_norm = nn.LayerNorm(hidden_dim) if pre_norm else None
        self.encoder = ACTTransformerEncoder(encoder_layer, enc_layers, encoder_norm)
        decoder_layer = ACTTransformerDecoderLayer(
            hidden_dim, nheads, dim_feedforward, dropout, pre_norm
        )
        self.decoder = ACTTransformerDecoder(
            decoder_layer,
            dec_layers,
            nn.LayerNorm(hidden_dim),
            return_intermediate=True,
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(
        self,
        src: torch.Tensor,
        mask: torch.Tensor | None,
        query_embed: torch.Tensor,
        pos_embed: torch.Tensor,
        latent_input: torch.Tensor | None = None,
        proprio_input: torch.Tensor | None = None,
        additional_pos_embed: torch.Tensor | None = None,
        task_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if src.dim() == 4:
            batch_size = src.shape[0]
            src = src.flatten(2).permute(2, 0, 1)
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
            query_embed = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
            if (
                latent_input is None
                or proprio_input is None
                or additional_pos_embed is None
            ):
                raise ValueError("Image ACT transformer requires latent/proprio inputs.")
            additional_pos_embed = additional_pos_embed.unsqueeze(1).repeat(
                1, batch_size, 1
            )
            pos_embed = torch.cat([additional_pos_embed, pos_embed], dim=0)
            addition_inputs = [latent_input, proprio_input]
            if task_emb is not None:
                addition_inputs.append(task_emb)
            addition_input = torch.stack(addition_inputs, dim=0)
            src = torch.cat([addition_input, src], dim=0)
        elif src.dim() == 3:
            batch_size = src.shape[1]
            query_embed = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
        else:
            raise ValueError(f"Unexpected transformer source shape {tuple(src.shape)}.")

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        output = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
        )
        return output.transpose(1, 2)


class MultiViewTransformerEncoderDecoderACT(nn.Module):
    """Reference-style ACT CVAE transformer."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        nheads: int,
        dim_feedforward: int,
        enc_layers: int,
        dec_layers: int,
        pre_norm: bool,
        state_dim: int,
        action_dim: int,
        num_queries: int,
        kl_weight: float,
        latent_dim: int,
        use_lang_cond: bool = False,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.num_queries = int(num_queries)
        self.kl_weight = float(kl_weight)
        self.latent_dim = int(latent_dim)
        self.use_lang_cond = bool(use_lang_cond)

        style_encoder_layer = ACTTransformerEncoderLayer(
            hidden_dim,
            nheads,
            dim_feedforward,
            dropout,
            pre_norm,
        )
        style_encoder_norm = nn.LayerNorm(hidden_dim) if pre_norm else None
        self.style_encoder = ACTTransformerEncoder(
            style_encoder_layer,
            enc_layers,
            style_encoder_norm,
        )
        self.transformer = ACTDetrTransformer(
            hidden_dim=hidden_dim,
            nheads=nheads,
            enc_layers=enc_layers,
            dec_layers=dec_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pre_norm=pre_norm,
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Match Mobile-GENIMA's initialization order: construct the original
        # single-layer robot-state projection, then replace it with the Genima
        # two-layer projection after all actor submodules have been initialized.
        self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, latent_dim * 2)
        self.register_buffer(
            "pos_table",
            _get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim),
        )
        self.latent_out_proj = nn.Linear(latent_dim, hidden_dim)
        self.additional_pos_embed = nn.Embedding(3 if self.use_lang_cond else 2, hidden_dim)
        self.input_proj_robot_state = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _style_variable_encoder(
        self, actions: torch.Tensor, qpos: torch.Tensor, is_pad: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = qpos.shape[0]
        action_embed = self.encoder_action_proj(actions)
        qpos_embed = self.encoder_joint_proj(qpos).unsqueeze(1)
        cls_embed = self.cls_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], dim=1)
        encoder_input = encoder_input.permute(1, 0, 2)
        cls_joint_is_pad = torch.zeros(
            (batch_size, 2), dtype=torch.bool, device=qpos.device
        )
        style_is_pad = torch.cat([cls_joint_is_pad, is_pad], dim=1)
        pos_embed = self.pos_table[:, : encoder_input.shape[0]].permute(1, 0, 2)
        encoder_output = self.style_encoder(
            encoder_input,
            src_key_padding_mask=style_is_pad,
            pos=pos_embed,
        )[0]
        latent_info = self.latent_proj(encoder_output)
        mu = latent_info[:, : self.latent_dim]
        logvar = latent_info[:, self.latent_dim :]
        std = (logvar / 2).exp()
        latent_sample = mu + std * torch.randn_like(std)
        return self.latent_out_proj(latent_sample), mu, logvar

    def forward(
        self,
        image_features: tuple[torch.Tensor, torch.Tensor] | None,
        qpos: torch.Tensor,
        actions: torch.Tensor | None = None,
        is_pad: torch.Tensor | None = None,
        task_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, torch.Tensor | None]]:
        batch_size = qpos.shape[0]
        if actions is not None:
            actions = actions[:, : self.num_queries]
            if is_pad is None:
                is_pad = torch.zeros(
                    actions.shape[:2], dtype=torch.bool, device=actions.device
                )
            is_pad = is_pad[:, : self.num_queries].bool()
            latent_input, mu, logvar = self._style_variable_encoder(actions, qpos, is_pad)
        else:
            mu = logvar = None
            latent_sample = torch.zeros(
                (batch_size, self.latent_dim), dtype=torch.float32, device=qpos.device
            )
            latent_input = self.latent_out_proj(latent_sample)

        proprio_input = self.input_proj_robot_state(qpos)
        additional = torch.stack([latent_input, proprio_input], dim=0)
        additional_pos = self.additional_pos_embed.weight[:, None, :].repeat(
            1, batch_size, 1
        )

        if image_features is None:
            hs = self.transformer(
                additional,
                None,
                self.query_embed.weight,
                additional_pos,
            )[-1]
        else:
            feat, img_pos = image_features
            hs = self.transformer(
                feat,
                None,
                self.query_embed.weight,
                img_pos,
                latent_input,
                proprio_input,
                self.additional_pos_embed.weight,
                task_emb=task_emb,
            )[-1]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, (mu, logvar)

    def calculate_loss(
        self,
        model_output: tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        actions: torch.Tensor,
        is_pad: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        a_hat, _is_pad_hat, (mu, logvar) = model_output
        actions = actions[:, : self.num_queries]
        is_pad = is_pad[:, : self.num_queries].bool()
        valid = (~is_pad).unsqueeze(-1)
        n_grip = 2 if self.action_dim >= 2 else 0
        if n_grip:
            all_l1 = F.l1_loss(
                actions[..., :-n_grip],
                a_hat[..., :-n_grip],
                reduction="none",
            )
            gripper_loss = F.binary_cross_entropy_with_logits(
                a_hat[..., -n_grip:],
                actions[..., -n_grip:],
                reduction="none",
            ) * 0.05
            gripper_loss = (gripper_loss * valid).mean()
        else:
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            gripper_loss = torch.zeros((), dtype=actions.dtype, device=actions.device)
        l1 = (all_l1 * valid).mean()
        kl = _kl_divergence(mu, logvar)
        loss = l1 + gripper_loss + kl * self.kl_weight
        return loss, {
            "actor_loss": loss,
            "l1": l1,
            "gripper_loss": gripper_loss,
            "kl": kl,
        }


class ACT(nn.Module, Method):
    """PyTorch ACT method with trainable ResNet, CVAE latent, and L1+KL loss."""

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
        jit: bool = True,
        platform: str | None = None,
        seed: int = 0,
        is_rl: bool = False,
        use_ema: bool = False,
        weight_decay: float = 1e-4,
        lr_backbone: float = 1e-5,
        update_block_every_steps: int = 1,
    ):
        del jit, platform, is_rl, use_ema, update_block_every_steps
        super().__init__()
        if intrinsic_reward_module is not None:
            raise NotImplementedError("ACT does not support intrinsic rewards.")
        if not frame_stack_on_channel:
            raise NotImplementedError("ACT requires frame_stack_on_channel=true.")
        if len(action_space.shape) != 2:
            raise ValueError("ACT expects action_space.shape == (sequence, action_dim).")

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

        self.lr = float(lr)
        self.lr_backbone = float(lr_backbone)
        self.adaptive_lr = bool(adaptive_lr)
        self.num_train_steps = max(1, int(num_train_steps))
        self.actor_grad_clip = actor_grad_clip
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_train_envs = num_train_envs
        self.num_eval_envs = num_eval_envs
        self.replay_alpha = replay_alpha
        self.replay_beta = replay_beta
        self.frame_stack_on_channel = frame_stack_on_channel
        self.logging = False
        self._eval_env_running = False
        self._update_step_count = 0
        self._first_update_completed = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_spec = model
        obs_layout = bc_observation_layout(observation_space)
        self.rgb_keys = obs_layout.rgb_keys
        self.low_dim_size = int(obs_layout.low_dim_size)
        self.action_sequence = int(action_space.shape[0])
        self.action_dim = int(action_space.shape[1])

        actor_spec = model.actor_model
        if actor_spec.num_queries != self.action_sequence:
            raise ValueError("ACT num_queries must match action_sequence.")

        self.data_augmentation = bool(actor_spec.data_augmentation)
        self.use_lang_cond = bool(actor_spec.use_lang_cond)
        self._clip_text_cache: dict[tuple[int, ...], torch.Tensor] = {}
        self.aug_transforms: dict[tuple[int, int], nn.Module] = {}
        if self.data_augmentation:
            try:
                import torchvision.transforms  # noqa: F401
                from torchvision.transforms import v2  # noqa: F401
            except ImportError as exc:
                raise ImportError("ACT data augmentation requires torchvision.") from exc

        self.encoder_model = None
        if obs_layout.use_pixels:
            if model.encoder_model is None:
                raise ValueError("Pixel ACT requires encoder_model.")
            if obs_layout.rgb_input_shape is None:
                raise ValueError("Pixel ACT expected RGB observation spaces.")
            self.encoder_model = ImageEncoderACT(
                input_shape=obs_layout.rgb_input_shape,
                hidden_dim=actor_spec.hidden_dim,
                backbone=model.encoder_model.model,
                pretrained=True,
                use_lang_cond=actor_spec.use_lang_cond,
            )
            self.encoder_model.to(self.device)
            _ = self.encoder_model.output_shape

        self.actor_model = MultiViewTransformerEncoderDecoderACT(
            hidden_dim=actor_spec.hidden_dim,
            dropout=actor_spec.dropout,
            nheads=actor_spec.nheads,
            dim_feedforward=actor_spec.dim_feedforward,
            enc_layers=actor_spec.enc_layers,
            dec_layers=actor_spec.dec_layers,
            pre_norm=actor_spec.pre_norm,
            state_dim=self.low_dim_size,
            action_dim=self.action_dim,
            num_queries=self.action_sequence,
            kl_weight=actor_spec.kl_weight,
            latent_dim=actor_spec.latent_dim,
            use_lang_cond=actor_spec.use_lang_cond,
        )
        self.to(self.device)

        param_groups = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ],
                "lr": self.lr,
            }
        ]
        backbone_params = [
            p
            for n, p in self.named_parameters()
            if "backbone" in n and p.requires_grad
        ]
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": self.lr_backbone})
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.lr,
            weight_decay=float(weight_decay),
        )
        self.encoder_optimizer = None
        if self.encoder_model is not None:
            self.encoder_optimizer = torch.optim.Adam(
                self.encoder_model.parameters(),
                lr=self.lr,
            )
        self.scheduler = None
        if self.adaptive_lr:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_train_steps,
            )

    def _to_tensor(self, value, dtype=None) -> torch.Tensor:
        if torch.is_tensor(value):
            tensor = value
        else:
            tensor = torch.as_tensor(value)
        tensor = tensor.to(self.device, non_blocking=True)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor

    def _extract_low_dim(self, batch_or_obs: dict) -> torch.Tensor:
        if "low_dim_state" not in batch_or_obs:
            raise ValueError("ACT requires low_dim_state observations.")
        low_dim = self._to_tensor(batch_or_obs["low_dim_state"], torch.float32)
        return low_dim.reshape(low_dim.shape[0], -1)

    def _extract_images(self, batch_or_obs: dict) -> torch.Tensor | None:
        if self.encoder_model is None:
            return None
        images = []
        for key in self.rgb_keys:
            if key not in batch_or_obs:
                raise ValueError(f"Missing RGB observation key '{key}'.")
            value = self._to_tensor(batch_or_obs[key])
            images.append(value)
        rgb = torch.stack(images, dim=1)
        batch_size, num_views, time_dim, channels = rgb.shape[:4]
        return rgb.reshape(batch_size, num_views, time_dim * channels, *rgb.shape[4:])

    def _extract_lang_tokens(self, batch_or_obs: dict) -> torch.Tensor | None:
        if not self.use_lang_cond:
            return None
        if "lang_tokens" not in batch_or_obs:
            raise ValueError("Language-conditioned ACT requires 'lang_tokens' observations.")
        tokens = self._to_tensor(batch_or_obs["lang_tokens"], torch.long)
        while tokens.ndim > 2:
            tokens = tokens[:, 0]
        return tokens

    def encode_clip_text(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.to(self.device, dtype=torch.long, non_blocking=True)
        if tokens.ndim != 2:
            raise ValueError(f"Expected CLIP tokens of shape (B,77), got {tokens.shape}.")
        first_key = tuple(int(v) for v in tokens[0].detach().cpu().tolist())
        if tokens.shape[0] > 1 and torch.all(tokens == tokens[:1]):
            cached = self._clip_text_cache.get(first_key)
            if cached is not None:
                return cached.expand(tokens.shape[0], -1)
        if not hasattr(self, "clip_model"):
            print("Loading CLIP model...")
            import clip

            clip_model, _ = clip.load("ViT-B/32", device=self.device)
            del clip_model.visual
            clip_model.eval()
            for param in clip_model.parameters():
                param.requires_grad_(False)
            self.__dict__["clip_model"] = clip_model
        with torch.no_grad():
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            x = self.clip_model.token_embedding(tokens).type(dtype)
            x = x + self.clip_model.positional_embedding.type(dtype)
            x = x.permute(1, 0, 2)
            x = self.clip_model.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.clip_model.ln_final(x).type(dtype)
            text_features = (
                x[torch.arange(x.shape[0], device=x.device), tokens.argmax(dim=-1)]
                @ self.clip_model.text_projection
            ).to(torch.float32)
        if tokens.shape[0] > 1 and torch.all(tokens == tokens[:1]):
            self._clip_text_cache[first_key] = text_features[:1].detach()
            return self._clip_text_cache[first_key].expand(tokens.shape[0], -1)
        return text_features

    def _augment_images(self, image: torch.Tensor) -> torch.Tensor:
        if not self.data_augmentation or not self.training:
            return image
        batch_size, num_views, channels, height, width = image.shape
        if channels != 3:
            return image
        aug_key = (int(height), int(width))
        if aug_key not in self.aug_transforms:
            import torchvision.transforms as transforms
            from torchvision.transforms import v2

            self.aug_transforms[aug_key] = transforms.Compose(
                [
                    v2.RandomApply([v2.ElasticTransform(alpha=80.0, sigma=10.0)]),
                    v2.RandomApply(
                        [
                            v2.ColorJitter(
                                brightness=0.2,
                                contrast=0.2,
                                saturation=0.1,
                                hue=0.05,
                            )
                        ]
                    ),
                    v2.RandomApply([v2.RandomCrop(size=aug_key, padding=4)]),
                    AddGaussianNoise(0, 5.0),
                ]
            )
        flat = image.reshape(batch_size * num_views, channels, height, width).to(
            dtype=torch.uint8
        )
        flat = self.aug_transforms[aug_key](flat)
        return flat.reshape(batch_size, num_views, channels, height, width)

    def _forward_policy(
        self,
        batch_or_obs: dict,
        actions: torch.Tensor | None = None,
        is_pad: torch.Tensor | None = None,
    ):
        qpos = self._extract_low_dim(batch_or_obs)
        image = self._extract_images(batch_or_obs)
        if image is not None and actions is not None:
            image = self._augment_images(image)
        task_emb = None
        if self.use_lang_cond:
            task_emb = self.encode_clip_text(self._extract_lang_tokens(batch_or_obs))
        image_features = None
        if image is not None:
            encoded = self.encoder_model(image, task_emb=task_emb)
            if self.use_lang_cond:
                feat, pos, task_emb = encoded
                image_features = (feat, pos)
            else:
                image_features = encoded
        return self.actor_model(
            image_features,
            qpos,
            actions=actions,
            is_pad=is_pad,
            task_emb=task_emb,
        )

    def act(self, observations: dict, step: int, eval_mode: bool):
        del step, eval_mode
        was_training = self.training
        self.train(False)
        with torch.no_grad():
            actions = self._forward_policy(observations)[0]
        if was_training:
            self.train(True)
        return actions.detach().cpu().numpy().astype(np.float32)

    def update(
        self,
        replay_iter: Iterator[dict],
        step: int,
        replay_buffer: ReplayBuffer = None,
    ) -> dict[str, np.ndarray]:
        del step, replay_buffer
        was_training = self.training
        self.train(True)
        try:
            batch = next(replay_iter)
            actions = self._to_tensor(batch["action"], torch.float32)
            if "action_pad_mask" in batch:
                action_pad_mask = self._to_tensor(batch["action_pad_mask"], torch.bool)
            else:
                action_pad_mask = torch.zeros(
                    actions.shape[:2], dtype=torch.bool, device=self.device
                )

            start_time = time.perf_counter()
            model_output = self._forward_policy(
                batch, actions=actions, is_pad=action_pad_mask
            )
            loss, loss_dict = self.actor_model.calculate_loss(
                model_output,
                actions=actions,
                is_pad=action_pad_mask,
            )
            if self.encoder_optimizer is not None:
                self.encoder_optimizer.zero_grad(set_to_none=True)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.actor_grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), float(self.actor_grad_clip)
                )
            self.optimizer.step()
            if self.encoder_optimizer is not None:
                if self.actor_grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.encoder_model.parameters(),
                        float(self.actor_grad_clip),
                    )
                self.encoder_optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            elapsed = time.perf_counter() - start_time
            self._update_step_count += 1
            self._first_update_completed = True

            metrics = {}
            if self.logging:
                metrics["actor_loss"] = float(loss_dict["actor_loss"].detach().cpu())
                metrics["l1"] = float(loss_dict["l1"].detach().cpu())
                metrics["gripper_loss"] = float(loss_dict["gripper_loss"].detach().cpu())
                metrics["kl"] = float(loss_dict["kl"].detach().cpu())
                metrics["backend/update_time_sec"] = elapsed
                metrics["backend/update_steps_per_second"] = actions.shape[0] / max(
                    elapsed, 1e-12
                )
                if self._update_step_count == 1:
                    metrics["backend/first_update_time_sec"] = elapsed
            return metrics
        finally:
            self.train(was_training)

    def update_many(
        self,
        replay_iter: Iterator[dict],
        num_updates: int,
        replay_buffer: ReplayBuffer = None,
    ) -> None:
        was_logging = self.logging
        was_training = self.training
        self.logging = False
        self.train(True)
        try:
            for update_idx in range(int(num_updates)):
                self.update(replay_iter, update_idx, replay_buffer)
        finally:
            self.logging = was_logging
            self.train(was_training)

    def reset(self, step: int, agents_to_reset: list[int]):
        del step, agents_to_reset

    def state_dict(self) -> dict:
        return {"model": super().state_dict()}

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        model_state = state_dict.get("model", state_dict)
        return super().load_state_dict(model_state, strict=strict)

    def checkpoint_state_dict(self) -> dict[str, dict]:
        return {
            "model": {k: v.detach().cpu() for k, v in super().state_dict().items()},
            "optimizer": self.optimizer.state_dict(),
            "encoder_optimizer": None
            if self.encoder_optimizer is None
            else self.encoder_optimizer.state_dict(),
            "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
        }

    def load_checkpoint_state_dict(self, state_dict: dict[str, dict]):
        model_state = state_dict.get("model", state_dict)
        super().load_state_dict(model_state)
        if "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])
        if (
            self.encoder_optimizer is not None
            and state_dict.get("encoder_optimizer") is not None
        ):
            self.encoder_optimizer.load_state_dict(state_dict["encoder_optimizer"])
        if self.scheduler is not None and state_dict.get("scheduler") is not None:
            self.scheduler.load_state_dict(state_dict["scheduler"])


__all__ = [
    "ACT",
    "ACTActorModelSpec",
    "ACTModelSpec",
    "ACTSpec",
    "ImageEncoderACT",
    "MultiViewTransformerEncoderDecoderACT",
    "act_model_spec_from_cfg",
    "act_spec_from_cfg",
]
