"""PyTorch side of the BiGym-shaped pixel policy update benchmark.

ACT uses the original RoboBase PyTorch implementation (``ActBCAgent`` from
the PyTorch checkout, FiLM ResNet18 backbone, DETR-style transformer) with the
hyper-parameters of the JAX ``act.yaml``. Diffusion / Flow Matching use a
matched PyTorch stack: torchvision ResNet18 trunk (frozen BN statistics,
trainable weights, 224 bilinear resize, ImageNet normalisation, global pool,
views flattened) feeding CleanDiffuser's ``ChiTransformer``, trained with the
same DDPM (cosine, 50 steps) / rectified-flow objectives, AdamW and EMA as the
JAX ``dp_pixel_bigym_transformer_ddpm`` / ``fm_pixel_bigym_transformer``
launches.

Batches live on the GPU as uint8 before ``update`` is called, so the PyTorch
numbers exclude any host-side conversion cost (best case for PyTorch).
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from functools import partial
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gpu_sampler import ProcessGpuMemorySampler  # noqa: E402

TORCH_ROBOBASE = Path(
    os.environ.get("TORCH_ROBOBASE_ROOT", str(Path.home() / "robobase"))
)
CLEAN_DIFFUSER = Path(
    os.environ.get("CLEAN_DIFFUSER_ROOT", str(Path.home() / "CleanDiffuser"))
)
CAMERAS = ("head", "left_wrist", "right_wrist")


def _autocast_context(args):
    """``torch.autocast(bfloat16)`` when ``--amp-bf16`` is set, else a no-op."""

    import contextlib

    if getattr(args, "amp_bf16", False):
        return lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _summary(seconds: list[float]) -> dict[str, float]:
    ordered = sorted(seconds)
    mean = statistics.fmean(seconds)
    return {
        "mean_ms": mean * 1000.0,
        "median_ms": statistics.median(ordered) * 1000.0,
        "p95_ms": float(np.percentile(seconds, 95)) * 1000.0,
        "min_ms": ordered[0] * 1000.0,
        "max_ms": ordered[-1] * 1000.0,
        "cv_percent": (statistics.pstdev(seconds) / mean * 100.0) if mean else 0.0,
    }


def build_spaces(args):
    from gymnasium import spaces

    obs_spaces = {}
    for camera in CAMERAS[: args.num_views]:
        obs_spaces[f"rgb_{camera}"] = spaces.Box(
            0, 255, (1, 3, args.image_size, args.image_size), np.uint8
        )
    obs_spaces["low_dim_state"] = spaces.Box(
        -np.inf, np.inf, (1, args.low_dim), np.float32
    )
    if args.lang:
        obs_spaces["lang_tokens"] = spaces.Box(0, 49407, (1, 77), np.int32)
    observation_space = spaces.Dict(obs_spaces)
    action_space = spaces.Box(
        -1.0, 1.0, (args.action_seq, args.action_dim), np.float32
    )
    return observation_space, action_space


def make_batches(args, count: int, device) -> list[dict]:
    rng = np.random.default_rng(args.seed)
    batches = []
    tokens = rng.integers(0, 49407, (77,), dtype=np.int32)
    for _ in range(count):
        batch = {}
        for camera in CAMERAS[: args.num_views]:
            batch[f"rgb_{camera}"] = rng.integers(
                0,
                256,
                (args.batch_size, 1, 3, args.image_size, args.image_size),
                dtype=np.uint8,
            )
        batch["low_dim_state"] = rng.standard_normal(
            (args.batch_size, 1, args.low_dim), dtype=np.float32
        )
        if args.lang:
            batch["lang_tokens"] = np.tile(
                tokens[None, None], (args.batch_size, 1, 1)
            ).astype(np.int32)
        batch["action"] = rng.uniform(
            -1.0, 1.0, (args.batch_size, args.action_seq, args.action_dim)
        ).astype(np.float32)
        pad = np.zeros((args.batch_size, args.action_seq), dtype=bool)
        tail_rows = rng.integers(0, args.batch_size, (max(1, args.batch_size // 16),))
        for row in tail_rows:
            pad[row, args.action_seq - int(rng.integers(1, args.action_seq // 2 + 1)) :] = True
        batch["action_pad_mask"] = pad
        batch["reward"] = np.zeros((args.batch_size, 1), dtype=np.float32)
        batches.append(
            {key: torch.as_tensor(value, device=device) for key, value in batch.items()}
        )
    return batches


# ---------------------------------------------------------------------------
# ACT: original RoboBase PyTorch implementation
# ---------------------------------------------------------------------------


def build_act_agent(args, device, observation_space, action_space):
    sys.path.insert(0, str(TORCH_ROBOBASE))
    import robobase.models.act.backbone as act_backbone

    # No network access on the benchmark host: build ImageNet-shaped ResNet18
    # trunks with random weights. Speed and memory do not depend on weights.
    act_backbone.is_main_process = lambda: False
    original_film_init = act_backbone.ResNetFilmBackbone.__init__

    def film_init(self, embedding_name, pretrained=True, film_config=None):
        original_film_init(
            self, embedding_name, pretrained=False, film_config=film_config
        )

    act_backbone.ResNetFilmBackbone.__init__ = film_init

    if getattr(args, "amp_bf16", False):
        # The reference FrozenBatchNorm2d multiplies by float32 buffers, which
        # promotes every bf16 convolution output back to float32 under
        # autocast (activations stored in float32, bandwidth doubled). Cast the
        # folded scale/bias to the activation dtype so the PyTorch arm really
        # runs its trunk in bf16, like the JAX bf16 trunk. Numerically this is
        # the standard autocast treatment of an affine op.
        def frozen_bn_forward(self, x):
            w = self.weight.reshape(1, -1, 1, 1)
            b = self.bias.reshape(1, -1, 1, 1)
            rv = self.running_var.reshape(1, -1, 1, 1)
            rm = self.running_mean.reshape(1, -1, 1, 1)
            scale = w * (rv + 1e-5).rsqrt()
            bias = b - rm * scale
            return x * scale.to(x.dtype) + bias.to(x.dtype)

        act_backbone.FrozenBatchNorm2d.forward = frozen_bn_forward

    from robobase.method.act import ActBCAgent, ImageEncoderACT
    from robobase.models.multi_view_transformer import (
        MultiViewTransformerEncoderDecoderACT,
    )

    class BenchActBCAgent(ActBCAgent):
        def encode_clip_text(self, lang_tokens):
            # CLIP is not installed in the PyTorch checkout. The parity profile
            # supplies the same fixed zero feature as the JAX side.
            return self._bench_task_emb[: lang_tokens.shape[0]], None

    agent = BenchActBCAgent(
        lr_backbone=1e-5,
        weight_decay=1e-4,
        use_lang_cond=bool(args.lang),
        lr=1e-5,
        adaptive_lr=False,
        num_train_steps=100000,
        actor_grad_clip=1.0,
        actor_model=partial(
            MultiViewTransformerEncoderDecoderACT,
            hidden_dim=256,
            dropout=0.1,
            nheads=8,
            dim_feedforward=2048,
            enc_layers=4,
            dec_layers=6,
            pre_norm=False,
            num_queries=args.action_seq,
            kl_weight=10,
            use_lang_cond=bool(args.lang),
        ),
        encoder_model=partial(
            ImageEncoderACT,
            hidden_dim=256,
            position_embedding="sine",
            lr_backbone=1e-5,
            masks=False,
            backbone="resnet18",
            dilation=False,
            use_lang_cond=bool(args.lang),
        ),
        view_fusion_model=None,
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        num_train_envs=1,
        num_eval_envs=1,
        replay_alpha=0.7,
        replay_beta=0.5,
        frame_stack_on_channel=True,
    )
    if getattr(args, "parity_profile", False):
        # The JAX parity arm receives the same fixed zero feature directly.
        agent._bench_task_emb = torch.zeros(args.batch_size, 512, device=device)
        # These modules are present in the historical Torch policy but do no
        # work in this benchmark: frame_stack=1 bypasses projection_layer and
        # calculate_loss ignores the padding prediction.
        agent.actor.projection_layer = nn.Identity()
        agent.actor.actor_model.is_pad_head = nn.Identity()
    else:
        agent._bench_task_emb = torch.randn(args.batch_size, 512, device=device)
    agent.train(True)
    agent.logging = False
    parameters = list(agent.actor.parameters())
    parameter_count = sum(p.numel() for p in parameters)

    autocast = _autocast_context(args)

    def step(replay_iter):
        # autocast(bf16) covers the forward, loss and (for the reference
        # implementation) backward inside ``update``; bf16 needs no GradScaler.
        with autocast():
            agent.update(replay_iter, 0, None)

    return step, parameter_count, parameter_count


# ---------------------------------------------------------------------------
# Diffusion / Flow Matching: torchvision ResNet18 + CleanDiffuser ChiTransformer
# ---------------------------------------------------------------------------


class _FrozenAffineNorm(nn.Module):
    """BatchNorm2d with frozen statistics as a dtype-following affine op."""

    def __init__(self, bn: nn.BatchNorm2d):
        super().__init__()
        self.weight = nn.Parameter(bn.weight.detach().clone())
        self.bias = nn.Parameter(bn.bias.detach().clone())
        self.register_buffer("running_mean", bn.running_mean.detach().clone())
        self.register_buffer("running_var", bn.running_var.detach().clone())
        self.eps = float(bn.eps)

    def forward(self, x):
        scale = self.weight * torch.rsqrt(self.running_var + self.eps)
        bias = self.bias - self.running_mean * scale
        return x * scale.to(x.dtype).view(1, -1, 1, 1) + bias.to(x.dtype).view(1, -1, 1, 1)


def _replace_bn_with_frozen_affine(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, _FrozenAffineNorm(child))
        else:
            _replace_bn_with_frozen_affine(child)


class MultiViewResNet18Condition(nn.Module):
    """Mirror of the JAX trainable ResNet18 encoder + flatten fusion + concat."""

    def __init__(self, resize_to: int | None = 224, bf16_affine_bn: bool = False):
        super().__init__()
        import torchvision

        trunk = torchvision.models.resnet18(weights=None)
        trunk.fc = nn.Identity()
        if bf16_affine_bn:
            # Frozen-statistics BatchNorm is an affine map. torch.autocast keeps
            # ``batch_norm`` in float32, which would promote every bf16
            # convolution output back to float32; fold the frozen statistics
            # into a per-channel affine that computes in the activation dtype
            # (same numerics, trainable weight/bias kept in float32), so the
            # PyTorch trunk really runs in bf16 like the JAX bf16 trunk.
            _replace_bn_with_frozen_affine(trunk)
        self.trunk = trunk
        self.resize_to = resize_to
        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )
        self._freeze_bn_stats()

    def _freeze_bn_stats(self):
        for module in self.trunk.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Running statistics stay frozen (JAX uses use_running_average=True);
        # the affine parameters and convolutions remain trainable.
        self._freeze_bn_stats()
        return self

    def forward(self, rgb_u8: torch.Tensor, low_dim: torch.Tensor, lang):
        batch_size, num_views = rgb_u8.shape[:2]
        x = rgb_u8.reshape(batch_size * num_views, *rgb_u8.shape[2:]).float() / 255.0
        if self.resize_to is not None and x.shape[-1] != self.resize_to:
            x = F.interpolate(
                x,
                size=(self.resize_to, self.resize_to),
                mode="bilinear",
                align_corners=False,
            )
        x = (x - self.mean) / self.std
        features = self.trunk(x).reshape(batch_size, -1)
        parts = [low_dim, features]
        if lang is not None:
            parts.append(lang)
        return torch.cat(parts, dim=-1)


def _cosine_alphas_cumprod(num_steps: int, max_beta: float = 0.999) -> np.ndarray:
    def alpha_bar(t):
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = [
        min(1.0 - alpha_bar((i + 1) / num_steps) / alpha_bar(i / num_steps), max_beta)
        for i in range(num_steps)
    ]
    return np.cumprod(1.0 - np.asarray(betas, dtype=np.float64)).astype(np.float32)


def build_generative_agent(args, device, ema_decay_constant: bool):
    sys.path.insert(0, str(CLEAN_DIFFUSER))
    from cleandiffuser.nn_diffusion.chitransformer import ChiTransformer

    lang_dim = 512 if args.lang else 0
    condition_dim = args.low_dim + 512 * args.num_views + lang_dim
    encoder = MultiViewResNet18Condition(
        resize_to=224, bf16_affine_bn=bool(getattr(args, "amp_bf16", False))
    ).to(device)
    net = ChiTransformer(
        act_dim=args.action_dim,
        obs_dim=condition_dim,
        Ta=args.action_seq,
        To=1,
        d_model=256,
        nhead=4,
        num_layers=8,
        p_drop_emb=0.0,
        p_drop_attn=0.3,
        n_cond_layers=0,
        timestep_emb_type="positional",
    ).to(device)
    if getattr(args, "parity_profile", False):
        # CleanDiffuser constructs cond_encoder but never calls it in forward;
        # the JAX port correctly has no counterpart.
        net.cond_encoder = nn.Identity()
    model = nn.ModuleDict({"encoder": encoder, "diffusion": net})
    model.train()
    ema_model = deepcopy(model).requires_grad_(False)
    ema_model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    parameter_count = sum(p.numel() for p in model.parameters())
    differentiable_parameter_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    task_emb = None
    if args.lang:
        task_emb = (
            torch.zeros(args.batch_size, 512, device=device)
            if getattr(args, "parity_profile", False)
            else torch.randn(args.batch_size, 512, device=device)
        )
    num_diffusion_iters = 50
    alphas_cumprod = torch.as_tensor(_cosine_alphas_cumprod(num_diffusion_iters), device=device)
    sqrt_ac = alphas_cumprod.sqrt()
    sqrt_1m_ac = (1.0 - alphas_cumprod).sqrt()
    time_scale = 1000.0
    ema_step = 0

    def ema_update():
        nonlocal ema_step
        ema_step += 1
        if ema_decay_constant:
            decay = 0.995
        else:
            # diffusers schedule as implemented in the JAX methods
            decay = min(0.995, (1.0 + max(0, ema_step - 1)) / (10.0 + max(0, ema_step - 1)))
        with torch.no_grad():
            for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                p_ema.lerp_(p, 1.0 - decay)

    autocast = _autocast_context(args)

    def features_from_batch(batch):
        rgb = torch.stack(
            [batch[f"rgb_{camera}"] for camera in CAMERAS[: args.num_views]], dim=1
        )  # (B, V, T, 3, H, W)
        rgb = rgb.reshape(rgb.shape[0], rgb.shape[1], -1, *rgb.shape[-2:])
        low_dim = batch["low_dim_state"].reshape(rgb.shape[0], -1)
        return model["encoder"](rgb, low_dim, task_emb)

    def step(replay_iter):
        batch = next(replay_iter)
        actions = batch["action"]
        valid = ~batch["action_pad_mask"]
        valid_f = valid.unsqueeze(-1).float()
        if args.method == "diffusion":
            noise = torch.randn_like(actions)
            timesteps = torch.randint(
                0, num_diffusion_iters, (actions.shape[0],), device=device
            )
            noisy = (
                sqrt_ac[timesteps][:, None, None] * actions
                + sqrt_1m_ac[timesteps][:, None, None] * noise
            )
            noisy = torch.where(valid.unsqueeze(-1), noisy, torch.zeros_like(noisy))
            target = torch.where(valid.unsqueeze(-1), noise, torch.zeros_like(noise))
            model_time = timesteps
            model_input = noisy
        else:
            x1 = torch.randn_like(actions)
            t = torch.rand((actions.shape[0],), device=device)
            xt = t[:, None, None] * x1 + (1.0 - t[:, None, None]) * actions
            target = actions - x1
            xt = torch.where(valid.unsqueeze(-1), xt, torch.zeros_like(xt))
            target = torch.where(valid.unsqueeze(-1), target, torch.zeros_like(target))
            model_time = t * time_scale
            model_input = xt
        with autocast():
            condition = features_from_batch(batch)[:, None, :]
            pred = model["diffusion"](model_input, model_time, condition)
            per_token = (pred.float() - target) ** 2
            denom = valid_f.expand_as(per_token).sum(dim=(1, 2)).clamp(min=1.0)
            per_sample = (per_token * valid_f).sum(dim=(1, 2)) / denom
            loss = per_sample.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        ema_update()

    return step, parameter_count, differentiable_parameter_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", choices=("act", "diffusion", "flow_matching"), required=True
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-views", type=int, default=3)
    parser.add_argument("--low-dim", type=int, default=63)
    parser.add_argument("--action-dim", type=int, default=16)
    parser.add_argument("--action-seq", type=int, default=20)
    parser.add_argument("--lang", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--amp-bf16",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run forward/loss under torch.autocast(bfloat16): bf16 convolutions, "
        "matmuls and flash attention; batch norm, layer norm, softmax and the "
        "optimizer stay float32 (the PyTorch counterpart of the JAX bf16 trunk + "
        "cudnn attention configuration).",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", default="")
    parser.add_argument(
        "--parity-profile", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    sampler = ProcessGpuMemorySampler().start()
    torch.manual_seed(args.seed)
    # Set both switches explicitly: recent PyTorch builds may default cuDNN
    # convolutions to TF32 even when matmul TF32 is disabled.  ``--no-tf32``
    # must therefore be an actual strict-float32 control, not merely whatever
    # defaults happen to be installed on the benchmark host.
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    torch.set_float32_matmul_precision("high" if args.tf32 else "highest")
    device = torch.device("cuda:0")
    observation_space, action_space = build_spaces(args)

    build_start = time.perf_counter()
    if args.method == "act":
        step, parameter_count, differentiable_parameter_count = build_act_agent(
            args, device, observation_space, action_space
        )
    else:
        step, parameter_count, differentiable_parameter_count = build_generative_agent(
            args, device, ema_decay_constant=(args.method == "diffusion")
        )
    build_seconds = time.perf_counter() - build_start

    batches = make_batches(args, args.num_batches, device)
    replay_iter = itertools.cycle(batches)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    first_start = time.perf_counter()
    step(replay_iter)
    torch.cuda.synchronize()
    first_seconds = time.perf_counter() - first_start
    compile_peak_mib = sampler.reset_peak()

    for _ in range(args.warmup):
        step(replay_iter)
    torch.cuda.synchronize()

    throughput_start = time.perf_counter()
    for _ in range(args.iterations):
        step(replay_iter)
    torch.cuda.synchronize()
    throughput_seconds = time.perf_counter() - throughput_start

    blocked_times = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        step(replay_iter)
        torch.cuda.synchronize()
        blocked_times.append(time.perf_counter() - start)

    smi = sampler.stop()
    result = {
        "backend": "torch",
        "label": args.label,
        "method": args.method,
        "torch_version": torch.__version__,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "num_views": args.num_views,
        "low_dim": args.low_dim,
        "action_dim": args.action_dim,
        "action_seq": args.action_seq,
        "lang": bool(args.lang),
        "act_use_remat": False,
        "parity_profile": bool(args.parity_profile),
        "large_inputs_device_resident": True,
        "language_input": (
            "precomputed_zero_feature"
            if args.lang and args.parity_profile
            else "fixed_random_feature"
        ),
        "amp_bf16": bool(args.amp_bf16),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "fused_steps": 1,
        "parameter_count": int(parameter_count),
        "differentiable_parameter_count": int(differentiable_parameter_count),
        "build_seconds": build_seconds,
        "first_call_seconds": first_seconds,
        "throughput_updates_per_second": args.iterations / throughput_seconds,
        "throughput_ms_per_update": throughput_seconds / args.iterations * 1000.0,
        "throughput_samples_per_second": args.iterations
        * args.batch_size
        / throughput_seconds,
        "blocked_call": _summary(blocked_times),
        "blocked_ms_per_update": statistics.median(blocked_times) * 1000.0,
        "max_memory_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "max_memory_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "memory_allocated_mib": torch.cuda.memory_allocated() / 2**20,
        "nvidia_smi_compile_peak_mib": compile_peak_mib,
        **smi,
        "env": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")},
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
