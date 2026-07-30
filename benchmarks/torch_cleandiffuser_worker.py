"""CleanDiffuser side of the state-only UNet parity benchmark.

This adapter intentionally lives outside ``robobase`` so importing or installing
the runtime package never requires PyTorch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import cleandiffuser
import numpy as np
import torch

from cleandiffuser.diffusion import ContinuousRectifiedFlow
from cleandiffuser.diffusion.ddpm import DDPM
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import ChiUNet1d


def _summary(seconds: list[float]) -> dict[str, float]:
    ordered = sorted(seconds)
    mean = statistics.fmean(seconds)
    return {
        "mean_ms": mean * 1000.0,
        "median_ms": statistics.median(ordered) * 1000.0,
        "p95_ms": float(np.percentile(seconds, 95)) * 1000.0,
        "min_ms": ordered[0] * 1000.0,
        "max_ms": ordered[-1] * 1000.0,
        "cv_percent": statistics.pstdev(seconds) / mean * 100.0,
    }


def _synchronize() -> None:
    torch.cuda.synchronize()


def _time_call(fn):
    _synchronize()
    start = time.perf_counter()
    result = fn()
    _synchronize()
    return time.perf_counter() - start, result


def _dim_mult(down_dims: list[int], model_dim: int) -> list[int]:
    previous = model_dim
    multipliers = []
    for index, width in enumerate(down_dims):
        denominator = model_dim if index == 0 else previous
        if width % denominator:
            raise ValueError(
                "CleanDiffuser dim_mult cannot represent down_dims="
                f"{down_dims} from model_dim={model_dim}."
            )
        multiplier = width // denominator
        if multiplier < 1:
            raise ValueError("down_dims must be non-decreasing positive multiples.")
        multipliers.append(multiplier)
        previous = width
    return multipliers


def _build_agent(args, device: torch.device):
    if args.n_groups != 8:
        raise ValueError("CleanDiffuser ChiUNet1d fixes GroupNorm to 8 groups.")
    model_dim = args.down_dims[0]
    nn_diffusion = ChiUNet1d(
        act_dim=args.action_dim,
        obs_dim=args.obs_dim,
        To=args.obs_steps,
        model_dim=model_dim,
        emb_dim=args.embed_dim,
        kernel_size=args.kernel_size,
        cond_predict_scale=True,
        obs_as_global_cond=True,
        dim_mult=_dim_mult(args.down_dims, model_dim),
        timestep_emb_type="positional",
    ).to(device)
    nn_condition = IdentityCondition(dropout=0.0).to(device)
    bounds_shape = (1, args.horizon, args.action_dim)
    x_max = torch.ones(bounds_shape, device=device)
    x_min = -torch.ones(bounds_shape, device=device)
    common = {
        "nn_diffusion": nn_diffusion,
        "nn_condition": nn_condition,
        "device": device,
        "x_max": x_max,
        "x_min": x_min,
        "ema_rate": args.ema_decay,
        "optim_params": {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
    }
    if args.objective == "diffusion":
        agent = DDPM(diffusion_steps=args.sample_steps, **common)
    else:
        agent = ContinuousRectifiedFlow(**common)

    # CleanDiffuser's reference pipeline uses an identity condition with zero
    # dropout. Eval mode avoids generating a redundant all-one dropout mask.
    agent.model["condition"].eval()
    agent.model_ema["condition"].eval()
    return agent


def _compile_diffusion_modules(agent, mode: str) -> None:
    agent.model["diffusion"] = torch.compile(
        agent.model["diffusion"], mode=mode, fullgraph=False
    )
    agent.model_ema["diffusion"] = torch.compile(
        agent.model_ema["diffusion"], mode=mode, fullgraph=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--objective", choices=("diffusion", "flow_matching"), required=True
    )
    parser.add_argument("--backbone", choices=("unet1d",), default="unet1d")
    parser.add_argument("--encoder", choices=("none",), default="none")
    parser.add_argument("--fusion", choices=("none",), default="none")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--action-dim", type=int, default=10)
    parser.add_argument("--obs-steps", type=int, default=2)
    parser.add_argument("--obs-dim", type=int, default=23)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--down-dims", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--n-groups", type=int, default=8)
    parser.add_argument("--sample-steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This performance worker requires a CUDA device.")
    if args.encoder != "none" or args.fusion != "none":
        raise ValueError("The CleanDiffuser parity profile is state-only.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    agent = _build_agent(args, device)
    parameter_count = sum(parameter.numel() for parameter in agent.model.parameters())
    if args.torch_compile:
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        _compile_diffusion_modules(agent, args.compile_mode)

    generator = np.random.default_rng(args.seed)
    actions = torch.as_tensor(
        generator.standard_normal(
            (args.batch_size, args.horizon, args.action_dim), dtype=np.float32
        ),
        device=device,
    )
    condition = torch.as_tensor(
        generator.standard_normal(
            (args.batch_size, args.obs_steps * args.obs_dim), dtype=np.float32
        ),
        device=device,
    )

    def update_once():
        return agent.update(actions, condition, update_ema=args.ema)

    update_first_seconds, _ = _time_call(update_once)
    for _ in range(args.warmup):
        update_once()
    update_times = [_time_call(update_once)[0] for _ in range(args.iterations)]

    eval_condition_host = generator.standard_normal(
        (args.eval_batch_size, args.obs_steps * args.obs_dim), dtype=np.float32
    )
    prior = torch.zeros(
        (args.eval_batch_size, args.horizon, args.action_dim), device=device
    )

    def sample_once():
        eval_condition = torch.as_tensor(eval_condition_host, device=device)
        sample_kwargs = {
            "prior": prior,
            "n_samples": args.eval_batch_size,
            "sample_steps": args.sample_steps,
            "use_ema": args.ema,
            "condition_cfg": eval_condition,
            "w_cfg": 1.0,
            "temperature": 1.0,
        }
        if args.objective == "flow_matching":
            sample_kwargs["sample_step_schedule"] = "uniform_continuous"
        with torch.no_grad():
            sample, _ = agent.sample(**sample_kwargs)
        return sample.detach().cpu().numpy()

    sample_first_seconds, _ = _time_call(sample_once)
    for _ in range(args.warmup):
        sample_once()
    sample_times = [_time_call(sample_once)[0] for _ in range(args.iterations)]

    update_summary = _summary(update_times)
    sample_summary = _summary(sample_times)
    graph_breaks = 0
    unique_graphs = 0
    compiler_counter_totals = {}
    if args.torch_compile:
        counters = torch._dynamo.utils.counters
        graph_breaks = int(sum(counters.get("graph_break", {}).values()))
        unique_graphs = int(counters.get("stats", {}).get("unique_graphs", 0))
        compiler_counter_totals = {
            str(category): int(sum(values.values()))
            for category, values in counters.items()
        }
    result = {
        "backend": "torch",
        "implementation": "CleanDiffuser",
        "implementation_source": str(Path(cleandiffuser.__file__).resolve().parent),
        "framework_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "objective": args.objective,
        "backbone": args.backbone,
        "encoder": args.encoder,
        "fusion": args.fusion,
        "dtype": "float32",
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "horizon": args.horizon,
        "action_dim": args.action_dim,
        "obs_shape": [args.obs_steps, args.obs_dim],
        "condition_dim": args.obs_steps * args.obs_dim,
        "embed_dim": args.embed_dim,
        "down_dims": args.down_dims,
        "kernel_size": args.kernel_size,
        "n_groups": args.n_groups,
        "cond_predict_scale": True,
        "conditioning_mode": "global",
        "timestep_embedding_type": "clean_diffuser",
        "operator_variant": "torch",
        "compatibility_mode": "clean_diffuser",
        "global_condition_embed_dim": args.embed_dim,
        "sample_steps": args.sample_steps,
        "sampler": "ddpm" if args.objective == "diffusion" else "euler",
        "ema": args.ema,
        "ema_decay": args.ema_decay,
        "ema_decay_schedule": "constant",
        "optimizer": "adamw",
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "training_schedule": (
            "cosine_discrete" if args.objective == "diffusion" else "uniform_continuous"
        ),
        "loss_reduction": "mse_mean",
        "sample_temperature": 1.0,
        "action_bounds": [-1.0, 1.0],
        "parameter_count": parameter_count,
        "compile_mode": args.compile_mode if args.torch_compile else "eager",
        "graph_breaks": graph_breaks,
        "unique_graphs": unique_graphs,
        "compiler_counter_totals": compiler_counter_totals,
        "update_first_call_seconds": update_first_seconds,
        "sample_first_call_seconds": sample_first_seconds,
        "update": update_summary,
        "update_samples_per_second": args.batch_size
        / (update_summary["mean_ms"] / 1000.0),
        "sample": sample_summary,
        "sample_actions_per_second": args.eval_batch_size
        / (sample_summary["mean_ms"] / 1000.0),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
