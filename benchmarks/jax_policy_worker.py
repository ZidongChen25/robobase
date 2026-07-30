"""JAX side of the state-only CleanDiffuser parity benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import jax
import numpy as np
from gymnasium import spaces

from robobase.method.diffusion import Diffusion, DiffusionModelSpec
from robobase.method.flow_matching import FlowMatching, FlowMatchingModelSpec
from robobase.models.backbone import DiffusionBackboneSpec


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


def _build_agent(args, observation_space, action_space):
    backbone = DiffusionBackboneSpec(
        type=args.backbone,
        sequence_length=args.horizon,
        diffusion_step_embed_dim=args.embed_dim,
        down_dims=tuple(args.down_dims),
        kernel_size=args.kernel_size,
        n_groups=args.n_groups,
        conditioning_mode="global",
        cond_predict_scale=True,
        global_condition_embed_dim=args.embed_dim,
        timestep_embedding_type="clean_diffuser",
        operator_variant="torch",
        compatibility_mode="clean_diffuser",
    )
    common = dict(
        lr=args.lr,
        adaptive_lr=False,
        num_train_steps=max(1000, args.warmup + args.iterations + 1),
        observation_space=observation_space,
        action_space=action_space,
        num_train_envs=1,
        num_eval_envs=args.eval_batch_size,
        replay_alpha=0.6,
        replay_beta=0.4,
        frame_stack_on_channel=True,
        actor_grad_clip=None,
        jit=True,
        platform=args.platform,
        seed=args.seed,
        use_ema=args.ema,
        ema_decay=args.ema_decay,
        weight_decay=args.weight_decay,
        update_block_every_steps=1,
    )
    if args.objective == "diffusion":
        sampler = args.sampler or "ddpm"
        if sampler not in {"ddpm", "ddim"}:
            raise ValueError("Diffusion sampler must be ddpm or ddim.")
        return Diffusion(
            model=DiffusionModelSpec(backbone, None, None),
            num_diffusion_iters=args.sample_steps,
            objective_type="ddpm",
            sampler=sampler,
            ema_decay_schedule="constant",
            **common,
        )
    if args.sampler not in {None, "euler"}:
        raise ValueError("Flow Matching sampler must be euler.")
    return FlowMatching(
        model=FlowMatchingModelSpec(backbone, None, None),
        num_flow_steps=args.sample_steps,
        objective_type="rectified_flow",
        sampler="euler",
        sample_schedule="uniform",
        train_time_schedule="uniform",
        time_scale=1.0,
        ema_decay_schedule="constant",
        **common,
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
    parser.add_argument("--sampler", choices=("ddpm", "ddim", "euler"), default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--platform", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.encoder != "none" or args.fusion != "none":
        raise ValueError("The CleanDiffuser parity profile is state-only.")

    observation_space = spaces.Dict(
        {
            "low_dim_state": spaces.Box(
                -np.inf,
                np.inf,
                shape=(args.obs_steps, args.obs_dim),
                dtype=np.float32,
            )
        }
    )
    action_space = spaces.Box(
        -1.0,
        1.0,
        shape=(args.horizon, args.action_dim),
        dtype=np.float32,
    )
    agent = _build_agent(args, observation_space, action_space)

    rng = np.random.default_rng(args.seed)
    host_batch = {
        "low_dim_state": rng.standard_normal(
            (args.batch_size, args.obs_steps, args.obs_dim), dtype=np.float32
        ),
        "action": rng.standard_normal(
            (args.batch_size, args.horizon, args.action_dim), dtype=np.float32
        ),
    }
    batch = jax.tree.map(jax.device_put, host_batch)
    replay_iter = iter(lambda: batch, None)
    jax.tree.map(lambda value: value.block_until_ready(), (agent.params, batch))

    start = time.perf_counter()
    agent.update(replay_iter, 0)
    update_first_seconds = time.perf_counter() - start
    for _ in range(args.warmup):
        agent.update(replay_iter, 0)
    update_times = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        agent.update(replay_iter, 0)
        update_times.append(time.perf_counter() - start)

    eval_obs = {
        "low_dim_state": rng.standard_normal(
            (args.eval_batch_size, args.obs_steps, args.obs_dim), dtype=np.float32
        )
    }
    start = time.perf_counter()
    agent.act(eval_obs, 0, eval_mode=args.ema)
    sample_first_seconds = time.perf_counter() - start
    for _ in range(args.warmup):
        agent.act(eval_obs, 0, eval_mode=args.ema)
    sample_times = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        agent.act(eval_obs, 0, eval_mode=args.ema)
        sample_times.append(time.perf_counter() - start)

    update_summary = _summary(update_times)
    sample_summary = _summary(sample_times)
    params = sum(leaf.size for leaf in jax.tree.leaves(agent.params))
    result = {
        "backend": "jax",
        "implementation": "RoboBase JAX",
        "framework_version": jax.__version__,
        "device": str(jax.devices()[0]),
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
        "sampler": args.sampler
        or ("ddpm" if args.objective == "diffusion" else "euler"),
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
        "parameter_count": params,
        "compile_mode": "jax.jit",
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
