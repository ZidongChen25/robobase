"""JAX side of the BiGym-shaped pixel policy update benchmark.

Builds the production ACT / Diffusion / Flow Matching agents through the same
hydra launch configs and factory as ``train.py``, feeds them synthetic batches
with the exact BiGym replay layout (three 256x256 uint8 cameras, low-dim
state, CLIP-style language tokens, chunked actions with a pad mask), and
reports update throughput plus peak device memory.

Run with ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` so the nvidia-smi footprint
reflects real allocations instead of the default 75% pool.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "robobase" / "cfgs"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT))

from gpu_sampler import ProcessGpuMemorySampler  # noqa: E402

LAUNCHES = {
    "act": "act_pixel_bigym",
    "diffusion": "dp_pixel_bigym_transformer_ddpm",
    "flow_matching": "fm_pixel_bigym_transformer",
}
CAMERAS = ("head", "left_wrist", "right_wrist")


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover - metadata only
        return "unknown"


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


def build_cfg(args):
    from hydra import compose, initialize_config_dir

    overrides = [
        f"launch={LAUNCHES[args.method]}",
        f"batch_size={args.batch_size}",
        f"visual_observation_shape=[{args.image_size},{args.image_size}]",
        f"action_sequence={args.action_seq}",
        "pixels=true",
        "num_train_envs=1",
        "num_eval_envs=1",
        f"backend.fused_update_steps={max(1, args.fused_steps)}",
        f"backend.update_block_every_steps={args.block_every}",
        f"method.use_lang_cond={'true' if args.lang else 'false'}",
    ]
    if args.method == "act":
        overrides.append(
            "method.actor_model.data_augmentation="
            + ("true" if args.augmentation else "false")
        )
        if getattr(args, "parity_profile", False):
            overrides.extend(
                [
                    "method.actor_model.use_remat=false",
                    "method.actor_model.proprio_projection_type=campose_single",
                    "method.actor_model.gripper_dims=0",
                ]
            )
    elif getattr(args, "parity_profile", False):
        # Match CleanDiffuser's exact GELU, LayerNorm epsilon, attention-output
        # dropout, positional embedding and zero embedding-dropout setting.
        overrides.extend(
            [
                "method.backbone.operator_variant=torch",
                "+method.backbone.embedding_dropout=0.0",
            ]
        )
    overrides.extend(args.override)
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="robobase_config", overrides=overrides)


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
        if getattr(args, "parity_profile", False):
            obs_spaces["lang_features"] = spaces.Box(
                -np.inf, np.inf, (1, 512), np.float32
            )
        else:
            obs_spaces["lang_tokens"] = spaces.Box(0, 49407, (1, 77), np.int32)
    observation_space = spaces.Dict(obs_spaces)
    action_space = spaces.Box(
        -1.0, 1.0, (args.action_seq, args.action_dim), np.float32
    )
    return observation_space, action_space


def make_batches(args, count: int) -> list[dict]:
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
            if getattr(args, "parity_profile", False):
                batch["lang_features"] = np.zeros(
                    (args.batch_size, 1, 512), dtype=np.float32
                )
            else:
                batch["lang_tokens"] = np.tile(
                    tokens[None, None], (args.batch_size, 1, 1)
                ).astype(np.int32)
        batch["action"] = rng.uniform(
            -1.0, 1.0, (args.batch_size, args.action_seq, args.action_dim)
        ).astype(np.float32)
        pad = np.zeros((args.batch_size, args.action_seq), dtype=bool)
        # A handful of rows end early, as episode tails do in BiGym replay.
        tail_rows = rng.integers(0, args.batch_size, (max(1, args.batch_size // 16),))
        for row in tail_rows:
            pad[row, args.action_seq - int(rng.integers(1, args.action_seq // 2 + 1)) :] = True
        batch["action_pad_mask"] = pad
        batch["reward"] = np.zeros((args.batch_size, 1), dtype=np.float32)
        batch["indices"] = np.arange(args.batch_size, dtype=np.int64)
        batches.append(batch)
    return batches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=tuple(LAUNCHES), required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-views", type=int, default=3)
    parser.add_argument("--low-dim", type=int, default=63)
    parser.add_argument("--action-dim", type=int, default=16)
    parser.add_argument("--action-seq", type=int, default=20)
    parser.add_argument("--lang", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--augmentation", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--fused-steps", type=int, default=1)
    parser.add_argument("--block-every", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", default="")
    parser.add_argument(
        "--parity-profile", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--device-resident", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    sampler = ProcessGpuMemorySampler().start()
    import jax

    device = jax.local_devices()[0]
    cfg = build_cfg(args)
    observation_space, action_space = build_spaces(args)

    from robobase.factory import create_agent

    build_start = time.perf_counter()
    agent = create_agent(
        cfg, observation_space=observation_space, action_space=action_space
    )
    build_seconds = time.perf_counter() - build_start
    agent.train(True)
    agent.logging = False
    path_leaves, _ = jax.tree_util.tree_flatten_with_path(agent.params)
    parameter_count = int(
        sum(int(np.prod(leaf.shape)) for _, leaf in path_leaves)
    )
    frozen_parameter_count = 0
    if args.method == "act":
        frozen_parameter_count = int(
            sum(
                int(np.prod(leaf.shape))
                for path, leaf in path_leaves
                if any(
                    str(getattr(entry, "key", entry)) == "BatchNorm_0"
                    for entry in path
                )
            )
        )

    batches = make_batches(args, args.num_batches)
    if args.device_resident:
        # Match the PyTorch worker, whose update inputs already live on GPU.
        # Language rows remain on host because the shared validation helper
        # intentionally normalises them through NumPy; they are tiny and the
        # resulting 512-vector is transferred before the compiled update.
        batches = [
            {
                key: value
                if key in {"lang_features", "lang_tokens"}
                else jax.device_put(value, device)
                for key, value in batch.items()
            }
            for batch in batches
        ]
    replay_iter = itertools.cycle(batches)
    fused = max(1, int(args.fused_steps))

    def step():
        if fused > 1:
            agent.update_many(replay_iter, fused, None)
        else:
            agent.update(replay_iter, 0, None)

    def block():
        jax.block_until_ready(agent.params)

    first_start = time.perf_counter()
    step()
    block()
    first_seconds = time.perf_counter() - first_start
    after_compile_stats = dict(device.memory_stats() or {})
    # Compilation-time autotuning scratch is not steady-state usage; keep it
    # as its own number and track the training-loop peak separately.
    compile_peak_mib = sampler.reset_peak()

    for _ in range(args.warmup):
        step()
    block()

    # Throughput: dispatch ``iterations`` calls back to back, one final block.
    # This is how the training loop runs (host prefetch overlaps device work).
    throughput_start = time.perf_counter()
    for _ in range(args.iterations):
        step()
    block()
    throughput_seconds = time.perf_counter() - throughput_start

    # Latency: block after every call.
    blocked_times = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        step()
        block()
        blocked_times.append(time.perf_counter() - start)

    final_stats = dict(device.memory_stats() or {})
    smi = sampler.stop()
    updates = args.iterations * fused
    if args.method == "act":
        augmentation = bool(cfg.method.actor_model.data_augmentation)
        image_augmentation_type = str(cfg.method.actor_model.image_augmentation_type)
    else:
        image_augmentation_type = str(cfg.method.image_augmentation_type)
        augmentation = image_augmentation_type.lower() != "none"
    result = {
        "backend": "jax",
        "label": args.label,
        "method": args.method,
        "launch": LAUNCHES[args.method],
        "commit": _git_commit(),
        "jax_version": jax.__version__,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "num_views": args.num_views,
        "low_dim": args.low_dim,
        "action_dim": args.action_dim,
        "action_seq": args.action_seq,
        "lang": bool(args.lang),
        "augmentation": augmentation,
        "image_augmentation_type": image_augmentation_type,
        "act_use_remat": (
            bool(cfg.method.actor_model.get("use_remat", True))
            if args.method == "act"
            else False
        ),
        "parity_profile": bool(args.parity_profile),
        "large_inputs_device_resident": bool(args.device_resident),
        "language_input": (
            "precomputed_zero_feature"
            if args.lang and args.parity_profile
            else "token_projection"
        ),
        "fused_steps": fused,
        "block_every": args.block_every,
        "overrides": list(args.override),
        "parameter_count": parameter_count,
        "differentiable_parameter_count": (
            parameter_count - frozen_parameter_count
        ),
        "build_seconds": build_seconds,
        "first_call_seconds": first_seconds,
        "throughput_updates_per_second": updates / throughput_seconds,
        "throughput_ms_per_update": throughput_seconds / updates * 1000.0,
        "throughput_samples_per_second": updates * args.batch_size / throughput_seconds,
        "blocked_call": _summary(blocked_times),
        "blocked_ms_per_update": statistics.median(blocked_times) / fused * 1000.0,
        "peak_bytes_in_use_after_compile_mib": after_compile_stats.get(
            "peak_bytes_in_use", 0
        )
        / 2**20,
        "peak_bytes_in_use_mib": final_stats.get("peak_bytes_in_use", 0) / 2**20,
        "bytes_in_use_mib": final_stats.get("bytes_in_use", 0) / 2**20,
        "bytes_limit_mib": final_stats.get("bytes_limit", 0) / 2**20,
        "nvidia_smi_compile_peak_mib": compile_peak_mib,
        **smi,
        "env": {
            key: os.environ.get(key)
            for key in (
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "XLA_PYTHON_CLIENT_MEM_FRACTION",
                "XLA_PYTHON_CLIENT_ALLOCATOR",
                "XLA_FLAGS",
                "JAX_DEFAULT_MATMUL_PRECISION",
                "CUDA_VISIBLE_DEVICES",
            )
        },
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
