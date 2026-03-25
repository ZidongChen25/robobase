from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robobase.envs.robomimic import RobomimicEnvFactory
from robobase.gpu import apply_requested_gpu

CFG_DIR = REPO_ROOT / "robobase" / "cfgs"
DEFAULT_STATE_DATASET = (
    "/home/zc1525/robobase/third_party_datasets/robomimic/tool_hang/ph/low_dim_v141.hdf5"
)
DEFAULT_IMAGE_DATASET = (
    "/home/zc1525/robobase/third_party_datasets/robomimic/tool_hang/ph/image_v141.hdf5"
)


def _configure_jax_logging():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)


def _torch_sync(device):
    import torch

    if device is None:
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std())
    return {
        "mean_sec": mean,
        "std_sec": std,
        "min_sec": float(arr.min()),
        "max_sec": float(arr.max()),
        "p50_sec": float(np.percentile(arr, 50)),
        "p90_sec": float(np.percentile(arr, 90)),
        "cv": float(std / mean) if mean > 0 else 0.0,
    }


def _compose_cfg(
    *,
    backend: str,
    modality: str,
    gpu_id: int,
    demos: str,
    cache_backends: str,
    batch_size: int,
    replay_num_workers: int,
    replay_save_dir: str,
    state_dataset_path: str,
    image_dataset_path: str,
) -> Any:
    GlobalHydra.instance().clear()
    dataset_path = state_dataset_path if modality == "state" else image_dataset_path
    overrides = [
        "launch=dp_state_robomimic",
        "env=robomimic/tool_hang",
        f"backend={backend}",
        f"gpu_id={gpu_id}",
        "num_gpus=1",
        "create_train_env=false",
        "num_train_frames=0",
        "num_pretrain_steps=200000",
        "num_eval_episodes=0",
        "num_eval_envs=1",
        "num_train_envs=1",
        f"demos={demos}",
        "env.use_live_env=false",
        f"pixels={'true' if modality == 'image' else 'false'}",
        f"env.dataset_path={dataset_path}",
        f"batch_size={batch_size}",
        "save_snapshot=false",
        "wandb.use=false",
        "tb.use=false",
        "log_eval_video=false",
        "log_pretrain_every=100",
        "method.use_ema=false",
        "method.num_diffusion_iters=100",
        f"replay.save_dir={replay_save_dir}",
        "replay.persist=true",
        "replay.reuse_saved=true",
        f"replay.cache_frozen_image_feature_backends=[{cache_backends}]",
        f"replay.num_workers={replay_num_workers}",
    ]
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CFG_DIR),
        job_name="benchmark_diffusion_breakdown",
    ):
        return compose(config_name="robobase_config", overrides=overrides)


def _benchmark_torch_step(agent, replay_iter, replay_buffer) -> dict[str, float]:
    from robobase.method.utils import extract_from_batch, loss_weights

    total_start = time.perf_counter()

    replay_start = time.perf_counter()
    batch = next(replay_iter)
    replay_next = time.perf_counter() - replay_start

    cached_pixel_features = bool(
        agent.use_pixels
        and hasattr(agent, "_has_cached_pixel_features")
        and agent._has_cached_pixel_features(batch)
    )
    transfer_start = time.perf_counter()
    keys_to_skip = set()
    if cached_pixel_features:
        keys_to_skip.update(agent.obs_layout.rgb_keys)
        keys_to_skip.update(f"{key}_tp1" for key in agent.obs_layout.rgb_keys)
    batch = {
        key: value.to(agent.device, non_blocking=True)
        for key, value in batch.items()
        if key not in keys_to_skip
    }
    _torch_sync(agent.device)
    device_transfer = time.perf_counter() - transfer_start

    prep_start = time.perf_counter()
    action = batch["action"]
    action_pad_mask = extract_from_batch(batch, "action_pad_mask", missing_ok=True)
    loss_coeff = loss_weights(batch, agent.replay_beta)
    low_dim_obs = None
    if agent.low_dim_size > 0:
        low_dim_obs, _ = agent.extract_low_dim_state(batch)
    _torch_sync(agent.device)
    prep_low_dim = time.perf_counter() - prep_start

    vision_time = 0.0
    fused_view_feats = None
    if agent.use_pixels:
        if cached_pixel_features:
            fused_view_feats = agent._extract_cached_pixel_features(batch)
        else:
            vision_start = time.perf_counter()
            rgb_obs, _, _ = agent.extract_pixels(batch)
            _, rgb_feats = agent.encode(rgb_obs)
            _, fused_view_feats = agent.multi_view_fusion(rgb_obs, rgb_feats)
            if fused_view_feats is not None:
                fused_view_feats = fused_view_feats.detach()
            _torch_sync(agent.device)
            vision_time = time.perf_counter() - vision_start

    core_start = time.perf_counter()
    agent.update_actor(
        low_dim_obs,
        fused_view_feats,
        action,
        loss_coeff,
        action_pad_mask=action_pad_mask,
        used_cached_pixel_features=cached_pixel_features,
    )
    _torch_sync(agent.device)
    core_update = time.perf_counter() - core_start

    priority_start = time.perf_counter()
    if replay_buffer.__class__.__name__ == "PrioritizedReplayBuffer":
        replay_buffer.set_priority(
            indices=batch["indices"].cpu().detach().numpy(),
            priorities=agent._new_priority**agent.replay_alpha,
        )
    priority_update = time.perf_counter() - priority_start

    total = time.perf_counter() - total_start
    return {
        "replay_next_sec": replay_next,
        "device_transfer_sec": device_transfer,
        "prep_low_dim_sec": prep_low_dim,
        "vision_encode_fuse_sec": vision_time,
        "core_update_sec": core_update,
        "priority_update_sec": priority_update,
        "total_sec": total,
        "segment_sum_sec": replay_next
        + device_transfer
        + prep_low_dim
        + vision_time
        + core_update
        + priority_update,
    }


def _benchmark_jax_step(agent, replay_iter, replay_buffer) -> dict[str, float]:
    total_start = time.perf_counter()

    replay_start = time.perf_counter()
    batch = next(replay_iter)
    replay_next = time.perf_counter() - replay_start

    prep_start = time.perf_counter()
    low_dim_obs = agent._extract_low_dim_batch(batch)
    actions = np.asarray(batch["action"], dtype=np.float32)
    action_pad_mask = agent._extract_action_pad_mask(batch)
    loss_coeff = agent._loss_weights(batch)
    fused_view_feats = None
    rgb_obs = None
    if agent.use_pixels and hasattr(agent, "_extract_cached_pixel_features"):
        fused_view_feats = agent._extract_cached_pixel_features(batch)
    if agent.use_pixels and fused_view_feats is None:
        rgb_obs, _ = agent._extract_rgb_obs(batch)
    prep_host = time.perf_counter() - prep_start

    vision_time = 0.0
    if rgb_obs is not None:
        vision_start = time.perf_counter()
        fused_view_feats = agent._fuse_multi_view(agent._encode_pixels(rgb_obs))
        agent._block(fused_view_feats)
        vision_time = time.perf_counter() - vision_start

    feature_start = time.perf_counter()
    obs_features = agent._combine_features(low_dim_obs, fused_view_feats)
    feature_combine = time.perf_counter() - feature_start

    core_start = time.perf_counter()
    (
        agent.params,
        agent.opt_state,
        agent.rng_key,
        actor_loss,
        new_priority,
        agent.ema_params,
        agent._ema_optimization_step,
    ) = agent._update_impl(
        agent.params,
        agent.opt_state,
        agent.rng_key,
        agent.jnp.asarray(obs_features),
        agent.jnp.asarray(actions),
        agent.jnp.asarray(loss_coeff),
        None if action_pad_mask is None else agent.jnp.asarray(action_pad_mask),
        agent.params if agent.ema_params is None else agent.ema_params,
        agent._ema_optimization_step,
    )
    agent._block(actor_loss, new_priority)
    core_update = time.perf_counter() - core_start

    priority_start = time.perf_counter()
    if replay_buffer.__class__.__name__ == "PrioritizedReplayBuffer":
        replay_buffer.set_priority(
            indices=np.asarray(batch["indices"]),
            priorities=np.asarray(agent.jax.device_get(new_priority), dtype=np.float32)
            ** agent.replay_alpha,
        )
    priority_update = time.perf_counter() - priority_start

    total = time.perf_counter() - total_start
    return {
        "replay_next_sec": replay_next,
        "prep_host_sec": prep_host,
        "feature_combine_sec": feature_combine,
        "vision_encode_fuse_sec": vision_time,
        "core_update_sec": core_update,
        "priority_update_sec": priority_update,
        "total_sec": total,
        "segment_sum_sec": replay_next
        + prep_host
        + feature_combine
        + vision_time
        + core_update
        + priority_update,
    }


def _run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    cfg = _compose_cfg(
        backend=args.backend,
        modality=args.modality,
        gpu_id=args.gpu_id,
        demos=args.demos,
        cache_backends=args.cache_backends,
        batch_size=args.batch_size,
        replay_num_workers=args.replay_num_workers,
        replay_save_dir=args.replay_save_dir,
        state_dataset_path=args.state_dataset_path,
        image_dataset_path=args.image_dataset_path,
    )
    apply_requested_gpu(cfg)
    if args.backend == "jax":
        _configure_jax_logging()

    work_dir = REPO_ROOT / "exp_local" / f"bench_workspace_{args.backend}_{args.modality}"
    work_dir.mkdir(parents=True, exist_ok=True)

    from robobase.workspace import Workspace

    workspace = Workspace(
        cfg,
        env_factory=RobomimicEnvFactory(),
        work_dir=work_dir,
    )
    try:
        workspace._load_demos()
        agent = workspace.agent
        agent.train(True)

        replay_dir = Path(args.replay_save_dir)
        replay_file_count = len(list(replay_dir.glob("*.npz"))) if replay_dir.exists() else 0

        if args.backend == "torch":
            step_fn = lambda: _benchmark_torch_step(
                agent,
                workspace.replay_iter,
                workspace.replay_buffer,
            )
        else:
            step_fn = lambda: _benchmark_jax_step(
                agent,
                workspace.replay_iter,
                workspace.replay_buffer,
            )

        for _ in range(args.warmup_iters):
            step_fn()

        records = [step_fn() for _ in range(args.measure_iters)]
        summary = {key: _summarize([record[key] for record in records]) for key in records[0]}

        total_mean = summary["total_sec"]["mean_sec"]
        throughput = {
            "batched_updates_per_second": 1.0 / total_mean if total_mean > 0 else 0.0,
            "samples_per_second": args.batch_size / total_mean if total_mean > 0 else 0.0,
        }

        return {
            "backend": args.backend,
            "modality": args.modality,
            "gpu_id": args.gpu_id,
            "batch_size": args.batch_size,
            "replay_num_workers": args.replay_num_workers,
            "warmup_iters": args.warmup_iters,
            "measure_iters": args.measure_iters,
            "replay_save_dir": args.replay_save_dir,
            "replay_file_count": replay_file_count,
            "summary": summary,
            "throughput": throughput,
        }
    finally:
        workspace.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("torch", "jax"), required=True)
    parser.add_argument("--modality", choices=("state", "image"), required=True)
    parser.add_argument("--gpu-id", type=int, default=3)
    parser.add_argument("--demos", default=".inf")
    parser.add_argument("--cache-backends", default="torch,jax")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-num-workers", type=int, default=4)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--measure-iters", type=int, default=12)
    parser.add_argument(
        "--state-dataset-path",
        default=DEFAULT_STATE_DATASET,
    )
    parser.add_argument(
        "--image-dataset-path",
        default=DEFAULT_IMAGE_DATASET,
    )
    parser.add_argument("--replay-save-dir", required=True)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    result = _run_benchmark(args)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
