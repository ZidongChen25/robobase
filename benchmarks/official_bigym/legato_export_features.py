#!/usr/bin/env python3
"""Export aligned H8 BiGym windows through a restored FM visual encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--fm-snapshot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--pixel-dataset-root", type=Path)
    parser.add_argument("--state-dataset-root", type=Path)
    parser.add_argument("--lang-feature-path", type=Path)
    parser.add_argument("--gpu-id", type=int)
    return parser.parse_args()


def _configure_process(gpu_id: int | None) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if gpu_id is not None and gpu_id >= 0:
        gpu = str(gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        os.environ["JAX_CUDA_VISIBLE_DEVICES"] = gpu
        os.environ["MUJOCO_EGL_DEVICE_ID"] = gpu


def _set_if_present(cfg, dotted_key: str, value) -> None:
    node = cfg
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in node or node[part] is None:
            return
        node = node[part]
    node[parts[-1]] = value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(args: argparse.Namespace) -> dict:
    import jax
    import numpy as np
    from omegaconf import OmegaConf

    from benchmarks.official_bigym.legato_data import (
        WindowDataset,
        save_window_dataset,
    )
    from benchmarks.official_bigym.legato_features import FrozenFMVisualFeatures
    from benchmarks.official_bigym.legato_upstream import UPSTREAM_COMMIT
    from robobase.replay_buffer.uniform_replay_buffer import ACTION, ACTION_PAD_MASK
    from robobase.workspace import Workspace

    cfg_path = args.run_dir / ".hydra" / "config.yaml"
    for path in (cfg_path, args.fm_snapshot):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.horizon <= 0 or args.stride <= 0 or args.batch_size <= 0:
        raise ValueError("horizon, stride, and batch size must be positive.")
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError("max-episodes must be positive when specified.")

    cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_struct(cfg, False)
    if str(cfg.env.task_name) != "flip_cutlery":
        raise ValueError(
            f"This exporter is locked to flip_cutlery, got {cfg.env.task_name!r}."
        )
    cfg.create_train_env = False
    cfg.num_train_envs = 0
    cfg.num_train_frames = 0
    cfg.num_eval_episodes = 0
    cfg.num_eval_envs = 0
    cfg.log_train_video = False
    cfg.log_eval_video = False
    cfg.save_snapshot = False
    cfg.save_csv = False
    cfg.gpu_id = None
    if args.pixel_dataset_root is not None:
        cfg.env.pixel_dataset_root = str(args.pixel_dataset_root.expanduser().resolve())
    if args.state_dataset_root is not None:
        cfg.env.state_dataset_root = str(args.state_dataset_root.expanduser().resolve())
    if args.lang_feature_path is not None:
        cfg.method.lang_feature_source = "precomputed"
        cfg.method.lang_feature_path = str(args.lang_feature_path.expanduser().resolve())
    elif str(cfg.method.get("lang_feature_source", "tokens")) not in {
        "tokens",
        "jax",
        "hash",
        "precomputed",
    }:
        cfg.method.lang_feature_source = "tokens"
    _set_if_present(cfg, "wandb.use", False)
    _set_if_present(cfg, "tb.use", False)
    _set_if_present(cfg, "replay.num_workers", 0)
    _set_if_present(cfg, "lazy_replay.num_workers", 0)
    _set_if_present(cfg, "lazy_replay.persistent_workers", False)
    _set_if_present(cfg, "backend.replay_prefetch_size", 0)
    _set_if_present(cfg, "backend.replay_device_prefetch", False)
    OmegaConf.resolve(cfg)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    workspace = Workspace(cfg, work_dir=str(args.work_dir))
    try:
        workspace.load_snapshot(args.fm_snapshot, load_replay_buffer=False)
        replay = workspace.replay_buffer
        required = ("episode_index_metadata", "sample_batch_indices")
        if not all(callable(getattr(replay, name, None)) for name in required):
            raise TypeError("Feature export requires the BiGym lazy replay buffer.")
        if int(getattr(replay, "_action_seq_len", 0)) < args.horizon:
            raise ValueError(
                "The FM replay action horizon is shorter than the export horizon."
            )
        boundary = FrozenFMVisualFeatures(workspace.agent)
        episode_metadata = replay.episode_index_metadata()
        if args.max_episodes is not None:
            episode_metadata = episode_metadata[: args.max_episodes]

        features = []
        chunks = []
        episode_indices = []
        start_indices = []
        for episode_index, (global_start, transition_len) in enumerate(
            episode_metadata
        ):
            local_starts = np.arange(
                0,
                max(0, int(transition_len) - args.horizon + 1),
                args.stride,
                dtype=np.int64,
            )
            for offset in range(0, len(local_starts), args.batch_size):
                local = local_starts[offset : offset + args.batch_size]
                indices = int(global_start) + local
                batch = replay.sample_batch_indices(indices)
                valid = ~np.asarray(batch[ACTION_PAD_MASK])[:, : args.horizon].any(1)
                if not valid.any():
                    continue
                observation_keys = getattr(replay, "observation_elements", {})
                observations = {
                    key: np.asarray(batch[key])[valid] for key in observation_keys
                }
                encoded = boundary.encode_numpy(observations)
                features.append(encoded)
                chunks.append(
                    np.asarray(batch[ACTION], dtype=np.float32)[
                        valid, : args.horizon
                    ]
                )
                selected = local[valid]
                episode_indices.append(
                    np.full(len(selected), episode_index, dtype=np.int32)
                )
                start_indices.append(selected.astype(np.int32))

        if not features:
            raise ValueError("No full, unpadded action windows were exported.")
        dataset = WindowDataset(
            np.concatenate(features),
            np.concatenate(chunks),
            np.concatenate(episode_indices),
            np.concatenate(start_indices),
        )
        save_window_dataset(args.output, dataset)
    finally:
        workspace.shutdown()

    return {
        "status": "ok",
        "task": "flip_cutlery",
        "run_dir": str(args.run_dir.resolve()),
        "fm_snapshot": str(args.fm_snapshot.resolve()),
        "fm_snapshot_sha256": _sha256(args.fm_snapshot),
        "official_upstream_commit": UPSTREAM_COMMIT,
        "output": str(args.output.resolve()),
        "num_episodes": len(episode_metadata),
        "num_windows": len(dataset),
        "feature_dim": int(dataset.features.shape[-1]),
        "action_horizon": int(dataset.action_chunks.shape[1]),
        "action_dim": int(dataset.action_chunks.shape[-1]),
        "stride": int(args.stride),
        "lang_feature_source": str(cfg.method.get("lang_feature_source", "tokens")),
        "lang_feature_path": cfg.method.get("lang_feature_path", None),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
    }


def main() -> int:
    args = _parse_args()
    _configure_process(args.gpu_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    started = time.time()
    try:
        payload = _run(args)
        payload["elapsed_seconds"] = time.time() - started
        metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return 0
    except BaseException as exc:
        payload = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
