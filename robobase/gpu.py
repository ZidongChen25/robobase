from __future__ import annotations

import os

from omegaconf import DictConfig, open_dict


def _visible_gpu_ids() -> list[int]:
    raw_value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw_value:
        return []
    gpu_ids = []
    for token in raw_value.split(","):
        token = token.strip()
        if token.isdigit():
            gpu_ids.append(int(token))
    return gpu_ids


def apply_requested_gpu(cfg: DictConfig):
    requested_gpu = cfg.get("gpu_id", None)
    selected_gpu = None
    if requested_gpu is not None:
        requested_gpu = int(requested_gpu)
        if requested_gpu >= 0:
            selected_gpu = requested_gpu
            gpu_str = str(requested_gpu)
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
            os.environ["JAX_CUDA_VISIBLE_DEVICES"] = gpu_str
            with open_dict(cfg):
                cfg.num_gpus = 1
    else:
        visible_gpu_ids = _visible_gpu_ids()
        if visible_gpu_ids:
            selected_gpu = visible_gpu_ids[0]

    if selected_gpu is None:
        return

    # Respect an externally chosen EGL device: EGL enumeration order can
    # differ from CUDA order (e.g. a degraded card shifting indices), so
    # callers may need to render on a different EGL device than they
    # compute on.
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", str(selected_gpu))

    with open_dict(cfg):
        if cfg.get("env", None) is not None:
            cfg.env.render_gpu_device_id = selected_gpu
