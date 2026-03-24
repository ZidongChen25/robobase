from __future__ import annotations

import os

from omegaconf import DictConfig, open_dict


def apply_requested_gpu(cfg: DictConfig):
    requested_gpu = cfg.get("gpu_id", None)
    if requested_gpu is None:
        return

    requested_gpu = int(requested_gpu)
    if requested_gpu < 0:
        return

    gpu_str = str(requested_gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    os.environ["JAX_CUDA_VISIBLE_DEVICES"] = gpu_str
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"

    with open_dict(cfg):
        cfg.num_gpus = 1
        if cfg.get("env", None) is not None:
            cfg.env.render_gpu_device_id = 0
