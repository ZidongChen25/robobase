from __future__ import annotations

import hydra
from omegaconf import DictConfig, open_dict

from robobase.method.bc import BCSpec
from robobase.method.diffusion import DiffusionSpec


def _instantiate_method_cfg(
    method_cfg: DictConfig,
    *,
    device,
    common_kwargs: dict,
    override_kwargs: dict | None = None,
):
    original_name = method_cfg.get("name", None)
    with open_dict(method_cfg):
        if "name" in method_cfg:
            del method_cfg["name"]
    try:
        instantiate_kwargs = dict(common_kwargs)
        if override_kwargs is not None:
            instantiate_kwargs.update(override_kwargs)
        return hydra.utils.instantiate(
            method_cfg,
            device=device,
            **instantiate_kwargs,
        )
    finally:
        with open_dict(method_cfg):
            if original_name is not None:
                method_cfg.name = original_name


def create_torch_agent(
    cfg: DictConfig,
    *,
    method_name: str,
    device,
    common_kwargs: dict,
):
    if method_name not in {"bc", "diffusion"}:
        return _instantiate_method_cfg(
            cfg.method,
            device=device,
            common_kwargs=common_kwargs,
        )

    raise AssertionError(
        f"{method_name.upper()} should be created through the shared backend path."
    )


def create_torch_bc_agent(
    cfg: DictConfig,
    *,
    spec: BCSpec,
    device,
    common_kwargs: dict,
):
    from robobase.backends.torch.method.bc import BC

    return BC(
        lr=spec.lr,
        adaptive_lr=spec.adaptive_lr,
        num_train_steps=spec.num_train_steps,
        actor_grad_clip=spec.actor_grad_clip,
        model=spec.model,
        device=device,
        **common_kwargs,
    )


def create_torch_diffusion_agent(
    cfg: DictConfig,
    *,
    spec: DiffusionSpec,
    device,
    common_kwargs: dict,
):
    from robobase.backends.torch.method.diffusion import Diffusion

    return Diffusion(
        lr=spec.lr,
        adaptive_lr=spec.adaptive_lr,
        num_train_steps=spec.num_train_steps,
        actor_grad_clip=spec.actor_grad_clip,
        num_diffusion_iters=spec.num_diffusion_iters,
        use_ema=spec.use_ema,
        model=spec.model,
        device=device,
