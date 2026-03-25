from __future__ import annotations

from omegaconf import DictConfig

from robobase.method.bc import BCSpec
from robobase.method.diffusion import DiffusionSpec


def create_jax_agent(
    cfg: DictConfig,
    *,
    method_name: str,
    common_kwargs: dict,
):
    if common_kwargs["intrinsic_reward_module"] is not None:
        raise NotImplementedError(
            "The minimal JAX backend does not support intrinsic reward modules yet."
        )

    if method_name not in {"bc", "diffusion"}:
        raise NotImplementedError(
            "The minimal JAX backend currently supports only method=bc and method=diffusion."
        )

    raise AssertionError(
        f"{method_name.upper()} should be created through the shared backend path."
    )


def create_jax_bc_agent(
    cfg: DictConfig,
    *,
    spec: BCSpec,
    common_kwargs: dict,
    device=None,
):
    from robobase.backends.jax.method.bc import JaxBC

    return JaxBC(
        lr=spec.lr,
        adaptive_lr=spec.adaptive_lr,
        num_train_steps=spec.num_train_steps,
        actor_grad_clip=spec.actor_grad_clip,
        model=spec.model,
        jit=bool(cfg.backend.get("jit", True)),
        platform=cfg.backend.get("platform", None),
        seed=int(cfg.seed),
        **common_kwargs,
    )


def create_jax_diffusion_agent(
    cfg: DictConfig,
    *,
    spec: DiffusionSpec,
    common_kwargs: dict,
    device=None,
):
    del device
    from robobase.backends.jax.method.diffusion import JaxDiffusion

    return JaxDiffusion(
        lr=spec.lr,
        adaptive_lr=spec.adaptive_lr,
        num_train_steps=spec.num_train_steps,
        actor_grad_clip=spec.actor_grad_clip,
        num_diffusion_iters=spec.num_diffusion_iters,
        use_ema=spec.use_ema,
        model=spec.model,
        jit=bool(cfg.backend.get("jit", True)),
