from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from robobase.method.bc import bc_spec_from_cfg
from robobase.method.diffusion import diffusion_spec_from_cfg


def method_name_from_cfg(cfg: DictConfig) -> str:
    method_cfg = cfg.get("method", None)
    if method_cfg is None:
        raise ValueError("Method config is missing.")

    method_name = method_cfg.get("name", None)
    if method_name is not None:
        return str(method_name).lower()

    method_target = str(method_cfg.get("_target_", "")).strip()
    if not method_target:
        raise ValueError("Method config does not define a name or _target_.")

    target_to_name = {
        "robobase.method.bc.BC": "bc",
        "robobase.method.diffusion.Diffusion": "diffusion",
    }
    if method_target in target_to_name:
        return target_to_name[method_target]

    return method_target.rsplit(".", maxsplit=1)[-1].lower()


def create_agent(
    cfg: DictConfig,
    *,
    observation_space: Any,
    action_space: Any,
    intrinsic_reward_module: Any = None,
    **_ignored,
):
    method_name = method_name_from_cfg(cfg)
    common_kwargs = dict(
        observation_space=observation_space,
        action_space=action_space,
        num_train_envs=cfg.num_train_envs,
        num_eval_envs=cfg.num_eval_envs,
        replay_alpha=cfg.replay.alpha,
        replay_beta=cfg.replay.beta,
        frame_stack_on_channel=cfg.frame_stack_on_channel,
        intrinsic_reward_module=intrinsic_reward_module,
    )

    if method_name == "bc":
        from robobase.method.bc import BC

        spec = bc_spec_from_cfg(cfg)
        return BC(
            lr=spec.lr,
            adaptive_lr=spec.adaptive_lr,
            num_train_steps=spec.num_train_steps,
            actor_grad_clip=spec.actor_grad_clip,
            model=spec.model,
            jit=bool(cfg.get("backend", {}).get("jit", True) if cfg.get("backend") else True),
            platform=cfg.get("backend", {}).get("platform", None) if cfg.get("backend") else None,
            seed=int(cfg.seed),
            **common_kwargs,
        )

    if method_name == "diffusion":
        from robobase.method.diffusion import Diffusion

        spec = diffusion_spec_from_cfg(cfg)
        return Diffusion(
            lr=spec.lr,
            adaptive_lr=spec.adaptive_lr,
            num_train_steps=spec.num_train_steps,
            actor_grad_clip=spec.actor_grad_clip,
            num_diffusion_iters=spec.num_diffusion_iters,
            use_ema=spec.use_ema,
            ema_decay=spec.ema_decay,
            weight_decay=spec.weight_decay,
            model=spec.model,
            jit=bool(cfg.get("backend", {}).get("jit", True) if cfg.get("backend") else True),
            platform=cfg.get("backend", {}).get("platform", None) if cfg.get("backend") else None,
            seed=int(cfg.seed),
            **common_kwargs,
        )

    raise NotImplementedError(
        f"Unsupported method '{method_name}'. Supported: bc, diffusion."
    )
