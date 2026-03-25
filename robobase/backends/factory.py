from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from robobase.method.bc import bc_spec_from_cfg
from robobase.method.diffusion import diffusion_spec_from_cfg


def backend_name_from_cfg(cfg: DictConfig) -> str:
    backend_cfg = cfg.get("backend", None)
    if backend_cfg is None:
        return "torch"
    return str(backend_cfg.get("name", "torch")).lower()


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
        "robobase.backends.torch.method.bc.BC": "bc",
        "robobase.backends.jax.bc.JaxBC": "bc",
        "robobase.backends.jax.method.bc.JaxBC": "bc",
        "robobase.method.diffusion.Diffusion": "diffusion",
        "robobase.backends.torch.method.diffusion.Diffusion": "diffusion",
        "robobase.backends.jax.method.diffusion.JaxDiffusion": "diffusion",
    }
    if method_target in target_to_name:
        return target_to_name[method_target]

    return method_target.rsplit(".", maxsplit=1)[-1].lower()


def create_agent(
    cfg: DictConfig,
    *,
    observation_space: Any,
    action_space: Any,
    device: Any,
    intrinsic_reward_module: Any = None,
):
    backend_name = backend_name_from_cfg(cfg)
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
        spec = bc_spec_from_cfg(cfg)

        if backend_name == "torch":
            from robobase.backends.torch.factory import create_torch_bc_agent

            return create_torch_bc_agent(
                cfg,
                spec=spec,
                device=device,
                common_kwargs=common_kwargs,
            )

        if backend_name == "jax":
            from robobase.backends.jax.factory import create_jax_bc_agent

            return create_jax_bc_agent(
                cfg,
                spec=spec,
                device=device,
                common_kwargs=common_kwargs,
            )

        raise ValueError(f"Unsupported backend '{backend_name}' for BC.")
    if method_name == "diffusion":
        spec = diffusion_spec_from_cfg(cfg)

        if backend_name == "torch":
            from robobase.backends.torch.factory import create_torch_diffusion_agent

            return create_torch_diffusion_agent(
                cfg,
                spec=spec,
                device=device,
                common_kwargs=common_kwargs,
            )

        if backend_name == "jax":
            from robobase.backends.jax.factory import create_jax_diffusion_agent

            return create_jax_diffusion_agent(
                cfg,
                spec=spec,
                common_kwargs=common_kwargs,
            )

    if backend_name == "torch":
        from robobase.backends.torch.factory import create_torch_agent

        return create_torch_agent(
            cfg,
            method_name=method_name,
            device=device,
            common_kwargs=common_kwargs,
        )

    if backend_name != "jax":
        raise ValueError(f"Unsupported backend '{backend_name}'.")

    from robobase.backends.jax.factory import create_jax_agent

    return create_jax_agent(
        cfg,
        method_name=method_name,
        common_kwargs=common_kwargs,
    )
