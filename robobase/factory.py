from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from robobase.method.act import act_spec_from_cfg
from robobase.method.bc import bc_spec_from_cfg
from robobase.method.diffusion import diffusion_spec_from_cfg
from robobase.method.cqn import cqn_spec_from_cfg
from robobase.method.cqn_as import cqn_as_spec_from_cfg


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
        "robobase.method.act.ACT": "act",
        "robobase.method.act.ActBCAgent": "act",
        "robobase.method.bc.BC": "bc",
        "robobase.method.diffusion.Diffusion": "diffusion",
        "robobase.method.cqn.CQN": "cqn",
        "robobase.method.cqn_as.CQNAS": "cqn_as",
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
        update_block_every_steps=int(
            cfg.get("backend", {}).get("update_block_every_steps", 1)
            if cfg.get("backend", None)
            else 1
        ),
    )

    jit = bool(cfg.get("backend", {}).get("jit", True) if cfg.get("backend") else True)
    platform = (
        cfg.get("backend", {}).get("platform", None) if cfg.get("backend") else None
    )

    if method_name == "cqn":
        from robobase.method.cqn import CQN

        spec = cqn_spec_from_cfg(cfg)
        return CQN(
            critic_lr=spec.critic_lr,
            num_train_steps=spec.num_train_steps,
            num_explore_steps=spec.num_explore_steps,
            critic_target_tau=spec.critic_target_tau,
            critic_grad_clip=spec.critic_grad_clip,
            weight_decay=spec.weight_decay,
            levels=spec.levels,
            bins=spec.bins,
            atoms=spec.atoms,
            v_min=spec.v_min,
            v_max=spec.v_max,
            critic_lambda=spec.critic_lambda,
            centralized_critic=spec.centralized_critic,
            use_dueling=spec.use_dueling,
            always_bootstrap=spec.always_bootstrap,
            stddev_schedule=spec.stddev_schedule,
            bc_lambda=spec.bc_lambda,
            bc_margin=spec.bc_margin,
            use_target_network_for_rollout=spec.use_target_network_for_rollout,
            num_update_steps=spec.num_update_steps,
            model=spec.model,
            jit=jit,
            platform=platform,
            seed=int(cfg.seed),
            **common_kwargs,
        )

    if method_name == "cqn_as":
        from robobase.method.cqn_as import CQNAS

        spec = cqn_as_spec_from_cfg(cfg)
        if int(cfg.execution_length) != 1:
            raise ValueError("CQN-AS requires execution_length=1.")
        if int(cfg.get("action_execution_start", 0)) != 0:
            raise ValueError("CQN-AS requires action_execution_start=0.")
        if bool(cfg.get("temporal_ensemble", False)):
            raise ValueError(
                "CQN-AS rollout control is implemented inside the agent so "
                "exploration noise is applied after action selection; set the "
                "root temporal_ensemble=false."
            )
        return CQNAS(
            critic_lr=spec.critic_lr,
            num_train_steps=spec.num_train_steps,
            num_explore_steps=spec.num_explore_steps,
            critic_target_tau=spec.critic_target_tau,
            critic_grad_clip=spec.critic_grad_clip,
            weight_decay=spec.weight_decay,
            levels=spec.levels,
            bins=spec.bins,
            atoms=spec.atoms,
            v_min=spec.v_min,
            v_max=spec.v_max,
            critic_lambda=spec.critic_lambda,
            centralized_critic=spec.centralized_critic,
            use_dueling=spec.use_dueling,
            always_bootstrap=spec.always_bootstrap,
            stddev_schedule=spec.stddev_schedule,
            bc_lambda=spec.bc_lambda,
            bc_margin=spec.bc_margin,
            use_target_network_for_rollout=spec.use_target_network_for_rollout,
            num_update_steps=spec.num_update_steps,
            gru_layers=spec.gru_layers,
            temporal_ensemble=spec.temporal_ensemble,
            temporal_ensemble_replan_interval=(
                spec.temporal_ensemble_replan_interval
            ),
            temporal_ensemble_gain=spec.temporal_ensemble_gain,
            tie_break_delta=spec.tie_break_delta,
            model=spec.model,
            jit=jit,
            platform=platform,
            seed=int(cfg.seed),
            **common_kwargs,
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

    if method_name == "act":
        from robobase.method.act import ACT

        spec = act_spec_from_cfg(cfg)
        return ACT(
            lr=spec.lr,
            adaptive_lr=spec.adaptive_lr,
            num_train_steps=spec.num_train_steps,
            actor_grad_clip=spec.actor_grad_clip,
            weight_decay=spec.weight_decay,
            lr_backbone=spec.lr_backbone,
            horizon_dropout_lengths=spec.horizon_dropout_lengths,
            horizon_dropout_probs=spec.horizon_dropout_probs,
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
            lr_schedule=spec.lr_schedule,
            actor_grad_clip=spec.actor_grad_clip,
            objective_type=spec.objective_type,
            num_diffusion_iters=spec.num_diffusion_iters,
            sampler=spec.sampler,
            use_ema=spec.use_ema,
            ema_decay=spec.ema_decay,
            ema_decay_schedule=spec.ema_decay_schedule,
            weight_decay=spec.weight_decay,
            model=spec.model,
            jit=bool(cfg.get("backend", {}).get("jit", True) if cfg.get("backend") else True),
            platform=cfg.get("backend", {}).get("platform", None) if cfg.get("backend") else None,
            seed=int(cfg.seed),
            **common_kwargs,
        )

    raise NotImplementedError(
        f"Unsupported method '{method_name}'. Supported: act, bc, cqn, cqn_as, "
        "diffusion."
    )
