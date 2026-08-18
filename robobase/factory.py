from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from robobase.method.act import act_spec_from_cfg
from robobase.method.bc import bc_spec_from_cfg
from robobase.method.diffusion import diffusion_spec_from_cfg
from robobase.method.flow_matching import (
    flow_matching_spec_from_cfg,
    validate_legato_overlap,
)
from robobase.method.cqn import cqn_spec_from_cfg
from robobase.method.cqn_as import cqn_as_spec_from_cfg
from robobase.method.cqn_flow import cqn_flow_spec_from_cfg
from robobase.method.drqv2 import drqv2_spec_from_cfg
from robobase.method.djcqn import djcqn_spec_from_cfg, validate_djcqn_config
from robobase.method.ppo import ppo_spec_from_cfg
from robobase.method.q_chunking import (
    q_chunking_spec_from_cfg,
    validate_q_chunking_config,
)
from robobase.method.sac import sac_spec_from_cfg


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
        "robobase.method.flow_matching.FlowMatching": "flow_matching",
        "robobase.method.a2a.A2A": "a2a",
        "robobase.method.legato.Legato": "legato",
        "robobase.method.ppo.PPO": "ppo",
        "robobase.method.q_chunking.QChunking": "q_chunking",
        "robobase.method.sac.SAC": "sac",
        "robobase.method.alix.ALIX": "alix",
        "robobase.method.cqn.CQN": "cqn",
        "robobase.method.cqn_as.CQNAS": "cqn_as",
        "robobase.method.cqn_flow.CQNFlowAS": "cqn_flow",
        "robobase.method.dreamerv3.DreamerV3": "dreamerv3",
        "robobase.method.drm.DrM": "drm",
        "robobase.method.drqv2.DrQV2": "drqv2",
        "robobase.method.djcqn.DJCQN": "djcqn",
        "robobase.method.edp.DiffusionRL": "edp",
        "robobase.method.iql_drqv2.IQLDrQV2": "iql_drqv2",
        "robobase.method.mwm.MaskedWorldModel": "mwm",
        "robobase.method.sac_lix.SACLix": "sac_lix",
        "robobase.method.value_based.ValueBased": "value_based",
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
    removed_methods = {
        "alix",
        "dreamerv3",
        "drm",
        "edp",
        "iql_drqv2",
        "mwm",
        "sac_lix",
        "value_based",
    }
    if method_name in removed_methods:
        raise NotImplementedError(
            f"Method '{method_name}' is a historical Torch configuration and "
            "has no implementation in this JAX-only runtime. Supported JAX "
            "methods: a2a, act, bc, cqn, cqn_as, cqn_flow, diffusion, drqv2, "
            "flow_matching, "
            "legato, ppo, q_chunking, djcqn, sac."
        )
    unsupported_reason = cfg.method.get("unsupported_reason", None)
    if unsupported_reason is not None:
        raise NotImplementedError(
            f"Method '{method_name}' is disabled: {unsupported_reason}. "
            "Supported JAX methods: a2a, act, bc, cqn, cqn_as, cqn_flow, "
            "diffusion, drqv2, flow_matching, legato, ppo, q_chunking, djcqn, sac."
        )
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

    if method_name == "ppo":
        from robobase.method.ppo import PPO

        spec = ppo_spec_from_cfg(cfg)
        return PPO(
            lr=spec.lr,
            num_train_steps=spec.num_train_steps,
            rollout_steps=spec.rollout_steps,
            batch_size=spec.batch_size,
            num_epochs=spec.num_epochs,
            gamma=spec.gamma,
            gae_lambda=spec.gae_lambda,
            clip_range=spec.clip_range,
            clip_range_vf=spec.clip_range_vf,
            normalize_advantage=spec.normalize_advantage,
            entropy_coef=spec.entropy_coef,
            value_coef=spec.value_coef,
            max_grad_norm=spec.max_grad_norm,
            target_kl=spec.target_kl,
            model=spec.model,
            jit=jit,
            platform=platform,
            seed=int(cfg.seed),
            **common_kwargs,
        )

    if method_name == "sac":
        from robobase.method.sac import SAC

        spec = sac_spec_from_cfg(cfg)
        return SAC(
            actor_lr=spec.actor_lr,
            critic_lr=spec.critic_lr,
            alpha_lr=spec.alpha_lr,
            num_train_steps=spec.num_train_steps,
            num_explore_steps=spec.num_explore_steps,
            critic_target_tau=spec.critic_target_tau,
            init_temperature=spec.init_temperature,
            target_entropy=spec.target_entropy,
            actor_grad_clip=spec.actor_grad_clip,
            critic_grad_clip=spec.critic_grad_clip,
            weight_decay=spec.weight_decay,
            model=spec.model,
            jit=jit,
            platform=platform,
            seed=int(cfg.seed),
            **common_kwargs,
        )

    if method_name == "drqv2":
        from robobase.method.drqv2 import DrQV2

        spec = drqv2_spec_from_cfg(cfg)
        return DrQV2(
            actor_lr=spec.actor_lr,
            critic_lr=spec.critic_lr,
            encoder_lr=spec.encoder_lr,
            num_train_steps=spec.num_train_steps,
            num_explore_steps=spec.num_explore_steps,
            critic_target_tau=spec.critic_target_tau,
            stddev_schedule=spec.stddev_schedule,
            stddev_clip=spec.stddev_clip,
            use_augmentation=spec.use_augmentation,
            augmentation_pad=spec.augmentation_pad,
            num_critics=spec.num_critics,
            feature_dim=spec.feature_dim,
            actor_uses_time=spec.actor_uses_time,
            always_bootstrap=spec.always_bootstrap,
            bc_lambda=spec.bc_lambda,
            actor_grad_clip=spec.actor_grad_clip,
            critic_grad_clip=spec.critic_grad_clip,
            weight_decay=spec.weight_decay,
            model=spec.model,
            jit=jit,
            platform=platform,
            seed=int(cfg.seed),
            **common_kwargs,
        )

    if method_name == "q_chunking":
        from robobase.method.q_chunking import QChunking

        validate_q_chunking_config(
            action_sequence=int(cfg.action_sequence),
            execution_length=int(cfg.execution_length),
            replay_nstep=int(cfg.replay.nstep),
            temporal_ensemble=bool(cfg.get("temporal_ensemble", False)),
            action_execution_start=int(cfg.get("action_execution_start", 0)),
        )
        spec = q_chunking_spec_from_cfg(cfg)
        return QChunking(
            actor_lr=spec.actor_lr,
            critic_lr=spec.critic_lr,
            num_train_steps=spec.num_train_steps,
            num_explore_steps=spec.num_explore_steps,
            critic_target_tau=spec.critic_target_tau,
            flow_steps=spec.flow_steps,
            actor_num_samples=spec.actor_num_samples,
            q_aggregate=spec.q_aggregate,
            actor_grad_clip=spec.actor_grad_clip,
            critic_grad_clip=spec.critic_grad_clip,
            weight_decay=spec.weight_decay,
            model=spec.model,
            jit=jit,
            platform=platform,
            seed=int(cfg.seed),
            **common_kwargs,
        )

    if method_name == "djcqn":
        from robobase.method.djcqn import DJCQN

        validate_djcqn_config(
            action_sequence=int(cfg.action_sequence),
            execution_length=int(cfg.execution_length),
            replay_nstep=int(cfg.replay.nstep),
            temporal_ensemble=bool(cfg.get("temporal_ensemble", False)),
            prefix_horizon=int(cfg.method.get("prefix_horizon", 1)),
            action_execution_start=int(cfg.get("action_execution_start", 0)),
        )
        spec = djcqn_spec_from_cfg(cfg)
        return DJCQN(
            critic_lr=spec.critic_lr,
            num_train_steps=spec.num_train_steps,
            num_explore_steps=spec.num_explore_steps,
            critic_target_tau=spec.critic_target_tau,
            levels=spec.levels,
            bins=spec.bins,
            beam_width=spec.beam_width,
            num_critics=spec.num_critics,
            prefix_expectile=spec.prefix_expectile,
            q_aggregate=spec.q_aggregate,
            eval_lcb_beta=spec.eval_lcb_beta,
            sibling_exploration_prob=spec.sibling_exploration_prob,
            sibling_level=spec.sibling_level,
            critic_grad_clip=spec.critic_grad_clip,
            weight_decay=spec.weight_decay,
            model=spec.model,
            jit=jit,
            platform=platform,
            seed=int(cfg.seed),
            **common_kwargs,
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
        if bool(cfg.method.get("direct_scalar_q", False)):
            from robobase.method.cqn_direct_q import CQNDirectQAS

            cqn_as_cls = CQNDirectQAS
        else:
            from robobase.method.cqn_as import CQNAS

            cqn_as_cls = CQNAS

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
        direct_q_kwargs = {}
        if bool(cfg.method.get("direct_scalar_q", False)):
            direct_q_kwargs["direct_q_loss"] = str(
                cfg.method.get("direct_q_loss", "mse")
            )
            direct_q_kwargs["direct_q_huber_delta"] = float(
                cfg.method.get("direct_q_huber_delta", 1.0)
            )
            direct_q_kwargs["causal_rct_weight"] = float(
                cfg.method.get("causal_rct_weight", 0.0)
            )
            causal_rct_level = cfg.method.get("causal_rct_level", None)
            direct_q_kwargs["causal_rct_level"] = (
                None
                if causal_rct_level is None
                else int(causal_rct_level)
            )
            direct_q_kwargs["freeze_bc_policy"] = bool(
                cfg.method.get("freeze_bc_policy", False)
            )
            direct_q_kwargs["bc_policy_mode"] = str(
                cfg.method.get("bc_policy_mode", "behavior_logits")
            ).lower()
            frozen_policy_snapshot = cfg.method.get(
                "frozen_policy_snapshot",
                None,
            )
            direct_q_kwargs["frozen_policy_snapshot"] = (
                None
                if frozen_policy_snapshot is None
                else str(frozen_policy_snapshot)
            )
        return cqn_as_cls(
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
            bc_lambda_schedule=spec.bc_lambda_schedule,
            bin_flip_prob=spec.bin_flip_prob,
            bin_flip_level=spec.bin_flip_level,
            bin_explore_probs=spec.bin_explore_probs,
            bin_explore_schedule=spec.bin_explore_schedule,
            bin_explore_persist_plans=spec.bin_explore_persist_plans,
            low_dim_mask_prob=spec.low_dim_mask_prob,
            low_dim_mask_keep_last=spec.low_dim_mask_keep_last,
            bc_margin=spec.bc_margin,
            demo_fosd=spec.demo_fosd,
            use_target_network_for_rollout=spec.use_target_network_for_rollout,
            num_update_steps=spec.num_update_steps,
            gru_layers=spec.gru_layers,
            temporal_ensemble=spec.temporal_ensemble,
            temporal_ensemble_replan_interval=(
                spec.temporal_ensemble_replan_interval
            ),
            temporal_ensemble_gain=spec.temporal_ensemble_gain,
            tie_break_delta=spec.tie_break_delta,
            random_levels_from=spec.random_levels_from,
            level_override_mode=spec.level_override_mode,
            post_ensemble_random_keep_levels=(
                spec.post_ensemble_random_keep_levels
            ),
            post_ensemble_fixed_leaf=spec.post_ensemble_fixed_leaf,
            post_ensemble_l1_flip_prob=spec.post_ensemble_l1_flip_prob,
            post_ensemble_l2_flip_prob=spec.post_ensemble_l2_flip_prob,
            post_ensemble_l1_flip_horizon=(
                spec.post_ensemble_l1_flip_horizon
            ),
            structured_exploration_prob=spec.structured_exploration_prob,
            structured_exploration_level=spec.structured_exploration_level,
            structured_exploration_horizon=(
                spec.structured_exploration_horizon
            ),
            separate_bc_policy=spec.separate_bc_policy,
            bc_policy_stop_gradient=spec.bc_policy_stop_gradient,
            distinct_policy_encoder=spec.distinct_policy_encoder,
            td_target_action_source=spec.td_target_action_source,
            demo_behavior_force_probability=(
                spec.demo_behavior_force_probability
            ),
            td_target_policy_value_beta=(
                spec.td_target_policy_value_beta
            ),
            critic_sequence_mode=spec.critic_sequence_mode,
            token_split_horizon_targets=spec.token_split_horizon_targets,
            token_split_boundary=spec.token_split_boundary,
            mc_return_weight=spec.mc_return_weight,
            mc_lower_bound_target=spec.mc_lower_bound_target,
            mc_return_stop_gradient_encoder=(
                spec.mc_return_stop_gradient_encoder
            ),
            mc_return_value_only=spec.mc_return_value_only,
            policy_value_beta=spec.policy_value_beta,
            strict_demo_rl_only=spec.strict_demo_rl_only,
                autoregressive_action_dims=(
                    spec.autoregressive_action_dims
                ),
                pessimistic_twin_critic=spec.pessimistic_twin_critic,
                auxiliary_td_loss_weight=(
                    spec.auxiliary_td_loss_weight
                ),
                episodic_twin_head_exploration=(
                    spec.episodic_twin_head_exploration
                ),
                twin_rollout_beam_width=spec.twin_rollout_beam_width,
                dense_return_q_target=spec.dense_return_q_target,
            dense_return_positive_only=(
                spec.dense_return_positive_only
            ),
            dense_return_expected_q_loss=(
                spec.dense_return_expected_q_loss
            ),
            dense_return_advantage_alpha=(
                spec.dense_return_advantage_alpha
            ),
            dense_return_advantage_clip_ratio=(
                spec.dense_return_advantage_clip_ratio
            ),
            q_reward_scale=spec.q_reward_scale,
            dense_return_label_smoothing=spec.dense_return_label_smoothing,
            dense_return_floor_satisfaction_margin=(
                spec.dense_return_floor_satisfaction_margin
            ),
            dense_return_relative_floor_margin=(
                spec.dense_return_relative_floor_margin
            ),
            return_gated_margin=spec.return_gated_margin,
            return_gated_margin_weight=spec.return_gated_margin_weight,
            dense_return_finest_neighbor_weight=(
                spec.dense_return_finest_neighbor_weight
            ),
            episodic_success_q_target=(
                spec.episodic_success_q_target
            ),
            ordered_success_return_mix=(
                spec.ordered_success_return_mix
            ),
            sequence_aligned_mc_discount=(
                spec.sequence_aligned_mc_discount
            ),
            unseen_return_floor_weight=(
                spec.unseen_return_floor_weight
            ),
            unseen_return_floor_value=spec.unseen_return_floor_value,
            unseen_return_floor_reduction=(
                spec.unseen_return_floor_reduction
            ),
            unseen_return_floor_topk=spec.unseen_return_floor_topk,
            cv_rct_weight=spec.cv_rct_weight,
            cv_rct_level=spec.cv_rct_level,
            cv_rct_baseline=spec.cv_rct_baseline,
            awr_beta=spec.awr_beta,
            awr_weight_max=spec.awr_weight_max,
            awr_expectile_tau=spec.awr_expectile_tau,
            progress_potential_weight=spec.progress_potential_weight,
            progress_potential_schedule=spec.progress_potential_schedule,
            progress_head_weight=spec.progress_head_weight,
            progress_expectile_tau=spec.progress_expectile_tau,
            progress_success_gated=spec.progress_success_gated,
            flow_policy=spec.flow_policy,
            flow_policy_candidates=spec.flow_policy_candidates,
            flow_policy_steps=spec.flow_policy_steps,
            flow_policy_lambda=spec.flow_policy_lambda,
            flow_policy_ema=spec.flow_policy_ema,
            flow_policy_hidden_dims=spec.flow_policy_hidden_dims,
            flow_policy_gru_layers=spec.flow_policy_gru_layers,
            coarse_flow=spec.coarse_flow,
            coarse_flow_pure=spec.coarse_flow_pure,
            coarse_flow_selfdistill_weight=(
                spec.coarse_flow_selfdistill_weight
            ),
            coarse_flow_selfdistill_threshold=(
                spec.coarse_flow_selfdistill_threshold
            ),
            use_frozen_support_mask=spec.use_frozen_support_mask,
        support_mask_decode=spec.support_mask_decode,
            support_mask_tau=spec.support_mask_tau,
            support_mask_freeze_step=spec.support_mask_freeze_step,
            model=spec.model,
            jit=jit,
            platform=platform,
            seed=int(cfg.seed),
            **direct_q_kwargs,
            **common_kwargs,
        )

    if method_name == "cqn_flow":
        from robobase.method.cqn_flow import CQNFlowAS

        spec = cqn_flow_spec_from_cfg(cfg)
        if int(cfg.execution_length) != 1:
            raise ValueError("CQN-Flow requires execution_length=1.")
        if int(cfg.get("action_execution_start", 0)) != 0:
            raise ValueError("CQN-Flow requires action_execution_start=0.")
        if bool(cfg.get("temporal_ensemble", False)):
            raise ValueError(
                "CQN-Flow rollout control is implemented inside the agent so "
                "exploration noise is applied after action selection; set the "
                "root temporal_ensemble=false."
            )
        return CQNFlowAS(
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
            structured_exploration_prob=spec.structured_exploration_prob,
            structured_exploration_level=spec.structured_exploration_level,
            structured_exploration_horizon=(
                spec.structured_exploration_horizon
            ),
            separate_bc_policy=spec.separate_bc_policy,
            bc_policy_stop_gradient=spec.bc_policy_stop_gradient,
            distinct_policy_encoder=spec.distinct_policy_encoder,
            td_target_action_source=spec.td_target_action_source,
            td_target_policy_value_beta=(
                spec.td_target_policy_value_beta
            ),
            critic_sequence_mode=spec.critic_sequence_mode,
            mc_return_weight=spec.mc_return_weight,
            mc_return_stop_gradient_encoder=(
                spec.mc_return_stop_gradient_encoder
            ),
            mc_return_value_only=spec.mc_return_value_only,
            value_mode=spec.value_mode,
            num_flow_steps=spec.num_flow_steps,
            num_flow_samples=spec.num_flow_samples,
            num_target_flow_samples=spec.num_target_flow_samples,
            num_action_flow_samples=spec.num_action_flow_samples,
            flow_source_type=spec.flow_source_type,
            flow_source_std=spec.flow_source_std,
            flow_source_min=spec.flow_source_min,
            flow_source_max=spec.flow_source_max,
            antithetic_flow_sources=spec.antithetic_flow_sources,
            fixed_action_flow_sources=spec.fixed_action_flow_sources,
            action_flow_quantile_grid=spec.action_flow_quantile_grid,
            flow_iqn_quantile_coupling=(
                spec.flow_iqn_quantile_coupling
            ),
            quantile_endpoint_lambda=spec.quantile_endpoint_lambda,
            quantile_huber_kappa=spec.quantile_huber_kappa,
            return_sample_aggregation=spec.return_sample_aggregation,
            return_sample_temperature=spec.return_sample_temperature,
            return_sample_truncate_top=(
                spec.return_sample_truncate_top
            ),
            flow_q_action_readout=spec.flow_q_action_readout,
            atom_ce_lambda=spec.atom_ce_lambda,
            bcfm_lambda=spec.bcfm_lambda,
            dcfm_lambda=spec.dcfm_lambda,
            evor_td_lambda=spec.evor_td_lambda,
            confidence_weight_temp=spec.confidence_weight_temp,
            pcbf_loss_coeff=spec.pcbf_loss_coeff,
            pcbf_lambda=spec.pcbf_lambda,
            endpoint_q_lambda=spec.endpoint_q_lambda,
            source_consistency_lambda=spec.source_consistency_lambda,
            flow_distill_lambda=spec.flow_distill_lambda,
            flow_distill_action_readout=(
                spec.flow_distill_action_readout
            ),
            demo_flow_steps=spec.demo_flow_steps,
            demo_fosd=spec.demo_fosd,
            query_hidden_dim=spec.query_hidden_dim,
            time_embedding_type=spec.time_embedding_type,
            time_embed_dim=spec.time_embed_dim,
            time_scale=spec.time_scale,
            clip_scalar_targets=spec.clip_scalar_targets,
            clip_flow_trajectory=spec.clip_flow_trajectory,
            scalar_value_embedding=spec.scalar_value_embedding,
            scalar_embed_bins=spec.scalar_embed_bins,
            scalar_embed_sigma=spec.scalar_embed_sigma,
            critic_architecture=spec.critic_architecture,
            advantage_c51_lambda=spec.advantage_c51_lambda,
            advantage_q_lambda=spec.advantage_q_lambda,
            causal_branch_cache=spec.causal_branch_cache,
            causal_branch_weight=spec.causal_branch_weight,
            causal_branch_delta_weight=spec.causal_branch_delta_weight,
            causal_branch_temperature=spec.causal_branch_temperature,
            causal_branch_batch_size=spec.causal_branch_batch_size,
            causal_branch_level=spec.causal_branch_level,
            policy_value_beta=spec.policy_value_beta,
            freeze_bc_policy=spec.freeze_bc_policy,
            bc_policy_mode=spec.bc_policy_mode,
            demo_batch_size=(
                None
                if cfg.get("demo_batch_size", None) is None
                else int(cfg.demo_batch_size)
            ),
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
            jit=bool(
                cfg.get("backend", {}).get("jit", True) if cfg.get("backend") else True
            ),
            platform=cfg.get("backend", {}).get("platform", None)
            if cfg.get("backend")
            else None,
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
            jit=bool(
                cfg.get("backend", {}).get("jit", True) if cfg.get("backend") else True
            ),
            platform=cfg.get("backend", {}).get("platform", None)
            if cfg.get("backend")
            else None,
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
            image_augmentation_type=spec.image_augmentation_type,
            mask_padded_model_input=spec.mask_padded_model_input,
            model=spec.model,
            jit=bool(
                cfg.get("backend", {}).get("jit", True) if cfg.get("backend") else True
            ),
            platform=cfg.get("backend", {}).get("platform", None)
            if cfg.get("backend")
            else None,
            seed=int(cfg.seed),
            **common_kwargs,
        )

    if method_name in {"flow_matching", "a2a", "legato"}:
        spec = flow_matching_spec_from_cfg(cfg)
        stateful_flow_source = spec.flow_source.type in {
            "a2a",
            "a2a_noise",
            "legato",
        }
        if bool(cfg.get("temporal_ensemble", False)) and stateful_flow_source:
            raise ValueError(
                f"{spec.flow_source.type} rollout state requires "
                "temporal_ensemble=false "
                "until executed actions are fed back to the agent."
            )
        if spec.flow_source.type in {"a2a", "a2a_noise", "legato"}:
            execution_start = int(cfg.get("action_execution_start", 0))
            replay_start = int(
                cfg.get("replay", {}).get("action_sequence_start_offset", 0)
            )
            if execution_start != replay_start:
                policy_name = (
                    "A2A" if spec.flow_source.type in {"a2a", "a2a_noise"} else "Legato"
                )
                raise ValueError(
                    f"{policy_name} requires replay.action_sequence_start_offset "
                    "to equal action_execution_start so train and rollout "
                    "continuation align."
                )
        if spec.flow_source.type == "legato":
            validate_legato_overlap(
                spec.flow_source,
                action_sequence=int(cfg.action_sequence),
                execution_length=int(cfg.execution_length),
                action_execution_start=int(cfg.get("action_execution_start", 0)),
            )
        if method_name == "a2a":
            from robobase.method.a2a import A2A as flow_policy_cls
        elif method_name == "legato":
            from robobase.method.legato import Legato as flow_policy_cls
        else:
            from robobase.method.flow_matching import FlowMatching as flow_policy_cls

        return flow_policy_cls(
            lr=spec.lr,
            adaptive_lr=spec.adaptive_lr,
            num_train_steps=spec.num_train_steps,
            actor_grad_clip=spec.actor_grad_clip,
            objective_type=spec.objective_type,
            lr_schedule=spec.lr_schedule,
            num_flow_steps=spec.num_flow_steps,
            sampler=spec.sampler,
            sample_schedule=spec.sample_schedule,
            train_time_schedule=spec.train_time_schedule,
            time_scale=spec.time_scale,
            image_augmentation_type=spec.image_augmentation_type,
            horizon_dropout_lengths=spec.horizon_dropout_lengths,
            horizon_dropout_probs=spec.horizon_dropout_probs,
            horizon_loss_weights=spec.horizon_loss_weights,
            use_ema=spec.use_ema,
            ema_decay=spec.ema_decay,
            ema_decay_schedule=spec.ema_decay_schedule,
            weight_decay=spec.weight_decay,
            flow_source=spec.flow_source,
            execution_length=int(cfg.execution_length),
            action_execution_start=int(cfg.get("action_execution_start", 0)),
            model=spec.model,
            jit=bool(
                cfg.get("backend", {}).get("jit", True) if cfg.get("backend") else True
            ),
            platform=cfg.get("backend", {}).get("platform", None)
            if cfg.get("backend")
            else None,
            seed=int(cfg.seed),
            **common_kwargs,
        )

    raise NotImplementedError(
        f"Unsupported method '{method_name}'. Supported: a2a, act, bc, cqn, "
        "cqn_as, cqn_flow, diffusion, drqv2, flow_matching, legato, ppo, "
        "q_chunking, djcqn, sac."
    )
