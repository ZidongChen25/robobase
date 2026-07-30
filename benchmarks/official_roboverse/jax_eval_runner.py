"""RoboVerse simulator boundary for the strict JAX A2A policy.

RoboVerse/Isaac Sim represents observations and actions as Torch tensors.  This
adapter is the only Torch-facing layer; model loading, normalization and policy
inference remain in :mod:`benchmarks.official_roboverse.jax_a2a`.
"""

from __future__ import annotations

from collections import deque

from benchmarks.official_roboverse.jax_a2a import JaxA2APredictor


def make_jax_eval_runner(base_eval_runner, diffusion_policy_cfg, torch_module):
    class JaxA2AEvalRunner(base_eval_runner):
        def _init_policy(self, default_runner, **kwargs):
            del default_runner
            self.predictor = JaxA2APredictor(kwargs["checkpoint_path"])
            import jax.dlpack

            warm_images = torch_module.zeros(
                (self.num_envs, 8, 3, 256, 256),
                dtype=torch_module.float32,
                device=self.device,
            )
            warm_states = torch_module.zeros(
                (self.num_envs, 8, 9),
                dtype=torch_module.float32,
                device=self.device,
            )
            warm_actions = self.predictor.predict_device(
                jax.dlpack.from_dlpack(warm_images),
                jax.dlpack.from_dlpack(warm_states),
            )
            torch_module.from_dlpack(warm_actions).sum().item()
            self.policy_cfg = diffusion_policy_cfg()
            self.policy_cfg.obs_config.obs_type = "joint_pos"
            self.policy_cfg.obs_config.obs_dim = 9
            self.policy_cfg.obs_config.norm_image = True
            self.policy_cfg.action_config.action_type = "joint_pos"
            self.policy_cfg.action_config.action_dim = 9
            self.policy_cfg.action_config.action_chunk_steps = 8
            self.policy_cfg.action_config.delta = False
            self.policy_cfg.action_config.temporal_agg = False
            self.obs = deque(maxlen=9)
            self.env = None

        @staticmethod
        def _stack_last_n_obs(all_obs, n_steps=8):
            all_obs = list(all_obs)
            latest = all_obs[-1]
            result = []
            missing = max(0, n_steps - len(all_obs))
            result.extend([all_obs[0]] * missing)
            result.extend(all_obs[-n_steps:])
            return torch_module.stack(result, dim=1).to(latest.device)

        def reset(self):
            self.obs.clear()
            super().reset()

        def update_obs(self, current_obs):
            self.obs.append(current_obs)

        def predict_action(self, observaton=None):
            if observaton is not None:
                self.obs.append(observaton)
            images = self._stack_last_n_obs(
                [item["head_cam"] for item in self.obs]
            )
            states = self._stack_last_n_obs(
                [item["agent_pos"] for item in self.obs]
            )
            import jax.dlpack

            actions = self.predictor.predict_device(
                jax.dlpack.from_dlpack(images.contiguous()),
                jax.dlpack.from_dlpack(states.contiguous()),
            )
            return torch_module.from_dlpack(actions).to(torch_module.float32).transpose(0, 1)

    return JaxA2AEvalRunner


def patch_default_eval_runner() -> None:
    import hydra
    import torch
    from roboverse_learn.il.configs.base_config import DiffusionPolicyCfg
    from roboverse_learn.il.runners.base_runner import BaseRunner
    from roboverse_learn.il.runners.base_eval_runner import BaseEvalRunner
    from roboverse_learn.il.runners import default_eval_runner, default_runner

    def _jax_only_runner_init(self, cfg, output_dir=None):
        # The upstream evaluator normally constructs an unused Torch policy and
        # optimizer before DefaultEvalRunner loads a checkpoint. Skip that work;
        # Isaac Sim remains Torch-based, but the policy is exclusively JAX.
        BaseRunner.__init__(self, cfg, output_dir=output_dir)
        self.model = None
        self.ema_model = None
        self.optimizer = None
        self.policy_name = "a2a_jax"
        self.global_step = 0
        self.epoch = 0
        self.eval_args = hydra.utils.instantiate(cfg.eval_config.eval_args)

    default_eval_runner.DefaultEvalRunner = make_jax_eval_runner(
        BaseEvalRunner, DiffusionPolicyCfg, torch
    )
    default_runner.DefaultRunner.__init__ = _jax_only_runner_init


__all__ = ["make_jax_eval_runner", "patch_default_eval_runner"]
