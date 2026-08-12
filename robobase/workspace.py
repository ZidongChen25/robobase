import signal
import time
import random
import gc
import json
import math
from collections import deque
from typing import Callable, Any
from functools import partial
import logging
import os

from gymnasium import spaces
from omegaconf import DictConfig

from robobase import utils
from robobase.factory import create_agent, method_name_from_cfg
from robobase.envs.env import EnvFactory
from robobase.logger import Logger
from robobase.replay_buffer.iterator import (
    PrefetchReplayBatchIterator,
    create_epoch_replay_iterator,
    create_jax_replay_iterator,
    normalize_replay_num_workers,
)
from robobase.replay_buffer.prioritized_replay_buffer import PrioritizedReplayBuffer
from robobase.replay_buffer.replay_buffer import ReplayBuffer
from robobase.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from robobase.replay_buffer.shared_demo_cache import (
    SharedDemoReplayCache,
    demo_cache_key,
)
from robobase.replay_buffer.bigym_lazy_replay import (
    LazyBiGymReplayBuffer,
    lazy_replay_enabled,
)
from robobase.replay_buffer.vision_feature_cache import (
    build_vision_feature_cache_plan,
)


from pathlib import Path

import pickle

import hydra
import numpy as np
import gymnasium as gym


def _set_seed_everywhere(seed: int):
    np.random.seed(seed)
    random.seed(seed)


class _Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        return step < self._until // self._action_repeat


class _Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None or self._every == 0:
            return False
        every = self._every // self._action_repeat
        return step % every == 0


class _Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed = time.time() - self._last_time
        self._last_time = time.time()
        total = time.time() - self._start_time
        return elapsed, total

    def total_time(self):
        return time.time() - self._start_time


class _DemoMergedIterator:
    def __init__(self, replay_iter, demo_replay_iter):
        self.replay_iter = replay_iter
        self.demo_replay_iter = demo_replay_iter
        self._is_safe = False

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self.replay_iter)
        demo_batch = next(self.demo_replay_iter)
        if not self._is_safe:
            assert set(batch.keys()) == set(demo_batch.keys())
            self._is_safe = True
        demo_batch["demo"] = np.ones_like(demo_batch["demo"])
        return {
            k: np.concatenate([batch[k], demo_batch[k]], axis=0) for k in batch.keys()
        }

    def close(self):
        for iterator in (self.replay_iter, self.demo_replay_iter):
            close = getattr(iterator, "close", None)
            if callable(close):
                close()


class _NonFiniteForensicTap:
    """Pass-through replay iterator that retains the last few batch slices.

    A non-finite update poisons every parameter in one step, so by the time the
    guard fires the triggering inputs are gone -- which is why three NaN events
    (offqc8_s2 @8k, stage-173 @31k, QC+truncation @1k/4k) were all closed as
    "rare numerical event" without a root cause. This tap keeps the inputs
    alive so the guard can dump them.

    By default only non-uint8 keys are retained. Diagnostic runs can retain
    uint8 images too, which is necessary for exact forward/update replay.
    Nothing is copied to host and no statistic is computed per step, so the
    numerical path is unchanged.
    """

    def __init__(self, iterator, keep: int = 3, include_uint8: bool = False):
        self._iterator = iterator
        self.recent = deque(maxlen=max(1, int(keep)))
        self.include_uint8 = bool(include_uint8)

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self._iterator)
        try:
            if isinstance(batch, dict):
                self.recent.append(
                    {
                        k: v
                        for k, v in batch.items()
                        if getattr(v, "dtype", None) is not None
                        and (self.include_uint8 or v.dtype != np.uint8)
                    }
                )
        except Exception:  # never let forensics break training
            pass
        return batch

    def close(self):
        close = getattr(self._iterator, "close", None)
        if callable(close):
            close()


class _DemoOnlyIterator:
    """Route updates exclusively from the dedicated demonstration replay."""

    def __init__(self, replay_iter, demo_replay_iter):
        self.replay_iter = replay_iter
        self.demo_replay_iter = demo_replay_iter

    def __iter__(self):
        return self

    def __next__(self):
        demo_batch = next(self.demo_replay_iter)
        demo_batch["demo"] = np.ones_like(demo_batch["demo"])
        return demo_batch

    def close(self):
        for iterator in (self.replay_iter, self.demo_replay_iter):
            close = getattr(iterator, "close", None)
            if callable(close):
                close()


def _merge_replay_demo_iter(replay_iter, demo_replay_iter):
    return iter(_DemoMergedIterator(replay_iter, demo_replay_iter))


def _normalize_demos_count(value):
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"inf", "+inf", ".inf", "+.inf"}:
            return np.inf
        return float(normalized)
    return value


def _executed_action_steps(info: dict[str, Any]) -> int:
    mask = info.get("action_sequence_mask") if isinstance(info, dict) else None
    if mask is None:
        return 1
    return max(1, int(np.asarray(mask, dtype=np.int32).sum()))


def _effective_episode_length(cfg: DictConfig) -> int:
    """Return the number of wrapped environment decisions in one episode."""

    episode_length = int(cfg.env.episode_length)
    if (
        str(cfg.env.get("env_name", "")).lower() == "bigym"
        and not bool(cfg.env.get("episode_length_is_env_steps", False))
    ):
        episode_length //= int(cfg.env.demo_down_sample_rate)
    return max(1, episode_length)


def _replay_action_from_step(
    commanded_action: np.ndarray,
    next_info: dict[str, Any],
) -> np.ndarray:
    """Select the single action that the environment actually executed."""

    replay_action = np.asarray(commanded_action)[0]
    transition_info = next_info if isinstance(next_info, dict) else {}
    final_info = transition_info.get("final_info")
    if bool(transition_info.get("_final_info", False)) and isinstance(
        final_info, dict
    ):
        transition_info = final_info
    executed = transition_info.get("executed_action")
    if executed is None:
        return replay_action

    executed = np.asarray(executed)
    if executed.ndim == replay_action.ndim + 1:
        if executed.shape[0] != 1:
            raise ValueError(
                "RL replay requires execution_length=1, but the environment "
                "reported multiple executed actions."
            )
        executed = executed[0]
    if executed.shape != replay_action.shape:
        raise ValueError(
            "Executed action shape does not match replay action shape: "
            f"{executed.shape} != {replay_action.shape}."
        )
    return executed


def _validate_eval_env_counts(cfg: DictConfig) -> None:
    num_eval_envs = int(cfg.num_eval_envs)
    num_eval_episodes = int(cfg.num_eval_episodes)
    if num_eval_episodes < 0:
        raise ValueError(f"num_eval_episodes must be >= 0, got {num_eval_episodes}.")
    if num_eval_envs < 0:
        raise ValueError(f"num_eval_envs must be >= 0, got {num_eval_envs}.")
    if num_eval_episodes > 0 and num_eval_envs < 1:
        raise ValueError(
            "num_eval_envs must be >= 1 when num_eval_episodes > 0, got "
            f"num_eval_envs={num_eval_envs} and num_eval_episodes={num_eval_episodes}."
        )


def _validate_rl_action_sequence(cfg: DictConfig) -> None:
    """Restrict multi-step RL chunks to methods that implement chunk rollout."""

    method_name = method_name_from_cfg(cfg)
    supported_methods = {"cqn_as", "cqn_flow", "q_chunking"}
    if (
        cfg.method.is_rl
        and cfg.action_sequence != 1
        and method_name not in supported_methods
    ):
        raise ValueError(
            "Action sequence > 1 is only supported for the CQN-AS, "
            "CQN-Flow, and Q-Chunking RL methods"
        )


def _online_updates_ready(
    cfg: DictConfig,
    *,
    main_loop_iterations: int,
    replay_size: int,
) -> bool:
    """Return whether replay and online-delay gates both permit updates."""

    return bool(
        replay_size >= int(cfg.replay_size_before_train)
        and main_loop_iterations >= int(cfg.get("online_update_after_steps", 0))
    )


def _mc_return_anchor_enabled(cfg: DictConfig) -> bool:
    method_name = method_name_from_cfg(cfg)
    return bool(
        method_name in {"cqn_as", "cqn_flow"}
        and (
            float(cfg.method.get("mc_return_weight", 0.0)) > 0.0
            or bool(cfg.method.get("mc_lower_bound_target", False))
            or bool(cfg.method.get("episodic_success_q_target", False))
            or (
                method_name == "cqn_flow"
                and float(cfg.method.get("evor_td_lambda", 0.0)) > 0.0
            )
        )
    )


def _structured_exploration_enabled(cfg: DictConfig) -> bool:
    return (
        method_name_from_cfg(cfg) in {"cqn_as", "cqn_flow"}
        and float(cfg.method.get("structured_exploration_prob", 0.0)) > 0.0
    )


def _create_default_replay_buffer(
    cfg: DictConfig,
    observation_space: gym.Space,
    action_space: gym.Space,
    demo_replay: bool = False,
    save_dir: str | None = None,
    purge_replay_on_shutdown: bool = True,
    save_snapshot: bool = False,
    reuse_saved: bool = False,
) -> ReplayBuffer:
    use_mc_return_anchor = _mc_return_anchor_enabled(cfg)
    use_structured_exploration = _structured_exploration_enabled(cfg)
    if lazy_replay_enabled(cfg):
        if use_mc_return_anchor:
            raise NotImplementedError(
                "CQN-AS MC-return targets require episode-backed replay; "
                "set lazy_replay.use=false."
            )
        if demo_replay:
            raise NotImplementedError("lazy_replay does not support demo replay.")
        if cfg.env.env_name != "bigym":
            raise NotImplementedError("lazy_replay is currently implemented for BiGym.")
        extra_replay_elements = spaces.Dict({})
        if cfg.demos != 0:
            extra_replay_elements["demo"] = spaces.Box(0, 1, shape=(), dtype=np.uint8)
        if cfg.replay.get("nstep_explore_truncate", False):
            # Per-step flag: executed action came from a bin-explore-shifted
            # registered plan. Consumed by explore-aware n-step truncation.
            extra_replay_elements["explored"] = spaces.Box(
                0, 1, shape=(), dtype=np.uint8
            )
        if use_structured_exploration:
            extra_replay_elements["structured_explore"] = spaces.Box(
                0,
                1,
                shape=(),
                dtype=np.uint8,
            )
            extra_replay_elements["structured_explore_start"] = spaces.Box(
                0, 1, shape=(), dtype=np.uint8
            )
            extra_replay_elements["structured_explore_dimension"] = spaces.Box(
                -1,
                int(action_space.shape[-1]) - 1,
                shape=(),
                dtype=np.int16,
            )
            extra_replay_elements["structured_explore_delta"] = spaces.Box(
                -np.inf, np.inf, shape=(), dtype=np.float32
            )
            extra_replay_elements[
                "structured_explore_assignment_prob"
            ] = spaces.Box(0.0, 1.0, shape=(), dtype=np.float32)
        return LazyBiGymReplayBuffer(
            cfg,
            observation_space,
            action_space,
            batch_size=cfg.batch_size,
            extra_replay_elements=extra_replay_elements,
        )

    multiprocessing_context = "spawn"
    method_name = method_name_from_cfg(cfg)
    source_cfg = cfg.method.get("flow_source", {})
    default_source_type = (
        method_name if method_name in {"a2a", "legato"} else "gaussian"
    )
    flow_source_type = str(source_cfg.get("type", default_source_type)).lower()
    action_history_len = (
        int(source_cfg.get("history_horizon", cfg.action_sequence))
        if flow_source_type in {"a2a", "a2a_noise"}
        else 0
    )
    action_history_padding = str(source_cfg.get("history_padding", "zero")).lower()
    action_history_source = str(
        source_cfg.get("history_source", "commanded_action")
    ).lower()
    cache_plan = build_vision_feature_cache_plan(
        cfg=cfg,
        observation_space=observation_space,
        save_dir=save_dir if save_dir is not None else cfg.replay.save_dir,
        reuse_saved=bool(reuse_saved),
    )
    replay_observation_space = (
        cache_plan.observation_space if cache_plan is not None else observation_space
    )
    if cache_plan is not None:
        logging.info(
            "Replay image-feature cache enabled with keys=%s (feature_dim=%d).",
            list(cache_plan.feature_keys),
            cache_plan.feature_dim,
        )
    extra_replay_elements = spaces.Dict({})
    if cfg.demos != 0:
        extra_replay_elements["demo"] = spaces.Box(0, 1, shape=(), dtype=np.uint8)
    if cfg.replay.get("nstep_explore_truncate", False):
        # Per-step flag: executed action came from a bin-explore-shifted
        # registered plan. Consumed by explore-aware n-step truncation.
        extra_replay_elements["explored"] = spaces.Box(
            0, 1, shape=(), dtype=np.uint8
        )
    if use_mc_return_anchor:
        extra_replay_elements["mc_return"] = spaces.Box(
            -np.inf,
            np.inf,
            shape=(),
            dtype=np.float32,
        )
    if use_structured_exploration:
        extra_replay_elements["structured_explore"] = spaces.Box(
            0,
            1,
            shape=(),
            dtype=np.uint8,
        )
        extra_replay_elements["structured_explore_start"] = spaces.Box(
            0, 1, shape=(), dtype=np.uint8
        )
        extra_replay_elements["structured_explore_dimension"] = spaces.Box(
            -1,
            int(action_space.shape[-1]) - 1,
            shape=(),
            dtype=np.int16,
        )
        extra_replay_elements["structured_explore_delta"] = spaces.Box(
            -np.inf, np.inf, shape=(), dtype=np.float32
        )
        extra_replay_elements[
            "structured_explore_assignment_prob"
        ] = spaces.Box(0.0, 1.0, shape=(), dtype=np.float32)
    # Create replay_class with buffer-specific hyperparameters
    replay_class = UniformReplayBuffer
    if cfg.replay.prioritization:
        replay_class = PrioritizedReplayBuffer
    replay_class = partial(
        replay_class,
        nstep=cfg.replay.nstep,
        gamma=cfg.replay.gamma,
    )
    # Create replay_class with common hyperparameters
    return replay_class(
        save_dir=save_dir if save_dir is not None else cfg.replay.save_dir,
        purge_replay_on_shutdown=purge_replay_on_shutdown,
        save_snapshot=save_snapshot,
        episode_compression=str(cfg.replay.get("compression", "none")),
        reuse_saved=reuse_saved,
        batch_size=cfg.batch_size if not demo_replay else cfg.demo_batch_size,
        replay_capacity=cfg.replay.size if not demo_replay else cfg.replay.demo_size,
        action_shape=action_space.shape,
        action_dtype=action_space.dtype,
        reward_shape=(),
        reward_dtype=np.float32,
        observation_elements=replay_observation_space,
        extra_replay_elements=extra_replay_elements,
        preprocessing_fn=None if cache_plan is None else cache_plan.preprocessing_fn,
        num_workers=cfg.replay.num_workers,
        sequential=cfg.replay.sequential,
        transition_seq_len=cfg.replay.transition_seq_len,
        action_sequence_start_offset=int(
            cfg.replay.get("action_sequence_start_offset", 0)
        ),
        action_padding=str(cfg.replay.get("action_padding", "zero")),
        action_history_len=action_history_len,
        action_history_padding=action_history_padding,
        action_history_source=action_history_source,
        max_cached_episodes=cfg.replay.get("max_cached_episodes", None),
        max_cached_episode_bytes=cfg.replay.get("max_cached_episode_bytes", None),
        include_tp1=bool(cfg.replay.get("include_tp1", True)),
        include_next_action=bool(
            cfg.replay.get("include_next_action", False)
        ),
        auxiliary_nstep=cfg.replay.get("auxiliary_nstep", None),
        nstep_explore_truncate=cfg.replay.get("nstep_explore_truncate", False),
        multiprocessing_context=multiprocessing_context,
        transition_uniform_sampling=bool(
            cfg.replay.get("transition_uniform_sampling", False)
        ),
    )


def _create_default_envs(cfg: DictConfig) -> EnvFactory:
    factory = None
    if cfg.env.env_name == "rlbench":
        from robobase.envs.rlbench import RLBenchEnvFactory

        factory = RLBenchEnvFactory()
    elif cfg.env.env_name == "dmc":
        from robobase.envs.dmc import DMCEnvFactory

        factory = DMCEnvFactory()
    elif cfg.env.env_name == "bigym":
        from robobase.envs.bigym import BiGymEnvFactory

        factory = BiGymEnvFactory()
    elif cfg.env.env_name == "d4rl":
        from robobase.envs.d4rl import D4RLEnvFactory

        factory = D4RLEnvFactory()
    elif cfg.env.env_name == "robomimic":
        from robobase.envs.robomimic import RobomimicEnvFactory

        factory = RobomimicEnvFactory()
    elif cfg.env.env_name == "pusht":
        from robobase.envs.pusht import PushTEnvFactory

        factory = PushTEnvFactory()
    else:
        ValueError()
    return factory


class Workspace:
    def __init__(
        self,
        cfg: DictConfig,
        env_factory: EnvFactory = None,
        create_replay_fn: Callable[[DictConfig], ReplayBuffer] = None,
        work_dir: str = None,
    ):
        if env_factory is None:
            env_factory = _create_default_envs(cfg)
        if create_replay_fn is None:
            create_replay_fn = _create_default_replay_buffer

        self.work_dir = Path(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            if work_dir is None
            else work_dir
        )
        print(f"workspace: {self.work_dir}")

        # Sanity checks
        _validate_eval_env_counts(cfg)
        if cfg.execution_length > cfg.action_sequence:
            raise ValueError(
                "execution_length must be <= action_sequence, got "
                f"{cfg.execution_length} and {cfg.action_sequence}."
            )
        action_execution_start = int(cfg.get("action_execution_start", 0))
        if action_execution_start < 0:
            raise ValueError(
                f"action_execution_start must be >= 0, got {action_execution_start}."
            )
        if action_execution_start + cfg.execution_length > cfg.action_sequence:
            raise ValueError(
                "action_execution_start + execution_length must be <= "
                f"action_sequence, got {action_execution_start} + "
                f"{cfg.execution_length} > {cfg.action_sequence}."
            )
        obs_delay = int(cfg.get("obs_delay", 0) or 0)
        if obs_delay < 0:
            raise ValueError(f"obs_delay must be >= 0, got {obs_delay}.")
        if obs_delay > 0:
            logging.info(
                "Delayed-policy conditioning enabled: acting on o_{t-%d} "
                "(obs_delay=%d environment steps).",
                obs_delay,
                obs_delay,
            )
        noise_mask_steps = int(cfg.method.get("noise_mask_steps", 0))
        if noise_mask_steps < 0 or noise_mask_steps >= cfg.action_sequence:
            raise ValueError(
                "method.noise_mask_steps must satisfy "
                f"0 <= noise_mask_steps < action_sequence, got {noise_mask_steps} "
                f"and {cfg.action_sequence}."
            )
        available_action_steps = cfg.action_sequence - noise_mask_steps
        if cfg.execution_length > available_action_steps:
            raise ValueError(
                "execution_length must be <= the number of available actions "
                f"({available_action_steps} = action_sequence - noise_mask_steps), "
                f"got execution_length={cfg.execution_length}, "
                f"action_sequence={cfg.action_sequence}, "
                f"noise_mask_steps={noise_mask_steps}."
            )
        effective_episode_length = _effective_episode_length(cfg)
        if (
            not cfg.is_imitation_learning
            and str(cfg.method.get("name", "")).lower() != "ppo"
            and cfg.replay_size_before_train * cfg.action_repeat * cfg.execution_length
            < effective_episode_length
            and cfg.replay_size_before_train > 0
        ):
            raise ValueError(
                "replay_size_before_train * action_repeat * execution_length "
                f"({cfg.replay_size_before_train} * {cfg.action_repeat} * "
                f"{cfg.execution_length}) "
                f"must be >= effective episode length "
                f"({effective_episode_length})."
            )

        _validate_rl_action_sequence(cfg)
        if cfg.method.is_rl and cfg.execution_length != 1:
            raise ValueError("execution_length > 1 is not supported for RL methods")
        if not cfg.method.is_rl and cfg.replay.nstep != 1:
            raise ValueError("replay.nstep != 1 is not supported for IL methods")

        self.cfg = cfg
        _set_seed_everywhere(cfg.seed)
        self.use_demo_replay = cfg.demo_batch_size is not None
        self.demo_only_updates = bool(
            cfg.replay.get("demo_only_updates", False)
        )
        if self.demo_only_updates and not self.use_demo_replay:
            raise ValueError(
                "replay.demo_only_updates=true requires demo_batch_size."
            )
        self._skip_demo_loading_from_dataset = False

        # create logger
        self.logger = Logger(self.work_dir, cfg=self.cfg)
        self.env_factory = env_factory

        if (num_demos := cfg.demos) != 0:
            # Collect demos or fetch saved demos before making environments
            # to consider demo-based action space (e.g., standardization)
            if self._should_use_lazy_replay():
                prepare_lazy_replay = getattr(
                    self.env_factory,
                    "prepare_lazy_replay",
                    None,
                )
                if not callable(prepare_lazy_replay):
                    raise NotImplementedError(
                        "lazy_replay requires env_factory.prepare_lazy_replay()."
                    )
                prepare_lazy_replay(cfg, num_demos)
                self._skip_demo_loading_from_dataset = True
            elif self._should_collect_demos_from_dataset():
                self.env_factory.collect_or_fetch_demos(cfg, num_demos)
            else:
                self._skip_demo_loading_from_dataset = True

        self.train_envs = None
        self.eval_env = None
        self.eval_envs = None
        self._defer_live_eval_env_creation = self._should_defer_live_eval_env_creation()

        # Make training environment
        if self._should_create_train_env():
            self.train_envs = self.env_factory.make_train_env(cfg)
        elif cfg.num_train_envs > 0:
            logging.warning(
                "Train env is not created. Offline pretraining can still run, "
                "but online rollouts are disabled."
            )

        if num_demos != 0:
            # Post-process demos using the information from environments
            if not self._skip_demo_loading_from_dataset:
                self.env_factory.post_collect_or_fetch_demos(cfg)

        observation_space, action_space = self.env_factory.get_spaces(cfg)
        if self.cfg.num_eval_episodes > 0 and not self._defer_live_eval_env_creation:
            self._ensure_eval_envs_created()

        if cfg.get("intrinsic_reward_module", None):
            raise NotImplementedError("Intrinsic reward modules are not yet supported.")

        self.agent = create_agent(
            cfg,
            observation_space=observation_space,
            action_space=action_space,
        )
        self.agent.train(False)

        self.replay_buffer = create_replay_fn(
            cfg,
            observation_space,
            action_space,
            save_dir=self._resolve_replay_save_dir(demo_replay=False),
            purge_replay_on_shutdown=not self._should_persist_replay_files(),
            save_snapshot=self._should_persist_replay_files(),
            reuse_saved=bool(cfg.replay.reuse_saved),
        )
        self.prioritized_replay = cfg.replay.prioritization
        self.extra_replay_elements = self.replay_buffer.extra_replay_elements
        self.replay_num_workers = normalize_replay_num_workers(
            "jax", cfg.replay.num_workers
        )
        self._replay_iter = None

        # Create a separate demo replay that contains successful episodes.
        # This is designed for RL. IL algorithms don't have to use this!
        # TODO: Change the name to `self_imitation_buffer` or other names
        # Note that original buffer also contains demos, but they are not protected
        # TODO: Support demo protection in a buffer
        if self.use_demo_replay:
            self.demo_replay_buffer = create_replay_fn(
                cfg,
                observation_space,
                action_space,
                demo_replay=True,
                save_dir=self._resolve_replay_save_dir(demo_replay=True),
                purge_replay_on_shutdown=not self._should_persist_replay_files(),
                save_snapshot=self._should_persist_replay_files(),
                reuse_saved=bool(cfg.replay.reuse_saved),
            )
            self.demo_replay_num_workers = self.replay_num_workers

        self._shared_demo_cache = None
        demo_cache_dir = cfg.replay.get("demo_cache_dir", None)
        if self.use_demo_replay and demo_cache_dir:
            signatures = {
                "all_demos": self.replay_buffer.storage_signature(),
                "expert_demos": self.demo_replay_buffer.storage_signature(),
            }
            cache_key = demo_cache_key(cfg, signatures)
            self._shared_demo_cache = SharedDemoReplayCache(
                demo_cache_dir,
                cache_key,
            )
            logging.info(
                "Shared demo replay cache: %s",
                self._shared_demo_cache.path,
            )

        if self.prioritized_replay:
            if self.use_demo_replay:
                raise NotImplementedError(
                    "Demo replay is not compatible with prioritized replay"
                )

        # RLBench doesn't like it when we import cv2 before it, so moving
        # import here.
        from robobase.video import VideoRecorder

        self.eval_video_recorder = VideoRecorder(
            (self.work_dir / "eval_videos") if self.cfg.log_eval_video else None
        )

        self._timer = _Timer()
        self._pretrain_step = 0
        self._main_loop_iterations = 0
        self._global_env_episode = 0
        self._act_dim = action_space.shape[0]
        if self.train_envs:
            self._episode_rollouts = [[] for _ in range(self.train_envs.num_envs)]
        else:
            self._episode_rollouts = []

        self._shutting_down = False
        self._sigint_count = 0
        self._previous_sigint_handler = None
        self._snapshot_loaded = False
        self._snapshot_cfg = None
        train_agent_count = (
            self.train_envs.num_envs if self.train_envs is not None else 0
        )
        self._eval_agent_indices = list(
            range(train_agent_count, train_agent_count + self.cfg.num_eval_envs)
        )

    @property
    def pretrain_steps(self):
        return self._pretrain_step

    @property
    def main_loop_iterations(self):
        return self._main_loop_iterations

    @property
    def global_env_episodes(self):
        return self._global_env_episode

    def _calculate_global_env_steps(
        self,
        *,
        main_loop_iterations: int | None = None,
        pretrain_steps: int | None = None,
    ) -> int:
        if not self.train_envs:
            return pretrain_steps if pretrain_steps is not None else self.pretrain_steps

        if main_loop_iterations is None:
            main_loop_iterations = self._main_loop_iterations
        if pretrain_steps is None:
            pretrain_steps = self.pretrain_steps

        # TODO: Pretrain_steps should not be included in env_steps, because it's
        # training steps but not environment steps. We need another PR to address this
        return (
            main_loop_iterations
            * self.cfg.action_repeat
            * self.train_envs.num_envs
            * self.cfg.execution_length
            + pretrain_steps
        )

    @property
    def global_env_steps(self):
        """Total number of environment steps taken."""
        return self._calculate_global_env_steps()

    def _make_merged_replay_iter(self, replay_iter, demo_replay_iter):
        """Build the online+demo merged iterator.

        Overridable hook: subclasses swap the merge implementation (e.g.
        device-side concat) and it lands *inside* any prefetch wrapper the
        property adds afterwards.
        """
        if getattr(self, "demo_only_updates", False):
            return iter(_DemoOnlyIterator(replay_iter, demo_replay_iter))
        return _merge_replay_demo_iter(replay_iter, demo_replay_iter)

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            if self._should_use_epoch_style_replay():
                if self.use_demo_replay:
                    raise NotImplementedError(
                        "Epoch-style replay sampling does not support demo replay."
                    )
                if self.prioritized_replay:
                    raise NotImplementedError(
                        "Epoch-style replay sampling is not compatible with prioritized replay."
                    )
                _replay_iter = create_epoch_replay_iterator(
                    self.replay_buffer,
                    execution_length=self.cfg.execution_length,
                    shuffle=True,
                    seed=int(self.cfg.seed),
                    load_all_episodes=bool(
                        self.cfg.replay.get("epoch_load_all_episodes", False)
                    ),
                    batch_chunk_size=self.cfg.replay.get("epoch_batch_chunk_size", 0),
                )
            else:
                _replay_iter = create_jax_replay_iterator(
                    self.replay_buffer,
                    num_workers=self.replay_num_workers,
                )
            if self.use_demo_replay:
                _demo_replay_iter = create_jax_replay_iterator(
                    self.demo_replay_buffer,
                    num_workers=self.demo_replay_num_workers,
                )
                _replay_iter = self._make_merged_replay_iter(
                    _replay_iter, _demo_replay_iter
                )
            prefetch_size = int(
                self.cfg.get("backend", {}).get("replay_prefetch_size", 0)
                if self.cfg.get("backend", None)
                else 0
            )
            prefetch_workers = 1
            if self._should_use_lazy_replay():
                lazy_cfg = self.cfg.get("lazy_replay", {})
                prefetch_workers = max(1, int(lazy_cfg.get("num_workers", 1)))
                if prefetch_size <= 0:
                    prefetch_size = prefetch_workers * int(
                        lazy_cfg.get("prefetch_factor", 2)
                    )
            if prefetch_size > 0:
                backend_cfg = self.cfg.get("backend", {})
                map_fn = None
                if bool(backend_cfg.get("replay_device_prefetch", False)):
                    map_fn = getattr(self.agent, "prefetch_batch", None)
                _replay_iter = PrefetchReplayBatchIterator(
                    _replay_iter,
                    queue_size=prefetch_size,
                    worker_name="jax_replay_prefetch",
                    map_fn=map_fn,
                    num_workers=prefetch_workers,
                )
            if bool(self.cfg.get("nonfinite_dump", True)):
                _replay_iter = _NonFiniteForensicTap(
                    _replay_iter,
                    keep=int(self.cfg.get("nonfinite_dump_keep_batches", 3)),
                    include_uint8=bool(
                        self.cfg.get("nonfinite_dump_include_uint8", False)
                    ),
                )
            self._replay_iter = _replay_iter
        return self._replay_iter

    def _should_use_epoch_style_replay(self) -> bool:
        return bool(
            self.cfg.is_imitation_learning
            and self.cfg.num_train_frames <= 0
            and self.cfg.replay.get("epoch_style_sampling", False)
        )

    def _offline_batches_per_epoch(self) -> int:
        if not self._should_use_epoch_style_replay():
            raise ValueError(
                "num_pretrain_epochs/eval_every_epochs require offline imitation "
                "learning with replay.epoch_style_sampling=true."
            )
        epoch_iter = create_epoch_replay_iterator(
            self.replay_buffer,
            execution_length=self.cfg.execution_length,
            shuffle=False,
            seed=int(self.cfg.seed),
            load_all_episodes=False,
            batch_chunk_size=self.cfg.replay.get("epoch_batch_chunk_size", 0),
        )
        try:
            return int(epoch_iter.batches_per_epoch)
        finally:
            close = getattr(epoch_iter, "close", None)
            if callable(close):
                close()

    def _resolve_pretrain_schedule(self) -> tuple[int, int, int | None]:
        num_pretrain_steps = int(self.cfg.num_pretrain_steps)
        eval_every_steps = int(self.cfg.eval_every_steps)
        num_pretrain_epochs = self.cfg.get("num_pretrain_epochs", None)
        eval_every_epochs = self.cfg.get("eval_every_epochs", None)
        snapshot_every_epochs = self.cfg.get("snapshot_every_epochs", None)

        if (
            num_pretrain_epochs is None
            and eval_every_epochs is None
            and snapshot_every_epochs is None
        ):
            return num_pretrain_steps, eval_every_steps, None

        batches_per_epoch = self._offline_batches_per_epoch()
        if num_pretrain_epochs is not None:
            num_pretrain_steps = int(num_pretrain_epochs) * batches_per_epoch
        if eval_every_epochs is not None:
            eval_every_steps = int(eval_every_epochs) * batches_per_epoch
        snapshot_every_steps = None
        if snapshot_every_epochs is not None:
            snapshot_every_steps = int(snapshot_every_epochs) * batches_per_epoch

        logging.info(
            "Resolved offline pretrain schedule: %d batches/epoch, "
            "%d pretrain steps, eval every %d steps, snapshot every %s steps.",
            batches_per_epoch,
            num_pretrain_steps,
            eval_every_steps,
            "disabled" if snapshot_every_steps is None else str(snapshot_every_steps),
        )
        return num_pretrain_steps, eval_every_steps, snapshot_every_steps

    def _close_replay_iter(self):
        if self._replay_iter is None:
            return
        close = getattr(self._replay_iter, "close", None)
        if callable(close):
            close()
        self._replay_iter = None

    def _resolve_replay_save_dir(self, demo_replay: bool) -> str | None:
        if self.cfg.replay.save_dir is not None:
            base_dir = Path(self.cfg.replay.save_dir)
            return str(base_dir / "demo" if demo_replay else base_dir)
        if not self._should_persist_replay_files():
            return None
        return str(self.work_dir / ("demo_replay" if demo_replay else "replay"))

    def _should_persist_replay_files(self) -> bool:
        return bool(self.cfg.save_snapshot or self.cfg.replay.persist)

    def _requires_demo_stats(self) -> bool:
        return bool(
            self.cfg.use_standardization
            or self.cfg.use_min_max_normalization
            or self.cfg.norm_obs
        )

    def _saved_replay_exists(self, demo_replay: bool = False) -> bool:
        replay_dir = self._resolve_replay_save_dir(demo_replay=demo_replay)
        if replay_dir is None:
            return False
        replay_path = Path(replay_dir)
        return replay_path.exists() and any(replay_path.glob("*.npz"))

    def _should_collect_demos_from_dataset(self) -> bool:
        if self._should_use_lazy_replay():
            return False
        if self.use_demo_replay:
            return True
        if not bool(self.cfg.replay.reuse_saved):
            return True
        if self._requires_demo_stats():
            return True
        return not self._saved_replay_exists(demo_replay=False)

    def _should_use_lazy_replay(self) -> bool:
        return lazy_replay_enabled(self.cfg)

    def _timer_state_dict(self) -> dict[str, float]:
        return {"elapsed_total": self._timer.total_time()}

    def _restore_timer_state(self, state_dict: dict[str, float]):
        elapsed_total = float(state_dict["elapsed_total"])
        self._timer = _Timer()
        now = time.time()
        self._timer._start_time = now - elapsed_total
        self._timer._last_time = now

    def _rng_state_dict(self) -> dict[str, Any]:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
        }

    def _restore_rng_state(self, state_dict: dict[str, Any]):
        random.setstate(state_dict["python"])
        np.random.set_state(state_dict["numpy"])

    def train(self):
        self._previous_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._signal_handler)
        if self.cfg.num_train_frames > 0 and not self.train_envs:
            raise Exception("Train envs not created! Train can't be called!")
        try:
            self._train()
        except KeyboardInterrupt:
            logging.warning("Interrupted by Ctrl+C. Shutting down...")
            self.shutdown()
            raise SystemExit(130)
        except Exception as e:
            self.shutdown()
            raise e
        finally:
            if self._previous_sigint_handler is not None:
                signal.signal(signal.SIGINT, self._previous_sigint_handler)
                self._previous_sigint_handler = None

    def _train(self):
        # Load Demo
        self._load_demos()

        # Perform pretraining. This is suitable for behaviour cloning or Offline RL
        self._pretrain_on_demos()

        # Perform online rl with exploration.
        self._online_rl()

        if self.cfg.save_snapshot:
            self.save_snapshot()

        self._finalize_completed_training_artifacts()

        self.shutdown()

    def eval(self) -> dict[str, Any]:
        return self._eval(eval_record_all_episode=True)

    def _extract_vector_env_info(
        self, infos: dict[str, Any], env_idx: int, prefer_final: bool = False
    ) -> dict[str, Any]:
        if prefer_final:
            final_infos = infos.get("final_info")
            final_info_mask = infos.get("_final_info")
            if (
                final_infos is not None
                and final_info_mask is not None
                and final_info_mask[env_idx]
            ):
                return final_infos[env_idx]

        extracted = {}
        for key, value in infos.items():
            if key.startswith("_"):
                continue
            mask = infos.get(f"_{key}")
            if mask is not None and not mask[env_idx]:
                continue
            extracted[key] = value[env_idx]
        return extracted

    def _executed_vector_action_steps(self, infos: dict[str, Any]) -> np.ndarray:
        return np.asarray(
            [
                _executed_action_steps(
                    self._extract_vector_env_info(
                        infos,
                        env_idx,
                        prefer_final=True,
                    )
                )
                for env_idx in range(self.eval_envs.num_envs)
            ],
            dtype=np.int32,
        )

    def _eval_seed_for_episode(self, episode: int) -> int | None:
        env_cfg = self.cfg.get("env", {})
        eval_seeds = env_cfg.get("eval_seeds", None)
        if eval_seeds is not None:
            if len(eval_seeds) == 0:
                raise ValueError("env.eval_seeds must contain at least one seed.")
            return int(eval_seeds[int(episode) % len(eval_seeds)])
        eval_seed_start = env_cfg.get("eval_seed_start", None)
        if eval_seed_start is None:
            return None
        return int(eval_seed_start) + int(episode)

    def _set_agent_active_eval_seeds(self, seeds: list[int] | None) -> None:
        setter = getattr(self.agent, "set_active_eval_seeds", None)
        if callable(setter):
            setter(seeds)

    def _reset_vector_env_slots(
        self,
        observation,
        reset_requests: list[tuple[int, int]],
    ):
        envs = getattr(self.eval_envs, "envs", None)
        if envs is None:
            raise NotImplementedError(
                "Per-episode vector evaluation seeds require a vector environment "
                "that exposes its individual envs."
            )

        observations = list(
            gym.vector.utils.iterate(
                self.eval_envs.single_observation_space,
                observation,
            )
        )
        # Gymnasium 0.29 autoresets done slots without a seed inside step().
        # Reset those slots again and replace the observation before the next act().
        for env_idx, seed in reset_requests:
            observations[env_idx], _ = envs[env_idx].reset(seed=int(seed))

        out = getattr(self.eval_envs, "observations", None)
        if out is None:
            out = gym.vector.utils.create_empty_array(
                self.eval_envs.single_observation_space,
                n=self.eval_envs.num_envs,
            )
        observation = gym.vector.utils.concatenate(
            self.eval_envs.single_observation_space,
            observations,
            out,
        )
        if hasattr(self.eval_envs, "observations"):
            self.eval_envs.observations = observation
        return observation

    def _run_single_env_eval(
        self, eval_record_all_episode: bool = False
    ) -> dict[str, Any]:
        step, episode, total_reward, successes = 0, 0, 0, 0
        eval_until_episode = _Until(self.cfg.num_eval_episodes)
        first_rollout = []
        metrics = {}
        while eval_until_episode(episode):
            reset_seed = self._eval_seed_for_episode(episode)
            observation, info = self.eval_env.reset(seed=reset_seed)
            self._set_agent_active_eval_seeds(
                None if reset_seed is None else [reset_seed]
            )
            # eval agent always has last id (ids start from 0)
            self.agent.reset(self.main_loop_iterations, [self._eval_agent_indices[-1]])
            enabled = eval_record_all_episode or episode == 0
            self.eval_video_recorder.init(self.eval_env, enabled=enabled)
            termination, truncation = False, False
            while not (termination or truncation):
                (
                    action,
                    (next_observation, reward, termination, truncation, next_info),
                    env_metrics,
                ) = self._perform_env_steps(observation, self.eval_env, True)
                observation = next_observation
                info = next_info
                metrics.update(env_metrics)
                # Below is testing a feature wich can be enforced in v6.
                # The ability will allow agent info to be passed to envirionments.
                # This will be habdy for rednering any auxiliary outputs.
                if "agent_act_info" in env_metrics:
                    if hasattr(self.eval_env, "give_agent_info"):
                        self.eval_env.give_agent_info(env_metrics["agent_act_info"])
                self.eval_video_recorder.record(self.eval_env)
                total_reward += float(np.asarray(reward).item())
                step += _executed_action_steps(next_info)
            if episode == 0:
                first_rollout = np.array(self.eval_video_recorder.frames)
            self.eval_video_recorder.save(f"{self.global_env_steps}.mp4")
            success = info.get("task_success")
            if success is not None:
                successes += np.array(success).astype(int).item()
            else:
                successes = None
            episode += 1
        metrics.update(
            {
                "episode_reward": float(total_reward / episode),
                "episode_length": float(step * self.cfg.action_repeat / episode),
            }
        )
        if successes is not None:
            metrics["episode_success"] = successes / episode
        if self.cfg.log_eval_video and len(first_rollout) > 0:
            metrics["eval_rollout"] = dict(video=first_rollout, fps=4)
        return metrics

    def _run_vector_env_eval(
        self, eval_record_all_episode: bool = False
    ) -> dict[str, Any]:
        del eval_record_all_episode
        active_eval_seeds = [
            self._eval_seed_for_episode(episode)
            for episode in range(self.eval_envs.num_envs)
        ]
        use_fixed_eval_seeds = active_eval_seeds[0] is not None
        if use_fixed_eval_seeds:
            assert all(seed is not None for seed in active_eval_seeds)
            observation, _ = self.eval_envs.reset(seed=active_eval_seeds)
            self._set_agent_active_eval_seeds(active_eval_seeds)
        else:
            observation, _ = self.eval_envs.reset()
            self._set_agent_active_eval_seeds(None)
        active_episode_is_target = np.arange(self.eval_envs.num_envs) < int(
            self.cfg.num_eval_episodes
        )
        next_target_episode = min(
            self.eval_envs.num_envs,
            int(self.cfg.num_eval_episodes),
        )
        next_filler_episode = max(
            self.eval_envs.num_envs,
            int(self.cfg.num_eval_episodes),
        )
        self.agent.reset(self.main_loop_iterations, self._eval_agent_indices)

        completed_rewards = []
        completed_lengths = []
        completed_successes = []
        success_supported = True
        episode_rewards = np.zeros(self.eval_envs.num_envs, dtype=np.float64)
        episode_lengths = np.zeros(self.eval_envs.num_envs, dtype=np.int32)
        metrics = {}
        video_complete = not self.cfg.log_eval_video
        self.eval_video_recorder.init(self.eval_envs, enabled=self.cfg.log_eval_video)

        while len(completed_rewards) < self.cfg.num_eval_episodes or not video_complete:
            (
                _,
                (next_observation, reward, termination, truncation, next_info),
                env_metrics,
            ) = self._perform_env_steps(observation, self.eval_envs, True)
            metrics.update(env_metrics)
            episode_rewards += reward
            episode_lengths += self._executed_vector_action_steps(next_info)

            if not video_complete:
                self.eval_video_recorder.record(self.eval_envs)

            done_mask = np.logical_or(termination, truncation)
            if not video_complete and done_mask[0]:
                video_complete = True

            if not np.any(done_mask):
                observation = next_observation
                continue

            done_env_indices = np.flatnonzero(done_mask)
            self.agent.reset(
                self.main_loop_iterations,
                [self._eval_agent_indices[idx] for idx in done_env_indices],
            )
            for env_idx in done_env_indices:
                if active_episode_is_target[env_idx]:
                    completed_rewards.append(float(episode_rewards[env_idx]))
                    completed_lengths.append(int(episode_lengths[env_idx]))
                    info = self._extract_vector_env_info(
                        next_info, env_idx, prefer_final=True
                    )
                    if success_supported:
                        success = info.get("task_success")
                        if success is None:
                            success_supported = False
                            completed_successes = []
                        else:
                            completed_successes.append(
                                int(np.asarray(success).astype(int).item())
                            )
                active_episode_is_target[env_idx] = False
                episode_rewards[env_idx] = 0.0
                episode_lengths[env_idx] = 0

            needs_another_step = (
                len(completed_rewards) < self.cfg.num_eval_episodes
                or not video_complete
            )
            if needs_another_step:
                reset_requests = []
                for env_idx in done_env_indices:
                    if next_target_episode < self.cfg.num_eval_episodes:
                        next_episode = next_target_episode
                        next_target_episode += 1
                        active_episode_is_target[env_idx] = True
                    else:
                        next_episode = next_filler_episode
                        next_filler_episode += 1
                    if use_fixed_eval_seeds:
                        next_seed = self._eval_seed_for_episode(next_episode)
                        assert next_seed is not None
                        active_eval_seeds[env_idx] = next_seed
                        reset_requests.append((int(env_idx), next_seed))
                if use_fixed_eval_seeds:
                    next_observation = self._reset_vector_env_slots(
                        next_observation,
                        reset_requests,
                    )
                    self._set_agent_active_eval_seeds(active_eval_seeds)
            observation = next_observation

        if self.cfg.log_eval_video and len(self.eval_video_recorder.frames) > 0:
            self.eval_video_recorder.save(f"{self.global_env_steps}.mp4")
            metrics["eval_rollout"] = dict(
                video=np.array(self.eval_video_recorder.frames), fps=4
            )

        metrics.update(
            {
                "episode_reward": float(np.mean(completed_rewards)),
                "episode_length": float(
                    np.mean(completed_lengths) * self.cfg.action_repeat
                ),
            }
        )
        if success_supported and completed_successes:
            metrics["episode_success"] = float(np.mean(completed_successes))
        return metrics

    def _get_rollout_diagnostics(self) -> dict[str, Any]:
        rollout_diagnostics = getattr(self.agent, "rollout_diagnostics", None)
        if not callable(rollout_diagnostics):
            return {}
        return dict(rollout_diagnostics())

    def _eval(self, eval_record_all_episode: bool = False) -> dict[str, Any]:
        # TODO: In future, this func could do with a further refactor
        self._ensure_eval_envs_created()
        if self.eval_env is None and self.eval_envs is None:
            raise ValueError(
                "Evaluation requested but no evaluation environment is configured."
            )
        reset_aligned_eval_noise = getattr(
            self.agent,
            "reset_aligned_eval_noise",
            None,
        )
        if callable(reset_aligned_eval_noise):
            reset_aligned_eval_noise()
        self.agent.set_eval_env_running(True)
        try:
            if self.eval_envs is not None:
                metrics = self._run_vector_env_eval(
                    eval_record_all_episode=eval_record_all_episode
                )
            else:
                metrics = self._run_single_env_eval(
                    eval_record_all_episode=eval_record_all_episode
                )
            metrics.update(self._get_rollout_diagnostics())
            return metrics
        finally:
            self._set_agent_active_eval_seeds(None)
            self.agent.set_eval_env_running(False)
            if self._defer_live_eval_env_creation:
                self._close_eval_envs()

    def _add_to_replay(
        self,
        actions,
        observations,
        rewards,
        terminations,
        truncations,
        infos,
        next_infos,
    ):
        # TODO: In future, this func could do with a further refactor
        # TODO: Add transitions into replay buffer in sliding window fashion??
        #      Currently, as train env has action sequence wrapper which only gives
        #      total reward and final obs for the full sequence, we can't perform
        #      sliding window.

        # Convert observation to list of observations ordered by train_env index
        list_of_obs_dicts = [
            dict(zip(observations, t)) for t in zip(*observations.values())
        ]
        store_structured_explore = (
            "structured_explore" in self.extra_replay_elements.keys()
        )
        store_explored = "explored" in self.extra_replay_elements.keys()
        if os.environ.get("ROBOBASE_DEBUG_EXPLORED") and store_explored:
            logging.info(
                "[dbg] flag=%s applied=%s countdown=%s",
                getattr(self.agent, "_last_bin_explored", "NOATTR"),
                getattr(self.agent, "_last_bin_explore_applied", "NOATTR"),
                getattr(self.agent, "_bin_explored_exec_remaining", "NOATTR"),
            )
        bin_explored = np.asarray(
            getattr(
                self.agent,
                "_last_bin_explored",
                np.zeros((self.train_envs.num_envs,), dtype=np.bool_),
            ),
            dtype=np.bool_,
        )
        structured_explore_mask = np.asarray(
            getattr(
                self.agent,
                "_last_structured_exploration_mask",
                np.zeros((self.train_envs.num_envs,), dtype=np.bool_),
            ),
            dtype=np.bool_,
        )
        structured_explore_start = np.asarray(
            getattr(
                self.agent,
                "_last_structured_exploration_start",
                np.zeros((self.train_envs.num_envs,), dtype=np.bool_),
            ),
            dtype=np.bool_,
        )
        structured_explore_dimension = np.asarray(
            getattr(
                self.agent,
                "_last_structured_exploration_dimension",
                np.full((self.train_envs.num_envs,), -1, dtype=np.int16),
            ),
            dtype=np.int16,
        )
        structured_explore_delta = np.asarray(
            getattr(
                self.agent,
                "_last_structured_exploration_delta",
                np.zeros((self.train_envs.num_envs,), dtype=np.float32),
            ),
            dtype=np.float32,
        )
        structured_explore_assignment_prob = np.asarray(
            getattr(
                self.agent,
                "_last_structured_exploration_assignment_prob",
                np.ones((self.train_envs.num_envs,), dtype=np.float32),
            ),
            dtype=np.float32,
        )
        if store_structured_explore and structured_explore_mask.shape != (
            self.train_envs.num_envs,
        ):
            raise ValueError(
                "agent structured-exploration mask does not match train envs"
            )
        if store_structured_explore:
            expected_shape = (self.train_envs.num_envs,)
            metadata = {
                "start": structured_explore_start,
                "dimension": structured_explore_dimension,
                "delta": structured_explore_delta,
                "assignment_prob": structured_explore_assignment_prob,
            }
            mismatched = {
                name: value.shape
                for name, value in metadata.items()
                if value.shape != expected_shape
            }
            if mismatched:
                raise ValueError(
                    "agent structured-exploration metadata does not match "
                    f"train envs: {mismatched}"
                )
        agents_reset = []
        for i in range(self.train_envs.num_envs):
            # Add transitions to episode rollout
            transition_info = {k: infos[k][i] for k in infos.keys()}
            if store_explored:
                transition_info["explored"] = np.uint8(bin_explored[i])
            if store_structured_explore:
                transition_info["structured_explore"] = np.uint8(
                    structured_explore_mask[i]
                )
                transition_info["structured_explore_start"] = np.uint8(
                    structured_explore_start[i]
                )
                transition_info["structured_explore_dimension"] = np.int16(
                    structured_explore_dimension[i]
                )
                transition_info["structured_explore_delta"] = np.float32(
                    structured_explore_delta[i]
                )
                transition_info[
                    "structured_explore_assignment_prob"
                ] = np.float32(structured_explore_assignment_prob[i])
            self._episode_rollouts[i].append(
                (
                    actions[i],
                    list_of_obs_dicts[i],
                    rewards[i],
                    terminations[i],
                    truncations[i],
                    transition_info,
                    {k: next_infos[k][i] for k in next_infos.keys()},
                )
            )

            # If episode finishes, add to replay buffer.
            if terminations[i] or truncations[i]:
                agents_reset.append(i)
                ep = self._episode_rollouts[i]
                last_next_info = ep[-1][-1]
                assert last_next_info["_final_observation"]
                # `next_info` containing `final_info` is the first info of next episode
                # we need to extract `final_info` and use it as true next_info
                final_obs = last_next_info["final_observation"]
                final_info = last_next_info["final_info"]
                task_success = int(final_info.get("task_success", 0) > 0.0)

                # Re-labeling successful demonstrations as success, following CQN
                relabeling_as_demo = (
                    task_success
                    and self.use_demo_replay
                    and self.cfg.use_self_imitation
                )
                store_mc_return = "mc_return" in self.extra_replay_elements.keys()
                mc_returns = utils.discounted_episode_returns(
                    [transition[2] for transition in ep],
                    self.cfg.replay.gamma,
                )
                for transition_index, (
                    act,
                    obs,
                    rew,
                    term,
                    trunc,
                    info,
                    next_info,
                ) in enumerate(ep):
                    # Only keep the last frames regardless of frame stacks because
                    # replay buffer always store single-step transitions
                    obs = {k: v[-1] for k, v in obs.items()}

                    # Online replay stores one transition per actually executed
                    # environment action. Receding-horizon wrappers expose that
                    # action explicitly because it may be a temporal ensemble of
                    # several predicted chunks rather than ``act[0]``.
                    replay_act = (
                        _replay_action_from_step(act, next_info)
                        if self.cfg.method.is_rl
                        else np.asarray(act)[0]
                    )

                    if relabeling_as_demo:
                        info["demo"] = 1
                    else:
                        info["demo"] = 0
                    if store_mc_return:
                        info["mc_return"] = mc_returns[transition_index]

                    # Filter out unwanted keys in info
                    extra_replay_elements = {
                        k: v
                        for k, v in info.items()
                        if k in self.extra_replay_elements.keys()
                    }

                    self.replay_buffer.add(
                        obs, replay_act, rew, term, trunc, **extra_replay_elements
                    )
                    if relabeling_as_demo:
                        self.demo_replay_buffer.add(
                            obs, replay_act, rew, term, trunc, **extra_replay_elements
                        )

                # Add final obs
                # Only keep the last frames regardless of frame stacks because
                # replay buffer always store single-step transitions
                final_obs = {k: v[-1] for k, v in final_obs.items()}
                self.replay_buffer.add_final(final_obs)
                if relabeling_as_demo:
                    self.demo_replay_buffer.add_final(final_obs)

                # clean up
                self._global_env_episode += 1
                self._episode_rollouts[i].clear()

        self.agent.reset(self.main_loop_iterations, agents_reset)  # clear hidden dim

    def _handle_on_policy_resets(self, terminations, truncations):
        agents_reset = [
            index
            for index, (terminated, truncated) in enumerate(
                zip(terminations, truncations, strict=True)
            )
            if terminated or truncated
        ]
        self._global_env_episode += len(agents_reset)
        self.agent.reset(self.main_loop_iterations, agents_reset)

    def _signal_handler(self, sig, frame):
        del sig, frame
        self._sigint_count += 1
        print("\nCtrl+C detected. Preparing to shutdown...")
        self._shutting_down = True
        if self._sigint_count == 1:
            raise KeyboardInterrupt
        raise KeyboardInterrupt

    def _load_demos(self):
        if self._snapshot_loaded:
            return
        if self._should_use_lazy_replay():
            logging.info("Using lazy replay; skipping DemoEnv replay import.")
            return
        if self.replay_buffer.reused_existing and not self.use_demo_replay:
            logging.info(
                "Reusing saved replay episodes from disk. Skipping demo import."
            )
            return
        if (num_demos := self.cfg.demos) != 0:
            if self._shared_demo_cache is not None:
                self._load_demos_with_shared_cache()
            else:
                # NOTE: Currently we do not protect demos from being evicted from replay
                self.env_factory.load_demos_into_replay(
                    self.cfg,
                    self.replay_buffer,
                    is_demo_buffer=(
                        True if self.cfg.is_imitation_learning else False
                    ),
                )
                if self.use_demo_replay:
                    # Load demos to the dedicated demo_replay_buffer
                    self.env_factory.load_demos_into_replay(
                        self.cfg, self.demo_replay_buffer, is_demo_buffer=True
                    )
            if bool(self.cfg.replay.get("discard_loaded_demos_after_replay", True)):
                clear_loaded_demos = getattr(
                    self.env_factory,
                    "clear_loaded_demos",
                    None,
                )
                if callable(clear_loaded_demos):
                    clear_loaded_demos()
                    gc.collect()

        if self.cfg.replay_size_before_train > 0:
            num_demos = _normalize_demos_count(num_demos)
            diff = self.cfg.replay_size_before_train - len(self.replay_buffer)
            if num_demos > 0 and diff > 0:
                logging.warning(
                    f"Collecting additional {diff} random samples even though there "
                    f"are {len(self.replay_buffer)} demo samples inside the buffer. "
                    "Please make sure that this is an intended behavior."
                )

    def _load_demos_with_shared_cache(self) -> None:
        cache = self._shared_demo_cache
        if cache is None:
            raise RuntimeError("shared demo cache was not configured")
        if self.replay_buffer.reused_existing:
            raise ValueError(
                "Shared demo cache cannot seed a non-empty reused online replay."
            )
        if self.demo_replay_buffer.reused_existing:
            raise ValueError(
                "Shared demo cache cannot seed a non-empty reused demo replay."
            )
        with cache.lock():
            if cache.is_complete():
                online_count = self.replay_buffer.seed_from_replay_directory(
                    cache.source("all_demos")
                )
                expert_count = self.demo_replay_buffer.seed_from_replay_directory(
                    cache.source("expert_demos")
                )
                logging.info(
                    "Reused shared demo replay cache (%d all-demo, %d expert files).",
                    online_count,
                    expert_count,
                )
                return

            self.env_factory.load_demos_into_replay(
                self.cfg,
                self.replay_buffer,
                is_demo_buffer=False,
            )
            self.env_factory.load_demos_into_replay(
                self.cfg,
                self.demo_replay_buffer,
                is_demo_buffer=True,
            )
            manifest = cache.publish(
                {
                    "all_demos": self.replay_buffer.replay_dir,
                    "expert_demos": self.demo_replay_buffer.replay_dir,
                }
            )
            logging.info(
                "Published shared demo replay cache (%d all-demo, %d expert files).",
                manifest["files"]["all_demos"],
                manifest["files"]["expert_demos"],
            )

    def _guard_non_finite_update(self, metrics: dict[str, Any]) -> None:
        """Abort as soon as the loss or any instrumented update stage is bad.

        Divergence poisons every parameter in one step, after which training
        keeps running for hours producing zero-success rollouts that read like
        a behavioural collapse (cqn-flow.md 64.2 retracted a whole verdict to
        this). Checking once per update block costs one scalar sync.
        """
        self._last_update_metrics = dict(metrics)
        loss = metrics.get("critic_loss", None)
        if loss is None:
            return
        try:
            value = float(loss)
        except (TypeError, ValueError):
            return
        committed = metrics.get("nan_diag/update_committed", 1.0)
        try:
            committed = float(np.asarray(committed))
        except (TypeError, ValueError):
            committed = 0.0
        if math.isfinite(value) and committed >= 0.5:
            return
        step = self.main_loop_iterations
        dump_dir = self._dump_non_finite_forensics(step, value)
        logging.error(
            "Non-finite update (critic_loss=%s, committed=%s) at iteration %d; "
            "aborting. "
            "Forensics: %s",
            value,
            committed,
            step,
            dump_dir or "(dump failed; see traceback above)",
        )
        raise FloatingPointError(
            f"critic update became non-finite at iteration {step} "
            f"(loss={value}, committed={committed})"
        )

    def _dump_non_finite_forensics(self, step: int, loss_value: float):
        """Write the inputs and parameter state behind a non-finite update.

        Everything here is best-effort: a forensics failure must never mask the
        abort itself, so each section is guarded independently.
        """
        try:
            out = self.work_dir / "nonfinite_dump"
            out.mkdir(parents=True, exist_ok=True)
            summary: dict[str, Any] = {
                "iteration": int(step),
                "critic_loss": str(loss_value),
                "env_steps": int(getattr(self, "_global_env_episode_step", -1)),
            }
            summary["nan_diag"] = {
                key: float(np.asarray(value))
                for key, value in getattr(self, "_last_update_metrics", {}).items()
                if str(key).startswith("nan_diag/")
            }

            def leaf_report(tree, label):
                import jax

                rows = []
                try:
                    leaves, treedef = jax.tree_util.tree_flatten(tree)
                    paths = [str(p) for p in range(len(leaves))]
                    try:
                        flat = jax.tree_util.tree_flatten_with_path(tree)[0]
                        paths = ["/".join(str(k) for k in p) for p, _ in flat]
                        leaves = [v for _, v in flat]
                    except Exception:
                        pass
                    for name, leaf in zip(paths, leaves):
                        arr = np.asarray(leaf)
                        if arr.dtype.kind not in "fc":
                            continue
                        finite = np.isfinite(arr)
                        rows.append(
                            {
                                "leaf": name,
                                "shape": list(arr.shape),
                                "finite_fraction": float(finite.mean()),
                                "max_abs_finite": (
                                    float(np.abs(arr[finite]).max())
                                    if finite.any()
                                    else None
                                ),
                            }
                        )
                except Exception as exc:  # pragma: no cover - diagnostics only
                    rows = [{"error": repr(exc)}]
                summary[label] = rows

            leaf_report(getattr(self.agent, "params", None), "params")
            leaf_report(getattr(self.agent, "opt_state", None), "opt_state")
            leaf_report(
                getattr(self.agent, "target_critic_params", None),
                "target_critic_params",
            )

            if bool(self.cfg.get("nonfinite_dump_save_state", False)):
                self._atomic_pickle_dump(
                    {
                        "agent": self.agent.state_dict(),
                        "agent_checkpoint_state": (
                            self.agent.checkpoint_state_dict()
                        ),
                    },
                    out / f"pre_update_state_iter{step}.pkl",
                )

            # The batches that produced the blow-up. `recent[-1]` is the one
            # the failing update consumed.
            tap = self._replay_iter
            recent = list(getattr(tap, "recent", []) or [])
            arrays: dict[str, np.ndarray] = {}
            batch_reports = []
            for i, batch in enumerate(recent):
                rep = {}
                for k, v in batch.items():
                    try:
                        arr = np.asarray(v)
                    except Exception:
                        continue
                    arrays[f"b{i}__{k}"] = arr
                    if arr.dtype.kind in "fc":
                        finite = np.isfinite(arr)
                        entry = {
                            "shape": list(arr.shape),
                            "finite_fraction": float(finite.mean()),
                        }
                        if finite.any():
                            absv = np.abs(np.where(finite, arr, 0.0))
                            entry["max_abs"] = float(absv.max())
                            entry["argmax"] = [
                                int(x) for x in np.unravel_index(absv.argmax(), absv.shape)
                            ]
                        rep[k] = entry
                    else:
                        rep[k] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
                batch_reports.append(rep)
            summary["batches"] = batch_reports
            summary["batches_retained"] = len(recent)
            if arrays:
                np.savez_compressed(out / f"batches_iter{step}.npz", **arrays)
            (out / f"summary_iter{step}.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True)
            )
            return str(out)
        except Exception as exc:  # pragma: no cover - diagnostics only
            logging.exception("Non-finite forensics dump failed: %r", exc)
            return None

    def _apply_demo_batch_fraction_schedule(self) -> float | None:
        """Scheduled demo/online sample split at constant total batch.

        Mutates both buffers' default sample sizes before an update block.
        The consuming iterators re-read .batch_size per __next__, so the
        change is immediate (up to replay_prefetch_size updates of skew).
        """
        schedule_spec = self.cfg.get("demo_batch_fraction_schedule", None)
        if schedule_spec is None:
            return None
        if self.demo_replay_buffer is None:
            raise ValueError(
                "demo_batch_fraction_schedule requires demo_batch_size."
            )
        if int(self.cfg.replay.get("num_workers", 0)) != 0:
            raise ValueError(
                "demo_batch_fraction_schedule requires replay.num_workers=0."
            )
        total = int(self.cfg.batch_size) + int(self.cfg.demo_batch_size)
        step = self.pretrain_steps + self.main_loop_iterations
        fraction = float(utils.schedule(str(schedule_spec), step))
        fraction = min(max(fraction, 0.0), 1.0)
        demo_count = int(round(fraction * total))
        demo_count = min(max(demo_count, 1), total - 1)
        self.demo_replay_buffer.set_batch_size(demo_count)
        self.replay_buffer.set_batch_size(total - demo_count)
        return demo_count / total

    def _perform_updates(self) -> dict[str, Any]:
        if self.agent.logging:
            start_time = time.time()
        metrics = {}
        realized_demo_fraction = self._apply_demo_batch_fraction_schedule()
        if realized_demo_fraction is not None and self.agent.logging:
            metrics["demo_batch_fraction"] = realized_demo_fraction
        update_slots = self._num_update_slots()
        self.agent.train(True)
        for i in range(update_slots):
            if (self.main_loop_iterations + i) % self.cfg.update_every_steps != 0:
                # Skip update
                continue
            metrics.update(
                self.agent.update(
                    self.replay_iter, self.main_loop_iterations + i, self.replay_buffer
                )
            )
        self.agent.train(False)
        self._guard_non_finite_update(metrics)
        if self.agent.logging:
            execution_time_for_update = time.time() - start_time
            metrics["agent_batched_updates_per_second"] = (
                update_slots / execution_time_for_update
            )
            metrics["agent_updates_per_second"] = (
                update_slots * self.cfg.batch_size
            ) / execution_time_for_update
        return metrics

    def _perform_env_steps(
        self, observations: dict[str, np.ndarray], env: gym.Env, eval_mode: bool
    ) -> tuple[np.ndarray, tuple, dict[str, Any]]:
        if self.agent.logging:
            start_time = time.time()
        env_batch_size = env.num_envs if getattr(env, "is_vector_env", False) else 1
        metrics = {}
        backend_observations = observations
        if eval_mode and not getattr(env, "is_vector_env", False):
            backend_observations = {
                k: np.expand_dims(v, axis=0) for k, v in observations.items()
            }
        action = self.agent.act(
            backend_observations, self.main_loop_iterations, eval_mode=eval_mode
        )
        if isinstance(action, tuple):
            action, act_info = action
            metrics["agent_act_info"] = act_info
        action = np.asarray(action)

        if action.ndim != 3:
            raise ValueError(
                "Expected actions from `agent.act` to have shape "
                "(Batch, Timesteps, Action Dim)."
            )
        if eval_mode and not getattr(env, "is_vector_env", False):
            action = action[0]  # we expect batch of 1 for eval

        if self.agent.logging:
            execution_time_for_act = time.time() - start_time
            metrics["agent_act_steps_per_second"] = (
                env_batch_size / execution_time_for_act
            )
            start_time = time.time()

        *env_step_tuple, next_info = env.step(action)

        if self.agent.logging:
            execution_time_for_env_step = time.time() - start_time
            metrics["env_steps_per_second"] = (
                env_batch_size / execution_time_for_env_step
            )
            for k, v in next_info.items():
                # if train env, then will be vectorised, so get first elem
                if getattr(env, "is_vector_env", False):
                    metrics[f"env_info/{k}"] = v[0] if len(v) > 0 else v
                else:
                    metrics[f"env_info/{k}"] = v if eval_mode else v[0]

        return action, (*env_step_tuple, next_info), metrics

    def _pretrain_on_demos(self):
        (
            num_pretrain_steps,
            eval_every_steps,
            snapshot_every_steps,
        ) = self._resolve_pretrain_schedule()
        if num_pretrain_steps > 0:
            pre_train_until_step = _Until(num_pretrain_steps)
            should_pretrain_log = _Every(self.cfg.log_pretrain_every)
            should_pretrain_eval = _Every(eval_every_steps)
            snapshot_every_n = 0
            if self.cfg.save_snapshot:
                snapshot_every_n = (
                    snapshot_every_steps
                    if snapshot_every_steps is not None
                    else self.cfg.snapshot_every_n
                )
            should_pretrain_save_snapshot = _Every(snapshot_every_n)
            snapshot_save_start_step = int(self.cfg.get("snapshot_save_start_step", 0))
            if len(self.replay_buffer) <= 0:
                raise ValueError(
                    "there is no sample to pre-train with in the replay buffer "
                    f"but num_pretrain_steps ({num_pretrain_steps}) is > 0"
                )

            while pre_train_until_step(self.pretrain_steps):
                if self._shutting_down:
                    break
                self.agent.logging = False

                fused_update_steps = int(
                    self.cfg.get("backend", {}).get("fused_update_steps", 1)
                    if self.cfg.get("backend", None)
                    else 1
                )
                if (
                    fused_update_steps > 1
                    and hasattr(self.agent, "update_many")
                    and not should_pretrain_log(self.pretrain_steps)
                ):
                    boundary_step = int(num_pretrain_steps)
                    if self.cfg.log_pretrain_every:
                        next_log_step = (
                            self.pretrain_steps // self.cfg.log_pretrain_every + 1
                        ) * self.cfg.log_pretrain_every
                        boundary_step = min(boundary_step, next_log_step)
                    if eval_every_steps:
                        next_eval_step = (
                            self.pretrain_steps // eval_every_steps + 1
                        ) * eval_every_steps
                        boundary_step = min(boundary_step, next_eval_step - 1)
                    if snapshot_every_n:
                        next_snapshot_step = (
                            self.pretrain_steps // snapshot_every_n + 1
                        ) * snapshot_every_n
                        boundary_step = min(boundary_step, next_snapshot_step - 1)
                    block_steps = min(
                        fused_update_steps,
                        max(0, boundary_step - self.pretrain_steps),
                    )
                    if block_steps > 1:
                        self.agent.update_many(
                            self.replay_iter, block_steps, self.replay_buffer
                        )
                        self._pretrain_step += block_steps
                        continue

                if should_pretrain_log(self.pretrain_steps):
                    self.agent.logging = True
                pretrain_metrics = self._perform_updates()

                if should_pretrain_log(self.pretrain_steps):
                    pretrain_metrics.update(self._get_common_metrics())
                    self.logger.log_metrics(
                        pretrain_metrics, self.pretrain_steps, prefix="pretrain"
                    )

                next_pretrain_step = self.pretrain_steps + 1
                if should_pretrain_eval(next_pretrain_step):
                    eval_metrics = self._eval()
                    eval_metrics.update(
                        self._get_common_metrics(pretrain_steps=next_pretrain_step)
                    )
                    self.logger.log_metrics(
                        eval_metrics, next_pretrain_step, prefix="pretrain_eval"
                    )

                if (
                    next_pretrain_step >= snapshot_save_start_step
                    and should_pretrain_save_snapshot(next_pretrain_step)
                ):
                    self.save_snapshot(counter_increments={"_pretrain_step": 1})

                self._pretrain_step += 1

    def _online_rl(self):
        if self.train_envs is None or self.cfg.num_train_frames <= 0:
            return
        train_until_frame = _Until(self.cfg.num_train_frames)
        should_log = _Every(self.cfg.log_every)
        eval_every_n = self.cfg.eval_every_steps if self.eval_env is not None else 0
        should_eval = _Every(eval_every_n)
        snapshot_every_n = self.cfg.snapshot_every_n if self.cfg.save_snapshot else 0
        should_save_snapshot = _Every(snapshot_every_n)
        snapshot_save_start_step = int(self.cfg.get("snapshot_save_start_step", 0))
        observations, info = self.train_envs.reset()
        on_policy = bool(getattr(self.agent, "on_policy", False))
        #  We use agent 0 to accumulate stats about how the training agents are doing
        agent_0_ep_len = agent_0_reward = 0
        agent_0_prev_ep_len = agent_0_prev_reward = None
        while train_until_frame(self.global_env_steps):
            metrics = {}
            self.agent.logging = False
            if should_log(self.main_loop_iterations):
                self.agent.logging = True
            if not on_policy and _online_updates_ready(
                self.cfg,
                main_loop_iterations=self.main_loop_iterations,
                replay_size=len(self.replay_buffer),
            ):
                update_metrics = self._perform_updates()
                metrics.update(update_metrics)

            (
                action,
                (next_observations, rewards, terminations, truncations, next_info),
                env_metrics,
            ) = self._perform_env_steps(observations, self.train_envs, False)

            agent_0_reward += rewards[0]
            agent_0_ep_len += 1
            if terminations[0] or truncations[0]:
                agent_0_prev_ep_len = agent_0_ep_len
                agent_0_prev_reward = agent_0_reward
                agent_0_ep_len = agent_0_reward = 0

            metrics.update(env_metrics)
            if on_policy:
                self.agent.observe_transition(
                    rewards=rewards,
                    terminations=terminations,
                    truncations=truncations,
                    next_observations=next_observations,
                    next_info=next_info,
                )
                if self.agent.rollout_ready:
                    self.agent.train(True)
                    metrics.update(
                        self.agent.update(
                            None,
                            self.main_loop_iterations,
                            None,
                        )
                    )
                    self.agent.train(False)
            if on_policy:
                self._handle_on_policy_resets(terminations, truncations)
            else:
                self._add_to_replay(
                    action,
                    observations,
                    rewards,
                    terminations,
                    truncations,
                    info,
                    next_info,
                )
            observations = next_observations
            info = next_info
            if should_log(self.main_loop_iterations):
                metrics.update(self._get_common_metrics())
                metrics.update(self._get_rollout_diagnostics())
                if agent_0_prev_reward is not None and agent_0_prev_ep_len is not None:
                    metrics.update(
                        {
                            "episode_reward": agent_0_prev_reward,
                            "episode_length": agent_0_prev_ep_len
                            * self.cfg.action_repeat,
                        }
                    )
                self.logger.log_metrics(metrics, self.global_env_steps, prefix="train")

            next_main_loop_iteration = self.main_loop_iterations + 1
            if should_eval(next_main_loop_iteration):
                eval_metrics = self._eval()
                eval_metrics.update(
                    self._get_common_metrics(
                        main_loop_iterations=next_main_loop_iteration
                    )
                )
                self.logger.log_metrics(
                    eval_metrics,
                    self._calculate_global_env_steps(
                        main_loop_iterations=next_main_loop_iteration
                    ),
                    prefix="eval",
                )

            if (
                next_main_loop_iteration >= snapshot_save_start_step
                and should_save_snapshot(next_main_loop_iteration)
            ):
                self.save_snapshot(counter_increments={"_main_loop_iterations": 1})

            if self._shutting_down:
                break

            self._main_loop_iterations += 1

    def _get_common_metrics(
        self,
        *,
        main_loop_iterations: int | None = None,
        pretrain_steps: int | None = None,
    ) -> dict[str, Any]:
        _, total_time = self._timer.reset()
        if main_loop_iterations is None:
            main_loop_iterations = self.main_loop_iterations
        if pretrain_steps is None:
            pretrain_steps = self.pretrain_steps
        iteration = pretrain_steps if self.train_envs is None else main_loop_iterations
        metrics = {
            "total_time": total_time,
            "iteration": iteration,
            "env_steps": self._calculate_global_env_steps(
                main_loop_iterations=main_loop_iterations,
                pretrain_steps=pretrain_steps,
            ),
            "env_episodes": self.global_env_episodes,
            "buffer_size": len(self.replay_buffer),
        }
        if self.use_demo_replay:
            metrics["demo_buffer_size"] = len(self.demo_replay_buffer)
        return metrics

    def shutdown(self):
        self._close_replay_iter()
        self._close_eval_envs()

        if self.train_envs is not None:
            self.train_envs.close()
        self.replay_buffer.shutdown()
        if self.use_demo_replay:
            self.demo_replay_buffer.shutdown()

    def _should_create_train_env(self) -> bool:
        return (
            bool(self.cfg.create_train_env)
            and self.cfg.num_train_envs > 0
            and self.cfg.num_train_frames > 0
        )

    def _should_defer_live_eval_env_creation(self) -> bool:
        return (
            self.cfg.num_eval_episodes > 0
            and self.cfg.num_train_frames == 0
            and getattr(self.cfg.env, "env_name", None) == "robomimic"
            and bool(getattr(self.cfg.env, "use_live_env", False))
        )

    def _num_update_slots(self) -> int:
        if self.train_envs is not None:
            return self.train_envs.num_envs
        return max(int(self.cfg.num_train_envs), 1)

    def _ensure_eval_envs_created(self):
        if self.cfg.num_eval_episodes <= 0:
            return
        if self.cfg.num_eval_envs > 1:
            if self.eval_env is not None:
                self.eval_env.close()
                self.eval_env = None
            if self.eval_envs is None:
                self.eval_envs = self.env_factory.make_eval_envs(self.cfg)
        else:
            if self.eval_envs is not None:
                self.eval_envs.close()
                self.eval_envs = None
            if self.eval_env is None:
                self.eval_env = self.env_factory.make_eval_env(self.cfg)

    def _close_eval_envs(self):
        if self.eval_envs is not None:
            self.eval_envs.close()
            self.eval_envs = None
        if self.eval_env is not None:
            self.eval_env.close()
            self.eval_env = None

    def save_snapshot(self, counter_increments: dict[str, int] | None = None):
        keys_to_save = [
            "_pretrain_step",
            "_main_loop_iterations",
            "_global_env_episode",
            "cfg",
        ]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        if counter_increments is not None:
            for key, value in counter_increments.items():
                payload[key] += value
        snapshot_step = self._calculate_global_env_steps(
            main_loop_iterations=payload["_main_loop_iterations"],
            pretrain_steps=payload["_pretrain_step"],
        )
        snapshot = self.work_dir / "snapshots" / f"{snapshot_step}_snapshot.pkl"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        payload["snapshot_version"] = 3
        payload["snapshot_step"] = snapshot_step
        payload["agent"] = self.agent.state_dict()
        payload["replay_buffer"] = self.replay_buffer.state_dict()
        payload["rng_state"] = self._rng_state_dict()
        payload["timer_state"] = self._timer_state_dict()
        if self.logger.wandb_run_id is not None:
            payload["wandb_run_id"] = self.logger.wandb_run_id
        if self.use_demo_replay:
            payload["demo_replay_buffer"] = self.demo_replay_buffer.state_dict()
        self._atomic_pickle_dump(payload, snapshot)
        # The optimizer tree is ~half of a snapshot and is only ever read by a
        # resume, which always resumes from the newest one. Keeping it in a
        # single overwritten sidecar leaves every numbered snapshot a valid
        # evaluation checkpoint at half the size.
        self._atomic_pickle_dump(
            {
                "resume_state_version": 1,
                "snapshot_step": snapshot_step,
                "agent_checkpoint_state": self.agent.checkpoint_state_dict(),
            },
            self.work_dir / "snapshots" / "resume_state.pkl",
        )
        latest_snapshot = self.work_dir / "snapshots" / "latest_snapshot.pkl"
        # Point `latest` at the just-written snapshot via a relative symlink
        # instead of a full copy, so we don't duplicate the (large) checkpoint
        # on disk. Resume reads latest_snapshot.pkl and transparently follows
        # the link; the numbered `<step>_snapshot.pkl` is kept as-is.
        latest_snapshot.unlink(missing_ok=True)
        latest_snapshot.symlink_to(snapshot.name)
        if bool(self.cfg.get("artifacts", {}).get("save_eval_checkpoints", False)):
            self._save_eval_checkpoint(payload, snapshot_step)
        self._prune_resume_snapshots()

    @staticmethod
    def _atomic_pickle_dump(payload: dict[str, Any], destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.tmp.{os.getpid()}"
        )
        try:
            with temporary.open("wb") as handle:
                pickle.dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _save_eval_checkpoint(self, snapshot_payload: dict, snapshot_step: int) -> Path:
        """Write a params-only checkpoint suitable for evaluation, not resume."""

        destination = (
            self.work_dir
            / "eval_checkpoints"
            / f"{snapshot_step}_checkpoint.pkl"
        )
        payload = {
            "eval_checkpoint_version": 1,
            "snapshot_step": int(snapshot_step),
            "_pretrain_step": snapshot_payload["_pretrain_step"],
            "_main_loop_iterations": snapshot_payload["_main_loop_iterations"],
            "_global_env_episode": snapshot_payload["_global_env_episode"],
            "cfg": snapshot_payload["cfg"],
            "agent": snapshot_payload["agent"],
        }
        self._atomic_pickle_dump(payload, destination)
        return destination

    def _prune_resume_snapshots(self) -> list[Path]:
        keep = int(self.cfg.get("artifacts", {}).get("resume_keep_last", 0))
        if keep <= 0:
            return []
        snapshot_dir = self.work_dir / "snapshots"
        numbered = []
        for path in snapshot_dir.glob("*_snapshot.pkl"):
            if path.name == "latest_snapshot.pkl":
                continue
            try:
                step = int(path.name.removesuffix("_snapshot.pkl"))
            except ValueError:
                continue
            numbered.append((step, path))
        removed = []
        for _, path in sorted(numbered)[:-keep]:
            path.unlink(missing_ok=True)
            removed.append(path)
        if removed:
            logging.info(
                "Pruned %d obsolete resume snapshots; retaining latest %d.",
                len(removed),
                keep,
            )
        return removed

    def _finalize_completed_training_artifacts(self) -> None:
        """Drop resume-only state after a natural, successfully saved finish."""

        artifacts = self.cfg.get("artifacts", {})
        delete_replay = bool(
            artifacts.get("delete_replay_on_train_complete", False)
        )
        delete_resume = bool(
            artifacts.get("delete_resume_on_train_complete", False)
        )
        if not (delete_replay or delete_resume):
            return
        if not bool(artifacts.get("save_eval_checkpoints", False)):
            raise ValueError(
                "Training artifact cleanup requires params-only eval checkpoints."
            )
        eval_checkpoints = sorted(
            (self.work_dir / "eval_checkpoints").glob("*_checkpoint.pkl")
        )
        if not eval_checkpoints:
            raise RuntimeError("Refusing cleanup: no eval checkpoint was written.")

        self._close_replay_iter()
        removed_replay_files = 0
        if delete_replay:
            clear = getattr(self.replay_buffer, "clear_persisted_episodes", None)
            if callable(clear):
                removed_replay_files += int(clear())
            if self.use_demo_replay:
                clear_demo = getattr(
                    self.demo_replay_buffer, "clear_persisted_episodes", None
                )
                if callable(clear_demo):
                    removed_replay_files += int(clear_demo())

        removed_resume_files = 0
        if delete_resume:
            snapshot_dir = self.work_dir / "snapshots"
            for path in snapshot_dir.glob("*_snapshot.pkl"):
                path.unlink(missing_ok=True)
                removed_resume_files += 1
            # The optimizer sidecar is resume-only state and is meaningless
            # once its snapshots are gone.
            resume_state = snapshot_dir / "resume_state.pkl"
            if resume_state.is_file():
                resume_state.unlink(missing_ok=True)
                removed_resume_files += 1

        record = {
            "completed_env_steps": int(self.global_env_steps),
            "eval_checkpoint_count": len(eval_checkpoints),
            "removed_replay_files": removed_replay_files,
            "removed_resume_files": removed_resume_files,
            "shared_demo_cache": (
                str(self._shared_demo_cache.path)
                if self._shared_demo_cache is not None
                else None
            ),
        }
        (self.work_dir / "training_artifacts_finalized.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )
        logging.info("Finalized completed training artifacts: %s", record)

    @staticmethod
    def _sidecar_checkpoint_state(
        snapshot_path: Path,
        snapshot_step: int | None,
    ) -> dict | None:
        """Optimizer state for ``snapshot_path`` from the resume sidecar.

        Snapshot version 3 stores the optimizer tree once, beside the numbered
        snapshots, because only the newest snapshot is ever resumed from. An
        older or stripped snapshot has no matching sidecar; the caller then
        restarts the optimizer from scratch, which is exact for evaluation and
        approximate for training continuation.
        """

        sidecar = snapshot_path.resolve().parent / "resume_state.pkl"
        if snapshot_step is None or not sidecar.is_file():
            logging.warning(
                "No optimizer state for %s; the optimizer will be "
                "reinitialized. Evaluation is unaffected; an exact training "
                "resume is not possible from this snapshot.",
                snapshot_path,
            )
            return None
        try:
            with sidecar.open("rb") as handle:
                stored = pickle.load(handle)
        except (OSError, pickle.UnpicklingError) as exc:
            logging.warning("Unreadable resume sidecar %s: %r", sidecar, exc)
            return None
        if int(stored.get("snapshot_step", -1)) != int(snapshot_step):
            # The sidecar always tracks the newest snapshot, so this is the
            # normal outcome when loading an earlier checkpoint for evaluation.
            logging.info(
                "Resume sidecar is for step %s, not %s; reinitializing the "
                "optimizer.",
                stored.get("snapshot_step"),
                snapshot_step,
            )
            return None
        return stored.get("agent_checkpoint_state")

    def load_snapshot(
        self,
        path_to_snapshot_to_load=None,
        load_replay_buffer: bool = True,
    ):
        if path_to_snapshot_to_load is None:
            path_to_snapshot_to_load = (
                self.work_dir / "snapshots" / "latest_snapshot.pkl"
            )
        else:
            path_to_snapshot_to_load = Path(path_to_snapshot_to_load)
        if not path_to_snapshot_to_load.is_file():
            raise ValueError(
                f"Provided file '{str(path_to_snapshot_to_load)}' is not a snapshot."
            )
        with path_to_snapshot_to_load.open("rb") as f:
            payload = pickle.load(f)
        payload.pop("snapshot_version", None)
        payload.pop("eval_checkpoint_version", None)
        snapshot_step = payload.pop("snapshot_step", None)
        self.agent.load_state_dict(payload.pop("agent"))
        checkpoint_state = payload.pop("agent_checkpoint_state", None)
        if checkpoint_state is None:
            checkpoint_state = self._sidecar_checkpoint_state(
                path_to_snapshot_to_load,
                snapshot_step,
            )
        self.agent.load_checkpoint_state_dict(checkpoint_state or {})
        snapshot_cfg = payload.pop("cfg", None)
        replay_state = payload.pop("replay_buffer", None)
        if load_replay_buffer and replay_state is not None:
            self.replay_buffer.load_state_dict(replay_state)
        if self.use_demo_replay:
            demo_replay_state = payload.pop("demo_replay_buffer", None)
            if load_replay_buffer and demo_replay_state is not None:
                self.demo_replay_buffer.load_state_dict(demo_replay_state)
        timer_state = payload.pop("timer_state", None)
        rng_state = payload.pop("rng_state", None)
        payload.pop("wandb_run_id", None)
        for k, v in payload.items():
            self.__dict__[k] = v
        self._snapshot_cfg = snapshot_cfg
        if timer_state is not None:
            self._restore_timer_state(timer_state)
        if rng_state is not None:
            self._restore_rng_state(rng_state)
        self._close_replay_iter()
        self._snapshot_loaded = True
