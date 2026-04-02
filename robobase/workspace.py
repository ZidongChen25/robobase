import shutil
import signal
import sys
import time
import random
from typing import Callable, Any
from functools import partial
import logging

from gymnasium import spaces
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from robobase.factory import create_agent, method_name_from_cfg
from robobase.envs.env import EnvFactory
from robobase.logger import Logger
from robobase.replay_buffer.iterator import (
    create_jax_replay_iterator,
    normalize_replay_num_workers,
)
from robobase.replay_buffer.prioritized_replay_buffer import PrioritizedReplayBuffer
from robobase.replay_buffer.replay_buffer import ReplayBuffer
from robobase.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
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
        return {k: np.concatenate([batch[k], demo_batch[k]], axis=0) for k in batch.keys()}

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
    multiprocessing_context = "spawn"
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
        multiprocessing_context=multiprocessing_context,
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
        if cfg.num_eval_envs < 1:
            raise ValueError(f"num_eval_envs must be >= 1, got {cfg.num_eval_envs}.")
        if cfg.execution_length > cfg.action_sequence:
            raise ValueError(
                "execution_length must be <= action_sequence, got "
                f"{cfg.execution_length} and {cfg.action_sequence}."
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
        if (
            not cfg.is_imitation_learning
            and cfg.replay_size_before_train * cfg.action_repeat * cfg.action_sequence
            < cfg.env.episode_length
            and cfg.replay_size_before_train > 0
        ):
            raise ValueError(
                "replay_size_before_train * action_repeat "
                f"({cfg.replay_size_before_train} * {cfg.action_repeat}) "
                f"must be >= episode_length ({cfg.env.episode_length})."
            )

        if cfg.method.is_rl and cfg.action_sequence != 1:
            raise ValueError("Action sequence > 1 is not supported for RL methods")
        if cfg.method.is_rl and cfg.execution_length != 1:
            raise ValueError("execution_length > 1 is not supported for RL methods")
        if not cfg.method.is_rl and cfg.replay.nstep != 1:
            raise ValueError("replay.nstep != 1 is not supported for IL methods")

        self.cfg = cfg
        _set_seed_everywhere(cfg.seed)
        self.use_demo_replay = cfg.demo_batch_size is not None
        self._skip_demo_loading_from_dataset = False

        # create logger
        self.logger = Logger(self.work_dir, cfg=self.cfg)
        self.env_factory = env_factory

        if (num_demos := cfg.demos) != 0:
            # Collect demos or fetch saved demos before making environments
            # to consider demo-based action space (e.g., standardization)
            if self._should_collect_demos_from_dataset():
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
            raise NotImplementedError(
                "Intrinsic reward modules are not yet supported."
            )

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
        train_agent_count = self.train_envs.num_envs if self.train_envs is not None else 0
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
            * self.cfg.action_sequence
            + pretrain_steps
        )

    @property
    def global_env_steps(self):
        """Total number of environment steps taken."""
        return self._calculate_global_env_steps()

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            _replay_iter = create_jax_replay_iterator(
                self.replay_buffer,
                num_workers=self.replay_num_workers,
            )
            if self.use_demo_replay:
                _demo_replay_iter = create_jax_replay_iterator(
                    self.demo_replay_buffer,
                    num_workers=self.demo_replay_num_workers,
                )
                _replay_iter = _merge_replay_demo_iter(
                    _replay_iter, _demo_replay_iter
                )
            self._replay_iter = _replay_iter
        return self._replay_iter

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
        if self.use_demo_replay:
            return True
        if not bool(self.cfg.replay.reuse_saved):
            return True
        if self._requires_demo_stats():
            return True
        return not self._saved_replay_exists(demo_replay=False)

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

        self.shutdown()

    def eval(self) -> dict[str, Any]:
        return self._eval(eval_record_all_episode=True)

    def _extract_vector_env_info(
        self, infos: dict[str, Any], env_idx: int, prefer_final: bool = False
    ) -> dict[str, Any]:
        if prefer_final:
            final_infos = infos.get("final_info")
            final_info_mask = infos.get("_final_info")
            if final_infos is not None and final_info_mask is not None and final_info_mask[env_idx]:
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

    def _run_single_env_eval(self, eval_record_all_episode: bool = False) -> dict[str, Any]:
        step, episode, total_reward, successes = 0, 0, 0, 0
        eval_until_episode = _Until(self.cfg.num_eval_episodes)
        first_rollout = []
        metrics = {}
        while eval_until_episode(episode):
            observation, info = self.eval_env.reset()
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
                total_reward += reward
                step += 1
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
                "episode_reward": total_reward / episode,
                "episode_length": step * self.cfg.action_repeat / episode,
            }
        )
        if successes is not None:
            metrics["episode_success"] = successes / episode
        if self.cfg.log_eval_video and len(first_rollout) > 0:
            metrics["eval_rollout"] = dict(video=first_rollout, fps=4)
        return metrics

    def _run_vector_env_eval(self, eval_record_all_episode: bool = False) -> dict[str, Any]:
        del eval_record_all_episode
        observation, _ = self.eval_envs.reset()
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
            observation = next_observation
            metrics.update(env_metrics)
            episode_rewards += reward
            episode_lengths += 1

            if not video_complete:
                self.eval_video_recorder.record(self.eval_envs)

            done_mask = np.logical_or(termination, truncation)
            if not video_complete and done_mask[0]:
                video_complete = True

            if not np.any(done_mask):
                continue

            done_env_indices = np.flatnonzero(done_mask)
            self.agent.reset(
                self.main_loop_iterations,
                [self._eval_agent_indices[idx] for idx in done_env_indices],
            )
            for env_idx in done_env_indices:
                if len(completed_rewards) < self.cfg.num_eval_episodes:
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
                            completed_successes.append(int(np.asarray(success).astype(int).item()))
                episode_rewards[env_idx] = 0.0
                episode_lengths[env_idx] = 0

        if self.cfg.log_eval_video and len(self.eval_video_recorder.frames) > 0:
            self.eval_video_recorder.save(f"{self.global_env_steps}.mp4")
            metrics["eval_rollout"] = dict(
                video=np.array(self.eval_video_recorder.frames), fps=4
            )

        metrics.update(
            {
                "episode_reward": float(np.mean(completed_rewards)),
                "episode_length": float(np.mean(completed_lengths) * self.cfg.action_repeat),
            }
        )
        if success_supported and completed_successes:
            metrics["episode_success"] = float(np.mean(completed_successes))
        return metrics

    def _eval(self, eval_record_all_episode: bool = False) -> dict[str, Any]:
        # TODO: In future, this func could do with a further refactor
        self._ensure_eval_envs_created()
        if self.eval_env is None:
            raise ValueError("Evaluation requested but no evaluation environment is configured.")
        self.agent.set_eval_env_running(True)
        try:
            if self.eval_envs is not None:
                return self._run_vector_env_eval(
                    eval_record_all_episode=eval_record_all_episode
                )
            return self._run_single_env_eval(
                eval_record_all_episode=eval_record_all_episode
            )
        finally:
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
        agents_reset = []
        for i in range(self.train_envs.num_envs):
            # Add transitions to episode rollout
            self._episode_rollouts[i].append(
                (
                    actions[i],
                    list_of_obs_dicts[i],
                    rewards[i],
                    terminations[i],
                    truncations[i],
                    {k: infos[k][i] for k in infos.keys()},
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
                for act, obs, rew, term, trunc, info, next_info in ep:
                    # Only keep the last frames regardless of frame stacks because
                    # replay buffer always store single-step transitions
                    obs = {k: v[-1] for k, v in obs.items()}

                    # Strip out temporal dimension as action_sequence = 1
                    act = act[0]

                    if relabeling_as_demo:
                        info["demo"] = 1
                    else:
                        info["demo"] = 0

                    # Filter out unwanted keys in info
                    extra_replay_elements = {
                        k: v
                        for k, v in info.items()
                        if k in self.extra_replay_elements.keys()
                    }

                    self.replay_buffer.add(
                        obs, act, rew, term, trunc, **extra_replay_elements
                    )
                    if relabeling_as_demo:
                        self.demo_replay_buffer.add(
                            obs, act, rew, term, trunc, **extra_replay_elements
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
        if self.replay_buffer.reused_existing and not self.use_demo_replay:
            logging.info("Reusing saved replay episodes from disk. Skipping demo import.")
            return
        if (num_demos := self.cfg.demos) != 0:
            # NOTE: Currently we do not protect demos from being evicted from replay
            self.env_factory.load_demos_into_replay(
                self.cfg,
                self.replay_buffer,
                is_demo_buffer=True if self.cfg.is_imitation_learning else False,
            )
            if self.use_demo_replay:
                # Load demos to the dedicated demo_replay_buffer
                self.env_factory.load_demos_into_replay(
                    self.cfg, self.demo_replay_buffer, is_demo_buffer=True
                )

        if self.cfg.replay_size_before_train > 0:
            num_demos = _normalize_demos_count(num_demos)
            diff = self.cfg.replay_size_before_train - len(self.replay_buffer)
            if num_demos > 0 and diff > 0:
                logging.warning(
                    f"Collecting additional {diff} random samples even though there "
                    f"are {len(self.replay_buffer)} demo samples inside the buffer. "
                    "Please make sure that this is an intended behavior."
                )

    def _perform_updates(self) -> dict[str, Any]:
        if self.agent.logging:
            start_time = time.time()
        metrics = {}
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
        if self.cfg.num_pretrain_steps > 0:
            pre_train_until_step = _Until(self.cfg.num_pretrain_steps)
            should_pretrain_log = _Every(self.cfg.log_pretrain_every)
            should_pretrain_eval = _Every(self.cfg.eval_every_steps)
            snapshot_every_n = (
                self.cfg.snapshot_every_n if self.cfg.save_snapshot else 0
            )
            should_pretrain_save_snapshot = _Every(snapshot_every_n)
            snapshot_save_start_step = int(
                self.cfg.get("snapshot_save_start_step", 0)
            )
            if len(self.replay_buffer) <= 0:
                raise ValueError(
                    "there is no sample to pre-train with in the replay buffer "
                    f"but num_pretrain_steps ({self.cfg.num_pretrain_steps}) is > 0"
                )

            while pre_train_until_step(self.pretrain_steps):
                if self._shutting_down:
                    break
                self.agent.logging = False

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
        seed_until_size = _Until(self.cfg.replay_size_before_train)
        should_log = _Every(self.cfg.log_every)
        eval_every_n = self.cfg.eval_every_steps if self.eval_env is not None else 0
        should_eval = _Every(eval_every_n)
        snapshot_every_n = self.cfg.snapshot_every_n if self.cfg.save_snapshot else 0
        should_save_snapshot = _Every(snapshot_every_n)
        snapshot_save_start_step = int(self.cfg.get("snapshot_save_start_step", 0))
        observations, info = self.train_envs.reset()
        #  We use agent 0 to accumulate stats about how the training agents are doing
        agent_0_ep_len = agent_0_reward = 0
        agent_0_prev_ep_len = agent_0_prev_reward = None
        while train_until_frame(self.global_env_steps):
            metrics = {}
            self.agent.logging = False
            if should_log(self.main_loop_iterations):
                self.agent.logging = True
            if not seed_until_size(len(self.replay_buffer)):
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
        metrics = {
            "total_time": total_time,
            "iteration": main_loop_iterations,
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
        if self.eval_env is None:
            self.eval_env = self.env_factory.make_eval_env(self.cfg)
        if self.cfg.num_eval_envs > 1 and self.eval_envs is None:
            self.eval_envs = self.env_factory.make_eval_envs(self.cfg)

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
        payload["snapshot_version"] = 2
        payload["agent"] = self.agent.state_dict()
        payload["agent_checkpoint_state"] = self.agent.checkpoint_state_dict()
        payload["replay_buffer"] = self.replay_buffer.state_dict()
        payload["rng_state"] = self._rng_state_dict()
        payload["timer_state"] = self._timer_state_dict()
        if self.logger.wandb_run_id is not None:
            payload["wandb_run_id"] = self.logger.wandb_run_id
        if self.use_demo_replay:
            payload["demo_replay_buffer"] = self.demo_replay_buffer.state_dict()
        with snapshot.open("wb") as f:
            pickle.dump(payload, f)
        latest_snapshot = self.work_dir / "snapshots" / "latest_snapshot.pkl"
        shutil.copy(snapshot, latest_snapshot)

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
        self.agent.load_state_dict(payload.pop("agent"))
        self.agent.load_checkpoint_state_dict(payload.pop("agent_checkpoint_state", {}))
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
