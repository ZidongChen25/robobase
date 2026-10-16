"""Throughput-oriented Workspace for RLPD (opt in via ``train_rlpd.py``).

Reuses :class:`robobase.workspace.Workspace` for config validation, the
environment/demo factories, logging and output directories.  Differences:

1. Offline data.  Demos pass through the same ``EnvFactory`` wrappers as for
   every other method (``load_demos_into_replay``) into a flat transition
   recorder; the packed matrix then lives on the GPU (see ``robobase.method.rlpd``).
2. Online replay.  A fixed-capacity GPU ring buffer updated inside RLPD's
   compiled step; RoboBase's episode replay is not used.
3. Loop order.  ``act -> dispatch(insert previous transition + update) ->
   env.step``: the simulator step runs while the GPU executes the update, so
   the device never waits on Python/MuJoCo.  Every update still sees every
   transition collected so far, exactly as in the reference loop; the price is
   that the action at step t comes from parameters one update older (the
   update for transition t-1 is still in flight).  ``method.overlap_env_step:
   false`` restores the strict reference order act -> env.step -> insert ->
   update.
4. Evaluation.  Deterministic ``tanh(mean)`` episodes run in CPU worker
   processes on a NumPy copy of the actor while training continues; results
   are logged at the step whose parameters they evaluated.  Every evaluation
   uses the same per-episode reset seeds (``env.eval_seed_start``/``eval_seeds``).
"""

from __future__ import annotations

import atexit
import concurrent.futures as futures
import contextlib
import gc
import json
import logging
import multiprocessing as mp
import os
import shutil
import tempfile
import time
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from robobase.method.rlpd import RLPD, numpy_actor_mode, pack_transitions
from robobase.workspace import Workspace

log = logging.getLogger(__name__)


def use_private_tempdir() -> str:
    """Give this process its own temp dir (inherited by forked env workers).

    gymnasium-robotics maze envs write their generated MJCF to a fixed
    ``<tempdir>/<robot>_maze.xml``; processes building envs concurrently
    (parallel runs, eval workers) otherwise race on that file and MuJoCo
    reads it half-written ("mjParseXML: empty file").
    """

    path = tempfile.mkdtemp(prefix=f"robobase_{os.getpid()}_")
    os.environ["TMPDIR"] = path
    tempfile.tempdir = path
    atexit.register(shutil.rmtree, path, ignore_errors=True)
    return path


def _flat_low_dim(observation: dict) -> np.ndarray:
    return np.asarray(observation["low_dim_state"], np.float32).reshape(-1)


def legacy_d4rl_rows(cfg: DictConfig) -> np.ndarray | None:
    """Offline rows for gym-era D4RL tasks, built like RLPD's ``D4RLDataset``.

    ``d4rl.qlearning_dataset`` with actions clipped to +-(1 - 1e-5) and
    ``mask = 1 - terminals`` (timeouts bootstrap).  RoboBase's demo
    conversion instead treats every trajectory boundary as terminal, so it is
    bypassed here.  Returns None for every other environment.
    """

    if cfg.env.env_name != "d4rl":
        return None
    from robobase.envs import d4rl as d4rl_env

    if d4rl_env._is_minari_cfg(cfg):
        return None
    env = d4rl_env._make_env(cfg)
    try:
        data = d4rl_env.D4RLDataset(env)
    finally:
        env.close()
    return pack_transitions(
        data.observations,
        data.actions,
        data.rewards,
        data.masks,
        data.next_observations,
    )


class FlatTransitionRecorder:
    """Duck-typed replay sink for ``EnvFactory.load_demos_into_replay``.

    Collects the wrapped demo transitions as packed RLPD rows
    ``[obs | action | reward | 1 - terminal | next_obs]``.
    """

    sequential = False

    def __init__(self):
        self.extra_replay_elements = {}
        self._gamma = 1.0
        self._episode: list[tuple] = []
        self._chunks: list[np.ndarray] = []
        self.num_episodes = 0

    def add(self, observation, action, reward, terminal, truncated, **kwargs):
        del truncated, kwargs
        self._episode.append(
            (
                _flat_low_dim(observation),
                np.asarray(action, np.float32).reshape(-1),
                float(reward),
                bool(terminal),
            )
        )

    def add_final(self, final_observation):
        if not self._episode:
            raise ValueError("add_final() called on an empty episode.")
        obs = [step[0] for step in self._episode] + [_flat_low_dim(final_observation)]
        self._chunks.append(
            pack_transitions(
                np.stack(obs[:-1]),
                np.stack([step[1] for step in self._episode]),
                np.asarray([step[2] for step in self._episode]),
                1.0 - np.asarray([step[3] for step in self._episode], np.float32),
                np.stack(obs[1:]),
            )
        )
        self._episode = []
        self.num_episodes += 1

    def rows(self) -> np.ndarray:
        if not self._chunks:
            raise ValueError("No demo transitions were recorded.")
        return np.concatenate(self._chunks, axis=0)


# ----------------------------------------------------------------------------
# Out-of-process evaluation
# ----------------------------------------------------------------------------

_EVAL_ENV = None


def _eval_worker_init(cfg_container: dict) -> None:
    use_private_tempdir()
    os.nice(10)  # evaluation is background work; the learner keeps priority
    global _EVAL_ENV
    from robobase.workspace import _create_default_envs

    cfg = OmegaConf.create(cfg_container)
    _EVAL_ENV = _create_default_envs(cfg).make_eval_env(cfg)


@contextlib.contextmanager
def _environment(**overrides):
    saved = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# Environment of the spawned eval workers.  Importing robobase.factory runs
# JAX ops at module import, which would otherwise open a CUDA context (~0.5 GB
# of GPU memory) and ~200 XLA/OpenBLAS threads in every worker; spinning
# OpenBLAS pools in 4 runs x 10 workers drove the load average past 2000 and
# starved the learners (1 s GPU-utilisation gaps).
_EVAL_WORKER_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "JAX_PLATFORMS": "cpu",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
}


def _eval_worker_run(actor_params, low, high, seeds) -> list[tuple]:
    env = _EVAL_ENV
    results = []
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        episode_return, length = 0.0, 0
        while True:
            unit = numpy_actor_mode(actor_params, obs["low_dim_state"].reshape(1, -1))
            action = low + 0.5 * (unit[0] + 1.0) * (high - low)
            obs, reward, terminated, truncated, info = env.step(
                action.reshape(env.action_space.shape).astype(np.float32)
            )
            episode_return += float(reward)
            length += 1
            if terminated or truncated:
                break
        success = info.get("task_success")
        results.append(
            (seed, episode_return, length, None if success is None else float(success))
        )
    return results


class _AsyncEvaluator:
    def __init__(self, cfg: DictConfig, num_workers: int):
        container = OmegaConf.to_container(cfg, resolve=True)
        self._pool = futures.ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_eval_worker_init,
            initargs=(container,),
        )
        # Spawned children copy os.environ at start, so start every worker
        # now, inside the CPU-only environment; this process is unaffected
        # (its backends and thread pools already exist).
        with _environment(**_EVAL_WORKER_ENV):
            list(self._pool.map(time.sleep, [0.5] * num_workers))
        self._num_workers = num_workers
        self._pending: list[tuple[int, float, list[futures.Future]]] = []

    def submit(self, step: int, actor_params, low, high, seeds: list) -> None:
        chunks = [seeds[i :: self._num_workers] for i in range(self._num_workers)]
        jobs = [
            self._pool.submit(_eval_worker_run, actor_params, low, high, chunk)
            for chunk in chunks
            if chunk
        ]
        self._pending.append((step, time.perf_counter(), jobs))

    def completed(self, wait: bool = False) -> list[tuple[int, float, list]]:
        done = []
        while self._pending:
            step, start, jobs = self._pending[0]
            if not wait and not all(job.done() for job in jobs):
                break
            results = [item for job in jobs for item in job.result()]
            done.append((step, time.perf_counter() - start, results))
            self._pending.pop(0)
        return done

    def shutdown(self):
        self._pool.shutdown(wait=True, cancel_futures=True)


# ----------------------------------------------------------------------------
# Workspace
# ----------------------------------------------------------------------------


def _validate_rlpd_cfg(cfg: DictConfig) -> None:
    if str(cfg.method.get("name", "")).lower() != "rlpd":
        raise ValueError("RLPDWorkspace requires method=rlpd.")
    problems = []
    if int(cfg.num_train_envs) != 1:
        problems.append("num_train_envs must be 1")
    if int(cfg.action_sequence) != 1 or int(cfg.execution_length) != 1:
        problems.append("action_sequence/execution_length must be 1")
    if int(cfg.frame_stack) != 1:
        problems.append("frame_stack must be 1")
    if bool(cfg.pixels):
        problems.append("pixels must be false (state-based RLPD)")
    if int(cfg.action_repeat) != 1:
        problems.append("action_repeat must be 1")
    if cfg.get("demo_batch_size", None) is not None:
        problems.append("demo_batch_size must be null; use method.offline_batch_size")
    if problems:
        raise ValueError("Unsupported RLPD config: " + "; ".join(problems))


class RLPDWorkspace(Workspace):
    def __init__(self, cfg: DictConfig, work_dir: str | None = None):
        _validate_rlpd_cfg(cfg)
        use_private_tempdir()
        super().__init__(cfg, work_dir=work_dir)
        if not isinstance(self.agent, RLPD):
            raise TypeError("RLPDWorkspace requires an RLPD agent.")
        self._overlap = bool(cfg.method.get("overlap_env_step", True))
        self._evaluator = None
        self._eval_workers = int(cfg.method.get("eval_workers", 10))
        ref = cfg.method.get("eval_ref_scores", None)
        self._ref_scores = None if ref is None else (float(ref[0]), float(ref[1]))
        self._unit_low = np.asarray(self.agent.action_low)
        self._unit_high = np.asarray(self.agent.action_high)
        self._profile = {"act": 0.0, "dispatch": 0.0, "env": 0.0, "other": 0.0}

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _load_demos(self):
        agent: RLPD = self.agent
        rows = None
        if agent.spec.offline_batch_size > 0:
            start = time.perf_counter()
            rows = legacy_d4rl_rows(self.cfg)
            episodes = None
            if rows is None:
                if self.cfg.demos == 0:
                    raise ValueError("RLPD offline_batch_size > 0 requires demos != 0.")
                recorder = FlatTransitionRecorder()
                self.env_factory.load_demos_into_replay(self.cfg, recorder)
                rows, episodes = recorder.rows(), recorder.num_episodes
                for attr in ("_raw_demos", "_demos"):
                    if hasattr(self.env_factory, attr):
                        setattr(self.env_factory, attr, [])
                gc.collect()
            r0 = agent.obs_dim + agent.action_dim
            info = {
                "episodes": episodes,
                "transitions": int(rows.shape[0]),
                "terminals": int((rows[:, r0 + 1] == 0).sum()),
                "reward_sum": float(rows[:, r0].sum()),
                "load_seconds": time.perf_counter() - start,
            }
            log.info("RLPD offline dataset: %s", info)
            (self.work_dir / "offline_dataset.json").write_text(
                json.dumps(info, indent=2)
            )
        agent.attach_device_replay(rows)

    def _pretrain_on_demos(self):
        steps = int(self.cfg.num_pretrain_steps or 0)
        for _ in range(steps):
            self.agent.pretrain_step()
            self._pretrain_step += 1
            if self._pretrain_step % int(self.cfg.log_pretrain_every) == 0:
                metrics = self.agent.last_metrics()
                metrics.update(self._get_common_metrics())
                self.logger.log_metrics(metrics, self.pretrain_steps, prefix="pretrain")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _eval_seeds(self) -> list:
        return [
            self._eval_seed_for_episode(episode)
            for episode in range(int(self.cfg.num_eval_episodes))
        ]

    def _submit_eval(self, step: int) -> None:
        if self._evaluator is None:
            self._evaluator = _AsyncEvaluator(self.cfg, self._eval_workers)
        self._evaluator.submit(
            step,
            self.agent.actor_params_numpy(),
            self._unit_low,
            self._unit_high,
            self._eval_seeds(),
        )

    def _log_completed_evals(self, wait: bool = False) -> None:
        if self._evaluator is None:
            return
        for step, seconds, results in self._evaluator.completed(wait=wait):
            returns = np.asarray([r[1] for r in results], np.float64)
            metrics = {
                "episode_reward": float(returns.mean()),
                "episode_reward_std": float(returns.std()),
                "episode_length": float(np.mean([r[2] for r in results])),
                "num_episodes": len(results),
                "eval_wall_seconds": seconds,
            }
            successes = [r[3] for r in results]
            if all(s is not None for s in successes):
                metrics["episode_success"] = float(np.mean(successes))
            if self._ref_scores is not None:
                low, high = self._ref_scores
                metrics["normalized_score"] = (
                    100.0 * (metrics["episode_reward"] - low) / (high - low)
                )
            metrics.update(
                {
                    "iteration": step,
                    "env_steps": step,
                    "env_episodes": self.global_env_episodes,
                    "total_time": time.perf_counter() - self._loop_start,
                }
            )
            self.logger.log_metrics(metrics, step, prefix="eval")

    # ------------------------------------------------------------------
    # Online loop
    # ------------------------------------------------------------------

    def _online_rl(self):
        cfg, agent, env = self.cfg, self.agent, self.train_envs
        if env is None or cfg.num_train_frames <= 0:
            return
        total = int(cfg.num_train_frames)
        start_training = int(agent.num_explore_steps)
        eval_every = int(cfg.eval_every_steps) if cfg.num_eval_episodes > 0 else 0
        log_every = int(cfg.log_every)
        perf = time.perf_counter
        prof = self._profile

        observations, _ = env.reset(seed=int(cfg.seed))
        obs = _flat_low_dim({"low_dim_state": observations["low_dim_state"][0]})
        pending = None  # (row, step) collected but not yet handed to the agent
        episode_return, episode_length = 0.0, 0
        last_episode = (float("nan"), float("nan"), float("nan"))
        self._loop_start = window_start = perf()
        window_steps = 0

        for step in range(self._main_loop_iterations, total):
            t0 = perf()
            action = agent.act(
                {"low_dim_state": obs[None, None]}, step, eval_mode=False
            )
            t1 = perf()
            prof["act"] += t1 - t0
            if eval_every and step % eval_every == 0:
                self._submit_eval(step)
            if self._overlap and pending is not None:
                agent.observe(pending[0], update=pending[1] >= start_training)
            t2 = perf()
            prof["dispatch"] += t2 - t1
            next_observations, rewards, terminations, truncations, infos = env.step(
                action
            )
            t3 = perf()
            prof["env"] += t3 - t2

            terminated, truncated = bool(terminations[0]), bool(truncations[0])
            next_obs = _flat_low_dim(
                {"low_dim_state": next_observations["low_dim_state"][0]}
            )
            if terminated or truncated:
                final = infos["final_observation"][0]
                real_next_obs = _flat_low_dim(final)
            else:
                real_next_obs = next_obs
            reward = float(rewards[0])
            row = pack_transitions(
                obs[None],
                action.reshape(1, -1),
                [reward],
                [0.0 if terminated else 1.0],
                real_next_obs[None],
            )[0]
            if self._overlap:
                pending = (row, step)
            else:
                agent.observe(row, update=step >= start_training)
            obs = next_obs
            episode_return += reward
            episode_length += 1
            if terminated or truncated:
                success = None
                final_info = infos.get("final_info", [None])[0]
                if isinstance(final_info, dict) and "task_success" in final_info:
                    success = float(final_info["task_success"])
                last_episode = (episode_return, episode_length, success)
                self._global_env_episode += 1
                episode_return, episode_length = 0.0, 0
            self._main_loop_iterations += 1
            window_steps += 1
            prof["other"] += perf() - t3

            if (step + 1) % log_every == 0:
                now = perf()
                metrics = agent.last_metrics()
                metrics.update(
                    {
                        "episode_reward": last_episode[0],
                        "episode_length": last_episode[1],
                        "env_steps_per_second": window_steps / (now - window_start),
                        "iteration": step + 1,
                        "env_steps": step + 1,
                        "env_episodes": self.global_env_episodes,
                        "buffer_size": agent._online_count,
                        "total_time": now - self._loop_start,
                    }
                )
                if last_episode[2] is not None:
                    metrics["episode_success"] = last_episode[2]
                for key, value in prof.items():
                    metrics[f"profile_ms/{key}"] = 1e3 * value / window_steps
                    prof[key] = 0.0
                self.logger.log_metrics(metrics, step + 1, prefix="train")
                self._log_completed_evals()
                window_start, window_steps = perf(), 0
            if self._shutting_down:
                break

        if pending is not None:
            agent.observe(pending[0], update=pending[1] >= start_training)
        if eval_every and not self._shutting_down:
            self._submit_eval(self._main_loop_iterations)
        self._log_completed_evals(wait=True)

    def save_snapshot(self, counter_increments: dict[str, int] | None = None):
        # Params + optimizer + online replay; RoboBase's episode replay is unused.
        del counter_increments
        import pickle

        path = self.work_dir / "snapshots" / f"{self.global_env_steps}_rlpd.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "cfg": OmegaConf.to_container(self.cfg, resolve=False),
            "main_loop_iterations": self._main_loop_iterations,
            "agent": self.agent.state_dict(),
            "agent_checkpoint_state": self.agent.checkpoint_state_dict(),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def _finalize_completed_training_artifacts(self) -> None:
        return

    def shutdown(self):
        if self._evaluator is not None:
            self._evaluator.shutdown()
            self._evaluator = None
        super().shutdown()
