from __future__ import annotations

from typing import Any, Optional, List
import copy
import collections
import heapq
import logging
import random
import math
import multiprocessing as mp

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from tqdm import tqdm
from omegaconf import DictConfig
from gymnasium.wrappers import TimeLimit, EnvCompatibility

from robobase.envs.wrappers import (
    OnehotTime,
    FrameStack,
    RescaleFromTanh,
    ActionSequence,
    RecedingHorizonControl,
    AppendDemoInfo,
)
from robobase.utils import add_demo_to_replay_buffer
from robobase.envs.env import EnvFactory, DemoEnv

try:
    import gym as gym_old
except ImportError:  # pragma: no cover - exercised when old D4RL deps are absent
    gym_old = None

try:
    import d4rl
except ImportError:  # pragma: no cover - exercised when old D4RL deps are absent
    d4rl = None

try:
    import minari
except ImportError:  # pragma: no cover - exercised when Minari deps are absent
    minari = None

SUPPORTED_OLD_D4RL_ENVS = ["ant", "antmaze", "halfcheetah", "hopper", "walker2d"]
MINARI_ADROIT_EXPERT_DATASETS = {
    "door": "D4RL/door/expert-v2",
    "hammer": "D4RL/hammer/expert-v2",
    "pen": "D4RL/pen/expert-v2",
    "relocate": "D4RL/relocate/expert-v2",
}
SUPPORTED_ENVS = SUPPORTED_OLD_D4RL_ENVS + sorted(MINARI_ADROIT_EXPERT_DATASETS)

Batch = collections.namedtuple(
    "Batch", ["observations", "actions", "rewards", "masks", "next_observations"]
)


def compute_returns(traj):
    episode_return = 0
    for _, _, rew, _, _, _ in traj:
        episode_return += rew

    return episode_return


def split_into_trajectories(
    observations, actions, rewards, masks, dones_float, next_observations
):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append(
            (
                observations[i],
                actions[i],
                rewards[i],
                masks[i],
                dones_float[i],
                next_observations[i],
            )
        )
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


def get_traj_dataset(env, sorting=True):
    dataset = D4RLDataset(env)
    trajs = split_into_trajectories(
        dataset.observations,
        dataset.actions,
        dataset.rewards,
        dataset.masks,
        dataset.dones_float,
        dataset.next_observations,
    )
    if sorting:
        trajs.sort(key=compute_returns, reverse=True)

    # Convert traj to RoboBase demo
    converted_trajs = [[]]
    for traj in trajs:
        # The first transition only contains (obs, info),
        # corresponding to the ouput of env.reset()
        converted_trajs[-1].append([traj[0][0], {"demo": 1}])

        # For the subsequent transitions. we convert
        # (obs, actions, rew, masks, dones_float, next_obs)
        # to (next_obs, rew, term, trunc, next_info) required by robobase.DemoEnv.
        for ts in traj:
            # truncation is always False as the time limit is handled by
            # the `TimeLimit` wrapper.
            converted_trajs[-1].append(
                [ts[5], ts[2], ts[4], False, {"demo_action": ts[1], "demo": 1}]
            )

        # If traj length equals to max_episode_len, then termination=False and
        # truncated=True
        # NOTE: For d4rl, the collected trajectory has 1 less step then
        # max_episode_steps.
        if len(traj) == env.spec.max_episode_steps - 1:
            converted_trajs[-1][-1][2] = False

        converted_trajs.append([])
    converted_trajs.pop()  # Remove the last empty traj

    # NOTE: this raw_dataset is not sorted
    return converted_trajs, dataset.raw_dataset


def _require_old_d4rl():
    if gym_old is None or d4rl is None:
        raise ImportError(
            "Old D4RL tasks require the optional d4rl dependencies. "
            "Install with `uv sync --extra d4rl` or install `gym` and `d4rl`."
        )


def _require_minari():
    if minari is None:
        raise ImportError(
            "Minari D4RL Adroit datasets require `minari`. Live Adroit "
            "environments also require `gymnasium-robotics`."
        )


def _env_cfg_get(cfg: DictConfig, name: str, default: Any = None):
    if not hasattr(cfg, "env"):
        return default
    try:
        return cfg.env.get(name, default)
    except AttributeError:
        return getattr(cfg.env, name, default)


def _normalize_num_demos(value):
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"inf", "+inf", ".inf", "+.inf"}:
            return math.inf
        return float(normalized)
    return value


def _is_old_gym_env(env: Any) -> bool:
    return gym_old is not None and isinstance(env, gym_old.Env)


def _is_minari_cfg(cfg: DictConfig) -> bool:
    dataset_format = _env_cfg_get(cfg, "dataset_format", None)
    if dataset_format is not None:
        return str(dataset_format).lower() == "minari"
    if _env_cfg_get(cfg, "dataset_id", None) is not None:
        return True
    task_name = str(cfg.env.task_name)
    return task_name.startswith("D4RL/") or task_name in MINARI_ADROIT_EXPERT_DATASETS


def _resolve_minari_dataset_id(cfg: DictConfig) -> str:
    dataset_id = _env_cfg_get(cfg, "dataset_id", None)
    task_name = str(cfg.env.task_name)
    if dataset_id is None:
        if task_name.startswith("D4RL/"):
            dataset_id = task_name
        elif task_name in MINARI_ADROIT_EXPERT_DATASETS:
            dataset_id = MINARI_ADROIT_EXPERT_DATASETS[task_name]
        else:
            task_key = task_name.split("-")[0]
            dataset_id = MINARI_ADROIT_EXPERT_DATASETS.get(task_key)

    if dataset_id not in MINARI_ADROIT_EXPERT_DATASETS.values():
        supported = ", ".join(MINARI_ADROIT_EXPERT_DATASETS.values())
        raise AssertionError(
            f"{dataset_id or task_name} is not a supported Minari Adroit expert "
            f"dataset. Supported: {supported}"
        )
    return str(dataset_id)


def _get_minari_d4rl_dataset(dataset_id: str):
    _require_minari()
    try:
        return minari.load_dataset(dataset_id)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load Minari dataset {dataset_id!r}. "
            f"Download it first with `minari download {dataset_id}`."
        ) from exc


def _get_minari_dataset_spaces(cfg: DictConfig):
    dataset = _get_minari_d4rl_dataset(_resolve_minari_dataset_id(cfg))
    return dataset.spec.observation_space, dataset.spec.action_space


def _episode_return(demo):
    return sum(float(step[1]) for step in demo[1:])


def _convert_minari_episode(episode):
    observations = np.asarray(episode.observations, dtype=np.float32)
    actions = np.asarray(episode.actions, dtype=np.float32)
    rewards = np.asarray(episode.rewards, dtype=np.float32)
    terminations = np.asarray(episode.terminations, dtype=bool)
    truncations = np.asarray(episode.truncations, dtype=bool)

    if len(observations) != len(actions) + 1:
        raise ValueError(
            "Expected Minari episode observations to include reset and final "
            f"observation ({len(actions) + 1}), got {len(observations)}."
        )

    demo = [[observations[0], {"demo": 1}]]
    for i, action in enumerate(actions):
        demo.append(
            [
                observations[i + 1],
                rewards[i],
                bool(terminations[i]),
                bool(truncations[i]),
                {"demo_action": action, "demo": 1},
            ]
        )
    return demo


def get_minari_traj_dataset(
    dataset_id: str,
    sorting: bool = True,
    *,
    num_demos: int | None = None,
    random_selection: bool = False,
    selection: str = "top_return",
    num_transitions: int | None = None,
):
    dataset = _get_minari_d4rl_dataset(dataset_id)
    limit = None if num_demos is None else max(0, int(num_demos))
    if random_selection:
        # Backwards compatibility with the original boolean config.
        selection = "random"
    selection = str(selection).strip().lower()
    if selection not in {"top_return", "first", "random"}:
        raise ValueError(
            "Minari demo selection must be one of "
            f"'top_return', 'first', or 'random'; got {selection!r}."
        )

    transition_limit = (
        None if num_transitions is None else max(0, int(num_transitions))
    )
    if transition_limit is not None and selection != "first":
        raise ValueError(
            "num_transitions is only defined for selection='first', so the "
            "selected data has an unambiguous original dataset order."
        )

    converted_trajs = []
    top_demos = []
    selected_transitions = 0
    if limit == 0 or transition_limit == 0:
        return converted_trajs, dataset

    for episode_index, episode in enumerate(tqdm(dataset.iterate_episodes())):
        if selection == "first":
            if limit is not None and len(converted_trajs) >= limit:
                break

            converted = _convert_minari_episode(episode)
            episode_transitions = len(converted) - 1
            if transition_limit is not None:
                remaining = transition_limit - selected_transitions
                if remaining <= 0:
                    break
                if episode_transitions > remaining:
                    # A RoboBase demo has one reset item followed by one item per
                    # transition. Mark an artificial cut as a truncation so it is
                    # still a valid episode boundary for replay ingestion.
                    converted = converted[: remaining + 1]
                    converted[-1][2] = False
                    converted[-1][3] = True
                    episode_transitions = remaining

            converted_trajs.append(converted)
            selected_transitions += episode_transitions
            if (
                transition_limit is not None
                and selected_transitions == transition_limit
            ):
                break
            continue

        if limit is None:
            converted_trajs.append(_convert_minari_episode(episode))
            continue
        if selection == "random":
            if episode_index < limit:
                converted_trajs.append(_convert_minari_episode(episode))
                continue
            replacement_index = random.randrange(episode_index + 1)
            if replacement_index < limit:
                converted_trajs[replacement_index] = _convert_minari_episode(episode)
            continue

        episode_return = float(np.asarray(episode.rewards, dtype=np.float32).sum())
        candidate_key = (episode_return, episode_index)
        if len(top_demos) >= limit and candidate_key <= top_demos[0][:2]:
            continue
        candidate = (*candidate_key, _convert_minari_episode(episode))
        if len(top_demos) < limit:
            heapq.heappush(top_demos, candidate)
        else:
            heapq.heapreplace(top_demos, candidate)

    if transition_limit is not None and selected_transitions != transition_limit:
        raise ValueError(
            f"Requested the first {transition_limit} Minari transitions, but "
            f"the dataset only provided {selected_transitions}."
        )

    if top_demos:
        converted_trajs = [item[2] for item in sorted(top_demos, reverse=True)]
    elif sorting and selection == "top_return":
        converted_trajs.sort(key=_episode_return, reverse=True)
    return converted_trajs, dataset


class D4RLDataset:
    def __init__(self, env: gym_old.Env, clip_to_eps: bool = True, eps: float = 1e-5):
        _require_old_d4rl()
        logging.warning("Collecting dataset from d4rl")
        self.raw_dataset = dataset = d4rl.qlearning_dataset(env.env)

        # Clip actions.
        # NOTE: sometimes action could be 1 which is not reachable for a tanh policy.
        if clip_to_eps:
            lim = 1 - eps
            dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

        # Fix d4rl termination state
        # NOTE: Due to dataset bugs, we manually add termination flag if next
        # observation is far away from current.
        dones_float = np.zeros_like(dataset["rewards"])
        for i in range(len(dones_float) - 1):
            if (
                np.linalg.norm(
                    dataset["observations"][i + 1] - dataset["next_observations"][i]
                )
                > 1e-6
                or dataset["terminals"][i] == 1.0
            ):
                dones_float[i] = 1
            else:
                dones_float[i] = 0
        dones_float[-1] = 1

        self.observations = dataset["observations"].astype(np.float32)
        self.actions = dataset["actions"].astype(np.float32)
        self.rewards = dataset["rewards"].astype(np.float32)
        self.masks = 1.0 - dataset["terminals"].astype(np.float32)
        self.dones_float = dones_float.astype(np.float32)
        self.next_observations = dataset["next_observations"].astype(np.float32)
        self.size = len(dataset["observations"])


class ConvertObsToDict(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """
    A wrapper to warp raw observation to space.Dict with key "low_dim_state".
    """

    def __init__(self, env: gym.Env):
        """Init.

        Args:
            env (gym.Env): the environment to apply wrapper on.
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)
        self.env = env

        assert isinstance(self.env.observation_space, spaces.Box)
        obs_space = self.env.observation_space
        self.observation_space = spaces.Dict(
            {
                "low_dim_state": spaces.Box(
                    low=np.asarray(obs_space.low, dtype=np.float32),
                    high=np.asarray(obs_space.high, dtype=np.float32),
                    shape=obs_space.shape,
                    dtype=np.float32,
                )
            }
        )

    def step(self, action):
        """Steps through the environment, incrementing the time step.

        Args:
            action: the action to take

        Returns:
            The environment's step using the action, with observation wrapped in a
            Dict with key "low_dim_state".
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._convert_obs(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment setting the time to zero.

        Args:
            **kwargs: Kwargs to apply to env.reset()

        Returns:
            The reset environment,  with observation wrapped in a Dict with key
            "low_dim_state".
        """
        obs, info = self.env.reset(**kwargs)
        return self._convert_obs(obs), info

    def _convert_obs(self, obs):
        return {"low_dim_state": obs.astype(np.float32)}


def _extract_success(info: dict) -> bool | None:
    if "task_success" in info:
        success = info["task_success"]
    elif "success" in info:
        success = info["success"]
    else:
        return None

    if isinstance(success, dict):
        return bool(success.get("task", any(success.values())))
    return bool(np.asarray(success).astype(bool).item())


class AddTaskSuccessInfo(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Normalize Adroit/Gymnasium Robotics success info for workspace eval."""

    def __init__(self, env: gym.Env):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        success = _extract_success(info)
        if success is not None:
            info["task_success"] = int(success)
        return obs, reward, terminated, truncated, info


class D4RLPlaceholderEnv(gym.Env):
    def __init__(self, observation_space: spaces.Box, action_space: spaces.Box):
        self.observation_space = copy.deepcopy(observation_space)
        self.action_space = copy.deepcopy(action_space)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        raise RuntimeError(
            "This placeholder only exposes dataset spaces. Enable the live "
            "D4RL/Adroit environment for online rollouts or evaluation."
        )


class D4RLEnvCompatibility(EnvCompatibility):
    """
    D4RL uses old gym environments. This Wrapper updates them to new
    gymnaisum environments with the updated API syntax.
    """

    def __init__(self, old_env: gym_old.Env, render_mode: Optional[str] = None):
        """Init.

        Args:
            old_env (gym_old.Env): an environment written with old_gym format.
            render_mode (Optional[str], optional): render mode. Defaults to None.
        """
        _require_old_d4rl()
        super().__init__(old_env, render_mode)

        # Assert the observation for old_env is box.
        assert isinstance(self.env.action_space, gym_old.spaces.Box)
        assert isinstance(self.env.observation_space, gym_old.spaces.Box)

        # Transform observation and action space from gym_old.Space into gymnasium.Space
        # Also force dtype to float32.
        self.observation_space = spaces.Box(
            low=self.env.observation_space.low,
            high=self.env.observation_space.high,
            shape=self.env.observation_space.shape,
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=self.env.action_space.low,
            high=self.env.action_space.high,
            shape=self.env.action_space.shape,
            dtype=np.float32,
        )

    def get_normalized_score(self, score: float) -> float:
        """Get episode returns normalized against expert demos

        Args:
            score (float): episode returns

        Returns:
            (float): normalized episode returns
        """
        return self.env.get_normalized_score(score)


def _get_demo_fn(cfg: DictConfig, num_demos: int, demo_list: List):
    env = _make_env(cfg)
    d4rl_trajs, _ = get_traj_dataset(env)
    demo_list.extend(d4rl_trajs)
    env.close()


def _make_env(cfg: DictConfig) -> gym_old.Env:
    if _is_minari_cfg(cfg):
        dataset_id = _resolve_minari_dataset_id(cfg)
        dataset = _get_minari_d4rl_dataset(dataset_id)
        return dataset.recover_environment()

    _require_old_d4rl()
    task_name = cfg.env.task_name

    # check task_name is supported
    task_env_name = task_name.split("-")[0]
    assert task_env_name in SUPPORTED_OLD_D4RL_ENVS, f"{task_name} is not supported!"

    return gym_old.make(task_name)


class D4RLEnvFactory(EnvFactory):
    def _wrap_env(
        self,
        env: gym_old.Env | gym.Env,
        cfg: DictConfig,
        return_raw_spaces=False,
        demo_env=False,
    ):
        if return_raw_spaces:
            action_space = copy.deepcopy(env.action_space)
            observation_space = copy.deepcopy(env.observation_space)
        # sanity check
        assert not cfg.pixels, "D4RL is state-only environment"
        if _is_old_gym_env(env):
            # NOTE: For d4rl, the collected trajectory has 1 less step then
            # max_episode_steps.
            assert (
                cfg.env.episode_length == env.spec.max_episode_steps - 1
            ), "For D4RL, episode_length must be the same as the collected demo length."

        if _is_old_gym_env(env):
            env = D4RLEnvCompatibility(env)

        env = ConvertObsToDict(env)
        env = AddTaskSuccessInfo(env)
        if cfg.use_standardization:
            raise NotImplementedError("Not implemented and tested for D4RL")
        elif cfg.use_min_max_normalization:
            raise NotImplementedError("Not implemented and tested for D4RL")
        else:
            rescale_from_tanh_cls = RescaleFromTanh

        env = rescale_from_tanh_cls(env)
        env = TimeLimit(env, cfg.env.episode_length)
        if cfg.use_onehot_time_and_no_bootstrap:
            env = OnehotTime(env, cfg.env.episode_length)
        if not demo_env:
            env = FrameStack(env, cfg.frame_stack)
            execution_length = int(cfg.get("execution_length", cfg.action_sequence))
            if int(cfg.action_sequence) == execution_length:
                env = ActionSequence(env, cfg.action_sequence)
            else:
                env = RecedingHorizonControl(
                    env,
                    cfg.action_sequence,
                    cfg.env.episode_length,
                    execution_length,
                    bool(cfg.get("temporal_ensemble", True)),
                    float(cfg.get("temporal_ensemble_gain", 0.01)),
                    int(cfg.get("action_execution_start", 0)),
                )
        env = AppendDemoInfo(env)

        if return_raw_spaces:
            return env, (action_space, observation_space)
        else:
            return env

    def make_train_env(self, cfg: DictConfig) -> gym.vector.VectorEnv:
        """See base class for documentation."""
        return gym.vector.AsyncVectorEnv(
            [
                lambda: self._wrap_env(_make_env(cfg), cfg)
                for _ in range(cfg.num_train_envs)
            ]
        )

    def make_eval_env(self, cfg: DictConfig) -> gym.Env:
        """See base class for documentation."""
        # NOTE: Assumes workspace always creates eval_env in the main thread
        env, (self._action_space, self._observation_space) = self._wrap_env(
            _make_env(cfg), cfg, return_raw_spaces=True
        )
        return env

    def get_spaces(self, cfg: DictConfig) -> tuple[gym.Space, gym.Space]:
        if _is_minari_cfg(cfg):
            observation_space, action_space = _get_minari_dataset_spaces(cfg)
            env = self._wrap_env(
                D4RLPlaceholderEnv(observation_space, action_space),
                cfg,
            )
            self._action_space = copy.deepcopy(action_space)
            self._observation_space = copy.deepcopy(observation_space)
            return env.observation_space, env.action_space
        return super().get_spaces(cfg)

    def collect_or_fetch_demos(self, cfg: DictConfig, num_demos: int):
        """See base class for documentation."""
        num_demos = _normalize_num_demos(num_demos)
        if num_demos == 0:
            self._raw_demos = []
            return

        if _is_minari_cfg(cfg):
            finite_num_demos = int(num_demos) if math.isfinite(num_demos) else None
            random_selection = bool(_env_cfg_get(cfg, "random_traj", False))
            selection = _env_cfg_get(cfg, "demo_selection", None)
            if selection is None:
                selection = "random" if random_selection else "top_return"
            self._raw_demos, _ = get_minari_traj_dataset(
                _resolve_minari_dataset_id(cfg),
                num_demos=finite_num_demos,
                random_selection=random_selection,
                selection=str(selection),
                num_transitions=_env_cfg_get(cfg, "num_transitions", None),
            )
            return
        else:
            # collect all demos
            manager = mp.Manager()
            mp_list = manager.list()
            p = mp.Process(
                target=_get_demo_fn,
                args=(
                    cfg,
                    num_demos,
                    mp_list,
                ),
            )
            p.start()
            p.join()

            # Only extract num_demos from the full dataset
            all_demos = list(mp_list)
        if not math.isfinite(num_demos):
            num_demos = len(all_demos)

        if cfg.env.random_traj:
            self._raw_demos = random.sample(all_demos, num_demos)
        else:
            self._raw_demos = all_demos[:num_demos]

    def post_collect_or_fetch_demos(self, cfg: DictConfig):
        self._demos = self._raw_demos

    def load_demos_into_replay(
        self, cfg: DictConfig, buffer, is_demo_buffer: bool = False
    ):
        """See base class for documentation."""
        assert hasattr(self, "_demos"), (
            "There's no _demo attribute inside the factory, "
            "Check `collect_or_fetch_demos` is called before calling this method."
        )
        demo_env = self._wrap_env(
            DemoEnv(
                copy.deepcopy(self._demos), self._action_space, self._observation_space
            ),
            cfg,
            demo_env=True,
        )
        for _ in range(len(self._demos)):
            add_demo_to_replay_buffer(demo_env, buffer)
