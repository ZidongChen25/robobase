import copy
import json
import logging
import math
import os
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, open_dict

from robobase.envs.env import Demo, DemoEnv, EnvFactory
from robobase.envs.wrappers import (
    ActionSequence,
    AppendDemoInfo,
    ConcatDim,
    FrameStack,
    OnehotTime,
    RescaleFromTanh,
)
from robobase.utils import add_demo_to_replay_buffer

try:
    import h5py
except ImportError:
    h5py = None


def _require_h5py():
    if h5py is None:
        raise ImportError(
            "robomimic support requires h5py. Install the optional dependency first."
        )


def _resolve_dataset_path(dataset_path: str) -> Path:
    path = Path(os.path.expanduser(dataset_path))
    if path.is_absolute():
        return path
    try:
        base_dir = Path(get_original_cwd())
    except ValueError:
        base_dir = Path.cwd()
    return (base_dir / path).resolve()


def _sorted_demo_keys(keys: list[str]) -> list[str]:
    return sorted(keys, key=lambda key: int(key.split("_")[-1]))


def _get_demo_keys(dataset_file, filter_key: str | None) -> list[str]:
    if filter_key and filter_key != "all":
        mask_key = f"mask/{filter_key}"
        if mask_key not in dataset_file:
            raise KeyError(
                f"Filter key '{filter_key}' is not present in the robomimic dataset."
            )
        return _sorted_demo_keys(
            [key.decode("utf-8") for key in np.asarray(dataset_file[mask_key])]
        )
    return _sorted_demo_keys(list(dataset_file["data"].keys()))


def _normalise_rgb_key(key: str) -> str:
    if key.endswith("_image"):
        return f"{key[:-6]}_rgb"
    if "image" in key and "rgb" not in key:
        return key.replace("image", "rgb")
    return key


def _convert_obs_value(key: str, value) -> tuple[str, np.ndarray]:
    array = np.asarray(value)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim == 3 and array.shape[-1] in (1, 3, 4):
        array = np.moveaxis(array, -1, 0)
        key = _normalise_rgb_key(key)
    if array.dtype == np.uint8 and array.ndim == 3:
        return key, array
    return key, array.astype(np.float32, copy=False)


def _convert_obs_dict(obs_dict) -> dict[str, np.ndarray]:
    converted = {}
    for key in sorted(obs_dict.keys()):
        new_key, value = _convert_obs_value(key, obs_dict[key])
        converted[new_key] = value
    return converted


def _convert_obs_group(obs_group, index: int) -> dict[str, np.ndarray]:
    return _convert_obs_dict({key: obs_group[key][index] for key in obs_group.keys()})


def _space_from_array(array: np.ndarray) -> spaces.Box:
    if array.dtype == np.uint8 and array.ndim == 3:
        low = np.zeros(array.shape, dtype=np.uint8)
        high = np.full(array.shape, 255, dtype=np.uint8)
        return spaces.Box(low=low, high=high, dtype=np.uint8)
    low = np.full(array.shape, -np.inf, dtype=np.float32)
    high = np.full(array.shape, np.inf, dtype=np.float32)
    return spaces.Box(low=low, high=high, dtype=np.float32)


def _extract_success(info: dict, env) -> bool:
    if "success" in info:
        success = info["success"]
    elif hasattr(env, "_check_success"):
        success = env._check_success()
    else:
        success = False
    if isinstance(success, dict):
        return bool(success.get("task", any(success.values())))
    return bool(success)


def _normalize_num_demos(num_demos) -> float:
    if isinstance(num_demos, str):
        value = num_demos.strip().lower()
        if value in {"inf", "+inf", ".inf", "+.inf"}:
            return math.inf
        return float(value)
    return float(num_demos)


class RobomimicPlaceholderEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        reset_observation: dict[str, np.ndarray],
        render_shape: tuple[int, int] = (84, 84),
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self._reset_observation = copy.deepcopy(reset_observation)
        self._render_shape = tuple(render_shape)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return copy.deepcopy(self._reset_observation), {"demo": 0}

    def step(self, action):
        return (
            copy.deepcopy(self._reset_observation),
            0.0,
            False,
            True,
            {"demo": 0, "task_success": 0, "placeholder_env": 1},
        )

    def render(self):
        for key, value in self._reset_observation.items():
            if "rgb" in key and value.ndim == 3:
                return np.moveaxis(value, 0, -1)
        height, width = self._render_shape
        return np.zeros((height, width, 3), dtype=np.uint8)


class RobomimicRobosuiteEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        env_name: str,
        env_kwargs: dict,
        pixels: bool,
        visual_observation_shape: tuple[int, int],
        cameras: list[str],
    ):
        try:
            import robosuite
        except ImportError as exc:
            raise ImportError(
                "Live robomimic evaluation requires robosuite to be installed."
            ) from exc

        self._is_v1 = robosuite.__version__.split(".")[0] == "1"
        self._render_camera = cameras[0] if cameras else "agentview"
        kwargs = copy.deepcopy(env_kwargs)
        kwargs.update(
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=pixels,
        )
        if pixels:
            kwargs["camera_names"] = cameras
            kwargs["camera_heights"] = int(visual_observation_shape[0])
            kwargs["camera_widths"] = int(visual_observation_shape[1])
            kwargs["camera_depths"] = False

        self._env = robosuite.make(env_name, **kwargs)
        if self._is_v1:
            # Match robomimic's robosuite wrapper by explicitly enabling
            # joint position and eef velocity observations.
            for observable_name in self._env.observation_names:
                if ("joint_pos" in observable_name) or ("eef_vel" in observable_name):
                    self._env.modify_observable(
                        observable_name=observable_name,
                        attribute="active",
                        modifier=True,
                    )
        action_low, action_high = self._env.action_spec
        self.action_space = spaces.Box(
            low=np.asarray(action_low, dtype=np.float32),
            high=np.asarray(action_high, dtype=np.float32),
            dtype=np.float32,
        )
        self._env.reset()
        reset_obs = self._get_observation()
        self.observation_space = spaces.Dict(
            {key: _space_from_array(value) for key, value in reset_obs.items()}
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._env.reset()
        return self._get_observation(), {"demo": 0}

    def step(self, action):
        _, reward, done, info = self._env.step(action)
        task_success = _extract_success(info, self._env)
        return (
            self._get_observation(),
            float(reward),
            bool(done or task_success),
            False,
            {"demo": 0, "task_success": int(task_success)},
        )

    def _get_observation(self) -> dict[str, np.ndarray]:
        if self._is_v1:
            raw_obs = self._env._get_observations(force_update=True)
        else:
            raw_obs = self._env._get_observation()

        observation = {}
        if "object-state" in raw_obs:
            observation["object"] = np.asarray(raw_obs["object-state"], dtype=np.float32)

        for robot in self._env.robots:
            prefix = robot.robot_model.naming_prefix
            for key, value in raw_obs.items():
                if key.startswith(prefix) and not key.endswith("proprio-state"):
                    new_key, converted = _convert_obs_value(key, value)
                    observation[new_key] = converted

        for key, value in raw_obs.items():
            if key in observation or key.endswith("object-state"):
                continue
            if "image" in key:
                new_key, converted = _convert_obs_value(key, value)
                observation[new_key] = converted

        return observation

    def render(self):
        return self._env.sim.render(camera_name=self._render_camera)[::-1]

    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()


class RobomimicEnvFactory(EnvFactory):
    def __init__(self):
        self._warned_about_placeholder = False

    def _ensure_dataset_metadata(self, cfg: DictConfig):
        _require_h5py()
        dataset_path = _resolve_dataset_path(cfg.env.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"robomimic dataset not found: {dataset_path}")
        if getattr(self, "_dataset_path", None) == dataset_path:
            return

        with h5py.File(dataset_path, "r") as dataset_file:
            demo_keys = _get_demo_keys(dataset_file, cfg.env.filter_key)
            if len(demo_keys) == 0:
                raise ValueError("robomimic dataset does not contain any demos.")
            env_meta = json.loads(dataset_file["data"].attrs["env_args"])
            first_demo = dataset_file[f"data/{demo_keys[0]}"]
            reset_observation = _convert_obs_group(first_demo["obs"], 0)
            max_episode_length = max(
                int(dataset_file[f"data/{key}"].attrs["num_samples"])
                for key in demo_keys
            )
            action_dim = int(first_demo["actions"].shape[-1])

        self._dataset_path = dataset_path
        self._env_meta = env_meta
        self._raw_observation_space = spaces.Dict(
            {key: _space_from_array(value) for key, value in reset_observation.items()}
        )
        self._raw_action_space = spaces.Box(
            low=-np.ones(action_dim, dtype=np.float32),
            high=np.ones(action_dim, dtype=np.float32),
            dtype=np.float32,
        )
        self._reset_observation = reset_observation
        self._dataset_has_images = any(
            value.dtype == np.uint8 and value.ndim == 3
            for value in reset_observation.values()
        )

        if cfg.pixels and not self._dataset_has_images:
            raise ValueError(
                "pixels=true requires a robomimic image dataset, but the current "
                f"dataset only contains low-dimensional observations: {dataset_path}"
            )

        with open_dict(cfg):
            if not cfg.env.task_name:
                cfg.env.task_name = env_meta["env_name"]
            if not cfg.env.episode_length:
                cfg.env.episode_length = max_episode_length

    def _load_demos_from_dataset(
        self, dataset_path: Path, filter_key: str | None, num_demos: int, random_traj: bool
    ) -> list[Demo]:
        _require_h5py()
        num_demos = _normalize_num_demos(num_demos)
        with h5py.File(dataset_path, "r") as dataset_file:
            demo_keys = _get_demo_keys(dataset_file, filter_key)
            if math.isfinite(num_demos):
                num_demos = min(int(num_demos), len(demo_keys))
            else:
                num_demos = len(demo_keys)
            if random_traj:
                demo_keys = random.sample(demo_keys, num_demos)
            else:
                demo_keys = demo_keys[:num_demos]

            demos = []
            for demo_key in demo_keys:
                episode = dataset_file[f"data/{demo_key}"]
                actions = np.asarray(episode["actions"], dtype=np.float32)
                rewards = np.asarray(episode["rewards"], dtype=np.float32)
                dones = np.asarray(episode["dones"], dtype=np.uint8).astype(bool)
                demo_timesteps = [(_convert_obs_group(episode["obs"], 0), {"demo": 1})]
                for index in range(actions.shape[0]):
                    terminated = bool(dones[index])
                    truncated = index == (actions.shape[0] - 1) and not terminated
                    info = {"demo_action": actions[index], "demo": 1}
                    demo_timesteps.append(
                        (
                            _convert_obs_group(episode["next_obs"], index),
                            float(rewards[index]),
                            terminated,
                            truncated,
                            info,
                        )
                    )
                demos.append(Demo(demo_timesteps))
        return demos

    def _wrap_env(
        self,
        env: gym.Env,
        cfg: DictConfig,
        return_raw_spaces: bool = False,
        demo_env: bool = False,
    ):
        if return_raw_spaces:
            action_space = copy.deepcopy(env.action_space)
            observation_space = copy.deepcopy(env.observation_space)

        env = RescaleFromTanh(env)

        has_low_dim_keys = any(
            len(space.shape) == 1 for space in env.observation_space.values()
        )
        if has_low_dim_keys:
            env = ConcatDim(
                env=env,
                shape_length=1,
                dim=-1,
                new_name="low_dim_state",
            )

        env = TimeLimit(env, cfg.env.episode_length)
        if cfg.use_onehot_time_and_no_bootstrap:
            env = OnehotTime(env, cfg.env.episode_length)
        if not demo_env:
            env = FrameStack(env, cfg.frame_stack)
            env = ActionSequence(env, cfg.action_sequence)
        env = AppendDemoInfo(env)

        if return_raw_spaces:
            return env, (action_space, observation_space)
        return env

    def _make_base_env(self, cfg: DictConfig):
        self._ensure_dataset_metadata(cfg)
        if cfg.env.use_live_env:
            return RobomimicRobosuiteEnv(
                env_name=self._env_meta["env_name"],
                env_kwargs=self._env_meta["env_kwargs"],
                pixels=cfg.pixels,
                visual_observation_shape=tuple(cfg.visual_observation_shape),
                cameras=list(cfg.env.cameras),
            )

        if not self._warned_about_placeholder:
            logging.warning(
                "robomimic is using a placeholder gym env because "
                "env.use_live_env=false. Demo loading works, but online rollouts "
                "and evaluation require robosuite."
            )
            self._warned_about_placeholder = True
        return RobomimicPlaceholderEnv(
            observation_space=self._raw_observation_space,
            action_space=self._raw_action_space,
            reset_observation=self._reset_observation,
            render_shape=tuple(cfg.visual_observation_shape),
        )

    def make_train_env(self, cfg: DictConfig) -> gym.vector.VectorEnv:
        self._ensure_dataset_metadata(cfg)
        return gym.vector.SyncVectorEnv(
            [lambda: self._wrap_env(self._make_base_env(cfg), cfg) for _ in range(cfg.num_train_envs)]
        )

    def make_eval_env(self, cfg: DictConfig) -> gym.Env:
        self._ensure_dataset_metadata(cfg)
        env = self._make_base_env(cfg)
        wrapped_env, _ = self._wrap_env(env, cfg, return_raw_spaces=True)
        return wrapped_env

    def collect_or_fetch_demos(self, cfg: DictConfig, num_demos: int):
        self._ensure_dataset_metadata(cfg)
        self._raw_demos = self._load_demos_from_dataset(
            dataset_path=self._dataset_path,
            filter_key=cfg.env.filter_key,
            num_demos=num_demos,
            random_traj=cfg.env.random_traj,
        )

    def post_collect_or_fetch_demos(self, cfg: DictConfig):
        self._demos = self._raw_demos

    def load_demos_into_replay(
        self, cfg: DictConfig, buffer, is_demo_buffer: bool = False
    ):
        assert hasattr(self, "_demos"), (
            "There's no _demo attribute inside the factory, "
            "Check `collect_or_fetch_demos` is called before calling this method."
        )
        demo_env = self._wrap_env(
            DemoEnv(
                copy.deepcopy(self._demos),
                copy.deepcopy(self._raw_action_space),
                copy.deepcopy(self._raw_observation_space),
            ),
            cfg,
            demo_env=True,
        )
        for _ in range(len(self._demos)):
            add_demo_to_replay_buffer(demo_env, buffer)
