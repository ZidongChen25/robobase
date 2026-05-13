import copy
import json
import logging
import math
import os
import random
from functools import partial
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, open_dict
from scipy.spatial.transform import Rotation

from robobase.envs.env import Demo, DemoEnv, EnvFactory
from robobase.envs.wrappers import (
    ActionSequence,
    AppendDemoInfo,
    ConcatDim,
    FrameStack,
    OnehotTime,
    RecedingHorizonControl,
    RescaleFromTanh,
    RescaleFromTanhWithMinMax,
    RescaleFromTanhWithStandardization,
)
from robobase.utils import add_demo_to_replay_buffer, rescale_demo_actions

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

    candidates = []
    try:
        base_dir = Path(get_original_cwd())
    except ValueError:
        base_dir = Path.cwd()
    candidates.append((base_dir / path).resolve())
    candidates.append((Path.cwd() / path).resolve())

    # Allow this backend repo to reuse datasets that live in the sibling
    # /home/.../robobase workspace without copying them locally.
    sibling_robobase_root = Path(__file__).resolve().parents[3] / "robobase"
    candidates.append((sibling_robobase_root / path).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


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


def _sort_obs_dict(obs_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: obs_dict[key] for key in sorted(obs_dict.keys())}


def _camera_name_to_rgb_key(camera_name: str) -> str:
    normalized = _normalise_rgb_key(str(camera_name))
    if normalized.endswith("_rgb"):
        return normalized
    return f"{normalized}_rgb"


def _rgb_keys_from_obs_dict(obs_dict: dict[str, np.ndarray]) -> list[str]:
    return [
        key
        for key, value in sorted(obs_dict.items())
        if value.dtype == np.uint8 and value.ndim == 3 and key.endswith("_rgb")
    ]


def _resize_rgb_obs(
    value: np.ndarray, visual_observation_shape: tuple[int, int]
) -> np.ndarray:
    target_height, target_width = (int(dim) for dim in visual_observation_shape)
    if value.shape[-2:] == (target_height, target_width):
        return value

    resized = cv2.resize(
        np.moveaxis(value, 0, -1),
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )
    if resized.ndim == 2:
        resized = resized[:, :, None]
    return np.moveaxis(resized, -1, 0).astype(value.dtype, copy=False)


def _filter_obs_dict_for_cfg(
    obs_dict: dict[str, np.ndarray], cfg: DictConfig
) -> dict[str, np.ndarray]:
    obs_keys = list(_cfg_get(cfg.env, "obs_keys", []))
    if obs_keys:
        missing_keys = [key for key in obs_keys if key not in obs_dict]
        if missing_keys:
            raise KeyError(
                "Configured robomimic obs_keys are missing from the dataset/env: "
                f"{missing_keys}"
            )
        selected = {key: obs_dict[key] for key in obs_keys}
        if cfg.pixels:
            selected.update(
                {
                    key: value
                    for key, value in obs_dict.items()
                    if value.dtype == np.uint8
                    and value.ndim == 3
                    and key.endswith("_rgb")
                }
            )
        obs_dict = selected

    if not cfg.pixels:
        return {
            key: value
            for key, value in obs_dict.items()
            if not (value.dtype == np.uint8 and value.ndim == 3 and key.endswith("_rgb"))
        }

    dataset_rgb_keys = _rgb_keys_from_obs_dict(obs_dict)
    if not dataset_rgb_keys:
        return obs_dict

    configured_cameras = list(cfg.env.cameras)
    configured_rgb_keys = [
        _camera_name_to_rgb_key(camera_name) for camera_name in configured_cameras
    ]
    selected_rgb_keys = [key for key in configured_rgb_keys if key in dataset_rgb_keys]

    if not selected_rgb_keys:
        selected_rgb_keys = dataset_rgb_keys
        inferred_cameras = [key[: -len("_rgb")] for key in selected_rgb_keys]
        if configured_cameras and configured_cameras != inferred_cameras:
            logging.warning(
                "robomimic dataset cameras %s do not match cfg.env.cameras=%s. "
                "Falling back to dataset cameras.",
                inferred_cameras,
                configured_cameras,
            )
        if configured_cameras != inferred_cameras:
            with open_dict(cfg):
                cfg.env.cameras = inferred_cameras

    visual_observation_shape = tuple(cfg.visual_observation_shape)
    filtered_obs = {}
    for key, value in obs_dict.items():
        is_rgb = value.dtype == np.uint8 and value.ndim == 3 and key.endswith("_rgb")
        if is_rgb:
            if key in selected_rgb_keys:
                filtered_obs[key] = _resize_rgb_obs(value, visual_observation_shape)
            continue
        filtered_obs[key] = value
    return filtered_obs


def _convert_obs_group(obs_group, index: int) -> dict[str, np.ndarray]:
    return _convert_obs_dict({key: obs_group[key][index] for key in obs_group.keys()})


def _load_obs_sequence_for_cfg(
    obs_group,
    length: int,
    cfg: DictConfig,
) -> list[dict[str, np.ndarray]]:
    obs_keys = list(_cfg_get(cfg.env, "obs_keys", []))
    if obs_keys and not cfg.pixels:
        arrays = {
            key: np.asarray(obs_group[key][:length], dtype=np.float32)
            for key in obs_keys
        }
        return [
            {key: arrays[key][index] for key in obs_keys}
            for index in range(length)
        ]
    return [
        _filter_obs_dict_for_cfg(_convert_obs_group(obs_group, index), cfg)
        for index in range(length)
    ]


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


def _cfg_get(cfg: DictConfig, key: str, default):
    return cfg.get(key, default)


def _normalize_vectors(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, eps)


def _axis_angle_to_rotation_6d(axis_angle: np.ndarray) -> np.ndarray:
    matrices = Rotation.from_rotvec(axis_angle.reshape(-1, 3)).as_matrix()
    matrices = matrices.reshape(*axis_angle.shape[:-1], 3, 3)
    return matrices[..., :2, :].reshape(*axis_angle.shape[:-1], 6)


def _rotation_6d_to_axis_angle(rotation_6d: np.ndarray) -> np.ndarray:
    a1 = rotation_6d[..., :3]
    a2 = rotation_6d[..., 3:]
    b1 = _normalize_vectors(a1)
    b2 = _normalize_vectors(a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1)
    b3 = np.cross(b1, b2, axis=-1)
    matrices = np.stack([b1, b2, b3], axis=-2)
    return Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_rotvec().reshape(
        *rotation_6d.shape[:-1], 3
    )


def _raw_action_to_abs_action(raw_actions: np.ndarray) -> np.ndarray:
    raw_shape = raw_actions.shape
    is_dual_arm = raw_shape[-1] == 14
    actions = raw_actions.reshape(*raw_shape[:-1], 2, 7) if is_dual_arm else raw_actions
    pos = actions[..., :3]
    rot = _axis_angle_to_rotation_6d(actions[..., 3:6])
    gripper = actions[..., 6:]
    abs_actions = np.concatenate([pos, rot, gripper], axis=-1).astype(
        np.float32, copy=False
    )
    if is_dual_arm:
        abs_actions = abs_actions.reshape(*raw_shape[:-1], 20)
    return abs_actions


def _abs_action_to_raw_action(abs_actions: np.ndarray) -> np.ndarray:
    raw_shape = abs_actions.shape
    is_dual_arm = raw_shape[-1] == 20
    actions = abs_actions.reshape(*raw_shape[:-1], 2, 10) if is_dual_arm else abs_actions
    pos = actions[..., :3]
    rot = _rotation_6d_to_axis_angle(actions[..., 3:-1])
    gripper = actions[..., -1:]
    raw_actions = np.concatenate([pos, rot, gripper], axis=-1).astype(
        np.float32, copy=False
    )
    if is_dual_arm:
        raw_actions = raw_actions.reshape(*raw_shape[:-1], 14)
    return raw_actions


class RobomimicAbsoluteAction(gym.ActionWrapper, gym.utils.RecordConstructorArgs):
    """Expose CleanDiffuser's absolute 6D-rotation action representation."""

    def __init__(self, env: gym.Env):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ActionWrapper.__init__(self, env)
        raw_shape = env.action_space.shape
        if len(raw_shape) != 1 or raw_shape[-1] not in (7, 14):
            raise ValueError(
                "robomimic absolute action conversion expects a 7D or 14D raw "
                f"action space, got {raw_shape}."
            )
        action_dim = 20 if raw_shape[-1] == 14 else 10
        self.action_space = spaces.Box(
            low=np.full((action_dim,), -np.inf, dtype=np.float32),
            high=np.full((action_dim,), np.inf, dtype=np.float32),
            dtype=np.float32,
        )
        self.is_vector_env = getattr(env, "is_vector_env", False)

    def action(self, action):
        return _abs_action_to_raw_action(np.asarray(action, dtype=np.float32))


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
        has_offscreen_renderer: bool,
        render_gpu_device_id: int,
        abs_action: bool = False,
        obs_keys: list[str] | None = None,
    ):
        try:
            import robosuite
        except ImportError as exc:
            raise ImportError(
                "Live robomimic evaluation requires robosuite to be installed."
            ) from exc

        self._is_v1 = robosuite.__version__.split(".")[0] == "1"
        self._render_camera = cameras[0] if cameras else "agentview"
        self._obs_keys = list(obs_keys or [])
        kwargs = copy.deepcopy(env_kwargs)
        if abs_action:
            kwargs.setdefault("controller_configs", {})
            kwargs["controller_configs"] = copy.deepcopy(kwargs["controller_configs"])
            kwargs["controller_configs"]["control_delta"] = False
        kwargs.update(
            has_renderer=False,
            has_offscreen_renderer=has_offscreen_renderer,
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=pixels,
            render_gpu_device_id=render_gpu_device_id,
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

        observation = _sort_obs_dict(observation)
        if self._obs_keys:
            missing_keys = [key for key in self._obs_keys if key not in observation]
            if missing_keys:
                raise KeyError(
                    "Configured robomimic obs_keys are missing from live env: "
                    f"{missing_keys}"
                )
            selected = {key: observation[key] for key in self._obs_keys}
            if self._render_camera and any(
                value.dtype == np.uint8 and value.ndim == 3
                for value in observation.values()
            ):
                selected.update(
                    {
                        key: value
                        for key, value in observation.items()
                        if value.dtype == np.uint8
                        and value.ndim == 3
                        and key.endswith("_rgb")
                    }
                )
            observation = selected
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
            reset_observation = _filter_obs_dict_for_cfg(
                _convert_obs_group(first_demo["obs"], 0), cfg
            )
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
        self,
        cfg: DictConfig,
        dataset_path: Path,
        filter_key: str | None,
        num_demos: int,
        random_traj: bool,
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
            selected_demo_keys = []
            for demo_key in demo_keys:
                episode = dataset_file[f"data/{demo_key}"]
                actions = np.asarray(episode["actions"], dtype=np.float32)
                if bool(_cfg_get(cfg.env, "abs_action", False)):
                    actions = _raw_action_to_abs_action(actions)
                rewards = np.asarray(episode["rewards"], dtype=np.float32)
                dones = np.asarray(episode["dones"], dtype=np.uint8).astype(bool)
                obs_sequence = _load_obs_sequence_for_cfg(
                    episode["obs"],
                    actions.shape[0],
                    cfg,
                )
                next_obs_sequence = _load_obs_sequence_for_cfg(
                    episode["next_obs"],
                    actions.shape[0],
                    cfg,
                )
                demo_timesteps = [(obs_sequence[0], {"demo": 1})]
                for index in range(actions.shape[0]):
                    terminated = bool(dones[index])
                    truncated = index == (actions.shape[0] - 1) and not terminated
                    info = {"demo_action": actions[index], "demo": 1}
                    demo_timesteps.append(
                        (
                            next_obs_sequence[index],
                            float(rewards[index]),
                            terminated,
                            truncated,
                            info,
                        )
                    )
                demos.append(Demo(demo_timesteps))
                selected_demo_keys.append(demo_key)
            self._selected_demo_keys = selected_demo_keys
        return demos

    def _compute_action_stats(self, demos: list[Demo]) -> dict[str, np.ndarray]:
        actions = []
        for demo in demos:
            for step in demo:
                *_, info = step
                if isinstance(info, dict) and "demo_action" in info:
                    actions.append(info["demo_action"])
        if len(actions) == 0:
            raise ValueError("robomimic demos do not contain any actions.")

        actions = np.stack(actions, axis=0)
        action_stats = {
            "mean": np.mean(actions, axis=0),
            "std": np.std(actions, axis=0),
            "max": np.max(actions, axis=0),
            "min": np.min(actions, axis=0),
        }
        action_stats["std"][action_stats["std"] == 0] = 1.0
        return action_stats

    def _compute_obs_stats(self, demos: list[Demo]) -> dict[str, dict[str, np.ndarray]]:
        observations = []
        for demo in demos:
            for step in demo:
                observation = step[0]
                if isinstance(observation, dict):
                    observations.append(observation)
        if len(observations) == 0:
            raise ValueError("robomimic demos do not contain any observations.")

        keys = observations[0].keys()
        obs_arrays = {
            key: np.stack([obs[key] for obs in observations], axis=0) for key in keys
        }
        obs_stats = {
            "mean": {key: np.mean(obs_arrays[key], axis=0) for key in keys},
            "std": {key: np.std(obs_arrays[key], axis=0) for key in keys},
            "max": {key: np.max(obs_arrays[key], axis=0) for key in keys},
            "min": {key: np.min(obs_arrays[key], axis=0) for key in keys},
        }
        for key in keys:
            obs_stats["std"][key][obs_stats["std"][key] == 0] = 1.0
        return obs_stats

    def _compute_obs_stats_from_dataset_obs(
        self, cfg: DictConfig
    ) -> dict[str, dict[str, np.ndarray]]:
        """Match CleanDiffuser stats: fit on dataset obs, not final next_obs."""

        selected_demo_keys = getattr(self, "_selected_demo_keys", None)
        if not selected_demo_keys:
            return self._compute_obs_stats(self._raw_demos)

        obs_arrays = {}
        obs_keys = list(_cfg_get(cfg.env, "obs_keys", []))
        with h5py.File(self._dataset_path, "r") as dataset_file:
            for demo_key in selected_demo_keys:
                obs_group = dataset_file[f"data/{demo_key}/obs"]
                episode_length = dataset_file[f"data/{demo_key}/actions"].shape[0]
                if obs_keys and not cfg.pixels:
                    for key in obs_keys:
                        obs_arrays.setdefault(key, []).append(
                            np.asarray(
                                obs_group[key][:episode_length],
                                dtype=np.float32,
                            )
                        )
                    continue

                for index in range(episode_length):
                    obs = _filter_obs_dict_for_cfg(
                        _convert_obs_group(obs_group, index),
                        cfg,
                    )
                    for key, value in obs.items():
                        if value.dtype == np.uint8:
                            continue
                        obs_arrays.setdefault(key, []).append(value[None])

        if len(obs_arrays) == 0:
            raise ValueError("robomimic demos do not contain low-dimensional observations.")

        stacked = {
            key: np.concatenate(values, axis=0) for key, values in obs_arrays.items()
        }
        obs_stats = {
            "mean": {key: np.mean(stacked[key], axis=0) for key in stacked},
            "std": {key: np.std(stacked[key], axis=0) for key in stacked},
            "max": {key: np.max(stacked[key], axis=0) for key in stacked},
            "min": {key: np.min(stacked[key], axis=0) for key in stacked},
        }
        for key in stacked:
            obs_stats["std"][key][obs_stats["std"][key] == 0] = 1.0
        return obs_stats

    def _make_rescale_from_tanh_cls(self, cfg: DictConfig):
        use_standardization = _cfg_get(cfg, "use_standardization", False)
        use_min_max_normalization = _cfg_get(
            cfg, "use_min_max_normalization", False
        )
        demos = _cfg_get(cfg, "demos", 0)
        min_max_margin = _cfg_get(cfg, "min_max_margin", 0.0)

        assert not (
            use_standardization and use_min_max_normalization
        ), "You can't use both standardization and min/max normalization."
        if bool(_cfg_get(cfg.env, "abs_action", False)) and not use_min_max_normalization:
            raise ValueError(
                "env.abs_action=true requires use_min_max_normalization=true so "
                "the 10D absolute action representation can be mapped to [-1, 1]."
            )

        if use_standardization:
            assert demos != 0
            return partial(
                RescaleFromTanhWithStandardization,
                action_stats=self._action_stats,
            )
        if use_min_max_normalization:
            assert demos != 0
            return partial(
                RescaleFromTanhWithMinMax,
                action_stats=self._action_stats,
                min_max_margin=min_max_margin,
            )
        return RescaleFromTanh

    def _rescale_demo_action_helper(self, info, cfg: DictConfig):
        use_standardization = _cfg_get(cfg, "use_standardization", False)
        use_min_max_normalization = _cfg_get(
            cfg, "use_min_max_normalization", False
        )
        min_max_margin = _cfg_get(cfg, "min_max_margin", 0.0)

        if use_standardization:
            return RescaleFromTanhWithStandardization.transform_to_tanh(
                info["demo_action"],
                action_stats=self._action_stats,
            )
        if use_min_max_normalization:
            return RescaleFromTanhWithMinMax.transform_to_tanh(
                info["demo_action"],
                action_stats=self._action_stats,
                min_max_margin=min_max_margin,
            )
        return RescaleFromTanh.transform_to_tanh(
            info["demo_action"], self._raw_action_space
        )

    def _wrap_env(
        self,
        env: gym.Env,
        cfg: DictConfig,
        return_raw_spaces: bool = False,
        demo_env: bool = False,
    ):
        if bool(_cfg_get(cfg.env, "abs_action", False)):
            env = RobomimicAbsoluteAction(env)

        if return_raw_spaces:
            action_space = copy.deepcopy(env.action_space)
            observation_space = copy.deepcopy(env.observation_space)

        rescale_from_tanh_cls = self._make_rescale_from_tanh_cls(cfg)
        env = rescale_from_tanh_cls(env)

        obs_stats = None
        norm_obs = _cfg_get(cfg, "norm_obs", False)
        demos = _cfg_get(cfg, "demos", 0)
        execution_length = _cfg_get(cfg, "execution_length", cfg.action_sequence)
        temporal_ensemble = _cfg_get(cfg, "temporal_ensemble", True)
        temporal_ensemble_gain = _cfg_get(cfg, "temporal_ensemble_gain", 0.01)

        if norm_obs:
            assert demos != 0
            obs_stats = self._obs_stats
        obs_norm_type = str(_cfg_get(cfg, "obs_norm_type", "standardization")).lower()

        has_low_dim_keys = any(
            len(space.shape) == 1 for space in env.observation_space.values()
        )
        if has_low_dim_keys:
            env = ConcatDim(
                env=env,
                shape_length=1,
                dim=-1,
                new_name="low_dim_state",
                norm_obs=norm_obs,
                obs_stats=obs_stats,
                obs_norm_type=obs_norm_type,
            )

        env = TimeLimit(env, cfg.env.episode_length)
        if cfg.use_onehot_time_and_no_bootstrap:
            env = OnehotTime(env, cfg.env.episode_length)
        if not demo_env:
            env = FrameStack(env, cfg.frame_stack)
            if cfg.action_sequence == execution_length:
                env = ActionSequence(
                    env,
                    cfg.action_sequence,
                )
            else:
                env = RecedingHorizonControl(
                    env,
                    cfg.action_sequence,
                    cfg.env.episode_length,
                    execution_length,
                    temporal_ensemble,
                    temporal_ensemble_gain,
                    _cfg_get(cfg, "action_execution_start", 0),
                )
        env = AppendDemoInfo(env)

        if return_raw_spaces:
            return env, (action_space, observation_space)
        return env

    def _make_base_env(self, cfg: DictConfig):
        self._ensure_dataset_metadata(cfg)
        if cfg.env.use_live_env:
            needs_offscreen_renderer = cfg.pixels or cfg.log_eval_video
            return RobomimicRobosuiteEnv(
                env_name=self._env_meta["env_name"],
                env_kwargs=self._env_meta["env_kwargs"],
                pixels=cfg.pixels,
                visual_observation_shape=tuple(cfg.visual_observation_shape),
                cameras=list(cfg.env.cameras),
                has_offscreen_renderer=needs_offscreen_renderer,
                render_gpu_device_id=cfg.env.render_gpu_device_id,
                abs_action=bool(_cfg_get(cfg.env, "abs_action", False)),
                obs_keys=list(_cfg_get(cfg.env, "obs_keys", [])),
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

    def get_spaces(self, cfg: DictConfig) -> tuple[gym.Space, gym.Space]:
        self._ensure_dataset_metadata(cfg)
        wrapped_env = self._wrap_env(
            RobomimicPlaceholderEnv(
                observation_space=copy.deepcopy(self._raw_observation_space),
                action_space=copy.deepcopy(self._raw_action_space),
                reset_observation=copy.deepcopy(self._reset_observation),
                render_shape=tuple(cfg.visual_observation_shape),
            ),
            cfg,
        )
        return wrapped_env.observation_space, wrapped_env.action_space

    def make_eval_env(self, cfg: DictConfig) -> gym.Env:
        self._ensure_dataset_metadata(cfg)
        env = self._make_base_env(cfg)
        wrapped_env, _ = self._wrap_env(env, cfg, return_raw_spaces=True)
        return wrapped_env

    def make_eval_envs(self, cfg: DictConfig) -> gym.vector.VectorEnv:
        self._ensure_dataset_metadata(cfg)
        return gym.vector.SyncVectorEnv(
            [
                lambda: self._wrap_env(self._make_base_env(cfg), cfg)
                for _ in range(cfg.num_eval_envs)
            ]
        )

    def collect_or_fetch_demos(self, cfg: DictConfig, num_demos: int):
        self._ensure_dataset_metadata(cfg)
        self._raw_demos = self._load_demos_from_dataset(
            cfg=cfg,
            dataset_path=self._dataset_path,
            filter_key=cfg.env.filter_key,
            num_demos=num_demos,
            random_traj=cfg.env.random_traj,
        )
        if _cfg_get(cfg, "use_standardization", False) or _cfg_get(
            cfg, "use_min_max_normalization", False
        ):
            self._action_stats = self._compute_action_stats(self._raw_demos)
        if _cfg_get(cfg, "norm_obs", False):
            self._obs_stats = self._compute_obs_stats_from_dataset_obs(cfg)

    def post_collect_or_fetch_demos(self, cfg: DictConfig):
        self._demos = rescale_demo_actions(
            self._rescale_demo_action_helper, self._raw_demos, cfg
        )

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
