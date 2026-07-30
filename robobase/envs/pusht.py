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
import imageio.v3 as iio
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
    RecedingHorizonControl,
    RescaleFromTanh,
    RescaleFromTanhWithMinMax,
    RescaleFromTanhWithStandardization,
)
from robobase.utils import add_demo_to_replay_buffer, rescale_demo_actions

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None


_DEFAULT_REPO_ID = "lerobot/pusht"
_DEFAULT_TASK_NAME = "PushT"
_DEFAULT_ENV_ID = "gym_pusht/PushT-v0"
_DEFAULT_EPISODE_LENGTH = 300
_DEFAULT_IMAGE_KEY = "image_rgb"
_DEFAULT_STATE_KEY = "agent_pos"


def _require_pyarrow():
    if pq is None:
        raise ImportError(
            "Push-T LeRobot dataset support requires pyarrow. Install the "
            "optional pusht dependencies first."
        )


def _cfg_get(cfg, key: str, default):
    return cfg.get(key, default)


def _normalize_num_demos(num_demos) -> float:
    if isinstance(num_demos, str):
        value = num_demos.strip().lower()
        if value in {"inf", "+inf", ".inf", "+.inf"}:
            return math.inf
        return float(value)
    return float(num_demos)


def _resolve_path(path: str | os.PathLike) -> Path:
    path = Path(os.path.expanduser(str(path)))
    if path.is_absolute():
        return path
    candidates = []
    try:
        candidates.append((Path(get_original_cwd()) / path).resolve())
    except ValueError:
        pass
    candidates.append((Path.cwd() / path).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _parse_split_range(split: str, episode_indices: list[int]) -> list[int]:
    if not split:
        return episode_indices
    if ":" not in split:
        raise ValueError(f"Unsupported LeRobot split format: {split!r}")
    start_s, end_s = split.split(":", maxsplit=1)
    start = int(start_s) if start_s else 0
    end = int(end_s) if end_s else len(episode_indices)
    return episode_indices[start:end]


def _space_from_array(array: np.ndarray) -> spaces.Box:
    if array.dtype == np.uint8 and array.ndim == 3:
        return spaces.Box(
            low=np.zeros(array.shape, dtype=np.uint8),
            high=np.full(array.shape, 255, dtype=np.uint8),
            dtype=np.uint8,
        )
    return spaces.Box(
        low=np.full(array.shape, -np.inf, dtype=np.float32),
        high=np.full(array.shape, np.inf, dtype=np.float32),
        dtype=np.float32,
    )


def _resize_hwc_rgb(
    value: np.ndarray, visual_observation_shape: tuple[int, int]
) -> np.ndarray:
    target_height, target_width = (int(dim) for dim in visual_observation_shape)
    if value.shape[:2] == (target_height, target_width):
        return value.astype(np.uint8, copy=False)
    resized = cv2.resize(
        value,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )
    if resized.ndim == 2:
        resized = resized[:, :, None]
    return resized.astype(np.uint8, copy=False)


def _hwc_to_chw_rgb(
    value: np.ndarray, visual_observation_shape: tuple[int, int]
) -> np.ndarray:
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.floating) and array.max(initial=0.0) <= 1.0:
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    else:
        array = array.astype(np.uint8, copy=False)
    if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected HWC RGB image, got shape {array.shape}.")
    array = array[..., :3]
    array = _resize_hwc_rgb(array, visual_observation_shape)
    return np.moveaxis(array, -1, 0)


def _decode_video_frames(
    video_path: Path,
    required_indices: set[int],
    visual_observation_shape: tuple[int, int],
) -> dict[int, np.ndarray]:
    if len(required_indices) == 0:
        return {}
    if not video_path.exists():
        raise FileNotFoundError(f"Push-T video file not found: {video_path}")

    targets = sorted(required_indices)
    decoded = {}
    target_pos = 0
    for frame_index, frame in enumerate(iio.imiter(video_path)):
        while target_pos < len(targets) and targets[target_pos] < frame_index:
            target_pos += 1
        if target_pos >= len(targets):
            break
        if frame_index == targets[target_pos]:
            decoded[frame_index] = _hwc_to_chw_rgb(frame, visual_observation_shape)
            target_pos += 1
            while target_pos < len(targets) and targets[target_pos] == frame_index:
                target_pos += 1
        if target_pos >= len(targets):
            break

    missing = sorted(set(targets) - set(decoded))
    if missing:
        raise ValueError(
            f"Push-T video {video_path} ended before frames {missing[:5]} "
            "could be decoded."
        )
    return decoded


def _read_parquet_table(path: Path):
    _require_pyarrow()
    if not path.exists():
        raise FileNotFoundError(path)
    return pq.read_table(path)


def _concat_parquet_files(paths: list[Path]):
    _require_pyarrow()
    tables = [_read_parquet_table(path) for path in sorted(paths)]
    if len(tables) == 0:
        raise FileNotFoundError("No LeRobot parquet files found.")
    if len(tables) == 1:
        return tables[0]
    return pa.concat_tables(tables, promote_options="default")


def _load_lerobot_frame_table(dataset_root: Path) -> dict:
    table = _concat_parquet_files(list(dataset_root.glob("data/**/*.parquet")))
    if "index" in table.column_names:
        order = np.argsort(np.asarray(table["index"].to_pylist(), dtype=np.int64))
        table = table.take(pa.array(order))
    return table.to_pydict()


def _load_info(dataset_root: Path) -> dict:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"LeRobot info.json not found: {info_path}")
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _video_path_from_info(dataset_root: Path, info: dict, video_key: str) -> Path:
    video_path_template = info.get(
        "video_path",
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    )
    return dataset_root / video_path_template.format(
        video_key=video_key,
        chunk_index=0,
        file_index=0,
        episode_chunk=0,
        episode_index=0,
    )


def _obs_from_raw(
    agent_pos: np.ndarray,
    image: np.ndarray | None,
    *,
    pixels: bool,
    image_key: str,
) -> dict[str, np.ndarray]:
    obs = {_DEFAULT_STATE_KEY: np.asarray(agent_pos, dtype=np.float32)}
    if pixels:
        if image is None:
            raise ValueError("pixels=true requires Push-T image observations.")
        obs[image_key] = np.asarray(image, dtype=np.uint8)
    return obs


class PushTPlaceholderEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        reset_observation: dict[str, np.ndarray],
        render_shape: tuple[int, int],
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
        for value in self._reset_observation.values():
            if value.dtype == np.uint8 and value.ndim == 3:
                return np.moveaxis(value, 0, -1)
        height, width = self._render_shape
        return np.zeros((height, width, 3), dtype=np.uint8)


class PushTGymEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        cfg: DictConfig,
        *,
        image_key: str,
    ):
        try:
            import gym_pusht  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Live Push-T evaluation requires gym-pusht. Install the optional "
                "pusht dependencies first."
            ) from exc

        self._pixels = bool(cfg.pixels)
        self._include_environment_state = bool(
            _cfg_get(cfg.env, "include_environment_state", False)
        )
        self._image_key = image_key
        obs_type = (
            "pixels_agent_pos"
            if self._pixels
            else "environment_state_agent_pos"
        )
        configured_obs_type = _cfg_get(cfg.env, "obs_type", None)
        if configured_obs_type:
            obs_type = str(configured_obs_type)
        visual_shape = tuple(int(dim) for dim in cfg.visual_observation_shape)

        kwargs = {
            "obs_type": obs_type,
            "render_mode": str(_cfg_get(cfg.env, "render_mode", "rgb_array")),
            "observation_height": visual_shape[0],
            "observation_width": visual_shape[1],
            "visualization_height": int(_cfg_get(cfg.env, "visualization_height", 384)),
            "visualization_width": int(_cfg_get(cfg.env, "visualization_width", 384)),
            "max_episode_steps": int(cfg.env.episode_length),
            "disable_env_checker": True,
        }
        self._env = gym.make(_DEFAULT_ENV_ID, **kwargs)
        self.action_space = spaces.Box(
            low=np.zeros((2,), dtype=np.float32),
            high=np.full((2,), 512.0, dtype=np.float32),
            dtype=np.float32,
        )

        reset_obs, _ = self._env.reset(seed=int(cfg.seed))
        converted = self._convert_obs(reset_obs)
        self.observation_space = spaces.Dict(
            {key: _space_from_array(value) for key, value in converted.items()}
        )

    def _convert_obs(self, observation) -> dict[str, np.ndarray]:
        if not isinstance(observation, dict):
            observation = {"agent_pos": np.asarray(observation)[:2]}

        converted = {}
        if "agent_pos" in observation:
            converted[_DEFAULT_STATE_KEY] = np.asarray(
                observation["agent_pos"], dtype=np.float32
            )
        if (
            self._include_environment_state
            and "environment_state" in observation
        ):
            converted["environment_state"] = np.asarray(
                observation["environment_state"], dtype=np.float32
            )
        if self._pixels and "pixels" in observation:
            converted[self._image_key] = np.moveaxis(
                np.asarray(observation["pixels"], dtype=np.uint8), -1, 0
            )
        return converted

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        info = dict(info)
        info["task_success"] = int(bool(info.get("is_success", False)))
        return self._convert_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(
            np.asarray(action, dtype=np.float32)
        )
        info = dict(info)
        info["task_success"] = int(bool(info.get("is_success", False)))
        return self._convert_obs(obs), float(reward), terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


class PushTEnvFactory(EnvFactory):
    def __init__(self):
        self._warned_about_placeholder = False

    def _download_dataset(self, cfg: DictConfig, *, pixels: bool) -> Path:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "Downloading Push-T from the Hugging Face Hub requires "
                "huggingface_hub."
            ) from exc

        repo_id = str(_cfg_get(cfg.env, "repo_id", _DEFAULT_REPO_ID))
        allow_patterns = ["meta/*", "data/**"]
        if pixels:
            allow_patterns.append("videos/**")
        dataset_path = str(_cfg_get(cfg.env, "dataset_path", "") or "").strip()
        local_dir = None
        if dataset_path:
            local_dir = str(_resolve_path(dataset_path))
        cache_dir = _cfg_get(cfg.env, "cache_dir", None)
        return Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                cache_dir=cache_dir,
                local_dir=local_dir,
                allow_patterns=allow_patterns,
            )
        )

    def _resolve_dataset_root(self, cfg: DictConfig, *, pixels: bool) -> Path:
        dataset_path = str(_cfg_get(cfg.env, "dataset_path", "") or "").strip()
        if dataset_path:
            path = _resolve_path(dataset_path)
            if path.exists():
                return path
            if not bool(_cfg_get(cfg.env, "download", True)):
                raise FileNotFoundError(f"Push-T dataset not found: {path}")
            return self._download_dataset(cfg, pixels=pixels)

        if not bool(_cfg_get(cfg.env, "download", True)):
            raise FileNotFoundError(
                "env.dataset_path is empty and env.download=false for Push-T."
            )
        return self._download_dataset(cfg, pixels=pixels)

    def _ensure_dataset_metadata(self, cfg: DictConfig):
        pixels = bool(cfg.pixels)
        dataset_root = self._resolve_dataset_root(cfg, pixels=pixels)
        cache_key = (dataset_root, pixels, tuple(cfg.visual_observation_shape))
        if getattr(self, "_cache_key", None) == cache_key:
            return

        info = _load_info(dataset_root)
        data = _load_lerobot_frame_table(dataset_root)
        if "observation.state" not in data or "action" not in data:
            raise KeyError(
                "Push-T LeRobot data must contain observation.state and action."
            )
        episode_indices = sorted({int(v) for v in data["episode_index"]})
        if len(episode_indices) == 0:
            raise ValueError("Push-T LeRobot dataset does not contain any episodes.")

        first_state = np.asarray(data["observation.state"][0], dtype=np.float32)
        first_obs = {_DEFAULT_STATE_KEY: first_state}
        if pixels:
            first_image = self._first_image_from_dataset(cfg, dataset_root, info, data)
            first_obs[self._image_key(cfg)] = first_image

        max_episode_length = max(
            sum(1 for ep in data["episode_index"] if int(ep) == episode_index)
            for episode_index in episode_indices
        )

        self._cache_key = cache_key
        self._dataset_root = dataset_root
        self._info = info
        self._data = data
        self._episode_indices = episode_indices
        self._raw_observation_space = spaces.Dict(
            {key: _space_from_array(value) for key, value in first_obs.items()}
        )
        self._raw_action_space = spaces.Box(
            low=np.zeros((2,), dtype=np.float32),
            high=np.full((2,), 512.0, dtype=np.float32),
            dtype=np.float32,
        )
        self._reset_observation = first_obs

        with open_dict(cfg):
            if not cfg.env.task_name:
                cfg.env.task_name = _DEFAULT_TASK_NAME
            if not cfg.env.episode_length:
                cfg.env.episode_length = int(
                    _cfg_get(cfg.env, "max_episode_steps", _DEFAULT_EPISODE_LENGTH)
                )
            cfg.env.dataset_path = str(dataset_root)
            cfg.env.max_demo_episode_length = max_episode_length

    def _image_key(self, cfg: DictConfig) -> str:
        return str(_cfg_get(cfg.env, "image_key", _DEFAULT_IMAGE_KEY))

    def _first_image_from_dataset(
        self,
        cfg: DictConfig,
        dataset_root: Path,
        info: dict,
        data: dict,
    ) -> np.ndarray:
        visual_shape = tuple(int(dim) for dim in cfg.visual_observation_shape)
        if "observation.image" in data:
            return _hwc_to_chw_rgb(data["observation.image"][0], visual_shape)

        video_path = _video_path_from_info(dataset_root, info, "observation.image")
        return _decode_video_frames(video_path, {0}, visual_shape)[0]

    def _episode_indices_for_cfg(
        self, cfg: DictConfig, num_demos, random_traj: bool
    ) -> list[int]:
        split_name = str(_cfg_get(cfg.env, "split", "train"))
        split_spec = self._info.get("splits", {}).get(split_name, "")
        episode_indices = _parse_split_range(split_spec, self._episode_indices)
        if len(episode_indices) == 0:
            raise ValueError(f"Push-T split {split_name!r} contains no episodes.")

        num_demos = _normalize_num_demos(num_demos)
        if math.isfinite(num_demos):
            episode_indices = episode_indices[: min(int(num_demos), len(episode_indices))]
        if random_traj:
            episode_indices = random.sample(episode_indices, len(episode_indices))
        return episode_indices

    def _row_indices_by_episode(self, selected_episode_indices: list[int]) -> dict[int, list[int]]:
        selected = set(int(v) for v in selected_episode_indices)
        rows_by_episode = {episode_index: [] for episode_index in selected_episode_indices}
        for row_index, episode_index in enumerate(self._data["episode_index"]):
            episode_index = int(episode_index)
            if episode_index in selected:
                rows_by_episode[episode_index].append(row_index)
        return rows_by_episode

    def _load_images_for_rows(
        self,
        cfg: DictConfig,
        rows_by_episode: dict[int, list[int]],
    ) -> dict[int, np.ndarray]:
        if not bool(cfg.pixels):
            return {}

        visual_shape = tuple(int(dim) for dim in cfg.visual_observation_shape)
        image_column = self._data.get("observation.image", None)
        all_rows = [row for rows in rows_by_episode.values() for row in rows]
        if image_column is not None:
            return {
                row: _hwc_to_chw_rgb(image_column[row], visual_shape)
                for row in all_rows
            }

        frame_indices = {int(self._data["index"][row]) for row in all_rows}
        video_path = _video_path_from_info(
            self._dataset_root,
            self._info,
            "observation.image",
        )
        return _decode_video_frames(video_path, frame_indices, visual_shape)

    def _load_demos_from_dataset(
        self,
        cfg: DictConfig,
        num_demos,
        random_traj: bool,
    ) -> list[Demo]:
        selected_episode_indices = self._episode_indices_for_cfg(
            cfg,
            num_demos,
            random_traj=random_traj,
        )
        rows_by_episode = self._row_indices_by_episode(selected_episode_indices)
        images_by_index = self._load_images_for_rows(cfg, rows_by_episode)
        image_key = self._image_key(cfg)

        demos = []
        for episode_index in selected_episode_indices:
            rows = rows_by_episode[episode_index]
            if len(rows) == 0:
                continue
            obs_sequence = []
            actions = []
            rewards = []
            dones = []
            successes = []
            for row in rows:
                image = None
                if cfg.pixels:
                    image_lookup_key = (
                        row
                        if "observation.image" in self._data
                        else int(self._data["index"][row])
                    )
                    image = images_by_index[image_lookup_key]
                obs_sequence.append(
                    _obs_from_raw(
                        self._data["observation.state"][row],
                        image,
                        pixels=bool(cfg.pixels),
                        image_key=image_key,
                    )
                )
                actions.append(np.asarray(self._data["action"][row], dtype=np.float32))
                rewards.append(float(self._data.get("next.reward", [0.0])[row]))
                dones.append(bool(self._data.get("next.done", [False])[row]))
                successes.append(bool(self._data.get("next.success", [False])[row]))

            demo_timesteps = [(obs_sequence[0], {"demo": 1})]
            for index, action in enumerate(actions):
                next_obs = (
                    obs_sequence[index + 1]
                    if index + 1 < len(obs_sequence)
                    else obs_sequence[index]
                )
                is_last = index == len(actions) - 1
                terminated = bool(dones[index]) if is_last else False
                truncated = bool(is_last and not terminated)
                info = {
                    "demo_action": action,
                    "demo": 1,
                    "task_success": int(successes[index]),
                }
                demo_timesteps.append(
                    (
                        next_obs,
                        rewards[index],
                        terminated,
                        truncated,
                        info,
                    )
                )
            demos.append(Demo(demo_timesteps))

        self._selected_episode_indices = selected_episode_indices
        return demos

    def _compute_action_stats(self, demos: list[Demo]) -> dict[str, np.ndarray]:
        actions = []
        for demo in demos:
            for step in demo:
                *_, info = step
                if isinstance(info, dict) and "demo_action" in info:
                    actions.append(info["demo_action"])
        if len(actions) == 0:
            raise ValueError("Push-T demos do not contain any actions.")

        actions = np.stack(actions, axis=0)
        action_stats = {
            "mean": np.mean(actions, axis=0),
            "std": np.std(actions, axis=0),
            "max": np.max(actions, axis=0),
            "min": np.min(actions, axis=0),
        }
        action_stats["std"][action_stats["std"] == 0] = 1.0
        return action_stats

    def _compute_obs_stats(self) -> dict[str, dict[str, np.ndarray]]:
        selected = set(getattr(self, "_selected_episode_indices", self._episode_indices))
        states = np.asarray(
            [
                state
                for state, episode_index in zip(
                    self._data["observation.state"], self._data["episode_index"]
                )
                if int(episode_index) in selected
            ],
            dtype=np.float32,
        )
        if states.size == 0:
            raise ValueError("Push-T demos do not contain low-dimensional observations.")
        obs_stats = {
            "mean": {_DEFAULT_STATE_KEY: np.mean(states, axis=0)},
            "std": {_DEFAULT_STATE_KEY: np.std(states, axis=0)},
            "max": {_DEFAULT_STATE_KEY: np.max(states, axis=0)},
            "min": {_DEFAULT_STATE_KEY: np.min(states, axis=0)},
        }
        obs_stats["std"][_DEFAULT_STATE_KEY][
            obs_stats["std"][_DEFAULT_STATE_KEY] == 0
        ] = 1.0
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
        if return_raw_spaces:
            action_space = copy.deepcopy(env.action_space)
            observation_space = copy.deepcopy(env.observation_space)

        env = self._make_rescale_from_tanh_cls(cfg)(env)

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
                env = ActionSequence(env, cfg.action_sequence)
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
        if bool(_cfg_get(cfg.env, "use_live_env", True)):
            return PushTGymEnv(cfg, image_key=self._image_key(cfg))

        if not self._warned_about_placeholder:
            logging.warning(
                "Push-T is using a placeholder gym env because "
                "env.use_live_env=false. Demo loading works, but online rollouts "
                "and evaluation require gym-pusht."
            )
            self._warned_about_placeholder = True
        return PushTPlaceholderEnv(
            observation_space=self._raw_observation_space,
            action_space=self._raw_action_space,
            reset_observation=self._reset_observation,
            render_shape=tuple(cfg.visual_observation_shape),
        )

    def make_train_env(self, cfg: DictConfig) -> gym.vector.VectorEnv:
        self._ensure_dataset_metadata(cfg)
        env_fns = [
            lambda: self._wrap_env(self._make_base_env(cfg), cfg)
            for _ in range(cfg.num_train_envs)
        ]
        return gym.vector.SyncVectorEnv(
            env_fns
        )

    def get_spaces(self, cfg: DictConfig) -> tuple[gym.Space, gym.Space]:
        self._ensure_dataset_metadata(cfg)
        wrapped_env = self._wrap_env(
            PushTPlaceholderEnv(
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
            num_demos=num_demos,
            random_traj=bool(_cfg_get(cfg.env, "random_traj", False)),
        )
        if _cfg_get(cfg, "use_standardization", False) or _cfg_get(
            cfg, "use_min_max_normalization", False
        ):
            self._action_stats = self._compute_action_stats(self._raw_demos)
        if _cfg_get(cfg, "norm_obs", False):
            self._obs_stats = self._compute_obs_stats()

    def post_collect_or_fetch_demos(self, cfg: DictConfig):
        self._demos = rescale_demo_actions(
            self._rescale_demo_action_helper, self._raw_demos, cfg
        )

    def load_demos_into_replay(
        self, cfg: DictConfig, buffer, is_demo_buffer: bool = False
    ):
        assert hasattr(self, "_demos"), (
            "There's no _demos attribute inside the factory. "
            "Check collect_or_fetch_demos is called before load_demos_into_replay."
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
