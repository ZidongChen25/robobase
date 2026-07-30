from __future__ import annotations

import copy
import logging
import threading
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from bigym.action_modes import JointPositionActionMode, PelvisDof
from bigym.bigym_env import CONTROL_FREQUENCY_MAX
from bigym.utils.observation_config import CameraConfig, ObservationConfig
from demonstrations.demo_store import DemoStore
from demonstrations.utils import Metadata, ObservationMode
from gymnasium import spaces
from omegaconf import DictConfig
from safetensors import safe_open

from robobase.envs.utils.bigym_utils import TASK_MAP
from robobase.envs.wrappers import RescaleFromStandardization, RescaleFromTanhWithMinMax
from robobase.language import (
    clip_text_feature_array,
    clip_tokenize_text,
    tokenize_text,
    tokens_to_feature_array,
)
from robobase.replay_buffer.replay_buffer import ReplayBuffer
from robobase.replay_buffer.uniform_replay_buffer import (
    ACTION,
    ACTION_PAD_MASK,
    DISCOUNT,
    INDICES,
    REWARD,
    TERMINAL,
    TRUNCATED,
)


@dataclass(frozen=True)
class LazyBiGymEpisode:
    uuid: str
    pixel_path: Path
    state_path: Path
    num_obs: int
    num_actions: int
    transition_len: int
    global_start: int
    successful: bool


@dataclass(frozen=True)
class LazyBiGymManifest:
    episodes: tuple[LazyBiGymEpisode, ...]
    action_stats: dict[str, np.ndarray]
    obs_stats: dict[str, dict[str, np.ndarray]]


def lazy_replay_enabled(cfg: DictConfig) -> bool:
    env_cfg = cfg.get("env", {})
    lazy_cfg = cfg.get("lazy_replay", None)
    is_bigym_imitation = bool(
        cfg.get("is_imitation_learning", False)
        and env_cfg is not None
        and str(env_cfg.get("env_name", "")).lower() == "bigym"
    )
    if not is_bigym_imitation or lazy_cfg is None:
        return False

    setting = lazy_cfg.get("use", "auto")
    if isinstance(setting, bool):
        mode = "true" if setting else "false"
    else:
        mode = str(setting).strip().lower()
    if mode in {"false", "0", "off", "no"}:
        return False
    if mode in {"true", "1", "on", "yes"}:
        return True
    if mode != "auto":
        raise ValueError(
            "lazy_replay.use must be one of true, false, or auto; "
            f"got {setting!r}."
        )

    demos = cfg.get("demos", 0)
    try:
        has_demos = float(demos) != 0.0
    except (TypeError, ValueError):
        has_demos = bool(demos)
    return bool(cfg.get("pixels", False) and has_demos)


def _lazy_observation_timing(cfg: DictConfig) -> str:
    timing = str(
        cfg.get("lazy_replay", {}).get(
            "observation_timing",
            cfg.replay.get("observation_timing", "pre_action"),
        )
    ).lower()
    aliases = {
        "pre": "pre_action",
        "pre-action": "pre_action",
        "post": "post_action",
        "post-action": "post_action",
    }
    timing = aliases.get(timing, timing)
    if timing not in {"pre_action", "post_action"}:
        raise ValueError(
            "lazy_replay.observation_timing must be one of "
            "'pre_action' or 'post_action'."
        )
    return timing


def _make_metadata_env(cfg: DictConfig):
    task_class = TASK_MAP[cfg.env.task_name]
    camera_configs = [
        CameraConfig(
            name=camera_name,
            rgb=True,
            depth=False,
            resolution=cfg.visual_observation_shape,
        )
        for camera_name in cfg.env.cameras
    ]
    if cfg.env.enable_all_floating_dof:
        action_mode = JointPositionActionMode(
            absolute=cfg.env.action_mode == "absolute",
            floating_base=True,
            floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ],
        )
    else:
        action_mode = JointPositionActionMode(
            absolute=cfg.env.action_mode == "absolute",
            floating_base=True,
        )
    return task_class(
        render_mode=cfg.env.render_mode,
        action_mode=action_mode,
        observation_config=ObservationConfig(
            cameras=camera_configs if cfg.pixels else [],
            proprioception=True,
            privileged_information=False if cfg.pixels else True,
        ),
        control_frequency=CONTROL_FREQUENCY_MAX // int(cfg.env.demo_down_sample_rate),
    )


def _metadata_dirs(cfg: DictConfig) -> tuple[Path, Path]:
    frequency = CONTROL_FREQUENCY_MAX // int(cfg.env.demo_down_sample_rate)
    env = _make_metadata_env(cfg)
    try:
        metadata = Metadata.from_env(env)
    finally:
        env.close()

    def cache_root(key: str, fallback):
        value = cfg.env.get(key, fallback)
        value = str(value).strip() if value is not None else ""
        return Path(value).expanduser() if value else None

    dataset_root = cfg.env.get("dataset_root", "")
    pixel_root = cache_root("pixel_dataset_root", dataset_root)
    state_root = cache_root("state_dataset_root", dataset_root)
    pixel_store = DemoStore(cache_root=pixel_root)
    state_store = (
        pixel_store
        if state_root == pixel_root
        else DemoStore(cache_root=state_root)
    )
    pixel_metadata = copy.deepcopy(metadata)
    pixel_metadata.observation_mode = ObservationMode.Pixel
    state_metadata = copy.deepcopy(metadata)
    state_metadata.observation_mode = ObservationMode.State
    pixel_dir = pixel_store._create_path(pixel_metadata, frequency).parent
    state_dir = state_store._create_path(state_metadata, frequency).parent
    return pixel_dir, state_dir


def _slice_array(path: Path, key: str, indices=None) -> np.ndarray:
    with safe_open(path, framework="numpy") as handle:
        if indices is None:
            return np.asarray(handle.get_tensor(key))
        tensor_slice = handle.get_slice(key)
        return np.asarray(tensor_slice[indices])


def _tensor_shape(path: Path, key: str) -> list[int]:
    with safe_open(path, framework="numpy") as handle:
        return list(handle.get_slice(key).get_shape())


def _take_rows(tensor_slice, indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    flat = indices.reshape(-1)
    rows = [np.asarray(tensor_slice[int(index)]) for index in flat.tolist()]
    if not rows:
        return np.empty(indices.shape, dtype=np.float32)
    return np.stack(rows, axis=0).reshape(indices.shape + rows[0].shape)


def _state_key(obs_key: str) -> str:
    return f"obs_{obs_key}"


def _rgb_key(camera_name: str) -> str:
    return f"rgb_{camera_name}"


def _camera_intrinsic_obs_key(camera_name: str) -> str:
    return f"camera_intrinsic_{camera_name}"


def _camera_c2w_obs_key(camera_name: str) -> str:
    return f"camera_c2w_{camera_name}"


def _camera_intrinsic_key(camera_name: str) -> str:
    return f"obs_camera_intrinsic_{camera_name}"


def _camera_c2w_key(camera_name: str) -> str:
    return f"obs_camera_c2w_{camera_name}"


def _state_rgb_key(camera_name: str) -> str:
    return f"obs_rgb_{camera_name}"


def _supported_observation_keys(cameras: tuple[str, ...]) -> set[str]:
    keys = {
        "low_dim_state",
        "proprioception_floating_base",
        "proprioception_floating_base_actions",
        "lang_tokens",
        "lang_features",
        "time",
    }
    for camera in cameras:
        keys.update(
            {
                _rgb_key(camera),
                _camera_intrinsic_obs_key(camera),
                _camera_c2w_obs_key(camera),
            }
        )
    return keys


def _load_non_rgb_episode(path: Path) -> dict[str, np.ndarray]:
    arrays = {}
    with safe_open(path, framework="numpy") as handle:
        for key in handle.keys():
            if key.startswith("obs_rgb_"):
                continue
            arrays[key] = np.asarray(handle.get_tensor(key))
    return arrays


def _load_camera_params_episode(path: Path, cameras: tuple[str, ...]) -> dict[str, np.ndarray]:
    arrays = {}
    required_keys = []
    for camera in cameras:
        required_keys.extend((_camera_intrinsic_key(camera), _camera_c2w_key(camera)))
    with safe_open(path, framework="numpy") as handle:
        available_keys = set(handle.keys())
        missing_keys = [key for key in required_keys if key not in available_keys]
        if missing_keys:
            raise ValueError(
                "BiGym lazy replay Plucker conditioning needs per-frame camera "
                f"parameters in {path}; missing {missing_keys}. Rebuild the pixel "
                "demo cache with scripts/cache_bigym_pixel_demos.py "
                "--include-camera-params --force-recache."
            )
        for key in required_keys:
            arrays[key] = np.asarray(handle.get_tensor(key), dtype=np.float32)
    return arrays


def _successful_episode(path: Path) -> bool:
    rewards = _slice_array(path, "reward")
    return float(np.sum(rewards)) > 0.25


def _normalize_demos_count(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"inf", "+inf", ".inf", "+.inf"}:
            return None
        return int(float(normalized))
    if np.isinf(value):
        return None
    return int(value)


def _action_stats(cfg: DictConfig, actions: np.ndarray) -> dict[str, np.ndarray]:
    actions = np.asarray(actions, dtype=np.float32)
    if cfg.use_standardization:
        action_mean = np.mean(actions, 0)
        action_std = np.clip(np.std(actions, 0), 1e-6, np.inf)
        if action_mean.shape[0] >= 2:
            action_mean[-2:] = 0.0
            action_std[-2:] = 1.0
        action_max = np.max(actions, 0)
        action_min = np.min(actions, 0)
    else:
        mean, std, gmax, gmin = (0.5, 0.25, 1.0, 0.0)
        action_mean = np.hstack([np.mean(actions, 0)[:-2], mean, mean])
        action_std = np.hstack([np.std(actions, 0)[:-2], std, std])
        action_max = np.hstack([np.max(actions, 0)[:-2], gmax, gmax])
        action_min = np.hstack([np.min(actions, 0)[:-2], gmin, gmin])
    return {
        "mean": action_mean.astype(np.float32),
        "std": action_std.astype(np.float32),
        "max": action_max.astype(np.float32),
        "min": action_min.astype(np.float32),
    }


def _obs_stats(cfg: DictConfig, obs_arrays: dict[str, list[np.ndarray]]):
    stacked = {
        key: np.concatenate([np.asarray(value) for value in values], axis=0)
        for key, values in obs_arrays.items()
        if values
    }
    obs_mean = {key: np.mean(value, 0) for key, value in stacked.items()}
    obs_std = {
        key: np.clip(np.std(value, 0), 1e-10, np.inf)
        for key, value in stacked.items()
    }
    obs_min = {key: np.min(value, 0) for key, value in stacked.items()}
    obs_max = {key: np.max(value, 0) for key, value in stacked.items()}
    if cfg.obs_norm_type == "standardization":
        if "proprioception" in obs_mean and obs_mean["proprioception"].shape[0] > 0:
            obs_mean["proprioception"][0] = 0.0
            obs_std["proprioception"][0] = 1.0
        if "proprioception_grippers" in obs_mean:
            obs_mean["proprioception_grippers"] = np.zeros_like(
                obs_mean["proprioception_grippers"]
            )
            obs_std["proprioception_grippers"] = np.ones_like(
                obs_std["proprioception_grippers"]
            )
            obs_max["proprioception_grippers"] = np.ones_like(
                obs_max["proprioception_grippers"]
            )
            obs_min["proprioception_grippers"] = np.zeros_like(
                obs_min["proprioception_grippers"]
            )
    return {
        "mean": {key: value.astype(np.float32) for key, value in obs_mean.items()},
        "std": {key: value.astype(np.float32) for key, value in obs_std.items()},
        "max": {key: value.astype(np.float32) for key, value in obs_max.items()},
        "min": {key: value.astype(np.float32) for key, value in obs_min.items()},
    }


def build_bigym_lazy_manifest(cfg: DictConfig, num_demos=None) -> LazyBiGymManifest:
    pixel_dir, state_dir = _metadata_dirs(cfg)
    observation_timing = _lazy_observation_timing(cfg)
    action_index_offset = 1 if observation_timing == "post_action" else 0
    if not pixel_dir.exists():
        raise FileNotFoundError(
            f"BiGym pixel demos are not cached at {pixel_dir}. "
            "Run scripts/cache_bigym_pixel_demos.py first."
        )
    pixel_files = sorted(pixel_dir.glob("*.safetensors"))
    max_demos = _normalize_demos_count(num_demos if num_demos is not None else cfg.demos)
    if max_demos is not None:
        pixel_files = pixel_files[:max_demos]
    if not pixel_files:
        raise ValueError(f"No BiGym pixel demos found in {pixel_dir}.")

    episodes = []
    raw_actions = []
    obs_arrays = defaultdict(list)
    global_start = 0
    for pixel_path in pixel_files:
        state_path = state_dir / pixel_path.name
        metadata_path = state_path if state_path.exists() else pixel_path
        num_obs = int(_tensor_shape(pixel_path, _state_rgb_key(cfg.env.cameras[0]))[0])
        successful = _successful_episode(metadata_path)
        if bool(cfg.env.get("filter_successful_demos", True)) and not successful:
            continue

        episode = _load_non_rgb_episode(metadata_path)
        num_actions = int(episode["info_demo_action"].shape[0])
        transition_len = max(0, min(num_obs - 1, num_actions - action_index_offset))
        if transition_len <= 0:
            continue
        raw_actions.append(
            episode["info_demo_action"][
                action_index_offset : action_index_offset + transition_len
            ]
        )
        for obs_key in (
            "proprioception",
            "proprioception_grippers",
            "proprioception_floating_base",
            "proprioception_floating_base_actions",
        ):
            key = _state_key(obs_key)
            if key in episode:
                obs_arrays[obs_key].append(episode[key])
        episodes.append(
            LazyBiGymEpisode(
                uuid=pixel_path.stem,
                pixel_path=pixel_path,
                state_path=metadata_path,
                num_obs=num_obs,
                num_actions=num_actions,
                transition_len=transition_len,
                global_start=global_start,
                successful=successful,
            )
        )
        global_start += transition_len

    if not episodes:
        raise ValueError("No usable BiGym lazy replay episodes were found.")
    action_stats = _action_stats(cfg, np.concatenate(raw_actions, axis=0))
    obs_stats = _obs_stats(cfg, obs_arrays)
    logging.info(
        "Built lazy BiGym replay manifest task=%s episodes=%d transitions=%d.",
        cfg.env.task_name,
        len(episodes),
        global_start,
    )
    return LazyBiGymManifest(tuple(episodes), action_stats, obs_stats)


class LazyBiGymReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        cfg: DictConfig,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        *,
        batch_size: int,
        extra_replay_elements: spaces.Dict | None = None,
    ):
        self.cfg = cfg
        self.observation_elements = observation_space.spaces
        self._camera_param_cameras = tuple(
            str(camera)
            for camera in cfg.env.cameras
            if _camera_intrinsic_obs_key(str(camera)) in self.observation_elements
            or _camera_c2w_obs_key(str(camera)) in self.observation_elements
        )
        supported_camera_param_keys = {
            key
            for camera in self._camera_param_cameras
            for key in (
                _camera_intrinsic_obs_key(camera),
                _camera_c2w_obs_key(camera),
            )
        }
        camera_param_keys = {
            key
            for key in self.observation_elements
            if str(key).startswith("camera_intrinsic_")
            or str(key).startswith("camera_c2w_")
        }
        missing_or_unsupported_camera_params = [
            key for key in camera_param_keys if key not in supported_camera_param_keys
        ]
        if missing_or_unsupported_camera_params:
            raise ValueError(
                "BiGym lazy replay camera parameter observation keys must match "
                "env.cameras and include both intrinsic/c2w for each conditioned "
                f"camera; unsupported keys {missing_or_unsupported_camera_params}."
            )
        for camera in self._camera_param_cameras:
            missing_pair = [
                key
                for key in (
                    _camera_intrinsic_obs_key(camera),
                    _camera_c2w_obs_key(camera),
                )
                if key not in self.observation_elements
            ]
            if missing_pair:
                raise ValueError(
                    "BiGym lazy replay camera conditioning requires both "
                    f"intrinsic and c2w observations for camera {camera!r}; "
                    f"missing {missing_pair}."
                )
        raymap_keys = [key for key in self.observation_elements if "raymap" in str(key)]
        if raymap_keys:
            raise ValueError(
                "BiGym lazy replay no longer generates raymap observations in CPU "
                "workers. Expose camera_intrinsic_* and camera_c2w_* observations "
                f"instead of {raymap_keys}."
            )
        unsupported_observation_keys = sorted(
            set(self.observation_elements)
            - _supported_observation_keys(tuple(str(camera) for camera in cfg.env.cameras))
        )
        if unsupported_observation_keys:
            raise ValueError(
                "BiGym lazy replay does not know how to populate observation keys "
                f"{unsupported_observation_keys}."
            )
        self.extra_replay_elements = (
            spaces.Dict({}) if extra_replay_elements is None else extra_replay_elements
        )
        self._batch_size = int(batch_size)
        self._action_shape = tuple(action_space.shape[1:])
        self._action_seq_len = int(action_space.shape[0])
        self._frame_stacks = int(cfg.frame_stack)
        self._nstep = int(cfg.replay.nstep)
        self._gamma = float(cfg.replay.gamma)
        self._action_sequence_start_offset = max(
            0, int(cfg.replay.get("action_sequence_start_offset", 0))
        )
        self._observation_timing = _lazy_observation_timing(cfg)
        self._action_index_offset = 1 if self._observation_timing == "post_action" else 0
        self._drop_reset_frame = bool(cfg.lazy_replay.get("drop_reset_frame", False))
        self._action_padding = str(cfg.replay.get("action_padding", "zero")).lower()
        if self._action_padding not in {"zero", "edge", "repeat"}:
            raise ValueError("action_padding must be one of zero, edge, repeat.")

        self._manifest = build_bigym_lazy_manifest(cfg, cfg.demos)
        self._episodes = self._manifest.episodes
        self._starts = np.asarray(
            [episode.global_start for episode in self._episodes], dtype=np.int64
        )
        self._ends = np.asarray(
            [
                episode.global_start + episode.transition_len
                for episode in self._episodes
            ],
            dtype=np.int64,
        )
        self._size = int(self._ends[-1]) if self._ends.size else 0
        self._episode_cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self._episode_cache_lock = threading.RLock()
        self._max_cached_episodes = int(
            cfg.get("lazy_replay", {}).get("max_cached_episodes_per_worker", 2)
        )
        self._include_tp1 = bool(cfg.get("lazy_replay", {}).get("include_tp1", False))

        self._lang_tokens = self._build_lang_tokens()
        self._lang_features = self._build_lang_features()
        logging.info(
            "LazyBiGymReplayBuffer streams %d transitions from %d episode files "
            "(observation_timing=%s).",
            self._size,
            len(self._episodes),
            self._observation_timing,
        )

    @property
    def frame_stack(self):
        return self._frame_stacks

    @property
    def action_seq(self):
        return self._action_seq_len

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sequential(self):
        return False

    @property
    def reused_existing(self) -> bool:
        return True

    def __len__(self):
        return self._size

    def is_empty(self):
        return self._size == 0

    def is_full(self):
        return False

    def replay_capacity(self):
        return self._size

    def _task_description(self) -> str:
        from robobase.envs.bigym import BIGYM_TASK_DESCRIPTIONS, bigym_task_description

        method_cfg = self.cfg.get("method", {})
        explicit_description = method_cfg.get("lang_description", None)
        if explicit_description is not None:
            return str(explicit_description)
        if "lang_feature_source" not in method_cfg:
            return BIGYM_TASK_DESCRIPTIONS.get(
                str(self.cfg.env.task_name),
                "reach the target",
            )
        return bigym_task_description(self.cfg.env.task_name)

    def _lang_feature_source(self) -> str:
        method_cfg = self.cfg.get("method", {})
        # Missing means a legacy config. Those runs used CLIP features; using
        # the new hashed-token path here would also make replay conditioning
        # disagree with the live evaluation environment.
        return str(method_cfg.get("lang_feature_source", "clip")).lower()

    def _lang_feature_device(self) -> str:
        method_cfg = self.cfg.get("method", {})
        return str(method_cfg.get("lang_feature_device", "cpu"))

    def _build_lang_tokens(self) -> np.ndarray:
        if "lang_tokens" not in self.observation_elements:
            return np.zeros((1, 77), dtype=np.int32)
        description = self._task_description()
        if self._lang_feature_source() in {"clip", "clip_text"}:
            return clip_tokenize_text(description)
        return tokenize_text(description).astype(np.int32, copy=False)

    def _build_lang_features(self) -> np.ndarray:
        if "lang_features" not in self.observation_elements:
            return np.zeros((1, 512), dtype=np.float32)
        description = self._task_description()
        if self._lang_feature_source() in {"clip", "clip_text"}:
            return clip_text_feature_array(
                description,
                device=self._lang_feature_device(),
            )
        return tokens_to_feature_array(self._lang_tokens)

    def _cached_episode(self, episode_idx: int) -> dict[str, np.ndarray]:
        with self._episode_cache_lock:
            cached = self._episode_cache.pop(episode_idx, None)
            if cached is not None:
                self._episode_cache[episode_idx] = cached
                return cached
            episode_meta = self._episodes[episode_idx]
            episode = _load_non_rgb_episode(episode_meta.state_path)
            camera_param_cameras = getattr(self, "_camera_param_cameras", ())
            if camera_param_cameras:
                episode.update(
                    _load_camera_params_episode(
                        episode_meta.pixel_path,
                        tuple(camera_param_cameras),
                    )
                )
            self._episode_cache[episode_idx] = episode
            while (
                self._max_cached_episodes >= 0
                and len(self._episode_cache) > self._max_cached_episodes
            ):
                self._episode_cache.popitem(last=False)
            return episode

    def _locate_indices(self, indices: np.ndarray):
        episode_idxs = np.searchsorted(self._ends, indices, side="right")
        if np.any(episode_idxs >= len(self._episodes)):
            raise ValueError("Lazy replay sample index is out of range.")
        local_idxs = indices - self._starts[episode_idxs]
        return episode_idxs.astype(np.int64), local_idxs.astype(np.int64)

    def _normalize_obs(self, key: str, value: np.ndarray) -> np.ndarray:
        value = value.astype(np.float32, copy=False)
        if not bool(self.cfg.norm_obs) or key not in self._manifest.obs_stats["mean"]:
            return value
        if str(self.cfg.obs_norm_type).lower() in {"min_max", "minmax"}:
            obs_min = self._manifest.obs_stats["min"][key]
            obs_max = self._manifest.obs_stats["max"][key]
            obs_range = obs_max - obs_min
            mask = (obs_range != 0).astype(value.dtype, copy=False)
            obs_range = np.where(obs_range == 0, 1.0, obs_range)
            return ((value - obs_min) / obs_range * 2.0 - 1.0) * mask
        return (value - self._manifest.obs_stats["mean"][key]) / (
            self._manifest.obs_stats["std"][key] + 1e-10
        )

    def _transform_actions(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        if self.cfg.use_standardization:
            return RescaleFromStandardization.transform_to_standardization(
                actions,
                self._manifest.action_stats,
            )
        return RescaleFromTanhWithMinMax.transform_to_tanh(
            actions,
            self._manifest.action_stats,
            float(self.cfg.min_max_margin),
        )

    def _time_observations(self, observation_indices: np.ndarray) -> np.ndarray:
        time_space = self.observation_elements["time"]
        if len(time_space.shape) != 2 or time_space.shape[0] != self._frame_stacks:
            raise ValueError(
                "BiGym lazy replay expected time observation shape "
                f"({self._frame_stacks}, T), got {time_space.shape}."
            )
        time_dim = int(time_space.shape[-1])
        time_indices = np.clip(
            observation_indices + self._action_index_offset,
            0,
            time_dim - 1,
        )
        return np.eye(time_dim, dtype=time_space.dtype)[time_indices]

    def episode_index_metadata(self) -> list[tuple[int, int]]:
        return [
            (episode.global_start, episode.transition_len)
            for episode in self._episodes
        ]

    def load_all_episodes(self):
        logging.info("LazyBiGymReplayBuffer ignores load_all_episodes().")

    def sample(self, batch_size=None, indices=None):
        if indices is None:
            batch_size = self._batch_size if batch_size is None else int(batch_size)
            indices = np.random.randint(0, self._size, size=batch_size)
        return self.sample_batch_indices(indices)

    def sample_batch_indices(self, indices: np.ndarray | list[int]) -> dict:
        indices = np.asarray(indices, dtype=np.int64)
        batch_size = int(indices.shape[0])
        episode_idxs, local_idxs = self._locate_indices(indices)
        batch = {
            ACTION: np.empty(
                (batch_size, self._action_seq_len, *self._action_shape),
                dtype=np.float32,
            ),
            ACTION_PAD_MASK: np.zeros(
                (batch_size, self._action_seq_len),
                dtype=np.bool_,
            ),
            REWARD: np.zeros((batch_size,), dtype=np.float32),
            TERMINAL: np.zeros((batch_size,), dtype=np.bool_),
            TRUNCATED: np.zeros((batch_size,), dtype=np.bool_),
            DISCOUNT: np.full((batch_size,), self._gamma, dtype=np.float64),
            INDICES: indices.copy(),
            "demo": np.ones((batch_size,), dtype=np.uint8),
        }

        for key, space in self.observation_elements.items():
            batch[key] = np.empty((batch_size, *space.shape), dtype=space.dtype)
            if self._include_tp1:
                batch[key + "_tp1"] = np.empty_like(batch[key])

        obs_offsets = np.arange(-(self._frame_stacks - 1), 1, dtype=np.int64)
        action_offsets = np.arange(self._action_seq_len, dtype=np.int64)
        for episode_idx in np.unique(episode_idxs):
            mask = episode_idxs == episode_idx
            out_idxs = np.nonzero(mask)[0]
            idxs = local_idxs[mask]
            episode_meta = self._episodes[int(episode_idx)]
            episode = self._cached_episode(int(episode_idx))
            ep_len = int(episode_meta.transition_len)

            obs_shift = 1 if getattr(self, "_drop_reset_frame", False) else 0
            obs_idxs = np.clip(
                idxs[:, None] + obs_shift + obs_offsets[None, :], 0, ep_len
            )
            next_idxs = np.clip(
                idxs[:, None] + obs_shift + self._nstep + obs_offsets[None, :],
                0,
                ep_len,
            )
            rgb_cameras = [
                camera
                for camera in self.cfg.env.cameras
                if _rgb_key(camera) in self.observation_elements
            ]
            if rgb_cameras:
                with safe_open(episode_meta.pixel_path, framework="numpy") as handle:
                    for camera in rgb_cameras:
                        key = _rgb_key(camera)
                        tensor_key = _state_rgb_key(camera)
                        tensor = handle.get_slice(tensor_key)
                        batch[key][out_idxs] = _take_rows(tensor, obs_idxs)
                        if self._include_tp1:
                            batch[key + "_tp1"][out_idxs] = _take_rows(
                                tensor, next_idxs
                            )

            for camera in self.cfg.env.cameras:
                intrinsic_obs_key = _camera_intrinsic_obs_key(camera)
                c2w_obs_key = _camera_c2w_obs_key(camera)
                if intrinsic_obs_key in batch:
                    batch[intrinsic_obs_key][out_idxs] = episode[
                        _camera_intrinsic_key(camera)
                    ][obs_idxs]
                    if self._include_tp1:
                        batch[intrinsic_obs_key + "_tp1"][out_idxs] = episode[
                            _camera_intrinsic_key(camera)
                        ][next_idxs]
                if c2w_obs_key in batch:
                    batch[c2w_obs_key][out_idxs] = episode[_camera_c2w_key(camera)][
                        obs_idxs
                    ]
                    if self._include_tp1:
                        batch[c2w_obs_key + "_tp1"][out_idxs] = episode[
                            _camera_c2w_key(camera)
                        ][next_idxs]

            low_dim_parts = []
            for obs_key in ("proprioception", "proprioception_grippers"):
                tensor_key = _state_key(obs_key)
                if tensor_key not in episode:
                    continue
                values = self._normalize_obs(obs_key, episode[tensor_key][obs_idxs])
                low_dim_parts.append(values)
            if low_dim_parts and "low_dim_state" in batch:
                batch["low_dim_state"][out_idxs] = np.concatenate(low_dim_parts, axis=-1)
                if self._include_tp1:
                    batch["low_dim_state_tp1"][out_idxs] = np.concatenate(
                        [
                            self._normalize_obs(
                                obs_key,
                                episode[_state_key(obs_key)][next_idxs],
                            )
                            for obs_key in (
                                "proprioception",
                                "proprioception_grippers",
                            )
                            if _state_key(obs_key) in episode
                        ],
                        axis=-1,
                    )
            elif "low_dim_state" in batch:
                raise ValueError(
                    "BiGym lazy replay could not populate low_dim_state because the "
                    "cached episode has no proprioception observations."
                )

            for obs_key in (
                "proprioception_floating_base",
                "proprioception_floating_base_actions",
            ):
                if obs_key not in batch:
                    continue
                tensor_key = _state_key(obs_key)
                if tensor_key not in episode:
                    raise ValueError(
                        "BiGym lazy replay could not populate observation "
                        f"{obs_key!r}; cached tensor {tensor_key!r} is missing."
                    )
                batch[obs_key][out_idxs] = episode[tensor_key][obs_idxs].astype(
                    np.float32,
                    copy=False,
                )
                if self._include_tp1:
                    batch[obs_key + "_tp1"][out_idxs] = episode[tensor_key][
                        next_idxs
                    ].astype(np.float32, copy=False)

            if "lang_tokens" in batch:
                batch["lang_tokens"][out_idxs] = self._lang_tokens[None, :, :]
                if self._include_tp1:
                    batch["lang_tokens_tp1"][out_idxs] = self._lang_tokens[None, :, :]
            if "lang_features" in batch:
                batch["lang_features"][out_idxs] = self._lang_features[None, :, :]
                if self._include_tp1:
                    batch["lang_features_tp1"][out_idxs] = (
                        self._lang_features[None, :, :]
                    )
            if "time" in batch:
                batch["time"][out_idxs] = self._time_observations(obs_idxs)
                if self._include_tp1:
                    batch["time_tp1"][out_idxs] = self._time_observations(next_idxs)

            action_start_idxs = (
                idxs + self._action_index_offset - self._action_sequence_start_offset
            )
            action_idxs = action_start_idxs[:, None] + action_offsets[None, :]
            num_actions = int(episode_meta.num_actions)
            invalid_action_idxs = (action_idxs < 0) | (action_idxs >= num_actions)
            clipped_action_idxs = np.clip(action_idxs, 0, max(num_actions - 1, 0))
            actions = episode["info_demo_action"][clipped_action_idxs]
            actions = self._transform_actions(actions)
            if self._action_padding == "zero":
                actions = actions.copy()
                actions[invalid_action_idxs] = 0
                batch[ACTION_PAD_MASK][out_idxs] = invalid_action_idxs
            batch[ACTION][out_idxs] = actions

            reward_idxs = np.clip(idxs + self._nstep, 0, ep_len)
            batch[REWARD][out_idxs] = episode.get("reward", np.zeros(ep_len + 1))[
                reward_idxs
            ].astype(np.float32, copy=False)
            batch[TERMINAL][out_idxs] = episode.get(
                "termination",
                np.zeros(ep_len + 1, dtype=np.bool_),
            )[reward_idxs]
            batch[TRUNCATED][out_idxs] = episode.get(
                "truncation",
                np.zeros(ep_len + 1, dtype=np.bool_),
            )[reward_idxs]
        return batch
