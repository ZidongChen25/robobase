from bigym.bigym_env import BiGymEnv, CONTROL_FREQUENCY_MAX
from bigym.action_modes import JointPositionActionMode
from robobase.utils import DemoEnv, add_demo_to_replay_buffer
from robobase.envs.utils.bigym_utils import (
    TASK_MAP,
    build_actuated_qpos_state,
    build_actuated_qpos_stats,
)
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from robobase.envs.env import EnvFactory
from robobase.envs.wrappers import (
    RescaleFromTanhWithMinMax,
    RescaleFromStandardization,
    OnehotTime,
    ActionSequence,
    AppendDemoInfo,
    FrameStack,
    ConcatDim,
    AppendKeysToLowDim,
    RecedingHorizonControl,
    RawProprioDropout,
    maybe_delay_observations,
)
from robobase.envs.camera_conditioning import (
    camera_conditioning_enabled,
    intrinsic_from_fovy,
)
from robobase.language import load_precomputed_language_features, tokenize_text
from omegaconf import DictConfig
from bigym.utils.observation_config import ObservationConfig, CameraConfig
from bigym.action_modes import PelvisDof
import logging
import numpy as np

from demonstrations.demo import DemoStep
from demonstrations.demo_store import DemoStore
from demonstrations.utils import Metadata

from typing import List, Dict, Tuple, Callable
import copy

UNIT_TEST = False


BIGYM_TASK_DESCRIPTIONS = {
    "move_plate": "Move the plate between two draining racks.",
    "MovePlate": "Move the plate between two draining racks.",
    "flip_cup": "Flip the cup upright.",
    "FlipCup": "Flip the cup upright.",
    "dishwasher_load_cups": "Load the cups into the dishwasher.",
    "DishwasherLoadCups": "Load the cups into the dishwasher.",
    "put_cups": "Put the cups away.",
    "PutCups": "Put the cups away.",
    "sandwich_remove": "Remove the sandwich from the toaster.",
    "RemoveSandwich": "Remove the sandwich from the toaster.",
    "dishwasher_open": "Open the dishwasher door.",
    "DishwasherOpen": "Open the dishwasher door.",
}


def bigym_task_description(task_name: str) -> str:
    task_name = str(task_name)
    description = BIGYM_TASK_DESCRIPTIONS.get(task_name)
    if description is not None:
        return description
    description = " ".join(task_name.replace("_", " ").split())
    if not description:
        return "Reach the target."
    return f"{description.capitalize()}."


def _episode_limit_steps(cfg: DictConfig) -> int:
    episode_length = int(cfg.env.episode_length)
    if bool(cfg.env.get("episode_length_is_env_steps", False)):
        return episode_length
    return episode_length // int(cfg.env.demo_down_sample_rate)


def _demo_observation_array(value):
    array = np.asarray(value)
    if array.dtype == np.uint8:
        return array
    return array.astype(np.float32, copy=False)


class AddBiGymExecutedActionFeedback(gym.ObservationWrapper):
    """Expose measured H1 actuator positions in policy action coordinates."""

    KEY = "executed_action_feedback"

    def __init__(self, env: gym.Env, *, obs_stats, norm_obs, norm_type):
        gym.ObservationWrapper.__init__(self, env)
        self._feedback_stats = build_actuated_qpos_stats(obs_stats)
        self._norm_obs = bool(norm_obs)
        self._norm_type = str(norm_type).lower()
        obs_spaces = dict(self.observation_space.spaces)
        obs_spaces[self.KEY] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.action_space.shape,
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(obs_spaces)

    def observation(self, observation):
        obs = dict(observation)
        feedback = build_actuated_qpos_state(
            obs["proprioception"],
            obs["proprioception_floating_base"],
            obs["proprioception_grippers"],
        )
        if feedback.shape != self.action_space.shape:
            raise ValueError(
                "BiGym executed-action feedback must match action shape; got "
                f"{feedback.shape} and {self.action_space.shape}."
            )
        if self._norm_obs and self._norm_type in {"min_max", "minmax"}:
            minimum = self._feedback_stats["min"]
            value_range = self._feedback_stats["max"] - minimum
            nonconstant = value_range != 0
            value_range = np.where(nonconstant, value_range, 1.0)
            feedback = np.where(
                nonconstant, (feedback - minimum) / value_range * 2.0 - 1.0, 0.0
            )
        elif self._norm_obs:
            feedback = (feedback - self._feedback_stats["mean"]) / (
                self._feedback_stats["std"] + 1e-10
            )
        obs[self.KEY] = feedback.astype(np.float32, copy=False)
        return obs


class AddBiGymLanguageTokens(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        task_name: str,
        *,
        lang_feature_source: str = "tokens",
        lang_feature_device: str = "cpu",
        lang_feature_path=None,
        lang_feature_dim: int = 512,
        description: str | None = None,
    ):
        gym.ObservationWrapper.__init__(self, env)
        description = (
            bigym_task_description(task_name)
            if description is None
            else str(description)
        )
        del lang_feature_device
        source = str(lang_feature_source).lower()
        if source in {"tokens", "jax", "hash"}:
            self._lang_tokens = tokenize_text(description).astype(np.int32, copy=False)
            self._lang_features = None
        elif source in {"clip", "clip_text"}:
            # Pre-06a61d4 checkpoints trained on CLIP token ids.
            from robobase.language import clip_tokenize_text

            self._lang_tokens = clip_tokenize_text(description)
            self._lang_features = None
        elif source == "precomputed":
            self._lang_tokens = tokenize_text(description).astype(np.int32, copy=False)
            self._lang_features = load_precomputed_language_features(
                lang_feature_path,
                feature_dim=lang_feature_dim,
            )
        else:
            raise ValueError(
                "The JAX-only runtime supports lang_feature_source='tokens' "
                "(aliases: 'jax', 'hash') or 'precomputed'; "
                f"got {lang_feature_source!r}."
            )
        obs_spaces = dict(self.observation_space.spaces)
        obs_spaces["lang_tokens"] = spaces.Box(
            low=0,
            high=np.iinfo(np.int32).max,
            shape=(1, 77),
            dtype=np.int32,
        )
        if self._lang_features is not None:
            obs_spaces["lang_features"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=self._lang_features.shape,
                dtype=np.float32,
            )
        self.observation_space = spaces.Dict(obs_spaces)

    def observation(self, observation):
        obs = dict(observation)
        obs["lang_tokens"] = self._lang_tokens.copy()
        if self._lang_features is not None:
            obs["lang_features"] = self._lang_features.copy()
        return obs


class AddBiGymCameraConditioning(gym.ObservationWrapper):
    """Expose per-frame camera parameters for model-side Plucker generation."""

    def __init__(
        self,
        env: gym.Env,
        cameras: list[str],
        image_size: tuple[int, int],
    ):
        gym.ObservationWrapper.__init__(self, env)
        self._cameras = tuple(str(camera) for camera in cameras)
        self._image_size = tuple(int(dim) for dim in image_size)
        obs_spaces = dict(self.observation_space.spaces)
        for camera in self._cameras:
            obs_spaces[f"camera_intrinsic_{camera}"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(3, 3),
                dtype=np.float32,
            )
            obs_spaces[f"camera_c2w_{camera}"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(4, 4),
                dtype=np.float32,
            )
        self.observation_space = spaces.Dict(obs_spaces)

    def _find_camera_id(self, camera_name: str) -> int:
        physics = self.env.unwrapped.mojo.physics
        try:
            return int(physics.model.name2id(camera_name, "camera"))
        except Exception:
            for camera_id in range(int(physics.model.ncam)):
                name = physics.model.id2name(camera_id, "camera")
                if name == camera_name or str(name).endswith(f"/{camera_name}"):
                    return int(camera_id)
        raise ValueError(f"Camera {camera_name!r} not found in BiGym physics model.")

    def _camera_params(self, camera_name: str) -> tuple[np.ndarray, np.ndarray]:
        physics = self.env.unwrapped.mojo.physics
        camera_id = self._find_camera_id(camera_name)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 3] = np.asarray(physics.data.cam_xpos[camera_id], dtype=np.float32)
        c2w[:3, :3] = np.asarray(
            physics.data.cam_xmat[camera_id],
            dtype=np.float32,
        ).reshape(3, 3)
        height, width = self._image_size
        fovy = float(physics.model.cam_fovy[camera_id])
        return intrinsic_from_fovy(fovy, height, width), c2w

    def observation(self, observation):
        obs = dict(observation)
        for camera in self._cameras:
            intrinsic_key = f"camera_intrinsic_{camera}"
            c2w_key = f"camera_c2w_{camera}"
            has_intrinsic = intrinsic_key in obs
            has_c2w = c2w_key in obs
            if has_intrinsic != has_c2w:
                raise ValueError(
                    "BiGym camera conditioning requires paired intrinsic/c2w "
                    f"values for camera {camera!r}."
                )
            if has_intrinsic:
                intrinsic = np.asarray(obs[intrinsic_key], dtype=np.float32)
                c2w = np.asarray(obs[c2w_key], dtype=np.float32)
                if intrinsic.shape != (3, 3) or c2w.shape != (4, 4):
                    raise ValueError(
                        f"Invalid cached camera parameters for {camera!r}: "
                        f"intrinsic={intrinsic.shape}, c2w={c2w.shape}."
                    )
                if not np.all(np.isfinite(intrinsic)) or not np.all(np.isfinite(c2w)):
                    raise ValueError(
                        f"Cached camera parameters for {camera!r} must be finite."
                    )
            else:
                if not hasattr(self.env.unwrapped, "mojo"):
                    raise ValueError(
                        "BiGym demonstration camera conditioning needs cached "
                        "per-frame camera parameters. Rebuild the pixel demos with "
                        "scripts/cache_bigym_pixel_demos.py --include-camera-params "
                        "--force-recache."
                    )
                intrinsic, c2w = self._camera_params(camera)
            obs[intrinsic_key] = intrinsic
            obs[c2w_key] = c2w
        return obs


def rescale_demo_actions(
    rescale_fn: Callable, demos: List[List[DemoStep]], cfg: DictConfig
):
    """Rescale actions in demonstrations to [-1, 1] Tanh space.
    This is because RoboBase assumes everything to be in [-1, 1] space.

    Args:
        rescale_fn: callable that takes info containing demo action and cfg and
            outputs the rescaled action
        demos: list of demo episodes whose actions are raw, i.e., not scaled
        cfg: Configs

    Returns:
        List[Demo]: list of demo episodes whose actions are rescaled
    """
    for demo in demos:
        for step in demo:
            info = step.info
            if "demo_action" in info:
                # Rescale demo actions
                info["demo_action"] = rescale_fn(info, cfg)
    return demos


def _task_name_to_env_class(task_name: str) -> type[BiGymEnv]:
    return TASK_MAP[task_name]


class BiGymEnvFactory(EnvFactory):
    @staticmethod
    def _demo_success(demo) -> bool:
        return sum(float(step.reward) for step in demo.timesteps) > 0.25

    @staticmethod
    def _training_timesteps(demo):
        for step in demo.timesteps:
            yield step

    def _wrap_env(self, env, cfg, demo_env=False, train=True, return_raw_spaces=False):
        # last two are grippers
        assert cfg.demos != 0
        assert cfg.action_repeat == 1

        action_space = copy.deepcopy(env.action_space)
        observation_space = copy.deepcopy(env.observation_space)

        if cfg.use_standardization:
            env = RescaleFromStandardization(
                env=env,
                action_stats=self._action_stats,
            )
        else:
            env = RescaleFromTanhWithMinMax(
                env=env,
                action_stats=self._action_stats,
                min_max_margin=cfg.min_max_margin,
            )
        source_cfg = cfg.method.get("flow_source", {})
        if str(source_cfg.get("history_source", "commanded_action")).lower() == (
            "executed_action_feedback"
        ):
            env = AddBiGymExecutedActionFeedback(
                env,
                obs_stats=self._obs_stats,
                norm_obs=cfg.norm_obs,
                norm_type=cfg.obs_norm_type,
            )
        obs_stats = None
        if cfg.norm_obs:
            obs_stats = self._obs_stats

        proprio_dropout_stage = str(
            cfg.method.get("proprio_dropout_stage", "model")
        ).lower()
        if proprio_dropout_stage not in {"model", "raw"}:
            raise ValueError("method.proprio_dropout_stage must be 'model' or 'raw'.")
        if proprio_dropout_stage == "raw":
            probability = float(cfg.method.get("proprio_dropout_prob", 0.0))
            if probability not in {0.0, 1.0}:
                raise ValueError(
                    "BiGym offline raw proprio dropout supports probabilities 0 or 1; "
                    "fractional dropout must be implemented in the replay sampler."
                )
            if probability > 0.0:
                env = RawProprioDropout(
                    env,
                    keys=tuple(
                        str(key)
                        for key in cfg.method.get(
                            "proprio_dropout_keys", ["proprioception"]
                        )
                    ),
                    probability=probability,
                )

        # We normalize the low dimensional observations in the ConcatDim wrapper.
        # This is to be consistent with the original ACT implementation.
        env = ConcatDim(
            env,
            shape_length=1,
            dim=-1,
            new_name="low_dim_state",
            norm_obs=cfg.norm_obs,
            obs_stats=obs_stats,
            obs_norm_type=cfg.obs_norm_type,
            keys_to_ignore=[
                "proprioception_floating_base",
                "proprioception_floating_base_actions",
                AddBiGymExecutedActionFeedback.KEY,
            ],
        )
        if bool(cfg.env.get("append_floating_base_to_low_dim", False)):
            # Stage-160: give the floating-base state a fixed position (the
            # last dims of low_dim_state) so training-time low-dim masking
            # can zero everything except it.
            env = AppendKeysToLowDim(
                env,
                keys=["proprioception_floating_base"],
                norm_obs=cfg.norm_obs,
                obs_stats=obs_stats,
                obs_norm_type=cfg.obs_norm_type,
            )
        if bool(cfg.get("pixels", False)) and camera_conditioning_enabled(
            cfg.env.get("camera_conditioning", "none")
        ):
            env = AddBiGymCameraConditioning(
                env,
                cameras=list(cfg.env.cameras),
                image_size=tuple(cfg.visual_observation_shape),
            )
        if bool(cfg.method.get("use_lang_cond", False)):
            lang_feature_source = cfg.method.get("lang_feature_source", None)
            lang_description = cfg.method.get("lang_description", None)
            if lang_feature_source is None:
                # Era fingerprint (see _compute_obs_stats): the hash tokenizer
                # replaced clip.tokenize in the same commit that introduced
                # env.truncate_demo_at_success (06a61d4, 2026-07-30). Saved
                # configs lacking that key predate the switch and their
                # checkpoints expect CLIP token ids.
                lang_feature_source = (
                    "tokens" if "truncate_demo_at_success" in cfg.env else "clip"
                )
                if lang_description is None:
                    lang_description = BIGYM_TASK_DESCRIPTIONS.get(
                        str(cfg.env.task_name),
                        "reach the target",
                    )
            env = AddBiGymLanguageTokens(
                env,
                cfg.env.task_name,
                lang_feature_source=lang_feature_source,
                lang_feature_device=cfg.method.get("lang_feature_device", "cpu"),
                lang_feature_path=cfg.method.get("lang_feature_path", None),
                lang_feature_dim=int(cfg.method.get("lang_feature_dim", 512)),
                description=lang_description,
            )
        episode_limit_steps = _episode_limit_steps(cfg)
        if cfg.use_onehot_time_and_no_bootstrap:
            env = OnehotTime(env, episode_limit_steps)
        if not demo_env:
            env = FrameStack(env, cfg.frame_stack)
        # Delayed-policy conditioning. Applied to the demo env too so imported
        # demos land in replay already paired as (o_{t-h}, a_t), and kept inside
        # the action-sequence wrapper so h counts environment steps.
        env = maybe_delay_observations(env, cfg)
        env = TimeLimit(env, episode_limit_steps)

        if not demo_env:
            if int(cfg.execution_length) == int(cfg.action_sequence):
                env = ActionSequence(
                    env,
                    cfg.action_sequence,
                )
            else:
                env = RecedingHorizonControl(
                    env,
                    cfg.action_sequence,
                    episode_limit_steps,
                    cfg.execution_length,
                    temporal_ensemble=cfg.temporal_ensemble,
                    gain=cfg.temporal_ensemble_gain,
                    execution_start=cfg.get("action_execution_start", 0),
                )

        env = AppendDemoInfo(env)

        if return_raw_spaces:
            return env, action_space, observation_space
        else:
            return env

    def _create_env(self, cfg: DictConfig) -> BiGymEnv:
        bigym_class = _task_name_to_env_class(cfg.env.task_name)
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

        return bigym_class(
            render_mode=cfg.env.render_mode,
            action_mode=action_mode,
            observation_config=ObservationConfig(
                cameras=camera_configs if cfg.pixels else [],
                proprioception=True,
                privileged_information=False if cfg.pixels else True,
            ),
            control_frequency=CONTROL_FREQUENCY_MAX // cfg.env.demo_down_sample_rate,
        )

    def make_train_env(self, cfg: DictConfig) -> gym.vector.VectorEnv:
        vec_env_class = gym.vector.SyncVectorEnv
        return vec_env_class(
            [
                lambda: self._wrap_env(
                    self._create_env(cfg),
                    cfg,
                    demo_env=False,
                    train=True,
                )
                for _ in range(cfg.num_train_envs)
            ],
        )

    def make_eval_envs(self, cfg: DictConfig) -> gym.vector.VectorEnv:
        vec_env_class = gym.vector.SyncVectorEnv
        return vec_env_class(
            [
                lambda: self._wrap_env(
                    self._create_env(cfg),
                    cfg,
                    demo_env=False,
                    train=False,
                )
                for _ in range(cfg.num_eval_envs)
            ],
        )

    def make_eval_env(self, cfg: DictConfig) -> gym.Env:
        env, self._action_space, self._observation_space = self._wrap_env(
            env=self._create_env(cfg),
            cfg=cfg,
            demo_env=False,
            train=False,
            return_raw_spaces=True,
        )
        return env

    def _get_demo_fn(self, cfg: DictConfig, num_demos: int):
        demos = []

        logging.info("Start to load demos.")
        env = self._create_env(cfg)

        demo_store = DemoStore()
        if np.isinf(num_demos):
            num_demos = -1

        demos = demo_store.get_demos(
            Metadata.from_env(env),
            amount=num_demos,
            frequency=CONTROL_FREQUENCY_MAX // cfg.env.demo_down_sample_rate,
        )
        camera_wrapper = None
        if bool(cfg.get("pixels", False)) and camera_conditioning_enabled(
            cfg.env.get("camera_conditioning", "none")
        ):
            camera_wrapper = AddBiGymCameraConditioning(
                env,
                cameras=list(cfg.env.cameras),
                image_size=tuple(cfg.visual_observation_shape),
            )

        demo_alignment = str(cfg.env.get("demo_alignment", "reset_prepend")).lower()
        for demo in demos:
            timesteps = list(demo.timesteps)
            if demo_alignment in {
                "reference_post_action",
                "mobile_genima",
                "post_action_previous_action",
            }:
                # Match controller/env/bigym_utils.py in Mobile-GENIMA:
                # generated data.pkl stores observations after env.step(action),
                # then the loader uses raw_obs[i] with raw_actions[i - 1].
                original_actions = [
                    np.asarray(ts.info["demo_action"], dtype=np.float32).copy()
                    for ts in timesteps
                ]
                for i, ts in enumerate(timesteps):
                    ts.observation = {
                        k: _demo_observation_array(v) for k, v in ts.observation.items()
                    }
                    ts.info = dict(ts.info)
                    if i == 0:
                        ts.info.pop("demo_action", None)
                    else:
                        ts.info["demo_action"] = original_actions[i - 1]
                demo._steps = timesteps
            else:
                # DemoStore timesteps are post-action observations; prepend reset obs
                # so the first replay transition is reset_obs -> first demo action.
                reset_obs, _ = env.reset(seed=demo.seed)
                if camera_wrapper is not None:
                    reset_obs = camera_wrapper.observation(reset_obs)
                reset_obs = {
                    k: _demo_observation_array(v) for k, v in reset_obs.items()
                }
                reset_step = DemoStep(
                    reset_obs,
                    0.0,
                    False,
                    False,
                    {"demo": 1},
                    np.zeros(env.action_space.shape, dtype=np.float32),
                )
                reset_step.info.pop("demo_action", None)
                demo._steps = [reset_step] + timesteps
                for ts in demo.timesteps:
                    ts.observation = {
                        k: _demo_observation_array(v) for k, v in ts.observation.items()
                    }

        env.close()
        logging.info("Finished loading demos.")
        return demos

    def collect_or_fetch_demos(self, cfg: DictConfig, num_demos: int):
        if bool(cfg.get("pixels", False)) and camera_conditioning_enabled(
            cfg.env.get("camera_conditioning", "none")
        ):
            lazy_setting = cfg.get("lazy_replay", {}).get("use", "auto")
            lazy_disabled = (
                not lazy_setting
                if isinstance(lazy_setting, bool)
                else str(lazy_setting).strip().lower() in {"false", "0", "off", "no"}
            )
            if lazy_disabled:
                raise ValueError(
                    "BiGym camera-conditioned demonstrations require lazy replay "
                    "with cached per-frame camera parameters. Set "
                    "lazy_replay.use=auto or true and rebuild pixel demos with "
                    "scripts/cache_bigym_pixel_demos.py --include-camera-params "
                    "--force-recache."
                )
        demos = self._get_demo_fn(cfg, num_demos)
        self._all_raw_demos = demos
        success_flags = [self._demo_success(demo) for demo in demos]
        successful_count = sum(success_flags)
        logging.info(
            "Loaded %d BiGym demonstrations; %d are successful.",
            len(demos),
            successful_count,
        )

        expected_successful = cfg.env.get("expected_successful_demos", None)
        if expected_successful is not None and successful_count != int(
            expected_successful
        ):
            raise ValueError(
                "BiGym successful demonstration count mismatch: "
                f"expected {expected_successful}, got {successful_count}."
            )

        if bool(cfg.env.get("filter_successful_demos", True)):
            demos = [
                demo for demo, successful in zip(demos, success_flags) if successful
            ]
            if not demos:
                raise ValueError("No successful BiGym demonstrations were found.")
            logging.info("Keeping %d successful BiGym demonstrations.", len(demos))

        self._raw_demos = demos
        self._action_stats = self._compute_action_stats(cfg, demos)
        self._obs_stats = self._compute_obs_stats(cfg, demos)

    def prepare_lazy_replay(self, cfg: DictConfig, num_demos: int):
        from robobase.replay_buffer.bigym_lazy_replay import (
            build_bigym_lazy_manifest,
        )

        manifest = build_bigym_lazy_manifest(cfg, num_demos)
        self._lazy_replay_manifest = manifest
        self._action_stats = manifest.action_stats
        self._obs_stats = manifest.obs_stats

    def post_collect_or_fetch_demos(self, cfg: DictConfig):
        demo_list = [demo.timesteps for demo in self._raw_demos]
        demo_list = rescale_demo_actions(
            self._rescale_demo_action_helper, demo_list, cfg
        )
        self._demos = self._demo_to_steps(cfg, demo_list)

    def load_demos_into_replay(self, cfg: DictConfig, buffer, is_demo_buffer):
        """See base class for documentation."""
        assert hasattr(self, "_demos"), (
            "There's no _demo attribute inside the factory, "
            "Check `collect_or_fetch_demos` is called before calling this method."
        )

        if is_demo_buffer:
            # Keep this guard so older cached factories that were not filtered during
            # collection still only load successful demonstrations for IL/demo replay.
            demos = []
            for i, demo in enumerate(self._demos):
                successful = demo[0][-1]["demo"] == 1
                if successful:
                    demos.append(demo)
                else:
                    logging.info("Skipping failed demonstration %d.", i)
        else:
            demos = self._demos

        demo_env = self._wrap_env(
            DemoEnv(
                [list(demo) for demo in demos],
                self._action_space,
                self._observation_space,
            ),
            cfg,
            demo_env=True,
            train=False,
        )
        for _ in range(len(demos)):
            add_demo_to_replay_buffer(demo_env, buffer)

    def _demo_to_steps(
        self, cfg: DictConfig, demo_list: List[List[DemoStep]]
    ) -> List[DemoStep]:
        ret_demos = []
        truncate_at_success = bool(
            cfg.env.get("truncate_demo_at_success", False)
        )

        for demo in demo_list:
            cur_demo = []
            last_timestep = False

            # Detect whether this demo is successful or not
            rewards = []
            for step in demo:
                reward = step.reward
                rewards.append(reward)
            successful_demo = sum(rewards) > 0.25
            if bool(cfg.env.get("treat_all_demos_as_expert", False)):
                # Unlabeled-demo-quality regime: the practitioner has no
                # success labels, so every demonstration is marked as
                # expert. Imitation objectives then imitate failed demos
                # too; reward-derived objectives keep reading the true
                # returns. Rewards themselves are never altered.
                successful_demo = True

            for i, step in enumerate(demo):
                step.info.update({"demo": int(successful_demo)})
                if i == 0:
                    cur_demo.append((step.observation, step.info))
                else:
                    term, trunc = step.termination, step.truncation
                    reward = step.reward
                    if truncate_at_success and reward > 0.25:
                        # Match the live BiGym MDP, whose terminate property is
                        # true immediately on success. This also prevents a
                        # recorded post-success tail from contributing the same
                        # sparse reward dozens of times.
                        term = True
                        trunc = False
                        last_timestep = True
                    elif i == len(demo) - 1:
                        if not (term or trunc):
                            term = False
                            trunc = True
                        last_timestep = True
                    else:
                        term = False
                        trunc = False

                    cur_demo.append((step.observation, reward, term, trunc, step.info))
                if last_timestep:
                    break
            ret_demos.append(cur_demo)

        return ret_demos

    def _compute_action_stats(
        self, cfg: DictConfig, demos: List[List[DemoStep]]
    ) -> Dict:
        actions = []
        for demo in demos:
            for step in self._training_timesteps(demo):
                info = step.info
                if "demo_action" in info:
                    actions.append(info["demo_action"])
        actions = np.stack(actions)

        if cfg.use_standardization:
            action_mean = np.mean(actions, 0)
            action_std = np.std(actions, 0)
            action_std = np.clip(action_std, 1e-6, np.inf)
            if action_mean.shape[0] >= 2:
                action_mean[-2:] = 0.0
                action_std[-2:] = 1.0
            action_max = np.max(actions, 0)
            action_min = np.min(actions, 0)
        else:
            mean, std, gmax, gmin = self._get_gripper_action_stats(cfg)
            action_mean = np.hstack([np.mean(actions, 0)[:-2], mean, mean])
            action_std = np.hstack([np.std(actions, 0)[:-2], std, std])
            action_max = np.hstack([np.max(actions, 0)[:-2], gmax, gmax])
            action_min = np.hstack([np.min(actions, 0)[:-2], gmin, gmin])
        action_stats = {
            "mean": action_mean,
            "std": action_std,
            "max": action_max,
            "min": action_min,
        }
        return action_stats

    def _compute_obs_stats(self, cfg: DictConfig, demos: List[List[DemoStep]]) -> Dict:
        obs = []
        for demo in demos:
            for step in self._training_timesteps(demo):
                obs.append(step.observation)

        keys = [
            key for key in obs[0].keys() if np.asarray(obs[0][key]).dtype != np.uint8
        ]
        if not keys:
            raise ValueError("BiGym demos do not contain low-dimensional observations.")
        obs = {key: np.stack([o[key] for o in obs], axis=0) for key in keys}
        obs_mean = {key: np.mean(obs[key], 0) for key in keys}
        obs_std = {key: np.clip(np.std(obs[key], 0), 1e-10, np.inf) for key in keys}
        obs_min = {key: np.min(obs[key], 0) for key in keys}
        obs_max = {key: np.max(obs[key], 0) for key in keys}

        # Identity-normalization override for degenerate-scale dims
        # (proprioception[0] and the gripper dims). NOTE: this predates the
        # May-2026 IL checkpoints (verified: their era emits exactly these
        # identity-normalized values), so it applies unconditionally -- do NOT
        # era-gate it. The 2026-08-08 checkpoint-compat audit found the real
        # old-checkpoint breaker was the lang tokenizer switch (see
        # AddBiGymLanguageTokens), not these stats.
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
        obs_stats = {
            "mean": obs_mean,
            "std": obs_std,
            "max": obs_max,
            "min": obs_min,
        }
        return obs_stats

    def _get_gripper_action_stats(
        self, cfg: DictConfig
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if cfg.env.action_mode in ["absolute", "delta"]:
            return (0.5, 0.25, 1, 0)
        else:
            raise NotImplementedError("Unsupported action mode.")

    def _rescale_demo_action_helper(self, info, cfg: DictConfig):
        if cfg.use_standardization:
            return RescaleFromStandardization.transform_to_standardization(
                info["demo_action"],
                action_stats=self._action_stats,
            )
        return RescaleFromTanhWithMinMax.transform_to_tanh(
            info["demo_action"],
            action_stats=self._action_stats,
            min_max_margin=cfg.min_max_margin,
        )
