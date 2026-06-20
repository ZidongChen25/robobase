from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig

from robobase.method.bc_runtime import bc_observation_layout


JAX_BC_FEATURE_KEY = "vision_features_jax_bc"
JAX_DIFFUSION_FEATURE_KEY = "vision_features_jax_diffusion"
JAX_FLOW_MATCHING_FEATURE_KEY = "vision_features_jax_flow_matching"
JAX_ACT_FEATURE_KEY = "vision_features_jax_act"
DEFAULT_CACHE_BACKENDS = ("jax",)
_SUPPORTED_FUSION_MODES = {"flatten", "average", "sum"}
_SUPPORTED_RESNETS = {"resnet18": 512, "resnet34": 512}


def cached_feature_observation_key(method_name: str, backend_name: str = "jax") -> str:
    method_name = str(method_name).lower()
    mapping = {
        "act": JAX_ACT_FEATURE_KEY,
        "bc": JAX_BC_FEATURE_KEY,
        "diffusion": JAX_DIFFUSION_FEATURE_KEY,
        "flow_matching": JAX_FLOW_MATCHING_FEATURE_KEY,
    }
    try:
        return mapping[method_name]
    except KeyError as exc:
        raise ValueError(
            f"No cached feature observation key for method={method_name!r}."
        ) from exc


@dataclass(frozen=True)
class VisionFeatureCachePlan:
    observation_space: spaces.Dict
    preprocessing_fn: list[Any]
    feature_keys: tuple[str, ...]
    feature_dim: int


@dataclass
class FrozenVisionFeaturePreprocessor:
    rgb_keys: tuple[str, ...]
    encoder_model: str
    fusion_mode: str
    feature_keys: tuple[str, ...]
    method_name: str
    input_shape: tuple[int, int, int, int]
    _torch_encoder: Any = field(default=None, init=False, repr=False)
    _jax_encoder: Any = field(default=None, init=False, repr=False)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_torch_encoder"] = None
        state["_jax_encoder"] = None
        return state

    def __call__(self, transitions: list[dict]) -> list[dict]:
        if not transitions:
            return transitions
        rgb_batch = np.stack(
            [
                np.stack(
                    [
                        np.asarray(transition[key], dtype=np.uint8)
                        for key in self.rgb_keys
                    ],
                    axis=0,
                )
                for transition in transitions
            ],
            axis=0,
        )

        cached_batches: dict[str, np.ndarray] = {}
        for feature_key in self.feature_keys:
            cached_batches[feature_key] = self._encode_jax(rgb_batch)

        processed = []
        for index, transition in enumerate(transitions):
            new_transition = {
                key: value
                for key, value in transition.items()
                if key not in self.rgb_keys
            }
            for feature_key, feature_batch in cached_batches.items():
                new_transition[feature_key] = feature_batch[index]
            processed.append(new_transition)
        return processed

    def _fuse_multi_view(self, feats: np.ndarray) -> np.ndarray:
        if feats.shape[1] == 1:
            return feats[:, 0].astype(np.float32, copy=False)
        if self.fusion_mode == "flatten":
            return feats.reshape((feats.shape[0], -1)).astype(np.float32, copy=False)
        if self.fusion_mode == "average":
            return feats.mean(axis=1).astype(np.float32, copy=False)
        if self.fusion_mode == "sum":
            return feats.sum(axis=1).astype(np.float32, copy=False)
        raise ValueError(f"Unsupported fusion mode '{self.fusion_mode}'.")

    def _encode_jax(self, rgb_batch: np.ndarray) -> np.ndarray:
        if self._jax_encoder is None:
            import jax
            from robobase.models.encoder import JaxResNetEncoder

            self._jax_encoder = JaxResNetEncoder(
                input_shape=self.input_shape,
                model=self.encoder_model,
                jit=True,
            )
        import jax
        feats = self._jax_encoder.encode(rgb_batch)
        return self._fuse_multi_view(np.asarray(jax.device_get(feats)))


def build_vision_feature_cache_plan(
    *,
    cfg: DictConfig,
    observation_space: spaces.Dict,
    save_dir: str | None,
    reuse_saved: bool,
) -> VisionFeatureCachePlan | None:
    cache_setting = str(
        cfg.replay.get("cache_frozen_image_features", "auto")
    ).strip().lower()
    if cache_setting in {"0", "false", "off", "no"}:
        return None
    if cache_setting not in {"1", "true", "on", "yes", "auto"}:
        raise ValueError(
            "replay.cache_frozen_image_features must be one of "
            "{auto,true,false,on,off,1,0}."
        )

    method_name = str(cfg.method.get("name", "")).lower()
    if method_name not in {"act", "bc", "diffusion", "flow_matching"}:
        if cache_setting != "auto":
            raise ValueError(
                "Frozen image-feature replay caching is only supported for "
                "method=act, method=bc, method=diffusion, and method=flow_matching."
            )
        return None

    obs_layout = bc_observation_layout(observation_space)
    if not obs_layout.use_pixels:
        return None
    if obs_layout.time_dim != 1:
        message = (
            "Frozen image-feature replay caching currently requires frame_stack=1 "
            f"(got time_dim={obs_layout.time_dim})."
        )
        if cache_setting == "auto":
            logging.info("%s Falling back to raw replay images.", message)
            return None
        raise ValueError(message)
    if obs_layout.rgb_input_shape is None or obs_layout.rgb_input_shape[1] != 3:
        message = (
            "Frozen image-feature replay caching requires RGB observations with "
            f"3 channels, got rgb_input_shape={obs_layout.rgb_input_shape}."
        )
        if cache_setting == "auto":
            logging.info("%s Falling back to raw replay images.", message)
            return None
        raise ValueError(message)

    if method_name == "act":
        from robobase.method.act import act_model_spec_from_cfg
        model_spec = act_model_spec_from_cfg(cfg)
    elif method_name == "bc":
        from robobase.method.bc import bc_model_spec_from_cfg
        model_spec = bc_model_spec_from_cfg(cfg)
    elif method_name == "diffusion":
        from robobase.method.diffusion import diffusion_model_spec_from_cfg
        model_spec = diffusion_model_spec_from_cfg(cfg)
    else:
        from robobase.method.flow_matching import flow_matching_model_spec_from_cfg
        model_spec = flow_matching_model_spec_from_cfg(cfg)

    if model_spec.encoder_model is None or model_spec.encoder_model.type != "resnet":
        message = "Frozen image-feature replay caching currently supports only ResNet encoders."
        if cache_setting == "auto":
            logging.info("%s Falling back to raw replay images.", message)
            return None
        raise ValueError(message)
    if getattr(model_spec.encoder_model, "trainable", False):
        message = "Frozen image-feature replay caching is incompatible with trainable encoders."
        if cache_setting == "auto":
            logging.info("%s Falling back to raw replay images.", message)
            return None
        raise ValueError(message)
    if model_spec.encoder_model.model not in _SUPPORTED_RESNETS:
        message = (
            "Frozen image-feature replay caching supports only "
            f"{sorted(_SUPPORTED_RESNETS)}, got {model_spec.encoder_model.model!r}."
        )
        if cache_setting == "auto":
            logging.info("%s Falling back to raw replay images.", message)
            return None
        raise ValueError(message)

    if obs_layout.use_multicam_fusion:
        if (
            model_spec.view_fusion_model is None
            or model_spec.view_fusion_model.type != "multicam_feature"
            or model_spec.view_fusion_model.mode not in _SUPPORTED_FUSION_MODES
        ):
            message = (
                "Frozen image-feature replay caching supports only parameter-free "
                f"fusion modes {sorted(_SUPPORTED_FUSION_MODES)}."
            )
            if cache_setting == "auto":
                logging.info("%s Falling back to raw replay images.", message)
                return None
            raise ValueError(message)
        fusion_mode = model_spec.view_fusion_model.mode
    else:
        fusion_mode = "flatten"

    requested_backends = tuple(
        str(name).lower()
        for name in cfg.replay.get(
            "cache_frozen_image_feature_backends",
            list(DEFAULT_CACHE_BACKENDS),
        )
    )
    if not requested_backends:
        return None
    unsupported_backends = set(requested_backends) - set(DEFAULT_CACHE_BACKENDS)
    if unsupported_backends:
        raise ValueError(
            "Unsupported replay.cache_frozen_image_feature_backends: "
            f"{sorted(unsupported_backends)}."
        )

    feature_keys = tuple(
        cached_feature_observation_key(method_name, backend_name)
        for backend_name in requested_backends
    )
    feature_dim = _SUPPORTED_RESNETS[model_spec.encoder_model.model]
    if obs_layout.use_multicam_fusion and fusion_mode == "flatten":
        feature_dim *= len(obs_layout.rgb_keys)

    if reuse_saved and save_dir is not None:
        existing_replay = Path(save_dir)
        saved_keys = _first_saved_episode_keys(existing_replay)
        if saved_keys is not None and not all(key in saved_keys for key in feature_keys):
            message = (
                "Saved replay does not contain cached image features for "
                f"{sorted(feature_keys)} in {existing_replay}."
            )
            if cache_setting == "auto":
                logging.info("%s Reusing raw replay instead.", message)
                return None
            raise ValueError(message)

    replay_observation_space = spaces.Dict(
        {
            key: value
            for key, value in observation_space.items()
            if key not in obs_layout.rgb_keys
        }
    )
    for feature_key in feature_keys:
        replay_observation_space[feature_key] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1, feature_dim),
            dtype=np.float32,
        )

    preprocessor = FrozenVisionFeaturePreprocessor(
        rgb_keys=obs_layout.rgb_keys,
        encoder_model=model_spec.encoder_model.model,
        fusion_mode=fusion_mode,
        feature_keys=feature_keys,
        method_name=method_name,
        input_shape=obs_layout.rgb_input_shape,
    )
    return VisionFeatureCachePlan(
        observation_space=replay_observation_space,
        preprocessing_fn=[preprocessor],
        feature_keys=feature_keys,
        feature_dim=feature_dim,
    )


def _first_saved_episode_keys(replay_dir: Path) -> set[str] | None:
    if not replay_dir.exists():
        return None
    episode_files = sorted(replay_dir.glob("*.npz"))
    if not episode_files:
        return None
    with np.load(episode_files[0], allow_pickle=False) as episode:
        return set(episode.files)
