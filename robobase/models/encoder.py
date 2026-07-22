"""Pure JAX ResNet encoder (Flax)."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict, freeze
from flax.traverse_util import unflatten_dict
import flax.linen as nn


_IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
_SUPPORTED_RESNETS = {"resnet18": 18, "resnet34": 34}
_RESNET_FEATURE_SIZES = {"resnet18": 512, "resnet34": 512}
_DEFAULT_RESNET18_JAX_NPZ = (
    Path.home() / ".cache" / "robobase_jaxflat" / "resnet18_imagenet_timm_jax_resnet.npz"
)


class _CQNMultiViewCNN(nn.Module):
    """Paper CQN-AS encoder: one independent 4-layer CNN per camera."""

    num_views: int

    @nn.compact
    def __call__(self, rgb_obs: jax.Array) -> jax.Array:
        x = rgb_obs.astype(jnp.float32) / 255.0 - 0.5
        outputs = []
        conv_init = nn.initializers.orthogonal(np.sqrt(2.0))
        for view in range(self.num_views):
            y = jnp.transpose(x[:, view], (0, 2, 3, 1))
            for layer, channels in enumerate((32, 64, 128, 256)):
                y = nn.Conv(
                    channels,
                    kernel_size=(4, 4),
                    strides=(2, 2),
                    padding=((1, 1), (1, 1)),
                    kernel_init=conv_init,
                    bias_init=nn.initializers.zeros_init(),
                    name=f"view_{view}_conv_{layer}",
                )(y)
                # This is ImgChLayerNorm from the reference implementation:
                # normalize channels independently at every spatial location.
                y = nn.LayerNorm(name=f"view_{view}_norm_{layer}")(y)
                y = nn.silu(y)
            outputs.append(y.reshape((y.shape[0], -1)))
        return jnp.stack(outputs, axis=1)


class JaxCQNEncoder:
    """Trainable adapter for the exact vision backbone used by CQN-AS."""

    def __init__(
        self,
        input_shape: tuple[int, int, int, int],
        *,
        jit: bool = True,
        seed: int = 0,
        **unused,
    ):
        del unused
        if len(input_shape) != 4:
            raise ValueError(f"CQN RGB input must be [views, channels, H, W], got {input_shape}.")
        self._input_shape = tuple(int(value) for value in input_shape)
        self._model = _CQNMultiViewCNN(num_views=self._input_shape[0])
        dummy = jnp.zeros((1, *self._input_shape), dtype=jnp.float32)
        self._variables = self._model.init(jax.random.PRNGKey(int(seed)), dummy)
        output = self._model.apply(self._variables, dummy)
        self._output_shape = tuple(int(value) for value in output.shape[1:])
        self._jit = bool(jit)
        self._encode = jax.jit(self._encode_impl) if self._jit else self._encode_impl

    @property
    def output_shape(self) -> tuple[int, int]:
        return self._output_shape

    @property
    def trainable_params(self):
        return self._variables["params"]

    @property
    def batch_stats(self):
        return None

    def frozen_state_dict(self) -> dict:
        return {}

    def load_frozen_state_dict(self, state_dict) -> None:
        del state_dict

    def _encode_impl(self, rgb_obs, *, params=None, stop_gradient=True):
        variables = self._variables if params is None else {"params": params}
        features = self._model.apply(variables, rgb_obs)
        return jax.lax.stop_gradient(features) if stop_gradient else features

    def encode(
        self,
        rgb_obs,
        raymap_obs=None,
        camera_intrinsic_obs=None,
        camera_c2w_obs=None,
    ):
        del raymap_obs, camera_intrinsic_obs, camera_c2w_obs
        return self._encode(jnp.asarray(rgb_obs, dtype=jnp.float32))

    def apply_trainable(
        self,
        params,
        rgb_obs,
        raymap_obs=None,
        camera_intrinsic_obs=None,
        camera_c2w_obs=None,
        task_emb=None,
    ):
        del raymap_obs, camera_intrinsic_obs, camera_c2w_obs, task_emb
        return self._encode_impl(
            jnp.asarray(rgb_obs, dtype=jnp.float32),
            params=params,
            stop_gradient=False,
        )


def _pretrained_resnet_candidates(model_name: str) -> list[Path]:
    env_name = f"ROBOBASE_{model_name.upper()}_JAX_NPZ"
    candidates = []
    env_value = os.environ.get(env_name)
    if env_value:
        candidates.append(Path(env_value).expanduser())
    if model_name == "resnet18":
        candidates.append(_DEFAULT_RESNET18_JAX_NPZ)
    return candidates


def _resolve_pretrained_resnet_npz(model_name: str) -> Path | None:
    return next(
        (path for path in _pretrained_resnet_candidates(model_name) if path.exists()),
        None,
    )


@lru_cache(maxsize=16)
def _file_sha256(
    path: str,
    size: int,
    mtime_ns: int,
    ctime_ns: int,
) -> str:
    del size, mtime_ns, ctime_ns
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resnet_weight_fingerprint(model_name: str, pretrained: bool) -> str:
    """Return a stable identity for the exact frozen encoder weights."""
    if not pretrained:
        return "jax-resnet-random-init-seed-0"
    path = _resolve_pretrained_resnet_npz(model_name)
    if path is None:
        # Legacy RoboBase pixel checkpoints were trained with timm weights
        # converted through jax_resnet at model construction time.  Keep a
        # distinct identity for that compatibility path when no exported NPZ
        # override is installed.
        try:
            import timm

            timm_version = str(timm.__version__)
        except Exception:
            timm_version = "unavailable"
        return f"legacy-timm-jax-resnet:{model_name}:{timm_version}"
    stat = path.stat()
    return "sha256:" + _file_sha256(
        str(path.resolve()),
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _linear(params, x: jnp.ndarray) -> jnp.ndarray:
    return x @ params["kernel"] + params["bias"]


def _as_jax_input(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return jnp.asarray(value)


def _init_linear_params(
    key,
    in_features: int,
    out_features: int,
) -> dict[str, jnp.ndarray]:
    bound = 1.0 / np.sqrt(float(in_features))
    kernel_key, bias_key = jax.random.split(key)
    return {
        "kernel": jax.random.uniform(
            kernel_key,
            (int(in_features), int(out_features)),
            minval=-bound,
            maxval=bound,
            dtype=jnp.float32,
        ),
        "bias": jax.random.uniform(
            bias_key,
            (int(out_features),),
            minval=-bound,
            maxval=bound,
            dtype=jnp.float32,
        ),
    }


def _init_resnet18_film_variables(
    key,
    *,
    task_input_dim: int,
    task_hidden_dim: int,
) -> FrozenDict:
    """Initialize the ACT FiLM language projection used by ResNet18."""

    keys = jax.random.split(key, 4)
    params = {
        "text_proj": _init_linear_params(keys[0], task_input_dim, task_hidden_dim),
        "layer_1": _init_linear_params(keys[1], task_hidden_dim, 2 * 2 * 128),
        "layer_2": _init_linear_params(keys[2], task_hidden_dim, 2 * 2 * 256),
        "layer_3": _init_linear_params(keys[3], task_hidden_dim, 2 * 2 * 512),
    }
    return freeze({"params": params})


def _conv2d(
    x: jnp.ndarray,
    kernel: jnp.ndarray,
    *,
    strides: tuple[int, int],
    padding,
) -> jnp.ndarray:
    return jax.lax.conv_general_dilated(
        x,
        kernel,
        window_strides=strides,
        padding=padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )


def _batch_norm(
    x: jnp.ndarray,
    params,
    stats,
    *,
    eps: float = 1e-5,
) -> jnp.ndarray:
    mean = stats["mean"].reshape((1, 1, 1, -1))
    var = stats["var"].reshape((1, 1, 1, -1))
    scale = params["scale"].reshape((1, 1, 1, -1))
    bias = params["bias"].reshape((1, 1, 1, -1))
    return (x - mean) * jax.lax.rsqrt(var + jnp.asarray(eps, dtype=x.dtype)) * scale + bias


def _resnet_conv_block(
    x: jnp.ndarray,
    params,
    stats,
    name: str,
    *,
    strides: tuple[int, int] = (1, 1),
    padding=((0, 0), (0, 0)),
    activate: bool = True,
) -> jnp.ndarray:
    block_params = params[name]
    block_stats = stats[name]
    x = _conv2d(
        x,
        block_params["Conv_0"]["kernel"],
        strides=strides,
        padding=padding,
    )
    x = _batch_norm(
        x,
        block_params["BatchNorm_0"],
        block_stats["BatchNorm_0"],
    )
    return jax.nn.relu(x) if activate else x


def _apply_resnet18_block(
    x: jnp.ndarray,
    params,
    stats,
    *,
    strides: tuple[int, int],
    film: jnp.ndarray | None = None,
) -> jnp.ndarray:
    residual = x
    y = _resnet_conv_block(
        x,
        params,
        stats,
        "ConvBlock_0",
        strides=strides,
        padding=((1, 1), (1, 1)),
        activate=True,
    )
    y = _resnet_conv_block(
        y,
        params,
        stats,
        "ConvBlock_1",
        strides=(1, 1),
        padding=((1, 1), (1, 1)),
        activate=False,
    )
    if film is not None:
        gamma = film[:, 0].reshape((film.shape[0], 1, 1, -1))
        beta = film[:, 1].reshape((film.shape[0], 1, 1, -1))
        y = (1.0 + gamma) * y + beta
    if "ResNetSkipConnection_0" in params:
        residual = _resnet_conv_block(
            residual,
            params["ResNetSkipConnection_0"],
            stats["ResNetSkipConnection_0"],
            "ConvBlock_0",
            strides=strides,
            padding=((0, 0), (0, 0)),
            activate=False,
        )
    return jax.nn.relu(y + residual)


def _apply_resnet18_film(
    x: jnp.ndarray,
    resnet_variables,
    film_params,
    task_emb: jnp.ndarray,
) -> jnp.ndarray:
    params = resnet_variables["params"]
    stats = resnet_variables["batch_stats"]
    x = _resnet_conv_block(
        x,
        params["layers_0"],
        stats["layers_0"],
        "ConvBlock_0",
        strides=(2, 2),
        padding=((3, 3), (3, 3)),
        activate=True,
    )
    x = nn.max_pool(
        x,
        window_shape=(3, 3),
        strides=(2, 2),
        padding=((1, 1), (1, 1)),
    )

    block_specs = (
        ("layers_2", (1, 1), None, None, None),
        ("layers_3", (1, 1), None, None, None),
        ("layers_4", (2, 2), "layer_1", 0, 128),
        ("layers_5", (1, 1), "layer_1", 1, 128),
        ("layers_6", (2, 2), "layer_2", 0, 256),
        ("layers_7", (1, 1), "layer_2", 1, 256),
        ("layers_8", (2, 2), "layer_3", 0, 512),
        ("layers_9", (1, 1), "layer_3", 1, 512),
    )
    film_cache = {}
    for layer_name, strides, film_name, block_idx, planes in block_specs:
        block_film = None
        if film_name is not None:
            if film_name not in film_cache:
                film_cache[film_name] = _linear(film_params[film_name], task_emb).reshape(
                    (task_emb.shape[0], 2, 2, int(planes))
                )
            block_film = film_cache[film_name][:, :, int(block_idx)]

        def apply_block(
            block_input,
            block_params,
            block_stats,
            film_values,
            block_strides=strides,
        ):
            return _apply_resnet18_block(
                block_input,
                block_params,
                block_stats,
                strides=block_strides,
                film=film_values,
            )

        x = jax.checkpoint(apply_block)(
            x,
            params[layer_name],
            stats[layer_name],
            block_film,
        )
    return x


class JaxPluckerEncoder(nn.Module):
    """Strided CNN adapter for per-pixel Plucker ray maps."""

    out_channels: int
    hidden_channels: int = 64

    @nn.compact
    def __call__(self, raymap: jnp.ndarray) -> jnp.ndarray:
        x = raymap.astype(jnp.float32)
        channels = (
            self.hidden_channels,
            self.hidden_channels * 2,
            self.hidden_channels * 4,
            self.out_channels,
            self.out_channels,
        )
        kernels = ((7, 7), (3, 3), (3, 3), (3, 3), (3, 3))
        for index, (features, kernel_size) in enumerate(
            zip(channels, kernels, strict=True)
        ):
            x = nn.Conv(
                features=int(features),
                kernel_size=kernel_size,
                strides=(2, 2),
                padding="SAME",
                use_bias=False,
                kernel_init=nn.initializers.kaiming_normal(),
                name=f"conv_{index}",
            )(x)
            x = _frozen_batch_norm_identity(x)
            x = nn.relu(x)
        return x


def _frozen_batch_norm_identity(x: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    """Apply fixed identity batch-normalization buffers."""

    return x * jax.lax.rsqrt(jnp.asarray(1.0 + eps, dtype=x.dtype))


class JaxPluckerLateFusion(nn.Module):
    """Official late concat + 1x1 projection for RGB and Plucker feature maps."""

    out_channels: int

    @nn.compact
    def __call__(self, rgb_feat: jnp.ndarray, plucker_feat: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([rgb_feat, plucker_feat], axis=-1)
        return nn.Conv(
            features=int(self.out_channels),
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            use_bias=True,
            kernel_init=nn.initializers.kaiming_normal(),
            name="input_proj",
        )(x)


def _identity_plucker_late_fusion_variables(num_features: int) -> FrozenDict:
    kernel = jnp.zeros((1, 1, int(num_features) * 2, int(num_features)), dtype=jnp.float32)
    kernel = kernel.at[0, 0, : int(num_features), :].set(
        jnp.eye(int(num_features), dtype=jnp.float32)
    )
    bias = jnp.zeros((int(num_features),), dtype=jnp.float32)
    return freeze({"params": {"input_proj": {"kernel": kernel, "bias": bias}}})


def _plucker_raymap_from_camera_params_jax(
    intrinsics: jnp.ndarray,
    c2ws: jnp.ndarray,
    height: int,
    width: int,
) -> jnp.ndarray:
    """Build direction+moment Plucker maps on the JAX device."""

    intrinsics = intrinsics.astype(jnp.float32)
    c2ws = c2ws.astype(jnp.float32)
    v, u = jnp.meshgrid(
        jnp.arange(height, dtype=jnp.float32) + 0.5,
        jnp.arange(width, dtype=jnp.float32) + 0.5,
        indexing="ij",
    )
    u = u.reshape((1, -1))
    v = v.reshape((1, -1))
    x = (u - intrinsics[:, 0, 2:3]) / intrinsics[:, 0, 0:1]
    y = -(v - intrinsics[:, 1, 2:3]) / intrinsics[:, 1, 1:2]
    camera_dirs = jnp.stack([x, y, -jnp.ones_like(x)], axis=-1)
    camera_dirs = camera_dirs / jnp.maximum(
        jnp.linalg.norm(camera_dirs, axis=-1, keepdims=True),
        jnp.asarray(1e-9, dtype=jnp.float32),
    )
    directions = jnp.einsum("npc,nkc->npk", camera_dirs, c2ws[:, :3, :3])
    origins = jnp.broadcast_to(c2ws[:, None, :3, 3], directions.shape)
    viewdirs = directions / jnp.maximum(
        jnp.linalg.norm(directions, axis=-1, keepdims=True),
        jnp.asarray(1e-9, dtype=jnp.float32),
    )
    moments = jnp.cross(origins, viewdirs, axis=-1)
    return jnp.concatenate([viewdirs, moments], axis=-1).reshape(
        (intrinsics.shape[0], height, width, 6)
    )


def _load_pretrained_resnet_npz(model_name: str) -> FrozenDict:
    env_name = f"ROBOBASE_{model_name.upper()}_JAX_NPZ"
    candidates = _pretrained_resnet_candidates(model_name)
    path = _resolve_pretrained_resnet_npz(model_name)
    if path is not None:
        with np.load(path, allow_pickle=False) as arrays:
            flat_variables = {
                tuple(str(key).split("/")): jnp.asarray(
                    arrays[key], dtype=jnp.float32
                )
                for key in arrays.files
            }
        return freeze(unflatten_dict(flat_variables))

    searched = ", ".join(str(path) for path in candidates) or f"${env_name} (unset)"
    raise FileNotFoundError(
        f"pretrained=true for '{model_name}' requires a converted JAX ResNet npz. "
        f"Searched: {searched}. Set {env_name} to override."
    )


@lru_cache(maxsize=4)
def _load_legacy_timm_pretrained_resnet_variables(model_name: str) -> FrozenDict:
    """Load the exact ImageNet variables used by legacy pixel baselines.

    Before the pure-JAX encoder rewrite, ``pretrained`` was implicit and the
    encoder always converted timm's ResNet weights with ``jax_resnet``.  Old
    snapshots store trainable parameters but not frozen batch statistics, so
    substituting a different ImageNet checkpoint (or random batch statistics)
    makes otherwise valid policies produce saturated actions.
    """

    if model_name not in _SUPPORTED_RESNETS:
        raise NotImplementedError(
            f"JAX encoder supports only {sorted(_SUPPORTED_RESNETS)}. Got '{model_name}'."
        )
    try:
        import jax_resnet
        import timm
        from jax_resnet.common import slice_variables
    except ImportError as exc:
        raise ImportError(
            "Legacy pretrained ResNet compatibility requires `jax-resnet` and `timm`."
        ) from exc

    state_dict = timm.create_model(model_name, pretrained=True).state_dict()
    _, variables = jax_resnet.pretrained_resnet(
        _SUPPORTED_RESNETS[model_name],
        state_dict=state_dict,
    )
    # Exclude only the classifier. The average-pool layer has no variables, so
    # the same tree can drive both pooled and spatial feature-model variants.
    return freeze(slice_variables(variables, end=-1))


def _load_resnet_feature_model(model_name: str, pretrained: bool = False):
    weight_fingerprint = resnet_weight_fingerprint(model_name, pretrained)
    return _load_resnet_feature_model_cached(
        model_name,
        pretrained,
        weight_fingerprint,
    )


@lru_cache(maxsize=8)
def _load_resnet_feature_model_cached(
    model_name: str,
    pretrained: bool,
    weight_fingerprint: str,
):
    del weight_fingerprint
    if model_name not in _SUPPORTED_RESNETS:
        raise NotImplementedError(
            f"JAX encoder supports only {sorted(_SUPPORTED_RESNETS)}. Got '{model_name}'."
        )
    try:
        from jax_resnet import resnet
    except ImportError as exc:
        raise ImportError("JAX ResNet encoder requires `flax` and `jax-resnet`.") from exc

    model = getattr(resnet, f"ResNet{_SUPPORTED_RESNETS[model_name]}")(n_classes=1000)
    feature_model = nn.Sequential(model.layers[:-2])
    if pretrained:
        feature_variables = (
            _load_pretrained_resnet_npz(model_name)
            if _resolve_pretrained_resnet_npz(model_name) is not None
            else _load_legacy_timm_pretrained_resnet_variables(model_name)
        )
    else:
        feature_variables = feature_model.init(
            jax.random.PRNGKey(0),
            jnp.zeros((1, 224, 224, 3), dtype=jnp.float32),
        )
    return feature_model, feature_variables, _RESNET_FEATURE_SIZES[model_name]


class JaxResNetEncoder:
    """JAX ResNet feature extractor.

    This is *not* a trainable ``nn.Module`` — parameters are loaded once and
    frozen via ``stop_gradient``.  Only the JIT-compiled ``encode`` path is
    exposed.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int, int],
        model: str,
        jit: bool = True,
        pretrained: bool = False,
        resize_to_224: bool = True,
        use_plucker: bool = False,
        plucker_hidden_channels: int = 64,
        plucker_identity_init: bool = False,
        use_film: bool = False,
        film_task_input_dim: int = 512,
        film_task_hidden_dim: int = 256,
    ):
        if input_shape[1] % 3 != 0:
            raise ValueError(
                "ResNet RGB input channels must be a multiple of 3; "
                f"got input_shape={input_shape}."
            )

        feature_model, feature_variables, num_features = (
            _load_resnet_feature_model(model, bool(pretrained))
        )
        self._feature_model = feature_model
        self._feature_variables = jax.tree.map(
            lambda x: jnp.asarray(x, dtype=jnp.float32), feature_variables,
        )
        self._num_features = int(num_features)
        self._input_shape = input_shape
        self._resize_to_224 = bool(resize_to_224)
        self._use_plucker = bool(use_plucker)
        self._use_film = bool(use_film)
        if self._use_film and model != "resnet18":
            raise NotImplementedError("ACT FiLM conditioning currently supports resnet18 only.")
        self._plucker_model = None
        self._plucker_feature_variables = None
        self._plucker_fusion_model = None
        self._plucker_fusion_variables = None
        self._film_variables = None
        self._mean = jnp.asarray(_IMAGENET_MEAN.reshape((1, 1, 1, 3)))
        self._std = jnp.asarray(_IMAGENET_STD.reshape((1, 1, 1, 3)))

        if self._use_film:
            self._film_variables = _init_resnet18_film_variables(
                jax.random.PRNGKey(2),
                task_input_dim=int(film_task_input_dim),
                task_hidden_dim=int(film_task_hidden_dim),
            )

        if self._use_plucker:
            self._plucker_model = JaxPluckerEncoder(
                out_channels=self._num_features,
                hidden_channels=int(plucker_hidden_channels),
            )
            dummy_raymap = jnp.zeros(
                (1, int(input_shape[2]), int(input_shape[3]), 6),
                dtype=jnp.float32,
            )
            self._plucker_feature_variables = self._plucker_model.init(
                jax.random.PRNGKey(0), dummy_raymap,
            )
            self._plucker_fusion_model = JaxPluckerLateFusion(
                out_channels=self._num_features,
            )
            if bool(plucker_identity_init):
                self._plucker_fusion_variables = _identity_plucker_late_fusion_variables(
                    self._num_features
                )
            else:
                dummy_feat = jnp.zeros(
                    (1, 1, 1, self._num_features),
                    dtype=jnp.float32,
                )
                self._plucker_fusion_variables = self._plucker_fusion_model.init(
                    jax.random.PRNGKey(1), dummy_feat, dummy_feat,
                )

        self._jit = bool(jit)
        self._refresh_encode_fn()

    @property
    def output_shape(self) -> tuple[int, int]:
        return (self._input_shape[0], self._num_features)

    @property
    def trainable_params(self):
        resnet_params = self._feature_variables.get("params", {})
        if not self._use_plucker and not self._use_film:
            return resnet_params
        params = {
            "resnet": resnet_params,
        }
        if self._use_film:
            params["film"] = self._film_variables.get("params", {})
        if self._use_plucker:
            params["plucker"] = self._plucker_feature_variables.get("params", {})
            params["fusion"] = self._plucker_fusion_variables.get("params", {})
        return params

    @property
    def batch_stats(self):
        return self._feature_variables.get("batch_stats", None)

    def _refresh_encode_fn(self) -> None:
        encode_impl = self._encode_impl
        self._encode = jax.jit(encode_impl) if self._jit else encode_impl

    def frozen_state_dict(self) -> dict:
        """Return non-parameter ResNet state required for deterministic restore."""

        if self.batch_stats is None:
            return {}
        return {"batch_stats": self.batch_stats}

    def load_frozen_state_dict(self, state_dict: dict | None) -> None:
        """Restore frozen ResNet collections while preserving trainable params."""

        if not state_dict or state_dict.get("batch_stats") is None:
            return
        feature_variables = dict(self._feature_variables)
        feature_variables["batch_stats"] = jax.tree.map(
            lambda value: jnp.asarray(value, dtype=jnp.float32),
            state_dict["batch_stats"],
        )
        self._feature_variables = freeze(feature_variables)
        # ``encode`` can have been traced before a state restore in standalone
        # use. Rebuild it so the restored frozen buffers cannot remain captured
        # as stale JIT constants.
        self._refresh_encode_fn()

    def encode(
        self,
        rgb_obs,
        raymap_obs=None,
        camera_intrinsic_obs=None,
        camera_c2w_obs=None,
    ) -> jnp.ndarray:
        return self._encode(
            _as_jax_input(rgb_obs),
            _as_jax_input(raymap_obs),
            _as_jax_input(camera_intrinsic_obs),
            _as_jax_input(camera_c2w_obs),
        )

    def apply_trainable(
        self,
        params,
        rgb_obs: jnp.ndarray,
        raymap_obs: jnp.ndarray | None = None,
        camera_intrinsic_obs: jnp.ndarray | None = None,
        camera_c2w_obs: jnp.ndarray | None = None,
        task_emb: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        if self._use_plucker or self._use_film:
            variables = {"resnet": {"params": params["resnet"]}}
            if self._use_film:
                variables["film"] = {"params": params["film"]}
            if self._use_plucker:
                variables["plucker"] = {"params": params["plucker"]}
                variables["fusion"] = {"params": params["fusion"]}
        else:
            variables = {"params": params}
        if self.batch_stats is not None:
            variables["batch_stats"] = self.batch_stats
        return self._encode_impl(
            rgb_obs,
            raymap_obs,
            camera_intrinsic_obs,
            camera_c2w_obs,
            task_emb=task_emb,
            variables=freeze(variables),
            stop_gradient=False,
        )

    def apply_trainable_spatial(
        self,
        params,
        rgb_obs: jnp.ndarray,
        raymap_obs: jnp.ndarray | None = None,
        camera_intrinsic_obs: jnp.ndarray | None = None,
        camera_c2w_obs: jnp.ndarray | None = None,
        task_emb: jnp.ndarray | None = None,
        return_task_emb: bool = False,
    ) -> jnp.ndarray:
        if self._use_plucker or self._use_film:
            variables = {"resnet": {"params": params["resnet"]}}
            if self._use_film:
                variables["film"] = {"params": params["film"]}
            if self._use_plucker:
                variables["plucker"] = {"params": params["plucker"]}
                variables["fusion"] = {"params": params["fusion"]}
        else:
            variables = {"params": params}
        if self.batch_stats is not None:
            variables["batch_stats"] = self.batch_stats
        return self._encode_impl(
            rgb_obs,
            raymap_obs,
            camera_intrinsic_obs,
            camera_c2w_obs,
            task_emb=task_emb,
            variables=freeze(variables),
            stop_gradient=False,
            spatial=True,
            return_task_emb=return_task_emb,
        )

    def _preprocess(
        self,
        rgb_obs: jnp.ndarray,
        *,
        resize: bool | None = None,
    ) -> tuple[jnp.ndarray, int, int, int, int, int]:
        x = rgb_obs.astype(jnp.float32)
        batch_size, num_views, channels, height, width = x.shape
        if channels % 3 != 0:
            raise ValueError(f"Expected RGB channels divisible by 3, got {channels}.")
        frame_stack = channels // 3
        x = x.reshape((batch_size, num_views, frame_stack, 3, height, width))
        x = jnp.transpose(x, (0, 1, 2, 4, 5, 3))
        x = x.reshape((batch_size * num_views * frame_stack, height, width, 3))
        if resize is None:
            resize = self._resize_to_224 and not self._use_plucker
        if resize:
            x = jax.image.resize(
                x, shape=(x.shape[0], 224, 224, x.shape[-1]), method="bilinear",
            )
        x = x / 255.0
        x = (x - self._mean) / self._std
        return x, batch_size, num_views, frame_stack, int(height), int(width)

    def _preprocess_raymap(
        self, raymap_obs: jnp.ndarray, batch_size: int, num_views: int
    ) -> tuple[jnp.ndarray, int]:
        raymap = raymap_obs.astype(jnp.float32)
        if raymap.ndim != 5:
            raise ValueError(
                "Expected raymap observations with shape "
                f"(batch, views, channels, height, width), got {raymap.shape}."
            )
        if raymap.shape[0] != batch_size or raymap.shape[1] != num_views:
            raise ValueError(
                "Raymap observations must match RGB batch/view dimensions; "
                f"got raymap={raymap.shape[:2]}, rgb={(batch_size, num_views)}."
            )
        channels = int(raymap.shape[2])
        if channels % 6 != 0:
            raise ValueError(f"Expected raymap channels divisible by 6, got {channels}.")
        frame_stack = channels // 6
        height, width = int(raymap.shape[-2]), int(raymap.shape[-1])
        raymap = raymap.reshape((batch_size, num_views, frame_stack, 6, height, width))
        raymap = jnp.transpose(raymap, (0, 1, 2, 4, 5, 3))
        raymap = raymap.reshape((batch_size * num_views * frame_stack, height, width, 6))
        return raymap, frame_stack

    def _preprocess_camera_params(
        self,
        camera_intrinsic_obs: jnp.ndarray,
        camera_c2w_obs: jnp.ndarray,
        batch_size: int,
        num_views: int,
        height: int,
        width: int,
    ) -> tuple[jnp.ndarray, int]:
        intrinsics = camera_intrinsic_obs.astype(jnp.float32)
        c2ws = camera_c2w_obs.astype(jnp.float32)
        if intrinsics.ndim != 5 or c2ws.ndim != 5:
            raise ValueError(
                "Expected camera parameter observations with shapes "
                "(batch, views, time, 3, 3) and (batch, views, time, 4, 4); "
                f"got {intrinsics.shape} and {c2ws.shape}."
            )
        if intrinsics.shape[:2] != (batch_size, num_views) or c2ws.shape[:2] != (
            batch_size,
            num_views,
        ):
            raise ValueError(
                "Camera parameter observations must match RGB batch/view dimensions; "
                f"got intrinsic={intrinsics.shape[:2]}, c2w={c2ws.shape[:2]}, "
                f"rgb={(batch_size, num_views)}."
            )
        if intrinsics.shape[2] != c2ws.shape[2]:
            raise ValueError(
                "Camera intrinsic and c2w frame stacks must match; "
                f"got {intrinsics.shape[2]} vs {c2ws.shape[2]}."
            )
        if intrinsics.shape[-2:] != (3, 3) or c2ws.shape[-2:] != (4, 4):
            raise ValueError(
                "Expected camera parameter trailing shapes (3, 3) and (4, 4); "
                f"got {intrinsics.shape[-2:]} and {c2ws.shape[-2:]}."
            )
        frame_stack = int(intrinsics.shape[2])
        intrinsics = intrinsics.reshape((-1, 3, 3))
        c2ws = c2ws.reshape((-1, 4, 4))
        raymap = _plucker_raymap_from_camera_params_jax(
            intrinsics,
            c2ws,
            height=height,
            width=width,
        )
        return raymap, frame_stack

    def _resnet_variables(self, variables: FrozenDict | None):
        if variables is None:
            return self._feature_variables
        if "resnet" in variables:
            resnet_variables = dict(variables["resnet"])
            if self.batch_stats is not None:
                resnet_variables["batch_stats"] = self.batch_stats
            return freeze(resnet_variables)
        if self.batch_stats is not None and "batch_stats" not in variables:
            resnet_variables = dict(variables)
            resnet_variables["batch_stats"] = self.batch_stats
            return freeze(resnet_variables)
        return variables

    def _get_film_params(self, variables: FrozenDict | None):
        if not self._use_film:
            return None
        if variables is None:
            return self._film_variables["params"]
        if "film" in variables:
            return variables["film"]["params"]
        return self._film_variables["params"]

    def _get_plucker_variables(self, variables: FrozenDict | None):
        if variables is None:
            return self._plucker_feature_variables
        if "plucker" in variables:
            return freeze({"params": variables["plucker"]["params"]})
        return self._plucker_feature_variables

    def _get_plucker_fusion_variables(self, variables: FrozenDict | None):
        if variables is None:
            return self._plucker_fusion_variables
        if "fusion" in variables:
            return freeze({"params": variables["fusion"]["params"]})
        return self._plucker_fusion_variables

    def _encode_impl(
        self,
        rgb_obs: jnp.ndarray,
        raymap_obs: jnp.ndarray | None = None,
        camera_intrinsic_obs: jnp.ndarray | None = None,
        camera_c2w_obs: jnp.ndarray | None = None,
        *,
        task_emb: jnp.ndarray | None = None,
        variables: FrozenDict | None = None,
        stop_gradient: bool = True,
        spatial: bool = False,
        return_task_emb: bool = False,
    ) -> jnp.ndarray:
        x, batch_size, num_views, frame_stack, height, width = self._preprocess(
            rgb_obs,
            resize=(False if spatial else None),
        )
        encoded_task_emb = task_emb
        if self._use_film:
            if task_emb is None:
                raise ValueError("ACT FiLM ResNet conditioning requires task_emb.")
            film_params = self._get_film_params(variables)
            encoded_task_emb = _linear(film_params["text_proj"], task_emb.astype(jnp.float32))
            expanded_task = jnp.broadcast_to(
                encoded_task_emb[:, None, None, :],
                (batch_size, num_views, frame_stack, encoded_task_emb.shape[-1]),
            ).reshape((batch_size * num_views * frame_stack, encoded_task_emb.shape[-1]))
            x = _apply_resnet18_film(
                x,
                self._resnet_variables(variables),
                film_params,
                expanded_task,
            )
        else:
            x = self._feature_model.apply(
                self._resnet_variables(variables),
                x,
            )

        def finish(features):
            if stop_gradient:
                features = jax.lax.stop_gradient(features)
            if return_task_emb:
                return features, encoded_task_emb
            return features

        if self._use_plucker:
            if raymap_obs is None and (
                camera_intrinsic_obs is None or camera_c2w_obs is None
            ):
                raise ValueError(
                    "JaxResNetEncoder(use_plucker=True) requires raymap_obs or "
                    "camera_intrinsic_obs/camera_c2w_obs."
                )
            if x.ndim != 4:
                raise ValueError(
                    "Plucker late fusion expects spatial RGB feature maps; "
                    f"got shape {x.shape}."
                )
            if raymap_obs is not None:
                raymap, raymap_frame_stack = self._preprocess_raymap(
                    raymap_obs, batch_size, num_views,
                )
            else:
                raymap, raymap_frame_stack = self._preprocess_camera_params(
                    camera_intrinsic_obs,
                    camera_c2w_obs,
                    batch_size,
                    num_views,
                    height,
                    width,
                )
            if raymap_frame_stack != frame_stack:
                raise ValueError(
                    "Raymap frame stack must match RGB frame stack; "
                    f"got {raymap_frame_stack} vs {frame_stack}."
                )
            plucker = self._plucker_model.apply(
                self._get_plucker_variables(variables),
                raymap,
            )
            if plucker.shape[-3:-1] != x.shape[-3:-1]:
                raise ValueError(
                    "Plucker feature spatial shape does not match RGB feature shape: "
                    f"{tuple(plucker.shape[-3:-1])} vs {tuple(x.shape[-3:-1])}."
                )
            x = self._plucker_fusion_model.apply(
                self._get_plucker_fusion_variables(variables),
                x,
                plucker,
            )
            x = x.reshape(
                (
                    batch_size,
                    num_views,
                    frame_stack,
                    x.shape[1],
                    x.shape[2],
                    self._num_features,
                )
            )
            x = x.mean(axis=2)
            if spatial:
                return finish(x)
            x = x.mean(axis=(2, 3))
        else:
            if spatial:
                if x.ndim != 4:
                    raise ValueError(
                        "Spatial ResNet encoding expects a feature map output; "
                        f"got shape {x.shape}."
                    )
                x = x.reshape(
                    (
                        batch_size,
                        num_views,
                        frame_stack,
                        x.shape[1],
                        x.shape[2],
                        self._num_features,
                    )
                )
                x = x.mean(axis=2)
                return finish(x)
            if x.ndim == 4:
                x = x.mean(axis=(1, 2))
            x = x.reshape((batch_size, num_views, frame_stack, self._num_features))
            x = x.mean(axis=2)
        return finish(x)
