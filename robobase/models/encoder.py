"""Pure JAX ResNet encoder (Flax)."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
import tempfile
import urllib.request

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict, freeze
from flax.traverse_util import unflatten_dict
import flax.linen as nn

from robobase.models.resnet import resnet_feature_model
from robobase.models.pooling import max_pool_3x3_stride2_pad1


_IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
_SUPPORTED_RESNETS = {"resnet18": 18, "resnet34": 34}
_RESNET_FEATURE_SIZES = {"resnet18": 512, "resnet34": 512}
_PYTORCH_CONV_KERNEL_INIT = nn.initializers.variance_scaling(
    1.0 / 3.0,
    "fan_in",
    "uniform",
)
_PYTORCH_RESNET_CONV_KERNEL_INIT = nn.initializers.variance_scaling(
    2.0,
    "fan_out",
    "normal",
)
_DEFAULT_RESNET18_JAX_NPZ = (
    Path.home()
    / ".cache"
    / "robobase_jaxflat"
    / "resnet18_imagenet_timm_jax_resnet.npz"
)
_RESNET18_SAFETENSORS_URL = (
    "https://huggingface.co/timm/resnet18.tv_in1k/resolve/"
    "a987fd6b60f845221bfff84b6f0191273ba56ead/model.safetensors"
)
_RESNET18_SAFETENSORS_SHA256 = (
    "694f673df6520a3158624e8a89af086f59923ee4cd7436fe5bc3bc71d295ad81"
)
_DEFAULT_RESNET18_SAFETENSORS = (
    Path.home()
    / ".cache"
    / "robobase_jaxflat"
    / "resnet18.tv_in1k.a987fd6.model.safetensors"
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


def _drqv2_activation(x: jax.Array, name: str) -> jax.Array:
    """Activation set supported by the configurable DrQ-v2 CNN."""

    name = str(name).lower()
    if name == "relu":
        return nn.relu(x)
    if name in {"silu", "swish"}:
        return nn.silu(x)
    if name == "gelu":
        return nn.gelu(x)
    if name == "tanh":
        return jnp.tanh(x)
    raise ValueError(f"Unsupported DrQ-v2 encoder activation {name!r}.")


class _DrQV2MultiViewCNN(nn.Module):
    """Official DrQ-v2 convolution pattern, independently per camera.

    The reference agent has one camera and uses a stride-2 convolution followed
    by three stride-1 convolutions.  RoboBase observations can contain multiple
    cameras, so this keeps the historical RoboBase behavior of assigning each
    view its own encoder while retaining the exact single-view architecture.
    """

    num_views: int
    num_downsample_convs: int = 1
    num_post_downsample_convs: int = 3
    channels: int = 32
    kernel_size: int = 3
    padding: int = 0
    channels_multiplier: int = 1
    activation_name: str = "relu"
    norm: str = "none"
    normalize_inputs: bool = True

    @nn.compact
    def __call__(self, rgb_obs: jax.Array) -> jax.Array:
        x = rgb_obs.astype(jnp.float32)
        if self.normalize_inputs:
            x = x / 255.0 - 0.5
        kernel_init = nn.initializers.orthogonal(np.sqrt(2.0))
        padding = (
            "VALID"
            if self.padding == 0
            else ((self.padding, self.padding), (self.padding, self.padding))
        )
        norm = str(self.norm).lower()
        if norm not in {"none", "identity", "layer"}:
            raise ValueError(
                "DrQ-v2 encoder norm must be one of "
                f"{{'none', 'identity', 'layer'}}, got {self.norm!r}."
            )

        outputs = []
        for view in range(self.num_views):
            y = jnp.transpose(x[:, view], (0, 2, 3, 1))
            output_channels = int(self.channels)
            layer_index = 0
            for stride, count in (
                (2, int(self.num_downsample_convs)),
                (1, int(self.num_post_downsample_convs)),
            ):
                for _ in range(count):
                    y = nn.Conv(
                        output_channels,
                        kernel_size=(self.kernel_size, self.kernel_size),
                        strides=(stride, stride),
                        padding=padding,
                        kernel_init=kernel_init,
                        bias_init=nn.initializers.zeros_init(),
                        name=f"view_{view}_conv_{layer_index}",
                    )(y)
                    if norm == "layer":
                        y = nn.LayerNorm(
                            name=f"view_{view}_norm_{layer_index}"
                        )(y)
                    y = _drqv2_activation(y, self.activation_name)
                    output_channels *= int(self.channels_multiplier)
                    layer_index += 1
            outputs.append(y.reshape((y.shape[0], -1)))
        return jnp.stack(outputs, axis=1)


class JaxDrQV2Encoder:
    """Trainable Flax adapter for the official DrQ-v2 pixel encoder."""

    def __init__(
        self,
        input_shape: tuple[int, int, int, int],
        *,
        num_downsample_convs: int = 1,
        num_post_downsample_convs: int = 3,
        channels: int = 32,
        kernel_size: int = 3,
        padding: int = 0,
        channels_multiplier: int = 1,
        activation: str = "relu",
        norm: str = "none",
        normalize_inputs: bool = True,
        jit: bool = True,
        seed: int = 0,
        **unused,
    ):
        del unused
        if len(input_shape) != 4:
            raise ValueError(
                "DrQ-v2 RGB input must be [views, channels, H, W], "
                f"got {input_shape}."
            )
        if num_downsample_convs < 0 or num_post_downsample_convs < 0:
            raise ValueError("DrQ-v2 convolution counts must be non-negative.")
        if num_downsample_convs + num_post_downsample_convs < 1:
            raise ValueError("DrQ-v2 encoder requires at least one convolution.")
        if channels < 1 or channels_multiplier < 1:
            raise ValueError("DrQ-v2 encoder channels must be positive.")
        if kernel_size < 1 or padding < 0:
            raise ValueError("DrQ-v2 kernel_size must be positive and padding >= 0.")

        self._input_shape = tuple(int(value) for value in input_shape)
        self._model = _DrQV2MultiViewCNN(
            num_views=self._input_shape[0],
            num_downsample_convs=int(num_downsample_convs),
            num_post_downsample_convs=int(num_post_downsample_convs),
            channels=int(channels),
            kernel_size=int(kernel_size),
            padding=int(padding),
            channels_multiplier=int(channels_multiplier),
            activation_name=str(activation).lower(),
            norm=str(norm).lower(),
            normalize_inputs=bool(normalize_inputs),
        )
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


def _normalize_pretrained_weights_path(
    pretrained_weights_path: str | os.PathLike[str] | None,
) -> Path | None:
    if pretrained_weights_path is None:
        return None
    return Path(pretrained_weights_path).expanduser().resolve()


def _pretrained_resnet_candidates(
    model_name: str,
    pretrained_weights_path: str | os.PathLike[str] | None = None,
) -> list[Path]:
    env_name = f"ROBOBASE_{model_name.upper()}_JAX_NPZ"
    candidates = []
    explicit_path = _normalize_pretrained_weights_path(pretrained_weights_path)
    if explicit_path is not None:
        candidates.append(explicit_path)
    env_value = os.environ.get(env_name)
    if env_value:
        candidates.append(Path(env_value).expanduser())
    if model_name == "resnet18":
        candidates.append(_DEFAULT_RESNET18_JAX_NPZ)
    return candidates


def _resolve_pretrained_resnet_npz(
    model_name: str,
    pretrained_weights_path: str | os.PathLike[str] | None = None,
) -> Path | None:
    explicit_path = _normalize_pretrained_weights_path(pretrained_weights_path)
    if explicit_path is not None and not explicit_path.is_file():
        raise FileNotFoundError(
            f"Explicit pretrained ResNet weights do not exist: {explicit_path}"
        )
    return next(
        (
            path
            for path in _pretrained_resnet_candidates(
                model_name,
                pretrained_weights_path,
            )
            if path.exists()
        ),
        None,
    )


def _downloads_enabled() -> bool:
    value = os.environ.get("ROBOBASE_DISABLE_PRETRAINED_DOWNLOAD", "")
    return value.strip().lower() not in {"1", "true", "yes", "on"}


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


def _sha256(path: Path) -> str:
    stat = path.stat()
    return _file_sha256(
        str(path.resolve()),
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _ensure_resnet18_safetensors() -> Path:
    path = _DEFAULT_RESNET18_SAFETENSORS
    if path.exists() and _sha256(path) == _RESNET18_SAFETENSORS_SHA256:
        return path
    if not _downloads_enabled():
        raise FileNotFoundError(
            "ResNet18 pretrained weights are not cached and automatic download is "
            "disabled by ROBOBASE_DISABLE_PRETRAINED_DOWNLOAD."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            try:
                with urllib.request.urlopen(  # noqa: S310 - pinned HTTPS + SHA256
                    _RESNET18_SAFETENSORS_URL,
                    timeout=60,
                ) as response:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        output.write(chunk)
            except Exception as exc:
                raise FileNotFoundError(
                    "Unable to download the pinned ResNet18 ImageNet weights from "
                    f"{_RESNET18_SAFETENSORS_URL}. Set ROBOBASE_RESNET18_JAX_NPZ "
                    "to a converted local checkpoint when running offline."
                ) from exc
        if _sha256(temporary_path) != _RESNET18_SAFETENSORS_SHA256:
            raise ValueError(
                "Downloaded ResNet18 weights failed SHA256 verification; refusing "
                "to load them."
            )
        os.replace(temporary_path, path)
        temporary_path = None
        return path
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def resnet_weight_fingerprint(
    model_name: str,
    pretrained: bool,
    pretrained_weights_path: str | os.PathLike[str] | None = None,
    *,
    seed: int = 0,
) -> str:
    """Return a stable identity for the exact frozen encoder weights."""
    if not pretrained:
        return f"flax-resnet-random-init-seed-{int(seed)}"
    path = _resolve_pretrained_resnet_npz(model_name, pretrained_weights_path)
    if path is None and model_name == "resnet18":
        path = _ensure_resnet18_safetensors()
    if path is None:
        return f"missing-jax-npz:{model_name}"
    return "sha256:" + _sha256(path)


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


_COMPUTE_DTYPES = {
    "float32": jnp.float32,
    "bfloat16": jnp.bfloat16,
}


def resolve_compute_dtype(compute_dtype) -> jnp.dtype | None:
    """Map an ``encoder_model.compute_dtype`` string to a JAX dtype.

    ``None``/``"float32"`` keep the historical float32 path bit-identical.
    ``"bfloat16"`` runs the ResNet convolutions and stores their activations
    in bfloat16 (parameters, batch statistics, the normalisation arithmetic
    and every consumer stay float32), which halves the trunk's activation
    memory and uses the tensor-core bf16 path instead of TF32.
    """

    if compute_dtype is None:
        return None
    if isinstance(compute_dtype, str):
        key = compute_dtype.strip().lower()
        if key not in _COMPUTE_DTYPES:
            raise ValueError(
                "encoder_model.compute_dtype must be one of "
                f"{sorted(_COMPUTE_DTYPES)}, got {compute_dtype!r}."
            )
        resolved = _COMPUTE_DTYPES[key]
    else:
        resolved = jnp.dtype(compute_dtype)
    if resolved == jnp.float32:
        return None
    return resolved


def _conv2d(
    x: jnp.ndarray,
    kernel: jnp.ndarray,
    *,
    strides: tuple[int, int],
    padding,
    compute_dtype=None,
) -> jnp.ndarray:
    if compute_dtype is not None:
        x = x.astype(compute_dtype)
        kernel = kernel.astype(compute_dtype)
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
    # Match Torch FrozenBatchNorm2d: the affine values participate in the
    # forward pass but are buffers, not differentiable parameters.  They stay
    # in the JAX checkpoint tree for backward compatibility.
    scale = jax.lax.stop_gradient(params["scale"]).reshape((1, 1, 1, -1))
    bias = jax.lax.stop_gradient(params["bias"]).reshape((1, 1, 1, -1))
    return (x - mean) * jax.lax.rsqrt(
        var + jnp.asarray(eps, dtype=x.dtype)
    ) * scale + bias


def _resnet_conv_block(
    x: jnp.ndarray,
    params,
    stats,
    name: str,
    *,
    strides: tuple[int, int] = (1, 1),
    padding=((0, 0), (0, 0)),
    activate: bool = True,
    compute_dtype=None,
) -> jnp.ndarray:
    block_params = params[name]
    block_stats = stats[name]
    x = _conv2d(
        x,
        block_params["Conv_0"]["kernel"],
        strides=strides,
        padding=padding,
        compute_dtype=compute_dtype,
    )
    # Frozen-statistics normalisation promotes to float32 (the parameters and
    # statistics are float32); the stored activation is cast back afterwards.
    x = _batch_norm(
        x,
        block_params["BatchNorm_0"],
        block_stats["BatchNorm_0"],
    )
    x = jax.nn.relu(x) if activate else x
    return x if compute_dtype is None else x.astype(compute_dtype)


def _apply_resnet18_block(
    x: jnp.ndarray,
    params,
    stats,
    *,
    strides: tuple[int, int],
    film: jnp.ndarray | None = None,
    compute_dtype=None,
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
        compute_dtype=compute_dtype,
    )
    y = _resnet_conv_block(
        y,
        params,
        stats,
        "ConvBlock_1",
        strides=(1, 1),
        padding=((1, 1), (1, 1)),
        activate=False,
        compute_dtype=compute_dtype,
    )
    if film is not None:
        gamma = film[:, 0].reshape((film.shape[0], 1, 1, -1))
        beta = film[:, 1].reshape((film.shape[0], 1, 1, -1))
        y = (1.0 + gamma) * y + beta
        if compute_dtype is not None:
            y = y.astype(compute_dtype)
    if "ResNetSkipConnection_0" in params:
        residual = _resnet_conv_block(
            residual,
            params["ResNetSkipConnection_0"],
            stats["ResNetSkipConnection_0"],
            "ConvBlock_0",
            strides=strides,
            padding=((0, 0), (0, 0)),
            activate=False,
            compute_dtype=compute_dtype,
        )
    return jax.nn.relu(y + residual)


def _apply_resnet18_film(
    x: jnp.ndarray,
    resnet_variables,
    film_params,
    task_emb: jnp.ndarray,
    compute_dtype=None,
    use_remat: bool = True,
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
        compute_dtype=compute_dtype,
    )
    x = max_pool_3x3_stride2_pad1(x)

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
                film_cache[film_name] = _linear(
                    film_params[film_name], task_emb
                ).reshape((task_emb.shape[0], 2, 2, int(planes)))
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
                compute_dtype=compute_dtype,
            )

        block_fn = jax.checkpoint(apply_block) if use_remat else apply_block
        x = block_fn(
            x,
            params[layer_name],
            stats[layer_name],
            block_film,
        )
    return x if compute_dtype is None else x.astype(jnp.float32)


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
            pad = int(kernel_size[0]) // 2
            x = nn.Conv(
                features=int(features),
                kernel_size=kernel_size,
                strides=(2, 2),
                padding=((pad, pad), (pad, pad)),
                use_bias=False,
                # Match torch.nn.Conv2d.reset_parameters():
                # kaiming_uniform_(a=sqrt(5), mode="fan_in").
                kernel_init=_PYTORCH_CONV_KERNEL_INIT,
                name=f"conv_{index}",
            )(x)
            x = _frozen_batch_norm_identity(x)
            x = nn.relu(x)
        return x


def normalize_plucker_fusion_mode(
    mode,
    *,
    use_plucker: bool | None = None,
) -> str:
    """Normalize the explicit CamPose integration strategy.

    ``projected_late`` is retained only for checkpoints created before the
    official ACT/DP paths were separated. New code should select ``act_late``
    or ``dp_early`` explicitly. A non-``None`` ``use_plucker`` acts as a config
    enable gate; standalone fusion modules omit it so their explicit mode wins.
    """

    if use_plucker is False:
        # ``use_plucker`` is the runtime ablation gate. Keep accepting the
        # configured family marker so dp_resnet can switch between its 9-channel
        # official path and the otherwise identical 3-channel RGB control.
        return "none"
    if mode is None:
        return "projected_late" if use_plucker is True else "none"
    normalized = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "off": "none",
        "false": "none",
        "rgb": "none",
        "late": "act_late",
        "act": "act_late",
        "early": "dp_early",
        "dp": "dp_early",
        "diffusion": "dp_early",
        "legacy": "projected_late",
        "legacy_projected_late": "projected_late",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"none", "act_late", "dp_early", "projected_late"}:
        raise ValueError(
            f"plucker_fusion_mode must be one of none/act_late/dp_early; got {mode!r}."
        )
    return normalized


class JaxPluckerFusion(nn.Module):
    """Plug-and-play CamPose fusion primitive operating on NHWC tensors.

    ``act_late`` applies the official five-layer ray CNN and returns the
    unprojected RGB/ray concatenation. The ACT image projection can therefore
    perform the official single ``1024 -> hidden_dim`` 1x1 projection.
    ``dp_early`` returns raw ``RGB(3) + Plucker(6)`` channels for the diffusion
    ResNet. ``none`` is an exact identity and creates no parameters.
    """

    mode: str
    plucker_out_channels: int = 512
    plucker_hidden_channels: int = 64

    @nn.compact
    def __call__(
        self,
        rgb: jnp.ndarray,
        raymap: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        mode = normalize_plucker_fusion_mode(self.mode)
        if mode == "none":
            return rgb
        if mode == "projected_late":
            raise ValueError("JaxPluckerFusion does not expose legacy projected_late.")
        if raymap is None:
            raise ValueError(f"Plucker fusion mode {mode!r} requires a raymap.")
        if rgb.ndim != 4 or raymap.ndim != 4:
            raise ValueError(
                "Plucker fusion expects NHWC rank-4 tensors; "
                f"got rgb={rgb.shape}, raymap={raymap.shape}."
            )
        if rgb.shape[0] != raymap.shape[0]:
            raise ValueError("RGB and raymap batch dimensions must match.")
        if int(raymap.shape[-1]) != 6:
            raise ValueError(f"Expected six Plucker channels, got {raymap.shape[-1]}.")
        if mode == "dp_early":
            if int(rgb.shape[-1]) != 3:
                raise ValueError(
                    f"DP early fusion expects RGB(3), got {rgb.shape[-1]}."
                )
            if rgb.shape[:3] != raymap.shape[:3]:
                raise ValueError(
                    "DP early fusion requires matching RGB/raymap spatial shapes; "
                    f"got rgb={rgb.shape[:3]}, raymap={raymap.shape[:3]}."
                )
            return jnp.concatenate([rgb, raymap], axis=-1)

        plucker = JaxPluckerEncoder(
            out_channels=int(self.plucker_out_channels),
            hidden_channels=int(self.plucker_hidden_channels),
            name="plucker_encoder",
        )(raymap)
        if plucker.shape[:3] != rgb.shape[:3]:
            raise ValueError(
                "ACT late fusion requires matching RGB/ray feature maps; "
                f"got rgb={rgb.shape[:3]}, ray={plucker.shape[:3]}."
            )
        return jnp.concatenate([rgb, plucker], axis=-1)


def _frozen_batch_norm_identity(x: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    """Apply fixed identity batch-normalization buffers."""

    return x * jax.lax.rsqrt(jnp.asarray(1.0 + eps, dtype=x.dtype))


class JaxPluckerLateFusion(nn.Module):
    """Official late concat + 1x1 projection for RGB and Plucker feature maps."""

    out_channels: int

    @nn.compact
    def __call__(self, rgb_feat: jnp.ndarray, plucker_feat: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([rgb_feat, plucker_feat], axis=-1)
        fan_in = int(x.shape[-1])
        return nn.Conv(
            features=int(self.out_channels),
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            use_bias=True,
            kernel_init=_PYTORCH_CONV_KERNEL_INIT,
            bias_init=nn.initializers.uniform(1.0 / np.sqrt(float(fan_in))),
            name="input_proj",
        )(x)


def _pytorch_default_bias_init(fan_in: int):
    return nn.initializers.uniform(1.0 / np.sqrt(float(max(1, fan_in))))


class _JaxDPResNet18Block(nn.Module):
    """Torchvision ResNet18 basic block with BatchNorm replaced by GroupNorm."""

    features: int
    stride: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        y = nn.Conv(
            self.features,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            kernel_init=_PYTORCH_RESNET_CONV_KERNEL_INIT,
            name="conv1",
        )(x)
        y = nn.GroupNorm(
            num_groups=max(1, self.features // 16),
            epsilon=1e-5,
            name="norm1",
        )(y)
        y = nn.relu(y)
        y = nn.Conv(
            self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            kernel_init=_PYTORCH_RESNET_CONV_KERNEL_INIT,
            name="conv2",
        )(y)
        y = nn.GroupNorm(
            num_groups=max(1, self.features // 16),
            epsilon=1e-5,
            name="norm2",
        )(y)

        if self.stride != 1 or int(x.shape[-1]) != self.features:
            residual = nn.Conv(
                self.features,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                padding="VALID",
                use_bias=False,
                kernel_init=_PYTORCH_RESNET_CONV_KERNEL_INIT,
                name="downsample_conv",
            )(residual)
            residual = nn.GroupNorm(
                num_groups=max(1, self.features // 16),
                epsilon=1e-5,
                name="downsample_norm",
            )(residual)
        return nn.relu(y + residual)


class JaxDPEarlyConcatResNet18(nn.Module):
    """Pure-JAX port of CamPoseOpensource DP's ``RgbEncoder``.

    This preserves the official early 9-channel fusion, GroupNorm ResNet18,
    32-keypoint spatial softmax, and final 64-dimensional projection. It has no
    ImageNet/pretrained path, matching the official ``weights=None`` model.
    """

    use_plucker: bool = True
    num_keypoints: int = 32

    @nn.compact
    def __call__(
        self,
        rgb: jnp.ndarray,
        raymap: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        mode = "dp_early" if self.use_plucker else "none"
        x = JaxPluckerFusion(mode=mode, name="input_fusion")(rgb, raymap)
        expected_channels = 9 if self.use_plucker else 3
        if int(x.shape[-1]) != expected_channels:
            raise ValueError(
                f"DP encoder expected {expected_channels} input channels, got {x.shape[-1]}."
            )

        first_kernel_init = (
            _PYTORCH_CONV_KERNEL_INIT
            if self.use_plucker
            else _PYTORCH_RESNET_CONV_KERNEL_INIT
        )
        x = nn.Conv(
            64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding=((3, 3), (3, 3)),
            use_bias=False,
            kernel_init=first_kernel_init,
            name="conv1",
        )(x)
        x = nn.GroupNorm(num_groups=4, epsilon=1e-5, name="norm1")(x)
        x = nn.relu(x)
        x = max_pool_3x3_stride2_pad1(x)

        for stage, features in enumerate((64, 128, 256, 512)):
            for block in range(2):
                stride = 2 if stage > 0 and block == 0 else 1
                x = _JaxDPResNet18Block(
                    features=features,
                    stride=stride,
                    name=f"layer{stage + 1}_{block}",
                )(x)

        batch_size, height, width, channels = x.shape
        x = nn.Conv(
            self.num_keypoints,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            kernel_init=_PYTORCH_CONV_KERNEL_INIT,
            bias_init=_pytorch_default_bias_init(int(channels)),
            name="spatial_softmax_conv",
        )(x)
        x = jnp.transpose(x, (0, 3, 1, 2)).reshape(
            (batch_size * self.num_keypoints, height * width)
        )
        attention = jax.nn.softmax(x, axis=-1)
        pos_y, pos_x = jnp.meshgrid(
            jnp.linspace(-1.0, 1.0, height, dtype=jnp.float32),
            jnp.linspace(-1.0, 1.0, width, dtype=jnp.float32),
            indexing="ij",
        )
        pos_grid = jnp.stack([pos_x.reshape(-1), pos_y.reshape(-1)], axis=-1)
        x = (attention @ pos_grid).reshape((batch_size, self.num_keypoints * 2))
        feature_dim = self.num_keypoints * 2
        x = nn.Dense(
            feature_dim,
            kernel_init=_PYTORCH_CONV_KERNEL_INIT,
            bias_init=_pytorch_default_bias_init(feature_dim),
            name="proj",
        )(x)
        return nn.relu(x)


def _identity_plucker_late_fusion_variables(num_features: int) -> FrozenDict:
    kernel = jnp.zeros(
        (1, 1, int(num_features) * 2, int(num_features)), dtype=jnp.float32
    )
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
    """Build CamPose policy-compatible moment+direction maps on device."""

    intrinsics = intrinsics.astype(jnp.float32)
    c2ws = c2ws.astype(jnp.float32)
    v, u = jnp.meshgrid(
        jnp.arange(height, dtype=jnp.float32) + 0.5,
        jnp.arange(width, dtype=jnp.float32) + 0.5,
        indexing="ij",
    )
    u = u.reshape((1, -1))
    v = v.reshape((1, -1))
    # CamPoseOpensource's policy embedder applies this second +0.5 after its
    # pixel-center grid has already been shifted by +0.5.
    x = (u - intrinsics[:, 0, 2:3] + 0.5) / intrinsics[:, 0, 0:1]
    y = -(v - intrinsics[:, 1, 2:3] + 0.5) / intrinsics[:, 1, 1:2]
    camera_dirs = jnp.stack([x, y, -jnp.ones_like(x)], axis=-1)
    directions = jnp.einsum("npc,nkc->npk", camera_dirs, c2ws[:, :3, :3])
    origins = jnp.broadcast_to(c2ws[:, None, :3, 3], directions.shape)
    viewdirs = directions / (
        jnp.linalg.norm(directions, axis=-1, keepdims=True)
        + jnp.asarray(1e-8, dtype=jnp.float32)
    )
    moments = jnp.cross(origins, viewdirs, axis=-1)
    return jnp.concatenate([moments, viewdirs], axis=-1).reshape(
        (intrinsics.shape[0], height, width, 6)
    )


def _resnet18_safetensors_to_variables(tensors: dict[str, np.ndarray]) -> FrozenDict:
    """Convert torchvision/timm ResNet18 tensors without importing Torch."""

    flat_variables: dict[tuple[str, ...], jnp.ndarray] = {}

    def add_conv(source: str, target: tuple[str, ...]) -> None:
        kernel = np.asarray(tensors[source], dtype=np.float32)
        flat_variables[("params", *target, "kernel")] = jnp.asarray(
            kernel.transpose(2, 3, 1, 0)
        )

    def add_batch_norm(source: str, target: tuple[str, ...]) -> None:
        for source_name, collection, target_name in (
            ("weight", "params", "scale"),
            ("bias", "params", "bias"),
            ("running_mean", "batch_stats", "mean"),
            ("running_var", "batch_stats", "var"),
        ):
            flat_variables[(collection, *target, target_name)] = jnp.asarray(
                tensors[f"{source}.{source_name}"], dtype=jnp.float32
            )

    stem = ("layers_0", "ConvBlock_0")
    add_conv("conv1.weight", (*stem, "Conv_0"))
    add_batch_norm("bn1", (*stem, "BatchNorm_0"))

    for stage in range(1, 5):
        for block in range(2):
            layer = f"layers_{2 + (stage - 1) * 2 + block}"
            source = f"layer{stage}.{block}"
            for conv_index in (1, 2):
                block_name = f"ConvBlock_{conv_index - 1}"
                add_conv(
                    f"{source}.conv{conv_index}.weight",
                    (layer, block_name, "Conv_0"),
                )
                add_batch_norm(
                    f"{source}.bn{conv_index}",
                    (layer, block_name, "BatchNorm_0"),
                )
            if stage > 1 and block == 0:
                skip = (layer, "ResNetSkipConnection_0", "ConvBlock_0")
                add_conv(f"{source}.downsample.0.weight", (*skip, "Conv_0"))
                add_batch_norm(f"{source}.downsample.1", (*skip, "BatchNorm_0"))

    if len(flat_variables) != 100:
        raise ValueError(
            "Pinned ResNet18 checkpoint conversion produced an incomplete Flax tree: "
            f"expected 100 leaves, got {len(flat_variables)}."
        )
    return freeze(unflatten_dict(flat_variables))


def _load_pretrained_resnet_npz(
    model_name: str,
    pretrained_weights_path: str | os.PathLike[str] | None = None,
) -> FrozenDict:
    env_name = f"ROBOBASE_{model_name.upper()}_JAX_NPZ"
    candidates = _pretrained_resnet_candidates(model_name, pretrained_weights_path)
    path = _resolve_pretrained_resnet_npz(model_name, pretrained_weights_path)
    if path is not None:
        if path.suffix == ".safetensors":
            if model_name != "resnet18":
                raise ValueError(
                    "Direct safetensors conversion currently supports resnet18 only."
                )
            try:
                from safetensors.numpy import load_file
            except ImportError as exc:
                raise ImportError(
                    "Loading ResNet safetensors requires the safetensors package."
                ) from exc
            return _resnet18_safetensors_to_variables(load_file(path))
        if path.suffix != ".npz":
            raise ValueError(
                "Pretrained ResNet weights must be a JAX .npz or a supported "
                f".safetensors file, got: {path}"
            )
        with np.load(path, allow_pickle=False) as arrays:
            flat_variables = {
                tuple(str(key).split("/")): jnp.asarray(arrays[key], dtype=jnp.float32)
                for key in arrays.files
            }
        return freeze(unflatten_dict(flat_variables))

    if model_name == "resnet18":
        try:
            from safetensors.numpy import load_file
        except ImportError as exc:
            raise ImportError(
                "Loading the pinned ResNet18 checkpoint requires safetensors. "
                "Install RoboBase with the 'jax' or 'jax-cuda12' extra."
            ) from exc
        return _resnet18_safetensors_to_variables(
            load_file(_ensure_resnet18_safetensors())
        )

    searched = ", ".join(str(path) for path in candidates) or f"${env_name} (unset)"
    raise FileNotFoundError(
        f"pretrained=true for '{model_name}' requires a converted JAX ResNet npz. "
        f"Searched: {searched}. Set {env_name} to override."
    )


def _load_resnet_feature_model(
    model_name: str,
    pretrained: bool = False,
    seed: int = 0,
    pretrained_weights_path: str | os.PathLike[str] | None = None,
):
    explicit_path = _normalize_pretrained_weights_path(pretrained_weights_path)
    weight_fingerprint = resnet_weight_fingerprint(
        model_name,
        pretrained,
        explicit_path,
    )
    return _load_resnet_feature_model_cached(
        model_name,
        pretrained,
        weight_fingerprint,
        int(seed),
        None if explicit_path is None else str(explicit_path),
    )


@lru_cache(maxsize=8)
def _load_resnet_feature_model_cached(
    model_name: str,
    pretrained: bool,
    weight_fingerprint: str,
    seed: int = 0,
    pretrained_weights_path: str | None = None,
):
    del weight_fingerprint
    if model_name not in _SUPPORTED_RESNETS:
        raise NotImplementedError(
            f"JAX encoder supports only {sorted(_SUPPORTED_RESNETS)}. Got '{model_name}'."
        )
    feature_model = resnet_feature_model(_SUPPORTED_RESNETS[model_name])
    if pretrained:
        feature_variables = _load_pretrained_resnet_npz(
            model_name,
            pretrained_weights_path,
        )
    else:
        feature_variables = feature_model.init(
            jax.random.PRNGKey(seed),
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
        plucker_fusion_mode: str | None = None,
        plucker_hidden_channels: int = 64,
        plucker_identity_init: bool = False,
        use_film: bool = False,
        film_task_input_dim: int = 512,
        film_task_hidden_dim: int = 256,
        seed: int = 0,
        pretrained_weights_path: str | os.PathLike[str] | None = None,
        compute_dtype=None,
        use_remat: bool = True,
    ):
        if input_shape[1] % 3 != 0:
            raise ValueError(
                "ResNet RGB input channels must be a multiple of 3; "
                f"got input_shape={input_shape}."
            )

        seed = int(seed)
        self._compute_dtype = resolve_compute_dtype(compute_dtype)
        self._use_remat = bool(use_remat)

        def init_key(legacy_index: int):
            if seed == 0:
                return jax.random.PRNGKey(legacy_index)
            return jax.random.fold_in(jax.random.PRNGKey(seed), legacy_index)

        if pretrained_weights_path is not None and not pretrained:
            raise ValueError(
                "pretrained_weights_path requires pretrained=true."
            )
        if pretrained_weights_path is not None:
            feature_model, feature_variables, num_features = (
                _load_resnet_feature_model(
                    model,
                    bool(pretrained),
                    seed=seed,
                    pretrained_weights_path=pretrained_weights_path,
                )
            )
        elif seed == 0:
            feature_model, feature_variables, num_features = _load_resnet_feature_model(
                model,
                bool(pretrained),
            )
        else:
            feature_model, feature_variables, num_features = _load_resnet_feature_model(
                model,
                bool(pretrained),
                seed=seed,
            )
        if self._compute_dtype is not None:
            # Same variable tree; only the convolution/activation dtype differs.
            feature_model = resnet_feature_model(
                _SUPPORTED_RESNETS[model], dtype=self._compute_dtype
            )
        self._feature_model = feature_model
        self._feature_variables = jax.tree.map(
            lambda x: jnp.asarray(x, dtype=jnp.float32),
            feature_variables,
        )
        self._num_features = int(num_features)
        self._input_shape = input_shape
        self._resize_to_224 = bool(resize_to_224)
        self._plucker_fusion_mode = normalize_plucker_fusion_mode(
            plucker_fusion_mode,
            use_plucker=bool(use_plucker),
        )
        if self._plucker_fusion_mode == "dp_early":
            raise ValueError(
                "Use JaxDPEarlyFusionEncoder for plucker_fusion_mode='dp_early'; "
                "JaxResNetEncoder implements RGB and ACT spatial backbones."
            )
        self._use_plucker = self._plucker_fusion_mode != "none"
        self._use_film = bool(use_film)
        if self._use_film and model != "resnet18":
            raise NotImplementedError(
                "ACT FiLM conditioning currently supports resnet18 only."
            )
        self._plucker_model = None
        self._plucker_feature_variables = None
        self._plucker_fusion_model = None
        self._plucker_fusion_variables = None
        self._film_variables = None
        self._mean = jnp.asarray(_IMAGENET_MEAN.reshape((1, 1, 1, 3)))
        self._std = jnp.asarray(_IMAGENET_STD.reshape((1, 1, 1, 3)))

        if self._use_film:
            self._film_variables = _init_resnet18_film_variables(
                init_key(2),
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
                init_key(0),
                dummy_raymap,
            )
            if self._plucker_fusion_mode == "projected_late":
                self._plucker_fusion_model = JaxPluckerLateFusion(
                    out_channels=self._num_features,
                )
                if bool(plucker_identity_init):
                    self._plucker_fusion_variables = (
                        _identity_plucker_late_fusion_variables(self._num_features)
                    )
                else:
                    dummy_feat = jnp.zeros(
                        (1, 1, 1, self._num_features),
                        dtype=jnp.float32,
                    )
                    self._plucker_fusion_variables = self._plucker_fusion_model.init(
                        init_key(1),
                        dummy_feat,
                        dummy_feat,
                    )

        self._jit = bool(jit)
        self._refresh_encode_fn()

    @property
    def output_shape(self) -> tuple[int, int]:
        feature_size = (
            self._num_features * 2
            if self._plucker_fusion_mode == "act_late"
            else self._num_features
        )
        return (self._input_shape[0], feature_size)

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
            if self._plucker_fusion_variables is not None:
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
                if "fusion" in params:
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
                if "fusion" in params:
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
                x,
                shape=(x.shape[0], 224, 224, x.shape[-1]),
                method="bilinear",
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
            raise ValueError(
                f"Expected raymap channels divisible by 6, got {channels}."
            )
        frame_stack = channels // 6
        height, width = int(raymap.shape[-2]), int(raymap.shape[-1])
        raymap = raymap.reshape((batch_size, num_views, frame_stack, 6, height, width))
        raymap = jnp.transpose(raymap, (0, 1, 2, 4, 5, 3))
        raymap = raymap.reshape(
            (batch_size * num_views * frame_stack, height, width, 6)
        )
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
            encoded_task_emb = _linear(
                film_params["text_proj"], task_emb.astype(jnp.float32)
            )
            expanded_task = jnp.broadcast_to(
                encoded_task_emb[:, None, None, :],
                (batch_size, num_views, frame_stack, encoded_task_emb.shape[-1]),
            ).reshape(
                (batch_size * num_views * frame_stack, encoded_task_emb.shape[-1])
            )
            x = _apply_resnet18_film(
                x,
                self._resnet_variables(variables),
                film_params,
                expanded_task,
                compute_dtype=self._compute_dtype,
                use_remat=self._use_remat,
            )
        else:
            x = self._feature_model.apply(
                self._resnet_variables(variables),
                x,
            )
            if self._compute_dtype is not None:
                x = x.astype(jnp.float32)

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
                    raymap_obs,
                    batch_size,
                    num_views,
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
            if self._plucker_fusion_mode == "act_late":
                x = jnp.concatenate([x, plucker], axis=-1)
            else:
                x = self._plucker_fusion_model.apply(
                    self._get_plucker_fusion_variables(variables),
                    x,
                    plucker,
                )
            feature_size = int(x.shape[-1])
            x = x.reshape(
                (
                    batch_size,
                    num_views,
                    frame_stack,
                    x.shape[1],
                    x.shape[2],
                    feature_size,
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


class JaxDPEarlyFusionEncoder:
    """Runtime wrapper for the official CamPose diffusion image encoder.

    Its public training surface mirrors ``JaxResNetEncoder`` so Diffusion and
    Flow Matching can select it in their encoder factory without special
    forward/update logic. The model is intentionally trainable and random-init,
    matching CamPoseOpensource's ``torchvision.resnet18(weights=None)``.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int, int],
        model: str = "resnet18",
        jit: bool = True,
        pretrained: bool = False,
        use_plucker: bool = True,
        plucker_fusion_mode: str | None = "dp_early",
        num_keypoints: int = 32,
        seed: int = 0,
        **unused_kwargs,
    ):
        del unused_kwargs
        if model != "resnet18":
            raise NotImplementedError(
                "CamPose DP early fusion is defined only for ResNet18."
            )
        if pretrained:
            raise ValueError(
                "CamPose DP early fusion uses random ResNet18 weights; "
                "pretrained=true would not match the official policy."
            )
        mode = normalize_plucker_fusion_mode(
            plucker_fusion_mode,
            use_plucker=bool(use_plucker),
        )
        if mode not in {"none", "dp_early"}:
            raise ValueError(
                f"JaxDPEarlyFusionEncoder supports only none/dp_early, got {mode!r}."
            )
        if int(input_shape[1]) % 3 != 0:
            raise ValueError(
                "DP RGB input channels must be a multiple of 3; "
                f"got input_shape={input_shape}."
            )

        self._input_shape = tuple(int(value) for value in input_shape)
        self._mode = mode
        self._use_plucker = mode == "dp_early"
        self._num_keypoints = int(num_keypoints)
        self._feature_size = self._num_keypoints * 2
        self._model = JaxDPEarlyConcatResNet18(
            use_plucker=self._use_plucker,
            num_keypoints=self._num_keypoints,
        )
        height, width = self._input_shape[2:]
        dummy_rgb = jnp.zeros((1, height, width, 3), dtype=jnp.float32)
        dummy_raymap = (
            jnp.zeros((1, height, width, 6), dtype=jnp.float32)
            if self._use_plucker
            else None
        )
        self._variables = self._model.init(
            jax.random.PRNGKey(int(seed)),
            dummy_rgb,
            dummy_raymap,
        )
        self._jit = bool(jit)

        def encode_impl(params, rgb, raymap, intrinsics, c2ws):
            return self._encode_impl(
                params,
                rgb,
                raymap,
                intrinsics,
                c2ws,
                True,
            )

        self._encode = jax.jit(encode_impl) if self._jit else encode_impl

    @property
    def output_shape(self) -> tuple[int, int]:
        return (self._input_shape[0], self._feature_size)

    @property
    def trainable_params(self):
        return self._variables["params"]

    @property
    def batch_stats(self):
        return None

    def frozen_state_dict(self) -> dict:
        return {}

    def load_frozen_state_dict(self, state_dict: dict | None) -> None:
        del state_dict

    def encode(
        self,
        rgb_obs,
        raymap_obs=None,
        camera_intrinsic_obs=None,
        camera_c2w_obs=None,
    ) -> jnp.ndarray:
        return self._encode(
            self.trainable_params,
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
        del task_emb
        return self._encode_impl(
            params,
            rgb_obs,
            raymap_obs,
            camera_intrinsic_obs,
            camera_c2w_obs,
            False,
        )

    def _preprocess_rgb(
        self,
        rgb_obs: jnp.ndarray,
    ) -> tuple[jnp.ndarray, int, int, int, int, int]:
        if rgb_obs.ndim != 5:
            raise ValueError(
                "Expected RGB shape (batch, views, channels, height, width), "
                f"got {rgb_obs.shape}."
            )
        x = rgb_obs.astype(jnp.float32)
        batch_size, num_views, channels, height, width = x.shape
        if channels % 3 != 0:
            raise ValueError(f"Expected RGB channels divisible by 3, got {channels}.")
        frame_stack = channels // 3
        x = x.reshape((batch_size, num_views, frame_stack, 3, height, width))
        x = jnp.transpose(x, (0, 1, 2, 4, 5, 3)).reshape(
            (batch_size * num_views * frame_stack, height, width, 3)
        )
        # Official DP receives ToTensor RGB in [0, 1] and does not apply
        # ImageNet normalization because its ResNet is random-initialized.
        return x / 255.0, batch_size, num_views, frame_stack, height, width

    @staticmethod
    def _preprocess_raymap(
        raymap_obs: jnp.ndarray,
        batch_size: int,
        num_views: int,
    ) -> tuple[jnp.ndarray, int]:
        if raymap_obs.ndim != 5:
            raise ValueError(
                "Expected raymap shape (batch, views, channels, height, width), "
                f"got {raymap_obs.shape}."
            )
        if raymap_obs.shape[:2] != (batch_size, num_views):
            raise ValueError("Raymap batch/view dimensions must match RGB.")
        channels = int(raymap_obs.shape[2])
        if channels % 6 != 0:
            raise ValueError(
                f"Expected raymap channels divisible by 6, got {channels}."
            )
        frame_stack = channels // 6
        height, width = raymap_obs.shape[-2:]
        raymap = raymap_obs.astype(jnp.float32).reshape(
            (batch_size, num_views, frame_stack, 6, height, width)
        )
        raymap = jnp.transpose(raymap, (0, 1, 2, 4, 5, 3)).reshape(
            (batch_size * num_views * frame_stack, height, width, 6)
        )
        return raymap, frame_stack

    @staticmethod
    def _preprocess_camera_params(
        intrinsic_obs: jnp.ndarray,
        c2w_obs: jnp.ndarray,
        batch_size: int,
        num_views: int,
        height: int,
        width: int,
    ) -> tuple[jnp.ndarray, int]:
        if intrinsic_obs.ndim != 5 or c2w_obs.ndim != 5:
            raise ValueError("Expected camera params shaped (B,V,T,3,3)/(B,V,T,4,4).")
        if intrinsic_obs.shape[:2] != (batch_size, num_views):
            raise ValueError("Camera intrinsic batch/view dimensions must match RGB.")
        if c2w_obs.shape[:2] != (batch_size, num_views):
            raise ValueError("Camera c2w batch/view dimensions must match RGB.")
        if intrinsic_obs.shape[2] != c2w_obs.shape[2]:
            raise ValueError("Camera intrinsic/c2w frame stacks must match.")
        if intrinsic_obs.shape[-2:] != (3, 3) or c2w_obs.shape[-2:] != (4, 4):
            raise ValueError("Camera params must end in (3,3)/(4,4).")
        frame_stack = int(intrinsic_obs.shape[2])
        raymap = _plucker_raymap_from_camera_params_jax(
            intrinsic_obs.astype(jnp.float32).reshape((-1, 3, 3)),
            c2w_obs.astype(jnp.float32).reshape((-1, 4, 4)),
            height,
            width,
        )
        return raymap, frame_stack

    def _encode_impl(
        self,
        params,
        rgb_obs: jnp.ndarray,
        raymap_obs: jnp.ndarray | None,
        camera_intrinsic_obs: jnp.ndarray | None,
        camera_c2w_obs: jnp.ndarray | None,
        stop_gradient: bool,
    ) -> jnp.ndarray:
        rgb, batch_size, num_views, frame_stack, height, width = self._preprocess_rgb(
            rgb_obs
        )
        raymap = None
        if self._use_plucker:
            if raymap_obs is not None:
                raymap, ray_frames = self._preprocess_raymap(
                    raymap_obs,
                    batch_size,
                    num_views,
                )
            elif camera_intrinsic_obs is not None and camera_c2w_obs is not None:
                raymap, ray_frames = self._preprocess_camera_params(
                    camera_intrinsic_obs,
                    camera_c2w_obs,
                    batch_size,
                    num_views,
                    height,
                    width,
                )
            else:
                raise ValueError(
                    "DP early fusion requires raymap_obs or intrinsic/c2w observations."
                )
            if ray_frames != frame_stack:
                raise ValueError(
                    f"Ray/RGB frame stacks must match, got {ray_frames}/{frame_stack}."
                )
        features = self._model.apply({"params": params}, rgb, raymap)
        features = features.reshape(
            (batch_size, num_views, frame_stack, self._feature_size)
        ).mean(axis=2)
        return jax.lax.stop_gradient(features) if stop_gradient else features


# --- Legacy (pre-06a61d4) frozen pretrained ResNet -------------------------
# Verbatim port of the 8bf7999-era JaxResNetEncoder. Checkpoints trained
# before the 2026-07-30 pure-JAX encoder rewrite hold parameters in the
# jax_resnet module structure (timm torchvision weights, frozen batch stats);
# the rewritten JaxResNetEncoder above silently mis-applies them (2026-08-08
# compat audit: May FM sandwich 40% -> 4%, May DP -> NaN). Era-gated
# construction in bc/act/diffusion/flow_matching routes old configs here.

_LEGACY_IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_LEGACY_IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
_LEGACY_SUPPORTED_RESNETS = {"resnet18": 18, "resnet34": 34}
_LEGACY_RESNET_FEATURE_SIZES = {"resnet18": 512, "resnet34": 512}


@lru_cache(maxsize=4)
def _load_legacy_pretrained_resnet_feature_model(model_name: str):
    if model_name not in _LEGACY_SUPPORTED_RESNETS:
        raise NotImplementedError(
            "Legacy JAX encoder supports only "
            f"{sorted(_LEGACY_SUPPORTED_RESNETS)}. Got '{model_name}'."
        )
    try:
        import flax.linen as legacy_nn
        import jax_resnet
        import timm
        from jax_resnet.common import slice_variables
    except ImportError as exc:
        raise ImportError(
            "LegacyJaxResNetEncoder requires `flax`, `jax-resnet`, and `timm`."
        ) from exc

    state_dict = timm.create_model(model_name, pretrained=True).state_dict()
    model_cls, variables = jax_resnet.pretrained_resnet(
        _LEGACY_SUPPORTED_RESNETS[model_name],
        state_dict=state_dict,
    )
    model = model_cls()
    feature_model = legacy_nn.Sequential(model.layers[:-1])
    feature_variables = slice_variables(variables, end=-1)
    return (
        feature_model,
        feature_variables,
        _LEGACY_RESNET_FEATURE_SIZES[model_name],
    )


class LegacyJaxResNetEncoder:
    """Frozen pretrained ResNet feature extractor (pre-06a61d4 semantics)."""

    def __init__(
        self,
        input_shape: tuple[int, int, int, int],
        model: str,
        jit: bool = True,
        **_new_era_kwargs,
    ):
        # Tolerate new-era construction kwargs (pretrained, seed, ...); the
        # legacy path is always pretrained with frozen batch statistics.
        if _new_era_kwargs.get("use_plucker"):
            raise ValueError(
                "LegacyJaxResNetEncoder does not support use_plucker."
            )
        assert input_shape[1] == 3, "ResNet only supports channel of size 3"

        feature_model, feature_variables, num_features = (
            _load_legacy_pretrained_resnet_feature_model(model)
        )
        self._feature_model = feature_model
        self._feature_variables = jax.tree.map(
            lambda x: jnp.asarray(x, dtype=jnp.float32),
            feature_variables,
        )
        self._num_features = int(num_features)
        self._input_shape = input_shape
        self._mean = jnp.asarray(_LEGACY_IMAGENET_MEAN.reshape((1, 1, 1, 3)))
        self._std = jnp.asarray(_LEGACY_IMAGENET_STD.reshape((1, 1, 1, 3)))

        encode_impl = self._encode_impl
        if jit:
            encode_impl = jax.jit(encode_impl)
        self._encode = encode_impl

    @property
    def output_shape(self) -> tuple[int, int]:
        return (self._input_shape[0], self._num_features)

    @property
    def trainable_params(self):
        return self._feature_variables["params"]

    @property
    def batch_stats(self):
        return self._feature_variables.get("batch_stats", None)

    def encode(self, rgb_obs) -> jnp.ndarray:
        if hasattr(rgb_obs, "detach"):
            rgb_obs = rgb_obs.detach().cpu().numpy()
        else:
            rgb_obs = np.asarray(rgb_obs)
        return self._encode(jnp.asarray(rgb_obs))

    def apply_trainable(
        self,
        params,
        rgb_obs: jnp.ndarray,
        raymap_obs=None,
        **_new_era_kwargs,
    ) -> jnp.ndarray:
        if raymap_obs is not None:
            raise ValueError(
                "LegacyJaxResNetEncoder does not support raymap conditioning."
            )
        variables = {"params": params}
        if self.batch_stats is not None:
            variables["batch_stats"] = self.batch_stats
        return self._encode_impl(
            rgb_obs,
            variables=freeze(variables),
            stop_gradient=False,
        )

    def _preprocess(self, rgb_obs: jnp.ndarray) -> tuple[jnp.ndarray, int, int]:
        x = rgb_obs.astype(jnp.float32)
        batch_size, num_views, channels, height, width = x.shape
        x = jnp.transpose(x, (0, 1, 3, 4, 2))
        x = x.reshape((batch_size * num_views, height, width, channels))
        x = jax.image.resize(
            x,
            shape=(x.shape[0], 224, 224, x.shape[-1]),
            method="bilinear",
        )
        x = x / 255.0
        x = (x - self._mean) / self._std
        return x, batch_size, num_views

    def _encode_impl(
        self,
        rgb_obs: jnp.ndarray,
        *,
        variables: FrozenDict | None = None,
        stop_gradient: bool = True,
    ) -> jnp.ndarray:
        x, batch_size, num_views = self._preprocess(rgb_obs)
        x = self._feature_model.apply(
            self._feature_variables if variables is None else variables,
            x,
        )
        x = x.reshape((batch_size, num_views, self._num_features))
        if stop_gradient:
            x = jax.lax.stop_gradient(x)
        return x
