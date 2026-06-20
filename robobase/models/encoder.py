"""Frozen pretrained ResNet encoder (Flax)."""

from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict, freeze


_IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
_SUPPORTED_RESNETS = {"resnet18": 18, "resnet34": 34}
_RESNET_FEATURE_SIZES = {"resnet18": 512, "resnet34": 512}


@lru_cache(maxsize=4)
def _load_pretrained_resnet_feature_model(model_name: str):
    if model_name not in _SUPPORTED_RESNETS:
        raise NotImplementedError(
            f"JAX encoder supports only {sorted(_SUPPORTED_RESNETS)}. Got '{model_name}'."
        )
    try:
        import flax.linen as nn
        import jax_resnet
        import timm
        from jax_resnet.common import slice_variables
    except ImportError as exc:
        raise ImportError(
            "JAX ResNet encoder requires `flax`, `jax-resnet`, and `timm`."
        ) from exc

    state_dict = timm.create_model(model_name, pretrained=True).state_dict()
    model_cls, variables = jax_resnet.pretrained_resnet(
        _SUPPORTED_RESNETS[model_name], state_dict=state_dict,
    )
    model = model_cls()
    feature_model = nn.Sequential(model.layers[:-1])
    feature_variables = slice_variables(variables, end=-1)
    return feature_model, feature_variables, _RESNET_FEATURE_SIZES[model_name]


class JaxResNetEncoder:
    """Frozen pretrained ResNet feature extractor.

    This is *not* a trainable ``nn.Module`` — parameters are loaded once and
    frozen via ``stop_gradient``.  Only the JIT-compiled ``encode`` path is
    exposed.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int, int],
        model: str,
        jit: bool = True,
    ):
        assert input_shape[1] == 3, "ResNet only supports channel of size 3"

        feature_model, feature_variables, num_features = (
            _load_pretrained_resnet_feature_model(model)
        )
        self._feature_model = feature_model
        self._feature_variables = jax.tree.map(
            lambda x: jnp.asarray(x, dtype=jnp.float32), feature_variables,
        )
        self._num_features = int(num_features)
        self._input_shape = input_shape
        self._mean = jnp.asarray(_IMAGENET_MEAN.reshape((1, 1, 1, 3)))
        self._std = jnp.asarray(_IMAGENET_STD.reshape((1, 1, 1, 3)))

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

    def apply_trainable(self, params, rgb_obs: jnp.ndarray) -> jnp.ndarray:
        variables = {"params": params}
        if self.batch_stats is not None:
            variables["batch_stats"] = self.batch_stats
        return self._encode_impl(rgb_obs, variables=freeze(variables), stop_gradient=False)

    def _preprocess(self, rgb_obs: jnp.ndarray) -> tuple[jnp.ndarray, int, int]:
        x = rgb_obs.astype(jnp.float32)
        batch_size, num_views, channels, height, width = x.shape
        x = jnp.transpose(x, (0, 1, 3, 4, 2))
        x = x.reshape((batch_size * num_views, height, width, channels))
        x = jax.image.resize(
            x, shape=(x.shape[0], 224, 224, x.shape[-1]), method="bilinear",
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
