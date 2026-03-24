from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from collections.abc import Mapping

import numpy as np


_IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
_SUPPORTED_RESNETS = {
    "resnet18": 18,
    "resnet34": 34,
}
_RESNET_FEATURE_SIZES = {
    "resnet18": 512,
    "resnet34": 512,
}


def _to_device_tree(jnp, tree):
    if tree is None:
        return None
    if isinstance(tree, Mapping):
        return {key: _to_device_tree(jnp, value) for key, value in tree.items()}
    if isinstance(tree, tuple):
        return tuple(_to_device_tree(jnp, value) for value in tree)
    if isinstance(tree, list):
        return [_to_device_tree(jnp, value) for value in tree]
    return jnp.asarray(tree, dtype=jnp.float32)


@lru_cache(maxsize=4)
def _load_pretrained_resnet_feature_model(model_name: str):
    if model_name not in _SUPPORTED_RESNETS:
        raise NotImplementedError(
            "The JAX BC encoder currently supports only "
            f"{sorted(_SUPPORTED_RESNETS)}. Got '{model_name}'."
        )

    try:
        import flax.linen as nn
        import jax_resnet
        import timm
        from jax_resnet.common import slice_variables
    except ImportError as exc:
        raise ImportError(
            "JAX ResNet encoder support requires `flax` and `jax-resnet`. "
            "Install the project with the JAX extras."
        ) from exc

    state_dict = timm.create_model(model_name, pretrained=True).state_dict()
    model_cls, variables = jax_resnet.pretrained_resnet(
        _SUPPORTED_RESNETS[model_name],
        state_dict=state_dict,
    )
    model = model_cls()
    feature_model = nn.Sequential(model.layers[:-1])
    feature_variables = slice_variables(variables, end=-1)
    return feature_model, feature_variables, _RESNET_FEATURE_SIZES[model_name]


@dataclass
class JaxResNetEncoder:
    input_shape: tuple[int, int, int, int]
    model: str
    jit: bool = True

    def __post_init__(self):
        assert self.input_shape[1] == 3, "ResNet only supports channel of size 3"

        import jax
        import jax.numpy as jnp

        feature_model, feature_variables, num_features = (
            _load_pretrained_resnet_feature_model(self.model)
        )
        self.jax = jax
        self.jnp = jnp
        self._feature_model = feature_model
        self._feature_variables = _to_device_tree(jnp, feature_variables)
        self._num_features = int(num_features)
        self._mean = jnp.asarray(_IMAGENET_MEAN.reshape((1, 1, 1, 3)))
        self._std = jnp.asarray(_IMAGENET_STD.reshape((1, 1, 1, 3)))

        encode_impl = self._encode_impl
        if self.jit:
            encode_impl = jax.jit(encode_impl)
        self._encode = encode_impl

    @property
    def output_shape(self):
        return (self.input_shape[0], self._num_features)

    def calculate_loss(self, *_args, **_kwargs):
        return None

    def encode(self, rgb_obs):
        if hasattr(rgb_obs, "detach"):
            rgb_obs = rgb_obs.detach().cpu().numpy()
        else:
            rgb_obs = np.asarray(rgb_obs)
        return self._encode(self.jnp.asarray(rgb_obs))

    def _encode_impl(self, rgb_obs):
        x = rgb_obs.astype(self.jnp.float32)
        batch_size, num_views, channels, height, width = x.shape
        if channels != self.input_shape[1]:
            raise ValueError(
                f"Expected RGB channels {self.input_shape[1]}, got {channels}."
            )
        x = self.jnp.transpose(x, (0, 1, 3, 4, 2))
        x = x.reshape((batch_size * num_views, height, width, channels))
        x = self.jax.image.resize(
            x,
            shape=(x.shape[0], 224, 224, x.shape[-1]),
            method="bilinear",
        )
        x = x / 255.0
        x = (x - self._mean) / self._std
        x = self._feature_model.apply(self._feature_variables, x)
        x = x.reshape((batch_size, num_views, self._num_features))
        return self.jax.lax.stop_gradient(x)
