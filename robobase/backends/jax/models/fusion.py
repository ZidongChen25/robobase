from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JaxFusionMultiCamFeature:
    input_shape: tuple[int, int]
    mode: str = "flatten"

    def __post_init__(self):
        if self.mode == "flatten":
            output_shape = (int(np.prod(self.input_shape)),)
        elif self.mode in {"average", "sum"}:
            output_shape = (self.input_shape[1],)
        else:
            raise ValueError(f"Mode {self.mode} is not supported.")
        object.__setattr__(self, "_output_shape", output_shape)

    @property
    def output_shape(self):
        return self._output_shape

    def calculate_loss(self, *_args, **_kwargs):
        return None

    def apply(self, jnp, x):
        if self.mode == "flatten":
            return x.reshape((x.shape[0], -1))
        if self.mode == "average":
            return x.mean(axis=1)
        return x.sum(axis=1)

