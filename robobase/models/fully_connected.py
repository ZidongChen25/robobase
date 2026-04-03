"""MLP with sequence output head (Flax)."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class JaxMLPWithSequenceOutput(nn.Module):
    """MLP backbone with either an MLP or RNN sequence output head."""

    hidden_dims: tuple[int, ...]
    action_sequence: int
    action_dim: int
    output_sequence_network_type: str = "mlp"

    @property
    def _use_recurrent_head(self) -> bool:
        return self.action_sequence > 1 and self.output_sequence_network_type == "rnn"

    @nn.compact
    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        x = features.astype(jnp.float32)
        if x.ndim > 2:
            x = x.reshape((x.shape[0], -1))

        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(dim, name=f"dense_{i}")(x)
            x = nn.relu(x)

        if self._use_recurrent_head:
            batch_size = x.shape[0]
            hidden_dim = x.shape[-1]
            repeated = jnp.repeat(x[:, None, :], self.action_sequence, axis=1)
            hidden0 = jnp.zeros((batch_size, hidden_dim), dtype=jnp.float32)

            # Use nn.scan to wrap GRUCell so params are lifted out of the
            # scan body, avoiding tracer leaks.
            ScanGRU = nn.scan(
                nn.GRUCell,
                variable_broadcast="params",
                split_rngs={"params": False},
                in_axes=1,
                out_axes=1,
            )
            scan_gru = ScanGRU(features=hidden_dim, name="gru")
            _, outputs = scan_gru(hidden0, repeated)

            out = nn.Dense(self.action_dim, name="out")(outputs)
            return out.reshape((batch_size, self.action_sequence, self.action_dim))

        out = nn.Dense(self.action_sequence * self.action_dim, name="out")(x)
        return out.reshape((x.shape[0], self.action_sequence, self.action_dim))
