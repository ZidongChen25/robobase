from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JaxMLPWithSequenceOutput:
    hidden_dims: tuple[int, ...]
    action_sequence: int
    action_dim: int
    output_sequence_network_type: str

    def init_params(self, jax, jnp, key, input_dim: int):
        num_dense_layers = len(self.hidden_dims)
        extra_keys = 2 if self._use_recurrent_head else 1
        keys = jax.random.split(key, num_dense_layers + extra_keys)

        layers = []
        current_dim = input_dim
        for hidden_dim, layer_key in zip(self.hidden_dims, keys[:num_dense_layers]):
            layers.append(self._init_dense(jax, jnp, layer_key, current_dim, hidden_dim))
            current_dim = hidden_dim

        params = {"layers": layers}
        if self._use_recurrent_head:
            params["gru"] = self._init_gru(jax, jnp, keys[-2], current_dim)
            params["out"] = self._init_dense(
                jax, jnp, keys[-1], current_dim, self.action_dim
            )
        else:
            output_dim = self.action_sequence * self.action_dim
            params["out"] = self._init_dense(jax, jnp, keys[-1], current_dim, output_dim)
        return params

    def apply(self, jax, jnp, params, features):
        x = features.astype(jnp.float32)
        if x.ndim > 2:
            x = x.reshape((x.shape[0], -1))

        for layer in params["layers"]:
            x = jax.nn.relu(x @ layer["w"] + layer["b"])

        if self._use_recurrent_head:
            batch_size = x.shape[0]
            repeated = jnp.repeat(x[:, None, :], self.action_sequence, axis=1)
            hidden0 = jnp.zeros((batch_size, x.shape[-1]), dtype=jnp.float32)

            def step(hidden, inputs):
                next_hidden = self._gru_step(jax, jnp, params["gru"], inputs, hidden)
                return next_hidden, next_hidden

            _, outputs = jax.lax.scan(
                step,
                hidden0,
                jnp.swapaxes(repeated, 0, 1),
            )
            outputs = jnp.swapaxes(outputs, 0, 1)
            out = outputs @ params["out"]["w"] + params["out"]["b"]
            return out.reshape((batch_size, self.action_sequence, self.action_dim))

        x = x @ params["out"]["w"] + params["out"]["b"]
        return x.reshape((x.shape[0], self.action_sequence, self.action_dim))

    @property
    def _use_recurrent_head(self) -> bool:
        return self.action_sequence > 1 and self.output_sequence_network_type == "rnn"

    @staticmethod
    def _gru_step(jax, jnp, params, inputs, hidden):
        gates_x = inputs @ params["w_ih"] + params["b_ih"]
        gates_h = hidden @ params["w_hh"] + params["b_hh"]
        i_r, i_z, i_n = jnp.split(gates_x, 3, axis=-1)
        h_r, h_z, h_n = jnp.split(gates_h, 3, axis=-1)
        reset_gate = jax.nn.sigmoid(i_r + h_r)
        update_gate = jax.nn.sigmoid(i_z + h_z)
        candidate = jnp.tanh(i_n + reset_gate * h_n)
        return (1.0 - update_gate) * candidate + update_gate * hidden

    @staticmethod
    def _init_gru(jax, jnp, key, hidden_dim: int):
        key_ih, key_hh = jax.random.split(key)
        return {
            "w_ih": JaxMLPWithSequenceOutput._init_weight(
                jax, jnp, key_ih, hidden_dim, hidden_dim * 3
            ),
            "w_hh": JaxMLPWithSequenceOutput._init_weight(
                jax, jnp, key_hh, hidden_dim, hidden_dim * 3
            ),
            "b_ih": jnp.zeros((hidden_dim * 3,), dtype=jnp.float32),
            "b_hh": jnp.zeros((hidden_dim * 3,), dtype=jnp.float32),
        }

    @staticmethod
    def _init_dense(jax, jnp, key, in_dim: int, out_dim: int):
        return {
            "w": JaxMLPWithSequenceOutput._init_weight(jax, jnp, key, in_dim, out_dim),
            "b": jnp.zeros((out_dim,), dtype=jnp.float32),
        }

    @staticmethod
    def _init_weight(jax, jnp, key, in_dim: int, out_dim: int):
        limit = np.sqrt(6.0 / float(in_dim + out_dim))
        return jax.random.uniform(
            key,
            shape=(in_dim, out_dim),
            minval=-limit,
            maxval=limit,
            dtype=jnp.float32,
        )

