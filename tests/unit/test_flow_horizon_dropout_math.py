import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy


def _no_padding_weights(
    prefix_lengths: np.ndarray,
    prefix_probs: np.ndarray,
    sequence_length: int,
) -> np.ndarray:
    """Expected per-step weights for a prefix loss normalized by H."""
    positions = np.arange(1, sequence_length + 1)
    return np.asarray(
        [
            np.sum(prefix_probs[prefix_lengths >= position] / prefix_lengths[
                prefix_lengths >= position
            ])
            for position in positions
        ],
        dtype=np.float64,
    )


def _padded_dynamic_coefficients(
    prefix_lengths: np.ndarray,
    prefix_probs: np.ndarray,
    valid_tokens: np.ndarray,
) -> np.ndarray:
    """Expected weights when each sampled prefix is normalized after masking."""
    coefficients = np.zeros(valid_tokens.shape, dtype=np.float64)
    positions = np.arange(1, valid_tokens.size + 1)
    for prefix_length, probability in zip(prefix_lengths, prefix_probs):
        prefix_valid = valid_tokens & (positions <= prefix_length)
        valid_count = prefix_valid.sum()
        assert valid_count > 0
        coefficients += probability * prefix_valid / valid_count
    return coefficients


def test_prefix_dropout_expected_loss_and_gradient_match_static_weights():
    """Lock the exact no-padding Flow horizon-dropout expectation identity."""
    prefix_lengths = np.asarray([1, 3, 5], dtype=np.int32)
    prefix_probs = np.asarray([0.2, 0.3, 0.5], dtype=np.float64)
    sequence_length = 5
    weights = jnp.asarray(
        _no_padding_weights(prefix_lengths, prefix_probs, sequence_length),
        dtype=jnp.float32,
    )

    inputs = jnp.asarray(
        [
            [0.2, -0.4],
            [1.0, 0.3],
            [-0.7, 0.5],
            [0.6, 1.2],
            [-1.1, -0.2],
        ],
        dtype=jnp.float32,
    )
    targets = jnp.asarray(
        [
            [0.1, 0.8],
            [-0.3, 0.2],
            [0.5, -0.6],
            [0.9, 0.4],
            [-0.2, 0.7],
        ],
        dtype=jnp.float32,
    )
    params = {
        "kernel": jnp.asarray([[0.7, -0.1], [0.25, 0.6]], dtype=jnp.float32),
        "bias": jnp.asarray([0.05, -0.15], dtype=jnp.float32),
    }

    def per_step_mse(current_params):
        prediction = inputs @ current_params["kernel"] + current_params["bias"]
        return jnp.square(prediction - targets).mean(axis=-1)

    def enumerated_dropout_expectation(current_params):
        losses = per_step_mse(current_params)
        return sum(
            probability * losses[:prefix_length].mean()
            for prefix_length, probability in zip(prefix_lengths, prefix_probs)
        )

    def deterministic_weighted_loss(current_params):
        return jnp.sum(per_step_mse(current_params) * weights)

    dropout_value, dropout_grad = jax.value_and_grad(
        enumerated_dropout_expectation
    )(params)
    weighted_value, weighted_grad = jax.value_and_grad(
        deterministic_weighted_loss
    )(params)

    np.testing.assert_allclose(np.asarray(weights).sum(), 1.0, atol=1e-7)
    np.testing.assert_allclose(dropout_value, weighted_value, rtol=1e-6, atol=1e-7)
    dropout_leaves = jax.tree_util.tree_leaves(dropout_grad)
    weighted_leaves = jax.tree_util.tree_leaves(weighted_grad)
    for dropout_leaf, weighted_leaf in zip(dropout_leaves, weighted_leaves):
        np.testing.assert_allclose(
            dropout_leaf,
            weighted_leaf,
            rtol=1e-6,
            atol=1e-7,
        )


def test_linear_two_to_one_weights_invert_to_prefix_distribution():
    """Verify the unique prefix distribution represented by linear 2:1 weights."""
    sequence_length = 20
    horizons = np.arange(1, sequence_length + 1, dtype=np.float64)
    weights = np.linspace(1.0 / 15.0, 1.0 / 30.0, sequence_length)

    prefix_probs = np.empty(sequence_length, dtype=np.float64)
    prefix_probs[:-1] = horizons[:-1] * (weights[:-1] - weights[1:])
    prefix_probs[-1] = sequence_length * weights[-1]

    expected_probs = np.concatenate(
        [np.arange(1, sequence_length, dtype=np.float64) / 570.0, [2.0 / 3.0]]
    )
    reconstructed_weights = np.asarray(
        [
            np.sum(prefix_probs[position - 1 :] / horizons[position - 1 :])
            for position in range(1, sequence_length + 1)
        ]
    )

    np.testing.assert_allclose(prefix_probs, expected_probs, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(prefix_probs.sum(), 1.0, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(
        np.dot(horizons, prefix_probs),
        53.0 / 3.0,
        rtol=0.0,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        reconstructed_weights,
        weights,
        rtol=0.0,
        atol=1e-14,
    )


def test_padding_requires_per_sample_dynamic_prefix_coefficients():
    """Show why globally derived static weights are not exact after padding."""
    prefix_lengths = np.asarray([1, 2, 4], dtype=np.int32)
    prefix_probs = np.asarray([0.5, 0.25, 0.25], dtype=np.float64)
    valid_tokens = np.asarray([True, True, False, False])
    per_step_loss = np.asarray([1.0, 4.0, 9.0, 16.0])

    dynamic_coefficients = _padded_dynamic_coefficients(
        prefix_lengths,
        prefix_probs,
        valid_tokens,
    )
    unpadded_weights = _no_padding_weights(
        prefix_lengths,
        prefix_probs,
        sequence_length=valid_tokens.size,
    )
    static_coefficients = unpadded_weights * valid_tokens
    static_coefficients /= static_coefficients.sum()

    positions = np.arange(1, valid_tokens.size + 1)
    enumerated_loss = 0.0
    for prefix_length, probability in zip(prefix_lengths, prefix_probs):
        prefix_valid = valid_tokens & (positions <= prefix_length)
        enumerated_loss += probability * per_step_loss[prefix_valid].mean()

    np.testing.assert_allclose(
        dynamic_coefficients,
        np.asarray([0.75, 0.25, 0.0, 0.0]),
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        static_coefficients,
        np.asarray([11.0 / 14.0, 3.0 / 14.0, 0.0, 0.0]),
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        enumerated_loss,
        np.dot(dynamic_coefficients, per_step_loss),
        rtol=0.0,
        atol=1e-14,
    )
    assert not np.isclose(
        enumerated_loss,
        np.dot(static_coefficients, per_step_loss),
    )
