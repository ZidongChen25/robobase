import jax
import jax.numpy as jnp
import numpy as np
import pytest

from robobase.models.a2a import ActionChunkDecoder, TemporalActionEncoder


@pytest.mark.parametrize("horizon", [8, 16, 20])
def test_a2a_encoder_decoder_shapes_for_supported_horizons(horizon):
    batch_size = 3
    action_dim = 6
    latent_dim = 12
    actions = jnp.linspace(
        -1.0,
        1.0,
        batch_size * horizon * action_dim,
        dtype=jnp.float32,
    ).reshape(batch_size, horizon, action_dim)

    encoder = TemporalActionEncoder(latent_dim=latent_dim, hidden_dim=16)
    encoder_variables = encoder.init(jax.random.PRNGKey(0), actions)
    latent = encoder.apply(encoder_variables, actions)

    decoder = ActionChunkDecoder(
        horizon=horizon,
        action_dim=action_dim,
        latent_dim=latent_dim,
        hidden_dim=16,
        num_layers=2,
    )
    decoder_variables = decoder.init(jax.random.PRNGKey(1), latent)
    decoded = decoder.apply(decoder_variables, latent, train=False)
    decoded_from_token = decoder.apply(
        decoder_variables,
        latent[:, None, :],
        train=False,
    )

    assert latent.shape == (batch_size, latent_dim)
    assert decoded.shape == actions.shape
    np.testing.assert_allclose(decoded_from_token, decoded)
    assert np.isfinite(np.asarray(latent)).all()
    assert np.isfinite(np.asarray(decoded)).all()


@pytest.mark.parametrize("horizon", [8, 16, 20])
def test_a2a_encoder_decoder_apply_is_jittable(horizon):
    actions = jnp.ones((2, horizon, 4), dtype=jnp.float32)
    encoder = TemporalActionEncoder(latent_dim=10, hidden_dim=16)
    encoder_variables = encoder.init(jax.random.PRNGKey(2), actions)

    encode = jax.jit(lambda variables, value: encoder.apply(variables, value))
    latent = encode(encoder_variables, actions)

    decoder = ActionChunkDecoder(
        horizon=horizon,
        action_dim=4,
        latent_dim=10,
        hidden_dim=16,
        num_layers=2,
    )
    decoder_variables = decoder.init(jax.random.PRNGKey(3), latent)
    decode = jax.jit(
        lambda variables, value: decoder.apply(variables, value, train=False)
    )
    decoded = decode(decoder_variables, latent)
    decoded.block_until_ready()

    assert latent.shape == (2, 10)
    assert decoded.shape == (2, horizon, 4)
    assert np.isfinite(np.asarray(decoded)).all()


@pytest.mark.parametrize("horizon", [8, 16, 20])
def test_a2a_encoder_decoder_have_finite_end_to_end_gradients(horizon):
    actions = jax.random.normal(
        jax.random.PRNGKey(4),
        shape=(2, horizon, 4),
        dtype=jnp.float32,
    )
    encoder = TemporalActionEncoder(latent_dim=10, hidden_dim=16)
    encoder_params = encoder.init(jax.random.PRNGKey(5), actions)["params"]

    latent = encoder.apply({"params": encoder_params}, actions)
    decoder = ActionChunkDecoder(
        horizon=horizon,
        action_dim=4,
        latent_dim=10,
        hidden_dim=16,
        num_layers=2,
    )
    decoder_params = decoder.init(jax.random.PRNGKey(6), latent)["params"]

    def reconstruction_loss(current_encoder_params, current_decoder_params):
        encoded = encoder.apply({"params": current_encoder_params}, actions)
        reconstructed = decoder.apply(
            {"params": current_decoder_params},
            encoded,
            train=False,
        )
        return jnp.mean(jnp.square(reconstructed - actions))

    loss, (encoder_grads, decoder_grads) = jax.jit(
        jax.value_and_grad(reconstruction_loss, argnums=(0, 1))
    )(encoder_params, decoder_params)
    gradient_groups = (encoder_grads, decoder_grads)

    assert np.isfinite(float(loss))
    for gradients in gradient_groups:
        leaves = jax.tree_util.tree_leaves(gradients)
        assert leaves
        assert all(np.isfinite(np.asarray(leaf)).all() for leaf in leaves)
        squared_norm = sum(jnp.sum(jnp.square(leaf)) for leaf in leaves)
        assert float(squared_norm) > 0.0


def test_a2a_encoder_is_invariant_to_values_under_padding_mask():
    encoder = TemporalActionEncoder(latent_dim=10, hidden_dim=16)
    actions = jax.random.normal(
        jax.random.PRNGKey(7),
        shape=(2, 20, 4),
        dtype=jnp.float32,
    )
    padding_mask = jnp.asarray(
        [[True] * 5 + [False] * 15, [False] * 13 + [True] * 7],
        dtype=jnp.bool_,
    )
    variables = encoder.init(jax.random.PRNGKey(8), actions, padding_mask)
    corrupted = jnp.where(padding_mask[..., None], 1.0e6, actions)

    expected = encoder.apply(variables, actions, padding_mask)
    actual = encoder.apply(variables, corrupted, padding_mask)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    assert variables["params"]["conv_0"]["kernel"].shape[1] == actions.shape[-1] + 1
