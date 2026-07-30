from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from robobase.method.diffusion import Diffusion, _cosine_betas
from robobase.method.flow_matching import FlowMatching


def test_clean_diffuser_cosine_beta_schedule_golden():
    expected = np.asarray(
        [
            0.042195409536361694,
            0.11567331105470657,
            0.19218802452087402,
            0.2782455086708069,
            0.38362085819244385,
            0.5260363817214966,
            0.740268886089325,
            0.9990000128746033,
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(_cosine_betas(8), expected)


def test_clean_diffuser_ddpm_single_posterior_step_matches_numpy_reference():
    betas = np.asarray([0.05, 0.10, 0.20, 0.30], dtype=np.float32)
    alphas_cumprod = np.cumprod(1.0 - betas).astype(np.float32)
    timestep = 2
    shape = (1, 4, 2)
    eps_prediction = jnp.asarray(
        [[[-0.7, -0.5], [-0.3, -0.1], [0.1, 0.3], [0.5, 0.7]]],
        dtype=jnp.float32,
    )

    runtime = SimpleNamespace(
        jax=jax,
        jnp=jnp,
        action_sequence=shape[1],
        action_dim=shape[2],
        sampler="ddpm",
        inference_timesteps=jnp.asarray([timestep], dtype=jnp.int32),
        betas=jnp.asarray(betas),
        alphas_cumprod=jnp.asarray(alphas_cumprod),
        _sample_clip_bounds=(-1.0, 1.0),
        _condition_as_local=False,
        _features_from_inputs=lambda _params, obs_inputs: obs_inputs,
        _apply_actor=lambda _params, _sample, _time, _features, **_kwargs: (
            eps_prediction
        ),
    )
    sample_fn = Diffusion._build_sample_fn(runtime)
    rng_key = jax.random.PRNGKey(17)
    obs_features = jnp.zeros((shape[0], 3), dtype=jnp.float32)

    actual = np.asarray(sample_fn(None, rng_key, obs_features))

    init_key, loop_key = jax.random.split(rng_key)
    _, posterior_noise_key = jax.random.split(loop_key)
    x_t = np.asarray(jax.random.normal(init_key, shape=shape))
    posterior_noise = np.asarray(jax.random.normal(posterior_noise_key, shape=shape))
    eps = np.asarray(eps_prediction)
    beta_t = betas[timestep]
    alpha_t = np.float32(1.0) - beta_t
    bar_alpha_t = alphas_cumprod[timestep]
    bar_alpha_prev = alphas_cumprod[timestep - 1]
    sqrt_bar_alpha = np.sqrt(bar_alpha_t)
    sqrt_one_minus_bar_alpha = np.sqrt(np.float32(1.0) - bar_alpha_t)
    eps = np.clip(
        eps,
        (x_t - sqrt_bar_alpha) / sqrt_one_minus_bar_alpha,
        (x_t + sqrt_bar_alpha) / sqrt_one_minus_bar_alpha,
    )
    posterior_mean = (x_t - beta_t / sqrt_one_minus_bar_alpha * eps) / np.sqrt(alpha_t)
    posterior_std = np.sqrt(
        beta_t * (np.float32(1.0) - bar_alpha_prev) / (np.float32(1.0) - bar_alpha_t)
    )
    expected = posterior_mean + posterior_std * posterior_noise

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_clean_diffuser_continuous_rf_uniform_euler_trajectory():
    num_steps = 4
    initial_noise = np.asarray(
        [[[0.4, -0.8], [0.2, 0.6], [-0.3, 0.9]]], dtype=np.float32
    )
    obs_features = np.asarray([[0.6]], dtype=np.float32)
    schedule = np.linspace(0.0, 1.0, num_steps + 1, dtype=np.float32)

    def velocity(_params, current_sample, timestep, features, **_kwargs):
        return (
            np.float32(0.2) * current_sample
            + np.float32(0.1) * timestep[:, None, None]
            + np.float32(0.05) * features[:, None, :]
        )

    runtime = SimpleNamespace(
        jnp=jnp,
        num_flow_steps=num_steps,
        _sample_clip_bounds=(-1.0, 1.0),
        _condition_as_local=False,
        _apply_actor=velocity,
    )
    actual = np.asarray(
        FlowMatching._integrate_sample(
            runtime,
            None,
            jnp.asarray(initial_noise),
            jnp.asarray(obs_features),
            jnp.asarray(schedule),
            1.0,
        )
    )

    expected = initial_noise.copy()
    for step in range(num_steps, 0, -1):
        time_value = schedule[step]
        delta_time = time_value - schedule[step - 1]
        expected_velocity = (
            np.float32(0.2) * expected
            + np.float32(0.1) * time_value
            + np.float32(0.05) * obs_features[:, None, :]
        )
        expected = expected + delta_time * expected_velocity
    expected = np.clip(expected, -1.0, 1.0)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
