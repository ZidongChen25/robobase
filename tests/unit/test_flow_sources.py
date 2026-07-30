import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = jax.numpy

from robobase.method.flow_sources import (  # noqa: E402
    A2AFlowSource,
    GaussianFlowSource,
    LegatoFlowSource,
    a2a_flow_training_pair,
    gaussian_flow_training_pair,
    guided_euler_step,
    legato_inference_source,
    legato_schedule,
    legato_training_pair,
    linear_flow_training_pair,
)


def test_linear_and_a2a_pairs_use_reverse_time_path():
    source = jnp.asarray([[[-2.0], [0.0]]], dtype=jnp.float32)
    target = jnp.asarray([[[2.0], [4.0]]], dtype=jnp.float32)
    tau = jnp.asarray([0.25], dtype=jnp.float32)

    expected = linear_flow_training_pair(source, target, tau)
    actual = a2a_flow_training_pair(source, target, tau)

    np.testing.assert_allclose(actual.sample, 0.25 * source + 0.75 * target)
    np.testing.assert_allclose(actual.target_velocity, target - source)
    np.testing.assert_allclose(actual.sample, expected.sample)
    assert actual.schedule_channel is None


def test_gaussian_pair_uses_keyed_jax_noise():
    key = jax.random.PRNGKey(7)
    target = jnp.ones((2, 3, 1), dtype=jnp.float32)
    tau = jnp.asarray([0.0, 1.0], dtype=jnp.float32)
    expected_noise = jax.random.normal(key, target.shape, dtype=target.dtype)

    pair = gaussian_flow_training_pair(key, target, tau)

    np.testing.assert_allclose(pair.sample[0], target[0])
    np.testing.assert_allclose(pair.sample[1], expected_noise[1])
    np.testing.assert_allclose(pair.target_velocity, target - expected_noise)


def test_legato_schedule_supports_batched_delay_and_ramp():
    delay = jnp.asarray([2, 1], dtype=jnp.int32)
    ramp = jnp.asarray([2, 3], dtype=jnp.int32)

    hard = legato_schedule(5, delay, ramp, kind="hard")
    linear = legato_schedule(5, delay, ramp, kind="linear")
    cosine = legato_schedule(5, delay, ramp, kind="cosine")

    assert hard.shape == linear.shape == cosine.shape == (2, 5, 1)
    np.testing.assert_allclose(hard[..., 0], [[1, 1, 0, 0, 0], [1, 0, 0, 0, 0]])
    np.testing.assert_allclose(linear[0, :, 0], [1, 1, 1, 0.5, 0])
    np.testing.assert_allclose(linear[1, :, 0], [1, 1, 2 / 3, 1 / 3, 0])
    np.testing.assert_allclose(cosine[0, :, 0], [1, 1, 1, 0.5, 0])
    assert np.all(np.asarray(cosine) >= 0.0)
    assert np.all(np.asarray(cosine) <= 1.0)


def test_legato_schedule_supports_execution_offset_and_rejects_clipping():
    schedule = legato_schedule(5, delay=1, ramp=1, start=2, kind="hard")

    np.testing.assert_array_equal(
        schedule[:, 0],
        np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="start \\+ delay \\+ ramp"):
        legato_schedule(4, delay=3, ramp=2, kind="linear")

    invalid_dynamic = jax.jit(
        lambda start, delay, ramp: legato_schedule(
            5,
            delay,
            ramp,
            start=start,
            kind="linear",
        )
    )(jnp.asarray(2), jnp.asarray(3), jnp.asarray(1))
    assert np.isnan(np.asarray(invalid_dynamic)).all()


def test_legato_omega_zero_is_exact_standard_flow_matching():
    noise = jnp.asarray([[[-1.0], [3.0]]], dtype=jnp.float32)
    target = jnp.asarray([[[2.0], [5.0]]], dtype=jnp.float32)
    tau = jnp.asarray([0.4], dtype=jnp.float32)

    standard = linear_flow_training_pair(noise, target, tau)
    legato = legato_training_pair(target, noise, tau, omega=0.0, dt=0.2)

    np.testing.assert_allclose(legato.sample, standard.sample)
    np.testing.assert_allclose(legato.target_velocity, standard.target_velocity)
    np.testing.assert_allclose(legato.schedule_channel, 1.0)


def test_legato_omega_one_collapses_mixture_to_target():
    noise = jnp.zeros((2, 4, 3), dtype=jnp.float32)
    target = jnp.arange(24, dtype=jnp.float32).reshape(2, 4, 3)
    tau = jnp.asarray([0.2, 0.9], dtype=jnp.float32)

    pair = legato_training_pair(target, noise, tau, omega=1.0, dt=0.25)

    np.testing.assert_allclose(pair.sample, target)
    np.testing.assert_allclose(pair.schedule_channel, 0.0)


def test_legato_target_velocity_matches_paper_v2_closed_form():
    noise = jnp.asarray([[[-2.0], [1.0], [3.0]]], dtype=jnp.float32)
    target = jnp.asarray([[[2.0], [5.0], [7.0]]], dtype=jnp.float32)
    omega = jnp.asarray([[[0.0], [0.25], [0.75]]], dtype=jnp.float32)
    tau = jnp.asarray([0.4], dtype=jnp.float32)
    dt = 0.2

    pair = legato_training_pair(target, noise, tau, omega, dt)
    effective_source = omega * target + (1.0 - omega) * noise
    expected_sample = 0.4 * effective_source + 0.6 * target
    expected_velocity = (1.0 - (omega / dt) * 0.4) * (target - noise)

    np.testing.assert_allclose(pair.sample, expected_sample)
    np.testing.assert_allclose(pair.target_velocity, expected_velocity)
    np.testing.assert_allclose(pair.schedule_channel, 1.0 - omega)


def test_guided_source_and_euler_step_preserve_full_guidance():
    sample = jnp.full((1, 3, 2), -5.0, dtype=jnp.float32)
    reference = jnp.arange(6, dtype=jnp.float32).reshape(1, 3, 2)
    velocity = jnp.zeros_like(sample)
    omega = jnp.asarray([1.0, 0.5, 0.0], dtype=jnp.float32)

    source = legato_inference_source(sample, reference, omega)
    stepped = guided_euler_step(sample, reference, velocity, omega, dt=0.1)

    np.testing.assert_allclose(source.sample[:, 0], reference[:, 0])
    np.testing.assert_allclose(source.sample[:, 2], sample[:, 2])
    np.testing.assert_allclose(stepped, source.sample)
    np.testing.assert_allclose(source.schedule_channel[0, :, 0], [0.0, 0.5, 1.0])


def test_flow_sources_are_jittable_and_differentiable():
    noise = jnp.zeros((2, 4, 3), dtype=jnp.float32)
    target = jnp.ones_like(noise)
    tau = jnp.asarray([0.2, 0.8], dtype=jnp.float32)
    omega = jax.jit(lambda d, r: legato_schedule(4, d, r, kind="cosine"))(
        jnp.asarray([1, 2]), jnp.asarray([3, 2])
    )
    pair_fn = jax.jit(legato_training_pair)

    pair = pair_fn(target, noise, tau, omega, 0.25)

    def loss_fn(value):
        result = legato_training_pair(value, noise, tau, omega, 0.25)
        return jnp.sum(result.sample) + jnp.sum(result.target_velocity)

    gradient = jax.jit(jax.grad(loss_fn))(target)
    assert pair.sample.shape == target.shape
    assert pair.schedule_channel.shape == (2, 4, 1)
    assert gradient.shape == target.shape
    assert np.all(np.isfinite(np.asarray(gradient)))


def test_plug_in_source_classes_share_training_and_inference_contract():
    key = jax.random.PRNGKey(3)
    source = jnp.zeros((2, 4, 3), dtype=jnp.float32)
    target = jnp.ones_like(source)
    tau = jnp.asarray([0.25, 0.75], dtype=jnp.float32)

    gaussian = GaussianFlowSource()
    a2a = A2AFlowSource()
    legato = LegatoFlowSource(target_mode="public_kinetix_plus")
    gaussian_pair = jax.jit(gaussian.build_training_pair)(key, target, tau)
    a2a_pair = jax.jit(a2a.build_training_pair)(key, target, tau, source=source)
    legato_pair = jax.jit(legato.build_training_pair)(
        key, target, tau, omega=0.25, dt=0.25
    )

    assert gaussian_pair.sample.shape == target.shape
    np.testing.assert_allclose(a2a_pair.target_velocity, 1.0)
    assert legato_pair.schedule_channel.shape == (2, 4, 1)
    assert np.all(np.isfinite(np.asarray(legato_pair.target_velocity)))
    np.testing.assert_allclose(
        a2a.build_inference_source(key, source=source).sample, source
    )
