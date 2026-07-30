import jax.numpy as jnp
import numpy as np

from scripts.eval_cqn_structured_delta_sidecar import (
    select_structured_delta_plan,
)


def _fixture():
    baseline = jnp.zeros((1, 2, 2), dtype=jnp.float32)
    deltas = jnp.asarray(
        [[[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]],
        dtype=jnp.float32,
    )
    candidates = jnp.broadcast_to(
        baseline[:, None, None],
        (1, 2, 3, 2, 2),
    )
    candidates = candidates.at[0, 0, :, :, 0].set(
        deltas[0, 0, :, None]
    )
    candidates = candidates.at[0, 1, :, :, 1].set(
        deltas[0, 1, :, None]
    )
    policy = jnp.asarray(
        [[[-1.0, 0.0, -0.2], [-1.0, 0.0, -0.1]]],
        dtype=jnp.float32,
    )
    return baseline, candidates, deltas, policy


def test_structured_delta_selects_one_best_supported_dimension():
    baseline, candidates, deltas, policy = _fixture()

    result = select_structured_delta_plan(
        baseline,
        candidates,
        deltas,
        policy,
        jnp.asarray([[0.9, 0.2]], dtype=jnp.float32),
        jnp.asarray([1.0], dtype=jnp.float32),
        min_value_margin=0.1,
        max_bc_logprob_drop=0.5,
        max_state_rms=2.0,
    )

    assert bool(result.applied_override[0])
    assert int(result.selected_dimension[0]) == 0
    np.testing.assert_allclose(result.action[0, :, 0], 1.0)
    np.testing.assert_allclose(result.action[0, :, 1], 0.0)


def test_structured_delta_exactly_falls_back_outside_state_support():
    baseline, candidates, deltas, policy = _fixture()

    result = select_structured_delta_plan(
        baseline,
        candidates,
        deltas,
        policy,
        jnp.asarray([[0.9, 0.9]], dtype=jnp.float32),
        jnp.asarray([3.0], dtype=jnp.float32),
        min_value_margin=0.1,
        max_bc_logprob_drop=0.5,
        max_state_rms=2.0,
    )

    assert not bool(result.applied_override[0])
    np.testing.assert_array_equal(result.action, baseline)


def test_structured_delta_rejects_value_best_candidate_outside_bc_support():
    baseline, candidates, deltas, policy = _fixture()

    result = select_structured_delta_plan(
        baseline,
        candidates,
        deltas,
        policy,
        jnp.asarray([[-0.9, 0.0]], dtype=jnp.float32),
        jnp.asarray([1.0], dtype=jnp.float32),
        min_value_margin=0.1,
        max_bc_logprob_drop=0.5,
        max_state_rms=2.0,
    )

    assert not bool(result.applied_override[0])
    np.testing.assert_array_equal(result.action, baseline)


def test_structured_delta_reliability_mask_forces_exact_bc_fallback():
    baseline, candidates, deltas, policy = _fixture()

    result = select_structured_delta_plan(
        baseline,
        candidates,
        deltas,
        policy,
        jnp.asarray([[0.9, 0.9]], dtype=jnp.float32),
        jnp.asarray([1.0], dtype=jnp.float32),
        reliability_mask=jnp.asarray([[False, False]]),
        min_value_margin=0.1,
        max_bc_logprob_drop=0.5,
        max_state_rms=2.0,
    )

    assert not bool(result.applied_override[0])
    assert not bool(result.eligible_override_mask.any())
    np.testing.assert_array_equal(result.action, baseline)


def test_structured_delta_reliability_mask_keeps_supported_dimension():
    baseline, candidates, deltas, policy = _fixture()

    result = select_structured_delta_plan(
        baseline,
        candidates,
        deltas,
        policy,
        jnp.asarray([[0.9, 0.9]], dtype=jnp.float32),
        jnp.asarray([1.0], dtype=jnp.float32),
        reliability_mask=jnp.asarray([[False, True]]),
        min_value_margin=0.1,
        max_bc_logprob_drop=0.5,
        max_state_rms=2.0,
    )

    assert bool(result.applied_override[0])
    assert int(result.selected_dimension[0]) == 1
