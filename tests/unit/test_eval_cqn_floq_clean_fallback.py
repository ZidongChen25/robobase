import jax.numpy as jnp
import numpy as np
import pytest

from scripts.eval_cqn_floq_clean_fallback import (
    select_flow_fallback_plan,
)


def _fixture():
    baseline = jnp.zeros((1, 2, 2), dtype=jnp.float32)
    candidates = jnp.broadcast_to(
        baseline[:, None, None],
        (1, 2, 3, 2, 2),
    )
    candidates = candidates.at[0, 0, :, :, 0].set(
        jnp.asarray([-1.0, 0.0, 1.0])[:, None]
    )
    candidates = candidates.at[0, 1, :, :, 1].set(
        jnp.asarray([-1.0, 0.0, 1.0])[:, None]
    )
    baseline_indices = jnp.asarray([[1, 1]], dtype=jnp.int32)
    distilled_q = jnp.asarray(
        [[[-1.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        dtype=jnp.float32,
    )
    policy_logits = jnp.asarray(
        [[[-2.0, 0.0, -0.1], [-2.0, 0.0, -0.1]]],
        dtype=jnp.float32,
    )
    flow_samples = jnp.asarray(
        [
            [
                [[-1.0, 0.0, 1.5], [-1.0, 0.0, 0.5]],
                [[-1.0, 0.0, 1.0], [-1.0, 0.0, -0.1]],
                [[-1.0, 0.0, 1.2], [-1.0, 0.0, 0.2]],
                [[-1.0, 0.0, 0.8], [-1.0, 0.0, -0.2]],
            ]
        ],
        dtype=jnp.float32,
    )
    return (
        baseline,
        candidates,
        baseline_indices,
        distilled_q,
        policy_logits,
        flow_samples,
    )


def _select(**overrides):
    values = _fixture()
    kwargs = {
        "policy_value_beta": 0.1,
        "min_value_margin": 0.5,
        "max_bc_logprob_drop": 1.0,
        "max_best_bc_logprob_drop": 1.0,
        "min_source_win_fraction": 0.75,
        "min_source_mean_delta": 0.0,
    }
    kwargs.update(overrides)
    return select_flow_fallback_plan(*values, **kwargs)


def test_selects_one_dimension_with_supported_consistent_flow_gain():
    result = _select()

    assert bool(result.applied_override[0])
    assert int(result.selected_dimension[0]) == 0
    np.testing.assert_allclose(result.action[0, :, 0], 1.0)
    np.testing.assert_allclose(result.action[0, :, 1], 0.0)
    assert float(result.selected_source_win_fraction[0]) == 1.0


def test_falls_back_exactly_when_sources_disagree():
    result = _select(min_source_win_fraction=1.0)
    # Dimension 0 remains eligible because all four sources agree.
    assert bool(result.applied_override[0])

    values = list(_fixture())
    values[-1] = values[-1].at[0, :2, 0, 2].set(-1.0)
    result = select_flow_fallback_plan(
        *values,
        policy_value_beta=0.1,
        min_value_margin=0.5,
        max_bc_logprob_drop=1.0,
        max_best_bc_logprob_drop=1.0,
        min_source_win_fraction=0.75,
        min_source_mean_delta=0.0,
    )

    assert not bool(result.applied_override[0])
    np.testing.assert_array_equal(result.action, values[0])


def test_rejects_candidate_outside_behavior_support():
    result = _select(
        max_bc_logprob_drop=0.01,
        max_best_bc_logprob_drop=0.01,
    )

    assert not bool(result.applied_override[0])
    np.testing.assert_array_equal(result.action, _fixture()[0])


def test_validates_source_sample_shape():
    values = list(_fixture())
    values[-1] = jnp.zeros((1, 4, 2), dtype=jnp.float32)

    with pytest.raises(ValueError, match="flow_q_samples"):
        select_flow_fallback_plan(
            *values,
            policy_value_beta=1.0,
            min_value_margin=0.0,
            max_bc_logprob_drop=1.0,
            max_best_bc_logprob_drop=1.0,
            min_source_win_fraction=0.5,
            min_source_mean_delta=0.0,
        )
