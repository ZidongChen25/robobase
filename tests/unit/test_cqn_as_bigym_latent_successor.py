from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from robobase.method.cqn_as_bigym_latent_successor import (
    _capture_agent_rollout_state,
    _direct_plus_other_bins,
    _restore_agent_rollout_state,
    _safe_candidate_choice,
)


def test_direct_plus_other_bins_keeps_true_direct_and_drops_nearest_bin():
    direct = np.zeros((16, 3), dtype=np.float32)
    direct[0, 1] = 0.12
    siblings = np.zeros((5, 16, 3), dtype=np.float32)
    siblings[:, 0, 1] = np.asarray([-0.8, -0.4, 0.0, 0.1, 0.8])

    candidates, dropped = _direct_plus_other_bins(
        direct,
        siblings,
        action_dimension=1,
    )

    assert dropped == 3
    assert candidates.shape == (5, 16, 3)
    np.testing.assert_array_equal(candidates[0], direct)
    np.testing.assert_allclose(candidates[1:, 0, 1], [-0.8, -0.4, 0.0, 0.8])


def test_agent_rollout_state_round_trip_copies_arrays_and_rng_key():
    agent = SimpleNamespace(
        rng_key=jnp.asarray([3, 7], dtype=jnp.uint32),
        _eval_action_history=np.ones((1, 2, 2, 1), dtype=np.float32),
        _eval_action_history_valid=np.asarray([[True, False]]),
        _eval_open_loop_plan=None,
        _eval_open_loop_position=None,
        _eval_open_loop_valid=None,
    )
    state = _capture_agent_rollout_state(agent)
    agent.rng_key = jnp.asarray([0, 0], dtype=jnp.uint32)
    agent._eval_action_history[...] = 9.0
    agent._eval_action_history_valid[...] = False

    _restore_agent_rollout_state(agent, state)

    np.testing.assert_array_equal(np.asarray(agent.rng_key), [3, 7])
    np.testing.assert_array_equal(agent._eval_action_history, 1.0)
    np.testing.assert_array_equal(agent._eval_action_history_valid, [[True, False]])


def test_safe_candidate_choice_rejects_nonreward_terminal_even_if_q_is_highest():
    selected = _safe_candidate_choice(
        np.asarray([0.2, 0.9, 0.4], dtype=np.float32),
        np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        switch_margin=1e-5,
    )

    assert selected["unmasked_choice"] == 1
    assert selected["unmasked_invalid_win"]
    assert selected["raw_choice"] == 2
    assert selected["choice"] == 2
    assert not selected["all_invalid"]


def test_safe_candidate_choice_falls_back_to_direct_when_all_are_invalid():
    selected = _safe_candidate_choice(
        np.asarray([0.0, 0.3], dtype=np.float32),
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([1.0, 1.0], dtype=np.float32),
        switch_margin=1e-5,
    )

    assert selected["all_invalid"]
    assert selected["choice"] == 0
    assert not selected["switch"]
