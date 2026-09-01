from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

import robobase.method.cqn_as_bigym_latent_successor as successor_module
from robobase.method.cqn_as_bigym_latent_successor import (
    CQNASBigymGroundTruthLatentSuccessor,
    _capture_agent_rollout_state,
    _direct_plus_all_dimension_bins,
    _direct_plus_other_bins,
    _restore_agent_rollout_state,
    _safe_candidate_choice,
)


def test_policy_to_terminal_branches_use_frozen_cqn_continuation(monkeypatch):
    class FakeEnv:
        def __init__(self):
            self.steps = 0
            self.first_action = 0.0

        def step(self, action):
            self.steps += 1
            if self.steps == 1:
                self.first_action = float(np.asarray(action)[0, 0])
            terminal = self.steps == 3
            reward = float(terminal and self.first_action > 0.5)
            return (
                {"state": np.asarray([self.steps], dtype=np.float32)},
                reward,
                terminal,
                False,
                {},
            )

    class FakeAgent:
        def __init__(self):
            self.calls = 0

        def act(self, observations, step, eval_mode):
            assert eval_mode
            assert observations["state"].shape[0] == 1
            self.calls += 1
            return np.zeros((1, 2, 1), dtype=np.float32)

    env = FakeEnv()
    agent = FakeAgent()
    wrapper = object.__new__(CQNASBigymGroundTruthLatentSuccessor)
    wrapper.base = agent
    wrapper._rollout_env = env
    wrapper.discount = 0.9
    wrapper.horizon = 1
    wrapper.policy_to_terminal = True
    wrapper.max_branch_steps = 5
    wrapper.terminal_first_success = False
    wrapper._branch_steps = 0

    monkeypatch.setattr(
        successor_module,
        "restore_bigym_branch_state",
        lambda rollout_env, _state: setattr(rollout_env, "steps", 0),
    )
    monkeypatch.setattr(
        successor_module,
        "_restore_agent_rollout_state",
        lambda _agent, _state: None,
    )
    monkeypatch.setattr(
        successor_module,
        "_register_candidate",
        lambda _agent, candidate: np.asarray(candidate, dtype=np.float32),
    )

    candidates = np.zeros((2, 2, 1), dtype=np.float32)
    candidates[1, 0, 0] = 1.0
    observations, returns, discounts, done = wrapper._branch_candidates(
        candidates,
        env_state=None,
        agent_state={},
        policy_step=17,
    )

    assert len(observations) == 2
    np.testing.assert_allclose(returns, [0.0, 0.9**2])
    np.testing.assert_allclose(discounts, [0.9**3, 0.9**3])
    np.testing.assert_array_equal(done, [1.0, 1.0])
    assert agent.calls == 4
    assert wrapper._branch_steps == 6


def test_policy_to_terminal_can_stop_at_first_realized_success(monkeypatch):
    class FakeEnv:
        def __init__(self):
            self.action = 0.0

        def step(self, action):
            self.action = float(np.asarray(action)[0, 0])
            return (
                {"state": np.asarray([self.action], dtype=np.float32)},
                float(self.action > 0.5),
                True,
                False,
                {},
            )

    wrapper = object.__new__(CQNASBigymGroundTruthLatentSuccessor)
    wrapper.base = SimpleNamespace()
    wrapper._rollout_env = FakeEnv()
    wrapper.discount = 0.9
    wrapper.horizon = 1
    wrapper.policy_to_terminal = True
    wrapper.max_branch_steps = 5
    wrapper.terminal_first_success = True
    wrapper._branch_steps = 0
    monkeypatch.setattr(successor_module, "restore_bigym_branch_state", lambda *_: None)
    monkeypatch.setattr(successor_module, "_restore_agent_rollout_state", lambda *_: None)
    monkeypatch.setattr(
        successor_module,
        "_register_candidate",
        lambda _agent, candidate: np.asarray(candidate, dtype=np.float32),
    )
    candidates = np.asarray([[[0.0]], [[1.0]], [[-1.0]]], dtype=np.float32)

    observations, returns, _, done = wrapper._branch_candidates(
        candidates,
        env_state=None,
        agent_state={},
    )

    assert len(observations) == 2
    np.testing.assert_allclose(returns, [0.0, 1.0])
    np.testing.assert_array_equal(done, [1.0, 1.0])
    assert wrapper._branch_steps == 2


def test_direct_plus_all_dimension_bins_drops_one_nearest_plan_per_dimension():
    direct = np.zeros((2, 2), dtype=np.float32)
    direct[0] = [0.1, -0.1]
    siblings = np.zeros((6, 2, 2), dtype=np.float32)
    dimensions = np.repeat(np.arange(2), 3)
    siblings[:3, 0, 0] = [-0.8, 0.0, 0.8]
    siblings[3:, 0, 1] = [-0.8, 0.0, 0.8]

    candidates, dropped = _direct_plus_all_dimension_bins(
        direct,
        siblings,
        dimensions,
    )

    assert candidates.shape == (5, 2, 2)
    np.testing.assert_array_equal(candidates[0], direct)
    np.testing.assert_array_equal(dropped, [1, 4])
    np.testing.assert_allclose(candidates[1:3, 0, 0], [-0.8, 0.8])
    np.testing.assert_allclose(candidates[3:, 0, 1], [-0.8, 0.8])


def test_policy_to_terminal_scores_realized_return_without_calling_critic():
    wrapper = object.__new__(CQNASBigymGroundTruthLatentSuccessor)
    wrapper.policy_to_terminal = True
    wrapper._score_ground_truth_latents = lambda *_args: (_ for _ in ()).throw(
        AssertionError("terminal-return oracle must not query Q/V")
    )
    observations = [
        {"state": np.asarray([0.0], dtype=np.float32)},
        {"state": np.asarray([1.0], dtype=np.float32)},
    ]

    scores, features = wrapper._score_candidates(
        observations,
        np.asarray([0.0, 0.81], dtype=np.float32),
        np.asarray([0.7, 0.7], dtype=np.float32),
        np.asarray([1.0, 1.0], dtype=np.float32),
    )

    np.testing.assert_allclose(scores, [0.0, 0.81])
    np.testing.assert_array_equal(features, np.zeros((2, 1), dtype=np.float32))


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
