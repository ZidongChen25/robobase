"""Bin-explore state lifecycle: replan-mask gating, episode reset, resume.

Regression tests for cqn-flow.md 48.2 (review items #2-#4): rows outside the
temporal-ensemble replan mask must not draw, shift, or burn persist windows;
episode reset must clear persisted sibling shifts; snapshots must carry the
NumPy exploration RNG streams and persisted windows so a resumed run draws
the same assignment sequence as an uninterrupted one.
"""

from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from robobase.factory import create_agent
from tests.unit.test_cqn_as import _spaces

CONFIG_DIR = str((Path(__file__).parents[2] / "robobase" / "cfgs").resolve())


def _explore_agent(num_train_envs=2, probs="[1.0,1.0,1.0]"):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "method=cqn_as",
                "action_sequence=3",
                f"num_train_envs={num_train_envs}",
                "num_eval_envs=1",
                "backend.jit=false",
                "backend.platform=cpu",
                "method.model.hidden_dims=[32,32]",
                "method.atoms=11",
                f"method.bin_explore_probs={probs}",
                "method.bin_explore_persist_plans=2",
            ],
        )
    observation_space, action_space = _spaces()
    return create_agent(
        cfg, observation_space=observation_space, action_space=action_space
    )


def _chunk(batch=2):
    return np.zeros((batch, 3, 2), dtype=np.float32)


def test_register_mask_gates_draws_shifts_and_persist_counters():
    agent = _explore_agent()

    shifted = agent._apply_bin_explore(
        _chunk(), np.asarray([True, False])
    )
    np.testing.assert_array_equal(agent._bin_explore_remaining, [1, 0])
    assert agent._bin_explore_dimension[1] == -1, "masked row must not draw"
    assert not np.array_equal(shifted[0], _chunk()[0])
    np.testing.assert_array_equal(shifted[1], _chunk()[1])

    before = agent._bin_explore_remaining.copy()
    shifted = agent._apply_bin_explore(
        _chunk(), np.asarray([False, False])
    )
    np.testing.assert_array_equal(agent._bin_explore_remaining, before)
    np.testing.assert_array_equal(shifted, _chunk())

    agent._apply_bin_explore(_chunk(), np.asarray([True, True]))
    np.testing.assert_array_equal(agent._bin_explore_remaining, [0, 1])


def test_reset_clears_persisted_shift_only_for_reset_envs():
    agent = _explore_agent()
    agent._apply_bin_explore(_chunk(), np.asarray([True, True]))
    assert (agent._bin_explore_remaining > 0).all()

    agent.reset(step=0, agents_to_reset=[0])

    assert agent._bin_explore_remaining[0] == 0
    assert agent._bin_explore_dimension[0] == -1
    assert agent._bin_explore_level[0] == -1
    assert agent._bin_explore_sibling[0] == -1
    assert agent._bin_explore_remaining[1] > 0
    assert agent._bin_explore_dimension[1] != -1


def test_checkpoint_roundtrip_resumes_exploration_rng_stream():
    agent = _explore_agent(num_train_envs=1, probs="[0.5,0.3,0.2]")
    # Advance the exploration RNG, then snapshot at a windowless point
    # (remaining == 0), where an uninterrupted process and a resumed one
    # are in identical exploration states.
    for _ in range(50):
        agent._apply_bin_explore(_chunk(1), np.asarray([True]))
        if agent._bin_explore_remaining[0] == 0:
            break
    assert agent._bin_explore_remaining[0] == 0
    state = agent.checkpoint_state_dict()

    # The uninterrupted process continues drawing...
    continued = [
        (
            agent._apply_bin_explore(_chunk(1), np.asarray([True])),
            agent._bin_explore_dimension.copy(),
            agent._bin_explore_level.copy(),
            agent._bin_explore_sibling.copy(),
        )
        for _ in range(6)
    ]

    # ...and a resumed process must reproduce the same sequence.
    resumed = _explore_agent(num_train_envs=1, probs="[0.5,0.3,0.2]")
    resumed.load_checkpoint_state_dict(state)
    for expected_chunk, dim, level, sibling in continued:
        chunk = resumed._apply_bin_explore(_chunk(1), np.asarray([True]))
        np.testing.assert_array_equal(chunk, expected_chunk)
        np.testing.assert_array_equal(resumed._bin_explore_dimension, dim)
        np.testing.assert_array_equal(resumed._bin_explore_level, level)
        np.testing.assert_array_equal(resumed._bin_explore_sibling, sibling)


def test_resume_never_restores_mid_episode_window():
    # Workspace snapshots carry no env state and resume starts fresh
    # episodes without agent.reset(); a mid-episode persist window must
    # therefore never survive a checkpoint roundtrip.
    agent = _explore_agent(num_train_envs=2)
    agent._apply_bin_explore(_chunk(), np.asarray([True, True]))
    assert (agent._bin_explore_remaining > 0).all()
    state = agent.checkpoint_state_dict()
    assert not any(key.startswith("bin_explore_remaining") for key in state)

    resumed = _explore_agent(num_train_envs=2)
    resumed.load_checkpoint_state_dict(state)
    np.testing.assert_array_equal(resumed._bin_explore_remaining, [0, 0])
    np.testing.assert_array_equal(resumed._bin_explore_dimension, [-1, -1])
    # The RNG stream itself must still continue from the snapshot.
    assert (
        resumed._bin_explore_rng.bit_generator.state["state"]
        == agent._bin_explore_rng.bit_generator.state["state"]
    )

    # A snapshot from the short-lived format that did store windows must
    # load without applying them.
    legacy = dict(state)
    legacy["bin_explore_remaining"] = np.asarray([2, 2], dtype=np.int32)
    legacy["bin_explore_dimension"] = np.asarray([1, 1], dtype=np.int16)
    fresh = _explore_agent(num_train_envs=2)
    fresh.load_checkpoint_state_dict(legacy)
    np.testing.assert_array_equal(fresh._bin_explore_remaining, [0, 0])
    np.testing.assert_array_equal(fresh._bin_explore_dimension, [-1, -1])


def test_old_snapshots_without_exploration_keys_still_load():
    agent = _explore_agent(num_train_envs=1)
    state = agent.checkpoint_state_dict()
    for key in list(state):
        if key.startswith("bin_"):
            state.pop(key)
    agent.load_checkpoint_state_dict(state)  # must not raise
    assert agent._bin_explore_remaining.shape == (1,)
