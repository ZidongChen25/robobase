"""Focused checks for single-critic twin_rollout_beam_width rollout search.

The beam must (1) survive the config chain on the default single-critic
method (yaml -> spec -> factory table -> __init__), (2) leave the width-1
default untouched, (3) route eval-time action selection through the joint
beam with the greedy contract (same chunk shape, complete-chunk rerank
score at least the greedy chunk's), and (4) keep the twin-path and
bypassed-action-path validation gates.
"""

from pathlib import Path

import jax
import numpy as np
import pytest
from gymnasium import spaces
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from robobase.factory import create_agent


def _params_equal(left, right):
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    assert left_tree == right_tree
    return all(
        np.allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-7)
        for a, b in zip(left_leaves, right_leaves, strict=True)
    )

CONFIG_DIR = str((Path(__file__).parents[2] / "robobase" / "cfgs").resolve())

ACTION_SEQUENCE = 3
ACTION_DIM = 2


def _compose(*overrides):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(
            config_name="robobase_config",
            overrides=[
                "method=cqn_as",
                f"action_sequence={ACTION_SEQUENCE}",
                "num_train_envs=1",
                "num_eval_envs=1",
                "backend.jit=false",
                "backend.platform=cpu",
                "method.model.hidden_dims=[32,32]",
                "method.atoms=11",
                *overrides,
            ],
        )


def _twin_cfg(*overrides):
    return _compose(
        "num_pretrain_steps=0",
        "is_imitation_learning=false",
        "use_self_imitation=false",
        "method.strict_demo_rl_only=true",
        "method.bc_lambda=0",
        "method.bc_lambda_schedule=null",
        "method.bc_margin=0",
        "method.demo_fosd=false",
        "method.separate_bc_policy=false",
        "method.use_dueling=false",
        "method.mc_return_value_only=false",
        "method.mc_lower_bound_target=true",
        "method.td_target_action_source=critic_replay_max",
        "replay.include_next_action=true",
        "method.weight_decay=0.0",
        "method.unseen_return_floor_weight=0.0",
        "method.pessimistic_twin_critic=true",
        "method.twin_rollout_beam_width=2",
        *overrides,
    )


def _spaces():
    observation_space = spaces.Dict(
        {
            "low_dim_state": spaces.Box(
                -np.inf,
                np.inf,
                shape=(1, 5),
                dtype=np.float32,
            )
        }
    )
    action_space = spaces.Box(
        -1.0,
        1.0,
        shape=(ACTION_SEQUENCE, ACTION_DIM),
        dtype=np.float32,
    )
    return observation_space, action_space


def _batch(batch_size=8, seed=11):
    rng = np.random.default_rng(seed)
    return {
        "low_dim_state": rng.normal(size=(batch_size, 1, 5)).astype(
            np.float32
        ),
        "low_dim_state_tp1": rng.normal(size=(batch_size, 1, 5)).astype(
            np.float32
        ),
        "action": rng.uniform(
            -1.0,
            1.0,
            size=(batch_size, ACTION_SEQUENCE, ACTION_DIM),
        ).astype(np.float32),
        "reward": rng.normal(size=(batch_size,)).astype(np.float32),
        "discount": np.full((batch_size,), 0.99, dtype=np.float32),
        "terminal": np.zeros((batch_size,), dtype=bool),
        "truncated": np.zeros((batch_size,), dtype=bool),
        "demo": np.ones((batch_size,), dtype=np.uint8),
    }


def _make_agent(cfg):
    observation_space, action_space = _spaces()
    return create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )


def test_beam_single_critic_attribute_round_trip():
    agent = _make_agent(_compose("method.twin_rollout_beam_width=8"))
    assert agent.twin_rollout_beam_width == 8
    assert agent.pessimistic_twin_critic is False
    assert agent.episodic_twin_head_exploration is False


def test_beam_default_width_one_construction_unchanged():
    agent = _make_agent(_compose())
    assert agent.twin_rollout_beam_width == 1
    assert agent.pessimistic_twin_critic is False


def test_beam_eval_action_matches_greedy_contract_and_reranks():
    """After identical updates the beam agent's eval action keeps the
    greedy chunk contract (shape, bounds), while the beam's complete-chunk
    rerank score is at least the greedy chunk's (the joint top-1 greedy
    path is always a beam member) and the selected chunks differ."""
    greedy = _make_agent(_compose("num_eval_envs=8"))
    beam = _make_agent(
        _compose(
            "num_eval_envs=8",
            "method.twin_rollout_beam_width=8",
        )
    )
    for step in range(1, 4):
        batch = _batch(seed=11 + step)
        greedy.update(
            iter([{k: np.array(v, copy=True) for k, v in batch.items()}]),
            step=step,
        )
        beam.update(
            iter([{k: np.array(v, copy=True) for k, v in batch.items()}]),
            step=step,
        )
    assert _params_equal(greedy.params, beam.params)

    obs_rng = np.random.default_rng(1011)
    obs = {
        "low_dim_state": obs_rng.normal(size=(8, 1, 5)).astype(np.float32)
    }
    greedy_action = np.asarray(greedy.act(dict(obs), step=1, eval_mode=True))
    beam_action = np.asarray(beam.act(dict(obs), step=1, eval_mode=True))
    assert beam_action.shape == greedy_action.shape
    for action in (greedy_action, beam_action):
        assert np.all(np.isfinite(action))
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)
    assert np.any(np.abs(beam_action - greedy_action) > 1e-6)

    obs_inputs = beam._prepare_rl_obs_inputs(obs)
    features = beam._rl_features(
        beam.params.get("encoder", None),
        obs_inputs,
        stop_gradient=True,
    )
    critic_params = beam.target_critic_params
    beam_chunk, _ = beam._joint_beam_action(critic_params, features)
    greedy_chunk, _ = beam._greedy_action(critic_params, features)
    assert beam_chunk.shape == greedy_chunk.shape
    beam_score = np.asarray(
        beam._score_action_sequence_for_backup(
            critic_params, features, beam_chunk
        )
    )
    greedy_score = np.asarray(
        beam._score_action_sequence_for_backup(
            critic_params, features, greedy_chunk
        )
    )
    assert np.all(beam_score >= greedy_score - 1e-5)
    assert np.any(
        np.abs(np.asarray(beam_chunk) - np.asarray(greedy_chunk)) > 1e-6
    )


def test_beam_twin_path_validation_unchanged():
    agent = _make_agent(
        _twin_cfg("method.episodic_twin_head_exploration=true")
    )
    assert agent.twin_rollout_beam_width == 2
    assert agent.pessimistic_twin_critic is True
    with pytest.raises(
        ValueError, match="episodic_twin_head_exploration=true"
    ):
        _make_agent(_twin_cfg())


def test_beam_rejects_bypassed_action_paths():
    with pytest.raises(ValueError, match="separate_bc_policy=false"):
        _make_agent(
            _compose(
                "method.twin_rollout_beam_width=8",
                "method.separate_bc_policy=true",
            )
        )
    with pytest.raises(ValueError, match="coarse_flow=false"):
        _make_agent(
            _compose(
                "method.twin_rollout_beam_width=8",
                "method.coarse_flow=true",
            )
        )
