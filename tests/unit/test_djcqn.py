from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnasium import spaces
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from robobase.factory import create_agent, method_name_from_cfg
from robobase.method.djcqn import (
    PrefixC2FCritic,
    absolute_topk,
    c2f_prefix_beam_search,
    chosen_bin_upper_expectile_loss,
    rank_adjacent_sibling_disagreement,
    validate_djcqn_config,
)
from robobase.method.q_chunking import q_chunking_td_target
from robobase.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from robobase.workspace import _replay_action_from_step, _validate_rl_action_sequence


CONFIG_DIR = str((Path(__file__).parents[2] / "robobase" / "cfgs").resolve())


def _spaces(horizon=3, action_dim=2):
    observation_space = spaces.Dict(
        {
            "low_dim_state": spaces.Box(
                -np.inf, np.inf, shape=(1, 5), dtype=np.float32
            )
        }
    )
    action_space = spaces.Box(
        -1.0, 1.0, shape=(horizon, action_dim), dtype=np.float32
    )
    return observation_space, action_space


def _config(*, jit=False, horizon=3, overrides=()):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(
            config_name="robobase_config",
            overrides=[
                "method=djcqn",
                f"action_sequence={horizon}",
                f"replay.nstep={horizon}",
                "num_train_envs=2",
                "num_eval_envs=1",
                f"backend.jit={'true' if jit else 'false'}",
                "method.levels=2",
                "method.bins=3",
                "method.beam_width=2",
                "method.num_critics=3",
                "method.model.hidden_dims=[16,16]",
                *overrides,
            ],
        )


def _agent(*, jit=False, overrides=()):
    cfg = _config(jit=jit, overrides=overrides)
    observation_space, action_space = _spaces()
    return create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )


def _batch(batch_size=4, horizon=3):
    rng = np.random.default_rng(4)
    return {
        "low_dim_state": rng.normal(size=(batch_size, 1, 5)).astype(np.float32),
        "low_dim_state_tp1": rng.normal(size=(batch_size, 1, 5)).astype(np.float32),
        "action": rng.uniform(-1, 1, size=(batch_size, horizon, 2)).astype(
            np.float32
        ),
        "action_pad_mask": np.zeros((batch_size, horizon), dtype=bool),
        # Replay already supplies R_H and gamma^H here.
        "reward": rng.uniform(0, 1, size=(batch_size,)).astype(np.float32),
        "discount": np.full((batch_size,), 0.99**horizon, dtype=np.float32),
        "terminal": np.zeros((batch_size,), dtype=bool),
        "truncated": np.zeros((batch_size,), dtype=bool),
    }


def _tree_changed(before, after):
    left, left_def = jax.tree.flatten(before)
    right, right_def = jax.tree.flatten(after)
    assert left_def == right_def
    return any(
        not np.allclose(np.asarray(a), np.asarray(b))
        for a, b in zip(left, right, strict=True)
    )


def test_djcqn_config_is_new_and_legacy_q_chunking_is_unchanged():
    cfg = _config()
    assert method_name_from_cfg(cfg) == "djcqn"
    assert cfg.method._target_ == "robobase.method.djcqn.DJCQN"
    assert cfg.action_sequence == cfg.replay.nstep == 3
    assert cfg.execution_length == 1
    assert cfg.method.prefix_horizon == 1
    assert cfg.method.sibling_exploration_prob == 0.0
    assert "actor_lr" not in cfg.method
    assert "bc_lambda" not in cfg.method
    _validate_rl_action_sequence(cfg)

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        legacy = compose(config_name="robobase_config", overrides=["method=q_chunking"])
    assert legacy.method._target_ == "robobase.method.q_chunking.QChunking"
    assert legacy.method.flow_steps == 10
    assert legacy.method.actor_num_samples == 32


def test_djcqn_bigym_launch_uses_corrected_demo_and_replanning_contract():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=djcqn_pixel_bigym",
                "env=bigym/move_plate",
            ],
        )
    assert cfg.action_sequence == cfg.replay.nstep == 5
    assert cfg.execution_length == 1
    assert not cfg.temporal_ensemble
    assert cfg.env.truncate_demo_at_success
    assert not cfg.use_self_imitation
    assert cfg.num_pretrain_steps == 0
    assert cfg.num_train_frames == 100_000
    assert cfg.online_update_after_steps == 0
    assert cfg.batch_size == cfg.demo_batch_size == 128
    assert cfg.replay.demo_size == 1_000_000
    assert cfg.artifacts.save_eval_checkpoints
    assert cfg.method.prefix_horizon == 1
    assert cfg.method.sibling_level == -1
    assert cfg.method.encoder_model.type == "cqn"
    assert "actor_lr" not in cfg.method
    assert "bc_lambda" not in cfg.method
    _validate_rl_action_sequence(cfg)


def test_djcqn_state_dmc_launch_does_not_merge_q_chunking_actor_fields():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=djcqn_state_dmc",
                "env=dmc/cartpole_balance",
            ],
        )
    assert cfg.method.name == "djcqn"
    assert cfg.method._target_ == "robobase.method.djcqn.DJCQN"
    assert not cfg.pixels
    assert cfg.action_sequence == cfg.replay.nstep == 5
    assert "actor_lr" not in cfg.method
    assert "flow_steps" not in cfg.method
    _validate_rl_action_sequence(cfg)


def test_djcqn_rejects_nonmatching_h_step_or_nonreplanning_contract():
    validate_djcqn_config(
        action_sequence=5,
        execution_length=1,
        replay_nstep=5,
        temporal_ensemble=False,
    )
    for changed in (
        {"execution_length": 2},
        {"replay_nstep": 4},
        {"temporal_ensemble": True},
        {"prefix_horizon": 2},
    ):
        values = dict(
            action_sequence=5,
            execution_length=1,
            replay_nstep=5,
            temporal_ensemble=False,
            prefix_horizon=1,
        )
        values.update(changed)
        try:
            validate_djcqn_config(**values)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid contract unexpectedly accepted: {changed}")


def test_joint_and_prefix_critic_shapes_and_prefix_causality():
    agent = _agent()
    obs = {"low_dim_state": np.zeros((2, 1, 5), dtype=np.float32)}
    features = agent._features(agent.params, agent._prepare_rl_obs_inputs(obs))
    joint = agent.joint_model.apply(
        agent.params["joint"], features, jnp.zeros((2, 6), dtype=jnp.float32)
    )
    assert joint.shape == (2, 3)

    zeros = jnp.zeros((2, 2), dtype=jnp.float32)
    prefix_a = agent.prefix_model.apply(
        agent.params["prefix"],
        features,
        zeros - 1,
        zeros + 1,
        zeros,
        zeros,
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.ones((2,), dtype=jnp.int32),
    )
    changed_prefix = zeros.at[:, 0].set(0.75)
    changed_mask = zeros.at[:, 0].set(1.0)
    prefix_b = agent.prefix_model.apply(
        agent.params["prefix"],
        features,
        zeros - 1,
        zeros + 1,
        changed_prefix,
        changed_mask,
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.ones((2,), dtype=jnp.int32),
    )
    assert prefix_a.shape == (2, 3, 3)
    assert not np.allclose(prefix_a, prefix_b)


def test_expectile_loss_has_no_unseen_sibling_output_gradients():
    values = jnp.asarray([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])
    chosen = jnp.asarray([1])
    target = jnp.asarray([[0.9, 0.8]])

    def loss_fn(x):
        return jnp.sum(chosen_bin_upper_expectile_loss(x, chosen, target, 0.8))

    grads = np.asarray(jax.grad(loss_fn)(values))
    assert np.any(grads[..., 1] != 0)
    np.testing.assert_array_equal(grads[..., 0], 0)
    np.testing.assert_array_equal(grads[..., 2], 0)


def test_adjacent_sibling_is_ranked_by_per_env_ensemble_disagreement():
    # [batch, level*factor queries, critic heads, bins].  Environment zero is
    # most uncertain at factor 1 / right sibling; environment one is most
    # uncertain at factor 0 / left sibling.
    values = np.zeros((2, 2, 3, 3), dtype=np.float32)
    values[0, 1, :, 2] = [0.0, 2.0, 4.0]
    values[1, 0, :, 0] = [-3.0, 0.0, 3.0]
    factors, offsets, disagreement = rank_adjacent_sibling_disagreement(
        jnp.asarray(values),
        jnp.ones((2, 2), dtype=jnp.int32),
        level=0,
        factors=2,
        bins=3,
    )
    np.testing.assert_array_equal(factors, [1, 0])
    np.testing.assert_array_equal(offsets, [1, -1])
    assert np.all(np.asarray(disagreement) > 0.0)


def test_absolute_topk_does_not_accumulate_previous_prefix_scores():
    previous = jnp.asarray([[100.0, 0.0]])
    current_absolute = jnp.asarray([[0.1, 0.9]])
    values, indices = absolute_topk(current_absolute, 1)
    np.testing.assert_allclose(values, [[0.9]])
    np.testing.assert_array_equal(indices, [[1]])
    assert int(jnp.argmax(previous + current_absolute, axis=-1)[0]) == 0


class _ConditionalMockPrefix:
    num_critics = 2

    def apply(
        self,
        params,
        features,
        low,
        high,
        selected,
        selected_mask,
        level_indices,
        factor_indices,
    ):
        del params, features, low, high, selected_mask, level_indices
        batch = selected.shape[0]
        # Factor 0 greedily takes bin 0. Factor 1 takes bin 0 after a negative
        # factor-0 choice and bin 2 after the forced zero-valued sibling.
        factor0 = jnp.broadcast_to(jnp.asarray([3.0, 2.0, 1.0]), (batch, 3))
        choose_right = selected[:, 0] >= -0.1
        factor1 = jnp.where(
            choose_right[:, None],
            jnp.asarray([0.0, 1.0, 4.0]),
            jnp.asarray([4.0, 1.0, 0.0]),
        )
        scores = jnp.where(factor_indices[:, None] == 0, factor0, factor1)
        return jnp.repeat(scores[:, None, :], 2, axis=1)


def test_forced_sibling_recomputes_suffix_under_changed_prefix():
    kwargs = dict(
        model=_ConditionalMockPrefix(),
        params={},
        features=jnp.zeros((1, 1)),
        action_low=-jnp.ones((2,)),
        action_high=jnp.ones((2,)),
        levels=1,
        bins=3,
        beam_width=1,
        head_indices=jnp.zeros((1,), dtype=jnp.int32),
        eval_lcb_beta=None,
    )
    greedy, _, _ = c2f_prefix_beam_search(**kwargs)
    sibling, _, _ = c2f_prefix_beam_search(
        **kwargs,
        forced_sibling_level=0,
        forced_sibling_factor=0,
        forced_sibling_offset=1,
    )
    assert float(greedy[0, 0]) < -0.5
    assert float(greedy[0, 1]) < -0.5
    assert abs(float(sibling[0, 0])) < 0.1
    assert float(sibling[0, 1]) > 0.5


def test_episode_head_persists_until_reset_but_not_across_fresh_env_resume():
    agent = _agent()
    first = agent._episode_heads(2)
    second = agent._episode_heads(2)
    np.testing.assert_array_equal(first, second)
    state = agent.state_dict()
    checkpoint = agent.checkpoint_state_dict()
    assert "train_episode_heads" not in state
    saved_rng = np.asarray(checkpoint["rng_key"])
    agent.load_state_dict(state)
    agent.load_checkpoint_state_dict(checkpoint)
    np.testing.assert_array_equal(agent._train_episode_heads, [-1, -1])
    np.testing.assert_array_equal(agent.rng_key, saved_rng)
    _, expected_head_key = jax.random.split(jnp.asarray(saved_rng))
    expected_resumed = np.asarray(
        jax.random.randint(expected_head_key, (2,), 0, agent.num_critics)
    )
    resumed = agent._episode_heads(2)
    np.testing.assert_array_equal(resumed, expected_resumed)
    agent.reset(10, [0])
    assert agent._train_episode_heads[0] == -1
    assert agent._train_episode_heads[1] == resumed[1]


def test_jitted_online_sibling_path_is_runnable_and_default_remains_disabled():
    default_agent = _agent(jit=True)
    assert default_agent.sibling_exploration_prob == 0.0
    exploring_agent = _agent(
        jit=True,
        overrides=(
            "num_explore_steps=0",
            "method.sibling_exploration_prob=1.0",
        ),
    )
    obs = {"low_dim_state": np.zeros((2, 1, 5), dtype=np.float32)}
    command = exploring_agent.act(obs, 0, eval_mode=False)
    assert command.shape == (2, 3, 2)
    diagnostics = exploring_agent.rollout_diagnostics()
    assert diagnostics["djcqn_sibling_explored"] == 1.0
    assert np.isfinite(diagnostics["djcqn_sibling_disagreement"])
    assert 0.0 <= diagnostics["djcqn_sibling_factor"] < 2.0
    np.testing.assert_array_equal(command[:, 1:], 0.0)


def test_vector_envs_apply_their_own_ranked_sibling_and_recomputed_action():
    agent = _agent(
        overrides=(
            "num_explore_steps=0",
            "method.sibling_exploration_prob=1.0",
        )
    )
    calls = []

    def fake_policy(params, obs_inputs, heads, factor, offset):
        del params
        batch = obs_inputs.shape[0]
        calls.append((int(factor), int(offset), np.asarray(heads).copy()))
        marker = 0.0 if factor < 0 else float((int(factor) + 1) * int(offset)) / 4
        action = jnp.full((batch, 2), marker, dtype=jnp.float32)
        head_values = jnp.zeros((batch, 3), dtype=jnp.float32)
        score = jnp.full((batch,), marker, dtype=jnp.float32)
        return action, head_values, score

    agent._train_policy_impl = fake_policy
    agent._sibling_rank_impl = lambda params, obs_inputs, action: (
        jnp.asarray([0, 1], dtype=jnp.int32),
        jnp.asarray([-1, 1], dtype=jnp.int32),
        jnp.asarray([0.4, 0.8], dtype=jnp.float32),
    )
    obs = {"low_dim_state": np.zeros((2, 1, 5), dtype=np.float32)}
    command = agent.act(obs, 0, eval_mode=False)
    np.testing.assert_allclose(command[0, 0], [-0.25, -0.25])
    np.testing.assert_allclose(command[1, 0], [0.5, 0.5])
    assert [(factor, offset) for factor, offset, _ in calls] == [
        (-1, 0),
        (0, -1),
        (1, 1),
    ]
    diagnostics = agent.rollout_diagnostics()
    assert diagnostics["djcqn_sibling_explored"] == 1.0
    assert diagnostics["djcqn_sibling_disagreement"] == pytest.approx(0.6)
    assert diagnostics["djcqn_sibling_factor"] == pytest.approx(0.5)


def test_initial_random_exploration_precedes_value_and_sibling_selection():
    agent = _agent(
        overrides=(
            "num_explore_steps=10",
            "method.sibling_exploration_prob=1.0",
        )
    )
    obs = {"low_dim_state": np.zeros((2, 1, 5), dtype=np.float32)}
    command = agent.act(obs, 9, eval_mode=False)
    assert command.shape == (2, 3, 2)
    np.testing.assert_array_equal(command[:, 1:], 0.0)
    np.testing.assert_array_equal(agent._train_episode_heads, [-1, -1])
    assert agent.rollout_diagnostics()["djcqn_sibling_explored"] == 0.0


def test_h_step_target_and_only_fresh_first_command_is_authoritative():
    target = q_chunking_td_target(
        jnp.asarray([0.4]),
        jnp.asarray([0.99**3]),
        jnp.asarray([1.0]),
        jnp.asarray([0.7]),
    )
    np.testing.assert_allclose(target, [0.4 + 0.99**3 * 0.7])

    agent = _agent()
    obs = {"low_dim_state": np.zeros((2, 1, 5), dtype=np.float32)}
    command = agent.act(obs, 0, eval_mode=False)
    assert command.shape == (2, 3, 2)
    np.testing.assert_array_equal(command[:, 1:], 0.0)
    np.testing.assert_array_equal(
        _replay_action_from_step(command[0], {}), command[0, 0]
    )


def test_replay_assembles_actual_consecutive_h_step_action_and_return(tmp_path):
    horizon = 3
    gamma = 0.9
    replay = UniformReplayBuffer(
        batch_size=1,
        replay_capacity=16,
        nstep=horizon,
        gamma=gamma,
        action_shape=(horizon, 1),
        action_dtype=np.float32,
        reward_shape=(),
        reward_dtype=np.float32,
        observation_elements=spaces.Dict(
            {
                "low_dim_state": spaces.Box(
                    -np.inf, np.inf, shape=(1, 1), dtype=np.float32
                )
            }
        ),
        extra_replay_elements=spaces.Dict({}),
        save_dir=str(tmp_path),
        purge_replay_on_shutdown=False,
        save_snapshot=True,
        num_workers=0,
    )
    for index in range(5):
        replay.add(
            {"low_dim_state": np.asarray([index], dtype=np.float32)},
            np.asarray([index / 10.0], dtype=np.float32),
            np.float32(index + 1),
            terminal=index == 4,
            truncated=False,
        )
    replay.add_final({"low_dim_state": np.asarray([5], dtype=np.float32)})
    batch = replay.sample_batch_indices(np.asarray([0]))
    np.testing.assert_allclose(batch["action"][0, :, 0], [0.0, 0.1, 0.2])
    np.testing.assert_allclose(
        batch["reward"], [1.0 + gamma * 2.0 + gamma**2 * 3.0]
    )
    np.testing.assert_allclose(batch["discount"], [gamma**horizon])
    assert not batch["action_pad_mask"].any()


def test_jitted_update_is_finite_and_updates_both_value_heads():
    agent = _agent(jit=True)
    agent.logging = True
    before_joint = jax.tree.map(np.asarray, agent.params["joint"])
    before_prefix = jax.tree.map(np.asarray, agent.params["prefix"])
    metrics = agent.update(iter([_batch()]), step=0)
    assert np.isfinite(metrics["critic_loss"])
    assert np.isfinite(metrics["joint_critic_loss"])
    assert np.isfinite(metrics["prefix_critic_loss"])
    assert _tree_changed(before_joint, agent.params["joint"])
    assert _tree_changed(before_prefix, agent.params["prefix"])
