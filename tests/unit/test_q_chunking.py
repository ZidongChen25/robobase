from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnasium import spaces
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from robobase.factory import create_agent, method_name_from_cfg
from robobase.method.q_chunking import (
    q_chunking_td_target,
    validate_q_chunking_config,
)
from robobase.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from robobase.replay_buffer.vision_feature_cache import JAX_Q_CHUNKING_FEATURE_KEY
from robobase.workspace import _online_updates_ready, _validate_rl_action_sequence


CONFIG_DIR = str((Path(__file__).parents[2] / "robobase" / "cfgs").resolve())


def _spaces(horizon=3):
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
        shape=(horizon, 2),
        dtype=np.float32,
    )
    return observation_space, action_space


def _config(horizon=3, *, jit=True, overrides=()):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(
            config_name="robobase_config",
            overrides=[
                "method=q_chunking",
                f"action_sequence={horizon}",
                f"replay.nstep={horizon}",
                "num_train_envs=1",
                "num_eval_envs=1",
                f"backend.jit={'true' if jit else 'false'}",
                "method.flow_steps=2",
                "method.actor_num_samples=4",
                "method.model.hidden_dims=[32,32]",
                *overrides,
            ],
        )


def _batch(batch_size=4, horizon=3):
    rng = np.random.default_rng(7)
    return {
        "low_dim_state": rng.normal(size=(batch_size, 1, 5)).astype(np.float32),
        "low_dim_state_tp1": rng.normal(size=(batch_size, 1, 5)).astype(np.float32),
        "action": rng.uniform(
            -1.0,
            1.0,
            size=(batch_size, horizon, 2),
        ).astype(np.float32),
        "action_pad_mask": np.zeros((batch_size, horizon), dtype=bool),
        "reward": rng.normal(size=(batch_size,)).astype(np.float32),
        "discount": np.full((batch_size,), 0.99**horizon, dtype=np.float32),
        "terminal": np.zeros((batch_size,), dtype=bool),
        "truncated": np.zeros((batch_size,), dtype=bool),
    }


def _tree_changed(before, after):
    before_leaves, before_tree = jax.tree.flatten(before)
    after_leaves, after_tree = jax.tree.flatten(after)
    assert before_tree == after_tree
    return any(
        not np.allclose(np.asarray(left), np.asarray(right))
        for left, right in zip(before_leaves, after_leaves, strict=True)
    )


def test_q_chunking_method_target_is_registered():
    cfg = _config()
    assert method_name_from_cfg(cfg) == "q_chunking"
    assert cfg.method._target_ == "robobase.method.q_chunking.QChunking"


def test_q_chunking_launch_has_official_replay_contract():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=q_chunking_pixel_bigym",
                "env=bigym/move_plate",
            ],
        )

    assert cfg.method.name == "q_chunking"
    assert cfg.action_sequence == 5
    assert cfg.execution_length == 1
    assert cfg.replay.nstep == cfg.action_sequence
    assert cfg.replay.include_tp1
    assert cfg.replay.action_padding == "zero"
    assert not cfg.temporal_ensemble
    assert cfg.method.flow_steps == 10
    assert cfg.method.actor_num_samples == 32
    assert cfg.method.q_aggregate == "mean"
    assert cfg.method.model.activation == "gelu"
    assert cfg.method.encoder_model.type == "cqn"
    assert cfg.env.cameras == ["head", "left_wrist", "right_wrist"]
    assert cfg.num_pretrain_steps == 1_000_000
    assert cfg.num_train_frames == 2_000_000
    assert cfg.online_update_after_steps == 5_000
    assert cfg.save_csv


def test_workspace_allows_q_chunking_action_sequence():
    _validate_rl_action_sequence(_config())


def test_q_chunking_online_update_delay_is_independent_of_replay_size():
    cfg = _config()
    cfg.online_update_after_steps = 5_000
    cfg.replay_size_before_train = 500

    assert not _online_updates_ready(
        cfg,
        main_loop_iterations=4_999,
        replay_size=50_000,
    )
    assert not _online_updates_ready(
        cfg,
        main_loop_iterations=5_000,
        replay_size=499,
    )
    assert _online_updates_ready(
        cfg,
        main_loop_iterations=5_000,
        replay_size=500,
    )


def test_q_chunking_state_dmc_launch_composes():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=q_chunking_state_dmc",
                "env=dmc/cartpole_balance",
            ],
        )

    assert not cfg.pixels
    assert cfg.action_sequence == 5
    assert cfg.execution_length == 1
    assert cfg.replay.nstep == 5
    assert not cfg.temporal_ensemble


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"action_sequence": 1}, "action_sequence"),
        ({"execution_length": 2}, "execution_length"),
        ({"replay_nstep": 1}, "replay.nstep"),
        ({"temporal_ensemble": True}, "temporal_ensemble"),
        ({"action_execution_start": 1}, "action_execution_start"),
    ],
)
def test_q_chunking_rejects_biased_replay_or_rollout_contract(kwargs, message):
    values = {
        "action_sequence": 3,
        "execution_length": 1,
        "replay_nstep": 3,
        "temporal_ensemble": False,
        "action_execution_start": 0,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        validate_q_chunking_config(**values)


def test_q_chunking_td_target_uses_replay_k_step_discount():
    target = q_chunking_td_target(
        jnp.asarray([1.0, 2.0]),
        jnp.asarray([0.99**3, 0.99**3]),
        jnp.asarray([1.0, 0.0]),
        jnp.asarray([4.0, 9.0]),
    )
    np.testing.assert_allclose(
        np.asarray(target),
        np.asarray([1.0 + 0.99**3 * 4.0, 2.0]),
        rtol=1e-6,
    )


def test_q_chunking_replay_builds_matching_action_and_return_chunks(tmp_path):
    horizon = 3
    gamma = 0.9
    replay = UniformReplayBuffer(
        batch_size=2,
        replay_capacity=32,
        nstep=horizon,
        gamma=gamma,
        action_shape=(horizon, 1),
        action_dtype=np.float32,
        reward_shape=(),
        reward_dtype=np.float32,
        observation_elements=spaces.Dict(
            {
                "low_dim_state": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(1, 1),
                    dtype=np.float32,
                )
            }
        ),
        extra_replay_elements=spaces.Dict({}),
        save_dir=str(tmp_path),
        purge_replay_on_shutdown=False,
        save_snapshot=True,
        num_workers=0,
    )
    for index in range(6):
        replay.add(
            {"low_dim_state": np.asarray([index], dtype=np.float32)},
            np.asarray([index / 10.0], dtype=np.float32),
            np.float32(index + 1),
            terminal=index == 5,
            truncated=False,
        )
    replay.add_final(
        {"low_dim_state": np.asarray([6], dtype=np.float32)},
    )

    batch = replay.sample_batch_indices(np.asarray([0, 1]))

    assert batch["action"].shape == (2, horizon, 1)
    np.testing.assert_allclose(batch["action"][0, :, 0], [0.0, 0.1, 0.2])
    assert not batch["action_pad_mask"].any()
    np.testing.assert_allclose(
        batch["reward"],
        [
            1.0 + gamma * 2.0 + gamma**2 * 3.0,
            2.0 + gamma * 3.0 + gamma**2 * 4.0,
        ],
    )
    np.testing.assert_allclose(batch["discount"], gamma**horizon)
    assert batch["low_dim_state_tp1"].shape == (2, 1, 1)


def test_q_chunking_jitted_update_has_finite_losses_and_updates_both_networks():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _config(jit=True),
        observation_space=observation_space,
        action_space=action_space,
    )
    agent.logging = True
    before_actor = jax.tree.map(np.asarray, agent.params["actor"])
    before_critic = jax.tree.map(np.asarray, agent.params["critic"])

    metrics = agent.update(iter([_batch()]), step=1)

    assert np.isfinite(metrics["actor_loss"])
    assert np.isfinite(metrics["critic_loss"])
    assert metrics["chunk_valid_fraction"] == pytest.approx(1.0)
    assert _tree_changed(before_actor, agent.params["actor"])
    assert _tree_changed(before_critic, agent.params["critic"])


def test_q_chunking_pixel_actor_and_critic_encoders_initialize_independently():
    observation_space = spaces.Dict(
        {
            "low_dim_state": spaces.Box(
                -np.inf,
                np.inf,
                shape=(1, 5),
                dtype=np.float32,
            ),
            "rgb_front": spaces.Box(
                0,
                255,
                shape=(1, 3, 16, 16),
                dtype=np.uint8,
            ),
        }
    )
    _, action_space = _spaces()
    agent = create_agent(
        _config(
            jit=False,
            overrides=[
                "pixels=true",
                "method.encoder_model.type=cqn",
                "method.encoder_model.model=cqn_cnn",
                "method.encoder_model.trainable=true",
                "method.encoder_model.pretrained=false",
            ],
        ),
        observation_space=observation_space,
        action_space=action_space,
    )

    assert agent._cached_pixel_feature_key == JAX_Q_CHUNKING_FEATURE_KEY
    assert _tree_changed(
        agent.params["actor_encoder"],
        agent.params["critic_encoder"],
    )
    for critic, target in zip(
        jax.tree.leaves(agent.params["critic_encoder"]),
        jax.tree.leaves(agent.target_critic_params["critic_encoder"]),
        strict=True,
    ):
        np.testing.assert_allclose(np.asarray(target), np.asarray(critic))

    action = agent.act(
        {
            "low_dim_state": np.zeros((1, 1, 5), dtype=np.float32),
            "rgb_front": np.zeros((1, 1, 3, 16, 16), dtype=np.uint8),
        },
        step=3000,
        eval_mode=True,
    )
    assert action.shape == (1, 3, 2)
    assert np.all(np.isfinite(action))

    batch = _batch(batch_size=2)
    batch["rgb_front"] = np.zeros((2, 1, 3, 16, 16), dtype=np.uint8)
    batch["rgb_front_tp1"] = np.ones((2, 1, 3, 16, 16), dtype=np.uint8)
    before_actor_encoder = jax.tree.map(np.asarray, agent.params["actor_encoder"])
    before_critic_encoder = jax.tree.map(np.asarray, agent.params["critic_encoder"])
    agent.update(iter([batch]), step=1)
    assert _tree_changed(before_actor_encoder, agent.params["actor_encoder"])
    assert _tree_changed(before_critic_encoder, agent.params["critic_encoder"])


def test_q_chunking_masks_incomplete_chunks_from_critic_but_trains_flow_tokens():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _config(jit=False),
        observation_space=observation_space,
        action_space=action_space,
    )
    agent.logging = True
    batch = _batch()
    batch["action_pad_mask"][0, -1] = True

    metrics = agent.update(iter([batch]), step=1)

    assert metrics["chunk_valid_fraction"] == pytest.approx(0.75)
    assert metrics["action_valid_fraction"] == pytest.approx(11.0 / 12.0)
    assert np.isfinite(metrics["actor_loss"])
    assert np.isfinite(metrics["critic_loss"])


def test_q_chunking_executes_one_sampled_chunk_open_loop_and_reset_refreshes():
    observation_space, action_space = _spaces()
    agent = create_agent(
        _config(jit=False),
        observation_space=observation_space,
        action_space=action_space,
    )
    expected = jnp.asarray(
        [[[-0.8, -0.7], [-0.2, -0.1], [0.6, 0.7]]],
        dtype=jnp.float32,
    )
    calls = []

    def fixed_policy(params, obs_inputs, key):
        del params, obs_inputs, key
        calls.append(1)
        return expected.reshape((1, -1)), jnp.asarray([2.0])

    agent._policy_action_impl = fixed_policy
    observations = {"low_dim_state": np.zeros((1, 1, 5), dtype=np.float32)}

    first = agent.act(observations, step=3000, eval_mode=False)
    second = agent.act(observations, step=3001, eval_mode=False)
    third = agent.act(observations, step=3002, eval_mode=False)

    np.testing.assert_allclose(first, np.asarray(expected), atol=1e-6)
    np.testing.assert_allclose(
        second,
        np.asarray([[[-0.2, -0.1], [0.6, 0.7], [0.6, 0.7]]]),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        third,
        np.asarray([[[0.6, 0.7], [0.6, 0.7], [0.6, 0.7]]]),
        atol=1e-6,
    )
    assert len(calls) == 1

    agent.reset(step=3002, agents_to_reset=[0])
    agent.act(observations, step=3003, eval_mode=False)
    assert len(calls) == 2


def test_q_chunking_checkpoint_restores_target_critic():
    observation_space, action_space = _spaces()
    cfg = _config(jit=False)
    agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    agent.update(iter([_batch()]), step=1)
    restored = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    restored.load_state_dict(agent.state_dict())
    restored.load_checkpoint_state_dict(agent.checkpoint_state_dict())

    left = jax.tree.leaves(agent.target_critic_params)
    right = jax.tree.leaves(restored.target_critic_params)
    for expected, actual in zip(left, right, strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))


def test_q_chunking_robomimic_launch_matches_official_acfql_protocol():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=q_chunking_state_robomimic",
                "env=robomimic/square_mh",
            ],
        )

    assert cfg.method.name == "q_chunking"
    assert not cfg.pixels
    assert cfg.frame_stack == 1
    assert cfg.action_repeat == 1
    assert cfg.action_sequence == 5
    assert cfg.execution_length == 1
    assert not cfg.temporal_ensemble
    assert cfg.update_every_steps == 1
    assert cfg.batch_size == 256
    assert cfg.num_pretrain_steps == 1_000_000
    assert cfg.num_train_frames == 2_000_000
    assert cfg.online_update_after_steps == 5_000
    assert cfg.num_explore_steps == 0
    assert cfg.replay.nstep == 5
    assert cfg.replay.gamma == 0.99
    assert cfg.replay.size == 2_000_000
    assert cfg.replay.include_tp1
    assert cfg.replay.action_padding == "zero"
    assert not cfg.replay.prioritization
    assert cfg.replay.transition_uniform_sampling
    assert cfg.method.actor_lr == 3e-4
    assert cfg.method.critic_lr == 3e-4
    assert cfg.method.critic_target_tau == 0.005
    assert cfg.method.flow_steps == 10
    assert cfg.method.actor_num_samples == 32
    assert cfg.method.q_aggregate == "mean"
    assert list(cfg.method.model.hidden_dims) == [512, 512, 512, 512]
    assert cfg.method.model.activation == "gelu"
    # Official robomimic square environment contract.
    assert cfg.env.episode_length == 400
    assert cfg.env.reward_shift == -1.0
    assert cfg.env.filter_key == "all"
    assert cfg.env.use_live_env
    assert list(cfg.env.obs_keys) == [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "object",
    ]
    assert not cfg.use_min_max_normalization
    assert not cfg.use_standardization
    assert not cfg.norm_obs
    # Async-eval protocol: no in-loop evaluation, snapshots for the watcher.
    assert cfg.num_eval_episodes == 0
    assert cfg.save_snapshot
    assert cfg.snapshot_every_n == 5_000
    assert not cfg.wandb.use
    assert not cfg.log_eval_video
