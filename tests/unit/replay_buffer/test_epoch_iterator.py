import numpy as np
from gymnasium import spaces

from robobase.envs.env import DemoEnv
from robobase.envs.wrappers import ActionSequence, FrameStack, RecedingHorizonControl
from robobase.replay_buffer.iterator import create_epoch_replay_iterator
from robobase.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from robobase.utils import add_demo_to_replay_buffer
from tests.unit.wrappers.utils import DummyEnv


def _collect_demo_from_dummy_env(env, num_demo):
    demos = []
    for traj_idx in range(num_demo):
        obs, info = env.reset()
        info["demo"] = 1
        current_demo = [[obs, info]]
        terminated = truncated = False
        while not terminated and not truncated:
            action = np.ones_like(env.action_space.sample()) * (traj_idx + 1) / 100.0
            obs, reward, terminated, truncated, info = env.step(action)
            info["demo_action"] = action
            info["demo"] = 1
            current_demo.append([obs, reward, terminated, truncated, info])
        demos.append(current_demo)
    return demos


def _wrap_env(env, frame_stack, action_sequence, execution_length, *, demo_env=False):
    if demo_env:
        return env
    env = FrameStack(env, frame_stack)
    if action_sequence == execution_length:
        return ActionSequence(env, action_sequence)
    return RecedingHorizonControl(
        env,
        action_sequence,
        5,
        execution_length,
        temporal_ensemble=False,
    )


def test_epoch_iterator_covers_each_valid_anchor_once_per_epoch():
    episode_len = 20
    num_episodes = 3
    frame_stack = 2
    action_sequence = 5
    execution_length = 2
    batch_size = 6

    raw_env = DummyEnv(episode_len=episode_len)
    demos = _collect_demo_from_dummy_env(raw_env, num_episodes)
    demo_env = DemoEnv(demos, raw_env.action_space, raw_env.observation_space)
    env = _wrap_env(
        raw_env,
        frame_stack,
        action_sequence,
        execution_length,
        demo_env=False,
    )

    replay_buffer = UniformReplayBuffer(
        batch_size=batch_size,
        replay_capacity=256,
        action_shape=env.action_space.shape,
        action_dtype=env.action_space.dtype,
        nstep=1,
        reward_shape=(),
        reward_dtype=np.float32,
        observation_elements=env.observation_space,
        extra_replay_elements=spaces.Dict(
            {"demo": spaces.Box(0, 1, shape=(), dtype=np.uint8)}
        ),
        num_workers=0,
    )

    for _ in range(len(demos)):
        add_demo_to_replay_buffer(demo_env, replay_buffer)

    iterator = create_epoch_replay_iterator(
        replay_buffer,
        execution_length=execution_length,
        seed=0,
    )

    expected_per_episode = 18
    assert iterator.sample_indices.shape == (expected_per_episode * num_episodes,)
    assert iterator.batches_per_epoch == 9

    collected_indices = []
    for _ in range(iterator.batches_per_epoch):
        batch = next(iterator)
        assert batch["indices"].shape == (batch_size,)
        collected_indices.extend(batch["indices"].tolist())

    assert len(collected_indices) == expected_per_episode * num_episodes
    assert len(set(collected_indices)) == expected_per_episode * num_episodes
    assert sorted(collected_indices) == sorted(iterator.sample_indices.tolist())

    assert iterator.sample_indices.min() == 0
    assert iterator.sample_indices.max() == 57
    assert 18 not in iterator.sample_indices
    assert 19 not in iterator.sample_indices
    assert 38 not in iterator.sample_indices
    assert 39 not in iterator.sample_indices
    assert 58 not in iterator.sample_indices
    assert 59 not in iterator.sample_indices


def test_epoch_iterator_streams_without_eager_loading():
    episode_len = 8
    num_episodes = 2
    frame_stack = 1
    action_sequence = 3
    execution_length = 1
    batch_size = 4

    raw_env = DummyEnv(episode_len=episode_len)
    demos = _collect_demo_from_dummy_env(raw_env, num_episodes)
    demo_env = DemoEnv(demos, raw_env.action_space, raw_env.observation_space)
    env = _wrap_env(
        raw_env,
        frame_stack,
        action_sequence,
        execution_length,
        demo_env=False,
    )

    replay_buffer = UniformReplayBuffer(
        batch_size=batch_size,
        replay_capacity=256,
        action_shape=env.action_space.shape,
        action_dtype=env.action_space.dtype,
        nstep=1,
        reward_shape=(),
        reward_dtype=np.float32,
        observation_elements=env.observation_space,
        extra_replay_elements=spaces.Dict(
            {"demo": spaces.Box(0, 1, shape=(), dtype=np.uint8)}
        ),
        num_workers=0,
        max_cached_episodes=1,
    )

    for _ in range(len(demos)):
        add_demo_to_replay_buffer(demo_env, replay_buffer)

    def fail_load_all():
        raise AssertionError("epoch iterator should not eagerly load all episodes")

    replay_buffer.load_all_episodes = fail_load_all
    iterator = create_epoch_replay_iterator(
        replay_buffer,
        execution_length=execution_length,
        seed=0,
    )

    batch = next(iterator)

    assert batch["indices"].shape == (batch_size,)
    assert len(replay_buffer._episodes) <= 1


def test_epoch_iterator_can_skip_tp1_observations():
    episode_len = 8
    batch_size = 4

    raw_env = DummyEnv(episode_len=episode_len)
    demos = _collect_demo_from_dummy_env(raw_env, num_demo=1)
    demo_env = DemoEnv(demos, raw_env.action_space, raw_env.observation_space)
    env = _wrap_env(
        raw_env,
        frame_stack=1,
        action_sequence=3,
        execution_length=1,
        demo_env=False,
    )

    replay_buffer = UniformReplayBuffer(
        batch_size=batch_size,
        replay_capacity=256,
        action_shape=env.action_space.shape,
        action_dtype=env.action_space.dtype,
        nstep=1,
        reward_shape=(),
        reward_dtype=np.float32,
        observation_elements=env.observation_space,
        extra_replay_elements=spaces.Dict(
            {"demo": spaces.Box(0, 1, shape=(), dtype=np.uint8)}
        ),
        num_workers=0,
        include_tp1=False,
    )

    add_demo_to_replay_buffer(demo_env, replay_buffer)
    batch = replay_buffer.sample_batch_indices(np.arange(batch_size))

    obs_key = next(iter(env.observation_space.spaces.keys()))
    assert obs_key in batch
    assert not any(key.endswith("_tp1") for key in batch)


def test_epoch_iterator_can_group_batches_by_episode_chunks():
    episode_len = 20
    num_episodes = 3
    frame_stack = 2
    action_sequence = 5
    execution_length = 2
    batch_size = 6
    batch_chunk_size = 3

    raw_env = DummyEnv(episode_len=episode_len)
    demos = _collect_demo_from_dummy_env(raw_env, num_episodes)
    demo_env = DemoEnv(demos, raw_env.action_space, raw_env.observation_space)
    env = _wrap_env(
        raw_env,
        frame_stack,
        action_sequence,
        execution_length,
        demo_env=False,
    )

    replay_buffer = UniformReplayBuffer(
        batch_size=batch_size,
        replay_capacity=256,
        action_shape=env.action_space.shape,
        action_dtype=env.action_space.dtype,
        nstep=1,
        reward_shape=(),
        reward_dtype=np.float32,
        observation_elements=env.observation_space,
        extra_replay_elements=spaces.Dict(
            {"demo": spaces.Box(0, 1, shape=(), dtype=np.uint8)}
        ),
        num_workers=0,
    )

    for _ in range(len(demos)):
        add_demo_to_replay_buffer(demo_env, replay_buffer)

    iterator = create_epoch_replay_iterator(
        replay_buffer,
        execution_length=execution_length,
        seed=0,
        batch_chunk_size=batch_chunk_size,
    )

    collected_indices = []
    for _ in range(iterator.batches_per_epoch):
        batch = next(iterator)
        indices = batch["indices"]
        assert indices.shape == (batch_size,)
        # The locality-aware iterator should cap each full batch at
        # batch_size / batch_chunk_size episode chunks.
        episode_ids = indices // episode_len
        assert np.unique(episode_ids).size <= batch_size // batch_chunk_size
        collected_indices.extend(indices.tolist())

    assert len(collected_indices) == iterator.sample_indices.size
    assert len(set(collected_indices)) == iterator.sample_indices.size
    assert sorted(collected_indices) == sorted(iterator.sample_indices.tolist())


def test_action_sequence_start_offset_uses_edge_padding():
    observation_space = spaces.Dict(
        {
            "low_dim_state": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(2, 1),
                dtype=np.float32,
            )
        }
    )
    replay_buffer = UniformReplayBuffer(
        batch_size=1,
        replay_capacity=16,
        action_shape=(4, 1),
        action_dtype=np.float32,
        nstep=1,
        reward_shape=(),
        reward_dtype=np.float32,
        observation_elements=observation_space,
        extra_replay_elements=spaces.Dict({}),
        num_workers=0,
        action_sequence_start_offset=1,
        action_padding="edge",
    )

    for index in range(4):
        replay_buffer.add(
            {"low_dim_state": np.asarray([index], dtype=np.float32)},
            np.asarray([index], dtype=np.float32),
            0.0,
            index == 3,
            False,
        )
    replay_buffer.add_final({"low_dim_state": np.asarray([4], dtype=np.float32)})

    start_sample = replay_buffer.sample(batch_size=1, indices=[0])
    end_sample = replay_buffer.sample(batch_size=1, indices=[3])
    vector_sample = replay_buffer.sample_batch_indices(np.asarray([0, 3]))

    np.testing.assert_array_equal(start_sample["action"][0, :, 0], [0, 0, 1, 2])
    np.testing.assert_array_equal(end_sample["action"][0, :, 0], [2, 3, 3, 3])
    np.testing.assert_array_equal(vector_sample["action"][:, :, 0], [[0, 0, 1, 2], [2, 3, 3, 3]])
    assert not start_sample["action_pad_mask"].any()
    assert not end_sample["action_pad_mask"].any()
    assert not vector_sample["action_pad_mask"].any()
