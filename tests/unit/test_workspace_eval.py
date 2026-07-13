import numpy as np
import pytest
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from omegaconf import OmegaConf

from robobase.envs.wrappers import ActionSequence
from robobase.workspace import Workspace, _validate_eval_env_counts
from tests.unit.wrappers.utils import DummyEnv


def test_vector_action_step_count_handles_object_masks_and_final_info():
    env = SyncVectorEnv(
        [lambda: ActionSequence(DummyEnv(episode_len=7), 5) for _ in range(2)]
    )
    workspace = Workspace.__new__(Workspace)
    workspace.eval_envs = env

    try:
        env.reset()
        *_, running_info = env.step(env.action_space.sample())
        np.testing.assert_array_equal(
            workspace._executed_vector_action_steps(running_info),
            np.asarray([5, 5], dtype=np.int32),
        )

        *_, final_info = env.step(env.action_space.sample())
        np.testing.assert_array_equal(
            workspace._executed_vector_action_steps(final_info),
            np.asarray([2, 2], dtype=np.int32),
        )
    finally:
        env.close()


def test_zero_eval_allows_zero_envs_and_does_not_create_them():
    cfg = OmegaConf.create({"num_eval_episodes": 0, "num_eval_envs": 0})
    _validate_eval_env_counts(cfg)

    class NoEvalFactory:
        def make_eval_env(self, cfg):
            raise AssertionError("single eval env should not be created")

        def make_eval_envs(self, cfg):
            raise AssertionError("vector eval envs should not be created")

    workspace = Workspace.__new__(Workspace)
    workspace.cfg = cfg
    workspace.env_factory = NoEvalFactory()
    workspace.eval_env = None
    workspace.eval_envs = None

    workspace._ensure_eval_envs_created()

    assert workspace.eval_env is None
    assert workspace.eval_envs is None


@pytest.mark.parametrize(
    "num_eval_episodes,num_eval_envs,error_field",
    [
        (1, 0, "num_eval_envs"),
        (0, -1, "num_eval_envs"),
        (-1, 0, "num_eval_episodes"),
    ],
)
def test_invalid_eval_env_counts_are_rejected(
    num_eval_episodes,
    num_eval_envs,
    error_field,
):
    cfg = OmegaConf.create(
        {
            "num_eval_episodes": num_eval_episodes,
            "num_eval_envs": num_eval_envs,
        }
    )

    with pytest.raises(ValueError, match=error_field):
        _validate_eval_env_counts(cfg)


@pytest.mark.parametrize(
    "num_eval_envs,expected_single_calls,expected_vector_calls",
    [(1, 1, 0), (2, 0, 1)],
)
def test_eval_env_creation_only_builds_selected_mode(
    num_eval_envs,
    expected_single_calls,
    expected_vector_calls,
):
    class CountingFactory:
        def __init__(self):
            self.single_calls = 0
            self.vector_calls = 0

        def make_eval_env(self, cfg):
            del cfg
            self.single_calls += 1
            return object()

        def make_eval_envs(self, cfg):
            del cfg
            self.vector_calls += 1
            return object()

    factory = CountingFactory()
    workspace = Workspace.__new__(Workspace)
    workspace.cfg = OmegaConf.create(
        {"num_eval_episodes": 1, "num_eval_envs": num_eval_envs}
    )
    workspace.env_factory = factory
    workspace.eval_env = None
    workspace.eval_envs = None

    workspace._ensure_eval_envs_created()

    assert factory.single_calls == expected_single_calls
    assert factory.vector_calls == expected_vector_calls


class _SeedRecordingEnv(gym.Env):
    def __init__(
        self,
        *,
        slow_seed=None,
        slow_episode_length=1,
        reward_with_seed=False,
    ):
        self.observation_space = gym.spaces.Dict(
            {
                "low_dim_state": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1, 1),
                    dtype=np.float32,
                )
            }
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1, 1),
            dtype=np.float32,
        )
        self.reset_seeds = []
        self.slow_seed = slow_seed
        self.slow_episode_length = int(slow_episode_length)
        self.reward_with_seed = bool(reward_with_seed)
        self.current_seed = None
        self.episode_step = 0

    def reset(self, *, seed=None, options=None):
        del options
        super().reset(seed=seed)
        self.reset_seeds.append(seed)
        self.current_seed = seed
        self.episode_step = 0
        value = -1 if seed is None else seed
        return {
            "low_dim_state": np.asarray([[value]], dtype=np.float32)
        }, {}

    def step(self, action):
        del action
        self.episode_step += 1
        episode_length = (
            self.slow_episode_length
            if self.current_seed == self.slow_seed
            else 1
        )
        terminated = self.episode_step >= episode_length
        reward = (
            float(self.current_seed)
            if self.reward_with_seed and terminated
            else float(terminated)
        )
        return (
            {"low_dim_state": np.zeros((1, 1), dtype=np.float32)},
            reward,
            terminated,
            False,
            {"task_success": int(terminated)},
        )


class _SeedAwareAgent:
    logging = False

    def __init__(self):
        self.active_seed_history = []
        self.observed_reset_values = []
        self.aligned_noise_resets = 0
        self.eval_env_running = False

    def reset_aligned_eval_noise(self):
        self.aligned_noise_resets += 1

    def set_active_eval_seeds(self, seeds):
        self.active_seed_history.append(None if seeds is None else list(seeds))

    def set_eval_env_running(self, value):
        self.eval_env_running = bool(value)

    def reset(self, step, agents_to_reset):
        del step, agents_to_reset

    def act(self, observations, step, eval_mode):
        del step, eval_mode
        values = np.asarray(observations["low_dim_state"])[:, 0, 0]
        self.observed_reset_values.append(values.astype(int).tolist())
        return np.zeros((len(values), 1, 1), dtype=np.float32)


class _NoopVideoRecorder:
    frames = []

    def init(self, env, enabled):
        del env, enabled

    def record(self, env):
        del env

    def save(self, file_name):
        del file_name


def test_vector_eval_reseeds_autoreset_slots_and_aligns_agent_noise():
    env = SyncVectorEnv([_SeedRecordingEnv, _SeedRecordingEnv])
    workspace = Workspace.__new__(Workspace)
    workspace.cfg = OmegaConf.create(
        {
            "num_eval_episodes": 4,
            "num_eval_envs": 2,
            "action_repeat": 1,
            "log_eval_video": False,
            "env": {"eval_seed_start": 10},
        }
    )
    workspace.env_factory = None
    workspace.eval_env = None
    workspace.eval_envs = env
    workspace.agent = _SeedAwareAgent()
    workspace.eval_video_recorder = _NoopVideoRecorder()
    workspace._eval_agent_indices = [0, 1]
    workspace._main_loop_iterations = 0
    workspace._defer_live_eval_env_creation = False

    try:
        metrics = workspace._eval()

        assert metrics["episode_reward"] == pytest.approx(1.0)
        assert metrics["episode_length"] == pytest.approx(1.0)
        assert metrics["episode_success"] == pytest.approx(1.0)
        assert workspace.agent.observed_reset_values == [[10, 11], [12, 13]]
        assert workspace.agent.active_seed_history == [
            [10, 11],
            [12, 13],
            None,
        ]
        assert workspace.agent.aligned_noise_resets == 1
        assert workspace.agent.eval_env_running is False
        assert env.envs[0].reset_seeds == [10, None, 12, None]
        assert env.envs[1].reset_seeds == [11, None, 13, None]
    finally:
        env.close()


def test_vector_eval_waits_for_each_target_seed_instead_of_fast_fillers():
    env = SyncVectorEnv(
        [
            lambda: _SeedRecordingEnv(
                slow_seed=11,
                slow_episode_length=5,
                reward_with_seed=True,
            ),
            lambda: _SeedRecordingEnv(
                slow_seed=11,
                slow_episode_length=5,
                reward_with_seed=True,
            ),
        ]
    )
    workspace = Workspace.__new__(Workspace)
    workspace.cfg = OmegaConf.create(
        {
            "num_eval_episodes": 4,
            "num_eval_envs": 2,
            "action_repeat": 1,
            "log_eval_video": False,
            "env": {"eval_seed_start": 10},
        }
    )
    workspace.env_factory = None
    workspace.eval_env = None
    workspace.eval_envs = env
    workspace.agent = _SeedAwareAgent()
    workspace.eval_video_recorder = _NoopVideoRecorder()
    workspace._eval_agent_indices = [0, 1]
    workspace._main_loop_iterations = 0
    workspace._defer_live_eval_env_creation = False

    try:
        metrics = workspace._eval()

        assert metrics["episode_reward"] == pytest.approx((10 + 11 + 12 + 13) / 4)
        assert metrics["episode_length"] == pytest.approx((1 + 5 + 1 + 1) / 4)
        assert workspace.agent.observed_reset_values == [
            [10, 11],
            [12, 0],
            [13, 0],
            [14, 0],
            [15, 0],
        ]
        assert workspace.agent.active_seed_history == [
            [10, 11],
            [12, 11],
            [13, 11],
            [14, 11],
            [15, 11],
            None,
        ]
    finally:
        env.close()
