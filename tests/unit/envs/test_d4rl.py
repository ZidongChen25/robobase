import pytest
import gymnasium as gym
import numpy as np
from types import SimpleNamespace
from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize

from robobase.envs.d4rl import (
    get_traj_dataset,
    get_minari_traj_dataset,
    AddTaskSuccessInfo,
    D4RLEnvCompatibility,
    ConvertObsToDict,
    D4RLEnvFactory,
)
import robobase.envs.d4rl as d4rl_env
from robobase.envs.env import DemoEnv
from robobase.envs.wrappers import RecedingHorizonControl

try:
    import gym as gym_old
except ImportError:
    gym_old = None


def _find_wrapper(env, wrapper_cls):
    current = env
    while current is not None:
        if isinstance(current, wrapper_cls):
            return current
        current = getattr(current, "env", None)
    return None


def _skip_without_old_d4rl():
    if gym_old is None or d4rl_env.d4rl is None:
        pytest.skip("old gym/d4rl dependencies are not installed")


@pytest.mark.parametrize(
    "task_name, expected_len",
    [
        ("halfcheetah-medium-v2", 1000),
        ("hopper-medium-v2", 2187),
        ("walker2d-medium-v2", 1191),
    ],
)
def test_get_trajectory_dataset(task_name, expected_len):
    _skip_without_old_d4rl()
    env = gym_old.make(task_name)
    d4rl_trajs, _ = get_traj_dataset(env)

    for traj in d4rl_trajs:  # for each trajecotry
        assert len(traj[0]) == 2  # first transition only contains obs and info
        for i in range(1, len(traj)):
            assert len(traj[i]) == 5  # subsequent transitons contain 5 items
            assert "demo_action" in traj[i][4]
            assert traj[i][4]["demo"] == 1

    assert len(d4rl_trajs) == expected_len


@pytest.mark.parametrize(
    "task_name",
    [("halfcheetah-medium-v2"), ("hopper-medium-v2"), ("walker2d-medium-v2")],
)
def test_env_compatilibility_wrapper(task_name):
    _skip_without_old_d4rl()
    env = gym_old.make(task_name)
    env = D4RLEnvCompatibility(env)

    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Box)
    assert env.observation_space.dtype == np.float32
    assert env.action_space.dtype == np.float32

    env.reset()
    action = env.action_space.sample()
    res = env.step(action)
    assert len(res) == 5  # In the new gym api, step should return 5 items.


@pytest.mark.parametrize(
    "task_name", [("HalfCheetah-v4"), ("Hopper-v4"), ("Walker2d-v4")]
)
def test_convert_obs_to_dict_wrapper(task_name):
    env = gym.make(task_name)
    env = ConvertObsToDict(env)

    assert isinstance(env.observation_space, gym.spaces.Dict)

    obs, info = env.reset()
    assert "low_dim_state" in obs

    action = env.action_space.sample()
    obs, _, _, _, _ = env.step(action)
    assert "low_dim_state" in obs


def test_add_task_success_info_maps_success_key():
    class SuccessEnv(gym.Env):
        observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            del seed, options
            return np.zeros((1,), dtype=np.float32), {}

        def step(self, action):
            del action
            return np.zeros((1,), dtype=np.float32), 0.0, True, False, {"success": True}

    env = AddTaskSuccessInfo(SuccessEnv())
    _, _, _, _, info = env.step(env.action_space.sample())

    assert info["task_success"] == 1


@pytest.fixture()
def compose_cfg():
    GlobalHydra.instance().clear()
    initialize(config_path="../../../robobase/cfgs")
    method = ["method=" + "iql_drqv2"]
    cfg = compose(
        config_name="robobase_config",
        overrides=method
        + [
            "pixels=false",
            "env=d4rl/hopper",
            "save_snapshot=true",
            "snapshot_every_n=1",
        ],
    )
    return cfg


@pytest.mark.parametrize(
    "num_demos, desired_num_demos", [(float("inf"), 2187), (0, 0), (100, 100)]
)
def test_collect_demo(num_demos, desired_num_demos, compose_cfg):
    _skip_without_old_d4rl()
    factory = D4RLEnvFactory()
    factory.collect_or_fetch_demos(compose_cfg, num_demos)

    assert len(factory._raw_demos) == desired_num_demos


class _FakeMinariDataset:
    def __init__(self):
        self.spec = SimpleNamespace(
            observation_space=gym.spaces.Box(
                -np.inf, np.inf, shape=(3,), dtype=np.float64
            ),
            action_space=gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
        )

    def iterate_episodes(self):
        yield SimpleNamespace(
            observations=np.asarray(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                dtype=np.float64,
            ),
            actions=np.asarray([[0.1, -0.1], [0.2, -0.2]], dtype=np.float32),
            rewards=np.asarray([1.0, 2.0], dtype=np.float32),
            terminations=np.asarray([False, True]),
            truncations=np.asarray([False, False]),
        )


class _FakeMultiEpisodeMinariDataset(_FakeMinariDataset):
    def iterate_episodes(self):
        for episode_return in (1.0, 3.0, 2.0):
            yield SimpleNamespace(
                observations=np.zeros((2, 3), dtype=np.float32),
                actions=np.zeros((1, 2), dtype=np.float32),
                rewards=np.asarray([episode_return], dtype=np.float32),
                terminations=np.asarray([True]),
                truncations=np.asarray([False]),
            )


class _FakeMinari:
    def load_dataset(self, dataset_id):
        assert dataset_id == "D4RL/pen/expert-v2"
        return _FakeMinariDataset()


@pytest.fixture()
def fake_minari(monkeypatch):
    monkeypatch.setattr(d4rl_env, "minari", _FakeMinari())


class _CountingEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(3,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        del action
        self.steps += 1
        return np.full(3, self.steps, dtype=np.float32), 1.0, False, False, {}


def test_get_minari_trajectory_dataset(fake_minari):
    demos, raw_dataset = get_minari_traj_dataset("D4RL/pen/expert-v2")

    assert isinstance(raw_dataset, _FakeMinariDataset)
    assert len(demos) == 1
    demo = demos[0]
    assert len(demo[0]) == 2
    assert demo[0][0].dtype == np.float32
    assert demo[1][-1]["demo_action"].shape == (2,)
    assert demo[2][2] is True
    assert demo[2][3] is False


def test_get_minari_trajectory_dataset_sorts_highest_return_first(monkeypatch):
    dataset = _FakeMultiEpisodeMinariDataset()
    monkeypatch.setattr(
        d4rl_env,
        "minari",
        SimpleNamespace(load_dataset=lambda dataset_id: dataset),
    )

    demos, _ = get_minari_traj_dataset("D4RL/pen/expert-v2")

    assert [d4rl_env._episode_return(demo) for demo in demos] == [3.0, 2.0, 1.0]


def test_get_minari_trajectory_dataset_streams_only_top_k(monkeypatch):
    dataset = _FakeMultiEpisodeMinariDataset()
    monkeypatch.setattr(
        d4rl_env,
        "minari",
        SimpleNamespace(load_dataset=lambda dataset_id: dataset),
    )

    demos, _ = get_minari_traj_dataset("D4RL/pen/expert-v2", num_demos=2)

    assert [d4rl_env._episode_return(demo) for demo in demos] == [3.0, 2.0]


def test_get_minari_trajectory_dataset_selects_first_episodes(monkeypatch):
    dataset = _FakeMultiEpisodeMinariDataset()
    monkeypatch.setattr(
        d4rl_env,
        "minari",
        SimpleNamespace(load_dataset=lambda dataset_id: dataset),
    )

    demos, _ = get_minari_traj_dataset(
        "D4RL/pen/expert-v2", num_demos=2, selection="first"
    )

    assert [d4rl_env._episode_return(demo) for demo in demos] == [1.0, 3.0]


def test_get_minari_trajectory_dataset_selects_exact_first_transitions(monkeypatch):
    dataset = _FakeMinariDataset()
    monkeypatch.setattr(
        d4rl_env,
        "minari",
        SimpleNamespace(load_dataset=lambda dataset_id: dataset),
    )

    demos, _ = get_minari_traj_dataset(
        "D4RL/pen/expert-v2", selection="first", num_transitions=1
    )

    assert sum(len(demo) - 1 for demo in demos) == 1
    assert np.allclose(demos[0][1][-1]["demo_action"], [0.1, -0.1])
    assert demos[0][-1][2] is False
    assert demos[0][-1][3] is True


def test_minari_get_spaces_uses_dataset_metadata(fake_minari):
    GlobalHydra.instance().clear()
    with initialize(config_path="../../../robobase/cfgs", version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=bc_state_robomimic",
                "env=d4rl/pen",
                "num_pretrain_steps=0",
            ],
        )

    obs_space, action_space = D4RLEnvFactory().get_spaces(cfg)

    assert obs_space["low_dim_state"].shape == (1, 3)
    assert obs_space["low_dim_state"].dtype == np.float32
    assert action_space.shape == (1, 2)


def test_minari_collect_zero_demos_does_not_load_dataset(monkeypatch):
    class FailingMinari:
        def load_dataset(self, dataset_id):
            raise AssertionError("dataset should not be loaded")

    monkeypatch.setattr(d4rl_env, "minari", FailingMinari())
    GlobalHydra.instance().clear()
    with initialize(config_path="../../../robobase/cfgs", version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=bc_state_robomimic",
                "env=d4rl/pen",
                "num_pretrain_steps=0",
            ],
        )

    factory = D4RLEnvFactory()
    factory.collect_or_fetch_demos(cfg, 0)

    assert factory._raw_demos == []


def test_minari_collect_accepts_string_inf(fake_minari):
    GlobalHydra.instance().clear()
    with initialize(config_path="../../../robobase/cfgs", version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "launch=bc_state_robomimic",
                "env=d4rl/pen",
                "num_pretrain_steps=0",
            ],
        )

    factory = D4RLEnvFactory()
    factory.collect_or_fetch_demos(cfg, ".inf")

    assert len(factory._raw_demos) == 1


def test_d4rl_demo_env_keeps_raw_demo_actions_for_replay(fake_minari):
    GlobalHydra.instance().clear()
    with initialize(config_path="../../../robobase/cfgs", version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "method=flow_matching",
                "env=d4rl/pen",
                "action_sequence=20",
                "execution_length=20",
                "num_train_envs=0",
            ],
        )

    dataset = _FakeMinariDataset()
    factory = D4RLEnvFactory()
    normal_env = factory._wrap_env(
        d4rl_env.D4RLPlaceholderEnv(
            dataset.spec.observation_space,
            dataset.spec.action_space,
        ),
        cfg,
    )
    assert normal_env.action_space.shape == (20, 2)

    raw_demo = [
        [np.zeros(3, dtype=np.float32), {"demo": 1}],
        [
            np.ones(3, dtype=np.float32),
            1.0,
            True,
            False,
            {"demo_action": np.asarray([0.1, -0.1], dtype=np.float32), "demo": 1},
        ],
    ]
    demo_env = factory._wrap_env(
        DemoEnv(
            [raw_demo],
            dataset.spec.action_space,
            dataset.spec.observation_space,
        ),
        cfg,
        demo_env=True,
    )

    demo_env.reset()
    _, _, _, _, info = demo_env.step(dataset.spec.action_space.sample())

    assert info["demo_action"].shape == (2,)


def test_d4rl_uses_receding_horizon_for_shorter_execution_length(fake_minari):
    GlobalHydra.instance().clear()
    with initialize(config_path="../../../robobase/cfgs", version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "method=flow_matching",
                "env=d4rl/pen",
                "action_sequence=20",
                "execution_length=1",
                "temporal_ensemble=false",
                "num_train_envs=0",
            ],
        )

    env = D4RLEnvFactory()._wrap_env(_CountingEnv(), cfg)

    assert _find_wrapper(env, RecedingHorizonControl) is not None
    env.reset()
    _, reward, termination, truncation, info = env.step(env.action_space.sample())

    assert reward == 1.0
    assert not termination
    assert not truncation
    assert info["action_sequence_mask"].tolist() == [1] + [0] * 19
