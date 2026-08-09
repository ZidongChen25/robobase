import numpy as np
import pytest
from omegaconf import OmegaConf

pytest.importorskip("bigym")

from robobase.envs.bigym import (
    AddBiGymCameraConditioning,
    AddBiGymLanguageTokens,
    BIGYM_TASK_DESCRIPTIONS,
    BiGymEnvFactory,
    bigym_task_description,
)
from robobase.envs.utils.bigym_utils import TASK_MAP
from robobase.envs.wrappers import (
    ConcatDim,
    ObservationDelay,
    RawProprioDropout,
    RecedingHorizonControl,
)
from tests.unit.wrappers.utils import DummyEnv


def _find_wrapper(env, wrapper_type):
    current = env
    while current is not None:
        if isinstance(current, wrapper_type):
            return current
        current = getattr(current, "env", None)
    return None


@pytest.mark.parametrize("train", [False, True])
def test_bigym_receding_horizon_uses_action_execution_start(train):
    factory = BiGymEnvFactory()
    factory._action_stats = {
        "mean": np.zeros(2, dtype=np.float32),
        "std": np.ones(2, dtype=np.float32),
        "min": -np.ones(2, dtype=np.float32),
        "max": np.ones(2, dtype=np.float32),
    }
    factory._obs_stats = None

    cfg = OmegaConf.create(
        {
            "demos": 1,
            "action_repeat": 1,
            "use_standardization": False,
            "min_max_margin": 0.0,
            "norm_obs": False,
            "obs_norm_type": "standardization",
            "use_onehot_time_and_no_bootstrap": False,
            "frame_stack": 1,
            "action_sequence": 5,
            "execution_length": 2,
            "action_execution_start": 2,
            "temporal_ensemble": False,
            "temporal_ensemble_gain": 0.01,
            "method": {"use_lang_cond": False},
            "env": {
                "episode_length": 50,
                "demo_down_sample_rate": 1,
                "task_name": "move_plate",
            },
        }
    )

    env = factory._wrap_env(DummyEnv(episode_len=50), cfg, demo_env=False, train=train)
    receding = _find_wrapper(env, RecedingHorizonControl)

    assert receding is not None
    assert receding._execution_start == 2


@pytest.mark.parametrize("demo_env", [False, True])
def test_bigym_obs_delay_wraps_both_live_and_demo_envs(demo_env):
    # The demo env must be delayed too: demos imported through it are what the
    # non-lazy replay stores, so the (o_{t-h}, a_t) pairing is baked in there.
    factory = BiGymEnvFactory()
    factory._action_stats = {
        "mean": np.zeros(2, dtype=np.float32),
        "std": np.ones(2, dtype=np.float32),
        "min": -np.ones(2, dtype=np.float32),
        "max": np.ones(2, dtype=np.float32),
    }
    factory._obs_stats = None

    cfg = OmegaConf.create(
        {
            "demos": 1,
            "action_repeat": 1,
            "use_standardization": False,
            "min_max_margin": 0.0,
            "norm_obs": False,
            "obs_norm_type": "standardization",
            "use_onehot_time_and_no_bootstrap": False,
            "frame_stack": 1,
            "action_sequence": 1,
            "execution_length": 1,
            "obs_delay": 3,
            "temporal_ensemble": False,
            "temporal_ensemble_gain": 0.01,
            "method": {"use_lang_cond": False},
            "env": {
                "episode_length": 50,
                "demo_down_sample_rate": 1,
                "task_name": "move_plate",
            },
        }
    )

    env = factory._wrap_env(DummyEnv(episode_len=50), cfg, demo_env=demo_env)
    delay = _find_wrapper(env, ObservationDelay)

    assert delay is not None
    assert delay.delay == 3

    cfg.obs_delay = 0
    undelayed = factory._wrap_env(DummyEnv(episode_len=50), cfg, demo_env=demo_env)

    assert _find_wrapper(undelayed, ObservationDelay) is None


def test_bigym_language_wrapper_emits_jax_tokens_only():
    env = AddBiGymLanguageTokens(
        DummyEnv(episode_len=5),
        "move_plate",
        lang_feature_source="tokens",
        description="reach the target",
    )
    obs = env.observation({})

    assert env.observation_space["lang_tokens"].shape == (1, 77)
    assert "lang_features" not in env.observation_space
    assert obs["lang_tokens"].dtype == np.int32
    assert np.any(obs["lang_tokens"] != 0)


def test_bigym_language_wrapper_emits_precomputed_features(tmp_path):
    path = tmp_path / "language.npy"
    expected = np.linspace(-0.5, 0.5, 8, dtype=np.float32)[None, :]
    np.save(path, expected)
    env = AddBiGymLanguageTokens(
        DummyEnv(episode_len=5),
        "move_plate",
        lang_feature_source="precomputed",
        lang_feature_path=path,
        lang_feature_dim=8,
        description="reach the target",
    )

    obs = env.observation({})

    assert env.observation_space["lang_features"].shape == (1, 8)
    assert obs["lang_features"].dtype == np.float32
    np.testing.assert_array_equal(obs["lang_features"], expected)


def test_bigym_camera_wrapper_preserves_cached_demo_parameters():
    env = AddBiGymCameraConditioning(
        DummyEnv(episode_len=5),
        cameras=["head"],
        image_size=(8, 8),
    )
    intrinsic = np.asarray(
        [[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    c2w = np.eye(4, dtype=np.float32)

    obs = env.observation({"camera_intrinsic_head": intrinsic, "camera_c2w_head": c2w})

    np.testing.assert_array_equal(obs["camera_intrinsic_head"], intrinsic)
    np.testing.assert_array_equal(obs["camera_c2w_head"], c2w)


def test_bigym_camera_wrapper_rejects_demo_without_cached_parameters():
    env = AddBiGymCameraConditioning(
        DummyEnv(episode_len=5),
        cameras=["head"],
        image_size=(8, 8),
    )

    with pytest.raises(ValueError, match="--include-camera-params"):
        env.observation({})


def test_raw_proprio_dropout_runs_before_standardization():
    raw_env = RawProprioDropout(
        DummyEnv(episode_len=5),
        keys=("obs0",),
        probability=1.0,
    )
    env = ConcatDim(
        raw_env,
        shape_length=1,
        dim=-1,
        new_name="low_dim_state",
        norm_obs=True,
        obs_stats={
            "mean": {"obs0": np.full((100,), 2.0, dtype=np.float32)},
            "std": {"obs0": np.full((100,), 4.0, dtype=np.float32)},
            "min": {},
            "max": {},
        },
        keys_to_ignore=["obs1"],
    )
    observation, _ = raw_env.env.reset()

    actual = env.observation(raw_env.observation(observation))["low_dim_state"]

    np.testing.assert_allclose(actual, -0.5)


def test_bigym_language_wrapper_rejects_torch_clip_source():
    with pytest.raises(ValueError, match="JAX-only"):
        AddBiGymLanguageTokens(
            DummyEnv(episode_len=5),
            "move_plate",
            lang_feature_source="clip",
        )


def test_bigym_task_description_humanizes_unmapped_tasks():
    unmapped_tasks = sorted(set(TASK_MAP) - set(BIGYM_TASK_DESCRIPTIONS))
    fallback_descriptions = {
        bigym_task_description(task_name) for task_name in unmapped_tasks
    }

    assert bigym_task_description("flip_cutlery") == "Flip cutlery."
    assert bigym_task_description("stack_blocks") == "Stack blocks."
    assert len(fallback_descriptions) == len(unmapped_tasks)
