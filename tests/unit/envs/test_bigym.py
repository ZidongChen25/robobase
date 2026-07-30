import numpy as np
import pytest
from omegaconf import OmegaConf

pytest.importorskip("bigym")

from robobase.envs.bigym import (
    AddBiGymLanguageTokens,
    BIGYM_TASK_DESCRIPTIONS,
    BiGymEnvFactory,
    bigym_task_description,
)
from robobase.envs.utils.bigym_utils import TASK_MAP
from robobase.envs.wrappers import RecedingHorizonControl
from tests.unit.wrappers.utils import DummyEnv


def _find_wrapper(env, wrapper_type):
    current = env
    while current is not None:
        if isinstance(current, wrapper_type):
            return current
        current = getattr(current, "env", None)
    return None


def test_bigym_eval_receding_horizon_uses_action_execution_start():
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

    env = factory._wrap_env(DummyEnv(episode_len=50), cfg, demo_env=False, train=False)
    receding = _find_wrapper(env, RecedingHorizonControl)

    assert receding is not None
    assert receding._execution_start == 2


def test_bigym_language_wrapper_can_emit_clip_features(monkeypatch):
    seen_descriptions = []
    monkeypatch.setattr(
        "robobase.envs.bigym.clip_tokenize_text",
        lambda description: (
            seen_descriptions.append(description)
            or np.full((1, 77), 7, dtype=np.int32)
        ),
    )
    monkeypatch.setattr(
        "robobase.envs.bigym.clip_text_feature_array",
        lambda description, device="cpu": np.full((1, 512), 0.25, dtype=np.float32),
    )

    env = AddBiGymLanguageTokens(
        DummyEnv(episode_len=5),
        "move_plate",
        lang_feature_source="clip",
        description="reach the target",
    )
    obs = env.observation({})

    assert env.observation_space["lang_tokens"].shape == (1, 77)
    assert env.observation_space["lang_features"].shape == (1, 512)
    np.testing.assert_array_equal(obs["lang_tokens"], np.full((1, 77), 7))
    np.testing.assert_allclose(obs["lang_features"], np.full((1, 512), 0.25))
    assert seen_descriptions == ["reach the target"]


def test_bigym_task_description_humanizes_unmapped_tasks():
    unmapped_tasks = sorted(set(TASK_MAP) - set(BIGYM_TASK_DESCRIPTIONS))
    fallback_descriptions = {
        bigym_task_description(task_name) for task_name in unmapped_tasks
    }

    assert bigym_task_description("flip_cutlery") == "Flip cutlery."
    assert bigym_task_description("stack_blocks") == "Stack blocks."
    assert len(fallback_descriptions) == len(unmapped_tasks)
