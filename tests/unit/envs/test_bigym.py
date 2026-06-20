import numpy as np
import pytest
from omegaconf import OmegaConf

pytest.importorskip("bigym")

from robobase.envs.bigym import BiGymEnvFactory
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
