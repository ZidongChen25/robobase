from types import SimpleNamespace

import numpy as np

from robobase.envs.bigym_branch_state import (
    capture_bigym_branch_state,
    restore_bigym_branch_state,
)


class _Physics:
    def __init__(self, data):
        self.data = data

    def get_state(self):
        return self.data.qpos.copy()

    def set_state(self, state):
        self.data.qpos[...] = state

    def forward(self):
        pass


class _StepCache:
    def __init__(self):
        self.cleaned = False

    def clean(self):
        self.cleaned = True


class _RawEnv:
    def __init__(self):
        data = SimpleNamespace(
            time=1.5,
            qpos=np.array([1.0, 2.0]),
            ctrl=np.array([3.0]),
            qacc_warmstart=np.array([4.0]),
            qfrc_applied=np.zeros(2),
            xfrc_applied=np.zeros((1, 6)),
            mocap_pos=np.zeros((1, 3)),
            mocap_quat=np.zeros((1, 4)),
            eq_active=np.ones(1, dtype=np.int32),
            userdata=np.zeros(1),
        )
        model = SimpleNamespace(
            body_pos=np.zeros((1, 3)),
            body_quat=np.ones((1, 4)),
        )
        physics = _Physics(data)
        self.mojo = SimpleNamespace(data=data, model=model, physics=physics)
        floating_base = SimpleNamespace(
            _accumulated_actions=np.array([5.0]),
            _last_action=np.array([6.0]),
        )
        self.robot = SimpleNamespace(floating_base=floating_base)
        self._action = np.array([7.0])
        self._step_cache = _StepCache()

    @property
    def unwrapped(self):
        return self


class _Wrapper:
    def __init__(self, env):
        self.env = env
        self._elapsed_steps = 8
        self.frames = {"state": np.array([[9.0]])}

    @property
    def unwrapped(self):
        return self.env.unwrapped


def test_restore_does_not_mutate_abandoned_candidate_alias():
    raw = _RawEnv()
    env = _Wrapper(raw)
    branch = capture_bigym_branch_state(env)

    candidate = np.array([99.0])
    raw._action = candidate  # BiGym step aliases the caller's array this way.
    raw.mojo.data.qpos[...] = -1
    raw.mojo.data.ctrl[...] = -2
    raw.robot.floating_base._last_action[...] = -3
    env._elapsed_steps = 100
    env.frames["state"][...] = -4

    restore_bigym_branch_state(env, branch)

    np.testing.assert_array_equal(candidate, [99.0])
    np.testing.assert_array_equal(raw._action, [7.0])
    assert not np.shares_memory(raw._action, candidate)
    np.testing.assert_array_equal(raw.mojo.data.qpos, [1.0, 2.0])
    np.testing.assert_array_equal(raw.mojo.data.ctrl, [3.0])
    np.testing.assert_array_equal(raw.robot.floating_base._last_action, [6.0])
    assert env._elapsed_steps == 8
    np.testing.assert_array_equal(env.frames["state"], [[9.0]])
    assert raw._step_cache.cleaned
