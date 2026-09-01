"""Exact in-process branch snapshots for wrapped BiGym environments.

MuJoCo's ``Physics.get_state`` does not include actuator controls, and RoboBase
wrappers keep additional frame/action history.  A counterfactual value audit
must restore all of them before comparing candidate actions from one state.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

_PHYSICS_ARRAYS = (
    "ctrl",
    "qacc_warmstart",
    "qfrc_applied",
    "xfrc_applied",
    "mocap_pos",
    "mocap_quat",
    "eq_active",
    "userdata",
)

_WRAPPER_STATE_ATTRIBUTES = (
    "_elapsed_steps",  # gymnasium TimeLimit
    "frames",  # RoboBase FrameStack
    "_action_history",  # RecedingHorizonControl
    "_action_history_valid",
    "_cur_step",
    "_t",  # OnehotTime
)

# BiGym's animated floating-base legs update model body quaternions at every
# control step. Task reset also moves static bodies (for example plate racks),
# so qpos/qvel alone is not a complete branch state.
_MODEL_ARRAYS = (
    "body_pos",
    "body_quat",
    "geom_pos",
    "geom_quat",
    "site_pos",
    "site_quat",
    "cam_pos",
    "cam_quat",
)


@dataclass(frozen=True)
class BiGymBranchState:
    """State needed to restart deterministic rollouts in the same env."""

    physics_state: np.ndarray
    integration_state: np.ndarray | None
    physics_time: float
    physics_arrays: dict[str, np.ndarray]
    model_arrays: dict[str, np.ndarray]
    raw_action: np.ndarray
    floating_base_accumulated_actions: np.ndarray
    floating_base_last_action: np.ndarray
    wrapper_types: tuple[str, ...]
    wrapper_values: tuple[dict[str, Any], ...]
    env_health_values: dict[str, Any]
    numpy_random_state: tuple[Any, ...]


def _wrapper_chain(env) -> list[Any]:
    chain = []
    current = env
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        if "env" not in vars(current):
            break
        current = vars(current)["env"]
    return chain


def _copy_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _copy_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        copied = [_copy_value(item) for item in value]
        return type(value)(copied)
    return copy.deepcopy(value)


def capture_bigym_branch_state(env) -> BiGymBranchState:
    """Capture MuJoCo, controller, wrapper, and NumPy RNG state."""

    wrappers = _wrapper_chain(env)
    raw_env = env.unwrapped
    if not hasattr(raw_env, "mojo") or not hasattr(raw_env, "robot"):
        raise TypeError("capture_bigym_branch_state requires a BiGym environment")

    data = raw_env.mojo.data
    integration_state = None
    try:
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        integration_state = np.empty(
            mujoco.mj_stateSize(raw_env.mojo.model, state_spec),
            dtype=np.float64,
        )
        mujoco.mj_getState(
            raw_env.mojo.model,
            data,
            integration_state,
            state_spec,
        )
    except TypeError:
        # Lightweight unit-test fakes are not native MjModel/MjData objects.
        integration_state = None
    # get_state() is intentionally compact and excludes controls, applied
    # forces, mocap/equality inputs, and the solver warm start.
    physics_arrays = {
        name: np.asarray(getattr(data, name)).copy()
        for name in _PHYSICS_ARRAYS
        if hasattr(data, name)
    }
    model = raw_env.mojo.model
    model_arrays = {
        name: np.asarray(getattr(model, name)).copy()
        for name in _MODEL_ARRAYS
        if hasattr(model, name)
    }
    wrapper_values = tuple(
        {
            name: _copy_value(getattr(wrapper, name))
            for name in _WRAPPER_STATE_ATTRIBUTES
            if name in vars(wrapper)
        }
        for wrapper in wrappers
    )
    floating_base = raw_env.robot.floating_base
    env_health = getattr(raw_env, "_env_health", None)
    return BiGymBranchState(
        physics_state=raw_env.mojo.physics.get_state().copy(),
        integration_state=integration_state,
        physics_time=float(data.time),
        physics_arrays=physics_arrays,
        model_arrays=model_arrays,
        raw_action=np.asarray(raw_env._action).copy(),
        floating_base_accumulated_actions=np.asarray(
            floating_base._accumulated_actions
        ).copy(),
        floating_base_last_action=np.asarray(floating_base._last_action).copy(),
        wrapper_types=tuple(type(wrapper).__qualname__ for wrapper in wrappers),
        wrapper_values=wrapper_values,
        env_health_values=(
            {
                "_current_error": getattr(env_health, "_current_error", None),
                "_consecutive_errors": list(
                    getattr(env_health, "_consecutive_errors", [])
                ),
            }
            if env_health is not None
            else {}
        ),
        numpy_random_state=copy.deepcopy(np.random.get_state()),
    )


def restore_bigym_branch_state(env, state: BiGymBranchState) -> None:
    """Restore a state captured from this same wrapped environment."""

    wrappers = _wrapper_chain(env)
    wrapper_types = tuple(type(wrapper).__qualname__ for wrapper in wrappers)
    if wrapper_types != state.wrapper_types:
        raise ValueError(
            "BiGym wrapper chain changed between capture and restore: "
            f"{state.wrapper_types} != {wrapper_types}"
        )

    raw_env = env.unwrapped
    for name, value in state.model_arrays.items():
        getattr(raw_env.mojo.model, name)[...] = value
    if state.integration_state is None:
        raw_env.mojo.physics.set_state(state.physics_state)
        raw_env.mojo.data.time = state.physics_time
    else:
        mujoco.mj_setState(
            raw_env.mojo.model,
            raw_env.mojo.data,
            state.integration_state,
            mujoco.mjtState.mjSTATE_INTEGRATION,
        )
    # Restore mocap/equality inputs before forward so derived kinematics match
    # the branch point. dm_control's forward callbacks can rewrite actuator
    # controls, so reapply every external integration input afterwards too.
    for name, value in state.physics_arrays.items():
        getattr(raw_env.mojo.data, name)[...] = value
    raw_env.mojo.physics.forward()
    if state.integration_state is not None:
        # ``forward`` recomputes derived quantities but may rewrite warm-start
        # and control inputs.  Restore the complete integration vector again;
        # qpos/qvel are unchanged, so the derived geometry remains valid.
        mujoco.mj_setState(
            raw_env.mojo.model,
            raw_env.mojo.data,
            state.integration_state,
            mujoco.mjtState.mjSTATE_INTEGRATION,
        )
    for name, value in state.physics_arrays.items():
        getattr(raw_env.mojo.data, name)[...] = value
    # BiGym assigns ``self._action = action`` and can therefore alias the
    # caller's candidate array. Rebinding avoids corrupting that array while
    # abandoning a counterfactual branch.
    raw_env._action = state.raw_action.copy()
    floating_base = raw_env.robot.floating_base
    np.copyto(
        floating_base._accumulated_actions,
        state.floating_base_accumulated_actions,
    )
    np.copyto(floating_base._last_action, state.floating_base_last_action)

    for wrapper, values in zip(wrappers, state.wrapper_values):
        for name, value in values.items():
            setattr(wrapper, name, _copy_value(value))

    # Cached reward/success predicates refer to the abandoned branch.
    raw_env._step_cache.clean()
    env_health = getattr(raw_env, "_env_health", None)
    if env_health is not None:
        for name, value in state.env_health_values.items():
            setattr(env_health, name, list(value) if isinstance(value, list) else value)
    np.random.set_state(copy.deepcopy(state.numpy_random_state))
