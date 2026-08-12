import random
import re
import time
import warnings
import selectors
import sys

from gymnasium.spaces import Box
from omegaconf import DictConfig
import numpy as np
from typing import List, Callable
from scipy.spatial.transform import Rotation as R

from robobase.envs.env import Demo, DemoEnv
from robobase.replay_buffer.replay_buffer import ReplayBuffer


def discounted_episode_returns(rewards, gamma: float) -> np.ndarray:
    """Compute discounted reward-to-go for one completed episode."""

    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    returns = np.empty_like(rewards)
    running_return = np.float32(0.0)
    gamma = np.float32(gamma)
    for index in range(len(rewards) - 1, -1, -1):
        running_return = rewards[index] + gamma * running_return
        returns[index] = running_return
    return returns


class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def check_for_kill_input(timeout: int = 0.0001):
    sel = selectors.DefaultSelector()
    try:
        # pytest will throw value error on this line
        sel.register(sys.stdin, selectors.EVENT_READ)
    except Exception:
        return False
    events = sel.select(timeout)
    if events:
        key, _ = events[0]
        return key.fileobj.readline().rstrip("\n").lower() == "q"
    else:
        return False


def set_seed_everywhere(seed):
    np.random.seed(seed)
    random.seed(seed)


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None or self._every == 0:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+)\)", schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r"step_linear\((.+),(.+),(.+),(.+),(.+)\)", schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


class DemoStep(dict):
    """A step of a demo which holds state along with joint and gripper positions."""

    def __init__(
        self,
        joint_positions: np.ndarray,
        gripper_open: float,
        state: dict,
        gripper_matrix: np.array = None,
        misc: dict = {},
    ):
        """Init.

        Args:
            joint_positions (np.ndarray): joint positions excluding the gripper.
            gripper_open (float): value between 0.0 and 1.0 representing open and
                closed respectively.
            state (dict): state observations expected as inputs to the model.
        """
        super().__init__(**state)
        self.joint_positions = joint_positions
        self.gripper_open = gripper_open
        self.gripper_matrix = gripper_matrix
        self.misc = misc


def observations_to_action_with_onehot_gripper(
    current_observation: DemoStep,
    next_observation: DemoStep,
    action_space: Box,
):
    """Calculates the action linking two sequential observations.

    Args:
        current_observation (DemoStep): the observation made before the action.
        next_observation (DemoStep): the observation made after the action.
        action_space (Box): the action space of the unwrapped env.

    Returns:
        np.ndarray: action taken at current observation. Returns None if action
            outside action_space.
    """
    action = np.concatenate(
        [
            (
                next_observation.misc["joint_position_action"][:-1]
                - current_observation.joint_positions
                if "joint_position_action" in next_observation.misc
                else next_observation.joint_positions
                - current_observation.joint_positions
            ),
            [1.0 if next_observation.gripper_open == 1 else 0.0],
        ]
    ).astype(np.float32)
    if np.any(action[:-1] > action_space.high[:-1]) or np.any(
        action[:-1] < action_space.low[:-1]
    ):
        return None
    return action


def observations_to_action_with_onehot_gripper_nbp(
    current_observation: DemoStep,
    next_observation: DemoStep,
    action_space: Box,
):
    """Calculates the action linking two sequential observations.

    Args:
        current_observation (DemoStep): the observation made before the action.
        next_observation (DemoStep): the observation made after the action.
        action_space (Box): the action space of the unwrapped env.

    Returns:
        np.ndarray: action taken at current observation. Returns None if action
            outside action_space.
    """

    action_trans = next_observation.gripper_matrix[:3, 3]

    rot = R.from_matrix(next_observation.gripper_matrix[:3, :3])
    action_orien = rot.as_quat(
        canonical=True
    )  # Enforces w component always positive and unit vector

    action_gripper = [1.0 if next_observation.gripper_open == 1 else 0.0]
    action = np.concatenate(
        [
            action_trans,
            action_orien,
            action_gripper,
        ]
    )

    if np.any(action[:-1] > action_space.high[:-1]) or np.any(
        action[:-1] < action_space.low[:-1]
    ):
        warnings.warn(
            "Action outside action space.",
            UserWarning,
        )
        return None
    return action


def observations_to_timesteps(
    demo: List[DemoStep],
    action_space: Box,
    skipping: bool = True,
    obs_to_act_func: Callable[
        [DemoStep, DemoStep, Box], np.ndarray
    ] = observations_to_action_with_onehot_gripper,
):
    """Converts demo steps into timesteps.

    Args:
        demo (List[DemoStep]): an episode.
        action_space (Box): the actions space of the unwrapped env.
        skipping (bool): option to augment demonstration data through observations.
        obs_to_act_func: function to call for determining action.

    Returns:
        List[List[Tuple]]: a list of timestep demonstrations. Each demonstration
            ends with the following format where a_t is stored in info_{t+1}:

                [(s_0), (s_1, r_1, term_1, trunc_1, info_1), ...]
    """
    loaded_demos = []
    skip = 1
    first_step = demo[0]
    # enter loop until skipping more observations goes outside action_space
    while True:
        info = {"demo": 1}
        # add first observation to demo_timesteps following format defined above
        demo_timesteps = [(first_step, info)]
        i = 0
        while i < len(demo[:-1]):
            demo_step = demo[i]
            r = 0.0
            done = False
            # find the next observation
            for j in range(1, 1 + skip):
                next_demo_step = demo[i + j]
                if (i + j) >= (len(demo) - 1):
                    r = 1.0
                    done = True
                    break
                if next_demo_step.gripper_open != demo_step.gripper_open:
                    break
            i += j
            # calculate action
            action = obs_to_act_func(demo_step, next_demo_step, action_space)
            # wipe demo_timesteps if action outside action_space
            if action is None:
                demo_timesteps = []
                break
            # add action into info to be extracted later
            info = {"demo_action": action, "demo": 1}
            demo_timesteps.append(
                (
                    next_demo_step,
                    r,
                    done,
                    False,
                    info,
                )
            )
        if len(demo_timesteps) == 0:
            break
        loaded_demos.append(Demo(demo_timesteps))
        if skipping:
            skip += 1
        else:
            break
    return loaded_demos


def rescale_demo_actions(rescale_fn: Callable, demos: List[Demo], cfg: DictConfig):
    """Rescale actions in demonstrations to [-1, 1] Tanh space.
    This is because RoboBase assumes everything to be in [-1, 1] space.

    Args:
        rescale_fn: callable that takes info containing demo action and cfg and
            outputs the rescaled action
        demos: list of demo episodes whose actions are raw, i.e., not scaled
        cfg: Configs

    Returns:
        List[Demo]: list of demo episodes whose actions are rescaled
    """
    for demo in demos:
        for step in demo:
            *_, info = step
            if "demo_action" in info:
                # Rescale demo actions
                info["demo_action"] = rescale_fn(info, cfg)
    return demos


def add_demo_to_replay_buffer(wrapped_env: DemoEnv, replay_buffer: ReplayBuffer):
    """Loads demos into replay buffer by passing observations through wrappers.

    CYCLING THROUGH DEMOS IS HANDLED BY WRAPPED ENV.

    Args:
        wrapped_env: the fully wrapped environment.
        replay_buffer: replay buffer to be loaded.
    """
    is_sequential = replay_buffer.sequential
    ep = []

    # Extract demonstration episode in replay buffer transitions
    obs, info = wrapped_env.reset()
    fake_action = wrapped_env.action_space.sample()
    term, trunc = False, False
    while not (term or trunc):
        next_obs, rew, term, trunc, next_info = wrapped_env.step(fake_action)
        # Demo steps can be loaded into both the online and protected demo
        # buffers. Do not consume the shared source dict on the first pass.
        next_info = dict(next_info)
        action = next_info.pop("demo_action")
        action_space = wrapped_env.action_space
        if np.all(np.isfinite(action_space.low)) and np.all(
            np.isfinite(action_space.high)
        ):
            assert np.all(action <= action_space.high)
            assert np.all(action >= action_space.low)
        ep.append([obs, action, rew, term, trunc, info, next_info])
        obs = next_obs
        info = next_info
    final_obs, _ = obs, info

    store_mc_return = "mc_return" in replay_buffer.extra_replay_elements.keys()
    store_structured_explore = (
        "structured_explore" in replay_buffer.extra_replay_elements.keys()
    )
    mc_returns = discounted_episode_returns(
        [transition[2] for transition in ep],
        getattr(replay_buffer, "_gamma", 1.0),
    )
    for index, (obs, action, rew, term, trunc, info, _) in enumerate(ep):
        extra = {"demo": info["demo"]}
        if "explored" in replay_buffer.extra_replay_elements.keys():
            extra["explored"] = np.uint8(0)
        if store_mc_return:
            extra["mc_return"] = mc_returns[index]
        if store_structured_explore:
            extra["structured_explore"] = np.uint8(0)
            if (
                "structured_explore_start"
                in replay_buffer.extra_replay_elements.keys()
            ):
                extra["structured_explore_start"] = np.uint8(0)
            if (
                "structured_explore_dimension"
                in replay_buffer.extra_replay_elements.keys()
            ):
                extra["structured_explore_dimension"] = np.int16(-1)
            if (
                "structured_explore_delta"
                in replay_buffer.extra_replay_elements.keys()
            ):
                extra["structured_explore_delta"] = np.float32(0.0)
            if (
                "structured_explore_assignment_prob"
                in replay_buffer.extra_replay_elements.keys()
            ):
                extra["structured_explore_assignment_prob"] = np.float32(1.0)
        replay_buffer.add(obs, action, rew, term, trunc, **extra)

    if not is_sequential:
        replay_buffer.add_final(final_obs)


def merge_replay_demo_iter(replay_iter, demo_replay_iter):
    return iter(DemoMergedIterator(replay_iter, demo_replay_iter))


class DemoMergedIterator:
    def __init__(self, replay_iter, demo_replay_iter):
        self.replay_iter = replay_iter
        self.demo_replay_iter = demo_replay_iter
        self._is_safe = False

    def __iter__(self):
        return self

    def _check_keys(self, batch, demo_batch):
        assert set(batch.keys()) == set(demo_batch.keys()), (
            f"Keys in demo batch are different: {batch.keys()}, {demo_batch.keys()}"
        )

    def _ones_like(self, value):
        return np.ones_like(value)

    def _cat(self, lhs, rhs):
        return np.concatenate([lhs, rhs], axis=0)

    def __next__(self):
        batch = next(self.replay_iter)
        demo_batch = next(self.demo_replay_iter)
        if not self._is_safe:
            self._check_keys(batch, demo_batch)
            self._is_safe = True
        # Override demo to be 1 for demo_batch
        demo_batch["demo"] = self._ones_like(demo_batch["demo"])
        return {k: self._cat(batch[k], demo_batch[k]) for k in batch.keys()}

    def close(self):
        for iterator in (self.replay_iter, self.demo_replay_iter):
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
