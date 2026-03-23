from functools import partial

import numpy as np
import torch
from gymnasium import spaces

from robobase.method.diffusion import Diffusion
from robobase.models.diffusion_models import ConditionalUnet1D


def _make_diffusion_method(use_ema: bool = True):
    observation_space = spaces.Dict(
        {
            "low_dim_state": spaces.Box(
                low=-1.0, high=1.0, shape=(1, 3), dtype=np.float32
            )
        }
    )
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(20, 2), dtype=np.float32)
    actor_model = partial(
        ConditionalUnet1D,
        sequence_length=20,
        diffusion_step_embed_dim=32,
        down_dims=[32, 64],
        kernel_size=3,
        n_groups=4,
    )
    return Diffusion(
        observation_space=observation_space,
        action_space=action_space,
        device=torch.device("cpu"),
        num_train_envs=1,
        num_eval_envs=1,
        replay_alpha=0.7,
        replay_beta=0.5,
        frame_stack_on_channel=True,
        is_rl=False,
        lr=1e-4,
        adaptive_lr=False,
        num_train_steps=4,
        actor_model=actor_model,
        encoder_model=None,
        view_fusion_model=None,
        num_diffusion_iters=4,
        use_ema=use_ema,
    )


def _clone_state_dict(state_dict):
    return {k: v.detach().clone() for k, v in state_dict.items()}


def _make_obs():
    return {"low_dim_state": torch.zeros((1, 1, 3), dtype=torch.float32)}


def test_diffusion_snapshot_restores_ema_state_when_enabled():
    method = _make_diffusion_method(use_ema=True)

    with torch.no_grad():
        for idx, param in enumerate(method.actor.actor.parameters(), start=1):
            param.add_(0.01 * idx)
        method.actor.ema.step(method.actor.actor.parameters())

        for idx, param in enumerate(method.actor.actor.parameters(), start=1):
            param.add_(0.02 * idx)
        method.actor.ema.step(method.actor.actor.parameters())

    state_dict = _clone_state_dict(method.state_dict())
    assert any(k.startswith("_actor_ema_state.") for k in state_dict)
    assert not any(k.startswith("actor.ema_actor.") for k in state_dict)

    reloaded_method = _make_diffusion_method(use_ema=True)
    reloaded_method.load_state_dict(state_dict)

    reloaded_state_dict = reloaded_method.state_dict()
    assert method.actor.ema.optimization_step == reloaded_method.actor.ema.optimization_step
    assert len(state_dict) == len(reloaded_state_dict)
    for key, value in state_dict.items():
        assert torch.allclose(value, reloaded_state_dict[key])


def test_diffusion_snapshot_omits_ema_state_when_disabled():
    method = _make_diffusion_method(use_ema=False)
    state_dict = _clone_state_dict(method.state_dict())

    assert not any(k.startswith("_actor_ema_state.") for k in state_dict)
    assert not any(k.startswith("actor.ema_actor.") for k in state_dict)

    reloaded_method = _make_diffusion_method(use_ema=False)
    reloaded_method.load_state_dict(state_dict)

    reloaded_state_dict = reloaded_method.state_dict()
    assert len(state_dict) == len(reloaded_state_dict)
    for key, value in state_dict.items():
        assert torch.allclose(value, reloaded_state_dict[key])


def test_diffusion_without_ema_ignores_ema_snapshot_state():
    method = _make_diffusion_method(use_ema=True)
    with torch.no_grad():
        method.actor.ema.step(method.actor.actor.parameters())

    state_dict = _clone_state_dict(method.state_dict())
    reloaded_method = _make_diffusion_method(use_ema=False)
    reloaded_method.load_state_dict(state_dict)

    assert not any(
        k.startswith("_actor_ema_state.") for k in reloaded_method.state_dict()
    )


def test_diffusion_with_ema_can_load_non_ema_snapshot():
    method = _make_diffusion_method(use_ema=False)
    with torch.no_grad():
        for idx, param in enumerate(method.actor.actor.parameters(), start=1):
            param.add_(0.03 * idx)

    state_dict = _clone_state_dict(method.state_dict())
    reloaded_method = _make_diffusion_method(use_ema=True)
    reloaded_method.load_state_dict(state_dict)

    for ema_param, actor_param in zip(
        reloaded_method.actor.ema.shadow_params,
        reloaded_method.actor.actor.parameters(),
    ):
        assert torch.allclose(ema_param, actor_param)


def test_diffusion_eval_mode_only_uses_ema_when_enabled():
    method = _make_diffusion_method(use_ema=True)
    obs = _make_obs()
    ema_calls = []

    def fake_copy_to(parameters):
        ema_calls.append(len(list(parameters)))

    method.actor.ema.copy_to = fake_copy_to

    torch.manual_seed(0)
    method._act(obs, eval_mode=False)
    assert not ema_calls

    torch.manual_seed(0)
    method._act(obs, eval_mode=True)
    assert ema_calls == [len(list(method.actor.ema_actor.parameters()))]

    method_without_ema = _make_diffusion_method(use_ema=False)
    torch.manual_seed(0)
    method_without_ema._act(obs, eval_mode=True)
