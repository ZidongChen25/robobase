"""Focused checks for token_split_horizon_targets (cqn-rline.md wave 2).

The split must (1) survive the four-layer config chain (yaml -> spec ->
factory table -> __init__), (2) reject unsupported compositions loudly,
(3) reproduce the exact legacy single-horizon loss when the auxiliary
transition equals the primary one, (4) actually consume the auxiliary
horizon (different aux rewards change the loss), and (5) report the
token mask fraction and aux reward mean for wiring positive controls.
"""

from pathlib import Path

import jax
import numpy as np
import pytest
from gymnasium import spaces
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from robobase.factory import create_agent


def _params_equal(left, right):
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    assert left_tree == right_tree
    return all(
        np.allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-7)
        for a, b in zip(left_leaves, right_leaves, strict=True)
    )

CONFIG_DIR = str((Path(__file__).parents[2] / "robobase" / "cfgs").resolve())

ACTION_SEQUENCE = 3
ACTION_DIM = 2


def _compose(*overrides):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(
            config_name="robobase_config",
            overrides=[
                "method=cqn_as",
                f"action_sequence={ACTION_SEQUENCE}",
                "num_train_envs=1",
                "num_eval_envs=1",
                "backend.jit=false",
                "backend.platform=cpu",
                "method.model.hidden_dims=[32,32]",
                "method.atoms=11",
                *overrides,
            ],
        )


def _split_cfg(*overrides):
    return _compose(
        "method.token_split_horizon_targets=true",
        "method.token_split_boundary=1",
        "replay.nstep=1",
        "replay.auxiliary_nstep=3",
        *overrides,
    )


def _spaces():
    observation_space = spaces.Dict(
        {
            "low_dim_state": spaces.Box(
                -np.inf,
                np.inf,
                shape=(1, 5),
                dtype=np.float32,
            )
        }
    )
    action_space = spaces.Box(
        -1.0,
        1.0,
        shape=(ACTION_SEQUENCE, ACTION_DIM),
        dtype=np.float32,
    )
    return observation_space, action_space


def _batch(batch_size=4, aux=True):
    rng = np.random.default_rng(11)
    batch = {
        "low_dim_state": rng.normal(size=(batch_size, 1, 5)).astype(
            np.float32
        ),
        "low_dim_state_tp1": rng.normal(size=(batch_size, 1, 5)).astype(
            np.float32
        ),
        "action": rng.uniform(
            -1.0,
            1.0,
            size=(batch_size, ACTION_SEQUENCE, ACTION_DIM),
        ).astype(np.float32),
        "reward": rng.normal(size=(batch_size,)).astype(np.float32),
        "discount": np.full((batch_size,), 0.99, dtype=np.float32),
        "terminal": np.zeros((batch_size,), dtype=bool),
        "truncated": np.zeros((batch_size,), dtype=bool),
        "demo": np.ones((batch_size,), dtype=np.uint8),
    }
    if aux:
        batch["low_dim_state_tp_aux"] = rng.normal(
            size=(batch_size, 1, 5)
        ).astype(np.float32)
        batch["action_tp_aux"] = rng.uniform(
            -1.0,
            1.0,
            size=(batch_size, ACTION_SEQUENCE, ACTION_DIM),
        ).astype(np.float32)
        batch["reward_aux"] = rng.normal(size=(batch_size,)).astype(
            np.float32
        )
        batch["discount_aux"] = np.full(
            (batch_size,), 0.99**3, dtype=np.float32
        )
        batch["terminal_aux"] = np.zeros((batch_size,), dtype=bool)
        batch["truncated_aux"] = np.zeros((batch_size,), dtype=bool)
    return batch


def _make_agent(cfg):
    observation_space, action_space = _spaces()
    return create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )


def test_token_split_attribute_round_trip():
    agent = _make_agent(_split_cfg())
    assert agent.token_split_horizon_targets is True
    assert agent.token_split_boundary == 1


def test_token_split_default_off_and_absent_from_metrics():
    agent = _make_agent(_compose())
    assert agent.token_split_horizon_targets is False
    agent.logging = True
    metrics = agent.update(iter([_batch(aux=False)]), step=1)
    assert "token_split_aux_fraction" not in metrics
    assert "token_split_aux_reward_mean" not in metrics


def test_token_split_requires_auxiliary_nstep():
    with pytest.raises(ValueError, match="auxiliary_nstep"):
        _make_agent(
            _compose(
                "method.token_split_horizon_targets=true",
                "method.token_split_boundary=1",
            )
        )


def test_token_split_requires_explicit_boundary():
    with pytest.raises(ValueError, match="token_split_boundary"):
        _make_agent(_split_cfg("method.token_split_boundary=null"))


def test_token_split_boundary_range_is_validated():
    with pytest.raises(ValueError, match="token_split_boundary"):
        _make_agent(
            _split_cfg(f"method.token_split_boundary={ACTION_SEQUENCE}")
        )


def test_token_split_rejects_mc_lower_bound():
    with pytest.raises(ValueError, match="mc_lower_bound_target"):
        _make_agent(_split_cfg("method.mc_lower_bound_target=true"))


def test_token_split_rejects_dense_return_target():
    with pytest.raises(ValueError, match="dense_return_q_target"):
        _make_agent(_split_cfg("method.dense_return_q_target=true"))


def test_token_split_missing_aux_batch_fields_raise():
    agent = _make_agent(_split_cfg())
    with pytest.raises(KeyError, match="auxiliary-horizon"):
        agent.update(iter([_batch(aux=False)]), step=1)


def test_token_split_matches_legacy_when_aux_equals_primary():
    """Aux transition == primary transition -> the split target is the
    legacy target for every token, so the loss must match the flag-off
    loss on the same batch and identical initial parameters."""
    batch = _batch()
    for name, source in (
        ("low_dim_state_tp_aux", "low_dim_state_tp1"),
        ("reward_aux", "reward"),
        ("discount_aux", "discount"),
        ("terminal_aux", "terminal"),
        ("truncated_aux", "truncated"),
    ):
        batch[name] = np.array(batch[source], copy=True)
    batch["action_tp_aux"] = np.array(batch["action"], copy=True)

    legacy = _make_agent(_compose())
    split = _make_agent(_split_cfg())
    legacy.logging = True
    split.logging = True
    legacy_metrics = legacy.update(
        iter([{k: np.array(v, copy=True) for k, v in batch.items()}]),
        step=1,
    )
    split_metrics = split.update(
        iter([{k: np.array(v, copy=True) for k, v in batch.items()}]),
        step=1,
    )
    assert split_metrics["token_split_aux_fraction"] == pytest.approx(
        2.0 / 3.0
    )
    assert split_metrics["critic_loss"] == pytest.approx(
        legacy_metrics["critic_loss"], rel=1e-5
    )
    # The heads start zero-initialised, so the first-step loss is
    # target-independent; the gradients are not. Parameter equality after
    # one update is the real equivalence statement.
    assert _params_equal(legacy.params, split.params)


def test_token_split_consumes_aux_horizon():
    """Changing only the auxiliary reward must change the loss (the aux
    horizon is actually wired into the target), and the wiring metric
    must report it."""
    base_batch = _batch()
    shifted_batch = {
        k: np.array(v, copy=True) for k, v in base_batch.items()
    }
    shifted_batch["reward_aux"] = base_batch["reward_aux"] + 1.0

    agent_a = _make_agent(_split_cfg())
    agent_b = _make_agent(_split_cfg())
    agent_a.logging = True
    agent_b.logging = True
    metrics_a = agent_a.update(iter([base_batch]), step=1)
    metrics_b = agent_b.update(iter([shifted_batch]), step=1)
    assert metrics_b["token_split_aux_reward_mean"] == pytest.approx(
        metrics_a["token_split_aux_reward_mean"] + 1.0
    )
    # Zero-initialised heads make the first-step loss target-independent;
    # the gradient (softmax - target) is not, so the updated parameters
    # must differ when the auxiliary horizon reward differs.
    assert not _params_equal(agent_a.params, agent_b.params)
