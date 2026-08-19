"""Focused checks for FS-CQN (frozen-support CQN-AS, fscqn proposal 2026-08-14).

The mask must (1) leave the legacy graph bit-identical when off, (2) never
let decode or the target-side greedy pick a bin outside the admissible set
{b: pi_b >= tau * max pi_b}, (3) stop training the bin-probability head after
support_mask_freeze_step gradient updates, (4) gate the dense/unseen floor so
in-mask unexecuted siblings receive no floor gradient, and (5) reject decode
paths the mask does not cover (autoregressive, twin, beam).
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnasium import spaces
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from robobase.factory import create_agent
from robobase.method.cqn_research import (
    dense_return_distributional_loss,
    unseen_return_floor_loss,
)

CONFIG_DIR = str((Path(__file__).parents[2] / "robobase" / "cfgs").resolve())

ACTION_SEQUENCE = 3
ACTION_DIM = 2
BINS = 5
ATOMS = 11


def _params_equal(left, right, exact=False):
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    assert left_tree == right_tree
    if exact:
        return all(
            np.array_equal(np.asarray(a), np.asarray(b))
            for a, b in zip(left_leaves, right_leaves, strict=True)
        )
    return all(
        np.allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-7)
        for a, b in zip(left_leaves, right_leaves, strict=True)
    )


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
                f"method.atoms={ATOMS}",
                *overrides,
            ],
        )


def _fscqn_overrides(*extra):
    return (
        "method.bc_lambda=0.0",
        "method.bc_margin=0.0",
        "method.demo_fosd=false",
        "method.use_frozen_support_mask=true",
        "method.support_mask_tau=0.3",
        "method.support_mask_freeze_step=10000",
        "method.dense_return_q_target=true",
        "method.dense_return_positive_only=true",
        "method.mc_lower_bound_target=true",
        *extra,
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


def _batch(batch_size=4, mc_return=0.7):
    rng = np.random.default_rng(11)
    return {
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
        "mc_return": np.full((batch_size,), mc_return, dtype=np.float32),
    }


def _copy_batch(batch):
    return {k: np.array(v, copy=True) for k, v in batch.items()}


def _make_agent(cfg):
    observation_space, action_space = _spaces()
    return create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )


class _CriticStub:
    """Constant per-bin C51 logits: each bin's mass sits on one atom."""

    def __init__(self, atom_by_bin):
        self._atom_by_bin = list(atom_by_bin)

    def apply(self, params, features, one_hot, midpoint, **kwargs):
        del params, one_hot, midpoint, kwargs
        table = np.full((BINS, ATOMS), -20.0, dtype=np.float32)
        for bin_index, atom in enumerate(self._atom_by_bin):
            table[bin_index, atom] = 20.0
        return jnp.asarray(
            np.broadcast_to(
                table[None, None, None],
                (
                    features.shape[0],
                    ACTION_SEQUENCE,
                    ACTION_DIM,
                    BINS,
                    ATOMS,
                ),
            )
        )


class _PolicyStub:
    """Constant per-bin behavior logits with known softmax probabilities."""

    def __init__(self, strengths):
        self._strengths = np.asarray(strengths, dtype=np.float32)

    def apply(self, params, features, one_hot, midpoint, **kwargs):
        del params, one_hot, midpoint, kwargs
        return jnp.asarray(
            np.broadcast_to(
                self._strengths[None, None, None, :, None],
                (
                    features.shape[0],
                    ACTION_SEQUENCE,
                    ACTION_DIM,
                    BINS,
                    1,
                ),
            )
        )


def _features(agent, batch):
    obs_inputs = agent._prepare_rl_obs_inputs(batch)
    return agent._rl_features(agent.params.get("encoder"), obs_inputs)


def test_flags_off_is_bit_identical_to_legacy():
    batch = _batch()
    legacy = _make_agent(_compose())
    explicit = _make_agent(
        _compose(
            "method.use_frozen_support_mask=false",
            "method.support_mask_tau=0.3",
            "method.support_mask_freeze_step=10000",
        )
    )
    assert "policy" not in legacy.params
    assert "policy" not in explicit.params
    legacy.logging = True
    explicit.logging = True
    legacy_metrics = legacy.update(iter([_copy_batch(batch)]), step=1)
    explicit_metrics = explicit.update(iter([_copy_batch(batch)]), step=1)
    assert "support_mask_ce_loss" not in legacy_metrics
    assert "support_mask_ce_loss" not in explicit_metrics
    assert _params_equal(legacy.params, explicit.params, exact=True)


def test_mask_on_update_wires_head_and_metrics():
    agent = _make_agent(_compose(*_fscqn_overrides()))
    assert "policy" in agent.params
    agent.logging = True
    metrics = agent.update(iter([_batch()]), step=1)
    assert metrics["support_mask_ce_loss"] == pytest.approx(
        np.log(BINS), rel=1e-3
    )
    assert metrics["support_mask_ce_weight"] == pytest.approx(1.0)
    assert 0.0 < metrics["support_mask_width"] <= 1.0


def test_masked_decode_never_leaves_admissible_set():
    agent = _make_agent(_compose(*_fscqn_overrides()))
    features = _features(agent, _batch())
    agent.critic_model = _CriticStub([0, 0, 0, ATOMS - 1, 0])
    agent.policy_model = _PolicyStub([8.0, 0.0, 0.0, 0.0, 0.0])
    _, unmasked = agent._greedy_action(
        agent.params["critic"],
        features,
        key=jax.random.PRNGKey(0),
    )
    assert np.all(np.asarray(unmasked) == 3)
    _, masked = agent._greedy_action(
        agent.params["critic"],
        features,
        key=jax.random.PRNGKey(0),
        policy_params=agent.params["policy"],
    )
    assert np.all(np.asarray(masked) == 0)


def test_masked_tie_break_samples_only_admissible_bins():
    agent = _make_agent(_compose(*_fscqn_overrides()))
    features = _features(agent, _batch())
    agent.critic_model = _CriticStub([0, 0, 0, 0, 0])
    agent.policy_model = _PolicyStub([8.0, 8.0, 0.0, 0.0, 0.0])
    for seed in range(5):
        _, selected = agent._greedy_action(
            agent.params["critic"],
            features,
            key=jax.random.PRNGKey(seed),
            policy_params=agent.params["policy"],
        )
        assert np.all(np.asarray(selected) <= 1)


def test_target_side_greedy_inherits_the_mask():
    agent = _make_agent(_compose(*_fscqn_overrides()))
    batch = _batch()
    features = _features(agent, batch)
    agent.critic_model = _CriticStub([0, 0, 0, ATOMS - 1, 0])
    agent.policy_model = _PolicyStub([8.0, 0.0, 0.0, 0.0, 0.0])
    actions = jnp.asarray(batch["action"].reshape(batch["action"].shape[0], -1))
    demos = jnp.ones((features.shape[0],), dtype=jnp.float32)
    target_action, _ = agent._td_target_action_for_update(
        agent.params["critic"],
        features,
        actions,
        actions,
        demos,
        jax.random.PRNGKey(3),
        policy_params=agent.params["policy"],
    )
    masked_action, masked_selected = agent._greedy_action(
        agent.params["critic"],
        features,
        key=jax.random.PRNGKey(3),
        policy_params=agent.params["policy"],
    )
    assert np.all(np.asarray(masked_selected) == 0)
    assert np.allclose(np.asarray(target_action), np.asarray(masked_action))
    unmasked_action, _ = agent._greedy_action(
        agent.params["critic"],
        features,
        key=jax.random.PRNGKey(3),
    )
    assert not np.allclose(
        np.asarray(target_action), np.asarray(unmasked_action)
    )


def test_target_mask_only_unmasks_decode_but_keeps_target_masked():
    agent = _make_agent(
        _compose(
            *_fscqn_overrides("method.support_mask_decode=false")
        )
    )
    batch = _batch()
    obs_inputs = agent._prepare_rl_obs_inputs(batch)
    features = agent._rl_features(
        agent.params.get("encoder"), obs_inputs, stop_gradient=True
    )
    agent.critic_model = _CriticStub([0, 0, 0, ATOMS - 1, 0])
    agent.policy_model = _PolicyStub([8.0, 0.0, 0.0, 0.0, 0.0])

    rollout_action = agent._greedy_action_impl(
        agent.params,
        agent.target_critic_params,
        obs_inputs,
        False,
        jax.random.PRNGKey(7),
        jnp.full((features.shape[0],), -1, dtype=jnp.int32),
    )
    unmasked_action, unmasked_bins = agent._greedy_action(
        agent.params["critic"],
        features,
        key=jax.random.PRNGKey(7),
    )
    assert np.all(np.asarray(unmasked_bins) == 3)
    assert np.allclose(np.asarray(rollout_action), np.asarray(unmasked_action))

    actions = jnp.asarray(
        batch["action"].reshape(batch["action"].shape[0], -1)
    )
    demos = jnp.ones((features.shape[0],), dtype=jnp.float32)
    target_action, _ = agent._td_target_action_for_update(
        agent.params["critic"],
        features,
        actions,
        actions,
        demos,
        jax.random.PRNGKey(7),
        policy_params=agent.params["policy"],
    )
    masked_action, masked_bins = agent._greedy_action(
        agent.params["critic"],
        features,
        key=jax.random.PRNGKey(7),
        policy_params=agent.params["policy"],
    )
    assert np.all(np.asarray(masked_bins) == 0)
    assert np.allclose(np.asarray(target_action), np.asarray(masked_action))


def test_head_frozen_after_freeze_step():
    agent = _make_agent(
        _compose(*_fscqn_overrides("method.support_mask_freeze_step=1"))
    )
    batch = _batch()
    before = jax.tree.map(np.asarray, agent.params["policy"])
    agent.logging = True
    metrics = agent.update(iter([_copy_batch(batch)]), step=1)
    assert metrics["support_mask_ce_weight"] == pytest.approx(1.0)
    after_first = jax.tree.map(np.asarray, agent.params["policy"])
    assert not _params_equal(before, after_first, exact=True)
    critic_before = jax.tree.map(np.asarray, agent.params["critic"])
    metrics = agent.update(iter([_copy_batch(batch)]), step=2)
    assert metrics["support_mask_ce_weight"] == pytest.approx(0.0)
    agent.update(iter([_copy_batch(batch)]), step=3)
    assert _params_equal(
        after_first, agent.params["policy"], exact=True
    )
    assert not _params_equal(
        critic_before, agent.params["critic"], exact=True
    )


def test_head_keeps_training_before_freeze_step():
    agent = _make_agent(_compose(*_fscqn_overrides()))
    batch = _batch()
    agent.update(iter([_copy_batch(batch)]), step=1)
    after_first = jax.tree.map(np.asarray, agent.params["policy"])
    agent.update(iter([_copy_batch(batch)]), step=2)
    assert not _params_equal(
        after_first, agent.params["policy"], exact=True
    )


def _floor_gating_fixtures():
    rng = np.random.default_rng(5)
    logits = jnp.asarray(
        rng.normal(size=(1, 1, 1, 3, 5)).astype(np.float32)
    )
    discrete_action = jnp.asarray([[[1]]], dtype=jnp.int32)
    support = jnp.linspace(-2.0, 2.0, 5)
    support_mask = jnp.asarray(
        [[[[True, True, False]]]], dtype=jnp.bool_
    )
    chosen_target = jnp.zeros((1, 1, 1, 5), dtype=jnp.float32)
    chosen_target = chosen_target.at[..., 4].set(1.0)
    return logits, discrete_action, support, support_mask, chosen_target


def test_dense_floor_gating_spares_in_mask_siblings():
    (
        logits,
        discrete_action,
        support,
        support_mask,
        chosen_target,
    ) = _floor_gating_fixtures()

    def loss(current_logits, mask):
        per_sample, _, _ = dense_return_distributional_loss(
            current_logits,
            discrete_action,
            chosen_target,
            support,
            0.0,
            support_mask=mask,
        )
        return per_sample.sum()

    gated_grad = jax.grad(loss)(logits, support_mask)
    ungated_grad = jax.grad(loss)(logits, None)
    assert np.allclose(np.asarray(gated_grad)[0, 0, 0, 0], 0.0)
    assert np.abs(np.asarray(gated_grad)[0, 0, 0, 2]).sum() > 0.0
    assert np.abs(np.asarray(gated_grad)[0, 0, 0, 1]).sum() > 0.0
    assert np.abs(np.asarray(ungated_grad)[0, 0, 0, 0]).sum() > 0.0


def test_unseen_floor_gating_spares_in_mask_siblings():
    (
        logits,
        discrete_action,
        support,
        support_mask,
        _,
    ) = _floor_gating_fixtures()

    def loss(current_logits, mask):
        per_sample, _ = unseen_return_floor_loss(
            current_logits,
            discrete_action,
            support,
            0.0,
            support_mask=mask,
        )
        return per_sample.sum()

    gated_grad = jax.grad(loss)(logits, support_mask)
    ungated_grad = jax.grad(loss)(logits, None)
    assert np.allclose(np.asarray(gated_grad)[0, 0, 0, 0], 0.0)
    assert np.abs(np.asarray(gated_grad)[0, 0, 0, 2]).sum() > 0.0
    assert np.allclose(np.asarray(gated_grad)[0, 0, 0, 1], 0.0)
    assert np.abs(np.asarray(ungated_grad)[0, 0, 0, 0]).sum() > 0.0


def test_unseen_floor_max_reduction_finite_when_all_bins_admissible():
    logits, discrete_action, support, _, _ = _floor_gating_fixtures()
    all_admissible = jnp.ones((1, 1, 1, 3), dtype=jnp.bool_)
    per_sample, _ = unseen_return_floor_loss(
        logits,
        discrete_action,
        support,
        0.0,
        reduction="max",
        support_mask=all_admissible,
    )
    assert np.all(np.isfinite(np.asarray(per_sample)))
    assert np.asarray(per_sample).sum() == pytest.approx(0.0)


def test_rejects_autoregressive_combo():
    with pytest.raises(ValueError, match="autoregressive_action_dims"):
        _make_agent(
            _compose(
                *_fscqn_overrides(
                    "method.autoregressive_action_dims=true"
                )
            )
        )


def test_rejects_separate_bc_policy_combo():
    with pytest.raises(ValueError, match="separate_bc_policy"):
        _make_agent(
            _compose(
                *_fscqn_overrides("method.separate_bc_policy=true")
            )
        )


def test_rejects_out_of_range_tau():
    with pytest.raises(ValueError, match="support_mask_tau"):
        _make_agent(
            _compose(*_fscqn_overrides("method.support_mask_tau=0.0"))
        )
