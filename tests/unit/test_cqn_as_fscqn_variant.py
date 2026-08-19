"""R2 extraction checks for the FS-CQN frozen-support-mask variant.

``robobase/method/cqn_as_fscqn.py`` subclasses the FROZEN pristine
``CQNAS``.  These tests encode the R2 contract:

1. flags off  -> bit-identical to the pristine official class;
2. flags on   -> act()/update() run, metrics finite, mask keys present,
                 and the mask actually constrains decode and the TD target;
3. flags on   -> same ``critic_loss`` as the research monolith.

The behavioral spec is ``tests/unit/test_fscqn_support_mask.py`` (read-only);
its assertions are adapted here to the pristine-base call shapes.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnasium import spaces
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from robobase.factory import create_agent
from robobase.method.cqn_as_fscqn import (
    CQNASFrozenSupportMask,
    cqn_as_fscqn_spec_from_cfg,
    support_gated_per_bin_loss,
    support_gated_tail_unseen_q,
    support_gated_unseen_mask,
)

CONFIG_DIR = str((Path(__file__).parents[2] / "robobase" / "cfgs").resolve())

ACTION_SEQUENCE = 3
ACTION_DIM = 2
BINS = 5
ATOMS = 11


def _compose(method: str, *overrides: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(
            config_name="robobase_config",
            overrides=[
                f"method={method}",
                f"action_sequence={ACTION_SEQUENCE}",
                "num_train_envs=1",
                "num_eval_envs=1",
                "num_explore_steps=0",
                "backend.jit=false",
                "backend.platform=cpu",
                "method.model.hidden_dims=[32,32]",
                f"method.atoms={ATOMS}",
                *overrides,
            ],
        )


def _mask_on(*extra: str) -> tuple[str, ...]:
    return (
        "method.use_frozen_support_mask=true",
        "method.support_mask_tau=0.3",
        "method.support_mask_freeze_step=10000",
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


def _batch(batch_size=4):
    rng = np.random.default_rng(11)
    return {
        "low_dim_state": rng.normal(size=(batch_size, 1, 5)).astype(np.float32),
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


def _copy_batch(batch):
    return {key: np.array(value, copy=True) for key, value in batch.items()}


def _observation(batch_size=1):
    rng = np.random.default_rng(23)
    return {
        "low_dim_state": rng.normal(size=(batch_size, 1, 5)).astype(np.float32)
    }


def _fscqn_kwargs(cfg, observation_space, action_space):
    """Exactly the construction an integrator must add to ``factory.py``."""

    spec = cqn_as_fscqn_spec_from_cfg(cfg)
    return dict(
        critic_lr=spec.critic_lr,
        num_train_steps=spec.num_train_steps,
        num_explore_steps=spec.num_explore_steps,
        critic_target_tau=spec.critic_target_tau,
        critic_grad_clip=spec.critic_grad_clip,
        weight_decay=spec.weight_decay,
        levels=spec.levels,
        bins=spec.bins,
        atoms=spec.atoms,
        v_min=spec.v_min,
        v_max=spec.v_max,
        critic_lambda=spec.critic_lambda,
        centralized_critic=spec.centralized_critic,
        use_dueling=spec.use_dueling,
        always_bootstrap=spec.always_bootstrap,
        stddev_schedule=spec.stddev_schedule,
        bc_lambda=spec.bc_lambda,
        bc_margin=spec.bc_margin,
        use_target_network_for_rollout=spec.use_target_network_for_rollout,
        num_update_steps=spec.num_update_steps,
        gru_layers=spec.gru_layers,
        temporal_ensemble=spec.temporal_ensemble,
        temporal_ensemble_replan_interval=(
            spec.temporal_ensemble_replan_interval
        ),
        temporal_ensemble_gain=spec.temporal_ensemble_gain,
        tie_break_delta=spec.tie_break_delta,
        use_frozen_support_mask=spec.use_frozen_support_mask,
        support_mask_decode=spec.support_mask_decode,
        support_mask_tau=spec.support_mask_tau,
        support_mask_freeze_step=spec.support_mask_freeze_step,
        model=spec.model,
        jit=bool(cfg.backend.jit),
        platform=cfg.backend.platform,
        seed=int(cfg.seed),
        observation_space=observation_space,
        action_space=action_space,
        num_train_envs=cfg.num_train_envs,
        num_eval_envs=cfg.num_eval_envs,
        replay_alpha=cfg.replay.alpha,
        replay_beta=cfg.replay.beta,
        frame_stack_on_channel=cfg.frame_stack_on_channel,
    )


def _make_fscqn(*overrides: str, cls=CQNASFrozenSupportMask):
    cfg = _compose("cqn_as_fscqn", *overrides)
    observation_space, action_space = _spaces()
    return cls(**_fscqn_kwargs(cfg, observation_space, action_space))


def _make_agent(method: str, *overrides: str):
    cfg = _compose(method, *overrides)
    observation_space, action_space = _spaces()
    return create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )


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


def _features(agent, batch, stop_gradient=False):
    obs_inputs = agent._prepare_rl_obs_inputs(batch)
    return agent._rl_features(
        agent.params.get("encoder"),
        obs_inputs,
        stop_gradient=stop_gradient,
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
                (features.shape[0], ACTION_SEQUENCE, ACTION_DIM, BINS, ATOMS),
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
                (features.shape[0], ACTION_SEQUENCE, ACTION_DIM, BINS, 1),
            )
        )


# --------------------------------------------------------------------------
# 1. flags off == pristine official CQN-AS
# --------------------------------------------------------------------------


def test_flags_off_is_bit_identical_to_pristine():
    batch = _batch()
    pristine = _make_agent("cqn_as_official")
    variant = _make_fscqn()
    explicit = _make_fscqn(
        "method.use_frozen_support_mask=false",
        "method.support_mask_decode=true",
        "method.support_mask_tau=0.3",
        "method.support_mask_freeze_step=10000",
    )
    assert "policy" not in pristine.params
    assert "policy" not in variant.params
    assert "policy" not in explicit.params
    assert _params_equal(pristine.params, variant.params, exact=True)

    pristine.logging = True
    variant.logging = True
    explicit.logging = True
    pristine_metrics = pristine.update(iter([_copy_batch(batch)]), step=1)
    variant_metrics = variant.update(iter([_copy_batch(batch)]), step=1)
    explicit_metrics = explicit.update(iter([_copy_batch(batch)]), step=1)

    assert "support_mask_ce_loss" not in variant_metrics
    assert "support_mask_ce_loss" not in explicit_metrics
    assert set(variant_metrics) == set(pristine_metrics)
    assert variant_metrics["critic_loss"] == pytest.approx(
        pristine_metrics["critic_loss"], abs=1e-6
    )
    assert explicit_metrics["critic_loss"] == pytest.approx(
        pristine_metrics["critic_loss"], abs=1e-6
    )
    assert _params_equal(pristine.params, variant.params, exact=True)
    assert _params_equal(
        pristine.target_critic_params,
        variant.target_critic_params,
        exact=True,
    )


# --------------------------------------------------------------------------
# 2. flags on: runs, reports, and constrains
# --------------------------------------------------------------------------


def test_mask_on_update_wires_head_and_metrics():
    agent = _make_fscqn(*_mask_on())
    assert "policy" in agent.params
    agent.logging = True
    action = agent.act(_observation(), step=100, eval_mode=True)
    assert np.all(np.isfinite(np.asarray(action)))
    metrics = agent.update(iter([_batch()]), step=1)
    for key in (
        "support_mask_ce_loss",
        "support_mask_ce_weight",
        "support_mask_width",
    ):
        assert key in metrics
    for key, value in metrics.items():
        assert np.isfinite(value), key
    # A zero-initialized head is uniform over bins on the first update.
    assert metrics["support_mask_ce_loss"] == pytest.approx(
        np.log(BINS), rel=1e-3
    )
    assert metrics["support_mask_ce_weight"] == pytest.approx(1.0)
    assert 0.0 < metrics["support_mask_width"] <= 1.0


def test_masked_decode_never_leaves_admissible_set():
    agent = _make_fscqn(*_mask_on())
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
    agent = _make_fscqn(*_mask_on())
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
    agent = _make_fscqn(*_mask_on())
    features = _features(agent, _batch())
    agent.critic_model = _CriticStub([0, 0, 0, ATOMS - 1, 0])
    agent.policy_model = _PolicyStub([8.0, 0.0, 0.0, 0.0, 0.0])
    # ``_greedy_action_for_update`` is the hook the TD target uses inside
    # ``_build_update_fn``; on the pristine base there is no separate
    # ``_td_target_action_for_update`` indirection.
    target_action, target_bins = agent._greedy_action_for_update(
        agent.params["critic"],
        features,
        jax.random.PRNGKey(3),
        policy_params=agent.params["policy"],
    )
    masked_action, masked_bins = agent._greedy_action(
        agent.params["critic"],
        features,
        key=jax.random.PRNGKey(3),
        policy_params=agent.params["policy"],
    )
    assert np.all(np.asarray(target_bins) == 0)
    assert np.all(np.asarray(masked_bins) == 0)
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
    agent = _make_fscqn(*_mask_on("method.support_mask_decode=false"))
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
    )
    unmasked_action, unmasked_bins = agent._greedy_action(
        agent.params["critic"],
        features,
        key=jax.random.PRNGKey(7),
    )
    assert np.all(np.asarray(unmasked_bins) == 3)
    assert np.allclose(np.asarray(rollout_action), np.asarray(unmasked_action))

    target_action, target_bins = agent._greedy_action_for_update(
        agent.params["critic"],
        features,
        jax.random.PRNGKey(7),
        policy_params=agent.params["policy"],
    )
    assert np.all(np.asarray(target_bins) == 0)
    masked_action, _ = agent._greedy_action(
        agent.params["critic"],
        features,
        key=jax.random.PRNGKey(7),
        policy_params=agent.params["policy"],
    )
    assert np.allclose(np.asarray(target_action), np.asarray(masked_action))


def test_head_frozen_after_freeze_step():
    agent = _make_fscqn(*_mask_on("method.support_mask_freeze_step=1"))
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
    assert _params_equal(after_first, agent.params["policy"], exact=True)
    assert not _params_equal(
        critic_before, agent.params["critic"], exact=True
    )


def test_head_keeps_training_before_freeze_step():
    agent = _make_fscqn(*_mask_on())
    batch = _batch()
    agent.update(iter([_copy_batch(batch)]), step=1)
    after_first = jax.tree.map(np.asarray, agent.params["policy"])
    agent.update(iter([_copy_batch(batch)]), step=2)
    assert not _params_equal(after_first, agent.params["policy"], exact=True)


# --------------------------------------------------------------------------
# 3. flags on == research monolith
# --------------------------------------------------------------------------


def test_flags_on_matches_research_critic_loss():
    batch = _batch()
    variant = _make_fscqn(*_mask_on())
    research = _make_agent("cqn_as", *_mask_on())
    assert type(research).__module__ == "robobase.method.cqn_as_research"
    assert _params_equal(variant.params, research.params, exact=True)
    variant.logging = True
    research.logging = True
    variant_metrics = variant.update(iter([_copy_batch(batch)]), step=1)
    research_metrics = research.update(iter([_copy_batch(batch)]), step=1)
    assert variant_metrics["critic_loss"] == pytest.approx(
        research_metrics["critic_loss"], abs=1e-5
    )
    assert variant_metrics["support_mask_ce_loss"] == pytest.approx(
        research_metrics["support_mask_ce_loss"], abs=1e-6
    )
    assert variant_metrics["support_mask_ce_weight"] == pytest.approx(
        research_metrics["support_mask_ce_weight"]
    )
    assert _params_equal(variant.params, research.params)


# --------------------------------------------------------------------------
# 4. floor-loss gating primitives (dense-return coupling)
# --------------------------------------------------------------------------


def _gating_fixtures():
    rng = np.random.default_rng(5)
    logits = jnp.asarray(rng.normal(size=(1, 1, 1, 3, 5)).astype(np.float32))
    discrete_action = jnp.asarray([[[1]]], dtype=jnp.int32)
    support = jnp.linspace(-2.0, 2.0, 5)
    support_mask = jnp.asarray([[[[True, True, False]]]], dtype=jnp.bool_)
    return logits, discrete_action, support, support_mask


def _bin_q(all_logits, support):
    return jnp.sum(jax.nn.softmax(all_logits, axis=-1) * support, axis=-1)


def test_support_gated_unseen_mask_spares_in_mask_siblings():
    logits, discrete_action, support, support_mask = _gating_fixtures()

    def loss(current_logits, mask):
        # Local stand-in for the dense-return line's
        # ``unseen_return_floor_loss`` mean reduction, exercising only the
        # gating primitive this line contributes.
        all_q = _bin_q(current_logits, support)
        chosen_mask = jax.nn.one_hot(
            discrete_action, all_q.shape[-1], dtype=all_q.dtype
        )
        unseen_mask = support_gated_unseen_mask(1.0 - chosen_mask, mask)
        return jnp.sum(jnp.square(all_q - 0.0) * unseen_mask)

    gated_grad = np.asarray(jax.grad(loss)(logits, support_mask))
    ungated_grad = np.asarray(jax.grad(loss)(logits, None))
    assert np.allclose(gated_grad[0, 0, 0, 0], 0.0)
    assert np.allclose(gated_grad[0, 0, 0, 1], 0.0)
    assert np.abs(gated_grad[0, 0, 0, 2]).sum() > 0.0
    assert np.abs(ungated_grad[0, 0, 0, 0]).sum() > 0.0


def test_support_gated_per_bin_loss_spares_in_mask_siblings():
    logits, discrete_action, support, support_mask = _gating_fixtures()

    def loss(current_logits, mask):
        # Local stand-in for the tail of the dense-return line's
        # ``dense_return_distributional_loss``.
        all_q = _bin_q(current_logits, support)
        chosen_mask = jax.nn.one_hot(
            discrete_action, all_q.shape[-1], dtype=all_q.dtype
        )
        per_bin_loss = jnp.square(all_q - 1.0)
        return jnp.sum(
            support_gated_per_bin_loss(per_bin_loss, chosen_mask, mask)
        )

    gated_grad = np.asarray(jax.grad(loss)(logits, support_mask))
    ungated_grad = np.asarray(jax.grad(loss)(logits, None))
    assert np.allclose(gated_grad[0, 0, 0, 0], 0.0)
    # The executed bin keeps its return target, unlike the unseen-floor gate.
    assert np.abs(gated_grad[0, 0, 0, 1]).sum() > 0.0
    assert np.abs(gated_grad[0, 0, 0, 2]).sum() > 0.0
    assert np.abs(ungated_grad[0, 0, 0, 0]).sum() > 0.0


def test_support_gated_tail_unseen_q_is_finite_when_all_bins_admissible():
    all_admissible = jnp.ones((1, 1, 1, 3), dtype=jnp.bool_)
    tail = jnp.asarray([[[[-jnp.inf]]]], dtype=jnp.float32)
    gated = support_gated_tail_unseen_q(tail, 0.0, all_admissible)
    assert np.all(np.isfinite(np.asarray(gated)))
    assert np.square(np.asarray(gated) - 0.0).sum() == pytest.approx(0.0)
    ungated = support_gated_tail_unseen_q(tail, 0.0, None)
    assert not np.all(np.isfinite(np.asarray(ungated)))


# --------------------------------------------------------------------------
# 5. configuration guards
# --------------------------------------------------------------------------


def test_rejects_out_of_range_tau():
    with pytest.raises(ValueError, match="support_mask_tau"):
        _make_fscqn(*_mask_on("method.support_mask_tau=0.0"))


def test_rejects_negative_freeze_step():
    with pytest.raises(ValueError, match="support_mask_freeze_step"):
        _make_fscqn(*_mask_on("method.support_mask_freeze_step=-1"))


class _SeparateBCPolicyCombo(CQNASFrozenSupportMask):
    """Stands in for a future mix-in with the bc-policy line."""

    def __init__(self, **kwargs):
        self.separate_bc_policy = True
        super().__init__(**kwargs)


class _AutoregressiveCombo(CQNASFrozenSupportMask):
    """Stands in for a future mix-in with the td-variants line."""

    def __init__(self, **kwargs):
        self.autoregressive_action_dims = True
        super().__init__(**kwargs)


def test_rejects_separate_bc_policy_combo():
    with pytest.raises(ValueError, match="separate_bc_policy"):
        _make_fscqn(*_mask_on(), cls=_SeparateBCPolicyCombo)


def test_rejects_autoregressive_combo():
    with pytest.raises(ValueError, match="autoregressive_action_dims"):
        _make_fscqn(*_mask_on(), cls=_AutoregressiveCombo)


def test_uncovered_decode_paths_are_allowed_when_the_mask_is_off():
    agent = _make_fscqn(cls=_SeparateBCPolicyCombo)
    assert agent.use_frozen_support_mask is False
    assert "policy" not in agent.params
