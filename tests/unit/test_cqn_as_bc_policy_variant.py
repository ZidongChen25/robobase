"""CPU-only checks for the R2 ``bc-policy`` CQN-AS variant.

Covers the R2 verification protocol for
``robobase/method/cqn_as_bc_policy.py``:

1. flags-off is numerically identical to the frozen pristine ``CQNAS``;
2. flags-on runs and emits the line's metric keys, all finite;
3. flags-on matches the research monolith (``cqn_as_research.CQNAS``)
   configured with the same flags.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from gymnasium import spaces  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402

from robobase.factory import create_agent  # noqa: E402
from robobase.method.cqn_as import CQNAS, cqn_as_spec_from_cfg  # noqa: E402
from robobase.method.cqn_as_bc_policy import (  # noqa: E402
    CQNASBcPolicy,
    cqn_as_bc_policy_spec_from_cfg,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = str((REPO_ROOT / "robobase" / "cfgs").resolve())

ACTION_SEQUENCE = 4
ACTION_DIM = 8
LOW_DIM = 5
RGB_KEY = "rgb_head"
RGB_SHAPE = (1, 3, 84, 84)
BATCH = 2

_LINE_FLAGS = (
    "separate_bc_policy",
    "bc_policy_stop_gradient",
    "distinct_policy_encoder",
    "td_target_action_source",
    "demo_behavior_force_probability",
    "freeze_bc_policy",
    "bc_policy_mode",
    "frozen_policy_snapshot",
)


def _compose(method: str, *overrides: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(
            config_name="robobase_config",
            overrides=[
                f"method={method}",
                "num_train_envs=2",
                "num_eval_envs=2",
                "num_explore_steps=0",
                "backend.jit=false",
                "backend.platform=cpu",
                "method.model.hidden_dims=[32,32]",
                "method.atoms=11",
                f"action_sequence={ACTION_SEQUENCE}",
                *overrides,
            ],
        )


def _spaces(*, pixels: bool):
    obs = {
        "low_dim_state": spaces.Box(
            -np.inf, np.inf, shape=(1, LOW_DIM), dtype=np.float32
        )
    }
    if pixels:
        obs[RGB_KEY] = spaces.Box(0, 255, shape=RGB_SHAPE, dtype=np.uint8)
    action_space = spaces.Box(
        -1.0, 1.0, shape=(ACTION_SEQUENCE, ACTION_DIM), dtype=np.float32
    )
    return spaces.Dict(obs), action_space


def _observation(*, pixels: bool):
    rng = np.random.default_rng(3)
    obs = {
        "low_dim_state": rng.normal(size=(BATCH, 1, LOW_DIM)).astype(
            np.float32
        ),
    }
    if pixels:
        obs[RGB_KEY] = rng.integers(
            0, 256, size=(BATCH, *RGB_SHAPE), dtype=np.uint8
        )
    return obs


def _batch(*, pixels: bool, demo=None, next_action=None):
    rng = np.random.default_rng(7)
    batch = {
        "low_dim_state": rng.normal(size=(BATCH, 1, LOW_DIM)).astype(
            np.float32
        ),
        "low_dim_state_tp1": rng.normal(size=(BATCH, 1, LOW_DIM)).astype(
            np.float32
        ),
        "action": rng.uniform(
            -1.0, 1.0, size=(BATCH, ACTION_SEQUENCE, ACTION_DIM)
        ).astype(np.float32),
        "reward": rng.normal(size=(BATCH,)).astype(np.float32),
        "discount": np.full((BATCH,), 0.99, dtype=np.float32),
        "terminal": np.zeros((BATCH,), dtype=bool),
        "truncated": np.zeros((BATCH,), dtype=bool),
        "demo": (
            np.ones((BATCH,), dtype=np.uint8)
            if demo is None
            else np.asarray(demo, dtype=np.uint8)
        ),
    }
    if next_action is not None:
        batch["action_tp1"] = np.asarray(next_action, dtype=np.float32)
    if pixels:
        batch[RGB_KEY] = rng.integers(
            0, 256, size=(BATCH, *RGB_SHAPE), dtype=np.uint8
        )
        batch[f"{RGB_KEY}_tp1"] = rng.integers(
            0, 256, size=(BATCH, *RGB_SHAPE), dtype=np.uint8
        )
    return batch


def _common_kwargs(cfg, observation_space, action_space, *, jit=False):
    return dict(
        observation_space=observation_space,
        action_space=action_space,
        num_train_envs=cfg.num_train_envs,
        num_eval_envs=cfg.num_eval_envs,
        replay_alpha=cfg.replay.alpha,
        replay_beta=cfg.replay.beta,
        frame_stack_on_channel=cfg.frame_stack_on_channel,
        jit=jit,
        platform="cpu",
        seed=int(cfg.seed),
    )


def _pristine_kwargs(spec):
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
        model=spec.model,
    )


def _make_variant(cfg, observation_space, action_space, *, jit=False):
    spec = cqn_as_bc_policy_spec_from_cfg(cfg)
    return CQNASBcPolicy(
        **_pristine_kwargs(spec),
        separate_bc_policy=spec.separate_bc_policy,
        bc_policy_stop_gradient=spec.bc_policy_stop_gradient,
        distinct_policy_encoder=spec.distinct_policy_encoder,
        td_target_action_source=spec.td_target_action_source,
        demo_behavior_force_probability=(
            spec.demo_behavior_force_probability
        ),
        freeze_bc_policy=spec.freeze_bc_policy,
        bc_policy_mode=spec.bc_policy_mode,
        frozen_policy_snapshot=spec.frozen_policy_snapshot,
        **_common_kwargs(cfg, observation_space, action_space, jit=jit),
    )


def _make_pristine(cfg, observation_space, action_space):
    spec = cqn_as_spec_from_cfg(cfg)
    return CQNAS(
        **_pristine_kwargs(spec),
        **_common_kwargs(cfg, observation_space, action_space),
    )


def _shapes(tree):
    return jax.tree.map(lambda leaf: tuple(np.shape(leaf)), tree)


def _tree_changed(before, after) -> bool:
    return any(
        not np.allclose(np.asarray(a), np.asarray(b))
        for a, b in zip(
            jax.tree.leaves(before), jax.tree.leaves(after), strict=True
        )
    )


def _assert_all_finite(metrics: dict) -> None:
    for key, value in metrics.items():
        array = np.asarray(value, dtype=np.float64)
        assert np.all(np.isfinite(array)), f"{key} is not finite: {array}"


# ----------------------------------------------------------------------
# 1. flags-off == pristine CQN-AS
# ----------------------------------------------------------------------
def test_bc_policy_variant_defaults_match_pristine_cqn_as():
    cfg = _compose("cqn_as_bc_policy")
    observation_space, action_space = _spaces(pixels=False)

    variant = _make_variant(cfg, observation_space, action_space)
    pristine = _make_pristine(cfg, observation_space, action_space)

    assert set(variant.params) == set(pristine.params) == {"critic"}
    assert _shapes(variant.params) == _shapes(pristine.params)
    assert not _tree_changed(pristine.params, variant.params)

    variant.logging = True
    pristine.logging = True
    variant_metrics = variant.update(iter([_batch(pixels=False)]), step=1)
    pristine_metrics = pristine.update(iter([_batch(pixels=False)]), step=1)

    assert _shapes(variant.params) == _shapes(pristine.params)
    assert variant_metrics["critic_loss"] == pytest.approx(
        pristine_metrics["critic_loss"], abs=1e-6
    )
    assert not _tree_changed(pristine.params, variant.params)
    assert not _tree_changed(
        pristine.target_critic_params, variant.target_critic_params
    )


def test_bc_policy_variant_defaults_match_pristine_act():
    cfg = _compose("cqn_as_bc_policy")
    observation_space, action_space = _spaces(pixels=False)

    variant = _make_variant(cfg, observation_space, action_space)
    pristine = _make_pristine(cfg, observation_space, action_space)
    observation = _observation(pixels=False)

    variant_action = np.asarray(
        variant.act(observation, step=100, eval_mode=True)
    )
    pristine_action = np.asarray(
        pristine.act(observation, step=100, eval_mode=True)
    )
    assert variant_action.shape == pristine_action.shape
    np.testing.assert_allclose(variant_action, pristine_action, atol=1e-6)


# ----------------------------------------------------------------------
# 2. flags-on sanity
# ----------------------------------------------------------------------
def test_separate_bc_policy_trains_policy_head_with_td_only_critic():
    cfg = _compose(
        "cqn_as_bc_policy",
        "method.separate_bc_policy=true",
        "method.bc_policy_stop_gradient=true",
        "method.critic_lambda=0.0",
        "method.weight_decay=0.0",
    )
    observation_space, action_space = _spaces(pixels=False)
    agent = _make_variant(cfg, observation_space, action_space)

    assert "policy" in agent.params
    assert "policy_encoder" not in agent.params

    action = np.asarray(
        agent.act(_observation(pixels=False), step=100, eval_mode=True)
    )
    assert action.shape == (BATCH, ACTION_SEQUENCE, ACTION_DIM)
    assert np.all(np.isfinite(action))

    critic_before = jax.tree.map(np.asarray, agent.params["critic"])
    policy_before = jax.tree.map(np.asarray, agent.params["policy"])
    agent.logging = True
    metrics = agent.update(iter([_batch(pixels=False)]), step=1)

    _assert_all_finite(metrics)
    for key in (
        "policy_bc_loss",
        "policy_ce",
        "policy_demo_top1",
        "policy_entropy",
        "policy_encoder_grad_norm",
        "td_critic_loss",
        "total_loss",
        "critic_q_span",
    ):
        assert key in metrics, sorted(metrics)
    # critic_lambda=0 leaves the TD critic untouched; only the BC head learns.
    assert not _tree_changed(critic_before, agent.params["critic"])
    assert _tree_changed(policy_before, agent.params["policy"])
    assert metrics["critic_loss"] == pytest.approx(0.0)
    assert metrics["policy_bc_loss"] > 0.0
    assert 0.0 <= metrics["policy_demo_top1"] <= 1.0


def test_separate_bc_policy_bc_lambda_zero_is_rejected():
    cfg = _compose(
        "cqn_as_bc_policy",
        "method.separate_bc_policy=true",
        "method.bc_lambda=0.0",
    )
    observation_space, action_space = _spaces(pixels=False)
    with pytest.raises(ValueError, match="requires bc_lambda"):
        _make_variant(cfg, observation_space, action_space)


def test_distinct_policy_encoder_isolates_the_value_tower():
    cfg = _compose(
        "cqn_as_bc_policy",
        "pixels=true",
        "method.separate_bc_policy=true",
        "method.distinct_policy_encoder=true",
        "method.bc_policy_stop_gradient=false",
        "method.critic_lambda=0.0",
        "method.weight_decay=0.0",
    )
    observation_space, action_space = _spaces(pixels=True)
    agent = _make_variant(cfg, observation_space, action_space)

    assert "policy_encoder" in agent.params
    value_encoder_before = jax.tree.map(np.asarray, agent.params["encoder"])
    policy_encoder_before = jax.tree.map(
        np.asarray, agent.params["policy_encoder"]
    )
    # Both towers must start from the same visual initialization.
    assert not _tree_changed(value_encoder_before, policy_encoder_before)

    agent.logging = True
    batch = _batch(pixels=True)
    # The policy head is zero-initialized, so the first update cannot yet
    # backpropagate a feature gradient; the second exercises the tower.
    agent.update(iter([batch]), step=1)
    metrics = agent.update(iter([batch]), step=2)

    _assert_all_finite(metrics)
    assert not _tree_changed(value_encoder_before, agent.params["encoder"])
    assert _tree_changed(
        policy_encoder_before, agent.params["policy_encoder"]
    )
    assert metrics["policy_encoder_grad_norm"] > 0.0
    assert metrics["critic_loss"] == pytest.approx(0.0)
    assert metrics["policy_bc_loss"] > 0.0


def test_distinct_policy_encoder_requires_separate_bc_policy():
    cfg = _compose(
        "cqn_as_bc_policy",
        "method.distinct_policy_encoder=true",
    )
    observation_space, action_space = _spaces(pixels=False)
    with pytest.raises(ValueError, match="requires separate_bc_policy"):
        _make_variant(cfg, observation_space, action_space)


def test_demo_behavior_forcing_overrides_the_bootstrap_action():
    rng = np.random.default_rng(11)
    next_action = rng.uniform(
        -1.0, 1.0, size=(BATCH, ACTION_SEQUENCE, ACTION_DIM)
    ).astype(np.float32)
    observation_space, action_space = _spaces(pixels=False)

    forced_cfg = _compose(
        "cqn_as_bc_policy",
        "method.td_target_action_source=critic_replay_max",
        "method.demo_behavior_force_probability=1.0",
    )
    forced = _make_variant(forced_cfg, observation_space, action_space)
    forced.logging = True
    # The critic heads are zero-initialized, so at step 1 the greedy and the
    # behaviour chunk score identically and candidate-max is degenerate. The
    # second update exercises a genuinely discriminating comparison.
    forced.update(
        iter([_batch(pixels=False, demo=[1, 0], next_action=next_action)]),
        step=1,
    )
    forced_metrics = forced.update(
        iter([_batch(pixels=False, demo=[1, 0], next_action=next_action)]),
        step=2,
    )

    _assert_all_finite(forced_metrics)
    for key in (
        "behavior_candidate_fraction",
        "behavior_candidate_score",
        "greedy_candidate_score",
        "behavior_minus_greedy_q",
        "demo_behavior_force_fraction",
        "demo_behavior_force_probability",
    ):
        assert key in forced_metrics, sorted(forced_metrics)
    # One of the two samples is a demo and p=1, so exactly half are forced.
    assert forced_metrics["demo_behavior_force_fraction"] == pytest.approx(0.5)
    assert forced_metrics["demo_behavior_force_probability"] == pytest.approx(
        1.0
    )
    # The behaviour chunk scores strictly below greedy here, so the only
    # sample bootstrapped from it is the forced demo.
    assert forced_metrics["behavior_minus_greedy_q"] < 0.0
    assert forced_metrics["behavior_candidate_fraction"] == pytest.approx(0.5)

    unforced_cfg = _compose(
        "cqn_as_bc_policy",
        "method.td_target_action_source=critic_replay_max",
        "method.demo_behavior_force_probability=0.0",
    )
    unforced = _make_variant(unforced_cfg, observation_space, action_space)
    unforced.logging = True
    unforced.update(
        iter([_batch(pixels=False, demo=[1, 0], next_action=next_action)]),
        step=1,
    )
    unforced_metrics = unforced.update(
        iter([_batch(pixels=False, demo=[1, 0], next_action=next_action)]),
        step=2,
    )
    _assert_all_finite(unforced_metrics)
    assert unforced_metrics["demo_behavior_force_fraction"] == pytest.approx(
        0.0
    )
    assert unforced_metrics["behavior_candidate_fraction"] == pytest.approx(
        0.0
    )


def test_candidate_target_requires_next_action_in_batch():
    cfg = _compose(
        "cqn_as_bc_policy",
        "method.td_target_action_source=critic_replay_max",
    )
    observation_space, action_space = _spaces(pixels=False)
    agent = _make_variant(cfg, observation_space, action_space)
    with pytest.raises(KeyError, match="action_tp1"):
        agent.update(iter([_batch(pixels=False)]), step=1)


def test_demo_behavior_forcing_requires_its_carrier():
    cfg = _compose(
        "cqn_as_bc_policy",
        "method.demo_behavior_force_probability=1.0",
    )
    observation_space, action_space = _spaces(pixels=False)
    with pytest.raises(ValueError, match="critic_replay_max"):
        _make_variant(cfg, observation_space, action_space)


@pytest.mark.parametrize(
    "override, match",
    [
        ("method.freeze_bc_policy=true", "freeze_bc_policy"),
        ("method.bc_policy_mode=legacy_c51", "bc_policy_mode"),
        ("method.td_target_action_source=replay_next", "td-variants"),
        ("method.td_target_action_source=bc_policy", "td-variants"),
        ("method.td_target_action_source=policy_value", "td-variants"),
    ],
)
def test_out_of_scope_flags_fail_loudly(override, match):
    cfg = _compose("cqn_as_bc_policy", override)
    observation_space, action_space = _spaces(pixels=False)
    with pytest.raises(NotImplementedError, match=match):
        _make_variant(cfg, observation_space, action_space)


@pytest.mark.parametrize(
    "overrides, needs_next_action",
    [
        ((), False),
        (("method.separate_bc_policy=true",), False),
        (
            (
                "method.td_target_action_source=critic_replay_max",
                "method.demo_behavior_force_probability=1.0",
            ),
            True,
        ),
    ],
)
def test_every_branch_runs_under_jit(overrides, needs_next_action):
    """All three update graphs must also trace/compile under jax.jit."""

    cfg = _compose("cqn_as_bc_policy", *overrides)
    observation_space, action_space = _spaces(pixels=False)
    agent = _make_variant(cfg, observation_space, action_space, jit=True)
    agent.logging = True

    action = np.asarray(
        agent.act(_observation(pixels=False), step=100, eval_mode=False)
    )
    assert action.shape == (BATCH, ACTION_SEQUENCE, ACTION_DIM)
    assert np.all(np.isfinite(action))

    next_action = None
    if needs_next_action:
        next_action = np.random.default_rng(11).uniform(
            -1.0, 1.0, size=(BATCH, ACTION_SEQUENCE, ACTION_DIM)
        ).astype(np.float32)
    metrics = agent.update(
        iter([_batch(pixels=False, demo=[1, 0], next_action=next_action)]),
        step=1,
    )
    _assert_all_finite(metrics)


def test_variant_yaml_exposes_every_line_flag():
    cfg = _compose("cqn_as_bc_policy")
    assert cfg.method.name == "cqn_as_bc_policy"
    assert (
        cfg.method._target_
        == "robobase.method.cqn_as_bc_policy.CQNASBcPolicy"
    )
    for flag in _LINE_FLAGS:
        assert flag in cfg.method, flag
    assert not cfg.method.separate_bc_policy
    assert not cfg.method.bc_policy_stop_gradient
    assert not cfg.method.distinct_policy_encoder
    assert cfg.method.td_target_action_source == "critic"
    assert cfg.method.demo_behavior_force_probability == 0.0
    assert not cfg.method.freeze_bc_policy
    assert cfg.method.bc_policy_mode == "behavior_logits"
    assert cfg.method.frozen_policy_snapshot is None


# ----------------------------------------------------------------------
# 3. flags-on == research monolith
# ----------------------------------------------------------------------
def test_separate_bc_policy_matches_research_monolith():
    flags = (
        "method.separate_bc_policy=true",
        "method.bc_policy_stop_gradient=false",
        "method.critic_lambda=0.1",
        "method.weight_decay=0.1",
    )
    observation_space, action_space = _spaces(pixels=False)

    variant_cfg = _compose("cqn_as_bc_policy", *flags)
    variant = _make_variant(variant_cfg, observation_space, action_space)

    research_cfg = _compose("cqn_as", *flags)
    research = create_agent(
        research_cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    assert type(research).__module__ == "robobase.method.cqn_as_research"

    assert _shapes(variant.params) == _shapes(research.params)
    assert not _tree_changed(research.params, variant.params)

    variant.logging = True
    research.logging = True
    variant_metrics = variant.update(iter([_batch(pixels=False)]), step=1)
    research_metrics = research.update(iter([_batch(pixels=False)]), step=1)

    _assert_all_finite(variant_metrics)
    assert variant_metrics["critic_loss"] == pytest.approx(
        research_metrics["critic_loss"], abs=1e-5
    )
    assert variant_metrics["policy_bc_loss"] == pytest.approx(
        research_metrics["policy_bc_loss"], abs=1e-5
    )
    assert variant_metrics["policy_demo_top1"] == pytest.approx(
        research_metrics["policy_demo_top1"], abs=1e-5
    )
    assert not _tree_changed(research.params, variant.params)


def test_demo_behavior_forcing_matches_research_monolith():
    rng = np.random.default_rng(11)
    next_action = rng.uniform(
        -1.0, 1.0, size=(BATCH, ACTION_SEQUENCE, ACTION_DIM)
    ).astype(np.float32)
    flags = (
        "method.td_target_action_source=critic_replay_max",
        "method.demo_behavior_force_probability=1.0",
        "replay.include_next_action=true",
    )
    observation_space, action_space = _spaces(pixels=False)

    variant_cfg = _compose("cqn_as_bc_policy", *flags)
    variant = _make_variant(variant_cfg, observation_space, action_space)

    research_cfg = _compose("cqn_as", *flags)
    research = create_agent(
        research_cfg,
        observation_space=observation_space,
        action_space=action_space,
    )

    def _make_batch():
        return _batch(pixels=False, demo=[1, 0], next_action=next_action)

    variant.logging = True
    research.logging = True
    # Step 1 is degenerate (zero-initialized heads); compare on step 2 where
    # candidate-max actually discriminates.
    variant.update(iter([_make_batch()]), step=1)
    research.update(iter([_make_batch()]), step=1)
    variant_metrics = variant.update(iter([_make_batch()]), step=2)
    research_metrics = research.update(iter([_make_batch()]), step=2)

    _assert_all_finite(variant_metrics)
    assert variant_metrics["behavior_minus_greedy_q"] < 0.0
    for key in (
        "critic_loss",
        "behavior_candidate_fraction",
        "behavior_candidate_score",
        "greedy_candidate_score",
        "behavior_minus_greedy_q",
        "demo_behavior_force_fraction",
    ):
        assert variant_metrics[key] == pytest.approx(
            research_metrics[key], abs=1e-5
        ), key
