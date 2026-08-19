"""CPU-only R2 checks for the ``mc-rct`` CQN-AS research line.

Verification contract (R2_COMMON_BRIEF):

1. flags-off ``CQNASMcRct`` is numerically identical to the frozen pristine
   ``CQNAS`` after one ``update()`` (same seed, same synthetic batch);
2. flags-on runs ``act()`` + ``update()`` finite and emits the line's metrics;
3. flags-on matches ``cqn_as_research.CQNAS`` configured the same way.

The behavioural spec is ``tests/unit/test_cqn_as.py`` (``mc_lower_bound`` and
``mc_return`` tests); its assertions are adapted here, that file is not edited.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from flax.traverse_util import flatten_dict  # noqa: E402
from gymnasium import spaces  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402

from robobase.factory import create_agent  # noqa: E402
from robobase.method.cqn_as import CQNAS, cqn_as_spec_from_cfg  # noqa: E402
from robobase.method.cqn_as_mc_rct import (  # noqa: E402
    CQNASMcRct,
    cqn_as_mc_rct_spec_from_cfg,
)

CONFIG_DIR = str((Path(__file__).parents[2] / "robobase" / "cfgs").resolve())

ACTION_SEQUENCE = 3
ACTION_DIM = 2
LOW_DIM = 5
BATCH = 4
RGB_KEY = "rgb_front"
RGB_SHAPE = (1, 3, 84, 84)


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
                "backend.jit=false",
                "backend.platform=cpu",
                "method.model.hidden_dims=[32,32]",
                "method.atoms=11",
                *overrides,
            ],
        )


def _spaces(*, pixels: bool = False):
    observation = {
        "low_dim_state": spaces.Box(
            -np.inf, np.inf, shape=(1, LOW_DIM), dtype=np.float32
        )
    }
    if pixels:
        observation[RGB_KEY] = spaces.Box(
            0, 255, shape=RGB_SHAPE, dtype=np.uint8
        )
    action_space = spaces.Box(
        -1.0, 1.0, shape=(ACTION_SEQUENCE, ACTION_DIM), dtype=np.float32
    )
    return spaces.Dict(observation), action_space


def _batch(*, batch_size: int = BATCH, pixels: bool = False):
    rng = np.random.default_rng(7)
    batch = {
        "low_dim_state": rng.normal(size=(batch_size, 1, LOW_DIM)).astype(
            np.float32
        ),
        "low_dim_state_tp1": rng.normal(size=(batch_size, 1, LOW_DIM)).astype(
            np.float32
        ),
        "action": rng.uniform(
            -1.0, 1.0, size=(batch_size, ACTION_SEQUENCE, ACTION_DIM)
        ).astype(np.float32),
        "reward": rng.normal(size=(batch_size,)).astype(np.float32),
        "discount": np.full((batch_size,), 0.99, dtype=np.float32),
        "terminal": np.zeros((batch_size,), dtype=bool),
        "truncated": np.zeros((batch_size,), dtype=bool),
        "demo": np.ones((batch_size,), dtype=np.uint8),
    }
    if pixels:
        batch[RGB_KEY] = np.zeros((batch_size, *RGB_SHAPE), dtype=np.uint8)
        batch[f"{RGB_KEY}_tp1"] = np.ones(
            (batch_size, *RGB_SHAPE), dtype=np.uint8
        )
    return batch


def _observation(*, batch_size: int = 1, pixels: bool = False):
    rng = np.random.default_rng(3)
    observation = {
        "low_dim_state": rng.normal(size=(batch_size, 1, LOW_DIM)).astype(
            np.float32
        )
    }
    if pixels:
        observation[RGB_KEY] = rng.integers(
            0, 256, size=(batch_size, *RGB_SHAPE), dtype=np.uint8
        )
    return observation


def _pristine_kwargs(spec, cfg, observation_space, action_space):
    """The exact argument set ``factory.create_agent`` feeds pristine CQN-AS."""

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
        observation_space=observation_space,
        action_space=action_space,
        num_train_envs=cfg.num_train_envs,
        num_eval_envs=cfg.num_eval_envs,
        replay_alpha=cfg.replay.alpha,
        replay_beta=cfg.replay.beta,
        frame_stack_on_channel=cfg.frame_stack_on_channel,
        jit=bool(cfg.backend.jit),
        platform=str(cfg.backend.platform),
        seed=int(cfg.seed),
        update_block_every_steps=int(cfg.backend.update_block_every_steps),
    )


def _official_agent(*overrides: str, pixels: bool = False):
    cfg = _compose("cqn_as_official", *overrides)
    observation_space, action_space = _spaces(pixels=pixels)
    spec = cqn_as_spec_from_cfg(cfg)
    return CQNAS(
        **_pristine_kwargs(spec, cfg, observation_space, action_space)
    )


def _mc_rct_agent(*overrides: str, pixels: bool = False):
    cfg = _compose("cqn_as_mc_rct", *overrides)
    observation_space, action_space = _spaces(pixels=pixels)
    spec = cqn_as_mc_rct_spec_from_cfg(cfg)
    return CQNASMcRct(
        **_pristine_kwargs(spec, cfg, observation_space, action_space),
        mc_return_weight=spec.mc_return_weight,
        mc_lower_bound_target=spec.mc_lower_bound_target,
        mc_return_stop_gradient_encoder=spec.mc_return_stop_gradient_encoder,
        mc_return_value_only=spec.mc_return_value_only,
        causal_rct_weight=spec.causal_rct_weight,
        causal_rct_level=spec.causal_rct_level,
        cv_rct_weight=spec.cv_rct_weight,
        cv_rct_level=spec.cv_rct_level,
        cv_rct_baseline=spec.cv_rct_baseline,
    )


def _run_updates(agent, batch, count):
    """Run ``count`` updates and return the last metrics dict.

    The critic output heads are zero-initialised, so at step 1 the C51 cross
    entropy is ``log(atoms)`` for *any* normalised target and the loss cannot
    discriminate target composition. Every comparison that must actually see
    the MC branch therefore warms the heads up first.
    """

    agent.logging = True
    metrics = {}
    for index in range(count):
        metrics = agent.update(iter([dict(batch)]), step=1 + index)
    return metrics


def _tree_changed(before, after):
    before_leaves, before_tree = jax.tree.flatten(before)
    after_leaves, after_tree = jax.tree.flatten(after)
    assert before_tree == after_tree
    return any(
        not np.allclose(np.asarray(left), np.asarray(right))
        for left, right in zip(before_leaves, after_leaves, strict=True)
    )


# ---------------------------------------------------------------------------
# 1. Flags off is bit-for-bit the frozen pristine class.
# ---------------------------------------------------------------------------


def test_mc_rct_flags_off_matches_pristine_cqn_as():
    official = _official_agent()
    variant = _mc_rct_agent()

    official_shapes = jax.tree.map(lambda leaf: leaf.shape, official.params)
    variant_shapes = jax.tree.map(lambda leaf: leaf.shape, variant.params)
    assert official_shapes == variant_shapes

    assert not variant._canonical_mc_anchor
    assert not variant._uses_canonical_mc_returns

    # Three updates: the first moves the zero-initialised heads, the rest run
    # on a non-degenerate critic where any divergence would compound.
    official_metrics = _run_updates(official, _batch(), 3)
    variant_metrics = _run_updates(variant, _batch(), 3)

    assert set(variant_metrics) == set(official_metrics)
    np.testing.assert_allclose(
        variant_metrics["critic_loss"],
        official_metrics["critic_loss"],
        atol=1e-6,
        rtol=0.0,
    )
    for key in ("entropy", "target_entropy", "loss_coeff"):
        np.testing.assert_allclose(
            variant_metrics[key], official_metrics[key], atol=1e-6, rtol=0.0
        )

    official_leaves, official_tree = jax.tree.flatten(official.params)
    variant_leaves, variant_tree = jax.tree.flatten(variant.params)
    assert official_tree == variant_tree
    for official_leaf, variant_leaf in zip(
        official_leaves, variant_leaves, strict=True
    ):
        np.testing.assert_allclose(
            np.asarray(variant_leaf),
            np.asarray(official_leaf),
            atol=1e-6,
            rtol=0.0,
        )


def test_mc_rct_flags_off_needs_no_mc_return_element():
    """The flags-off update keeps the pristine 12-argument replay contract."""

    variant = _mc_rct_agent()
    batch = _batch()
    assert "mc_return" not in batch
    variant.logging = True
    metrics = variant.update(iter([batch]), step=1)
    assert "mc_return_loss" not in metrics
    assert "mc_lower_bound_fraction" not in metrics


# ---------------------------------------------------------------------------
# 2. Flags on: finite, and the line's metrics appear.
# ---------------------------------------------------------------------------


def test_mc_rct_flags_on_act_and_update_are_finite():
    variant = _mc_rct_agent(
        "method.mc_lower_bound_target=true",
        "method.mc_return_weight=0.1",
    )
    assert variant._canonical_mc_anchor
    assert variant._uses_canonical_mc_returns

    action = np.asarray(
        variant.act(_observation(), step=100, eval_mode=False)
    )
    assert action.shape == (1, ACTION_SEQUENCE, ACTION_DIM)
    assert np.all(np.isfinite(action))

    batch = _batch()
    batch["mc_return"] = np.linspace(0.2, 0.8, BATCH).astype(np.float32)
    variant.logging = True
    metrics = variant.update(iter([batch]), step=1)

    for key in (
        "mc_return_loss",
        "mc_return_mae",
        "mc_lower_bound_fraction",
        "mc_return_mean",
    ):
        assert key in metrics, sorted(metrics)
    for key, value in metrics.items():
        assert np.all(np.isfinite(np.asarray(value))), key
    assert metrics["mc_return_mean"] == pytest.approx(0.5)
    assert metrics["mc_return_loss"] > 0.0
    assert metrics["mc_return_mae"] > 0.0


def test_mc_lower_bound_is_a_target_not_an_extra_loss():
    """Adapted from ``test_cqn_as_mc_lower_bound_is_reward_only_...``.

    A zero-reward batch whose completed-episode return is strictly positive
    must be routed entirely through the MC branch, must NOT emit an auxiliary
    ``mc_return_loss``, and must be independent of demo identity.
    """

    demo_agent = _mc_rct_agent("method.mc_lower_bound_target=true")
    online_agent = _mc_rct_agent("method.mc_lower_bound_target=true")
    assert not demo_agent._canonical_mc_anchor
    assert demo_agent._uses_canonical_mc_returns

    demo_batch = _batch()
    demo_batch["reward"][:] = 0.0
    demo_batch["mc_return"] = np.linspace(0.2, 0.8, BATCH).astype(np.float32)
    online_batch = {
        key: np.array(value, copy=True) for key, value in demo_batch.items()
    }
    online_batch["demo"][:] = 0

    demo_agent.logging = True
    online_agent.logging = True
    demo_metrics = demo_agent.update(iter([demo_batch]), step=1)
    online_metrics = online_agent.update(iter([online_batch]), step=1)

    assert demo_metrics["mc_lower_bound_fraction"] == pytest.approx(1.0)
    assert demo_metrics["mc_return_mean"] == pytest.approx(0.5)
    assert "mc_return_loss" not in demo_metrics

    # bc_lambda is a demo-gated loss, so demo/online params only agree when
    # the pristine BC term is off; the MC branch itself is demo-blind.
    demo_metrics.pop("backend/update_time_sec")
    online_metrics.pop("backend/update_time_sec")
    assert demo_metrics["mc_lower_bound_fraction"] == pytest.approx(
        online_metrics["mc_lower_bound_fraction"]
    )
    assert demo_metrics["mc_return_mean"] == pytest.approx(
        online_metrics["mc_return_mean"]
    )


def test_mc_lower_bound_only_bites_above_the_bellman_target():
    """The lower bound is a max: inert below the backup, active above it."""

    floor_batch = _batch()
    # v_min = -2.0: a return pinned at the support floor can never exceed the
    # bootstrapped expectation, so the target must stay pure TD.
    floor_batch["mc_return"] = np.full((BATCH,), -2.0, dtype=np.float32)
    ceiling_batch = _batch()
    # v_max = 2.0: a return pinned at the support ceiling always wins.
    ceiling_batch["mc_return"] = np.full((BATCH,), 2.0, dtype=np.float32)

    floor_metrics = _run_updates(
        _mc_rct_agent("method.mc_lower_bound_target=true"), floor_batch, 3
    )
    ceiling_metrics = _run_updates(
        _mc_rct_agent("method.mc_lower_bound_target=true"), ceiling_batch, 3
    )
    baseline_metrics = _run_updates(_mc_rct_agent(), _batch(), 3)

    assert floor_metrics["mc_lower_bound_fraction"] == pytest.approx(0.0)
    assert ceiling_metrics["mc_lower_bound_fraction"] == pytest.approx(1.0)

    # Inert branch reproduces the pure-TD agent exactly.
    np.testing.assert_allclose(
        floor_metrics["critic_loss"],
        baseline_metrics["critic_loss"],
        atol=1e-6,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        floor_metrics["target_entropy"],
        baseline_metrics["target_entropy"],
        atol=1e-6,
        rtol=0.0,
    )

    # The active branch really does swap the bootstrapped C51 target for a
    # projected point mass: target entropy collapses from the broad Bellman
    # distribution to (numerically) zero. This is visible on update 1 and is
    # independent of optimiser dynamics.
    assert baseline_metrics["target_entropy"] > 2.0
    assert abs(ceiling_metrics["target_entropy"]) < 1e-3
    # ...and that propagates into the loss once the zero-init heads have moved.
    assert (
        abs(ceiling_metrics["critic_loss"] - baseline_metrics["critic_loss"])
        > 1e-5
    )


def test_mc_return_value_only_preserves_advantage_parameters():
    """Adapted from ``test_cqn_as_mc_return_value_only_preserves_...``.

    ``critic_lambda=0`` + ``bc_lambda=0`` + ``weight_decay=0`` isolate the MC
    anchor as the only gradient source, so the advantage stream must be frozen
    while the value stream moves.
    """

    variant = _mc_rct_agent(
        "method.mc_return_weight=0.5",
        "method.mc_return_value_only=true",
        "method.critic_lambda=0.0",
        "method.bc_lambda=0.0",
        "method.weight_decay=0.0",
    )
    batch = _batch()
    batch["demo"] = np.zeros_like(batch["demo"])
    batch["mc_return"] = np.linspace(0.1, 0.9, BATCH).astype(np.float32)
    before = flatten_dict(variant.params["critic"])

    variant.update(iter([batch]), step=1)

    after = flatten_dict(variant.params["critic"])
    advantage_paths = [
        path for path in before if any("advantage" in str(part) for part in path)
    ]
    value_paths = [
        path for path in before if any("value" in str(part) for part in path)
    ]
    assert advantage_paths
    assert value_paths
    assert all(
        np.allclose(np.asarray(before[path]), np.asarray(after[path]))
        for path in advantage_paths
    )
    assert any(
        not np.allclose(np.asarray(before[path]), np.asarray(after[path]))
        for path in value_paths
    )


def test_mc_return_stop_gradient_encoder_protects_the_pixel_encoder():
    """Adapted from ``test_cqn_as_mc_return_can_protect_policy_trained_...``."""

    variant = _mc_rct_agent(
        "pixels=true",
        "method.mc_return_weight=0.5",
        "method.mc_return_stop_gradient_encoder=true",
        "method.critic_lambda=0.0",
        "method.bc_lambda=0.0",
        "method.weight_decay=0.0",
        pixels=True,
    )
    assert "encoder" in variant.params
    batch = _batch(batch_size=2, pixels=True)
    batch["demo"] = np.zeros_like(batch["demo"])
    batch["mc_return"] = np.asarray([0.2, 0.8], dtype=np.float32)
    encoder_before = jax.tree.map(np.asarray, variant.params["encoder"])
    critic_before = jax.tree.map(np.asarray, variant.params["critic"])

    variant.update(iter([batch]), step=1)

    assert _tree_changed(critic_before, variant.params["critic"])
    assert not _tree_changed(encoder_before, variant.params["encoder"])


def test_mc_return_without_stop_gradient_does_train_the_pixel_encoder():
    """Control for the previous test: the guard is what freezes the encoder."""

    variant = _mc_rct_agent(
        "pixels=true",
        "method.mc_return_weight=0.5",
        "method.mc_return_stop_gradient_encoder=false",
        "method.critic_lambda=0.0",
        "method.bc_lambda=0.0",
        "method.weight_decay=0.0",
        pixels=True,
    )
    batch = _batch(batch_size=2, pixels=True)
    batch["demo"] = np.zeros_like(batch["demo"])
    batch["mc_return"] = np.asarray([0.2, 0.8], dtype=np.float32)
    encoder_before = jax.tree.map(np.asarray, variant.params["encoder"])

    # The critic heads are zero-initialised, so the first update produces no
    # feature gradient; the second exercises the now-nonzero heads.
    variant.update(iter([batch]), step=1)
    variant.update(iter([batch]), step=2)

    assert _tree_changed(encoder_before, variant.params["encoder"])


# ---------------------------------------------------------------------------
# 3. Flags on matches the research monolith.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flags",
    [
        ("method.mc_lower_bound_target=true",),
        ("method.mc_return_weight=0.1",),
        ("method.mc_lower_bound_target=true", "method.mc_return_weight=0.1"),
    ],
)
def test_mc_rct_flags_on_matches_research_monolith(flags):
    observation_space, action_space = _spaces()
    research = create_agent(
        _compose("cqn_as", *flags),
        observation_space=observation_space,
        action_space=action_space,
    )
    variant = _mc_rct_agent(*flags)

    batch = _batch()
    # Straddle the support so the lower bound is active on some entries and
    # inert on others; three updates take the heads off their zero init.
    batch["mc_return"] = np.linspace(-1.5, 1.5, BATCH).astype(np.float32)
    research_metrics = _run_updates(research, batch, 3)
    variant_metrics = _run_updates(variant, batch, 3)

    if "mc_lower_bound_fraction" in variant_metrics:
        assert 0.0 < variant_metrics["mc_lower_bound_fraction"] < 1.0

    np.testing.assert_allclose(
        variant_metrics["critic_loss"],
        research_metrics["critic_loss"],
        atol=1e-5,
        rtol=0.0,
    )
    for key in research_metrics:
        if key.startswith("mc_"):
            np.testing.assert_allclose(
                variant_metrics[key],
                research_metrics[key],
                atol=1e-5,
                rtol=0.0,
                err_msg=key,
            )


def test_mc_return_value_only_deliberately_diverges_from_the_research_path():
    """``mc_return_value_only`` is dead code on the research canonical path.

    At ``ff9dfbf`` the flag is read ONLY inside the ``separate_bc_policy``
    branch (``cqn_as_research.py:5003``); the canonical MC anchor in
    ``cqn_research.py:1790-1820`` never looks at it. This line cannot own
    ``separate_bc_policy`` (bc-policy line), so the variant implements the
    documented value-stream semantics on the canonical path instead. The
    resulting mismatch against the monolith is intentional, and this test
    pins it so nobody "fixes" it into silence.
    """

    flags = ("method.mc_return_weight=0.1", "method.mc_return_value_only=true")
    observation_space, action_space = _spaces()
    research = create_agent(
        _compose("cqn_as", *flags),
        observation_space=observation_space,
        action_space=action_space,
    )
    variant = _mc_rct_agent(*flags)

    batch = _batch()
    batch["mc_return"] = np.linspace(-1.5, 1.5, BATCH).astype(np.float32)
    research_metrics = _run_updates(research, batch, 3)
    variant_metrics = _run_updates(variant, batch, 3)

    assert np.isfinite(research_metrics["critic_loss"])
    assert np.isfinite(variant_metrics["critic_loss"])
    # Research ignores the flag, so its loss equals the plain-anchor arm.
    plain = create_agent(
        _compose("cqn_as", "method.mc_return_weight=0.1"),
        observation_space=observation_space,
        action_space=action_space,
    )
    plain_metrics = _run_updates(plain, batch, 3)
    np.testing.assert_allclose(
        research_metrics["critic_loss"],
        plain_metrics["critic_loss"],
        atol=1e-9,
        rtol=0.0,
    )
    # The variant actually honours it, so it must differ from both.
    assert (
        abs(variant_metrics["critic_loss"] - research_metrics["critic_loss"])
        > 1e-6
    )


# ---------------------------------------------------------------------------
# Coupling: the RCT probes are owned by other lines and must fail loudly.
# ---------------------------------------------------------------------------


def test_causal_rct_weight_is_rejected_with_a_pointer_to_direct_scalar_q():
    with pytest.raises(ValueError, match="cqn_direct_q"):
        _mc_rct_agent("method.causal_rct_weight=0.1")


def test_cv_rct_weight_is_rejected_with_a_pointer_to_separate_bc_policy():
    with pytest.raises(ValueError, match="separate_bc_policy"):
        _mc_rct_agent("method.cv_rct_weight=0.0")


def test_rct_probe_flags_still_validate_their_cheap_invariants():
    with pytest.raises(ValueError, match="causal_rct_weight must be"):
        _mc_rct_agent("method.causal_rct_weight=-1.0")
    with pytest.raises(ValueError, match="causal_rct_level must be"):
        _mc_rct_agent("method.causal_rct_level=99")
    with pytest.raises(ValueError, match="cv_rct_level must be"):
        _mc_rct_agent("method.cv_rct_level=99")
    with pytest.raises(ValueError, match="cv_rct_baseline must be"):
        _mc_rct_agent("method.cv_rct_baseline=bogus")
    with pytest.raises(ValueError, match="mc_return_weight must be"):
        _mc_rct_agent("method.mc_return_weight=-1.0")
    with pytest.raises(ValueError, match="requires use_dueling=true"):
        _mc_rct_agent(
            "method.mc_return_weight=0.5",
            "method.mc_return_value_only=true",
            "method.use_dueling=false",
        )
