"""R2 extraction tests for the ``progress-shaping`` CQN-AS research line.

Covers the three contracts of ``R2_COMMON_BRIEF.md``:

1. flags-off is numerically and structurally identical to the pristine
   ``robobase.method.cqn_as.CQNAS``;
2. flags-on runs (``act`` + ``update``), stays finite and emits the line's
   metric keys;
3. flags-on matches the research monolith (``cqn_as_research.CQNAS``) on
   ``critic_loss``.

Plus the behavioral spec adapted from the research-era progress tests in
``tests/unit/test_cqn_as.py`` (Ng-form algebra, telescoping, zero/constant
potential equivalences, success gating, schedule, and the construction-time
refusals).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from flax.core import unfreeze  # noqa: E402
from flax.traverse_util import flatten_dict, unflatten_dict  # noqa: E402
from gymnasium import spaces  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402

from robobase.factory import create_agent  # noqa: E402
from robobase.method.cqn_as import CQNAS, cqn_as_spec_from_cfg  # noqa: E402
from robobase.method.cqn_as_progress_shaping import (  # noqa: E402
    CQNASProgressShaping,
    cqn_as_progress_shaping_spec_from_cfg,
    progress_shaped_rewards,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = str((REPO_ROOT / "robobase" / "cfgs").resolve())

# Mirrors scripts/refactor_equivalence_check.py.
ACTION_SEQUENCE = 4
ACTION_DIM = 8
LOW_DIM = 5
RGB_KEY = "rgb_head"
RGB_SHAPE = (1, 12, 84, 84)
BATCH = 4


def _compose(method: str, *overrides: str, pixels: bool = False):
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
                "method.weight_decay=0.0",
                f"pixels={'true' if pixels else 'false'}",
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
        "low_dim_state": rng.normal(size=(BATCH, 1, LOW_DIM)).astype(np.float32),
    }
    if pixels:
        obs[RGB_KEY] = rng.integers(0, 256, size=(BATCH, *RGB_SHAPE), dtype=np.uint8)
    return obs


def _batch(*, pixels: bool = False, terminal_last: bool = True, seed: int = 7):
    rng = np.random.default_rng(seed)
    terminal = np.zeros((BATCH,), dtype=bool)
    if terminal_last:
        terminal[-1] = True
    batch = {
        "low_dim_state": rng.normal(size=(BATCH, 1, LOW_DIM)).astype(np.float32),
        "low_dim_state_tp1": rng.normal(size=(BATCH, 1, LOW_DIM)).astype(np.float32),
        "action": rng.uniform(
            -1.0, 1.0, size=(BATCH, ACTION_SEQUENCE, ACTION_DIM)
        ).astype(np.float32),
        # Sparse {0, 1} success reward on the terminal transition: the regime
        # the C51 support ([v_min, v_max]) is sized for, so a shaped target
        # only leaves the support if the potential itself pushes it out.
        "reward": terminal.astype(np.float32),
        "discount": np.full((BATCH,), 0.99, dtype=np.float32),
        "terminal": terminal,
        "truncated": np.zeros((BATCH,), dtype=bool),
        "demo": np.ones((BATCH,), dtype=np.uint8),
    }
    if pixels:
        batch[RGB_KEY] = rng.integers(0, 256, size=(BATCH, *RGB_SHAPE), dtype=np.uint8)
        batch[f"{RGB_KEY}_tp1"] = rng.integers(
            0, 256, size=(BATCH, *RGB_SHAPE), dtype=np.uint8
        )
    return batch


def _progress_batch(*, pixels: bool = False, terminal_last: bool = True, seed: int = 7):
    batch = _batch(pixels=pixels, terminal_last=terminal_last, seed=seed)
    batch["progress"] = np.arange(1, BATCH + 1, dtype=np.float32) / BATCH
    batch["progress_valid"] = np.ones((BATCH,), dtype=np.uint8)
    return batch


# The critic's output layers are zero-initialised, so a single update from
# scratch produces a feature-independent loss. Every numeric equivalence check
# therefore runs several updates on distinct batches and compares the last one.
EQUIV_STEPS = 3


def _run_updates(agent, batches, *, first_step: int = 1):
    metrics = {}
    for offset, batch in enumerate(batches):
        metrics = agent.update(iter([dict(batch)]), step=first_step + offset)
    return metrics


def _strip_progress(batch):
    stripped = {key: np.array(value, copy=True) for key, value in batch.items()}
    stripped.pop("progress", None)
    stripped.pop("progress_valid", None)
    return stripped


def _pristine_kwargs(spec, cfg, observation_space, action_space):
    """Exactly the ``cqn_as_official`` construction branch of the factory."""

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
        temporal_ensemble_replan_interval=spec.temporal_ensemble_replan_interval,
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
        platform=cfg.backend.platform,
        seed=int(cfg.seed),
    )


def _variant(*overrides: str, pixels: bool = False):
    cfg = _compose("cqn_as_progress_shaping", *overrides, pixels=pixels)
    spec = cqn_as_progress_shaping_spec_from_cfg(cfg)
    observation_space, action_space = _spaces(pixels=pixels)
    agent = CQNASProgressShaping(
        **_pristine_kwargs(spec, cfg, observation_space, action_space),
        progress_potential_weight=spec.progress_potential_weight,
        progress_potential_schedule=spec.progress_potential_schedule,
        progress_head_weight=spec.progress_head_weight,
        progress_expectile_tau=spec.progress_expectile_tau,
        progress_success_gated=spec.progress_success_gated,
    )
    agent.logging = True
    return agent


def _pristine(*overrides: str, pixels: bool = False):
    cfg = _compose("cqn_as_official", *overrides, pixels=pixels)
    spec = cqn_as_spec_from_cfg(cfg)
    observation_space, action_space = _spaces(pixels=pixels)
    agent = CQNAS(**_pristine_kwargs(spec, cfg, observation_space, action_space))
    agent.logging = True
    return agent


def _research(*overrides: str, pixels: bool = False):
    cfg = _compose("cqn_as", *overrides, pixels=pixels)
    observation_space, action_space = _spaces(pixels=pixels)
    agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    agent.logging = True
    return agent


def _set_constant_progress_potential(agent, value):
    """Force ``Phi(s) == value`` by writing the zero-kernel head's bias."""

    flat = flatten_dict(unfreeze(agent.params["progress_value"]))
    bias_key = next(key for key in flat if key[-2:] == ("value_out", "bias"))
    flat[bias_key] = jnp.full_like(flat[bias_key], float(value))
    agent.params = {**agent.params, "progress_value": unflatten_dict(flat)}


def _tree_changed(before, after):
    return any(
        not np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(
            jax.tree.leaves(before), jax.tree.leaves(after), strict=True
        )
    )


def _shapes(tree):
    return jax.tree.map(lambda leaf: tuple(np.shape(leaf)), tree)


# ---------------------------------------------------------------------------
# 1. flags-off == pristine
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pixels", (False, True))
def test_flags_off_is_the_pristine_cqn_as(pixels):
    variant = _variant(pixels=pixels)
    pristine = _pristine(pixels=pixels)

    assert variant.progress_head_enabled is False
    assert variant.progress_shaping_enabled is False
    assert "progress_value" not in variant.params
    assert _shapes(variant.params) == _shapes(pristine.params)

    batches = [
        _batch(pixels=pixels, seed=7 + index) for index in range(EQUIV_STEPS)
    ]
    variant_metrics = _run_updates(variant, batches)
    pristine_metrics = _run_updates(pristine, batches)

    assert not [key for key in variant_metrics if key.startswith("progress")]
    assert sorted(variant_metrics) == sorted(pristine_metrics)
    np.testing.assert_allclose(
        variant_metrics["critic_loss"],
        pristine_metrics["critic_loss"],
        atol=1e-6,
    )
    assert _shapes(variant.params) == _shapes(pristine.params)
    for left, right in zip(
        jax.tree.leaves(variant.params),
        jax.tree.leaves(pristine.params),
        strict=True,
    ):
        np.testing.assert_allclose(np.asarray(left), np.asarray(right), atol=1e-6)


def test_flags_off_ignores_progress_elements_in_the_batch():
    """A batch carrying labels must not change the legacy computation."""

    variant = _variant()
    pristine = _pristine()
    labelled = [_progress_batch(seed=7 + index) for index in range(EQUIV_STEPS)]

    variant_metrics = _run_updates(variant, labelled)
    pristine_metrics = _run_updates(
        pristine, [_strip_progress(batch) for batch in labelled]
    )
    np.testing.assert_allclose(
        variant_metrics["critic_loss"], pristine_metrics["critic_loss"], atol=1e-6
    )


# ---------------------------------------------------------------------------
# 2. flags-on sanity
# ---------------------------------------------------------------------------


def test_flags_on_act_and_update_are_finite_and_log_the_line_metrics():
    agent = _variant(
        "method.progress_head_weight=1.0",
        "method.progress_expectile_tau=0.9",
        "method.progress_success_gated=true",
        "method.progress_potential_weight=0.25",
    )
    assert agent.progress_head_enabled is True
    assert agent.progress_shaping_enabled is True
    assert "progress_value" in agent.params

    action = np.asarray(agent.act(_observation(pixels=False), step=100, eval_mode=True))
    assert action.shape == (BATCH, ACTION_SEQUENCE, ACTION_DIM)
    assert np.all(np.isfinite(action))

    batch = _progress_batch()
    before = jax.tree.map(np.asarray, agent.params["progress_value"])
    metrics = agent.update(iter([batch]), step=1)

    expected = {
        "progress_head_loss",
        "progress_head_value_mean",
        "progress_label_mean",
        "progress_valid_fraction",
        "progress_potential_lambda",
        "progress_shaping_clip_frac",
    }
    assert expected <= set(metrics)
    for key, value in metrics.items():
        assert np.isfinite(value), key
    assert metrics["progress_head_loss"] > 0.0
    assert metrics["progress_valid_fraction"] == pytest.approx(1.0)
    assert metrics["progress_label_mean"] == pytest.approx(
        float(np.mean(batch["progress"])), abs=1e-6
    )
    assert metrics["progress_potential_lambda"] == pytest.approx(0.25)
    assert metrics["progress_shaping_clip_frac"] == pytest.approx(0.0)
    assert _tree_changed(before, agent.params["progress_value"])


def test_head_only_emits_no_shaping_metrics():
    agent = _variant("method.progress_head_weight=1.0")
    metrics = agent.update(iter([_progress_batch()]), step=1)
    assert "progress_head_loss" in metrics
    assert "progress_shaping_clip_frac" not in metrics
    assert "progress_potential_lambda" not in metrics


def test_missing_replay_elements_raise_a_named_key_error():
    agent = _variant("method.progress_head_weight=1.0")
    with pytest.raises(KeyError, match="progress"):
        agent.update(iter([_strip_progress(_progress_batch())]), step=1)


# ---------------------------------------------------------------------------
# 3. flags-on == research monolith
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    (
        ("method.progress_head_weight=1.0",),
        (
            "method.progress_head_weight=1.0",
            "method.progress_potential_weight=0.25",
        ),
        (
            "method.progress_potential_weight=0.25",
            "method.progress_head_weight=0.0",
        ),
        (
            "method.progress_head_weight=1.0",
            "method.progress_success_gated=false",
            "method.progress_expectile_tau=0.7",
        ),
    ),
)
def test_flags_on_matches_the_research_monolith(overrides):
    variant = _variant(*overrides)
    research = _research(*overrides)

    batches = [_progress_batch(seed=7 + index) for index in range(EQUIV_STEPS)]
    variant_metrics = _run_updates(variant, batches)
    research_metrics = _run_updates(research, batches)

    np.testing.assert_allclose(
        variant_metrics["critic_loss"],
        research_metrics["critic_loss"],
        atol=1e-5,
    )
    for key in (
        "progress_head_loss",
        "progress_head_value_mean",
        "progress_label_mean",
        "progress_valid_fraction",
    ):
        if key in variant_metrics:
            np.testing.assert_allclose(
                variant_metrics[key], research_metrics[key], atol=1e-5
            )


# ---------------------------------------------------------------------------
# Behavioral spec (adapted from the research-era tests in test_cqn_as.py)
# ---------------------------------------------------------------------------


def test_progress_shaped_rewards_terminal_drops_the_next_potential():
    rewards = jnp.asarray([0.0, 1.0], dtype=jnp.float32)
    discounts = jnp.asarray([0.99, 0.99], dtype=jnp.float32)
    bootstrap = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    phi = jnp.asarray([0.3, 0.8], dtype=jnp.float32)
    phi_next = jnp.asarray([0.5, 0.9], dtype=jnp.float32)

    np.testing.assert_allclose(
        progress_shaped_rewards(rewards, discounts, bootstrap, phi, phi_next, 0.25),
        [
            0.0 + 0.25 * (0.99 * 0.5 - 0.3),
            # bootstrap == 0 kills Phi(s'): the success target deflates to
            # exactly 1 - lambda * Phi(s).
            1.0 - 0.25 * 0.8,
        ],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        progress_shaped_rewards(rewards, discounts, bootstrap, phi, phi_next, 0.0),
        rewards,
        atol=0.0,
    )


def test_progress_shaping_telescopes_over_a_whole_episode():
    gamma, lam = 0.9, 0.4
    phi = np.asarray([0.1, 0.35, 0.6, 0.95], dtype=np.float32)
    horizon = phi.shape[0]
    rewards = np.zeros((horizon,), dtype=np.float32)
    discounts = np.full((horizon,), gamma, dtype=np.float32)
    bootstrap = np.ones((horizon,), dtype=np.float32)
    phi_next = np.concatenate([phi[1:], np.asarray([0.0], dtype=np.float32)])
    bootstrap[-1] = 0.0  # Phi(s_T) = 0 via the bootstrap mask.

    shaped = np.asarray(
        progress_shaped_rewards(
            jnp.asarray(rewards),
            jnp.asarray(discounts),
            jnp.asarray(bootstrap),
            jnp.asarray(phi),
            jnp.asarray(phi_next),
            lam,
        )
    )
    discounted_sum = float(sum(gamma**t * shaped[t] for t in range(horizon)))
    np.testing.assert_allclose(discounted_sum, -lam * phi[0], atol=1e-6)


def test_zero_initialized_potential_is_the_exact_legacy_target():
    variant = _variant(
        "method.progress_potential_weight=0.25",
        "method.progress_head_weight=0.0",
    )
    pristine = _pristine()
    batches = [_progress_batch(seed=7 + index) for index in range(EQUIV_STEPS)]

    metrics = _run_updates(variant, batches)
    _run_updates(pristine, [_strip_progress(batch) for batch in batches])

    # The head is zero-initialised and gets no gradient (head weight 0), so
    # Phi == 0 and the shaped target is the legacy target bit for bit.
    assert metrics["progress_head_value_mean"] == pytest.approx(0.0)
    assert metrics["progress_potential_lambda"] == pytest.approx(0.25)
    for left, right in zip(
        jax.tree.leaves(pristine.params["critic"]),
        jax.tree.leaves(variant.params["critic"]),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_constant_potential_equals_pre_shifted_rewards():
    lam, potential = 0.25, 0.4
    variant = _variant(
        f"method.progress_potential_weight={lam}",
        "method.progress_head_weight=0.0",
    )
    _set_constant_progress_potential(variant, potential)
    pristine = _pristine()

    batches = [_progress_batch(seed=7 + index) for index in range(EQUIV_STEPS)]
    shifted_batches = []
    for batch in batches:
        shifted = _strip_progress(batch)
        bootstrap = 1.0 - batch["terminal"].astype(np.float32)
        shifted["reward"] = (
            batch["reward"]
            + lam * (batch["discount"] * bootstrap * potential - potential)
        ).astype(np.float32)
        shifted_batches.append(shifted)

    metrics = _run_updates(variant, batches)
    _run_updates(pristine, shifted_batches)

    assert metrics["progress_head_value_mean"] == pytest.approx(potential)
    for left, right in zip(
        jax.tree.leaves(pristine.params["critic"]),
        jax.tree.leaves(variant.params["critic"]),
        strict=True,
    ):
        np.testing.assert_allclose(np.asarray(left), np.asarray(right), atol=1e-7)


def test_progress_head_leaves_the_legacy_critic_update_bitwise():
    variant = _variant("method.progress_head_weight=1.0")
    pristine = _pristine()
    batches = [_progress_batch(seed=7 + index) for index in range(EQUIV_STEPS)]

    _run_updates(variant, batches)
    _run_updates(pristine, [_strip_progress(batch) for batch in batches])

    for left, right in zip(
        jax.tree.leaves(pristine.params["critic"]),
        jax.tree.leaves(variant.params["critic"]),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_success_gate_censors_failed_episodes():
    gated = _variant(
        "method.progress_head_weight=1.0", "method.progress_success_gated=true"
    )
    ungated = _variant(
        "method.progress_head_weight=1.0", "method.progress_success_gated=false"
    )
    batch = _progress_batch()
    batch["progress_valid"] = np.zeros_like(batch["progress_valid"])

    gated_before = jax.tree.map(np.asarray, gated.params["progress_value"])
    ungated_before = jax.tree.map(np.asarray, ungated.params["progress_value"])
    gated_metrics = gated.update(iter([dict(batch)]), step=1)
    ungated_metrics = ungated.update(iter([dict(batch)]), step=1)

    assert gated_metrics["progress_valid_fraction"] == pytest.approx(0.0)
    assert gated_metrics["progress_head_loss"] == pytest.approx(0.0)
    assert not _tree_changed(gated_before, gated.params["progress_value"])
    assert ungated_metrics["progress_valid_fraction"] == pytest.approx(1.0)
    assert ungated_metrics["progress_head_loss"] > 0.0
    assert _tree_changed(ungated_before, ungated.params["progress_value"])


def test_potential_schedule_anneals_lambda():
    agent = _variant(
        "method.progress_potential_weight=0.25",
        "method.progress_head_weight=1.0",
        "method.progress_potential_schedule='linear(0.25,0.0,100)'",
    )
    early = agent.update(iter([_progress_batch()]), step=0)
    late = agent.update(iter([_progress_batch()]), step=100)
    assert early["progress_potential_lambda"] == pytest.approx(0.25)
    assert late["progress_potential_lambda"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Construction-time refusals
# ---------------------------------------------------------------------------


def test_potential_must_fit_the_c51_support():
    with pytest.raises(ValueError, match="exceeds v_max"):
        _variant("method.progress_potential_weight=1.5")


def test_negative_weights_and_out_of_range_tau_are_refused():
    with pytest.raises(ValueError, match="progress_head_weight"):
        _variant("method.progress_head_weight=-1.0")
    with pytest.raises(ValueError, match="progress_expectile_tau"):
        _variant(
            "method.progress_head_weight=1.0", "method.progress_expectile_tau=1.0"
        )


def test_schedule_requires_a_positive_weight_and_must_parse():
    with pytest.raises(ValueError, match="progress_potential_schedule requires"):
        _variant("method.progress_potential_schedule='linear(0.25,0.0,100)'")
    with pytest.raises(NotImplementedError):
        _variant(
            "method.progress_potential_weight=0.25",
            "method.progress_potential_schedule='cosine(1,2)'",
        )


def test_spec_defaults_are_the_exact_legacy_knobs():
    cfg = _compose("cqn_as_progress_shaping")
    assert cfg.method.progress_potential_weight == pytest.approx(0.0)
    assert cfg.method.progress_head_weight == pytest.approx(0.0)
    assert cfg.method.progress_potential_schedule is None
    assert cfg.method.progress_expectile_tau == pytest.approx(0.9)
    assert cfg.method.progress_success_gated is True

    spec = cqn_as_progress_shaping_spec_from_cfg(cfg)
    assert spec.progress_potential_weight == pytest.approx(0.0)
    assert spec.progress_head_weight == pytest.approx(0.0)
    assert spec.progress_potential_schedule is None
    assert spec.progress_expectile_tau == pytest.approx(0.9)
    assert spec.progress_success_gated is True


def test_spec_reads_the_enabled_knobs():
    cfg = _compose(
        "cqn_as_progress_shaping",
        "method.progress_head_weight=1.0",
        "method.progress_potential_weight=0.25",
        "method.progress_potential_schedule='linear(0.25,0.0,50000)'",
    )
    spec = cqn_as_progress_shaping_spec_from_cfg(cfg)
    assert spec.progress_potential_weight == pytest.approx(0.25)
    assert spec.progress_head_weight == pytest.approx(1.0)
    assert spec.progress_potential_schedule == "linear(0.25,0.0,50000)"
    assert spec.progress_success_gated is True


def test_spec_requires_truncated_demo_tails_on_bigym():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "method=cqn_as_progress_shaping",
                "env=bigym/move_plate",
                "pixels=true",
                "method.progress_head_weight=1.0",
                "env.truncate_demo_at_success=false",
            ],
        )
    with pytest.raises(ValueError, match="truncate_demo_at_success"):
        cqn_as_progress_shaping_spec_from_cfg(cfg)

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                "method=cqn_as_progress_shaping",
                "env=bigym/move_plate",
                "pixels=true",
                "method.progress_head_weight=1.0",
                "env.truncate_demo_at_success=true",
            ],
        )
    spec = cqn_as_progress_shaping_spec_from_cfg(cfg)
    assert spec.progress_head_weight == pytest.approx(1.0)
