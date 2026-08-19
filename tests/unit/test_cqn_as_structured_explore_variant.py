"""R2 line-1 (`structured-exploration`) variant tests.

Verification contract from ``R2_COMMON_BRIEF.md``:

1. flags-off  == pristine ``robobase.method.cqn_as.CQNAS`` (same seed, same
   synthetic batch, ``critic_loss`` atol <= 1e-6, identical param tree shapes);
2. flags-on runs (one ``act()`` + one ``update()``), all metrics finite, the
   line's diagnostic keys present;
3. flags-on == ``robobase.method.cqn_as_research.CQNAS`` with the same flags
   (``critic_loss`` atol <= 1e-5);
4. plus the behavioral spec adapted from
   ``tests/unit/test_cqn_as_bin_explore_state.py`` (replan-mask gating,
   episode reset, checkpoint resume of the NumPy exploration RNG streams).

CPU only.
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
from gymnasium import spaces  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402

from robobase.factory import create_agent  # noqa: E402
from robobase.method.cqn_as import CQNAS, cqn_as_spec_from_cfg  # noqa: E402
from robobase.method.cqn_as_structured_explore import (  # noqa: E402
    CQNASStructuredExplore,
    cqn_as_structured_explore_spec_from_cfg,
)

CONFIG_DIR = str((Path(__file__).parents[2] / "robobase" / "cfgs").resolve())

ACTION_SEQUENCE = 4
ACTION_DIM = 8
LOW_DIM = 5
RGB_KEY = "rgb_head"
RGB_SHAPE = (1, 12, 84, 84)
BATCH = 2

# Canonical wave values for this line (launch cfgs
# ``cqn_as_pixel_bigym_{coherent_exploration,stage153_bin_explore,
# stage160_lowdim_mask,stage162_edecay}_gate`` and ``scripts/run_sw_flip.sh``).
FLAGS_ON = (
    "method.structured_exploration_prob=0.06",
    "method.structured_exploration_level=1",
    "method.structured_exploration_horizon=4",
    "method.bin_explore_probs=[0.002,0.004,0.008]",
    "method.bin_explore_persist_plans=2",
    "method.bin_explore_schedule='linear(1.0,0.0,100000)'",
    "method.low_dim_mask_prob=0.2",
    "method.low_dim_mask_keep_last=3",
    "method.post_ensemble_l1_flip_prob=0.015",
    "method.post_ensemble_l2_flip_prob=0.05",
    "method.post_ensemble_l1_flip_horizon=4",
)


# ----------------------------------------------------------------------
# Synthetic spaces / batches (mirrors scripts/refactor_equivalence_check.py)
# ----------------------------------------------------------------------


def _compose(method: str, *overrides: str, action_sequence: int = ACTION_SEQUENCE):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(
            config_name="robobase_config",
            overrides=[
                f"method={method}",
                f"action_sequence={action_sequence}",
                "num_train_envs=2",
                "num_eval_envs=2",
                "num_explore_steps=0",
                "backend.jit=false",
                "backend.platform=cpu",
                "method.model.hidden_dims=[32,32]",
                "method.atoms=11",
                *overrides,
            ],
        )


def _spaces(*, action_sequence: int = ACTION_SEQUENCE, pixels: bool = True):
    obs = {
        "low_dim_state": spaces.Box(
            -np.inf, np.inf, shape=(1, LOW_DIM), dtype=np.float32
        )
    }
    if pixels:
        obs[RGB_KEY] = spaces.Box(0, 255, shape=RGB_SHAPE, dtype=np.uint8)
    action_space = spaces.Box(
        -1.0, 1.0, shape=(action_sequence, ACTION_DIM), dtype=np.float32
    )
    return spaces.Dict(obs), action_space


def _observation(*, pixels: bool = True):
    rng = np.random.default_rng(3)
    obs = {"low_dim_state": rng.normal(size=(BATCH, 1, LOW_DIM)).astype(np.float32)}
    if pixels:
        obs[RGB_KEY] = rng.integers(0, 256, size=(BATCH, *RGB_SHAPE), dtype=np.uint8)
    return obs


def _batch(*, action_sequence: int = ACTION_SEQUENCE, pixels: bool = True):
    rng = np.random.default_rng(7)
    batch = {
        "low_dim_state": rng.normal(size=(BATCH, 1, LOW_DIM)).astype(np.float32),
        "low_dim_state_tp1": rng.normal(size=(BATCH, 1, LOW_DIM)).astype(np.float32),
        "action": rng.uniform(
            -1.0, 1.0, size=(BATCH, action_sequence, ACTION_DIM)
        ).astype(np.float32),
        "reward": rng.normal(size=(BATCH,)).astype(np.float32),
        "discount": np.full((BATCH,), 0.99, dtype=np.float32),
        "terminal": np.zeros((BATCH,), dtype=bool),
        "truncated": np.zeros((BATCH,), dtype=bool),
        "demo": np.ones((BATCH,), dtype=np.uint8),
    }
    if pixels:
        batch[RGB_KEY] = rng.integers(
            0, 256, size=(BATCH, *RGB_SHAPE), dtype=np.uint8
        )
        batch[f"{RGB_KEY}_tp1"] = rng.integers(
            0, 256, size=(BATCH, *RGB_SHAPE), dtype=np.uint8
        )
    return batch


# ----------------------------------------------------------------------
# Direct construction (factory.py is off-limits for R2 line agents)
# ----------------------------------------------------------------------


def _common_kwargs(cfg, observation_space, action_space):
    return dict(
        observation_space=observation_space,
        action_space=action_space,
        num_train_envs=cfg.num_train_envs,
        num_eval_envs=cfg.num_eval_envs,
        replay_alpha=cfg.replay.alpha,
        replay_beta=cfg.replay.beta,
        frame_stack_on_channel=cfg.frame_stack_on_channel,
        intrinsic_reward_module=None,
        update_block_every_steps=int(
            cfg.get("backend", {}).get("update_block_every_steps", 1)
        ),
        jit=bool(cfg.backend.jit),
        platform=cfg.backend.platform,
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
        temporal_ensemble_replan_interval=spec.temporal_ensemble_replan_interval,
        temporal_ensemble_gain=spec.temporal_ensemble_gain,
        tie_break_delta=spec.tie_break_delta,
        model=spec.model,
    )


def _line_kwargs(spec):
    """Exactly the extra kwargs the factory registration snippet must pass."""

    return dict(
        structured_exploration_prob=spec.structured_exploration_prob,
        structured_exploration_level=spec.structured_exploration_level,
        structured_exploration_horizon=spec.structured_exploration_horizon,
        bin_flip_prob=spec.bin_flip_prob,
        bin_flip_level=spec.bin_flip_level,
        bin_explore_probs=spec.bin_explore_probs,
        bin_explore_schedule=spec.bin_explore_schedule,
        bin_explore_persist_plans=spec.bin_explore_persist_plans,
        low_dim_mask_prob=spec.low_dim_mask_prob,
        low_dim_mask_keep_last=spec.low_dim_mask_keep_last,
        random_levels_from=spec.random_levels_from,
        level_override_mode=spec.level_override_mode,
        post_ensemble_random_keep_levels=spec.post_ensemble_random_keep_levels,
        post_ensemble_fixed_leaf=spec.post_ensemble_fixed_leaf,
        post_ensemble_l1_flip_prob=spec.post_ensemble_l1_flip_prob,
        post_ensemble_l2_flip_prob=spec.post_ensemble_l2_flip_prob,
        post_ensemble_l1_flip_horizon=spec.post_ensemble_l1_flip_horizon,
    )


def _variant_agent(*overrides, action_sequence=ACTION_SEQUENCE, pixels=True):
    cfg = _compose(
        "cqn_as_structured_explore", *overrides, action_sequence=action_sequence
    )
    observation_space, action_space = _spaces(
        action_sequence=action_sequence, pixels=pixels
    )
    spec = cqn_as_structured_explore_spec_from_cfg(cfg)
    return CQNASStructuredExplore(
        **_pristine_kwargs(spec),
        **_line_kwargs(spec),
        **_common_kwargs(cfg, observation_space, action_space),
    )


def _official_agent(*overrides, action_sequence=ACTION_SEQUENCE, pixels=True):
    cfg = _compose("cqn_as_official", *overrides, action_sequence=action_sequence)
    observation_space, action_space = _spaces(
        action_sequence=action_sequence, pixels=pixels
    )
    spec = cqn_as_spec_from_cfg(cfg)
    return CQNAS(
        **_pristine_kwargs(spec),
        **_common_kwargs(cfg, observation_space, action_space),
    )


def _research_agent(*overrides, action_sequence=ACTION_SEQUENCE, pixels=True):
    cfg = _compose("cqn_as", *overrides, action_sequence=action_sequence)
    observation_space, action_space = _spaces(
        action_sequence=action_sequence, pixels=pixels
    )
    return create_agent(
        cfg, observation_space=observation_space, action_space=action_space
    )


def _shapes(tree):
    return [np.shape(leaf) for leaf in jax.tree_util.tree_leaves(tree)]


def _max_param_diff(a, b):
    return max(
        float(np.max(np.abs(np.asarray(x) - np.asarray(y))))
        for x, y in zip(
            jax.tree_util.tree_leaves(a.params), jax.tree_util.tree_leaves(b.params)
        )
    )


def _run_updates(agent, steps=3, pixels=True):
    """Several updates: the zero-initialised heads make step-1 loss degenerate
    (uniform logits => the loss does not depend on the selected bin), so a
    single update cannot discriminate action-path changes."""

    agent.logging = True
    losses = []
    for step in range(1, steps + 1):
        metrics = agent.update(iter([_batch(pixels=pixels)]), step=step)
        losses.append(float(metrics["critic_loss"]))
    return losses, metrics


def _assert_finite(name, value):
    array = np.asarray(value, dtype=np.float64)
    assert np.all(np.isfinite(array)), f"{name} not finite: {array}"


# ----------------------------------------------------------------------
# 1. flags-off == pristine
# ----------------------------------------------------------------------


def test_flags_off_matches_pristine_act_and_update():
    variant = _variant_agent("pixels=true")
    official = _official_agent("pixels=true")

    assert _shapes(variant.params) == _shapes(official.params)
    assert jax.tree_util.tree_structure(variant.params) == (
        jax.tree_util.tree_structure(official.params)
    )
    for a, b in zip(
        jax.tree_util.tree_leaves(variant.params),
        jax.tree_util.tree_leaves(official.params),
    ):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    observation = _observation()
    for eval_mode in (True, False):
        np.testing.assert_array_equal(
            np.asarray(variant.act(observation, step=100, eval_mode=eval_mode)),
            np.asarray(official.act(observation, step=100, eval_mode=eval_mode)),
        )

    variant_losses, variant_metrics = _run_updates(variant)
    official_losses, official_metrics = _run_updates(official)

    assert set(variant_metrics) == set(official_metrics)
    np.testing.assert_allclose(
        variant_losses, official_losses, atol=1e-6, rtol=0.0
    )
    assert _shapes(variant.params) == _shapes(official.params)
    assert _max_param_diff(variant, official) == 0.0
    np.testing.assert_array_equal(
        np.asarray(variant.rng_key), np.asarray(official.rng_key)
    )


def test_flags_off_yaml_defaults_are_all_off():
    cfg = _compose("cqn_as_structured_explore")
    assert (
        cfg.method._target_
        == "robobase.method.cqn_as_structured_explore.CQNASStructuredExplore"
    )
    assert cfg.method.name == "cqn_as_structured_explore"
    spec = cqn_as_structured_explore_spec_from_cfg(cfg)
    assert spec.structured_exploration_prob == 0.0
    assert spec.structured_exploration_level == 1
    assert spec.structured_exploration_horizon == 1
    assert spec.bin_flip_prob == 0.0
    assert spec.bin_flip_level is None
    assert spec.bin_explore_probs is None
    assert spec.bin_explore_schedule is None
    assert spec.bin_explore_persist_plans is None
    assert spec.low_dim_mask_prob == 0.0
    assert spec.low_dim_mask_keep_last == 0
    assert spec.random_levels_from is None
    assert spec.level_override_mode == "random"
    assert spec.post_ensemble_random_keep_levels is None
    assert spec.post_ensemble_fixed_leaf is None
    assert spec.post_ensemble_l1_flip_prob == 0.0
    assert spec.post_ensemble_l2_flip_prob == 0.0
    assert spec.post_ensemble_l1_flip_horizon == 4


# ----------------------------------------------------------------------
# 2. flags-on sanity
# ----------------------------------------------------------------------


def test_flags_on_act_update_finite_and_metrics_present():
    agent = _variant_agent("pixels=true", *FLAGS_ON)
    assert agent.bin_explore_persist_plans == 2

    np.random.seed(0)
    observation = _observation()
    action = agent.act(observation, step=100, eval_mode=False)
    assert action.shape == (BATCH, ACTION_SEQUENCE, ACTION_DIM)
    _assert_finite("act", action)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
    # The schedule multiplier is refreshed on every fresh-plan act().
    assert agent._bin_explore_scale == pytest.approx(1.0 - 100 / 100000)

    losses, metrics = _run_updates(agent)
    assert "critic_loss" in metrics
    for loss in losses:
        _assert_finite("critic_loss", loss)
    for key, value in metrics.items():
        _assert_finite(f"update[{key}]", value)

    diagnostics = agent.rollout_diagnostics()
    for key in (
        "structured_exploration_rate",
        "structured_exploration_applied",
        "structured_exploration_eligible",
        "structured_exploration_starts",
        "bin_explore_fired_total",
        "bin_explore_applied_total",
        "bin_explore_calls_total",
        "act_train_calls_total",
        "bin_explore_mask_rows_total",
    ):
        assert key in diagnostics
        _assert_finite(f"diagnostics[{key}]", diagnostics[key])
    assert diagnostics["structured_exploration_eligible"] == float(BATCH)
    assert diagnostics["bin_explore_calls_total"] == 1.0
    assert diagnostics["act_train_calls_total"] == 1.0

    # The replay-extra producers the workspace snapshots.
    assert agent._last_structured_exploration_mask.shape == (BATCH,)
    assert agent._last_structured_exploration_dimension.shape == (BATCH,)
    assert agent._last_bin_explored.shape == (BATCH,)


def test_low_dim_mask_only_masks_update_batches():
    agent = _variant_agent(
        "pixels=true",
        "method.low_dim_mask_prob=1.0",
        "method.low_dim_mask_keep_last=3",
    )
    low_dim = np.ones((4, LOW_DIM), dtype=np.float32)
    masked = np.asarray(
        agent._mask_low_dim(jnp.asarray(low_dim), jax.random.PRNGKey(0))
    )
    np.testing.assert_array_equal(masked[:, : LOW_DIM - 3], 0.0)
    np.testing.assert_array_equal(masked[:, LOW_DIM - 3 :], low_dim[:, LOW_DIM - 3 :])

    # act() is never masked: the same observation gives the same plan as the
    # unmasked agent.
    plain = _variant_agent("pixels=true")
    observation = _observation()
    np.testing.assert_array_equal(
        np.asarray(agent.act(observation, step=100, eval_mode=True)),
        np.asarray(plain.act(observation, step=100, eval_mode=True)),
    )

    # ... but the update path IS masked, so the learned params diverge. This
    # keeps the flags-off equivalence test from being vacuous.
    masked_losses, _ = _run_updates(agent)
    plain_losses, _ = _run_updates(plain)
    assert _max_param_diff(agent, plain) > 0.0
    assert masked_losses != plain_losses


def _low_dim_features(agent, batch=1):
    observation = {
        "low_dim_state": np.zeros((batch, 1, LOW_DIM), dtype=np.float32)
    }
    return agent._rl_features(
        agent.params.get("encoder", None),
        agent._prepare_rl_obs_inputs(observation),
    )


def test_random_levels_from_middle_selects_centre_bins():
    agent = _variant_agent(
        "method.random_levels_from=1",
        "method.level_override_mode=middle",
        action_sequence=3,
        pixels=False,
    )
    features = _low_dim_features(agent)
    action, indices = agent._greedy_action(agent.params["critic"], features, key=None)
    indices = np.asarray(indices)
    assert indices.shape[1] == agent.levels
    # Every level at/below random_levels_from emits the parent cell's centre.
    np.testing.assert_array_equal(indices[:, 1:], agent.bins // 2)
    # ... which makes the action the exact centre of its level-0 cell.
    width0 = 2.0 / agent.bins
    expected = -1.0 + (indices[:, 0] + 0.5) * width0
    np.testing.assert_allclose(np.asarray(action), expected, atol=1e-6)

    # `random` mode instead draws uniformly at those levels (needs a key).
    random_agent = _variant_agent(
        "method.random_levels_from=1", action_sequence=3, pixels=False
    )
    features = _low_dim_features(random_agent)
    with pytest.raises(ValueError, match="needs an rng key"):
        random_agent._greedy_action(random_agent.params["critic"], features, key=None)
    action, indices = random_agent._greedy_action(
        random_agent.params["critic"], features, key=jax.random.PRNGKey(0)
    )
    indices = np.asarray(indices)
    assert indices[:, 1:].min() >= 0 and indices[:, 1:].max() < random_agent.bins
    _assert_finite("random_levels_from=random action", action)


def test_post_ensemble_randomize_keeps_prefix_cell():
    agent = _variant_agent(
        "pixels=true",
        "method.post_ensemble_random_keep_levels=1",
        "method.post_ensemble_fixed_leaf=0",
    )
    action = np.zeros((3, ACTION_DIM), dtype=np.float32) + 0.31
    out = agent._post_ensemble_randomize(action)
    span = 2.0
    parent_w = span / agent.bins
    leaf_w = span / agent.bins**agent.levels
    parent = np.floor((action + 1.0) / parent_w)
    np.testing.assert_allclose(out, -1.0 + parent * parent_w + 0.5 * leaf_w)


# ----------------------------------------------------------------------
# 3. flags-on == research monolith
# ----------------------------------------------------------------------


def test_flags_on_matches_research_monolith():
    variant = _variant_agent("pixels=true", *FLAGS_ON)
    research = _research_agent("pixels=true", *FLAGS_ON)

    observation = _observation()
    for eval_mode in (True, False):
        np.random.seed(4242)
        variant_action = np.asarray(
            variant.act(observation, step=100, eval_mode=eval_mode)
        )
        np.random.seed(4242)
        research_action = np.asarray(
            research.act(observation, step=100, eval_mode=eval_mode)
        )
        np.testing.assert_allclose(
            variant_action, research_action, atol=1e-6, rtol=0.0
        )

    variant_losses, _ = _run_updates(variant)
    research_losses, _ = _run_updates(research)
    np.testing.assert_allclose(
        variant_losses, research_losses, atol=1e-5, rtol=0.0
    )
    assert _max_param_diff(variant, research) < 1e-5

    np.testing.assert_array_equal(
        variant._last_structured_exploration_mask,
        research._last_structured_exploration_mask,
    )
    np.testing.assert_array_equal(
        variant._last_bin_explored, research._last_bin_explored
    )
    for key in (
        "structured_exploration_applied",
        "structured_exploration_eligible",
        "bin_explore_fired_total",
        "bin_explore_applied_total",
        "bin_explore_calls_total",
    ):
        assert variant.rollout_diagnostics()[key] == (
            research.rollout_diagnostics()[key]
        ), key


def test_firing_exploration_matches_research_across_steps():
    """Same as above but with probabilities that actually fire every step, so
    the drawn assignments, persistence windows and replay-extra payloads are
    compared rather than just the plumbing."""

    overrides = (
        "method.structured_exploration_prob=1.0",
        "method.structured_exploration_level=1",
        "method.structured_exploration_horizon=3",
        "method.bin_explore_probs=[1.0,1.0,1.0]",
        "method.bin_explore_persist_plans=2",
        "method.post_ensemble_l1_flip_prob=1.0",
        "method.post_ensemble_l2_flip_prob=1.0",
        "method.post_ensemble_l1_flip_horizon=4",
    )
    variant = _variant_agent(*overrides, action_sequence=3, pixels=False)
    research = _research_agent(*overrides, action_sequence=3, pixels=False)
    observation = {
        "low_dim_state": np.zeros((BATCH, 1, LOW_DIM), dtype=np.float32)
    }
    for offset in range(6):
        np.random.seed(1000 + offset)
        variant_action = np.asarray(
            variant.act(observation, step=100 + offset, eval_mode=False)
        )
        np.random.seed(1000 + offset)
        research_action = np.asarray(
            research.act(observation, step=100 + offset, eval_mode=False)
        )
        np.testing.assert_allclose(
            variant_action, research_action, atol=1e-6, rtol=0.0
        )
        for attribute in (
            "_last_bin_explored",
            "_last_bin_explore_applied",
            "_bin_explore_remaining",
            "_bin_explore_dimension",
            "_bin_explore_level",
            "_bin_explore_sibling",
            "_last_structured_exploration_mask",
            "_last_structured_exploration_start",
            "_last_structured_exploration_dimension",
            "_last_structured_exploration_delta",
            "_last_structured_exploration_assignment_prob",
        ):
            np.testing.assert_array_equal(
                getattr(variant, attribute),
                getattr(research, attribute),
                err_msg=f"{attribute} @ offset {offset}",
            )

    variant_diagnostics = variant.rollout_diagnostics()
    research_diagnostics = research.rollout_diagnostics()
    assert variant_diagnostics["bin_explore_fired_total"] > 0.0
    assert variant_diagnostics["bin_explore_applied_total"] > 0.0
    assert variant_diagnostics["structured_exploration_applied"] > 0.0
    assert variant_diagnostics["structured_exploration_starts"] > 0.0
    for key, value in variant_diagnostics.items():
        assert value == research_diagnostics[key], key

    variant.reset(step=0, agents_to_reset=[0])
    research.reset(step=0, agents_to_reset=[0])
    np.testing.assert_array_equal(
        variant._bin_explore_remaining, research._bin_explore_remaining
    )
    np.testing.assert_array_equal(
        variant._structured_exploration_remaining,
        research._structured_exploration_remaining,
    )


# ----------------------------------------------------------------------
# 4. bin-explore state lifecycle (spec: tests/unit/test_cqn_as_bin_explore_state.py)
# ----------------------------------------------------------------------


def _explore_agent(num_train_envs=2, probs="[1.0,1.0,1.0]"):
    cfg = _compose(
        "cqn_as_structured_explore",
        f"num_train_envs={num_train_envs}",
        "num_eval_envs=1",
        f"method.bin_explore_probs={probs}",
        "method.bin_explore_persist_plans=2",
        action_sequence=3,
    )
    observation_space = spaces.Dict(
        {
            "low_dim_state": spaces.Box(
                -np.inf, np.inf, shape=(1, LOW_DIM), dtype=np.float32
            )
        }
    )
    action_space = spaces.Box(-1.0, 1.0, shape=(3, 2), dtype=np.float32)
    spec = cqn_as_structured_explore_spec_from_cfg(cfg)
    return CQNASStructuredExplore(
        **_pristine_kwargs(spec),
        **_line_kwargs(spec),
        **_common_kwargs(cfg, observation_space, action_space),
    )


def _chunk(batch=2):
    return np.zeros((batch, 3, 2), dtype=np.float32)


def test_register_mask_gates_draws_shifts_and_persist_counters():
    agent = _explore_agent()

    shifted = agent._apply_bin_explore(_chunk(), np.asarray([True, False]))
    np.testing.assert_array_equal(agent._bin_explore_remaining, [1, 0])
    assert agent._bin_explore_dimension[1] == -1, "masked row must not draw"
    assert not np.array_equal(shifted[0], _chunk()[0])
    np.testing.assert_array_equal(shifted[1], _chunk()[1])

    before = agent._bin_explore_remaining.copy()
    shifted = agent._apply_bin_explore(_chunk(), np.asarray([False, False]))
    np.testing.assert_array_equal(agent._bin_explore_remaining, before)
    np.testing.assert_array_equal(shifted, _chunk())

    agent._apply_bin_explore(_chunk(), np.asarray([True, True]))
    np.testing.assert_array_equal(agent._bin_explore_remaining, [0, 1])


def test_reset_clears_persisted_shift_only_for_reset_envs():
    agent = _explore_agent()
    agent._apply_bin_explore(_chunk(), np.asarray([True, True]))
    assert (agent._bin_explore_remaining > 0).all()

    agent.reset(step=0, agents_to_reset=[0])

    assert agent._bin_explore_remaining[0] == 0
    assert agent._bin_explore_dimension[0] == -1
    assert agent._bin_explore_level[0] == -1
    assert agent._bin_explore_sibling[0] == -1
    assert agent._bin_explore_remaining[1] > 0
    assert agent._bin_explore_dimension[1] != -1


def test_checkpoint_roundtrip_resumes_exploration_rng_stream():
    agent = _explore_agent(num_train_envs=1, probs="[0.5,0.3,0.2]")
    for _ in range(50):
        agent._apply_bin_explore(_chunk(1), np.asarray([True]))
        if agent._bin_explore_remaining[0] == 0:
            break
    assert agent._bin_explore_remaining[0] == 0
    state = agent.checkpoint_state_dict()

    continued = [
        (
            agent._apply_bin_explore(_chunk(1), np.asarray([True])),
            agent._bin_explore_dimension.copy(),
            agent._bin_explore_level.copy(),
            agent._bin_explore_sibling.copy(),
        )
        for _ in range(6)
    ]

    resumed = _explore_agent(num_train_envs=1, probs="[0.5,0.3,0.2]")
    resumed.load_checkpoint_state_dict(state)
    for expected_chunk, dim, level, sibling in continued:
        chunk = resumed._apply_bin_explore(_chunk(1), np.asarray([True]))
        np.testing.assert_array_equal(chunk, expected_chunk)
        np.testing.assert_array_equal(resumed._bin_explore_dimension, dim)
        np.testing.assert_array_equal(resumed._bin_explore_level, level)
        np.testing.assert_array_equal(resumed._bin_explore_sibling, sibling)


def test_resume_never_restores_mid_episode_window():
    agent = _explore_agent(num_train_envs=2)
    agent._apply_bin_explore(_chunk(), np.asarray([True, True]))
    assert (agent._bin_explore_remaining > 0).all()
    state = agent.checkpoint_state_dict()
    assert not any(key.startswith("bin_explore_remaining") for key in state)

    resumed = _explore_agent(num_train_envs=2)
    resumed.load_checkpoint_state_dict(state)
    np.testing.assert_array_equal(resumed._bin_explore_remaining, [0, 0])
    np.testing.assert_array_equal(resumed._bin_explore_dimension, [-1, -1])
    assert (
        resumed._bin_explore_rng.bit_generator.state["state"]
        == agent._bin_explore_rng.bit_generator.state["state"]
    )

    legacy = dict(state)
    legacy["bin_explore_remaining"] = np.asarray([2, 2], dtype=np.int32)
    legacy["bin_explore_dimension"] = np.asarray([1, 1], dtype=np.int16)
    fresh = _explore_agent(num_train_envs=2)
    fresh.load_checkpoint_state_dict(legacy)
    np.testing.assert_array_equal(fresh._bin_explore_remaining, [0, 0])
    np.testing.assert_array_equal(fresh._bin_explore_dimension, [-1, -1])


def test_old_snapshots_without_exploration_keys_still_load():
    agent = _explore_agent(num_train_envs=1)
    state = agent.checkpoint_state_dict()
    for key in list(state):
        if key.startswith("bin_"):
            state.pop(key)
    agent.load_checkpoint_state_dict(state)  # must not raise
    assert agent._bin_explore_remaining.shape == (1,)


def test_apply_records_applied_rows_for_explored_flagging():
    agent = _explore_agent()
    agent._apply_bin_explore(_chunk(), np.asarray([True, False]))
    np.testing.assert_array_equal(agent._last_bin_explore_applied, [True, False])
    agent._apply_bin_explore(_chunk(), np.asarray([False, False]))
    np.testing.assert_array_equal(agent._last_bin_explore_applied, [False, False])


def test_exploration_checkpoint_helpers_tolerate_missing_rngs():
    # Mirrors tests/unit/test_cqn_exploration_checkpoint_compat.py: subclasses
    # that never created the exploration arrays must degrade to a no-op.
    agent = object.__new__(CQNASStructuredExplore)
    assert agent._exploration_checkpoint_state() == {}
    agent._load_exploration_checkpoint_state(
        {
            "bin_flip_rng_state": {"ignored": True},
            "bin_explore_rng_state": {"ignored": True},
            "episodic_twin_head_rng_state": {"ignored": True},
        }
    )
    diagnostics = agent.rollout_diagnostics()
    assert diagnostics["structured_exploration_rate"] == 0.0
    assert diagnostics["bin_explore_fired_total"] == 0.0


# ----------------------------------------------------------------------
# Coherent structured exploration + open-loop bin flip
# ----------------------------------------------------------------------


def test_coherent_structured_exploration_persists_assignment():
    cfg = _compose(
        "cqn_as_structured_explore",
        "num_train_envs=1",
        "num_eval_envs=1",
        "method.structured_exploration_prob=1.0",
        "method.structured_exploration_level=1",
        "method.structured_exploration_horizon=3",
        action_sequence=3,
    )
    observation_space = spaces.Dict(
        {
            "low_dim_state": spaces.Box(
                -np.inf, np.inf, shape=(1, LOW_DIM), dtype=np.float32
            )
        }
    )
    action_space = spaces.Box(-1.0, 1.0, shape=(3, 2), dtype=np.float32)
    spec = cqn_as_structured_explore_spec_from_cfg(cfg)
    agent = CQNASStructuredExplore(
        **_pristine_kwargs(spec),
        **_line_kwargs(spec),
        **_common_kwargs(cfg, observation_space, action_space),
    )

    base = jnp.zeros((1, 2), dtype=jnp.float32)
    first = agent._coherent_structured_exploration_action(base, jax.random.PRNGKey(1))
    agent.structured_exploration_prob = 0.0
    second = agent._coherent_structured_exploration_action(base, jax.random.PRNGKey(2))
    third = agent._coherent_structured_exploration_action(base, jax.random.PRNGKey(3))
    fourth = agent._coherent_structured_exploration_action(base, jax.random.PRNGKey(4))

    assert bool(first[1][0]) and bool(first[2][0])
    assert bool(second[1][0]) and not bool(second[2][0])
    assert bool(third[1][0]) and not bool(third[2][0])
    assert not bool(fourth[1][0]) and not bool(fourth[2][0])
    assert first[3][0] == second[3][0] == third[3][0]
    assert fourth[3][0] == -1
    np.testing.assert_allclose(first[4], second[4])
    np.testing.assert_allclose(first[5], [1.0 / (2 * agent.action_dim)])
    np.testing.assert_allclose(second[5], [1.0])


def test_bin_flip_requires_open_loop_execution():
    with pytest.raises(ValueError, match="temporal_ensemble=false"):
        _variant_agent(
            "pixels=true",
            "method.bin_flip_prob=0.1",
            "method.temporal_ensemble=true",
        )
    with pytest.raises(ValueError, match="mutually"):
        _variant_agent(
            "pixels=true",
            "method.bin_flip_prob=0.1",
            "method.temporal_ensemble=false",
            "method.bin_explore_probs=[0.1,0.1,0.1]",
        )


def test_bin_flip_shifts_a_single_dimension_to_a_sibling_cell():
    agent = _variant_agent(
        "pixels=true",
        "method.temporal_ensemble=false",
        "method.bin_flip_prob=1.0",
        "method.bin_flip_level=0",
        action_sequence=ACTION_SEQUENCE,
    )
    chunk = np.zeros((2, ACTION_SEQUENCE, ACTION_DIM), dtype=np.float32)
    flipped = agent._apply_bin_flip(chunk)
    assert np.all(agent._bin_flip_remaining == ACTION_SEQUENCE)
    assert np.all(agent._bin_flip_dimension >= 0)
    changed = np.any(np.abs(flipped - chunk) > 0, axis=1)
    np.testing.assert_array_equal(changed.sum(axis=1), [1, 1])
    width = 2.0 / agent.bins
    for row in range(2):
        dim = int(agent._bin_flip_dimension[row])
        delta = float(flipped[row, 0, dim] - chunk[row, 0, dim])
        np.testing.assert_allclose(delta / width, round(delta / width), atol=1e-5)
