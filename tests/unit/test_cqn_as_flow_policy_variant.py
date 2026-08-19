"""CPU-only tests for the R2 flow-policy CQN-AS variant.

Verification contract from ``R2_COMMON_BRIEF.md``:

1. flags-off equals the frozen official :class:`robobase.method.cqn_as.CQNAS`
   (same seed, same synthetic batch, ``critic_loss`` within ``1e-6``, identical
   parameter tree);
2. flags-on (``coarse_flow`` / ``coarse_flow_pure``) runs ``act()`` +
   ``update()`` with finite metrics and the line's metric keys;
3. flags-on matches ``cqn_as_research.CQNAS`` configured the same way.
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
from flax.traverse_util import flatten_dict  # noqa: E402
from gymnasium import spaces  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402

from robobase.factory import create_agent  # noqa: E402
from robobase.method.cqn import encode_action  # noqa: E402
from robobase.method.cqn_as import CQNAS, cqn_as_spec_from_cfg  # noqa: E402
from robobase.method.cqn_as_flow_policy import (  # noqa: E402
    CQNASFlowPolicy,
    cqn_as_flow_policy_spec_from_cfg,
)

CONFIG_DIR = str((Path(__file__).parents[2] / "robobase" / "cfgs").resolve())

ACTION_SEQUENCE = 3
ACTION_DIM = 2
LOW_DIM = 5


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


RGB_KEY = "rgb_head"
RGB_SHAPE = (1, 12, 84, 84)


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


def _batch(batch_size=4, *, pixels: bool = False):
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
        batch[RGB_KEY] = rng.integers(
            0, 256, size=(batch_size, *RGB_SHAPE), dtype=np.uint8
        )
        batch[f"{RGB_KEY}_tp1"] = rng.integers(
            0, 256, size=(batch_size, *RGB_SHAPE), dtype=np.uint8
        )
    return batch


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
        update_block_every_steps=1,
    )


def _spec_kwargs(spec):
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


def _official_agent(*overrides, pixels: bool = False):
    cfg = _compose("cqn_as_official", *overrides)
    observation_space, action_space = _spaces(pixels=pixels)
    spec = cqn_as_spec_from_cfg(cfg)
    return CQNAS(
        **_spec_kwargs(spec),
        jit=False,
        platform="cpu",
        seed=int(cfg.seed),
        **_common_kwargs(cfg, observation_space, action_space),
    )


def _variant_agent(*overrides, pixels: bool = False):
    cfg = _compose("cqn_as_flow_policy", *overrides)
    observation_space, action_space = _spaces(pixels=pixels)
    spec = cqn_as_flow_policy_spec_from_cfg(cfg)
    return CQNASFlowPolicy(
        **_spec_kwargs(spec),
        jit=False,
        platform="cpu",
        seed=int(cfg.seed),
        flow_policy=spec.flow_policy,
        flow_policy_candidates=spec.flow_policy_candidates,
        flow_policy_steps=spec.flow_policy_steps,
        flow_policy_lambda=spec.flow_policy_lambda,
        flow_policy_ema=spec.flow_policy_ema,
        flow_policy_hidden_dims=spec.flow_policy_hidden_dims,
        flow_policy_gru_layers=spec.flow_policy_gru_layers,
        coarse_flow=spec.coarse_flow,
        coarse_flow_pure=spec.coarse_flow_pure,
        coarse_flow_selfdistill_weight=spec.coarse_flow_selfdistill_weight,
        coarse_flow_selfdistill_threshold=(
            spec.coarse_flow_selfdistill_threshold
        ),
        policy_value_beta=spec.policy_value_beta,
        **_common_kwargs(cfg, observation_space, action_space),
    )


_COARSE_FLOW_OVERRIDES = (
    "method.coarse_flow=true",
    "method.levels=1",
    "method.flow_policy_steps=2",
)


def _research_agent(*overrides, pixels: bool = False):
    """Research monolith agent, used only as the flags-on reference."""

    cfg = _compose(
        "cqn_as",
        # The research default structured_exploration_level=1 fails its own
        # levels-range validation at levels=1 even with probability 0.
        "method.structured_exploration_level=0",
        *overrides,
    )
    observation_space, action_space = _spaces(pixels=pixels)
    return create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )


def _tree_shapes(tree):
    return {
        key: tuple(np.asarray(value).shape)
        for key, value in flatten_dict(tree).items()
    }


def _tree_changed(before, after):
    before_leaves, before_tree = jax.tree.flatten(before)
    after_leaves, after_tree = jax.tree.flatten(after)
    assert before_tree == after_tree
    return any(
        not np.allclose(np.asarray(left), np.asarray(right))
        for left, right in zip(before_leaves, after_leaves, strict=True)
    )


# ----------------------------------------------------------------------
# 1. flags-off equals the frozen official class
# ----------------------------------------------------------------------
def test_flags_off_matches_official_cqn_as():
    official = _official_agent()
    variant = _variant_agent()

    assert "flow_policy" not in variant.params
    assert variant.flow_policy_ema_params is None
    assert "flow_policy_ema_params" not in variant.state_dict()
    assert _tree_shapes(official.params) == _tree_shapes(variant.params)
    official_flat = flatten_dict(official.params)
    variant_flat = flatten_dict(variant.params)
    for key, value in official_flat.items():
        np.testing.assert_allclose(
            np.asarray(value), np.asarray(variant_flat[key]), atol=0.0
        )

    official.logging = True
    variant.logging = True
    official_metrics = official.update(iter([_batch()]), step=1)
    variant_metrics = variant.update(iter([_batch()]), step=1)

    assert set(official_metrics) == set(variant_metrics)
    np.testing.assert_allclose(
        float(variant_metrics["critic_loss"]),
        float(official_metrics["critic_loss"]),
        atol=1e-6,
    )
    assert _tree_shapes(official.params) == _tree_shapes(variant.params)
    post_official = flatten_dict(official.params)
    post_variant = flatten_dict(variant.params)
    for key, value in post_official.items():
        np.testing.assert_allclose(
            np.asarray(value), np.asarray(post_variant[key]), atol=1e-6
        )

    observations = {
        "low_dim_state": np.zeros((1, 1, LOW_DIM), dtype=np.float32)
    }
    official_action = official.act(observations, step=3000, eval_mode=True)
    variant_action = variant.act(observations, step=3000, eval_mode=True)
    np.testing.assert_allclose(
        np.asarray(variant_action), np.asarray(official_action), atol=1e-6
    )


# ----------------------------------------------------------------------
# 2. flags-on sanity
# ----------------------------------------------------------------------
def test_coarse_flow_trains_both_towers_and_acts():
    agent = _variant_agent(*_COARSE_FLOW_OVERRIDES)
    assert "flow_policy" in agent.params

    critic_before = jax.tree.map(np.asarray, agent.params["critic"])
    flow_before = jax.tree.map(np.asarray, agent.params["flow_policy"])
    agent.logging = True
    agent.update(iter([_batch()]), step=1)
    metrics = agent.update(iter([_batch()]), step=2)

    assert _tree_changed(critic_before, agent.params["critic"])
    assert _tree_changed(flow_before, agent.params["flow_policy"])
    assert "coarse_flow_loss" in metrics
    assert metrics["coarse_flow_loss"] > 0.0
    for key, value in metrics.items():
        assert np.all(np.isfinite(np.asarray(value))), key

    observations = {
        "low_dim_state": np.zeros((1, 1, LOW_DIM), dtype=np.float32)
    }
    action = agent.act(observations, step=3000, eval_mode=True)
    assert action.shape == (1, ACTION_SEQUENCE, ACTION_DIM)
    assert np.all(np.isfinite(np.asarray(action)))
    assert np.all(np.asarray(action) >= -1.0)
    assert np.all(np.asarray(action) <= 1.0)
    train_action = agent.act(observations, step=3000, eval_mode=False)
    assert np.all(np.isfinite(np.asarray(train_action)))


def test_coarse_flow_cell_roundtrips_recorded_actions():
    agent = _variant_agent(*_COARSE_FLOW_OVERRIDES)
    rng = np.random.default_rng(3)
    actions = jnp.asarray(
        rng.uniform(-1.0, 1.0, size=(4, agent._flat_action_dim)),
        dtype=jnp.float32,
    )
    indices = encode_action(
        actions,
        agent.action_low,
        agent.action_high,
        agent.levels,
        agent.bins,
    )
    bin_context, cell_low, cell_width = agent._coarse_flow_cell(indices)
    assert bin_context.shape == (
        4,
        agent.action_sequence,
        agent.levels * agent.bins * agent.action_dim + agent.action_dim,
    )
    low = np.asarray(cell_low)
    width = np.asarray(cell_width)
    acts = np.asarray(actions)
    assert np.all(acts >= low - 1e-5)
    assert np.all(acts <= low + width + 1e-5)
    u1 = np.clip(2.0 * (acts - low) / width - 1.0, -1.0, 1.0)
    np.testing.assert_allclose(
        low + (u1 + 1.0) * 0.5 * width, acts, atol=1e-5
    )
    one_hot_block = np.asarray(bin_context)[
        ..., : agent.levels * agent.action_dim * agent.bins
    ].reshape((4, agent.action_sequence, -1, agent.bins))
    np.testing.assert_allclose(one_hot_block.sum(axis=-1), 1.0, atol=1e-6)


def test_coarse_flow_action_stays_inside_selected_cell():
    agent = _variant_agent(*_COARSE_FLOW_OVERRIDES)
    rng = np.random.default_rng(5)
    indices = jnp.asarray(
        rng.integers(
            0,
            agent.bins,
            size=(2, agent.levels, agent.action_sequence, agent.action_dim),
        ),
        dtype=jnp.int32,
    )
    features = jnp.zeros((2, LOW_DIM), dtype=jnp.float32)
    action = agent._coarse_flow_action(
        agent.params["flow_policy"],
        features,
        indices,
        jax.random.PRNGKey(0),
    )
    assert action.shape == (2, agent.action_sequence, agent.action_dim)
    _, cell_low, cell_width = agent._coarse_flow_cell(indices)
    low = np.asarray(cell_low).reshape(action.shape)
    high = low + np.asarray(cell_width).reshape(action.shape)
    assert np.all(np.asarray(action) >= low - 1e-5)
    assert np.all(np.asarray(action) <= high + 1e-5)


def test_coarse_flow_pure_trains_and_acts_full_range():
    agent = _variant_agent(
        *_COARSE_FLOW_OVERRIDES, "method.coarse_flow_pure=true"
    )
    conditioned = _variant_agent(*_COARSE_FLOW_OVERRIDES)
    flat_pure = flatten_dict(agent.params["flow_policy"])
    flat_cond = flatten_dict(conditioned.params["flow_policy"])
    key = next(k for k in flat_pure if "flow_dense_0" in k and k[-1] == "kernel")
    assert flat_pure[key].shape[0] < flat_cond[key].shape[0]

    flow_before = jax.tree.map(np.asarray, agent.params["flow_policy"])
    agent.logging = True
    agent.update(iter([_batch()]), step=1)
    metrics = agent.update(iter([_batch()]), step=2)
    assert _tree_changed(flow_before, agent.params["flow_policy"])
    assert metrics["coarse_flow_loss"] > 0.0

    features = jnp.zeros((2, LOW_DIM), dtype=jnp.float32)
    action = agent._coarse_flow_action(
        agent.params["flow_policy"], features, None, jax.random.PRNGKey(0)
    )
    assert action.shape == (2, agent.action_sequence, agent.action_dim)
    assert np.all(np.asarray(action) >= -1.0)
    assert np.all(np.asarray(action) <= 1.0)

    observations = {
        "low_dim_state": np.zeros((1, 1, LOW_DIM), dtype=np.float32)
    }
    out = agent.act(observations, step=3000, eval_mode=True)
    assert np.all(np.isfinite(np.asarray(out)))


def test_coarse_flow_ema_tracks_online_weights_and_checkpoints():
    agent = _variant_agent(
        *_COARSE_FLOW_OVERRIDES, "method.flow_policy_ema=0.5"
    )
    assert agent.flow_policy_ema_params is not None
    agent.update(iter([_batch()]), step=1)
    agent.update(iter([_batch()]), step=2)
    online = jax.tree.leaves(agent.params["flow_policy"])
    ema = jax.tree.leaves(agent.flow_policy_ema_params)
    assert any(
        not np.allclose(np.asarray(left), np.asarray(right))
        for left, right in zip(online, ema, strict=True)
    )

    state = agent.state_dict()
    assert "flow_policy_ema_params" in state
    restored = _variant_agent(
        *_COARSE_FLOW_OVERRIDES, "method.flow_policy_ema=0.5"
    )
    restored.load_state_dict(state)
    for left, right in zip(
        jax.tree.leaves(agent.flow_policy_ema_params),
        jax.tree.leaves(restored.flow_policy_ema_params),
        strict=True,
    ):
        np.testing.assert_allclose(np.asarray(left), np.asarray(right))

    observations = {
        "low_dim_state": np.zeros((1, 1, LOW_DIM), dtype=np.float32)
    }
    action = agent.act(observations, step=3000, eval_mode=True)
    assert np.all(np.isfinite(np.asarray(action)))


def test_flow_policy_sample_and_rerank_pick_argmax_candidate():
    # The rerank helpers are the decoupled-flow (Stage-146) mechanism; the
    # unconditioned coarse_flow_pure head has the same signature, so they can
    # be exercised without the bc-policy line's separate_bc_policy substrate.
    agent = _variant_agent(
        *_COARSE_FLOW_OVERRIDES, "method.coarse_flow_pure=true"
    )
    features = jnp.zeros((2, LOW_DIM), dtype=jnp.float32)
    chunks = agent._flow_policy_sample(
        agent.params["flow_policy"], features, jax.random.PRNGKey(0), 4
    )
    assert chunks.shape == (2, 4, agent.action_sequence, agent.action_dim)
    assert np.all(np.asarray(chunks) >= -1.0)
    assert np.all(np.asarray(chunks) <= 1.0)

    selected, scores = agent._flow_rerank_action(
        agent.target_critic_params, features, chunks
    )
    assert selected.shape == (2, agent.action_sequence, agent.action_dim)
    assert scores.shape == (2, 4)
    best = np.argmax(np.asarray(scores), axis=-1)
    for row in range(2):
        np.testing.assert_allclose(
            np.asarray(selected[row]), np.asarray(chunks[row, best[row]])
        )


def test_coarse_flow_selfdistill_reweights_online_chunks():
    agent = _variant_agent(
        *_COARSE_FLOW_OVERRIDES,
        "method.coarse_flow_selfdistill_weight=1.0",
        "method.coarse_flow_selfdistill_threshold=0.5",
    )
    baseline = _variant_agent(*_COARSE_FLOW_OVERRIDES)

    batch = _batch()
    batch["demo"] = np.zeros((batch["reward"].shape[0],), dtype=np.uint8)
    batch["mc_return"] = np.ones(
        (batch["reward"].shape[0],), dtype=np.float32
    )
    agent.logging = True
    baseline.logging = True
    with_selfdistill = agent.update(iter([batch]), step=1)
    without_selfdistill = baseline.update(iter([batch]), step=1)

    # No demos in the batch: the plain arm has no flow training signal while
    # the self-distilling arm trains on the high-return online chunks.
    assert without_selfdistill["coarse_flow_loss"] == pytest.approx(0.0)
    assert with_selfdistill["coarse_flow_loss"] > 0.0


def test_selfdistill_without_coarse_flow_is_inert():
    # Research ignores the self-distillation weight when coarse_flow is off
    # (use_coarse_flow gates the whole block); the variant must stay on the
    # pristine 12-argument update path rather than threading mc_return.
    variant = _variant_agent(
        "method.coarse_flow_selfdistill_weight=1.0",
    )
    official = _official_agent()
    assert "flow_policy" not in variant.params
    variant.logging = True
    official.logging = True
    variant_metrics = variant.update(iter([_batch()]), step=1)
    official_metrics = official.update(iter([_batch()]), step=1)
    assert set(variant_metrics) == set(official_metrics)
    np.testing.assert_allclose(
        float(variant_metrics["critic_loss"]),
        float(official_metrics["critic_loss"]),
        atol=1e-6,
    )


# ----------------------------------------------------------------------
# 3. flags-on equals the research monolith
# ----------------------------------------------------------------------
@pytest.mark.parametrize("extra", [(), ("method.coarse_flow_pure=true",)])
def test_coarse_flow_matches_research_monolith(extra):
    variant = _variant_agent(*_COARSE_FLOW_OVERRIDES, *extra)
    research = _research_agent(*_COARSE_FLOW_OVERRIDES, *extra)

    assert _tree_shapes(variant.params["flow_policy"]) == _tree_shapes(
        research.params["flow_policy"]
    )
    variant_flat = flatten_dict(variant.params["flow_policy"])
    research_flat = flatten_dict(research.params["flow_policy"])
    for key, value in variant_flat.items():
        np.testing.assert_allclose(
            np.asarray(value), np.asarray(research_flat[key]), atol=0.0
        )

    variant.logging = True
    research.logging = True
    variant_metrics = variant.update(iter([_batch()]), step=1)
    research_metrics = research.update(iter([_batch()]), step=1)
    np.testing.assert_allclose(
        float(variant_metrics["coarse_flow_loss"]),
        float(research_metrics["coarse_flow_loss"]),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        float(variant_metrics["critic_loss"]),
        float(research_metrics["critic_loss"]),
        atol=1e-5,
    )
    for key, value in flatten_dict(variant.params["flow_policy"]).items():
        np.testing.assert_allclose(
            np.asarray(value),
            np.asarray(flatten_dict(research.params["flow_policy"])[key]),
            atol=1e-6,
        )


def test_coarse_flow_matches_research_monolith_with_pixels():
    # The flow head only splits into rgb / low-dim projection streams when
    # low_dim_size < feature width, i.e. on the pixel platform the waves ran.
    overrides = (
        *_COARSE_FLOW_OVERRIDES,
        "pixels=true",
        "method.flow_policy_hidden_dims=[16,16]",
        "method.flow_policy_gru_layers=2",
    )
    variant = _variant_agent(*overrides, pixels=True)
    research = _research_agent(*overrides, pixels=True)
    assert _tree_shapes(variant.params["flow_policy"]) == _tree_shapes(
        research.params["flow_policy"]
    )
    assert any(
        "flow_rgb_projection" in key for key in flatten_dict(
            variant.params["flow_policy"]
        )
    )
    assert any(
        "flow_gru_1" in key for key in flatten_dict(
            variant.params["flow_policy"]
        )
    )

    variant.logging = True
    research.logging = True
    variant_metrics = variant.update(iter([_batch(pixels=True)]), step=1)
    research_metrics = research.update(iter([_batch(pixels=True)]), step=1)
    np.testing.assert_allclose(
        float(variant_metrics["coarse_flow_loss"]),
        float(research_metrics["coarse_flow_loss"]),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        float(variant_metrics["critic_loss"]),
        float(research_metrics["critic_loss"]),
        atol=1e-5,
    )


# ----------------------------------------------------------------------
# Validation / coupling guards
# ----------------------------------------------------------------------
def test_flow_policy_flag_reports_bc_policy_coupling():
    with pytest.raises(ValueError, match="separate_bc_policy"):
        _variant_agent("method.flow_policy=true")


def test_policy_value_beta_reports_bc_policy_coupling():
    with pytest.raises(ValueError, match="separate_bc_policy"):
        _variant_agent("method.policy_value_beta=1.0")


def test_coarse_flow_pure_requires_coarse_flow():
    with pytest.raises(ValueError, match="coarse_flow=true"):
        _variant_agent("method.coarse_flow_pure=true")


def test_flow_policy_hyperparameter_validation():
    with pytest.raises(ValueError, match="flow_policy_steps"):
        _variant_agent("method.flow_policy_steps=0")
    with pytest.raises(ValueError, match="flow_policy_lambda"):
        _variant_agent("method.flow_policy_lambda=-1.0")
    with pytest.raises(ValueError, match="flow_policy_ema"):
        _variant_agent("method.flow_policy_ema=1.5")
    with pytest.raises(ValueError, match="flow_policy_candidates"):
        _variant_agent("method.flow_policy_candidates=0")
    with pytest.raises(ValueError, match="selfdistill"):
        _variant_agent(
            *_COARSE_FLOW_OVERRIDES,
            "method.coarse_flow_selfdistill_weight=-0.5",
        )


def test_yaml_defaults_are_all_off():
    cfg = _compose("cqn_as_flow_policy")
    spec = cqn_as_flow_policy_spec_from_cfg(cfg)
    assert cfg.method.name == "cqn_as_flow_policy"
    assert (
        cfg.method._target_
        == "robobase.method.cqn_as_flow_policy.CQNASFlowPolicy"
    )
    assert spec.flow_policy is False
    assert spec.coarse_flow is False
    assert spec.coarse_flow_pure is False
    assert spec.coarse_flow_selfdistill_weight is None
    assert spec.coarse_flow_selfdistill_threshold == pytest.approx(0.5)
    assert spec.policy_value_beta is None
    assert spec.flow_policy_candidates == 8
    assert spec.flow_policy_steps == 8
    assert spec.flow_policy_lambda == pytest.approx(1.0)
    assert spec.flow_policy_ema is None
    assert spec.flow_policy_hidden_dims is None
    assert spec.flow_policy_gru_layers is None
