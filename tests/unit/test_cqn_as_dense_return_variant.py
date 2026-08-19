"""CPU-only verification for the dense-return CQN-AS variant (R2).

Covers the three checks required by ``R2_COMMON_BRIEF.md``:

1. flags-off ``CQNASDenseReturn`` is numerically identical to the pristine
   ``robobase.method.cqn_as.CQNAS`` (same seed, same synthetic batch);
2. flags-on runs ``act()`` + ``update()`` with finite metrics and this line's
   metric keys;
3. flags-on ``critic_loss`` matches ``cqn_as_research.CQNAS`` configured with
   the same flags.

It also re-encodes the behavioural spec of the four research-era tests
(``test_cqn_as_label_smoothing``, ``test_cqn_as_satisficing_floor``,
``test_cqn_as_relative_floor``, ``test_cqn_as_return_gated_margin``) against
this module's own copies of the loss functions, and pins the documented
coupling to the mc-rct line.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import functools  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from gymnasium import spaces  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402

from robobase.factory import create_agent  # noqa: E402
from robobase.method.cqn_as_dense_return import (  # noqa: E402
    CQNASDenseReturn,
    cqn_as_dense_return_spec_from_cfg,
    dense_return_distributional_loss,
    episodic_success_returns,
    ordered_success_returns,
    return_gated_margin_loss,
    sequence_aligned_sparse_returns,
    unseen_return_floor_loss,
)

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
CONFIG_DIR = os.path.join(REPO_ROOT, "robobase", "cfgs")

ACTION_SEQUENCE = 4
ACTION_DIM = 8
LOW_DIM = 5
RGB_KEY = "rgb_head"
# BiGym-shaped pixel observation: [time=1, frame_stack * 3 = 12, 84, 84].
RGB_SHAPE = (1, 12, 84, 84)
BATCH = 2


# ---------------------------------------------------------------------------
# synthetic spaces / batches (mirrors scripts/refactor_equivalence_check.py)
# ---------------------------------------------------------------------------


def _compose(method: str, *overrides: str, pixels: bool = True):
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
                f"pixels={'true' if pixels else 'false'}",
                f"action_sequence={ACTION_SEQUENCE}",
                *overrides,
            ],
        )


def _spaces(pixels: bool = True):
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


def _observation(pixels: bool = True):
    rng = np.random.default_rng(3)
    observation = {
        "low_dim_state": rng.normal(size=(BATCH, 1, LOW_DIM)).astype(
            np.float32
        )
    }
    if pixels:
        observation[RGB_KEY] = rng.integers(
            0, 256, size=(BATCH, *RGB_SHAPE), dtype=np.uint8
        )
    return observation


def _batch(
    pixels: bool = True,
    with_mc_return: bool = True,
    positive_reward: bool = False,
):
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
        "demo": np.ones((BATCH,), dtype=np.uint8),
    }
    if pixels:
        batch[RGB_KEY] = rng.integers(
            0, 256, size=(BATCH, *RGB_SHAPE), dtype=np.uint8
        )
        batch[f"{RGB_KEY}_tp1"] = rng.integers(
            0, 256, size=(BATCH, *RGB_SHAPE), dtype=np.uint8
        )
    if positive_reward:
        # A sparse-success shaped batch: without a strictly positive target
        # the relative floor clips straight back to the absolute floor and
        # several arms degenerate into the plain dense target.
        batch["reward"] = np.asarray([1.0, 0.0], dtype=np.float32)
    if with_mc_return:
        # One successful and one failed completed trajectory, so every
        # return-gated branch has both sides of its mask exercised.
        batch["mc_return"] = np.asarray([0.9, 0.0], dtype=np.float32)
    return batch


# ---------------------------------------------------------------------------
# construction (this is also the factory registration snippet)
# ---------------------------------------------------------------------------


def _build_dense_return_agent(cfg, observation_space, action_space):
    spec = cqn_as_dense_return_spec_from_cfg(cfg)
    return CQNASDenseReturn(
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
        strict_demo_rl_only=spec.strict_demo_rl_only,
        dense_return_q_target=spec.dense_return_q_target,
        dense_return_positive_only=spec.dense_return_positive_only,
        dense_return_expected_q_loss=spec.dense_return_expected_q_loss,
        dense_return_advantage_alpha=spec.dense_return_advantage_alpha,
        dense_return_advantage_clip_ratio=(
            spec.dense_return_advantage_clip_ratio
        ),
        q_reward_scale=spec.q_reward_scale,
        dense_return_label_smoothing=spec.dense_return_label_smoothing,
        dense_return_floor_satisfaction_margin=(
            spec.dense_return_floor_satisfaction_margin
        ),
        dense_return_relative_floor_margin=(
            spec.dense_return_relative_floor_margin
        ),
        return_gated_margin=spec.return_gated_margin,
        return_gated_margin_weight=spec.return_gated_margin_weight,
        dense_return_finest_neighbor_weight=(
            spec.dense_return_finest_neighbor_weight
        ),
        episodic_success_q_target=spec.episodic_success_q_target,
        ordered_success_return_mix=spec.ordered_success_return_mix,
        sequence_aligned_mc_discount=spec.sequence_aligned_mc_discount,
        unseen_return_floor_weight=spec.unseen_return_floor_weight,
        unseen_return_floor_value=spec.unseen_return_floor_value,
        unseen_return_floor_reduction=spec.unseen_return_floor_reduction,
        unseen_return_floor_topk=spec.unseen_return_floor_topk,
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


def _dense_return_agent(*overrides: str, pixels: bool = True):
    cfg = _compose("cqn_as_dense_return", *overrides, pixels=pixels)
    observation_space, action_space = _spaces(pixels)
    agent = _build_dense_return_agent(cfg, observation_space, action_space)
    agent.logging = True
    return agent


def _factory_agent(method: str, *overrides: str, pixels: bool = True):
    cfg = _compose(method, *overrides, pixels=pixels)
    observation_space, action_space = _spaces(pixels)
    agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )
    agent.logging = True
    return agent


def _param_shapes(params):
    return jax.tree_util.tree_map(lambda x: tuple(np.shape(x)), params)


def _one_update(agent, batch=None):
    return agent.update(iter([batch if batch is not None else _batch()]), step=1)


PARITY_STEPS = 4


def _loss_trace(agent, batch, steps: int = PARITY_STEPS):
    """critic_loss over ``steps`` sequential updates on the same batch.

    One update is not discriminating: the pristine CQN-AS heads are
    zero-initialised, so every bin's cross-entropy collapses to
    ``-log(1/atoms)`` regardless of its target and most dense-return arms
    coincide at step 1.  Stepping the optimiser makes the per-bin targets
    matter.
    """

    return [
        float(agent.update(iter([dict(batch)]), step=step)["critic_loss"])
        for step in range(1, steps + 1)
    ]


@functools.lru_cache(maxsize=1)
def _flags_off_low_dim_trace():
    """Flags-off reference trace on the low-dim parity batch."""

    return tuple(
        _loss_trace(
            _dense_return_agent(pixels=False),
            _batch(pixels=False, positive_reward=True),
        )
    )


# ---------------------------------------------------------------------------
# 1. flags-off == pristine CQNAS
# ---------------------------------------------------------------------------


def test_flags_off_matches_pristine_cqn_as():
    pristine = _factory_agent("cqn_as_official")
    variant = _dense_return_agent()

    assert _param_shapes(pristine.params) == _param_shapes(variant.params)
    assert _param_shapes(pristine.target_critic_params) == _param_shapes(
        variant.target_critic_params
    )

    batch = _batch()
    pristine_metrics = _one_update(pristine, dict(batch))
    variant_metrics = _one_update(variant, dict(batch))

    assert set(pristine_metrics) == set(variant_metrics)
    np.testing.assert_allclose(
        variant_metrics["critic_loss"],
        pristine_metrics["critic_loss"],
        atol=1e-6,
        rtol=0.0,
    )
    for key in ("entropy", "target_entropy", "loss_coeff"):
        np.testing.assert_allclose(
            variant_metrics[key], pristine_metrics[key], atol=1e-6, rtol=0.0
        )
    assert _param_shapes(pristine.params) == _param_shapes(variant.params)

    # keep stepping: an equal first loss would also follow from an unchanged
    # forward pass with a broken optimiser/target-update path.
    np.testing.assert_allclose(
        _loss_trace(variant, batch, steps=3),
        _loss_trace(pristine, batch, steps=3),
        atol=1e-6,
        rtol=0.0,
    )
    assert _param_shapes(pristine.params) == _param_shapes(variant.params)


def test_flags_off_act_matches_pristine_cqn_as():
    pristine = _factory_agent("cqn_as_official")
    variant = _dense_return_agent()
    observation = _observation()
    np.testing.assert_allclose(
        np.asarray(variant.act(observation, step=100, eval_mode=True)),
        np.asarray(pristine.act(observation, step=100, eval_mode=True)),
        atol=1e-6,
        rtol=0.0,
    )


# ---------------------------------------------------------------------------
# 2. flags-on sanity
# ---------------------------------------------------------------------------


CANONICAL_DENSE_ARM = (
    "method.bc_lambda=0.0",
    "method.bc_margin=0.0",
    "method.strict_demo_rl_only=true",
    "method.dense_return_q_target=true",
    "method.dense_return_positive_only=true",
    "method.q_reward_scale=2.0",
    "method.return_gated_margin=0.16",
    "method.return_gated_margin_weight=1.0",
)


def test_flags_on_runs_and_reports_line_metrics():
    agent = _dense_return_agent(*CANONICAL_DENSE_ARM)
    assert agent.strict_demo_rl_only

    action = np.asarray(agent.act(_observation(), step=100, eval_mode=False))
    assert action.shape == (BATCH, ACTION_SEQUENCE, ACTION_DIM)
    assert np.all(np.isfinite(action))

    metrics = _one_update(agent)
    for key in (
        "critic_loss",
        "dense_return_q_loss",
        "dense_return_positive_fraction",
        "unseen_q_mean",
        "chosen_q_mean",
        "chosen_unseen_q_gap",
        "q_reward_scale",
        "scaled_mc_return_mean",
    ):
        assert key in metrics, sorted(metrics)
    for key, value in metrics.items():
        assert np.all(np.isfinite(np.asarray(value, dtype=np.float64))), key
    # one of the two synthetic transitions has a positive completed return
    assert metrics["dense_return_positive_fraction"] == pytest.approx(0.5)
    assert metrics["q_reward_scale"] == pytest.approx(2.0)


def test_unseen_return_floor_flags_on():
    agent = _dense_return_agent(
        "method.unseen_return_floor_weight=1.0",
        "method.unseen_return_floor_value=0.0",
        "method.unseen_return_floor_reduction=max",
        pixels=False,
    )
    batch = _batch(pixels=False, positive_reward=True)
    metrics = _one_update(agent, dict(batch))
    for key in (
        "unseen_return_floor_loss",
        "unseen_q_mean",
        "chosen_q_mean",
        "chosen_unseen_q_gap",
    ):
        assert key in metrics, sorted(metrics)
    for key, value in metrics.items():
        assert np.all(np.isfinite(np.asarray(value, dtype=np.float64))), key
    assert metrics["unseen_return_floor_loss"] >= 0.0
    # the floor must actually add a term once the zero-initialised heads move
    fresh = _dense_return_agent(
        "method.unseen_return_floor_weight=1.0",
        "method.unseen_return_floor_value=0.0",
        "method.unseen_return_floor_reduction=max",
        pixels=False,
    )
    assert max(
        abs(a - b)
        for a, b in zip(_loss_trace(fresh, batch), _flags_off_low_dim_trace())
    ) > 1e-9


def test_episodic_success_q_target_flags_on():
    agent = _dense_return_agent(
        "method.dense_return_q_target=true",
        "method.episodic_success_q_target=true",
        pixels=False,
    )
    metrics = _one_update(agent, _batch(pixels=False))
    assert "episodic_success_fraction" in metrics
    assert metrics["episodic_success_fraction"] == pytest.approx(0.5)
    for key, value in metrics.items():
        assert np.all(np.isfinite(np.asarray(value, dtype=np.float64))), key


def test_dense_return_expected_q_and_advantage_flags_on():
    expected_q = _dense_return_agent(
        "method.dense_return_q_target=true",
        "method.dense_return_expected_q_loss=true",
        pixels=False,
    )
    metrics = _one_update(expected_q, _batch(pixels=False))
    assert metrics["dense_return_expected_q_target"] == pytest.approx(1.0)

    advantage = _dense_return_agent(
        "method.dense_return_q_target=true",
        "method.dense_return_advantage_alpha=0.1",
        "method.dense_return_advantage_clip_ratio=0.5",
        pixels=False,
    )
    metrics = _one_update(advantage, _batch(pixels=False))
    assert metrics["dense_return_advantage_alpha"] == pytest.approx(0.1)
    assert metrics["dense_return_advantage_clip_ratio"] == pytest.approx(0.5)
    for key, value in metrics.items():
        assert np.all(np.isfinite(np.asarray(value, dtype=np.float64))), key


def test_mc_return_element_is_required_by_return_gated_terms():
    agent = _dense_return_agent(
        "method.dense_return_q_target=true",
        "method.dense_return_positive_only=true",
        pixels=False,
    )
    with pytest.raises(KeyError, match="mc_return"):
        _one_update(agent, _batch(pixels=False, with_mc_return=False))


# ---------------------------------------------------------------------------
# 3. flags-on == research monolith
# ---------------------------------------------------------------------------


RESEARCH_PARITY_ARMS = (
    # dense categorical target with the full label-smoothing / neighbour /
    # relative-floor stack; none of these need an mc-rct flag.
    (
        "method.dense_return_q_target=true",
        "method.dense_return_label_smoothing=0.05",
        "method.dense_return_finest_neighbor_weight=0.25",
    ),
    (
        "method.dense_return_q_target=true",
        "method.dense_return_relative_floor_margin=0.16",
    ),
    (
        "method.dense_return_q_target=true",
        "method.dense_return_floor_satisfaction_margin=0.02",
    ),
    (
        "method.dense_return_q_target=true",
        "method.dense_return_advantage_alpha=0.1",
        "method.dense_return_advantage_clip_ratio=0.5",
    ),
    (
        "method.dense_return_q_target=true",
        "method.dense_return_expected_q_loss=true",
    ),
    (
        "method.unseen_return_floor_weight=1.0",
        "method.unseen_return_floor_reduction=max",
    ),
    (
        "method.unseen_return_floor_weight=1.0",
        "method.unseen_return_floor_reduction=topk",
        "method.unseen_return_floor_topk=2",
    ),
    # episodic_success_q_target is the one arm of this line that the research
    # class also threads real mc_returns into, so the data path is identical.
    (
        "method.dense_return_q_target=true",
        "method.episodic_success_q_target=true",
    ),
)


@pytest.mark.parametrize("arm", RESEARCH_PARITY_ARMS)
def test_flags_on_matches_research_monolith(arm):
    # low-dim only: the dense-return objective is encoder-independent and the
    # pixel path is already covered by the flags-off equivalence tests.
    batch = _batch(pixels=False, positive_reward=True)
    variant_trace = _loss_trace(
        _dense_return_agent(*arm, pixels=False), batch
    )
    research_trace = _loss_trace(
        _factory_agent("cqn_as", *arm, pixels=False), batch
    )
    np.testing.assert_allclose(
        variant_trace, research_trace, atol=1e-5, rtol=0.0
    )
    # guard against a vacuous comparison: each arm must actually move the
    # objective away from the flags-off baseline.
    baseline_trace = _flags_off_low_dim_trace()
    assert max(
        abs(a - b) for a, b in zip(variant_trace, baseline_trace)
    ) > 1e-9, (arm, variant_trace, baseline_trace)


# ---------------------------------------------------------------------------
# behavioural spec (adapted from the research-era focused tests)
# ---------------------------------------------------------------------------


def _loss_setup(seed=0):
    rng = np.random.default_rng(seed)
    batch, levels, heads, bins, atoms = 2, 2, 3, 5, 51
    logits = jnp.asarray(
        rng.normal(size=(batch, levels, heads, bins, atoms)),
        dtype=jnp.float32,
    )
    action = jnp.asarray(
        rng.integers(0, bins, size=(batch, levels, heads)),
        dtype=jnp.int32,
    )
    support = jnp.linspace(-2.0, 2.0, atoms, dtype=jnp.float32)
    return logits, action, support, atoms


def _point_mass(support, value, atoms):
    dist = np.zeros((atoms,), dtype=np.float32)
    dist[int(np.argmin(np.abs(np.asarray(support) - value)))] = 1.0
    return jnp.asarray(dist)


def _dense_loss(logits, action, chosen, support, **kwargs):
    per_sample, chosen_q, unseen_q = dense_return_distributional_loss(
        logits,
        action,
        jnp.broadcast_to(chosen, logits.shape[:-2] + (logits.shape[-1],)),
        support,
        0.0,
        kwargs.get("finest_neighbor_weight", 0.0),
        kwargs.get("advantage_alpha", 0.0),
        kwargs.get("advantage_clip_ratio", None),
        kwargs.get("label_smoothing", 0.0),
        kwargs.get("floor_satisfaction_margin", None),
        kwargs.get("relative_floor_margin", None),
    )
    return per_sample.sum(), chosen_q, unseen_q


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"label_smoothing": 0.05},
        {"floor_satisfaction_margin": 0.02},
        {"relative_floor_margin": 0.16},
        {"finest_neighbor_weight": 0.25},
    ],
)
def test_zero_return_action_label_invariance(kwargs):
    """The line's operational anti-imitation test.

    At G = 0 every bin target coincides with the floor, so loss AND logit
    gradients must be exactly invariant to the recorded action label.
    """

    logits, action, support, atoms = _loss_setup()
    floor = _point_mass(support, 0.0, atoms)
    loss_a, _, _ = _dense_loss(logits, action, floor, support, **kwargs)
    grads_a = jax.grad(
        lambda lg: _dense_loss(lg, action, floor, support, **kwargs)[0]
    )(logits)
    other = (action + 2) % 5
    loss_b, _, _ = _dense_loss(logits, other, floor, support, **kwargs)
    grads_b = jax.grad(
        lambda lg: _dense_loss(lg, other, floor, support, **kwargs)[0]
    )(logits)
    assert float(loss_a) == float(loss_b)
    np.testing.assert_array_equal(np.asarray(grads_a), np.asarray(grads_b))


def test_label_smoothing_changes_positive_return_loss():
    logits, action, support, atoms = _loss_setup()
    chosen = _point_mass(support, 1.0, atoms)
    base, _, _ = _dense_loss(logits, action, chosen, support)
    smoothed, _, _ = _dense_loss(
        logits, action, chosen, support, label_smoothing=0.05
    )
    assert float(base) != float(smoothed)


def test_satisficing_floor_suspends_only_satisfied_floor_bins():
    _, _, support, atoms = _loss_setup()
    zero_idx = int(np.argmin(np.abs(np.asarray(support))))
    logits = np.zeros((1, 1, 1, 3, atoms), dtype=np.float32)
    logits[0, 0, 0, 0, zero_idx] = 20.0  # chosen bin, at the floor
    logits[0, 0, 0, 1, zero_idx] = 20.0  # unseen, satisfied
    logits[0, 0, 0, 2, atoms - 1] = 20.0  # unseen, far above the floor
    logits = jnp.asarray(logits)
    action = jnp.zeros((1, 1, 1), dtype=jnp.int32)
    chosen = _point_mass(support, 1.0, atoms)
    grads = jax.grad(
        lambda lg: _dense_loss(
            lg, action, chosen, support, floor_satisfaction_margin=0.02
        )[0]
    )(logits)
    g = np.abs(np.asarray(grads))[0, 0, 0]
    assert g[1].max() == 0.0
    assert g[2].max() > 0.0
    assert g[0].max() > 0.0


def test_relative_floor_targets_chosen_minus_margin():
    logits, action, support, atoms = _loss_setup()
    chosen = _point_mass(support, 1.0, atoms)
    margin = 0.16
    grad_fn = jax.grad(
        lambda lg: _dense_loss(
            lg, action, chosen, support, relative_floor_margin=margin
        )[0]
    )
    lg = logits
    for _ in range(300):
        lg = lg - 2.0 * grad_fn(lg)
    _, chosen_q, unseen_q = _dense_loss(
        lg, action, chosen, support, relative_floor_margin=margin
    )
    np.testing.assert_allclose(
        np.asarray(unseen_q), np.asarray(chosen_q) - margin, atol=0.03
    )
    # the chosen bin's own target is untouched by the relative floor
    _, chosen_abs, _ = _dense_loss(logits, action, chosen, support)
    _, chosen_rel, _ = _dense_loss(
        logits, action, chosen, support, relative_floor_margin=margin
    )
    np.testing.assert_array_equal(
        np.asarray(chosen_abs), np.asarray(chosen_rel)
    )


def test_return_gated_margin_is_gated_per_sample_by_measured_return():
    rng = np.random.default_rng(0)
    logits = jnp.asarray(
        rng.normal(size=(4, 2, 3, 5, 51)), dtype=jnp.float32
    )
    action = jnp.asarray(
        rng.integers(0, 5, size=(4, 2, 3)), dtype=jnp.int32
    )
    support = jnp.linspace(-2.0, 2.0, 51, dtype=jnp.float32)

    zero = jnp.zeros((4,), dtype=bool)
    loss = return_gated_margin_loss(logits, action, support, 0.16, zero)
    assert float(loss.sum()) == 0.0
    grads = jax.grad(
        lambda lg: return_gated_margin_loss(
            lg, action, support, 0.16, zero
        ).sum()
    )(logits)
    assert float(jnp.abs(grads).max()) == 0.0
    other = (action + 2) % 5
    assert float(
        return_gated_margin_loss(logits, other, support, 0.16, zero).sum()
    ) == float(loss.sum())

    mixed = jnp.asarray([True, False, True, False])
    mixed_loss = np.asarray(
        return_gated_margin_loss(logits, action, support, 0.16, mixed)
    )
    assert mixed_loss[1] == 0.0 and mixed_loss[3] == 0.0
    assert mixed_loss[0] > 0.0 and mixed_loss[2] > 0.0


def test_unseen_return_floor_excludes_the_replayed_bin():
    logits, action, support, _ = _loss_setup()
    per_sample, unseen_q = unseen_return_floor_loss(
        logits, action, support, 0.0
    )
    assert per_sample.shape == (logits.shape[0],)
    assert np.all(np.isfinite(np.asarray(per_sample)))
    assert np.all(np.isfinite(np.asarray(unseen_q)))
    grads = jax.grad(
        lambda lg: unseen_return_floor_loss(lg, action, support, 0.0)[0].sum()
    )(logits)
    chosen_grad = np.take_along_axis(
        np.asarray(grads),
        np.asarray(action)[..., None, None],
        axis=-2,
    )
    assert np.abs(chosen_grad).max() == 0.0


def test_return_transform_helpers():
    returns = jnp.asarray([0.0, 0.5, -1.0, 1.0], dtype=jnp.float32)
    np.testing.assert_array_equal(
        np.asarray(episodic_success_returns(returns)),
        np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(ordered_success_returns(returns, 0.5)),
        np.asarray([0.0, 0.75, 0.0, 1.0], dtype=np.float32),
        atol=1e-6,
    )
    aligned = np.asarray(
        sequence_aligned_sparse_returns(
            jnp.asarray([0.0, 0.9], dtype=jnp.float32), 2, 2, 0.99
        )
    )
    np.testing.assert_array_equal(aligned[0], np.zeros((4,), dtype=np.float32))
    np.testing.assert_allclose(aligned[1][:2], np.asarray([0.9, 0.9]), atol=1e-6)
    np.testing.assert_allclose(
        aligned[1][2:], np.asarray([0.9 / 0.99, 0.9 / 0.99]), atol=1e-6
    )


# ---------------------------------------------------------------------------
# validation + documented mc-rct coupling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides,match",
    [
        (("method.dense_return_positive_only=true",), "dense_return_q_target"),
        (("method.dense_return_label_smoothing=0.05",), "dense_return_q_target"),
        (("method.q_reward_scale=2.0",), "dense_return_q_target"),
        (
            (
                "method.dense_return_q_target=true",
                "method.unseen_return_floor_weight=1.0",
            ),
            "unseen_return_floor_weight=0",
        ),
        (
            (
                "method.unseen_return_floor_weight=1.0",
                "method.unseen_return_floor_reduction=topk",
                "method.unseen_return_floor_topk=5",
            ),
            "unseen_return_floor_topk",
        ),
        (
            (
                "method.dense_return_q_target=true",
                "method.return_gated_margin=0.16",
            ),
            "return_gated_margin_weight",
        ),
    ],
)
def test_invalid_combinations_are_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        _dense_return_agent(*overrides, pixels=False)


@pytest.mark.parametrize(
    "overrides",
    [
        (
            "method.dense_return_q_target=true",
            "method.ordered_success_return_mix=0.5",
        ),
        (
            "method.dense_return_q_target=true",
            "method.sequence_aligned_mc_discount=0.99",
        ),
    ],
)
def test_mc_lower_bound_coupled_flags_are_refused(overrides):
    """Coupling protocol: these two transforms have no consumer here.

    Their only consumer in the monolith is the mc-rct ``mc_lower_bound_target``
    max(TD, MC) branch, so the variant refuses them instead of silently
    ignoring them or absorbing a foreign flag.
    """

    with pytest.raises(ValueError, match="mc_lower_bound_target"):
        _dense_return_agent(*overrides, pixels=False)


def test_strict_demo_rl_only_rejects_imitation_paths():
    with pytest.raises(ValueError, match="strict_demo_rl_only"):
        _dense_return_agent("method.strict_demo_rl_only=true", pixels=False)
    agent = _dense_return_agent(
        "method.strict_demo_rl_only=true",
        "method.bc_lambda=0.0",
        "method.bc_margin=0.0",
        pixels=False,
    )
    assert agent.strict_demo_rl_only
