"""CPU-only checks for the ``guards-schedules`` CQN-AS variant.

Covers the R2 verification protocol for ``robobase/method/cqn_as_guards.py``:

1. flags-off is numerically identical to the pristine official ``CQNAS``;
2. flags-on runs, is finite, and emits this line's metric keys;
3. flags-on matches the research monolith's ``critic_loss``;
4. the guard actually skips a non-finite update (the behaviour
   ``workspace._guard_non_finite_update`` depends on).
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
from robobase.method.cqn_as_guards import (  # noqa: E402
    CQNASGuarded,
    cqn_as_guards_spec_from_cfg,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = str((REPO_ROOT / "robobase" / "cfgs").resolve())

ACTION_SEQUENCE = 4
ACTION_DIM = 8
LOW_DIM = 5
BATCH = 4


def _compose(method: str, *overrides: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(
            config_name="robobase_config",
            overrides=[
                f"method={method}",
                f"action_sequence={ACTION_SEQUENCE}",
                "num_train_envs=2",
                "num_eval_envs=2",
                "num_explore_steps=0",
                "backend.jit=false",
                "backend.platform=cpu",
                "method.model.hidden_dims=[32,32]",
                "method.atoms=11",
                "method.weight_decay=0.0",
                *overrides,
            ],
        )


def _spaces():
    observation_space = spaces.Dict(
        {
            "low_dim_state": spaces.Box(
                -np.inf, np.inf, shape=(1, LOW_DIM), dtype=np.float32
            )
        }
    )
    action_space = spaces.Box(
        -1.0, 1.0, shape=(ACTION_SEQUENCE, ACTION_DIM), dtype=np.float32
    )
    return observation_space, action_space


def _observation():
    rng = np.random.default_rng(3)
    return {
        "low_dim_state": rng.normal(size=(BATCH, 1, LOW_DIM)).astype(np.float32),
    }


def _batch():
    rng = np.random.default_rng(7)
    demo = np.zeros((BATCH,), dtype=np.uint8)
    # Mixed demo/online rows so bc_agreement and bc_online_agreement both have
    # a non-degenerate normaliser.
    demo[: BATCH // 2] = 1
    return {
        "low_dim_state": rng.normal(size=(BATCH, 1, LOW_DIM)).astype(np.float32),
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
        "demo": demo,
    }


RGB_KEY = "rgb_head"
# BiGym-shaped pixel observation: [time=1, frame_stack * 3 = 12, 84, 84].
RGB_SHAPE = (1, 12, 84, 84)
PIXEL_BATCH = 2


def _pixel_spaces():
    observation_space, action_space = _spaces()
    observation_space = spaces.Dict(
        {
            **observation_space.spaces,
            RGB_KEY: spaces.Box(0, 255, shape=RGB_SHAPE, dtype=np.uint8),
        }
    )
    return observation_space, action_space


def _pixel_batch():
    rng = np.random.default_rng(11)
    batch = {
        "low_dim_state": rng.normal(size=(PIXEL_BATCH, 1, LOW_DIM)).astype(
            np.float32
        ),
        "low_dim_state_tp1": rng.normal(size=(PIXEL_BATCH, 1, LOW_DIM)).astype(
            np.float32
        ),
        "action": rng.uniform(
            -1.0, 1.0, size=(PIXEL_BATCH, ACTION_SEQUENCE, ACTION_DIM)
        ).astype(np.float32),
        "reward": rng.normal(size=(PIXEL_BATCH,)).astype(np.float32),
        "discount": np.full((PIXEL_BATCH,), 0.99, dtype=np.float32),
        "terminal": np.zeros((PIXEL_BATCH,), dtype=bool),
        "truncated": np.zeros((PIXEL_BATCH,), dtype=bool),
        "demo": np.ones((PIXEL_BATCH,), dtype=np.uint8),
        RGB_KEY: rng.integers(
            0, 256, size=(PIXEL_BATCH, *RGB_SHAPE), dtype=np.uint8
        ),
        f"{RGB_KEY}_tp1": rng.integers(
            0, 256, size=(PIXEL_BATCH, *RGB_SHAPE), dtype=np.uint8
        ),
    }
    return batch


def _pristine_kwargs(cfg, spec, observation_space, action_space):
    """The exact kwargs ``factory.create_agent`` builds for CQN-AS."""

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
        jit=False,
        platform="cpu",
        seed=int(cfg.seed),
    )


def _official_agent(*overrides: str):
    cfg = _compose("cqn_as_official", *overrides)
    observation_space, action_space = _spaces()
    spec = cqn_as_spec_from_cfg(cfg)
    return CQNAS(
        **_pristine_kwargs(cfg, spec, observation_space, action_space)
    )


def _guarded_agent(*overrides: str):
    """Mirror of the factory branch this variant needs (see report snippet)."""

    cfg = _compose("cqn_as_guards", *overrides)
    observation_space, action_space = _spaces()
    spec = cqn_as_guards_spec_from_cfg(cfg)
    return CQNASGuarded(
        **_pristine_kwargs(cfg, spec, observation_space, action_space),
        nonfinite_guard=spec.nonfinite_guard,
        bc_diagnostics=spec.bc_diagnostics,
        bc_lambda_schedule=spec.bc_lambda_schedule,
        demo_fosd=spec.demo_fosd,
    )


def _research_agent(*overrides: str):
    cfg = _compose("cqn_as", *overrides)
    observation_space, action_space = _spaces()
    return create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )


def _run_update(agent, *, step: int = 1, batch=None):
    agent.logging = True
    return agent.update(iter([batch if batch is not None else _batch()]), step=step)


# --------------------------------------------------------------------------
# 1. flags-off == pristine
# --------------------------------------------------------------------------


def test_yaml_defaults_declare_the_line_keys():
    cfg = _compose("cqn_as_guards")
    assert cfg.method.name == "cqn_as_guards"
    assert (
        cfg.method._target_ == "robobase.method.cqn_as_guards.CQNASGuarded"
    )
    assert cfg.method.nonfinite_guard is True
    assert cfg.method.bc_diagnostics is True
    assert cfg.method.bc_lambda_schedule is None
    assert cfg.method.demo_fosd is True
    spec = cqn_as_guards_spec_from_cfg(cfg)
    assert spec.nonfinite_guard and spec.bc_diagnostics
    assert spec.bc_lambda_schedule is None and spec.demo_fosd


def test_class_defaults_are_all_off():
    agent = _guarded_agent(
        "method.nonfinite_guard=false",
        "method.bc_diagnostics=false",
    )
    assert agent.nonfinite_guard is False
    assert agent.bc_diagnostics is False
    assert agent.bc_lambda_schedule is None
    assert agent.demo_fosd is True


def test_flags_off_update_matches_pristine_cqn_as():
    official = _official_agent()
    guarded = _guarded_agent(
        "method.nonfinite_guard=false",
        "method.bc_diagnostics=false",
    )

    batch = _batch()
    official_metrics = _run_update(official, batch=batch)
    guarded_metrics = _run_update(guarded, batch=dict(batch))

    assert guarded_metrics["critic_loss"] == pytest.approx(
        official_metrics["critic_loss"], abs=1e-6
    )
    for key in ("entropy", "target_entropy", "loss_coeff"):
        assert guarded_metrics[key] == pytest.approx(
            official_metrics[key], abs=1e-6
        )
    # No extra metric keys leak when the flags are off.
    assert set(guarded_metrics) == set(official_metrics)

    official_leaves, official_tree = jax.tree.flatten(official.params)
    guarded_leaves, guarded_tree = jax.tree.flatten(guarded.params)
    assert official_tree == guarded_tree
    for official_leaf, guarded_leaf in zip(
        official_leaves, guarded_leaves, strict=True
    ):
        assert official_leaf.shape == guarded_leaf.shape
        np.testing.assert_allclose(
            np.asarray(guarded_leaf), np.asarray(official_leaf), atol=1e-6
        )


def test_flags_off_update_matches_pristine_with_pixels():
    """Covers the RGB path, where _augment_update_obs_inputs splits the key.

    A mis-threaded schedule argument would desynchronise the augmentation /
    action RNG streams, which low-dim-only observations cannot detect.
    """

    observation_space, action_space = _pixel_spaces()
    official_cfg = _compose("cqn_as_official", "pixels=true")
    guarded_cfg = _compose(
        "cqn_as_guards",
        "pixels=true",
        "method.nonfinite_guard=false",
        "method.bc_diagnostics=false",
    )
    official = CQNAS(
        **_pristine_kwargs(
            official_cfg,
            cqn_as_spec_from_cfg(official_cfg),
            observation_space,
            action_space,
        )
    )
    guarded_spec = cqn_as_guards_spec_from_cfg(guarded_cfg)
    guarded = CQNASGuarded(
        **_pristine_kwargs(
            guarded_cfg, guarded_spec, observation_space, action_space
        ),
        nonfinite_guard=guarded_spec.nonfinite_guard,
        bc_diagnostics=guarded_spec.bc_diagnostics,
        bc_lambda_schedule=guarded_spec.bc_lambda_schedule,
        demo_fosd=guarded_spec.demo_fosd,
    )

    batch = _pixel_batch()
    official_metrics = _run_update(official, batch=dict(batch))
    guarded_metrics = _run_update(guarded, batch=dict(batch))
    assert guarded_metrics["critic_loss"] == pytest.approx(
        official_metrics["critic_loss"], abs=1e-6
    )
    for official_leaf, guarded_leaf in zip(
        jax.tree.leaves(official.params),
        jax.tree.leaves(guarded.params),
        strict=True,
    ):
        assert official_leaf.shape == guarded_leaf.shape
        np.testing.assert_allclose(
            np.asarray(guarded_leaf), np.asarray(official_leaf), atol=1e-6
        )


def test_flags_off_act_matches_pristine_cqn_as():
    official = _official_agent()
    guarded = _guarded_agent(
        "method.nonfinite_guard=false",
        "method.bc_diagnostics=false",
    )
    observation = _observation()
    np.testing.assert_allclose(
        np.asarray(guarded.act(observation, step=100, eval_mode=True)),
        np.asarray(official.act(observation, step=100, eval_mode=True)),
        atol=1e-6,
    )


# --------------------------------------------------------------------------
# 2. flags-on sanity
# --------------------------------------------------------------------------

NAN_DIAG_KEYS = (
    "nan_diag/update_committed",
    "nan_diag/features_all_finite",
    "nan_diag/next_features_all_finite",
    "nan_diag/target_logits_all_finite",
    "nan_diag/target_probabilities_all_finite",
    "nan_diag/target_distribution_all_finite",
    "nan_diag/chosen_logits_all_finite",
    "nan_diag/all_logits_all_finite",
    "nan_diag/chosen_log_probabilities_all_finite",
    "nan_diag/canonical_per_sample_all_finite",
    "nan_diag/bc_fosd_term_all_finite",
    "nan_diag/bc_margin_term_all_finite",
    "nan_diag/loss_all_finite",
    "nan_diag/features_max_abs_finite",
    "nan_diag/next_features_max_abs_finite",
    "nan_diag/target_logits_max_abs_finite",
    "nan_diag/chosen_logits_max_abs_finite",
    "nan_diag/pre_params_all_finite",
    "nan_diag/pre_target_all_finite",
    "nan_diag/pre_opt_state_all_finite",
    "nan_diag/grads_all_finite",
    "nan_diag/updates_all_finite",
    "nan_diag/candidate_opt_state_all_finite",
    "nan_diag/candidate_params_all_finite",
    "nan_diag/candidate_target_all_finite",
    "nan_diag/grads_max_abs_finite",
    "nan_diag/updates_max_abs_finite",
)

BC_DIAG_KEYS = (
    "bc_weight",
    "bc_agreement",
    "bc_binding_rate",
    "bc_margin_gap",
    "bc_sibling_q_span",
    "bc_online_agreement",
)


def test_flags_on_act_and_update_are_finite_and_emit_line_metrics():
    agent = _guarded_agent()
    action = agent.act(_observation(), step=100, eval_mode=False)
    assert np.all(np.isfinite(np.asarray(action)))
    assert np.asarray(action).shape == (BATCH, ACTION_SEQUENCE, ACTION_DIM)

    metrics = _run_update(agent)
    for key in NAN_DIAG_KEYS + BC_DIAG_KEYS:
        assert key in metrics, f"missing metric {key}"
    for key, value in metrics.items():
        assert np.isfinite(value), f"non-finite metric {key}={value}"
    assert metrics["nan_diag/update_committed"] == pytest.approx(1.0)


def test_flags_off_emits_no_guard_or_diagnostic_metrics():
    agent = _guarded_agent(
        "method.nonfinite_guard=false",
        "method.bc_diagnostics=false",
    )
    metrics = _run_update(agent)
    for key in NAN_DIAG_KEYS + BC_DIAG_KEYS:
        assert key not in metrics


def test_bc_diagnostics_are_emitted_even_at_bc_lambda_zero():
    """The research block is unconditional; lambda=0 must stay instrumented."""

    agent = _guarded_agent(
        "method.nonfinite_guard=false",
        "method.bc_lambda=0.0",
    )
    metrics = _run_update(agent)
    for key in BC_DIAG_KEYS:
        assert key in metrics
    assert metrics["bc_weight"] == pytest.approx(0.0)
    # Observational only: at lambda=0 the diagnostics must not have entered
    # the loss, so the update still equals the pristine one.
    official = _official_agent("method.bc_lambda=0.0")
    official_metrics = _run_update(official)
    assert metrics["critic_loss"] == pytest.approx(
        official_metrics["critic_loss"], abs=1e-6
    )


@pytest.mark.parametrize(
    "guard,diagnostics",
    [(True, False), (False, True)],
)
def test_the_two_flags_are_independent(guard, diagnostics):
    agent = _guarded_agent(
        f"method.nonfinite_guard={str(guard).lower()}",
        f"method.bc_diagnostics={str(diagnostics).lower()}",
    )
    metrics = _run_update(agent)
    assert all(np.isfinite(v) for v in metrics.values())
    assert all((k in metrics) is guard for k in NAN_DIAG_KEYS)
    assert all((k in metrics) is diagnostics for k in BC_DIAG_KEYS)
    if guard:
        # The guard still commits with only its own flags in the vote.
        assert metrics["nan_diag/update_committed"] == pytest.approx(1.0)
        poisoned = _guarded_agent(
            "method.nonfinite_guard=true",
            f"method.bc_diagnostics={str(diagnostics).lower()}",
        )
        bad = _run_update(poisoned, batch=_poisoned_batch())
        assert bad["nan_diag/update_committed"] == pytest.approx(0.0)


def test_bc_diagnostics_do_not_change_the_update():
    batch = _batch()
    off = _guarded_agent(
        "method.nonfinite_guard=false", "method.bc_diagnostics=false"
    )
    on = _guarded_agent(
        "method.nonfinite_guard=false", "method.bc_diagnostics=true"
    )
    off_metrics = _run_update(off, batch=batch)
    on_metrics = _run_update(on, batch=dict(batch))
    assert on_metrics["critic_loss"] == pytest.approx(
        off_metrics["critic_loss"], abs=1e-6
    )
    for off_leaf, on_leaf in zip(
        jax.tree.leaves(off.params), jax.tree.leaves(on.params), strict=True
    ):
        np.testing.assert_allclose(
            np.asarray(on_leaf), np.asarray(off_leaf), atol=1e-7
        )


def _poisoned_batch():
    """A batch that actually drives the update non-finite.

    NB a NaN *reward* does NOT: ``project_categorical`` (pristine
    ``robobase/method/cqn.py``) clips the atom targets and then floors them to
    integer indices, so NaN collapses to ``lower == upper`` with weight 1.0 and
    the projected distribution comes back finite. Poison the observation
    instead, which is the divergence route the guard was written for.
    """

    poisoned = _batch()
    poisoned["low_dim_state"] = np.full(
        (BATCH, 1, LOW_DIM), np.nan, dtype=np.float32
    )
    return poisoned


def test_nonfinite_guard_skips_a_bad_update_and_keeps_last_good_state():
    agent = _guarded_agent()
    good = jax.tree.map(np.asarray, agent.params)
    good_target = jax.tree.map(np.asarray, agent.target_critic_params)

    agent.logging = True
    metrics = agent.update(iter([_poisoned_batch()]), step=1)

    assert metrics["nan_diag/update_committed"] == pytest.approx(0.0)
    assert not np.isfinite(metrics["critic_loss"])
    assert metrics["nan_diag/features_all_finite"] == pytest.approx(0.0)
    for before, after in zip(
        jax.tree.leaves(good), jax.tree.leaves(agent.params), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(after), np.asarray(before))
    for before, after in zip(
        jax.tree.leaves(good_target),
        jax.tree.leaves(agent.target_critic_params),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(after), np.asarray(before))


def test_nonfinite_guard_surfaces_committed_metric_without_logging():
    """workspace._guard_non_finite_update must see the metric even then."""

    agent = _guarded_agent()
    agent.logging = False
    metrics = agent.update(iter([_poisoned_batch()]), step=1)
    assert metrics["nan_diag/update_committed"] == pytest.approx(0.0)

    agent = _guarded_agent()
    agent.logging = False
    assert agent.update(iter([_batch()]), step=1) == {}


def test_guard_off_lets_a_bad_update_through():
    """The guard is what stops the poison; without it the state is ruined."""

    agent = _guarded_agent("method.nonfinite_guard=false")
    agent.logging = True
    metrics = agent.update(iter([_poisoned_batch()]), step=1)
    assert "nan_diag/update_committed" not in metrics
    assert not np.isfinite(metrics["critic_loss"])
    assert any(
        not np.all(np.isfinite(np.asarray(leaf)))
        for leaf in jax.tree.leaves(agent.params)
    )


def test_nan_reward_is_swallowed_by_the_c51_projection_not_the_guard():
    """Documents a real blind spot (pristine code; reported, not fixed)."""

    agent = _guarded_agent()
    poisoned = _batch()
    poisoned["reward"] = np.full((BATCH,), np.nan, dtype=np.float32)
    metrics = _run_update(agent, batch=poisoned)
    assert np.isfinite(metrics["critic_loss"])
    assert metrics["nan_diag/update_committed"] == pytest.approx(1.0)
    assert metrics["nan_diag/target_distribution_all_finite"] == 1.0


def test_bc_lambda_schedule_decays_the_weight():
    agent = _guarded_agent("method.bc_lambda_schedule='linear(1.0,0.0,10)'")
    assert agent.bc_lambda_schedule == "linear(1.0,0.0,10)"
    early = _run_update(agent, step=1)
    late = _run_update(agent, step=100000)
    assert np.isfinite(early["critic_loss"])
    assert np.isfinite(late["critic_loss"])
    assert early["bc_weight"] > late["bc_weight"]
    assert late["bc_weight"] == pytest.approx(0.0)


def test_bc_lambda_schedule_null_keeps_the_constant_weight():
    agent = _guarded_agent()
    metrics = _run_update(agent, step=12345)
    assert metrics["bc_weight"] == pytest.approx(agent.bc_lambda)


def test_bc_lambda_schedule_at_the_constant_matches_no_schedule():
    batch = _batch()
    constant = _guarded_agent()
    scheduled = _guarded_agent(
        "method.bc_lambda_schedule='linear(1.0,1.0,10)'",
    )
    constant_metrics = _run_update(constant, batch=batch)
    scheduled_metrics = _run_update(scheduled, batch=dict(batch))
    assert scheduled_metrics["critic_loss"] == pytest.approx(
        constant_metrics["critic_loss"], abs=1e-6
    )


def test_demo_fosd_false_drops_the_fosd_term():
    batch = _batch()
    with_fosd = _guarded_agent()
    without_fosd = _guarded_agent("method.demo_fosd=false")

    # Step-1 losses are EQUAL by construction: the critic output heads are
    # zero-initialised, so every bin shares one distribution and the FOSD
    # hinge max(chosen_cdf - all_cdf, 0) is exactly 0. Asserting on the first
    # loss is the known false negative; assert on the parameter tree instead.
    with_metrics = _run_update(with_fosd, batch=batch)
    without_metrics = _run_update(without_fosd, batch=dict(batch))
    assert np.isfinite(without_metrics["critic_loss"])
    assert without_metrics["nan_diag/bc_fosd_term_all_finite"] == 1.0
    assert without_metrics["critic_loss"] == pytest.approx(
        with_metrics["critic_loss"], abs=1e-6
    )
    assert any(
        not np.allclose(np.asarray(a), np.asarray(b), atol=1e-9)
        for a, b in zip(
            jax.tree.leaves(with_fosd.params),
            jax.tree.leaves(without_fosd.params),
            strict=True,
        )
    )

    # Once the heads leave zero the FOSD penalty is a strictly positive
    # addition on demo rows, so dropping it lowers the loss.
    for step in (2, 3):
        with_metrics = _run_update(with_fosd, batch=dict(batch), step=step)
        without_metrics = _run_update(
            without_fosd, batch=dict(batch), step=step
        )
    assert without_metrics["critic_loss"] < with_metrics["critic_loss"]


# --------------------------------------------------------------------------
# 3. flags-on == research monolith
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        (),
        ("method.demo_fosd=false",),
        ("method.bc_lambda_schedule='linear(1.0,0.2,10)'",),
        ("method.bc_lambda=0.0",),
    ],
)
def test_flags_on_matches_research_monolith(overrides):
    batch = _batch()
    research = _research_agent(*overrides)
    guarded = _guarded_agent(*overrides)

    # Several sequential updates: one step from zero-initialised heads is a
    # degenerate comparison (FOSD and every sibling-based diagnostic are
    # identically zero there).
    for step in (1, 2, 3, 4):
        research_metrics = _run_update(research, batch=dict(batch), step=step)
        guarded_metrics = _run_update(guarded, batch=dict(batch), step=step)

        assert set(guarded_metrics) == set(research_metrics)
        for key in research_metrics:
            if key == "backend/update_time_sec":
                continue
            assert guarded_metrics[key] == pytest.approx(
                research_metrics[key], abs=1e-5
            ), f"step {step} key {key}"
        for research_leaf, guarded_leaf in zip(
            jax.tree.leaves(research.params),
            jax.tree.leaves(guarded.params),
            strict=True,
        ):
            np.testing.assert_allclose(
                np.asarray(guarded_leaf), np.asarray(research_leaf), atol=1e-6
            )

    # The comparison is not vacuous: by step 4 the sibling diagnostics have
    # left their zero-init values.
    assert guarded_metrics["bc_sibling_q_span"] > 0.0
