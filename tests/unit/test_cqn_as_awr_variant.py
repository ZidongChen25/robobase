"""CPU-only checks for the extracted AWR line (`robobase.method.cqn_as_awr`).

Verification protocol from ``R2_COMMON_BRIEF.md``:

1. flags-off is numerically the pristine :class:`robobase.method.cqn_as.CQNAS`;
2. flags-on runs and reports finite, correctly named metrics;
3. flags-on cannot be compared against ``cqn_as_research.CQNAS`` -- the research
   AWR block is only reachable with ``separate_bc_policy=true`` (a different
   update graph and parameter tree owned by the ``bc-policy`` line), so the
   mechanism is instead pinned against a NumPy transcription of the research
   formulas (``cqn_as_research.py`` lines 5287-5323).

Run with::

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
        PYTHONPATH=/home/zc1525/robobase_jaxflat_refactor \
        /home/zc1525/robobase_jaxflat/.venv/bin/python -m pytest \
        tests/unit/test_cqn_as_awr_variant.py -q
"""

from dataclasses import fields
from pathlib import Path

import jax
import numpy as np
import pytest
from gymnasium import spaces
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from robobase.method.cqn_as import CQNAS, CQNASpec, cqn_as_spec_from_cfg
from robobase.method.cqn_as_awr import (
    CQNASAwr,
    CQNASAwrSpec,
    cqn_as_awr_spec_from_cfg,
)

CONFIG_DIR = str((Path(__file__).parents[2] / "robobase" / "cfgs").resolve())

ACTION_SEQUENCE = 3
ACTION_DIM = 2
LOW_DIM = 5
BATCH = 4
SEED = 0


def _compose(method="cqn_as_official", *overrides):
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
                "method.atoms=11",
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


def _batch(with_mc_return=False):
    rng = np.random.default_rng(7)
    batch = {
        "low_dim_state": rng.normal(size=(BATCH, 1, LOW_DIM)).astype(np.float32),
        "low_dim_state_tp1": rng.normal(size=(BATCH, 1, LOW_DIM)).astype(np.float32),
        "action": rng.uniform(
            -1.0, 1.0, size=(BATCH, ACTION_SEQUENCE, ACTION_DIM)
        ).astype(np.float32),
        "reward": rng.normal(size=(BATCH,)).astype(np.float32),
        "discount": np.full((BATCH,), 0.99, dtype=np.float32),
        "terminal": np.zeros((BATCH,), dtype=bool),
        "truncated": np.zeros((BATCH,), dtype=bool),
        "demo": np.array([1, 1, 0, 0], dtype=np.uint8),
    }
    if with_mc_return:
        batch["mc_return"] = np.array([1.0, 0.8, 0.0, 0.6], dtype=np.float32)
    return batch


def _observation():
    rng = np.random.default_rng(3)
    return {
        "low_dim_state": rng.normal(size=(BATCH, 1, LOW_DIM)).astype(np.float32),
    }


def _base_kwargs(cfg, observation_space, action_space):
    """Pristine CQN-AS construction kwargs, mirroring ``factory.create_agent``."""

    spec = cqn_as_spec_from_cfg(cfg)
    kwargs = {field.name: getattr(spec, field.name) for field in fields(CQNASpec)}
    kwargs.update(
        observation_space=observation_space,
        action_space=action_space,
        num_train_envs=int(cfg.num_train_envs),
        num_eval_envs=int(cfg.num_eval_envs),
        replay_alpha=cfg.replay.alpha,
        replay_beta=cfg.replay.beta,
        frame_stack_on_channel=cfg.frame_stack_on_channel,
        jit=False,
        platform="cpu",
        seed=SEED,
    )
    return kwargs


def _shapes(tree):
    return jax.tree.map(lambda leaf: tuple(np.shape(leaf)), tree)


# ---------------------------------------------------------------- verification 1


def test_flags_off_matches_pristine_cqn_as():
    """Same seed + same batch -> identical critic_loss and param tree."""

    cfg = _compose()
    observation_space, action_space = _spaces()
    kwargs = _base_kwargs(cfg, observation_space, action_space)

    pristine = CQNAS(**kwargs)
    variant = CQNASAwr(**kwargs)

    assert variant.awr_beta is None
    assert "expectile_value" not in variant.params
    assert _shapes(pristine.params) == _shapes(variant.params)

    pristine.logging = True
    variant.logging = True
    pristine_metrics = pristine.update(iter([_batch()]), step=1)
    variant_metrics = variant.update(iter([_batch()]), step=1)

    assert variant_metrics["critic_loss"] == pytest.approx(
        pristine_metrics["critic_loss"], abs=1e-6
    )
    for key in ("entropy", "target_entropy", "loss_coeff"):
        assert variant_metrics[key] == pytest.approx(
            pristine_metrics[key], abs=1e-6
        )
    assert _shapes(pristine.params) == _shapes(variant.params)
    # Post-update parameters must coincide, not just the reported loss.
    pristine_leaves = jax.tree.leaves(pristine.params)
    variant_leaves = jax.tree.leaves(variant.params)
    assert len(pristine_leaves) == len(variant_leaves)
    for left, right in zip(pristine_leaves, variant_leaves, strict=True):
        np.testing.assert_allclose(
            np.asarray(left), np.asarray(right), atol=1e-6, rtol=0.0
        )
    # The auxiliary metrics exist and are exactly zero on the legacy path.
    for key in (
        "awr_value_loss",
        "awr_value_mean",
        "awr_weight_mean",
        "awr_weight_ess",
    ):
        assert variant_metrics[key] == pytest.approx(0.0)


def test_flags_off_matches_pristine_action():
    cfg = _compose()
    observation_space, action_space = _spaces()
    kwargs = _base_kwargs(cfg, observation_space, action_space)

    pristine = CQNAS(**kwargs)
    variant = CQNASAwr(**kwargs)
    observation = _observation()
    for eval_mode in (True, False):
        left = np.asarray(pristine.act(observation, step=10, eval_mode=eval_mode))
        right = np.asarray(variant.act(observation, step=10, eval_mode=eval_mode))
        np.testing.assert_allclose(left, right, atol=1e-6, rtol=0.0)


# ---------------------------------------------------------------- verification 2


def _awr_agent(**awr_flags):
    cfg = _compose()
    observation_space, action_space = _spaces()
    kwargs = _base_kwargs(cfg, observation_space, action_space)
    flags = dict(awr_beta=0.5, awr_weight_max=10.0, awr_expectile_tau=0.7)
    flags.update(awr_flags)
    return CQNASAwr(**kwargs, **flags)


def test_flags_on_adds_expectile_head_and_trains():
    agent = _awr_agent()
    assert "expectile_value" in agent.params

    value_before = jax.tree.map(np.asarray, agent.params["expectile_value"])
    critic_before = jax.tree.map(np.asarray, agent.params["critic"])

    action = np.asarray(agent.act(_observation(), step=10, eval_mode=True))
    assert action.shape == (BATCH, ACTION_SEQUENCE, ACTION_DIM)
    assert np.all(np.isfinite(action))

    agent.logging = True
    agent.update(iter([_batch(with_mc_return=True)]), step=1)
    metrics = agent.update(iter([_batch(with_mc_return=True)]), step=2)

    for key, value in metrics.items():
        assert np.isfinite(value), f"{key} is not finite: {value}"
    for key in (
        "awr_value_loss",
        "awr_value_mean",
        "awr_weight_mean",
        "awr_weight_ess",
    ):
        assert key in metrics
    assert metrics["awr_value_loss"] > 0.0
    assert metrics["awr_weight_mean"] > 0.0
    assert 0.0 < metrics["awr_weight_ess"] <= 1.0

    def _changed(before, after):
        return any(
            not np.allclose(np.asarray(left), np.asarray(right))
            for left, right in zip(
                jax.tree.leaves(before), jax.tree.leaves(after), strict=True
            )
        )

    assert _changed(value_before, agent.params["expectile_value"])
    assert _changed(critic_before, agent.params["critic"])


def _bc_terms(agent, batch):
    """NumPy transcription of the pristine FOSD + margin per-sample terms.

    ``robobase/method/cqn.py`` lines 582-605.
    """

    obs_inputs = agent._prepare_rl_obs_inputs(batch)
    features = agent._rl_features(agent.params.get("encoder", None), obs_inputs)
    actions = np.asarray(batch["action"], dtype=np.float32).reshape(
        (batch["action"].shape[0], -1)
    )
    chosen_logits, all_logits = agent._critic_logits_per_level(
        agent.params["critic"], features, actions
    )
    chosen_logits = np.asarray(chosen_logits, dtype=np.float64)
    all_logits = np.asarray(all_logits, dtype=np.float64)

    def _softmax(x):
        shifted = x - x.max(axis=-1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / exponentials.sum(axis=-1, keepdims=True)

    chosen_probabilities = _softmax(chosen_logits)
    all_probabilities = _softmax(all_logits)
    chosen_cdf = np.cumsum(chosen_probabilities, axis=-1)
    all_cdf = np.cumsum(all_probabilities, axis=-1)
    fosd = (
        np.maximum(chosen_cdf[..., None, :] - all_cdf, 0.0)
        .sum(axis=-1)
        .mean(axis=(1, 2, 3))
    )
    support = np.asarray(agent.support, dtype=np.float64)
    all_q = np.sum(all_probabilities * support, axis=-1)
    chosen_q = np.sum(chosen_probabilities * support, axis=-1)
    margin = np.maximum(
        agent.bc_margin - (chosen_q[..., None] - all_q), 0.0
    ).mean(axis=(1, 2, 3))
    return fosd, margin


def test_flags_on_replaces_the_demo_mask_in_the_bc_terms():
    """The exact substitution: demo mean -> self-normalised AWR-weighted mean."""

    beta, weight_max = 0.5, 10.0
    on = _awr_agent(awr_beta=beta, awr_weight_max=weight_max)
    off = _awr_agent(awr_beta=None)
    batch = _batch(with_mc_return=True)

    # The critic and the value head are zero-initialised, which makes every
    # per-sample BC term constant across the batch (any weighting then gives
    # the same mean). Warm them up so the substitution is observable.
    on.logging = True
    for step in range(1, 6):
        on.update(iter([batch]), step=step)

    # Share the warmed critic/encoder params and the RNG stream so the two
    # graphs differ ONLY in how the BC per-sample terms are averaged.
    off.params = {
        key: value for key, value in on.params.items() if key != "expectile_value"
    }
    off.target_critic_params = on.target_critic_params
    off.opt_state = off.optimizer.init(off.params)
    off.rng_key = on.rng_key
    off.logging = True

    fosd, margin = _bc_terms(on, batch)
    assert fosd.std() > 0.0 or margin.std() > 0.0

    obs_inputs = on._prepare_rl_obs_inputs(batch)
    features = on._rl_features(on.params.get("encoder", None), obs_inputs)
    state_value = np.asarray(
        on.expectile_value_model.apply(on.params["expectile_value"], features),
        dtype=np.float64,
    )
    value_error = batch["mc_return"].astype(np.float64) - state_value
    weights = np.clip(np.exp(value_error / beta), 0.0, weight_max)
    weight_sum = max(float(np.sum(weights)), 1e-6)
    demos = batch["demo"].astype(np.float64)
    demo_count = max(float(np.sum(demos)), 1.0)

    expected_delta = on.bc_lambda * (
        (np.sum(fosd * weights) + np.sum(margin * weights)) / weight_sum
        - (np.sum(fosd * demos) + np.sum(margin * demos)) / demo_count
    )

    on_metrics = on.update(iter([batch]), step=6)
    off_metrics = off.update(iter([batch]), step=6)
    observed_delta = on_metrics["critic_loss"] - off_metrics["critic_loss"]

    assert abs(expected_delta) > 1e-6
    assert observed_delta == pytest.approx(expected_delta, abs=1e-5)


def test_awr_beta_requires_bc_lambda():
    cfg = _compose("cqn_as_official", "method.bc_lambda=0.0")
    observation_space, action_space = _spaces()
    kwargs = _base_kwargs(cfg, observation_space, action_space)
    with pytest.raises(ValueError, match="bc_lambda"):
        CQNASAwr(**kwargs, awr_beta=0.5)


@pytest.mark.parametrize(
    "flags,message",
    [
        (dict(awr_beta=0.0), "awr_beta must be positive"),
        (dict(awr_beta=0.5, awr_weight_max=0.0), "awr_weight_max must be positive"),
        (dict(awr_beta=0.5, awr_expectile_tau=1.0), "awr_expectile_tau"),
    ],
)
def test_flag_validation(flags, message):
    cfg = _compose()
    observation_space, action_space = _spaces()
    kwargs = _base_kwargs(cfg, observation_space, action_space)
    with pytest.raises(ValueError, match=message):
        CQNASAwr(**kwargs, **flags)


# ---------------------------------------------------------------- verification 3


def test_awr_quantities_match_research_formulas():
    """NumPy transcription of ``cqn_as_research.py`` lines 5287-5323.

    The research AWR block itself is unreachable without
    ``separate_bc_policy=true``; this pins the arithmetic instead.
    """

    beta, weight_max, tau = 0.5, 10.0, 0.7
    agent = _awr_agent(
        awr_beta=beta, awr_weight_max=weight_max, awr_expectile_tau=tau
    )
    batch = _batch(with_mc_return=True)

    features = agent._rl_features(
        agent.params.get("encoder", None),
        agent._prepare_rl_obs_inputs(batch),
    )
    state_value = np.asarray(
        agent.expectile_value_model.apply(
            agent.params["expectile_value"], features
        )
    )
    mc_returns = batch["mc_return"].astype(np.float32)

    value_error = mc_returns - state_value
    expectile_weight = np.where(value_error < 0.0, 1.0 - tau, tau)
    expected_value_loss = float(np.mean(expectile_weight * np.square(value_error)))
    expected_value_mean = float(np.mean(state_value))
    weights = np.clip(np.exp(value_error / beta), 0.0, weight_max)
    weight_sum = max(float(np.sum(weights)), 1e-6)
    expected_weight_mean = float(np.mean(weights))
    expected_ess = float(
        weight_sum**2 / (max(float(np.sum(np.square(weights))), 1e-6) * weights.size)
    )

    agent.logging = True
    metrics = agent.update(iter([batch]), step=1)

    assert metrics["awr_value_loss"] == pytest.approx(expected_value_loss, abs=1e-6)
    assert metrics["awr_value_mean"] == pytest.approx(expected_value_mean, abs=1e-6)
    assert metrics["awr_weight_mean"] == pytest.approx(
        expected_weight_mean, abs=1e-6
    )
    assert metrics["awr_weight_ess"] == pytest.approx(expected_ess, abs=1e-6)


def test_missing_mc_return_regresses_towards_zero():
    """Research-era behaviour: no ``mc_return`` in replay -> zeros target.

    ``robobase.workspace._mc_return_anchor_enabled`` does not list ``awr_beta``
    among its conditions, so an AWR-only launch config silently trains the
    expectile head against zeros. Reproduced, not fixed (see the report).
    """

    beta, tau = 0.5, 0.7
    agent = _awr_agent(awr_beta=beta, awr_expectile_tau=tau)
    batch = _batch(with_mc_return=False)
    assert "mc_return" not in batch

    features = agent._rl_features(
        agent.params.get("encoder", None),
        agent._prepare_rl_obs_inputs(batch),
    )
    state_value = np.asarray(
        agent.expectile_value_model.apply(
            agent.params["expectile_value"], features
        )
    )
    value_error = -state_value
    expectile_weight = np.where(value_error < 0.0, 1.0 - tau, tau)
    expected_value_loss = float(np.mean(expectile_weight * np.square(value_error)))

    agent.logging = True
    metrics = agent.update(iter([batch]), step=1)
    assert metrics["awr_value_loss"] == pytest.approx(expected_value_loss, abs=1e-6)


RESEARCH_AWR_OVERRIDES = (
    # The canonical Stage-145 platform: the research AWR block is unreachable
    # without all of this (tests/unit/test_cqn_as.py::_awr_cfg).
    "method.separate_bc_policy=true",
    "method.bc_policy_stop_gradient=true",
    "method.td_target_action_source=replay_next",
    "method.critic_sequence_mode=effective_k0",
    "method.critic_lambda=0.0",
    "method.mc_return_weight=0.1",
    "method.weight_decay=0.0",
    "method.awr_beta=0.5",
    "method.awr_weight_max=10.0",
    "method.awr_expectile_tau=0.7",
)


def test_research_refuses_awr_without_the_bc_policy_line():
    """Why flags-on critic_loss cannot be compared against the monolith."""

    from robobase.factory import create_agent

    observation_space, action_space = _spaces()
    with pytest.raises(ValueError, match="separate_bc_policy"):
        create_agent(
            _compose("cqn_as", "method.awr_beta=0.5"),
            observation_space=observation_space,
            action_space=action_space,
        )


def test_awr_quantities_are_bit_identical_to_the_research_monolith():
    """Flags-on ≡ research, for the part that is extractable.

    ``critic_loss`` is not comparable: the research AWR block only exists on
    the ``separate_bc_policy`` update graph, whose parameter tree carries an
    extra ``policy`` head and whose losses are composed differently. What IS
    comparable is the AWR machinery itself -- transplant a *warmed* research
    expectile head into the extracted variant and every one of the four AWR
    quantities must agree exactly.
    """

    from robobase.factory import create_agent

    observation_space, action_space = _spaces()
    research = create_agent(
        _compose("cqn_as", *RESEARCH_AWR_OVERRIDES),
        observation_space=observation_space,
        action_space=action_space,
    )
    research.logging = True
    batch = _batch(with_mc_return=True)
    for step in range(1, 6):
        research.update(iter([batch]), step=step)

    mine = _awr_agent(awr_beta=0.5, awr_weight_max=10.0, awr_expectile_tau=0.7)
    mine.logging = True
    mine.params = dict(mine.params)
    mine.params["expectile_value"] = research.params["expectile_value"]
    mine.opt_state = mine.optimizer.init(mine.params)

    research_metrics = research.update(iter([batch]), step=6)
    mine_metrics = mine.update(iter([batch]), step=6)

    # A warmed head, so this is not the trivial zero-initialised comparison.
    assert abs(research_metrics["awr_value_mean"]) > 1e-8
    for key in (
        "awr_value_loss",
        "awr_value_mean",
        "awr_weight_mean",
        "awr_weight_ess",
    ):
        assert mine_metrics[key] == research_metrics[key], key


# ---------------------------------------------------------------- config wiring


def test_yaml_defaults_are_off_and_target_the_variant():
    cfg = _compose("cqn_as_awr")
    assert cfg.method.name == "cqn_as_awr"
    assert cfg.method._target_ == "robobase.method.cqn_as_awr.CQNASAwr"
    assert cfg.method.awr_beta is None
    assert cfg.method.awr_weight_max == pytest.approx(10.0)
    assert cfg.method.awr_expectile_tau == pytest.approx(0.7)
    # Everything else is the pristine official method config.
    official = _compose("cqn_as_official")
    for key in official.method:
        if key in {"name", "_target_"}:
            continue
        assert cfg.method[key] == official.method[key], key


def test_spec_builder_reads_the_flags():
    cfg = _compose(
        "cqn_as_awr",
        "method.awr_beta=0.5",
        "method.awr_weight_max=20.0",
        "method.awr_expectile_tau=0.9",
    )
    spec = cqn_as_awr_spec_from_cfg(cfg)
    assert isinstance(spec, CQNASAwrSpec)
    assert isinstance(spec, CQNASpec)
    assert spec.awr_beta == pytest.approx(0.5)
    assert spec.awr_weight_max == pytest.approx(20.0)
    assert spec.awr_expectile_tau == pytest.approx(0.9)
    # Base fields still come from the pristine builder.
    base = cqn_as_spec_from_cfg(cfg)
    for field in fields(CQNASpec):
        assert getattr(spec, field.name) == getattr(base, field.name), field.name

    default_spec = cqn_as_awr_spec_from_cfg(_compose("cqn_as_awr"))
    assert default_spec.awr_beta is None
