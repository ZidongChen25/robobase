"""R2 verification for the extracted token-split CQN-AS variant.

Covers the three contracts from ``R2_COMMON_BRIEF.md``:

1. flags-off ``CQNASTokenSplit`` is numerically identical to the pristine
   frozen ``CQNAS`` (same seed, same synthetic batch);
2. flags-on runs (``act`` + ``update``) with finite metrics and the two
   token-split wiring metrics present;
3. flags-on matches the research monolith ``cqn_as_research.CQNAS``.

Plus the behavioural assertions adapted from the research-era spec
``tests/unit/test_cqn_as_token_split.py`` (config round-trip, loud rejection
of unsupported compositions, legacy equality when the auxiliary transition
equals the primary one, and actual consumption of the auxiliary horizon).

CPU only::

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
        PYTHONPATH=/home/zc1525/robobase_jaxflat_refactor \
        /home/zc1525/robobase_jaxflat/.venv/bin/python -m pytest \
        tests/unit/test_cqn_as_token_split_variant.py -q
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

from robobase.method.cqn_as import CQNAS, cqn_as_spec_from_cfg  # noqa: E402
from robobase.method.cqn_as_token_split import (  # noqa: E402
    CQNASTokenSplit,
    cqn_as_token_split_spec_from_cfg,
)

CONFIG_DIR = str((Path(__file__).parents[2] / "robobase" / "cfgs").resolve())

ACTION_SEQUENCE = 4
ACTION_DIM = 2
LOW_DIM = 5
BATCH = 4
RGB_KEY = "rgb_head"
# BiGym-shaped pixel observation: [time=1, frame_stack * 3 = 12, 84, 84].
RGB_SHAPE = (1, 12, 84, 84)
# Canonical wave-2 value (cqn-rline.md arm "tokensplit"): tokens 1..2 keep the
# 1-step backup, the rest regress to the auxiliary horizon.
BOUNDARY = 2
AUX_NSTEP = 3


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
                "method.atoms=11",
                *overrides,
            ],
        )


def _split_overrides(*overrides: str):
    return (
        "method.token_split_horizon_targets=true",
        f"method.token_split_boundary={BOUNDARY}",
        "replay.nstep=1",
        f"replay.auxiliary_nstep={AUX_NSTEP}",
        *overrides,
    )


def _spaces(pixels: bool = False):
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


def _agent_kwargs(cfg, spec, observation_space, action_space):
    """The pristine CQN-AS construction kwargs used by ``factory.py``."""

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
        num_train_envs=int(cfg.num_train_envs),
        num_eval_envs=int(cfg.num_eval_envs),
        replay_alpha=cfg.replay.alpha,
        replay_beta=cfg.replay.beta,
        frame_stack_on_channel=cfg.frame_stack_on_channel,
        intrinsic_reward_module=None,
        jit=False,
        platform="cpu",
        seed=int(cfg.seed),
        update_block_every_steps=1,
    )


def _make_split(*overrides: str, pixels: bool = False):
    """yaml -> spec -> __init__ chain for the extracted variant."""

    cfg = _compose("cqn_as_token_split", *overrides)
    observation_space, action_space = _spaces(pixels)
    spec = cqn_as_token_split_spec_from_cfg(cfg)
    return CQNASTokenSplit(
        **_agent_kwargs(cfg, spec, observation_space, action_space),
        token_split_horizon_targets=spec.token_split_horizon_targets,
        token_split_boundary=spec.token_split_boundary,
    )


def _make_official(*overrides: str, pixels: bool = False):
    cfg = _compose("cqn_as_official", *overrides)
    observation_space, action_space = _spaces(pixels)
    spec = cqn_as_spec_from_cfg(cfg)
    return CQNAS(**_agent_kwargs(cfg, spec, observation_space, action_space))


def _make_research(*overrides: str, pixels: bool = False):
    from robobase.factory import create_agent

    cfg = _compose("cqn_as", *overrides)
    observation_space, action_space = _spaces(pixels)
    return create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )


def _observation():
    rng = np.random.default_rng(3)
    return {
        "low_dim_state": rng.normal(size=(1, 1, LOW_DIM)).astype(np.float32),
    }


def _batch(aux: bool = True, seed: int = 11, pixels: bool = False, size=BATCH):
    rng = np.random.default_rng(seed)
    batch = {
        "low_dim_state": rng.normal(size=(size, 1, LOW_DIM)).astype(np.float32),
        "low_dim_state_tp1": rng.normal(size=(size, 1, LOW_DIM)).astype(
            np.float32
        ),
        "action": rng.uniform(
            -1.0, 1.0, size=(size, ACTION_SEQUENCE, ACTION_DIM)
        ).astype(np.float32),
        "reward": rng.normal(size=(size,)).astype(np.float32),
        "discount": np.full((size,), 0.99, dtype=np.float32),
        "terminal": np.zeros((size,), dtype=bool),
        "truncated": np.zeros((size,), dtype=bool),
        "demo": np.ones((size,), dtype=np.uint8),
    }
    if pixels:
        for key in (RGB_KEY, f"{RGB_KEY}_tp1"):
            batch[key] = rng.integers(
                0, 256, size=(size, *RGB_SHAPE), dtype=np.uint8
            )
    if aux:
        batch["low_dim_state_tp_aux"] = rng.normal(
            size=(size, 1, LOW_DIM)
        ).astype(np.float32)
        batch["action_tp_aux"] = rng.uniform(
            -1.0, 1.0, size=(size, ACTION_SEQUENCE, ACTION_DIM)
        ).astype(np.float32)
        batch["reward_aux"] = rng.normal(size=(size,)).astype(np.float32)
        batch["discount_aux"] = np.full(
            (size,), 0.99**AUX_NSTEP, dtype=np.float32
        )
        batch["terminal_aux"] = np.zeros((size,), dtype=bool)
        batch["truncated_aux"] = np.zeros((size,), dtype=bool)
        if pixels:
            batch[f"{RGB_KEY}_tp_aux"] = rng.integers(
                0, 256, size=(size, *RGB_SHAPE), dtype=np.uint8
            )
    return batch


def _copy(batch):
    return {key: np.array(value, copy=True) for key, value in batch.items()}


def _params_equal(left, right, *, rtol=1e-5, atol=1e-7):
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    assert left_tree == right_tree
    return all(
        np.allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol)
        for a, b in zip(left_leaves, right_leaves, strict=True)
    )


def _shapes(tree):
    return jax.tree.map(lambda leaf: tuple(np.asarray(leaf).shape), tree)


# --------------------------------------------------------------------------
# 1. flags-off == pristine
# --------------------------------------------------------------------------


def test_flags_off_matches_pristine_official():
    official = _make_official()
    split = _make_split()
    assert split.token_split_horizon_targets is False
    assert split.token_split_boundary is None

    official.logging = True
    split.logging = True
    # The zero-initialised heads make the FIRST loss target-independent, so
    # several updates are needed before critic_loss is a real equivalence
    # statement; parameter equality is checked at every step regardless.
    for index in range(3):
        batch = _batch(aux=False, seed=11 + index)
        official_metrics = official.update(
            iter([_copy(batch)]), step=index + 1
        )
        split_metrics = split.update(iter([_copy(batch)]), step=index + 1)
        assert split_metrics["critic_loss"] == pytest.approx(
            official_metrics["critic_loss"], abs=1e-6
        ), index
        assert _shapes(split.params) == _shapes(official.params)
        assert _params_equal(split.params, official.params), index
        assert _params_equal(
            split.target_critic_params, official.target_critic_params
        ), index
        assert "token_split_aux_fraction" not in split_metrics
        assert "token_split_aux_reward_mean" not in split_metrics
    assert set(split_metrics) == set(official_metrics)


# --------------------------------------------------------------------------
# 2. flags-on sanity
# --------------------------------------------------------------------------


def test_flags_on_act_and_update_are_finite():
    agent = _make_split(*_split_overrides())
    assert agent.token_split_horizon_targets is True
    assert agent.token_split_boundary == BOUNDARY

    for eval_mode in (True, False):
        action = np.asarray(agent.act(_observation(), step=100, eval_mode=eval_mode))
        assert action.shape == (1, ACTION_SEQUENCE, ACTION_DIM)
        assert np.all(np.isfinite(action))

    agent.logging = True
    metrics = agent.update(iter([_batch()]), step=1)
    assert "token_split_aux_fraction" in metrics
    assert "token_split_aux_reward_mean" in metrics
    assert metrics["token_split_aux_fraction"] == pytest.approx(
        (ACTION_SEQUENCE - BOUNDARY) / ACTION_SEQUENCE
    )
    for key, value in metrics.items():
        assert np.all(np.isfinite(np.asarray(value, dtype=np.float64))), key


# --------------------------------------------------------------------------
# 3. flags-on == research monolith
# --------------------------------------------------------------------------


def test_flags_on_matches_research_monolith():
    split = _make_split(*_split_overrides())
    research = _make_research(*_split_overrides())
    assert type(research).__module__ == "robobase.method.cqn_as_research"

    split.logging = True
    research.logging = True
    for index in range(3):
        batch = _batch(seed=11 + index)
        split_metrics = split.update(iter([_copy(batch)]), step=index + 1)
        research_metrics = research.update(
            iter([_copy(batch)]), step=index + 1
        )
        assert split_metrics["critic_loss"] == pytest.approx(
            research_metrics["critic_loss"], abs=1e-5
        ), index
        assert split_metrics["token_split_aux_fraction"] == pytest.approx(
            research_metrics["token_split_aux_fraction"]
        ), index
        assert split_metrics["token_split_aux_reward_mean"] == pytest.approx(
            research_metrics["token_split_aux_reward_mean"], abs=1e-6
        ), index
        assert _params_equal(
            split.params, research.params, rtol=1e-5, atol=1e-7
        ), index


def test_flags_on_matches_research_monolith_with_pixels():
    """The pixel path is the only one that exercises the auxiliary-horizon
    RandomShift augmentation (``fold_in(action_key, 4243)``) and the dict
    observation branch."""

    overrides = _split_overrides("pixels=true")
    split = _make_split(*overrides, pixels=True)
    research = _make_research(*overrides, pixels=True)

    split.logging = True
    research.logging = True
    batch = _batch(pixels=True, size=2)
    split_metrics = split.update(iter([_copy(batch)]), step=1)
    research_metrics = research.update(iter([_copy(batch)]), step=1)

    assert split_metrics["critic_loss"] == pytest.approx(
        research_metrics["critic_loss"], abs=1e-5
    )
    assert _params_equal(split.params, research.params, rtol=1e-5, atol=1e-7)


# --------------------------------------------------------------------------
# Behavioural spec, adapted from tests/unit/test_cqn_as_token_split.py
# --------------------------------------------------------------------------


def test_token_split_requires_auxiliary_nstep():
    with pytest.raises(ValueError, match="auxiliary_nstep"):
        _make_split(
            "method.token_split_horizon_targets=true",
            f"method.token_split_boundary={BOUNDARY}",
        )


def test_token_split_requires_explicit_boundary():
    with pytest.raises(ValueError, match="token_split_boundary"):
        _make_split(
            *_split_overrides("method.token_split_boundary=null"),
        )


@pytest.mark.parametrize("boundary", [0, ACTION_SEQUENCE])
def test_token_split_boundary_range_is_validated(boundary):
    with pytest.raises(ValueError, match="token_split_boundary"):
        _make_split(
            *_split_overrides(f"method.token_split_boundary={boundary}"),
        )


def test_token_split_missing_aux_batch_fields_raise():
    agent = _make_split(*_split_overrides())
    with pytest.raises(KeyError, match="auxiliary-horizon"):
        agent.update(iter([_batch(aux=False)]), step=1)


def test_token_split_missing_aux_observation_raises():
    agent = _make_split(*_split_overrides())
    batch = _batch()
    del batch["low_dim_state_tp_aux"]
    with pytest.raises(KeyError, match="auxiliary TD replay batch is missing"):
        agent.update(iter([batch]), step=1)


def test_token_split_matches_legacy_when_aux_equals_primary():
    """Aux transition == primary transition -> the split target is the legacy
    target for every token, so the loss and the post-update parameters must
    match the flag-off run on the same batch."""

    batch = _batch()
    for name, source in (
        ("low_dim_state_tp_aux", "low_dim_state_tp1"),
        ("reward_aux", "reward"),
        ("discount_aux", "discount"),
        ("terminal_aux", "terminal"),
        ("truncated_aux", "truncated"),
    ):
        batch[name] = np.array(batch[source], copy=True)
    batch["action_tp_aux"] = np.array(batch["action"], copy=True)

    legacy = _make_split()
    split = _make_split(*_split_overrides())
    legacy.logging = True
    split.logging = True
    legacy_metrics = legacy.update(iter([_copy(batch)]), step=1)
    split_metrics = split.update(iter([_copy(batch)]), step=1)

    assert split_metrics["token_split_aux_fraction"] == pytest.approx(
        (ACTION_SEQUENCE - BOUNDARY) / ACTION_SEQUENCE
    )
    assert split_metrics["critic_loss"] == pytest.approx(
        legacy_metrics["critic_loss"], rel=1e-5
    )
    # The heads start zero-initialised, so the first-step loss is
    # target-independent; the gradients are not. Parameter equality after one
    # update is the real equivalence statement.
    assert _params_equal(legacy.params, split.params)


def test_token_split_consumes_aux_horizon():
    """Changing only the auxiliary reward must change the update (the aux
    horizon is really wired into the target) and be reported by the metric."""

    base_batch = _batch()
    shifted_batch = _copy(base_batch)
    shifted_batch["reward_aux"] = base_batch["reward_aux"] + 1.0

    agent_a = _make_split(*_split_overrides())
    agent_b = _make_split(*_split_overrides())
    agent_a.logging = True
    agent_b.logging = True
    metrics_a = agent_a.update(iter([base_batch]), step=1)
    metrics_b = agent_b.update(iter([shifted_batch]), step=1)

    assert metrics_b["token_split_aux_reward_mean"] == pytest.approx(
        metrics_a["token_split_aux_reward_mean"] + 1.0, abs=1e-5
    )
    assert not _params_equal(agent_a.params, agent_b.params)


def test_token_split_boundary_moves_the_aux_fraction():
    """The mask is per token, not per flat action dimension."""

    fractions = {}
    for boundary in (1, ACTION_SEQUENCE - 1):
        agent = _make_split(
            *_split_overrides(f"method.token_split_boundary={boundary}")
        )
        agent.logging = True
        fractions[boundary] = agent.update(iter([_batch()]), step=1)[
            "token_split_aux_fraction"
        ]
    assert fractions[1] == pytest.approx(
        (ACTION_SEQUENCE - 1) / ACTION_SEQUENCE
    )
    assert fractions[ACTION_SEQUENCE - 1] == pytest.approx(
        1.0 / ACTION_SEQUENCE
    )
