"""R2 verification for the extracted CQN-AS twin-critic line.

Covers the four flags owned by ``robobase/method/cqn_as_twin_critic.py``:
``pessimistic_twin_critic``, ``auxiliary_td_loss_weight``,
``episodic_twin_head_exploration`` and ``twin_rollout_beam_width``.

1. flags-off == pristine ``robobase.method.cqn_as.CQNAS`` (critic_loss and
   parameter tree after one ``update()``);
2. flags-on runs: ``act()`` + ``update()`` finite with the line's metric keys;
3. flags-on == ``robobase.method.cqn_as_research.CQNAS`` with the same flags;
4. the behavioural gates adapted from ``test_cqn_as_beam_single.py``.
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
from robobase.method.cqn_as_twin_critic import (  # noqa: E402
    CQNASTwinCritic,
    cqn_as_twin_critic_spec_from_cfg,
)

CONFIG_DIR = str((Path(__file__).parents[2] / "robobase" / "cfgs").resolve())

ACTION_SEQUENCE = 3
ACTION_DIM = 2
LOW_DIM = 5

# Hyperparameters shared by the pristine, extracted and research composes so
# every comparison differs only in the twin-critic flags.
_BASE_OVERRIDES = (
    f"action_sequence={ACTION_SEQUENCE}",
    "num_train_envs=1",
    "num_eval_envs=1",
    "num_explore_steps=0",
    "backend.jit=false",
    "backend.platform=cpu",
    "method.model.hidden_dims=[32,32]",
    "method.atoms=11",
)

# The isolated reward-only direct-C51 platform the twin path was run on.
_TWIN_PLATFORM_OVERRIDES = (
    "num_pretrain_steps=0",
    "is_imitation_learning=false",
    "use_self_imitation=false",
    "method.use_dueling=false",
    "method.bc_lambda=0",
    "method.bc_margin=0",
    "method.weight_decay=0.0",
    "replay.include_next_action=true",
)

# The foreign-line flags the research monolith *requires* to be set before it
# will build the same twin update graph (see the coupling notes in the report).
_RESEARCH_TWIN_PLATFORM_OVERRIDES = (
    "method.strict_demo_rl_only=true",
    "method.bc_lambda_schedule=null",
    "method.demo_fosd=false",
    "method.separate_bc_policy=false",
    "method.mc_return_value_only=false",
    "method.mc_lower_bound_target=true",
    "method.td_target_action_source=critic_replay_max",
    "method.unseen_return_floor_weight=0.0",
)


def _compose(method: str, *overrides: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(
            config_name="robobase_config",
            overrides=[f"method={method}", *_BASE_OVERRIDES, *overrides],
        )


def _twin_cfg(*overrides: str):
    return _compose(
        "cqn_as_twin_critic",
        *_TWIN_PLATFORM_OVERRIDES,
        "method.pessimistic_twin_critic=true",
        *overrides,
    )


def _research_twin_cfg(*overrides: str):
    return _compose(
        "cqn_as",
        *_TWIN_PLATFORM_OVERRIDES,
        *_RESEARCH_TWIN_PLATFORM_OVERRIDES,
        "method.pessimistic_twin_critic=true",
        *overrides,
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


def _common_kwargs(cfg):
    observation_space, action_space = _spaces()
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
        jit=bool(cfg.backend.jit),
        platform="cpu",
        seed=int(cfg.seed),
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


def _make_pristine(cfg):
    spec = cqn_as_spec_from_cfg(cfg)
    return CQNAS(**_spec_kwargs(spec), **_common_kwargs(cfg))


def _make_variant(cfg):
    """Mirror the factory construction this line's registration snippet adds."""

    spec = cqn_as_twin_critic_spec_from_cfg(cfg)
    return CQNASTwinCritic(
        **_spec_kwargs(spec),
        pessimistic_twin_critic=spec.pessimistic_twin_critic,
        auxiliary_td_loss_weight=spec.auxiliary_td_loss_weight,
        episodic_twin_head_exploration=spec.episodic_twin_head_exploration,
        twin_rollout_beam_width=spec.twin_rollout_beam_width,
        **_common_kwargs(cfg),
    )


def _make_research(cfg):
    from robobase.factory import create_agent

    observation_space, action_space = _spaces()
    return create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )


def _batch(batch_size=8, seed=11, *, auxiliary=False):
    rng = np.random.default_rng(seed)
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
        "action_tp1": rng.uniform(
            -1.0, 1.0, size=(batch_size, ACTION_SEQUENCE, ACTION_DIM)
        ).astype(np.float32),
        "reward": rng.normal(size=(batch_size,)).astype(np.float32),
        "discount": np.full((batch_size,), 0.99, dtype=np.float32),
        "terminal": np.zeros((batch_size,), dtype=bool),
        "truncated": np.zeros((batch_size,), dtype=bool),
        "demo": np.ones((batch_size,), dtype=np.uint8),
        "mc_return": rng.uniform(-1.0, 1.0, size=(batch_size,)).astype(
            np.float32
        ),
    }
    if auxiliary:
        batch["low_dim_state_tp_aux"] = rng.normal(
            size=(batch_size, 1, LOW_DIM)
        ).astype(np.float32)
        batch["action_tp_aux"] = rng.uniform(
            -1.0, 1.0, size=(batch_size, ACTION_SEQUENCE, ACTION_DIM)
        ).astype(np.float32)
        batch["reward_aux"] = rng.normal(size=(batch_size,)).astype(np.float32)
        batch["discount_aux"] = np.full(
            (batch_size,), 0.99**4, dtype=np.float32
        )
        batch["terminal_aux"] = np.zeros((batch_size,), dtype=bool)
    return batch


def _copy(batch):
    return {key: np.array(value, copy=True) for key, value in batch.items()}


def _run_update(agent, batch, step=1):
    agent.logging = True
    return agent.update(iter([_copy(batch)]), step=step)


def _tree_shapes(tree):
    leaves, structure = jax.tree_util.tree_flatten(tree)
    return structure, tuple(np.shape(leaf) for leaf in leaves)


def _params_equal(left, right):
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    assert left_tree == right_tree
    return all(
        np.allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-7)
        for a, b in zip(left_leaves, right_leaves, strict=True)
    )


# ----------------------------------------------------------------------
# 1. flags-off == pristine
# ----------------------------------------------------------------------


def test_flags_off_matches_pristine_update():
    cfg = _compose("cqn_as_twin_critic")
    pristine = _make_pristine(cfg)
    variant = _make_variant(cfg)

    assert variant.pessimistic_twin_critic is False
    assert variant.auxiliary_td_loss_weight == 0.0
    assert variant.episodic_twin_head_exploration is False
    assert variant.twin_rollout_beam_width == 1
    # Flags off: no twin subtree anywhere in the parameter tree.
    assert set(variant.params) == set(pristine.params)
    assert "critic2" not in variant.params
    assert _tree_shapes(variant.params) == _tree_shapes(pristine.params)
    assert _params_equal(variant.params, pristine.params)

    for step in range(1, 6):
        batch = _batch(seed=11 + step)
        pristine_metrics = _run_update(pristine, batch, step=step)
        variant_metrics = _run_update(variant, batch, step=step)

        assert set(variant_metrics) == set(pristine_metrics)
        assert np.allclose(
            variant_metrics["critic_loss"],
            pristine_metrics["critic_loss"],
            atol=1e-6,
            rtol=0.0,
        ), (step, variant_metrics["critic_loss"], pristine_metrics["critic_loss"])
        assert _tree_shapes(variant.params) == _tree_shapes(pristine.params)
        assert _params_equal(variant.params, pristine.params)
        assert _params_equal(
            variant.target_critic_params, pristine.target_critic_params
        )


def test_flags_off_matches_pristine_act():
    cfg = _compose("cqn_as_twin_critic")
    pristine = _make_pristine(cfg)
    variant = _make_variant(cfg)

    obs_rng = np.random.default_rng(1011)
    obs = {
        "low_dim_state": obs_rng.normal(size=(1, 1, LOW_DIM)).astype(np.float32)
    }
    for eval_mode in (True, False):
        pristine_action = np.asarray(
            pristine.act(dict(obs), step=100, eval_mode=eval_mode)
        )
        variant_action = np.asarray(
            variant.act(dict(obs), step=100, eval_mode=eval_mode)
        )
        assert variant_action.shape == pristine_action.shape
        assert np.allclose(variant_action, pristine_action, atol=1e-6)


# ----------------------------------------------------------------------
# 2. flags-on sanity
# ----------------------------------------------------------------------


_TWIN_METRIC_KEYS = (
    "critic_loss",
    "critic1_loss",
    "critic2_loss",
    "one_step_critic_loss",
    "auxiliary_critic_loss",
    "auxiliary_td_loss_weight",
    "twin_q_disagreement",
    "twin_target1_fraction",
    "behavior_critic_gap",
    "greedy_critic_gap",
    "mc_lower_bound_fraction",
    "behavior_candidate_fraction",
)


def test_twin_flags_on_act_and_update_finite():
    cfg = _twin_cfg(
        "method.episodic_twin_head_exploration=true",
        "method.twin_rollout_beam_width=8",
    )
    agent = _make_variant(cfg)

    assert agent.pessimistic_twin_critic is True
    assert agent.episodic_twin_head_exploration is True
    assert agent.twin_rollout_beam_width == 8
    assert "critic2" in agent.params
    assert isinstance(agent.target_critic_params, tuple)
    assert len(agent.target_critic_params) == 2

    obs_rng = np.random.default_rng(5)
    obs = {
        "low_dim_state": obs_rng.normal(size=(1, 1, LOW_DIM)).astype(np.float32)
    }
    # Episode start: reset samples the per-environment head.
    agent.reset(step=0, agents_to_reset=[0])
    assert agent._episodic_twin_heads[0] in (0, 1)
    for eval_mode in (False, True):
        action = np.asarray(agent.act(dict(obs), step=10, eval_mode=eval_mode))
        assert action.shape == (1, ACTION_SEQUENCE, ACTION_DIM)
        assert np.all(np.isfinite(action))
        assert np.all(action >= -1.0 - 1e-6)
        assert np.all(action <= 1.0 + 1e-6)

    metrics = _run_update(agent, _batch())
    for key in _TWIN_METRIC_KEYS:
        assert key in metrics, sorted(metrics)
    for key, value in metrics.items():
        assert np.all(np.isfinite(np.asarray(value, dtype=np.float64))), key

    diagnostics = agent.rollout_diagnostics()
    assert diagnostics["episodic_twin_head_assignments"] == 1.0
    assert (
        diagnostics["episodic_twin_head0_rate"]
        + diagnostics["episodic_twin_head1_rate"]
        == 1.0
    )


def test_auxiliary_td_loss_weight_runs_and_reweights():
    overrides = (
        "replay.nstep=1",
        "replay.auxiliary_nstep=4",
        "replay.include_tp1=true",
    )
    control = _make_variant(_twin_cfg(*overrides))
    treatment = _make_variant(
        _twin_cfg(*overrides, "method.auxiliary_td_loss_weight=1.0")
    )
    assert treatment.auxiliary_td_loss_weight == 1.0

    batch = _batch(auxiliary=True)
    control_metrics = _run_update(control, batch)
    treatment_metrics = _run_update(treatment, batch)

    for key, value in treatment_metrics.items():
        assert np.all(np.isfinite(np.asarray(value, dtype=np.float64))), key
    assert control_metrics["auxiliary_td_loss_weight"] == 0.0
    assert treatment_metrics["auxiliary_td_loss_weight"] == 1.0
    # 0.5 * (L_1step + L_Nstep): the reported one-step piece is unchanged, the
    # combined objective is the normalized mean of the two horizons.
    assert np.allclose(
        control_metrics["critic_loss"],
        control_metrics["one_step_critic_loss"],
        atol=1e-6,
    )
    assert np.allclose(
        treatment_metrics["critic_loss"],
        0.5
        * (
            treatment_metrics["one_step_critic_loss"]
            + treatment_metrics["auxiliary_critic_loss"]
        ),
        atol=1e-5,
    )
    assert treatment_metrics["auxiliary_critic_loss"] > 0.0


def test_twin_flags_on_under_jit():
    """The twin/beam rollout graph must also trace: jax.lax.cond over the two
    episode heads and the beam's dynamic gathers are jit-only failure modes."""

    agent = _make_variant(
        _twin_cfg(
            "backend.jit=true",
            "method.episodic_twin_head_exploration=true",
            "method.twin_rollout_beam_width=8",
        )
    )
    obs_rng = np.random.default_rng(23)
    obs = {
        "low_dim_state": obs_rng.normal(size=(1, 1, LOW_DIM)).astype(np.float32)
    }
    agent.reset(step=0, agents_to_reset=[0])
    for eval_mode in (True, False):
        action = np.asarray(agent.act(dict(obs), step=10, eval_mode=eval_mode))
        assert action.shape == (1, ACTION_SEQUENCE, ACTION_DIM)
        assert np.all(np.isfinite(action))
    metrics = _run_update(agent, _batch())
    for key, value in metrics.items():
        assert np.all(np.isfinite(np.asarray(value, dtype=np.float64))), key


def test_twin_state_dict_round_trip():
    agent = _make_variant(_twin_cfg("method.twin_rollout_beam_width=1"))
    _run_update(agent, _batch())
    state = agent.state_dict()

    restored = _make_variant(_twin_cfg("method.twin_rollout_beam_width=1"))
    restored.load_state_dict(state)
    assert set(restored.params) == {"critic", "critic2"} | (
        {"encoder"} if "encoder" in agent.params else set()
    )
    assert _params_equal(restored.params, agent.params)
    assert isinstance(restored.target_critic_params, tuple)
    assert _params_equal(
        restored.target_critic_params, agent.target_critic_params
    )


def test_auxiliary_td_loss_weight_requires_auxiliary_batch():
    agent = _make_variant(
        _twin_cfg(
            "replay.nstep=1",
            "replay.auxiliary_nstep=4",
            "replay.include_tp1=true",
            "method.auxiliary_td_loss_weight=1.0",
        )
    )
    with pytest.raises(KeyError, match="auxiliary-horizon targets"):
        _run_update(agent, _batch(auxiliary=False))


# ----------------------------------------------------------------------
# 3. flags-on == research monolith
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra",
    [
        (),
        ("method.episodic_twin_head_exploration=true",),
        (
            "method.episodic_twin_head_exploration=true",
            "method.twin_rollout_beam_width=8",
        ),
    ],
    ids=["twin", "twin+episodic", "twin+episodic+beam8"],
)
def test_flags_on_matches_research_update(extra):
    variant = _make_variant(_twin_cfg(*extra))
    research = _make_research(_research_twin_cfg(*extra))
    assert type(research).__module__ == "robobase.method.cqn_as_research"

    assert _tree_shapes(variant.params) == _tree_shapes(research.params)
    assert _params_equal(variant.params, research.params)

    # The C51 heads are zero-initialized, so both critics agree exactly on the
    # first update; several steps are needed before twin_q_disagreement and the
    # min-Q target selection actually discriminate.
    disagreement = 0.0
    for step in range(1, 6):
        batch = _batch(seed=11 + step)
        variant_metrics = _run_update(variant, batch, step=step)
        research_metrics = _run_update(research, batch, step=step)
        assert set(variant_metrics) == set(research_metrics)
        assert np.allclose(
            variant_metrics["critic_loss"],
            research_metrics["critic_loss"],
            atol=1e-5,
            rtol=0.0,
        ), (step, variant_metrics["critic_loss"], research_metrics["critic_loss"])
        for key in _TWIN_METRIC_KEYS:
            assert np.allclose(
                variant_metrics[key], research_metrics[key], atol=1e-5, rtol=0.0
            ), (step, key, variant_metrics[key], research_metrics[key])
        assert _params_equal(variant.params, research.params)
        assert _params_equal(
            variant.target_critic_params, research.target_critic_params
        )
        disagreement = max(disagreement, variant_metrics["twin_q_disagreement"])
    assert disagreement > 0.0, "twin critics never diverged; check is vacuous"


def test_flags_on_matches_research_auxiliary_update():
    overrides = (
        "replay.nstep=1",
        "replay.auxiliary_nstep=4",
        "replay.include_tp1=true",
        "method.auxiliary_td_loss_weight=1.0",
    )
    variant = _make_variant(_twin_cfg(*overrides))
    research = _make_research(_research_twin_cfg(*overrides))

    batch = _batch(auxiliary=True)
    variant_metrics = _run_update(variant, batch)
    research_metrics = _run_update(research, batch)
    assert np.allclose(
        variant_metrics["critic_loss"],
        research_metrics["critic_loss"],
        atol=1e-5,
        rtol=0.0,
    ), (variant_metrics["critic_loss"], research_metrics["critic_loss"])
    assert np.allclose(
        variant_metrics["auxiliary_critic_loss"],
        research_metrics["auxiliary_critic_loss"],
        atol=1e-5,
        rtol=0.0,
    )


def test_flags_on_matches_research_act():
    variant = _make_variant(
        _twin_cfg(
            "method.episodic_twin_head_exploration=true",
            "method.twin_rollout_beam_width=8",
        )
    )
    research = _make_research(
        _research_twin_cfg(
            "method.episodic_twin_head_exploration=true",
            "method.twin_rollout_beam_width=8",
        )
    )
    obs_rng = np.random.default_rng(17)
    obs = {
        "low_dim_state": obs_rng.normal(size=(1, 1, LOW_DIM)).astype(np.float32)
    }
    variant.reset(step=0, agents_to_reset=[0])
    research.reset(step=0, agents_to_reset=[0])
    assert (
        variant._episodic_twin_heads[0] == research._episodic_twin_heads[0]
    )
    for eval_mode in (True, False):
        variant_action = np.asarray(
            variant.act(dict(obs), step=10, eval_mode=eval_mode)
        )
        research_action = np.asarray(
            research.act(dict(obs), step=10, eval_mode=eval_mode)
        )
        assert variant_action.shape == research_action.shape
        assert np.allclose(variant_action, research_action, atol=1e-5)


# ----------------------------------------------------------------------
# 4. behavioural gates adapted from tests/unit/test_cqn_as_beam_single.py
# ----------------------------------------------------------------------


def test_beam_single_critic_attribute_round_trip():
    agent = _make_variant(
        _compose("cqn_as_twin_critic", "method.twin_rollout_beam_width=8")
    )
    assert agent.twin_rollout_beam_width == 8
    assert agent.pessimistic_twin_critic is False
    assert agent.episodic_twin_head_exploration is False


def test_beam_default_width_one_construction_unchanged():
    agent = _make_variant(_compose("cqn_as_twin_critic"))
    assert agent.twin_rollout_beam_width == 1
    assert agent.pessimistic_twin_critic is False


def test_beam_eval_action_matches_greedy_contract_and_reranks():
    """After identical updates the beam agent's eval action keeps the greedy
    chunk contract (shape, bounds), while the beam's complete-chunk rerank
    score is at least the greedy chunk's (the joint top-1 greedy path is
    always a beam member) and the selected chunks differ."""

    greedy = _make_variant(_compose("cqn_as_twin_critic", "num_eval_envs=8"))
    beam = _make_variant(
        _compose(
            "cqn_as_twin_critic",
            "num_eval_envs=8",
            "method.twin_rollout_beam_width=8",
        )
    )
    for step in range(1, 4):
        batch = _batch(seed=11 + step)
        greedy.update(iter([_copy(batch)]), step=step)
        beam.update(iter([_copy(batch)]), step=step)
    # A rollout-only maximizer: the update graph must be untouched.
    assert _params_equal(greedy.params, beam.params)

    obs_rng = np.random.default_rng(1011)
    obs = {
        "low_dim_state": obs_rng.normal(size=(8, 1, LOW_DIM)).astype(np.float32)
    }
    greedy_action = np.asarray(greedy.act(dict(obs), step=1, eval_mode=True))
    beam_action = np.asarray(beam.act(dict(obs), step=1, eval_mode=True))
    assert beam_action.shape == greedy_action.shape
    for action in (greedy_action, beam_action):
        assert np.all(np.isfinite(action))
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)
    assert np.any(np.abs(beam_action - greedy_action) > 1e-6)

    obs_inputs = beam._prepare_rl_obs_inputs(obs)
    features = beam._rl_features(
        beam.params.get("encoder", None), obs_inputs, stop_gradient=True
    )
    critic_params = beam.target_critic_params
    beam_chunk, _ = beam._joint_beam_action(critic_params, features)
    greedy_chunk, _ = beam._greedy_action(critic_params, features)
    assert beam_chunk.shape == greedy_chunk.shape
    beam_score = np.asarray(
        beam._score_action_sequence_for_backup(
            critic_params, features, beam_chunk
        )
    )
    greedy_score = np.asarray(
        beam._score_action_sequence_for_backup(
            critic_params, features, greedy_chunk
        )
    )
    assert np.all(beam_score >= greedy_score - 1e-5)
    assert np.any(
        np.abs(np.asarray(beam_chunk) - np.asarray(greedy_chunk)) > 1e-6
    )


def test_beam_twin_path_validation_unchanged():
    agent = _make_variant(
        _twin_cfg(
            "method.twin_rollout_beam_width=2",
            "method.episodic_twin_head_exploration=true",
        )
    )
    assert agent.twin_rollout_beam_width == 2
    assert agent.pessimistic_twin_critic is True
    with pytest.raises(
        ValueError, match="episodic_twin_head_exploration=true"
    ):
        _make_variant(_twin_cfg("method.twin_rollout_beam_width=2"))


def test_twin_requires_direct_c51_platform():
    with pytest.raises(ValueError, match="use_dueling=false"):
        _make_variant(
            _twin_cfg("method.use_dueling=true")
        )
    with pytest.raises(ValueError, match="centralized_critic=false"):
        _make_variant(_twin_cfg("method.centralized_critic=true"))


def test_auxiliary_and_exploration_require_twin_critic():
    with pytest.raises(ValueError, match="pessimistic_twin_critic=true"):
        _make_variant(
            _compose(
                "cqn_as_twin_critic",
                "method.auxiliary_td_loss_weight=1.0",
                "replay.nstep=1",
                "replay.auxiliary_nstep=4",
                "replay.include_next_action=true",
            )
        )
    with pytest.raises(ValueError, match="pessimistic_twin_critic=true"):
        _make_variant(
            _compose(
                "cqn_as_twin_critic",
                "method.episodic_twin_head_exploration=true",
            )
        )


def test_twin_requires_replay_next_action():
    with pytest.raises(ValueError, match="replay.include_next_action=true"):
        cqn_as_twin_critic_spec_from_cfg(
            _compose(
                "cqn_as_twin_critic",
                "method.pessimistic_twin_critic=true",
                "replay.include_next_action=false",
            )
        )


def test_twin_update_requires_action_tp1():
    agent = _make_variant(_twin_cfg())
    batch = _batch()
    batch.pop("action_tp1")
    with pytest.raises(KeyError, match="action_tp1"):
        _run_update(agent, batch)


def test_episodic_twin_head_rng_checkpoint_round_trip():
    def build():
        return _make_variant(
            _twin_cfg(
                "num_train_envs=4",
                "method.episodic_twin_head_exploration=true",
            )
        )

    agent = build()
    fresh_state = agent._episodic_twin_head_rng.bit_generator.state
    for _ in range(3):
        agent.reset(step=0, agents_to_reset=[0, 1, 2, 3])
    advanced_state = agent.checkpoint_state_dict()
    assert "episodic_twin_head_rng_state" in advanced_state
    assert advanced_state["episodic_twin_head_rng_state"] != fresh_state

    resumed = build()
    resumed.load_checkpoint_state_dict(advanced_state)
    assert (
        resumed._episodic_twin_head_rng.bit_generator.state
        == advanced_state["episodic_twin_head_rng_state"]
    )
    # A resume starts fresh episodes, so no head survives the load.
    assert np.all(resumed._episodic_twin_heads < 0)
    # The continued stream matches the uninterrupted process step for step.
    agent.reset(step=0, agents_to_reset=[0, 1, 2, 3])
    resumed.reset(step=0, agents_to_reset=[0, 1, 2, 3])
    assert np.array_equal(
        agent._episodic_twin_heads, resumed._episodic_twin_heads
    )
    # ... while a process restarted from the seed would have diverged.
    restarted = build()
    restarted.reset(step=0, agents_to_reset=[0, 1, 2, 3])
    assert (
        restarted._episodic_twin_head_rng.bit_generator.state
        != resumed._episodic_twin_head_rng.bit_generator.state
    )


def test_flags_off_checkpoint_state_still_carries_rng_slot():
    agent = _make_variant(_compose("cqn_as_twin_critic"))
    state = agent.checkpoint_state_dict()
    assert "episodic_twin_head_rng_state" in state
    agent.load_checkpoint_state_dict(state)
