"""R2 verification for the ``td-variants`` line.

1. flags-off == pristine ``robobase.method.cqn_as.CQNAS`` (critic_loss and
   param-tree shapes after one ``update()``);
2. flags-on sanity (``act()`` + ``update()`` finite, line metrics present);
3. flags-on == the research monolith ``cqn_as_research.CQNAS`` where the
   research path implements the same option.

Everything runs on CPU with ``backend.jit=false``.
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
from robobase.method.cqn_as import CQNAS  # noqa: E402
from robobase.method.cqn_as import (  # noqa: E402
    cqn_as_spec_from_cfg as pristine_spec_from_cfg,
)
from robobase.method.cqn_as_td_variants import (  # noqa: E402
    AutoregressiveSequenceDistributionalCritic,
    CQNASTdVariants,
    cqn_as_td_variants_spec_from_cfg,
    shift_replay_action_sequence,
)


CONFIG_DIR = str((Path(__file__).parents[2] / "robobase" / "cfgs").resolve())

ACTION_SEQUENCE = 3
ACTION_DIM = 4
LOW_DIM = 5
BATCH = 3
SEED = 0


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
                f"seed={SEED}",
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


def _batch(*, with_next_action: bool = False):
    rng = np.random.default_rng(7)
    batch = {
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
        "demo": np.ones((BATCH,), dtype=np.uint8),
    }
    if with_next_action:
        batch["action_tp1"] = rng.uniform(
            -1.0, 1.0, size=(BATCH, ACTION_SEQUENCE, ACTION_DIM)
        ).astype(np.float32)
    return batch


def _agent_kwargs(spec, cfg, observation_space, action_space):
    """Mirror ``factory.py``'s ``cqn_as_official`` construction branch."""

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


def _pristine_agent(*overrides: str):
    cfg = _compose("cqn_as_official", *overrides)
    observation_space, action_space = _spaces()
    spec = pristine_spec_from_cfg(cfg)
    return CQNAS(
        **_agent_kwargs(spec, cfg, observation_space, action_space)
    )


def _variant_agent(*overrides: str):
    cfg = _compose("cqn_as_td_variants", *overrides)
    observation_space, action_space = _spaces()
    spec = cqn_as_td_variants_spec_from_cfg(cfg)
    return CQNASTdVariants(
        **_agent_kwargs(spec, cfg, observation_space, action_space),
        td_target_action_source=spec.td_target_action_source,
        td_target_policy_value_beta=spec.td_target_policy_value_beta,
        critic_sequence_mode=spec.critic_sequence_mode,
        autoregressive_action_dims=spec.autoregressive_action_dims,
    )


def _research_agent(*overrides: str):
    cfg = _compose("cqn_as", *overrides)
    observation_space, action_space = _spaces()
    return create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )


def _shapes(tree):
    return jax.tree.map(lambda leaf: tuple(np.shape(leaf)), tree)


def _run_update(agent, batch):
    agent.logging = True
    return agent.update(iter([batch]), step=1)


def _perturb_critic(agent, seed=11):
    """Replace the zero-initialized critic head with a non-degenerate one.

    Both CQN-AS output heads (``advantage_head`` / ``value_head``) are
    zero-initialized, so a freshly built agent assigns the *same* Q to every
    bin of every action.  Under that degeneracy the bootstrap action is
    irrelevant and every ``td_target_action_source`` yields a bit-identical
    loss, which cannot distinguish the variants.  The perturbation is a pure
    function of ``seed`` and of the (identical) initial tree, so two agents
    built with the same config end up with identical critics.
    """

    leaves, treedef = jax.tree.flatten(agent.params["critic"])
    keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    critic = jax.tree.unflatten(
        treedef,
        [
            jnp.asarray(leaf)
            + 0.5 * jax.random.normal(key, jnp.shape(leaf), dtype=jnp.float32)
            for leaf, key in zip(leaves, keys)
        ],
    )
    agent.params = {**agent.params, "critic": critic}
    agent.target_critic_params = critic
    agent.opt_state = agent.optimizer.init(agent.params)
    return agent


# ----------------------------------------------------------------------
# 1. flags-off == pristine
# ----------------------------------------------------------------------
def test_flags_off_matches_pristine_cqn_as():
    batch = _batch()
    pristine = _pristine_agent()
    variant = _variant_agent()

    assert variant.td_target_action_source == "critic"
    assert variant.critic_sequence_mode == "full"
    assert variant.autoregressive_action_dims is False
    assert variant.td_target_policy_value_beta is None

    # Identical initialization: same seed, same critic module.
    np.testing.assert_equal(
        _shapes(pristine.params), _shapes(variant.params)
    )
    for pristine_leaf, variant_leaf in zip(
        jax.tree.leaves(pristine.params), jax.tree.leaves(variant.params)
    ):
        np.testing.assert_allclose(
            np.asarray(pristine_leaf), np.asarray(variant_leaf), atol=0.0
        )

    pristine_metrics = _run_update(pristine, batch)
    variant_metrics = _run_update(variant, batch)

    assert variant_metrics["critic_loss"] == pytest.approx(
        pristine_metrics["critic_loss"], abs=1e-6
    )
    for key in ("entropy", "target_entropy", "loss_coeff"):
        assert variant_metrics[key] == pytest.approx(
            pristine_metrics[key], abs=1e-6
        )
    # No td-variants metric leaks onto the legacy path.
    assert "td_target_replay_next" not in variant_metrics
    assert "behavior_candidate_fraction" not in variant_metrics

    np.testing.assert_equal(
        _shapes(pristine.params), _shapes(variant.params)
    )
    np.testing.assert_equal(
        _shapes(pristine.target_critic_params),
        _shapes(variant.target_critic_params),
    )
    for pristine_leaf, variant_leaf in zip(
        jax.tree.leaves(pristine.params), jax.tree.leaves(variant.params)
    ):
        np.testing.assert_allclose(
            np.asarray(pristine_leaf),
            np.asarray(variant_leaf),
            atol=1e-6,
            rtol=0.0,
        )


def test_flags_off_act_matches_pristine():
    observation = _observation()
    pristine = _pristine_agent()
    variant = _variant_agent()
    for eval_mode in (True, False):
        np.testing.assert_allclose(
            np.asarray(pristine.act(observation, step=100, eval_mode=eval_mode)),
            np.asarray(variant.act(observation, step=100, eval_mode=eval_mode)),
            atol=1e-6,
        )


# ----------------------------------------------------------------------
# 2. flags-on sanity
# ----------------------------------------------------------------------
def test_replay_next_td_target_runs_and_reports_metric():
    variant = _variant_agent("method.td_target_action_source=replay_next")
    assert variant.td_target_action_source == "replay_next"

    action = variant.act(_observation(), step=100, eval_mode=False)
    assert np.all(np.isfinite(np.asarray(action)))

    metrics = _run_update(variant, _batch())
    assert metrics["td_target_replay_next"] == pytest.approx(1.0)
    for key, value in metrics.items():
        assert np.all(np.isfinite(np.asarray(value))), key

    # The bootstrap action is exactly the once-shifted replay chunk.
    batch = _batch()
    replay_actions = batch["action"].reshape((BATCH, -1))
    next_action, info = variant._td_target_action_for_update(
        variant.params["critic"],
        np.zeros((BATCH, 8), dtype=np.float32),
        replay_actions,
        replay_actions,
        np.zeros((BATCH,), dtype=np.float32),
        jax.random.PRNGKey(0),
    )
    assert info == {}
    np.testing.assert_allclose(
        np.asarray(next_action),
        np.asarray(
            shift_replay_action_sequence(
                replay_actions, ACTION_SEQUENCE, ACTION_DIM
            )
        ),
    )

    later = _run_update(
        _perturb_critic(
            _variant_agent("method.td_target_action_source=replay_next")
        ),
        batch,
    )
    baseline = _run_update(_perturb_critic(_variant_agent()), batch)
    assert later["critic_loss"] != pytest.approx(
        baseline["critic_loss"], abs=1e-8
    )


def test_critic_replay_max_runs_and_reports_candidate_metrics():
    variant = _variant_agent(
        "method.td_target_action_source=critic_replay_max"
    )
    assert variant.td_target_action_source == "critic_replay_max"

    action = variant.act(_observation(), step=100, eval_mode=False)
    assert np.all(np.isfinite(np.asarray(action)))

    metrics = _run_update(variant, _batch(with_next_action=True))
    for key in (
        "behavior_candidate_fraction",
        "behavior_candidate_score",
        "greedy_candidate_score",
        "behavior_minus_greedy_q",
    ):
        assert key in metrics
    assert 0.0 <= metrics["behavior_candidate_fraction"] <= 1.0
    assert metrics["behavior_minus_greedy_q"] == pytest.approx(
        metrics["behavior_candidate_score"] - metrics["greedy_candidate_score"],
        abs=1e-6,
    )
    for key, value in metrics.items():
        assert np.all(np.isfinite(np.asarray(value))), key


def test_critic_replay_max_requires_action_tp1_in_batch():
    variant = _variant_agent(
        "method.td_target_action_source=critic_replay_max"
    )
    with pytest.raises(KeyError, match="include_next_action"):
        _run_update(variant, _batch(with_next_action=False))


def test_effective_k0_restricts_td_to_the_executed_token():
    variant = _variant_agent("method.critic_sequence_mode=effective_k0")
    assert variant.critic_sequence_mode == "effective_k0"

    dummy = np.arange(
        2 * 3 * ACTION_SEQUENCE * ACTION_DIM * 5, dtype=np.float32
    ).reshape((2, 3, ACTION_SEQUENCE * ACTION_DIM, 5))
    sliced = np.asarray(variant._critic_training_slice(dummy))
    assert sliced.shape == (2, 3, ACTION_DIM, 5)
    np.testing.assert_allclose(sliced, dummy[:, :, :ACTION_DIM])

    full = _variant_agent()
    np.testing.assert_allclose(
        np.asarray(full._critic_training_slice(dummy)), dummy
    )

    metrics = _run_update(_perturb_critic(variant), _batch())
    baseline = _run_update(_perturb_critic(full), _batch())
    for key, value in metrics.items():
        assert np.all(np.isfinite(np.asarray(value))), key
    assert metrics["critic_loss"] != pytest.approx(
        baseline["critic_loss"], abs=1e-8
    )


def test_autoregressive_action_dims_adds_causal_head_and_trains():
    variant = _variant_agent("method.autoregressive_action_dims=true")
    assert variant.autoregressive_action_dims is True
    assert isinstance(
        variant.critic_model, AutoregressiveSequenceDistributionalCritic
    )
    critic_keys = set(variant.params["critic"]["params"])
    assert "action_correction" in critic_keys
    assert "base_critic" in critic_keys

    action = variant.act(_observation(), step=100, eval_mode=False)
    assert np.asarray(action).shape == (BATCH, ACTION_SEQUENCE, ACTION_DIM)
    assert np.all(np.isfinite(np.asarray(action)))

    before = jax.tree.map(np.asarray, variant.params["critic"])
    metrics = _run_update(variant, _batch())
    for key, value in metrics.items():
        assert np.all(np.isfinite(np.asarray(value))), key
    after = jax.tree.map(np.asarray, variant.params["critic"])
    changed = any(
        not np.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after))
    )
    assert changed


def test_combined_flags_on_run_finite():
    variant = _variant_agent(
        "method.td_target_action_source=replay_next",
        "method.critic_sequence_mode=effective_k0",
        "method.autoregressive_action_dims=true",
    )
    variant.act(_observation(), step=100, eval_mode=True)
    metrics = _run_update(variant, _batch())
    for key, value in metrics.items():
        assert np.all(np.isfinite(np.asarray(value))), key
    assert metrics["td_target_replay_next"] == pytest.approx(1.0)


def test_shift_replay_action_sequence_matches_definition():
    actions = np.arange(
        BATCH * ACTION_SEQUENCE * ACTION_DIM, dtype=np.float32
    ).reshape((BATCH, ACTION_SEQUENCE * ACTION_DIM))
    shifted = np.asarray(
        shift_replay_action_sequence(actions, ACTION_SEQUENCE, ACTION_DIM)
    )
    reference = actions.reshape((BATCH, ACTION_SEQUENCE, ACTION_DIM))
    np.testing.assert_allclose(shifted[:, :-1], reference[:, 1:])
    np.testing.assert_allclose(shifted[:, -1], reference[:, -1])


# ----------------------------------------------------------------------
# Coupled options are rejected, never silently absorbed
# ----------------------------------------------------------------------
@pytest.mark.parametrize("source", ["bc_policy", "policy_value"])
def test_coupled_td_target_action_sources_are_rejected(source):
    with pytest.raises(ValueError, match="bc-policy"):
        _variant_agent(f"method.td_target_action_source={source}")


def test_td_target_policy_value_beta_is_rejected():
    with pytest.raises(ValueError, match="td_target_policy_value_beta"):
        _variant_agent("method.td_target_policy_value_beta=1.0")


def test_unknown_option_values_are_rejected():
    with pytest.raises(ValueError, match="td_target_action_source"):
        _variant_agent("method.td_target_action_source=nonsense")
    with pytest.raises(ValueError, match="critic_sequence_mode"):
        _variant_agent("method.critic_sequence_mode=nonsense")


# ----------------------------------------------------------------------
# 3. flags-on == research monolith (best effort)
# ----------------------------------------------------------------------
def test_replay_next_matches_research_monolith():
    variant = _variant_agent("method.td_target_action_source=replay_next")
    research = _research_agent("method.td_target_action_source=replay_next")
    variant_metrics = _run_update(_perturb_critic(variant), _batch())
    research_metrics = _run_update(_perturb_critic(research), _batch())
    assert variant_metrics["critic_loss"] == pytest.approx(
        research_metrics["critic_loss"], abs=1e-5
    )


def test_critic_replay_max_matches_research_monolith():
    variant = _variant_agent(
        "method.td_target_action_source=critic_replay_max"
    )
    research = _research_agent(
        "method.td_target_action_source=critic_replay_max"
    )
    batch = _batch(with_next_action=True)
    variant_metrics = _run_update(_perturb_critic(variant), batch)
    research_metrics = _run_update(_perturb_critic(research), batch)
    assert variant_metrics["critic_loss"] == pytest.approx(
        research_metrics["critic_loss"], abs=1e-5
    )
    for key in (
        "behavior_candidate_fraction",
        "behavior_candidate_score",
        "greedy_candidate_score",
    ):
        assert variant_metrics[key] == pytest.approx(
            research_metrics[key], abs=1e-5
        )


def test_autoregressive_action_dims_matches_research_monolith():
    variant = _variant_agent("method.autoregressive_action_dims=true")
    research = _research_agent("method.autoregressive_action_dims=true")
    np.testing.assert_equal(
        _shapes(variant.params["critic"]), _shapes(research.params["critic"])
    )
    variant_metrics = _run_update(_perturb_critic(variant), _batch())
    research_metrics = _run_update(_perturb_critic(research), _batch())
    assert variant_metrics["critic_loss"] == pytest.approx(
        research_metrics["critic_loss"], abs=1e-5
    )


def test_effective_k0_is_inert_in_the_research_single_objective_path():
    """Documented divergence, asserted so it cannot regress silently.

    In ``cqn_as_research.py`` ``_critic_training_slice`` is only called from
    the ``separate_bc_policy`` update path (``cqn_as_research.py:4909,
    4972-4973``); with ``separate_bc_policy=false`` the base CQN update
    (``cqn_research.py:1089``) never slices, so ``effective_k0`` is a no-op
    there.  This variant implements the slice on the pristine single-objective
    path, so the two intentionally disagree.
    """

    research_full = _research_agent("method.critic_sequence_mode=full")
    research_k0 = _research_agent("method.critic_sequence_mode=effective_k0")
    research_full_metrics = _run_update(_perturb_critic(research_full), _batch())
    research_k0_metrics = _run_update(_perturb_critic(research_k0), _batch())
    assert research_k0_metrics["critic_loss"] == pytest.approx(
        research_full_metrics["critic_loss"], abs=1e-8
    )

    variant_k0 = _variant_agent("method.critic_sequence_mode=effective_k0")
    variant_full = _variant_agent("method.critic_sequence_mode=full")
    variant_k0_metrics = _run_update(_perturb_critic(variant_k0), _batch())
    variant_full_metrics = _run_update(_perturb_critic(variant_full), _batch())
    assert variant_k0_metrics["critic_loss"] != pytest.approx(
        variant_full_metrics["critic_loss"], abs=1e-8
    )
