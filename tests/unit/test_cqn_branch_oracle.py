from types import SimpleNamespace

import numpy as np
import pytest

from scripts.finetune_cqn_branch_oracle import (
    _baseline_rollout_success,
    _all_scores,
    _branch_coverage_summary,
    _candidate_plans,
    _load_dataset_cache,
    _oracle_training_data,
    _policy_candidate_log_probabilities,
    _records_from_scores,
    _return_stochasticity_summary,
    _score_candidates,
    _sibling_candidate_plans,
    _subset_branch_dataset,
    _summarize_records,
    _train_oracle_critic,
    _trees_bitwise_equal,
    _write_dataset_cache,
)


def test_policy_candidate_log_probabilities_score_requested_level_and_dimension():
    import jax.numpy as jnp

    class Agent:
        params = {"policy": {}}

        def _policy_logits_per_level(self, params, features, actions):
            del params, actions
            batch = features.shape[0]
            logits = jnp.zeros((batch, 2, 2, 3), dtype=jnp.float32)
            logits = logits.at[:, 1, 1].set(
                jnp.asarray(
                    [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]
                )
            )
            encoded = jnp.zeros((batch, 2, 2), dtype=jnp.int32)
            encoded = encoded.at[:, 1, 1].set(jnp.arange(batch))
            return logits, encoded

    scores = _policy_candidate_log_probabilities(
        Agent(),
        np.zeros((4,), np.float32),
        np.zeros((3, 2, 2), np.float32),
        action_dimension=1,
        score_level=1,
    )

    np.testing.assert_allclose(scores, scores[0])
    assert scores[0] > -0.1


def test_baseline_rollout_success_uses_cumulative_sparse_reward():
    class Agent:
        def reset(self, env_index, env_ids):
            assert env_index == 0
            assert env_ids == [0]

        def act(self, observation, step, eval_mode):
            assert observation["state"].shape == (1, 1)
            assert eval_mode
            return np.full((1, 1, 1), step, np.float32)

    class Env:
        def __init__(self, rewards):
            self.rewards = rewards
            self.step_index = 0

        def reset(self, seed):
            assert seed == 17
            self.step_index = 0
            return {"state": np.zeros((1,), np.float32)}, {}

        def step(self, plan):
            assert plan.shape == (1, 1)
            reward = self.rewards[self.step_index]
            self.step_index += 1
            terminated = self.step_index == len(self.rewards)
            return (
                {"state": np.zeros((1,), np.float32)},
                reward,
                terminated,
                False,
                {},
            )

    assert _baseline_rollout_success(
        Env([0.1, 0.1, 0.1]),
        Agent(),
        eval_seed=17,
        max_steps=10,
    )
    assert not _baseline_rollout_success(
        Env([0.1, 0.1]),
        Agent(),
        eval_seed=17,
        max_steps=10,
    )


def test_branch_coverage_summary_groups_dimensions_and_anchor_steps():
    data = {
        "returns": np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 1.0],
                [0.2, 0.2, 0.2],
                [0.0, 0.0, 0.25],
            ],
            np.float32,
        ),
        "action_dimensions": np.asarray([0, 0, 1, 1], np.int32),
        "metadata": [
            {
                "anchor_step": 30,
                "outcomes": [{"success": False}, {"success": False}],
            },
            {
                "anchor_step": 75,
                "outcomes": [{"success": False}, {"success": True}],
            },
            {
                "anchor_step": 30,
                "outcomes": [{"success": True}, {"success": True}],
            },
            {
                "anchor_step": 75,
                "outcomes": [{"success": True}, {"success": True}],
            },
        ],
    }

    summary = _branch_coverage_summary(data, return_atol=1e-12)

    assert summary["overall"]["num_states"] == 4
    assert summary["overall"]["num_informative_states"] == 2
    assert summary["overall"]["informative_fraction"] == 0.5
    assert summary["overall"]["num_any_success_states"] == 3
    assert summary["overall"]["num_all_success_states"] == 2
    assert summary["by_dimension"]["0"]["num_informative_states"] == 1
    assert summary["by_dimension"]["1"]["num_informative_states"] == 1
    assert summary["by_anchor_step"]["30"]["informative_fraction"] == 0.0
    assert summary["by_anchor_step"]["75"]["informative_fraction"] == 1.0


def test_candidate_plans_apply_coherent_delta_and_report_clipping():
    agent = SimpleNamespace(
        action_sequence=5,
        action_dim=2,
        structured_exploration_level=1,
        bins=5,
        _step_action_low=np.asarray([-1.0, -1.0], np.float32),
        _step_action_high=np.asarray([1.0, 1.0], np.float32),
    )
    baseline = np.zeros((5, 2), np.float32)
    baseline[:3, 1] = np.asarray([0.97, 0.0, -0.97])

    candidates, requested_delta = _candidate_plans(
        agent,
        baseline,
        action_dimension=1,
        intervention_horizon=3,
    )

    assert candidates.shape == (3, 5, 2)
    np.testing.assert_allclose(requested_delta, [-0.08, 0.0, 0.08])
    np.testing.assert_allclose(candidates[:, 0, 1], [0.89, 0.97, 1.0])
    np.testing.assert_allclose(candidates[:, 1, 1], [-0.08, 0.0, 0.08])
    np.testing.assert_allclose(candidates[:, 2, 1], [-1.0, -0.97, -0.89])
    np.testing.assert_allclose(
        candidates[:, 3:],
        np.repeat(baseline[None, 3:], 3, axis=0),
    )


def test_sibling_candidate_plans_share_prefix_and_repeat_delta():
    agent = SimpleNamespace(
        action_sequence=4,
        action_dim=2,
        levels=3,
        bins=5,
        _step_action_low=np.asarray([-1.0, -1.0], np.float32),
        _step_action_high=np.asarray([1.0, 1.0], np.float32),
    )
    baseline = np.zeros((4, 2), np.float32)
    baseline[:2, 1] = np.asarray([-0.95, 0.9])

    candidates, deltas = _sibling_candidate_plans(
        agent,
        baseline,
        action_dimension=1,
        intervention_horizon=2,
        force_level=1,
    )

    np.testing.assert_allclose(
        candidates[:, 0, 1],
        [-0.96, -0.88, -0.80, -0.72, -0.64],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        deltas,
        [-0.01, 0.07, 0.15, 0.23, 0.31],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        candidates[:, 1, 1],
        [0.89, 0.97, 1.0, 1.0, 1.0],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        candidates[:, 2:],
        np.repeat(baseline[None, 2:], 5, axis=0),
    )


def test_score_candidates_uses_hybrid_advantage_without_flow():
    import jax.numpy as jnp

    class Agent:
        hybrid_flow_v_direct_a = True
        action_sequence = 1
        action_dim = 2
        levels = 2
        _flat_action_dim = 2

        def _advantage_outputs_per_level(
            self, params, features, actions
        ):
            del params, features
            chosen = jnp.stack([actions, 2.0 * actions], axis=1)
            return None, None, chosen, None

    actions = np.asarray(
        [[[[0.0, 0.1]], [[0.0, 0.2]], [[0.0, 0.3]]]],
        np.float32,
    )
    scores = _score_candidates(
        Agent(),
        {},
        np.zeros((1, 4), np.float32),
        actions,
        np.asarray([1], np.int32),
        score_level=1,
    )

    np.testing.assert_allclose(np.asarray(scores), [[0.2, 0.4, 0.6]])


def test_score_candidates_uses_direct_scalar_q_without_categorical_support():
    import jax.numpy as jnp

    class Agent:
        hybrid_flow_v_direct_a = False
        direct_scalar_q = True
        action_sequence = 1
        action_dim = 2
        levels = 2
        _flat_action_dim = 2

        def _direct_q_per_level(self, params, features, actions):
            del params, features
            chosen = jnp.stack([actions, 3.0 * actions], axis=1)
            return chosen, None

        def _critic_logits_per_level(self, *args, **kwargs):
            raise AssertionError("direct scalar-Q must not use C51 logits")

    actions = np.asarray(
        [[[[0.1, 0.0]], [[0.2, 0.0]], [[0.3, 0.0]]]],
        np.float32,
    )
    scores = _score_candidates(
        Agent(),
        {},
        np.zeros((1, 4), np.float32),
        actions,
        np.asarray([0], np.int32),
        score_level=1,
    )

    np.testing.assert_allclose(np.asarray(scores), [[0.3, 0.6, 0.9]])


def test_direct_scalar_q_oracle_fit_learns_same_state_action_ordering():
    import jax.numpy as jnp
    import optax as optax_module

    class Agent:
        hybrid_flow_v_direct_a = False
        direct_scalar_q = True
        action_sequence = 1
        action_dim = 1
        levels = 1
        _flat_action_dim = 1
        optax = optax_module

        def _direct_q_per_level(self, params, features, actions):
            del features
            chosen = params["scale"] * actions.reshape((-1, 1, 1))
            return chosen, None

    actions = np.asarray(
        [[[-1.0]], [[0.0]], [[1.0]]],
        np.float32,
    )
    dataset = {
        "features": np.zeros((4, 2), np.float32),
        "actions": np.repeat(actions[None], 4, axis=0),
        "returns": np.repeat(
            np.asarray([[-1.0, 0.0, 1.0]], np.float32),
            4,
            axis=0,
        ),
        "action_dimensions": np.zeros((4,), np.int32),
    }
    initial = {"scale": jnp.asarray(0.0, jnp.float32)}

    fitted, history, informative = _train_oracle_critic(
        Agent(),
        initial,
        dataset,
        updates=20,
        batch_size=4,
        learning_rate=0.1,
        temperature=0.05,
        weight_decay=0.0,
        return_atol=1e-12,
        seed=3,
        score_level=0,
        delta_regression_weight=1.0,
        sampling_mode="full_batch",
    )
    scores = _all_scores(
        Agent(),
        fitted,
        dataset,
        score_level=0,
    )

    assert informative == 4
    assert history[-1]["batch_pairwise_accuracy"] == 1.0
    assert float(fitted["scale"]) > 0.0
    assert np.all(np.diff(scores, axis=1) > 0.0)


def test_all_scores_chunks_without_changing_candidate_order():
    import jax.numpy as jnp

    class Agent:
        hybrid_flow_v_direct_a = True
        action_sequence = 1
        action_dim = 1
        levels = 1
        _flat_action_dim = 1

        def _advantage_outputs_per_level(
            self, params, features, actions
        ):
            del params, features
            chosen = actions[:, None]
            return None, None, chosen, None

    dataset = {
        "features": np.zeros((5, 2), np.float32),
        "actions": np.arange(15, dtype=np.float32).reshape(5, 3, 1, 1),
        "action_dimensions": np.zeros((5,), np.int32),
    }
    scores = _all_scores(
        Agent(),
        {},
        dataset,
        score_level=0,
        score_batch_size=2,
    )

    np.testing.assert_array_equal(scores, np.arange(15).reshape(5, 3))


def test_trees_bitwise_equal_checks_structure_shape_and_values():
    left = {
        "encoder": [np.asarray([1.0, 2.0], np.float32)],
        "policy": {"kernel": np.asarray([[3]], np.int32)},
    }
    right = {
        "encoder": [np.asarray([1.0, 2.0], np.float32)],
        "policy": {"kernel": np.asarray([[3]], np.int32)},
    }
    changed = {
        "encoder": [np.asarray([1.0, 2.1], np.float32)],
        "policy": {"kernel": np.asarray([[3]], np.int32)},
    }

    assert _trees_bitwise_equal(left, right)
    assert not _trees_bitwise_equal(left, changed)
    assert not _trees_bitwise_equal(left, {"encoder": right["encoder"]})


def test_oracle_summary_uses_only_informative_states_and_state_bootstrap():
    scores = np.asarray([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
    returns = np.asarray([[0.0, 1.0, 2.0], [0.5, 0.5, 0.5]])
    metadata = [
        {
            "eval_seed": 1,
            "anchor_step": 30,
            "action_dimension": 13,
            "actual_first_delta": [-0.1, 0.0, 0.1],
            "policy_log_probability": [0.0, 1.0, 2.0],
        },
        {"eval_seed": 2, "anchor_step": 30, "action_dimension": 14},
    ]

    records = _records_from_scores(
        scores,
        returns,
        metadata,
        return_atol=1e-12,
    )
    summary = _summarize_records(
        records,
        bootstrap_replicates=100,
        bootstrap_seed=7,
    )

    assert summary["num_states"] == 2
    assert summary["num_informative_states"] == 1
    assert summary["num_informative_pairs"] == 3
    assert summary["pairwise_sign_accuracy"] == 1.0
    assert summary["mean_spearman"] == 1.0
    assert summary["top1_match_rate"] == 1.0
    assert summary["random_top1_probability"] == pytest.approx(1.0 / 3.0)
    assert summary["behavior_proxy_top1_match_rate"] == 0.0
    assert summary["behavior_proxy_mean_realized_regret"] == 1.0
    assert summary["policy_prior_pairwise_sign_accuracy"] == 1.0
    assert summary["policy_prior_top1_match_rate"] == 1.0
    assert summary["policy_prior_mean_realized_regret"] == 0.0
    assert summary["state_bootstrap"]["num_states"] == 1
    assert summary["state_bootstrap"]["pairwise_sign_accuracy_ci"] == [
        1.0,
        1.0,
    ]
    assert summary["seed_bootstrap"]["unit"] == "informative_eval_seed"
    assert summary["seed_bootstrap"]["num_seeds"] == 1
    assert summary["seed_bootstrap"]["pairwise_sign_accuracy_ci"] == [
        1.0,
        1.0,
    ]


def test_branch_dataset_cache_round_trip_and_metadata_validation(tmp_path):
    data = {
        "features": np.arange(6, dtype=np.float32).reshape(2, 3),
        "actions": np.zeros((2, 3, 4, 2), np.float32),
        "returns": np.asarray([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]),
        "return_samples": np.asarray(
            [
                [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
                [[2.0, 2.0], [1.0, 1.0], [0.0, 0.0]],
            ]
        ),
        "action_dimensions": np.asarray([0, 1], np.int32),
        "metadata": [{"eval_seed": 1}, {"eval_seed": 2}],
    }
    metadata = {"train_seeds": [1], "heldout_seeds": [2]}
    path = tmp_path / "branches.npz"

    _write_dataset_cache(path, data, data, metadata)
    train, heldout = _load_dataset_cache(path, metadata)

    np.testing.assert_array_equal(train["features"], data["features"])
    np.testing.assert_array_equal(heldout["returns"], data["returns"])
    np.testing.assert_array_equal(
        heldout["return_samples"], data["return_samples"]
    )
    assert train["metadata"] == data["metadata"]
    with pytest.raises(ValueError, match="metadata mismatch"):
        _load_dataset_cache(path, {"train_seeds": [99]})


def test_return_stochasticity_summary_detects_return_and_success_variation():
    dataset = {
        "returns": np.asarray([[0.5, 0.2]], np.float32),
        "return_samples": np.asarray(
            [[[0.0, 0.5, 1.0], [0.2, 0.2, 0.2]]],
            np.float32,
        ),
        "metadata": [
            {
                "repeat_outcomes": [
                    [
                        {"success": False},
                        {"success": True},
                        {"success": True},
                    ],
                    [
                        {"success": False},
                        {"success": False},
                        {"success": False},
                    ],
                ]
            }
        ],
    }

    summary = _return_stochasticity_summary(dataset, return_atol=1e-12)

    assert summary["repeats"] == 3
    assert summary["num_candidates"] == 2
    assert summary["num_variable_return_candidates"] == 1
    assert summary["variable_return_fraction"] == 0.5
    assert summary["max_return_span"] == 1.0
    assert summary["num_stochastic_success_candidates"] == 1


def test_oracle_training_shuffle_preserves_each_state_return_multiset():
    dataset = {
        "returns": np.asarray(
            [[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0]],
            np.float32,
        ),
        "features": np.zeros((2, 3), np.float32),
    }

    shuffled = _oracle_training_data(
        dataset,
        shuffle_mode="within_state",
        seed=7,
    )

    assert shuffled is not dataset
    np.testing.assert_array_equal(
        np.sort(shuffled["returns"], axis=1),
        np.sort(dataset["returns"], axis=1),
    )
    np.testing.assert_array_equal(
        shuffled["features"], dataset["features"]
    )
    np.testing.assert_array_equal(
        dataset["returns"],
        [[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0]],
    )


def test_subset_branch_dataset_keeps_arrays_and_metadata_aligned():
    dataset = {
        "features": np.arange(6).reshape(3, 2),
        "actions": np.arange(12).reshape(3, 2, 2),
        "returns": np.arange(6).reshape(3, 2),
        "action_dimensions": np.asarray([0, 1, 2]),
        "metadata": [
            {"eval_seed": 10},
            {"eval_seed": 11},
            {"eval_seed": 12},
        ],
    }

    subset = _subset_branch_dataset(
        dataset,
        np.asarray([True, False, True]),
    )

    np.testing.assert_array_equal(subset["features"], [[0, 1], [4, 5]])
    np.testing.assert_array_equal(subset["action_dimensions"], [0, 2])
    assert subset["metadata"] == [
        {"eval_seed": 10},
        {"eval_seed": 12},
    ]
