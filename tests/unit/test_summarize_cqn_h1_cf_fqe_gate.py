import copy

from scripts.summarize_cqn_h1_cf_fqe_gate import summarize


def _record(seed, anchor, dimension, q_accuracy, q_spearman):
    return {
        "eval_seed": seed,
        "anchor_step": anchor,
        "action_dimension": dimension,
        "realized_return": [0.0, 0.25, 0.5],
        "num_informative_pairs": 3,
        "pairwise_sign_accuracy": q_accuracy,
        "spearman": q_spearman,
        "behavior_proxy_num_informative_pairs": 3,
        "behavior_proxy_pairwise_sign_accuracy": 1.0 / 3.0,
        "policy_prior_num_informative_pairs": 3,
        "policy_prior_pairwise_sign_accuracy": 1.0 / 3.0,
    }


def _artifact(shuffle_mode, accuracy, spearman):
    records = [
        _record(seed, 30, dimension, accuracy, spearman)
        for seed in range(10, 14)
        for dimension in range(6)
    ]
    return {
        "status": "ok",
        "_path": f"/tmp/{shuffle_mode}.json",
        "target_estimator": "simulator_branch_monte_carlo",
        "continuation_policy": "frozen_independent_bc",
        "continuation_policy_value_beta": None,
        "source_snapshot": "/tmp/source.pkl",
        "dataset_cache": "/tmp/cache.npz",
        "intervention_horizon": 1,
        "candidate_mode": "sibling_bins",
        "critic_parameterization": "direct_scalar_q",
        "train_return_shuffle": shuffle_mode,
        "frozen_policy_bitwise_equal_after_fit": True,
        "num_informative_train_states": 24,
        "results": {
            "heldout_after": {
                "num_informative_states": 24,
                "records": records,
            }
        },
    }


def test_h1_cf_fqe_gate_passes_strong_causal_fit_and_matched_shuffle():
    positive = _artifact("none", 1.0, 1.0)
    shuffle = _artifact("within_state", 0.0, -1.0)

    payload = summarize(
        positive,
        shuffle,
        min_informative_states=24,
        min_eval_seeds=4,
        bootstrap_replicates=200,
        bootstrap_seed=7,
    )

    assert payload["gate"] == "pass"
    assert all(payload["gate_checks"].values())
    assert payload["heldout_q_minus_shuffle_pairwise"] == 1.0


def test_h1_cf_fqe_gate_rejects_h4_or_imitation_level_fit():
    positive = _artifact("none", 1.0 / 3.0, 0.0)
    positive["intervention_horizon"] = 4
    shuffle = _artifact("within_state", 1.0 / 3.0, 0.0)

    payload = summarize(
        positive,
        shuffle,
        min_informative_states=24,
        min_eval_seeds=4,
        bootstrap_replicates=100,
        bootstrap_seed=9,
    )

    assert payload["gate"] == "fail"
    assert not payload["gate_checks"]["one_step_intervention"]
    assert not payload["gate_checks"]["beats_independent_bc_proxy"]
    assert not payload["gate_checks"]["beats_within_state_shuffle_control"]


def test_h1_cf_fqe_gate_requires_identical_realized_returns():
    positive = _artifact("none", 1.0, 1.0)
    shuffle = _artifact("within_state", 0.0, -1.0)
    shuffle = copy.deepcopy(shuffle)
    shuffle["results"]["heldout_after"]["records"][0][
        "realized_return"
    ][0] = 0.1

    try:
        summarize(
            positive,
            shuffle,
            min_informative_states=24,
            min_eval_seeds=4,
            bootstrap_replicates=100,
            bootstrap_seed=11,
        )
    except ValueError as error:
        assert "realized returns differ" in str(error)
    else:
        raise AssertionError("mismatched held-out returns must be rejected")
