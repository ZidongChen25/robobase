from scripts.run_cqn_branch_cv_gate import (
    candidate_label,
    select_candidate,
    summarize_init_gate,
)


def _row(
    *,
    pairwise,
    spearman,
    top1=0.4,
    random=0.25,
    regret=0.05,
    before_regret=0.08,
    behavior_pairwise=0.54,
    behavior_top1=0.30,
    behavior_regret=0.07,
    policy_pairwise=0.53,
    policy_top1=0.31,
    policy_regret=0.075,
    updates=10,
    weight_decay=1e-3,
):
    return {
        "updates": updates,
        "weight_decay": weight_decay,
        "metrics": {
            "pairwise_sign_accuracy": pairwise,
            "mean_spearman": spearman,
            "top1_match_rate": top1,
            "random_top1_probability": random,
            "mean_realized_regret": regret,
            "before_mean_realized_regret": before_regret,
            "behavior_proxy_pairwise_sign_accuracy": behavior_pairwise,
            "behavior_proxy_top1_match_rate": behavior_top1,
            "behavior_proxy_mean_realized_regret": behavior_regret,
            "policy_prior_pairwise_sign_accuracy": policy_pairwise,
            "policy_prior_top1_match_rate": policy_top1,
            "policy_prior_mean_realized_regret": policy_regret,
        },
    }


def test_candidate_label_is_filesystem_stable():
    assert candidate_label(20, 1e-5) == "u20_wd1em05"
    assert candidate_label(5, 1e-2) == "u5_wd1em02"


def test_select_candidate_uses_fixed_lexicographic_protocol():
    rows = [
        _row(
            pairwise=0.56,
            spearman=0.20,
            updates=50,
            weight_decay=1e-5,
        ),
        _row(
            pairwise=0.57,
            spearman=0.10,
            updates=50,
            weight_decay=1e-5,
        ),
        _row(
            pairwise=0.57,
            spearman=0.20,
            regret=0.06,
            updates=50,
            weight_decay=1e-5,
        ),
        _row(
            pairwise=0.57,
            spearman=0.20,
            regret=0.05,
            updates=10,
            weight_decay=1e-2,
        ),
    ]

    assert select_candidate(rows) is rows[-1]


def test_initialization_gate_uses_medians_and_all_checks():
    passing = [
        _row(pairwise=0.56, spearman=0.11),
        _row(pairwise=0.58, spearman=0.20),
        _row(pairwise=0.54, spearman=0.09),
    ]
    result = summarize_init_gate(
        passing,
        min_pairwise=0.55,
        min_spearman=0.10,
    )
    assert result["gate"] == "pass"
    assert all(result["checks"].values())

    failing = [
        _row(
            pairwise=0.56,
            spearman=0.11,
            behavior_pairwise=0.57,
        ),
        _row(
            pairwise=0.56,
            spearman=0.20,
            behavior_pairwise=0.57,
        ),
        _row(
            pairwise=0.57,
            spearman=0.09,
            behavior_pairwise=0.58,
        ),
    ]
    result = summarize_init_gate(
        failing,
        min_pairwise=0.55,
        min_spearman=0.10,
    )
    assert result["gate"] == "fail"
    assert not result["checks"]["pairwise_above_nonreturn_proxies"]
