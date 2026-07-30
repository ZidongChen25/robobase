import json
from pathlib import Path

import pytest

from scripts.run_cqn_flow_branch_multiseed_gate import (
    POLICY_RNG_PROTOCOL,
    Checkpoint,
    _checkpoint,
    _probe_is_compatible,
    _resolve_continuation_beta,
    build_probe_command,
    summarize,
)


def test_checkpoint_parser_and_probe_command_freeze_readout():
    checkpoint = _checkpoint("seed1=run,snapshot.pkl")
    assert checkpoint == Checkpoint(
        "seed1", Path("run"), Path("snapshot.pkl")
    )
    with pytest.raises(Exception, match="checkpoint must be"):
        _checkpoint("seed1=run")

    command = build_probe_command(
        checkpoint,
        output=Path("probe.json"),
        gpu_id=5,
        eval_seeds=[94_000, 94_001],
        anchor_steps="30,75,120",
        force_level=1,
        intervention_mode="sibling_horizon",
        intervention_horizon=1,
        max_continuation_steps=300,
        flow_readout="distill",
        num_flow_steps=None,
        policy_value_beta=1.0,
        bootstrap_replicates=20_000,
        probe_seed=94_100,
    )
    assert command[command.index("--flow-readout") + 1] == "distill"
    assert command[command.index("--policy-value-beta") + 1] == "1"
    assert command[command.index("--eval-seeds") + 1] == "94000,94001"
    assert command[command.index("--intervention-horizon") + 1] == "1"
    assert command[command.index("--dimension-selection") + 1] == "q_span"

    integrated = build_probe_command(
        checkpoint,
        output=Path("integrated.json"),
        gpu_id=1,
        eval_seeds=[94_000],
        anchor_steps="30",
        force_level=1,
        intervention_mode="sibling_horizon",
        intervention_horizon=1,
        max_continuation_steps=300,
        flow_readout="integrated",
        num_flow_steps=8,
        policy_value_beta=1.0,
        bootstrap_replicates=100,
        probe_seed=94_100,
    )
    assert integrated[integrated.index("--num-flow-steps") + 1] == "8"

    truncated = build_probe_command(
        checkpoint,
        output=Path("truncated.json"),
        gpu_id=1,
        eval_seeds=[94_000],
        anchor_steps="30",
        force_level=1,
        intervention_mode="sibling_horizon",
        intervention_horizon=1,
        max_continuation_steps=300,
        flow_readout="integrated",
        num_flow_steps=10,
        policy_value_beta=0.3,
        bootstrap_replicates=100,
        probe_seed=94_100,
        return_sample_aggregation="truncated_mean",
        num_action_flow_samples=10,
        return_sample_truncate_top=1,
    )
    assert truncated[
        truncated.index("--return-sample-aggregation") + 1
    ] == "truncated_mean"
    assert truncated[
        truncated.index("--num-action-flow-samples") + 1
    ] == "10"
    assert truncated[
        truncated.index("--return-sample-truncate-top") + 1
    ] == "1"

    direct = build_probe_command(
        checkpoint,
        output=Path("direct.json"),
        gpu_id=1,
        eval_seeds=[94_000],
        anchor_steps="30",
        force_level=1,
        intervention_mode="sibling_horizon",
        intervention_horizon=1,
        max_continuation_steps=300,
        flow_readout="auto",
        num_flow_steps=None,
        policy_value_beta=1.0,
        bootstrap_replicates=100,
        probe_seed=94_100,
    )
    assert direct[direct.index("--flow-readout") + 1] == "auto"

    bc_continuation = build_probe_command(
        checkpoint,
        output=Path("bc_continuation.json"),
        gpu_id=1,
        eval_seeds=[94_000],
        anchor_steps="30",
        force_level=1,
        intervention_mode="sibling_horizon",
        intervention_horizon=1,
        max_continuation_steps=300,
        flow_readout="auto",
        num_flow_steps=None,
        policy_value_beta="bc",
        bootstrap_replicates=100,
        probe_seed=94_100,
    )
    assert bc_continuation[
        bc_continuation.index("--policy-value-beta") + 1
    ] == "bc"


def test_continuation_policy_separates_causal_estimand_from_deployment():
    command_beta, artifact_beta = _resolve_continuation_beta("bc", 0.3)
    assert command_beta == "bc"
    assert artifact_beta is None

    command_beta, artifact_beta = _resolve_continuation_beta(
        "deployment", 0.3
    )
    assert command_beta == pytest.approx(0.3)
    assert artifact_beta == pytest.approx(0.3)

    with pytest.raises(ValueError, match="unknown continuation policy"):
        _resolve_continuation_beta("invalid", 0.3)


def test_probe_cache_requires_common_per_eval_seed_rng(tmp_path):
    path = tmp_path / "probe.json"
    path.write_text(
        json.dumps(
            {
                "status": "ok",
                "policy_rng_protocol": "legacy_checkpoint_rng",
                "policy_value_beta": None,
                "dimension_selection": "round_robin",
                "intervention_horizon": 1,
                "records": [{"policy_path_proxy": {}}],
            }
        )
    )
    kwargs = {
        "intervention_horizon": 1,
        "require_anti_cheat_proxies": True,
        "expected_policy_value_beta": None,
        "dimension_selection": "round_robin",
    }

    assert not _probe_is_compatible(path, **kwargs)
    payload = json.loads(path.read_text())
    payload["policy_rng_protocol"] = POLICY_RNG_PROTOCOL
    path.write_text(json.dumps(payload))
    assert _probe_is_compatible(path, **kwargs)


def _write_probe(
    path,
    *,
    correct,
    policy_proxy_correct=None,
    path_proxy_correct=None,
    nearness_proxy_correct=None,
    dimension_selection="q_span",
    num_action_dimensions=None,
):
    records = []
    for seed in range(94_000, 94_004):
        record = {
            "eval_seed": seed,
            "realized_return_span": 1.0,
            "num_informative_pairs": 10,
            "pairwise_sign_accuracy": 1.0 if correct else 0.0,
            "spearman": 1.0 if correct else -1.0,
        }
        if num_action_dimensions is not None:
            record["action_dimension"] = (
                seed - 94_000
            ) % num_action_dimensions
        for name, proxy_correct in (
            ("policy_prior", policy_proxy_correct),
            ("policy_path", path_proxy_correct),
            ("action_nearness", nearness_proxy_correct),
        ):
            if proxy_correct is not None:
                record[f"{name}_proxy"] = {
                    "num_informative_pairs": 10,
                    "pairwise_sign_accuracy": (
                        1.0 if proxy_correct else 0.0
                    ),
                }
        records.append(record)
    pairwise = 1.0 if correct else 0.0
    spearman = 1.0 if correct else -1.0
    payload = {
        "status": "ok",
        "eval_seeds": list(range(94_000, 94_004)),
        "value_readout": "distill",
        "num_flow_steps": None,
        "policy_value_beta": 1.0,
        "intervention_horizon": 1,
        "dimension_selection": dimension_selection,
        "num_action_dimensions": num_action_dimensions,
        "informative_states_per_dimension": (
            {
                str(dimension): sum(
                    record.get("action_dimension") == dimension
                    for record in records
                )
                for dimension in range(num_action_dimensions)
            }
            if num_action_dimensions is not None
            else {}
        ),
        "num_states": 4,
        "num_informative_states": 4,
        "pairwise_sign_accuracy": pairwise,
        "mean_spearman": spearman,
        "top1_match_rate": pairwise,
        "mean_realized_regret": 1.0 - pairwise,
        "state_bootstrap": {
            "pairwise_sign_accuracy_ci": [pairwise, pairwise],
            "mean_spearman_ci": [spearman, spearman],
        },
        "records": records,
    }
    path.write_text(json.dumps(payload))


def test_multiseed_causal_summary_uses_crossed_seed_gate(tmp_path):
    paths = []
    for index, correct in enumerate((True, True, False), start=1):
        path = tmp_path / f"seed{index}.json"
        _write_probe(path, correct=correct)
        paths.append((f"seed{index}", path))

    summary = summarize(
        paths,
        bootstrap_replicates=1_000,
        bootstrap_seed=7,
        min_informative_states=4,
        required_positive_training_seeds=2,
    )

    assert summary["positive_training_seeds"] == 2
    assert summary["aggregate_pairwise_sign_accuracy"] == pytest.approx(
        2.0 / 3.0
    )
    assert summary["gate"] == "fail"
    assert not summary["gate_checks"][
        "aggregate_causal_ci_strictly_positive"
    ]


def test_multiseed_causal_summary_passes_when_all_seeds_are_positive(tmp_path):
    paths = []
    for index in range(3):
        path = tmp_path / f"seed{index}.json"
        _write_probe(path, correct=True)
        paths.append((f"seed{index}", path))

    summary = summarize(
        paths,
        bootstrap_replicates=100,
        bootstrap_seed=11,
        min_informative_states=4,
        required_positive_training_seeds=2,
    )

    assert summary["gate"] == "pass"


def test_anti_cheat_gate_requires_q_to_beat_both_proxies(tmp_path):
    paths = []
    for index in range(3):
        path = tmp_path / f"seed{index}.json"
        _write_probe(
            path,
            correct=True,
            policy_proxy_correct=False,
            path_proxy_correct=False,
            nearness_proxy_correct=False,
        )
        paths.append((f"seed{index}", path))

    summary = summarize(
        paths,
        bootstrap_replicates=100,
        bootstrap_seed=13,
        min_informative_states=4,
        required_positive_training_seeds=2,
        require_anti_cheat_proxies=True,
    )

    assert summary["gate"] == "pass"
    assert summary["aggregate_q_minus_proxy_pairwise_ci"][
        "policy_prior"
    ] == pytest.approx([1.0, 1.0])
    assert summary["aggregate_q_minus_proxy_pairwise_ci"][
        "policy_path"
    ] == pytest.approx([1.0, 1.0])
    assert summary["aggregate_q_minus_proxy_pairwise_ci"][
        "action_nearness"
    ] == pytest.approx([1.0, 1.0])


def test_anti_cheat_gate_rejects_q_that_only_matches_policy_prior(tmp_path):
    paths = []
    for index in range(3):
        path = tmp_path / f"seed{index}.json"
        _write_probe(
            path,
            correct=True,
            policy_proxy_correct=True,
            path_proxy_correct=False,
            nearness_proxy_correct=False,
        )
        paths.append((f"seed{index}", path))

    summary = summarize(
        paths,
        bootstrap_replicates=100,
        bootstrap_seed=17,
        min_informative_states=4,
        required_positive_training_seeds=2,
        require_anti_cheat_proxies=True,
    )

    assert summary["gate"] == "fail"
    assert not summary["gate_checks"][
        "q_pairwise_above_policy_prior_proxy_ci"
    ]


def test_anti_cheat_gate_rejects_q_that_matches_full_bc_path(tmp_path):
    paths = []
    for index in range(3):
        path = tmp_path / f"seed{index}.json"
        _write_probe(
            path,
            correct=True,
            policy_proxy_correct=False,
            path_proxy_correct=True,
            nearness_proxy_correct=False,
        )
        paths.append((f"seed{index}", path))

    summary = summarize(
        paths,
        bootstrap_replicates=100,
        bootstrap_seed=19,
        min_informative_states=4,
        required_positive_training_seeds=2,
        require_anti_cheat_proxies=True,
    )

    assert summary["gate"] == "fail"
    assert not summary["gate_checks"][
        "q_pairwise_above_policy_path_proxy_ci"
    ]


def test_unbiased_dimension_gate_requires_round_robin_coverage(tmp_path):
    paths = []
    for index in range(3):
        path = tmp_path / f"seed{index}.json"
        _write_probe(
            path,
            correct=True,
            dimension_selection="round_robin",
            num_action_dimensions=2,
        )
        paths.append((f"seed{index}", path))

    summary = summarize(
        paths,
        bootstrap_replicates=100,
        bootstrap_seed=23,
        min_informative_states=4,
        required_positive_training_seeds=2,
        min_informative_dimensions=2,
        min_informative_states_per_dimension=2,
    )

    assert summary["gate"] == "pass"
    assert summary["dimension_selection"] == "round_robin"
    assert summary["gate_checks"][
        "dimension_selection_is_value_independent"
    ]
    assert summary["gate_checks"][
        "informative_dimension_coverage_per_training_seed"
    ]


def test_unbiased_dimension_gate_rejects_q_selected_dimension(tmp_path):
    paths = []
    for index in range(3):
        path = tmp_path / f"seed{index}.json"
        _write_probe(
            path,
            correct=True,
            dimension_selection="q_span",
            num_action_dimensions=2,
        )
        paths.append((f"seed{index}", path))

    with pytest.raises(ValueError, match="round_robin"):
        summarize(
            paths,
            bootstrap_replicates=100,
            bootstrap_seed=29,
            min_informative_states=4,
            required_positive_training_seeds=2,
            min_informative_dimensions=2,
            min_informative_states_per_dimension=2,
        )
