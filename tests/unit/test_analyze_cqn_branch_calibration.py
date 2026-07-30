import json
from pathlib import Path

import pytest

from scripts.analyze_cqn_branch_calibration import (
    Probe,
    analyze,
    probes_from_causal_summary,
)


def _write_probe(
    path: Path,
    *,
    q_scale: float = 1.0,
    include_proxies: bool = True,
) -> None:
    eval_seeds = list(range(209_000, 209_008))
    records = []
    returns = [-2.0, -1.0, 0.0, 1.0, 2.0]
    proxy = [1.0, -1.0, 0.0, -1.0, 1.0]
    for eval_seed in eval_seeds:
        outcomes = []
        for bin_index, realised in enumerate(returns):
            outcome = {
                "bin": bin_index,
                "predicted_q": q_scale * realised,
                "discounted_return": realised,
            }
            if include_proxies:
                outcome.update(
                    {
                        "policy_prior_score": proxy[bin_index],
                        "policy_path_score": proxy[bin_index],
                        "action_nearness_score": proxy[bin_index],
                    }
                )
            outcomes.append(outcome)
        records.append(
            {
                "eval_seed": eval_seed,
                "anchor_step": 30,
                "action_dimension": (eval_seed - eval_seeds[0]) % 16,
                "realized_return_span": 4.0,
                "outcomes": outcomes,
            }
        )
    path.write_text(
        json.dumps(
            {
                "status": "ok",
                "eval_seeds": eval_seeds,
                "intervention_horizon": 1,
                "dimension_selection": "round_robin",
                "policy_value_beta": None,
                "policy_rng_protocol": (
                    "common_prngkey_probe_seed_plus_eval_seed"
                ),
                "num_states": len(records),
                "num_informative_states": len(records),
                "records": records,
            }
        )
    )


def _analyze(paths: list[Path], **overrides):
    kwargs = {
        "bootstrap_replicates": 500,
        "bootstrap_seed": 7,
        "permutation_replicates": 100,
        "permutation_seed": 8,
        "min_training_seeds": len(paths),
        "min_informative_states_per_split": 4,
        "native_slope_lower": 0.5,
        "native_slope_upper": 2.0,
        "strict_protocol": True,
    }
    kwargs.update(overrides)
    return analyze(
        [
            Probe(label=f"seed{index}", path=path)
            for index, path in enumerate(paths, start=1)
        ],
        **kwargs,
    )


def test_formal_gate_passes_calibrated_q_and_beats_imitation_placebos(
    tmp_path,
):
    paths = []
    for index in range(3):
        path = tmp_path / f"probe{index}.json"
        _write_probe(path)
        paths.append(path)

    result = _analyze(paths)

    assert result["gate"] == "pass"
    assert result["route_a_claim_allowed"]
    assert result["calibration_seeds"] == list(range(209_000, 209_004))
    assert result["heldout_seeds"] == list(range(209_004, 209_008))
    assert result["metrics"]["native_heldout_slope"] == pytest.approx(1.0)
    assert result["metrics"]["native_mse_skill"] == pytest.approx(1.0)
    assert result["crossed_bootstrap"][
        "native_heldout_slope_ci"
    ] == pytest.approx([1.0, 1.0])
    assert result["bin_permutation_placebo"][
        "p_value_placebo_at_least_as_good"
    ] <= 0.05
    assert all(result["gate_checks"].values())


def test_recoverable_ranking_does_not_substitute_for_native_return_units(
    tmp_path,
):
    paths = []
    for index in range(3):
        path = tmp_path / f"probe{index}.json"
        _write_probe(path, q_scale=0.1)
        paths.append(path)

    result = _analyze(paths)

    assert result["metrics"]["q_calibration_slope"] == pytest.approx(10.0)
    assert result["metrics"]["recalibrated_q_mse_skill"] == pytest.approx(
        1.0
    )
    assert not result["gate_checks"][
        "native_q_slope_ci_inside_equivalence_interval"
    ]
    assert result["gate"] == "fail"


def test_formal_gate_rejects_missing_anti_imitation_scores(tmp_path):
    path = tmp_path / "probe.json"
    _write_probe(path, include_proxies=False)

    with pytest.raises(ValueError, match="missing policy_prior_score"):
        _analyze([path])


def test_causal_summary_resolves_exact_raw_probe_sources(tmp_path):
    probe = tmp_path / "probe.json"
    _write_probe(probe)
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "status": "ok",
                "sources": {"seed1": str(probe)},
            }
        )
    )

    assert probes_from_causal_summary(summary) == [
        Probe("seed1", probe)
    ]
