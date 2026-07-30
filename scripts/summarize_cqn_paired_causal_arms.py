#!/usr/bin/env python3
"""Compare matched frozen-policy control/RCT causal branch probes.

Every pair must use the same clean behavior policy, simulator seeds, H=1
round-robin intervention, action bins, and independent-BC continuation.
Control and treatment are bootstrapped as a pair over both training seeds and
simulator seeds; sibling actions from one restored state are never treated as
independent samples.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PROXY_NAMES = ("policy_prior", "policy_path", "action_nearness")
POLICY_RNG_PROTOCOL = "common_prngkey_probe_seed_plus_eval_seed"


@dataclass(frozen=True)
class ArmPair:
    label: str
    control: Path
    treatment: Path


def _arm_pair(value: str) -> ArmPair:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "pair must be LABEL=CONTROL_PROBE,TREATMENT_PROBE"
        )
    label, paths = value.split("=", 1)
    split = paths.split(",")
    if not label or len(split) != 2 or not all(split):
        raise argparse.ArgumentTypeError(
            "pair must be LABEL=CONTROL_PROBE,TREATMENT_PROBE"
        )
    return ArmPair(label, Path(split[0]), Path(split[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", action="append", required=True, type=_arm_pair)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=214_200)
    parser.add_argument("--min-training-seeds", type=int, default=3)
    parser.add_argument("--min-eval-seeds", type=int, default=16)
    parser.add_argument("--min-informative-states", type=int, default=24)
    parser.add_argument(
        "--required-positive-training-seeds",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--strict-ci",
        action="store_true",
        help=(
            "Require crossed-bootstrap causal and treatment-over-control "
            "confidence bounds. Omit only for explicitly labeled discovery."
        ),
    )
    return parser.parse_args()


def _load(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text())
    if payload.get("status") != "ok":
        raise ValueError(f"probe is not complete: {resolved}")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"probe has no branch records: {resolved}")
    return resolved, payload


def _record_key(record: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(record["eval_seed"]),
        int(record["anchor_step"]),
        int(record["action_dimension"]),
    )


def _by_key(payload: dict[str, Any]) -> dict[tuple[int, int, int], dict]:
    records = {_record_key(record): record for record in payload["records"]}
    if len(records) != len(payload["records"]):
        raise ValueError("probe contains duplicate branch-state keys")
    return records


def _outcome_signature(outcome: dict[str, Any]) -> tuple[Any, ...]:
    keys = (
        "bin",
        "discounted_return",
        "rollout_length",
        "success",
        "terminated",
        "truncated",
        "raw_forced_action",
        "effective_forced_action",
        "intervention_delta",
        "intervention_horizon",
    )
    return tuple(outcome.get(key) for key in keys)


def _centred_outcomes(
    record: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    outcomes = record.get("outcomes")
    if not isinstance(outcomes, list) or len(outcomes) < 2:
        raise ValueError("branch state needs at least two candidate outcomes")
    q = np.asarray(
        [item["predicted_q"] for item in outcomes],
        dtype=np.float64,
    )
    realised = np.asarray(
        [item["discounted_return"] for item in outcomes],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(realised)):
        raise ValueError("branch outcomes contain non-finite Q/return")
    return q - q.mean(), realised - realised.mean()


def _interval(values: np.ndarray) -> list[float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return [float("nan"), float("nan")]
    return [
        float(np.quantile(finite, 0.025)),
        float(np.quantile(finite, 0.975)),
    ]


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator > 0.0,
    )


def _aggregate_metrics(
    stats: dict[str, np.ndarray],
    model_indices: np.ndarray | None = None,
    seed_indices: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    if model_indices is None or seed_indices is None:
        selected = {key: value.sum() for key, value in stats.items()}
    else:
        selected = {
            key: value[
                model_indices[:, :, None],
                seed_indices[:, None, :],
            ].sum(axis=(1, 2))
            for key, value in stats.items()
        }
    control_pairwise = _ratio(
        np.asarray(selected["control_pair_correct"]),
        np.asarray(selected["pair_count"]),
    )
    treatment_pairwise = _ratio(
        np.asarray(selected["treatment_pair_correct"]),
        np.asarray(selected["pair_count"]),
    )
    control_spearman = _ratio(
        np.asarray(selected["control_spearman_sum"]),
        np.asarray(selected["spearman_count"]),
    )
    treatment_spearman = _ratio(
        np.asarray(selected["treatment_spearman_sum"]),
        np.asarray(selected["spearman_count"]),
    )
    control_mse = _ratio(
        np.asarray(selected["control_squared_error"]),
        np.asarray(selected["contrast_count"]),
    )
    treatment_mse = _ratio(
        np.asarray(selected["treatment_squared_error"]),
        np.asarray(selected["contrast_count"]),
    )
    metrics: dict[str, np.ndarray | float] = {
        "control_pairwise": control_pairwise,
        "treatment_pairwise": treatment_pairwise,
        "treatment_minus_control_pairwise": (
            treatment_pairwise - control_pairwise
        ),
        "control_spearman": control_spearman,
        "treatment_spearman": treatment_spearman,
        "treatment_minus_control_spearman": (
            treatment_spearman - control_spearman
        ),
        "control_native_mse": control_mse,
        "treatment_native_mse": treatment_mse,
        "control_minus_treatment_native_mse": (
            control_mse - treatment_mse
        ),
    }
    for name in PROXY_NAMES:
        proxy = _ratio(
            np.asarray(selected[f"{name}_pair_correct"]),
            np.asarray(selected[f"{name}_pair_count"]),
        )
        metrics[f"{name}_pairwise"] = proxy
        metrics[f"treatment_minus_{name}_pairwise"] = (
            treatment_pairwise - proxy
        )
    return metrics


def summarize(
    pairs: list[ArmPair],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    min_training_seeds: int,
    min_eval_seeds: int,
    min_informative_states: int,
    required_positive_training_seeds: int,
    strict_ci: bool,
) -> dict[str, Any]:
    if not pairs or len({pair.label for pair in pairs}) != len(pairs):
        raise ValueError("arm-pair labels must be non-empty and unique")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    if not 1 <= required_positive_training_seeds <= len(pairs):
        raise ValueError("invalid positive-training-seed requirement")

    loaded = []
    reference_eval_seeds: list[int] | None = None
    reference_num_dimensions: int | None = None
    sources = {}
    for pair in pairs:
        control_path, control = _load(pair.control)
        treatment_path, treatment = _load(pair.treatment)
        for name, payload in (("control", control), ("treatment", treatment)):
            checks = {
                "H1": payload.get("intervention_horizon") == 1,
                "round_robin": (
                    payload.get("dimension_selection") == "round_robin"
                ),
                "independent_BC": payload.get("policy_value_beta") is None,
                "direct_scalar_q": (
                    payload.get("value_readout") == "direct_scalar_q"
                ),
                "common_policy_rng": (
                    payload.get("policy_rng_protocol")
                    == POLICY_RNG_PROTOCOL
                ),
            }
            if not all(checks.values()):
                failed = [key for key, value in checks.items() if not value]
                raise ValueError(
                    f"{pair.label} {name} violates causal protocol: {failed}"
                )
        eval_seeds = [int(seed) for seed in control["eval_seeds"]]
        if eval_seeds != [int(seed) for seed in treatment["eval_seeds"]]:
            raise ValueError(f"{pair.label} arm eval seeds differ")
        if reference_eval_seeds is None:
            reference_eval_seeds = eval_seeds
            reference_num_dimensions = int(
                control["num_action_dimensions"]
            )
        elif eval_seeds != reference_eval_seeds:
            raise ValueError("training-seed pairs do not share eval seeds")
        if int(control["num_action_dimensions"]) != reference_num_dimensions:
            raise ValueError("control action dimensionality differs")
        if int(treatment["num_action_dimensions"]) != reference_num_dimensions:
            raise ValueError("treatment action dimensionality differs")
        control_by_key = _by_key(control)
        treatment_by_key = _by_key(treatment)
        if control_by_key.keys() != treatment_by_key.keys():
            raise ValueError(f"{pair.label} branch-state keys differ")
        for key in control_by_key:
            left = control_by_key[key].get("outcomes", [])
            right = treatment_by_key[key].get("outcomes", [])
            if [_outcome_signature(item) for item in left] != [
                _outcome_signature(item) for item in right
            ]:
                raise ValueError(
                    f"{pair.label} counterfactual outcomes differ at {key}"
                )
            for proxy in PROXY_NAMES:
                if control_by_key[key].get(f"{proxy}_proxy") != (
                    treatment_by_key[key].get(f"{proxy}_proxy")
                ):
                    raise ValueError(
                        f"{pair.label} {proxy} proxy differs at {key}"
                    )
        sources[pair.label] = {
            "control": str(control_path),
            "treatment": str(treatment_path),
        }
        loaded.append((pair.label, control_by_key, treatment_by_key))

    assert reference_eval_seeds is not None
    shape = (len(loaded), len(reference_eval_seeds))
    stats = {
        key: np.zeros(shape, dtype=np.float64)
        for key in (
            "control_pair_correct",
            "treatment_pair_correct",
            "pair_count",
            "control_spearman_sum",
            "treatment_spearman_sum",
            "spearman_count",
            "control_squared_error",
            "treatment_squared_error",
            "contrast_count",
            *(
                item
                for name in PROXY_NAMES
                for item in (
                    f"{name}_pair_correct",
                    f"{name}_pair_count",
                )
            ),
        )
    }
    seed_index = {
        seed: index for index, seed in enumerate(reference_eval_seeds)
    }
    informative_per_model = np.zeros(len(loaded), dtype=np.int64)
    per_training_seed = {}
    positive_training_seeds = 0
    for model_index, (label, control, treatment) in enumerate(loaded):
        for key in sorted(control):
            eval_index = seed_index[key[0]]
            control_record = control[key]
            treatment_record = treatment[key]
            pair_count = float(control_record["num_informative_pairs"])
            if pair_count > 0.0:
                stats["pair_count"][model_index, eval_index] += pair_count
                stats["control_pair_correct"][model_index, eval_index] += (
                    pair_count
                    * float(control_record["pairwise_sign_accuracy"])
                )
                stats["treatment_pair_correct"][model_index, eval_index] += (
                    pair_count
                    * float(treatment_record["pairwise_sign_accuracy"])
                )
                informative_per_model[model_index] += 1
            for prefix, record in (
                ("control", control_record),
                ("treatment", treatment_record),
            ):
                spearman = float(record["spearman"])
                if math.isfinite(spearman):
                    stats[f"{prefix}_spearman_sum"][
                        model_index,
                        eval_index,
                    ] += spearman
            if math.isfinite(float(control_record["spearman"])) and math.isfinite(
                float(treatment_record["spearman"])
            ):
                stats["spearman_count"][model_index, eval_index] += 1.0
            control_q, realised = _centred_outcomes(control_record)
            treatment_q, treatment_realised = _centred_outcomes(
                treatment_record
            )
            if not np.array_equal(realised, treatment_realised):
                raise ValueError(f"{label} centred returns differ at {key}")
            stats["control_squared_error"][model_index, eval_index] += float(
                np.sum(np.square(control_q - realised))
            )
            stats["treatment_squared_error"][model_index, eval_index] += float(
                np.sum(np.square(treatment_q - realised))
            )
            stats["contrast_count"][model_index, eval_index] += realised.size
            for name in PROXY_NAMES:
                proxy = treatment_record.get(f"{name}_proxy")
                if not isinstance(proxy, dict):
                    raise ValueError(f"{label} is missing {name} proxy")
                count = float(proxy["num_informative_pairs"])
                stats[f"{name}_pair_count"][
                    model_index,
                    eval_index,
                ] += count
                stats[f"{name}_pair_correct"][
                    model_index,
                    eval_index,
                ] += count * float(proxy["pairwise_sign_accuracy"])

        one = {key: value[model_index : model_index + 1] for key, value in stats.items()}
        metrics = _aggregate_metrics(one)
        positive = float(
            metrics["treatment_minus_control_pairwise"]
        ) > 0.0
        positive_training_seeds += int(positive)
        per_training_seed[label] = {
            "informative_states": int(informative_per_model[model_index]),
            **{key: float(value) for key, value in metrics.items()},
            "treatment_pairwise_improves_control": positive,
        }

    point = {
        key: float(value)
        for key, value in _aggregate_metrics(stats).items()
    }
    rng = np.random.default_rng(int(bootstrap_seed))
    model_draws = rng.integers(
        0,
        len(loaded),
        size=(bootstrap_replicates, len(loaded)),
    )
    seed_draws = rng.integers(
        0,
        len(reference_eval_seeds),
        size=(bootstrap_replicates, len(reference_eval_seeds)),
    )
    samples = _aggregate_metrics(stats, model_draws, seed_draws)
    ci95 = {
        key: _interval(np.asarray(value, dtype=np.float64))
        for key, value in samples.items()
    }
    causal_direction = (
        point["treatment_pairwise"] > 0.5
        or point["treatment_spearman"] > 0.0
    )
    beats_proxies = all(
        point[f"treatment_minus_{name}_pairwise"] > 0.0
        for name in PROXY_NAMES
    )
    checks = {
        "minimum_training_seed_coverage": (
            len(loaded) >= int(min_training_seeds)
        ),
        "minimum_eval_seed_coverage": (
            len(reference_eval_seeds) >= int(min_eval_seeds)
        ),
        "informative_coverage_per_training_seed": bool(
            np.all(informative_per_model >= int(min_informative_states))
        ),
        "matched_counterfactual_outcomes_exact": True,
        "treatment_point_causal_direction_positive": causal_direction,
        "treatment_point_improves_control_pairwise": (
            point["treatment_minus_control_pairwise"] > 0.0
        ),
        "treatment_point_beats_all_imitation_proxies": beats_proxies,
        "positive_training_seed_requirement": (
            positive_training_seeds
            >= int(required_positive_training_seeds)
        ),
    }
    if strict_ci:
        checks.update(
            {
                "treatment_causal_ci_strictly_positive": (
                    ci95["treatment_pairwise"][0] > 0.5
                    or ci95["treatment_spearman"][0] > 0.0
                ),
                "treatment_improves_control_pairwise_ci": (
                    ci95["treatment_minus_control_pairwise"][0] > 0.0
                ),
                **{
                    f"treatment_beats_{name}_proxy_ci": (
                        ci95[f"treatment_minus_{name}_pairwise"][0] > 0.0
                    )
                    for name in PROXY_NAMES
                },
            }
        )
    return {
        "status": "ok",
        "claim_scope": (
            "formal_matched_frozen_policy_rct_effect"
            if strict_ci
            else "seed1_discovery_only"
        ),
        "selection_use_forbidden": True,
        "num_training_seeds": len(loaded),
        "num_eval_seeds": len(reference_eval_seeds),
        "eval_seed_start": reference_eval_seeds[0],
        "eval_seed_end": reference_eval_seeds[-1],
        "num_action_dimensions": reference_num_dimensions,
        "sources": sources,
        "per_training_seed": per_training_seed,
        "positive_training_seeds": positive_training_seeds,
        "point": point,
        "crossed_bootstrap_ci95": ci95,
        "bootstrap": {
            "unit": "matched_training_seed_x_simulator_seed",
            "replicates": int(bootstrap_replicates),
            "seed": int(bootstrap_seed),
        },
        "thresholds": {
            "min_training_seeds": int(min_training_seeds),
            "min_eval_seeds": int(min_eval_seeds),
            "min_informative_states": int(min_informative_states),
            "required_positive_training_seeds": int(
                required_positive_training_seeds
            ),
            "strict_ci": bool(strict_ci),
        },
        "gate_checks": checks,
        "gate": "pass" if all(checks.values()) else "fail",
    }


def main() -> int:
    args = parse_args()
    result = summarize(
        args.pair,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        min_training_seeds=args.min_training_seeds,
        min_eval_seeds=args.min_eval_seeds,
        min_informative_states=args.min_informative_states,
        required_positive_training_seeds=(
            args.required_positive_training_seeds
        ),
        strict_ci=bool(args.strict_ci),
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
