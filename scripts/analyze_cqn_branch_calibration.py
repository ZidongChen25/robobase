#!/usr/bin/env python3
"""Audit numerical calibration of CQN action-bin values on held-out branches.

The existing branch counterfactual gate tests ordering.  A monotone rescaling
can pass that gate while no longer having return units.  This audit therefore
centres both predicted Q and realised discounted return within each same-state
set of action bins and evaluates the resulting action contrasts.

The ordered simulator seeds are split once: the first half is calibration-only
and the second half is sealed held-out evaluation.  A crossed bootstrap
resamples training checkpoints and simulator seeds.  Imitation scores are
allowed an affine-through-origin calibration on the calibration split, so Q
does not win merely because its native units happen to resemble return units.
"""

from __future__ import annotations

import argparse
import json
import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROXY_KEYS = {
    "policy_prior": "policy_prior_score",
    "policy_path": "policy_path_score",
    "action_nearness": "action_nearness_score",
}
POLICY_RNG_PROTOCOL = "common_prngkey_probe_seed_plus_eval_seed"


@dataclass(frozen=True)
class Probe:
    label: str
    path: Path


@dataclass(frozen=True)
class StateContrasts:
    model_index: int
    eval_seed: int
    predicted_q: np.ndarray
    realised_return: np.ndarray
    proxies: dict[str, np.ndarray]


def _probe(value: str) -> Probe:
    if "=" not in value:
        raise argparse.ArgumentTypeError("probe must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("probe must be LABEL=PATH")
    return Probe(label=label, path=Path(raw_path))


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def _finite_nonnegative(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise argparse.ArgumentTypeError(
            "expected a finite non-negative number"
        )
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--probe", action="append", type=_probe)
    source.add_argument("--causal-summary", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=_positive_int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=209_200)
    parser.add_argument(
        "--permutation-replicates", type=_positive_int, default=2_000
    )
    parser.add_argument("--permutation-seed", type=int, default=209_201)
    parser.add_argument("--min-training-seeds", type=_positive_int, default=3)
    parser.add_argument(
        "--min-informative-states-per-split",
        type=_positive_int,
        default=12,
    )
    parser.add_argument(
        "--native-slope-lower",
        type=_finite_nonnegative,
        default=0.5,
    )
    parser.add_argument(
        "--native-slope-upper",
        type=_finite_nonnegative,
        default=2.0,
    )
    parser.add_argument(
        "--allow-legacy-discovery",
        action="store_true",
        help=(
            "Allow non-H1/non-round-robin probes and missing imitation "
            "proxies. The resulting artifact is explicitly discovery-only."
        ),
    )
    return parser.parse_args()


def probes_from_causal_summary(path: Path) -> list[Probe]:
    payload = json.loads(path.expanduser().resolve().read_text())
    if payload.get("status") != "ok":
        raise ValueError(f"causal summary is not complete: {path}")
    sources = payload.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("causal summary does not contain probe sources")
    return [
        Probe(str(label), Path(raw_path))
        for label, raw_path in sources.items()
    ]


def _centred(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError("each branch state needs at least two finite outcomes")
    return array - float(array.mean())


def _load_probes(
    probes: list[Probe],
    *,
    strict_protocol: bool,
) -> tuple[
    list[str],
    list[int],
    list[StateContrasts],
    dict[str, dict[str, Any]],
]:
    if not probes:
        raise ValueError("at least one probe is required")
    if len({item.label for item in probes}) != len(probes):
        raise ValueError("probe labels must be unique")

    labels: list[str] = []
    reference_eval_seeds: list[int] | None = None
    states: list[StateContrasts] = []
    metadata: dict[str, dict[str, Any]] = {}
    for model_index, probe in enumerate(probes):
        path = probe.path.expanduser().resolve()
        payload = json.loads(path.read_text())
        if payload.get("status") != "ok":
            raise ValueError(f"probe is not complete: {path}")
        eval_seeds = [int(item) for item in payload.get("eval_seeds", [])]
        if len(eval_seeds) < 4 or len(eval_seeds) % 2:
            raise ValueError(
                "probe needs an even number of at least four eval seeds"
            )
        if len(set(eval_seeds)) != len(eval_seeds):
            raise ValueError("eval seeds must be unique")
        if reference_eval_seeds is None:
            reference_eval_seeds = eval_seeds
        elif eval_seeds != reference_eval_seeds:
            raise ValueError("all probes must share ordered eval seeds")

        if strict_protocol:
            checks = {
                "intervention_horizon_is_one": (
                    payload.get("intervention_horizon") == 1
                ),
                "dimension_selection_is_round_robin": (
                    payload.get("dimension_selection") == "round_robin"
                ),
                "independent_bc_continuation": (
                    payload.get("policy_value_beta") is None
                ),
                "common_policy_rng": (
                    payload.get("policy_rng_protocol")
                    == POLICY_RNG_PROTOCOL
                ),
            }
            if not all(checks.values()):
                failed = [name for name, value in checks.items() if not value]
                raise ValueError(
                    "formal calibration requires H1, round_robin, and "
                    f"independent BC continuation; failed={failed}"
                )

        records = payload.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError(f"probe has no branch records: {path}")
        seen_state_keys: set[tuple[int, int, int]] = set()
        for record in records:
            eval_seed = int(record["eval_seed"])
            if eval_seed not in eval_seeds:
                raise ValueError("record eval seed is outside probe seed set")
            state_key = (
                eval_seed,
                int(record["anchor_step"]),
                int(record["action_dimension"]),
            )
            if state_key in seen_state_keys:
                raise ValueError(f"duplicate branch state: {state_key}")
            seen_state_keys.add(state_key)
            outcomes = record.get("outcomes")
            if not isinstance(outcomes, list) or len(outcomes) < 2:
                raise ValueError("branch record has fewer than two outcomes")
            q = _centred(item["predicted_q"] for item in outcomes)
            realised = _centred(
                item["discounted_return"] for item in outcomes
            )
            if q.shape != realised.shape:
                raise ValueError("Q and return candidate counts do not match")
            proxies: dict[str, np.ndarray] = {}
            for name, key in PROXY_KEYS.items():
                if all(key in item for item in outcomes):
                    proxy = _centred(item[key] for item in outcomes)
                    if proxy.shape != q.shape:
                        raise ValueError(
                            f"{name} candidate count does not match Q"
                        )
                    proxies[name] = proxy
                elif strict_protocol:
                    raise ValueError(
                        f"formal calibration probe is missing {key}"
                    )
            states.append(
                StateContrasts(
                    model_index=model_index,
                    eval_seed=eval_seed,
                    predicted_q=q,
                    realised_return=realised,
                    proxies=proxies,
                )
            )
        labels.append(probe.label)
        metadata[probe.label] = {
            "source": str(path),
            "num_states": len(records),
            "num_informative_states": int(
                sum(
                    float(record.get("realized_return_span", 0.0)) > 0.0
                    for record in records
                )
            ),
            "intervention_horizon": payload.get("intervention_horizon"),
            "dimension_selection": payload.get("dimension_selection"),
            "policy_value_beta": payload.get("policy_value_beta"),
        }

    assert reference_eval_seeds is not None
    return labels, reference_eval_seeds, states, metadata


def _sufficient_statistics(
    labels: list[str],
    eval_seeds: list[int],
    states: list[StateContrasts],
    proxy_names: list[str],
) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    # Features: n, y2, q2, qy, then proxy2/proxy_y pairs.
    feature_index = {"n": 0, "y2": 1, "q2": 2, "qy": 3}
    next_index = 4
    for name in proxy_names:
        feature_index[f"{name}2"] = next_index
        feature_index[f"{name}_y"] = next_index + 1
        next_index += 2
    stats = np.zeros(
        (len(labels), len(eval_seeds), next_index), dtype=np.float64
    )
    informative = np.zeros(
        (len(labels), len(eval_seeds)), dtype=np.int64
    )
    seed_index = {seed: index for index, seed in enumerate(eval_seeds)}
    for state in states:
        row = stats[state.model_index, seed_index[state.eval_seed]]
        y = state.realised_return
        q = state.predicted_q
        row[feature_index["n"]] += y.size
        row[feature_index["y2"]] += float(np.dot(y, y))
        row[feature_index["q2"]] += float(np.dot(q, q))
        row[feature_index["qy"]] += float(np.dot(q, y))
        informative[state.model_index, seed_index[state.eval_seed]] += int(
            np.ptp(y) > 0.0
        )
        for name in proxy_names:
            proxy = state.proxies[name]
            row[feature_index[f"{name}2"]] += float(
                np.dot(proxy, proxy)
            )
            row[feature_index[f"{name}_y"]] += float(
                np.dot(proxy, y)
            )
    return stats, feature_index, informative


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator > 1e-18,
    )


def _mse_from_stats(
    stats: np.ndarray,
    *,
    slope: np.ndarray,
    x2_index: int,
    xy_index: int,
    feature_index: dict[str, int],
) -> np.ndarray:
    n = stats[..., feature_index["n"]]
    y2 = stats[..., feature_index["y2"]]
    x2 = stats[..., x2_index]
    xy = stats[..., xy_index]
    return _safe_ratio(
        slope * slope * x2 - 2.0 * slope * xy + y2,
        n,
    )


def _interval(values: np.ndarray) -> list[float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return [float("nan"), float("nan")]
    return [
        float(np.percentile(finite, 2.5)),
        float(np.percentile(finite, 97.5)),
    ]


def _metric_bundle(
    calibration: np.ndarray,
    heldout: np.ndarray,
    feature_index: dict[str, int],
    proxy_names: list[str],
) -> dict[str, Any]:
    q_slope = float(
        _safe_ratio(
            np.asarray(calibration[feature_index["qy"]]),
            np.asarray(calibration[feature_index["q2"]]),
        )
    )
    native_slope = float(
        _safe_ratio(
            np.asarray(heldout[feature_index["qy"]]),
            np.asarray(heldout[feature_index["q2"]]),
        )
    )
    null_mse = float(
        heldout[feature_index["y2"]]
        / heldout[feature_index["n"]]
    )
    native_mse = float(
        (
            heldout[feature_index["q2"]]
            - 2.0 * heldout[feature_index["qy"]]
            + heldout[feature_index["y2"]]
        )
        / heldout[feature_index["n"]]
    )
    q_mse = float(
        _mse_from_stats(
            heldout,
            slope=np.asarray(q_slope),
            x2_index=feature_index["q2"],
            xy_index=feature_index["qy"],
            feature_index=feature_index,
        )
    )
    proxy_metrics = {}
    for name in proxy_names:
        proxy_slope = float(
            _safe_ratio(
                np.asarray(calibration[feature_index[f"{name}_y"]]),
                np.asarray(calibration[feature_index[f"{name}2"]]),
            )
        )
        proxy_mse = float(
            _mse_from_stats(
                heldout,
                slope=np.asarray(proxy_slope),
                x2_index=feature_index[f"{name}2"],
                xy_index=feature_index[f"{name}_y"],
                feature_index=feature_index,
            )
        )
        proxy_metrics[name] = {
            "calibration_slope": proxy_slope,
            "heldout_mse": proxy_mse,
            "q_minus_proxy_mse_improvement": proxy_mse - q_mse,
        }
    return {
        "native_heldout_slope": native_slope,
        "native_heldout_mse": native_mse,
        "null_heldout_mse": null_mse,
        "native_mse_skill": (
            1.0 - native_mse / null_mse
            if null_mse > 0.0
            else float("nan")
        ),
        "q_calibration_slope": q_slope,
        "recalibrated_q_heldout_mse": q_mse,
        "recalibrated_q_mse_skill": (
            1.0 - q_mse / null_mse
            if null_mse > 0.0
            else float("nan")
        ),
        "proxies": proxy_metrics,
    }


def _crossed_bootstrap(
    stats: np.ndarray,
    *,
    calibration_indices: np.ndarray,
    heldout_indices: np.ndarray,
    feature_index: dict[str, int],
    proxy_names: list[str],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    models = stats.shape[0]
    model_indices = rng.integers(0, models, size=(replicates, models))
    calibration_draws = calibration_indices[
        rng.integers(
            0,
            calibration_indices.size,
            size=(replicates, calibration_indices.size),
        )
    ]
    heldout_draws = heldout_indices[
        rng.integers(
            0,
            heldout_indices.size,
            size=(replicates, heldout_indices.size),
        )
    ]
    calibration = stats[
        model_indices[:, :, None],
        calibration_draws[:, None, :],
    ].sum(axis=(1, 2))
    heldout = stats[
        model_indices[:, :, None],
        heldout_draws[:, None, :],
    ].sum(axis=(1, 2))

    q_calibration_slope = _safe_ratio(
        calibration[:, feature_index["qy"]],
        calibration[:, feature_index["q2"]],
    )
    native_slope = _safe_ratio(
        heldout[:, feature_index["qy"]],
        heldout[:, feature_index["q2"]],
    )
    null_mse = _safe_ratio(
        heldout[:, feature_index["y2"]],
        heldout[:, feature_index["n"]],
    )
    native_mse = _safe_ratio(
        heldout[:, feature_index["q2"]]
        - 2.0 * heldout[:, feature_index["qy"]]
        + heldout[:, feature_index["y2"]],
        heldout[:, feature_index["n"]],
    )
    q_mse = _mse_from_stats(
        heldout,
        slope=q_calibration_slope,
        x2_index=feature_index["q2"],
        xy_index=feature_index["qy"],
        feature_index=feature_index,
    )
    output: dict[str, Any] = {
        "native_heldout_slope_ci": _interval(native_slope),
        "native_mse_skill_ci": _interval(1.0 - native_mse / null_mse),
        "q_calibration_slope_ci": _interval(q_calibration_slope),
        "recalibrated_q_mse_skill_ci": _interval(1.0 - q_mse / null_mse),
        "q_minus_proxy_mse_improvement_ci": {},
    }
    for name in proxy_names:
        proxy_slope = _safe_ratio(
            calibration[:, feature_index[f"{name}_y"]],
            calibration[:, feature_index[f"{name}2"]],
        )
        proxy_mse = _mse_from_stats(
            heldout,
            slope=proxy_slope,
            x2_index=feature_index[f"{name}2"],
            xy_index=feature_index[f"{name}_y"],
            feature_index=feature_index,
        )
        output["q_minus_proxy_mse_improvement_ci"][name] = _interval(
            proxy_mse - q_mse
        )
    return output


def _permutation_test(
    states: list[StateContrasts],
    *,
    calibration_seeds: set[int],
    heldout_seeds: set[int],
    replicates: int,
    seed: int,
    actual_mse: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    placebo_mse = np.full(replicates, np.nan, dtype=np.float64)
    for replicate in range(replicates):
        calibration_q2 = 0.0
        calibration_qy = 0.0
        heldout_n = 0
        heldout_y2 = 0.0
        heldout_q2 = 0.0
        heldout_qy = 0.0
        for state in states:
            permuted = rng.permutation(state.predicted_q)
            q2 = float(np.dot(permuted, permuted))
            qy = float(np.dot(permuted, state.realised_return))
            if state.eval_seed in calibration_seeds:
                calibration_q2 += q2
                calibration_qy += qy
            elif state.eval_seed in heldout_seeds:
                heldout_n += state.realised_return.size
                heldout_y2 += float(
                    np.dot(
                        state.realised_return,
                        state.realised_return,
                    )
                )
                heldout_q2 += q2
                heldout_qy += qy
        if calibration_q2 <= 1e-18 or heldout_n == 0:
            continue
        slope = calibration_qy / calibration_q2
        placebo_mse[replicate] = (
            slope * slope * heldout_q2
            - 2.0 * slope * heldout_qy
            + heldout_y2
        ) / heldout_n
    finite = placebo_mse[np.isfinite(placebo_mse)]
    if not finite.size:
        return {
            "replicates": replicates,
            "finite_replicates": 0,
            "p_value_placebo_at_least_as_good": float("nan"),
            "placebo_mse_interval": [float("nan"), float("nan")],
        }
    p_value = (1.0 + float(np.sum(finite <= actual_mse))) / (
        finite.size + 1.0
    )
    return {
        "replicates": int(replicates),
        "finite_replicates": int(finite.size),
        "p_value_placebo_at_least_as_good": p_value,
        "placebo_mse_interval": _interval(finite),
        "placebo_mse_median": float(np.median(finite)),
    }


def analyze(
    probes: list[Probe],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    permutation_replicates: int,
    permutation_seed: int,
    min_training_seeds: int,
    min_informative_states_per_split: int,
    native_slope_lower: float,
    native_slope_upper: float,
    strict_protocol: bool,
) -> dict[str, Any]:
    if native_slope_upper <= native_slope_lower:
        raise ValueError("native slope upper bound must exceed lower bound")
    labels, eval_seeds, states, metadata = _load_probes(
        probes, strict_protocol=strict_protocol
    )
    proxy_names = (
        list(PROXY_KEYS)
        if strict_protocol
        else [
            name
            for name in PROXY_KEYS
            if all(name in state.proxies for state in states)
        ]
    )
    stats, feature_index, informative = _sufficient_statistics(
        labels, eval_seeds, states, proxy_names
    )
    split = len(eval_seeds) // 2
    calibration_indices = np.arange(split, dtype=np.int64)
    heldout_indices = np.arange(split, len(eval_seeds), dtype=np.int64)
    calibration_seeds = eval_seeds[:split]
    heldout_seeds = eval_seeds[split:]
    calibration = stats[:, calibration_indices].sum(axis=(0, 1))
    heldout = stats[:, heldout_indices].sum(axis=(0, 1))
    metrics = _metric_bundle(
        calibration, heldout, feature_index, proxy_names
    )
    bootstrap = _crossed_bootstrap(
        stats,
        calibration_indices=calibration_indices,
        heldout_indices=heldout_indices,
        feature_index=feature_index,
        proxy_names=proxy_names,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    permutation = _permutation_test(
        states,
        calibration_seeds=set(calibration_seeds),
        heldout_seeds=set(heldout_seeds),
        replicates=permutation_replicates,
        seed=permutation_seed,
        actual_mse=metrics["recalibrated_q_heldout_mse"],
    )
    calibration_informative = informative[
        :, calibration_indices
    ].sum(axis=1)
    heldout_informative = informative[:, heldout_indices].sum(axis=1)
    heldout_native_slopes = []
    for model_index in range(len(labels)):
        one_model = stats[model_index, heldout_indices].sum(axis=0)
        heldout_native_slopes.append(
            float(
                _safe_ratio(
                    np.asarray(one_model[feature_index["qy"]]),
                    np.asarray(one_model[feature_index["q2"]]),
                )
            )
        )
    positive_native_slope_seeds = sum(
        math.isfinite(value) and value > 0.0
        for value in heldout_native_slopes
    )
    slope_ci = bootstrap["native_heldout_slope_ci"]
    checks = {
        "minimum_training_seed_coverage": (
            len(labels) >= min_training_seeds
        ),
        "calibration_informative_coverage_per_training_seed": bool(
            np.all(
                calibration_informative
                >= min_informative_states_per_split
            )
        ),
        "heldout_informative_coverage_per_training_seed": bool(
            np.all(
                heldout_informative
                >= min_informative_states_per_split
            )
        ),
        "native_q_contrast_skill_ci_strictly_positive": (
            bootstrap["native_mse_skill_ci"][0] > 0.0
        ),
        "native_q_slope_ci_inside_equivalence_interval": (
            slope_ci[0] >= native_slope_lower
            and slope_ci[1] <= native_slope_upper
        ),
        "positive_native_slope_training_seed_majority": (
            positive_native_slope_seeds >= math.ceil(len(labels) / 2)
        ),
        "recalibrated_q_skill_ci_strictly_positive": (
            bootstrap["recalibrated_q_mse_skill_ci"][0] > 0.0
        ),
        "q_beats_bin_permutation_placebo": (
            permutation["p_value_placebo_at_least_as_good"] <= 0.05
        ),
    }
    if strict_protocol:
        checks.update(
            {
                f"recalibrated_q_beats_{name}_proxy_ci": (
                    bootstrap["q_minus_proxy_mse_improvement_ci"][name][0]
                    > 0.0
                )
                for name in proxy_names
            }
        )
    per_training_seed = {
        label: {
            **metadata[label],
            "calibration_informative_states": int(
                calibration_informative[index]
            ),
            "heldout_informative_states": int(
                heldout_informative[index]
            ),
            "heldout_native_slope": heldout_native_slopes[index],
        }
        for index, label in enumerate(labels)
    }
    return {
        "status": "ok",
        "claim_scope": (
            "formal_heldout_numeric_value_calibration"
            if strict_protocol
            else "legacy_single_stage_discovery_only"
        ),
        "route_a_claim_allowed": bool(strict_protocol),
        "estimand": (
            "within_state_action_bin_Q_contrast_to_H1_discounted_return_"
            "contrast_under_independent_BC_continuation"
        ),
        "num_training_seeds": len(labels),
        "num_eval_seeds": len(eval_seeds),
        "calibration_seeds": calibration_seeds,
        "heldout_seeds": heldout_seeds,
        "selection_use_forbidden": True,
        "per_training_seed": per_training_seed,
        "metrics": metrics,
        "crossed_bootstrap": {
            **bootstrap,
            "unit": "training_checkpoint_x_simulator_seed",
            "replicates": int(bootstrap_replicates),
            "seed": int(bootstrap_seed),
        },
        "bin_permutation_placebo": {
            **permutation,
            "seed": int(permutation_seed),
            "permutation_unit": "within_same_branch_state_action_bins",
        },
        "thresholds": {
            "min_training_seeds": int(min_training_seeds),
            "min_informative_states_per_split": int(
                min_informative_states_per_split
            ),
            "native_slope_equivalence_interval": [
                float(native_slope_lower),
                float(native_slope_upper),
            ],
            "mse_skill_ci_lower_strictly_above": 0.0,
            "q_minus_proxy_mse_ci_lower_strictly_above": 0.0,
            "permutation_p_value_at_most": 0.05,
        },
        "gate_checks": checks,
        "gate": "pass" if all(checks.values()) else "fail",
        "interpretation": {
            "native_scale": (
                "Native delta-Q is evaluated without rescaling; its slope and "
                "MSE skill test whether Q is already expressed in return units."
            ),
            "recoverable_signal": (
                "Calibration-split slopes test predictive information beyond "
                "rank while every reported MSE is measured only on held-out "
                "simulator seeds."
            ),
            "anti_imitation": (
                "BC prior, full BC path, and action-nearness receive the same "
                "calibration budget as Q."
            ),
        },
    }


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        probes = (
            probes_from_causal_summary(args.causal_summary)
            if args.causal_summary is not None
            else args.probe
        )
        payload = analyze(
            probes,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            permutation_replicates=args.permutation_replicates,
            permutation_seed=args.permutation_seed,
            min_training_seeds=args.min_training_seeds,
            min_informative_states_per_split=(
                args.min_informative_states_per_split
            ),
            native_slope_lower=args.native_slope_lower,
            native_slope_upper=args.native_slope_upper,
            strict_protocol=not args.allow_legacy_discovery,
        )
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except BaseException as exc:
        payload = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        print(payload["traceback"])
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
