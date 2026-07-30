#!/usr/bin/env python3
"""Run a multi-training-seed causal value audit for CQN-Flow.

Each checkpoint is probed on the same simulator seeds. One subprocess runs per
GPU worker, and completed JSON artifacts are reusable. The final bootstrap
resamples both training checkpoints and simulator seeds; action pairs from the
same branch state are never treated as independent samples.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


POLICY_RNG_PROTOCOL = "common_prngkey_probe_seed_plus_eval_seed"


@dataclass(frozen=True)
class Checkpoint:
    label: str
    run_dir: Path
    snapshot: Path


def _checkpoint(value: str) -> Checkpoint:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "checkpoint must be LABEL=RUN_DIR,SNAPSHOT"
        )
    label, raw_paths = value.split("=", 1)
    paths = raw_paths.split(",")
    if not label or len(paths) != 2 or not all(paths):
        raise argparse.ArgumentTypeError(
            "checkpoint must be LABEL=RUN_DIR,SNAPSHOT"
        )
    return Checkpoint(label, Path(paths[0]), Path(paths[1]))


def _nonnegative(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise argparse.ArgumentTypeError("expected a non-negative number")
    return number


def _nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError(
            "expected a non-negative integer"
        )
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        required=True,
        action="append",
        type=_checkpoint,
    )
    parser.add_argument("--gpu-id", required=True, action="append", type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--eval-seed-start", type=int, default=94_000)
    parser.add_argument("--num-eval-seeds", type=int, default=64)
    parser.add_argument("--anchor-steps", default="30,75,120")
    parser.add_argument("--force-level", type=int, default=1)
    parser.add_argument(
        "--dimension-selection",
        choices=("q_span", "round_robin"),
        default="q_span",
    )
    parser.add_argument(
        "--intervention-mode",
        choices=(
            "effective_policy",
            "raw_plan",
            "structured_k0",
            "structured_horizon",
            "sibling_horizon",
        ),
        default="sibling_horizon",
    )
    parser.add_argument("--intervention-horizon", type=int)
    parser.add_argument("--max-continuation-steps", type=int, default=300)
    parser.add_argument(
        "--flow-readout",
        choices=("auto", "distill", "integrated"),
        default="distill",
    )
    parser.add_argument("--num-flow-steps", type=int)
    parser.add_argument(
        "--return-sample-aggregation",
        choices=("config", "mean", "entropic", "truncated_mean"),
        default="config",
    )
    parser.add_argument("--num-action-flow-samples", type=int)
    parser.add_argument(
        "--return-sample-truncate-top",
        type=_nonnegative_int,
    )
    parser.add_argument(
        "--policy-value-beta",
        type=_nonnegative,
        default=1.0,
        help=(
            "Validation-selected deployment beta retained as task-policy "
            "metadata. The causal continuation is controlled separately."
        ),
    )
    parser.add_argument(
        "--continuation-policy",
        choices=("bc", "deployment"),
        default="bc",
        help=(
            "Policy used after the forced action. The default independent BC "
            "continuation audits the value estimand without letting Q alter "
            "its own counterfactual outcomes."
        ),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=94_100)
    parser.add_argument("--min-informative-states", type=int, default=24)
    parser.add_argument(
        "--min-informative-dimensions",
        type=_nonnegative_int,
        default=0,
        help=(
            "Minimum number of action dimensions with enough informative "
            "states in every training-seed probe. A positive value also "
            "requires value-independent round-robin dimension selection."
        ),
    )
    parser.add_argument(
        "--min-informative-states-per-dimension",
        type=_nonnegative_int,
        default=1,
    )
    parser.add_argument(
        "--required-positive-training-seeds",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--require-anti-cheat-proxies",
        action="store_true",
        help=(
            "Require Q ranking to beat both the independent BC-policy prior "
            "and action-nearness proxy under paired crossed bootstrap."
        ),
    )
    return parser.parse_args()


def _completed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text()).get("status") == "ok"
    except (OSError, json.JSONDecodeError):
        return False


def _resolved(checkpoint: Checkpoint) -> Checkpoint:
    return Checkpoint(
        checkpoint.label,
        checkpoint.run_dir.expanduser().resolve(),
        checkpoint.snapshot.expanduser().resolve(),
    )


def _resolve_continuation_beta(
    continuation_policy: str,
    deployment_policy_value_beta: float,
) -> tuple[float | str, float | None]:
    if continuation_policy == "bc":
        return "bc", None
    if continuation_policy == "deployment":
        return float(deployment_policy_value_beta), float(
            deployment_policy_value_beta
        )
    raise ValueError(f"unknown continuation policy: {continuation_policy}")


def build_probe_command(
    checkpoint: Checkpoint,
    *,
    output: Path,
    gpu_id: int,
    eval_seeds: list[int],
    anchor_steps: str,
    force_level: int,
    intervention_mode: str,
    intervention_horizon: int | None,
    max_continuation_steps: int,
    flow_readout: str,
    num_flow_steps: int | None,
    policy_value_beta: float | str,
    bootstrap_replicates: int,
    probe_seed: int,
    dimension_selection: str = "q_span",
    return_sample_aggregation: str = "config",
    num_action_flow_samples: int | None = None,
    return_sample_truncate_top: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("analyze_cqn_branch_counterfactual.py")),
        "--run-dir",
        str(checkpoint.run_dir),
        "--snapshot",
        str(checkpoint.snapshot),
        "--output",
        str(output),
        "--gpu-id",
        str(gpu_id),
        "--eval-seeds",
        ",".join(str(seed) for seed in eval_seeds),
        "--anchor-steps",
        anchor_steps,
        "--force-level",
        str(force_level),
        "--dimension-selection",
        dimension_selection,
        "--intervention-mode",
        intervention_mode,
        "--max-continuation-steps",
        str(max_continuation_steps),
        "--flow-readout",
        flow_readout,
        "--policy-value-beta",
        (
            policy_value_beta
            if isinstance(policy_value_beta, str)
            else f"{policy_value_beta:g}"
        ),
        "--bootstrap-replicates",
        str(bootstrap_replicates),
        "--probe-seed",
        str(probe_seed),
    ]
    if intervention_horizon is not None:
        command.extend(
            ["--intervention-horizon", str(intervention_horizon)]
        )
    if num_flow_steps is not None:
        command.extend(["--num-flow-steps", str(num_flow_steps)])
    if return_sample_aggregation != "config":
        command.extend(
            [
                "--return-sample-aggregation",
                return_sample_aggregation,
            ]
        )
    if num_action_flow_samples is not None:
        command.extend(
            [
                "--num-action-flow-samples",
                str(num_action_flow_samples),
            ]
        )
    if return_sample_truncate_top is not None:
        command.extend(
            [
                "--return-sample-truncate-top",
                str(return_sample_truncate_top),
            ]
        )
    return command


def _run_logged(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        stream.write("$ " + " ".join(command) + "\n")
        stream.flush()
        subprocess.run(
            command,
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )


def _load_probe(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("status") != "ok":
        raise ValueError(f"incomplete branch probe: {path}")
    if not isinstance(payload.get("records"), list):
        raise ValueError(f"branch probe has no records: {path}")
    return payload


def _probe_is_compatible(
    path: Path,
    *,
    intervention_horizon: int | None,
    require_anti_cheat_proxies: bool,
    expected_policy_value_beta: float | None,
    dimension_selection: str = "q_span",
    return_sample_aggregation: str = "config",
    num_action_flow_samples: int | None = None,
    return_sample_truncate_top: int | None = None,
) -> bool:
    if not _completed(path):
        return False
    payload = json.loads(path.read_text())
    if payload.get("policy_rng_protocol") != POLICY_RNG_PROTOCOL:
        return False
    if payload.get("policy_value_beta") != expected_policy_value_beta:
        return False
    if payload.get("dimension_selection", "q_span") != dimension_selection:
        return False
    if (
        intervention_horizon is not None
        and int(payload.get("intervention_horizon", -1))
        != int(intervention_horizon)
    ):
        return False
    if (
        return_sample_aggregation != "config"
        and payload.get("return_sample_aggregation")
        != return_sample_aggregation
    ):
        return False
    if (
        num_action_flow_samples is not None
        and payload.get("num_action_flow_samples")
        != num_action_flow_samples
    ):
        return False
    if (
        return_sample_truncate_top is not None
        and payload.get("return_sample_truncate_top")
        != return_sample_truncate_top
    ):
        return False
    if require_anti_cheat_proxies:
        records = payload.get("records")
        if not isinstance(records, list) or not records:
            return False
        if any(
            not isinstance(record.get("policy_path_proxy"), dict)
            for record in records
        ):
            return False
    return True


def _metric_interval(samples: np.ndarray) -> list[float]:
    finite = samples[np.isfinite(samples)]
    if not finite.size:
        return [float("nan"), float("nan")]
    return [
        float(np.quantile(finite, 0.025)),
        float(np.quantile(finite, 0.975)),
    ]


def summarize(
    labeled_paths: list[tuple[str, Path]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    min_informative_states: int,
    required_positive_training_seeds: int,
    require_anti_cheat_proxies: bool = False,
    min_informative_dimensions: int = 0,
    min_informative_states_per_dimension: int = 1,
) -> dict:
    if len(labeled_paths) < 2:
        raise ValueError("at least two training checkpoints are required")
    labels = [label for label, _ in labeled_paths]
    if len(set(labels)) != len(labels):
        raise ValueError("training checkpoint labels must be unique")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    if not 1 <= required_positive_training_seeds <= len(labeled_paths):
        raise ValueError("invalid required-positive-training-seeds")
    if min_informative_dimensions < 0:
        raise ValueError("min-informative-dimensions must be non-negative")
    if min_informative_states_per_dimension < 1:
        raise ValueError(
            "min-informative-states-per-dimension must be positive"
        )

    payloads = []
    reference_eval_seeds = None
    reference_readout = None
    reference_flow_steps = None
    reference_beta = None
    reference_intervention_horizon = None
    reference_return_sample_aggregation = None
    reference_num_action_flow_samples = None
    reference_return_sample_truncate_top = None
    reference_dimension_selection = None
    reference_num_action_dimensions = None
    for label, path in labeled_paths:
        payload = _load_probe(path.expanduser().resolve())
        eval_seeds = [int(seed) for seed in payload["eval_seeds"]]
        if reference_eval_seeds is None:
            reference_eval_seeds = eval_seeds
            reference_readout = payload.get("value_readout")
            reference_flow_steps = payload.get("num_flow_steps")
            reference_beta = payload.get("policy_value_beta")
            reference_intervention_horizon = payload.get(
                "intervention_horizon"
            )
            reference_return_sample_aggregation = payload.get(
                "return_sample_aggregation"
            )
            reference_num_action_flow_samples = payload.get(
                "num_action_flow_samples"
            )
            reference_return_sample_truncate_top = payload.get(
                "return_sample_truncate_top"
            )
            reference_dimension_selection = payload.get(
                "dimension_selection", "q_span"
            )
            reference_num_action_dimensions = payload.get(
                "num_action_dimensions"
            )
        elif eval_seeds != reference_eval_seeds:
            raise ValueError("branch probes do not share simulator seeds")
        elif payload.get("value_readout") != reference_readout:
            raise ValueError("branch probes do not share the value readout")
        elif payload.get("num_flow_steps") != reference_flow_steps:
            raise ValueError("branch probes do not share flow steps")
        elif payload.get("policy_value_beta") != reference_beta:
            raise ValueError("branch probes do not share policy-value beta")
        elif (
            payload.get("intervention_horizon")
            != reference_intervention_horizon
        ):
            raise ValueError(
                "branch probes do not share intervention horizon"
            )
        elif (
            payload.get("return_sample_aggregation")
            != reference_return_sample_aggregation
        ):
            raise ValueError(
                "branch probes do not share return aggregation"
            )
        elif (
            payload.get("num_action_flow_samples")
            != reference_num_action_flow_samples
        ):
            raise ValueError(
                "branch probes do not share action-flow samples"
            )
        elif (
            payload.get("return_sample_truncate_top")
            != reference_return_sample_truncate_top
        ):
            raise ValueError(
                "branch probes do not share return truncation"
            )
        elif (
            payload.get("dimension_selection", "q_span")
            != reference_dimension_selection
        ):
            raise ValueError(
                "branch probes do not share dimension selection"
            )
        elif (
            payload.get("num_action_dimensions")
            != reference_num_action_dimensions
        ):
            raise ValueError(
                "branch probes do not share action dimensionality"
            )
        payloads.append((label, path.expanduser().resolve(), payload))

    if min_informative_dimensions > 0:
        if reference_dimension_selection != "round_robin":
            raise ValueError(
                "informative-dimension coverage requires round_robin "
                "dimension selection"
            )
        if (
            not isinstance(reference_num_action_dimensions, int)
            or reference_num_action_dimensions < min_informative_dimensions
        ):
            raise ValueError(
                "minimum informative dimensions exceeds action dimensionality"
            )

    assert reference_eval_seeds is not None
    eval_index = {
        seed: index for index, seed in enumerate(reference_eval_seeds)
    }
    shape = (len(payloads), len(reference_eval_seeds))
    pair_correct = np.zeros(shape, dtype=np.float64)
    pair_count = np.zeros(shape, dtype=np.float64)
    spearman_sum = np.zeros(shape, dtype=np.float64)
    spearman_count = np.zeros(shape, dtype=np.float64)
    proxy_names = (
        "policy_prior",
        "policy_path",
        "action_nearness",
    )
    proxy_pair_correct = {
        name: np.zeros(shape, dtype=np.float64) for name in proxy_names
    }
    proxy_pair_count = {
        name: np.zeros(shape, dtype=np.float64) for name in proxy_names
    }
    per_training_seed = {}
    sources = {}
    positive_training_seeds = 0

    for model_index, (label, path, payload) in enumerate(payloads):
        for record in payload["records"]:
            if float(record["realized_return_span"]) <= 0.0:
                continue
            seed = int(record["eval_seed"])
            if seed not in eval_index:
                raise ValueError(f"record seed outside requested set: {seed}")
            environment_index = eval_index[seed]
            count = float(record["num_informative_pairs"])
            accuracy = float(record["pairwise_sign_accuracy"])
            if count > 0.0 and math.isfinite(accuracy):
                pair_correct[model_index, environment_index] += (
                    accuracy * count
                )
                pair_count[model_index, environment_index] += count
            spearman = float(record["spearman"])
            if math.isfinite(spearman):
                spearman_sum[model_index, environment_index] += spearman
                spearman_count[model_index, environment_index] += 1.0
            for proxy_name in proxy_names:
                proxy = record.get(f"{proxy_name}_proxy")
                if not isinstance(proxy, dict):
                    continue
                proxy_count = float(proxy.get("num_informative_pairs", 0))
                proxy_accuracy = float(
                    proxy.get("pairwise_sign_accuracy", float("nan"))
                )
                if proxy_count > 0.0 and math.isfinite(proxy_accuracy):
                    proxy_pair_correct[proxy_name][
                        model_index,
                        environment_index,
                    ] += proxy_accuracy * proxy_count
                    proxy_pair_count[proxy_name][
                        model_index,
                        environment_index,
                    ] += proxy_count

        point_pairwise = float(payload["pairwise_sign_accuracy"])
        point_spearman = float(payload["mean_spearman"])
        causal_direction_positive = (
            point_pairwise > 0.5 or point_spearman > 0.0
        )
        proxy_point_pairwise = {}
        q_minus_proxy = {}
        proxy_coverage = True
        for proxy_name in proxy_names:
            denominator = float(
                proxy_pair_count[proxy_name][model_index].sum()
            )
            value = (
                float(
                    proxy_pair_correct[proxy_name][model_index].sum()
                    / denominator
                )
                if denominator > 0.0
                else float("nan")
            )
            proxy_point_pairwise[proxy_name] = value
            q_minus_proxy[proxy_name] = (
                float(point_pairwise - value)
                if math.isfinite(value)
                else float("nan")
            )
            proxy_coverage = (
                proxy_coverage
                and denominator > 0.0
                and np.array_equal(
                    proxy_pair_count[proxy_name][model_index],
                    pair_count[model_index],
                )
            )
        anti_cheat_direction_positive = (
            proxy_coverage
            and all(q_minus_proxy[name] > 0.0 for name in proxy_names)
        )
        informative_dimension_counts = {
            int(dimension): int(count)
            for dimension, count in payload.get(
                "informative_states_per_dimension", {}
            ).items()
        }
        sufficiently_informative_dimensions = sum(
            count >= min_informative_states_per_dimension
            for count in informative_dimension_counts.values()
        )
        direction_positive = causal_direction_positive and (
            not require_anti_cheat_proxies
            or anti_cheat_direction_positive
        )
        positive_training_seeds += int(direction_positive)
        per_training_seed[label] = {
            "num_states": int(payload["num_states"]),
            "num_informative_states": int(payload["num_informative_states"]),
            "pairwise_sign_accuracy": point_pairwise,
            "pairwise_sign_accuracy_ci": payload["state_bootstrap"][
                "pairwise_sign_accuracy_ci"
            ],
            "mean_spearman": point_spearman,
            "mean_spearman_ci": payload["state_bootstrap"][
                "mean_spearman_ci"
            ],
            "top1_match_rate": float(payload["top1_match_rate"]),
            "mean_realized_regret": float(
                payload["mean_realized_regret"]
            ),
            "causal_direction_positive": causal_direction_positive,
            "proxy_pairwise_sign_accuracy": proxy_point_pairwise,
            "q_minus_proxy_pairwise": q_minus_proxy,
            "anti_cheat_proxy_coverage": proxy_coverage,
            "anti_cheat_direction_positive": (
                anti_cheat_direction_positive
            ),
            "informative_states_per_dimension": (
                informative_dimension_counts
            ),
            "sufficiently_informative_dimensions": int(
                sufficiently_informative_dimensions
            ),
            "direction_positive": direction_positive,
        }
        sources[label] = str(path)

    total_pairs = float(pair_count.sum())
    total_spearman = float(spearman_count.sum())
    aggregate_pairwise = (
        float(pair_correct.sum() / total_pairs)
        if total_pairs > 0.0
        else float("nan")
    )
    aggregate_spearman = (
        float(spearman_sum.sum() / total_spearman)
        if total_spearman > 0.0
        else float("nan")
    )

    rng = np.random.default_rng(int(bootstrap_seed))
    model_indices = rng.integers(
        0,
        shape[0],
        size=(bootstrap_replicates, shape[0]),
    )
    environment_indices = rng.integers(
        0,
        shape[1],
        size=(bootstrap_replicates, shape[1]),
    )
    sampled_correct = pair_correct[
        model_indices[:, :, None],
        environment_indices[:, None, :],
    ].sum(axis=(1, 2))
    sampled_pair_count = pair_count[
        model_indices[:, :, None],
        environment_indices[:, None, :],
    ].sum(axis=(1, 2))
    pairwise_samples = np.divide(
        sampled_correct,
        sampled_pair_count,
        out=np.full(bootstrap_replicates, np.nan, dtype=np.float64),
        where=sampled_pair_count > 0.0,
    )
    sampled_spearman = spearman_sum[
        model_indices[:, :, None],
        environment_indices[:, None, :],
    ].sum(axis=(1, 2))
    sampled_spearman_count = spearman_count[
        model_indices[:, :, None],
        environment_indices[:, None, :],
    ].sum(axis=(1, 2))
    spearman_samples = np.divide(
        sampled_spearman,
        sampled_spearman_count,
        out=np.full(bootstrap_replicates, np.nan, dtype=np.float64),
        where=sampled_spearman_count > 0.0,
    )
    pairwise_ci = _metric_interval(pairwise_samples)
    spearman_ci = _metric_interval(spearman_samples)
    aggregate_proxy_pairwise = {}
    q_minus_proxy_ci = {}
    proxy_coverage_per_training_seed = {}
    for proxy_name in proxy_names:
        proxy_total_count = float(proxy_pair_count[proxy_name].sum())
        aggregate_proxy_pairwise[proxy_name] = (
            float(
                proxy_pair_correct[proxy_name].sum()
                / proxy_total_count
            )
            if proxy_total_count > 0.0
            else float("nan")
        )
        sampled_proxy_correct = proxy_pair_correct[proxy_name][
            model_indices[:, :, None],
            environment_indices[:, None, :],
        ].sum(axis=(1, 2))
        sampled_proxy_count = proxy_pair_count[proxy_name][
            model_indices[:, :, None],
            environment_indices[:, None, :],
        ].sum(axis=(1, 2))
        proxy_samples = np.divide(
            sampled_proxy_correct,
            sampled_proxy_count,
            out=np.full(
                bootstrap_replicates,
                np.nan,
                dtype=np.float64,
            ),
            where=sampled_proxy_count > 0.0,
        )
        q_minus_proxy_ci[proxy_name] = _metric_interval(
            pairwise_samples - proxy_samples
        )
        proxy_coverage_per_training_seed[proxy_name] = [
            bool(
                proxy_pair_count[proxy_name][index].sum() > 0.0
                and np.array_equal(
                    proxy_pair_count[proxy_name][index],
                    pair_count[index],
                )
            )
            for index in range(shape[0])
        ]
    causal_ci_pass = pairwise_ci[0] > 0.5 or spearman_ci[0] > 0.0
    checks = {
        "informative_coverage_per_training_seed": all(
            item["num_informative_states"] >= min_informative_states
            for item in per_training_seed.values()
        ),
        "aggregate_point_direction_positive": (
            aggregate_pairwise > 0.5 or aggregate_spearman > 0.0
        ),
        "aggregate_causal_ci_strictly_positive": causal_ci_pass,
        "positive_training_seed_requirement": (
            positive_training_seeds >= required_positive_training_seeds
        ),
    }
    if require_anti_cheat_proxies:
        checks.update(
            {
                "anti_cheat_proxy_coverage_per_training_seed": all(
                    all(coverage)
                    for coverage in (
                        proxy_coverage_per_training_seed.values()
                    )
                ),
                "q_pairwise_above_policy_prior_proxy_ci": (
                    q_minus_proxy_ci["policy_prior"][0] > 0.0
                ),
                "q_pairwise_above_policy_path_proxy_ci": (
                    q_minus_proxy_ci["policy_path"][0] > 0.0
                ),
                "q_pairwise_above_action_nearness_proxy_ci": (
                    q_minus_proxy_ci["action_nearness"][0] > 0.0
                ),
            }
        )
    if min_informative_dimensions > 0:
        checks.update(
            {
                "dimension_selection_is_value_independent": (
                    reference_dimension_selection == "round_robin"
                ),
                "informative_dimension_coverage_per_training_seed": all(
                    item["sufficiently_informative_dimensions"]
                    >= min_informative_dimensions
                    for item in per_training_seed.values()
                ),
            }
        )
    return {
        "status": "ok",
        "labels": labels,
        "sources": sources,
        "value_readout": reference_readout,
        "num_flow_steps": reference_flow_steps,
        "policy_value_beta": reference_beta,
        "intervention_horizon": reference_intervention_horizon,
        "return_sample_aggregation": (
            reference_return_sample_aggregation
        ),
        "num_action_flow_samples": reference_num_action_flow_samples,
        "return_sample_truncate_top": (
            reference_return_sample_truncate_top
        ),
        "dimension_selection": reference_dimension_selection,
        "num_action_dimensions": reference_num_action_dimensions,
        "num_training_seeds": shape[0],
        "num_eval_seeds": shape[1],
        "eval_seed_start": int(reference_eval_seeds[0]),
        "eval_seed_end": int(reference_eval_seeds[-1]),
        "per_training_seed": per_training_seed,
        "positive_training_seeds": positive_training_seeds,
        "aggregate_pairwise_sign_accuracy": aggregate_pairwise,
        "aggregate_pairwise_sign_accuracy_ci": pairwise_ci,
        "aggregate_mean_spearman": aggregate_spearman,
        "aggregate_mean_spearman_ci": spearman_ci,
        "anti_cheat_proxies_required": bool(
            require_anti_cheat_proxies
        ),
        "aggregate_proxy_pairwise_sign_accuracy": (
            aggregate_proxy_pairwise
        ),
        "aggregate_q_minus_proxy_pairwise_ci": q_minus_proxy_ci,
        "proxy_coverage_per_training_seed": (
            proxy_coverage_per_training_seed
        ),
        "bootstrap_unit": "crossed_training_checkpoint_x_simulator_seed",
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_seed": int(bootstrap_seed),
        "thresholds": {
            "min_informative_states": int(min_informative_states),
            "required_positive_training_seeds": int(
                required_positive_training_seeds
            ),
            "min_informative_dimensions": int(
                min_informative_dimensions
            ),
            "min_informative_states_per_dimension": int(
                min_informative_states_per_dimension
            ),
            "pairwise_ci_lower_strictly_above": 0.5,
            "spearman_ci_lower_strictly_above": 0.0,
            "q_minus_proxy_pairwise_ci_lower_strictly_above": (
                0.0 if require_anti_cheat_proxies else None
            ),
        },
        "gate_checks": checks,
        "gate": "pass" if all(checks.values()) else "fail",
    }


def run_gate(args: argparse.Namespace) -> dict:
    if len(args.checkpoint) < 2:
        raise ValueError("at least two training checkpoints are required")
    args.checkpoint = [_resolved(item) for item in args.checkpoint]
    if len({item.label for item in args.checkpoint}) != len(args.checkpoint):
        raise ValueError("checkpoint labels must be unique")
    if not args.gpu_id or len(set(args.gpu_id)) != len(args.gpu_id):
        raise ValueError("gpu-id workers must be non-empty and unique")
    if args.num_eval_seeds < 1 or args.bootstrap_replicates < 1:
        raise ValueError("seed and bootstrap counts must be positive")
    if args.max_continuation_steps < 1:
        raise ValueError("max-continuation-steps must be positive")
    if args.intervention_horizon is not None and (
        args.intervention_horizon < 1
        or args.intervention_mode
        not in {"structured_horizon", "sibling_horizon"}
    ):
        raise ValueError(
            "intervention-horizon requires a positive repeated-horizon mode"
        )
    if (
        args.dimension_selection == "round_robin"
        and args.intervention_mode != "sibling_horizon"
    ):
        raise ValueError(
            "round_robin dimension selection requires sibling_horizon"
        )
    if args.min_informative_dimensions > 0 and (
        args.dimension_selection != "round_robin"
        or args.min_informative_states_per_dimension < 1
    ):
        raise ValueError(
            "informative-dimension coverage requires round_robin selection "
            "and a positive per-dimension minimum"
        )
    if args.num_flow_steps is not None and (
        args.num_flow_steps < 1 or args.flow_readout != "integrated"
    ):
        raise ValueError(
            "num-flow-steps requires a positive integrated readout"
        )
    if (
        args.num_action_flow_samples is not None
        and args.num_action_flow_samples < 1
    ):
        raise ValueError("num-action-flow-samples must be positive")
    if args.return_sample_aggregation == "truncated_mean" and (
        args.num_action_flow_samples is None
        or args.return_sample_truncate_top is None
        or args.return_sample_truncate_top < 1
        or args.return_sample_truncate_top
        >= args.num_action_flow_samples
    ):
        raise ValueError(
            "truncated_mean requires an explicit action sample count and "
            "truncation in [1, samples)."
        )
    for checkpoint in args.checkpoint:
        if not (checkpoint.run_dir / ".hydra" / "config.yaml").is_file():
            raise FileNotFoundError(checkpoint.run_dir)
        if not checkpoint.snapshot.is_file():
            raise FileNotFoundError(checkpoint.snapshot)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_seeds = list(
        range(
            int(args.eval_seed_start),
            int(args.eval_seed_start) + int(args.num_eval_seeds),
        )
    )
    (
        probe_policy_value_beta,
        expected_probe_policy_value_beta,
    ) = _resolve_continuation_beta(
        args.continuation_policy,
        args.policy_value_beta,
    )
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "status": "running",
        "checkpoints": [
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(checkpoint).items()
            }
            for checkpoint in args.checkpoint
        ],
        "gpu_ids": list(args.gpu_id),
        "eval_seed_start": eval_seeds[0],
        "eval_seed_end": eval_seeds[-1],
        "anchor_steps": args.anchor_steps,
        "force_level": args.force_level,
        "dimension_selection": args.dimension_selection,
        "intervention_mode": args.intervention_mode,
        "intervention_horizon": args.intervention_horizon,
        "flow_readout": args.flow_readout,
        "num_flow_steps": args.num_flow_steps,
        "deployment_policy_value_beta": args.policy_value_beta,
        "continuation_policy": args.continuation_policy,
        "policy_value_beta": expected_probe_policy_value_beta,
        "return_sample_aggregation": args.return_sample_aggregation,
        "num_action_flow_samples": args.num_action_flow_samples,
        "return_sample_truncate_top": (
            args.return_sample_truncate_top
        ),
        "require_anti_cheat_proxies": bool(
            args.require_anti_cheat_proxies
        ),
        "min_informative_dimensions": args.min_informative_dimensions,
        "min_informative_states_per_dimension": (
            args.min_informative_states_per_dimension
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    started = time.time()
    work_queue: queue.Queue[Checkpoint] = queue.Queue()
    for checkpoint in args.checkpoint:
        work_queue.put(checkpoint)
    failures: list[BaseException] = []
    failure_lock = threading.Lock()

    def worker(gpu_id: int) -> None:
        while True:
            try:
                checkpoint = work_queue.get_nowait()
            except queue.Empty:
                return
            try:
                result_dir = output_dir / checkpoint.label
                output = result_dir / "probe.json"
                if not _probe_is_compatible(
                    output,
                    intervention_horizon=args.intervention_horizon,
                    require_anti_cheat_proxies=(
                        args.require_anti_cheat_proxies
                    ),
                    expected_policy_value_beta=(
                        expected_probe_policy_value_beta
                    ),
                    dimension_selection=args.dimension_selection,
                    return_sample_aggregation=(
                        args.return_sample_aggregation
                    ),
                    num_action_flow_samples=(
                        args.num_action_flow_samples
                    ),
                    return_sample_truncate_top=(
                        args.return_sample_truncate_top
                    ),
                ):
                    command = build_probe_command(
                        checkpoint,
                        output=output,
                        gpu_id=gpu_id,
                        eval_seeds=eval_seeds,
                        anchor_steps=args.anchor_steps,
                        force_level=args.force_level,
                        intervention_mode=args.intervention_mode,
                        intervention_horizon=args.intervention_horizon,
                        max_continuation_steps=args.max_continuation_steps,
                        flow_readout=args.flow_readout,
                        num_flow_steps=args.num_flow_steps,
                        policy_value_beta=probe_policy_value_beta,
                        bootstrap_replicates=args.bootstrap_replicates,
                        probe_seed=args.bootstrap_seed,
                        dimension_selection=args.dimension_selection,
                        return_sample_aggregation=(
                            args.return_sample_aggregation
                        ),
                        num_action_flow_samples=(
                            args.num_action_flow_samples
                        ),
                        return_sample_truncate_top=(
                            args.return_sample_truncate_top
                        ),
                    )
                    _run_logged(command, result_dir / "probe.log")
                if not _probe_is_compatible(
                    output,
                    intervention_horizon=args.intervention_horizon,
                    require_anti_cheat_proxies=(
                        args.require_anti_cheat_proxies
                    ),
                    expected_policy_value_beta=(
                        expected_probe_policy_value_beta
                    ),
                    dimension_selection=args.dimension_selection,
                    return_sample_aggregation=(
                        args.return_sample_aggregation
                    ),
                    num_action_flow_samples=(
                        args.num_action_flow_samples
                    ),
                    return_sample_truncate_top=(
                        args.return_sample_truncate_top
                    ),
                ):
                    raise RuntimeError(f"probe did not complete: {output}")
            except BaseException as exc:
                with failure_lock:
                    failures.append(exc)
            finally:
                work_queue.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu_id,), daemon=False)
        for gpu_id in args.gpu_id
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(
            f"{len(failures)} GPU worker(s) failed"
        ) from failures[0]

    summary = summarize(
        [
            (
                checkpoint.label,
                output_dir / checkpoint.label / "probe.json",
            )
            for checkpoint in args.checkpoint
        ],
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        min_informative_states=args.min_informative_states,
        required_positive_training_seeds=(
            args.required_positive_training_seeds
        ),
        require_anti_cheat_proxies=args.require_anti_cheat_proxies,
        min_informative_dimensions=args.min_informative_dimensions,
        min_informative_states_per_dimension=(
            args.min_informative_states_per_dimension
        ),
    )
    summary["elapsed_seconds"] = time.time() - started
    summary["manifest"] = str(manifest_path)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    manifest["status"] = "ok"
    manifest["summary"] = str(summary_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> int:
    args = parse_args()
    summary = run_gate(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
