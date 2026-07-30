#!/usr/bin/env python3
"""Summarize final A/B research gates across multiple Flow candidates.

Route A has one matched monolithic scalar-Q candidate.  Route B can contain
several predeclared Flow mechanisms (fixed-budget distill, integrated
readout, validation-selected checkpoint, or policy-value TD targets).  A Flow
candidate is deployable only when both its sealed task gate and its
multi-training-seed causal gate pass.  Among deployable candidates, selection
is conservative: maximize the lower endpoint of the task bootstrap interval,
then mean paired improvement.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CandidatePaths:
    label: str
    task: Path
    causal: Path | None


def _candidate(value: str) -> CandidatePaths:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "candidate must be LABEL=TASK_JSON[,CAUSAL_JSON]"
        )
    label, raw_paths = value.split("=", 1)
    paths = raw_paths.split(",")
    if not label or len(paths) not in {1, 2} or not all(paths):
        raise argparse.ArgumentTypeError(
            "candidate must be LABEL=TASK_JSON[,CAUSAL_JSON]"
        )
    return CandidatePaths(
        label=label,
        task=Path(paths[0]),
        causal=Path(paths[1]) if len(paths) == 2 else None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-summary",
        type=Path,
        help=(
            "Optional prior autoresearch summary whose A/B candidates are "
            "retained before appending new candidates."
        ),
    )
    parser.add_argument("--a-task", type=Path)
    parser.add_argument("--a-causal", type=Path)
    parser.add_argument(
        "--a-candidate",
        action="append",
        default=[],
        type=_candidate,
    )
    parser.add_argument(
        "--b-candidate",
        action="append",
        default=[],
        type=_candidate,
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text())
    if payload.get("status") != "ok":
        raise ValueError(f"incomplete artifact: {resolved}")
    payload["_path"] = str(resolved)
    return payload


def _restore_evidence(
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Restore a compact summary row to the payload shape used internally."""

    artifact = evidence.get("artifact")
    if not isinstance(artifact, str) or not artifact:
        raise ValueError("base-summary evidence is missing its artifact")
    payload = dict(evidence)
    payload.pop("artifact", None)
    payload["status"] = "ok"
    payload["_path"] = artifact
    return payload


def _base_candidates(
    payload: dict[str, Any],
    route_key: str,
) -> list[tuple[str, dict[str, Any], dict[str, Any] | None]]:
    route = payload.get(route_key)
    if not isinstance(route, dict):
        raise ValueError(f"base summary is missing {route_key}")
    rows = route.get("candidates")
    if not isinstance(rows, list):
        raise ValueError(f"base summary {route_key} is missing candidates")
    restored = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("label"), str):
            raise ValueError(f"invalid candidate in base summary {route_key}")
        task = row.get("task")
        if not isinstance(task, dict):
            raise ValueError("base-summary candidate is missing task evidence")
        causal = row.get("causal_value")
        if causal is not None and not isinstance(causal, dict):
            raise ValueError("invalid causal evidence in base summary")
        restored.append(
            (
                row["label"],
                _restore_evidence(task),
                _restore_evidence(causal) if causal is not None else None,
            )
        )
    return restored


def _task_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact": payload["_path"],
        "gate": payload.get("gate"),
        "num_training_seeds": payload.get("num_training_seeds"),
        "num_eval_seeds": payload.get("num_eval_seeds"),
        "eval_seed_start": payload.get("eval_seed_start"),
        "eval_seed_end": payload.get("eval_seed_end"),
        "mean_baseline_success": payload.get("mean_baseline_success"),
        "mean_candidate_success": payload.get("mean_candidate_success"),
        "mean_paired_delta": payload.get("mean_paired_delta"),
        "crossed_bootstrap_ci95": payload.get("crossed_bootstrap_ci95"),
        "aggregate_paired_wins": payload.get("aggregate_paired_wins"),
        "aggregate_paired_losses": payload.get("aggregate_paired_losses"),
        "per_training_seed": payload.get("per_training_seed"),
        "selection": payload.get("selection"),
        "selected_steps": payload.get("selected_steps"),
        "candidate_readout": payload.get("candidate_readout"),
        "num_flow_steps": payload.get("num_flow_steps"),
        "return_sample_aggregation": payload.get(
            "return_sample_aggregation"
        ),
        "num_action_flow_samples": payload.get(
            "num_action_flow_samples"
        ),
        "return_sample_truncate_top": payload.get(
            "return_sample_truncate_top"
        ),
        "thresholds": payload.get("thresholds"),
        "gate_checks": payload.get("gate_checks"),
    }


def _causal_evidence(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    return {
        "artifact": payload["_path"],
        "gate": payload.get("gate"),
        "value_readout": payload.get("value_readout"),
        "policy_value_beta": payload.get("policy_value_beta"),
        "intervention_horizon": payload.get("intervention_horizon"),
        "dimension_selection": payload.get("dimension_selection"),
        "num_action_dimensions": payload.get("num_action_dimensions"),
        "num_training_seeds": payload.get("num_training_seeds"),
        "num_eval_seeds": payload.get("num_eval_seeds"),
        "eval_seed_start": payload.get("eval_seed_start"),
        "eval_seed_end": payload.get("eval_seed_end"),
        "informative_dimension_thresholds": {
            "min_informative_dimensions": payload.get(
                "thresholds", {}
            ).get("min_informative_dimensions"),
            "min_informative_states_per_dimension": payload.get(
                "thresholds", {}
            ).get("min_informative_states_per_dimension"),
        },
        "return_sample_aggregation": payload.get(
            "return_sample_aggregation"
        ),
        "num_action_flow_samples": payload.get(
            "num_action_flow_samples"
        ),
        "return_sample_truncate_top": payload.get(
            "return_sample_truncate_top"
        ),
        "aggregate_pairwise_sign_accuracy": payload.get(
            "aggregate_pairwise_sign_accuracy"
        ),
        "aggregate_pairwise_sign_accuracy_ci": payload.get(
            "aggregate_pairwise_sign_accuracy_ci"
        ),
        "aggregate_mean_spearman": payload.get("aggregate_mean_spearman"),
        "aggregate_mean_spearman_ci": payload.get(
            "aggregate_mean_spearman_ci"
        ),
        "anti_cheat_proxies_required": payload.get(
            "anti_cheat_proxies_required"
        ),
        "aggregate_proxy_pairwise_sign_accuracy": payload.get(
            "aggregate_proxy_pairwise_sign_accuracy"
        ),
        "aggregate_q_minus_proxy_pairwise_ci": payload.get(
            "aggregate_q_minus_proxy_pairwise_ci"
        ),
        "positive_training_seeds": payload.get("positive_training_seeds"),
        "per_training_seed": payload.get("per_training_seed"),
        "gate_checks": payload.get("gate_checks"),
    }


def _ci_lower(task: dict[str, Any]) -> float:
    interval = task.get("crossed_bootstrap_ci95")
    if not isinstance(interval, list) or len(interval) != 2:
        return float("-inf")
    return float(interval[0])


def _mean_delta(task: dict[str, Any]) -> float:
    value = task.get("mean_paired_delta")
    return float("-inf") if value is None else float(value)


def _at_least(value: Any, threshold: float) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number >= float(threshold)


def _strictly_above(value: Any, threshold: float) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > float(threshold)


def _candidate_rows(
    candidates: list[
        tuple[str, dict[str, Any], dict[str, Any] | None]
    ],
    *,
    strict_task_improvement: bool,
) -> list[dict[str, Any]]:
    rows = []
    for label, task, causal in candidates:
        task_interval = task.get("crossed_bootstrap_ci95")
        task_thresholds = task.get("thresholds")
        task_checks = task.get("gate_checks")
        task_protocol_valid = bool(
            isinstance(task_interval, list)
            and len(task_interval) == 2
            and isinstance(task_thresholds, dict)
            and isinstance(task_checks, dict)
            and _at_least(task.get("num_training_seeds"), 3)
            and _at_least(task.get("num_eval_seeds"), 200)
            and _at_least(task_thresholds.get("min_mean_delta"), 0.0)
            and _at_least(task_thresholds.get("min_ci_lower"), 0.0)
            and _at_least(task.get("mean_paired_delta"), 0.0)
            and _at_least(task_interval[0], 0.0)
            and task_checks.get(
                "mean_delta_strictly_above_threshold"
            )
            is True
            and task_checks.get(
                "crossed_ci_lower_at_least_threshold"
            )
            is True
            and task_checks.get("aggregate_wins_above_losses") is True
            and task_checks.get("positive_training_seed_majority") is True
        )
        if strict_task_improvement and task_protocol_valid:
            task_protocol_valid = bool(
                _strictly_above(task.get("mean_paired_delta"), 0.0)
                and _strictly_above(task_interval[0], 0.0)
            )
        task_pass = bool(
            task.get("gate") == "pass" and task_protocol_valid
        )
        causal_checks = (
            causal.get("gate_checks", {})
            if isinstance(causal, dict)
            else {}
        )
        unbiased_dimension_confirmation = bool(
            isinstance(causal, dict)
            and causal.get("dimension_selection") == "round_robin"
            and causal_checks.get(
                "dimension_selection_is_value_independent"
            )
            is True
            and causal_checks.get(
                "informative_dimension_coverage_per_training_seed"
            )
            is True
        )
        causal_protocol_valid = bool(
            isinstance(causal, dict)
            and _at_least(causal.get("num_training_seeds"), 3)
            and _at_least(causal.get("num_eval_seeds"), 32)
            and causal.get("intervention_horizon") == 1
            and causal.get("policy_value_beta") is None
            and causal.get("anti_cheat_proxies_required") is True
            and causal_checks.get(
                "anti_cheat_proxy_coverage_per_training_seed"
            )
            is True
            and causal_checks.get(
                "q_pairwise_above_policy_prior_proxy_ci"
            )
            is True
            and causal_checks.get(
                "q_pairwise_above_policy_path_proxy_ci"
            )
            is True
            and causal_checks.get(
                "q_pairwise_above_action_nearness_proxy_ci"
            )
            is True
        )
        causal_pass = bool(
            causal is not None
            and causal.get("gate") == "pass"
            and unbiased_dimension_confirmation
            and causal_protocol_valid
        )
        rows.append(
            {
                "label": label,
                "overall_gate": (
                    "pass" if task_pass and causal_pass else "fail"
                ),
                "checks": {
                    "task_requirement": task_pass,
                    "sealed_multiseed_task_protocol": task_protocol_valid,
                    "causally_meaningful_value": causal_pass,
                    "q_independent_dimension_confirmation": (
                        unbiased_dimension_confirmation
                    ),
                    "independent_bc_causal_protocol": (
                        causal_protocol_valid
                    ),
                },
                "task": _task_evidence(task),
                "causal_value": _causal_evidence(causal),
            }
        )
    return rows


def _select(rows: list[dict[str, Any]]) -> str | None:
    qualified = [row for row in rows if row["overall_gate"] == "pass"]
    if not qualified:
        return None
    return max(
        qualified,
        key=lambda row: (
            _ci_lower(row["task"]),
            _mean_delta(row["task"]),
            row["label"],
        ),
    )["label"]


def summarize_multi(
    a_candidates: list[
        tuple[str, dict[str, Any], dict[str, Any] | None]
    ],
    b_candidates: list[
        tuple[str, dict[str, Any], dict[str, Any] | None]
    ],
) -> dict[str, Any]:
    if not a_candidates:
        raise ValueError("Route A requires at least one candidate")
    a_rows = _candidate_rows(
        a_candidates,
        strict_task_improvement=False,
    )
    b_rows = _candidate_rows(
        b_candidates,
        strict_task_improvement=True,
    )
    selected_a = _select(a_rows)
    selected_b = _select(b_rows)
    route_a = {
        "route": "A",
        "overall_gate": "pass" if selected_a is not None else "fail",
        "task_requirement": (
            "not worse than validation-selected clean CQN-AS"
        ),
        "selection_rule": (
            "among task-and-causal passes, maximize task CI lower bound, "
            "then mean paired delta"
        ),
        "selected_candidate": selected_a,
        "candidates": a_rows,
        "unmet_gates": (
            []
            if selected_a is not None
            else [
                "no non-Flow candidate passed both task and causal gates"
            ]
        ),
    }
    route_b = {
        "route": "B",
        "overall_gate": "pass" if selected_b is not None else "fail",
        "task_requirement": (
            "strictly better than validation-selected clean CQN-AS"
        ),
        "selection_rule": (
            "among task-and-causal passes, maximize task CI lower bound, "
            "then mean paired delta"
        ),
        "selected_candidate": selected_b,
        "candidates": b_rows,
        "unmet_gates": (
            []
            if selected_b is not None
            else [
                "no Flow candidate passed both strict task and causal gates"
            ]
        ),
    }
    both_pass = (
        route_a["overall_gate"] == "pass"
        and route_b["overall_gate"] == "pass"
    )
    return {
        "status": "ok",
        "protocol": {
            "checkpoint_selection": (
                "screen and validation precede sealed confirmation"
            ),
            "task_confirmation": (
                "paired environment seeds and crossed bootstrap"
            ),
            "causal_confirmation": (
                "three training seeds with sibling-horizon interventions; "
                "a final recommendation additionally requires "
                "Q-independent dimension confirmation"
            ),
            "training_loss_used_as_policy_metric": False,
        },
        "route_a": route_a,
        "route_b": route_b,
        "research_goal_gate": "pass" if both_pass else "fail",
        "next_action": (
            "freeze both routes and prepare reproducibility table"
            if both_pass
            else {
                "A": route_a["unmet_gates"],
                "B": route_b["unmet_gates"],
            }
        ),
    }


def summarize(
    a_task: dict[str, Any],
    a_causal: dict[str, Any],
    b_candidates: list[
        tuple[str, dict[str, Any], dict[str, Any] | None]
    ],
) -> dict[str, Any]:
    """Backward-compatible single-candidate Route-A entry point."""

    return summarize_multi(
        [("direct_scalar_q_replay_next", a_task, a_causal)],
        b_candidates,
    )


def main() -> int:
    args = parse_args()
    base_a = []
    base_b = []
    if args.base_summary is not None:
        base_payload = _load(args.base_summary)
        base_a = _base_candidates(base_payload, "route_a")
        base_b = _base_candidates(base_payload, "route_b")

    if args.a_candidate:
        if args.a_task is not None or args.a_causal is not None:
            raise ValueError(
                "use either --a-candidate or --a-task/--a-causal"
            )
        a_paths = args.a_candidate
    elif args.a_task is not None or args.a_causal is not None:
        if args.a_task is None or args.a_causal is None:
            raise ValueError(
                "--a-task and --a-causal are required without --a-candidate"
            )
        a_paths = [
            CandidatePaths(
                "direct_scalar_q_replay_next",
                args.a_task,
                args.a_causal,
            )
        ]
    else:
        a_paths = []
    if not base_a and not a_paths:
        raise ValueError(
            "Route A requires --base-summary, --a-candidate, or "
            "--a-task/--a-causal"
        )

    labels = [
        *[label for label, _, _ in [*base_a, *base_b]],
        *[item.label for item in [*a_paths, *args.b_candidate]],
    ]
    if len(set(labels)) != len(labels):
        raise ValueError("candidate labels must be unique")
    a_candidates = base_a + [
        (
            item.label,
            _load(item.task),
            _load(item.causal) if item.causal is not None else None,
        )
        for item in a_paths
    ]
    b_candidates = base_b + [
        (
            item.label,
            _load(item.task),
            _load(item.causal) if item.causal is not None else None,
        )
        for item in args.b_candidate
    ]
    payload = summarize_multi(a_candidates, b_candidates)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
