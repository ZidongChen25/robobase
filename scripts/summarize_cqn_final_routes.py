#!/usr/bin/env python3
"""Combine task and causal gates into final, route-specific CQN verdicts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-task", required=True, type=Path)
    parser.add_argument("--a-causal", required=True, type=Path)
    parser.add_argument("--b-task", type=Path)
    parser.add_argument("--b-causal", type=Path)
    parser.add_argument("--b-source", default="none")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text())
    if payload.get("status") != "ok":
        raise ValueError(f"incomplete artifact: {path}")
    payload["_path"] = str(path)
    return payload


def _task_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact": payload["_path"],
        "gate": payload["gate"],
        "mean_baseline_success": payload.get("mean_baseline_success"),
        "mean_candidate_success": payload.get("mean_candidate_success"),
        "mean_paired_delta": payload.get("mean_paired_delta"),
        "crossed_bootstrap_ci95": payload.get("crossed_bootstrap_ci95"),
        "aggregate_paired_wins": payload.get("aggregate_paired_wins"),
        "aggregate_paired_losses": payload.get("aggregate_paired_losses"),
        "aggregate_paired_ties": payload.get("aggregate_paired_ties"),
        "per_training_seed": payload.get("per_training_seed"),
        "selection": payload.get("selection"),
        "gate_checks": payload.get("gate_checks"),
    }


def _causal_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact": payload["_path"],
        "gate": payload["gate"],
        "value_readout": payload.get("value_readout"),
        "policy_value_beta": payload.get("policy_value_beta"),
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
        "positive_training_seeds": payload.get("positive_training_seeds"),
        "per_training_seed": payload.get("per_training_seed"),
        "gate_checks": payload.get("gate_checks"),
    }


def _route(
    *,
    name: str,
    task: dict[str, Any] | None,
    causal: dict[str, Any] | None,
    candidate: str,
    task_requirement: str,
) -> dict[str, Any]:
    task_pass = task is not None and task.get("gate") == "pass"
    causal_pass = causal is not None and causal.get("gate") == "pass"
    if task_pass and causal is None:
        raise ValueError(
            f"route {name} has a task-qualified candidate without causal audit"
        )
    checks = {
        "task_requirement": task_pass,
        "causal_value_requirement": causal_pass,
    }
    overall_pass = all(checks.values())
    gaps = [label for label, passed in checks.items() if not passed]
    if overall_pass:
        recommendation = (
            f"promote {candidate}; both the held-out task gate and the "
            "multi-training-seed causal-value gate passed"
        )
    elif task_pass:
        recommendation = (
            f"do not promote {candidate}; task performance passed but its "
            "action-conditioned value ordering is not causally established"
        )
    elif causal_pass:
        recommendation = (
            f"retain {candidate} as an audit-only value model; causal value "
            "passed but the held-out task requirement did not"
        )
    else:
        recommendation = (
            f"reject {candidate} for deployment and continue this route; "
            "neither required gate is currently satisfied"
        )
    return {
        "route": name,
        "candidate": candidate,
        "task_requirement": task_requirement,
        "overall_gate": "pass" if overall_pass else "fail",
        "checks": checks,
        "unmet_gates": gaps,
        "task": _task_evidence(task) if task is not None else None,
        "causal_value": (
            _causal_evidence(causal) if causal is not None else None
        ),
        "recommendation": recommendation,
    }


def summarize_final_routes(
    a_task: dict[str, Any],
    a_causal: dict[str, Any],
    b_task: dict[str, Any] | None,
    b_causal: dict[str, Any] | None,
    *,
    b_source: str,
) -> dict[str, Any]:
    route_a = _route(
        name="A",
        task=a_task,
        causal=a_causal,
        candidate="direct scalar-Q CQN-AS",
        task_requirement="not worse than validation-selected clean CQN-AS",
    )
    route_b = _route(
        name="B",
        task=b_task,
        causal=b_causal,
        candidate=(
            f"CQN-AS + Flow Matching ({b_source})"
            if b_source != "none"
            else "no task-qualified Flow candidate"
        ),
        task_requirement="strictly better than validation-selected clean CQN-AS",
    )
    both_pass = (
        route_a["overall_gate"] == "pass"
        and route_b["overall_gate"] == "pass"
    )
    return {
        "status": "ok",
        "protocol": {
            "task_policy": "validation-selected checkpoint on disjoint seeds",
            "task_confirmation": (
                "paired 200-seed confirmation with crossed bootstrap"
            ),
            "causal_confirmation": (
                "three training seeds with sibling-horizon interventions"
            ),
            "test_data_used_for_selection": False,
        },
        "route_a": route_a,
        "route_b": route_b,
        "research_goal_gate": "pass" if both_pass else "fail",
        "next_action": (
            "freeze both recommendations and prepare the reproducibility table"
            if both_pass
            else {
                "A": route_a["unmet_gates"],
                "B": route_b["unmet_gates"],
            }
        ),
    }


def main() -> int:
    args = parse_args()
    a_task = _load(args.a_task)
    a_causal = _load(args.a_causal)
    b_task = _load(args.b_task) if args.b_task is not None else None
    b_causal = _load(args.b_causal) if args.b_causal is not None else None
    payload = summarize_final_routes(
        a_task,
        a_causal,
        b_task,
        b_causal,
        b_source=args.b_source,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
