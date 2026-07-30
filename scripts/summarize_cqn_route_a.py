#!/usr/bin/env python3
"""Produce the final two-level conclusion for CQN-AS value authenticity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structured-gate", required=True, type=Path)
    parser.add_argument("--policy-calibration", required=True, type=Path)
    parser.add_argument("--direct-ensemble-gate", required=True, type=Path)
    parser.add_argument("--blend-gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text())
    if payload.get("status") != "ok":
        raise ValueError(f"incomplete artifact: {path}")
    payload["_path"] = str(path)
    return payload


def summarize_route_a(
    structured: dict[str, Any],
    policy: dict[str, Any],
    direct: dict[str, Any],
    blend: dict[str, Any],
) -> dict[str, Any]:
    structured_checks = structured.get("checks", {})
    structured_pass = (
        structured.get("gate") == "pass"
        and bool(structured_checks)
        and all(bool(value) for value in structured_checks.values())
    )
    bc_success = float(policy["bc"]["episode_success"])
    exact_fallback_variants = []
    for label, row in policy.get("variants", {}).items():
        if (
            int(row.get("total_applied_overrides", -1)) == 0
            and float(row["episode_success"]) == bc_success
            and float(row["success_delta_vs_bc"]) == 0.0
            and int(row["paired_wins"]) == 0
            and int(row["paired_losses"]) == 0
        ):
            exact_fallback_variants.append(label)
    clean_fallback_reproduced = bool(exact_fallback_variants)
    direct_closed = direct.get("gate") == "fail"
    blend_closed = blend.get("gate") == "fail"
    safe_audit_pass = structured_pass and clean_fallback_reproduced
    action_facing_pass = (
        structured_pass
        and policy.get("gate_passed") is True
        and direct.get("gate") == "pass"
        and blend.get("gate") == "pass"
    )
    return {
        "status": "ok",
        "route": "A",
        "safe_audit_gate": "pass" if safe_audit_pass else "fail",
        "action_facing_gate": "pass" if action_facing_pass else "fail",
        "checks": {
            "structured_counterfactual_value_passed": structured_pass,
            "clean_policy_fallback_reproduced": clean_fallback_reproduced,
            "direct_ensemble_deployment_closed": direct_closed,
            "direct_proxy_blend_deployment_closed": blend_closed,
        },
        "structured_value": {
            "artifact": structured.get("_path"),
            "model": structured.get("model"),
            "metrics": structured.get("metrics"),
            "seed_bootstrap": structured.get("seed_bootstrap"),
            "num_checks": len(structured_checks),
        },
        "task_noninferiority": {
            "artifact": policy.get("_path"),
            "baseline_success": bc_success,
            "exact_fallback_variants": exact_fallback_variants,
            "semantics": (
                "audit-only value; executed policy remains exact clean CQN-AS"
            ),
        },
        "closed_action_facing_candidates": {
            "direct_ensemble": {
                "artifact": direct.get("_path"),
                "gate": direct.get("gate"),
                "metrics": direct.get("metrics"),
            },
            "direct_proxy_blend": {
                "artifact": blend.get("_path"),
                "gate": blend.get("gate"),
                "selected_blend": blend.get("selected_blend"),
                "metrics": blend.get("metrics"),
            },
        },
        "recommendation": {
            "policy": "validation-selected clean CQN-AS",
            "value": "structured-delta causal audit sidecar",
            "deployment": "do not route the current sidecar into actions",
            "claim": (
                "counterfactual value is reproducible and task performance "
                "is non-inferior because the clean action path is unchanged; "
                "policy improvement from value remains unproven"
            ),
        },
    }


def main() -> int:
    args = parse_args()
    payload = summarize_route_a(
        _load(args.structured_gate),
        _load(args.policy_calibration),
        _load(args.direct_ensemble_gate),
        _load(args.blend_gate),
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
