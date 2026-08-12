#!/usr/bin/env python3
"""Summarize the frozen, matched seed-3 confirmation for Stage 32."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.summarize_cqn_no_bc_stage30 import _phase_contract, _treatment_arm
from scripts.summarize_cqn_no_bc_stage32 import _twin_contract


_TOL = 1e-12


def confirmation_decision(
    *,
    primary_mechanism_pass: bool,
    direct_success: float,
    twin_success: float,
) -> tuple[str, dict[str, object]]:
    delta = twin_success - direct_success
    confirmation_pass = (
        primary_mechanism_pass
        and delta >= -_TOL
        and twin_success >= 0.20 - _TOL
    )
    if not primary_mechanism_pass:
        decision = "supplemental_only_primary_gate_failed"
    elif confirmation_pass:
        decision = "advance_pessimistic_twin_to_full_101k_protocol"
    else:
        decision = "pessimistic_twin_seed3_confirmation_failed"
    return decision, {
        "primary_mechanism_pass": primary_mechanism_pass,
        "seed3_delta": delta,
        "seed3_twin_at_least_20pct": twin_success >= 0.20 - _TOL,
        "seed3_noninferior_to_direct": delta >= -_TOL,
        "confirmation_pass": confirmation_pass,
    }


def summarize(primary_summary: Path, direct: Path, twin: Path) -> dict:
    primary = json.loads(primary_summary.read_text())
    direct_arm = _treatment_arm(direct)
    twin_arm = _treatment_arm(twin)
    direct_phase = _phase_contract(direct, 3, treatment=True)
    twin_phase = _phase_contract(twin, 3, treatment=True)
    direct_architecture = _twin_contract(direct, 3, False)
    twin_architecture = _twin_contract(twin, 3, True)
    primary_pass = bool(primary["decision_flags"]["mechanism_pass"])
    decision, flags = confirmation_decision(
        primary_mechanism_pass=primary_pass,
        direct_success=float(direct_arm["best_success"]),
        twin_success=float(twin_arm["best_success"]),
    )
    return {
        "protocol": {
            "role": "pre_registered_seed3_matched_confirmation",
            "primary_decision_frozen_before_confirmation_evaluation": True,
            "does_not_change_primary_seed1_seed2_gate": True,
            "training_seed": 3,
            "selection_seeds": [400, 449],
            "episodes_per_checkpoint": 50,
            "heldout_seeds_800_999": "sealed",
            "only_method_difference": (
                "one direct C51 critic versus two independently initialized "
                "critics with min-Q action selection and clipped target"
            ),
            "optimized_objective": (
                "reward-based C51 TD/MC cross-entropy only; no actor, BC, "
                "margin, likelihood, or conservative auxiliary loss"
            ),
            "confirmation_gate": (
                "primary mechanism pass AND seed3 twin >= direct AND "
                "seed3 twin validation-best >=20%"
            ),
        },
        "primary_summary": str(primary_summary.resolve()),
        "primary_next_decision": primary["next_decision"],
        "primary_decision_flags": primary["decision_flags"],
        "phase_contracts": {
            "direct": direct_phase,
            "twin": twin_phase,
        },
        "architecture_contracts": {
            "direct": direct_architecture,
            "twin": twin_architecture,
        },
        "direct_seed3": direct_arm,
        "pessimistic_twin_seed3": twin_arm,
        "seed3_improvement": (
            twin_arm["best_success"] - direct_arm["best_success"]
        ),
        "decision_flags": flags,
        "next_decision": decision,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-summary", type=Path, required=True)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--twin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.primary_summary, args.direct, args.twin)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
