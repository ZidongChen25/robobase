#!/usr/bin/env python3
"""Summarize Stage 32 clipped twin-C51 against the direct-head baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from scripts.summarize_cqn_no_bc_stage30 import _phase_contract, _treatment_arm
from scripts.summarize_cqn_no_bc_stage31 import _decision


_DECISION_MAP = {
    "run_direct_head_seed3_then_update_matched_confirmation": (
        "run_pessimistic_twin_seed3_then_update_matched_confirmation"
    ),
    "extend_direct_head_to50k_before_rejection": (
        "extend_pessimistic_twin_to50k_before_rejection"
    ),
    "stop_direct_head_and_test_pessimistic_double_q": (
        "stop_pessimistic_twin_and_revisit_supported_exploration"
    ),
}


def _map_decision(decision: str) -> str:
    return _DECISION_MAP[decision]


def _twin_contract(run_dir: Path, seed: int, expected: bool) -> dict[str, object]:
    phase_dir = run_dir.parent / "phase_configs"
    phases = {}
    for phase in ("offline", "online"):
        path = phase_dir / f"{phase}_seed{seed}.yaml"
        cfg = OmegaConf.load(path)
        actual = bool(cfg.method.get("pessimistic_twin_critic", False))
        if actual is not expected:
            raise ValueError(
                f"{path}: pessimistic_twin_critic={actual}, expected {expected}"
            )
        if bool(cfg.method.use_dueling):
            raise ValueError(f"{path}: use_dueling must remain false")
        phases[phase] = {
            "config": str(path.resolve()),
            "pessimistic_twin_critic": actual,
            "use_dueling": False,
        }
    return phases


def summarize(
    stage31_summary: Path,
    twin_seed1: Path,
    twin_seed2: Path,
) -> dict:
    baseline_document = json.loads(stage31_summary.read_text())
    baseline = baseline_document["direct_head_offline_then_online"]
    twin = {
        "seed1": _treatment_arm(twin_seed1),
        "seed2": _treatment_arm(twin_seed2),
    }
    contracts = {
        "twin_seed1": _phase_contract(twin_seed1, 1, treatment=True),
        "twin_seed2": _phase_contract(twin_seed2, 2, treatment=True),
    }
    twin_contract = {
        "seed1": _twin_contract(twin_seed1, 1, True),
        "seed2": _twin_contract(twin_seed2, 2, True),
    }
    baseline_dirs = {
        seed: Path(arm["run_dir"]) for seed, arm in baseline.items()
    }
    baseline_contract = {
        "seed1": _twin_contract(baseline_dirs["seed1"], 1, False),
        "seed2": _twin_contract(baseline_dirs["seed2"], 2, False),
    }
    improvements = {
        seed: twin[seed]["best_success"] - baseline[seed]["best_success"]
        for seed in twin
    }
    baseline_mean = sum(
        arm["best_success"] for arm in baseline.values()
    ) / len(baseline)
    twin_mean = sum(arm["best_success"] for arm in twin.values()) / len(twin)
    decision, flags = _decision(improvements, twin)
    mapped_decision = _map_decision(decision)
    return {
        "protocol": {
            "development_only": True,
            "research_question": (
                "Does clipped agreement between two direct C51 critics "
                "suppress unsupported actions after offline reward learning?"
            ),
            "only_method_difference": (
                "one direct C51 critic -> two independently initialized "
                "critics with min-Q selection and clipped target distribution"
            ),
            "optimized_objective": (
                "mean of two copies of the same reward-based C51 TD/MC "
                "cross-entropy; no auxiliary or imitation loss"
            ),
            "selection_seeds": [400, 449],
            "episodes_per_checkpoint": 50,
            "heldout_seeds_800_999": "sealed",
            "mechanism_gate": (
                "mean selected gain >=10pp, both seed deltas nonnegative, "
                "and at least one seed >=20%"
            ),
            "scale_gate": (
                "mean selected gain >=5pp and any 20k endpoint >=20% and "
                "nondecreasing from 17.5k"
            ),
        },
        "stage31_summary": str(stage31_summary.resolve()),
        "phase_contracts": contracts,
        "direct_head_baseline_contract": baseline_contract,
        "pessimistic_twin_contract": twin_contract,
        "direct_head_offline_then_online_baseline": baseline,
        "pessimistic_twin_offline_then_online": twin,
        "baseline_selected_mean": baseline_mean,
        "twin_selected_mean": twin_mean,
        "per_seed_improvement": improvements,
        "mean_improvement": twin_mean - baseline_mean,
        "decision_flags": flags,
        "next_decision": mapped_decision,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage31-summary", type=Path, required=True)
    parser.add_argument("--twin-seed1", type=Path, required=True)
    parser.add_argument("--twin-seed2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.stage31_summary, args.twin_seed1, args.twin_seed2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
