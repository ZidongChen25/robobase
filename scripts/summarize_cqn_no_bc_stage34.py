#!/usr/bin/env python3
"""Summarize Stage 34 joint top-two beam rollout search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from scripts.summarize_cqn_no_bc_stage30 import (
    ONLINE_STEPS,
    _phase_contract,
    _treatment_arm,
)
from scripts.summarize_cqn_no_bc_stage31 import _decision
from scripts.summarize_cqn_no_bc_stage33 import _final_exploration_metrics


_DECISION_MAP = {
    "run_direct_head_seed3_then_update_matched_confirmation": (
        "run_joint_beam_seed3_matched_confirmation"
    ),
    "extend_direct_head_to50k_before_rejection": (
        "extend_joint_beam_to50k_before_rejection"
    ),
    "stop_direct_head_and_test_pessimistic_double_q": (
        "stop_joint_beam_primary_and_continue_registered_scale_sentinel"
    ),
}


def _map_decision(decision: str) -> str:
    return _DECISION_MAP[decision]


def _beam_contract(
    run_dir: Path,
    seed: int,
    expected_width: int,
) -> dict[str, object]:
    phase_dir = run_dir.parent / "phase_configs"
    phases = {}
    for phase in ("offline", "online"):
        path = phase_dir / f"{phase}_seed{seed}.yaml"
        cfg = OmegaConf.load(path)
        width = int(cfg.method.get("twin_rollout_beam_width", 1))
        if width != expected_width:
            raise ValueError(
                f"{path}: twin_rollout_beam_width={width}, "
                f"expected {expected_width}"
            )
        if not bool(cfg.method.pessimistic_twin_critic):
            raise ValueError(f"{path}: pessimistic_twin_critic must remain true")
        if not bool(cfg.method.episodic_twin_head_exploration):
            raise ValueError(
                f"{path}: episodic_twin_head_exploration must remain true"
            )
        if bool(cfg.method.autoregressive_action_dims):
            raise ValueError(
                f"{path}: autoregressive_action_dims must remain false"
            )
        if bool(cfg.method.use_dueling):
            raise ValueError(f"{path}: use_dueling must remain false")
        phases[phase] = {
            "config": str(path.resolve()),
            "twin_rollout_beam_width": width,
            "pessimistic_twin_critic": True,
            "episodic_twin_head_exploration": True,
            "autoregressive_action_dims": False,
            "use_dueling": False,
        }
    return phases


def summarize(
    stage33_summary: Path,
    beam_seed1: Path,
    beam_seed2: Path,
) -> dict:
    baseline_document = json.loads(stage33_summary.read_text())
    baseline = baseline_document["episodic_twin_offline_then_online"]
    beam = {
        "seed1": _treatment_arm(beam_seed1),
        "seed2": _treatment_arm(beam_seed2),
    }
    phase_contracts = {
        "beam_seed1": _phase_contract(beam_seed1, 1, treatment=True),
        "beam_seed2": _phase_contract(beam_seed2, 2, treatment=True),
    }
    baseline_contract = {
        seed: _beam_contract(Path(arm["run_dir"]), int(seed[-1]), 1)
        for seed, arm in baseline.items()
    }
    beam_contract = {
        "seed1": _beam_contract(beam_seed1, 1, 8),
        "seed2": _beam_contract(beam_seed2, 2, 8),
    }
    for arm in beam.values():
        arm["final_exploration_metrics"] = _final_exploration_metrics(
            Path(arm["run_dir"])
        )

    improvements = {
        seed: beam[seed]["best_success"] - baseline[seed]["best_success"]
        for seed in beam
    }
    baseline_mean = sum(x["best_success"] for x in baseline.values()) / 2
    beam_mean = sum(x["best_success"] for x in beam.values()) / 2
    decision, flags = _decision(improvements, beam)
    return {
        "protocol": {
            "development_only": True,
            "research_question": (
                "Does joint top-two beam rollout maximization turn calibrated "
                "reward-only twin values into a successful policy?"
            ),
            "only_method_difference": (
                "rollout C2F maximization uses a width-8 complete-assignment "
                "beam; Bellman/MC training and sampled episodic head are unchanged"
            ),
            "optimized_objective": (
                "mean of two reward-based C51 TD/MC cross-entropies only; "
                "no actor, auxiliary, conservative, or imitation loss"
            ),
            "offline_updates": 10_000,
            "online_steps": list(ONLINE_STEPS),
            "training_seeds": [1, 2],
            "selection_seeds": [400, 449],
            "episodes_per_checkpoint": 50,
            "checkpoint_tie_break": "earliest checkpoint",
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
        "stage33_summary": str(stage33_summary.resolve()),
        "phase_contracts": phase_contracts,
        "episodic_twin_baseline_contract": baseline_contract,
        "joint_beam_contract": beam_contract,
        "episodic_twin_baseline": baseline,
        "joint_beam_offline_then_online": beam,
        "baseline_selected_mean": baseline_mean,
        "beam_selected_mean": beam_mean,
        "per_seed_improvement": improvements,
        "mean_improvement": beam_mean - baseline_mean,
        "decision_flags": flags,
        "next_decision": _map_decision(decision),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage33-summary", type=Path, required=True)
    parser.add_argument("--beam-seed1", type=Path, required=True)
    parser.add_argument("--beam-seed2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.stage33_summary, args.beam_seed1, args.beam_seed2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
