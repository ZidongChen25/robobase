#!/usr/bin/env python3
"""Summarize Stage 33 episode-persistent twin-head exploration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from scripts.summarize_cqn_no_bc_stage30 import (
    ONLINE_STEPS,
    _nearest_metrics,
    _phase_contract,
    _treatment_arm,
)
from scripts.summarize_cqn_no_bc_stage31 import _decision


_EXPLORATION_METRICS = (
    "episodic_twin_head_assignments",
    "episodic_twin_head0_rate",
    "episodic_twin_head1_rate",
)

_DECISION_MAP = {
    "run_direct_head_seed3_then_update_matched_confirmation": (
        "run_episodic_twin_seed3_matched_confirmation"
    ),
    "extend_direct_head_to50k_before_rejection": (
        "extend_episodic_twin_to50k_before_rejection"
    ),
    "stop_direct_head_and_test_pessimistic_double_q": (
        "stop_episodic_twin_and_test_supported_beam_exploration"
    ),
}


def _map_decision(decision: str) -> str:
    return _DECISION_MAP[decision]


def _exploration_contract(
    run_dir: Path,
    seed: int,
    expected: bool,
) -> dict[str, object]:
    phase_dir = run_dir.parent / "phase_configs"
    phases = {}
    for phase in ("offline", "online"):
        path = phase_dir / f"{phase}_seed{seed}.yaml"
        cfg = OmegaConf.load(path)
        actual = bool(
            cfg.method.get("episodic_twin_head_exploration", False)
        )
        if actual is not expected:
            raise ValueError(
                f"{path}: episodic_twin_head_exploration={actual}, "
                f"expected {expected}"
            )
        if not bool(cfg.method.pessimistic_twin_critic):
            raise ValueError(f"{path}: pessimistic_twin_critic must remain true")
        if bool(cfg.method.use_dueling):
            raise ValueError(f"{path}: use_dueling must remain false")
        phases[phase] = {
            "config": str(path.resolve()),
            "episodic_twin_head_exploration": actual,
            "pessimistic_twin_critic": True,
            "use_dueling": False,
        }
    return phases


def _final_exploration_metrics(run_dir: Path) -> dict[str, float]:
    return _nearest_metrics(
        run_dir / "train.csv",
        30_000,
        _EXPLORATION_METRICS,
    )


def summarize(
    stage32_summary: Path,
    explore_seed1: Path,
    explore_seed2: Path,
) -> dict:
    baseline_document = json.loads(stage32_summary.read_text())
    baseline = baseline_document["pessimistic_twin_offline_then_online"]
    exploration = {
        "seed1": _treatment_arm(explore_seed1),
        "seed2": _treatment_arm(explore_seed2),
    }
    contracts = {
        "explore_seed1": _phase_contract(
            explore_seed1, 1, treatment=True
        ),
        "explore_seed2": _phase_contract(
            explore_seed2, 2, treatment=True
        ),
    }
    exploration_contract = {
        "seed1": _exploration_contract(explore_seed1, 1, True),
        "seed2": _exploration_contract(explore_seed2, 2, True),
    }
    baseline_dirs = {
        seed: Path(arm["run_dir"]) for seed, arm in baseline.items()
    }
    baseline_contract = {
        "seed1": _exploration_contract(baseline_dirs["seed1"], 1, False),
        "seed2": _exploration_contract(baseline_dirs["seed2"], 2, False),
    }
    for arm in exploration.values():
        arm["final_exploration_metrics"] = _final_exploration_metrics(
            Path(arm["run_dir"])
        )

    improvements = {
        seed: exploration[seed]["best_success"]
        - baseline[seed]["best_success"]
        for seed in exploration
    }
    baseline_mean = sum(
        arm["best_success"] for arm in baseline.values()
    ) / len(baseline)
    exploration_mean = sum(
        arm["best_success"] for arm in exploration.values()
    ) / len(exploration)
    decision, flags = _decision(improvements, exploration)
    return {
        "protocol": {
            "development_only": True,
            "research_question": (
                "Does episode-persistent randomized twin-critic behavior "
                "exploration turn calibrated reward-only offline values into "
                "successful online data collection?"
            ),
            "only_method_difference": (
                "online behavior uses one sampled critic head per environment "
                "for the full episode; training targets and pessimistic "
                "evaluation are unchanged"
            ),
            "optimized_objective": (
                "mean of two copies of the same reward-based C51 TD/MC "
                "cross-entropy; no auxiliary or imitation loss"
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
        "stage32_summary": str(stage32_summary.resolve()),
        "phase_contracts": contracts,
        "pessimistic_twin_baseline_contract": baseline_contract,
        "episodic_twin_exploration_contract": exploration_contract,
        "pessimistic_twin_offline_then_online_baseline": baseline,
        "episodic_twin_offline_then_online": exploration,
        "baseline_selected_mean": baseline_mean,
        "exploration_selected_mean": exploration_mean,
        "per_seed_improvement": improvements,
        "mean_improvement": exploration_mean - baseline_mean,
        "decision_flags": flags,
        "next_decision": _map_decision(decision),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage32-summary", type=Path, required=True)
    parser.add_argument("--explore-seed1", type=Path, required=True)
    parser.add_argument("--explore-seed2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.stage32_summary,
        args.explore_seed1,
        args.explore_seed2,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
