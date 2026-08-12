#!/usr/bin/env python3
"""Summarize the matched 101k-online Stage-33 scale sentinel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from scripts.summarize_cqn_no_bc_stage30 import (
    _nearest_metrics,
    _read_curve,
    _selected,
)


OFFLINE_UPDATES = 10_000
ONLINE_STEPS = tuple(range(10_000, 100_001, 10_000)) + (101_000,)
RAW_STEPS = (OFFLINE_UPDATES,) + tuple(
    OFFLINE_UPDATES + step for step in ONLINE_STEPS
)
_TOL = 1e-12
_FINAL_METRICS = (
    "critic_loss",
    "episodic_twin_head_assignments",
    "episodic_twin_head0_rate",
    "episodic_twin_head1_rate",
)


def _scale_decision(
    baseline_best: float,
    treatment_best: float,
) -> tuple[str, dict[str, object]]:
    delta = treatment_best - baseline_best
    strong_scale_signal = (
        treatment_best >= 0.40 - _TOL and delta >= 0.10 - _TOL
    )
    weak_scale_signal = (
        treatment_best >= 0.20 - _TOL and delta >= -_TOL
    )
    if strong_scale_signal:
        decision = "launch_stage33_full_multiseed_101k_protocol"
    elif weak_scale_signal:
        decision = "run_second_matched_scale_seed_before_decision"
    else:
        decision = "scale_sentinel_does_not_support_episodic_twin"
    return decision, {
        "paired_best_delta": delta,
        "treatment_best_at_least_40pct": treatment_best >= 0.40 - _TOL,
        "treatment_best_at_least_20pct": treatment_best >= 0.20 - _TOL,
        "strong_scale_signal": strong_scale_signal,
        "weak_scale_signal": weak_scale_signal,
    }


def _scale_phase_contract(
    run_dir: Path,
    *,
    expected_exploration: bool,
) -> dict[str, object]:
    phase_dir = run_dir.parent / "phase_configs"
    phases = {}
    violations = []
    for phase in ("offline", "online"):
        path = phase_dir / f"{phase}_seed4.yaml"
        cfg = OmegaConf.load(path)
        item = {
            "config": str(path.resolve()),
            "num_pretrain_steps": int(cfg.num_pretrain_steps),
            "num_train_frames_global_clock": int(cfg.num_train_frames),
            "demo_batch_size": int(cfg.demo_batch_size),
            "demo_only_updates": bool(cfg.replay.demo_only_updates),
            "include_next_action": bool(cfg.replay.include_next_action),
            "strict_demo_rl_only": bool(cfg.method.strict_demo_rl_only),
            "is_imitation_learning": bool(cfg.is_imitation_learning),
            "use_self_imitation": bool(cfg.use_self_imitation),
            "bc_lambda": float(cfg.method.bc_lambda),
            "bc_margin": float(cfg.method.bc_margin),
            "demo_fosd": bool(cfg.method.demo_fosd),
            "separate_bc_policy": bool(cfg.method.separate_bc_policy),
            "flow_policy": bool(cfg.method.flow_policy),
            "critic_lambda": float(cfg.method.critic_lambda),
            "use_dueling": bool(cfg.method.use_dueling),
            "pessimistic_twin_critic": bool(
                cfg.method.pessimistic_twin_critic
            ),
            "episodic_twin_head_exploration": bool(
                cfg.method.get("episodic_twin_head_exploration", False)
            ),
            "mc_lower_bound_target": bool(cfg.method.mc_lower_bound_target),
            "td_target_action_source": str(
                cfg.method.td_target_action_source
            ),
            "demo_behavior_force_probability": float(
                cfg.method.demo_behavior_force_probability
            ),
        }
        phases[phase] = item
        expected_common = {
            "include_next_action": True,
            "strict_demo_rl_only": True,
            "is_imitation_learning": False,
            "use_self_imitation": False,
            "bc_lambda": 0.0,
            "bc_margin": 0.0,
            "demo_fosd": False,
            "separate_bc_policy": False,
            "flow_policy": False,
            "critic_lambda": 1.0,
            "use_dueling": False,
            "pessimistic_twin_critic": True,
            "episodic_twin_head_exploration": expected_exploration,
            "mc_lower_bound_target": True,
            "td_target_action_source": "critic_replay_max",
        }
        for key, expected in expected_common.items():
            if item[key] != expected:
                violations.append(
                    f"{phase}.{key}={item[key]!r}, expected {expected!r}"
                )

    expected_phase = {
        "offline.num_pretrain_steps": (
            phases["offline"]["num_pretrain_steps"],
            OFFLINE_UPDATES,
        ),
        "offline.num_train_frames_global_clock": (
            phases["offline"]["num_train_frames_global_clock"],
            OFFLINE_UPDATES,
        ),
        "offline.demo_batch_size": (
            phases["offline"]["demo_batch_size"],
            32,
        ),
        "offline.demo_only_updates": (
            phases["offline"]["demo_only_updates"],
            True,
        ),
        "offline.demo_behavior_force_probability": (
            phases["offline"]["demo_behavior_force_probability"],
            1.0,
        ),
        "online.num_pretrain_steps": (
            phases["online"]["num_pretrain_steps"],
            OFFLINE_UPDATES,
        ),
        "online.num_train_frames_global_clock": (
            phases["online"]["num_train_frames_global_clock"],
            OFFLINE_UPDATES + 101_000,
        ),
        "online.demo_batch_size": (
            phases["online"]["demo_batch_size"],
            16,
        ),
        "online.demo_only_updates": (
            phases["online"]["demo_only_updates"],
            False,
        ),
        "online.demo_behavior_force_probability": (
            phases["online"]["demo_behavior_force_probability"],
            0.0,
        ),
    }
    for name, (actual, expected) in expected_phase.items():
        if actual != expected:
            violations.append(f"{name}={actual!r}, expected {expected!r}")
    if violations:
        raise ValueError("Stage-33 scale contract failed: " + "; ".join(violations))
    return {"verified": True, "phases": phases}


def _scale_arm(run_dir: Path) -> dict[str, object]:
    raw_curve = _read_curve(
        run_dir / "val50_seeds400_scale_raw_steps.csv",
        RAW_STEPS,
    )
    online_curve = {
        step: raw_curve[OFFLINE_UPDATES + step] for step in ONLINE_STEPS
    }
    best_step, best_success = _selected(online_curve)
    best_raw_step = OFFLINE_UPDATES + best_step
    return {
        "run_dir": str(run_dir.resolve()),
        "offline_updates": OFFLINE_UPDATES,
        "online_environment_steps": 101_000,
        "offline_endpoint_success": raw_curve[OFFLINE_UPDATES],
        "online_curve": {
            str(step): online_curve[step] for step in ONLINE_STEPS
        },
        "best_online_step": best_step,
        "best_raw_snapshot_step": best_raw_step,
        "best_success": best_success,
        "selected_snapshot": str(
            (run_dir / "snapshots" / f"{best_raw_step}_snapshot.pkl").resolve()
        ),
        "final_101k_success": online_curve[101_000],
        "final_snapshot": str(
            (run_dir / "snapshots" / "111000_snapshot.pkl").resolve()
        ),
        "final_metrics": _nearest_metrics(
            run_dir / "train.csv",
            111_000,
            _FINAL_METRICS,
        ),
    }


def summarize(
    primary_summary: Path,
    baseline: Path,
    treatment: Path,
) -> dict[str, object]:
    primary = json.loads(primary_summary.read_text())
    baseline_arm = _scale_arm(baseline)
    treatment_arm = _scale_arm(treatment)
    decision, flags = _scale_decision(
        float(baseline_arm["best_success"]),
        float(treatment_arm["best_success"]),
    )
    return {
        "protocol": {
            "role": "pre_registered_matched_full_scale_sentinel",
            "primary_decision_frozen_before_sentinel_evaluation": True,
            "does_not_change_primary_seed1_seed2_gate": True,
            "training_seed": 4,
            "offline_updates": OFFLINE_UPDATES,
            "online_environment_steps": 101_000,
            "selection_seeds": [400, 449],
            "episodes_per_checkpoint": 50,
            "online_selection_steps": list(ONLINE_STEPS),
            "checkpoint_tie_break": "earliest checkpoint",
            "heldout_seeds_800_999": "sealed",
            "only_method_difference": (
                "deterministic pessimistic twin behavior versus one sampled "
                "critic head held fixed for each online episode"
            ),
            "optimized_objective": (
                "the same twin reward-based C51 TD/MC cross-entropy in both "
                "arms; no imitation or auxiliary loss"
            ),
            "strong_scale_gate": (
                "treatment validation-best >=40% and paired gain >=10pp"
            ),
            "weak_scale_gate": (
                "treatment validation-best >=20% and noninferior to baseline"
            ),
        },
        "primary_summary": str(primary_summary.resolve()),
        "primary_next_decision": primary["next_decision"],
        "primary_decision_flags": primary["decision_flags"],
        "phase_contracts": {
            "baseline": _scale_phase_contract(
                baseline, expected_exploration=False
            ),
            "treatment": _scale_phase_contract(
                treatment, expected_exploration=True
            ),
        },
        "deterministic_twin_seed4": baseline_arm,
        "episodic_twin_seed4": treatment_arm,
        "paired_best_improvement": (
            treatment_arm["best_success"] - baseline_arm["best_success"]
        ),
        "decision_flags": flags,
        "next_decision": decision,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-summary", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.primary_summary,
        args.baseline,
        args.treatment,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
