#!/usr/bin/env python3
"""Summarize the full-scale matched Stage-35 one-step plus four-step test."""

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
_METRICS = (
    "critic_loss",
    "one_step_critic_loss",
    "auxiliary_critic_loss",
    "auxiliary_td_loss_weight",
    "behavior_candidate_fraction",
    "auxiliary_behavior_candidate_fraction",
    "mc_lower_bound_fraction",
    "auxiliary_mc_lower_bound_fraction",
    "behavior_minus_greedy_q",
    "auxiliary_behavior_minus_greedy_q",
    "twin_q_disagreement",
)


def _run_dir(base: Path, seed: int, arm: str) -> Path:
    return base / f"seed{seed}" / arm / f"offline_twin_seed{seed}"


def _arm(run_dir: Path) -> dict[str, object]:
    raw_curve = _read_curve(
        run_dir / "val50_seeds400_full_raw_steps.csv",
        RAW_STEPS,
    )
    online_curve = {
        step: raw_curve[OFFLINE_UPDATES + step] for step in ONLINE_STEPS
    }
    best_step, best_success = _selected(online_curve)
    best_raw_step = OFFLINE_UPDATES + best_step
    return {
        "run_dir": str(run_dir.resolve()),
        "offline_endpoint_success": raw_curve[OFFLINE_UPDATES],
        "offline_endpoint_snapshot": str(
            (run_dir / "snapshots" / "10000_snapshot.pkl").resolve()
        ),
        "online_curve": {
            str(step): online_curve[step] for step in ONLINE_STEPS
        },
        "best_online_step": best_step,
        "best_raw_snapshot_step": best_raw_step,
        "best_success": best_success,
        "selected_snapshot": str(
            (run_dir / "snapshots" / f"{best_raw_step}_snapshot.pkl").resolve()
        ),
        "selected_metrics": _nearest_metrics(
            run_dir / "train.csv", best_raw_step, _METRICS
        ),
        "final_101k_success": online_curve[101_000],
        "final_snapshot": str(
            (run_dir / "snapshots" / "111000_snapshot.pkl").resolve()
        ),
        "final_metrics": _nearest_metrics(
            run_dir / "train.csv", 111_000, _METRICS
        ),
    }


def _phase_contract(
    base: Path,
    seed: int,
    arm: str,
    expected_weight: float,
) -> dict[str, object]:
    arm_dir = base / f"seed{seed}" / arm
    phases = {}
    violations = []
    for phase in ("offline", "online"):
        path = arm_dir / "phase_configs" / f"{phase}_seed{seed}.yaml"
        cfg = OmegaConf.load(path)
        item = {
            "config": str(path.resolve()),
            "num_pretrain_steps": int(cfg.num_pretrain_steps),
            "num_train_frames_global_clock": int(cfg.num_train_frames),
            "demo_batch_size": int(cfg.demo_batch_size),
            "demo_only_updates": bool(cfg.replay.demo_only_updates),
            "nstep": int(cfg.replay.nstep),
            "auxiliary_nstep": int(cfg.replay.auxiliary_nstep),
            "include_tp1": bool(cfg.replay.include_tp1),
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
                cfg.method.episodic_twin_head_exploration
            ),
            "twin_rollout_beam_width": int(
                cfg.method.twin_rollout_beam_width
            ),
            "mc_lower_bound_target": bool(cfg.method.mc_lower_bound_target),
            "td_target_action_source": str(
                cfg.method.td_target_action_source
            ),
            "demo_behavior_force_probability": float(
                cfg.method.demo_behavior_force_probability
            ),
            "auxiliary_td_loss_weight": float(
                cfg.method.auxiliary_td_loss_weight
            ),
        }
        phases[phase] = item
        expected_common = {
            "nstep": 1,
            "auxiliary_nstep": 4,
            "include_tp1": True,
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
            "episodic_twin_head_exploration": False,
            "twin_rollout_beam_width": 1,
            "mc_lower_bound_target": True,
            "td_target_action_source": "critic_replay_max",
            "auxiliary_td_loss_weight": expected_weight,
        }
        for key, expected in expected_common.items():
            if item[key] != expected:
                violations.append(
                    f"{phase}.{key}={item[key]!r}, expected {expected!r}"
                )

    expected_phases = {
        "offline.num_pretrain_steps": (
            phases["offline"]["num_pretrain_steps"], OFFLINE_UPDATES
        ),
        "offline.num_train_frames_global_clock": (
            phases["offline"]["num_train_frames_global_clock"], OFFLINE_UPDATES
        ),
        "offline.demo_batch_size": (
            phases["offline"]["demo_batch_size"], 32
        ),
        "offline.demo_only_updates": (
            phases["offline"]["demo_only_updates"], True
        ),
        "offline.demo_behavior_force_probability": (
            phases["offline"]["demo_behavior_force_probability"], 1.0
        ),
        "online.num_pretrain_steps": (
            phases["online"]["num_pretrain_steps"], OFFLINE_UPDATES
        ),
        "online.num_train_frames_global_clock": (
            phases["online"]["num_train_frames_global_clock"],
            OFFLINE_UPDATES + 101_000,
        ),
        "online.demo_batch_size": (
            phases["online"]["demo_batch_size"], 16
        ),
        "online.demo_only_updates": (
            phases["online"]["demo_only_updates"], False
        ),
        "online.demo_behavior_force_probability": (
            phases["online"]["demo_behavior_force_probability"], 0.0
        ),
    }
    for name, (actual, expected) in expected_phases.items():
        if actual != expected:
            violations.append(f"{name}={actual!r}, expected {expected!r}")
    if violations:
        raise ValueError("Stage-35 phase contract failed: " + "; ".join(violations))
    return {"verified": True, "phases": phases}


def _decision(
    controls: dict[str, dict[str, object]],
    treatments: dict[str, dict[str, object]],
) -> tuple[str, dict[str, object]]:
    deltas = {
        seed: float(treatments[seed]["best_success"])
        - float(controls[seed]["best_success"])
        for seed in treatments
    }
    treatment_mean = sum(
        float(arm["best_success"]) for arm in treatments.values()
    ) / len(treatments)
    mean_delta = sum(deltas.values()) / len(deltas)
    both_nonnegative = all(delta >= -_TOL for delta in deltas.values())
    any_positive = any(delta > _TOL for delta in deltas.values())
    strong_gate = (
        treatment_mean >= 0.40 - _TOL
        and mean_delta >= 0.10 - _TOL
        and both_nonnegative
    )
    weak_gate = (
        treatment_mean >= 0.20 - _TOL
        and mean_delta >= -_TOL
        and any_positive
    )
    if strong_gate:
        decision = "add_seeds3_4_then_run_four_seed_sealed_endpoint_comparison"
    elif weak_gate:
        decision = "run_third_matched_full_scale_seed_before_decision"
    else:
        decision = "reject_simultaneous_one_plus_four_mechanism"
    return decision, {
        "per_seed_selected_delta": deltas,
        "treatment_selected_mean": treatment_mean,
        "mean_selected_delta": mean_delta,
        "both_seed_deltas_nonnegative": both_nonnegative,
        "any_seed_delta_strictly_positive": any_positive,
        "strong_gate": strong_gate,
        "weak_gate": weak_gate,
    }


def summarize(base: Path) -> dict[str, object]:
    controls = {
        f"seed{seed}": _arm(_run_dir(base, seed, "control"))
        for seed in (1, 2)
    }
    treatments = {
        f"seed{seed}": _arm(_run_dir(base, seed, "treatment"))
        for seed in (1, 2)
    }
    control_mean = sum(
        float(arm["best_success"]) for arm in controls.values()
    ) / len(controls)
    treatment_mean = sum(
        float(arm["best_success"]) for arm in treatments.values()
    ) / len(treatments)
    decision, flags = _decision(controls, treatments)
    return {
        "protocol": {
            "research_question": (
                "Does a normalized simultaneous one-step plus four-step "
                "reward Bellman objective improve offline-to-online no-BC "
                "CQN-AS over the matched one-step objective?"
            ),
            "training_seeds": [1, 2],
            "offline_updates": OFFLINE_UPDATES,
            "online_environment_steps": 101_000,
            "selection_seeds": [400, 449],
            "episodes_per_checkpoint": 50,
            "online_selection_steps": list(ONLINE_STEPS),
            "checkpoint_tie_break": "earliest checkpoint",
            "offline_endpoint": "reported separately; excluded from selection",
            "fixed_endpoint": "raw step 111000 reported without reselection",
            "heldout_seeds_800_999": "sealed",
            "only_arm_difference": (
                "auxiliary_td_loss_weight: 0.0 versus 1.0; both replay "
                "buffers expose the exact same one-step and four-step fields"
            ),
            "control_objective": "L_C51_1step",
            "treatment_objective": "0.5 * (L_C51_1step + L_C51_4step)",
            "gradient_sources": (
                "reward-derived clipped double-Q Bellman and Monte-Carlo "
                "return targets only"
            ),
            "forbidden_and_absent": (
                "BC, FOSD, margin, likelihood, action regression, actor, "
                "conservative loss, representation auxiliary loss"
            ),
            "strong_gate": (
                "treatment selected mean >=40%, mean paired gain >=10pp, "
                "and both seed deltas nonnegative"
            ),
            "weak_gate": (
                "treatment selected mean >=20%, mean gain nonnegative, "
                "and at least one strictly positive seed delta"
            ),
        },
        "phase_contracts": {
            f"seed{seed}_{arm}": _phase_contract(
                base,
                seed,
                arm,
                0.0 if arm == "control" else 1.0,
            )
            for seed in (1, 2)
            for arm in ("control", "treatment")
        },
        "one_step_control": controls,
        "one_plus_four_treatment": treatments,
        "control_selected_mean": control_mean,
        "treatment_selected_mean": treatment_mean,
        "mean_selected_improvement": treatment_mean - control_mean,
        "decision_flags": flags,
        "next_decision": decision,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.base)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
