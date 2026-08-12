#!/usr/bin/env python3
"""Summarize the Stage-30 reward-only offline->online experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from omegaconf import OmegaConf


OFFLINE_UPDATES = 10_000
ONLINE_STEPS = (2_500, 5_000, 7_500, 10_000, 12_500, 15_000, 17_500, 20_000)
TREATMENT_RAW_STEPS = (OFFLINE_UPDATES,) + tuple(
    OFFLINE_UPDATES + step for step in ONLINE_STEPS
)
_TOL = 1e-12
ONLINE_METRICS = (
    "critic_loss",
    "behavior_candidate_fraction",
    "behavior_candidate_score",
    "greedy_candidate_score",
    "behavior_minus_greedy_q",
    "mc_lower_bound_fraction",
)
OFFLINE_METRICS = ONLINE_METRICS + (
    "demo_behavior_force_fraction",
    "demo_behavior_force_probability",
)


def _read_curve(path: Path, expected_steps: tuple[int, ...]) -> dict[int, float]:
    curve: dict[int, float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                step = int(float(row["env_steps"]))
                success = float(row["episode_success"])
            except (KeyError, TypeError, ValueError):
                continue
            if step in expected_steps:
                curve[step] = success
    missing = sorted(set(expected_steps) - set(curve))
    if missing:
        raise ValueError(f"{path} is missing validation steps {missing}")
    return curve


def _nearest_metrics(
    path: Path,
    target_step: int,
    fields: tuple[str, ...],
) -> dict[str, float]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no metric rows")
    valid_rows = []
    for row in rows:
        try:
            step = int(float(row["env_steps"]))
        except (KeyError, TypeError, ValueError):
            continue
        valid_rows.append((step, row))
    if not valid_rows:
        raise ValueError(f"{path} has no numeric env_steps")
    metric_step, row = min(
        valid_rows,
        key=lambda item: (
            abs(item[0] - target_step),
            item[0] > target_step,
        ),
    )
    missing = [field for field in fields if not row.get(field)]
    if missing:
        raise ValueError(f"{path} is missing metrics {missing}")
    return {
        "metric_raw_step": float(metric_step),
        **{field: float(row[field]) for field in fields},
    }


def _selected(curve: dict[int, float]) -> tuple[int, float]:
    return max(curve.items(), key=lambda item: (item[1], -item[0]))


def _control_arm(run_dir: Path) -> dict:
    curve = _read_curve(
        run_dir / "val50_seeds400_raw_steps.csv",
        ONLINE_STEPS,
    )
    best_step, best_success = _selected(curve)
    return {
        "run_dir": str(run_dir.resolve()),
        "online_curve": {str(step): curve[step] for step in ONLINE_STEPS},
        "best_online_step": best_step,
        "best_success": best_success,
        "selected_snapshot": str(
            (run_dir / "snapshots" / f"{best_step}_snapshot.pkl").resolve()
        ),
        "selected_metrics": _nearest_metrics(
            run_dir / "train.csv",
            best_step,
            ONLINE_METRICS,
        ),
    }


def _treatment_arm(run_dir: Path) -> dict:
    raw_curve = _read_curve(
        run_dir / "val50_seeds400_raw_steps.csv",
        TREATMENT_RAW_STEPS,
    )
    online_curve = {
        step: raw_curve[OFFLINE_UPDATES + step] for step in ONLINE_STEPS
    }
    best_step, best_success = _selected(online_curve)
    best_raw_step = OFFLINE_UPDATES + best_step
    return {
        "run_dir": str(run_dir.resolve()),
        "offline_updates": OFFLINE_UPDATES,
        "offline_endpoint_success": raw_curve[OFFLINE_UPDATES],
        "offline_endpoint_snapshot": str(
            (
                run_dir
                / "snapshots"
                / f"{OFFLINE_UPDATES}_snapshot.pkl"
            ).resolve()
        ),
        "offline_endpoint_metrics": _nearest_metrics(
            run_dir / "pretrain.csv",
            OFFLINE_UPDATES,
            OFFLINE_METRICS,
        ),
        "online_curve": {
            str(step): online_curve[step] for step in ONLINE_STEPS
        },
        "best_online_step": best_step,
        "best_raw_snapshot_step": best_raw_step,
        "best_success": best_success,
        "selected_snapshot": str(
            (
                run_dir
                / "snapshots"
                / f"{best_raw_step}_snapshot.pkl"
            ).resolve()
        ),
        "selected_metrics": _nearest_metrics(
            run_dir / "train.csv",
            best_raw_step,
            ONLINE_METRICS,
        ),
    }


def _phase_contract(run_dir: Path, seed: int, *, treatment: bool) -> dict:
    base = run_dir.parent
    phase_dir = base / "phase_configs"
    if treatment:
        paths = {
            "offline": phase_dir / f"offline_seed{seed}.yaml",
            "online": phase_dir / f"online_seed{seed}.yaml",
        }
    else:
        paths = {"online_control": phase_dir / f"control_seed{seed}.yaml"}

    phases = {}
    violations = []
    for phase, path in paths.items():
        cfg = OmegaConf.load(path)
        item = {
            "config": str(path.resolve()),
            "num_pretrain_steps": int(cfg.num_pretrain_steps),
            "num_train_frames_global_clock": int(cfg.num_train_frames),
            "batch_size": int(cfg.batch_size),
            "demo_batch_size": int(cfg.demo_batch_size),
            "demo_only_updates": bool(cfg.replay.demo_only_updates),
            "strict_demo_rl_only": bool(cfg.method.strict_demo_rl_only),
            "is_imitation_learning": bool(cfg.is_imitation_learning),
            "use_self_imitation": bool(cfg.use_self_imitation),
            "bc_lambda": float(cfg.method.bc_lambda),
            "bc_margin": float(cfg.method.bc_margin),
            "demo_fosd": bool(cfg.method.demo_fosd),
            "separate_bc_policy": bool(cfg.method.separate_bc_policy),
            "flow_policy": bool(cfg.method.flow_policy),
            "dense_return_q_target": bool(cfg.method.dense_return_q_target),
            "mc_lower_bound_target": bool(cfg.method.mc_lower_bound_target),
            "td_target_action_source": str(cfg.method.td_target_action_source),
            "demo_behavior_force_probability": float(
                cfg.method.demo_behavior_force_probability
            ),
            "include_next_action": bool(cfg.replay.include_next_action),
        }
        phases[phase] = item

        common_expected = {
            "strict_demo_rl_only": True,
            "is_imitation_learning": False,
            "use_self_imitation": False,
            "bc_lambda": 0.0,
            "bc_margin": 0.0,
            "demo_fosd": False,
            "separate_bc_policy": False,
            "flow_policy": False,
            "dense_return_q_target": False,
            "mc_lower_bound_target": True,
            "td_target_action_source": "critic_replay_max",
            "include_next_action": True,
        }
        for key, expected in common_expected.items():
            if item[key] != expected:
                violations.append(
                    f"{phase}.{key}={item[key]!r}, expected {expected!r}"
                )

    if treatment:
        offline = phases["offline"]
        online = phases["online"]
        expected_values = {
            "offline.num_pretrain_steps": (
                offline["num_pretrain_steps"],
                OFFLINE_UPDATES,
            ),
            "offline.num_train_frames_global_clock": (
                offline["num_train_frames_global_clock"],
                OFFLINE_UPDATES,
            ),
            "offline.demo_batch_size": (
                offline["demo_batch_size"],
                32,
            ),
            "offline.demo_only_updates": (
                offline["demo_only_updates"],
                True,
            ),
            "offline.demo_behavior_force_probability": (
                offline["demo_behavior_force_probability"],
                1.0,
            ),
            "online.num_pretrain_steps": (
                online["num_pretrain_steps"],
                OFFLINE_UPDATES,
            ),
            "online.num_train_frames_global_clock": (
                online["num_train_frames_global_clock"],
                OFFLINE_UPDATES + ONLINE_STEPS[-1],
            ),
            "online.demo_batch_size": (online["demo_batch_size"], 16),
            "online.demo_only_updates": (
                online["demo_only_updates"],
                False,
            ),
            "online.demo_behavior_force_probability": (
                online["demo_behavior_force_probability"],
                0.0,
            ),
        }
    else:
        control = phases["online_control"]
        expected_values = {
            "online_control.num_pretrain_steps": (
                control["num_pretrain_steps"],
                0,
            ),
            "online_control.num_train_frames_global_clock": (
                control["num_train_frames_global_clock"],
                ONLINE_STEPS[-1],
            ),
            "online_control.demo_batch_size": (
                control["demo_batch_size"],
                16,
            ),
            "online_control.demo_only_updates": (
                control["demo_only_updates"],
                False,
            ),
            "online_control.demo_behavior_force_probability": (
                control["demo_behavior_force_probability"],
                0.0,
            ),
        }
    for name, (actual, expected) in expected_values.items():
        if actual != expected:
            violations.append(
                f"{name}={actual!r}, expected {expected!r}"
            )
    if violations:
        raise ValueError("Stage-30 phase contract failed: " + "; ".join(violations))
    return {"verified": True, "phases": phases}


def _decision(
    improvements: dict[str, float],
    treatments: dict[str, dict],
) -> tuple[str, dict[str, object]]:
    mean_improvement = sum(improvements.values()) / len(improvements)
    both_nonnegative = all(delta >= -_TOL for delta in improvements.values())
    strong_pass = mean_improvement >= 0.05 - _TOL and both_nonnegative
    mixed_positive = (
        mean_improvement > _TOL
        and any(delta > _TOL for delta in improvements.values())
        and any(delta < -_TOL for delta in improvements.values())
    )
    good_20k_boundary = {
        seed: (
            arm["online_curve"]["20000"] >= 0.50 - _TOL
            and arm["online_curve"]["20000"]
            >= arm["online_curve"]["17500"] - _TOL
        )
        for seed, arm in treatments.items()
    }
    scale_continuation = (
        mean_improvement >= -0.05 - _TOL
        and any(good_20k_boundary.values())
    )
    if strong_pass:
        decision = (
            "run_seed3_then_update_count_matched_control_and_independent100"
        )
    elif mixed_positive:
        decision = "run_seed3_to_resolve_mixed_signs"
    elif scale_continuation:
        decision = "extend_offline_then_online_to50k_before_rejection"
    else:
        decision = (
            "stop_this_offline_recipe_without_full_budget_no_bc_claim"
        )
    return decision, {
        "mean_improvement": mean_improvement,
        "both_nonnegative": both_nonnegative,
        "strong_pass": strong_pass,
        "mixed_positive": mixed_positive,
        "good_20k_boundary": good_20k_boundary,
        "scale_continuation": scale_continuation,
    }


def summarize(
    control_seed1: Path,
    control_seed2: Path,
    treatment_seed1: Path,
    treatment_seed2: Path,
) -> dict:
    controls = {
        "seed1": _control_arm(control_seed1),
        "seed2": _control_arm(control_seed2),
    }
    treatments = {
        "seed1": _treatment_arm(treatment_seed1),
        "seed2": _treatment_arm(treatment_seed2),
    }
    contracts = {
        "control_seed1": _phase_contract(
            control_seed1, 1, treatment=False
        ),
        "control_seed2": _phase_contract(
            control_seed2, 2, treatment=False
        ),
        "treatment_seed1": _phase_contract(
            treatment_seed1, 1, treatment=True
        ),
        "treatment_seed2": _phase_contract(
            treatment_seed2, 2, treatment=True
        ),
    }
    improvements = {
        seed: treatments[seed]["best_success"]
        - controls[seed]["best_success"]
        for seed in treatments
    }
    control_mean = sum(
        arm["best_success"] for arm in controls.values()
    ) / len(controls)
    treatment_mean = sum(
        arm["best_success"] for arm in treatments.values()
    ) / len(treatments)
    decision, flags = _decision(improvements, treatments)
    return {
        "protocol": {
            "development_only": True,
            "official_parity_claim": False,
            "research_question": (
                "Does reward-only demo offline Q-learning before interaction "
                "improve the same no-BC online Q learner?"
            ),
            "selection_seeds": [400, 449],
            "episodes_per_checkpoint": 50,
            "online_selection_steps": list(ONLINE_STEPS),
            "checkpoint_tie_break": "earliest checkpoint",
            "offline_policy_endpoint": (
                "reported separately and excluded from online checkpoint selection"
            ),
            "treatment_offline_updates": OFFLINE_UPDATES,
            "treatment_offline_data": "100% protected expert replay",
            "treatment_offline_bootstrap": (
                "exact action_tp1 forced for demo reward Bellman continuation"
            ),
            "online_data": "16 main-replay + 16 protected-demo samples",
            "online_bootstrap": (
                "max over critic-greedy and exact replay action_tp1 candidates"
            ),
            "optimized_objective": (
                "single canonical reward-based C51 TD/MC Q cross-entropy "
                "on replayed actions"
            ),
            "not_compute_matched": (
                "treatment intentionally adds 10k offline Q updates; a pass "
                "requires a later update-count-matched ordering control"
            ),
            "mechanism_gate": (
                "two-seed selected mean improvement >=5pp and both deltas "
                "nonnegative"
            ),
            "scale_gate": (
                "selected mean trails by <=5pp and at least one 20k endpoint "
                "is >=50% and nondecreasing from 17.5k"
            ),
            "heldout_seeds_800_999": "sealed",
            "full_run_reference": {
                "online_budget": 101000,
                "fixed_endpoint": True,
                "episodes_per_training_seed": 200,
                "official_four_seed_mean": 0.646,
            },
        },
        "phase_contracts": contracts,
        "matched_online_only_controls": controls,
        "offline_then_online_treatments": treatments,
        "control_selected_mean": control_mean,
        "treatment_selected_mean": treatment_mean,
        "per_seed_improvement": improvements,
        "mean_improvement": treatment_mean - control_mean,
        "decision_flags": flags,
        "next_decision": decision,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-seed1", type=Path, required=True)
    parser.add_argument("--control-seed2", type=Path, required=True)
    parser.add_argument("--treatment-seed1", type=Path, required=True)
    parser.add_argument("--treatment-seed2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.control_seed1,
        args.control_seed2,
        args.treatment_seed1,
        args.treatment_seed2,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
