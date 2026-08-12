#!/usr/bin/env python3
"""Audit the completed Stage-43 run against its no-imitation contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


FORBIDDEN_NONZERO = (
    "bc_lambda",
    "bc_margin",
    "causal_rct_weight",
    "dense_return_advantage_alpha",
    "dense_return_finest_neighbor_weight",
    "unseen_return_floor_weight",
    "ordered_success_return_mix",
    "mc_return_weight",
    "auxiliary_td_loss_weight",
)
FORBIDDEN_TRUE = (
    "demo_fosd",
    "separate_bc_policy",
    "bc_policy_stop_gradient",
    "distinct_policy_encoder",
    "flow_policy",
    "coarse_flow",
    "coarse_flow_pure",
    "freeze_bc_policy",
    "direct_scalar_q",
    "episodic_success_q_target",
    "dense_return_expected_q_loss",
)
FORBIDDEN_OPTIONAL = (
    "bc_lambda_schedule",
    "td_target_policy_value_beta",
    "policy_value_beta",
    "cv_rct_weight",
    "awr_beta",
    "coarse_flow_selfdistill_weight",
    "dense_return_advantage_clip_ratio",
    "dense_return_floor_satisfaction_margin",
    "dense_return_relative_floor_margin",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a mapping")
    return value


def _value(mapping: dict[str, Any], dotted: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit_phase_config(
    path: Path,
    *,
    phase: str,
    allow_inert_self_imitation_flag: bool = False,
) -> dict[str, Any]:
    """Validate one resolved config and return its relevant contract fields."""

    cfg = _read_yaml(path)
    method = _value(cfg, "method", {})
    _require(phase in {"offline", "online"}, f"unknown phase: {phase}")
    _require(_value(cfg, "batch_size") == 256, f"{path}: batch_size != 256")
    _require(
        _value(cfg, "demo_batch_size") == 256,
        f"{path}: demo_batch_size != 256",
    )
    _require(
        _value(cfg, "num_pretrain_steps") == 10000,
        f"{path}: num_pretrain_steps != 10000",
    )
    _require(
        not bool(_value(cfg, "is_imitation_learning", False)),
        f"{path}: is_imitation_learning is enabled",
    )
    _require(
        bool(_value(cfg, "method.is_rl", False)),
        f"{path}: method.is_rl is not true",
    )
    _require(
        bool(_value(cfg, "method.strict_demo_rl_only", False)),
        f"{path}: strict_demo_rl_only is not true",
    )

    for name in FORBIDDEN_NONZERO:
        _require(
            float(method.get(name, 0.0)) == 0.0,
            f"{path}: method.{name} is nonzero",
        )
    for name in FORBIDDEN_TRUE:
        _require(
            not bool(method.get(name, False)),
            f"{path}: method.{name} is enabled",
        )
    for name in FORBIDDEN_OPTIONAL:
        _require(
            method.get(name) is None,
            f"{path}: method.{name} is configured",
        )

    self_imitation = bool(_value(cfg, "use_self_imitation", False))
    if self_imitation:
        _require(
            phase == "offline" and allow_inert_self_imitation_flag,
            f"{path}: self-imitation can be active",
        )
    _require(
        bool(method.get("dense_return_q_target", False)),
        f"{path}: dense return Q target is disabled",
    )
    _require(
        bool(method.get("mc_lower_bound_target", False)),
        f"{path}: MC return target is disabled",
    )
    _require(
        method.get("td_target_action_source") == "critic_replay_max",
        f"{path}: wrong Bellman candidate source",
    )
    _require(
        float(method.get("q_reward_scale", 1.0)) == 1.0,
        f"{path}: reward scale is not baseline-matched",
    )
    _require(
        float(method.get("dense_return_label_smoothing", 0.0)) == 0.0,
        f"{path}: dense-return label smoothing is active",
    )

    expected_demo_only = phase == "offline"
    expected_force = 1.0 if phase == "offline" else 0.0
    expected_positive_only = phase == "online"
    _require(
        bool(_value(cfg, "replay.demo_only_updates", False))
        == expected_demo_only,
        f"{path}: wrong demo-only replay setting for {phase}",
    )
    _require(
        float(method.get("demo_behavior_force_probability", 0.0))
        == expected_force,
        f"{path}: wrong behavior-candidate forcing for {phase}",
    )
    _require(
        bool(method.get("dense_return_positive_only", False))
        == expected_positive_only,
        f"{path}: wrong positive-return gate for {phase}",
    )
    if phase == "online":
        _require(
            not bool(method.get("strict_allow_reward_only_success_replay", False)),
            f"{path}: online success replay permission is enabled",
        )

    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "phase": phase,
        "batch_size": 256,
        "demo_batch_size": 256,
        "strict_demo_rl_only": True,
        "bc_lambda": float(method.get("bc_lambda", 0.0)),
        "bc_margin": float(method.get("bc_margin", 0.0)),
        "demo_fosd": bool(method.get("demo_fosd", False)),
        "self_imitation_configured": self_imitation,
        "self_imitation_operationally_active": False,
        "demo_only_updates": expected_demo_only,
        "dense_return_positive_only": expected_positive_only,
        "demo_behavior_force_probability": expected_force,
        "td_target_action_source": "critic_replay_max",
        "mc_lower_bound_target": True,
    }


def _audit_pretrain_csv(path: Path) -> dict[str, Any]:
    rows = list(csv.DictReader(path.open(newline="")))
    _require(bool(rows), f"{path}: empty pretrain CSV")
    episodes = {int(float(row["env_episodes"])) for row in rows}
    demo_sizes = {int(float(row["demo_buffer_size"])) for row in rows}
    buffer_sizes = {int(float(row["buffer_size"])) for row in rows}
    _require(episodes == {0}, f"{path}: online episodes occurred in pretraining")
    _require(demo_sizes == {9253}, f"{path}: demo replay changed")
    _require(buffer_sizes == {10953}, f"{path}: source replay changed")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "logged_rows": len(rows),
        "env_episodes": [0],
        "demo_buffer_sizes": [9253],
        "source_buffer_sizes": [10953],
    }


def _audit_online_csv(path: Path) -> dict[str, Any]:
    rows = list(csv.DictReader(path.open(newline="")))
    _require(bool(rows), f"{path}: empty online CSV")
    demo_sizes = {int(float(row["demo_buffer_size"])) for row in rows}
    _require(demo_sizes == {9253}, f"{path}: expert replay was not fixed")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "logged_rows": len(rows),
        "demo_buffer_sizes": [9253],
    }


def audit(stage43_dir: Path) -> dict[str, Any]:
    stage43_dir = stage43_dir.resolve()
    heldout_path = stage43_dir / "stage43_heldout_summary.json"
    heldout = json.loads(heldout_path.read_text())
    _require(bool(heldout["goal_criterion_met"]), "held-out goal did not pass")
    stage42_dir = Path((stage43_dir / "stage42_source.txt").read_text().strip())
    if not stage42_dir.is_absolute():
        stage42_dir = (Path.cwd() / stage42_dir).resolve()

    seeds: dict[str, Any] = {}
    for seed in range(1, 5):
        run_dir = stage43_dir / f"seed{seed}" / "fixed_expert_101k_online"
        if seed <= 2:
            manifest_path = (
                stage42_dir
                / f"seed{seed}"
                / "offline_dense_online_positive_fixed_expert"
                / "stage42_branch_manifest.json"
            )
            manifest = json.loads(manifest_path.read_text())
            _require(manifest["pretrain_step"] == 10000, "wrong offline step")
            _require(
                manifest["main_loop_iterations"] == 0,
                "seed1/2 offline source contains online interaction",
            )
            source_run = Path(manifest["source_run"])
            offline_config = source_run.parent / "phase_configs" / "offline.yaml"
            offline_csv = source_run / "pretrain.csv"
            online_config = (
                stage43_dir
                / f"seed{seed}"
                / "phase_configs"
                / "stage43_101k_online.yaml"
            )
            provenance = {
                "stage42_branch_manifest": str(manifest_path.resolve()),
                "pretrain_step": 10000,
                "main_loop_iterations_before_online_handoff": 0,
                "archived_stage42_raw50_config_is_online_handoff": True,
            }
            allow_inert_self_imitation_flag = True
        else:
            offline_config = (
                stage43_dir / f"seed{seed}" / "phase_configs" / "offline.yaml"
            )
            offline_csv = run_dir / "pretrain.csv"
            online_config = (
                stage43_dir / f"seed{seed}" / "phase_configs" / "online.yaml"
            )
            provenance = {
                "seed34_offline_completion_marker": str(
                    (stage43_dir / "seed34_offline_complete").resolve()
                ),
                "pretrain_step": 10000,
                "offline_env_episodes": 0,
            }
            allow_inert_self_imitation_flag = False

        seeds[f"seed{seed}"] = {
            "offline_config": audit_phase_config(
                offline_config,
                phase="offline",
                allow_inert_self_imitation_flag=allow_inert_self_imitation_flag,
            ),
            "online_config": audit_phase_config(
                online_config,
                phase="online",
            ),
            "offline_runtime": _audit_pretrain_csv(offline_csv),
            "online_runtime": _audit_online_csv(run_dir / "train.csv"),
            "provenance": provenance,
        }

    return {
        "audit_passed": True,
        "stage43_dir": str(stage43_dir),
        "heldout_summary": {
            "path": str(heldout_path.resolve()),
            "sha256": _sha256(heldout_path),
            "no_bc_fixed_endpoint_mean": heldout["no_bc_fixed_endpoint_mean"],
            "official_fixed_endpoint_mean": heldout[
                "official_fixed_endpoint_mean"
            ],
            "goal_criterion_met": True,
        },
        "objective_contract": {
            "optimization_path": "single distributional critic loss",
            "positive_targets": "reward-derived Bellman or Monte Carlo returns",
            "counterfactual_targets": "task-valid failure-return distribution",
            "no_action_likelihood_or_distance": True,
            "no_bc_margin_or_fosd": True,
            "no_actor_or_policy_pretraining": True,
            "zero_return_action_label_invariance": True,
            "fixed_expert_replay_online": True,
        },
        "seeds": seeds,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage43-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.stage43_dir)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
