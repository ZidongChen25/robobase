#!/usr/bin/env python3
"""Validate matched H=1 randomized direct-Q control/treatment smoke runs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import csv
import hashlib
import json
import math
from pathlib import Path
import pickle
from typing import Any

import numpy as np
from omegaconf import OmegaConf


BASE_METRICS = (
    "critic_loss",
    "td_critic_loss",
    "mc_return_loss",
    "direct_q_grad_norm",
    "direct_q_grad_nonfinite_fraction",
)
CAUSAL_METRICS = (
    "causal_rct_loss",
    "causal_rct_moment_loss",
    "causal_rct_valid_fraction",
    "causal_rct_treated_fraction",
    "causal_rct_tau_abs_mean",
    "causal_rct_assignment_error_max",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-causal-rct-weight", required=True, type=float)
    parser.add_argument("--expected-exploration-prob", type=float, default=0.2)
    parser.add_argument("--expected-level", type=int, default=1)
    parser.add_argument("--required-snapshot-step", type=int, default=1_000)
    parser.add_argument("--min-log-rows", type=int, default=1)
    parser.add_argument("--min-online-starts", type=int, default=30)
    parser.add_argument("--min-starts-per-dimension", type=int, default=1)
    parser.add_argument(
        "--expected-frozen-policy-snapshot",
        type=Path,
        help=(
            "Require a legacy-C51 policy imported from this clean CQN-AS "
            "snapshot and prove that the trained policy/encoder remain "
            "bitwise identical."
        ),
    )
    return parser.parse_args()


def _numeric_rows(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                parsed = {
                    key: float(row[key])
                    for key in (*BASE_METRICS, *CAUSAL_METRICS)
                }
                parsed["iteration"] = float(row["iteration"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(parsed)
    return rows


def _numeric_column(path: Path, key: str) -> list[float]:
    values = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                values.append(float(row[key]))
            except (KeyError, TypeError, ValueError):
                continue
    return values


def _flatten_tree(
    tree: Any,
    *,
    prefix: tuple[str, ...] = (),
) -> list[tuple[tuple[str, ...], np.ndarray]]:
    if isinstance(tree, Mapping):
        leaves = []
        for key in sorted(tree, key=lambda value: repr(value)):
            leaves.extend(
                _flatten_tree(
                    tree[key],
                    prefix=(*prefix, f"key:{key!r}"),
                )
            )
        return leaves
    if isinstance(tree, (tuple, list)):
        leaves = []
        container = type(tree).__name__
        for index, value in enumerate(tree):
            leaves.extend(
                _flatten_tree(
                    value,
                    prefix=(*prefix, f"{container}:{index}"),
                )
            )
        return leaves
    return [(prefix, np.asarray(tree))]


def _tree_sha256(
    leaves: list[tuple[tuple[str, ...], np.ndarray]],
) -> str:
    digest = hashlib.sha256()
    for path, array in leaves:
        digest.update(repr(path).encode())
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _tree_bitwise_evidence(source: Any, trained: Any) -> dict[str, Any]:
    source_leaves = _flatten_tree(source)
    trained_leaves = _flatten_tree(trained)
    compatible = (
        len(source_leaves) == len(trained_leaves)
        and all(
            left_path == right_path
            and left.shape == right.shape
            and left.dtype == right.dtype
            for (left_path, left), (right_path, right) in zip(
                source_leaves,
                trained_leaves,
                strict=True,
            )
        )
    )
    bitwise_equal = compatible and all(
        np.array_equal(left, right)
        for (_, left), (_, right) in zip(
            source_leaves,
            trained_leaves,
            strict=True,
        )
    )
    return {
        "compatible": bool(compatible),
        "bitwise_equal": bool(bitwise_equal),
        "source_num_leaves": len(source_leaves),
        "trained_num_leaves": len(trained_leaves),
        "source_sha256": _tree_sha256(source_leaves),
        "trained_sha256": _tree_sha256(trained_leaves),
    }


def _frozen_policy_audit(
    *,
    cfg,
    csv_path: Path,
    trained_snapshot: Path,
    expected_snapshot: Path,
) -> dict[str, Any]:
    expected_snapshot = expected_snapshot.expanduser().resolve()
    configured_value = cfg.method.get("frozen_policy_snapshot", None)
    configured_snapshot = (
        None
        if configured_value is None
        else Path(str(configured_value)).expanduser().resolve()
    )
    policy_gradients = _numeric_column(csv_path, "policy_grad_norm")
    policy_encoder_gradients = _numeric_column(
        csv_path,
        "policy_encoder_grad_norm",
    )
    evidence: dict[str, Any] = {
        "expected_snapshot": str(expected_snapshot),
        "configured_snapshot": (
            None
            if configured_snapshot is None
            else str(configured_snapshot)
        ),
        "configured_snapshot_matches": (
            configured_snapshot == expected_snapshot
        ),
        "freeze_bc_policy": bool(
            cfg.method.get("freeze_bc_policy", False)
        ),
        "legacy_c51_policy": (
            str(cfg.method.get("bc_policy_mode", "")).lower()
            == "legacy_c51"
        ),
        "policy_gradient_rows": len(policy_gradients),
        "policy_encoder_gradient_rows": len(policy_encoder_gradients),
        "max_abs_policy_grad_norm": (
            max((abs(value) for value in policy_gradients), default=None)
        ),
        "max_abs_policy_encoder_grad_norm": (
            max(
                (abs(value) for value in policy_encoder_gradients),
                default=None,
            )
        ),
    }
    if not expected_snapshot.is_file():
        evidence["error"] = "expected frozen policy snapshot is missing"
        return evidence
    if not trained_snapshot.is_file():
        evidence["error"] = "trained snapshot is missing"
        return evidence
    with expected_snapshot.open("rb") as stream:
        source_payload = pickle.load(stream)
    with trained_snapshot.open("rb") as stream:
        trained_payload = pickle.load(stream)
    source_agent = source_payload.get("agent", {})
    trained_agent = trained_payload.get("agent", {})
    source_params = source_agent.get("params", {})
    trained_params = trained_agent.get("params", {})
    source_policy = source_agent.get(
        "target_critic_params",
        source_params.get("critic"),
    )
    source_encoder = source_params.get("encoder")
    trained_policy = trained_params.get("policy")
    trained_policy_encoder = trained_params.get("policy_encoder")
    if source_policy is None or trained_policy is None:
        evidence["error"] = "source or trained policy tree is missing"
        return evidence
    if source_encoder is None or trained_policy_encoder is None:
        evidence["error"] = "source or trained policy encoder is missing"
        return evidence
    evidence["policy"] = _tree_bitwise_evidence(
        source_policy,
        trained_policy,
    )
    evidence["policy_encoder"] = _tree_bitwise_evidence(
        source_encoder,
        trained_policy_encoder,
    )
    return evidence


def _episode_length(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-2])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid replay archive name: {path}") from exc


def _replay_audit(
    replay_dir: Path,
    *,
    action_dim: int,
    expected_probability: float,
) -> dict[str, Any]:
    online_transitions = 0
    starts = 0
    active = 0
    controls = 0
    per_dimension = np.zeros((action_dim,), np.int64)
    max_assignment_error = 0.0
    num_archives = 0
    for path in sorted(replay_dir.glob("*.npz")):
        length = _episode_length(path)
        with np.load(path, allow_pickle=False) as episode:
            required = {
                "demo",
                "structured_explore",
                "structured_explore_start",
                "structured_explore_dimension",
                "structured_explore_delta",
                "structured_explore_assignment_prob",
            }
            missing = required - set(episode.files)
            if missing:
                raise ValueError(
                    f"{path} lacks structured replay fields {sorted(missing)}"
                )
            demo = np.asarray(episode["demo"][:length], np.bool_)
            online = ~demo
            if not np.any(online):
                continue
            mask = np.asarray(
                episode["structured_explore"][:length], np.bool_
            )[online]
            start = np.asarray(
                episode["structured_explore_start"][:length], np.bool_
            )[online]
            dimension = np.asarray(
                episode["structured_explore_dimension"][:length], np.int64
            )[online]
            delta = np.asarray(
                episode["structured_explore_delta"][:length], np.float64
            )[online]
            assignment = np.asarray(
                episode["structured_explore_assignment_prob"][:length],
                np.float64,
            )[online]
        num_archives += 1
        online_transitions += int(online.sum())
        starts += int(start.sum())
        active += int(mask.sum())
        control = ~start
        controls += int(control.sum())
        if not np.array_equal(mask, start):
            raise ValueError(
                f"{path} contains non-start active transitions; H=1 violated"
            )
        if np.any(dimension[start] < 0) or np.any(
            dimension[start] >= action_dim
        ):
            raise ValueError(f"{path} contains invalid treatment dimensions")
        if np.any(dimension[control] != -1):
            raise ValueError(f"{path} controls contain treatment dimensions")
        # A randomized treatment can have zero realized delta when the
        # proposed direction points outside an action bound and clipping maps
        # it back to the baseline action.  That is a valid zero-effect
        # assignment, not a malformed intervention.
        if np.any(~np.isfinite(delta[start])) or np.any(
            np.abs(delta[control]) > 1e-8
        ):
            raise ValueError(f"{path} has inconsistent intervention deltas")
        per_dimension += np.bincount(
            dimension[start],
            minlength=action_dim,
        )[:action_dim]
        expected = np.where(
            start,
            expected_probability / float(2 * action_dim),
            1.0 - expected_probability,
        )
        max_assignment_error = max(
            max_assignment_error,
            float(np.max(np.abs(assignment - expected))),
        )
    start_rate = starts / online_transitions if online_transitions else 0.0
    return {
        "num_online_archives": num_archives,
        "online_transitions": online_transitions,
        "starts": starts,
        "active": active,
        "controls": controls,
        "start_rate": start_rate,
        "starts_per_dimension": per_dimension.tolist(),
        "min_starts_per_dimension": (
            int(per_dimension.min()) if per_dimension.size else 0
        ),
        "max_assignment_probability_error": max_assignment_error,
    }


def check_run(
    run_dir: Path,
    *,
    expected_causal_rct_weight: float,
    expected_exploration_prob: float,
    expected_level: int,
    required_snapshot_step: int,
    min_log_rows: int,
    min_online_starts: int,
    min_starts_per_dimension: int,
    expected_frozen_policy_snapshot: Path | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    config_path = run_dir / ".hydra" / "config.yaml"
    csv_path = run_dir / "train.csv"
    snapshot = (
        run_dir
        / "snapshots"
        / f"{int(required_snapshot_step)}_snapshot.pkl"
    )
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    cfg = OmegaConf.load(config_path)
    rows = _numeric_rows(csv_path)
    action_dim = int(cfg.get("action_dim", 15))
    # BiGym does not expose action_dim in Hydra; infer it from replay.
    first_archive = next(iter(sorted((run_dir / "replay").glob("*.npz"))), None)
    if first_archive is None:
        raise FileNotFoundError(run_dir / "replay")
    with np.load(first_archive, allow_pickle=False) as episode:
        action_dim = int(np.asarray(episode["action"]).shape[-1])
    replay = _replay_audit(
        run_dir / "replay",
        action_dim=action_dim,
        expected_probability=float(expected_exploration_prob),
    )
    all_finite = bool(rows) and all(
        math.isfinite(value)
        for row in rows
        for value in row.values()
    )
    is_treatment = expected_causal_rct_weight > 0.0
    max_valid_fraction = max(
        (row["causal_rct_valid_fraction"] for row in rows),
        default=0.0,
    )
    max_treated_fraction = max(
        (row["causal_rct_treated_fraction"] for row in rows),
        default=0.0,
    )
    max_tau = max(
        (row["causal_rct_tau_abs_mean"] for row in rows),
        default=0.0,
    )
    max_metric_assignment_error = max(
        (row["causal_rct_assignment_error_max"] for row in rows),
        default=float("inf"),
    )
    max_control_causal_metric = max(
        (
            abs(row[key])
            for row in rows
            for key in CAUSAL_METRICS
        ),
        default=float("inf"),
    )
    checks = {
        "direct_scalar_q": bool(cfg.method.get("direct_scalar_q", False)),
        "separate_bc_policy": bool(
            cfg.method.get("separate_bc_policy", False)
        ),
        "distinct_policy_encoder": bool(
            cfg.method.get("distinct_policy_encoder", False)
        ),
        "rollout_is_exact_bc": (
            cfg.method.get("policy_value_beta", None) is None
        ),
        "completed_return_present": (
            float(cfg.method.get("mc_return_weight", 0.0)) > 0.0
        ),
        "one_step_intervention": (
            int(cfg.method.get("structured_exploration_horizon", -1)) == 1
        ),
        "no_unlogged_random_warmup": (
            int(cfg.get("num_explore_steps", -1)) == 0
            and int(cfg.method.get("num_explore_steps", -1)) == 0
        ),
        "no_unlogged_gaussian_exploration": math.isclose(
            float(cfg.method.get("stddev_schedule", float("nan"))),
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "uniform_replay_sampling": not bool(
            cfg.replay.get("prioritization", True)
        ),
        "one_step_replay_target": int(cfg.replay.get("nstep", -1)) == 1,
        "effective_first_action_value": (
            str(cfg.method.get("critic_sequence_mode", "")).lower()
            == "effective_k0"
        ),
        "replay_next_td_target": (
            str(cfg.method.get("td_target_action_source", "")).lower()
            == "replay_next"
        ),
        "matched_exploration_probability": math.isclose(
            float(cfg.method.get("structured_exploration_prob", -1.0)),
            float(expected_exploration_prob),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "matched_causal_level": (
            int(cfg.method.get("causal_rct_level", -1))
            == int(expected_level)
        ),
        "matched_causal_weight": math.isclose(
            float(cfg.method.get("causal_rct_weight", -1.0)),
            float(expected_causal_rct_weight),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "snapshot_exists": snapshot.is_file(),
        "enough_log_rows": len(rows) >= int(min_log_rows),
        "all_required_metrics_finite": all_finite,
        "zero_nonfinite_direct_q_gradients": (
            bool(rows)
            and max(
                row["direct_q_grad_nonfinite_fraction"] for row in rows
            )
            == 0.0
        ),
        "enough_randomized_starts": (
            replay["starts"] >= int(min_online_starts)
        ),
        "all_dimensions_randomized": (
            replay["min_starts_per_dimension"]
            >= int(min_starts_per_dimension)
        ),
        "replay_assignment_probability_exact": (
            replay["max_assignment_probability_error"] <= 1e-6
        ),
    }
    frozen_policy = None
    if expected_frozen_policy_snapshot is not None:
        frozen_policy = _frozen_policy_audit(
            cfg=cfg,
            csv_path=csv_path,
            trained_snapshot=snapshot,
            expected_snapshot=expected_frozen_policy_snapshot,
        )
        checks.update(
            {
                "frozen_policy_enabled": bool(
                    frozen_policy.get("freeze_bc_policy", False)
                ),
                "frozen_policy_is_legacy_c51": bool(
                    frozen_policy.get("legacy_c51_policy", False)
                ),
                "frozen_policy_source_matches": bool(
                    frozen_policy.get(
                        "configured_snapshot_matches",
                        False,
                    )
                ),
                "frozen_policy_bitwise_equal": bool(
                    frozen_policy.get("policy", {}).get(
                        "bitwise_equal",
                        False,
                    )
                ),
                "frozen_policy_encoder_bitwise_equal": bool(
                    frozen_policy.get("policy_encoder", {}).get(
                        "bitwise_equal",
                        False,
                    )
                ),
                "frozen_policy_gradient_exact_zero": (
                    frozen_policy.get("policy_gradient_rows", 0)
                    >= int(min_log_rows)
                    and frozen_policy.get(
                        "max_abs_policy_grad_norm",
                        None,
                    )
                    == 0.0
                ),
                "frozen_policy_encoder_gradient_exact_zero": (
                    frozen_policy.get(
                        "policy_encoder_gradient_rows",
                        0,
                    )
                    >= int(min_log_rows)
                    and frozen_policy.get(
                        "max_abs_policy_encoder_grad_norm",
                        None,
                    )
                    == 0.0
                ),
            }
        )
    if is_treatment:
        checks.update(
            {
                "causal_batches_have_online_samples": (
                    max_valid_fraction > 0.0
                ),
                "causal_batches_have_treatments": (
                    max_treated_fraction > 0.0
                ),
                "causal_advantage_leaves_zero": max_tau > 1e-8,
                "metric_assignment_probability_exact": (
                    max_metric_assignment_error <= 1e-6
                ),
            }
        )
    else:
        checks["causal_metrics_exact_noop"] = (
            max_control_causal_metric == 0.0
        )
    return {
        "status": "ok",
        "run_dir": str(run_dir),
        "expected_causal_rct_weight": float(expected_causal_rct_weight),
        "expected_exploration_prob": float(expected_exploration_prob),
        "required_snapshot": str(snapshot),
        "num_log_rows": len(rows),
        "last_iteration": rows[-1]["iteration"] if rows else None,
        "replay": replay,
        "max_causal_rct_valid_fraction": max_valid_fraction,
        "max_causal_rct_treated_fraction": max_treated_fraction,
        "max_causal_rct_tau_abs_mean": max_tau,
        "max_causal_rct_assignment_error": max_metric_assignment_error,
        "frozen_policy": frozen_policy,
        "checks": checks,
        "gate": "pass" if all(checks.values()) else "fail",
    }


def main() -> int:
    args = parse_args()
    payload = check_run(
        args.run_dir,
        expected_causal_rct_weight=float(args.expected_causal_rct_weight),
        expected_exploration_prob=float(args.expected_exploration_prob),
        expected_level=int(args.expected_level),
        required_snapshot_step=int(args.required_snapshot_step),
        min_log_rows=int(args.min_log_rows),
        min_online_starts=int(args.min_online_starts),
        min_starts_per_dimension=int(args.min_starts_per_dimension),
        expected_frozen_policy_snapshot=args.expected_frozen_policy_snapshot,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["gate"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
