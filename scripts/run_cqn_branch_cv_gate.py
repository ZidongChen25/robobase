#!/usr/bin/env python3
"""Select and confirm a seed-generalizing CQN branch-value sidecar.

The original final held-out branch seeds are never read unless a three-init
internal validation gate passes.  This prevents repeatedly tuning against the
small final test set.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--internal-cache", required=True, type=Path)
    parser.add_argument("--final-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpu-id", required=True, type=int)
    parser.add_argument(
        "--grid-updates",
        nargs="+",
        type=_positive_integer,
        default=[5, 10, 20, 50],
    )
    parser.add_argument(
        "--grid-weight-decays",
        nargs="+",
        type=_positive_float,
        default=[1e-5, 1e-3, 1e-2],
    )
    parser.add_argument(
        "--initialization-seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3],
    )
    parser.add_argument(
        "--delta-regression-weight",
        type=float,
        default=10.0,
    )
    parser.add_argument("--batch-size", type=_positive_integer, default=32)
    parser.add_argument(
        "--sampling-mode",
        choices=("random_balanced", "full_batch"),
        default="random_balanced",
    )
    parser.add_argument("--learning-rate", type=_positive_float, default=3e-4)
    parser.add_argument("--temperature", type=_positive_float, default=0.05)
    parser.add_argument(
        "--internal-bootstrap-replicates",
        type=int,
        default=2_000,
    )
    parser.add_argument(
        "--final-bootstrap-replicates",
        type=int,
        default=20_000,
    )
    parser.add_argument("--min-pairwise", type=float, default=0.55)
    parser.add_argument("--min-spearman", type=float, default=0.10)
    return parser.parse_args()


def candidate_label(updates: int, weight_decay: float) -> str:
    wd = f"{weight_decay:.0e}".replace("-", "m").replace("+", "")
    return f"u{updates}_wd{wd}"


def _load_cache_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["cache_metadata"].item()))
    required = {
        "source_snapshot",
        "train_seeds",
        "heldout_seeds",
        "anchor_steps",
        "action_dimensions",
        "candidate_mode",
        "force_level",
        "intervention_horizon",
        "max_continuation_steps",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"cache metadata missing fields: {missing}")
    return metadata


def _completed_payload(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return payload if payload.get("status") == "ok" else None


def metric_row(payload: dict[str, Any]) -> dict[str, float]:
    after = payload["results"]["heldout_after"]
    before = payload["results"]["heldout_before"]
    return {
        "pairwise_sign_accuracy": float(after["pairwise_sign_accuracy"]),
        "mean_spearman": float(after["mean_spearman"]),
        "top1_match_rate": float(after["top1_match_rate"]),
        "random_top1_probability": float(
            after["random_top1_probability"]
        ),
        "mean_realized_regret": float(after["mean_realized_regret"]),
        "before_mean_realized_regret": float(
            before["mean_realized_regret"]
        ),
        "behavior_proxy_pairwise_sign_accuracy": float(
            after["behavior_proxy_pairwise_sign_accuracy"]
        ),
        "behavior_proxy_top1_match_rate": float(
            after["behavior_proxy_top1_match_rate"]
        ),
        "behavior_proxy_mean_realized_regret": float(
            after["behavior_proxy_mean_realized_regret"]
        ),
        "policy_prior_pairwise_sign_accuracy": float(
            after["policy_prior_pairwise_sign_accuracy"]
        ),
        "policy_prior_top1_match_rate": float(
            after["policy_prior_top1_match_rate"]
        ),
        "policy_prior_mean_realized_regret": float(
            after["policy_prior_mean_realized_regret"]
        ),
    }


def select_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot select from an empty candidate list")
    # Selection protocol fixed before evaluation: ranking, monotonicity,
    # realized regret, then lower update count and stronger regularization.
    return max(
        rows,
        key=lambda row: (
            row["metrics"]["pairwise_sign_accuracy"],
            row["metrics"]["mean_spearman"],
            -row["metrics"]["mean_realized_regret"],
            -int(row["updates"]),
            float(row["weight_decay"]),
        ),
    )


def summarize_init_gate(
    rows: list[dict[str, Any]],
    *,
    min_pairwise: float,
    min_spearman: float,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("initialization gate requires at least one result")
    metric_names = (
        "pairwise_sign_accuracy",
        "mean_spearman",
        "top1_match_rate",
        "random_top1_probability",
        "mean_realized_regret",
        "before_mean_realized_regret",
        "behavior_proxy_pairwise_sign_accuracy",
        "behavior_proxy_top1_match_rate",
        "behavior_proxy_mean_realized_regret",
        "policy_prior_pairwise_sign_accuracy",
        "policy_prior_top1_match_rate",
        "policy_prior_mean_realized_regret",
    )
    medians = {
        name: float(np.median([row["metrics"][name] for row in rows]))
        for name in metric_names
    }
    checks = {
        "pairwise_above_absolute_threshold": (
            medians["pairwise_sign_accuracy"] > min_pairwise
        ),
        "pairwise_above_nonreturn_proxies": (
            medians["pairwise_sign_accuracy"]
            > max(
                medians["behavior_proxy_pairwise_sign_accuracy"],
                medians["policy_prior_pairwise_sign_accuracy"],
            )
        ),
        "spearman": medians["mean_spearman"] > min_spearman,
        "top1_above_random": (
            medians["top1_match_rate"]
            > medians["random_top1_probability"]
        ),
        "top1_above_nonreturn_proxies": (
            medians["top1_match_rate"]
            > max(
                medians["behavior_proxy_top1_match_rate"],
                medians["policy_prior_top1_match_rate"],
            )
        ),
        "regret_improved_from_untrained": (
            medians["mean_realized_regret"]
            < medians["before_mean_realized_regret"]
        ),
        "regret_below_nonreturn_proxies": (
            medians["mean_realized_regret"]
            < min(
                medians["behavior_proxy_mean_realized_regret"],
                medians["policy_prior_mean_realized_regret"],
            )
        ),
    }
    return {
        "gate": "pass" if all(checks.values()) else "fail",
        "thresholds": {
            "pairwise_sign_accuracy_strictly_above": float(min_pairwise),
            "pairwise_sign_accuracy_strictly_above_nonreturn_proxies": True,
            "mean_spearman_strictly_above": float(min_spearman),
            "top1_match_rate_strictly_above": "random_top1_probability",
            "top1_match_rate_strictly_above_nonreturn_proxies": True,
            "mean_realized_regret_strictly_below": (
                "heldout_before_mean_realized_regret"
            ),
            "mean_realized_regret_strictly_below_nonreturn_proxies": True,
        },
        "checks": checks,
        "median_metrics": medians,
        "initializations": rows,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as stream:
        stream.write("$ " + " ".join(command) + "\n")
        stream.flush()
        subprocess.run(
            command,
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )


def _fit_command(
    args: argparse.Namespace,
    *,
    cache: Path,
    metadata: dict[str, Any],
    output: Path,
    updates: int,
    weight_decay: float,
    initialization_seed: int,
    bootstrap_replicates: int,
    finetuned_snapshot: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("finetune_cqn_branch_oracle.py")),
        "--run-dir",
        str(args.run_dir),
        "--snapshot",
        str(args.snapshot),
        "--output",
        str(output),
        "--dataset-cache",
        str(cache),
        "--gpu-id",
        str(args.gpu_id),
        "--train-seeds",
        ",".join(str(seed) for seed in metadata["train_seeds"]),
        "--heldout-seeds",
        ",".join(str(seed) for seed in metadata["heldout_seeds"]),
        "--anchor-steps",
        ",".join(str(step) for step in metadata["anchor_steps"]),
        "--action-dimensions",
        ",".join(str(dim) for dim in metadata["action_dimensions"]),
        "--candidate-mode",
        str(metadata["candidate_mode"]),
        "--force-level",
        str(metadata["force_level"]),
        "--intervention-horizon",
        str(metadata["intervention_horizon"]),
        "--max-continuation-steps",
        str(metadata["max_continuation_steps"]),
        "--continuation-repeats",
        str(metadata.get("continuation_repeats", 1)),
        "--continuation-rng-mode",
        str(metadata.get("continuation_rng_mode", "restored")),
        "--updates",
        str(updates),
        "--batch-size",
        str(args.batch_size),
        "--sampling-mode",
        str(args.sampling_mode),
        "--learning-rate",
        f"{args.learning_rate:.12g}",
        "--temperature",
        f"{args.temperature:.12g}",
        "--delta-regression-weight",
        f"{args.delta_regression_weight:.12g}",
        "--weight-decay",
        f"{weight_decay:.12g}",
        "--bootstrap-replicates",
        str(bootstrap_replicates),
        "--seed",
        str(initialization_seed),
    ]
    if finetuned_snapshot is not None:
        command.extend(["--finetuned-snapshot", str(finetuned_snapshot)])
    return command


def _run_fit(
    args: argparse.Namespace,
    *,
    cache: Path,
    metadata: dict[str, Any],
    output: Path,
    updates: int,
    weight_decay: float,
    initialization_seed: int,
    bootstrap_replicates: int,
    finetuned_snapshot: Path | None = None,
) -> tuple[dict[str, Any], float]:
    cached = _completed_payload(output)
    if cached is not None:
        return cached, float(cached.get("elapsed_seconds", 0.0))
    command = _fit_command(
        args,
        cache=cache,
        metadata=metadata,
        output=output,
        updates=updates,
        weight_decay=weight_decay,
        initialization_seed=initialization_seed,
        bootstrap_replicates=bootstrap_replicates,
        finetuned_snapshot=finetuned_snapshot,
    )
    started = time.time()
    _run_logged(command, output.with_suffix(".log"))
    payload = _completed_payload(output)
    if payload is None:
        raise RuntimeError(f"branch fit did not complete: {output}")
    return payload, time.time() - started


def _progress(
    output_dir: Path,
    *,
    stage: str,
    completed: int,
    planned: int,
    durations: list[float],
) -> None:
    mean_seconds = float(np.mean(durations)) if durations else None
    remaining = planned - completed
    _write_json(
        output_dir / "progress.json",
        {
            "status": "running",
            "stage": stage,
            "completed_fits": completed,
            "currently_planned_fits": planned,
            "mean_fit_seconds": mean_seconds,
            "estimated_remaining_seconds": (
                mean_seconds * remaining
                if mean_seconds is not None
                else None
            ),
            "updated_at_unix": time.time(),
        },
    )


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    if args.delta_regression_weight < 0:
        raise ValueError("delta regression weight must be non-negative")
    if (
        args.internal_bootstrap_replicates < 0
        or args.final_bootstrap_replicates < 0
    ):
        raise ValueError("bootstrap replicate counts must be non-negative")
    if len(args.initialization_seeds) < 3:
        raise ValueError("at least three initialization seeds are required")
    if len(set(args.initialization_seeds)) != len(args.initialization_seeds):
        raise ValueError("initialization seeds must be unique")
    if len(set(args.grid_updates)) != len(args.grid_updates):
        raise ValueError("grid update counts must be unique")
    if len(set(args.grid_weight_decays)) != len(args.grid_weight_decays):
        raise ValueError("grid weight decays must be unique")

    args.run_dir = args.run_dir.expanduser().resolve()
    args.snapshot = args.snapshot.expanduser().resolve()
    args.internal_cache = args.internal_cache.expanduser().resolve()
    args.final_cache = args.final_cache.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    for path in (
        args.snapshot,
        args.internal_cache,
        args.final_cache,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    internal_metadata = _load_cache_metadata(args.internal_cache)
    final_metadata = _load_cache_metadata(args.final_cache)
    expected_snapshot = str(args.snapshot)
    for label, metadata in (
        ("internal", internal_metadata),
        ("final", final_metadata),
    ):
        if str(Path(metadata["source_snapshot"]).expanduser().resolve()) != (
            expected_snapshot
        ):
            raise ValueError(
                f"{label} cache source snapshot does not match --snapshot"
            )
    if set(internal_metadata["heldout_seeds"]) & set(
        final_metadata["heldout_seeds"]
    ):
        raise ValueError("internal validation and final test seeds overlap")

    started = time.time()
    durations: list[float] = []
    grid_rows = []
    grid_total = len(args.grid_updates) * len(args.grid_weight_decays)
    for updates in args.grid_updates:
        for weight_decay in args.grid_weight_decays:
            label = candidate_label(updates, weight_decay)
            output = args.output_dir / "grid" / f"{label}_seed1.json"
            payload, duration = _run_fit(
                args,
                cache=args.internal_cache,
                metadata=internal_metadata,
                output=output,
                updates=updates,
                weight_decay=weight_decay,
                initialization_seed=args.initialization_seeds[0],
                bootstrap_replicates=args.internal_bootstrap_replicates,
            )
            durations.append(duration)
            grid_rows.append(
                {
                    "label": label,
                    "updates": updates,
                    "weight_decay": weight_decay,
                    "initialization_seed": args.initialization_seeds[0],
                    "artifact": str(output),
                    "elapsed_seconds": float(
                        payload.get("elapsed_seconds", duration)
                    ),
                    "metrics": metric_row(payload),
                }
            )
            _progress(
                args.output_dir,
                stage="grid",
                completed=len(grid_rows),
                planned=grid_total,
                durations=durations,
            )

    selected = select_candidate(grid_rows)
    _write_json(
        args.output_dir / "grid_summary.json",
        {
            "status": "ok",
            "selection_protocol": (
                "max pairwise, max Spearman, min regret, fewer updates, "
                "stronger weight decay"
            ),
            "selected": selected,
            "candidates": grid_rows,
        },
    )

    internal_rows = [selected]
    internal_total = grid_total + len(args.initialization_seeds) - 1
    for initialization_seed in args.initialization_seeds[1:]:
        output = (
            args.output_dir
            / "internal_replication"
            / (
                f"{selected['label']}_seed{initialization_seed}.json"
            )
        )
        payload, duration = _run_fit(
            args,
            cache=args.internal_cache,
            metadata=internal_metadata,
            output=output,
            updates=int(selected["updates"]),
            weight_decay=float(selected["weight_decay"]),
            initialization_seed=initialization_seed,
            bootstrap_replicates=args.internal_bootstrap_replicates,
        )
        durations.append(duration)
        internal_rows.append(
            {
                "label": selected["label"],
                "updates": int(selected["updates"]),
                "weight_decay": float(selected["weight_decay"]),
                "initialization_seed": initialization_seed,
                "artifact": str(output),
                "elapsed_seconds": float(
                    payload.get("elapsed_seconds", duration)
                ),
                "metrics": metric_row(payload),
            }
        )
        _progress(
            args.output_dir,
            stage="internal_replication",
            completed=grid_total + len(internal_rows) - 1,
            planned=internal_total,
            durations=durations,
        )

    internal_gate = summarize_init_gate(
        internal_rows,
        min_pairwise=float(args.min_pairwise),
        min_spearman=float(args.min_spearman),
    )
    _write_json(args.output_dir / "internal_gate.json", internal_gate)
    result = {
        "status": "ok",
        "selection_protocol": (
            "seed1 grid on internal heldout; selected hyperparameters "
            "replicated over three initialization seeds"
        ),
        "sampling_mode": str(args.sampling_mode),
        "internal_cache": str(args.internal_cache),
        "final_cache": str(args.final_cache),
        "final_test_policy": (
            "unread unless the three-init internal gate passes"
        ),
        "selected": selected,
        "internal_gate": internal_gate,
        "final_gate": "not_run",
        "final_gate_summary": None,
    }
    if internal_gate["gate"] != "pass":
        result["final_gate_reason"] = (
            "internal seed-generalization gate failed; final test remained "
            "sealed"
        )
        result["elapsed_seconds"] = time.time() - started
        _write_json(args.output_dir / "gate_summary.json", result)
        _write_json(
            args.output_dir / "progress.json",
            {
                "status": "complete",
                "stage": "internal_gate_failed",
                "elapsed_seconds": result["elapsed_seconds"],
                "final_test_read": False,
            },
        )
        return result

    final_rows = []
    final_total = internal_total + len(args.initialization_seeds)
    for initialization_seed in args.initialization_seeds:
        output = (
            args.output_dir
            / "final_test"
            / f"{selected['label']}_seed{initialization_seed}.json"
        )
        snapshot = (
            args.output_dir
            / "snapshots"
            / f"{selected['label']}_seed{initialization_seed}.pkl"
        )
        payload, duration = _run_fit(
            args,
            cache=args.final_cache,
            metadata=final_metadata,
            output=output,
            updates=int(selected["updates"]),
            weight_decay=float(selected["weight_decay"]),
            initialization_seed=initialization_seed,
            bootstrap_replicates=args.final_bootstrap_replicates,
            finetuned_snapshot=snapshot,
        )
        durations.append(duration)
        final_rows.append(
            {
                "label": selected["label"],
                "updates": int(selected["updates"]),
                "weight_decay": float(selected["weight_decay"]),
                "initialization_seed": initialization_seed,
                "artifact": str(output),
                "snapshot": str(snapshot),
                "elapsed_seconds": float(
                    payload.get("elapsed_seconds", duration)
                ),
                "metrics": metric_row(payload),
            }
        )
        _progress(
            args.output_dir,
            stage="final_test",
            completed=internal_total + len(final_rows),
            planned=final_total,
            durations=durations,
        )

    final_gate = summarize_init_gate(
        final_rows,
        min_pairwise=float(args.min_pairwise),
        min_spearman=float(args.min_spearman),
    )
    result["final_gate"] = final_gate["gate"]
    result["final_gate_summary"] = final_gate
    result["elapsed_seconds"] = time.time() - started
    _write_json(args.output_dir / "gate_summary.json", result)
    _write_json(
        args.output_dir / "progress.json",
        {
            "status": "complete",
            "stage": "final_gate_complete",
            "elapsed_seconds": result["elapsed_seconds"],
            "final_test_read": True,
            "final_gate": result["final_gate"],
        },
    )
    return result


def main() -> int:
    args = parse_args()
    result = run_gate(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
