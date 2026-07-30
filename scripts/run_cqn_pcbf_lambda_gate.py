#!/usr/bin/env python3
"""Screen, validate, and independently confirm PCBF lambda choices.

The three data splits have distinct roles:

1. a small checkpoint-screen split chooses one frozen checkpoint per lambda;
2. a validation split chooses one lambda and compares it with clean CQN-AS;
3. only a positive validation margin opens an independent confirmation split.

All PCBF policies use the paper-consistent mean terminal-return score over
parallel return samples.  Evaluation jobs are distributed over the requested
GPUs, one process per GPU, and every result is restartable.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("value must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("value must be LABEL=PATH")
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flow-run",
        required=True,
        action="append",
        type=_labeled_path,
        help="LABEL=RUN_DIR; repeat once per PCBF lambda",
    )
    parser.add_argument("--clean-run-dir", required=True, type=Path)
    parser.add_argument("--clean-snapshot", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--gpu-ids", required=True, nargs="+", type=int)
    parser.add_argument("--num-flow-steps", type=int, default=4)
    parser.add_argument("--num-action-flow-samples", type=int, default=16)
    parser.add_argument("--screen-episodes", type=int, default=10)
    parser.add_argument("--screen-seed-start", type=int, default=50000)
    parser.add_argument("--validation-episodes", type=int, default=50)
    parser.add_argument("--validation-seed-start", type=int, default=51000)
    parser.add_argument("--confirmation-episodes", type=int, default=200)
    parser.add_argument("--confirmation-seed-start", type=int, default=52000)
    parser.add_argument("--min-validation-delta", type=float, default=0.02)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    return parser.parse_args()


@dataclass(frozen=True)
class FlowCheckpoint:
    run_label: str
    run_dir: Path
    snapshot: Path
    step: int
    run_order: int

    @property
    def label(self) -> str:
        return f"{self.run_label}_step{self.step}"


@dataclass(frozen=True)
class EvalJob:
    label: str
    run_dir: Path
    snapshot: Path
    output: Path
    work_dir: Path
    log: Path
    episodes: int
    seed_start: int
    is_flow: bool


def discover_flow_checkpoints(
    flow_runs: list[tuple[str, Path]],
) -> list[FlowCheckpoint]:
    checkpoints: list[FlowCheckpoint] = []
    labels = [label for label, _ in flow_runs]
    if len(labels) != len(set(labels)):
        raise ValueError("flow-run labels must be unique")
    for run_order, (run_label, run_dir) in enumerate(flow_runs):
        snapshot_dir = run_dir / "snapshots"
        found = []
        for snapshot in snapshot_dir.glob("*_snapshot.pkl"):
            match = re.fullmatch(r"([0-9]+)_snapshot[.]pkl", snapshot.name)
            if match is not None:
                found.append((int(match.group(1)), snapshot))
        if not found:
            raise FileNotFoundError(
                f"no numeric snapshots found under {snapshot_dir}"
            )
        checkpoints.extend(
            FlowCheckpoint(
                run_label=run_label,
                run_dir=run_dir,
                snapshot=snapshot,
                step=step,
                run_order=run_order,
            )
            for step, snapshot in sorted(found)
        )
    return checkpoints


def _completed_payload(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text()).get("status") == "ok"
    except (json.JSONDecodeError, OSError):
        return False


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _eval_command(
    job: EvalJob,
    *,
    gpu_id: int,
    num_flow_steps: int,
    num_action_flow_samples: int,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("eval_cqn_flow_policy_value.py")),
        "--run-dir",
        str(job.run_dir),
        "--snapshot",
        str(job.snapshot),
        "--output",
        str(job.output),
        "--work-dir",
        str(job.work_dir),
        "--gpu-id",
        str(gpu_id),
        "--num-eval-episodes",
        str(job.episodes),
        "--eval-seed-start",
        str(job.seed_start),
        "--policy-value-beta",
        "bc",
    ]
    if job.is_flow:
        command.extend(
            [
                "--flow-readout",
                "integrated",
                "--num-flow-steps",
                str(num_flow_steps),
                "--num-action-flow-samples",
                str(num_action_flow_samples),
                "--return-sample-aggregation",
                "mean",
            ]
        )
    return command


def run_jobs_parallel(
    jobs: list[EvalJob],
    *,
    gpu_ids: list[int],
    num_flow_steps: int,
    num_action_flow_samples: int,
    progress_path: Path,
    phase: str,
) -> dict[str, dict]:
    if not gpu_ids:
        raise ValueError("at least one GPU id is required")
    pending: queue.Queue[EvalJob] = queue.Queue()
    for job in jobs:
        pending.put(job)
    lock = threading.Lock()
    completed = 0
    durations: list[float] = []
    failures: list[str] = []
    payloads: dict[str, dict] = {}
    started = time.time()

    def write_progress() -> None:
        remaining = len(jobs) - completed
        mean_seconds = float(np.mean(durations)) if durations else None
        parallelism = min(len(gpu_ids), max(1, remaining))
        eta = (
            mean_seconds * remaining / parallelism
            if mean_seconds is not None
            else None
        )
        _atomic_json(
            progress_path,
            {
                "completed_jobs": completed,
                "elapsed_seconds": time.time() - started,
                "estimated_remaining_seconds": eta,
                "failed_jobs": failures,
                "mean_job_seconds": mean_seconds,
                "phase": phase,
                "status": "failed" if failures else "running",
                "total_jobs": len(jobs),
                "updated_at_unix": time.time(),
            },
        )

    write_progress()

    def worker(gpu_id: int) -> None:
        nonlocal completed
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            job.output.parent.mkdir(parents=True, exist_ok=True)
            job.log.parent.mkdir(parents=True, exist_ok=True)
            job.work_dir.parent.mkdir(parents=True, exist_ok=True)
            job_started = time.time()
            try:
                if not _completed_payload(job.output):
                    command = _eval_command(
                        job,
                        gpu_id=gpu_id,
                        num_flow_steps=num_flow_steps,
                        num_action_flow_samples=num_action_flow_samples,
                    )
                    with job.log.open("a") as stream:
                        stream.write("$ " + " ".join(command) + "\n")
                        stream.flush()
                        subprocess.run(
                            command,
                            check=True,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                        )
                if not _completed_payload(job.output):
                    raise RuntimeError(
                        f"job did not produce a valid result: {job.output}"
                    )
                payload = json.loads(job.output.read_text())
                with lock:
                    payloads[job.label] = payload
                    completed += 1
                    durations.append(time.time() - job_started)
                    write_progress()
            except Exception as exc:  # noqa: BLE001
                with lock:
                    failures.append(f"{job.label}: {type(exc).__name__}: {exc}")
                    write_progress()
                return
            finally:
                pending.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu_id,), daemon=False)
        for gpu_id in gpu_ids
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError("; ".join(failures))
    _atomic_json(
        progress_path,
        {
            "completed_jobs": completed,
            "elapsed_seconds": time.time() - started,
            "estimated_remaining_seconds": 0.0,
            "failed_jobs": [],
            "mean_job_seconds": (
                float(np.mean(durations)) if durations else 0.0
            ),
            "phase": phase,
            "status": "complete",
            "total_jobs": len(jobs),
            "updated_at_unix": time.time(),
        },
    )
    return payloads


def _success(payload: dict) -> float:
    return float(payload["episode_success"])


def select_checkpoint(
    checkpoints: list[FlowCheckpoint],
    payloads: dict[str, dict],
    *,
    run_label: str,
) -> FlowCheckpoint:
    eligible = [
        checkpoint
        for checkpoint in checkpoints
        if checkpoint.run_label == run_label
    ]
    if not eligible:
        raise ValueError(f"no checkpoints for {run_label}")
    return max(
        eligible,
        key=lambda checkpoint: (
            _success(payloads[checkpoint.label]),
            -checkpoint.step,
        ),
    )


def paired_statistics(
    candidate_payload: dict,
    baseline_payload: dict,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    candidate_by_seed = {
        int(row["seed"]): float(row["episode_success"])
        for row in candidate_payload["episode_results"]
    }
    baseline_by_seed = {
        int(row["seed"]): float(row["episode_success"])
        for row in baseline_payload["episode_results"]
    }
    if set(candidate_by_seed) != set(baseline_by_seed):
        raise ValueError("candidate and baseline do not share the same seeds")
    seeds = np.asarray(sorted(candidate_by_seed), dtype=np.int64)
    candidate = np.asarray(
        [candidate_by_seed[int(seed)] for seed in seeds],
        dtype=np.float64,
    )
    baseline = np.asarray(
        [baseline_by_seed[int(seed)] for seed in seeds],
        dtype=np.float64,
    )
    delta = candidate - baseline
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(
        0,
        len(seeds),
        size=(bootstrap_samples, len(seeds)),
    )
    boot = delta[indices].mean(axis=1)
    wins = int(np.sum(delta > 0))
    losses = int(np.sum(delta < 0))
    discordant = wins + losses
    if discordant:
        tail = sum(
            math.comb(discordant, value)
            for value in range(min(wins, losses) + 1)
        ) / (2**discordant)
        mcnemar = min(1.0, 2.0 * tail)
    else:
        mcnemar = 1.0
    return {
        "baseline_success": float(baseline.mean()),
        "candidate_success": float(candidate.mean()),
        "episodes": int(len(seeds)),
        "mcnemar_exact_p": float(mcnemar),
        "paired_delta": float(delta.mean()),
        "paired_delta_ci95": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "paired_losses": losses,
        "paired_ties": int(np.sum(delta == 0)),
        "paired_wins": wins,
    }


def confirmation_gate(row: dict) -> tuple[str, str]:
    if row["paired_delta"] <= 0.0:
        return "fail", "confirmation paired delta is not positive"
    if row["paired_delta_ci95"][0] < 0.0:
        return "fail", "confirmation CI lower bound is below zero"
    if row["paired_wins"] <= row["paired_losses"]:
        return "fail", "confirmation wins do not exceed losses"
    return "pass", "strict paired superiority gate passed"


def _screen_jobs(
    args: argparse.Namespace,
    checkpoints: list[FlowCheckpoint],
) -> list[EvalJob]:
    return [
        EvalJob(
            label=checkpoint.label,
            run_dir=checkpoint.run_dir,
            snapshot=checkpoint.snapshot,
            output=args.output_dir / "screen" / f"{checkpoint.label}.json",
            work_dir=args.work_root / "screen" / checkpoint.label,
            log=args.output_dir / "screen" / f"{checkpoint.label}.log",
            episodes=args.screen_episodes,
            seed_start=args.screen_seed_start,
            is_flow=True,
        )
        for checkpoint in checkpoints
    ]


def _selected_job(
    args: argparse.Namespace,
    checkpoint: FlowCheckpoint,
    *,
    split: str,
    episodes: int,
    seed_start: int,
) -> EvalJob:
    return EvalJob(
        label=checkpoint.run_label,
        run_dir=checkpoint.run_dir,
        snapshot=checkpoint.snapshot,
        output=args.output_dir / split / f"{checkpoint.run_label}.json",
        work_dir=args.work_root / split / checkpoint.run_label,
        log=args.output_dir / split / f"{checkpoint.run_label}.log",
        episodes=episodes,
        seed_start=seed_start,
        is_flow=True,
    )


def _clean_job(
    args: argparse.Namespace,
    *,
    split: str,
    episodes: int,
    seed_start: int,
) -> EvalJob:
    return EvalJob(
        label="clean_cqn_as",
        run_dir=args.clean_run_dir,
        snapshot=args.clean_snapshot,
        output=args.output_dir / split / "clean_cqn_as.json",
        work_dir=args.work_root / split / "clean_cqn_as",
        log=args.output_dir / split / "clean_cqn_as.log",
        episodes=episodes,
        seed_start=seed_start,
        is_flow=False,
    )


def run_gate(args: argparse.Namespace) -> dict:
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = discover_flow_checkpoints(args.flow_run)
    screen_payloads = run_jobs_parallel(
        _screen_jobs(args, checkpoints),
        gpu_ids=args.gpu_ids,
        num_flow_steps=args.num_flow_steps,
        num_action_flow_samples=args.num_action_flow_samples,
        progress_path=args.output_dir / "progress.json",
        phase="checkpoint_screen",
    )
    selected = [
        select_checkpoint(
            checkpoints,
            screen_payloads,
            run_label=run_label,
        )
        for run_label, _ in args.flow_run
    ]
    screen_summary = {
        checkpoint.run_label: {
            "screen_success": _success(screen_payloads[checkpoint.label]),
            "snapshot": str(checkpoint.snapshot),
            "step": checkpoint.step,
        }
        for checkpoint in selected
    }
    _atomic_json(args.output_dir / "screen_summary.json", screen_summary)

    validation_jobs = [
        _selected_job(
            args,
            checkpoint,
            split="validation",
            episodes=args.validation_episodes,
            seed_start=args.validation_seed_start,
        )
        for checkpoint in selected
    ]
    validation_jobs.append(
        _clean_job(
            args,
            split="validation",
            episodes=args.validation_episodes,
            seed_start=args.validation_seed_start,
        )
    )
    validation_payloads = run_jobs_parallel(
        validation_jobs,
        gpu_ids=args.gpu_ids,
        num_flow_steps=args.num_flow_steps,
        num_action_flow_samples=args.num_action_flow_samples,
        progress_path=args.output_dir / "progress.json",
        phase="validation",
    )
    order_by_label = {
        checkpoint.run_label: checkpoint.run_order
        for checkpoint in selected
    }
    candidate = max(
        selected,
        key=lambda checkpoint: (
            _success(validation_payloads[checkpoint.run_label]),
            screen_summary[checkpoint.run_label]["screen_success"],
            -order_by_label[checkpoint.run_label],
        ),
    )
    validation_row = paired_statistics(
        validation_payloads[candidate.run_label],
        validation_payloads["clean_cqn_as"],
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    validation_gate = (
        validation_row["paired_delta"] >= args.min_validation_delta
        and validation_row["paired_wins"] > validation_row["paired_losses"]
    )
    summary = {
        "action_flow_samples": args.num_action_flow_samples,
        "clean_run_dir": str(args.clean_run_dir),
        "clean_snapshot": str(args.clean_snapshot),
        "elapsed_seconds": time.time() - started,
        "flow_steps": args.num_flow_steps,
        "mean_terminal_return_readout": True,
        "screen_episodes": args.screen_episodes,
        "screen_seed_start": args.screen_seed_start,
        "screen_selected": screen_summary,
        "selected_candidate": {
            "label": candidate.run_label,
            "run_dir": str(candidate.run_dir),
            "snapshot": str(candidate.snapshot),
            "step": candidate.step,
        },
        "status": "ok",
        "validation_episodes": args.validation_episodes,
        "validation_gate": "pass" if validation_gate else "fail",
        "validation_gate_reason": (
            "positive validation margin opened confirmation"
            if validation_gate
            else "validation margin/win gate failed; confirmation remained sealed"
        ),
        "validation_results": {
            label: _success(payload)
            for label, payload in validation_payloads.items()
        },
        "validation_row": validation_row,
        "validation_seed_start": args.validation_seed_start,
    }
    if not validation_gate:
        summary["confirmation_gate"] = "not_run"
        _atomic_json(args.output_dir / "gate_summary.json", summary)
        return summary

    confirmation_jobs = [
        _selected_job(
            args,
            candidate,
            split="confirmation",
            episodes=args.confirmation_episodes,
            seed_start=args.confirmation_seed_start,
        ),
        _clean_job(
            args,
            split="confirmation",
            episodes=args.confirmation_episodes,
            seed_start=args.confirmation_seed_start,
        ),
    ]
    confirmation_payloads = run_jobs_parallel(
        confirmation_jobs,
        gpu_ids=args.gpu_ids,
        num_flow_steps=args.num_flow_steps,
        num_action_flow_samples=args.num_action_flow_samples,
        progress_path=args.output_dir / "progress.json",
        phase="confirmation",
    )
    confirmation_row = paired_statistics(
        confirmation_payloads[candidate.run_label],
        confirmation_payloads["clean_cqn_as"],
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed + 1,
    )
    gate, reason = confirmation_gate(confirmation_row)
    summary.update(
        {
            "confirmation_episodes": args.confirmation_episodes,
            "confirmation_gate": gate,
            "confirmation_gate_reason": reason,
            "confirmation_row": confirmation_row,
            "confirmation_seed_start": args.confirmation_seed_start,
            "elapsed_seconds": time.time() - started,
        }
    )
    _atomic_json(args.output_dir / "gate_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.num_flow_steps < 1 or args.num_action_flow_samples < 1:
        raise ValueError("flow steps and action-flow samples must be positive")
    for name in (
        "screen_episodes",
        "validation_episodes",
        "confirmation_episodes",
        "bootstrap_samples",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if not math.isfinite(args.min_validation_delta):
        raise ValueError("min-validation-delta must be finite")
    summary = run_gate(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
