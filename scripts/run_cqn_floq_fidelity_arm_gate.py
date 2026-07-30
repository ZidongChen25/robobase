#!/usr/bin/env python3
"""Select one expected-FLOQ fidelity arm, then confirm it once.

Every arm is first reduced to ``top_k`` checkpoints on a small screen split.
A disjoint validation split jointly selects one arm, checkpoint, and global
BC-prior beta.  Only that frozen choice is compared with validation-selected
clean CQN-AS on sealed confirmation seeds.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    from run_cqn_floq_checkpoint_selection_gate import (
        _positive_int,
        select_top_steps,
    )
except ImportError:
    from scripts.run_cqn_floq_checkpoint_selection_gate import (
        _positive_int,
        select_top_steps,
    )


@dataclass(frozen=True)
class Candidate:
    label: str
    run_dir: Path
    order: int
    return_sample_aggregation: str = "config"
    num_action_flow_samples: int | None = None
    return_sample_truncate_top: int | None = None


@dataclass(frozen=True)
class EvalJob:
    kind: str
    candidate: Candidate | None = None
    step: int | None = None
    beta: float | None = None


def _candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "candidate must be LABEL=RUN_DIR"
        )
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError(
            "candidate must be LABEL=RUN_DIR"
        )
    return label, Path(path)


def _finite_nonnegative(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise argparse.ArgumentTypeError(
            "expected a finite non-negative number"
        )
    return number


def _labeled_aggregation(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "candidate aggregation must be LABEL=AGGREGATION"
        )
    label, aggregation = value.split("=", 1)
    if aggregation not in {"config", "mean", "entropic", "truncated_mean"}:
        raise argparse.ArgumentTypeError(
            "candidate aggregation must be config, mean, entropic, or "
            "truncated_mean"
        )
    if not label:
        raise argparse.ArgumentTypeError("candidate label cannot be empty")
    return label, aggregation


def _labeled_positive_int(value: str) -> tuple[str, int]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "candidate sample count must be LABEL=COUNT"
        )
    label, raw_count = value.split("=", 1)
    count = int(raw_count)
    if not label or count < 1:
        raise argparse.ArgumentTypeError(
            "candidate sample count must use a non-empty label and "
            "positive count"
        )
    return label, count


def _labeled_nonnegative_int(value: str) -> tuple[str, int]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "candidate truncation must be LABEL=COUNT"
        )
    label, raw_count = value.split("=", 1)
    count = int(raw_count)
    if not label or count < 0:
        raise argparse.ArgumentTypeError(
            "candidate truncation must use a non-empty label and "
            "non-negative count"
        )
    return label, count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-run-dir", required=True, type=Path)
    parser.add_argument("--clean-snapshot", required=True, type=Path)
    parser.add_argument(
        "--candidate",
        required=True,
        action="append",
        type=_candidate,
    )
    parser.add_argument("--gpu-id", required=True, action="append", type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument(
        "--checkpoint-step",
        nargs="+",
        type=_positive_int,
        default=list(range(1_000, 10_001, 1_000)),
    )
    parser.add_argument("--screen-top-k", type=_positive_int, default=2)
    parser.add_argument(
        "--flow-readout",
        choices=("distill", "integrated"),
        default="distill",
    )
    parser.add_argument("--num-flow-steps", type=_positive_int)
    parser.add_argument(
        "--candidate-return-sample-aggregation",
        action="append",
        type=_labeled_aggregation,
        default=[],
    )
    parser.add_argument(
        "--candidate-action-flow-samples",
        action="append",
        type=_labeled_positive_int,
        default=[],
    )
    parser.add_argument(
        "--candidate-return-sample-truncate-top",
        action="append",
        type=_labeled_nonnegative_int,
        default=[],
    )
    parser.add_argument(
        "--beta",
        nargs="+",
        type=_finite_nonnegative,
        default=[0.3, 1.0, 3.0],
    )
    parser.add_argument(
        "--screen-beta",
        type=_finite_nonnegative,
        default=1.0,
    )
    parser.add_argument("--screen-episodes", type=_positive_int, default=10)
    parser.add_argument("--screen-seed-start", type=int, default=114_000)
    parser.add_argument(
        "--validation-episodes",
        type=_positive_int,
        default=50,
    )
    parser.add_argument(
        "--validation-seed-start",
        type=int,
        default=115_000,
    )
    parser.add_argument(
        "--confirmation-episodes",
        type=_positive_int,
        default=200,
    )
    parser.add_argument(
        "--confirmation-seed-start",
        type=int,
        default=116_000,
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=_positive_int,
        default=20_000,
    )
    parser.add_argument("--bootstrap-seed", type=int, default=116_200)
    parser.add_argument(
        "--min-validation-delta",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--min-confirmation-ci-lower",
        type=float,
        default=-0.05,
    )
    parser.add_argument("--required-selected-arm")
    parser.add_argument("--confirmation-baseline-arm")
    parser.add_argument(
        "--min-arm-confirmation-ci-lower",
        type=float,
        default=-0.05,
    )
    return parser.parse_args()


def _number_label(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _snapshot(candidate: Candidate, step: int) -> Path:
    return candidate.run_dir / "snapshots" / f"{step}_snapshot.pkl"


def _result_path(output_dir: Path, split: str, job: EvalJob) -> Path:
    if job.kind == "clean":
        filename = "clean.json"
    else:
        if job.candidate is None or job.step is None or job.beta is None:
            raise ValueError("candidate result requires arm, step, and beta")
        filename = (
            f"{job.candidate.label}_step{job.step}_"
            f"beta{_number_label(job.beta)}.json"
        )
    return output_dir / split / filename


def _completed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text()).get("status") == "ok"
    except (OSError, json.JSONDecodeError):
        return False


def build_eval_command(
    job: EvalJob,
    *,
    clean_run_dir: Path,
    clean_snapshot: Path,
    output: Path,
    work_dir: Path,
    gpu_id: int,
    episodes: int,
    seed_start: int,
    flow_readout: str,
    num_flow_steps: int | None,
) -> list[str]:
    candidate = job.kind == "candidate"
    if candidate:
        if job.candidate is None or job.step is None or job.beta is None:
            raise ValueError(
                "candidate evaluation requires arm, step, and beta"
            )
        run_dir = job.candidate.run_dir
        snapshot = _snapshot(job.candidate, job.step)
        beta = f"{job.beta:g}"
        readout = flow_readout
    else:
        if (
            job.kind != "clean"
            or job.candidate is not None
            or job.step is not None
            or job.beta is not None
        ):
            raise ValueError("invalid clean evaluation job")
        run_dir = clean_run_dir
        snapshot = clean_snapshot
        beta = "bc"
        readout = "auto"
    command = [
        sys.executable,
        str(Path(__file__).with_name("eval_cqn_flow_policy_value.py")),
        "--run-dir",
        str(run_dir),
        "--snapshot",
        str(snapshot),
        "--output",
        str(output),
        "--work-dir",
        str(work_dir),
        "--gpu-id",
        str(gpu_id),
        "--num-eval-episodes",
        str(episodes),
        "--eval-seed-start",
        str(seed_start),
        "--policy-value-beta",
        beta,
        "--flow-readout",
        readout,
    ]
    if candidate and num_flow_steps is not None:
        command.extend(["--num-flow-steps", str(num_flow_steps)])
    if candidate and job.candidate is not None:
        if job.candidate.return_sample_aggregation != "config":
            command.extend(
                [
                    "--return-sample-aggregation",
                    job.candidate.return_sample_aggregation,
                ]
            )
        if job.candidate.num_action_flow_samples is not None:
            command.extend(
                [
                    "--num-action-flow-samples",
                    str(job.candidate.num_action_flow_samples),
                ]
            )
        if job.candidate.return_sample_truncate_top is not None:
            command.extend(
                [
                    "--return-sample-truncate-top",
                    str(job.candidate.return_sample_truncate_top),
                ]
            )
    return command


def _run_logged(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        stream.write("$ " + " ".join(command) + "\n")
        stream.flush()
        subprocess.run(
            command,
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )


def _run_jobs(
    args: argparse.Namespace,
    *,
    split: str,
    jobs: list[EvalJob],
    episodes: int,
    seed_start: int,
) -> None:
    work_queue: queue.Queue[EvalJob] = queue.Queue()
    for job in jobs:
        work_queue.put(job)
    failures: list[BaseException] = []
    failure_lock = threading.Lock()

    def worker(gpu_id: int) -> None:
        while True:
            try:
                job = work_queue.get_nowait()
            except queue.Empty:
                return
            try:
                output = _result_path(args.output_dir, split, job)
                if not _completed(output):
                    command = build_eval_command(
                        job,
                        clean_run_dir=args.clean_run_dir,
                        clean_snapshot=args.clean_snapshot,
                        output=output,
                        work_dir=args.work_root / split / output.stem,
                        gpu_id=gpu_id,
                        episodes=episodes,
                        seed_start=seed_start,
                        flow_readout=args.flow_readout,
                        num_flow_steps=args.num_flow_steps,
                    )
                    _run_logged(command, output.with_suffix(".log"))
                if not _completed(output):
                    raise RuntimeError(
                        f"evaluation did not complete: {output}"
                    )
            except BaseException as exc:
                with failure_lock:
                    failures.append(exc)
            finally:
                work_queue.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu_id,), daemon=False)
        for gpu_id in args.gpu_id
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(
            f"{len(failures)} evaluation worker(s) failed"
        ) from failures[0]


def _payload(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("status") != "ok":
        raise ValueError(f"incomplete evaluation: {path}")
    return payload


def _success(path: Path) -> float:
    return float(_payload(path)["episode_success"])


def paired_result(
    clean_payload: dict,
    candidate_payload: dict,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict:
    clean_by_seed = {
        int(row["seed"]): float(row["episode_success"])
        for row in clean_payload["episode_results"]
    }
    candidate_by_seed = {
        int(row["seed"]): float(row["episode_success"])
        for row in candidate_payload["episode_results"]
    }
    if list(clean_by_seed) != list(candidate_by_seed):
        raise ValueError("clean and candidate episode seeds do not match")
    seeds = np.asarray(list(clean_by_seed), dtype=np.int64)
    clean = np.asarray(
        [clean_by_seed[int(seed)] for seed in seeds],
        dtype=np.float64,
    )
    candidate = np.asarray(
        [candidate_by_seed[int(seed)] for seed in seeds],
        dtype=np.float64,
    )
    delta = candidate - clean
    rng = np.random.default_rng(int(bootstrap_seed))
    indices = rng.integers(
        0,
        delta.size,
        size=(int(bootstrap_replicates), delta.size),
    )
    boot = delta[indices].mean(axis=1)
    wins = int(np.sum(delta > 0))
    losses = int(np.sum(delta < 0))
    return {
        "num_eval_seeds": int(delta.size),
        "eval_seed_start": int(seeds[0]),
        "eval_seed_end": int(seeds[-1]),
        "clean_success": float(clean.mean()),
        "candidate_success": float(candidate.mean()),
        "paired_delta": float(delta.mean()),
        "paired_delta_ci95": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "paired_wins": wins,
        "paired_losses": losses,
        "paired_ties": int(delta.size - wins - losses),
    }


def select_validation_winner(
    rows: list[tuple[Candidate, int, float, float]],
) -> tuple[Candidate, int, float, float]:
    """Select success first; ties prefer BC, early step, then arm order."""

    if not rows:
        raise ValueError("validation rows are empty")
    return max(
        rows,
        key=lambda row: (
            float(row[3]),
            float(row[2]),
            -int(row[1]),
            -int(row[0].order),
        ),
    )


def validation_gate(result: dict, min_delta: float) -> tuple[bool, dict]:
    checks = {
        "paired_delta_at_least_threshold": (
            float(result["paired_delta"]) >= float(min_delta)
        ),
        "paired_wins_above_losses": (
            int(result["paired_wins"]) > int(result["paired_losses"])
        ),
    }
    return all(checks.values()), checks


def confirmation_gate(
    result: dict,
    min_ci_lower: float,
) -> tuple[bool, dict]:
    checks = {
        "positive_paired_delta": float(result["paired_delta"]) > 0.0,
        "paired_wins_above_losses": (
            int(result["paired_wins"]) > int(result["paired_losses"])
        ),
        "ci_lower_at_least_promotion_margin": (
            float(result["paired_delta_ci95"][0])
            >= float(min_ci_lower)
        ),
    }
    return all(checks.values()), checks


def run_gate(args: argparse.Namespace) -> dict:
    if len(args.candidate) < 2:
        raise ValueError("at least two fidelity arms are required")
    labels = [label for label, _ in args.candidate]
    if len(set(labels)) != len(labels):
        raise ValueError("candidate labels must be unique")
    readout_maps = {}
    for name, entries in (
        (
            "return_sample_aggregation",
            args.candidate_return_sample_aggregation,
        ),
        (
            "num_action_flow_samples",
            args.candidate_action_flow_samples,
        ),
        (
            "return_sample_truncate_top",
            args.candidate_return_sample_truncate_top,
        ),
    ):
        mapping = {}
        for label, setting in entries:
            if label in mapping:
                raise ValueError(
                    f"duplicate {name} override for candidate {label}"
                )
            mapping[label] = setting
        unknown = set(mapping).difference(labels)
        if unknown:
            raise ValueError(
                f"{name} override references unknown candidates: "
                f"{sorted(unknown)}"
            )
        readout_maps[name] = mapping
    candidates = []
    for order, (label, path) in enumerate(args.candidate):
        candidate = Candidate(
            label,
            path.expanduser().resolve(),
            order,
            return_sample_aggregation=readout_maps[
                "return_sample_aggregation"
            ].get(label, "config"),
            num_action_flow_samples=readout_maps[
                "num_action_flow_samples"
            ].get(label),
            return_sample_truncate_top=readout_maps[
                "return_sample_truncate_top"
            ].get(label),
        )
        if candidate.return_sample_aggregation == "truncated_mean":
            if (
                candidate.num_action_flow_samples is None
                or candidate.return_sample_truncate_top is None
                or candidate.return_sample_truncate_top < 1
                or candidate.return_sample_truncate_top
                >= candidate.num_action_flow_samples
            ):
                raise ValueError(
                    "truncated_mean candidates require an explicit action "
                    "sample count and truncation in [1, samples)."
                )
        candidates.append(candidate)
    if not args.gpu_id or len(set(args.gpu_id)) != len(args.gpu_id):
        raise ValueError("gpu-id workers must be non-empty and unique")
    if len(set(args.checkpoint_step)) != len(args.checkpoint_step):
        raise ValueError("checkpoint steps must be unique")
    if args.screen_top_k > len(args.checkpoint_step):
        raise ValueError("screen-top-k exceeds checkpoint count")
    if not args.beta or len(set(args.beta)) != len(args.beta):
        raise ValueError("beta grid must be non-empty and unique")
    if args.num_flow_steps is not None and args.flow_readout != "integrated":
        raise ValueError("num-flow-steps requires integrated readout")
    if not all(
        math.isfinite(value)
        for value in (
            args.min_validation_delta,
            args.min_confirmation_ci_lower,
            args.min_arm_confirmation_ci_lower,
        )
    ):
        raise ValueError("gate thresholds must be finite")
    if (
        args.required_selected_arm is not None
        and args.required_selected_arm not in labels
    ):
        raise ValueError("required-selected-arm is not a declared candidate")
    if (
        args.confirmation_baseline_arm is not None
        and args.confirmation_baseline_arm not in labels
    ):
        raise ValueError(
            "confirmation-baseline-arm is not a declared candidate"
        )
    seed_ranges = [
        set(range(start, start + episodes))
        for start, episodes in (
            (args.screen_seed_start, args.screen_episodes),
            (args.validation_seed_start, args.validation_episodes),
            (
                args.confirmation_seed_start,
                args.confirmation_episodes,
            ),
        )
    ]
    if any(
        left & right
        for index, left in enumerate(seed_ranges)
        for right in seed_ranges[index + 1 :]
    ):
        raise ValueError(
            "screen/validation/confirmation seeds must be disjoint"
        )

    args.clean_run_dir = args.clean_run_dir.expanduser().resolve()
    args.clean_snapshot = args.clean_snapshot.expanduser().resolve()
    if not (args.clean_run_dir / ".hydra" / "config.yaml").is_file():
        raise FileNotFoundError(args.clean_run_dir)
    if not args.clean_snapshot.is_file():
        raise FileNotFoundError(args.clean_snapshot)
    for candidate in candidates:
        if not (candidate.run_dir / ".hydra" / "config.yaml").is_file():
            raise FileNotFoundError(candidate.run_dir)
        for step in args.checkpoint_step:
            snapshot = _snapshot(candidate, step)
            if not snapshot.is_file():
                raise FileNotFoundError(snapshot)

    args.output_dir = args.output_dir.expanduser().resolve()
    args.work_root = args.work_root.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = {
        "status": "running",
        "clean_run_dir": str(args.clean_run_dir),
        "clean_snapshot": str(args.clean_snapshot),
        "candidates": [asdict(candidate) for candidate in candidates],
        "checkpoint_steps": list(args.checkpoint_step),
        "screen_top_k": args.screen_top_k,
        "flow_readout": args.flow_readout,
        "num_flow_steps": args.num_flow_steps,
        "beta_grid": list(args.beta),
        "screen_beta": args.screen_beta,
        "screen_seed_start": args.screen_seed_start,
        "validation_seed_start": args.validation_seed_start,
        "confirmation_seed_start": args.confirmation_seed_start,
        "required_selected_arm": args.required_selected_arm,
        "confirmation_baseline_arm": args.confirmation_baseline_arm,
        "min_arm_confirmation_ci_lower": (
            args.min_arm_confirmation_ci_lower
        ),
    }
    manifest["candidates"] = [
        {
            **row,
            "run_dir": str(row["run_dir"]),
        }
        for row in manifest["candidates"]
    ]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    started = time.time()

    screen_jobs = [
        EvalJob("candidate", candidate, step, args.screen_beta)
        for candidate in candidates
        for step in args.checkpoint_step
    ]
    _run_jobs(
        args,
        split="screen",
        jobs=screen_jobs,
        episodes=args.screen_episodes,
        seed_start=args.screen_seed_start,
    )
    screen = {}
    retained = {}
    for candidate in candidates:
        rows = {
            step: _success(
                _result_path(
                    args.output_dir,
                    "screen",
                    EvalJob(
                        "candidate",
                        candidate,
                        step,
                        args.screen_beta,
                    ),
                )
            )
            for step in args.checkpoint_step
        }
        screen[candidate.label] = rows
        retained[candidate.label] = select_top_steps(
            rows,
            top_k=args.screen_top_k,
        )

    validation_jobs = [EvalJob("clean")]
    validation_jobs.extend(
        EvalJob("candidate", candidate, step, beta)
        for candidate in candidates
        for step in retained[candidate.label]
        for beta in args.beta
    )
    _run_jobs(
        args,
        split="validation",
        jobs=validation_jobs,
        episodes=args.validation_episodes,
        seed_start=args.validation_seed_start,
    )
    validation_rows = [
        (
            candidate,
            step,
            float(beta),
            _success(
                _result_path(
                    args.output_dir,
                    "validation",
                    EvalJob("candidate", candidate, step, beta),
                )
            ),
        )
        for candidate in candidates
        for step in retained[candidate.label]
        for beta in args.beta
    ]
    winner, winner_step, winner_beta, winner_success = (
        select_validation_winner(validation_rows)
    )
    clean_validation = _payload(
        _result_path(
            args.output_dir,
            "validation",
            EvalJob("clean"),
        )
    )
    winner_validation = _payload(
        _result_path(
            args.output_dir,
            "validation",
            EvalJob(
                "candidate",
                winner,
                winner_step,
                winner_beta,
            ),
        )
    )
    validation_result = paired_result(
        clean_validation,
        winner_validation,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed - 1,
    )
    validation_pass, validation_checks = validation_gate(
        validation_result,
        args.min_validation_delta,
    )
    required_arm_pass = (
        args.required_selected_arm is None
        or winner.label == args.required_selected_arm
    )
    validation_checks["required_arm_selected"] = required_arm_pass
    validation_pass = validation_pass and required_arm_pass
    baseline_selection = None
    baseline_candidate = None
    baseline_step = None
    baseline_beta = None
    if args.confirmation_baseline_arm is not None:
        baseline_rows = [
            row
            for row in validation_rows
            if row[0].label == args.confirmation_baseline_arm
        ]
        (
            baseline_candidate,
            baseline_step,
            baseline_beta,
            baseline_success,
        ) = select_validation_winner(baseline_rows)
        baseline_selection = {
            "arm": baseline_candidate.label,
            "step": baseline_step,
            "beta": baseline_beta,
            "validation_success": baseline_success,
        }
        if baseline_candidate.label == winner.label:
            validation_checks["winner_differs_from_confirmation_baseline"] = (
                False
            )
            validation_pass = False
        else:
            validation_checks["winner_differs_from_confirmation_baseline"] = (
                True
            )

    payload = {
        "status": "ok",
        "selection": {
            "screen": screen,
            "retained_steps": retained,
            "validation_rows": [
                {
                    "arm": candidate.label,
                    "step": step,
                    "beta": beta,
                    "success": success,
                }
                for candidate, step, beta, success in validation_rows
            ],
            "selected_arm": winner.label,
            "selected_step": winner_step,
            "selected_beta": winner_beta,
            "selected_validation_success": winner_success,
            "confirmation_baseline": baseline_selection,
            "tie_break": (
                "larger beta, earlier checkpoint, earlier declared arm"
            ),
        },
        "validation_result": validation_result,
        "validation_thresholds": {
            "min_paired_delta": args.min_validation_delta,
        },
        "validation_checks": validation_checks,
        "validation_gate": "pass" if validation_pass else "fail",
        "confirmation_result": None,
        "confirmation_checks": None,
        "confirmation_gate": "not_run",
        "arm_confirmation_result": None,
        "arm_confirmation_checks": None,
        "arm_confirmation_gate": (
            "not_requested"
            if args.confirmation_baseline_arm is None
            else "not_run"
        ),
        "promotion": "not_run",
        "flow_readout": args.flow_readout,
        "num_flow_steps": args.num_flow_steps,
        "manifest": str(manifest_path),
    }

    if validation_pass:
        confirmation_jobs = [
            EvalJob("clean"),
            EvalJob(
                "candidate",
                winner,
                winner_step,
                winner_beta,
            ),
        ]
        if baseline_candidate is not None:
            confirmation_jobs.append(
                EvalJob(
                    "candidate",
                    baseline_candidate,
                    baseline_step,
                    baseline_beta,
                )
            )
        _run_jobs(
            args,
            split="confirmation",
            jobs=confirmation_jobs,
            episodes=args.confirmation_episodes,
            seed_start=args.confirmation_seed_start,
        )
        confirmation_result = paired_result(
            _payload(
                _result_path(
                    args.output_dir,
                    "confirmation",
                    EvalJob("clean"),
                )
            ),
            _payload(
                _result_path(
                    args.output_dir,
                    "confirmation",
                    EvalJob(
                        "candidate",
                        winner,
                        winner_step,
                        winner_beta,
                    ),
                )
            ),
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
        confirmation_pass, confirmation_checks = confirmation_gate(
            confirmation_result,
            args.min_confirmation_ci_lower,
        )
        arm_confirmation_result = None
        arm_confirmation_checks = None
        arm_confirmation_pass = True
        if baseline_candidate is not None:
            arm_confirmation_result = paired_result(
                _payload(
                    _result_path(
                        args.output_dir,
                        "confirmation",
                        EvalJob(
                            "candidate",
                            baseline_candidate,
                            baseline_step,
                            baseline_beta,
                        ),
                    )
                ),
                _payload(
                    _result_path(
                        args.output_dir,
                        "confirmation",
                        EvalJob(
                            "candidate",
                            winner,
                            winner_step,
                            winner_beta,
                        ),
                    )
                ),
                bootstrap_replicates=args.bootstrap_replicates,
                bootstrap_seed=args.bootstrap_seed + 1,
            )
            (
                arm_confirmation_pass,
                arm_confirmation_checks,
            ) = confirmation_gate(
                arm_confirmation_result,
                args.min_arm_confirmation_ci_lower,
            )
        overall_confirmation_pass = (
            confirmation_pass and arm_confirmation_pass
        )
        payload.update(
            {
                "confirmation_result": confirmation_result,
                "confirmation_checks": confirmation_checks,
                "confirmation_gate": (
                    "pass" if confirmation_pass else "fail"
                ),
                "arm_confirmation_result": arm_confirmation_result,
                "arm_confirmation_checks": arm_confirmation_checks,
                "arm_confirmation_gate": (
                    "not_requested"
                    if baseline_candidate is None
                    else "pass"
                    if arm_confirmation_pass
                    else "fail"
                ),
                "promotion": (
                    "pass" if overall_confirmation_pass else "fail"
                ),
            }
        )
    else:
        payload["promotion"] = "fail"

    payload["elapsed_seconds"] = time.time() - started
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    manifest["status"] = "ok"
    manifest["summary"] = str(summary_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return payload


def main() -> int:
    args = parse_args()
    payload = run_gate(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
