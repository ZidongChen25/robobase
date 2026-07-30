#!/usr/bin/env python3
"""Select candidate checkpoints and one global BC-prior beta, then confirm.

The screen split uses one predeclared beta only to reduce each training run to
``top_k`` checkpoints.  A disjoint validation split evaluates every retained
checkpoint at every candidate beta, selects one global beta across training
seeds, and then selects one checkpoint per training seed at that beta.  The
frozen choices are compared with clean CQN-AS once on sealed confirmation
seeds.
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

try:
    from run_cqn_floq_checkpoint_selection_gate import (
        TrainingSeed,
        _positive_int,
        _resolved,
        _training_seed,
        select_top_steps,
        select_winner,
    )
    from summarize_cqn_multiseed_paired import summarize
except ImportError:
    from scripts.run_cqn_floq_checkpoint_selection_gate import (
        TrainingSeed,
        _positive_int,
        _resolved,
        _training_seed,
        select_top_steps,
        select_winner,
    )
    from scripts.summarize_cqn_multiseed_paired import summarize


@dataclass(frozen=True)
class EvalJob:
    training_seed: TrainingSeed
    kind: str
    step: int | None
    beta: float | None


def _finite_nonnegative(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise argparse.ArgumentTypeError(
            "expected a finite non-negative number"
        )
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-seed",
        required=True,
        action="append",
        type=_training_seed,
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
        "--candidate-readout",
        choices=("auto", "distill", "integrated"),
        default="auto",
    )
    parser.add_argument("--num-flow-steps", type=_positive_int)
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
    parser.add_argument("--screen-seed-start", type=int, default=111_000)
    parser.add_argument(
        "--validation-episodes",
        type=_positive_int,
        default=50,
    )
    parser.add_argument(
        "--validation-seed-start",
        type=int,
        default=112_000,
    )
    parser.add_argument(
        "--confirmation-episodes",
        type=_positive_int,
        default=200,
    )
    parser.add_argument(
        "--confirmation-seed-start",
        type=int,
        default=113_000,
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=_positive_int,
        default=20_000,
    )
    parser.add_argument("--bootstrap-seed", type=int, default=113_200)
    parser.add_argument("--min-mean-delta", type=float, default=0.0)
    parser.add_argument("--min-ci-lower", type=float, default=0.0)
    return parser.parse_args()


def _candidate_snapshot(item: TrainingSeed, step: int) -> Path:
    return item.flow_run_dir / "snapshots" / f"{step}_snapshot.pkl"


def _number_label(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _result_path(output_dir: Path, split: str, job: EvalJob) -> Path:
    if job.kind == "clean":
        filename = "clean.json"
    else:
        filename = (
            f"candidate_step{int(job.step)}_"
            f"beta{_number_label(float(job.beta))}.json"
        )
    return output_dir / split / job.training_seed.label / filename


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
    output: Path,
    work_dir: Path,
    gpu_id: int,
    episodes: int,
    seed_start: int,
    candidate_readout: str,
    num_flow_steps: int | None,
) -> list[str]:
    candidate = job.kind == "candidate"
    if candidate and (job.step is None or job.beta is None):
        raise ValueError("candidate evaluation requires step and beta")
    if not candidate and (job.step is not None or job.beta is not None):
        raise ValueError("clean evaluation cannot take step or beta")
    run_dir = (
        job.training_seed.flow_run_dir
        if candidate
        else job.training_seed.clean_run_dir
    )
    snapshot = (
        _candidate_snapshot(job.training_seed, int(job.step))
        if candidate
        else job.training_seed.clean_snapshot
    )
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
        f"{float(job.beta):g}" if candidate else "bc",
        "--flow-readout",
        candidate_readout if candidate else "auto",
    ]
    if candidate and num_flow_steps is not None:
        command.extend(["--num-flow-steps", str(num_flow_steps)])
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
                        output=output,
                        work_dir=(
                            args.work_root
                            / split
                            / job.training_seed.label
                            / output.stem
                        ),
                        gpu_id=gpu_id,
                        episodes=episodes,
                        seed_start=seed_start,
                        candidate_readout=args.candidate_readout,
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


def _success(path: Path) -> float:
    payload = json.loads(path.read_text())
    if payload.get("status") != "ok":
        raise ValueError(f"incomplete evaluation: {path}")
    return float(payload["episode_success"])


def select_global_beta_and_steps(
    validation: dict[str, dict[float, dict[int, float]]],
) -> tuple[float, dict[str, int], dict[float, float]]:
    """Choose one beta by mean best-per-seed validation success."""

    if len(validation) < 2:
        raise ValueError("at least two training seeds are required")
    beta_sets = [set(rows) for rows in validation.values()]
    if not beta_sets or any(beta_set != beta_sets[0] for beta_set in beta_sets):
        raise ValueError("training seeds do not share the same beta grid")
    if not beta_sets[0]:
        raise ValueError("validation beta grid is empty")

    selected_by_beta: dict[float, dict[str, int]] = {}
    mean_by_beta: dict[float, float] = {}
    for beta in sorted(beta_sets[0]):
        selected_by_beta[beta] = {
            label: select_winner(beta_rows[beta])
            for label, beta_rows in validation.items()
        }
        mean_by_beta[beta] = sum(
            validation[label][beta][step]
            for label, step in selected_by_beta[beta].items()
        ) / len(validation)
    selected_beta = max(
        mean_by_beta,
        key=lambda beta: (mean_by_beta[beta], beta),
    )
    return (
        float(selected_beta),
        selected_by_beta[selected_beta],
        mean_by_beta,
    )


def run_gate(args: argparse.Namespace) -> dict:
    if len(args.training_seed) < 2:
        raise ValueError("at least two training seeds are required")
    args.training_seed = [_resolved(item) for item in args.training_seed]
    if len({item.label for item in args.training_seed}) != len(
        args.training_seed
    ):
        raise ValueError("training seed labels must be unique")
    if not args.gpu_id or len(set(args.gpu_id)) != len(args.gpu_id):
        raise ValueError("gpu-id workers must be non-empty and unique")
    if len(set(args.checkpoint_step)) != len(args.checkpoint_step):
        raise ValueError("checkpoint steps must be unique")
    if args.screen_top_k > len(args.checkpoint_step):
        raise ValueError("screen-top-k exceeds checkpoint count")
    if not args.beta or len(set(args.beta)) != len(args.beta):
        raise ValueError("beta grid must be non-empty and unique")
    if (
        args.num_flow_steps is not None
        and args.candidate_readout != "integrated"
    ):
        raise ValueError("num-flow-steps requires integrated readout")
    if not all(
        math.isfinite(value)
        for value in (args.min_mean_delta, args.min_ci_lower)
    ):
        raise ValueError("gate thresholds must be finite")

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

    for item in args.training_seed:
        if not (item.clean_run_dir / ".hydra" / "config.yaml").is_file():
            raise FileNotFoundError(item.clean_run_dir)
        if not item.clean_snapshot.is_file():
            raise FileNotFoundError(item.clean_snapshot)
        if not (item.flow_run_dir / ".hydra" / "config.yaml").is_file():
            raise FileNotFoundError(item.flow_run_dir)
        for step in args.checkpoint_step:
            snapshot = _candidate_snapshot(item, step)
            if not snapshot.is_file():
                raise FileNotFoundError(snapshot)

    args.output_dir = args.output_dir.expanduser().resolve()
    args.work_root = args.work_root.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = {
        "status": "running",
        "training_seeds": [
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(item).items()
            }
            for item in args.training_seed
        ],
        "gpu_ids": list(args.gpu_id),
        "checkpoint_steps": list(args.checkpoint_step),
        "screen_top_k": args.screen_top_k,
        "candidate_readout": args.candidate_readout,
        "num_flow_steps": args.num_flow_steps,
        "beta_grid": list(args.beta),
        "screen_beta": args.screen_beta,
        "screen_seed_start": args.screen_seed_start,
        "validation_seed_start": args.validation_seed_start,
        "confirmation_seed_start": args.confirmation_seed_start,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    started = time.time()

    screen_jobs = [
        EvalJob(item, "candidate", step, args.screen_beta)
        for item in args.training_seed
        for step in args.checkpoint_step
    ]
    _run_jobs(
        args,
        split="screen",
        jobs=screen_jobs,
        episodes=args.screen_episodes,
        seed_start=args.screen_seed_start,
    )
    screen_rows = {}
    selected_for_validation = {}
    for item in args.training_seed:
        rows = {
            step: _success(
                _result_path(
                    args.output_dir,
                    "screen",
                    EvalJob(
                        item,
                        "candidate",
                        step,
                        args.screen_beta,
                    ),
                )
            )
            for step in args.checkpoint_step
        }
        screen_rows[item.label] = rows
        selected_for_validation[item.label] = select_top_steps(
            rows,
            top_k=args.screen_top_k,
        )

    validation_jobs = [
        EvalJob(item, "candidate", step, beta)
        for item in args.training_seed
        for step in selected_for_validation[item.label]
        for beta in args.beta
    ]
    _run_jobs(
        args,
        split="validation",
        jobs=validation_jobs,
        episodes=args.validation_episodes,
        seed_start=args.validation_seed_start,
    )
    validation_rows: dict[str, dict[float, dict[int, float]]] = {}
    for item in args.training_seed:
        validation_rows[item.label] = {
            float(beta): {
                step: _success(
                    _result_path(
                        args.output_dir,
                        "validation",
                        EvalJob(item, "candidate", step, beta),
                    )
                )
                for step in selected_for_validation[item.label]
            }
            for beta in args.beta
        }
    selected_beta, selected_steps, mean_by_beta = (
        select_global_beta_and_steps(validation_rows)
    )

    confirmation_jobs = []
    for item in args.training_seed:
        confirmation_jobs.extend(
            [
                EvalJob(item, "clean", None, None),
                EvalJob(
                    item,
                    "candidate",
                    selected_steps[item.label],
                    selected_beta,
                ),
            ]
        )
    _run_jobs(
        args,
        split="confirmation",
        jobs=confirmation_jobs,
        episodes=args.confirmation_episodes,
        seed_start=args.confirmation_seed_start,
    )
    summary = summarize(
        [
            (
                item.label,
                _result_path(
                    args.output_dir,
                    "confirmation",
                    EvalJob(item, "clean", None, None),
                ),
                _result_path(
                    args.output_dir,
                    "confirmation",
                    EvalJob(
                        item,
                        "candidate",
                        selected_steps[item.label],
                        selected_beta,
                    ),
                ),
            )
            for item in args.training_seed
        ],
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        min_mean_delta=args.min_mean_delta,
        min_ci_lower=args.min_ci_lower,
    )
    summary["selection"] = {
        "screen_beta": args.screen_beta,
        "screen": screen_rows,
        "selected_for_validation": selected_for_validation,
        "validation": validation_rows,
        "mean_best_validation_success_by_beta": mean_by_beta,
        "selected_global_beta": selected_beta,
        "selected_steps": selected_steps,
        "checkpoint_tie_break": "earlier_checkpoint",
        "beta_tie_break": "larger_bc_prior",
    }
    summary["candidate_readout"] = args.candidate_readout
    summary["num_flow_steps"] = args.num_flow_steps
    summary["elapsed_seconds"] = time.time() - started
    summary["manifest"] = str(manifest_path)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    manifest["status"] = "ok"
    manifest["summary"] = str(args.output_dir / "summary.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> int:
    args = parse_args()
    summary = run_gate(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
