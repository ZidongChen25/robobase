#!/usr/bin/env python3
"""Select the best candidate checkpoint per training seed, then confirm once.

A small screen narrows every training run to ``top_k`` checkpoints. A disjoint
validation split selects one checkpoint per training seed. Only those frozen
winners are compared with validation-selected clean CQN-AS checkpoints on a
third, sealed set of common simulator seeds. ``flow-readout=auto`` makes the
same protocol usable for a direct-C51 candidate without changing its action
readout.
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
    from summarize_cqn_multiseed_paired import summarize
except ImportError:
    from scripts.summarize_cqn_multiseed_paired import summarize


@dataclass(frozen=True)
class TrainingSeed:
    label: str
    clean_run_dir: Path
    clean_snapshot: Path
    flow_run_dir: Path


@dataclass(frozen=True)
class EvalJob:
    training_seed: TrainingSeed
    kind: str
    step: int | None


def _training_seed(value: str) -> TrainingSeed:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "training-seed must be "
            "LABEL=CLEAN_RUN,CLEAN_SNAPSHOT,FLOW_RUN"
        )
    label, raw_paths = value.split("=", 1)
    paths = raw_paths.split(",")
    if not label or len(paths) != 3 or not all(paths):
        raise argparse.ArgumentTypeError(
            "training-seed must be "
            "LABEL=CLEAN_RUN,CLEAN_SNAPSHOT,FLOW_RUN"
        )
    return TrainingSeed(label, *(Path(path) for path in paths))


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def _nonnegative(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise argparse.ArgumentTypeError("expected a non-negative number")
    return number


def _nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError(
            "expected a non-negative integer"
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
        "--flow-readout",
        choices=("auto", "distill", "integrated"),
        default="distill",
    )
    parser.add_argument("--num-flow-steps", type=_positive_int)
    parser.add_argument(
        "--return-sample-aggregation",
        choices=("config", "mean", "entropic", "truncated_mean"),
        default="config",
    )
    parser.add_argument(
        "--num-action-flow-samples",
        type=_positive_int,
    )
    parser.add_argument(
        "--return-sample-truncate-top",
        type=_nonnegative_int,
    )
    parser.add_argument(
        "--policy-value-beta",
        type=_nonnegative,
        default=1.0,
    )
    parser.add_argument("--screen-episodes", type=_positive_int, default=10)
    parser.add_argument("--screen-seed-start", type=int, default=101_000)
    parser.add_argument(
        "--validation-episodes",
        type=_positive_int,
        default=50,
    )
    parser.add_argument(
        "--validation-seed-start",
        type=int,
        default=102_000,
    )
    parser.add_argument(
        "--confirmation-episodes",
        type=_positive_int,
        default=200,
    )
    parser.add_argument(
        "--confirmation-seed-start",
        type=int,
        default=103_000,
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=_positive_int,
        default=20_000,
    )
    parser.add_argument("--bootstrap-seed", type=int, default=103_200)
    return parser.parse_args()


def _resolved(item: TrainingSeed) -> TrainingSeed:
    return TrainingSeed(
        item.label,
        item.clean_run_dir.expanduser().resolve(),
        item.clean_snapshot.expanduser().resolve(),
        item.flow_run_dir.expanduser().resolve(),
    )


def _flow_snapshot(item: TrainingSeed, step: int) -> Path:
    return item.flow_run_dir / "snapshots" / f"{step}_snapshot.pkl"


def _completed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text()).get("status") == "ok"
    except (OSError, json.JSONDecodeError):
        return False


def _result_path(output_dir: Path, split: str, job: EvalJob) -> Path:
    filename = (
        "clean.json"
        if job.kind == "clean"
        else f"flow_step{job.step}.json"
    )
    return output_dir / split / job.training_seed.label / filename


def build_eval_command(
    job: EvalJob,
    *,
    output: Path,
    work_dir: Path,
    gpu_id: int,
    episodes: int,
    seed_start: int,
    flow_readout: str,
    num_flow_steps: int | None,
    policy_value_beta: float,
    return_sample_aggregation: str = "config",
    num_action_flow_samples: int | None = None,
    return_sample_truncate_top: int | None = None,
) -> list[str]:
    candidate = job.kind == "flow"
    if candidate and job.step is None:
        raise ValueError("flow evaluation requires a checkpoint step")
    if not candidate and job.step is not None:
        raise ValueError("clean evaluation does not take a flow step")
    run_dir = (
        job.training_seed.flow_run_dir
        if candidate
        else job.training_seed.clean_run_dir
    )
    snapshot = (
        _flow_snapshot(job.training_seed, int(job.step))
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
        f"{policy_value_beta:g}" if candidate else "bc",
        "--flow-readout",
        flow_readout if candidate else "auto",
    ]
    if candidate and num_flow_steps is not None:
        command.extend(["--num-flow-steps", str(num_flow_steps)])
    if candidate and return_sample_aggregation != "config":
        command.extend(
            [
                "--return-sample-aggregation",
                return_sample_aggregation,
            ]
        )
    if candidate and num_action_flow_samples is not None:
        command.extend(
            [
                "--num-action-flow-samples",
                str(num_action_flow_samples),
            ]
        )
    if candidate and return_sample_truncate_top is not None:
        command.extend(
            [
                "--return-sample-truncate-top",
                str(return_sample_truncate_top),
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
                        flow_readout=args.flow_readout,
                        num_flow_steps=args.num_flow_steps,
                        policy_value_beta=args.policy_value_beta,
                        return_sample_aggregation=(
                            args.return_sample_aggregation
                        ),
                        num_action_flow_samples=(
                            args.num_action_flow_samples
                        ),
                        return_sample_truncate_top=(
                            args.return_sample_truncate_top
                        ),
                    )
                    _run_logged(command, output.with_suffix(".log"))
                if not _completed(output):
                    raise RuntimeError(f"evaluation did not complete: {output}")
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
            f"{len(failures)} checkpoint worker(s) failed"
        ) from failures[0]


def _success(path: Path) -> float:
    payload = json.loads(path.read_text())
    if payload.get("status") != "ok":
        raise ValueError(f"incomplete evaluation: {path}")
    return float(payload["episode_success"])


def select_top_steps(
    step_success: dict[int, float],
    *,
    top_k: int,
) -> list[int]:
    if not step_success or not 1 <= top_k <= len(step_success):
        raise ValueError("invalid step-success table or top-k")
    return sorted(
        step_success,
        key=lambda step: (-float(step_success[step]), int(step)),
    )[:top_k]


def select_winner(step_success: dict[int, float]) -> int:
    return select_top_steps(step_success, top_k=1)[0]


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
    if args.num_flow_steps is not None and args.flow_readout != "integrated":
        raise ValueError("num-flow-steps requires integrated readout")
    if args.return_sample_aggregation == "truncated_mean":
        if (
            args.num_action_flow_samples is None
            or args.return_sample_truncate_top is None
            or args.return_sample_truncate_top < 1
            or args.return_sample_truncate_top
            >= args.num_action_flow_samples
        ):
            raise ValueError(
                "truncated_mean requires an explicit action sample count "
                "and truncation in [1, samples)."
            )
    seed_ranges = [
        set(
            range(
                start,
                start + episodes,
            )
        )
        for start, episodes in (
            (args.screen_seed_start, args.screen_episodes),
            (args.validation_seed_start, args.validation_episodes),
            (args.confirmation_seed_start, args.confirmation_episodes),
        )
    ]
    if any(
        left & right
        for index, left in enumerate(seed_ranges)
        for right in seed_ranges[index + 1 :]
    ):
        raise ValueError("screen/validation/confirmation seeds must be disjoint")

    for item in args.training_seed:
        if not (item.clean_run_dir / ".hydra" / "config.yaml").is_file():
            raise FileNotFoundError(item.clean_run_dir)
        if not item.clean_snapshot.is_file():
            raise FileNotFoundError(item.clean_snapshot)
        if not (item.flow_run_dir / ".hydra" / "config.yaml").is_file():
            raise FileNotFoundError(item.flow_run_dir)
        for step in args.checkpoint_step:
            snapshot = _flow_snapshot(item, step)
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
        "flow_readout": args.flow_readout,
        "num_flow_steps": args.num_flow_steps,
        "policy_value_beta": args.policy_value_beta,
        "return_sample_aggregation": args.return_sample_aggregation,
        "num_action_flow_samples": args.num_action_flow_samples,
        "return_sample_truncate_top": (
            args.return_sample_truncate_top
        ),
        "screen_seed_start": args.screen_seed_start,
        "validation_seed_start": args.validation_seed_start,
        "confirmation_seed_start": args.confirmation_seed_start,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    started = time.time()

    screen_jobs = [
        EvalJob(item, "flow", step)
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
                    EvalJob(item, "flow", step),
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
        EvalJob(item, "flow", step)
        for item in args.training_seed
        for step in selected_for_validation[item.label]
    ]
    _run_jobs(
        args,
        split="validation",
        jobs=validation_jobs,
        episodes=args.validation_episodes,
        seed_start=args.validation_seed_start,
    )
    validation_rows = {}
    selected_steps = {}
    for item in args.training_seed:
        rows = {
            step: _success(
                _result_path(
                    args.output_dir,
                    "validation",
                    EvalJob(item, "flow", step),
                )
            )
            for step in selected_for_validation[item.label]
        }
        validation_rows[item.label] = rows
        selected_steps[item.label] = select_winner(rows)

    selection = {
        "screen": screen_rows,
        "selected_for_validation": selected_for_validation,
        "validation": validation_rows,
        "selected_steps": selected_steps,
        "tie_break": "earlier_checkpoint",
    }
    selection_path = args.output_dir / "selection.json"
    selection_path.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n"
    )

    confirmation_jobs = [
        job
        for item in args.training_seed
        for job in (
            EvalJob(item, "clean", None),
            EvalJob(item, "flow", selected_steps[item.label]),
        )
    ]
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
                    EvalJob(item, "clean", None),
                ),
                _result_path(
                    args.output_dir,
                    "confirmation",
                    EvalJob(
                        item,
                        "flow",
                        selected_steps[item.label],
                    ),
                ),
            )
            for item in args.training_seed
        ],
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        min_mean_delta=0.0,
        min_ci_lower=0.0,
    )
    summary["selection"] = str(selection_path)
    summary["selected_steps"] = selected_steps
    summary["candidate_readout"] = args.flow_readout
    summary["num_flow_steps"] = args.num_flow_steps
    summary["policy_value_beta"] = args.policy_value_beta
    summary["return_sample_aggregation"] = (
        args.return_sample_aggregation
    )
    summary["num_action_flow_samples"] = (
        args.num_action_flow_samples
    )
    summary["return_sample_truncate_top"] = (
        args.return_sample_truncate_top
    )
    summary["elapsed_seconds"] = time.time() - started
    summary["manifest"] = str(manifest_path)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    manifest["status"] = "ok"
    manifest["selection"] = str(selection_path)
    manifest["summary"] = str(summary_path)
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
