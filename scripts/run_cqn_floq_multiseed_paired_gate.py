#!/usr/bin/env python3
"""Run and summarize a frozen multi-seed CQN-AS/FLOQ comparison.

Pairs are drawn from a shared queue by one worker per GPU.  Each worker runs a
clean CQN-AS evaluation and its matched FLOQ evaluation sequentially, while
different training-seed pairs run concurrently.  Completed JSON artifacts are
reused, so a controller can safely resume an interrupted stage.
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
class Pair:
    label: str
    clean_run_dir: Path
    clean_snapshot: Path
    flow_run_dir: Path
    flow_snapshot: Path


@dataclass(frozen=True)
class EvalJob:
    pair: Pair
    candidate: bool


def _pair(value: str) -> Pair:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "pair must be LABEL=CLEAN_RUN,CLEAN_SNAPSHOT,FLOW_RUN,FLOW_SNAPSHOT"
        )
    label, raw_paths = value.split("=", 1)
    paths = raw_paths.split(",")
    if not label or len(paths) != 4 or not all(paths):
        raise argparse.ArgumentTypeError(
            "pair must be LABEL=CLEAN_RUN,CLEAN_SNAPSHOT,FLOW_RUN,FLOW_SNAPSHOT"
        )
    return Pair(label, *(Path(path) for path in paths))


def _finite_nonnegative(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise argparse.ArgumentTypeError(
            "policy-value-beta must be finite and non-negative"
        )
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True, action="append", type=_pair)
    parser.add_argument(
        "--gpu-id",
        required=True,
        action="append",
        type=int,
        help="Repeat once per GPU worker.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--num-eval-episodes", type=int, default=200)
    parser.add_argument("--eval-seed-start", type=int, default=92_000)
    parser.add_argument(
        "--policy-value-beta",
        type=_finite_nonnegative,
        default=1.0,
    )
    parser.add_argument(
        "--flow-readout",
        choices=("auto", "distill", "integrated"),
        default="distill",
        help=(
            "Candidate value readout. 'auto' also permits a non-Flow CQN-AS "
            "checkpoint, which is useful for matched direct-C51 controls."
        ),
    )
    parser.add_argument("--num-flow-steps", type=int)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=92_200)
    parser.add_argument("--min-mean-delta", type=float, default=0.0)
    parser.add_argument("--min-ci-lower", type=float, default=0.0)
    return parser.parse_args()


def _completed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text()).get("status") == "ok"
    except (OSError, json.JSONDecodeError):
        return False


def build_eval_command(
    pair: Pair,
    *,
    candidate: bool,
    output: Path,
    work_dir: Path,
    gpu_id: int,
    episodes: int,
    seed_start: int,
    policy_value_beta: float,
    flow_readout: str,
    num_flow_steps: int | None,
) -> list[str]:
    run_dir = pair.flow_run_dir if candidate else pair.clean_run_dir
    snapshot = pair.flow_snapshot if candidate else pair.clean_snapshot
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


def _run_eval(
    args: argparse.Namespace,
    pair: Pair,
    *,
    candidate: bool,
    gpu_id: int,
) -> Path:
    kind = "flow" if candidate else "clean"
    pair_dir = args.output_dir / pair.label
    output = pair_dir / f"{kind}.json"
    if _completed(output):
        return output
    command = build_eval_command(
        pair,
        candidate=candidate,
        output=output,
        work_dir=args.work_root / pair.label / kind,
        gpu_id=gpu_id,
        episodes=args.num_eval_episodes,
        seed_start=args.eval_seed_start,
        policy_value_beta=args.policy_value_beta,
        flow_readout=args.flow_readout,
        num_flow_steps=args.num_flow_steps,
    )
    _run_logged(command, pair_dir / f"{kind}.log")
    if not _completed(output):
        raise RuntimeError(f"evaluation did not complete: {output}")
    return output


def _resolved_pair(pair: Pair) -> Pair:
    return Pair(
        label=pair.label,
        clean_run_dir=pair.clean_run_dir.expanduser().resolve(),
        clean_snapshot=pair.clean_snapshot.expanduser().resolve(),
        flow_run_dir=pair.flow_run_dir.expanduser().resolve(),
        flow_snapshot=pair.flow_snapshot.expanduser().resolve(),
    )


def build_eval_jobs(pairs: list[Pair]) -> list[EvalJob]:
    """Expose clean/candidate evaluations as independent GPU work items."""

    return [
        EvalJob(pair=pair, candidate=candidate)
        for pair in pairs
        for candidate in (False, True)
    ]


def run_gate(args: argparse.Namespace) -> dict:
    if len(args.pair) < 2:
        raise ValueError("at least two training-seed pairs are required")
    args.pair = [_resolved_pair(pair) for pair in args.pair]
    if len({pair.label for pair in args.pair}) != len(args.pair):
        raise ValueError("pair labels must be unique")
    if not args.gpu_id or len(set(args.gpu_id)) != len(args.gpu_id):
        raise ValueError("gpu-id workers must be non-empty and unique")
    if args.num_eval_episodes < 1 or args.bootstrap_replicates < 1:
        raise ValueError("episode and bootstrap counts must be positive")
    if args.num_flow_steps is not None and args.num_flow_steps < 1:
        raise ValueError("num-flow-steps must be positive")
    for pair in args.pair:
        for path in (pair.clean_snapshot, pair.flow_snapshot):
            if not path.is_file():
                raise FileNotFoundError(path)

    args.output_dir = args.output_dir.expanduser().resolve()
    args.work_root = args.work_root.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    manifest = {
        "status": "running",
        "pairs": [
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(pair).items()
            }
            for pair in args.pair
        ],
        "gpu_ids": list(args.gpu_id),
        "num_eval_episodes": args.num_eval_episodes,
        "eval_seed_start": args.eval_seed_start,
        "eval_seed_end": (
            args.eval_seed_start + args.num_eval_episodes - 1
        ),
        "policy_value_beta": args.policy_value_beta,
        "flow_readout": args.flow_readout,
        "num_flow_steps": args.num_flow_steps,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    work_queue: queue.Queue[EvalJob] = queue.Queue()
    for job in build_eval_jobs(args.pair):
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
                _run_eval(
                    args,
                    job.pair,
                    candidate=job.candidate,
                    gpu_id=gpu_id,
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

    summary = summarize(
        [
            (
                pair.label,
                args.output_dir / pair.label / "clean.json",
                args.output_dir / pair.label / "flow.json",
            )
            for pair in args.pair
        ],
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        min_mean_delta=args.min_mean_delta,
        min_ci_lower=args.min_ci_lower,
    )
    summary["elapsed_seconds"] = time.time() - started
    summary["manifest"] = str(
        (args.output_dir / "manifest.json").resolve()
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    manifest["status"] = "ok"
    manifest["summary"] = str(
        (args.output_dir / "summary.json").resolve()
    )
    (args.output_dir / "manifest.json").write_text(
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
