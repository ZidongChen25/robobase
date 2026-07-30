#!/usr/bin/env python3
"""Compare distilled and integrated CQN-Flow readouts across training seeds.

The selection split chooses one globally shared integration-step count. Only a
candidate with positive paired direction against the distilled readout is
promoted. The selected readout is then rerun on a disjoint confirmation split
and judged with a crossed training-seed/environment-seed bootstrap.
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
class Checkpoint:
    label: str
    run_dir: Path
    snapshot: Path


@dataclass(frozen=True)
class Readout:
    label: str
    kind: str
    steps: int | None


def _checkpoint(value: str) -> Checkpoint:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "checkpoint must be LABEL=RUN_DIR,SNAPSHOT"
        )
    label, raw_paths = value.split("=", 1)
    paths = raw_paths.split(",")
    if not label or len(paths) != 2 or not all(paths):
        raise argparse.ArgumentTypeError(
            "checkpoint must be LABEL=RUN_DIR,SNAPSHOT"
        )
    return Checkpoint(label, Path(paths[0]), Path(paths[1]))


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        required=True,
        action="append",
        type=_checkpoint,
    )
    parser.add_argument("--gpu-id", required=True, action="append", type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument(
        "--steps",
        nargs="+",
        type=_positive_int,
        default=[2, 8],
    )
    parser.add_argument(
        "--policy-value-beta",
        type=_nonnegative,
        default=1.0,
    )
    parser.add_argument("--selection-episodes", type=int, default=50)
    parser.add_argument("--selection-seed-start", type=int, default=96_000)
    parser.add_argument("--confirmation-episodes", type=int, default=200)
    parser.add_argument(
        "--confirmation-seed-start",
        type=int,
        default=97_000,
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=97_200)
    return parser.parse_args()


def _completed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text()).get("status") == "ok"
    except (OSError, json.JSONDecodeError):
        return False


def _resolved(checkpoint: Checkpoint) -> Checkpoint:
    return Checkpoint(
        checkpoint.label,
        checkpoint.run_dir.expanduser().resolve(),
        checkpoint.snapshot.expanduser().resolve(),
    )


def build_eval_command(
    checkpoint: Checkpoint,
    readout: Readout,
    *,
    output: Path,
    work_dir: Path,
    gpu_id: int,
    episodes: int,
    seed_start: int,
    policy_value_beta: float,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("eval_cqn_flow_policy_value.py")),
        "--run-dir",
        str(checkpoint.run_dir),
        "--snapshot",
        str(checkpoint.snapshot),
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
        f"{policy_value_beta:g}",
        "--flow-readout",
        readout.kind,
    ]
    if readout.steps is not None:
        command.extend(["--num-flow-steps", str(readout.steps)])
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


def _result_path(
    output_dir: Path,
    split: str,
    checkpoint: Checkpoint,
    readout: Readout,
) -> Path:
    return output_dir / split / checkpoint.label / f"{readout.label}.json"


def _run_split(
    args: argparse.Namespace,
    *,
    split: str,
    readouts: list[Readout],
    episodes: int,
    seed_start: int,
) -> None:
    jobs: queue.Queue[tuple[Checkpoint, Readout]] = queue.Queue()
    for checkpoint in args.checkpoint:
        for readout in readouts:
            jobs.put((checkpoint, readout))
    failures: list[BaseException] = []
    failure_lock = threading.Lock()

    def worker(gpu_id: int) -> None:
        while True:
            try:
                checkpoint, readout = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                output = _result_path(
                    args.output_dir,
                    split,
                    checkpoint,
                    readout,
                )
                if not _completed(output):
                    command = build_eval_command(
                        checkpoint,
                        readout,
                        output=output,
                        work_dir=(
                            args.work_root
                            / split
                            / checkpoint.label
                            / readout.label
                        ),
                        gpu_id=gpu_id,
                        episodes=episodes,
                        seed_start=seed_start,
                        policy_value_beta=args.policy_value_beta,
                    )
                    _run_logged(command, output.with_suffix(".log"))
                if not _completed(output):
                    raise RuntimeError(f"evaluation did not complete: {output}")
            except BaseException as exc:
                with failure_lock:
                    failures.append(exc)
            finally:
                jobs.task_done()

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
            f"{len(failures)} readout worker(s) failed"
        ) from failures[0]


def _paired_summary(
    args: argparse.Namespace,
    *,
    split: str,
    candidate: Readout,
    bootstrap_seed: int,
) -> dict:
    distill = Readout("distill", "distill", None)
    return summarize(
        [
            (
                checkpoint.label,
                _result_path(
                    args.output_dir,
                    split,
                    checkpoint,
                    distill,
                ),
                _result_path(
                    args.output_dir,
                    split,
                    checkpoint,
                    candidate,
                ),
            )
            for checkpoint in args.checkpoint
        ],
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        min_mean_delta=0.0,
        min_ci_lower=0.0,
    )


def selection_checks(summary: dict) -> dict[str, bool]:
    return {
        "mean_delta_strictly_positive": (
            float(summary["mean_paired_delta"]) > 0.0
        ),
        "aggregate_wins_above_losses": (
            int(summary["aggregate_paired_wins"])
            > int(summary["aggregate_paired_losses"])
        ),
        "positive_training_seed_majority": bool(
            summary["gate_checks"]["positive_training_seed_majority"]
        ),
    }


def run_gate(args: argparse.Namespace) -> dict:
    if len(args.checkpoint) < 2:
        raise ValueError("at least two training checkpoints are required")
    args.checkpoint = [_resolved(item) for item in args.checkpoint]
    if len({item.label for item in args.checkpoint}) != len(args.checkpoint):
        raise ValueError("checkpoint labels must be unique")
    if not args.gpu_id or len(set(args.gpu_id)) != len(args.gpu_id):
        raise ValueError("gpu-id workers must be non-empty and unique")
    if len(set(args.steps)) != len(args.steps):
        raise ValueError("steps must be unique")
    if min(
        args.selection_episodes,
        args.confirmation_episodes,
        args.bootstrap_replicates,
    ) < 1:
        raise ValueError("episode and bootstrap counts must be positive")
    for checkpoint in args.checkpoint:
        if not (checkpoint.run_dir / ".hydra" / "config.yaml").is_file():
            raise FileNotFoundError(checkpoint.run_dir)
        if not checkpoint.snapshot.is_file():
            raise FileNotFoundError(checkpoint.snapshot)

    args.output_dir = args.output_dir.expanduser().resolve()
    args.work_root = args.work_root.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    distill = Readout("distill", "distill", None)
    integrated = [
        Readout(f"integrated_steps{steps}", "integrated", steps)
        for steps in args.steps
    ]
    manifest_path = args.output_dir / "manifest.json"
    manifest = {
        "status": "running",
        "checkpoints": [
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(checkpoint).items()
            }
            for checkpoint in args.checkpoint
        ],
        "gpu_ids": list(args.gpu_id),
        "steps": list(args.steps),
        "policy_value_beta": args.policy_value_beta,
        "selection_seed_start": args.selection_seed_start,
        "selection_episodes": args.selection_episodes,
        "confirmation_seed_start": args.confirmation_seed_start,
        "confirmation_episodes": args.confirmation_episodes,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    started = time.time()
    _run_split(
        args,
        split="selection",
        readouts=[distill, *integrated],
        episodes=args.selection_episodes,
        seed_start=args.selection_seed_start,
    )
    selection_rows = []
    for index, readout in enumerate(integrated):
        summary = _paired_summary(
            args,
            split="selection",
            candidate=readout,
            bootstrap_seed=args.bootstrap_seed + index,
        )
        checks = selection_checks(summary)
        selection_rows.append(
            {
                "readout": asdict(readout),
                "paired_summary": summary,
                "promotion_checks": checks,
                "promotion_gate": (
                    "pass" if all(checks.values()) else "fail"
                ),
            }
        )
    selected = max(
        selection_rows,
        key=lambda row: (
            row["paired_summary"]["mean_paired_delta"],
            -int(row["readout"]["steps"]),
        ),
    )
    selection_passed = selected["promotion_gate"] == "pass"
    payload = {
        "status": "ok",
        "selection_rows": selection_rows,
        "selected_readout": selected["readout"],
        "selection_gate": "pass" if selection_passed else "fail",
        "confirmation_summary": None,
        "confirmation_gate": "not_run",
        "gate": "fail",
    }

    if selection_passed:
        selected_readout = Readout(**selected["readout"])
        _run_split(
            args,
            split="confirmation",
            readouts=[distill, selected_readout],
            episodes=args.confirmation_episodes,
            seed_start=args.confirmation_seed_start,
        )
        confirmation = _paired_summary(
            args,
            split="confirmation",
            candidate=selected_readout,
            bootstrap_seed=args.bootstrap_seed + 10_000,
        )
        payload["confirmation_summary"] = confirmation
        payload["confirmation_gate"] = confirmation["gate"]
        payload["gate"] = confirmation["gate"]

    payload["elapsed_seconds"] = time.time() - started
    payload["manifest"] = str(manifest_path)
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
