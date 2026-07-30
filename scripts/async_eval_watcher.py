"""Asynchronous checkpoint evaluator (cqn-flow.md 42).

Watches a training run's ``snapshots/`` directory from a *different* GPU
and evaluates each new snapshot with a full 50-episode protocol, without
ever pausing the training process.  Results are appended to an eval CSV
that is column-compatible with the in-loop evaluator (``env_steps`` +
``episode_success``), so downstream tooling
(resolve_cqn_validation_best_snapshot.py) keeps working unchanged.

Backlog policy: ``--latest-only`` (default) always evaluates the newest
pending snapshot and marks older pending ones as skipped, so a single
eval GPU can serve a faster trainer without falling behind.

Typical launch (training itself runs with in-loop eval disabled):
  train_fast.py ... eval_every_steps=1000000 save_snapshot=true \
      snapshot_every_n=5000
  python scripts/async_eval_watcher.py --run-dir <dir> --gpu-id 4 \
      --num-episodes 50
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--gpu-id",
        type=str,
        required=True,
        help="CUDA device passed to the eval script: numeric id or GPU-<uuid>.",
    )
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--eval-seed-start", type=int, default=400)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--csv-name", default="eval.csv")
    parser.add_argument(
        "--latest-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When backlogged, evaluate only the newest pending snapshot.",
    )
    parser.add_argument("--max-evals", type=int, default=None)
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=3600.0,
        help="Exit after this many seconds with no new snapshot.",
    )
    parser.add_argument(
        "--delete-after-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete each snapshot once evaluated, except milestones.",
    )
    parser.add_argument(
        "--keep-every",
        type=int,
        default=20000,
        help="Milestone period: snapshots at multiples of this step count "
        "(and the final one) are never deleted.",
    )
    parser.add_argument(
        "--eval-script",
        default="eval_cqn_as_bigym_checkpoint.py",
        help="Checkpoint evaluator under scripts/ to run per snapshot; it must "
        "accept --run-dir/--snapshot/--gpu-id/--num-eval-episodes/"
        "--eval-seed-start/--output and emit JSON with success_percent.",
    )
    return parser.parse_args()


def snapshot_steps(run_dir: Path) -> list[tuple[int, Path]]:
    result = []
    for path in (run_dir / "snapshots").glob("*_snapshot.pkl"):
        match = re.match(r"(\d+)_snapshot\.pkl", path.name)
        if match:
            result.append((int(match.group(1)), path))
    return sorted(result)


def evaluated_steps(csv_path: Path) -> set[int]:
    if not csv_path.exists():
        return set()
    done = set()
    for row in csv.DictReader(open(csv_path)):
        value = row.get("env_steps")
        if value not in (None, ""):
            done.add(int(float(value)))
    return done


def append_row(csv_path: Path, row: dict):
    exists = csv_path.exists()
    fields = [
        "env_steps",
        "iteration",
        "episode_success",
        "episode_reward",
        "eval_episodes",
        "snapshot",
        "total_time",
    ]
    with open(csv_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def evaluate(args, run_dir: Path, step: int, snapshot: Path, start: float):
    out_json = run_dir / f"async_eval_{step}.json"
    log_path = run_dir / f"async_eval_{step}.log"
    command = [
        sys.executable,
        str(REPO / "scripts" / args.eval_script),
        "--run-dir",
        str(run_dir),
        "--snapshot",
        str(snapshot),
        "--gpu-id",
        str(args.gpu_id),
        "--num-eval-episodes",
        str(args.num_episodes),
        "--eval-seed-start",
        str(args.eval_seed_start),
        "--output",
        str(out_json),
    ]
    with open(log_path, "w") as log:
        completed = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, cwd=REPO
        )
    if completed.returncode != 0:
        print(f"[async-eval] step {step} FAILED, see {log_path}", flush=True)
        return
    payload = json.loads(open(out_json).read())
    success = float(payload["success_percent"]) / 100.0
    append_row(
        run_dir / args.csv_name,
        {
            "env_steps": step,
            "iteration": step,
            "episode_success": success,
            "episode_reward": payload.get("mean_reward", ""),
            "eval_episodes": args.num_episodes,
            "snapshot": snapshot.name,
            "total_time": round(time.time() - start, 1),
        },
    )
    print(
        f"[async-eval] step {step}: success {success * 100:.1f}% "
        f"({args.num_episodes} eps)",
        flush=True,
    )


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    csv_path = run_dir / args.csv_name
    start = time.time()
    last_new_snapshot = time.time()
    evals_done = 0
    known = set()
    while True:
        snaps = snapshot_steps(run_dir)
        done = evaluated_steps(csv_path)
        pending = [
            (step, path)
            for step, path in snaps
            if step not in done and step not in known
        ]
        if pending:
            last_new_snapshot = time.time()
            if args.latest_only and len(pending) > 1:
                skipped = pending[:-1]
                for step, _ in skipped:
                    known.add(step)
                    print(
                        f"[async-eval] skipping stale snapshot {step}",
                        flush=True,
                    )
                pending = pending[-1:]
            step, path = pending[-1]
            known.add(step)
            evaluate(args, run_dir, step, path, start)
            if args.delete_after_eval and step % max(1, args.keep_every):
                newest = max(s for s, _ in snaps)
                if step != newest and path.exists():
                    path.unlink()
                    print(
                        f"[async-eval] deleted non-milestone snapshot {step}",
                        flush=True,
                    )
            evals_done += 1
            if args.max_evals is not None and evals_done >= args.max_evals:
                print("[async-eval] max evals reached, exiting", flush=True)
                return
            continue
        if time.time() - last_new_snapshot > args.idle_timeout:
            print("[async-eval] idle timeout, exiting", flush=True)
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
