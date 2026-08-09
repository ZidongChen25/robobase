"""Sweep-evaluate every saved snapshot of a run in ONE process.

Builds the eval Workspace once (env creation + XLA compile paid once),
then loops over the run's numbered snapshots: load params -> eval ->
append a row.  Cuts the ~2min per-checkpoint startup of
eval_cqn_as_bigym_checkpoint.py to a single startup per run, and exposes
``--num-eval-envs`` (the workspace's existing vector-eval path) for
batched inference.

Output: <run_dir>/<csv-name> rows (env_steps, episode_success, ...) and a
JSON per snapshot under <run_dir>/sweep_evals/.
"""

import argparse
import csv
import json
import re
import shutil
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--dump-action-trace", default=None)
    parser.add_argument("--post-ensemble-fixed-leaf", type=int, default=None)
    parser.add_argument(
        "--log-ensemble-consensus",
        action="store_true",
    )
    parser.add_argument(
        "--post-ensemble-keep-levels",
        type=int,
        default=None,
        help=(
            "Keep the ensembled action's true C2F prefix down to this many "
            "levels; randomize the remainder POST-ensemble."
        ),
    )
    parser.add_argument(
        "--temporal-ensemble-gain",
        type=float,
        default=None,
        help=(
            "Override method.temporal_ensemble_gain at eval. Higher = newest "
            "plan dominates (0.01 default -> newest weight 6.7%%; 5.0 -> 99.3%%)."
        ),
    )
    parser.add_argument(
        "--level-override-mode",
        choices=("random", "middle"),
        default="random",
    )
    parser.add_argument(
        "--random-levels-from",
        type=int,
        default=None,
        help=(
            "Diagnostic: replace the critic bin choice with a uniform "
            "draw at this C2F level and below, to measure what the "
            "critic per-level ordering is worth on task success."
        ),
    )
    parser.add_argument("--num-eval-episodes", type=int, default=25)
    parser.add_argument("--eval-seed-start", type=int, default=400)
    parser.add_argument("--num-eval-envs", type=int, default=1)
    parser.add_argument(
        "--obs-delay",
        type=int,
        default=None,
        help=(
            "Delayed-policy conditioning h: act on the observation from h "
            "environment steps ago. Defaults to the value the run trained with."
        ),
    )
    parser.add_argument("--csv-name", default="sweep_eval.csv")
    parser.add_argument(
        "--skip-steps",
        default="",
        help="Comma-separated snapshot steps to skip.",
    )
    parser.add_argument(
        "--only-steps",
        default="",
        help="Optional comma-separated allowlist of snapshot steps.",
    )
    parser.add_argument(
        "--replan-interval",
        type=int,
        default=None,
        help="Override method.temporal_ensemble_replan_interval at eval.",
    )
    parser.add_argument(
        "--finalize-artifacts",
        action="store_true",
        help=(
            "After a complete sweep, retain only the validation-best and final "
            "params-only checkpoints and remove run-local resume/replay state."
        ),
    )
    parser.add_argument(
        "--selection-csv",
        default=None,
        help="Validation CSV used to select best; defaults to --csv-name.",
    )
    return parser.parse_args()


def parse_step_set(value) -> set[int]:
    return {int(v) for v in str(value).split(",") if v.strip()}


def configure_process(gpu_id):
    import os

    if gpu_id is not None and gpu_id >= 0:
        gpu = str(gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        os.environ["JAX_CUDA_VISIBLE_DEVICES"] = gpu
        os.environ["MUJOCO_EGL_DEVICE_ID"] = gpu
    os.environ.setdefault("MUJOCO_GL", "egl")


def discover_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """Prefer params-only checkpoints, with legacy snapshot fallback."""

    candidates = []
    for path in (run_dir / "eval_checkpoints").glob("*_checkpoint.pkl"):
        match = re.fullmatch(r"(\d+)_checkpoint\.pkl", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if candidates:
        return sorted(candidates)
    for path in (run_dir / "snapshots").glob("*_snapshot.pkl"):
        match = re.fullmatch(r"(\d+)_snapshot\.pkl", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    return sorted(candidates)


def finalize_run_artifacts(
    run_dir: Path,
    selection_csv: Path,
    checkpoints: list[tuple[int, Path]],
) -> dict:
    """Keep validation-best/final model parameters and discard resume state."""

    if not selection_csv.is_file():
        raise ValueError(f"Selection CSV does not exist: {selection_csv}")
    rows = list(csv.DictReader(selection_csv.open()))
    if not rows:
        raise ValueError(f"Selection CSV is empty: {selection_csv}")
    selection_seed_starts = {
        int(float(row["eval_seed_start"]))
        for row in rows
        if row.get("eval_seed_start") not in (None, "")
    }
    if not selection_seed_starts or max(selection_seed_starts) >= 800:
        raise ValueError(
            "Artifact selection must use validation seeds (<800), never sealed held-out."
        )

    checkpoint_by_step = dict(checkpoints)
    measured = {}
    for row in rows:
        if not row.get("env_steps") or row.get("episode_success") in (None, ""):
            continue
        measured[int(float(row["env_steps"]))] = float(row["episode_success"])
    missing = sorted(set(checkpoint_by_step) - set(measured))
    if missing:
        raise ValueError(
            "Refusing artifact finalization because validation did not cover "
            f"all checkpoints; missing steps: {missing}"
        )
    best_step = min(
        checkpoint_by_step,
        key=lambda step: (-measured[step], step),
    )
    final_step = max(checkpoint_by_step)
    retained_steps = sorted({best_step, final_step})

    if any(
        path.parent.name != "eval_checkpoints"
        for path in checkpoint_by_step.values()
    ):
        raise ValueError(
            "Refusing to finalize legacy full snapshots. Run a params-only "
            "checkpoint migration/evaluation first."
        )

    removed_checkpoints = 0
    for step, path in checkpoints:
        if step not in retained_steps:
            path.unlink(missing_ok=True)
            removed_checkpoints += 1

    removed_resume_files = 0
    snapshot_dir = run_dir / "snapshots"
    for path in snapshot_dir.glob("*_snapshot.pkl"):
        path.unlink(missing_ok=True)
        removed_resume_files += 1

    removed_replay_dirs = []
    for name in ("replay", "demo_replay"):
        path = run_dir / name
        if path.is_dir():
            shutil.rmtree(path)
            removed_replay_dirs.append(name)

    record = {
        "selection_csv": str(selection_csv),
        "selection_eval_seed_starts": sorted(selection_seed_starts),
        "best_step": best_step,
        "best_success": measured[best_step],
        "final_step": final_step,
        "retained_steps": retained_steps,
        "removed_eval_checkpoints": removed_checkpoints,
        "removed_resume_files": removed_resume_files,
        "removed_replay_dirs": removed_replay_dirs,
    }
    (run_dir / "artifact_finalization.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    return record


def main() -> int:
    args = parse_args()
    configure_process(args.gpu_id)

    from omegaconf import OmegaConf

    from robobase.workspace import Workspace

    run_dir = Path(args.run_dir).resolve()
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    OmegaConf.set_struct(cfg, False)
    cfg.create_train_env = False
    cfg.num_train_envs = 0
    cfg.num_train_frames = 0
    cfg.num_eval_envs = int(args.num_eval_envs)
    cfg.num_eval_episodes = int(args.num_eval_episodes)
    cfg.env.eval_seed_start = int(args.eval_seed_start)
    if args.obs_delay is not None:
        cfg.obs_delay = int(args.obs_delay)
    else:
        cfg.obs_delay = int(cfg.get("obs_delay", 0) or 0)
    cfg.demo_batch_size = None
    cfg.replay.demo_only_updates = False
    cfg.use_self_imitation = False
    cfg.log_train_video = False
    cfg.log_eval_video = False
    cfg.save_snapshot = False
    cfg.save_csv = False
    cfg.gpu_id = None
    cfg.wandb.use = False
    cfg.tb.use = False
    cfg.replay.num_workers = 0
    cfg.backend.replay_prefetch_size = 0
    cfg.backend.replay_device_prefetch = False
    cfg.backend.fused_update_steps = 1
    cfg.backend.update_block_every_steps = 1
    if args.post_ensemble_keep_levels is not None:
        cfg.method.post_ensemble_random_keep_levels = int(args.post_ensemble_keep_levels)
    if args.post_ensemble_fixed_leaf is not None:
        cfg.method.post_ensemble_fixed_leaf = int(args.post_ensemble_fixed_leaf)
    if args.temporal_ensemble_gain is not None:
        cfg.method.temporal_ensemble_gain = float(args.temporal_ensemble_gain)
    if args.random_levels_from is not None:
        cfg.method.random_levels_from = int(args.random_levels_from)
        cfg.method.level_override_mode = str(args.level_override_mode)
    if args.replan_interval is not None:
        cfg.method.temporal_ensemble = True
        cfg.method.temporal_ensemble_replan_interval = int(
            args.replan_interval
        )
    OmegaConf.resolve(cfg)

    skip = parse_step_set(args.skip_steps)
    only = parse_step_set(args.only_steps)
    all_checkpoints = discover_checkpoints(run_dir)
    snapshots = [
        (step, path)
        for step, path in all_checkpoints
        if step not in skip and (not only or step in only)
    ]
    if not snapshots:
        print("[sweep] no snapshots found")
        return 1

    out_dir = run_dir / "sweep_evals"
    out_dir.mkdir(exist_ok=True)
    csv_path = run_dir / args.csv_name
    done = set()
    if csv_path.exists():
        for row in csv.DictReader(open(csv_path)):
            if row.get("env_steps"):
                done.add(int(float(row["env_steps"])))

    work_dir = out_dir / "workspace"
    work_dir.mkdir(exist_ok=True)
    workspace = Workspace(cfg, work_dir=str(work_dir))
    try:
        for step, snapshot in snapshots:
            if step in done:
                print(f"[sweep] skip {step} (already in csv)", flush=True)
                continue
            start = time.time()
            workspace.load_snapshot(snapshot, load_replay_buffer=False)
            if args.log_ensemble_consensus:
                workspace.agent.log_ensemble_consensus = True
            if args.dump_action_trace:
                workspace.agent.log_executed_actions = True
                workspace.agent._action_trace = []
            metrics = workspace.eval()
            if args.dump_action_trace:
                import numpy as _np
                tr = workspace.agent._action_trace
                _np.savez(args.dump_action_trace,
                          pre=_np.array([t[0] for t in tr]),
                          post=_np.array([t[1] for t in tr]))
                print("[trace] %d steps -> %s" % (len(tr), args.dump_action_trace), flush=True)
            if args.log_ensemble_consensus:
                st = getattr(workspace.agent, "_ensemble_consensus_stats", None)
                if st and st["votes"] > 0:
                    print(
                        "[consensus] exact %.2f%%  adjacent(<=1格) %.2f%%  "
                        "全体一致的(维,步)占比 %.2f%%  votes %d"
                        % (
                            100 * st["exact"] / st["votes"],
                            100 * st["adjacent"] / st["votes"],
                            100 * st["full"] / st["entries"],
                            int(st["votes"]),
                        ),
                        flush=True,
                    )
                    workspace.agent._ensemble_consensus_stats = None
            numeric = {
                k: float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float))
            }
            row = {
                "env_steps": step,
                "iteration": step,
                "episode_success": numeric.get("episode_success"),
                "episode_reward": numeric.get("episode_reward", ""),
                "eval_episodes": int(args.num_eval_episodes),
                "eval_seed_start": int(args.eval_seed_start),
                "elapsed_sec": round(time.time() - start, 1),
            }
            exists = csv_path.exists()
            with open(csv_path, "a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
            json.dump(
                {"snapshot": str(snapshot), **row},
                open(out_dir / f"eval_{step}.json", "w"),
                indent=2,
            )
            print(
                f"[sweep] step {step}: success "
                f"{100 * (numeric.get('episode_success') or 0):.1f}% "
                f"({row['elapsed_sec']}s)",
                flush=True,
            )
    finally:
        workspace.shutdown()
    if args.finalize_artifacts:
        if args.selection_csv is None:
            selection_csv = csv_path
        else:
            selection_csv = Path(args.selection_csv)
            if not selection_csv.is_absolute():
                selection_csv = run_dir / selection_csv
            selection_csv = selection_csv.resolve()
        record = finalize_run_artifacts(
            run_dir,
            selection_csv,
            all_checkpoints,
        )
        print(f"[sweep] artifacts finalized: {record}", flush=True)
    print("[sweep] run complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
