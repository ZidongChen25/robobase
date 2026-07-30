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
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--num-eval-episodes", type=int, default=25)
    parser.add_argument("--eval-seed-start", type=int, default=400)
    parser.add_argument("--num-eval-envs", type=int, default=1)
    parser.add_argument("--csv-name", default="sweep_eval.csv")
    parser.add_argument(
        "--skip-steps",
        default="",
        help="Comma-separated snapshot steps to skip.",
    )
    parser.add_argument(
        "--replan-interval",
        type=int,
        default=None,
        help="Override method.temporal_ensemble_replan_interval at eval.",
    )
    return parser.parse_args()


def configure_process(gpu_id):
    import os

    if gpu_id is not None and gpu_id >= 0:
        gpu = str(gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        os.environ["JAX_CUDA_VISIBLE_DEVICES"] = gpu
        os.environ["MUJOCO_EGL_DEVICE_ID"] = gpu
    os.environ.setdefault("MUJOCO_GL", "egl")


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
    cfg.demo_batch_size = None
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
    if args.replan_interval is not None:
        cfg.method.temporal_ensemble = True
        cfg.method.temporal_ensemble_replan_interval = int(
            args.replan_interval
        )
    OmegaConf.resolve(cfg)

    skip = {
        int(v) for v in str(args.skip_steps).split(",") if v.strip()
    }
    snapshots = []
    for path in (run_dir / "snapshots").glob("*_snapshot.pkl"):
        match = re.match(r"(\d+)_snapshot\.pkl", path.name)
        if match and int(match.group(1)) not in skip:
            snapshots.append((int(match.group(1)), path))
    snapshots.sort()
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
            metrics = workspace.eval()
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
    print("[sweep] run complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
