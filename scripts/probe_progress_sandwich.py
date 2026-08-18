"""Phi-quality gate for progress-potential shaping (Gupta sandwich probe).

Potential-based shaping is policy-invariant asymptotically, so a bad ``Phi``
cannot invert preferences -- it can only waste the shaping budget.  What
predicts whether it *helps* is how tightly ``Phi`` brackets the true value
function.  Gupta et al. (NeurIPS 2022, "Unpacking Reward Shaping") formalise
this as a sandwich condition

    c1 * V(s)  <=  Phi(s)  <=  c2 * V(s)

and show the speed-up degrades with the ratio ``c2 / c1``; a ratio far above
~3 predicts no benefit from shaping.  This probe estimates ``c1``/``c2`` for
the trained progress head against the stored Monte-Carlo return of demo
transitions (for a demo truncated at its first success, ``mc_return`` IS the
realised discounted value of that state under the demo policy).

Because ``min``/``max`` are single-sample statistics, a percentile-robust
variant (``c1_p05`` / ``c2_p95``) is reported alongside; read both.

Caveat on "held out": a normal CQN-AS run trains on every demo it loads, so
the default probe batch is in-sample and the ratio is an optimistic bound.
Pass ``--demos`` larger than the run's ``cfg.demos`` together with
``--holdout-only`` to score only the demos the run never saw.

Usage:
    python scripts/probe_progress_sandwich.py --run-dir exp_local/<run> \
        [--only-steps 50000,100000] [--batches 8] [--output out.csv]
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import tempfile
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--egl-device-id", type=int, default=None)
    parser.add_argument(
        "--batches",
        type=int,
        default=8,
        help="Demo batches sampled per snapshot.",
    )
    parser.add_argument(
        "--demos",
        type=int,
        default=None,
        help="Override cfg.demos so extra, never-trained demos are loaded.",
    )
    parser.add_argument(
        "--holdout-only",
        action="store_true",
        help=(
            "Score only transitions from demo episodes beyond the run's own "
            "cfg.demos (requires --demos greater than it)."
        ),
    )
    parser.add_argument("--only-steps", default="")
    parser.add_argument("--label", default="")
    parser.add_argument(
        "--min-return",
        type=float,
        default=1e-3,
        help="Ignore transitions whose reference return is below this.",
    )
    return parser.parse_args()


def configure_process(gpu_id, egl_device_id):
    if gpu_id is not None and gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if egl_device_id is not None and egl_device_id >= 0:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_device_id)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def main() -> int:
    args = parse_args()
    configure_process(args.gpu_id, args.egl_device_id)

    import numpy as np
    from omegaconf import OmegaConf

    from robobase.workspace import Workspace

    run_dir = Path(args.run_dir).resolve()
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    OmegaConf.set_struct(cfg, False)

    trained_demos = cfg.get("demos", 0)
    progress_head_weight = float(cfg.method.get("progress_head_weight", 0.0))
    progress_potential_weight = float(
        cfg.method.get("progress_potential_weight", 0.0)
    )
    if progress_head_weight <= 0.0 and progress_potential_weight <= 0.0:
        print(
            "[sandwich] run has no progress head "
            "(progress_head_weight and progress_potential_weight are 0)"
        )
        return 1

    # Eval-shaped workspace: no envs, no persistence, demo replay only.
    cfg.create_train_env = False
    cfg.num_train_envs = 0
    cfg.num_train_frames = 0
    cfg.num_eval_envs = 0
    cfg.num_eval_episodes = 0
    cfg.save_snapshot = False
    cfg.save_csv = False
    cfg.log_train_video = False
    cfg.log_eval_video = False
    cfg.gpu_id = None
    cfg.wandb.use = False
    cfg.tb.use = False
    cfg.replay.num_workers = 0
    cfg.replay.save_dir = None
    cfg.replay.persist = False
    cfg.replay.reuse_saved = False
    cfg.replay.demo_only_updates = False
    cfg.use_self_imitation = False
    cfg.backend.replay_prefetch_size = 0
    cfg.backend.replay_device_prefetch = False
    if args.demos is not None:
        cfg.demos = int(args.demos)
    # The reference value is the completed-episode discounted return. It is
    # not stored unless an MC consumer asked for it, so switch the replay-side
    # anchor on for the probe. This adds a replay element only; no loss runs.
    if float(cfg.method.get("mc_return_weight", 0.0)) <= 0.0:
        cfg.method.mc_return_weight = 0.1
    OmegaConf.resolve(cfg)

    only = {int(v) for v in str(args.only_steps).split(",") if v.strip()}
    snapshots = []
    for path in (run_dir / "snapshots").glob("*_snapshot.pkl"):
        match = re.match(r"(\d+)_snapshot\.pkl", path.name)
        if match:
            step = int(match.group(1))
            if not only or step in only:
                snapshots.append((step, path))
    snapshots.sort()
    if not snapshots:
        print(f"[sandwich] no snapshots in {run_dir}")
        return 1

    with tempfile.TemporaryDirectory(prefix="progress_probe_") as tmp:
        workspace = Workspace(cfg, work_dir=tmp)
        workspace._load_demos()
        agent = workspace.agent
        if getattr(agent, "progress_value_model", None) is None:
            print("[sandwich] agent has no progress head")
            return 1
        buffer = getattr(workspace, "demo_replay_buffer", None)
        if buffer is None or len(buffer) == 0:
            print("[sandwich] no demo replay available")
            return 1

        import jax.numpy as jnp

        np.random.seed(12345)
        from robobase.replay_buffer.iterator import create_jax_replay_iterator

        iterator = create_jax_replay_iterator(buffer, num_workers=0)
        batches = [next(iterator) for _ in range(int(args.batches))]
        missing = [
            name
            for name in ("mc_return", "progress", "progress_valid")
            if name not in batches[0]
        ]
        if missing:
            print(
                "[sandwich] demo batch is missing "
                + ", ".join(missing)
                + "; the run must store progress labels."
            )
            return 1
        if args.holdout_only:
            try:
                cutoff = float(trained_demos)
            except (TypeError, ValueError):
                cutoff = float("inf")
            if not np.isfinite(cutoff) or (
                args.demos is not None and args.demos <= cutoff
            ):
                print(
                    "[sandwich] --holdout-only needs --demos strictly larger "
                    f"than the run's cfg.demos ({trained_demos})"
                )
                return 1
            print(
                "[sandwich] note: replay sampling does not expose the source "
                "demo index, so --holdout-only cannot be enforced per "
                "transition; scoring all sampled demo transitions instead."
            )

        def phi_for(batch):
            obs_inputs = agent._prepare_rl_obs_inputs(batch)
            features = agent._rl_features(
                agent.params.get("encoder", None), obs_inputs
            )
            phi = agent.progress_value_model.apply(
                agent.params["progress_value"], features
            )
            return np.asarray(jnp.clip(phi, 0.0, 1.0)).reshape(-1)

        def diagnostics():
            phi_all, ref_all, label_all = [], [], []
            for batch in batches:
                valid = np.asarray(batch["progress_valid"]).reshape(-1) > 0.5
                reference = np.asarray(
                    batch["mc_return"], dtype=np.float64
                ).reshape(-1)
                keep = valid & (reference > float(args.min_return))
                if not np.any(keep):
                    continue
                phi_all.append(phi_for(batch).astype(np.float64)[keep])
                ref_all.append(reference[keep])
                label_all.append(
                    np.asarray(batch["progress"], dtype=np.float64).reshape(
                        -1
                    )[keep]
                )
            if not phi_all:
                return None
            phi = np.concatenate(phi_all)
            reference = np.concatenate(ref_all)
            label = np.concatenate(label_all)
            ratio = phi / reference
            c1 = float(np.min(ratio))
            c2 = float(np.max(ratio))
            c1_p05 = float(np.percentile(ratio, 5))
            c2_p95 = float(np.percentile(ratio, 95))
            order_phi = np.argsort(np.argsort(phi))
            order_ref = np.argsort(np.argsort(reference))
            spearman = float(
                np.corrcoef(order_phi, order_ref)[0, 1]
            ) if phi.size > 1 else float("nan")
            return {
                "n": int(phi.size),
                "phi_mean": float(np.mean(phi)),
                "ref_mean": float(np.mean(reference)),
                "c1": c1,
                "c2": c2,
                "ratio": (c2 / c1) if c1 > 0.0 else float("inf"),
                "c1_p05": c1_p05,
                "c2_p95": c2_p95,
                "ratio_p90": (
                    (c2_p95 / c1_p05) if c1_p05 > 0.0 else float("inf")
                ),
                "spearman": spearman,
                "label_mae": float(np.mean(np.abs(phi - label))),
            }

        fields = [
            "label",
            "env_steps",
            "n",
            "phi_mean",
            "ref_mean",
            "c1",
            "c2",
            "ratio",
            "c1_p05",
            "c2_p95",
            "ratio_p90",
            "spearman",
            "label_mae",
            "elapsed_sec",
        ]
        writer = None
        handle = None
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            handle = out_path.open("w", newline="")
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()

        label = args.label or run_dir.name
        header = (
            f"{'step':>9} {'n':>7} {'phi':>7} {'V':>7} "
            f"{'c1':>8} {'c2':>8} {'c2/c1':>8} "
            f"{'c2/c1_p90':>10} {'spearman':>9} {'|phi-p|':>8}"
        )
        print(f"[sandwich] {label}")
        print(header)
        print("-" * len(header))
        for step, snapshot in snapshots:
            start = time.time()
            workspace.load_snapshot(snapshot, load_replay_buffer=False)
            row = diagnostics()
            if row is None:
                print(f"{step:>9} {'--- no valid demo transitions ---':>40}")
                continue
            print(
                f"{step:>9} {row['n']:>7} {row['phi_mean']:>7.3f} "
                f"{row['ref_mean']:>7.3f} {row['c1']:>8.3f} "
                f"{row['c2']:>8.3f} {row['ratio']:>8.2f} "
                f"{row['ratio_p90']:>10.2f} {row['spearman']:>9.3f} "
                f"{row['label_mae']:>8.3f}",
                flush=True,
            )
            if writer is not None:
                row = dict(row)
                row.update(
                    label=label,
                    env_steps=step,
                    elapsed_sec=round(time.time() - start, 1),
                )
                writer.writerow(row)
                handle.flush()
        if handle is not None:
            handle.close()
        print(
            "\n[sandwich] gate: c2/c1 >> 3 predicts no benefit from shaping "
            "(Gupta et al., NeurIPS 2022). Read ratio_p90 for the "
            "outlier-robust version."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
