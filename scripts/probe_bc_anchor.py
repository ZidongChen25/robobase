"""Retrospective BC-anchor diagnostics over a run's snapshot archive.

The margin hinge in the CQN/CQN-AS loss implements a constraint --
``Q(s, a_demo) >= max_sibling Q(s, a') + m`` on demonstration states -- and
``bc_lambda`` is its penalty coefficient, not a "BC weight".  A penalty
coefficient is only interpretable relative to the opposing force, whose scale
is task-specific (reward density, horizon, Q range, demo count), which is why
hand-tuned lambda schedules do not transfer between tasks.

This probe measures the constraint itself instead of its coefficient, on a
batch of demonstration transitions that is identical across runs (the base
BiGym demos are re-loaded from the dataset, in dataset order, for every run).
Per snapshot it reports:

    agreement       fraction of demo (level, dim) heads whose argmax bin is
                    the demonstrated one -- the constraint in behaviour space
    binding_rate    fraction of sibling bins violating the margin -- how much
                    of the hinge is actually exerting force
    margin_gap      mean (Q_demo - max_sibling Q), signed; the constraint's
                    slack in value units
    sibling_span    mean (max - min) Q across sibling bins -- how much
                    counterfactual structure the critic has learned at all
    chosen_q        mean Q of the demonstrated action, for scale context

Usage:
    python scripts/probe_bc_anchor.py --run-dir exp_local/<run> \
        --output exp_local/<run>/bc_anchor_probe.csv [--gpu-id 3]
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
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--egl-device-id", type=int, default=None)
    parser.add_argument(
        "--batches",
        type=int,
        default=4,
        help="Demo batches averaged per snapshot.",
    )
    parser.add_argument("--only-steps", default="")
    parser.add_argument("--label", default="")
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
    # Eval-shaped workspace: no envs, no persistence, but keep the demo replay
    # so the probe batch comes from the shared base demonstrations.
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
        print(f"[probe] no snapshots in {run_dir}")
        return 1

    with tempfile.TemporaryDirectory(prefix="bc_probe_") as tmp:
        workspace = Workspace(cfg, work_dir=tmp)
        # Demos are normally ingested at the top of _train(); the probe never
        # trains, so pull them in explicitly. This re-reads the dataset, which
        # is what makes the probe batch identical across runs.
        workspace._load_demos()
        agent = workspace.agent
        buffer = getattr(workspace, "demo_replay_buffer", None)
        if buffer is None or len(buffer) == 0:
            print("[probe] no demo replay available; cannot build probe batch")
            return 1

        import jax.numpy as jnp

        # Fixed probe batches: identical across runs given the same dataset,
        # demo count and seed.
        np.random.seed(12345)
        from robobase.replay_buffer.iterator import create_jax_replay_iterator

        iterator = create_jax_replay_iterator(buffer, num_workers=0)
        batches = [next(iterator) for _ in range(int(args.batches))]

        def diagnostics(batch):
            obs_inputs = agent._prepare_rl_obs_inputs(batch)
            actions = agent._as_jax_array(
                batch["action"], jnp.float32
            ).reshape((batch["action"].shape[0], -1))
            features = agent._rl_features(
                agent.params.get("encoder", None), obs_inputs
            )
            chosen_logits, all_logits = agent._critic_logits_per_level(
                agent.params["critic"], features, actions
            )
            chosen_q = jnp.sum(
                jax_softmax(chosen_logits) * agent.support, axis=-1
            )
            all_q = jnp.sum(jax_softmax(all_logits) * agent.support, axis=-1)
            best_q = jnp.max(all_q, axis=-1)
            gap = chosen_q - best_q
            sibling = jnp.abs(chosen_q[..., None] - all_q) > 1e-9
            violating = (
                (float(agent.bc_margin) - (chosen_q[..., None] - all_q)) > 0.0
            ) & sibling
            binding = jnp.sum(
                violating.astype(jnp.float32)
            ) / jnp.maximum(jnp.sum(sibling.astype(jnp.float32)), 1.0)
            return {
                "agreement": float(jnp.mean((gap >= -1e-6).astype(jnp.float32))),
                "binding_rate": float(binding),
                "margin_gap": float(jnp.mean(gap)),
                "sibling_span": float(
                    jnp.mean(jnp.max(all_q, axis=-1) - jnp.min(all_q, axis=-1))
                ),
                "chosen_q": float(jnp.mean(chosen_q)),
            }

        def jax_softmax(x):
            import jax

            return jax.nn.softmax(x, axis=-1)

        from robobase.utils import schedule as lambda_schedule

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "label",
            "env_steps",
            "bc_lambda",
            "agreement",
            "binding_rate",
            "margin_gap",
            "sibling_span",
            "chosen_q",
            "elapsed_sec",
        ]
        with out_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            label = args.label or run_dir.name
            sched = cfg.method.get("bc_lambda_schedule", None)
            for step, snapshot in snapshots:
                start = time.time()
                workspace.load_snapshot(snapshot, load_replay_buffer=False)
                rows = [diagnostics(b) for b in batches]
                merged = {
                    key: float(np.mean([r[key] for r in rows])) for key in rows[0]
                }
                lam = (
                    float(lambda_schedule(sched, step))
                    if sched
                    else float(cfg.method.get("bc_lambda", 0.0))
                )
                merged.update(
                    label=label,
                    env_steps=step,
                    bc_lambda=round(lam, 6),
                    elapsed_sec=round(time.time() - start, 1),
                )
                writer.writerow(merged)
                handle.flush()
                print(
                    f"[probe] {label}@{step}: lam={lam:.4f} "
                    f"agree={merged['agreement']:.3f} "
                    f"bind={merged['binding_rate']:.3f} "
                    f"gap={merged['margin_gap']:+.4f} "
                    f"span={merged['sibling_span']:.4f}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
