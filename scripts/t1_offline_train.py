#!/usr/bin/env python
"""T1: offline CQN-AS training on the frozen T1 dataset, one target operator.

Every arm trains from scratch, on the SAME frozen buffer (demos + collected
policy episodes minus a fixed held-out split), for the same number of
gradient updates, with the same seed. Only the value-target operator differs.

All arms carry ``mc_return`` in the replay signature (TD arms use
mc_return_weight=1e-9, i.e. zero to float32 precision) so that every arm shares
one demo cache, one storage signature and one batch layout.

Arms
  td      critic_lambda=0.1, mc weight ~0            1-step distributional TD
  mc      critic_lambda=0.0, mc_return_weight=0.1    pure MC return regression
  mclb    critic_lambda=0.1, mc_lower_bound_target   elementwise max(TD, MC)
  sarsa   td with td_target_action_source=replay_next in-sample (on-data) TD
  td_nobc td with bc_lambda=0                        collapse reference
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

ARMS = {
    "td": {
        "critic_lambda": 0.1,
        "mc_return_weight": 1e-9,
        "mc_lower_bound_target": False,
        "bc_lambda": 1.0,
    },
    "mc": {
        "critic_lambda": 0.0,
        "mc_return_weight": 0.1,
        "mc_lower_bound_target": False,
        "bc_lambda": 1.0,
    },
    "mclb": {
        "critic_lambda": 0.1,
        "mc_return_weight": 1e-9,
        "mc_lower_bound_target": True,
        "bc_lambda": 1.0,
    },
    "sarsa": {
        "critic_lambda": 0.1,
        "mc_return_weight": 1e-9,
        "mc_lower_bound_target": False,
        "bc_lambda": 1.0,
        "td_target_action_source": "replay_next",
    },
    "td_nobc": {
        "critic_lambda": 0.1,
        "mc_return_weight": 1e-9,
        "mc_lower_bound_target": False,
        "bc_lambda": 0.0,
        "bc_margin": 0.0,
        "demo_fosd": False,
    },
}

OBS_EXCLUDE = {"action", "reward", "terminal", "truncated", "demo", "mc_return"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="source run for .hydra cfg")
    p.add_argument("--frozen-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--arm", required=True, choices=sorted(ARMS))
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--snapshot-every", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--holdout-stride", type=int, default=5)
    p.add_argument("--smoke", type=int, default=0)
    return p.parse_args()


def holdout_split(frozen_dir: Path, stride: int):
    """Deterministic per-block every-Nth holdout, identical for all arms."""
    manifest = [
        json.loads(line)
        for line in (frozen_dir / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    by_block = {}
    for record in manifest:
        by_block.setdefault(record["block"], []).append(record)
    train, held = [], []
    for block in sorted(by_block):
        records = sorted(by_block[block], key=lambda r: r["seed"])
        for position, record in enumerate(records):
            (held if position % stride == 0 else train).append(record)
    return train, held


def inject_episodes(replay_buffer, frozen_dir: Path, records):
    from robobase.replay_buffer.uniform_replay_buffer import load_episode

    extras = set(replay_buffer.extra_replay_elements.keys())
    total = 0
    for record in records:
        episode = load_episode(frozen_dir / record["file"])
        obs_keys = [k for k in episode if k not in OBS_EXCLUDE]
        length = episode["action"].shape[0] - 1
        for index in range(length):
            observation = {key: episode[key][index] for key in obs_keys}
            extra = {}
            if "demo" in extras:
                extra["demo"] = np.uint8(0)
            if "mc_return" in extras:
                extra["mc_return"] = np.float32(episode["mc_return"][index])
            replay_buffer.add(
                observation,
                episode["action"][index],
                np.float32(episode["reward"][index]),
                bool(episode["terminal"][index] == 1),
                bool(episode["truncated"][index] == 1),
                **extra,
            )
        final = {key: episode[key][length] for key in obs_keys}
        replay_buffer.add_final(final)
        total += length
    return total


def main():
    args = parse_args()
    from omegaconf import OmegaConf
    from robobase.workspace_fast import WorkspaceFast

    run_dir = Path(args.run_dir).resolve()
    frozen_dir = Path(args.frozen_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    OmegaConf.set_struct(cfg, False)
    cfg.seed = int(args.seed)
    cfg.create_train_env = False
    cfg.num_train_envs = 0
    cfg.num_train_frames = 0
    cfg.num_eval_envs = 0
    cfg.num_eval_episodes = 0
    cfg.num_pretrain_steps = int(args.steps)
    cfg.num_pretrain_epochs = None
    cfg.eval_every_epochs = None
    cfg.snapshot_every_epochs = None
    cfg.eval_every_steps = 10**9
    cfg.snapshot_every_n = int(args.snapshot_every)
    cfg.snapshot_save_start_step = 0
    cfg.save_snapshot = True
    cfg.save_csv = True
    cfg.log_pretrain_every = 250
    cfg.log_train_video = False
    cfg.log_eval_video = False
    cfg.use_self_imitation = False
    cfg.replay_size_before_train = 0
    cfg.gpu_id = None
    cfg.wandb.use = False
    cfg.tb.use = False
    cfg.artifacts.save_eval_checkpoints = True
    cfg.artifacts.resume_keep_last = 1
    cfg.artifacts.delete_replay_on_train_complete = True
    cfg.artifacts.delete_resume_on_train_complete = True
    for key, value in ARMS[args.arm].items():
        cfg.method[key] = value
    OmegaConf.resolve(cfg)

    work_dir = out_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    hydra_dir = work_dir / ".hydra"
    hydra_dir.mkdir(exist_ok=True)
    OmegaConf.save(cfg, hydra_dir / "config.yaml")

    train_records, held_records = holdout_split(frozen_dir, args.holdout_stride)
    if args.smoke:
        train_records = train_records[: args.smoke]
    (work_dir / "t1_split.json").write_text(
        json.dumps(
            {
                "frozen_dir": str(frozen_dir),
                "holdout_stride": args.holdout_stride,
                "train_files": [r["file"] for r in train_records],
                "holdout_files": [r["file"] for r in held_records],
            },
            indent=2,
        )
        + "\n"
    )

    workspace = WorkspaceFast(cfg, work_dir=str(work_dir))
    print(
        f"[t1t] arm={args.arm} seed={args.seed} steps={args.steps} "
        f"work_dir={work_dir}",
        flush=True,
    )
    workspace._load_demos()
    demo_transitions = len(workspace.replay_buffer)
    injected = inject_episodes(workspace.replay_buffer, frozen_dir, train_records)
    print(
        f"[t1t] buffer: {demo_transitions} demo transitions + {injected} "
        f"collected transitions ({len(train_records)} episodes) = "
        f"{len(workspace.replay_buffer)}; demo buffer "
        f"{len(workspace.demo_replay_buffer)}",
        flush=True,
    )
    spec = {
        "arm": args.arm,
        "seed": int(args.seed),
        "steps": int(args.steps),
        "overrides": ARMS[args.arm],
        "critic_lambda": float(cfg.method.critic_lambda),
        "mc_return_weight": float(cfg.method.mc_return_weight),
        "mc_lower_bound_target": bool(cfg.method.mc_lower_bound_target),
        "td_target_action_source": str(cfg.method.td_target_action_source),
        "bc_lambda": float(cfg.method.bc_lambda),
        "dense_return_q_target": bool(cfg.method.dense_return_q_target),
        "replay_extra_elements": sorted(
            workspace.replay_buffer.extra_replay_elements.keys()
        ),
        "demo_transitions": int(demo_transitions),
        "collected_transitions": int(injected),
        "train_episodes": len(train_records),
        "holdout_episodes": len(held_records),
        "shared_demo_cache": (
            str(workspace._shared_demo_cache.path)
            if workspace._shared_demo_cache is not None
            else None
        ),
    }
    (work_dir / "t1_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(f"[t1t] spec {json.dumps(spec)}", flush=True)

    started = time.time()
    workspace._pretrain_on_demos()
    workspace.save_snapshot()
    workspace._finalize_completed_training_artifacts()
    workspace.shutdown()
    elapsed = time.time() - started
    (work_dir / "t1_done.json").write_text(
        json.dumps(
            {
                "arm": args.arm,
                "seed": int(args.seed),
                "steps": int(args.steps),
                "elapsed_sec": round(elapsed, 1),
                "updates_per_sec": round(args.steps / max(elapsed, 1e-9), 3),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"[t1t] arm={args.arm} done in {elapsed / 60:.1f} min "
        f"({args.steps / max(elapsed, 1e-9):.2f} updates/s)",
        flush=True,
    )


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    raise SystemExit(main())
