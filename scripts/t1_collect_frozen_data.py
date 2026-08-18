#!/usr/bin/env python
"""T1: collect a frozen mixed dataset with one frozen CQN-AS policy.

The dataset is built ONCE and reused by every T1 arm, so the only thing that
differs between arms is the value-target operator. Episodes are written in the
exact UniformReplayBuffer npz schema (plus ``mc_return``) so they can be
injected into a training buffer without any re-simulation.

Act-state trap (see scripts/eval_explore_dose.py): the agent keeps TWO
per-env act-state banks -- train slots [0, num_train_envs) and eval slots at
num_train_envs + i. Both must be reset between episode batches or later
rounds start by executing the previous round's stale open-loop chunk.
"""

import argparse
import datetime
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint-step", type=int, default=100000)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-envs", type=int, default=12)
    p.add_argument("--seed-start", type=int, default=2000)
    p.add_argument(
        "--blocks",
        default="greedy:100;asis:25:0.002,0.004,0.008;x4:25:0.008,0.016,0.032",
        help="semicolon list: greedy:N | name:N:l0,l1,l2",
    )
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    from omegaconf import OmegaConf
    from robobase.workspace import Workspace, _replay_action_from_step
    from robobase.replay_buffer.uniform_replay_buffer import save_episode
    from robobase import utils as rb_utils

    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    OmegaConf.set_struct(cfg, False)
    B = int(args.num_envs)
    cfg.create_train_env = False
    cfg.num_train_envs = B  # sizes the train-side act-state bank
    cfg.num_train_frames = 0
    cfg.num_eval_envs = B
    cfg.num_eval_episodes = 1
    cfg.env.eval_seed_start = int(args.seed_start)
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
    cfg.replay.demo_cache_dir = None
    cfg.backend.replay_prefetch_size = 0
    cfg.backend.replay_device_prefetch = False
    cfg.backend.fused_update_steps = 1
    cfg.backend.update_block_every_steps = 1
    OmegaConf.resolve(cfg)

    work_dir = out_dir / "_collect_workspace"
    work_dir.mkdir(parents=True, exist_ok=True)
    workspace = Workspace(cfg, work_dir=str(work_dir))
    ckpt = (
        run_dir / "eval_checkpoints" / f"{args.checkpoint_step}_checkpoint.pkl"
    )
    workspace.load_snapshot(ckpt, load_replay_buffer=False)
    agent = workspace.agent
    venv = workspace.eval_envs
    ckpt_step = int(args.checkpoint_step)
    gamma = float(cfg.replay.gamma)
    obs_keys = sorted(venv.single_observation_space.spaces.keys())
    print(f"[t1c] obs keys: {obs_keys}", flush=True)

    def reset_agent_state():
        n_train = int(getattr(agent, "num_train_envs", B))
        agent.reset(
            ckpt_step, list(range(n_train)) + [n_train + i for i in range(B)]
        )
        for attr in (
            "_bin_explored_exec_remaining",
            "_last_bin_explored",
            "_last_bin_explore_applied",
        ):
            if hasattr(agent, attr):
                delattr(agent, attr)

    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    manifest_path = out_dir / "manifest.jsonl"
    n_written = 0
    global_idx = 0
    seed_cursor = int(args.seed_start)

    def last_frame(value):
        return np.asarray(value[-1]).copy()

    def assemble_and_save(rows, final_frames, meta):
        """rows: list of (obs_frame_dict, action, reward, term, trunc)."""
        nonlocal n_written, global_idx
        n = len(rows)
        rewards = np.asarray([r[2] for r in rows], dtype=np.float32)
        mc = rb_utils.discounted_episode_returns(rewards, gamma)
        episode = {}
        for key in obs_keys:
            episode[key] = np.stack(
                [r[0][key] for r in rows] + [final_frames[key]], axis=0
            )
        action_dim = rows[0][1].shape[0]
        episode["action"] = np.concatenate(
            [
                np.stack([r[1] for r in rows], axis=0),
                np.zeros((1, action_dim), dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32)
        episode["reward"] = np.concatenate([rewards, [0.0]]).astype(np.float32)
        terminal = np.zeros(n + 1, dtype=np.int8)
        truncated = np.zeros(n + 1, dtype=np.int8)
        if rows[-1][3]:
            terminal[n - 1] = 1
        else:
            truncated[n - 1] = 1
        terminal[n] = -1
        truncated[n] = -1
        episode["terminal"] = terminal
        episode["truncated"] = truncated
        episode["demo"] = np.zeros(n + 1, dtype=np.uint8)
        episode["mc_return"] = np.concatenate(
            [np.asarray(mc, dtype=np.float32), [0.0]]
        ).astype(np.float32)
        name = f"{stamp}_{n_written}_{n}_{global_idx}.npz"
        save_episode(episode, out_dir / name, compression="zip")
        global_idx += n
        n_written += 1
        record = dict(meta)
        record.update(
            {
                "file": name,
                "length": n,
                "episode_return": float(rewards.sum()),
                "mc_return_0": float(mc[0]),
                "terminated": bool(rows[-1][3]),
            }
        )
        with manifest_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        return record

    def run_block(name, episodes, probs):
        nonlocal seed_cursor
        agent.bin_explore_probs = probs
        eval_mode = probs is None
        done_eps = 0
        successes = 0
        while done_eps < episodes:
            batch = min(B, episodes - done_eps)
            seeds = [seed_cursor + i for i in range(B)]
            seed_cursor += B
            obs, _ = venv.reset(seed=seeds)
            reset_agent_state()
            rows = [[] for _ in range(B)]
            ep_done = np.zeros(B, dtype=bool)
            ep_success = np.zeros(B, dtype=bool)
            final_frames = [None] * B
            explored_steps = np.zeros(B, dtype=np.int64)
            total_steps = np.zeros(B, dtype=np.int64)
            while not ep_done.all():
                action = np.asarray(
                    agent.act(obs, step=ckpt_step, eval_mode=eval_mode)
                )
                flags = np.asarray(
                    getattr(agent, "_last_bin_explored", np.zeros(B, bool))
                ).ravel()
                obs_frames = [
                    {k: last_frame(obs[k][i]) for k in obs_keys}
                    for i in range(B)
                ]
                next_obs, reward, term, trunc, infos = venv.step(action)
                reward = np.asarray(reward, np.float64)
                term = np.asarray(term)
                trunc = np.asarray(trunc)
                just_done = (term | trunc) & ~ep_done
                final_infos = infos.get("final_info")
                final_observations = infos.get("final_observation")
                for i in range(B):
                    if ep_done[i]:
                        continue
                    next_info_i = {k: infos[k][i] for k in infos.keys()}
                    executed = _replay_action_from_step(action[i], next_info_i)
                    rows[i].append(
                        (
                            obs_frames[i],
                            np.asarray(executed, np.float32),
                            float(reward[i]),
                            bool(term[i]),
                            bool(trunc[i]),
                        )
                    )
                    total_steps[i] += 1
                    if not eval_mode and flags.size > i:
                        explored_steps[i] += int(bool(flags[i]))
                    if just_done[i]:
                        fi = (
                            final_infos[i]
                            if final_infos is not None and final_infos[i]
                            else {}
                        )
                        ts = fi.get("task_success")
                        ep_success[i] = (
                            bool(np.asarray(ts).item())
                            if ts is not None
                            else False
                        )
                        fo = final_observations[i]
                        final_frames[i] = {
                            k: last_frame(fo[k]) for k in obs_keys
                        }
                ep_done |= just_done
                obs = next_obs
            for i in range(batch):
                meta = {
                    "block": name,
                    "eval_mode": eval_mode,
                    "bin_explore_probs": probs,
                    "seed": int(seeds[i]),
                    "success": bool(ep_success[i]),
                    "explored_step_fraction": float(
                        explored_steps[i] / max(1, total_steps[i])
                    ),
                }
                assemble_and_save(rows[i], final_frames[i], meta)
            successes += int(ep_success[:batch].sum())
            done_eps += batch
            print(
                f"[t1c] {name}: {done_eps}/{episodes} eps, "
                f"{successes} success so far",
                flush=True,
            )
        return successes

    summary = []
    for spec in args.blocks.split(";"):
        spec = spec.strip()
        if not spec:
            continue
        parts = spec.split(":")
        name = parts[0]
        episodes = int(parts[1])
        if args.smoke:
            episodes = min(episodes, B)
        probs = (
            None
            if len(parts) < 3 or not parts[2]
            else [float(x) for x in parts[2].split(",")]
        )
        successes = run_block(name, episodes, probs)
        summary.append(
            {
                "block": name,
                "episodes": episodes,
                "successes": successes,
                "success_rate": successes / max(1, episodes),
                "bin_explore_probs": probs,
            }
        )
        print(f"[t1c] block {name} done: {summary[-1]}", flush=True)
        if args.smoke:
            break

    (out_dir / "collection_summary.json").write_text(
        json.dumps(
            {
                "source_run": str(run_dir),
                "checkpoint_step": ckpt_step,
                "gamma": gamma,
                "num_envs": B,
                "seed_start": int(args.seed_start),
                "episodes_written": n_written,
                "blocks": summary,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[t1c] wrote {n_written} episodes to {out_dir}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
