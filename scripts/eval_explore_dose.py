"""Exploration dose-response: success rate vs exploration strength.

Runs N episodes on one checkpoint under several exploration settings.
The information an exploration scheme feeds the critic is bounded by the
success it destroys; a dose that leaves success unchanged explores only
outcome-irrelevant directions/scales.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint-step", type=int, required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed-start", type=int, default=400)
    p.add_argument("--out-json", required=True)
    p.add_argument(
        "--variants",
        default="greedy;noise;asis:0.002,0.004,0.008;"
        "x4:0.008,0.016,0.032;coarse:0.03,0,0",
        help="semicolon list: greedy | noise | name:l0,l1,l2",
    )
    return p.parse_args()


def main():
    args = parse_args()
    from omegaconf import OmegaConf
    from robobase.workspace import Workspace

    run_dir = Path(args.run_dir).resolve()
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    OmegaConf.set_struct(cfg, False)
    cfg.create_train_env = False
    cfg.num_train_envs = 25  # size train-side per-env state for batch acting
    cfg.num_train_frames = 0
    cfg.num_eval_envs = 25
    cfg.num_eval_episodes = 1
    cfg.env.eval_seed_start = int(args.seed_start)
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
    OmegaConf.resolve(cfg)

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    work_dir = out.parent / "workspace"
    work_dir.mkdir(exist_ok=True)
    workspace = Workspace(cfg, work_dir=str(work_dir))
    ckpt = (
        run_dir / "eval_checkpoints"
        / f"{args.checkpoint_step}_checkpoint.pkl"
    )
    workspace.load_snapshot(ckpt, load_replay_buffer=False)
    agent = workspace.agent
    venv = workspace.eval_envs
    B = venv.num_envs
    ckpt_step = int(args.checkpoint_step)

    def reset_agent_explore_state():
        # Reset BOTH act-state banks. Train slots are indices
        # [0, num_train_envs); eval slots live at num_train_envs + i and hold
        # the open-loop plan/position used by eval_mode acting — without this
        # reset, episodes after round 1 start by executing the previous
        # round's stale chunk (up to action_sequence-1 garbage actions).
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

    def run_variant(name, eval_mode, probs):
        agent.bin_explore_probs = probs
        successes = 0
        explored_steps = 0
        total_steps = 0
        rounds = (args.episodes + B - 1) // B
        for rd in range(rounds):
            seeds = [args.seed_start + rd * B + i for i in range(B)]
            obs, _ = venv.reset(seed=seeds)
            reset_agent_explore_state()
            ep_return = np.zeros(B)
            ep_done = np.zeros(B, dtype=bool)
            ep_success = np.zeros(B, dtype=bool)
            while not ep_done.all():
                action = np.asarray(
                    agent.act(obs, step=ckpt_step, eval_mode=eval_mode)
                )
                if not eval_mode:
                    flags = np.asarray(
                        getattr(agent, "_last_bin_explored",
                                np.zeros(B, bool))
                    ).ravel()
                    explored_steps += int(flags[~ep_done].sum())
                total_steps += int((~ep_done).sum())
                obs, reward, term, trunc, infos = venv.step(action)
                reward = np.asarray(reward, np.float64)
                ep_return[~ep_done] += reward[~ep_done]
                just_done = (
                    (np.asarray(term) | np.asarray(trunc)) & ~ep_done
                )
                final_infos = infos.get("final_info")
                for i in np.flatnonzero(just_done):
                    fi = (
                        final_infos[i]
                        if final_infos is not None and final_infos[i]
                        else {}
                    )
                    ts = fi.get("task_success")
                    ep_success[i] = (
                        bool(np.asarray(ts).item())
                        if ts is not None
                        else ep_return[i] > 0
                    )
                ep_done |= just_done
            successes += int(ep_success.sum())
            print(f"[dose]   round {rd}: {int(ep_success.sum())}/{B}",
                  flush=True)
        rec = {
            "variant": name,
            "eval_mode": eval_mode,
            "probs": probs,
            "episodes": args.episodes,
            "success_rate": successes / args.episodes,
            "explored_step_fraction": (
                explored_steps / max(1, total_steps) if not eval_mode else 0.0
            ),
        }
        print(f"[dose] {name}: success {rec['success_rate']:.2f} "
              f"explored_frac {rec['explored_step_fraction']:.3f}",
              flush=True)
        return rec

    results = []
    for spec in args.variants.split(";"):
        spec = spec.strip()
        if spec == "greedy":
            results.append(run_variant("greedy", True, None))
        elif spec == "noise":
            results.append(run_variant("noise_only", False, None))
        else:
            name, probs_str = spec.split(":")
            probs = [float(x) for x in probs_str.split(",")]
            results.append(run_variant(name, False, probs))

    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[dose] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
