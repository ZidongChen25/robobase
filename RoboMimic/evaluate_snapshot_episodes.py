#!/usr/bin/env python3
"""Evaluate a RoboBase RoboMimic snapshot and save per-episode results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf, open_dict


def _extract_vector_env_info(infos: dict, env_idx: int) -> dict:
    final_infos = infos.get("final_info")
    final_info_mask = infos.get("_final_info")
    if final_infos is not None and final_info_mask is not None and final_info_mask[env_idx]:
        return final_infos[env_idx]

    extracted = {}
    for key, value in infos.items():
        if key.startswith("_"):
            continue
        mask = infos.get(f"_{key}")
        if mask is not None and not mask[env_idx]:
            continue
        extracted[key] = value[env_idx]
    return extracted


def _run_vector_eval(workspace, output: Path, num_episodes: int) -> list[dict]:
    workspace._ensure_eval_envs_created()
    if workspace.eval_envs is None:
        raise ValueError("Expected vector eval envs; set num_eval_envs > 1.")

    env = workspace.eval_envs
    observation, _ = env.reset()
    workspace.agent.reset(workspace.main_loop_iterations, workspace._eval_agent_indices)
    workspace.agent.set_eval_env_running(True)

    episode_rewards = np.zeros(env.num_envs, dtype=np.float64)
    episode_lengths = np.zeros(env.num_envs, dtype=np.int32)
    rows = []
    try:
        while len(rows) < num_episodes:
            (
                _action,
                (next_observation, reward, termination, truncation, next_info),
                _env_metrics,
            ) = workspace._perform_env_steps(observation, env, True)
            observation = next_observation
            episode_rewards += reward
            episode_lengths += 1

            done_mask = np.logical_or(termination, truncation)
            if not np.any(done_mask):
                continue

            done_env_indices = np.flatnonzero(done_mask)
            workspace.agent.reset(
                workspace.main_loop_iterations,
                [workspace._eval_agent_indices[idx] for idx in done_env_indices],
            )
            for env_idx in done_env_indices:
                info = _extract_vector_env_info(next_info, int(env_idx))
                success_raw = info.get("task_success")
                success = (
                    int(np.asarray(success_raw).astype(int).item())
                    if success_raw is not None
                    else ""
                )
                rows.append(
                    {
                        "episode": len(rows),
                        "env_idx": int(env_idx),
                        "reward": float(episode_rewards[env_idx]),
                        "length": int(episode_lengths[env_idx]),
                        "success": success,
                        "termination": int(np.asarray(termination[env_idx]).item()),
                        "truncation": int(np.asarray(truncation[env_idx]).item()),
                        "task_success_raw": success_raw,
                    }
                )
                episode_rewards[env_idx] = 0.0
                episode_lengths[env_idx] = 0
                if len(rows) >= num_episodes:
                    break
    finally:
        workspace.agent.set_eval_env_running(False)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "env_idx",
                "reward",
                "length",
                "success",
                "termination",
                "truncation",
                "task_success_raw",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    snapshot = args.snapshot or run_dir / "snapshots" / "100000_snapshot.pkl"
    output = args.output or run_dir / "episode_eval_verify.csv"

    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    with open_dict(cfg):
        cfg.num_eval_episodes = int(args.episodes)
        cfg.log_eval_video = False
        cfg.wandb.use = False
        cfg.save_csv = False

    from robobase.workspace import Workspace

    workspace = Workspace(cfg, work_dir=run_dir)
    try:
        workspace.load_snapshot(snapshot, load_replay_buffer=False)
        rows = _run_vector_eval(workspace, output, int(args.episodes))
    finally:
        workspace.shutdown()

    successes = [float(row["success"]) for row in rows if row["success"] != ""]
    rewards = [float(row["reward"]) for row in rows]
    lengths = [int(row["length"]) for row in rows]
    print(output)
    print(f"episodes={len(rows)}")
    print(f"mean_success={np.mean(successes) if successes else float('nan')}")
    print(f"mean_reward={np.mean(rewards) if rewards else float('nan')}")
    print(f"mean_length={np.mean(lengths) if lengths else float('nan')}")
    print(f"success_count={int(np.sum(successes)) if successes else 0}")


if __name__ == "__main__":
    main()
