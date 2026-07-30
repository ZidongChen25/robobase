#!/usr/bin/env python3
"""On-path vs off-path Q reliability for CQN-family checkpoints (A-0b).

For executed replay transitions the discounted reward-to-go is ground truth
for Q(s, a_behavior).  This probe stratifies those transitions by how much
zoom-path support the executed action's level-0 bins have inside the replay
(phase-conditional bin share), then reports Q-vs-RTG reliability per stratum.

Prediction under the coverage mechanism: reliability degrades on transitions
whose executed level-0 bins are rare in their phase stratum, for every value
parameterization (C51, MC-anchored, flow) alike.

Read-only.  Reuses the loading machinery of analyze_cqn_value_fidelity.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import analyze_cqn_value_fidelity as base  # noqa: E402

PHASES = ("early", "middle", "late")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--samples-per-group", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offline-episode-count", type=int, default=60)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument(
        "--critic",
        choices=("config", "online", "target"),
        default="config",
    )
    return parser.parse_args()


def phase_of(transition_index: int, episode_length: int) -> str:
    fraction = transition_index / max(episode_length - 1, 1)
    return PHASES[min(int(fraction * 3), 2)]


def encode_level0(actions: np.ndarray, bins: int) -> np.ndarray:
    """Level-0 bin per dimension for actions in [-1, 1]; [T, D] -> int."""
    width = 2.0 / bins
    return np.clip(
        np.floor((actions + 1.0) / width).astype(np.int64), 0, bins - 1
    )


def build_phase_support(
    replay_dir: Path, bins: int
) -> dict[str, np.ndarray]:
    """Phase-conditional level-0 bin counts per dim from executed actions."""
    counts: dict[str, np.ndarray] = {}
    for path in sorted(replay_dir.glob("*.npz")):
        length = int(path.stem.split("_")[-2])
        with np.load(path) as data:
            actions = np.asarray(data["action"][:length], np.float64)
        level0 = encode_level0(actions, bins)
        dims = level0.shape[1]
        for phase in PHASES:
            counts.setdefault(phase, np.zeros((dims, bins), dtype=np.int64))
        for t in range(length):
            phase = phase_of(t, length)
            counts[phase][np.arange(dims), level0[t]] += 1
    return counts


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    args = parse_args()
    base.configure_process(args.gpu_id)

    from omegaconf import OmegaConf

    from robobase.workspace import Workspace

    run_dir = args.run_dir.expanduser().resolve()
    snapshot = (
        args.snapshot
        if args.snapshot is not None
        else run_dir / "snapshots" / "latest_snapshot.pkl"
    ).expanduser().resolve()
    replay_dir = run_dir / "replay"
    cfg_path = run_dir / ".hydra" / "config.yaml"
    for path in (cfg_path, snapshot, replay_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    cfg = OmegaConf.load(cfg_path)
    gamma = float(cfg.replay.gamma)
    started = time.time()

    samples, group_counts, _ = base.select_samples(
        replay_dir,
        gamma=gamma,
        samples_per_group=int(args.samples_per_group),
        samples_per_exploration_group=0,
        seed=int(args.seed),
        offline_episode_count=args.offline_episode_count,
    )
    support = build_phase_support(replay_dir, args.bins)

    OmegaConf.set_struct(cfg, False)
    cfg.create_train_env = False
    cfg.num_train_envs = 0
    cfg.num_train_frames = 0
    cfg.num_eval_envs = 0
    cfg.num_eval_episodes = 0
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
    cfg.replay.persist = False
    cfg.replay.reuse_saved = False
    cfg.backend.replay_prefetch_size = 0
    cfg.backend.replay_device_prefetch = False
    cfg.backend.fused_update_steps = 1
    cfg.backend.update_block_every_steps = 1
    OmegaConf.resolve(cfg)

    records: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="cqn-onpath-") as work_dir:
        workspace = Workspace(cfg, work_dir=work_dir)
        try:
            workspace.load_snapshot(snapshot, load_replay_buffer=False)
            agent = workspace.agent
            if args.critic == "config":
                use_target = bool(
                    cfg.method.get("use_target_network_for_rollout", True)
                )
            else:
                use_target = args.critic == "target"
            effective_k0 = (
                str(cfg.method.get("critic_sequence_mode", "full")).lower()
                == "effective_k0"
            )
            action_dim = int(agent.action_dim)

            batch_size = int(args.batch_size)
            for start in range(0, len(samples), batch_size):
                batch_samples = samples[start : start + batch_size]
                valid = len(batch_samples)
                if valid < batch_size:
                    batch_samples = batch_samples + [batch_samples[-1]] * (
                        batch_size - valid
                    )
                observations, actions = base._load_batch(agent, batch_samples)
                chosen_q, _, _, _, _ = base._checkpoint_q_batch(
                    agent,
                    observations,
                    actions,
                    use_target=use_target,
                    seed=int(args.seed) + start,
                )
                chosen_q = np.asarray(chosen_q)[:valid]
                if effective_k0:
                    chosen_q = chosen_q[:, :, :action_dim]
                # actions: [B, K, D]; executed current action is token 0.
                level0 = encode_level0(
                    np.asarray(actions)[:valid, 0, :], args.bins
                )
                for offset, sample in enumerate(batch_samples[:valid]):
                    phase = phase_of(
                        sample.transition_index, sample.episode_length
                    )
                    phase_counts = support[phase]
                    dims = phase_counts.shape[0]
                    shares = phase_counts[
                        np.arange(dims), level0[offset]
                    ] / np.maximum(phase_counts.sum(axis=1), 1)
                    modal = (
                        level0[offset]
                        == phase_counts.argmax(axis=1)
                    )
                    records.append(
                        {
                            "episode": sample.episode_path.name,
                            "transition_index": sample.transition_index,
                            "group": sample.group,
                            "phase": phase,
                            "discounted_return": sample.discounted_return,
                            "first_success_return": (
                                sample.first_success_return
                            ),
                            "predicted_q": float(
                                np.mean(chosen_q[offset, -1])
                            ),
                            "support_share_mean": float(shares.mean()),
                            "support_share_min": float(shares.min()),
                            "modal_dim_fraction": float(modal.mean()),
                        }
                    )
        finally:
            close = getattr(workspace, "close", None)
            if callable(close):
                close()

    # ---- stratified summary -------------------------------------------
    q = np.array([r["predicted_q"] for r in records])
    rtg = np.array([r["discounted_return"] for r in records])
    share = np.array([r["support_share_mean"] for r in records])
    modal_frac = np.array([r["modal_dim_fraction"] for r in records])
    groups = np.array([r["group"] for r in records])

    def stratum_summary(mask: np.ndarray) -> dict:
        if mask.sum() < 3:
            return {"count": int(mask.sum())}
        return {
            "count": int(mask.sum()),
            "spearman_q_rtg": spearman(q[mask], rtg[mask]),
            "mae_q_rtg": float(np.mean(np.abs(q[mask] - rtg[mask]))),
            "mean_q": float(q[mask].mean()),
            "mean_rtg": float(rtg[mask].mean()),
            "mean_support_share": float(share[mask].mean()),
        }

    quartiles = np.quantile(share, [0.25, 0.5, 0.75])
    strata = {
        "support_q1_lowest": share <= quartiles[0],
        "support_q2": (share > quartiles[0]) & (share <= quartiles[1]),
        "support_q3": (share > quartiles[1]) & (share <= quartiles[2]),
        "support_q4_highest": share > quartiles[2],
        "modal_majority": modal_frac >= 0.5,
        "modal_minority": modal_frac < 0.5,
    }
    summary = {name: stratum_summary(mask) for name, mask in strata.items()}
    summary["all"] = stratum_summary(np.ones_like(share, dtype=bool))
    by_group = {
        group: stratum_summary(groups == group)
        for group in sorted(set(groups))
    }

    result = {
        "run_dir": str(run_dir),
        "snapshot": str(snapshot),
        "critic": args.critic,
        "samples_per_group": int(args.samples_per_group),
        "episode_group_counts": group_counts,
        "share_quartiles": [float(v) for v in quartiles],
        "summary_by_support": summary,
        "summary_by_group": by_group,
        "records": records,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.output} ({len(records)} records)")


if __name__ == "__main__":
    main()
