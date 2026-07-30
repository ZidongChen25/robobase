#!/usr/bin/env python3
"""Stage-148: post-hoc sidecar value calibration on a frozen clean agent.

The canonical dose-response (cqn-flow.md 28.8) showed any online MC anchor
damages clean CQN-AS behavior.  This script trains a small value head on the
FROZEN clean checkpoint's encoder features, entirely offline, so behavior is
untouched by construction.  It reports episode-held-out Spearman/MAE of the
sidecar prediction against discounted reward-to-go of executed transitions.

Scope note: this is behavior-value (Q^pi_b on-path) calibration only; no
counterfactual claim is made (cqn-flow.md sections 25 and 28).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import analyze_cqn_value_fidelity as base  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--heldout-modulus", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--state-only",
        action="store_true",
        help="Exclude the executed action from sidecar inputs.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--extra-replay-dirs",
        default="",
        help="Comma-separated replay dirs contributing TRAIN-only samples "
        "(held-out split stays on the primary run's episodes).",
    )
    parser.add_argument(
        "--max-transitions-per-episode",
        type=int,
        default=40,
        help="Uniformly spread anchors per episode to bound compute.",
    )
    return parser.parse_args()


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    args = parse_args()
    base.configure_process(args.gpu_id)

    import jax
    import jax.numpy as jnp
    import optax
    from flax import linen as nn
    from omegaconf import OmegaConf

    from robobase.workspace import Workspace

    run_dir = args.run_dir.expanduser().resolve()
    snapshot = (
        args.snapshot
        if args.snapshot is not None
        else run_dir / "snapshots" / "latest_snapshot.pkl"
    ).expanduser().resolve()
    replay_dir = run_dir / "replay"
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    gamma = float(cfg.replay.gamma)
    started = time.time()

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

    # ---- gather transitions (episode-level split) ----------------------
    episodes = sorted(replay_dir.glob("*.npz"))
    extra_episodes = []
    for extra in str(args.extra_replay_dirs).split(","):
        extra = extra.strip()
        if extra:
            extra_episodes.extend(sorted(Path(extra).glob("*.npz")))
    samples: list[base.Sample] = []
    split: list[str] = []
    for path in episodes:
        episode_index = int(path.stem.split("_")[-3])
        length = int(path.stem.split("_")[-2])
        with np.load(path) as data:
            rewards = np.asarray(data["reward"][:length], np.float64)
        returns = np.zeros(length)
        running = 0.0
        for i in range(length - 1, -1, -1):
            running = rewards[i] + gamma * running
            returns[i] = running
        member = (
            "heldout"
            if episode_index % args.heldout_modulus == 0
            else "train"
        )
        count = min(args.max_transitions_per_episode, length)
        indices = np.linspace(0, length - 1, count).astype(int)
        for t in indices:
            samples.append(
                base.Sample(
                    episode_path=path,
                    episode_index=episode_index,
                    transition_index=int(t),
                    episode_length=length,
                    group=member,
                    discounted_return=float(returns[t]),
                    first_success_return=0.0,
                    future_success=False,
                )
            )
            split.append(member)
    for path in extra_episodes:
        length = int(path.stem.split("_")[-2])
        with np.load(path) as data:
            rewards = np.asarray(data["reward"][:length], np.float64)
        returns = np.zeros(length)
        running = 0.0
        for i in range(length - 1, -1, -1):
            running = rewards[i] + gamma * running
            returns[i] = running
        count = min(args.max_transitions_per_episode, length)
        indices = np.linspace(0, length - 1, count).astype(int)
        for t in indices:
            samples.append(
                base.Sample(
                    episode_path=path,
                    episode_index=-1,
                    transition_index=int(t),
                    episode_length=length,
                    group="train",
                    discounted_return=float(returns[t]),
                    first_success_return=0.0,
                    future_success=False,
                )
            )
            split.append("train")

    # ---- frozen feature extraction -------------------------------------
    features_list, action_list = [], []
    with tempfile.TemporaryDirectory(prefix="cqn-sidecar-") as work_dir:
        workspace = Workspace(cfg, work_dir=work_dir)
        workspace.load_snapshot(snapshot, load_replay_buffer=False)
        agent = workspace.agent
        for start in range(0, len(samples), args.batch_size):
            chunk = samples[start : start + args.batch_size]
            valid = len(chunk)
            if valid < args.batch_size:
                chunk = chunk + [chunk[-1]] * (args.batch_size - valid)
            observations, actions = base._load_batch(agent, chunk)
            obs_inputs = agent._prepare_rl_obs_inputs(observations)
            feats = agent._rl_features(
                agent.params.get("encoder", None),
                obs_inputs,
                stop_gradient=True,
            )
            features_list.append(np.asarray(feats)[:valid])
            action_list.append(
                np.asarray(actions)[:valid, 0, :]
            )  # executed step

    features = np.concatenate(features_list).astype(np.float32)
    exec_actions = np.concatenate(action_list).astype(np.float32)
    targets = np.array(
        [s.discounted_return for s in samples], dtype=np.float32
    )
    membership = np.array(split)
    inputs = (
        features
        if args.state_only
        else np.concatenate([features, exec_actions], axis=-1)
    )

    train_mask = membership == "train"
    held_mask = membership == "heldout"

    # ---- tiny sidecar head, trained fully offline ----------------------
    class Sidecar(nn.Module):
        hidden: int

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(self.hidden)(x)
            x = nn.LayerNorm()(x)
            x = nn.silu(x)
            x = nn.Dense(self.hidden)(x)
            x = nn.LayerNorm()(x)
            x = nn.silu(x)
            return nn.Dense(1)(x)[..., 0]

    model = Sidecar(hidden=args.hidden)
    rng = jax.random.PRNGKey(args.seed)
    params = model.init(rng, jnp.asarray(inputs[:2]))
    optimizer = (
        optax.adamw(args.lr, weight_decay=args.weight_decay)
        if args.weight_decay > 0.0
        else optax.adam(args.lr)
    )
    opt_state = optimizer.init(params)

    # Keep the dataset host-side; only the current minibatch enters the
    # device (a jit-closure array would be baked as a multi-GB constant).
    x_train_host = inputs[train_mask]
    y_train_host = targets[train_mask]

    @jax.jit
    def step(params, opt_state, x_batch, y_batch):
        def loss_fn(p):
            pred = model.apply(p, x_batch)
            return jnp.mean(jnp.square(pred - y_batch))
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    steps_per_epoch = max(1, x_train_host.shape[0] // 256)
    batch_rng = np.random.default_rng(args.seed)
    last_loss = None
    for epoch in range(args.epochs):
        for _ in range(steps_per_epoch):
            idx = batch_rng.integers(0, x_train_host.shape[0], size=256)
            params, opt_state, last_loss = step(
                params,
                opt_state,
                jnp.asarray(x_train_host[idx]),
                jnp.asarray(y_train_host[idx]),
            )

    prediction_chunks = []
    for start in range(0, inputs.shape[0], 512):
        prediction_chunks.append(
            np.asarray(
                model.apply(
                    params, jnp.asarray(inputs[start : start + 512])
                )
            )
        )
    predictions = np.concatenate(prediction_chunks)

    def report(mask):
        return {
            "count": int(mask.sum()),
            "spearman": spearman(predictions[mask], targets[mask]),
            "mae": float(
                np.mean(np.abs(predictions[mask] - targets[mask]))
            ),
            "mean_prediction": float(predictions[mask].mean()),
            "mean_target": float(targets[mask].mean()),
        }

    result = {
        "run_dir": str(run_dir),
        "snapshot": str(snapshot),
        "episodes": len(episodes),
        "heldout_modulus": args.heldout_modulus,
        "final_train_minibatch_loss": float(last_loss),
        "train": report(train_mask),
        "heldout": report(held_mask),
        "note": (
            "behavior-value calibration only (executed transitions); no "
            "counterfactual claim"
        ),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["heldout"], indent=2))


if __name__ == "__main__":
    main()
