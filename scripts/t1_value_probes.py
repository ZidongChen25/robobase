#!/usr/bin/env python
"""T1: value-quality probes for one trained arm (no rollouts, no env).

Everything is computed from stored transitions, so the readout depends only on
the critic, never on the behavior the BC hinge pins. Probe states come from the
held-out slice of the frozen dataset plus the demonstrations.

Probes
  a  per-level argmax agreement between the critic's bin choice and the
     recorded action's bin (teacher-forced on the recorded coarse bins)
  b  discrimination ranking of {healthy, hijacked, garbage} chunks
  c  Q-span (max-min over bins) per level, per state group
  d  failure verdict: Q(executed failing chunk) vs Q(nearest-demo chunk)
  e  return calibration: Q(executed chunk) vs true discounted return-to-go
  f  success/failure state separation (AUC of Q over held-out states)
"""

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np

OBS_EXCLUDE = {"action", "reward", "terminal", "truncated", "demo", "mc_return"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arm-dir", required=True)
    p.add_argument("--checkpoint-steps", default="", help="comma list; default all")
    p.add_argument("--frozen-dir", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--states-per-group", type=int, default=600)
    p.add_argument("--demo-states", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--seed", type=int, default=17)
    return p.parse_args()


def load_manifest(frozen_dir: Path):
    return {
        record["file"]: record
        for record in (
            json.loads(line)
            for line in (frozen_dir / "manifest.jsonl").read_text().splitlines()
            if line.strip()
        )
    }


class EpisodeStore:
    """Lazily materialized npz episodes with replay-identical slicing."""

    def __init__(self, paths, frame_stack, action_seq):
        self.paths = list(paths)
        self.frame_stack = int(frame_stack)
        self.action_seq = int(action_seq)
        self._cache = {}

    def episode(self, index):
        from robobase.replay_buffer.uniform_replay_buffer import load_episode

        if index not in self._cache:
            self._cache[index] = load_episode(Path(self.paths[index]))
        return self._cache[index]

    def length(self, index):
        return self.episode(index)["action"].shape[0] - 1

    def obs_keys(self, index):
        return [k for k in self.episode(index) if k not in OBS_EXCLUDE]

    def stacked_obs(self, index, step):
        episode = self.episode(index)
        ep_len = self.length(index)
        idxs = np.clip(
            np.arange(step - self.frame_stack + 1, step + 1), 0, ep_len
        )
        return {key: episode[key][idxs] for key in self.obs_keys(index)}

    def chunk(self, index, step):
        episode = self.episode(index)
        ep_len = self.length(index)
        stop = min(step + self.action_seq, ep_len)
        actions = episode["action"][step:stop]
        if actions.shape[0] < self.action_seq:
            pad = self.action_seq - actions.shape[0]
            actions = np.concatenate(
                [actions, np.repeat(actions[-1:], pad, axis=0)], axis=0
            )
        return actions.astype(np.float32)


def auc(positive, negative):
    positive = np.asarray(positive, np.float64)
    negative = np.asarray(negative, np.float64)
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    values = np.concatenate([positive, negative])
    order = values.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1)
    # average ranks for ties
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    rank_sum = ranks[: positive.size].sum()
    return float(
        (rank_sum - positive.size * (positive.size + 1) / 2.0)
        / (positive.size * negative.size)
    )


def main():
    args = parse_args()
    import jax
    import jax.numpy as jnp
    from omegaconf import OmegaConf
    from robobase.workspace import Workspace
    from robobase.method.cqn_research import encode_action

    arm_dir = Path(args.arm_dir).resolve()
    frozen_dir = Path(args.frozen_dir).resolve()
    spec = json.loads((arm_dir / "t1_spec.json").read_text())
    split = json.loads((arm_dir / "t1_split.json").read_text())
    manifest = load_manifest(frozen_dir)

    cfg = OmegaConf.load(arm_dir / ".hydra" / "config.yaml")
    OmegaConf.set_struct(cfg, False)
    cfg.create_train_env = False
    cfg.num_train_envs = 0
    cfg.num_train_frames = 0
    cfg.num_eval_envs = 0
    cfg.num_eval_episodes = 0
    cfg.num_pretrain_steps = 0
    cfg.save_snapshot = False
    cfg.save_csv = False
    cfg.wandb.use = False
    cfg.tb.use = False
    cfg.gpu_id = None
    cfg.replay.num_workers = 0
    cfg.replay.demo_cache_dir = None
    cfg.backend.replay_prefetch_size = 0
    cfg.backend.replay_device_prefetch = False
    OmegaConf.resolve(cfg)

    work_dir = arm_dir / "_probe_workspace"
    work_dir.mkdir(parents=True, exist_ok=True)
    workspace = Workspace(cfg, work_dir=str(work_dir))
    agent = workspace.agent

    demo_dir = Path(spec["shared_demo_cache"]) / "expert_demos"
    demo_paths = sorted(demo_dir.glob("*.npz"))
    held_paths = [frozen_dir / name for name in split["holdout_files"]]
    print(
        f"[t1p] {len(demo_paths)} demo episodes, {len(held_paths)} held-out "
        f"collected episodes",
        flush=True,
    )

    demos = EpisodeStore(demo_paths, cfg.frame_stack, cfg.action_sequence)
    held = EpisodeStore(held_paths, cfg.frame_stack, cfg.action_sequence)

    rng = np.random.default_rng(args.seed)

    def sample_states(store, indices, count):
        pool = []
        for index in indices:
            for step in range(store.length(index)):
                pool.append((index, step))
        if not pool:
            return []
        pick = rng.choice(len(pool), size=min(count, len(pool)), replace=False)
        return [pool[int(i)] for i in sorted(pick)]

    held_success, held_fail, held_explore = [], [], []
    for index, path in enumerate(held_paths):
        record = manifest[path.name]
        if record["block"] != "greedy":
            held_explore.append(index)
        if record["success"]:
            held_success.append(index)
        else:
            held_fail.append(index)

    groups = {
        "demo": sample_states(demos, range(len(demo_paths)), args.demo_states),
        "heldout_success": sample_states(
            held, held_success, args.states_per_group
        ),
        "heldout_fail": sample_states(held, held_fail, args.states_per_group),
        "heldout_explore": sample_states(
            held, held_explore, args.states_per_group
        ),
    }
    group_store = {
        "demo": demos,
        "heldout_success": held,
        "heldout_fail": held,
        "heldout_explore": held,
    }
    for name, states in groups.items():
        print(f"[t1p] group {name}: {len(states)} states", flush=True)

    # Nearest-demo lookup table over normalized low-dim state.
    demo_low_dim, demo_ref = [], []
    for index in range(len(demo_paths)):
        episode = demos.episode(index)
        length = demos.length(index)
        demo_low_dim.append(episode["low_dim_state"][:length])
        demo_ref.extend((index, step) for step in range(length))
    demo_low_dim = np.concatenate(demo_low_dim, axis=0).astype(np.float32)
    demo_sq = (demo_low_dim**2).sum(axis=1)

    def nearest_demo(query):
        distances = demo_sq - 2.0 * (demo_low_dim @ query) + float(query @ query)
        best = int(np.argmin(distances))
        return demo_ref[best], float(np.sqrt(max(distances[best], 0.0)))

    support = agent.support
    levels = int(agent.levels)
    action_low = float(np.min(np.asarray(agent.action_low)))
    action_high = float(np.max(np.asarray(agent.action_high)))
    print(f"[t1p] action range [{action_low}, {action_high}]", flush=True)

    def features_fn(params, obs_inputs):
        return agent._rl_features(
            params.get("encoder", None), obs_inputs, stop_gradient=True
        )

    def stats_fn(params, obs_inputs, chunk):
        features = features_fn(params, obs_inputs)
        chosen_logits, all_logits = agent._critic_logits_per_level(
            params["critic"], features, chunk
        )
        all_q = jnp.sum(jax.nn.softmax(all_logits, axis=-1) * support, axis=-1)
        chosen_q = jnp.sum(
            jax.nn.softmax(chosen_logits, axis=-1) * support, axis=-1
        )
        flat = jnp.asarray(chunk, jnp.float32).reshape((features.shape[0], -1))
        bins = encode_action(
            flat, agent.action_low, agent.action_high, agent.levels, agent.bins
        )
        return all_q, chosen_q, bins

    def score_fn(params, obs_inputs, chunk):
        features = features_fn(params, obs_inputs)
        return agent._score_action_sequence_for_backup(
            params["critic"], features, chunk
        )

    stats_jit = jax.jit(stats_fn)
    score_jit = jax.jit(score_fn)

    def batched(states, store, chunks=None):
        """Yield (obs_inputs, chunk_array, slice) over mini-batches."""
        size = args.batch_size
        for start in range(0, len(states), size):
            window = states[start : start + size]
            obs_batch = {}
            keys = store.obs_keys(window[0][0])
            for key in keys:
                obs_batch[key] = np.stack(
                    [store.stacked_obs(i, t)[key] for i, t in window], axis=0
                )
            if chunks is None:
                chunk = np.stack([store.chunk(i, t) for i, t in window], axis=0)
            else:
                chunk = np.stack(chunks[start : start + size], axis=0)
            yield (
                agent._prepare_rl_obs_inputs(obs_batch),
                jnp.asarray(chunk, jnp.float32),
                slice(start, start + len(window)),
            )

    steps = (
        [int(s) for s in args.checkpoint_steps.split(",") if s]
        if args.checkpoint_steps
        else sorted(
            int(p.stem.split("_")[0])
            for p in (arm_dir / "eval_checkpoints").glob("*_checkpoint.pkl")
        )
    )
    print(f"[t1p] checkpoints: {steps}", flush=True)

    results = {"arm": spec["arm"], "seed": spec["seed"], "checkpoints": {}}

    # Precompute per-group candidate chunks once (identical for every arm).
    candidates = {}
    for name, states in groups.items():
        store = group_store[name]
        healthy, hijack, garbage, nearest, nearest_dist = [], [], [], [], []
        mc_targets, time_fraction = [], []
        for index, step in states:
            healthy.append(store.chunk(index, step))
            other = int(rng.integers(len(store.paths)))
            other_len = store.length(other)
            hijack.append(store.chunk(other, int(rng.integers(other_len))))
            garbage.append(
                rng.uniform(
                    action_low,
                    action_high,
                    size=(int(cfg.action_sequence), healthy[-1].shape[1]),
                ).astype(np.float32)
            )
            query = np.asarray(
                store.episode(index)["low_dim_state"][step], np.float32
            )
            (demo_index, demo_step), distance = nearest_demo(query)
            nearest.append(demos.chunk(demo_index, demo_step))
            nearest_dist.append(distance)
            episode = store.episode(index)
            rewards = episode["reward"][: store.length(index)]
            discounts = float(cfg.replay.gamma) ** np.arange(
                rewards.shape[0] - step
            )
            mc_targets.append(float((rewards[step:] * discounts).sum()))
            time_fraction.append(step / max(1, store.length(index) - 1))
        candidates[name] = {
            "healthy": healthy,
            "hijack": hijack,
            "garbage": garbage,
            "nearest_demo": nearest,
            "nearest_demo_distance": np.asarray(nearest_dist, np.float32),
            "mc_target": np.asarray(mc_targets, np.float32),
            "time_fraction": np.asarray(time_fraction, np.float32),
        }

    for step_value in steps:
        checkpoint = (
            arm_dir / "eval_checkpoints" / f"{step_value}_checkpoint.pkl"
        )
        workspace.load_snapshot(checkpoint, load_replay_buffer=False)
        params = agent.params
        entry = {}
        for name, states in groups.items():
            if not states:
                continue
            store = group_store[name]
            cand = candidates[name]
            n = len(states)
            agree_sum = np.zeros(levels)
            span_sum = np.zeros(levels)
            chosen_level_q = np.zeros(levels)
            healthy_score = np.zeros(n, np.float64)
            for obs_inputs, chunk, window in batched(
                states, store, cand["healthy"]
            ):
                all_q, chosen_q, bins = stats_jit(params, obs_inputs, chunk)
                all_q = np.asarray(all_q)
                chosen_q = np.asarray(chosen_q)
                bins = np.asarray(bins)
                agree_sum += (
                    (all_q.argmax(axis=-1) == bins).mean(axis=2).sum(axis=0)
                )
                span_sum += (
                    (all_q.max(axis=-1) - all_q.min(axis=-1))
                    .mean(axis=2)
                    .sum(axis=0)
                )
                chosen_level_q += chosen_q.mean(axis=2).sum(axis=0)
                healthy_score[window] = chosen_q[:, -1].mean(axis=1)
            scores = {"healthy": healthy_score}
            for label in ("hijack", "garbage", "nearest_demo"):
                values = np.zeros(n, np.float64)
                for obs_inputs, chunk, window in batched(
                    states, store, cand[label]
                ):
                    values[window] = np.asarray(
                        score_jit(params, obs_inputs, chunk)
                    )
                scores[label] = values
            healthy_top = (
                (scores["healthy"] > scores["hijack"])
                & (scores["healthy"] > scores["garbage"])
            ).mean()
            entry[name] = {
                "states": n,
                "argmax_agreement_per_level": (agree_sum / n).round(4).tolist(),
                "q_span_per_level": (span_sum / n).round(4).tolist(),
                "chosen_q_per_level": (
                    chosen_level_q / n
                ).round(4).tolist(),
                "q_healthy_mean": float(scores["healthy"].mean()),
                "q_hijack_mean": float(scores["hijack"].mean()),
                "q_garbage_mean": float(scores["garbage"].mean()),
                "q_nearest_demo_mean": float(scores["nearest_demo"].mean()),
                "healthy_top_of_three": float(healthy_top),
                "healthy_beats_hijack": float(
                    (scores["healthy"] > scores["hijack"]).mean()
                ),
                "healthy_beats_garbage": float(
                    (scores["healthy"] > scores["garbage"]).mean()
                ),
                "healthy_beats_nearest_demo": float(
                    (scores["healthy"] > scores["nearest_demo"]).mean()
                ),
                "q_minus_nearest_demo_mean": float(
                    (scores["healthy"] - scores["nearest_demo"]).mean()
                ),
                "nearest_demo_distance_mean": float(
                    cand["nearest_demo_distance"].mean()
                ),
                "mc_target_mean": float(cand["mc_target"].mean()),
                "q_minus_mc_mae": float(
                    np.abs(scores["healthy"] - cand["mc_target"]).mean()
                ),
                "q_mc_pearson": float(
                    np.corrcoef(scores["healthy"], cand["mc_target"])[0, 1]
                )
                if np.std(scores["healthy"]) > 1e-9
                else float("nan"),
                "_scores": {k: v.tolist() for k, v in scores.items()},
                "_time_fraction": cand["time_fraction"].tolist(),
            }
        if "heldout_success" in entry and "heldout_fail" in entry:
            entry["success_fail_auc"] = auc(
                entry["heldout_success"]["_scores"]["healthy"],
                entry["heldout_fail"]["_scores"]["healthy"],
            )
        results["checkpoints"][str(step_value)] = entry
        summary = {
            name: {
                "agree_L0": value["argmax_agreement_per_level"][0],
                "agree_Lmax": value["argmax_agreement_per_level"][-1],
                "span_Lmax": value["q_span_per_level"][-1],
                "q": round(value["q_healthy_mean"], 4),
                "top3": round(value["healthy_top_of_three"], 3),
            }
            for name, value in entry.items()
            if isinstance(value, dict)
        }
        print(
            f"[t1p] {spec['arm']} @ {step_value}: {json.dumps(summary)}",
            flush=True,
        )

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"[t1p] wrote {out}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
