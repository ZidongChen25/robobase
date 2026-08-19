#!/usr/bin/env python
"""Dueling-stream autopsy for the CQN-AS unanchoring collapse (2026-08-18).

Hypothesis: the span collapse (Q-span over the 5 sibling bins ~0.76 -> ~0.02
while mean Q rises) is the ADVANTAGE stream dying while the VALUE stream stays
alive -- the dueling "lazy path".

The critic (C2FSequenceDistributionalCritic, robobase/method/cqn_as.py) has
two fully separate recurrent streams ("advantage_*" and "value_*" params) and
combines them at the LOGIT (per-atom) level, before the softmax over atoms:

    centered_advantages = advantages - advantages.mean(axis=-2)   # bins axis
    combined = values + centered_advantages                       # ~line 1064

so bin-spread of the advantage logits is identical before/after the mean
subtraction (the mean is constant across bins), while the raw pre-centering
offset is captured separately via flax capture_intermediates on the two heads.

For each checkpoint x state group we forward the recorded action chunk
(teacher-forced zoom path, same as every span metric in this repo) and report
per level:
    a_logit_span / a_logit_std   spread of advantage-head logits across bins
    a_raw_mean / a_raw_absmean   offset of the raw (pre-centering) A logits
    qa_span                      span of E[softmax(centered A logits) * support]
    v_expected                   E[softmax(value logits) * support]
    v_logit_absmean / v_logit_std  magnitude/shape of the value logits
    q_span / chosen_q            full combined-stream readout (sanity)
plus per-checkpoint L2 norms of the advantage_* vs value_* parameter groups.

No env, no rollouts; probe states are the frozen T1 dataset (demo + held-out).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path("/home/zc1525/robobase_jaxflat")
WSRL = ROOT / "exp_local/cqn_trunc_arms/wsrl_sandwich"
RUNS = {
    "armA_seed1": (WSRL / "seed1_20260818wsrlB", [120000, 125000, 142732]),
    "armA_seed2": (WSRL / "seed2_20260818wsrlB", [120000, 125000, 141466]),
    "ctrl_lam1": (WSRL / "ctrl_lam1_20260818wsrlB", [120000, 125000, 142900]),
}
FROZEN = ROOT / "exp_local/t1_td_mc/frozen_data"
SPLIT_FILE = ROOT / "exp_local/t1_td_mc/arms/td_s0/t1_split.json"
DEMO_DIR = (
    ROOT / "exp_local/_shared_demo_replay/c1afbe4b906976b8bfb97f2c/expert_demos"
)
OUT_JSON = ROOT / "reports/dueling_astream_autopsy_20260818.json"

OBS_EXCLUDE = {"action", "reward", "terminal", "truncated", "demo", "mc_return"}
EPISODES_PER_GROUP = 16
STATES_PER_EPISODE = 8
BATCH = 32
SEED = 17


def sample_group(paths, frame_stack, action_seq, rng):
    """Return stacked-obs arrays and recorded chunks for sampled states."""
    from robobase.replay_buffer.uniform_replay_buffer import load_episode

    picked = sorted(
        rng.choice(len(paths), size=min(EPISODES_PER_GROUP, len(paths)),
                   replace=False).tolist()
    )
    obs_lists: dict[str, list] = {}
    chunks = []
    for episode_index in picked:
        episode = load_episode(Path(paths[episode_index]))
        ep_len = episode["action"].shape[0] - 1
        keys = [k for k in episode if k not in OBS_EXCLUDE]
        steps = sorted(
            rng.choice(ep_len, size=min(STATES_PER_EPISODE, ep_len),
                       replace=False).tolist()
        )
        for step in steps:
            idxs = np.clip(
                np.arange(step - frame_stack + 1, step + 1), 0, ep_len
            )
            for key in keys:
                obs_lists.setdefault(key, []).append(episode[key][idxs])
            stop = min(step + action_seq, ep_len)
            actions = episode["action"][step:stop]
            if actions.shape[0] < action_seq:
                pad = action_seq - actions.shape[0]
                actions = np.concatenate(
                    [actions, np.repeat(actions[-1:], pad, axis=0)], axis=0
                )
            chunks.append(actions.astype(np.float32))
        del episode  # keep host RAM flat
    obs = {k: np.stack(v, axis=0) for k, v in obs_lists.items()}
    return obs, np.stack(chunks, axis=0)


def main():
    import jax
    import jax.numpy as jnp
    from omegaconf import OmegaConf
    from robobase.workspace import Workspace
    from robobase.method.cqn_research import encode_action, zoom_in

    cfg = OmegaConf.load(RUNS["armA_seed1"][0] / ".hydra" / "config.yaml")
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
    # probe_bc_anchor-style eval-shaped workspace (validated on these run
    # dirs): demos stay configured so _wrap_env is happy, but nothing is
    # persisted and _load_demos() is never called.
    cfg.log_train_video = False
    cfg.log_eval_video = False
    cfg.replay.num_workers = 0
    cfg.replay.save_dir = None
    cfg.replay.persist = False
    cfg.replay.reuse_saved = False
    cfg.replay.demo_only_updates = False
    cfg.use_self_imitation = False
    cfg.replay.size = 10000
    cfg.replay.demo_size = 10000
    cfg.backend.replay_prefetch_size = 0
    cfg.backend.replay_device_prefetch = False
    OmegaConf.resolve(cfg)

    work_dir = Path(os.environ.get("PROBE_WORK_DIR", "/tmp")) / "autopsy_ws"
    work_dir.mkdir(parents=True, exist_ok=True)
    workspace = Workspace(cfg, work_dir=str(work_dir))
    agent = workspace.agent

    frame_stack = int(cfg.frame_stack)
    action_seq = int(cfg.action_sequence)
    levels = int(agent.levels)
    bins = int(agent.bins)
    atoms = int(agent.atoms)
    seq = int(agent.action_sequence)
    adim = int(agent.action_dim)
    support = agent.support

    rng = np.random.default_rng(SEED)
    split = json.loads(SPLIT_FILE.read_text())
    held_paths = [FROZEN / name for name in split["holdout_files"]]
    demo_paths = sorted(DEMO_DIR.glob("*.npz"))
    print(f"[autopsy] {len(demo_paths)} demo eps, {len(held_paths)} held-out eps",
          flush=True)

    groups = {}
    for name, paths in (("demo", demo_paths), ("heldout", held_paths)):
        obs, chunks = sample_group(paths, frame_stack, action_seq, rng)
        groups[name] = (obs, chunks)
        print(f"[autopsy] group {name}: {chunks.shape[0]} states", flush=True)

    def head_filter(module, method):
        return module.name in ("advantage_head", "value_head")

    def stream_metrics(params, obs_inputs, chunk):
        features = agent._rl_features(
            params.get("encoder", None), obs_inputs, stop_gradient=True
        )
        n = features.shape[0]
        flat = jnp.asarray(chunk, jnp.float32).reshape((n, -1))
        discrete = encode_action(
            flat, agent.action_low, agent.action_high, levels, bins
        )
        low = jnp.broadcast_to(agent.action_low, flat.shape)
        high = jnp.broadcast_to(agent.action_high, flat.shape)
        out = {}
        for level in range(levels):
            one_hot = jnp.broadcast_to(
                jax.nn.one_hot(level, levels, dtype=jnp.float32), (n, levels)
            )
            midpoint = (0.5 * (low + high)).reshape((n, seq, adim))
            (combined, values, cadv), inter = agent.critic_model.apply(
                params["critic"],
                features,
                one_hot,
                midpoint,
                return_streams=True,
                capture_intermediates=head_filter,
                mutable=["intermediates"],
            )
            raw_adv = inter["intermediates"]["advantage_head"]["__call__"][
                0
            ].reshape((n, seq, adim, bins, atoms))
            # combined [n,seq,adim,bins,atoms]; values [n,seq,adim,1,atoms]
            q = jnp.sum(jax.nn.softmax(combined, axis=-1) * support, axis=-1)
            index = discrete[:, level, :].reshape((n, seq, adim))
            chosen_q = jnp.take_along_axis(q, index[..., None], axis=-1)[..., 0]
            v_logits = values[..., 0, :]
            qa = jnp.sum(jax.nn.softmax(cadv, axis=-1) * support, axis=-1)
            qa_raw = jnp.sum(
                jax.nn.softmax(raw_adv, axis=-1) * support, axis=-1
            )
            prefix = f"L{level}"
            out[f"{prefix}/q_span"] = (q.max(-1) - q.min(-1)).mean((1, 2))
            out[f"{prefix}/chosen_q"] = chosen_q.mean((1, 2))
            out[f"{prefix}/a_logit_span"] = (
                raw_adv.max(-2) - raw_adv.min(-2)
            ).mean((1, 2, 3))
            out[f"{prefix}/a_logit_std"] = raw_adv.std(-2).mean((1, 2, 3))
            out[f"{prefix}/a_centered_span"] = (
                cadv.max(-2) - cadv.min(-2)
            ).mean((1, 2, 3))
            out[f"{prefix}/a_raw_mean"] = raw_adv.mean((1, 2, 3, 4))
            out[f"{prefix}/a_raw_absmean"] = jnp.abs(raw_adv).mean((1, 2, 3, 4))
            out[f"{prefix}/qa_span"] = (qa.max(-1) - qa.min(-1)).mean((1, 2))
            out[f"{prefix}/qa_raw_span"] = (
                qa_raw.max(-1) - qa_raw.min(-1)
            ).mean((1, 2))
            out[f"{prefix}/v_expected"] = jnp.sum(
                jax.nn.softmax(v_logits, axis=-1) * support, axis=-1
            ).mean((1, 2))
            out[f"{prefix}/v_logit_absmean"] = jnp.abs(v_logits).mean((1, 2, 3))
            out[f"{prefix}/v_logit_std"] = v_logits.std(-1).mean((1, 2))
            low, high = zoom_in(
                low, high, discrete[:, level, :], bins,
                agent.action_low, agent.action_high,
            )
        return out

    metrics_jit = jax.jit(stream_metrics)

    def group_norms(tree):
        flat = {}
        for key, sub in tree.items():
            leaves = jax.tree_util.tree_leaves(sub)
            flat[key] = float(
                jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))
            )
        return flat

    results = {}
    for run_name, (run_dir, steps) in RUNS.items():
        for step in steps:
            checkpoint = run_dir / "eval_checkpoints" / f"{step}_checkpoint.pkl"
            workspace.load_snapshot(checkpoint, load_replay_buffer=False)
            params = agent.params
            entry = {"param_norms": {}}
            critic_tree = params["critic"]["params"]
            norms = group_norms(critic_tree)
            entry["param_norms"] = {
                "advantage_head": norms.get("advantage_head"),
                "value_head": norms.get("value_head"),
                "advantage_stack_total": float(
                    np.sqrt(sum(v**2 for k, v in norms.items()
                                if k.startswith("advantage")))
                ),
                "value_stack_total": float(
                    np.sqrt(sum(v**2 for k, v in norms.items()
                                if k.startswith("value")))
                ),
            }
            for group_name, (obs, chunks) in groups.items():
                total = chunks.shape[0]
                acc: dict[str, list] = {}
                for start in range(0, total, BATCH):
                    sl = slice(start, start + BATCH)
                    obs_batch = {k: v[sl] for k, v in obs.items()}
                    obs_inputs = agent._prepare_rl_obs_inputs(obs_batch)
                    out = metrics_jit(
                        params, obs_inputs, jnp.asarray(chunks[sl])
                    )
                    for key, value in out.items():
                        acc.setdefault(key, []).append(np.asarray(value))
                summary = {}
                for key, chunks_list in acc.items():
                    values = np.concatenate(chunks_list)
                    summary[key] = {
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                    }
                # all-level mean q_span for comparison with sibling_span probes
                summary["q_span_all_levels"] = float(
                    np.mean([summary[f"L{lv}/q_span"]["mean"]
                             for lv in range(levels)])
                )
                summary["chosen_q_all_levels"] = float(
                    np.mean([summary[f"L{lv}/chosen_q"]["mean"]
                             for lv in range(levels)])
                )
                entry[group_name] = summary
            results.setdefault(run_name, {})[str(step)] = entry
            demo_l0 = entry["demo"]
            print(
                f"[autopsy] {run_name}@{step} demo L0: "
                f"q_span={demo_l0['L0/q_span']['mean']:.4f} "
                f"a_logit_span={demo_l0['L0/a_logit_span']['mean']:.4f} "
                f"qa_span={demo_l0['L0/qa_span']['mean']:.4f} "
                f"v={demo_l0['L0/v_expected']['mean']:.4f} "
                f"chosen_q={demo_l0['L0/chosen_q']['mean']:.4f} "
                f"|A_head|={entry['param_norms']['advantage_head']:.3f} "
                f"|V_head|={entry['param_norms']['value_head']:.3f}",
                flush=True,
            )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2) + "\n")
    print(f"[autopsy] wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
