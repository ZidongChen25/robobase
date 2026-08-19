#!/usr/bin/env python
"""Stream-transplant follow-up to the dueling autopsy (2026-08-18).

Question left open by dueling_stream_autopsy.py: post-collapse the advantage
stream retains a residual bin-spread (~0.2 logits) which is still ~20x the
combined Q-span. Does the flatness live in the A stream or in the combination?

Test: parameter-level transplant between seed1 pre-collapse (120000) and
post-collapse (142732) checkpoints. Every critic param is prefixed
"advantage_*" or "value_*" (fully separate streams), so we can swap one stream
wholesale. Encoder params travel with their donor to keep features coherent;
an encoder-only swap control quantifies that confound.

Also computes, for the pure checkpoints, the first-order "effective A span":
    Q_bin ~= EV_V + Cov_{p_V}(support, A_bin),  p_V = softmax(value logits)
    eff_a_span = span_bins(Cov_{p_V}(support, A_bin))
i.e. how much of the A stream's structure is visible under the measure the
combined distribution actually uses.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path("/home/zc1525/robobase_jaxflat")
SEED1 = ROOT / "exp_local/cqn_trunc_arms/wsrl_sandwich/seed1_20260818wsrlB"
PRE_STEP, POST_STEP = 120000, 142732
OUT_JSON = ROOT / "reports/dueling_astream_autopsy_20260818_transplant.json"

spec = importlib.util.spec_from_file_location(
    "autopsy", ROOT / "scripts/dueling_stream_autopsy.py"
)
autopsy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(autopsy)


def main():
    import jax
    import jax.numpy as jnp
    from omegaconf import OmegaConf
    from robobase.workspace import Workspace
    from robobase.method.cqn_research import encode_action, zoom_in

    cfg = OmegaConf.load(SEED1 / ".hydra" / "config.yaml")
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

    work_dir = Path(os.environ.get("PROBE_WORK_DIR", "/tmp")) / "transplant_ws"
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

    # Same rng seed and order as the autopsy -> identical probe states.
    rng = np.random.default_rng(autopsy.SEED)
    split = json.loads(autopsy.SPLIT_FILE.read_text())
    held_paths = [autopsy.FROZEN / n for n in split["holdout_files"]]
    demo_paths = sorted(autopsy.DEMO_DIR.glob("*.npz"))
    groups = {}
    for name, paths in (("demo", demo_paths), ("heldout", held_paths)):
        obs, chunks = autopsy.sample_group(paths, frame_stack, action_seq, rng)
        groups[name] = (obs, chunks)
        print(f"[transplant] group {name}: {chunks.shape[0]} states", flush=True)

    def snapshot_params(step):
        workspace.load_snapshot(
            SEED1 / "eval_checkpoints" / f"{step}_checkpoint.pkl",
            load_replay_buffer=False,
        )
        return jax.tree.map(jnp.array, agent.params)

    pre = snapshot_params(PRE_STEP)
    post = snapshot_params(POST_STEP)

    def mix_critic(a_src, v_src):
        tree = {}
        for key in a_src["critic"]["params"]:
            donor = a_src if key.startswith("advantage") else v_src
            tree[key] = donor["critic"]["params"][key]
        return {"params": tree}

    configs = {
        "pure_pre": (pre, pre["critic"], "pre"),
        "pure_post": (post, post["critic"], "post"),
        "preV_postA": (pre, mix_critic(post, pre), "pre"),
        "postV_preA": (post, mix_critic(pre, post), "post"),
        # encoder-swap control: pre critic entirely, post encoder features
        "postEnc_preCritic": (post, pre["critic"], "post"),
    }

    def stream_metrics(enc_params, critic_params, obs_inputs, chunk):
        features = agent._rl_features(enc_params, obs_inputs, stop_gradient=True)
        n = features.shape[0]
        flat = jnp.asarray(chunk, jnp.float32).reshape((n, -1))
        discrete = encode_action(
            flat, agent.action_low, agent.action_high, levels, bins
        )
        low = jnp.broadcast_to(agent.action_low, flat.shape)
        high = jnp.broadcast_to(agent.action_high, flat.shape)
        out = {}
        for level in range(2):
            one_hot = jnp.broadcast_to(
                jax.nn.one_hot(level, levels, dtype=jnp.float32), (n, levels)
            )
            midpoint = (0.5 * (low + high)).reshape((n, seq, adim))
            combined, values, cadv = agent.critic_model.apply(
                critic_params, features, one_hot, midpoint, return_streams=True
            )
            q = jnp.sum(jax.nn.softmax(combined, axis=-1) * support, axis=-1)
            index = discrete[:, level, :].reshape((n, seq, adim))
            chosen_q = jnp.take_along_axis(q, index[..., None], axis=-1)[..., 0]
            # first-order effective A span under the V measure
            p_v = jax.nn.softmax(values, axis=-1)  # [n,seq,adim,1,atoms]
            ev_v = jnp.sum(p_v * support, axis=-1)  # [n,seq,adim,1]
            cov = jnp.sum(
                p_v * (support - ev_v[..., None]) * cadv, axis=-1
            )  # [n,seq,adim,bins]
            prefix = f"L{level}"
            out[f"{prefix}/q_span"] = (q.max(-1) - q.min(-1)).mean((1, 2))
            out[f"{prefix}/chosen_q"] = chosen_q.mean((1, 2))
            out[f"{prefix}/eff_a_span"] = (cov.max(-1) - cov.min(-1)).mean(
                (1, 2)
            )
            out[f"{prefix}/a_logit_span"] = (
                cadv.max(-2) - cadv.min(-2)
            ).mean((1, 2, 3))
            low, high = zoom_in(
                low, high, discrete[:, level, :], bins,
                agent.action_low, agent.action_high,
            )
        return out

    metrics_jit = jax.jit(stream_metrics)

    results = {}
    for config_name, (enc_donor, critic_tree, enc_label) in configs.items():
        enc_params = enc_donor.get("encoder", None)
        entry = {"encoder": enc_label}
        for group_name, (obs, chunks) in groups.items():
            total = chunks.shape[0]
            acc: dict[str, list] = {}
            for start in range(0, total, autopsy.BATCH):
                sl = slice(start, start + autopsy.BATCH)
                obs_inputs = agent._prepare_rl_obs_inputs(
                    {k: v[sl] for k, v in obs.items()}
                )
                out = metrics_jit(
                    enc_params, critic_tree, obs_inputs,
                    jnp.asarray(chunks[sl]),
                )
                for key, value in out.items():
                    acc.setdefault(key, []).append(np.asarray(value))
            entry[group_name] = {
                key: {
                    "mean": float(np.concatenate(v).mean()),
                    "std": float(np.concatenate(v).std()),
                }
                for key, v in acc.items()
            }
        results[config_name] = entry
        d = entry["demo"]
        print(
            f"[transplant] {config_name} (enc={enc_label}) demo: "
            f"L0 q_span={d['L0/q_span']['mean']:.4f} "
            f"eff_a_span={d['L0/eff_a_span']['mean']:.4f} "
            f"a_logit_span={d['L0/a_logit_span']['mean']:.4f} "
            f"chosen_q={d['L0/chosen_q']['mean']:.4f} | "
            f"L1 q_span={d['L1/q_span']['mean']:.4f}",
            flush=True,
        )

    OUT_JSON.write_text(json.dumps(results, indent=2) + "\n")
    print(f"[transplant] wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
