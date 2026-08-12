#!/usr/bin/env python
"""Value-steered multi-policy inference (Q-select) harness.

Tests whether a CQN-AS critic's value carries usable return information: N
candidate policies each propose their deployment-form action at every step of
a SHARED trajectory; a selector picks which one the env executes.

Selectors:
  q         judge critic scores each candidate's fresh greedy chunk
            (agent._last_plan_chunk) with _score_action_sequence_for_backup
            (deepest-level mean expected C51 value); execute the argmax
            candidate's post-ensemble action.
  vote      execute the candidate whose action is closest to the mean of all
            candidate actions (pure consensus -- no value information).
            Control for the ensemble effect.
  random    uniform-random candidate per step (ensemble floor).
  solo:NAME always execute candidate NAME (equivalence check vs direct eval).

Verdict logic: q > vote and q > best solo  => value adds information beyond
consensus. q ~= vote => the critic is a demo-frequency/consensus proxy.

v1 limitation: all candidates must share the judge's obs conventions
(frame_stack, norm_obs) -- i.e. CQN-AS checkpoints. IL candidates (ACT/DP/FM,
frame_stack 1, action standardization) need the v2 adapter layer.
"""
import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--judge",
        required=True,
        help="run_dir:snapshot_path of the judging critic",
    )
    p.add_argument(
        "--candidates",
        nargs="+",
        required=True,
        help="name=run_dir:snapshot_path per candidate policy",
    )
    p.add_argument(
        "--selectors",
        default="q,vote,random",
        help="Comma list: q,vote,random,solo:NAME,all-solos",
    )
    p.add_argument("--num-eval-episodes", type=int, default=50)
    p.add_argument("--eval-seed-start", type=int, default=400)
    p.add_argument("--output-csv", required=True)
    p.add_argument(
        "--trace-json",
        default=None,
        help="Optional per-step pick/score trace output",
    )
    return p.parse_args()


def build_workspace(run_dir: Path, label: str):
    from omegaconf import OmegaConf
    from robobase.workspace import Workspace

    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    cfg.num_eval_envs = 1  # single (non-vector) eval env path
    cfg.num_eval_episodes = 1
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
    work_dir = Path("exp_local/qselect_workspaces") / label
    work_dir.mkdir(parents=True, exist_ok=True)
    return Workspace(cfg, work_dir=str(work_dir))


def spec(arg: str):
    run, snap = arg.rsplit(":", 1)
    return Path(run), Path(snap)


def main():
    args = parse_args()
    import jax
    import jax.numpy as jnp

    judge_run, judge_snap = spec(args.judge)
    cand_specs = []
    for c in args.candidates:
        name, rest = c.split("=", 1)
        cand_specs.append((name, *spec(rest)))

    print(f"[qselect] building judge workspace: {judge_run}", flush=True)
    jw = build_workspace(judge_run, "judge")
    jw.load_snapshot(judge_snap, load_replay_buffer=False)
    judge = jw.agent

    cands = []
    for name, run, snap in cand_specs:
        print(f"[qselect] building candidate {name}: {run}", flush=True)
        ws = build_workspace(run, f"cand_{name}")
        ws.load_snapshot(snap, load_replay_buffer=False)
        cands.append((name, ws))
    names = [n for n, _ in cands]

    def judge_score_fn(params, obs_inputs, chunk):
        feats = judge._rl_features(
            params.get("encoder", None), obs_inputs, stop_gradient=True
        )
        return judge._score_action_sequence_for_backup(params["critic"], feats, chunk)

    judge_score = jax.jit(judge_score_fn)

    selectors = []
    for s in args.selectors.split(","):
        s = s.strip()
        if s == "all-solos":
            selectors += [f"solo:{n}" for n in names]
        elif s:
            selectors.append(s)

    # Shared eval env from the judge's config.
    jw._ensure_eval_envs_created()
    env = jw.eval_env
    assert env is not None, "expected single (non-vector) eval env"

    rng = np.random.default_rng(12345)
    out_rows = []
    trace = []
    for selector in selectors:
        successes, ep_rewards = 0, []
        picks = Counter()
        t0 = time.time()
        for ep in range(args.num_eval_episodes):
            seed = args.eval_seed_start + ep
            obs, info = env.reset(seed=seed)
            for _, ws in cands:
                ws.agent.reset(
                    ws.main_loop_iterations,
                    [getattr(ws, "_eval_agent_indices", [0])[-1]],
                )
            judge.reset(
                jw.main_loop_iterations,
                [getattr(jw, "_eval_agent_indices", [0])[-1]],
            )
            terminated = truncated = False
            total_reward = 0.0
            while not (terminated or truncated):
                b_obs = {k: np.expand_dims(v, 0) for k, v in obs.items()}
                actions, chunks = [], []
                for _, ws in cands:
                    a = ws.agent.act(
                        b_obs, ws.main_loop_iterations, eval_mode=True
                    )
                    if isinstance(a, tuple):
                        a = a[0]
                    actions.append(np.asarray(a))
                    chunks.append(
                        np.asarray(ws.agent._last_plan_chunk, dtype=np.float32)
                    )
                if selector == "q":
                    obs_inputs = judge._prepare_rl_obs_inputs(b_obs)
                    scores = [
                        float(
                            judge_score(
                                judge.params, obs_inputs, jnp.asarray(c)
                            )[0]
                        )
                        for c in chunks
                    ]
                    pick = int(np.argmax(scores))
                elif selector == "vote":
                    ex = np.stack([a[0, 0] for a in actions])
                    dist = np.linalg.norm(ex - ex.mean(axis=0), axis=-1)
                    pick = int(np.argmin(dist))
                    scores = None
                elif selector == "random":
                    pick = int(rng.integers(len(cands)))
                    scores = None
                elif selector.startswith("solo:"):
                    pick = names.index(selector.split(":", 1)[1])
                    scores = None
                else:
                    raise ValueError(f"unknown selector {selector}")
                if args.trace_json and selector == "q":
                    trace.append(
                        {"seed": seed, "pick": names[pick], "scores": scores}
                    )
                picks[names[pick]] += 1
                obs, reward, terminated, truncated, info = env.step(
                    actions[pick][0]
                )
                total_reward += float(np.asarray(reward).item())
            success = info.get("task_success")
            if success is not None:
                successes += int(np.array(success).astype(int).item())
            ep_rewards.append(total_reward)
        n = args.num_eval_episodes
        row = {
            "selector": selector,
            "episodes": n,
            "eval_seed_start": args.eval_seed_start,
            "success_rate": successes / n,
            "mean_reward": float(np.mean(ep_rewards)),
            "picks": json.dumps(dict(picks)),
            "elapsed_sec": round(time.time() - t0, 1),
        }
        out_rows.append(row)
        print(
            f"[qselect] {selector}: success {row['success_rate']:.1%} "
            f"picks {row['picks']} ({row['elapsed_sec']}s)",
            flush=True,
        )

    out = Path(args.output_csv)
    exists = out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(out_rows)
    if args.trace_json:
        Path(args.trace_json).write_text(json.dumps(trace))
    print(f"[qselect] wrote {len(out_rows)} rows -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
