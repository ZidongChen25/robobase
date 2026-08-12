#!/usr/bin/env python
"""Value-steered selection between heterogeneous IL policies (v2, sandwich arena).

Shared trajectory runs on the CANDIDATES' env convention (frame_stack 1,
lang_tokens, ActionSequence executing the full 20-step chunk per env.step).
Candidates (DP / FM sandwich checkpoints) consume the obs natively.

The CQN-AS judge never acts; it only scores. Adapters bridge its conventions:
  - frame stack: judge expects (4, ...) stacks; we feed 4 copies of the
    current obs -- the exact post-reset pattern its encoder sees at every
    episode start, so it is in-distribution.
  - low_dim normalization: candidates' env normalizes with THEIR demo stats
    (May, untruncated); the judge was trained on truncated-demo stats. We
    denormalize with the env's stats and renormalize with the judge's.
  - chunk length: judge scores the first `judge.action_sequence` (4) steps of
    each 20-step proposal.

Selectors: q (judge argmax), random, solo:NAME. (vote is degenerate with two
candidates.)

Run with --probe-only first: builds everything, prints layouts, asserts
compatibility, no episodes.
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
    p.add_argument("--judge", required=True, help="run_dir:snapshot_path")
    p.add_argument(
        "--candidates", nargs="+", required=True, help="name=run_dir:snapshot"
    )
    p.add_argument(
        "--env-from",
        required=True,
        help="candidate name whose workspace provides the shared env",
    )
    p.add_argument("--selectors", default="q,random,all-solos")
    p.add_argument("--num-eval-episodes", type=int, default=50)
    p.add_argument("--eval-seed-start", type=int, default=400)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--trace-json", default=None)
    p.add_argument("--probe-only", action="store_true")
    return p.parse_args()


def build_workspace(run_dir: Path, label: str):
    from omegaconf import OmegaConf
    from robobase.workspace import Workspace

    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    cfg.num_eval_envs = 1
    cfg.num_eval_episodes = 1
    cfg.log_train_video = False
    cfg.log_eval_video = False
    cfg.save_snapshot = False
    cfg.save_csv = False
    cfg.gpu_id = None
    cfg.wandb.use = False
    cfg.tb.use = False
    cfg.replay.num_workers = 0
    if "backend" in cfg:
        cfg.backend.replay_prefetch_size = 0
        cfg.backend.replay_device_prefetch = False
        cfg.backend.fused_update_steps = 1
        cfg.backend.update_block_every_steps = 1
    OmegaConf.resolve(cfg)
    work_dir = Path("exp_local/qselect_workspaces") / label
    work_dir.mkdir(parents=True, exist_ok=True)
    return Workspace(cfg, work_dir=str(work_dir))


def find_norm_stats(env):
    """Walk the wrapper chain for standardization stats keyed by obs name."""
    stats = {}
    node = env
    while node is not None:
        for attr in ("obs_stats", "_obs_stats"):
            s = getattr(node, attr, None)
            if isinstance(s, dict) and "mean" in s:
                for key in s["mean"]:
                    stats.setdefault(key, (s["mean"][key], s["std"][key]))
        node = getattr(node, "env", None)
    return stats


def spec(arg: str):
    run, snap = arg.rsplit(":", 1)
    return Path(run), Path(snap)


def main():
    args = parse_args()
    import jax
    import jax.numpy as jnp

    judge_run, judge_snap = spec(args.judge)
    print(f"[qs2] judge workspace: {judge_run}", flush=True)
    jw = build_workspace(judge_run, "qs2_judge")
    jw.load_snapshot(judge_snap, load_replay_buffer=False)
    judge = jw.agent

    cands = []
    for c in args.candidates:
        name, rest = c.split("=", 1)
        run, snap = spec(rest)
        print(f"[qs2] candidate {name}: {run}", flush=True)
        ws = build_workspace(run, f"qs2_{name}")
        ws.load_snapshot(snap, load_replay_buffer=False)
        cands.append((name, ws))
    names = [n for n, _ in cands]

    env_ws = dict(cands)[args.env_from]
    env_ws._ensure_eval_envs_created()
    env = env_ws.eval_env
    # Judge env exists only to expose its normalization stats and expected
    # observation shapes; never stepped.
    jw._ensure_eval_envs_created()
    judge_stats = find_norm_stats(jw.eval_env)
    judge_obs_shapes = {
        k: tuple(v.shape) for k, v in jw.eval_env.observation_space.items()
    }
    env_stats = find_norm_stats(env)
    jw._close_eval_envs()
    print(f"[qs2] judge expected obs shapes: {judge_obs_shapes}", flush=True)

    js = int(judge.action_sequence)
    jfs = int(jw.cfg.get("frame_stack", 1))
    print(f"[qs2] judge action_sequence={js} frame_stack={jfs}", flush=True)
    print(f"[qs2] shared env action space: {env.action_space}", flush=True)
    print(f"[qs2] shared env obs keys: {list(env.observation_space.keys())}", flush=True)
    print(f"[qs2] env低维stats keys: {sorted(env_stats)}", flush=True)
    print(f"[qs2] judge低维stats keys: {sorted(judge_stats)}", flush=True)

    # Renormalization: env stats -> raw -> judge stats. Separate obs keys are
    # bridged with their own stats; low_dim_state (a ConcatDim product) gets a
    # concatenated bridge assembled lazily once its width is known.
    shared_keys = sorted(k for k in env_stats if k in judge_stats)

    def make_key_bridge(em, es, jm, jsd):
        em, es = np.asarray(em).ravel(), np.asarray(es).ravel()
        jm, jsd = np.asarray(jm).ravel(), np.asarray(jsd).ravel()
        return lambda x: ((x * (es + 1e-10) + em) - jm) / (jsd + 1e-10)

    key_bridges = {
        k: make_key_bridge(*env_stats[k], *judge_stats[k]) for k in shared_keys
    }
    _lowdim_bridge_cache = {}

    def lowdim_bridge_for(width: int):
        if width in _lowdim_bridge_cache:
            return _lowdim_bridge_cache[width]
        # ConcatDim assembles low_dim_state from proprio keys in this order,
        # excluding whatever the config routed to separate keys. Find the
        # prefix combination matching the observed width.
        base_order = [
            "proprioception",
            "proprioception_grippers",
            "proprioception_floating_base",
            "proprioception_floating_base_actions",
        ]
        chosen = None
        for end in range(1, len(base_order) + 1):
            order = [k for k in base_order[:end] if k in shared_keys]
            w = sum(np.asarray(env_stats[k][0]).ravel().shape[0] for k in order)
            if w == width:
                chosen = order
                break
        if chosen is None:
            print(
                f"[qs2] WARN: no stats combo matches low_dim width {width}; "
                "passing through unbridged",
                flush=True,
            )
            _lowdim_bridge_cache[width] = lambda x: x
            return _lowdim_bridge_cache[width]
        em = np.concatenate([np.asarray(env_stats[k][0]).ravel() for k in chosen])
        es = np.concatenate([np.asarray(env_stats[k][1]).ravel() for k in chosen])
        jm = np.concatenate([np.asarray(judge_stats[k][0]).ravel() for k in chosen])
        jsd = np.concatenate(
            [np.asarray(judge_stats[k][1]).ravel() for k in chosen]
        )
        print(f"[qs2] low_dim bridge: {chosen} = {len(em)} dims", flush=True)
        _lowdim_bridge_cache[width] = make_key_bridge(em, es, jm, jsd)
        return _lowdim_bridge_cache[width]

    judge_keys = None  # resolved on first score call
    _rgb_layout_logged = set()

    def judge_obs_from(obs):
        """4-copy frame stack + normalization bridge, judge's keys only."""
        nonlocal judge_keys
        if judge_keys is None:
            judge_keys = [k for k in obs.keys() if k in judge_obs_shapes]
        out = {}
        for k in judge_keys:
            v = np.asarray(obs[k])
            # The candidate env's FrameStack already prepends a stack axis
            # (size 1); drop it (keep the newest frame) before re-stacking to
            # the judge's depth.
            if k in judge_obs_shapes and v.ndim == len(judge_obs_shapes[k]):
                v = v[-1]
            if k == "low_dim_state":
                v = np.asarray(v, dtype=np.float32)
                v = lowdim_bridge_for(v.shape[-1])(v)
            elif k in key_bridges:
                v = key_bridges[k](np.asarray(v, dtype=np.float32))
            if k.startswith("rgb") and k in judge_obs_shapes:
                # Judge trained at its own render resolution (e.g. 84x84);
                # the shared env renders larger. Area-downsample, layout-aware.
                import cv2

                th, tw = judge_obs_shapes[k][-2], judge_obs_shapes[k][-1]
                v = np.asarray(v)
                if k not in _rgb_layout_logged:
                    print(f"[qs2] {k} raw shape {v.shape} -> target ({th},{tw})", flush=True)
                    _rgb_layout_logged.add(k)
                if v.ndim == 3 and v.shape[-2:] != (th, tw):
                    if v.shape[0] in (1, 3, 4):  # CHW
                        img = np.moveaxis(v, 0, -1)
                        r = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
                        v = np.moveaxis(r, -1, 0)
                    elif v.shape[-1] in (1, 3, 4):  # HWC
                        v = cv2.resize(v, (tw, th), interpolation=cv2.INTER_AREA)
                    else:
                        raise ValueError(f"{k}: unrecognized rgb layout {v.shape}")
            stacked = np.stack([v] * jfs, axis=0)  # (stack, ...) oldest->newest
            out[k] = np.expand_dims(stacked, 0)  # batch of 1
        return out

    def judge_score_fn(params, obs_inputs, chunk):
        feats = judge._rl_features(
            params.get("encoder", None), obs_inputs, stop_gradient=True
        )
        return judge._score_action_sequence_for_backup(params["critic"], feats, chunk)

    judge_score = jax.jit(judge_score_fn)

    def score_chunks(obs, chunks):
        jobs = judge._prepare_rl_obs_inputs(judge_obs_from(obs))
        out = []
        for c in chunks:
            c4 = jnp.asarray(c[:, :js, :], dtype=jnp.float32)
            out.append(float(judge_score(judge.params, jobs, c4)[0]))
        return out

    if args.probe_only:
        obs, _ = env.reset(seed=args.eval_seed_start)
        chunks = []
        for name, ws in cands:
            b_obs = {k: np.expand_dims(v, 0) for k, v in obs.items()}
            a = ws.agent.act(b_obs, ws.main_loop_iterations, eval_mode=True)
            if isinstance(a, tuple):
                a = a[0]
            a = np.asarray(a)
            print(f"[qs2] {name} chunk shape: {a.shape}", flush=True)
            chunks.append(a)
        scores = score_chunks(obs, chunks)
        print(f"[qs2] probe scores: {dict(zip(names, scores))}", flush=True)
        print("[qs2] PROBE OK", flush=True)
        return 0

    selectors = []
    for s in args.selectors.split(","):
        s = s.strip()
        if s == "all-solos":
            selectors += [f"solo:{n}" for n in names]
        elif s:
            selectors.append(s)

    rng = np.random.default_rng(20260807)
    rows, trace = [], []
    for selector in selectors:
        successes = 0
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
            terminated = truncated = False
            while not (terminated or truncated):
                b_obs = {k: np.expand_dims(v, 0) for k, v in obs.items()}
                actions = []
                for cname, ws in cands:
                    a = ws.agent.act(
                        b_obs, ws.main_loop_iterations, eval_mode=True
                    )
                    if isinstance(a, tuple):
                        a = a[0]
                    a = np.asarray(a)
                    if not np.isfinite(a).all():
                        raise ValueError(
                            f"candidate {cname} produced non-finite actions "
                            f"(seed {seed})"
                        )
                    actions.append(a)
                if selector == "q":
                    scores = score_chunks(obs, actions)
                    pick = int(np.argmax(scores))
                elif selector == "random":
                    pick = int(rng.integers(len(cands)))
                    scores = None
                elif selector.startswith("solo:"):
                    pick = names.index(selector.split(":", 1)[1])
                    scores = None
                else:
                    raise ValueError(selector)
                if args.trace_json and selector == "q":
                    trace.append(
                        {"seed": seed, "pick": names[pick], "scores": scores}
                    )
                picks[names[pick]] += 1
                obs, reward, terminated, truncated, info = env.step(
                    actions[pick][0]
                )
            success = info.get("task_success")
            if success is not None:
                successes += int(np.array(success).astype(int).item())
        n = args.num_eval_episodes
        row = {
            "selector": selector,
            "episodes": n,
            "eval_seed_start": args.eval_seed_start,
            "success_rate": successes / n,
            "picks": json.dumps(dict(picks)),
            "elapsed_sec": round(time.time() - t0, 1),
        }
        rows.append(row)
        print(
            f"[qs2] {selector}: success {row['success_rate']:.1%} "
            f"picks {row['picks']} ({row['elapsed_sec']}s)",
            flush=True,
        )

    out = Path(args.output_csv)
    exists = out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(rows)
    if args.trace_json:
        Path(args.trace_json).write_text(json.dumps(trace))
    print(f"[qs2] wrote {len(rows)} rows -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
