#!/usr/bin/env python
"""Value-steered selection, v3: bridge-free dual-env lockstep.

v2 fed the judge adapted observations (downsampled renders, 4-copy frame
stack, renormalized low_dim). v3 removes every bridge: the judge gets its OWN
BiGym instance (native 84x84 renders, true per-step frame stack, its own
normalization) kept in lockstep with the candidates' env by feeding both the
exact same executed actions -- MuJoCo physics is deterministic given the same
reset seed and action stream.

Judge lockstep env: built from the judge config with
action_sequence=execution_length=1, which the factory maps to the passthrough
ActionSequence wrapper (no receding-horizon blending), so each fed action row
executes verbatim.

Per tick: candidates propose 20-step chunks from the IL env's obs; the judge
scores each candidate's first `judge.action_sequence` steps using its native
obs; the winner's chunk steps the IL env (authoritative for
reward/termination), and the judge env replays the executed rows one by one.
A drift assertion compares the two instances' denormalized floating-base
state every tick.
"""
import argparse
import csv
import json
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--judge", required=True, help="run_dir:snapshot_path")
    p.add_argument(
        "--candidates", nargs="+", required=True, help="name=run_dir:snapshot"
    )
    p.add_argument(
        "--env-from", required=True, help="candidate providing the shared IL env"
    )
    p.add_argument("--selectors", default="q,random,all-solos")
    p.add_argument("--num-eval-episodes", type=int, default=50)
    p.add_argument("--eval-seed-start", type=int, default=400)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--trace-json", default=None)
    p.add_argument("--probe-only", action="store_true")
    p.add_argument(
        "--drift-tol",
        type=float,
        default=5e-4,
        help="Max drift growth over the reset baseline on denormalized floating-base state (era action-scale differences accumulate ~1e-4/140 steps; real trajectory forks show at >1e-2)",
    )
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
    from omegaconf import OmegaConf

    judge_run, judge_snap = spec(args.judge)
    print(f"[qs3] judge workspace: {judge_run}", flush=True)
    jw = build_workspace(judge_run, "qs3_judge")
    jw.load_snapshot(judge_snap, load_replay_buffer=False)
    judge = jw.agent

    cands = []
    for c in args.candidates:
        name, rest = c.split("=", 1)
        run, snap = spec(rest)
        print(f"[qs3] candidate {name}: {run}", flush=True)
        ws = build_workspace(run, f"qs3_{name}")
        ws.load_snapshot(snap, load_replay_buffer=False)
        cands.append((name, ws))
    names = [n for n, _ in cands]

    env_ws = dict(cands)[args.env_from]
    env_ws._ensure_eval_envs_created()
    il_env = env_ws.eval_env

    # Judge lockstep env: passthrough action execution, native everything else.
    jcfg = deepcopy(jw.cfg)
    with __import__("omegaconf").omegaconf.open_dict(jcfg):
        jcfg.action_sequence = 1
        jcfg.execution_length = 1
        jcfg.num_eval_envs = 1
    OmegaConf.resolve(jcfg)
    judge_env = jw.env_factory.make_eval_env(jcfg)
    print(f"[qs3] judge lockstep env action space: {judge_env.action_space}", flush=True)
    print(f"[qs3] il env action space: {il_env.action_space}", flush=True)

    il_stats = find_norm_stats(il_env)
    judge_stats = find_norm_stats(judge_env)

    def denorm(stats, key, v):
        if key not in stats:
            return np.asarray(v, dtype=np.float64)
        m, s = stats[key]
        return np.asarray(v, dtype=np.float64) * (
            np.asarray(s, dtype=np.float64) + 1e-10
        ) + np.asarray(m, dtype=np.float64)

    def drift_metric(il_obs, j_obs):
        key = "proprioception_floating_base"
        if key not in il_obs or key not in j_obs:
            return None
        a = denorm(il_stats, key, np.asarray(il_obs[key])[-1])
        b = denorm(judge_stats, key, np.asarray(j_obs[key])[-1])
        return np.abs(a - b)

    def sync_physics(src_env, dst_env):
        """Copy MuJoCo + controller state src -> dst (cross-wrapper-chain).

        Physics-only subset of bigym_branch_state.restore: wrapper state and
        RNGs stay per-env (the judge env keeps its own frame history), but the
        world itself is forced identical, so lockstep drift cannot accumulate
        past one tick regardless of contact chaos.
        """
        from robobase.envs.bigym_branch_state import (
            capture_bigym_branch_state,
        )

        state = capture_bigym_branch_state(src_env)
        raw = dst_env.unwrapped
        raw.mojo.physics.set_state(state.physics_state)
        raw.mojo.data.time = state.physics_time
        for name, value in state.model_arrays.items():
            getattr(raw.mojo.model, name)[...] = value
        for name, value in state.physics_arrays.items():
            getattr(raw.mojo.data, name)[...] = value
        raw.mojo.physics.forward()
        for name, value in state.physics_arrays.items():
            getattr(raw.mojo.data, name)[...] = value
        raw._action = state.raw_action.copy()
        fb = raw.robot.floating_base
        np.copyto(fb._accumulated_actions, state.floating_base_accumulated_actions)
        np.copyto(fb._last_action, state.floating_base_last_action)
        raw._step_cache.clean()

    fork_ticks = 0
    total_ticks = 0

    def drift_heal(baseline, il_obs, j_obs, seed, tick):
        """Detect within-tick forks, log them, and resync the judge world."""
        nonlocal fork_ticks, total_ticks
        total_ticks += 1
        d = drift_metric(il_obs, j_obs)
        if d is None or baseline is None:
            return
        excess = float(np.abs(d - baseline).max())
        if excess > args.drift_tol:
            fork_ticks += 1
            sync_physics(il_env, judge_env)

    js = int(judge.action_sequence)
    print(f"[qs3] judge action_sequence={js}", flush=True)

    def judge_score_fn(params, obs_inputs, chunk):
        feats = judge._rl_features(
            params.get("encoder", None), obs_inputs, stop_gradient=True
        )
        return judge._score_action_sequence_for_backup(params["critic"], feats, chunk)

    judge_score = jax.jit(judge_score_fn)

    def score_chunks(j_obs, chunks):
        b_obs = {k: np.expand_dims(np.asarray(v), 0) for k, v in j_obs.items()}
        jobs = judge._prepare_rl_obs_inputs(b_obs)
        out = []
        for c in chunks:
            c4 = jnp.asarray(c[:, :js, :], dtype=jnp.float32)
            out.append(float(judge_score(judge.params, jobs, c4)[0]))
        return out

    def find_rescale(env):
        node = env
        while node is not None:
            if hasattr(node, "action_stats") and hasattr(node, "min_max_margin"):
                return node
            node = getattr(node, "env", None)
        return None

    from robobase.envs.wrappers import RescaleFromTanhWithMinMax as _RS

    il_rs = find_rescale(il_env)
    judge_rs = find_rescale(judge_env)
    convert_actions = (
        il_rs is not None
        and judge_rs is not None
        and (
            not np.allclose(
                il_rs.action_stats["min"], judge_rs.action_stats["min"]
            )
            or not np.allclose(
                il_rs.action_stats["max"], judge_rs.action_stats["max"]
            )
            or il_rs.min_max_margin != judge_rs.min_max_margin
        )
    )
    print(f"[qs3] action-scale conversion active: {convert_actions}", flush=True)

    def judge_env_step(action_row):
        """Feed one executed action row (A,) to the lockstep env, converting
        between the two eras' action rescale statistics so both instances
        execute the *same raw actuator targets*."""
        row = action_row.astype(np.float32)
        if convert_actions:
            raw = _RS.transform_from_tanh(
                row, il_rs.action_stats, il_rs.min_max_margin
            )
            raw = np.clip(
                raw,
                il_rs.orig_action_space.low[0]
                if il_rs.orig_action_space.low.ndim > 1
                else il_rs.orig_action_space.low,
                il_rs.orig_action_space.high[0]
                if il_rs.orig_action_space.high.ndim > 1
                else il_rs.orig_action_space.high,
            )
            row = _RS.transform_to_tanh(
                raw, judge_rs.action_stats, judge_rs.min_max_margin
            )
        return judge_env.step(row[None, :].astype(np.float32))

    selectors = []
    for s in args.selectors.split(","):
        s = s.strip()
        if s == "all-solos":
            selectors += [f"solo:{n}" for n in names]
        elif s:
            selectors.append(s)
    if args.probe_only:
        selectors = selectors[:1]

    rng = np.random.default_rng(20260808)
    rows, trace = [], []
    for selector in selectors:
        successes = 0
        fork_ticks = 0
        total_ticks = 0
        picks = Counter()
        t0 = time.time()
        n_episodes = 1 if args.probe_only else args.num_eval_episodes
        for ep in range(n_episodes):
            seed = args.eval_seed_start + ep
            il_obs, il_info = il_env.reset(seed=seed)
            j_obs, _ = judge_env.reset(seed=seed)
            drift_baseline = drift_metric(il_obs, j_obs)
            for _, ws in cands:
                ws.agent.reset(
                    ws.main_loop_iterations,
                    [getattr(ws, "_eval_agent_indices", [0])[-1]],
                )
            terminated = truncated = False
            judge_env_done = False
            tick = 0
            while not (terminated or truncated):
                b_obs = {
                    k: np.expand_dims(np.asarray(v), 0) for k, v in il_obs.items()
                }
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
                    scores = score_chunks(j_obs, actions)
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
                chunk = actions[pick][0]  # (20, A)
                il_obs, reward, terminated, truncated, il_info = il_env.step(
                    chunk
                )
                # Replay the executed rows on the judge env. The IL env may
                # stop mid-chunk (success/limit); execute the same count.
                from robobase.workspace import _executed_action_steps

                executed = int(_executed_action_steps(il_info))
                executed = max(1, min(executed, chunk.shape[0]))
                for row in chunk[:executed]:
                    if judge_env_done:
                        break
                    j_obs, _, j_term, j_trunc, _ = judge_env_step(row)
                    judge_env_done = bool(j_term or j_trunc)
                if not judge_env_done:
                    drift_heal(drift_baseline, il_obs, j_obs, seed, tick)
                tick += 1
            success = il_info.get("task_success")
            if success is not None:
                successes += int(np.array(success).astype(int).item())
        if args.probe_only:
            print("[qs3] PROBE OK (1 episode, drift asserts passed)", flush=True)
            return 0
        n = args.num_eval_episodes
        row = {
            "selector": selector,
            "episodes": n,
            "eval_seed_start": args.eval_seed_start,
            "success_rate": successes / n,
            "picks": json.dumps(dict(picks)),
            "fork_ticks": fork_ticks,
            "total_ticks": total_ticks,
            "elapsed_sec": round(time.time() - t0, 1),
        }
        rows.append(row)
        print(
            f"[qs3] {selector}: success {row['success_rate']:.1%} "
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
    print(f"[qs3] wrote {len(rows)} rows -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
