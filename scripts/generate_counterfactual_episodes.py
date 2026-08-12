"""Generate near-manifold counterfactual failure episodes (cqn-rline.md D1).

For each demonstration: replay it into the live wrapped env via its seed
(the verified cache_bigym_pixel_demos mechanism), capture branch states at
anchor steps, then per anchor: restore, perturb ONE action dimension by a
CQN L0-bin-scale delta for H consecutive steps, and continue with the
demo's remaining actions open-loop until the episode ends. The rolled-out
branch is recorded as an ordinary online-style episode (demo=0, real
rewards, mc_return = actual discounted reward-to-go) in the exact
UniformReplayBuffer npz schema, ready for injection via
`replay.reuse_saved=true` into a pre-populated <run>/replay dir.

Injection discipline (preregistered): failure data enters as the executed
action's real (mostly zero) return — never as blanket floors.
"""

import argparse
import datetime
import random
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIG_DIR = str(
    (Path(__file__).parents[1] / "robobase" / "cfgs").resolve()
)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--launch", default="cqn_as_pixel_bigym_demo_driven")
    p.add_argument("--task", default="move_plate")
    p.add_argument("--num-demos", type=int, default=60)
    p.add_argument("--anchors-per-demo", type=int, default=6)
    p.add_argument("--perturb-steps", type=int, default=4)
    p.add_argument("--perturb-delta", type=float, default=0.4,
                   help="tanh-space delta = one CQN L0 bin width")
    p.add_argument("--recovery-steps", type=int, default=0,
                   help="after the perturbation, servo back to the demo's "
                        "reference joint positions for this many steps "
                        "(ground-truth recovery; delta action mode makes "
                        "this an exact subtraction), then resume the demo")
    p.add_argument("--max-episodes", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--paired", action="store_true",
                   help="per anchor, roll BOTH the open-loop branch and the "
                        "recovery branch from the same perturbation; write "
                        "both episodes (suffix _open/_recov in the log)")
    p.add_argument("--smoke", action="store_true",
                   help="2 demos x 1 anchor, for wiring checks")
    return p.parse_args()


def main():
    args = _parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[
                f"launch={args.launch}",
                f"env=bigym/{args.task}",
                "num_train_envs=1",
                "num_eval_envs=1",
                "wandb.use=false",
            ],
        )
    OmegaConf.set_struct(cfg, False)

    from robobase.envs.bigym import BiGymEnvFactory
    from robobase.envs.bigym_branch_state import (
        capture_bigym_branch_state,
        restore_bigym_branch_state,
    )
    from robobase.envs.wrappers.rescale_from_tanh import (
        RescaleFromTanhWithMinMax,
    )
    from robobase.replay_buffer.uniform_replay_buffer import save_episode
    from robobase import utils as rb_utils

    factory = BiGymEnvFactory()
    factory.collect_or_fetch_demos(cfg, args.num_demos)
    factory.post_collect_or_fetch_demos(cfg)
    # DemoStore returns demos in nondeterministic order; sort by seed so
    # paired arms (e.g. recovery on/off) hit identical demos and anchors.
    demos = sorted(factory._raw_demos, key=lambda d: int(d.seed))
    env = factory.make_eval_env(cfg)
    action_stats = factory._action_stats
    margin = cfg.min_max_margin

    if args.smoke:
        demos = demos[:2]
        args.anchors_per_demo = 1

    rng = np.random.default_rng(args.seed)
    gamma = float(cfg.replay.gamma)
    n_written = 0
    global_idx = 0
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

    def to_tanh(raw_action):
        return RescaleFromTanhWithMinMax.transform_to_tanh(
            np.asarray(raw_action, dtype=np.float32),
            action_stats,
            margin,
        )

    # tanh space is affine in raw action space; a raw joint offset e maps to
    # a tanh correction of 2e/span with no midpoint bias. Grippers (last two
    # dims) are absolute commands, not deltas — never feedback-correct them.
    stats_min = np.asarray(action_stats["min"], np.float32)
    stats_max = np.asarray(action_stats["max"], np.float32)
    span = (
        (stats_max + np.fabs(stats_max) * margin)
        - (stats_min - np.fabs(stats_min) * margin)
        + 1e-8
    )

    def feedback_correction(raw_offset):
        corr = 2.0 * np.asarray(raw_offset, np.float32) / span
        corr[-2:] = 0.0
        return corr

    for demo_index, demo in enumerate(demos):
        if n_written >= args.max_episodes:
            break
        timesteps = demo.timesteps[1:]  # step 0 carries no executed action
        # info["demo_action"] is what the training pipeline replays — and
        # post_collect_or_fetch_demos has already rescaled it IN PLACE to
        # tanh space, so it is used verbatim. (executed_action is a
        # different, non-delta quantity; to_tanh(demo_action) would
        # double-transform. Both replay as garbage.)
        actions = [
            np.asarray(ts.info["demo_action"], np.float32)
            for ts in timesteps
            if ts.info.get("demo_action") is not None
        ]
        n = len(actions)
        if n < 12:
            continue
        anchor_candidates = list(range(max(1, n // 10), (n * 9) // 10))
        rng.shuffle(anchor_candidates)
        anchors = sorted(anchor_candidates[: args.anchors_per_demo])

        # Replay pass: drive the env along the demo, capturing branch
        # states + prefix rows at each anchor, and the actuated joint
        # reference trajectory for ground-truth recovery servoing.
        obs, _ = env.reset(seed=demo.seed)
        rows = []  # (obs_last_frame_dict, action, reward, term, trunc)
        captured = {}
        ref_qpos = []
        ref_qpos_full = []  # full physics qpos (robot + objects), MILES-style
        # scene-disturbance reference: deviation of the world state from the
        # demo's recorded state at the same step separates "servo failed"
        # from "servo succeeded but the correction disturbed the objects".

        def last_frames(observation):
            return {
                key: np.asarray(value[-1]).copy()
                for key, value in observation.items()
            }

        for step_index, action in enumerate(actions):
            if step_index in anchors:
                captured[step_index] = (
                    capture_bigym_branch_state(env),
                    [row for row in rows],
                    {k: v.copy() for k, v in last_frames(obs).items()},
                )
            chunk = np.repeat(
                action[None, :], int(cfg.action_sequence), axis=0
            )
            next_obs, reward, term, trunc, _ = env.step(chunk)
            rows.append(
                (last_frames(obs), action, float(reward), term, trunc)
            )
            ref_qpos.append(
                np.asarray(env.unwrapped.robot.qpos_actuated, np.float32)
            )
            ref_qpos_full.append(
                np.asarray(
                    env.unwrapped.mojo.physics.data.qpos, np.float64
                ).copy()
            )
            obs = next_obs
            if term or trunc:
                break
        print(
            f"[gen] demo {demo_index} replay return "
            f"{sum(r[2] for r in rows):.2f} len {len(rows)}",
            flush=True,
        )

        def rollout_branch(env_state, prefix_rows, anchor_frames, anchor,
                           dim, delta, recovery_steps):
            restore_bigym_branch_state(env, env_state)
            branch_rows = [row for row in prefix_rows]
            obs_frames = anchor_frames
            recovery_until = anchor + args.perturb_steps + recovery_steps
            joint_err_pre = joint_err_post = None
            obj_dev_rejoin = obj_dev_end = None

            def full_qpos_dev(step_index):
                ref_i = min(step_index, len(ref_qpos_full) - 1)
                now = np.asarray(
                    env.unwrapped.mojo.physics.data.qpos, np.float64
                )
                return float(np.abs(ref_qpos_full[ref_i] - now).max())

            for step_index in range(anchor, n):
                if (
                    recovery_steps > 0
                    and anchor + args.perturb_steps <= step_index
                    and step_index < recovery_until
                    and step_index < len(ref_qpos)
                ):
                    # Ground-truth recovery: demo action as feedforward (the
                    # env's per-step tracking gain is ~0.1-0.6, so a bare
                    # offset servo cannot catch the moving demo reference)
                    # plus proportional feedback toward the demo's recorded
                    # joint configuration at this step.
                    now = np.asarray(
                        env.unwrapped.robot.qpos_actuated, np.float32
                    )
                    err = float(
                        np.abs((ref_qpos[step_index] - now)[:-2]).max()
                    )
                    if joint_err_pre is None:
                        joint_err_pre = err
                    joint_err_post = err
                    action = np.clip(
                        actions[step_index]
                        + feedback_correction(ref_qpos[step_index] - now),
                        -1.0,
                        1.0,
                    )
                else:
                    action = actions[step_index].copy()
                    if step_index < anchor + args.perturb_steps:
                        action[dim] = np.clip(action[dim] + delta, -1.0, 1.0)
                chunk = np.repeat(
                    action[None, :], int(cfg.action_sequence), axis=0
                )
                observation, reward, term, trunc, _ = env.step(chunk)
                branch_rows.append(
                    (obs_frames, action, float(reward), term, trunc)
                )
                obs_frames = last_frames(observation)
                if (
                    obj_dev_rejoin is None
                    and step_index + 1 >= recovery_until
                ):
                    obj_dev_rejoin = full_qpos_dev(step_index)
                if term or trunc:
                    break
            obj_dev_end = full_qpos_dev(step_index)
            return (branch_rows, obs_frames, joint_err_pre, joint_err_post,
                    obj_dev_rejoin, obj_dev_end)

        # Branch pass per anchor.
        for anchor in anchors:
            if anchor not in captured or n_written >= args.max_episodes:
                continue
            env_state, prefix_rows, anchor_frames = captured[anchor]
            dim = int(rng.integers(0, len(actions[0])))
            delta = float(args.perturb_delta) * (
                1.0 if rng.random() < 0.5 else -1.0
            )
            variants = (
                [(0, "open"), (args.recovery_steps, "recov")]
                if args.paired
                else [(args.recovery_steps, "")]
            )
            for recovery_steps, tag in variants:
                (branch_rows, obs_frames, joint_err_pre, joint_err_post,
                 obj_dev_rejoin, obj_dev_end) = (
                    rollout_branch(env_state, prefix_rows, anchor_frames,
                                   anchor, dim, delta, recovery_steps)
                )
                if len(branch_rows) < 14:
                    continue
                episode = _assemble_episode(
                    branch_rows, obs_frames, gamma, cfg, rb_utils
                )
                length = len(branch_rows)
                name = f"{stamp}_{n_written}_{length}_{global_idx}.npz"
                save_episode(episode, out / name, compression="zip")
                global_idx += length
                n_written += 1
                total_reward = sum(r[2] for r in branch_rows)
                err_note = (
                    f" jerr {joint_err_pre:.3f}->{joint_err_post:.3f}"
                    if joint_err_pre is not None
                    else ""
                )
                tag_note = f" [{tag}]" if tag else ""
                print(
                    f"[gen] demo {demo_index} anchor {anchor} dim {dim} "
                    f"delta {delta:+.2f} len {length} "
                    f"return {total_reward:.2f}{err_note}{tag_note} -> {name}",
                    flush=True,
                )
                import json
                with open(out / "manifest.jsonl", "a") as mf:
                    mf.write(json.dumps({
                        "file": name,
                        "demo_seed": int(demo.seed),
                        "demo_index": demo_index,
                        "anchor": anchor,
                        "dim": dim,
                        "delta": delta,
                        "tag": tag or "single",
                        "return": total_reward,
                        "length": length,
                        "jerr_pre": joint_err_pre,
                        "jerr_post": joint_err_post,
                        "obj_dev_rejoin": obj_dev_rejoin,
                        "obj_dev_end": obj_dev_end,
                    }) + "\n")

    print(f"[gen] wrote {n_written} episodes to {out}", flush=True)


def _assemble_episode(rows, final_frames, gamma, cfg, rb_utils):
    n = len(rows)
    rewards = np.asarray([r[2] for r in rows], dtype=np.float32)
    mc = rb_utils.discounted_episode_returns(rewards, gamma).astype(
        np.float32
    )
    episode = {}
    obs_keys = rows[0][0].keys()
    for key in obs_keys:
        stacked = np.stack(
            [r[0][key] for r in rows] + [final_frames[key]], axis=0
        )
        episode[key] = stacked
    action_dim = rows[0][1].shape[0]
    episode["action"] = np.concatenate(
        [
            np.stack([r[1] for r in rows], axis=0),
            np.zeros((1, action_dim), dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)
    episode["reward"] = np.concatenate([rewards, [0.0]]).astype(np.float32)
    terminal = np.zeros(n + 1, dtype=np.int8)
    truncated = np.zeros(n + 1, dtype=np.int8)
    if rows[-1][3]:
        terminal[n - 1] = 1
    else:
        truncated[n - 1] = 1
    terminal[n] = -1
    truncated[n] = -1
    episode["terminal"] = terminal
    episode["truncated"] = truncated
    episode["demo"] = np.zeros(n + 1, dtype=np.uint8)
    # canonical CQN-AS stores no mc_return; only emit when the target run's
    # storage signature has the MC anchor enabled
    if bool(cfg.get("include_mc_return_in_generated", False)):
        episode["mc_return"] = np.concatenate([mc, [0.0]]).astype(
            np.float32
        )
    if bool(cfg.replay.get("nstep_explore_truncate", False)):
        episode["explored"] = np.zeros(n + 1, dtype=np.uint8)
    return episode


if __name__ == "__main__":
    main()
