"""Record online rollouts with per-step explore/rollout annotation.

Loads a trained checkpoint, runs episodes with TRAIN-mode action selection
(bin exploration active), captures frames, and overlays a red border +
"探索阶段" on steps whose executed action carries a bin-explore shift
(agent._last_bin_explored), green border + "rollout 阶段" otherwise.

Outputs per-episode mp4 + a JSON of per-step flags/rewards + a summary.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint-step", type=int, required=True)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed-start", type=int, default=400)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--render-size", type=int, default=480)
    return p.parse_args()


def find_font():
    for cand in (
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    ):
        if os.path.exists(cand):
            return cand
    return None


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from omegaconf import OmegaConf
    from robobase.workspace import Workspace

    run_dir = Path(args.run_dir).resolve()
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    OmegaConf.set_struct(cfg, False)
    cfg.create_train_env = False
    cfg.num_train_envs = 1  # keep agent train-side state machinery alive
    cfg.num_train_frames = 0
    cfg.num_eval_envs = 1
    cfg.num_eval_episodes = 1
    cfg.env.eval_seed_start = int(args.seed_start)
    cfg.demo_batch_size = None
    cfg.replay.demo_only_updates = False
    cfg.use_self_imitation = False
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

    work_dir = out / "workspace"
    work_dir.mkdir(exist_ok=True)
    workspace = Workspace(cfg, work_dir=str(work_dir))
    ckpt = (
        run_dir
        / "eval_checkpoints"
        / f"{args.checkpoint_step}_checkpoint.pkl"
    )
    if not ckpt.exists():
        ckpt = run_dir / "snapshots" / f"{args.checkpoint_step}_snapshot.pkl"
    workspace.load_snapshot(ckpt, load_replay_buffer=False)
    agent = workspace.agent

    envs = getattr(workspace.eval_envs, "envs", None)
    env = envs[0] if envs else workspace.eval_env

    from PIL import Image, ImageDraw, ImageFont

    font_path = find_font()
    font = (
        ImageFont.truetype(font_path, 30) if font_path else None
    )
    font_small = (
        ImageFont.truetype(font_path, 20) if font_path else None
    )

    def render_frame(observation):
        try:
            frame = env.unwrapped.render()
            if frame is not None and frame.ndim == 3:
                return np.asarray(frame, np.uint8)
        except Exception:
            pass
        # fallback: tile rgb observation keys (unstacked last frame)
        tiles = []
        for key, value in observation.items():
            if key.startswith("rgb"):
                arr = np.asarray(value)
                if arr.ndim == 4:  # (stack, C, H, W)
                    arr = arr[-1]
                if arr.shape[0] in (1, 3):  # CHW -> HWC
                    arr = np.transpose(arr, (1, 2, 0))
                tiles.append(arr.astype(np.uint8))
        if not tiles:
            raise RuntimeError("no frame source available")
        return np.concatenate(tiles, axis=1)

    def annotate(frame, explored, step_index, reward_total):
        size = args.render_size
        h, w = frame.shape[:2]
        scale = size / h
        img = Image.fromarray(frame).resize(
            (int(w * scale), size), Image.NEAREST
        )
        draw = ImageDraw.Draw(img)
        color = (220, 40, 40) if explored else (40, 180, 60)
        label = "探索阶段 EXPLORE" if explored else "rollout 阶段 ROLLOUT"
        bw = 14 if explored else 8
        for i in range(bw):
            draw.rectangle(
                [i, i, img.width - 1 - i, img.height - 1 - i],
                outline=color,
            )
        if font:
            draw.rectangle([bw + 2, bw + 2, bw + 320, bw + 44],
                           fill=(0, 0, 0))
            draw.text((bw + 8, bw + 6), label, fill=color, font=font)
            draw.text(
                (bw + 8, img.height - bw - 28),
                f"step {step_index}  R={reward_total:.0f}",
                fill=(255, 255, 255),
                font=font_small,
            )
        return np.asarray(img)

    import imageio.v2 as imageio

    ckpt_step = int(args.checkpoint_step)
    summary = []
    for ep in range(args.episodes):
        seed = args.seed_start + ep
        obs, _ = env.reset(seed=seed)
        # clear any per-episode explore/plan state on the train side
        try:
            agent.reset(ckpt_step, [0])
        except Exception:
            pass
        for attr in (
            "_bin_explored_exec_remaining",
            "_last_bin_explored",
            "_last_bin_explore_applied",
        ):
            if hasattr(agent, attr):
                delattr(agent, attr)

        frames = []
        flags = []
        rewards = []
        total = 0.0
        step_index = 0
        while True:
            batched = {
                k: np.asarray(v)[None] for k, v in obs.items()
            }
            action = agent.act(batched, step=ckpt_step, eval_mode=False)
            action = np.asarray(action)
            if action.ndim == 3:
                action = action[0]
            explored = bool(
                np.asarray(
                    getattr(agent, "_last_bin_explored", [False])
                ).ravel()[0]
            )
            obs, reward, term, trunc, _ = env.step(action)
            total += float(reward)
            flags.append(explored)
            rewards.append(float(reward))
            frames.append(
                annotate(render_frame(obs), explored, step_index, total)
            )
            step_index += 1
            if term or trunc:
                break

        name = f"ep{ep}_seed{seed}_R{total:.0f}"
        imageio.mimwrite(
            out / f"{name}.mp4", frames, fps=args.fps, quality=7
        )
        explored_count = int(np.sum(flags))
        # burst structure: lengths of consecutive explored runs
        bursts = []
        run = 0
        for f in flags:
            if f:
                run += 1
            elif run:
                bursts.append(run)
                run = 0
        if run:
            bursts.append(run)
        info = {
            "episode": ep,
            "seed": seed,
            "steps": len(flags),
            "return": total,
            "explored_steps": explored_count,
            "explored_fraction": explored_count / max(1, len(flags)),
            "explore_bursts": bursts,
            "flags": flags,
            "rewards": rewards,
        }
        summary.append(info)
        with open(out / f"{name}.json", "w") as fh:
            json.dump(info, fh)
        print(
            f"[vid] ep{ep} seed{seed}: {len(flags)} steps, return {total:.0f}, "
            f"explored {explored_count} ({100*info['explored_fraction']:.1f}%), "
            f"bursts {bursts}",
            flush=True,
        )

    agg_frac = float(np.mean([s["explored_fraction"] for s in summary]))
    agg_burst = [b for s in summary for b in s["explore_bursts"]]
    print(
        f"[vid] TOTAL: mean explored fraction {100*agg_frac:.1f}%, "
        f"bursts n={len(agg_burst)} "
        f"max_len={max(agg_burst) if agg_burst else 0}",
        flush=True,
    )


if __name__ == "__main__":
    main()
