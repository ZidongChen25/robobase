"""WSRL-style staged unanchoring for CQN-AS (research Arm B).

Pre-registered test (arXiv 2412.07762 recipe adapted to CQN-AS):

    Established fact -- deleting the DQfD BC hinge (bc_lambda -> 0) *cold*
    during online training collapses the critic within ~1k updates
    (bc_sibling_q_span 0.35 -> ~0.004, binding_rate -> 1.0, success -> 0).

    WSRL claims that collapse is a SEAM TRANSIENT caused by the
    offline/online distribution mismatch, and is avoidable by
      (1) freezing the anchored policy,
      (2) collecting warmup rollouts with it into the online buffer,
      (3) recalibrating the critic at high UTD on that data with the
          anchor already OFF,
      (4) only then continuing standard online training with the anchor off.

    Prediction on record: after warmup + recalibration, lambda=0 online
    training does NOT collapse for >=20k env steps, and rollout success
    stays near the anchored policy's level.

Implementation notes
--------------------
* The three phases run inside ONE process and ONE uninterrupted
  environment stream, so no episode is ever abandoned mid-flight and the
  replay accumulator (``Workspace._episode_rollouts``) is never orphaned.
  Phase switching is done by mutating ``cfg.online_update_after_steps``
  (read live by ``_online_updates_ready`` on every iteration) rather than
  by re-entering ``_online_rl``.
* ``bc_lambda`` is zeroed through ``method.bc_lambda_schedule`` rather than
  ``method.bc_lambda=0``: the whole BC block in ``cqn.py`` -- *including the
  bc_* diagnostics that carry the Q-span read-out* -- is gated behind
  ``if self.bc_lambda > 0.0 or use_bc_schedule``.  A static zero lambda
  would silently delete the metric this experiment is measured by.
  With the schedule, ``bc_weight`` is a traced 0.0 -> identical gradients,
  diagnostics preserved.
* Demos must be loaded BEFORE ``load_snapshot``: ``_load_demos`` early-returns
  when ``self._snapshot_loaded`` is set, which would leave both replay
  buffers empty (the exact params-only warm-start disaster of cqn-rline 890).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-run", required=True, help="run dir holding .hydra/config.yaml")
    p.add_argument("--checkpoint", required=True, help="path to the *_checkpoint.pkl to start from")
    p.add_argument("--run-dir", required=True, help="fresh output dir for this arm")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--warmup-episodes",
        type=int,
        default=50,
        help="frozen-policy rollout episodes before ANY gradient update (0 = cold control)",
    )
    p.add_argument(
        "--warmup-step-cap",
        type=int,
        default=40000,
        help="hard cap on warmup env steps in case episodes never terminate",
    )
    p.add_argument(
        "--recalib-updates",
        type=int,
        default=4000,
        help="total gradient updates in the recalibration phase (0 = skip)",
    )
    p.add_argument("--recalib-utd", type=int, default=4)
    p.add_argument("--online-steps", type=int, default=20000)
    p.add_argument("--online-utd", type=int, default=1)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--snapshot-every", type=int, default=5000)
    p.add_argument(
        "--bc-lambda-schedule",
        default="0.0",
        help="constant '0.0' unanchors while keeping the bc_* diagnostics alive",
    )
    p.add_argument("--recalib-log-every", type=int, default=50, help="in update calls")
    return p.parse_args()


def _configure_backend_logging():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)


def build_workspace_class():
    from omegaconf import open_dict

    from robobase.workspace_fast import WorkspaceFast

    class StagedUnanchorWorkspace(WorkspaceFast):
        """Workspace with a warmup / recalibration / online phase machine."""

        UPDATES_OFF = 10**12

        def configure_stages(
            self,
            *,
            warmup_episodes: int,
            warmup_step_cap: int,
            recalib_updates: int,
            recalib_utd: int,
            online_steps: int,
            online_utd: int,
            recalib_log_every: int,
            out_dir: Path,
        ):
            self._warmup_episodes = int(warmup_episodes)
            self._warmup_step_cap = int(warmup_step_cap)
            self._recalib_updates = int(recalib_updates)
            self._recalib_utd = int(recalib_utd)
            self._online_steps = int(online_steps)
            self._online_utd = int(online_utd)
            self._recalib_log_every = max(1, int(recalib_log_every))
            self._out_dir = Path(out_dir)
            self._episode_path = self._out_dir / "wsrl_episodes.jsonl"
            self._recalib_path = self._out_dir / "wsrl_recalib.jsonl"
            self._stage_path = self._out_dir / "wsrl_stages.json"

            self._start_step = int(self.global_env_steps)
            self._stage_episode_count = 0
            self._episode_step_count = np.zeros(self.train_envs.num_envs, dtype=np.int64)
            self._episode_return = np.zeros(self.train_envs.num_envs, dtype=np.float64)
            self._episode_index = 0
            self._seam_step = None
            self._online_end_step = None
            self._stage_records = []

            if self._warmup_episodes > 0:
                self._stage = "warmup"
                self.agent.num_update_steps = self._online_utd
                self._set_updates_enabled(False)
            else:
                # Cold control: gradients from the very first iteration.
                self._stage = "online"
                self._seam_step = self._start_step
                self._online_end_step = self._start_step + self._online_steps
                self.agent.num_update_steps = self._online_utd
                self._set_updates_enabled(True)
            self._record_stage(self._stage, "start")

        # ------------------------------------------------------------------
        def _set_updates_enabled(self, enabled: bool):
            with open_dict(self.cfg):
                self.cfg.online_update_after_steps = (
                    0 if enabled else self.UPDATES_OFF
                )

        def _record_stage(self, stage: str, event: str, **extra):
            record = {
                "stage": stage,
                "event": event,
                "env_step": int(self.global_env_steps),
                "main_loop_iterations": int(self.main_loop_iterations),
                "episodes": int(self._episode_index),
                "buffer_size": int(len(self.replay_buffer)),
                "demo_buffer_size": (
                    int(len(self.demo_replay_buffer)) if self.use_demo_replay else 0
                ),
                "wall_time": time.time(),
            }
            record.update(extra)
            self._stage_records.append(record)
            self._stage_path.write_text(json.dumps(self._stage_records, indent=2))
            logging.info("[wsrl] %s", json.dumps(record))
            print(f"[wsrl] {json.dumps(record)}", flush=True)

        # ------------------------------------------------------------------
        def _add_to_replay(
            self,
            actions,
            observations,
            rewards,
            terminations,
            truncations,
            infos,
            next_infos,
        ):
            num_envs = self.train_envs.num_envs
            # Never mutate/retype the arguments handed to the parent: the
            # replay buffer is dtype-sensitive.  Accounting uses a copy.
            reward_view = np.asarray(rewards, dtype=np.float64).reshape(-1)
            self._episode_step_count[:num_envs] += 1
            self._episode_return[:num_envs] += reward_view[:num_envs]

            finished = []
            final_infos = next_infos.get("final_info")
            for i in range(num_envs):
                if not (terminations[i] or truncations[i]):
                    continue
                final_info = {}
                if final_infos is not None and len(final_infos) > i and final_infos[i]:
                    final_info = final_infos[i]
                success_value = final_info.get("task_success", None)
                if success_value is None:
                    success = int(self._episode_return[i] > 0.0)
                    success_source = "return"
                else:
                    success = int(float(np.asarray(success_value).item()) > 0.0)
                    success_source = "task_success"
                finished.append(
                    {
                        "env": i,
                        "success": success,
                        "success_source": success_source,
                        "length": int(self._episode_step_count[i]),
                        "reward": float(self._episode_return[i]),
                    }
                )

            result = super()._add_to_replay(
                actions,
                observations,
                rewards,
                terminations,
                truncations,
                infos,
                next_infos,
            )

            for entry in finished:
                i = entry["env"]
                self._episode_step_count[i] = 0
                self._episode_return[i] = 0.0
                self._episode_index += 1
                self._stage_episode_count += 1
                entry.update(
                    {
                        "stage": self._stage,
                        "episode_index": self._episode_index,
                        "env_step": int(self.global_env_steps),
                        "steps_since_seam": (
                            None
                            if self._seam_step is None
                            else int(self.global_env_steps) - int(self._seam_step)
                        ),
                    }
                )
                with self._episode_path.open("a") as handle:
                    handle.write(json.dumps(entry) + "\n")

            self._maybe_advance_stage(bool(finished))
            return result

        # ------------------------------------------------------------------
        def _maybe_advance_stage(self, at_episode_boundary: bool):
            if self._stage == "warmup":
                warmup_steps = int(self.global_env_steps) - self._start_step
                budget_done = self._stage_episode_count >= self._warmup_episodes
                capped = warmup_steps >= self._warmup_step_cap
                if (budget_done and at_episode_boundary) or capped:
                    self._record_stage(
                        "warmup",
                        "end",
                        warmup_env_steps=warmup_steps,
                        warmup_episodes=int(self._stage_episode_count),
                        hit_step_cap=bool(capped and not budget_done),
                    )
                    self._run_recalibration()
                    self._stage = "online"
                    self._stage_episode_count = 0
                    self._seam_step = int(self.global_env_steps)
                    self._online_end_step = self._seam_step + self._online_steps
                    self.agent.num_update_steps = self._online_utd
                    self._set_updates_enabled(True)
                    self._record_stage(
                        "online",
                        "start",
                        online_end_step=int(self._online_end_step),
                    )
                return

            if (
                self._online_end_step is not None
                and int(self.global_env_steps) >= int(self._online_end_step)
            ):
                self._record_stage("online", "end")
                self._shutting_down = True

        # ------------------------------------------------------------------
        def _run_recalibration(self):
            if self._recalib_updates <= 0:
                self._record_stage("recalibration", "skipped")
                return
            utd = max(1, self._recalib_utd)
            calls = max(1, int(round(self._recalib_updates / utd)))
            self._record_stage(
                "recalibration",
                "start",
                planned_gradient_updates=calls * utd,
                utd=utd,
                calls=calls,
            )
            previous_logging = self.agent.logging
            self.agent.num_update_steps = utd
            start = time.time()
            for call_index in range(calls):
                should_log = (
                    call_index % self._recalib_log_every == 0
                    or call_index == calls - 1
                )
                self.agent.logging = should_log
                metrics = self._perform_updates()
                if not should_log:
                    continue
                row = {"call": call_index, "gradient_updates": (call_index + 1) * utd}
                for key, value in metrics.items():
                    try:
                        row[key] = float(np.asarray(value).item())
                    except (TypeError, ValueError):
                        continue
                row["elapsed_sec"] = time.time() - start
                with self._recalib_path.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(
                    "[wsrl-recalib] "
                    f"{row['gradient_updates']}/{calls * utd} "
                    f"span={row.get('bc_sibling_q_span', float('nan')):.5f} "
                    f"agree={row.get('bc_agreement', float('nan')):.4f} "
                    f"bind={row.get('bc_binding_rate', float('nan')):.4f} "
                    f"critic_loss={row.get('critic_loss', float('nan')):.5f}",
                    flush=True,
                )
            self.agent.logging = previous_logging
            self._record_stage(
                "recalibration",
                "end",
                gradient_updates=calls * utd,
                elapsed_sec=time.time() - start,
            )

    return StagedUnanchorWorkspace


def main():
    args = parse_args()
    _configure_backend_logging()

    from omegaconf import OmegaConf, open_dict

    base_run = Path(args.base_run).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(base_run / ".hydra" / "config.yaml")
    OmegaConf.set_struct(cfg, False)

    total_budget = (
        args.warmup_step_cap + args.online_steps + 20000
    )  # generous cap; the exact stop is driven by the stage machine
    with open_dict(cfg):
        cfg.seed = int(args.seed)
        cfg.num_train_frames = 100000 + total_budget
        cfg.log_every = int(args.log_every)
        cfg.snapshot_every_n = int(args.snapshot_every)
        cfg.save_snapshot = True
        cfg.save_csv = True
        cfg.wandb.use = False
        cfg.tb.use = False
        cfg.log_train_video = False
        cfg.log_eval_video = False
        cfg.num_eval_episodes = 0
        cfg.num_eval_envs = 0
        cfg.eval_every_steps = 10**9
        cfg.gpu_id = None
        # Keep every artifact: this arm is measured off the replay + checkpoints.
        cfg.artifacts.delete_replay_on_train_complete = False
        cfg.artifacts.delete_resume_on_train_complete = False
        cfg.artifacts.save_eval_checkpoints = True
        # Unanchor while keeping the bc_* diagnostics compiled in.
        cfg.method.bc_lambda_schedule = str(args.bc_lambda_schedule)
        cfg.method.num_update_steps = int(args.online_utd)

    OmegaConf.resolve(cfg)
    with open_dict(cfg):
        # Preserve the original schedule domain even though num_train_frames grew.
        cfg.method.num_train_steps = 101000

    from robobase.gpu import apply_requested_gpu

    apply_requested_gpu(cfg)

    (run_dir / ".hydra").mkdir(exist_ok=True)
    OmegaConf.save(cfg, run_dir / ".hydra" / "config.yaml")
    (run_dir / "wsrl_args.json").write_text(json.dumps(vars(args), indent=2))

    workspace_cls = build_workspace_class()
    workspace = workspace_cls(cfg, work_dir=str(run_dir))

    # Order matters: _load_demos() early-returns once a snapshot is loaded.
    workspace._load_demos()
    print(
        f"[wsrl] demos loaded: online={len(workspace.replay_buffer)} "
        f"demo={len(workspace.demo_replay_buffer) if workspace.use_demo_replay else 0}",
        flush=True,
    )
    if len(workspace.replay_buffer) <= 0:
        raise RuntimeError("online replay is empty after _load_demos()")

    workspace.load_snapshot(checkpoint, load_replay_buffer=False)
    print(
        f"[wsrl] loaded {checkpoint} -> env_step={workspace.global_env_steps} "
        f"iters={workspace.main_loop_iterations} "
        f"episodes={workspace.global_env_episodes}",
        flush=True,
    )
    # bc_lambda_schedule is what actually drives the loss weight; assert it landed.
    assert getattr(workspace.agent, "bc_lambda_schedule", None) is not None, (
        "bc_lambda_schedule did not reach the agent; the BC block would stay static"
    )
    print(
        f"[wsrl] agent bc_lambda={workspace.agent.bc_lambda} "
        f"bc_lambda_schedule={workspace.agent.bc_lambda_schedule} "
        f"num_update_steps={workspace.agent.num_update_steps}",
        flush=True,
    )

    workspace.configure_stages(
        warmup_episodes=args.warmup_episodes,
        warmup_step_cap=args.warmup_step_cap,
        recalib_updates=args.recalib_updates,
        recalib_utd=args.recalib_utd,
        online_steps=args.online_steps,
        online_utd=args.online_utd,
        recalib_log_every=args.recalib_log_every,
        out_dir=run_dir,
    )

    try:
        workspace._online_rl()
        if cfg.save_snapshot:
            workspace.save_snapshot()
        (run_dir / "train_complete").touch()
    finally:
        workspace.shutdown()
    print("[wsrl] done", flush=True)


if __name__ == "__main__":
    main()
