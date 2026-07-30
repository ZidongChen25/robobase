#!/usr/bin/env python3
"""Evaluate an official A2A checkpoint directly in a raw BiGym environment."""

from __future__ import annotations

import argparse
from collections import deque
import importlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np

from benchmarks.official_bigym.a2a_upstream import (
    OFFICIAL_A2A_COMMIT,
    file_sha256,
    load_official_checkpoint,
    validate_official_checkout,
)
from benchmarks.official_bigym.bigym_data import stack_recent_history


def _resolve_class(spec: str):
    if ":" not in spec:
        raise ValueError("--task-class must use module.path:ClassName syntax.")
    module_name, class_name = spec.split(":", maxsplit=1)
    return getattr(importlib.import_module(module_name), class_name)


def _smoothness(actions: np.ndarray, chunk_starts: Sequence[int]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if actions.shape[0] > 1:
        metrics["action_first_difference"] = float(
            np.linalg.norm(np.diff(actions, axis=0), axis=-1).mean()
        )
    if actions.shape[0] > 2:
        metrics["action_second_difference"] = float(
            np.linalg.norm(np.diff(actions, n=2, axis=0), axis=-1).mean()
        )
    if actions.shape[0] > 3:
        metrics["action_jerk"] = float(
            np.linalg.norm(np.diff(actions, n=3, axis=0), axis=-1).mean()
        )
    boundaries = [index for index in chunk_starts[1:] if 0 < index < actions.shape[0]]
    if boundaries:
        metrics["action_boundary_jump"] = float(
            np.linalg.norm(actions[boundaries] - actions[np.asarray(boundaries) - 1], axis=-1).mean()
        )
    return metrics


class OfficialPolicyRollout:
    """Observation-history adapter around the unmodified official policy."""

    def __init__(self, policy, *, camera: str, device: str, execution_length: int | None):
        import torch

        self.torch = torch
        self.policy = policy
        self.camera = camera
        self.device = torch.device(device)
        self.history_steps = int(policy.n_obs_steps)
        self.prediction_length = int(policy.n_action_steps)
        self.execution_length = (
            self.prediction_length if execution_length is None else int(execution_length)
        )
        if not 1 <= self.execution_length <= self.prediction_length:
            raise ValueError(
                "execution_length must be between one and the checkpoint's "
                f"prediction length {self.prediction_length}."
            )
        self.images: deque[np.ndarray] = deque(maxlen=self.history_steps)
        self.states: deque[np.ndarray] = deque(maxlen=self.history_steps)

    def reset(self) -> None:
        self.images.clear()
        self.states.clear()

    def observe(self, observation: dict, qpos_actuated: np.ndarray) -> None:
        image = np.asarray(observation[f"rgb_{self.camera}"])
        if image.ndim != 3:
            raise ValueError(f"Expected CHW image, got {image.shape}.")
        self.images.append(image.astype(np.float32) / 255.0)
        self.states.append(np.asarray(qpos_actuated, dtype=np.float32))

    def predict(self) -> tuple[np.ndarray, float]:
        torch = self.torch
        images = stack_recent_history(self.images, self.history_steps)[None]
        states = stack_recent_history(self.states, self.history_steps)[None]
        obs = {
            "head_cam": torch.from_numpy(images).to(self.device),
            "agent_pos": torch.from_numpy(states).to(self.device),
        }
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        with torch.no_grad():
            action = self.policy.predict_action(obs)["action"]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        action = action.detach().to(torch.float32).cpu().numpy()
        if action.shape != (1, self.prediction_length, states.shape[-1]):
            raise ValueError(
                "Official policy returned an unexpected action shape: "
                f"{action.shape}."
            )
        return action[0, : self.execution_length], elapsed_ms


def _make_env(args: argparse.Namespace):
    checkout = Path(args.bigym_checkout).expanduser().resolve()
    if str(checkout) not in sys.path:
        sys.path.insert(0, str(checkout))
    from bigym.action_modes import JointPositionActionMode, PelvisDof
    from bigym.utils.observation_config import CameraConfig, ObservationConfig

    task_class = _resolve_class(args.task_class)
    action_mode = JointPositionActionMode(
        absolute=True,
        floating_base=True,
        floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ],
    )
    camera_names = getattr(args, "cameras", None)
    if camera_names is None:
        camera_names = [args.camera]
    cameras = [
        CameraConfig(
            name=name,
            rgb=True,
            depth=False,
            resolution=(args.image_size, args.image_size),
        )
        for name in camera_names
    ]
    return task_class(
        render_mode="rgb_array",
        action_mode=action_mode,
        observation_config=ObservationConfig(
            cameras=cameras,
            proprioception=True,
            privileged_information=False,
        ),
        control_frequency=args.control_frequency_hz,
    )


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    expected = None if args.allow_unpinned_upstream else OFFICIAL_A2A_COMMIT
    checkout, commit = validate_official_checkout(
        args.official_checkout, expected_commit=expected
    )
    policy, cfg = load_official_checkpoint(
        args.checkpoint, checkout, device=args.device
    )
    checkpoint_method = str(cfg.policy_name)
    if args.method != "auto" and args.method != checkpoint_method:
        raise ValueError(
            f"Checkpoint policy_name={checkpoint_method!r}, but --method={args.method!r}."
        )
    if args.flow_steps is not None:
        if args.flow_steps < 1:
            raise ValueError("--flow-steps must be positive.")
        if hasattr(policy, "num_sampling_steps"):
            policy.num_sampling_steps = int(args.flow_steps)
            if hasattr(policy, "flow_matcher"):
                policy.flow_matcher.num_sampling_steps = int(args.flow_steps)
        elif hasattr(policy, "num_inference_steps"):
            policy.num_inference_steps = int(args.flow_steps)
        else:
            raise TypeError(
                f"Unsupported official policy type {type(policy).__name__}: "
                "no flow-step attribute."
            )
    rollout = OfficialPolicyRollout(
        policy,
        camera=args.camera,
        device=args.device,
        execution_length=args.execution_length,
    )
    env = _make_env(args)
    episode_records: list[dict[str, object]] = []
    try:
        if env.action_space.shape != (int(policy.action_dim),):
            raise ValueError(
                f"BiGym action space {env.action_space.shape} does not match "
                f"checkpoint action_dim={policy.action_dim}."
            )
        for episode_index in range(args.num_episodes):
            seed = args.seed_start + episode_index
            # Make stochastic Gaussian-source policies independently repeatable.
            rollout.torch.manual_seed(seed)
            if rollout.device.type == "cuda":
                rollout.torch.cuda.manual_seed_all(seed)
            observation, _ = env.reset(seed=seed)
            rollout.reset()
            rollout.observe(observation, env.robot.qpos_actuated)
            success = False
            terminated = False
            truncated = False
            actions: list[np.ndarray] = []
            chunk_starts: list[int] = []
            inference_times: list[float] = []
            clipped_values = 0
            total_action_values = 0
            while len(actions) < args.max_steps and not (terminated or truncated or success):
                chunk_starts.append(len(actions))
                chunk, inference_ms = rollout.predict()
                inference_times.append(inference_ms)
                for action in chunk:
                    action = np.asarray(action, dtype=np.float32)
                    outside_bounds = np.logical_or(
                        action < env.action_space.low,
                        action > env.action_space.high,
                    )
                    clipped_values += int(outside_bounds.sum())
                    total_action_values += int(action.size)
                    if args.clip_actions:
                        action = np.clip(action, env.action_space.low, env.action_space.high)
                    observation, reward, terminated, truncated, info = env.step(action)
                    actions.append(np.asarray(action, dtype=np.float32))
                    success = success or bool(info.get("task_success", False)) or float(reward) > 0.25
                    if terminated or truncated or success or len(actions) >= args.max_steps:
                        break
                    rollout.observe(observation, env.robot.qpos_actuated)
            action_array = np.stack(actions) if actions else np.empty((0, policy.action_dim))
            record: dict[str, object] = {
                "episode": episode_index,
                "seed": seed,
                "success": success,
                "length": len(actions),
                "policy_calls": len(inference_times),
                "clipped_action_fraction": (
                    clipped_values / total_action_values
                    if total_action_values
                    else 0.0
                ),
                "mean_policy_inference_ms": (
                    float(np.mean(inference_times)) if inference_times else None
                ),
            }
            if actions:
                record.update(_smoothness(action_array, chunk_starts))
            episode_records.append(record)
            print(
                f"episode={episode_index} seed={seed} success={int(success)} "
                f"length={len(actions)}"
            )
    finally:
        env.close()

    success_count = sum(bool(record["success"]) for record in episode_records)
    aggregate_names = (
        "length",
        "mean_policy_inference_ms",
        "action_first_difference",
        "action_second_difference",
        "action_jerk",
        "action_boundary_jump",
        "clipped_action_fraction",
    )
    aggregate = {}
    for name in aggregate_names:
        values = [record[name] for record in episode_records if record.get(name) is not None]
        if values:
            aggregate[f"mean_{name}"] = float(np.mean(values))
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    return {
        "method": f"official_{checkpoint_method}",
        "official_commit": commit,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_policy_target": str(cfg.policy_config._target_),
        "device": args.device,
        "task_class": args.task_class,
        "camera": args.camera,
        "control_frequency_hz": args.control_frequency_hz,
        "num_episodes": args.num_episodes,
        "seed_start": args.seed_start,
        "policy_seed_mode": "episode_seed",
        "max_steps": args.max_steps,
        "history_steps": rollout.history_steps,
        "prediction_length": rollout.prediction_length,
        "execution_length": rollout.execution_length,
        "flow_steps": int(
            policy.num_sampling_steps
            if hasattr(policy, "num_sampling_steps")
            else policy.num_inference_steps
        ),
        "clip_actions": bool(args.clip_actions),
        "success_count": success_count,
        "success_rate": success_count / max(len(episode_records), 1),
        **aggregate,
        "episodes": episode_records,
    }


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--official-checkout",
        type=Path,
        default=root / "third_party/A2A_Flow_Matching_official",
    )
    parser.add_argument(
        "--bigym-checkout", type=Path, default=root / "third_party/bigym"
    )
    parser.add_argument(
        "--task-class", default="bigym.envs.manipulation:FlipCutlery"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--method", choices=("auto", "a2a", "fm_unet"), default="auto")
    parser.add_argument("--camera", default="head")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--control-frequency-hz", type=int, default=20)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--execution-length", type=int, default=None)
    parser.add_argument("--flow-steps", type=int, default=None)
    parser.add_argument("--clip-actions", action="store_true")
    parser.add_argument("--allow-unpinned-upstream", action="store_true")
    return parser


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    args = build_parser().parse_args()
    for name in ("image_size", "control_frequency_hz", "num_episodes", "max_steps"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    result = evaluate(args)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, sort_keys=True)
    print(json.dumps({key: value for key, value in result.items() if key != "episodes"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
