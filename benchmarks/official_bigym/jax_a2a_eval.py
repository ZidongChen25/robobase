#!/usr/bin/env python3
"""Evaluate the strict JAX A2A checkpoint in a raw BiGym environment."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import time

import numpy as np

from benchmarks.official_bigym.a2a_eval import _make_env, _smoothness
from benchmarks.official_bigym.bigym_data import stack_recent_history
from benchmarks.official_roboverse.jax_a2a import JaxA2APredictor


class JaxA2ARollout:
    def __init__(
        self,
        predictor: JaxA2APredictor,
        *,
        cameras: tuple[str, ...],
        execution_length: int,
    ) -> None:
        self.predictor = predictor
        self.cameras = cameras
        if len(cameras) != predictor.config.num_cameras:
            raise ValueError(
                f"Checkpoint expects {predictor.config.num_cameras} cameras, "
                f"got {cameras}."
            )
        self.observation_steps = predictor.config.observation_steps
        self.history_steps = predictor.config.history_steps
        self.prediction_length = predictor.config.action_steps
        self.execution_length = execution_length
        if not 1 <= execution_length <= self.prediction_length:
            raise ValueError(
                "execution_length must be between one and "
                f"{self.prediction_length}."
            )
        self.images: deque[np.ndarray] = deque(maxlen=self.observation_steps)
        self.states: deque[np.ndarray] = deque(maxlen=self.history_steps)

    def reset(self) -> None:
        self.images.clear()
        self.states.clear()

    def observe(self, observation: dict, qpos_actuated: np.ndarray) -> None:
        images = np.stack(
            [np.asarray(observation[f"rgb_{camera}"]) for camera in self.cameras]
        )
        if images.ndim != 4:
            raise ValueError(f"Expected VCHW images, got {images.shape}.")
        self.images.append(images.astype(np.float32) / 255.0)
        self.states.append(np.asarray(qpos_actuated, dtype=np.float32))

    def predict(self) -> tuple[np.ndarray, float]:
        images = stack_recent_history(self.images, self.observation_steps)[None]
        states = stack_recent_history(self.states, self.history_steps)[None]
        started = time.perf_counter()
        actions = self.predictor.predict(images, states)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        expected = (1, self.prediction_length, states.shape[-1])
        if actions.shape != expected:
            raise ValueError(f"Expected actions {expected}, got {actions.shape}.")
        return actions[0, : self.execution_length], elapsed_ms


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    predictor = JaxA2APredictor(args.checkpoint)
    if args.flow_steps is not None:
        raise ValueError(
            "Strict checkpoints compile flow_steps into the model; train a matched "
            "checkpoint instead of mutating sampling semantics at evaluation."
        )
    predictor.warmup(image_size=args.image_size)
    rollout = JaxA2ARollout(
        predictor,
        cameras=tuple(args.cameras),
        execution_length=args.execution_length,
    )
    env = _make_env(args)
    episodes: list[dict[str, object]] = []
    try:
        if env.action_space.shape != (predictor.config.action_dim,):
            raise ValueError(
                f"BiGym action space {env.action_space.shape} does not match "
                f"checkpoint action_dim={predictor.config.action_dim}."
            )
        for episode_index in range(args.num_episodes):
            seed = args.seed_start + episode_index
            observation, _ = env.reset(seed=seed)
            rollout.reset()
            rollout.observe(observation, env.robot.qpos_actuated)
            success = terminated = truncated = False
            actions: list[np.ndarray] = []
            chunk_starts: list[int] = []
            inference_times: list[float] = []
            clipped_values = total_action_values = 0
            while len(actions) < args.max_steps and not (
                terminated or truncated or success
            ):
                chunk_starts.append(len(actions))
                chunk, inference_ms = rollout.predict()
                inference_times.append(inference_ms)
                for action in chunk:
                    action = np.asarray(action, dtype=np.float32)
                    outside = np.logical_or(
                        action < env.action_space.low,
                        action > env.action_space.high,
                    )
                    clipped_values += int(outside.sum())
                    total_action_values += int(action.size)
                    if args.clip_actions:
                        action = np.clip(
                            action, env.action_space.low, env.action_space.high
                        )
                    observation, reward, terminated, truncated, info = env.step(action)
                    actions.append(action)
                    success = success or bool(info.get("task_success", False)) or (
                        float(reward) > 0.25
                    )
                    if (
                        terminated
                        or truncated
                        or success
                        or len(actions) >= args.max_steps
                    ):
                        break
                    rollout.observe(observation, env.robot.qpos_actuated)
            action_array = np.stack(actions)
            record: dict[str, object] = {
                "episode": episode_index,
                "seed": seed,
                "success": success,
                "length": len(actions),
                "policy_calls": len(inference_times),
                "mean_policy_inference_ms": float(np.mean(inference_times)),
                "clipped_action_fraction": clipped_values
                / max(total_action_values, 1),
            }
            record.update(_smoothness(action_array, chunk_starts))
            episodes.append(record)
            print(
                f"episode={episode_index} seed={seed} success={int(success)} "
                f"length={len(actions)}",
                flush=True,
            )
    finally:
        env.close()

    success_count = sum(bool(record["success"]) for record in episodes)
    return {
        "schema": "official_a2a_jax_bigym_eval_v1",
        "method": "official_a2a_jax",
        "torch_policy_dependency": False,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "task_class": args.task_class,
        "cameras": list(args.cameras),
        "num_episodes": len(episodes),
        "seed_start": args.seed_start,
        "observation_steps": rollout.observation_steps,
        "history_steps": rollout.history_steps,
        "prediction_length": rollout.prediction_length,
        "execution_length": rollout.execution_length,
        "flow_steps": predictor.config.flow_steps,
        "success_count": success_count,
        "success_rate": success_count / max(len(episodes), 1),
        "episodes": episodes,
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bigym-checkout", type=Path, default=root / "third_party/bigym"
    )
    parser.add_argument(
        "--task-class", default="bigym.envs.manipulation:FlipCutlery"
    )
    parser.add_argument(
        "--cameras", nargs="+", default=["head"]
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--control-frequency-hz", type=int, default=20)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--execution-length", type=int, default=8)
    parser.add_argument("--flow-steps", type=int, default=None)
    parser.add_argument("--clip-actions", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate(args)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("success_count", "success_rate")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
