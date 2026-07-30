"""Minimal feature-dataset trainer for the pinned official JAX policies."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from benchmarks.official_bigym.legato_adapter import (
    OfficialBigymPolicy,
    OfficialPolicyConfig,
)
from benchmarks.official_bigym.legato_checkpoint import (
    load_checkpoint,
    save_checkpoint,
    warm_start_legato_from_vanilla,
)
from benchmarks.official_bigym.legato_data import load_window_dataset


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    grad_norm_clip: float = 10.0
    lr_warmup_steps: int = 1000

    def __post_init__(self) -> None:
        if (
            self.learning_rate <= 0
            or self.grad_norm_clip <= 0
            or self.lr_warmup_steps <= 0
        ):
            raise ValueError(
                "learning_rate, grad_norm_clip, and lr_warmup_steps must be positive."
            )
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")


def _train_step(policy, optimizer, key, features, action_chunks):
    def loss_fn(current_policy):
        return current_policy.loss(key, features, action_chunks)

    loss, gradients = nnx.value_and_grad(loss_fn)(policy)
    grad_norm = optax.global_norm(gradients)
    optimizer.update(gradients)
    return loss, grad_norm


class OfficialTrainer:
    """Optimizer boundary; the model loss remains the unmodified upstream loss."""

    def __init__(self, adapter: OfficialBigymPolicy, config: TrainConfig):
        self.adapter = adapter
        self.config = config
        self.learning_rate_schedule = optax.warmup_constant_schedule(
            0.0,
            config.learning_rate,
            config.lr_warmup_steps,
        )
        self.step_count = 0
        self.optimizer = nnx.Optimizer(
            adapter.policy,
            optax.chain(
                optax.clip_by_global_norm(config.grad_norm_clip),
                optax.adamw(
                    self.learning_rate_schedule,
                    weight_decay=config.weight_decay,
                ),
            ),
        )
        self._compiled_step = nnx.jit(_train_step)

    def step(
        self,
        key: jax.Array,
        features: np.ndarray | jax.Array,
        action_chunks: np.ndarray | jax.Array,
    ) -> dict[str, float]:
        features = jnp.asarray(features, dtype=jnp.float32)
        action_chunks = jnp.asarray(action_chunks, dtype=jnp.float32)
        self.adapter._validate_inputs(features, action_chunks)
        loss, grad_norm = self._compiled_step(
            self.adapter.policy,
            self.optimizer,
            key,
            features,
            action_chunks,
        )
        learning_rate = self.learning_rate_schedule(self.step_count)
        self.step_count += 1
        return {
            "loss": float(jax.device_get(loss)),
            "grad_norm": float(jax.device_get(grad_norm)),
            "learning_rate": float(jax.device_get(learning_rate)),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("vanilla", "legato"), required=True)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-flow-steps", type=int, default=5)
    parser.add_argument("--execute-horizon", type=int, default=4)
    parser.add_argument("--inference-delay", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-norm-clip", type=float, default=10.0)
    parser.add_argument("--lr-warmup-steps", type=int, default=1000)
    parser.add_argument(
        "--warm-start",
        type=Path,
        help=(
            "Non-official ablation: initialize Legato from a vanilla checkpoint. "
            "The matched official run starts Legato from random initialization."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive.")
    dataset = load_window_dataset(args.dataset)
    horizon = int(dataset.action_chunks.shape[1])
    policy_config = OfficialPolicyConfig(
        action_horizon=horizon,
        execute_horizon=args.execute_horizon,
        inference_delay=args.inference_delay,
        num_flow_steps=args.num_flow_steps,
        warmup_max=min(4, horizon),
    )
    adapter = OfficialBigymPolicy(
        mode=args.mode,
        obs_dim=int(dataset.features.shape[-1]),
        action_dim=int(dataset.action_chunks.shape[-1]),
        config=policy_config,
        seed=args.seed,
    )
    if args.warm_start is not None:
        if args.mode != "legato":
            raise ValueError("--warm-start is only valid for Legato training.")
        vanilla = OfficialBigymPolicy(
            mode="vanilla",
            obs_dim=adapter.obs_dim,
            action_dim=adapter.action_dim,
            config=policy_config,
            seed=args.seed,
        )
        load_checkpoint(args.warm_start, vanilla, strict=False)
        warm_start_legato_from_vanilla(vanilla, adapter)

    train_config = TrainConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_norm_clip=args.grad_norm_clip,
        lr_warmup_steps=args.lr_warmup_steps,
    )
    trainer = OfficialTrainer(adapter, train_config)
    rng = jax.random.key(args.seed)
    step = 0
    for epoch in range(args.epochs):
        metrics = []
        for features, action_chunks in dataset.batches(
            args.batch_size,
            seed=args.seed + epoch,
            drop_remainder=True,
        ):
            rng, key = jax.random.split(rng)
            metrics.append(trainer.step(key, features, action_chunks))
            step += 1
        mean_loss = np.mean([item["loss"] for item in metrics])
        mean_grad_norm = np.mean([item["grad_norm"] for item in metrics])
        learning_rate = metrics[-1]["learning_rate"]
        print(
            f"epoch={epoch + 1} step={step} loss={mean_loss:.6f} "
            f"grad_norm={mean_grad_norm:.6f} lr={learning_rate:.8f}"
        )
    dataset_path = args.dataset.expanduser().resolve()
    initialization = "random"
    warm_start_sha256 = None
    if args.warm_start is not None:
        initialization = "vanilla_warm_start_ablation"
        warm_start_sha256 = _sha256(args.warm_start.expanduser().resolve())
    save_checkpoint(
        args.output,
        adapter,
        step=step,
        extra={
            "dataset": str(dataset_path),
            "dataset_sha256": _sha256(dataset_path),
            "num_examples": len(dataset),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "drop_remainder": True,
            "seed": int(args.seed),
            "train_config": asdict(train_config),
            "initialization": initialization,
            "warm_start_sha256": warm_start_sha256,
        },
    )


if __name__ == "__main__":
    main()


__all__ = ["OfficialTrainer", "TrainConfig"]
