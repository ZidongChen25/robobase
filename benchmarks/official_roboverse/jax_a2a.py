#!/usr/bin/env python3
"""Train the strict JAX A2A port on an official RoboVerse Zarr dataset."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

from flax import serialization, struct
from flax.core import freeze, unfreeze
import jax
import jax.numpy as jnp
import numpy as np
import optax

from robobase.models.official_a2a import OfficialA2A, OfficialA2AConfig
from robobase.models.encoder import _load_resnet_feature_model


@dataclass(frozen=True)
class LimitsNormalizer:
    scale: np.ndarray
    offset: np.ndarray

    @classmethod
    def fit(cls, data: np.ndarray, range_eps: float = 1e-4):
        data = np.asarray(data, dtype=np.float32).reshape((-1, data.shape[-1]))
        minimum = data.min(axis=0)
        maximum = data.max(axis=0)
        value_range = maximum - minimum
        constant = value_range < range_eps
        safe_range = np.where(constant, 2.0, value_range)
        scale = 2.0 / safe_range
        offset = -1.0 - scale * minimum
        offset = np.where(constant, -minimum, offset)
        return cls(scale.astype(np.float32), offset.astype(np.float32))

    def normalize(self, value):
        return value * self.scale + self.offset

    def unnormalize(self, value):
        return (value - self.offset) / self.scale


def create_sequence_indices(
    episode_ends: np.ndarray,
    *,
    sequence_length: int = 16,
    pad_before: int = 7,
    pad_after: int = 7,
    episode_mask: np.ndarray | None = None,
) -> np.ndarray:
    """NumPy equivalent of the public SequenceSampler index construction."""

    if episode_mask is None:
        episode_mask = np.ones_like(episode_ends, dtype=bool)
    indices: list[tuple[int, int, int, int]] = []
    for episode, end in enumerate(episode_ends):
        if not episode_mask[episode]:
            continue
        start = 0 if episode == 0 else int(episode_ends[episode - 1])
        length = int(end) - start
        for index in range(-pad_before, length - sequence_length + pad_after + 1):
            buffer_start = max(index, 0) + start
            buffer_end = min(index + sequence_length, length) + start
            sample_start = buffer_start - (index + start)
            sample_end = sequence_length - (index + sequence_length + start - buffer_end)
            indices.append((buffer_start, buffer_end, sample_start, sample_end))
    return np.asarray(indices, dtype=np.int64)


class OfficialZarrDataset:
    """Torch-free implementation of the public RobotImageDataset."""

    def __init__(
        self,
        path: str | Path,
        *,
        observation_steps: int = 8,
        history_steps: int = 8,
        action_steps: int = 8,
        cameras: tuple[str, ...] = ("head",),
        cache_images: bool = True,
    ):
        import zarr

        root = zarr.open_group(str(Path(path).expanduser().resolve()), mode="r")
        self.cameras = tuple(cameras)
        if not self.cameras or len(set(self.cameras)) != len(self.cameras):
            raise ValueError("cameras must contain unique camera names.")
        image_arrays = tuple(root[f"data/{name}_camera"] for name in self.cameras)
        self.images = tuple(
            array[:] if cache_images else array for array in image_arrays
        )
        self.states = np.asarray(root["data/state"][:], dtype=np.float32)
        self.actions = np.asarray(root["data/action"][:], dtype=np.float32)
        self.episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
        if self.episode_ends.size < 2:
            raise ValueError("Official training requires at least two episodes.")
        self.observation_steps = int(observation_steps)
        self.history_steps = int(history_steps)
        self.action_steps = int(action_steps)
        if self.observation_steps > self.history_steps:
            raise ValueError("observation_steps cannot exceed history_steps.")
        self.sequence_length = self.history_steps + self.action_steps
        # get_val_mask(..., val_ratio=0.02) in the public code always selects -1.
        train_mask = np.ones(self.episode_ends.size, dtype=bool)
        train_mask[-1] = False
        self.train_indices = create_sequence_indices(
            self.episode_ends,
            sequence_length=self.sequence_length,
            pad_before=self.history_steps - 1,
            pad_after=self.action_steps - 1,
            episode_mask=train_mask,
        )
        self.val_indices = create_sequence_indices(
            self.episode_ends,
            sequence_length=self.sequence_length,
            pad_before=self.history_steps - 1,
            pad_after=self.action_steps - 1,
            episode_mask=~train_mask,
        )
        self.train_gather_indices = self._build_gather_indices(
            self.train_indices, self.sequence_length
        )
        # The public normalizer is fitted on the complete replay buffer, including val.
        self.state_normalizer = LimitsNormalizer.fit(self.states)
        self.action_normalizer = LimitsNormalizer.fit(self.actions)

    @staticmethod
    def _build_gather_indices(
        indices: np.ndarray, sequence_length: int = 16
    ) -> np.ndarray:
        gather = np.empty((len(indices), sequence_length), dtype=np.int64)
        for output, (start, end, sample_start, sample_end) in enumerate(indices):
            gather[output, :sample_start] = start
            gather[output, sample_start:sample_end] = np.arange(start, end)
            gather[output, sample_end:] = end - 1
        return gather

    def batch(self, sample_indices: np.ndarray) -> dict[str, np.ndarray]:
        gather = self.train_gather_indices[np.asarray(sample_indices)]
        image_start = self.history_steps - self.observation_steps
        images = np.stack(
            [
                np.asarray(array[gather[:, image_start : self.history_steps]])
                for array in self.images
            ],
            axis=2,
        )
        states = self.states[gather]
        actions = self.actions[gather]
        return {
            # Match upstream: transfer compact uint8 RGB and normalize on device.
            "images": images,
            "states": self.state_normalizer.normalize(states),
            "actions": self.action_normalizer.normalize(actions),
        }


@struct.dataclass
class TrainState:
    params: object
    ema_params: object
    model_state: object
    opt_state: object
    step: jax.Array


class JaxA2APredictor:
    """Inference-only policy with the same chunk API as upstream A2A."""

    def __init__(self, checkpoint: str | Path):
        payload = serialization.msgpack_restore(
            Path(checkpoint).expanduser().resolve().read_bytes()
        )
        model_config = dict(payload["config"]["model"])
        # Checkpoints written before official image-normalizer parity were
        # trained directly on [0, 1] RGB. Preserve their inference semantics.
        model_config.setdefault("image_range_normalization", False)
        config = OfficialA2AConfig(**model_config)
        self.config = config
        self.model = OfficialA2A(config)
        self.params = jax.device_put(payload["state"]["ema_params"])
        self.model_state = jax.device_put(
            payload["state"].get("model_state", {})
        )
        normalizer = payload["normalizer"]
        self.state_normalizer = LimitsNormalizer(
            np.asarray(normalizer["state_scale"]),
            np.asarray(normalizer["state_offset"]),
        )
        self.action_normalizer = LimitsNormalizer(
            np.asarray(normalizer["action_scale"]),
            np.asarray(normalizer["action_offset"]),
        )
        self._state_scale = jax.device_put(self.state_normalizer.scale)
        self._state_offset = jax.device_put(self.state_normalizer.offset)
        self._action_scale = jax.device_put(self.action_normalizer.scale)
        self._action_offset = jax.device_put(self.action_normalizer.offset)
        self._predict = jax.jit(
            lambda images, states: self.model.apply(
                {"params": self.params, **self.model_state},
                images,
                states * self._state_scale + self._state_offset,
                method=self.model.predict_normalized,
            )
        )

    def predict_device(self, images: jax.Array, states: jax.Array) -> jax.Array:
        normalized = self._predict(images, states)
        return (normalized - self._action_offset) / self._action_scale

    def warmup(self, *, batch_size: int = 1, image_size: int = 256) -> None:
        actions = self.predict_device(
            jnp.zeros(
                (
                    batch_size,
                    self.config.observation_steps,
                    self.config.num_cameras,
                    3,
                    image_size,
                    image_size,
                ),
                dtype=jnp.float32,
            ),
            jnp.zeros(
                (
                    batch_size,
                    self.config.history_steps,
                    self.config.action_dim,
                ),
                dtype=jnp.float32,
            ),
        )
        actions.block_until_ready()

    def predict(self, images: np.ndarray, states: np.ndarray) -> np.ndarray:
        images = np.asarray(images, dtype=np.float32)
        states = np.asarray(states, dtype=np.float32)
        actions = self.predict_device(
            jnp.asarray(images), jnp.asarray(states)
        )
        return np.asarray(jax.device_get(actions))


def cosine_schedule(step, *, warmup_steps: int, total_steps: int):
    step = jnp.asarray(step, dtype=jnp.float32)
    warmup = jnp.minimum(step / max(1, warmup_steps), 1.0)
    progress = jnp.clip(
        (step - warmup_steps) / max(1, total_steps - warmup_steps), 0.0, 1.0
    )
    cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
    return jnp.where(step < warmup_steps, warmup, cosine)


def ema_decay(optimization_step: jax.Array) -> jax.Array:
    step = jnp.maximum(0.0, optimization_step.astype(jnp.float32) - 1.0)
    value = 1.0 - (1.0 + step) ** -0.75
    return jnp.where(step <= 0, 0.0, jnp.minimum(value, 0.9999))


def save_checkpoint(
    output: Path,
    state: TrainState,
    *,
    epoch: int,
    config: dict[str, object],
    dataset: OfficialZarrDataset,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "state": serialization.to_state_dict(jax.device_get(state)),
        "epoch": epoch,
        "config": config,
        "normalizer": {
            "state_scale": dataset.state_normalizer.scale,
            "state_offset": dataset.state_normalizer.offset,
            "action_scale": dataset.action_normalizer.scale,
            "action_offset": dataset.action_normalizer.offset,
        },
    }
    path = output / f"epoch_{epoch:04d}.msgpack"
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(serialization.msgpack_serialize(payload))
    temporary.replace(path)
    return path


def train(args: argparse.Namespace) -> None:
    dataset = OfficialZarrDataset(
        args.dataset,
        observation_steps=args.observation_steps,
        history_steps=args.history_steps,
        action_steps=args.action_steps,
        cameras=tuple(args.cameras),
        cache_images=not args.no_image_cache,
    )
    if dataset.states.shape[-1] != dataset.actions.shape[-1]:
        raise ValueError(
            "Official A2A requires state/action dimensions to match; got "
            f"{dataset.states.shape[-1]} and {dataset.actions.shape[-1]}."
        )
    config = OfficialA2AConfig(
        action_dim=int(dataset.actions.shape[-1]),
        observation_steps=args.observation_steps,
        history_steps=args.history_steps,
        action_steps=args.action_steps,
        flow_steps=args.flow_steps,
        flow_matcher=args.flow_matcher,
        image_range_normalization=args.image_range_normalization,
        num_cameras=len(args.cameras),
        vision_encoder=args.vision_encoder,
        resize_to_224=args.resize_to_224,
    )
    batches_per_epoch = len(dataset.train_indices) // args.batch_size
    total_schedule_steps = batches_per_epoch * args.epochs
    model = OfficialA2A(config)
    init_batch = dataset.batch(np.arange(args.batch_size))
    init_key, train_key = jax.random.split(jax.random.PRNGKey(args.seed))
    variables = model.init(
        init_key,
        jnp.asarray(init_batch["images"]),
        jnp.asarray(init_batch["states"]),
        jnp.asarray(init_batch["actions"]),
        method=model.initialize_all,
    )
    if args.pretrained_resnet_weights is not None:
        if config.vision_encoder != "fm_resnet":
            raise ValueError(
                "--pretrained-resnet-weights requires --vision-encoder=fm_resnet."
            )
        _, pretrained, _ = _load_resnet_feature_model(
            "resnet18",
            pretrained=True,
            pretrained_weights_path=args.pretrained_resnet_weights,
        )
        variables = unfreeze(variables)
        variables["params"]["observation_encoder"]["rgb_model"]["trunk"] = unfreeze(
            pretrained["params"]
        )
        variables["batch_stats"]["observation_encoder"]["rgb_model"]["trunk"] = unfreeze(
            pretrained["batch_stats"]
        )
        variables = freeze(variables)
    model_state = {
        name: values for name, values in variables.items() if name != "params"
    }
    optimizer = optax.adamw(
        learning_rate=1.0,
        b1=0.95,
        b2=0.999,
        eps=1e-8,
        weight_decay=1e-6,
    )
    state = TrainState(
        params=variables["params"],
        ema_params=variables["params"],
        model_state=model_state,
        opt_state=optimizer.init(variables["params"]),
        step=jnp.asarray(0, dtype=jnp.int32),
    )

    metric_names = (
        "loss",
        "flow_loss",
        "consistency_loss",
        "flow_action_reconstruction_loss",
        "encoder_action_reconstruction_loss",
        "lr",
        "ema_decay",
    )

    @jax.jit
    def update(state: TrainState, metric_totals, batch, key):
        def loss_fn(params):
            return model.apply(
                {"params": params, **state.model_state},
                batch["images"],
                batch["states"],
                batch["actions"],
                key,
                method=model.compute_loss,
            )

        (loss, metrics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params
        )
        updates, opt_state = optimizer.update(gradients, state.opt_state, state.params)
        # diffusers creates LambdaLR at lambda(0), so the first public
        # optimizer.step() uses zero LR and scheduler.step() prepares step 1.
        lr = 1e-4 * cosine_schedule(
            state.step, warmup_steps=500, total_steps=total_schedule_steps
        )
        updates = jax.tree.map(lambda value: value * lr, updates)
        params = optax.apply_updates(state.params, updates)
        decay = ema_decay(state.step)
        ema_params = jax.tree.map(
            lambda old, new: old * decay + new * (1.0 - decay),
            state.ema_params,
            params,
        )
        metrics = {**metrics, "lr": lr, "ema_decay": decay}
        return (
            TrainState(
                params, ema_params, state.model_state, opt_state, state.step + 1
            ),
            jax.tree.map(jnp.add, metric_totals, metrics),
        )

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_config = {
        "schema": "official_a2a_jax_v1",
        "dataset": str(Path(args.dataset).expanduser().resolve()),
        "model": asdict(config),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "dataloader_seed": 0,
        "max_train_steps": args.max_train_steps,
        "batches_per_epoch": batches_per_epoch,
        "lr_schedule_steps": total_schedule_steps,
        "official_source_commit": "596f6220f87734c39dd1e7598bda05b83690a3f7",
        "torch_dependency": False,
        "cameras": list(args.cameras),
        "pretrained_resnet_weights": (
            None
            if args.pretrained_resnet_weights is None
            else str(Path(args.pretrained_resnet_weights).expanduser().resolve())
        ),
    }
    (output / "config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    log_path = output / "train.jsonl"
    # create_dataloader does not forward the training seed; its custom
    # BatchSampler therefore uses its own public default seed of zero.
    rng = np.random.default_rng(0)
    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        order = rng.permutation(len(dataset.train_indices))
        max_steps = (
            batches_per_epoch
            if args.max_train_steps <= 0
            else min(args.max_train_steps, batches_per_epoch)
        )
        totals = {name: jnp.asarray(0.0, dtype=jnp.float32) for name in metric_names}
        for batch_index in range(max_steps):
            selection = order[
                batch_index * args.batch_size : (batch_index + 1) * args.batch_size
            ]
            batch = jax.device_put(dataset.batch(selection))
            train_key, step_key = jax.random.split(train_key)
            state, totals = update(state, totals, batch, step_key)
        host_totals = jax.device_get(totals)
        record = {
            "epoch": epoch,
            "global_step": int(state.step),
            "seconds": time.monotonic() - started,
            **{name: float(value) / max_steps for name, value in host_totals.items()},
        }
        with log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)
        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            save_checkpoint(
                output / "checkpoints",
                state,
                epoch=epoch,
                config=run_config,
                dataset=dataset,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-steps", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=30)
    parser.add_argument(
        "--flow-matcher",
        choices=("conditional", "exact_ot"),
        default="conditional",
    )
    parser.add_argument("--observation-steps", type=int, default=8)
    parser.add_argument("--history-steps", type=int, default=8)
    parser.add_argument("--action-steps", type=int, default=8)
    parser.add_argument("--flow-steps", type=int, default=6)
    parser.add_argument("--cameras", nargs="+", default=["head"])
    parser.add_argument(
        "--vision-encoder",
        choices=("official_gn", "fm_resnet"),
        default="official_gn",
    )
    parser.add_argument("--pretrained-resnet-weights", type=Path, default=None)
    parser.add_argument(
        "--image-range-normalization",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--resize-to-224",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--no-image-cache", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
