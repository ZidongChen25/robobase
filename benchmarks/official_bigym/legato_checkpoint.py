"""Checkpoint IO and official vanilla-to-Legato core initialization."""

from __future__ import annotations

import copy
from dataclasses import asdict
import os
from pathlib import Path
import pickle
from typing import Any

from flax import nnx, traverse_util
import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.official_bigym.legato_adapter import OfficialBigymPolicy
from benchmarks.official_bigym.legato_upstream import UPSTREAM_COMMIT


CHECKPOINT_FORMAT = "official_legato_bigym_v1"


def checkpoint_payload(
    adapter: OfficialBigymPolicy,
    *,
    step: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if step < 0:
        raise ValueError("step must be non-negative.")
    return {
        "format": CHECKPOINT_FORMAT,
        "upstream_commit": UPSTREAM_COMMIT,
        "mode": adapter.mode,
        "obs_dim": adapter.obs_dim,
        "action_dim": adapter.action_dim,
        "config": asdict(adapter.config),
        "step": int(step),
        "policy_state": jax.device_get(nnx.state(adapter.policy).to_pure_dict()),
        "extra": {} if extra is None else dict(extra),
    }


def save_checkpoint(
    path: str | Path,
    adapter: OfficialBigymPolicy,
    *,
    step: int,
    extra: dict[str, Any] | None = None,
) -> None:
    """Atomically save model state plus enough metadata to reject mix-ups."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as file:
        pickle.dump(
            checkpoint_payload(adapter, step=step, extra=extra),
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def _load_payload(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as file:
        payload = pickle.load(file)
    if not isinstance(payload, dict):
        raise ValueError("Official Legato checkpoint must contain a dictionary.")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported checkpoint format: {payload.get('format')!r}.")
    if payload.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("Checkpoint was produced by a different upstream revision.")
    return payload


def read_checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    """Read validated checkpoint metadata without returning the parameter tree."""
    payload = _load_payload(path)
    return {key: value for key, value in payload.items() if key != "policy_state"}


def load_checkpoint(
    path: str | Path,
    adapter: OfficialBigymPolicy,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Restore an adapter after validating mode, dimensions, and configuration."""
    payload = _load_payload(path)
    expected = {
        "obs_dim": adapter.obs_dim,
        "action_dim": adapter.action_dim,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    compatible_modes = (
        {"vanilla", "rtc"}
        if adapter.mode in {"vanilla", "rtc"}
        else {"legato"}
    )
    if payload.get("mode") not in compatible_modes:
        mismatches["mode"] = (payload.get("mode"), adapter.mode)
    if strict and payload.get("config") != asdict(adapter.config):
        mismatches["config"] = (payload.get("config"), asdict(adapter.config))
    if mismatches:
        raise ValueError(f"Checkpoint metadata does not match adapter: {mismatches}.")

    graphdef, state = nnx.split(adapter.policy)
    try:
        state.replace_by_pure_dict(payload["policy_state"])
    except Exception as exc:
        raise ValueError("Checkpoint state is incompatible with the policy.") from exc
    adapter.policy = nnx.merge(graphdef, state)
    return payload


def warm_start_legato_from_vanilla(
    vanilla: OfficialBigymPolicy,
    legato: OfficialBigymPolicy,
) -> None:
    """Insert a zero schedule row and copy all official shared core parameters.

    The official model concatenates ``[action, schedule, observation]`` for
    Legato and ``[action, observation]`` for vanilla. Zero-initializing the new
    schedule row preserves the vanilla function before Legato fine-tuning.
    """
    if vanilla.mode not in {"vanilla", "rtc"} or legato.mode != "legato":
        raise ValueError("Expected a vanilla/RTC source and a Legato destination.")
    if (vanilla.obs_dim, vanilla.action_dim) != (legato.obs_dim, legato.action_dim):
        raise ValueError("Source and destination observation/action dimensions differ.")
    architecture_fields = (
        "action_horizon",
        "channel_dim",
        "channel_hidden_dim",
        "token_hidden_dim",
        "num_layers",
    )
    if any(
        getattr(vanilla.config, name) != getattr(legato.config, name)
        for name in architecture_fields
    ):
        raise ValueError("Source and destination core architectures differ.")

    source = traverse_util.flatten_dict(nnx.state(vanilla.policy).to_pure_dict())
    target_nested = copy.deepcopy(nnx.state(legato.policy).to_pure_dict())
    target = traverse_util.flatten_dict(target_nested)
    if source.keys() != target.keys():
        raise ValueError("Official vanilla and Legato state trees do not align.")

    input_kernel_key = ("in_proj", "kernel")
    converted = {}
    for key, target_value in target.items():
        source_value = jnp.asarray(source[key])
        target_value = jnp.asarray(target_value)
        if key == input_kernel_key:
            split = vanilla.action_dim
            source_value = jnp.concatenate(
                [
                    source_value[:split],
                    jnp.zeros((1, source_value.shape[1]), dtype=source_value.dtype),
                    source_value[split:],
                ],
                axis=0,
            )
        if source_value.shape != target_value.shape:
            raise ValueError(
                f"State shape mismatch at {key}: {source_value.shape} != "
                f"{target_value.shape}."
            )
        converted[key] = source_value.astype(target_value.dtype)

    graphdef, state = nnx.split(legato.policy)
    state.replace_by_pure_dict(traverse_util.unflatten_dict(converted))
    legato.policy = nnx.merge(graphdef, state)


def state_max_abs_difference(
    first: OfficialBigymPolicy, second: OfficialBigymPolicy
) -> float:
    """Small checkpoint diagnostic used by smoke tests and reports."""
    first_leaves = jax.tree.leaves(nnx.state(first.policy).to_pure_dict())
    second_leaves = jax.tree.leaves(nnx.state(second.policy).to_pure_dict())
    if len(first_leaves) != len(second_leaves):
        return float("inf")
    differences = []
    for left, right in zip(first_leaves, second_leaves, strict=True):
        if left.shape != right.shape:
            return float("inf")
        differences.append(np.max(np.abs(np.asarray(left) - np.asarray(right))))
    return float(max(differences, default=0.0))


__all__ = [
    "CHECKPOINT_FORMAT",
    "checkpoint_payload",
    "load_checkpoint",
    "read_checkpoint_metadata",
    "save_checkpoint",
    "state_max_abs_difference",
    "warm_start_legato_from_vanilla",
]
