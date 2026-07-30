#!/usr/bin/env python3
"""Compose a frozen-train CQN branch cache with an external evaluation cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


_ARRAY_NAMES = ("features", "actions", "returns", "action_dimensions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True, type=Path)
    parser.add_argument("--evaluation-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _load_split(path: Path, split: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        result = {
            name: np.asarray(payload[f"{split}_{name}"])
            for name in _ARRAY_NAMES
        }
        result["metadata"] = json.loads(
            str(np.asarray(payload[f"{split}_metadata"]).item())
        )
        sample_key = f"{split}_return_samples"
        result["return_samples"] = (
            np.asarray(payload[sample_key])
            if sample_key in payload
            else result["returns"][..., None]
        )
    if len(result["metadata"]) != result["features"].shape[0]:
        raise ValueError(f"{path}:{split} metadata count differs")
    return result


def _concatenate(parts: list[dict[str, Any]]) -> dict[str, Any]:
    shapes = [
        {
            name: tuple(np.asarray(part[name]).shape[1:])
            for name in (*_ARRAY_NAMES, "return_samples")
        }
        for part in parts
    ]
    if any(shape != shapes[0] for shape in shapes[1:]):
        raise ValueError(f"external evaluation shapes differ: {shapes}")
    result = {
        name: np.concatenate(
            [np.asarray(part[name]) for part in parts],
            axis=0,
        )
        for name in (*_ARRAY_NAMES, "return_samples")
    }
    result["metadata"] = [
        record for part in parts for record in part["metadata"]
    ]
    return result


def _seed_ids(data: dict[str, Any]) -> list[int]:
    seeds = sorted({int(record["eval_seed"]) for record in data["metadata"]})
    if not seeds:
        raise ValueError("cache split contains no simulator seeds")
    return seeds


def compose_external_eval_cache(
    *,
    train_cache: Path,
    evaluation_cache: Path,
    output: Path,
) -> dict[str, Any]:
    train_cache = train_cache.expanduser().resolve()
    evaluation_cache = evaluation_cache.expanduser().resolve()
    for path in (train_cache, evaluation_cache):
        if not path.is_file():
            raise FileNotFoundError(path)
    train = _load_split(train_cache, "train")
    heldout = _concatenate(
        [
            _load_split(evaluation_cache, "train"),
            _load_split(evaluation_cache, "heldout"),
        ]
    )
    train_shapes = {
        name: tuple(np.asarray(train[name]).shape[1:])
        for name in (*_ARRAY_NAMES, "return_samples")
    }
    heldout_shapes = {
        name: tuple(np.asarray(heldout[name]).shape[1:])
        for name in (*_ARRAY_NAMES, "return_samples")
    }
    if train_shapes != heldout_shapes:
        raise ValueError(
            f"train/evaluation array shapes differ: "
            f"{train_shapes} != {heldout_shapes}"
        )
    train_seeds = _seed_ids(train)
    heldout_seeds = _seed_ids(heldout)
    overlap = set(train_seeds).intersection(heldout_seeds)
    if overlap:
        raise ValueError(f"train/evaluation seeds overlap: {sorted(overlap)}")

    metadata = {
        "composition": "frozen_train_with_external_evaluation",
        "train_cache": str(train_cache),
        "evaluation_cache": str(evaluation_cache),
        "train_seeds": train_seeds,
        "heldout_seeds": heldout_seeds,
    }
    arrays: dict[str, Any] = {
        "cache_metadata": np.asarray(json.dumps(metadata, sort_keys=True))
    }
    for prefix, data in (("train", train), ("heldout", heldout)):
        for name in (*_ARRAY_NAMES, "return_samples"):
            arrays[f"{prefix}_{name}"] = data[name]
        arrays[f"{prefix}_metadata"] = np.asarray(
            json.dumps(data["metadata"])
        )

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    return {
        "status": "ok",
        "output": str(output),
        "train_seeds": train_seeds,
        "heldout_seeds": heldout_seeds,
        "num_train_records": int(train["features"].shape[0]),
        "num_heldout_records": int(heldout["features"].shape[0]),
    }


def main() -> None:
    args = parse_args()
    result = compose_external_eval_cache(
        train_cache=args.train_cache,
        evaluation_cache=args.evaluation_cache,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
