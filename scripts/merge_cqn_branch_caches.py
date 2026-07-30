#!/usr/bin/env python3
"""Merge disjoint CQN simulator-branch cache splits.

The branch collector stores one ``train`` and one ``heldout`` split per NPZ.
This utility promotes any number of already-collected splits into a larger
training split while keeping one or more disjoint splits for validation/test.
Only caches collected from the same snapshot and intervention protocol may be
merged.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


_ARRAY_NAMES = (
    "features",
    "actions",
    "returns",
    "action_dimensions",
)
_SEED_FIELDS = frozenset({"train_seeds", "heldout_seeds"})


@dataclass(frozen=True)
class CacheSplit:
    path: Path
    split: str


def _parse_source(value: str) -> CacheSplit:
    try:
        path_text, split = value.rsplit(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "cache source must have the form PATH:train or PATH:heldout"
        ) from exc
    if split not in {"train", "heldout"}:
        raise argparse.ArgumentTypeError(
            "cache source split must be 'train' or 'heldout'"
        )
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"cache does not exist: {path}")
    return CacheSplit(path=path, split=split)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-source",
        action="append",
        required=True,
        type=_parse_source,
        help="Repeatable PATH:train or PATH:heldout source for merged train.",
    )
    parser.add_argument(
        "--heldout-source",
        action="append",
        required=True,
        type=_parse_source,
        help="Repeatable PATH:train or PATH:heldout source for heldout.",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _load_split(source: CacheSplit) -> tuple[dict[str, Any], dict[str, Any]]:
    with np.load(source.path, allow_pickle=False) as payload:
        prefix = source.split
        data = {
            name: np.asarray(payload[f"{prefix}_{name}"])
            for name in _ARRAY_NAMES
        }
        data["metadata"] = json.loads(
            str(payload[f"{prefix}_metadata"].item())
        )
        samples_name = f"{prefix}_return_samples"
        data["return_samples"] = (
            np.asarray(payload[samples_name])
            if samples_name in payload
            else data["returns"][..., None]
        )
        cache_metadata = json.loads(str(payload["cache_metadata"].item()))
    return data, cache_metadata


def _protocol_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if key not in _SEED_FIELDS
    }


def _record_seeds(records: Iterable[dict[str, Any]]) -> list[int]:
    seeds = sorted({int(record["eval_seed"]) for record in records})
    if not seeds:
        raise ValueError("branch cache split contains no eval_seed records")
    return seeds


def _merge_split(
    sources: list[CacheSplit],
    *,
    expected_protocol: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    datasets: list[dict[str, Any]] = []
    protocol = expected_protocol
    observed_seeds: set[int] = set()
    reference_shapes: dict[str, tuple[int, ...]] | None = None
    for source in sources:
        data, metadata = _load_split(source)
        current_protocol = _protocol_metadata(metadata)
        if protocol is None:
            protocol = current_protocol
        elif current_protocol != protocol:
            raise ValueError(
                f"branch protocol mismatch for {source.path}:{source.split}"
            )
        if len(data["metadata"]) != data["features"].shape[0]:
            raise ValueError(
                f"metadata/feature length mismatch in "
                f"{source.path}:{source.split}"
            )
        seeds = set(_record_seeds(data["metadata"]))
        overlap = observed_seeds.intersection(seeds)
        if overlap:
            raise ValueError(
                f"duplicate simulator seeds across merged sources: "
                f"{sorted(overlap)}"
            )
        observed_seeds.update(seeds)
        shapes = {
            name: tuple(np.asarray(data[name]).shape[1:])
            for name in (*_ARRAY_NAMES, "return_samples")
        }
        if reference_shapes is None:
            reference_shapes = shapes
        elif shapes != reference_shapes:
            raise ValueError(
                f"array shape mismatch for {source.path}:{source.split}: "
                f"expected {reference_shapes}, found {shapes}"
            )
        datasets.append(data)
    assert protocol is not None
    merged = {
        name: np.concatenate(
            [np.asarray(dataset[name]) for dataset in datasets],
            axis=0,
        )
        for name in (*_ARRAY_NAMES, "return_samples")
    }
    merged["metadata"] = [
        record
        for dataset in datasets
        for record in dataset["metadata"]
    ]
    return merged, protocol, {"seeds": sorted(observed_seeds)}


def merge_caches(
    *,
    train_sources: list[CacheSplit],
    heldout_sources: list[CacheSplit],
    output: Path,
) -> dict[str, Any]:
    train, protocol, train_info = _merge_split(train_sources)
    heldout, _, heldout_info = _merge_split(
        heldout_sources,
        expected_protocol=protocol,
    )
    train_seeds = set(train_info["seeds"])
    heldout_seeds = set(heldout_info["seeds"])
    overlap = train_seeds.intersection(heldout_seeds)
    if overlap:
        raise ValueError(
            f"train and heldout simulator seeds overlap: {sorted(overlap)}"
        )

    cache_metadata = {
        **protocol,
        "train_seeds": sorted(train_seeds),
        "heldout_seeds": sorted(heldout_seeds),
    }
    arrays: dict[str, Any] = {
        "cache_metadata": np.asarray(
            json.dumps(cache_metadata, sort_keys=True)
        )
    }
    for prefix, data in (("train", train), ("heldout", heldout)):
        for name in _ARRAY_NAMES:
            arrays[f"{prefix}_{name}"] = data[name]
        arrays[f"{prefix}_metadata"] = np.asarray(
            json.dumps(data["metadata"])
        )
        arrays[f"{prefix}_return_samples"] = data["return_samples"]

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    return {
        "status": "ok",
        "output": str(output),
        "train_seeds": sorted(train_seeds),
        "heldout_seeds": sorted(heldout_seeds),
        "num_train_records": int(train["features"].shape[0]),
        "num_heldout_records": int(heldout["features"].shape[0]),
        "protocol": protocol,
    }


def main() -> None:
    args = parse_args()
    result = merge_caches(
        train_sources=args.train_source,
        heldout_sources=args.heldout_source,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
