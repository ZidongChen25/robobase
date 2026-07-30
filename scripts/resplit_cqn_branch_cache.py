#!/usr/bin/env python3
"""Repartition an existing branch cache by simulator seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.merge_cqn_branch_caches import (
        CacheSplit,
        _ARRAY_NAMES,
        _merge_split,
    )
except ModuleNotFoundError:
    from merge_cqn_branch_caches import (
        CacheSplit,
        _ARRAY_NAMES,
        _merge_split,
    )


def _integer_list(value: str) -> list[int]:
    result = sorted({int(item.strip()) for item in value.split(",") if item})
    if not result:
        raise argparse.ArgumentTypeError("expected at least one simulator seed")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--train-seeds", required=True, type=_integer_list)
    parser.add_argument("--heldout-seeds", required=True, type=_integer_list)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _subset(data: dict[str, Any], seeds: set[int]) -> dict[str, Any]:
    record_seeds = np.asarray(
        [int(record["eval_seed"]) for record in data["metadata"]],
        dtype=np.int64,
    )
    mask = np.isin(record_seeds, np.asarray(sorted(seeds), np.int64))
    result = {
        name: np.asarray(data[name])[mask]
        for name in (*_ARRAY_NAMES, "return_samples")
    }
    result["metadata"] = [
        record
        for record, keep in zip(data["metadata"], mask, strict=True)
        if bool(keep)
    ]
    return result


def resplit_cache(
    *,
    input_path: Path,
    train_seeds: list[int],
    heldout_seeds: list[int],
    output: Path,
) -> dict[str, Any]:
    input_path = input_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    train_seed_set = set(train_seeds)
    heldout_seed_set = set(heldout_seeds)
    overlap = train_seed_set.intersection(heldout_seed_set)
    if overlap:
        raise ValueError(
            f"train and heldout simulator seeds overlap: {sorted(overlap)}"
        )

    all_data, protocol, info = _merge_split(
        [
            CacheSplit(input_path, "train"),
            CacheSplit(input_path, "heldout"),
        ]
    )
    available = set(info["seeds"])
    requested = train_seed_set.union(heldout_seed_set)
    if requested != available:
        raise ValueError(
            "requested seeds must partition every input seed exactly; "
            f"missing={sorted(available - requested)}, "
            f"unknown={sorted(requested - available)}"
        )
    train = _subset(all_data, train_seed_set)
    heldout = _subset(all_data, heldout_seed_set)
    cache_metadata = {
        **protocol,
        "train_seeds": sorted(train_seed_set),
        "heldout_seeds": sorted(heldout_seed_set),
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
        "input": str(input_path),
        "output": str(output),
        "train_seeds": sorted(train_seed_set),
        "heldout_seeds": sorted(heldout_seed_set),
        "num_train_records": int(train["features"].shape[0]),
        "num_heldout_records": int(heldout["features"].shape[0]),
    }


def main() -> None:
    args = parse_args()
    result = resplit_cache(
        input_path=args.input,
        train_seeds=args.train_seeds,
        heldout_seeds=args.heldout_seeds,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
