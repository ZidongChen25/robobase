#!/usr/bin/env python3
"""Create a seed-disjoint train/validation cache from one branch-cache split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _integer_list(value: str) -> list[int]:
    try:
        result = [int(item) for item in value.split(",") if item]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError(
            "seed list must be nonempty and unique"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument(
        "--train-seeds",
        required=True,
        type=_integer_list,
    )
    parser.add_argument(
        "--heldout-seeds",
        required=True,
        type=_integer_list,
    )
    return parser.parse_args()


def split_cache(
    source: Path,
    output: Path,
    *,
    train_seeds: list[int],
    heldout_seeds: list[int],
) -> dict:
    if set(train_seeds) & set(heldout_seeds):
        raise ValueError("train and heldout seeds must be disjoint")
    with np.load(source, allow_pickle=False) as payload:
        source_metadata = json.loads(
            str(payload["cache_metadata"].item())
        )
        source_records = json.loads(
            str(payload["train_metadata"].item())
        )
        source_seed_set = {
            int(record["eval_seed"]) for record in source_records
        }
        requested = set(train_seeds) | set(heldout_seeds)
        missing = requested - source_seed_set
        if missing:
            raise ValueError(
                f"requested seeds are absent from source train split: "
                f"{sorted(missing)}"
            )
        seed_by_row = np.asarray(
            [int(record["eval_seed"]) for record in source_records],
            dtype=np.int64,
        )
        masks = {
            "train": np.isin(seed_by_row, train_seeds),
            "heldout": np.isin(seed_by_row, heldout_seeds),
        }
        arrays = {}
        for destination, mask in masks.items():
            records = [
                record
                for record, keep in zip(source_records, mask, strict=True)
                if keep
            ]
            arrays[f"{destination}_features"] = np.asarray(
                payload["train_features"]
            )[mask]
            arrays[f"{destination}_actions"] = np.asarray(
                payload["train_actions"]
            )[mask]
            arrays[f"{destination}_returns"] = np.asarray(
                payload["train_returns"]
            )[mask]
            arrays[f"{destination}_action_dimensions"] = np.asarray(
                payload["train_action_dimensions"],
                dtype=np.int32,
            )[mask]
            arrays[f"{destination}_metadata"] = np.asarray(
                json.dumps(records)
            )
            if "train_return_samples" in payload:
                arrays[f"{destination}_return_samples"] = np.asarray(
                    payload["train_return_samples"]
                )[mask]
        metadata = dict(source_metadata)
        metadata["train_seeds"] = list(train_seeds)
        metadata["heldout_seeds"] = list(heldout_seeds)
        arrays["cache_metadata"] = np.asarray(
            json.dumps(metadata, sort_keys=True)
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    summary = {
        "status": "ok",
        "source": str(source),
        "output": str(output),
        "train_seeds": list(train_seeds),
        "heldout_seeds": list(heldout_seeds),
        "num_train_states": int(arrays["train_features"].shape[0]),
        "num_heldout_states": int(arrays["heldout_features"].shape[0]),
    }
    return summary


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    summary = split_cache(
        source,
        output,
        train_seeds=args.train_seeds,
        heldout_seeds=args.heldout_seeds,
    )
    if args.summary is not None:
        summary_path = args.summary.expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
