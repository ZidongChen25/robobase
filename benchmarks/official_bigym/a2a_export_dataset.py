#!/usr/bin/env python3
"""Export reset-aligned BiGym demonstrations for the official A2A trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.official_bigym.bigym_data import (
    discover_demo_files,
    export_official_zarr,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Reset-aligned BiGym pixel cache containing .safetensors episodes.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", default="head")
    parser.add_argument("--cameras", nargs="+", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--include-failed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--observation-timing",
        choices=("pre_action", "post_action"),
        default="pre_action",
    )
    parser.add_argument("--control-frequency-hz", type=int, default=20)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_episodes is not None and args.max_episodes < 1:
        raise ValueError("--max-episodes must be positive.")
    manifest = export_official_zarr(
        discover_demo_files(args.cache_dir),
        args.output,
        camera=args.camera,
        cameras=args.cameras,
        successful_only=not args.include_failed,
        max_episodes=args.max_episodes,
        overwrite=args.overwrite,
        observation_timing=args.observation_timing,
        control_frequency_hz=args.control_frequency_hz,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
