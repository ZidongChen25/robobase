#!/usr/bin/env python3
"""Replace causal evidence and recompute the strict A/B research summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.summarize_cqn_autoresearch_routes import (
        _base_candidates,
        _load,
        summarize_multi,
    )
except ModuleNotFoundError:
    from summarize_cqn_autoresearch_routes import (
        _base_candidates,
        _load,
        summarize_multi,
    )


def _override(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "causal override must be LABEL=CAUSAL_JSON"
        )
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError(
            "causal override must be LABEL=CAUSAL_JSON"
        )
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-summary", required=True, type=Path)
    parser.add_argument(
        "--causal-override",
        action="append",
        default=[],
        type=_override,
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = _load(args.base_summary)
    routes = {
        "route_a": _base_candidates(base, "route_a"),
        "route_b": _base_candidates(base, "route_b"),
    }
    overrides = {}
    for label, path in args.causal_override:
        if label in overrides:
            raise ValueError(f"duplicate causal override: {label}")
        overrides[label] = _load(path)

    known_labels = {
        label
        for candidates in routes.values()
        for label, _, _ in candidates
    }
    unknown = sorted(set(overrides) - known_labels)
    if unknown:
        raise ValueError(f"causal overrides have unknown labels: {unknown}")

    def replace(candidates):
        return [
            (
                label,
                task,
                overrides.get(label, causal),
            )
            for label, task, causal in candidates
        ]

    payload = summarize_multi(
        replace(routes["route_a"]),
        replace(routes["route_b"]),
    )
    payload["strict_revalidation"] = {
        "base_summary": str(args.base_summary.expanduser().resolve()),
        "causal_overrides": {
            label: evidence["_path"]
            for label, evidence in overrides.items()
        },
        "required_dimension_selection": "round_robin",
        "selection_use_forbidden": True,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
