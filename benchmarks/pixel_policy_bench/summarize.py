"""Aggregate worker JSONL results into a markdown table.

Usage: python benchmarks/pixel_policy_bench/summarize.py results/*.jsonl
Rows with the same (backend, label, method, batch, fused, aug, tf32) are
merged: throughput is the median over repeats, memory the max.
"""

from __future__ import annotations

import collections
import json
import statistics
import sys


def _key(row):
    return (
        row["backend"],
        row.get("label", ""),
        row["method"],
        int(row["batch_size"]),
        int(row.get("fused_steps", 1)),
        # Only ACT has an image augmentation pipeline; the flag is meaningless
        # for the other methods and would split identical configurations.
        bool(row.get("augmentation", False)) if row["method"] == "act" else False,
        bool(row.get("tf32_matmul", False)),
    )


def _memory_mib(row):
    if row["backend"] == "jax":
        return row["peak_bytes_in_use_mib"]
    return row["max_memory_allocated_mib"]


def main(paths):
    groups = collections.defaultdict(list)
    for path in paths:
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if line.startswith("{"):
                    row = json.loads(line)
                    groups[_key(row)].append(row)
    header = (
        "| backend | label | method | batch | fused | aug | tf32 | n | "
        "upd/s (median) | ms/update | samples/s | peak MiB | footprint MiB |"
    )
    print(header)
    print("|" + "---|" * 13)
    for key in sorted(groups):
        rows = groups[key]
        ups = statistics.median(r["throughput_updates_per_second"] for r in rows)
        ms = statistics.median(r["throughput_ms_per_update"] for r in rows)
        sps = statistics.median(r["throughput_samples_per_second"] for r in rows)
        peak = max(_memory_mib(r) for r in rows)
        foot = max(r.get("nvidia_smi_peak_mib", 0) for r in rows)
        backend, label, method, batch, fused, aug, tf32 = key
        print(
            f"| {backend} | {label} | {method} | {batch} | {fused} | {aug} | {tf32} | "
            f"{len(rows)} | {ups:.2f} | {ms:.1f} | {sps:.0f} | {peak:.0f} | {foot:.0f} |"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
