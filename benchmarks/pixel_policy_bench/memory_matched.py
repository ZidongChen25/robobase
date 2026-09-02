"""Throughput at matched peak memory, from worker JSONL results.

For every method it lists the (peak MiB, samples/s) points of each backend
variant and, for each PyTorch point, the JAX throughput at that memory
obtained by linear interpolation between the JAX batch-size points.

Usage: python memory_matched.py results/*.jsonl [--label-prefix excl_]
"""

from __future__ import annotations

import collections
import json
import statistics
import sys


def _mem(row):
    if row["backend"] == "jax":
        return row["peak_bytes_in_use_mib"]
    return row["max_memory_allocated_mib"]


def _variant(row):
    if row["backend"] == "torch":
        return "torch_tf32" if row.get("tf32_matmul") else "torch"
    dtype = "bf16" if "bf16" in row.get("label", "") else "f32"
    return f"jax_{dtype}"


def main(argv):
    prefix = None
    paths = []
    for arg in argv:
        if arg.startswith("--label-prefix="):
            prefix = arg.split("=", 1)[1]
        else:
            paths.append(arg)
    points = collections.defaultdict(lambda: collections.defaultdict(list))
    for path in paths:
        for line in open(path):
            if not line.startswith("{"):
                continue
            row = json.loads(line)
            if prefix and not row.get("label", "").startswith(prefix):
                continue
            if row["method"] == "act" and row.get("augmentation") and row["backend"] == "jax":
                continue  # torch ACT has no augmentation; compare like with like
            if int(row.get("fused_steps", 1)) != 1:
                continue
            key = (row["method"], _variant(row), int(row["batch_size"]))
            points[row["method"]][key].append(row)
    for method in sorted(points):
        print(f"\n## {method}")
        curves = collections.defaultdict(list)
        for (_, variant, batch), rows in sorted(points[method].items()):
            mem = max(_mem(r) for r in rows)
            sps = statistics.median(r["throughput_samples_per_second"] for r in rows)
            curves[variant].append((mem, sps, batch))
            print(f"{variant:11s} batch {batch:4d}: peak {mem:7.0f} MiB  {sps:7.0f} samples/s  (n={len(rows)})")
        for torch_variant in ("torch", "torch_tf32"):
            for mem_t, sps_t, batch_t in curves.get(torch_variant, []):
                for jax_variant in ("jax_f32", "jax_bf16"):
                    pts = sorted(curves.get(jax_variant, []))
                    if len(pts) < 2:
                        continue
                    (m0, s0, b0), (m1, s1, b1) = pts[0], pts[-1]
                    if not (m0 <= mem_t <= m1):
                        note = "extrapolated"
                    else:
                        note = "interpolated"
                    est = s0 + (s1 - s0) * (mem_t - m0) / (m1 - m0)
                    print(
                        f"  at {torch_variant} batch {batch_t} memory ({mem_t:.0f} MiB): "
                        f"{jax_variant} ~{est:.0f} samples/s vs {torch_variant} {sps_t:.0f} "
                        f"-> {est / sps_t:.2f}x ({note} between JAX batch {b0} and {b1})"
                    )


if __name__ == "__main__":
    main(sys.argv[1:])
