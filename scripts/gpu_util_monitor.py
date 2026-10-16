"""Run a command while sampling one GPU's utilization every 100 ms.

    python scripts/gpu_util_monitor.py --gpu 5 --out util.csv -- <command ...>
    python scripts/gpu_util_monitor.py --summarize util.csv --skip 60

Use CUDA_DEVICE_ORDER=PCI_BUS_ID for the command so CUDA and nvidia-smi
agree on the index.  Summary statistics cover the samples after ``--skip``
seconds (startup/compilation) up to ``--tail`` seconds before the end.
"""

import argparse
import csv
import subprocess
import sys
import time

import numpy as np


def summarize(path: str, skip: float = 0.0, tail: float = 0.0) -> dict:
    with open(path) as f:
        rows = [
            (float(r["t"]), float(r["util"]), float(r["mem_mib"]))
            for r in csv.DictReader(f)
        ]
    if not rows:
        return {}
    t = np.array([r[0] for r in rows])
    util = np.array([r[1] for r in rows])
    keep = (t >= t[0] + skip) & (t <= t[-1] - tail)
    u = util[keep]
    if u.size == 0:
        return {}
    return {
        "samples": int(u.size),
        "seconds": float(t[keep][-1] - t[keep][0]),
        "mean": float(u.mean()),
        "std": float(u.std()),
        "p1": float(np.percentile(u, 1)),
        "p5": float(np.percentile(u, 5)),
        "median": float(np.median(u)),
        "frac_below_80": float((u < 80).mean()),
        "frac_below_50": float((u < 50).mean()),
        "peak_mem_mib": float(np.array([r[2] for r in rows]).max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--out")
    parser.add_argument("--interval-ms", type=int, default=100)
    parser.add_argument("--summarize")
    parser.add_argument("--skip", type=float, default=0.0)
    parser.add_argument("--tail", type=float, default=0.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.summarize:
        print(summarize(args.summarize, args.skip, args.tail))
        return 0
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    sampler = subprocess.Popen(
        [
            "nvidia-smi",
            "-i",
            str(args.gpu),
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
            "-lms",
            str(args.interval_ms),
        ],
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "util", "mem_mib"])
        proc = subprocess.Popen(command)
        try:
            while proc.poll() is None:
                line = sampler.stdout.readline()
                if not line:
                    break
                util, mem = (v.strip() for v in line.split(","))
                writer.writerow([f"{time.time():.3f}", util, mem])
        finally:
            sampler.terminate()
            code = proc.wait()
    print("gpu_util", summarize(args.out, args.skip, args.tail), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
