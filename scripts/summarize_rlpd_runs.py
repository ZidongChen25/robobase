"""Summarize RLPD seed runs: eval curves, throughput and GPU utilisation.

    python scripts/summarize_rlpd_runs.py \
        --group "D4RL umaze-v2=exp_local/rlpd_d4rl/antmaze_umaze_v2_20261016" \
        --group "Minari umaze-v2=exp_local/rlpd_minari/umaze_v2_20261016" \
        --out reports/rlpd_antmaze

Each group directory holds ``seed*/{eval.csv,train.csv,gpu_util.csv}`` as
written by ``scripts/run_rlpd_seeds.sh``.  Writes ``<out>.json`` and
``<out>.png`` (mean +- std normalized score over seeds).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_util_monitor import summarize as summarize_gpu  # noqa: E402

# Read off Fig. 14 (AntMaze Umaze, RLPD, 10 seeds) of Ball et al. 2023; the
# paper gives no table, so these are approximate visual readings.
PAPER_UMAZE_APPROX = {0: 0.0, 25_000: 75.0, 50_000: 95.0, 100_000: 98.0, 300_000: 99.0}


def _rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def summarize_seed(seed_dir: Path, steady_after: int) -> dict:
    evals = _rows(seed_dir / "eval.csv")
    steps = np.array([float(r["env_steps"]) for r in evals])
    score = np.array([float(r["normalized_score"]) for r in evals])
    order = np.argsort(steps)
    train = _rows(seed_dir / "train.csv")
    steady = [r for r in train if float(r["env_steps"]) > steady_after]
    fps = np.array([float(r["env_steps_per_second"]) for r in steady])
    profile = {
        key.removeprefix("profile_ms_"): float(np.mean([float(r[key]) for r in steady]))
        for key in train[-1]
        if key.startswith("profile_ms_") and steady
    }
    out = {
        "seed_dir": str(seed_dir),
        "eval_steps": steps[order].astype(int).tolist(),
        "normalized_score": score[order].tolist(),
        "final_env_steps": int(float(train[-1]["env_steps"])),
        "wall_seconds_online_loop": float(train[-1]["total_time"]),
        "steady_env_steps_per_second_median": float(np.median(fps))
        if fps.size
        else None,
        "steady_env_steps_per_second_p5": float(np.percentile(fps, 5))
        if fps.size
        else None,
        "profile_ms_per_step": profile,
    }
    gpu_csv = seed_dir / "gpu_util.csv"
    if gpu_csv.exists():
        out["gpu_util"] = summarize_gpu(str(gpu_csv), skip=120.0, tail=5.0)
    return out


def summarize_group(name: str, root: Path, steady_after: int) -> dict:
    seeds = [
        summarize_seed(d, steady_after)
        for d in sorted(root.glob("seed*"))
        if (d / "eval.csv").exists() and (d / "train.csv").exists()
    ]
    common = (
        sorted(set.intersection(*(set(s["eval_steps"]) for s in seeds)))
        if seeds
        else []
    )
    table = np.array(
        [
            [dict(zip(s["eval_steps"], s["normalized_score"]))[t] for t in common]
            for s in seeds
        ]
    )
    curve = {
        "env_steps": common,
        "mean": table.mean(0).tolist() if seeds else [],
        "std": table.std(0).tolist() if seeds else [],
    }

    def at(step):
        if step not in common:
            return None
        values = table[:, common.index(step)]
        return {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "per_seed": values.tolist(),
        }

    last3 = table[:, -3:].mean(1) if seeds and len(common) >= 3 else np.array([])
    gpu = [s["gpu_util"] for s in seeds if s.get("gpu_util")]
    fps = [
        s["steady_env_steps_per_second_median"]
        for s in seeds
        if s["steady_env_steps_per_second_median"]
    ]
    return {
        "name": name,
        "root": str(root),
        "num_seeds": len(seeds),
        "curve": curve,
        "score_at": {
            str(k): at(k) for k in (25_000, 50_000, 100_000, 200_000, 300_000)
        },
        "final_last3_mean": float(last3.mean()) if last3.size else None,
        "final_last3_std": float(last3.std()) if last3.size else None,
        "steady_env_steps_per_second_median_over_seeds": float(np.median(fps))
        if fps
        else None,
        "wall_minutes_mean": float(
            np.mean([s["wall_seconds_online_loop"] for s in seeds]) / 60
        )
        if seeds
        else None,
        "gpu_util_mean_over_seeds": float(np.mean([g["mean"] for g in gpu]))
        if gpu
        else None,
        "gpu_util_std_within_run_mean": float(np.mean([g["std"] for g in gpu]))
        if gpu
        else None,
        "gpu_util_p5_mean": float(np.mean([g["p5"] for g in gpu])) if gpu else None,
        "seeds": seeds,
    }


def plot(groups: list[dict], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    for g in groups:
        x = np.array(g["curve"]["env_steps"]) / 1e3
        m, s = np.array(g["curve"]["mean"]), np.array(g["curve"]["std"])
        (line,) = ax.plot(x, m, lw=2, label=f"{g['name']} (n={g['num_seeds']})")
        ax.fill_between(
            x,
            np.clip(m - s, 0, 100),
            np.clip(m + s, 0, 100),
            color=line.get_color(),
            alpha=0.2,
        )
    px = np.array(list(PAPER_UMAZE_APPROX)) / 1e3
    ax.plot(
        px,
        list(PAPER_UMAZE_APPROX.values()),
        "k--",
        marker="o",
        ms=4,
        lw=1.2,
        label="RLPD paper Fig. 14 (approx. reading)",
    )
    ax.set_xlabel("Environment steps (x1000)")
    ax.set_ylabel("Normalized score (success %)")
    ax.set_ylim(-2, 102)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("RLPD on AntMaze-UMaze (mean ± 1 std over seeds)")
    fig.tight_layout()
    fig.savefig(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", action="append", required=True, help="NAME=DIR")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steady-after", type=int, default=25_000)
    args = parser.parse_args()
    groups = [
        summarize_group(n, Path(d), args.steady_after)
        for n, d in (g.split("=", 1) for g in args.group)
    ]
    with open(f"{args.out}.json", "w") as f:
        json.dump(
            {"paper_umaze_approx": PAPER_UMAZE_APPROX, "groups": groups}, f, indent=2
        )
    plot(groups, f"{args.out}.png")
    for g in groups:
        print(
            json.dumps(
                {k: v for k, v in g.items() if k not in ("seeds", "curve")}, indent=2
            )
        )


if __name__ == "__main__":
    main()
