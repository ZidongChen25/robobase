"""Read-out for the WSRL staged-unanchoring arm (Arm B).

Reads train.csv BY HEADER (resumed csvs carry repeated header rows), the
per-episode jsonl written by scripts/wsrl_staged_unanchor.py, and the stage
record, then prints the span / agreement / success trajectories and applies
the pre-registered collapse verdict.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SPAN_KEY = "bc_sibling_q_span"


def load_csv(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            # Resumed csvs repeat the header; drop those rows by value, not index.
            if row.get("env_steps") in (None, "", "env_steps"):
                continue
            rows.append(row)
    return rows


def fnum(row: dict, key: str):
    value = row.get(key, "")
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fshow(row: dict, key: str) -> float:
    """Numeric or NaN -- never confuse a legitimate 0.0 with 'missing'."""
    value = fnum(row, key)
    return float("nan") if value is None else value


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--span-threshold", type=float, default=0.02)
    parser.add_argument("--success-threshold", type=float, default=0.10)
    parser.add_argument("--sustained-steps", type=int, default=3000)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    stages = json.loads((run_dir / "wsrl_stages.json").read_text())
    seam = next(
        (
            record["env_step"]
            for record in stages
            if record["stage"] == "online" and record["event"] == "start"
        ),
        None,
    )
    start = stages[0]["env_step"]
    rows = load_csv(run_dir / "train.csv")
    episodes = load_jsonl(run_dir / "wsrl_episodes.jsonl")
    recalib = load_jsonl(run_dir / "wsrl_recalib.jsonl")

    print(f"run_dir       : {run_dir}")
    print(f"start step    : {start}")
    print(f"seam step     : {seam}")
    for record in stages:
        print(
            f"  stage {record['stage']:<15} {record['event']:<8} "
            f"step={record['env_step']:<8} eps={record['episodes']:<5} "
            f"buf={record['buffer_size']:<8} demo_buf={record['demo_buffer_size']}"
        )

    if recalib:
        print("\n-- recalibration (gradient updates, anchor already off) --")
        print(f"{'updates':>8} {'span':>9} {'agree':>8} {'bind':>8} {'critic_loss':>12}")
        for row in recalib:
            print(
                f"{row.get('gradient_updates', 0):>8} "
                f"{row.get(SPAN_KEY, float('nan')):>9.5f} "
                f"{row.get('bc_agreement', float('nan')):>8.4f} "
                f"{row.get('bc_binding_rate', float('nan')):>8.4f} "
                f"{row.get('critic_loss', float('nan')):>12.5f}"
            )

    print("\n-- online train.csv (by header) --")
    print(
        f"{'env_steps':>10} {'since_seam':>11} {'bc_weight':>10} {'span':>9} "
        f"{'agree':>8} {'online_agr':>11} {'bind':>8} {'critic_loss':>12} {'buffer':>9}"
    )
    span_series = []
    for row in rows:
        steps = fnum(row, "env_steps")
        if steps is None:
            continue
        # Rows logged before the first gradient update carry restval 0.0 for
        # every update metric (the logger backfills columns it did not yet
        # know about). Only rows at/after the seam contain a measured span.
        if seam is not None and steps < seam:
            continue
        span = fnum(row, SPAN_KEY)
        if span is None:
            continue
        since = None if seam is None else steps - seam
        span_series.append((steps, span))
        print(
            f"{steps:>10.0f} {('' if since is None else f'{since:.0f}'):>11} "
            f"{fshow(row, 'bc_weight'):>10.4f} "
            f"{span:>9.5f} "
            f"{fshow(row, 'bc_agreement'):>8.4f} "
            f"{fshow(row, 'bc_online_agreement'):>11.4f} "
            f"{fshow(row, 'bc_binding_rate'):>8.4f} "
            f"{fshow(row, 'critic_loss'):>12.5f} "
            f"{fshow(row, 'buffer_size'):>9.0f}"
        )

    # Rollout success by stage and by 2k window after the seam.
    print("\n-- rollout success --")
    by_stage: dict[str, list[int]] = {}
    for episode in episodes:
        by_stage.setdefault(episode["stage"], []).append(int(episode["success"]))
    for stage, values in by_stage.items():
        rate = sum(values) / max(1, len(values))
        print(f"  {stage:<12} n={len(values):<4} success={rate:.3f}")

    windows = []
    if seam is not None:
        online_eps = [e for e in episodes if e["stage"] == "online"]
        if online_eps:
            width = 2000
            last = max(e["env_step"] for e in online_eps)
            edge = seam
            while edge < last:
                bucket = [
                    int(e["success"])
                    for e in online_eps
                    if edge <= e["env_step"] < edge + width
                ]
                if bucket:
                    rate = sum(bucket) / len(bucket)
                    windows.append((edge - seam, edge + width - seam, len(bucket), rate))
                    print(
                        f"  +{edge - seam:>6}..{edge + width - seam:<6} "
                        f"n={len(bucket):<3} success={rate:.3f}"
                    )
                edge += width

    # Pre-registered verdict.
    print("\n-- verdict --")
    verdict = {}
    online_span = [(s, v) for s, v in span_series if seam is not None and s >= seam]
    span_collapse_from = None
    if online_span:
        run_start = None
        for steps, span in online_span:
            if span < args.span_threshold:
                if run_start is None:
                    run_start = steps
                elif steps - run_start >= args.sustained_steps:
                    span_collapse_from = run_start
                    break
            else:
                run_start = None
    success_collapse = None
    low = [w for w in windows if w[3] < args.success_threshold]
    if low:
        contiguous = 0
        first = None
        for window in windows:
            if window[3] < args.success_threshold:
                if first is None:
                    first = window[0]
                contiguous += window[1] - window[0]
                if contiguous >= args.sustained_steps:
                    success_collapse = first
                    break
            else:
                contiguous = 0
                first = None

    verdict["span_collapse_from_step_after_seam"] = (
        None if span_collapse_from is None else span_collapse_from - seam
    )
    verdict["success_collapse_from_step_after_seam"] = success_collapse
    verdict["min_online_span"] = min((v for _, v in online_span), default=None)
    verdict["final_online_span"] = online_span[-1][1] if online_span else None
    verdict["online_success"] = (
        sum(by_stage.get("online", [])) / max(1, len(by_stage.get("online", [])))
        if by_stage.get("online")
        else None
    )
    verdict["warmup_success"] = (
        sum(by_stage.get("warmup", [])) / max(1, len(by_stage.get("warmup", [])))
        if by_stage.get("warmup")
        else None
    )
    verdict["collapsed"] = bool(
        span_collapse_from is not None or success_collapse is not None
    )
    print(json.dumps(verdict, indent=2))

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "start_step": start,
                    "seam_step": seam,
                    "stages": stages,
                    "recalibration": recalib,
                    "span_series": span_series,
                    "success_windows": windows,
                    "verdict": verdict,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
