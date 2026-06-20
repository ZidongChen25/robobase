#!/usr/bin/env python3
"""Run Flow Matching sampling-step sweeps and plot the result."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckpointJob:
    task: str
    run_dir: Path
    checkpoint_step: int
    snapshot: Path
    baseline_success: float | None
    baseline_reward: float | None
    baseline_length: float | None
    action_sequence: int | None
    execution_length: int | None
    observation_timing: str | None


@dataclass(frozen=True)
class EvalJob:
    checkpoint: CheckpointJob
    flow_steps: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--flow-steps", nargs="+", type=int, default=[2, 4, 6, 8, 15, 20])
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--num-eval-episodes", type=int, default=None)
    parser.add_argument("--num-eval-envs", type=int, default=1)
    parser.add_argument("--max-parallel-per-gpu", type=int, default=1)
    parser.add_argument("--execution-length", type=int, default=None)
    parser.add_argument("--flow-schedule", type=str, default=None)
    parser.add_argument("--rerun-flow-step-10", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _read_manifest(path: Path) -> list[CheckpointJob]:
    rows = json.loads(path.read_text())
    jobs = []
    for row in rows:
        run_dir = Path(row["run_dir"])
        snapshot = Path(row.get("snapshot") or run_dir / "snapshots" / f"{row['checkpoint_step']}_snapshot.pkl")
        jobs.append(
            CheckpointJob(
                task=row["task"],
                run_dir=run_dir,
                checkpoint_step=int(row["checkpoint_step"]),
                snapshot=snapshot,
                baseline_success=_optional_float(row.get("baseline_success")),
                baseline_reward=_optional_float(row.get("baseline_reward")),
                baseline_length=_optional_float(row.get("baseline_length")),
                action_sequence=_optional_int(row.get("action_sequence")),
                execution_length=_optional_int(row.get("execution_length")),
                observation_timing=row.get("observation_timing"),
            )
        )
    return jobs


def _optional_float(value) -> float | None:
    if value is None or value == "":
        return None
    value = float(value)
    if math.isnan(value):
        return None
    return value


def _optional_int(value) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def _result_path(output_dir: Path, job: EvalJob) -> Path:
    task = _safe_name(job.checkpoint.task)
    return output_dir / "results_json" / f"{task}_ckpt{job.checkpoint.checkpoint_step}_flow{job.flow_steps}.json"


def _work_dir(output_dir: Path, job: EvalJob) -> Path:
    task = _safe_name(job.checkpoint.task)
    return output_dir / "work_dirs" / f"{task}_ckpt{job.checkpoint.checkpoint_step}_flow{job.flow_steps}"


def _log_path(output_dir: Path, job: EvalJob) -> Path:
    task = _safe_name(job.checkpoint.task)
    return output_dir / "logs" / f"{task}_ckpt{job.checkpoint.checkpoint_step}_flow{job.flow_steps}.log"


def _launch(job: EvalJob, gpu_id: int, args: argparse.Namespace) -> subprocess.Popen:
    output = _result_path(args.output_dir, job)
    work_dir = _work_dir(args.output_dir, job)
    log_path = _log_path(args.output_dir, job)
    output.parent.mkdir(parents=True, exist_ok=True)
    work_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/eval_flow_step_checkpoint.py",
        "--run-dir",
        str(job.checkpoint.run_dir),
        "--snapshot",
        str(job.checkpoint.snapshot),
        "--flow-steps",
        str(job.flow_steps),
        "--output",
        str(output),
        "--work-dir",
        str(work_dir),
        "--gpu-id",
        str(gpu_id),
        "--num-eval-envs",
        str(args.num_eval_envs),
    ]
    if args.num_eval_episodes is not None:
        cmd.extend(["--num-eval-episodes", str(args.num_eval_episodes)])
    if args.execution_length is not None:
        cmd.extend(["--execution-length", str(args.execution_length)])
    if args.flow_schedule is not None:
        cmd.extend(["--flow-schedule", str(args.flow_schedule)])

    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.25")
    env.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
    with log_path.open("w") as log_file:
        return subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)


def _read_result(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {"status": "missing"}
    except json.JSONDecodeError as exc:
        return {"status": "failed", "error": f"invalid json: {exc}"}


def _has_successful_result(path: Path) -> bool:
    return _read_result(path).get("status") == "ok"


def _collect_rows(
    checkpoints: list[CheckpointJob],
    flow_steps: list[int],
    output_dir: Path,
    include_existing_baseline: bool,
) -> list[dict]:
    rows: list[dict] = []
    for checkpoint in checkpoints:
        if include_existing_baseline:
            rows.append(
                {
                    "task": checkpoint.task,
                    "flow_steps": 10,
                    "episode_success": checkpoint.baseline_success,
                    "episode_reward": checkpoint.baseline_reward,
                    "episode_length": checkpoint.baseline_length,
                    "source": "existing_10_step_eval",
                    "status": "ok" if checkpoint.baseline_success is not None else "missing",
                    "run_dir": str(checkpoint.run_dir),
                    "snapshot": str(checkpoint.snapshot),
                    "checkpoint_step": checkpoint.checkpoint_step,
                    "action_sequence": checkpoint.action_sequence,
                    "execution_length": checkpoint.execution_length,
                    "observation_timing": checkpoint.observation_timing,
                    "error": "",
                }
            )
        for flow_step in flow_steps:
            result = _read_result(
                _result_path(output_dir, EvalJob(checkpoint=checkpoint, flow_steps=flow_step))
            )
            metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
            execution_length = result.get("execution_length", checkpoint.execution_length)
            flow_schedule = result.get("flow_schedule", "uniform")
            rows.append(
                {
                    "task": checkpoint.task,
                    "flow_steps": flow_step,
                    "episode_success": metrics.get("episode_success"),
                    "episode_reward": metrics.get("episode_reward"),
                    "episode_length": metrics.get("episode_length"),
                    "source": "sweep_eval",
                    "status": result.get("status", "missing"),
                    "run_dir": str(checkpoint.run_dir),
                    "snapshot": str(checkpoint.snapshot),
                    "checkpoint_step": checkpoint.checkpoint_step,
                    "action_sequence": checkpoint.action_sequence,
                    "execution_length": execution_length,
                    "flow_schedule": flow_schedule,
                    "observation_timing": checkpoint.observation_timing,
                    "error": result.get("error", ""),
                }
            )
    return rows


def _write_csv(rows: list[dict], output_dir: Path) -> Path:
    path = output_dir / "flow_step_sweep_results.csv"
    fieldnames = [
        "task",
        "flow_steps",
        "episode_success",
        "episode_reward",
        "episode_length",
        "source",
        "status",
        "checkpoint_step",
        "action_sequence",
        "execution_length",
        "flow_schedule",
        "observation_timing",
        "run_dir",
        "snapshot",
        "error",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _plot(rows: list[dict], output_dir: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return _plot_svg(rows, output_dir)

    ok_rows = [
        row
        for row in rows
        if row["status"] == "ok" and row["episode_success"] not in (None, "")
    ]
    by_task: dict[str, list[dict]] = {}
    for row in ok_rows:
        by_task.setdefault(row["task"], []).append(row)
    for task_rows in by_task.values():
        task_rows.sort(key=lambda row: int(row["flow_steps"]))

    paths: list[Path] = []
    if not by_task:
        return paths

    plt.figure(figsize=(10, 6))
    for task, task_rows in sorted(by_task.items()):
        xs = [int(row["flow_steps"]) for row in task_rows]
        ys = [float(row["episode_success"]) for row in task_rows]
        plt.plot(xs, ys, marker="o", linewidth=1.8, label=task)
    plt.xlabel("Flow sampling steps")
    plt.ylabel("Episode success")
    plt.ylim(-0.02, 1.02)
    plt.xticks(sorted({int(row["flow_steps"]) for row in ok_rows}))
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    path = output_dir / "flow_step_sweep_success.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)

    n = len(by_task)
    cols = min(3, n)
    rows_count = math.ceil(n / cols)
    fig, axes = plt.subplots(rows_count, cols, figsize=(4.2 * cols, 3.1 * rows_count), squeeze=False)
    all_steps = sorted({int(row["flow_steps"]) for row in ok_rows})
    for axis in axes.flat:
        axis.axis("off")
    for axis, (task, task_rows) in zip(axes.flat, sorted(by_task.items())):
        axis.axis("on")
        xs = [int(row["flow_steps"]) for row in task_rows]
        ys = [float(row["episode_success"]) for row in task_rows]
        axis.plot(xs, ys, marker="o", linewidth=1.8)
        axis.set_title(task)
        axis.set_ylim(-0.02, 1.02)
        axis.set_xticks(all_steps)
        axis.grid(True, alpha=0.3)
    fig.supxlabel("Flow sampling steps")
    fig.supylabel("Episode success")
    fig.tight_layout()
    facet_path = output_dir / "flow_step_sweep_success_facets.png"
    fig.savefig(facet_path, dpi=180)
    plt.close(fig)
    paths.append(facet_path)
    return paths


def _plot_svg(rows: list[dict], output_dir: Path) -> list[Path]:
    ok_rows = [
        row
        for row in rows
        if row["status"] == "ok" and row["episode_success"] not in (None, "")
    ]
    by_task: dict[str, list[dict]] = {}
    for row in ok_rows:
        by_task.setdefault(row["task"], []).append(row)
    for task_rows in by_task.values():
        task_rows.sort(key=lambda row: int(row["flow_steps"]))
    if not by_task:
        return []

    width, height = 1200, 760
    left, right, top, bottom = 90, 260, 40, 80
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_steps = sorted({int(row["flow_steps"]) for row in ok_rows})
    min_step, max_step = min(all_steps), max(all_steps)

    def x_pos(step: int) -> float:
        if max_step == min_step:
            return left + plot_width / 2
        return left + (step - min_step) / (max_step - min_step) * plot_width

    def y_pos(success: float) -> float:
        return top + (1.0 - success) * plot_height

    palette = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#17becf",
        "#8c564b",
        "#7f7f7f",
    ]
    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;font-size:16px;fill:#222}.small{font-size:13px}.title{font-size:22px;font-weight:700}.axis{stroke:#222;stroke-width:1.5}.grid{stroke:#ddd;stroke-width:1}.line{fill:none;stroke-width:2.5}.dot{stroke:white;stroke-width:1.5}</style>',
        '<text class="title" x="90" y="28">Flow step sweep success</text>',
    ]
    for value in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = y_pos(value)
        svg.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}"/>')
        svg.append(f'<text class="small" x="{left - 12}" y="{y + 5:.1f}" text-anchor="end">{value:.2f}</text>')
    for step in all_steps:
        x = x_pos(step)
        svg.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}"/>')
        svg.append(f'<text class="small" x="{x:.1f}" y="{top + plot_height + 28}" text-anchor="middle">{step}</text>')
    svg.append(f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>')
    svg.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>')
    svg.append(f'<text x="{left + plot_width / 2:.1f}" y="{height - 25}" text-anchor="middle">Flow sampling steps</text>')
    svg.append(f'<text x="24" y="{top + plot_height / 2:.1f}" text-anchor="middle" transform="rotate(-90 24 {top + plot_height / 2:.1f})">Episode success</text>')

    for idx, (task, task_rows) in enumerate(sorted(by_task.items())):
        color = palette[idx % len(palette)]
        points = " ".join(
            f"{x_pos(int(row['flow_steps'])):.1f},{y_pos(float(row['episode_success'])):.1f}"
            for row in task_rows
        )
        svg.append(f'<polyline class="line" stroke="{color}" points="{points}"/>')
        for row in task_rows:
            x = x_pos(int(row["flow_steps"]))
            y = y_pos(float(row["episode_success"]))
            svg.append(f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')
        legend_y = top + 22 * idx
        svg.append(f'<line x1="{left + plot_width + 35}" y1="{legend_y}" x2="{left + plot_width + 60}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text class="small" x="{left + plot_width + 68}" y="{legend_y + 5}">{_xml_escape(task)}</text>')
    svg.append("</svg>")

    path = output_dir / "flow_step_sweep_success.svg"
    path.write_text("\n".join(svg) + "\n")
    return [path]


def _xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _summarize(rows: list[dict], output_dir: Path) -> Path:
    summary_path = output_dir / "summary.txt"
    lines = []
    by_task: dict[str, list[dict]] = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(row)
    for task, task_rows in sorted(by_task.items()):
        ok = [
            row
            for row in task_rows
            if row["status"] == "ok" and row["episode_success"] not in (None, "")
        ]
        if not ok:
            lines.append(f"{task}: no successful eval rows")
            continue
        best = max(ok, key=lambda row: float(row["episode_success"]))
        lines.append(
            f"{task}: best flow_steps={best['flow_steps']} "
            f"success={float(best['episode_success']):.3f} "
            f"(ckpt={best['checkpoint_step']}, baseline10="
            f"{next((r['episode_success'] for r in ok if int(r['flow_steps']) == 10), 'missing')})"
        )
    summary_path.write_text("\n".join(lines) + "\n")
    return summary_path


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = _read_manifest(args.manifest)
    flow_steps = list(args.flow_steps)
    if not args.rerun_flow_step_10:
        flow_steps = [step for step in flow_steps if step != 10]
    eval_jobs = [
        EvalJob(checkpoint=checkpoint, flow_steps=step)
        for checkpoint in checkpoints
        for step in flow_steps
    ]
    if args.resume:
        eval_jobs = [
            job
            for job in eval_jobs
            if not _has_successful_result(_result_path(args.output_dir, job))
        ]

    gpu_slots = deque(
        gpu
        for gpu in args.gpus
        for _ in range(max(int(args.max_parallel_per_gpu), 1))
    )
    running: dict[subprocess.Popen, tuple[EvalJob, int]] = {}
    pending = deque(eval_jobs)
    status_path = args.output_dir / "status.log"
    with status_path.open("a") as status:
        status.write(
            f"starting {len(eval_jobs)} eval jobs across GPUs {args.gpus}; "
            f"output_dir={args.output_dir}\n"
        )
        status.flush()
        while pending or running:
            while pending and gpu_slots:
                gpu = gpu_slots.popleft()
                job = pending.popleft()
                process = _launch(job, gpu, args)
                running[process] = (job, gpu)
                status.write(
                    f"launched task={job.checkpoint.task} ckpt={job.checkpoint.checkpoint_step} "
                    f"flow_steps={job.flow_steps} gpu={gpu} pid={process.pid}\n"
                )
                status.flush()
            time.sleep(5)
            for process in list(running):
                code = process.poll()
                if code is None:
                    continue
                job, gpu = running.pop(process)
                gpu_slots.append(gpu)
                result = _read_result(_result_path(args.output_dir, job))
                status.write(
                    f"finished task={job.checkpoint.task} ckpt={job.checkpoint.checkpoint_step} "
                    f"flow_steps={job.flow_steps} gpu={gpu} pid={process.pid} "
                    f"returncode={code} status={result.get('status')}\n"
                )
                if result.get("status") == "ok":
                    metrics = result.get("metrics", {})
                    status.write(
                        f"  success={metrics.get('episode_success')} "
                        f"reward={metrics.get('episode_reward')} length={metrics.get('episode_length')}\n"
                    )
                else:
                    status.write(f"  error={result.get('error')}\n")
                status.flush()

    rows = _collect_rows(
        checkpoints,
        flow_steps,
        args.output_dir,
        include_existing_baseline=not args.rerun_flow_step_10,
    )
    csv_path = _write_csv(rows, args.output_dir)
    plot_paths = _plot(rows, args.output_dir)
    summary_path = _summarize(rows, args.output_dir)
    print(f"wrote {csv_path}")
    for path in plot_paths:
        print(f"wrote {path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
