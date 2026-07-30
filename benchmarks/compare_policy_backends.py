"""Run matched RoboBase JAX and CleanDiffuser Torch policy benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIFFUSER_COMMIT = "05f17fc9dbeae7c19a5e264632c9ae9aaac5994e"


def _command_output(command: list[str], cwd: Path) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repo_metadata(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    patch = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
    ).stdout
    digest.update(patch)
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=path,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    for relative_bytes in sorted(value for value in untracked if value):
        relative = relative_bytes.decode()
        candidate = path / relative
        if not candidate.is_file():
            continue
        digest.update(relative_bytes)
        digest.update(candidate.read_bytes())
    return {
        "path": str(path),
        "commit": _command_output(["git", "rev-parse", "HEAD"], path),
        "status_short": _command_output(
            ["git", "status", "--short"], path
        ).splitlines(),
        "worktree_sha256": digest.hexdigest(),
    }


def _gpu_snapshot(index: int) -> dict[str, Any]:
    query = "index,uuid,name,utilization.gpu,memory.used,memory.total,temperature.gpu"
    command = [
        "nvidia-smi",
        f"--id={index}",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    try:
        gpu = _command_output(command, REPO_ROOT)
        processes = _command_output(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            REPO_ROOT,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {"error": str(exc)}
    fields = [item.strip() for item in gpu.split(",")]
    return {
        "index": int(fields[0]),
        "uuid": fields[1],
        "name": fields[2],
        "utilization_percent": float(fields[3]),
        "memory_used_mib": float(fields[4]),
        "memory_total_mib": float(fields[5]),
        "temperature_c": float(fields[6]),
        "compute_processes": processes.splitlines() if processes else [],
    }


def _parse_worker_output(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "backend" in value:
            return value
    raise RuntimeError(
        f"Worker did not emit a JSON result. Output tail:\n{stdout[-4000:]}"
    )


def _run_worker(
    *,
    backend: str,
    python: Path,
    clean_root: Path,
    common_args: list[str],
    torch_mode: str,
    gpu: int,
) -> dict[str, Any]:
    worker_name = (
        "jax_policy_worker.py" if backend == "jax" else "torch_cleandiffuser_worker.py"
    )
    worker = REPO_ROOT / "benchmarks" / worker_name
    command = [str(python), str(worker), *common_args]
    cwd = REPO_ROOT
    if backend == "jax":
        command.extend(["--platform", "cuda"])
    else:
        cwd = clean_root
        if torch_mode != "eager":
            command.extend(["--torch-compile", "--compile-mode", torch_mode])

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "JAX_PLATFORMS": "cuda",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "PYTHONHASHSEED": "0",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TF_CPP_MIN_LOG_LEVEL": "2",
        }
    )
    source_root = REPO_ROOT if backend == "jax" else clean_root
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(source_root) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    start = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    wall_seconds = time.perf_counter() - start
    if process.returncode:
        raise RuntimeError(
            f"{backend} worker failed with exit {process.returncode}.\n"
            f"stdout tail:\n{process.stdout[-4000:]}\n"
            f"stderr tail:\n{process.stderr[-8000:]}"
        )
    result = _parse_worker_output(process.stdout)
    result["worker_wall_seconds"] = wall_seconds
    result["stderr_tail"] = process.stderr[-2000:]
    return result


def _comparable(
    jax_result: dict[str, Any], torch_result: dict[str, Any]
) -> tuple[bool, list[str]]:
    fields = (
        "objective",
        "backbone",
        "encoder",
        "fusion",
        "dtype",
        "batch_size",
        "eval_batch_size",
        "horizon",
        "action_dim",
        "obs_shape",
        "condition_dim",
        "embed_dim",
        "down_dims",
        "kernel_size",
        "n_groups",
        "cond_predict_scale",
        "conditioning_mode",
        "timestep_embedding_type",
        "operator_variant",
        "compatibility_mode",
        "global_condition_embed_dim",
        "sample_steps",
        "sampler",
        "ema",
        "ema_decay",
        "ema_decay_schedule",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "training_schedule",
        "loss_reduction",
        "sample_temperature",
        "action_bounds",
        "parameter_count",
    )
    mismatches = []
    for field in fields:
        if field not in jax_result or field not in torch_result:
            missing = [
                backend
                for backend, result in (("jax", jax_result), ("torch", torch_result))
                if field not in result
            ]
            mismatches.append(f"{field}: missing from {', '.join(missing)} result")
        elif jax_result[field] != torch_result[field]:
            mismatches.append(
                f"{field}: jax={jax_result[field]!r}, torch={torch_result[field]!r}"
            )
    return not mismatches, mismatches


def _comparison(
    jax_result: dict[str, Any], torch_result: dict[str, Any], repeat: int
) -> dict[str, Any]:
    comparable, mismatches = _comparable(jax_result, torch_result)
    result = {
        "objective": jax_result["objective"],
        "ema": jax_result["ema"],
        "torch_mode": torch_result["compile_mode"],
        "repeat": repeat,
        "comparable": comparable,
        "mismatches": mismatches,
    }
    if not comparable:
        return result
    update_median_speedup = (
        torch_result["update"]["median_ms"] / jax_result["update"]["median_ms"] - 1.0
    )
    sample_median_speedup = (
        torch_result["sample"]["median_ms"] / jax_result["sample"]["median_ms"] - 1.0
    )
    update_mean_speedup = (
        torch_result["update"]["mean_ms"] / jax_result["update"]["mean_ms"] - 1.0
    )
    sample_mean_speedup = (
        torch_result["sample"]["mean_ms"] / jax_result["sample"]["mean_ms"] - 1.0
    )
    update_p95_speedup = (
        torch_result["update"]["p95_ms"] / jax_result["update"]["p95_ms"] - 1.0
    )
    sample_p95_speedup = (
        torch_result["sample"]["p95_ms"] / jax_result["sample"]["p95_ms"] - 1.0
    )
    result.update(
        {
            "primary_gate_metric": "median_ms",
            "jax_update_median_speedup_percent": update_median_speedup * 100.0,
            "jax_sample_median_speedup_percent": sample_median_speedup * 100.0,
            "jax_update_mean_speedup_percent": update_mean_speedup * 100.0,
            "jax_sample_mean_speedup_percent": sample_mean_speedup * 100.0,
            "jax_update_p95_speedup_percent": update_p95_speedup * 100.0,
            "jax_sample_p95_speedup_percent": sample_p95_speedup * 100.0,
            "update_meets_30_percent": update_median_speedup >= 0.30,
            "sample_meets_30_percent": sample_median_speedup >= 0.30,
            "update_p95_meets_30_percent": update_p95_speedup >= 0.30,
            "sample_p95_meets_30_percent": sample_p95_speedup >= 0.30,
            "jax_update_cv_percent": jax_result["update"]["cv_percent"],
            "torch_update_cv_percent": torch_result["update"]["cv_percent"],
            "jax_sample_cv_percent": jax_result["sample"]["cv_percent"],
            "torch_sample_cv_percent": torch_result["sample"]["cv_percent"],
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--clean-root",
        type=Path,
        default=Path(f"/tmp/CleanDiffuser-baseline-{CLEAN_DIFFUSER_COMMIT[:8]}"),
    )
    parser.add_argument("--clean-commit", default=CLEAN_DIFFUSER_COMMIT)
    parser.add_argument("--allow-dirty-clean", action="store_true")
    parser.add_argument(
        "--jax-python", type=Path, default=REPO_ROOT / ".venv/bin/python"
    )
    parser.add_argument("--clean-python", type=Path, default=None)
    parser.add_argument(
        "--objectives",
        nargs="+",
        choices=("diffusion", "flow_matching"),
        default=["diffusion", "flow_matching"],
    )
    parser.add_argument(
        "--ema-modes", nargs="+", choices=("off", "on"), default=["off", "on"]
    )
    parser.add_argument(
        "--torch-modes", nargs="+", choices=("eager", "default"), default=["eager"]
    )
    parser.add_argument("--pair-repeats", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--action-dim", type=int, default=10)
    parser.add_argument("--obs-steps", type=int, default=2)
    parser.add_argument("--obs-dim", type=int, default=23)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--down-dims", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--n-groups", type=int, default=8)
    parser.add_argument("--sample-steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/robobase_clean_policy_benchmark.json")
    )
    args = parser.parse_args()
    if args.pair_repeats < 1 or args.iterations < 2:
        raise ValueError("pair-repeats must be >= 1 and iterations must be >= 2.")
    clean_python = args.clean_python or args.clean_root / ".venv/bin/python"

    jax_metadata = _repo_metadata(REPO_ROOT)
    clean_metadata = _repo_metadata(args.clean_root)
    if clean_metadata["commit"] != args.clean_commit:
        raise ValueError(
            "CleanDiffuser baseline commit mismatch: "
            f"expected {args.clean_commit}, got {clean_metadata['commit']}."
        )
    if clean_metadata["status_short"] and not args.allow_dirty_clean:
        raise ValueError(
            "CleanDiffuser baseline worktree must be clean; pass "
            "--allow-dirty-clean only for exploratory runs."
        )

    report: dict[str, Any] = {
        "benchmark": "RoboBase JAX vs CleanDiffuser matched state-only global UNet",
        "timestamp_unix": time.time(),
        "gpu_nonexclusive_warning": (
            "GPU state is sampled only before and after each process. Other jobs may "
            "affect results; inspect CV and repeat on an exclusive GPU before "
            "release claims."
        ),
        "repos": {
            "jax": jax_metadata,
            "clean_diffuser": clean_metadata,
        },
        "command": [sys.executable, *sys.argv],
        "command_shell": shlex.join([sys.executable, *sys.argv]),
        "gpu_before": _gpu_snapshot(args.gpu),
        "runs": [],
        "comparisons": [],
    }

    for objective in args.objectives:
        for ema_mode in args.ema_modes:
            for torch_mode in args.torch_modes:
                for repeat in range(args.pair_repeats):
                    common_args = [
                        "--objective",
                        objective,
                        "--batch-size",
                        str(args.batch_size),
                        "--eval-batch-size",
                        str(args.eval_batch_size),
                        "--horizon",
                        str(args.horizon),
                        "--action-dim",
                        str(args.action_dim),
                        "--obs-steps",
                        str(args.obs_steps),
                        "--obs-dim",
                        str(args.obs_dim),
                        "--embed-dim",
                        str(args.embed_dim),
                        "--down-dims",
                        *map(str, args.down_dims),
                        "--kernel-size",
                        str(args.kernel_size),
                        "--n-groups",
                        str(args.n_groups),
                        "--sample-steps",
                        str(args.sample_steps),
                        "--warmup",
                        str(args.warmup),
                        "--iterations",
                        str(args.iterations),
                        "--lr",
                        str(args.lr),
                        "--weight-decay",
                        str(args.weight_decay),
                        "--ema-decay",
                        str(args.ema_decay),
                        "--seed",
                        str(args.seed + repeat),
                    ]
                    if ema_mode == "on":
                        common_args.append("--ema")

                    order = ("jax", "torch") if repeat % 2 == 0 else ("torch", "jax")
                    pair = {}
                    for backend in order:
                        pair[backend] = _run_worker(
                            backend=backend,
                            python=args.jax_python
                            if backend == "jax"
                            else clean_python,
                            clean_root=args.clean_root,
                            common_args=common_args,
                            torch_mode=torch_mode,
                            gpu=args.gpu,
                        )
                        pair[backend]["repeat"] = repeat
                        pair[backend]["gpu_snapshot_after"] = _gpu_snapshot(args.gpu)
                        report["runs"].append(pair[backend])
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        args.output.write_text(
                            json.dumps(report, indent=2, sort_keys=True)
                        )

                    report["comparisons"].append(
                        _comparison(pair["jax"], pair["torch"], repeat)
                    )
                    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))

    report["gpu_after"] = _gpu_snapshot(args.gpu)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report["comparisons"], indent=2, sort_keys=True))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
