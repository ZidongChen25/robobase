#!/usr/bin/env python3
"""Evaluate a strict JAX A2A checkpoint in the official RoboVerse runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def build_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    root = Path(__file__).resolve().parents[2]
    checkout = Path(args.checkout).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    dataset = Path(args.dataset).expanduser().resolve()
    trajectory = Path(args.trajectory).expanduser().resolve()
    for required in (checkpoint, dataset, trajectory):
        if not required.exists():
            raise FileNotFoundError(required)
    command = [
        str(Path(args.python).expanduser().absolute()),
        str(root / "benchmarks/official_bigym/a2a_official_entrypoint.py"),
        str(checkout / "roboverse_learn/il/train.py"),
        "--config-name=default_runner.yaml",
        "task_name=close_box",
        f"dataset_config.zarr_path={dataset}",
        "shape_meta.obs.head_cam.shape=[3,256,256]",
        "shape_meta.obs.agent_pos.shape=[9]",
        "shape_meta.action.shape=[9]",
        "horizon=16",
        "n_obs_steps=8",
        "n_action_steps=8",
        "train_config.training_params.device=cuda:0",
        "eval_config.policy_runner.obs.obs_type=joint_pos",
        "eval_config.policy_runner.action.action_type=joint_pos",
        "eval_config.policy_runner.action.delta=0",
        "eval_config.eval_args.task=close_box",
        "eval_config.eval_args.sim=isaacsim",
        "eval_config.eval_args.max_step=300",
        "eval_config.eval_args.num_envs=1",
        "+eval_config.eval_args.gpu_id=0",
        " +eval_config.eval_args.task_id_range_low=0".strip(),
        f"+eval_config.eval_args.task_id_range_high={args.episodes}",
        f"+eval_config.eval_args.max_demo={args.episodes}",
        "train_enable=false",
        "eval_enable=true",
        f"eval_path={checkpoint}",
        "logging.mode=disabled",
        f"checkpoint.save_root_dir={output}",
        f"hydra.run.dir={output}",
    ]
    main_site = root / ".venv/lib/python3.11/site-packages"
    official_site = (
        Path(args.official_venv).expanduser().resolve()
        / "lib/python3.11/site-packages"
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    environment["ROBOBASE_JAX_A2A_EVAL"] = "1"
    environment["ROBOBASE_EXTRA_SITE_PACKAGES"] = str(official_site)
    environment["ROBOBASE_OFFICIAL_EVAL_TASK"] = "close_box"
    environment["ROBOBASE_OFFICIAL_EVAL_TRAJECTORY"] = str(trajectory)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(official_site), str(checkout), environment.get("PYTHONPATH", ""))
    )
    # Both JAX and the Isaac/Torch stack load cuDNN.  The newer JAX copy is
    # backward compatible with Torch, while the reverse ordering is not.
    cudnn = main_site / "nvidia/cudnn/lib"
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        (str(cudnn), environment.get("LD_LIBRARY_PATH", ""))
    )
    return command, environment


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument(
        "--checkout",
        default="~/.local/share/a2a-roboverse-paper/source",
    )
    parser.add_argument(
        "--official-venv", default="~/.venvs/a2a-roboverse-paper"
    )
    parser.add_argument("--python", default=".venv/bin/python")
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    command, environment = build_command(args)
    manifest = {
        "schema": "official_a2a_jax_eval_v1",
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_sha256": file_sha256(
            Path(args.checkpoint).expanduser().resolve()
        ),
        "dataset": str(Path(args.dataset).expanduser().resolve()),
        "trajectory": str(Path(args.trajectory).expanduser().resolve()),
        "trajectory_sha256": file_sha256(
            Path(args.trajectory).expanduser().resolve()
        ),
        "episodes": args.episodes,
        "flow_steps": 6,
        "execution_steps": 8,
        "simulator": "isaacsim",
        "torch_policy_dependency": False,
        "torch_simulator_boundary": True,
        "command": command,
    }
    (output / "launch_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    process = subprocess.Popen(command, env=environment, start_new_session=True)
    completed = False
    final_stats: Path | None = None
    deadline = time.monotonic() + args.timeout_seconds
    while process.poll() is None:
        final_stats = next(output.rglob("final_stats.txt"), None)
        if final_stats is not None:
            completed = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            break
        if time.monotonic() >= deadline:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise TimeoutError(
                f"RoboVerse evaluation exceeded {args.timeout_seconds} seconds"
            )
        time.sleep(1.0)
    if not completed:
        final_stats = next(output.rglob("final_stats.txt"), None)
        completed = final_stats is not None and process.returncode == 0
    if completed and final_stats is not None:
        stats_text = final_stats.read_text(encoding="utf-8")
        match = re.search(r"Average Success Rate:\s*([0-9.]+)", stats_text)
        if match is None:
            raise RuntimeError(f"Missing success rate in {final_stats}")
        manifest["status"] = "pass"
        manifest["success_rate"] = float(match.group(1))
        manifest["final_stats"] = str(final_stats)
        temporary = output / ".eval_manifest.json.tmp"
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output / "eval_manifest.json")
        return 0
    return int(process.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
