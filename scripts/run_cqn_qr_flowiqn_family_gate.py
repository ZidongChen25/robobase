#!/usr/bin/env python3
"""Select and compare a matched FlowIQN objective family on common splits.

Every arm receives the same checkpoint and BC-prior-beta selection budget.
The two quantile-regularized treatments are selected against each other only
on validation seeds.  Their frozen winner is then compared once with both the
frozen anchor-only control and validation-best clean CQN-AS on sealed seeds.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from omegaconf import OmegaConf

try:
    from run_cqn_floq_checkpoint_selection_gate import select_top_steps
    from summarize_cqn_paired_eval import summarize as summarize_paired
except ImportError:
    from scripts.run_cqn_floq_checkpoint_selection_gate import (
        select_top_steps,
    )
    from scripts.summarize_cqn_paired_eval import (
        summarize as summarize_paired,
    )


@dataclass(frozen=True)
class Arm:
    label: str
    run_dir: Path
    role: str


@dataclass(frozen=True)
class EvalJob:
    label: str
    step: int | None
    beta: float | None


def _labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=RUN_DIR")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("expected LABEL=RUN_DIR")
    return label, Path(raw_path)


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def _nonnegative(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise argparse.ArgumentTypeError(
            "expected a finite non-negative number"
        )
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--anchor",
        required=True,
        type=_labeled_path,
    )
    parser.add_argument(
        "--treatment",
        required=True,
        action="append",
        type=_labeled_path,
    )
    parser.add_argument("--clean-run-dir", required=True, type=Path)
    parser.add_argument("--clean-snapshot", required=True, type=Path)
    parser.add_argument("--gpu-id", required=True, action="append", type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument(
        "--checkpoint-step",
        nargs="+",
        type=_positive_int,
        default=list(range(1_000, 10_001, 1_000)),
    )
    parser.add_argument("--screen-top-k", type=_positive_int, default=2)
    parser.add_argument(
        "--beta",
        nargs="+",
        type=_nonnegative,
        default=[0.3, 1.0, 3.0],
    )
    parser.add_argument("--screen-beta", type=_nonnegative, default=1.0)
    parser.add_argument("--num-flow-steps", type=_positive_int, default=8)
    parser.add_argument(
        "--num-action-flow-samples",
        type=_positive_int,
        default=8,
    )
    parser.add_argument("--screen-episodes", type=_positive_int, default=10)
    parser.add_argument("--screen-seed-start", type=int, default=210_000)
    parser.add_argument(
        "--validation-episodes",
        type=_positive_int,
        default=50,
    )
    parser.add_argument(
        "--validation-seed-start",
        type=int,
        default=211_000,
    )
    parser.add_argument(
        "--confirmation-episodes",
        type=_positive_int,
        default=200,
    )
    parser.add_argument(
        "--confirmation-seed-start",
        type=int,
        default=212_000,
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=_positive_int,
        default=20_000,
    )
    parser.add_argument("--bootstrap-seed", type=int, default=212_200)
    parser.add_argument(
        "--clean-min-ci-lower",
        type=float,
        default=-0.05,
    )
    return parser.parse_args()


def _number_label(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _result_path(
    output_dir: Path,
    split: str,
    job: EvalJob,
) -> Path:
    if job.label == "clean":
        filename = "clean.json"
    else:
        filename = (
            f"step{int(job.step)}_beta{_number_label(float(job.beta))}.json"
        )
    return output_dir / split / job.label / filename


def _completed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text()).get("status") == "ok"
    except (OSError, json.JSONDecodeError):
        return False


def build_eval_command(
    job: EvalJob,
    *,
    arms: dict[str, Arm],
    clean_run_dir: Path,
    clean_snapshot: Path,
    output: Path,
    work_dir: Path,
    gpu_id: int,
    episodes: int,
    seed_start: int,
    num_flow_steps: int,
    num_action_flow_samples: int,
) -> list[str]:
    clean = job.label == "clean"
    if clean:
        if job.step is not None or job.beta is not None:
            raise ValueError("clean evaluation cannot use step or beta")
        run_dir = clean_run_dir
        snapshot = clean_snapshot
    else:
        if job.label not in arms:
            raise ValueError(f"unknown arm: {job.label}")
        if job.step is None or job.beta is None:
            raise ValueError("flow evaluation requires step and beta")
        run_dir = arms[job.label].run_dir
        snapshot = run_dir / "snapshots" / f"{int(job.step)}_snapshot.pkl"

    command = [
        sys.executable,
        str(Path(__file__).with_name("eval_cqn_flow_policy_value.py")),
        "--run-dir",
        str(run_dir),
        "--snapshot",
        str(snapshot),
        "--output",
        str(output),
        "--work-dir",
        str(work_dir),
        "--gpu-id",
        str(gpu_id),
        "--num-eval-episodes",
        str(episodes),
        "--eval-seed-start",
        str(seed_start),
        "--policy-value-beta",
        "bc" if clean else f"{float(job.beta):g}",
        "--flow-readout",
        "auto" if clean else "integrated",
    ]
    if not clean:
        command.extend(
            [
                "--num-flow-steps",
                str(num_flow_steps),
                "--num-action-flow-samples",
                str(num_action_flow_samples),
                "--return-sample-aggregation",
                "mean",
            ]
        )
    return command


def _run_logged(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        stream.write("$ " + " ".join(command) + "\n")
        stream.flush()
        subprocess.run(
            command,
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )


def _run_jobs(
    *,
    jobs: list[EvalJob],
    split: str,
    args: argparse.Namespace,
    arms: dict[str, Arm],
    episodes: int,
    seed_start: int,
) -> None:
    work_queue: queue.Queue[EvalJob] = queue.Queue()
    for job in jobs:
        work_queue.put(job)
    failures: list[BaseException] = []
    failure_lock = threading.Lock()

    def worker(gpu_id: int) -> None:
        while True:
            try:
                job = work_queue.get_nowait()
            except queue.Empty:
                return
            try:
                output = _result_path(args.output_dir, split, job)
                if not _completed(output):
                    command = build_eval_command(
                        job,
                        arms=arms,
                        clean_run_dir=args.clean_run_dir,
                        clean_snapshot=args.clean_snapshot,
                        output=output,
                        work_dir=(
                            args.work_root
                            / split
                            / job.label
                            / output.stem
                        ),
                        gpu_id=gpu_id,
                        episodes=episodes,
                        seed_start=seed_start,
                        num_flow_steps=args.num_flow_steps,
                        num_action_flow_samples=(
                            args.num_action_flow_samples
                        ),
                    )
                    _run_logged(command, output.with_suffix(".log"))
                if not _completed(output):
                    raise RuntimeError(
                        f"evaluation did not complete: {output}"
                    )
            except BaseException as exc:
                with failure_lock:
                    failures.append(exc)
            finally:
                work_queue.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu_id,), daemon=False)
        for gpu_id in args.gpu_id
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(
            f"{len(failures)} evaluation worker(s) failed"
        ) from failures[0]


def _success(path: Path) -> float:
    payload = json.loads(path.read_text())
    if payload.get("status") != "ok":
        raise ValueError(f"incomplete evaluation: {path}")
    return float(payload["episode_success"])


def select_arm_winner(
    validation: dict[float, dict[int, float]],
) -> tuple[float, int, float]:
    """Select success, then larger beta, then earlier checkpoint."""

    rows = [
        (float(success), float(beta), int(step))
        for beta, by_step in validation.items()
        for step, success in by_step.items()
    ]
    if not rows:
        raise ValueError("validation grid is empty")
    success, beta, step = max(
        rows,
        key=lambda row: (row[0], row[1], -row[2]),
    )
    return beta, step, success


def select_treatment(
    labels: list[str],
    selected: dict[str, dict[str, float | int]],
) -> str:
    """Select validation success; preserve preregistered label order on ties."""

    if not labels:
        raise ValueError("at least one treatment is required")
    return max(
        enumerate(labels),
        key=lambda item: (
            float(selected[item[1]]["validation_success"]),
            -item[0],
        ),
    )[1]


def _load_eval(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("status") != "ok":
        raise ValueError(f"incomplete evaluation: {path}")
    return payload


def _validate_arm_config(arm: Arm) -> None:
    cfg_path = arm.run_dir / ".hydra" / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    method = OmegaConf.load(cfg_path).method
    quantile_lambda = float(method.get("quantile_endpoint_lambda", 0.0))
    if arm.role == "anchor" and quantile_lambda != 0.0:
        raise ValueError("anchor arm must disable quantile endpoint loss")
    if arm.role == "treatment" and quantile_lambda <= 0.0:
        raise ValueError("treatment arm must enable quantile endpoint loss")
    required = (
        str(method.get("value_mode", "")).lower() == "return_sample",
        bool(method.get("flow_iqn_quantile_coupling", False)),
        bool(method.get("separate_bc_policy", False)),
        bool(method.get("distinct_policy_encoder", False)),
        str(method.get("td_target_action_source", "")).lower() == "bc_policy",
        method.get("policy_value_beta", None) is None,
    )
    if not all(required):
        raise ValueError(f"arm config violates matched family protocol: {arm}")


def _normalize_args(args: argparse.Namespace) -> dict[str, Arm]:
    anchor_label, anchor_path = args.anchor
    treatment_pairs = list(args.treatment)
    labels = [anchor_label, *(label for label, _ in treatment_pairs)]
    if len(set(labels)) != len(labels) or "clean" in labels:
        raise ValueError("arm labels must be unique and cannot be 'clean'")
    if len(treatment_pairs) < 2:
        raise ValueError("family gate requires at least two treatments")
    if not args.gpu_id or len(set(args.gpu_id)) != len(args.gpu_id):
        raise ValueError("gpu-id workers must be nonempty and unique")
    if len(set(args.checkpoint_step)) != len(args.checkpoint_step):
        raise ValueError("checkpoint steps must be unique")
    if args.screen_top_k > len(args.checkpoint_step):
        raise ValueError("screen-top-k exceeds checkpoint count")
    if not args.beta or len(set(args.beta)) != len(args.beta):
        raise ValueError("beta grid must be nonempty and unique")
    if not math.isfinite(args.clean_min_ci_lower):
        raise ValueError("clean-min-ci-lower must be finite")

    seed_ranges = [
        set(range(start, start + episodes))
        for start, episodes in (
            (args.screen_seed_start, args.screen_episodes),
            (args.validation_seed_start, args.validation_episodes),
            (
                args.confirmation_seed_start,
                args.confirmation_episodes,
            ),
        )
    ]
    if any(
        left & right
        for index, left in enumerate(seed_ranges)
        for right in seed_ranges[index + 1 :]
    ):
        raise ValueError(
            "screen/validation/confirmation seeds must be disjoint"
        )

    args.clean_run_dir = args.clean_run_dir.expanduser().resolve()
    args.clean_snapshot = args.clean_snapshot.expanduser().resolve()
    if not (
        args.clean_run_dir / ".hydra" / "config.yaml"
    ).is_file():
        raise FileNotFoundError(args.clean_run_dir)
    if not args.clean_snapshot.is_file():
        raise FileNotFoundError(args.clean_snapshot)

    arms = {
        anchor_label: Arm(
            anchor_label,
            anchor_path.expanduser().resolve(),
            "anchor",
        ),
        **{
            label: Arm(
                label,
                path.expanduser().resolve(),
                "treatment",
            )
            for label, path in treatment_pairs
        },
    }
    for arm in arms.values():
        _validate_arm_config(arm)
        for step in args.checkpoint_step:
            snapshot = arm.run_dir / "snapshots" / f"{step}_snapshot.pkl"
            if not snapshot.is_file():
                raise FileNotFoundError(snapshot)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.work_root = args.work_root.expanduser().resolve()
    return arms


def run_gate(args: argparse.Namespace) -> dict:
    arms = _normalize_args(args)
    anchor_label = args.anchor[0]
    treatment_labels = [label for label, _ in args.treatment]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = {
        "status": "running",
        "arms": [asdict(arm) for arm in arms.values()],
        "clean_run_dir": str(args.clean_run_dir),
        "clean_snapshot": str(args.clean_snapshot),
        "gpu_ids": list(args.gpu_id),
        "checkpoint_steps": list(args.checkpoint_step),
        "screen_top_k": int(args.screen_top_k),
        "beta_grid": list(args.beta),
        "screen_beta": float(args.screen_beta),
        "num_flow_steps": int(args.num_flow_steps),
        "num_action_flow_samples": int(args.num_action_flow_samples),
        "screen_seed_start": int(args.screen_seed_start),
        "validation_seed_start": int(args.validation_seed_start),
        "confirmation_seed_start": int(args.confirmation_seed_start),
        "treatment_tie_break": "preregistered_cli_order",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
    )
    started = time.time()

    screen_jobs = [
        EvalJob(label, step, args.screen_beta)
        for label in arms
        for step in args.checkpoint_step
    ]
    _run_jobs(
        jobs=screen_jobs,
        split="screen",
        args=args,
        arms=arms,
        episodes=args.screen_episodes,
        seed_start=args.screen_seed_start,
    )
    screen = {}
    selected_for_validation = {}
    for label in arms:
        rows = {
            step: _success(
                _result_path(
                    args.output_dir,
                    "screen",
                    EvalJob(label, step, args.screen_beta),
                )
            )
            for step in args.checkpoint_step
        }
        screen[label] = rows
        selected_for_validation[label] = select_top_steps(
            rows,
            top_k=args.screen_top_k,
        )

    validation_jobs = [
        EvalJob(label, step, beta)
        for label in arms
        for step in selected_for_validation[label]
        for beta in args.beta
    ]
    _run_jobs(
        jobs=validation_jobs,
        split="validation",
        args=args,
        arms=arms,
        episodes=args.validation_episodes,
        seed_start=args.validation_seed_start,
    )
    validation = {
        label: {
            float(beta): {
                step: _success(
                    _result_path(
                        args.output_dir,
                        "validation",
                        EvalJob(label, step, beta),
                    )
                )
                for step in selected_for_validation[label]
            }
            for beta in args.beta
        }
        for label in arms
    }
    selected = {}
    for label in arms:
        beta, step, success = select_arm_winner(validation[label])
        selected[label] = {
            "beta": beta,
            "step": step,
            "validation_success": success,
        }
    winner = select_treatment(treatment_labels, selected)

    confirmation_jobs = [
        EvalJob("clean", None, None),
        EvalJob(
            anchor_label,
            int(selected[anchor_label]["step"]),
            float(selected[anchor_label]["beta"]),
        ),
        EvalJob(
            winner,
            int(selected[winner]["step"]),
            float(selected[winner]["beta"]),
        ),
    ]
    _run_jobs(
        jobs=confirmation_jobs,
        split="confirmation",
        args=args,
        arms=arms,
        episodes=args.confirmation_episodes,
        seed_start=args.confirmation_seed_start,
    )
    clean_path = _result_path(
        args.output_dir,
        "confirmation",
        EvalJob("clean", None, None),
    )
    anchor_path = _result_path(
        args.output_dir,
        "confirmation",
        confirmation_jobs[1],
    )
    treatment_path = _result_path(
        args.output_dir,
        "confirmation",
        confirmation_jobs[2],
    )
    treatment_vs_anchor = summarize_paired(
        _load_eval(anchor_path),
        _load_eval(treatment_path),
        baseline_path=anchor_path,
        candidate_path=treatment_path,
        bootstrap_samples=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        min_delta=0.0,
        min_ci_lower=0.0,
    )
    treatment_vs_clean = summarize_paired(
        _load_eval(clean_path),
        _load_eval(treatment_path),
        baseline_path=clean_path,
        candidate_path=treatment_path,
        bootstrap_samples=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed + 1,
        min_delta=-1.0,
        min_ci_lower=-1.0,
    )
    clean_checks = {
        "mean_noninferior_to_clean": (
            treatment_vs_clean["paired_delta"] >= 0.0
        ),
        "ci_lower_within_promotion_margin": (
            treatment_vs_clean["paired_delta_ci95"][0]
            >= float(args.clean_min_ci_lower)
        ),
    }
    checks = {
        "treatment_strictly_beats_matched_anchor": (
            treatment_vs_anchor["gate"] == "pass"
        ),
        **clean_checks,
    }
    summary = {
        "status": "ok",
        "gate": "pass" if all(checks.values()) else "fail",
        "gate_checks": checks,
        "selection": {
            "screen": screen,
            "selected_for_validation": selected_for_validation,
            "validation": validation,
            "selected": selected,
            "selected_treatment": winner,
            "checkpoint_tie_break": "earlier_checkpoint",
            "beta_tie_break": "larger_bc_prior",
            "treatment_tie_break": "preregistered_cli_order",
        },
        "confirmation": {
            "treatment_vs_anchor": treatment_vs_anchor,
            "treatment_vs_clean": treatment_vs_clean,
        },
        "thresholds": {
            "treatment_vs_anchor_min_delta": 0.0,
            "treatment_vs_anchor_min_ci_lower": 0.0,
            "treatment_vs_clean_min_delta": 0.0,
            "treatment_vs_clean_min_ci_lower": float(
                args.clean_min_ci_lower
            ),
        },
        "manifest": str(manifest_path),
        "elapsed_seconds": time.time() - started,
        "conclusion_scope": "single_training_seed_family_promotion_only",
        "route_b_claim_forbidden": True,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    manifest["status"] = "ok"
    manifest["summary"] = str(summary_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
    )
    return summary


def main() -> int:
    args = parse_args()
    summary = run_gate(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
