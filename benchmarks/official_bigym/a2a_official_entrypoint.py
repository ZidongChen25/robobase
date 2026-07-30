#!/usr/bin/env python3
"""Run an upstream script with narrowly scoped dependency compatibility fixes."""

from __future__ import annotations

from pathlib import Path
import os
import runpy
import site
import sys


def _disabled_latent_plotter(*args, **kwargs):
    """Skip CPU plotting after upstream has executed its diagnostic data path."""
    del args, kwargs
    return {"avg_tsne_distance": float("nan")}


def patch_latent_visualization_plotter(runner_module=None) -> None:
    """Disable only t-SNE/plotting while preserving upstream RNG consumption."""
    if runner_module is None:
        from roboverse_learn.il.runners import default_runner as runner_module

    runner_module.plot_all_latent_visualizations = _disabled_latent_plotter


def patch_eval_trajectory(
    trajectory: str | Path,
    *,
    task: str = "close_box",
    task_class=None,
) -> Path:
    """Point an upstream task at an audited evaluation-only trajectory file."""

    trajectory = Path(trajectory).expanduser().resolve()
    if not trajectory.is_file():
        raise FileNotFoundError(trajectory)
    if "v2" not in trajectory.name:
        raise ValueError(
            "RoboVerse dispatches trajectory parsers from the path; the evaluation "
            "trajectory filename must contain 'v2'."
        )
    if task != "close_box":
        raise ValueError(f"Unsupported evaluation trajectory task: {task!r}")
    if task_class is None:
        from roboverse_pack.tasks.rlbench.close_box import CloseBoxTask

        task_class = CloseBoxTask
    task_class.traj_filepath = str(trajectory)
    return trajectory


def patch_eval_policy_cfg_joint_pos(eval_runner_class=None) -> None:
    """Correct eval-only action metadata for custom joint-position checkpoints."""

    if eval_runner_class is None:
        from roboverse_learn.il.runners.default_eval_runner import DefaultEvalRunner

        eval_runner_class = DefaultEvalRunner
    original = eval_runner_class._init_policy

    def _init_policy_with_joint_pos(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.policy_cfg.obs_config.obs_type = "joint_pos"
        self.policy_cfg.action_config.action_type = "joint_pos"
        self.policy_cfg.action_config.delta = False

    eval_runner_class._init_policy = _init_policy_with_joint_pos


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: a2a_official_entrypoint.py UPSTREAM_SCRIPT [ARGS ...]")
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    extra_site_packages = os.environ.get("ROBOBASE_EXTRA_SITE_PACKAGES")
    if extra_site_packages:
        # PYTHONPATH does not execute editable-install .pth files.  The mixed
        # JAX + Isaac evaluator needs the official environment's IsaacLab and
        # RoboVerse editable finders without moving policy code into that venv.
        site.addsitedir(extra_site_packages)

    from benchmarks.official_bigym.a2a_upstream import patch_diffusers_compat

    script = Path(sys.argv[1]).expanduser().resolve()
    if not script.is_file():
        raise FileNotFoundError(script)
    patch_diffusers_compat()
    eval_trajectory = os.environ.get("ROBOBASE_OFFICIAL_EVAL_TRAJECTORY")
    if eval_trajectory:
        patch_eval_trajectory(
            eval_trajectory,
            task=os.environ.get("ROBOBASE_OFFICIAL_EVAL_TASK", "close_box"),
        )
    if os.environ.get("ROBOBASE_OFFICIAL_EVAL_FORCE_JOINT_POS") == "1":
        patch_eval_policy_cfg_joint_pos()
    if os.environ.get("ROBOBASE_OFFICIAL_SKIP_LATENT_VIZ") == "1":
        # Keep the second validation-loader traversal and flow diagnostic calls:
        # constructing that iterator and sampling the diagnostic both consume the
        # same RNG as upstream. Only the deterministic CPU t-SNE/plot generation
        # is replaced, so later training epochs follow the upstream RNG path.
        patch_latent_visualization_plotter()
    if os.environ.get("ROBOBASE_JAX_A2A_EVAL") == "1":
        from benchmarks.official_roboverse.jax_eval_runner import (
            patch_default_eval_runner,
        )

        patch_default_eval_runner()
    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name="__main__")
    if os.environ.get("ROBOBASE_JAX_A2A_EVAL") == "1":
        # Isaac Sim and JAX both leave native worker threads behind.  Once the
        # upstream runner has closed the simulator and returned, normal Python
        # teardown can hang indefinitely; all result files are already closed.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
