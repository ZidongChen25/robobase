# Environments and Git Hygiene

This repo is managed with `uv` through `pyproject.toml`.

## What To Commit

Commit source code, configs, launch scripts, tests, and documentation:

- `robobase/**`
- `robobase/cfgs/**`
- `scripts/**`
- `tests/**`
- `BiGym/*.sh`
- `RoboMimic/*.py` and `RoboMimic/*.sh`
- `pyproject.toml`, `uv.lock`, `setup.py`
- project docs such as `README.md`, `backend.md`, and reports you want preserved

For the current working tree, the untracked files that look commit-worthy are:

- `BiGym/run_bigym_dp_transformer_tasks_gpu2_gpu5.sh`
- `BiGym/run_move_plate_diffusion_transformer_gpu2.sh`
- `robobase/cfgs/env/pusht.yaml`
- `robobase/cfgs/launch/act_pixel_bigym_lazy.yaml`
- `robobase/cfgs/launch/dp_pixel_bigym_move_plate_transformer.yaml`
- `robobase/cfgs/launch/dp_pixel_bigym_transformer_ddpm.yaml`
- `robobase/cfgs/launch/dp_pixel_pusht.yaml`
- `robobase/cfgs/launch/fm_pixel_bigym_transformer.yaml`
- `robobase/envs/pusht.py`
- `robobase/replay_buffer/bigym_lazy_replay.py`
- `scripts/cache_bigym_pixel_demos.py`
- `scripts/eval_flow_step_checkpoint.py`
- `scripts/run_flow_step_sweep.py`
- `tests/unit/envs/test_bigym.py`
- `tests/unit/envs/test_pusht.py`
- `tests/unit/replay_buffer/test_bigym_lazy_replay.py`

## What To Ignore

Do not commit generated training outputs, local caches, process files, backup files,
datasets, or virtual environments:

- `exp_local/`
- `wandb/`
- `.venv/`
- `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`
- `__pycache__/`, `*.pyc`
- `.hydra/`, `multirun/`, `outputs/`
- `logs/`, `*.log`, `*.pid`, `latest_*_run`
- `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`
- `*.bak*`, `*.tmp`, editor swap files
- `third_party_datasets/`, `/datasets/`, `/data/`, local Hugging Face caches

In `BiGym/`, commit the shell scripts and ignore `BiGym/logs/`, `*.pid`, and
`latest_*_run` symlinks.

## Core Repo Environment

Core dependencies are the codebase/runtime pieces shared across tasks:

```bash
uv sync --extra jax-cuda12
```

Use CPU JAX instead:

```bash
uv sync --extra jax
```

`dev` is a default uv dependency group and installs `pytest` and `pre-commit`.

## Benchmark Extras

Benchmarks are optional and should be installed only when needed:

```bash
uv sync --extra jax-cuda12 --extra dmc
uv sync --extra jax-cuda12 --extra robomimic
uv sync --extra jax-cuda12 --extra bigym
uv sync --extra jax-cuda12 --extra pusht
uv sync --extra jax-cuda12 --extra rlbench
uv sync --extra jax-cuda12 --extra d4rl
```

Notes:

- `robomimic` here means this repo's HDF5 dataset reader plus `robosuite` live eval.
- `bigym` uses the pinned upstream BiGym git commit in `pyproject.toml`; local scripts
  still use repo configs and `scripts/cache_bigym_pixel_demos.py`.
- `pusht` uses LeRobot's `lerobot/pusht` dataset format and `gym-pusht`; `pymunk<7`
  is required for `gym-pusht==0.1.6`.
- `rlbench` still needs CoppeliaSim/PyRep system setup.
- `d4rl` still needs MuJoCo-compatible local setup.

## Lockfile

Update the uv lockfile after dependency changes:

```bash
uv lock
```

Do not maintain checked-in `requirements*.txt` files. Use `pyproject.toml` and
`uv.lock` as the canonical environment definition. If a one-off deployment
target requires a requirements file, generate it outside the repo or keep it
untracked:

```bash
uv export --no-hashes --no-emit-project --extra jax-cuda12
```
