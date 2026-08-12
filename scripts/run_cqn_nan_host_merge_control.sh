#!/usr/bin/env bash
# Matched NaN control: only replace WorkspaceFast's device-side demo merge.
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT="${1:?artifact root}"
GPU_UUID="${2:-GPU-80b9cc0d-df5c-be12-e848-042d37578544}"
EGL_ID="${3:-2}"
RUN_DIR="${ROOT}/seed2"
mkdir -p "${RUN_DIR}"

echo "[host-merge] start gpu=${GPU_UUID} egl=${EGL_ID} $(date --iso-8601=seconds)" \
  > "${ROOT}/controller.log"
set +e
env \
  ROBOBASE_HOST_MERGE=1 \
  CUDA_VISIBLE_DEVICES="${GPU_UUID}" \
  MUJOCO_EGL_DEVICE_ID="${EGL_ID}" \
  MUJOCO_GL=egl \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python train_fast.py \
    launch=cqn_as_pixel_bigym_stage163c_official_qc8 \
    env=bigym/move_plate \
    seed=2 \
    xla_mem_fraction=0.45 \
    env.truncate_demo_at_success=true \
    env.obs_std_floor_relative=0.0 \
    eval_every_steps=1000000 \
    num_eval_episodes=0 \
    num_eval_envs=0 \
    log_eval_video=false \
    save_snapshot=true \
    snapshot_every_n=5000 \
    save_csv=true \
    wandb.use=false \
    nonfinite_dump=true \
    nonfinite_dump_keep_batches=3 \
    nonfinite_dump_include_uint8=true \
    nonfinite_dump_save_state=true \
    hydra.run.dir="${RUN_DIR}" \
    > "${RUN_DIR}/train_fast.log" 2>&1
status=$?
set -e
echo "[host-merge] exit=${status} $(date --iso-8601=seconds)" \
  >> "${ROOT}/controller.log"
exit "${status}"
