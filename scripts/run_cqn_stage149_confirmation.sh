#!/usr/bin/env bash
# Stage-149: sealed confirmation for the flow+CQN line.
# Phase 1: matched-platform clean seeds 2/3 (seed1 exists: stage2 clean).
# Phase 2: sealed 50-episode evals (fresh eval-seed-start 600) of every
#          arm at (a) nearest-snapshot-to-validation-best [primary] and
#          (b) final snapshot [selection-free secondary].
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_A="${1:-1}"
GPU_B="${2:-5}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_stage149_confirmation"
mkdir -p "${BASE}"
cp "$0" "${BASE}/stage149_controller.${STAMP}.sh"

train_clean () {
  local SEED="$1" GPU="$2"
  local RUN_DIR="${BASE}/move_plate_clean_seed${SEED}_gpu${GPU}_${STAMP}"
  echo "[stage149] train clean seed${SEED} on GPU${GPU}"
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
    .venv/bin/python train.py \
    launch=cqn_as_pixel_bigym_value_fidelity_gate \
    env=bigym/move_plate \
    seed="${SEED}" \
    save_csv=true \
    wandb.name="cqn_as_stage149_clean_seed${SEED}_move_plate" \
    hydra.run.dir="${RUN_DIR}" \
    > "${RUN_DIR}.launch.log" 2>&1
  echo "[stage149] clean seed${SEED} done"
}

train_clean 2 "${GPU_A}" &
PID_A=$!
train_clean 3 "${GPU_B}" &
PID_B=$!
wait "${PID_A}" "${PID_B}"
echo "[stage149] clean training complete"
