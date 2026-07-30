#!/usr/bin/env bash
# Stage-147 (CQN-value line): clean CQN-AS + canonical MC anchor, 3 seeds.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_A="${1:-1}"
GPU_B="${2:-5}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_stage147_clean_mc"
mkdir -p "${BASE}"
cp "$0" "${BASE}/stage147_controller.${STAMP}.sh"

run_arm () {
  local SEED="$1" GPU="$2"
  local RUN_DIR="${BASE}/move_plate_cleanmc_seed${SEED}_gpu${GPU}_${STAMP}"
  echo "[stage147] start seed${SEED} on GPU${GPU}"
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
    .venv/bin/python train.py \
    launch=cqn_as_pixel_bigym_stage147_clean_mc_gate \
    env=bigym/move_plate \
    seed="${SEED}" \
    wandb.name="cqn_as_stage147_cleanmc_seed${SEED}_move_plate" \
    hydra.run.dir="${RUN_DIR}" \
    > "${RUN_DIR}.launch.log" 2>&1
  echo "[stage147] done seed${SEED}"
}

( run_arm 1 "${GPU_A}"; run_arm 3 "${GPU_A}" ) &
PID_A=$!
run_arm 2 "${GPU_B}" &
PID_B=$!
wait "${PID_A}" "${PID_B}"
echo "[stage147] all arms complete"
