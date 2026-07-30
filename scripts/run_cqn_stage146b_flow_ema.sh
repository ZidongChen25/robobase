#!/usr/bin/env bash
# Stage-146b (flow+CQN line): EMA rollout weights, single variable vs
# v1c M=8.  Three seeds.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_A="${1:-1}"
GPU_B="${2:-5}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_stage146_flow_rerank"
mkdir -p "${BASE}"
cp "$0" "${BASE}/stage146b_controller.${STAMP}.sh"

run_arm () {
  local SEED="$1" GPU="$2"
  local RUN_DIR="${BASE}/move_plate_flowema_m8_seed${SEED}_gpu${GPU}_${STAMP}"
  echo "[stage146b] start seed${SEED} on GPU${GPU}"
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
    .venv/bin/python train.py \
    launch=cqn_as_pixel_bigym_stage146b_flow_ema_gate \
    env=bigym/move_plate \
    seed="${SEED}" \
    wandb.name="cqn_as_stage146b_flowema_m8_seed${SEED}_move_plate" \
    hydra.run.dir="${RUN_DIR}" \
    > "${RUN_DIR}.launch.log" 2>&1
  echo "[stage146b] done seed${SEED}"
}

( run_arm 1 "${GPU_A}"; run_arm 3 "${GPU_A}" ) &
PID_A=$!
run_arm 2 "${GPU_B}" &
PID_B=$!
wait "${PID_A}" "${PID_B}"
echo "[stage146b] all arms complete"
