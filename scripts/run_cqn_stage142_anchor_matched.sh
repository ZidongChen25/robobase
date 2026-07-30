#!/usr/bin/env bash
# Stage-142: anchor-matched FLOQ arms, seeds 1 and 2, one per GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_A="${1:-1}"
GPU_B="${2:-5}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_stage142_anchor_matched"
mkdir -p "${BASE}"
cp "$0" "${BASE}/stage142_controller.${STAMP}.sh"

run_arm () {
  local SEED="$1" GPU="$2"
  local RUN_DIR="${BASE}/move_plate_floq_mc_seed${SEED}_gpu${GPU}_${STAMP}"
  echo "[stage142] start seed${SEED} on GPU${GPU}"
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
    .venv/bin/python train.py \
    launch=cqn_flow_floq_stage142_anchor_matched_gate \
    env=bigym/move_plate \
    seed="${SEED}" \
    wandb.name="cqn_flow_stage142_floq_mc_seed${SEED}_move_plate" \
    hydra.run.dir="${RUN_DIR}" \
    > "${RUN_DIR}.launch.log" 2>&1
  echo "[stage142] done seed${SEED}"
}

run_arm 1 "${GPU_A}" &
PID_A=$!
run_arm 2 "${GPU_B}" &
PID_B=$!
wait "${PID_A}" "${PID_B}"
echo "[stage142] all arms complete"
