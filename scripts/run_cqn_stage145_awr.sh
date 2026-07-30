#!/usr/bin/env bash
# Stage-145 / Route-(c): AWR-weighted BC gate, 3 training seeds, 10.5k
# matched budget.  Task target: clean CQN-AS 3-seed validation-best mean
# 72.00% (cqn-flow.md 21.104); internal validation-best convention.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_A="${1:-1}"
GPU_B="${2:-5}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_stage145_awr"
mkdir -p "${BASE}"
cp "$0" "${BASE}/stage145_controller.${STAMP}.sh"

run_arm () {
  local SEED="$1" GPU="$2"
  local RUN_DIR="${BASE}/move_plate_awr_seed${SEED}_gpu${GPU}_${STAMP}"
  echo "[stage145] start seed${SEED} on GPU${GPU}"
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
    .venv/bin/python train.py \
    launch=cqn_as_pixel_bigym_stage145_awr_gate \
    env=bigym/move_plate \
    seed="${SEED}" \
    wandb.name="cqn_as_stage145_awr_seed${SEED}_move_plate" \
    hydra.run.dir="${RUN_DIR}" \
    > "${RUN_DIR}.launch.log" 2>&1
  echo "[stage145] done seed${SEED}"
}

# Three seeds over two GPUs: A takes 1 and 3, B takes 2.
( run_arm 1 "${GPU_A}"; run_arm 3 "${GPU_A}" ) &
PID_A=$!
run_arm 2 "${GPU_B}" &
PID_B=$!
wait "${PID_A}" "${PID_B}"
echo "[stage145] all arms complete"
