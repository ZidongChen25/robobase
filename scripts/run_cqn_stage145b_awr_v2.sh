#!/usr/bin/env bash
# Stage-145b: AWR v2 (beta 0.1, single change) x 3 seeds, plus the missing
# no-AWR decoupled+MC0.1 control seeds 2/3 (seed1=52% exists historically).
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_A="${1:-1}"
GPU_B="${2:-5}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_stage145_awr"
mkdir -p "${BASE}"
cp "$0" "${BASE}/stage145b_controller.${STAMP}.sh"

run_arm () {
  local SEED="$1" GPU="$2" BETA="$3" LABEL="$4"
  local RUN_DIR="${BASE}/move_plate_${LABEL}_seed${SEED}_gpu${GPU}_${STAMP}"
  echo "[stage145b] start ${LABEL} seed${SEED} on GPU${GPU}"
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
    .venv/bin/python train.py \
    launch=cqn_as_pixel_bigym_stage145_awr_gate \
    env=bigym/move_plate \
    seed="${SEED}" \
    method.awr_beta="${BETA}" \
    wandb.name="cqn_as_stage145b_${LABEL}_seed${SEED}_move_plate" \
    hydra.run.dir="${RUN_DIR}" \
    > "${RUN_DIR}.launch.log" 2>&1
  echo "[stage145b] done ${LABEL} seed${SEED}"
}

# GPU_A: v2 seeds 1,3 then control seed 3.  GPU_B: v2 seed 2 then control 2.
(
  run_arm 1 "${GPU_A}" 0.1 awr_b0p1
  run_arm 3 "${GPU_A}" 0.1 awr_b0p1
  run_arm 3 "${GPU_A}" null mc_only
) &
PID_A=$!
(
  run_arm 2 "${GPU_B}" 0.1 awr_b0p1
  run_arm 2 "${GPU_B}" null mc_only
) &
PID_B=$!
wait "${PID_A}" "${PID_B}"
echo "[stage145b] all arms complete"
