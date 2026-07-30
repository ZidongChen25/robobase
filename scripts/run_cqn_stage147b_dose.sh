#!/usr/bin/env bash
# Stage-147b: canonical MC anchor dose fallback after weight 0.1 interfered
# with margin ranking (~40% of critic gradient on shared logits).
# Arms: w=0.02 (dose reduction) and w=0.1+value_only (gradient routing to
# the dueling value stream), 3 seeds each.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_A="${1:-1}"
GPU_B="${2:-5}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_stage147_clean_mc"
mkdir -p "${BASE}"
cp "$0" "${BASE}/stage147b_controller.${STAMP}.sh"

run_arm () {
  local SEED="$1" GPU="$2" LABEL="$3"; shift 3
  local RUN_DIR="${BASE}/move_plate_${LABEL}_seed${SEED}_gpu${GPU}_${STAMP}"
  echo "[stage147b] start ${LABEL} seed${SEED} on GPU${GPU}"
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
    .venv/bin/python train.py \
    launch=cqn_as_pixel_bigym_stage147_clean_mc_gate \
    env=bigym/move_plate \
    seed="${SEED}" \
    "$@" \
    wandb.name="cqn_as_stage147b_${LABEL}_seed${SEED}_move_plate" \
    hydra.run.dir="${RUN_DIR}" \
    > "${RUN_DIR}.launch.log" 2>&1
  echo "[stage147b] done ${LABEL} seed${SEED}"
}

(
  run_arm 1 "${GPU_A}" mcw0p02 method.mc_return_weight=0.02
  run_arm 3 "${GPU_A}" mcw0p02 method.mc_return_weight=0.02
  run_arm 3 "${GPU_A}" mcvonly method.mc_return_weight=0.1 method.mc_return_value_only=true
) &
PID_A=$!
(
  run_arm 2 "${GPU_B}" mcw0p02 method.mc_return_weight=0.02
  run_arm 1 "${GPU_B}" mcvonly method.mc_return_weight=0.1 method.mc_return_value_only=true
  run_arm 2 "${GPU_B}" mcvonly method.mc_return_weight=0.1 method.mc_return_value_only=true
) &
PID_B=$!
wait "${PID_A}" "${PID_B}"
echo "[stage147b] all arms complete"
