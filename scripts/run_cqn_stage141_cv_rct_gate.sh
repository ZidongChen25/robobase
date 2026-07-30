#!/usr/bin/env bash
# Stage-141: CV-adjusted causal RCT, matched control/treatment arms.
#
# 4 runs = 2 training seeds x {control cv_rct_weight=0.0, treatment 0.1}.
# One seed per GPU; the two arms of a seed run sequentially on the same GPU
# so each control/treatment pair shares identical hardware. 10.5k frames,
# fixed pre-registered endpoint; no post-hoc checkpoint selection.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_A="${1:-1}"
GPU_B="${2:-5}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_stage141_cv_rct"
mkdir -p "${BASE}"
cp "$0" "${BASE}/stage141_controller.${STAMP}.sh"

run_seed () {
  local SEED="$1" GPU="$2"
  for W in 0.0 0.1; do
    local TAG="seed${SEED}_w${W/0./0p}"
    local RUN_DIR="${BASE}/move_plate_cv_rct_${TAG}_gpu${GPU}_${STAMP}"
    echo "[stage141] start ${TAG} on GPU${GPU}"
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
      .venv/bin/python train.py \
      launch=cqn_as_pixel_bigym_stage141_cv_rct_gate \
      env=bigym/move_plate \
      seed="${SEED}" \
      method.cv_rct_weight="${W}" \
      wandb.name="cqn_as_stage141_cv_rct_${TAG}_move_plate" \
      hydra.run.dir="${RUN_DIR}" \
      > "${RUN_DIR}.launch.log" 2>&1
    echo "[stage141] done ${TAG}"
  done
}

run_seed 1 "${GPU_A}" &
PID_A=$!
run_seed 2 "${GPU_B}" &
PID_B=$!
wait "${PID_A}" "${PID_B}"
echo "[stage141] all arms complete"
