#!/usr/bin/env bash
# Stage-146c: budget and candidate-count scaling of the EMA flow+rerank.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_A="${1:-1}"
GPU_B="${2:-5}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_stage146_flow_rerank"
mkdir -p "${BASE}"
cp "$0" "${BASE}/stage146c_controller.${STAMP}.sh"

run_arm () {
  local SEED="$1" GPU="$2" LABEL="$3"; shift 3
  local RUN_DIR="${BASE}/move_plate_${LABEL}_seed${SEED}_gpu${GPU}_${STAMP}"
  echo "[stage146c] start ${LABEL} seed${SEED} on GPU${GPU}"
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
    .venv/bin/python train.py \
    "$@" \
    env=bigym/move_plate \
    seed="${SEED}" \
    wandb.name="cqn_as_stage146c_${LABEL}_seed${SEED}_move_plate" \
    hydra.run.dir="${RUN_DIR}" \
    > "${RUN_DIR}.launch.log" 2>&1
  echo "[stage146c] done ${LABEL} seed${SEED}"
}

(
  run_arm 1 "${GPU_A}" b20k launch=cqn_as_pixel_bigym_stage146c_flow_b20k_gate
  run_arm 3 "${GPU_A}" b20k launch=cqn_as_pixel_bigym_stage146c_flow_b20k_gate
  run_arm 3 "${GPU_A}" m16 launch=cqn_as_pixel_bigym_stage146b_flow_ema_gate method.flow_policy_candidates=16
) &
PID_A=$!
(
  run_arm 2 "${GPU_B}" b20k launch=cqn_as_pixel_bigym_stage146c_flow_b20k_gate
  run_arm 1 "${GPU_B}" m16 launch=cqn_as_pixel_bigym_stage146b_flow_ema_gate method.flow_policy_candidates=16
  run_arm 2 "${GPU_B}" m16 launch=cqn_as_pixel_bigym_stage146b_flow_ema_gate method.flow_policy_candidates=16
) &
PID_B=$!
wait "${PID_A}" "${PID_B}"
echo "[stage146c] all arms complete"
