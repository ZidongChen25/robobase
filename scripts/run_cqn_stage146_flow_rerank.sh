#!/usr/bin/env bash
# Stage-146 (flow+CQN line): flow-BC proposals + calibrated-Q rerank.
# Arms: M=1 (flow-BC-alone control) and M=8 (rerank treatment), 3 seeds
# each, 10.5k matched budget.  Verdict: M=8 must beat M=1 (rerank
# contribution) and both are reported against clean CQN-AS 72.00%.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_A="${1:-1}"
GPU_B="${2:-5}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_stage146_flow_rerank"
mkdir -p "${BASE}"
cp "$0" "${BASE}/stage146_controller.${STAMP}.sh"

run_arm () {
  local SEED="$1" GPU="$2" M="$3"
  local RUN_DIR="${BASE}/move_plate_flowrr_m${M}_seed${SEED}_gpu${GPU}_${STAMP}"
  echo "[stage146] start m${M} seed${SEED} on GPU${GPU}"
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
    .venv/bin/python train.py \
    launch=cqn_as_pixel_bigym_stage146_flow_rerank_gate \
    env=bigym/move_plate \
    seed="${SEED}" \
    method.flow_policy_candidates="${M}" \
    wandb.name="cqn_as_stage146_flowrr_m${M}_seed${SEED}_move_plate" \
    hydra.run.dir="${RUN_DIR}" \
    > "${RUN_DIR}.launch.log" 2>&1
  echo "[stage146] done m${M} seed${SEED}"
}

(
  run_arm 1 "${GPU_A}" 8
  run_arm 3 "${GPU_A}" 8
  run_arm 3 "${GPU_A}" 1
) &
PID_A=$!
(
  run_arm 2 "${GPU_B}" 8
  run_arm 1 "${GPU_B}" 1
  run_arm 2 "${GPU_B}" 1
) &
PID_B=$!
wait "${PID_A}" "${PID_B}"
echo "[stage146] all arms complete"
