#!/usr/bin/env bash
# Stage-155: pure-flow no-selection control (cqn-flow.md 37).
# Waits for stage154, then 3 seeds of CCFF-platform pure flow.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_A="${1:-1}"
GPU_B="${2:-5}"
WAIT_LOG="${3:-exp_local/cqn_stage154_ccff_tdoff/stage154_master.log}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_stage155_pure_flow"
mkdir -p "${BASE}"
cp "$0" "${BASE}/stage155_controller.${STAMP}.sh"

echo "[stage155] waiting for stage154 completion marker"
until grep -q "all arms complete" "${WAIT_LOG}" 2>/dev/null; do
  sleep 120
done
echo "[stage155] GPUs released, starting"

run_seed () {
  local SEED="$1" GPU="$2"
  local RUN_DIR="${BASE}/move_plate_pureflow_seed${SEED}_gpu${GPU}_${STAMP}"
  echo "[stage155] pureflow seed${SEED} on GPU${GPU}"
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
    .venv/bin/python train.py \
    launch=cqn_as_pixel_bigym_stage155_pure_flow_gate \
    env=bigym/move_plate \
    seed="${SEED}" \
    save_csv=true \
    wandb.name="cqn_as_stage155_pureflow_seed${SEED}_move_plate" \
    hydra.run.dir="${RUN_DIR}" \
    > "${RUN_DIR}.launch.log" 2>&1
  echo "[stage155] done pureflow seed${SEED}"
}

worker_a () {
  run_seed 1 "${GPU_A}"
  run_seed 3 "${GPU_A}"
}
worker_b () {
  run_seed 2 "${GPU_B}"
}

worker_a &
PID_A=$!
worker_b &
PID_B=$!
wait "${PID_A}" "${PID_B}"
echo "[stage155] all arms complete"
for CSV in "${BASE}"/move_plate_*_"${STAMP}"/eval.csv; do
  NAME="$(basename "$(dirname "${CSV}")")"
  echo -n "${NAME}: "
  awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="episode_success") c=i; next} {printf "%s%% ", $c*100}' "${CSV}"
  echo
done
