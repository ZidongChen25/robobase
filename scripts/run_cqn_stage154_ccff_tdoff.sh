#!/usr/bin/env bash
# Stage-154: TD-off control for coarse-flow (cqn-flow.md 36).
# Waits for stage153 to finish, then 3 seeds of CCFF with critic_lambda=0.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_A="${1:-1}"
GPU_B="${2:-5}"
WAIT_LOG="${3:-exp_local/cqn_stage153_bin_explore/stage153_master.log}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_stage154_ccff_tdoff"
mkdir -p "${BASE}"
cp "$0" "${BASE}/stage154_controller.${STAMP}.sh"

echo "[stage154] waiting for stage153 completion marker"
until grep -q "all arms complete" "${WAIT_LOG}" 2>/dev/null; do
  sleep 120
done
echo "[stage154] GPUs released, starting"

run_seed () {
  local SEED="$1" GPU="$2"
  local RUN_DIR="${BASE}/move_plate_tdoff_seed${SEED}_gpu${GPU}_${STAMP}"
  echo "[stage154] tdoff seed${SEED} on GPU${GPU}"
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
    .venv/bin/python train.py \
    launch=cqn_as_pixel_bigym_stage154_ccff_tdoff_gate \
    env=bigym/move_plate \
    seed="${SEED}" \
    save_csv=true \
    wandb.name="cqn_as_stage154_tdoff_seed${SEED}_move_plate" \
    hydra.run.dir="${RUN_DIR}" \
    > "${RUN_DIR}.launch.log" 2>&1
  echo "[stage154] done tdoff seed${SEED}"
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
echo "[stage154] all arms complete"
for CSV in "${BASE}"/move_plate_*_"${STAMP}"/eval.csv; do
  NAME="$(basename "$(dirname "${CSV}")")"
  echo -n "${NAME}: "
  awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="episode_success") c=i; next} {printf "%s%% ", $c*100}' "${CSV}"
  echo
done
