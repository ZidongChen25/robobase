#!/usr/bin/env bash
# Stage-153: hierarchical epsilon-bin exploration gate (cqn-flow.md 35).
# Waits for the stage152b chains to release GPUs, then runs 3 seeds of
# vanilla + bin_explore_probs=[0.002,0.004,0.008] at full demos.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_A="${1:-1}"
GPU_B="${2:-5}"
WAIT_LOG="${3:-exp_local/cqn_stage152_coarse_flow/stage152b_master.log}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_stage153_bin_explore"
mkdir -p "${BASE}"
cp "$0" "${BASE}/stage153_controller.${STAMP}.sh"

echo "[stage153] waiting for stage152b completion marker"
until grep -q "sealed+extension complete" "${WAIT_LOG}" 2>/dev/null; do
  sleep 120
done
echo "[stage153] GPUs released, starting"

run_seed () {
  local SEED="$1" GPU="$2"
  local RUN_DIR="${BASE}/move_plate_binexp_seed${SEED}_gpu${GPU}_${STAMP}"
  echo "[stage153] binexp seed${SEED} on GPU${GPU}"
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
    .venv/bin/python train.py \
    launch=cqn_as_pixel_bigym_stage153_bin_explore_gate \
    env=bigym/move_plate \
    seed="${SEED}" \
    save_csv=true \
    wandb.name="cqn_as_stage153_binexp_seed${SEED}_move_plate" \
    hydra.run.dir="${RUN_DIR}" \
    > "${RUN_DIR}.launch.log" 2>&1
  echo "[stage153] done binexp seed${SEED}"
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
echo "[stage153] all arms complete"
for CSV in "${BASE}"/move_plate_*_"${STAMP}"/eval.csv; do
  NAME="$(basename "$(dirname "${CSV}")")"
  echo -n "${NAME}: "
  awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="episode_success") c=i; next} {printf "%s%% ", $c*100}' "${CSV}"
  echo
done
