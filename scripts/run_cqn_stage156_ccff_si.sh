#!/usr/bin/env bash
# Stage-156: CCFF + self-imitation (S) and + coarse epsilon-bin explore (SE).
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_A="${1:-1}"
GPU_B="${2:-5}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_stage156_ccff_si"
mkdir -p "${BASE}"
cp "$0" "${BASE}/stage156_controller.${STAMP}.sh"

run_arm () {
  local LAUNCH="$1" ARM="$2" SEED="$3" GPU="$4"
  local RUN_DIR="${BASE}/move_plate_${ARM}_seed${SEED}_gpu${GPU}_${STAMP}"
  echo "[stage156] ${ARM} seed${SEED} on GPU${GPU}"
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
    .venv/bin/python train.py \
    launch="${LAUNCH}" \
    env=bigym/move_plate \
    seed="${SEED}" \
    save_csv=true \
    wandb.name="cqn_as_stage156_${ARM}_seed${SEED}_move_plate" \
    hydra.run.dir="${RUN_DIR}" \
    > "${RUN_DIR}.launch.log" 2>&1
  echo "[stage156] done ${ARM} seed${SEED}"
}

worker_a () {
  for SEED in 1 2 3; do
    run_arm cqn_as_pixel_bigym_stage156_ccff_si_gate si "${SEED}" "${GPU_A}"
  done
}
worker_b () {
  for SEED in 1 2 3; do
    run_arm cqn_as_pixel_bigym_stage156_ccff_si_explore_gate siexp "${SEED}" "${GPU_B}"
  done
}

worker_a &
PID_A=$!
worker_b &
PID_B=$!
wait "${PID_A}" "${PID_B}"
echo "[stage156] all arms complete"
for CSV in "${BASE}"/move_plate_*_"${STAMP}"/eval.csv; do
  NAME="$(basename "$(dirname "${CSV}")")"
  echo -n "${NAME}: "
  awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="episode_success") c=i; next} {printf "%s%% ", $c*100}' "${CSV}"
  echo
done
