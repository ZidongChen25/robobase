#!/usr/bin/env bash
# Stage-152: coarse-flow (CCFF) demo-scarce gate (cqn-flow.md 34).
# GPU_A chain: A=vanilla@demos10 seeds 1-3, then B-full seed 1.
# GPU_B chain: B=CCFF@demos10 seeds 1-3, then B-full seeds 2-3.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_A="${1:-1}"
GPU_B="${2:-5}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_stage152_coarse_flow"
mkdir -p "${BASE}"
cp "$0" "${BASE}/stage152_controller.${STAMP}.sh"

run_arm () {
  local LAUNCH="$1" ARM="$2" SEED="$3" GPU="$4" DEMOS="$5"
  local RUN_DIR="${BASE}/move_plate_${ARM}_seed${SEED}_gpu${GPU}_${STAMP}"
  echo "[stage152] ${ARM} seed${SEED} demos=${DEMOS} on GPU${GPU}"
  local DEMO_ARGS=()
  if [[ "${DEMOS}" != "full" ]]; then
    DEMO_ARGS=("demos=${DEMOS}" "env.expected_successful_demos=null")
  fi
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
    .venv/bin/python train.py \
    launch="${LAUNCH}" \
    env=bigym/move_plate \
    seed="${SEED}" \
    save_csv=true \
    "${DEMO_ARGS[@]}" \
    wandb.name="cqn_as_stage152_${ARM}_seed${SEED}_move_plate" \
    hydra.run.dir="${RUN_DIR}" \
    > "${RUN_DIR}.launch.log" 2>&1
  echo "[stage152] done ${ARM} seed${SEED}"
}

chain_a () {
  for SEED in 1 2 3; do
    run_arm cqn_as_pixel_bigym_value_fidelity_gate vanilla_d10 "${SEED}" "${GPU_A}" 10
  done
  run_arm cqn_as_pixel_bigym_stage152_coarse_flow_gate ccff_full 1 "${GPU_A}" full
}

chain_b () {
  for SEED in 1 2 3; do
    run_arm cqn_as_pixel_bigym_stage152_coarse_flow_gate ccff_d10 "${SEED}" "${GPU_B}" 10
  done
  for SEED in 2 3; do
    run_arm cqn_as_pixel_bigym_stage152_coarse_flow_gate ccff_full "${SEED}" "${GPU_B}" full
  done
}

chain_a &
PID_A=$!
chain_b &
PID_B=$!
wait "${PID_A}" "${PID_B}"
echo "[stage152] all arms complete"
for CSV in "${BASE}"/move_plate_*_"${STAMP}"/eval.csv; do
  NAME="$(basename "$(dirname "${CSV}")")"
  echo -n "${NAME}: "
  awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="episode_success") c=i; next} {printf "%s%% ", $c*100}' "${CSV}"
  echo
done
