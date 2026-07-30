#!/usr/bin/env bash
# Stage-158: 100k official config + epsilon-bin exploration + margin decay.
# One seed per GPU; on completion, run the sibling probe on the final
# snapshot of each (Stage-143/151/157 protocol).
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_A="${1:-1}"
GPU_B="${2:-5}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_stage158_explore_100k"
mkdir -p "${BASE}"
cp "$0" "${BASE}/stage158_controller.${STAMP}.sh"

train_seed () {
  local SEED="$1" GPU="$2"
  local RUN_DIR="${BASE}/move_plate_exp100k_seed${SEED}_gpu${GPU}_${STAMP}"
  echo "[stage158] train seed${SEED} on GPU${GPU}"
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
    .venv/bin/python train.py \
    launch=cqn_as_pixel_bigym_stage158_explore_100k \
    env=bigym/move_plate \
    seed="${SEED}" \
    save_csv=true \
    wandb.name="cqn_as_stage158_exp100k_seed${SEED}_move_plate" \
    hydra.run.dir="${RUN_DIR}" \
    > "${RUN_DIR}.launch.log" 2>&1
  echo "[stage158] done train seed${SEED}"
  echo "[stage158] probe seed${SEED} on GPU${GPU}"
  MUJOCO_GL=egl .venv/bin/python scripts/analyze_cqn_branch_counterfactual.py \
    --run-dir "${RUN_DIR}" \
    --snapshot "${RUN_DIR}/snapshots/101000_snapshot.pkl" \
    --output "${BASE}/sibling_exp100k_seed${SEED}.json" \
    --gpu-id "${GPU}" \
    --eval-seeds 700,701,702,703,704,705,706,707,708,709,710,711 \
    --anchor-steps 30,75,120 \
    --intervention-mode sibling_horizon \
    --intervention-horizon 4 \
    --force-level 0 \
    --dimension-selection round_robin \
    --bootstrap-replicates 10000 \
    > "${BASE}/sibling_exp100k_seed${SEED}.log" 2>&1
  echo "[stage158] probe seed${SEED} done"
}

train_seed 1 "${GPU_A}" &
PID_A=$!
train_seed 2 "${GPU_B}" &
PID_B=$!
wait "${PID_A}" "${PID_B}"
echo "[stage158] all arms complete"
