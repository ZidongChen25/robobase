#!/usr/bin/env bash
# 50-episode (seeds 400-449, 25 vector envs) sweep over every saved
# snapshot of the 100k-budget runs, on two GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_A="${1:-0}"
GPU_B="${2:-1}"
F="exp_local/cqn_stage159_factorial"
M="exp_local/cqn_stage160_lowdim_mask"
S="exp_local/cqn_stage158_explore_100k"
P="exp_local/pixel_cqn_as"

sweep50 () {
  local RUN_DIR="$1" GPU="$2"
  XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${RUN_DIR}" \
    --gpu-id "${GPU}" \
    --num-eval-episodes 50 \
    --eval-seed-start 400 \
    --num-eval-envs 25 \
    --csv-name sweep_eval50.csv \
    > "${RUN_DIR}/sweep_eval50.log" 2>&1
  echo "[sweep50] done: $(basename ${RUN_DIR})"
}

worker_a () {
  sweep50 "$(ls -d ${F}/move_plate_exponly_seed1_*/ | head -1)" "${GPU_A}"
  sweep50 "$(ls -d ${F}/move_plate_decayonly_seed1_*/ | head -1)" "${GPU_A}"
  sweep50 "$(ls -d ${M}/move_plate_ldmask_seed1_*/ | head -1)" "${GPU_A}"
}
worker_b () {
  sweep50 "${S}/move_plate_exp100k_seed1_gpu1_20260727083806" "${GPU_B}"
  sweep50 "${S}/move_plate_exp100k_seed2_gpu5_20260727083806" "${GPU_B}"
  for SEED in 1 2 3 4; do
    sweep50 "${P}/move_plate_paper_seed${SEED}_100k_nw0_20260721" "${GPU_B}"
  done
}

worker_a &
PA=$!
worker_b &
PB=$!
wait "${PA}" "${PB}"
echo "[sweep50] all complete"
