#!/usr/bin/env bash
# Stage-166 v2 (cqn-flow.md 53): BC weaning from the combined recipe.
# Resume combined seed1 @101k in fresh run dirs (snapshot + replay
# hardlinked from stage158), extend to 201k, BOTH ARMS ON ONE CARD
# (xla_mem_fraction=0.45, starts staggered 120s):
#   wean0:   bc_lambda step_linear(1.0,0.25,100000,0.0,50000)
#            -> lambda hits 0 at 150k, then a 50k pure-TD plateau
#   hold025: original schedule (clamps at 0.25) — control
# Then sealed 200-ep@800 on the final snapshot (official ensemble exec).
set -uo pipefail
cd "$(dirname "$0")/.."

STAMP="$(date +%Y%m%d%H%M%S)"
SRC="exp_local/cqn_stage158_explore_100k/move_plate_exp100k_seed1_gpu1_20260727083806"
C="exp_local/cqn_stage166_bc_wean"
mkdir -p "${C}"

ASYNC="eval_every_steps=1000000 num_eval_episodes=0 num_eval_envs=0 log_eval_video=false save_snapshot=true snapshot_every_n=5000"

setup_dir () {
  local RUN_DIR="$1"
  mkdir -p "${RUN_DIR}/snapshots" "${RUN_DIR}/replay" "${RUN_DIR}/demo_replay"
  cp -l "${SRC}/snapshots/101000_snapshot.pkl" "${RUN_DIR}/snapshots/"
  ln -sf 101000_snapshot.pkl "${RUN_DIR}/snapshots/latest_snapshot.pkl"
  ln "${SRC}"/replay/*.npz "${RUN_DIR}/replay/"
  ln "${SRC}"/demo_replay/*.npz "${RUN_DIR}/demo_replay/"
}

GPU="${1:-0}"

run_arm () {
  local ARM="$1" EXTRA="$2"
  local RUN_DIR="${C}/move_plate_${ARM}_seed1_gpu${GPU}_${STAMP}"
  setup_dir "${RUN_DIR}"
  echo "[166] resume ${ARM} on GPU${GPU} ($(date +%H:%M:%S))"
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
    .venv/bin/python train_fast.py \
    launch=cqn_as_pixel_bigym_stage158_explore_100k \
    env=bigym/move_plate \
    seed=1 \
    num_train_frames=201000 \
    xla_mem_fraction=0.45 \
    ${ASYNC} \
    save_csv=true \
    wandb.use=false \
    ${EXTRA} \
    hydra.run.dir="${RUN_DIR}" \
    > "${RUN_DIR}.launch.log" 2>&1
  echo "[166] done train ${ARM} ($(date +%H:%M:%S))"
  XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${RUN_DIR}" --gpu-id "${GPU}" --num-eval-episodes 200 \
    --eval-seed-start 800 --num-eval-envs 25 --csv-name ep200_seeds800.csv \
    --skip-steps "$(ls ${RUN_DIR}/snapshots/ | grep -oE '^[0-9]+' | sort -n | head -n -1 | tr '\n' ',' | sed 's/,$//')" \
    > "${RUN_DIR}/ep200.log" 2>&1
  echo "[166] eval ${ARM} done: $(tail -1 ${RUN_DIR}/ep200_seeds800.csv 2>/dev/null)"
}

run_arm wean0 "method.bc_lambda_schedule='step_linear(1.0,0.25,100000,0.0,50000)'" &
P1=$!
sleep 120
run_arm hold025 "" &
P2=$!
wait "${P1}" "${P2}"
echo "[166] all complete"
