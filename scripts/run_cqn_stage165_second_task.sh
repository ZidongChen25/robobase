#!/usr/bin/env bash
# Stage-165 (cqn-flow.md 52): second-task external validity on
# dishwasher_close_trays. Arms: combined (explore x decay, stage158) and
# official+QC (nstep8 + replan-8 train&eval, stage163c), seeds 1&2.
# Two-runs-per-card recipe (AGENTS.md): xla_mem_fraction=0.45, starts
# staggered 120s; arm-balanced across cards. Each run auto-chains the
# sealed 200-ep@800 eval (ne=25, final snapshot, matching exec mode).
set -uo pipefail
cd "$(dirname "$0")/.."

STAMP="$(date +%Y%m%d%H%M%S)"
C="exp_local/cqn_stage165_second_task"
TASK="dishwasher_close_trays"
mkdir -p "${C}"

run_one () {
  local LAUNCH="$1" ARM="$2" SEED="$3" GPU="$4" EVAL_EXTRA="$5" EXTRA_OVERRIDES="$6"
  local RUN_DIR="${C}/${TASK}_${ARM}_seed${SEED}_gpu${GPU}_${STAMP}"
  echo "[165] train ${ARM} seed${SEED} on GPU${GPU} ($(date +%H:%M:%S))"
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
    .venv/bin/python train_fast.py \
    launch="${LAUNCH}" \
    env=bigym/${TASK} \
    seed="${SEED}" \
    xla_mem_fraction=0.45 \
    save_csv=true \
    wandb.use=false \
    ${EXTRA_OVERRIDES} \
    hydra.run.dir="${RUN_DIR}" \
    > "${RUN_DIR}.launch.log" 2>&1
  echo "[165] done train ${ARM} seed${SEED} ($(date +%H:%M:%S))"
  XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${RUN_DIR}" --gpu-id "${GPU}" --num-eval-episodes 200 \
    --eval-seed-start 800 --num-eval-envs 25 --csv-name ep200_seeds800.csv \
    ${EVAL_EXTRA} \
    --skip-steps "$(ls ${RUN_DIR}/snapshots/ | grep -oE '^[0-9]+' | sort -n | head -n -1 | tr '\n' ',' | sed 's/,$//')" \
    > "${RUN_DIR}/ep200.log" 2>&1
  echo "[165] eval ${ARM} seed${SEED} done ($(date +%H:%M:%S)): $(tail -1 ${RUN_DIR}/ep200_seeds800.csv 2>/dev/null)"
}

# stage158 lacks the async-protocol overrides; stage163c has them baked in.
ASYNC="eval_every_steps=1000000 num_eval_episodes=0 num_eval_envs=0 log_eval_video=false save_snapshot=true snapshot_every_n=5000"

card () {
  local GPU="$1" SEED="$2"
  run_one cqn_as_pixel_bigym_stage158_explore_100k combined "${SEED}" "${GPU}" "" "${ASYNC}" &
  local P1=$!
  sleep 120
  run_one cqn_as_pixel_bigym_stage163c_official_qc8 offqc8 "${SEED}" "${GPU}" "--replan-interval 8" "" &
  local P2=$!
  wait "${P1}" "${P2}"
}

card 2 1 &
CARD1=$!
card 5 2 &
CARD2=$!
wait "${CARD1}" "${CARD2}"
echo "[165] all complete ($(date +%H:%M:%S))"
