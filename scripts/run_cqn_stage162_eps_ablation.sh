#!/usr/bin/env bash
# Stage-162: exploration-probability ablation, async-eval protocol.
# Trainings on GPUs 0/1/2 (seed 1); one async watcher per run on GPU5.
# After each training: sibling probe + 200-ep final eval on its own GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_stage162_eps_ablation"
mkdir -p "${BASE}/sealed"

run_arm () {
  local ARM="$1" GPU="$2"
  local RUN_DIR="${BASE}/move_plate_${ARM}_seed1_gpu${GPU}_${STAMP}"
  echo "[162] train ${ARM} on GPU${GPU}"
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
    .venv/bin/python train_fast.py \
    launch=cqn_as_pixel_bigym_stage162_${ARM} \
    env=bigym/move_plate \
    seed=1 \
    save_csv=true \
    wandb.use=false \
    hydra.run.dir="${RUN_DIR}" \
    > "${RUN_DIR}.launch.log" 2>&1 &
  local TRAIN_PID=$!
  sleep 90
  MUJOCO_GL=egl nohup .venv/bin/python scripts/async_eval_watcher.py \
    --run-dir "${RUN_DIR}" \
    --gpu-id 5 \
    --num-episodes 50 \
    --eval-seed-start 400 \
    > "${RUN_DIR}.watcher.log" 2>&1 &
  wait "${TRAIN_PID}"
  echo "[162] done train ${ARM}"
  MUJOCO_GL=egl .venv/bin/python scripts/analyze_cqn_branch_counterfactual.py \
    --run-dir "${RUN_DIR}" \
    --snapshot "${RUN_DIR}/snapshots/101000_snapshot.pkl" \
    --output "${BASE}/sibling_${ARM}_seed1.json" \
    --gpu-id "${GPU}" \
    --eval-seeds 700,701,702,703,704,705,706,707,708,709,710,711 \
    --anchor-steps 30,75,120 \
    --intervention-mode sibling_horizon \
    --intervention-horizon 4 \
    --force-level 0 \
    --dimension-selection round_robin \
    --bootstrap-replicates 10000 \
    > "${BASE}/sibling_${ARM}_seed1.log" 2>&1
  echo "[162] probe ${ARM} done"
  XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${RUN_DIR}" \
    --gpu-id "${GPU}" \
    --num-eval-episodes 200 \
    --eval-seed-start 800 \
    --num-eval-envs 25 \
    --csv-name ep200_seeds800.csv \
    --skip-steps "$(ls ${RUN_DIR}/snapshots/ | grep -oE '^[0-9]+' | sort -n | head -n -1 | tr '\n' ',' | sed 's/,$//')" \
    > "${RUN_DIR}/ep200.log" 2>&1
  echo "[162] eval200 ${ARM} done"
}

run_arm uniform 0 &
P1=$!
run_arm double 1 &
P2=$!
run_arm edecay 2 &
P3=$!
wait "${P1}" "${P2}" "${P3}"
echo "[162] all complete"
