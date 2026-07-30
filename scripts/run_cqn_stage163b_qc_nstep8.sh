#!/usr/bin/env bash
# Stage-163b: QC-style nstep=8 + replan-8, queued behind the 163 chain on
# GPU5; watcher on GPU4.
set -euo pipefail
cd "$(dirname "$0")/.."

STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_stage163_replan8"

echo "[163b] waiting for stage163 chain"
until grep -q "\[163\] all complete" "${BASE}/stage163_master.log" 2>/dev/null; do
  sleep 180
done
RUN_DIR="${BASE}/move_plate_qc8_seed1_gpu5_${STAMP}"
echo "[163b] train qc8 on GPU5"
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=5 MUJOCO_EGL_DEVICE_ID=5 \
  .venv/bin/python train_fast.py \
  launch=cqn_as_pixel_bigym_stage163b_qc_nstep8 \
  env=bigym/move_plate \
  seed=1 \
  save_csv=true \
  wandb.use=false \
  hydra.run.dir="${RUN_DIR}" \
  > "${RUN_DIR}.launch.log" 2>&1 &
TRAIN_PID=$!
sleep 90
XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl nohup \
  .venv/bin/python scripts/async_eval_watcher.py \
  --run-dir "${RUN_DIR}" --gpu-id 4 --num-episodes 50 --eval-seed-start 400 \
  > "${RUN_DIR}.watcher.log" 2>&1 &
wait "${TRAIN_PID}"
echo "[163b] done train qc8"

MUJOCO_GL=egl .venv/bin/python scripts/analyze_cqn_branch_counterfactual.py \
  --run-dir "${RUN_DIR}" --snapshot "${RUN_DIR}/snapshots/101000_snapshot.pkl" \
  --output "${BASE}/sibling_qc8_seed1.json" --gpu-id 5 \
  --eval-seeds 700,701,702,703,704,705,706,707,708,709,710,711 \
  --anchor-steps 30,75,120 --intervention-mode sibling_horizon \
  --intervention-horizon 4 --force-level 0 --dimension-selection round_robin \
  --bootstrap-replicates 10000 > "${BASE}/sibling_qc8_seed1.log" 2>&1
echo "[163b] probe done"

XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
  .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
  --run-dir "${RUN_DIR}" --gpu-id 5 --num-eval-episodes 200 \
  --eval-seed-start 800 --num-eval-envs 25 --csv-name ep200_seeds800.csv \
  --replan-interval 8 \
  --skip-steps "$(ls ${RUN_DIR}/snapshots/ | grep -oE '^[0-9]+' | sort -n | head -n -1 | tr '\n' ',' | sed 's/,$//')" \
  > "${RUN_DIR}/ep200.log" 2>&1
echo "[163b] eval200 done"
echo "[163b] all complete"
