#!/usr/bin/env bash
# Stage-162 arm N (nonoise) directly on the recovered GPU1 (no wait, no
# watcher — curves backfilled by sweep later). Full chain.
set -euo pipefail
cd "$(dirname "$0")/.."

STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_stage162_eps_ablation"
RUN_DIR="${BASE}/move_plate_nonoise_seed1_gpu1_${STAMP}"

echo "[162n] train nonoise on GPU1"
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 \
  .venv/bin/python train_fast.py \
  launch=cqn_as_pixel_bigym_stage162_nonoise \
  env=bigym/move_plate \
  seed=1 \
  save_csv=true \
  wandb.use=false \
  hydra.run.dir="${RUN_DIR}" \
  > "${RUN_DIR}.launch.log" 2>&1
echo "[162n] done train nonoise"

MUJOCO_GL=egl .venv/bin/python scripts/analyze_cqn_branch_counterfactual.py \
  --run-dir "${RUN_DIR}" --snapshot "${RUN_DIR}/snapshots/101000_snapshot.pkl" \
  --output "${BASE}/sibling_nonoise_seed1.json" --gpu-id 1 \
  --eval-seeds 700,701,702,703,704,705,706,707,708,709,710,711 \
  --anchor-steps 30,75,120 --intervention-mode sibling_horizon \
  --intervention-horizon 4 --force-level 0 --dimension-selection round_robin \
  --bootstrap-replicates 10000 > "${BASE}/sibling_nonoise_seed1.log" 2>&1
echo "[162n] probe nonoise done"

XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
  .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
  --run-dir "${RUN_DIR}" --gpu-id 1 --num-eval-episodes 200 \
  --eval-seed-start 800 --num-eval-envs 25 --csv-name ep200_seeds800.csv \
  --skip-steps "$(ls ${RUN_DIR}/snapshots/ | grep -oE '^[0-9]+' | sort -n | head -n -1 | tr '\n' ',' | sed 's/,$//')" \
  > "${RUN_DIR}/ep200.log" 2>&1
echo "[162n] eval200 nonoise done"
echo "[162n] all complete"
