#!/usr/bin/env bash
# Resume the crashed H arm (double) from its latest snapshot in the SAME
# run dir, then run the probe + 200-ep chain its original worker skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_DIR="exp_local/cqn_stage162_eps_ablation/move_plate_double_seed1_gpu1_20260728141948"
BASE="exp_local/cqn_stage162_eps_ablation"

echo "[162r] resume double on GPU4 from $(ls ${RUN_DIR}/snapshots | grep latest)"
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=4 MUJOCO_EGL_DEVICE_ID=4 \
  .venv/bin/python train_fast.py \
  launch=cqn_as_pixel_bigym_stage162_double \
  env=bigym/move_plate \
  seed=1 \
  save_csv=true \
  wandb.use=false \
  hydra.run.dir="${RUN_DIR}" \
  >> "${RUN_DIR}.launch.log" 2>&1
echo "[162r] done train double"

MUJOCO_GL=egl .venv/bin/python scripts/analyze_cqn_branch_counterfactual.py \
  --run-dir "${RUN_DIR}" --snapshot "${RUN_DIR}/snapshots/101000_snapshot.pkl" \
  --output "${BASE}/sibling_double_seed1.json" --gpu-id 4 \
  --eval-seeds 700,701,702,703,704,705,706,707,708,709,710,711 \
  --anchor-steps 30,75,120 --intervention-mode sibling_horizon \
  --intervention-horizon 4 --force-level 0 --dimension-selection round_robin \
  --bootstrap-replicates 10000 > "${BASE}/sibling_double_seed1.log" 2>&1
echo "[162r] probe double done"

XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
  .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
  --run-dir "${RUN_DIR}" --gpu-id 4 --num-eval-episodes 200 \
  --eval-seed-start 800 --num-eval-envs 25 --csv-name ep200_seeds800.csv \
  --skip-steps "$(ls ${RUN_DIR}/snapshots/ | grep -oE '^[0-9]+' | sort -n | head -n -1 | tr '\n' ',' | sed 's/,$//')" \
  > "${RUN_DIR}/ep200.log" 2>&1
echo "[162r] eval200 double done"
echo "[162r] all complete"
