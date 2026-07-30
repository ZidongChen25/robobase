#!/usr/bin/env bash
# Standalone official+mask seed1 full chain on GPU3.
set -euo pipefail
cd "$(dirname "$0")/.."

STAMP="$(date +%Y%m%d%H%M%S)"
O_BASE="exp_local/cqn_stage161_official_mask"
mkdir -p "${O_BASE}/sealed50"
GPU=3
RUN_DIR="${O_BASE}/move_plate_offmask_seed1_gpu${GPU}_${STAMP}"

echo "[161c] train offmask seed1 on GPU${GPU}"
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
  .venv/bin/python train.py \
  launch=cqn_as_pixel_bigym_stage161_official_mask \
  env=bigym/move_plate \
  seed=1 \
  save_csv=true \
  wandb.name="cqn_as_stage161_offmask_seed1_move_plate" \
  hydra.run.dir="${RUN_DIR}" \
  > "${RUN_DIR}.launch.log" 2>&1
echo "[161c] done train offmask seed1"

MUJOCO_GL=egl .venv/bin/python scripts/analyze_cqn_branch_counterfactual.py \
  --run-dir "${RUN_DIR}" \
  --snapshot "${RUN_DIR}/snapshots/101000_snapshot.pkl" \
  --output "${O_BASE}/sibling_offmask_seed1.json" \
  --gpu-id "${GPU}" \
  --eval-seeds 700,701,702,703,704,705,706,707,708,709,710,711 \
  --anchor-steps 30,75,120 \
  --intervention-mode sibling_horizon \
  --intervention-horizon 4 \
  --force-level 0 \
  --dimension-selection round_robin \
  --bootstrap-replicates 10000 \
  > "${O_BASE}/sibling_offmask_seed1.log" 2>&1
echo "[161c] probe offmask seed1 done"

MUJOCO_GL=egl .venv/bin/python scripts/eval_cqn_as_bigym_checkpoint.py \
  --run-dir "${RUN_DIR}" \
  --snapshot "${RUN_DIR}/snapshots/101000_snapshot.pkl" \
  --gpu-id "${GPU}" \
  --num-eval-episodes 50 \
  --eval-seed-start 800 \
  --output "${O_BASE}/sealed50/offmask_s1_final.json" \
  > "${O_BASE}/sealed50/offmask_s1_final.log" 2>&1
echo "[161c] eval offmask seed1 done"
echo "[161c] all complete"
