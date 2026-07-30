#!/usr/bin/env bash
# Stage-160: low-dim mask variant, 1 seed to 100k, then sibling probe and
# 50-ep@800 eval of the final snapshot.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-1}"
SEED="${2:-1}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_stage160_lowdim_mask"
OUT="${BASE}/sealed50"
mkdir -p "${OUT}"
cp "$0" "${BASE}/stage160_controller.${STAMP}.sh"

RUN_DIR="${BASE}/move_plate_ldmask_seed${SEED}_gpu${GPU}_${STAMP}"
echo "[stage160] train ldmask seed${SEED} on GPU${GPU}"
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
  .venv/bin/python train.py \
  launch=cqn_as_pixel_bigym_stage160_lowdim_mask \
  env=bigym/move_plate \
  seed="${SEED}" \
  save_csv=true \
  wandb.name="cqn_as_stage160_ldmask_seed${SEED}_move_plate" \
  hydra.run.dir="${RUN_DIR}" \
  > "${RUN_DIR}.launch.log" 2>&1
echo "[stage160] done train seed${SEED}"

MUJOCO_GL=egl .venv/bin/python scripts/analyze_cqn_branch_counterfactual.py \
  --run-dir "${RUN_DIR}" \
  --snapshot "${RUN_DIR}/snapshots/101000_snapshot.pkl" \
  --output "${BASE}/sibling_ldmask_seed${SEED}.json" \
  --gpu-id "${GPU}" \
  --eval-seeds 700,701,702,703,704,705,706,707,708,709,710,711 \
  --anchor-steps 30,75,120 \
  --intervention-mode sibling_horizon \
  --intervention-horizon 4 \
  --force-level 0 \
  --dimension-selection round_robin \
  --bootstrap-replicates 10000 \
  > "${BASE}/sibling_ldmask_seed${SEED}.log" 2>&1
echo "[stage160] probe seed${SEED} done"

MUJOCO_GL=egl .venv/bin/python scripts/eval_cqn_as_bigym_checkpoint.py \
  --run-dir "${RUN_DIR}" \
  --snapshot "${RUN_DIR}/snapshots/101000_snapshot.pkl" \
  --gpu-id "${GPU}" \
  --num-eval-episodes 50 \
  --eval-seed-start 800 \
  --output "${OUT}/ldmask_s${SEED}_final.json" \
  > "${OUT}/ldmask_s${SEED}_final.log" 2>&1
echo "[stage160] eval seed${SEED} done"
echo "[stage160] all complete"
