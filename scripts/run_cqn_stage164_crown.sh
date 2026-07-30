#!/usr/bin/env bash
# Stage-164: crown arm (combined recipe x QC: nstep8 + replan8), seeds 1&2,
# plus seed-2 replications of official+QC and official+mask.
# Each run: train 100k -> sibling probe -> 200-ep@800 final (matching exec
# mode), chained on its own (by-then-free) GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

STAMP="$(date +%Y%m%d%H%M%S)"
R="exp_local/cqn_stage163_replan8"
O="exp_local/cqn_stage161_official_mask"
C="exp_local/cqn_stage164_crown"
mkdir -p "${C}"

full () {
  local LAUNCH="$1" ARM="$2" SEED="$3" GPU="$4" BASE="$5" EVAL_EXTRA="$6"
  local RUN_DIR="${BASE}/move_plate_${ARM}_seed${SEED}_gpu${GPU}_${STAMP}"
  echo "[164] train ${ARM} seed${SEED} on GPU${GPU}"
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
    .venv/bin/python train_fast.py \
    launch="${LAUNCH}" \
    env=bigym/move_plate \
    seed="${SEED}" \
    save_csv=true \
    wandb.use=false \
    hydra.run.dir="${RUN_DIR}" \
    > "${RUN_DIR}.launch.log" 2>&1
  echo "[164] done train ${ARM} seed${SEED}"
  MUJOCO_GL=egl .venv/bin/python scripts/analyze_cqn_branch_counterfactual.py \
    --run-dir "${RUN_DIR}" --snapshot "${RUN_DIR}/snapshots/101000_snapshot.pkl" \
    --output "${BASE}/sibling_${ARM}_seed${SEED}.json" --gpu-id "${GPU}" \
    --eval-seeds 700,701,702,703,704,705,706,707,708,709,710,711 \
    --anchor-steps 30,75,120 --intervention-mode sibling_horizon \
    --intervention-horizon 4 --force-level 0 --dimension-selection round_robin \
    --bootstrap-replicates 10000 > "${BASE}/sibling_${ARM}_seed${SEED}.log" 2>&1
  echo "[164] probe ${ARM} seed${SEED} done"
  XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${RUN_DIR}" --gpu-id "${GPU}" --num-eval-episodes 200 \
    --eval-seed-start 800 --num-eval-envs 25 --csv-name ep200_seeds800.csv \
    ${EVAL_EXTRA} \
    --skip-steps "$(ls ${RUN_DIR}/snapshots/ | grep -oE '^[0-9]+' | sort -n | head -n -1 | tr '\n' ',' | sed 's/,$//')" \
    > "${RUN_DIR}/ep200.log" 2>&1
  echo "[164] eval ${ARM} seed${SEED} done"
}

full cqn_as_pixel_bigym_stage163b_qc_nstep8 crown 1 0 "${C}" "--replan-interval 8" &
P1=$!
full cqn_as_pixel_bigym_stage163b_qc_nstep8 crown 2 2 "${C}" "--replan-interval 8" &
P2=$!
full cqn_as_pixel_bigym_stage163c_official_qc8 offqc8 2 3 "${R}" "--replan-interval 8" &
P3=$!
full cqn_as_pixel_bigym_stage161_official_mask offmask 2 4 "${O}" "" &
P4=$!
wait "${P1}" "${P2}" "${P3}" "${P4}"
echo "[164] all complete"
