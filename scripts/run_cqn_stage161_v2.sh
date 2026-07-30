#!/usr/bin/env bash
# v2: vectorized sweeps start immediately on GPU1; official+mask waits
# only for the exponly seed1 chain to free GPU0.
set -euo pipefail
cd "$(dirname "$0")/.."

STAMP="$(date +%Y%m%d%H%M%S)"
F_BASE="exp_local/cqn_stage159_factorial"
M_BASE="exp_local/cqn_stage160_lowdim_mask"
O_BASE="exp_local/cqn_stage161_official_mask"
mkdir -p "${O_BASE}/sealed50"

sweep () {
  local RUN_DIR="$1" GPU="$2"
  XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${RUN_DIR}" \
    --gpu-id "${GPU}" \
    --num-eval-episodes 25 \
    --eval-seed-start 400 \
    --num-eval-envs 25 \
    > "${RUN_DIR}/sweep_eval.log" 2>&1
  echo "[161] sweep done: $(basename ${RUN_DIR})"
}

worker_sweeps () {
  sweep "$(ls -d ${M_BASE}/move_plate_ldmask_seed1_*/ | head -1)" 1
  sweep "$(ls -d ${F_BASE}/move_plate_exponly_seed1_*/ | head -1)" 1
  sweep "$(ls -d ${F_BASE}/move_plate_decayonly_seed1_*/ | head -1)" 1
  sweep "exp_local/cqn_stage158_explore_100k/move_plate_exp100k_seed1_gpu1_20260727083806" 1
  sweep "exp_local/cqn_stage158_explore_100k/move_plate_exp100k_seed2_gpu5_20260727083806" 1
  echo "[161] all sweeps complete"
}

worker_offmask () {
  echo "[161] waiting for exponly chain to free GPU0"
  until grep -q "eval exponly seed1 done" "${F_BASE}/stage159_master.log" 2>/dev/null; do
    sleep 120
  done
  local RUN_DIR="${O_BASE}/move_plate_offmask_seed1_gpu0_${STAMP}"
  echo "[161] train offmask seed1 on GPU0"
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
    .venv/bin/python train.py \
    launch=cqn_as_pixel_bigym_stage161_official_mask \
    env=bigym/move_plate \
    seed=1 \
    save_csv=true \
    wandb.name="cqn_as_stage161_offmask_seed1_move_plate" \
    hydra.run.dir="${RUN_DIR}" \
    > "${RUN_DIR}.launch.log" 2>&1
  echo "[161] done train offmask seed1"
  MUJOCO_GL=egl .venv/bin/python scripts/analyze_cqn_branch_counterfactual.py \
    --run-dir "${RUN_DIR}" \
    --snapshot "${RUN_DIR}/snapshots/101000_snapshot.pkl" \
    --output "${O_BASE}/sibling_offmask_seed1.json" \
    --gpu-id 0 \
    --eval-seeds 700,701,702,703,704,705,706,707,708,709,710,711 \
    --anchor-steps 30,75,120 \
    --intervention-mode sibling_horizon \
    --intervention-horizon 4 \
    --force-level 0 \
    --dimension-selection round_robin \
    --bootstrap-replicates 10000 \
    > "${O_BASE}/sibling_offmask_seed1.log" 2>&1
  echo "[161] probe offmask seed1 done"
  MUJOCO_GL=egl .venv/bin/python scripts/eval_cqn_as_bigym_checkpoint.py \
    --run-dir "${RUN_DIR}" \
    --snapshot "${RUN_DIR}/snapshots/101000_snapshot.pkl" \
    --gpu-id 0 \
    --num-eval-episodes 50 \
    --eval-seed-start 800 \
    --output "${O_BASE}/sealed50/offmask_s1_final.json" \
    > "${O_BASE}/sealed50/offmask_s1_final.log" 2>&1
  echo "[161] eval offmask seed1 done"
}

worker_sweeps &
PA=$!
worker_offmask &
PB=$!
wait "${PA}" "${PB}"
echo "[161] all complete"
