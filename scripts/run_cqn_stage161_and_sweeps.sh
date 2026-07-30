#!/usr/bin/env bash
# After the three seed-1 chains finish: launch official+mask (stage161)
# seed1 on GPU0, and snapshot 25-ep sweeps (seeds 400) for the seed-1 arms
# plus the 158 references on GPUs 1/3.
set -euo pipefail
cd "$(dirname "$0")/.."

STAMP="$(date +%Y%m%d%H%M%S)"
F_BASE="exp_local/cqn_stage159_factorial"
M_BASE="exp_local/cqn_stage160_lowdim_mask"
O_BASE="exp_local/cqn_stage161_official_mask"
mkdir -p "${O_BASE}/sealed50"

echo "[161] waiting for seed1 chains"
until grep -q "eval exponly seed1 done" "${F_BASE}/stage159_master.log" \
  && grep -q "eval decayonly seed1 done" "${F_BASE}/stage159_master.log" \
  && grep -q "all complete" "${M_BASE}/stage160_master.log"; do
  sleep 120
done
echo "[161] seed1 chains finished, starting"

sweep () {
  local RUN_DIR="$1" GPU="$2"
  MUJOCO_GL=egl .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${RUN_DIR}" \
    --gpu-id "${GPU}" \
    --num-eval-episodes 25 \
    --eval-seed-start 400 \
    > "${RUN_DIR}/sweep_eval.log" 2>&1
  echo "[161] sweep done: $(basename ${RUN_DIR})"
}

official_mask () {
  local SEED=1 GPU="$1"
  local RUN_DIR="${O_BASE}/move_plate_offmask_seed${SEED}_gpu${GPU}_${STAMP}"
  echo "[161] train offmask seed${SEED} on GPU${GPU}"
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
    .venv/bin/python train.py \
    launch=cqn_as_pixel_bigym_stage161_official_mask \
    env=bigym/move_plate \
    seed="${SEED}" \
    save_csv=true \
    wandb.name="cqn_as_stage161_offmask_seed${SEED}_move_plate" \
    hydra.run.dir="${RUN_DIR}" \
    > "${RUN_DIR}.launch.log" 2>&1
  echo "[161] done train offmask seed${SEED}"
  MUJOCO_GL=egl .venv/bin/python scripts/analyze_cqn_branch_counterfactual.py \
    --run-dir "${RUN_DIR}" \
    --snapshot "${RUN_DIR}/snapshots/101000_snapshot.pkl" \
    --output "${O_BASE}/sibling_offmask_seed${SEED}.json" \
    --gpu-id "${GPU}" \
    --eval-seeds 700,701,702,703,704,705,706,707,708,709,710,711 \
    --anchor-steps 30,75,120 \
    --intervention-mode sibling_horizon \
    --intervention-horizon 4 \
    --force-level 0 \
    --dimension-selection round_robin \
    --bootstrap-replicates 10000 \
    > "${O_BASE}/sibling_offmask_seed${SEED}.log" 2>&1
  echo "[161] probe offmask seed${SEED} done"
  MUJOCO_GL=egl .venv/bin/python scripts/eval_cqn_as_bigym_checkpoint.py \
    --run-dir "${RUN_DIR}" \
    --snapshot "${RUN_DIR}/snapshots/101000_snapshot.pkl" \
    --gpu-id "${GPU}" \
    --num-eval-episodes 50 \
    --eval-seed-start 800 \
    --output "${O_BASE}/sealed50/offmask_s${SEED}_final.json" \
    > "${O_BASE}/sealed50/offmask_s${SEED}_final.log" 2>&1
  echo "[161] eval offmask seed${SEED} done"
}

worker_a () { official_mask 0; }
worker_b () {
  sweep "$(ls -d ${F_BASE}/move_plate_exponly_seed1_*/ | head -1)" 1
  sweep "$(ls -d ${M_BASE}/move_plate_ldmask_seed1_*/ | head -1)" 1
  sweep "exp_local/cqn_stage158_explore_100k/move_plate_exp100k_seed1_gpu1_20260727083806" 1
}
worker_c () {
  sweep "$(ls -d ${F_BASE}/move_plate_decayonly_seed1_*/ | head -1)" 3
  sweep "exp_local/cqn_stage158_explore_100k/move_plate_exp100k_seed2_gpu5_20260727083806" 3
}

worker_a &
PA=$!
worker_b &
PB=$!
worker_c &
PC=$!
wait "${PA}" "${PB}" "${PC}"
echo "[161] all complete"
