#!/usr/bin/env bash
# Seed-2 companions for the three in-flight arms (protocol identical to
# their seed-1 siblings): exponly@GPU2, decayonly@GPU4, ldmask@GPU5.
# Each run: train 100k -> sibling probe -> 50-ep@800 eval.
set -euo pipefail
cd "$(dirname "$0")/.."

STAMP="$(date +%Y%m%d%H%M%S)"
F_BASE="exp_local/cqn_stage159_factorial"
M_BASE="exp_local/cqn_stage160_lowdim_mask"
mkdir -p "${F_BASE}/sealed50" "${M_BASE}/sealed50"
cp "$0" "${F_BASE}/stage159160_seed2_controller.${STAMP}.sh"

run_full () {
  local LAUNCH="$1" ARM="$2" BASE="$3" GPU="$4"
  local SEED=2
  local RUN_DIR="${BASE}/move_plate_${ARM}_seed${SEED}_gpu${GPU}_${STAMP}"
  echo "[seed2] train ${ARM} seed${SEED} on GPU${GPU}"
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
    .venv/bin/python train.py \
    launch="${LAUNCH}" \
    env=bigym/move_plate \
    seed="${SEED}" \
    save_csv=true \
    wandb.name="cqn_as_${ARM}_seed${SEED}_move_plate" \
    hydra.run.dir="${RUN_DIR}" \
    > "${RUN_DIR}.launch.log" 2>&1
  echo "[seed2] done train ${ARM} seed${SEED}"
  MUJOCO_GL=egl .venv/bin/python scripts/analyze_cqn_branch_counterfactual.py \
    --run-dir "${RUN_DIR}" \
    --snapshot "${RUN_DIR}/snapshots/101000_snapshot.pkl" \
    --output "${BASE}/sibling_${ARM}_seed${SEED}.json" \
    --gpu-id "${GPU}" \
    --eval-seeds 700,701,702,703,704,705,706,707,708,709,710,711 \
    --anchor-steps 30,75,120 \
    --intervention-mode sibling_horizon \
    --intervention-horizon 4 \
    --force-level 0 \
    --dimension-selection round_robin \
    --bootstrap-replicates 10000 \
    > "${BASE}/sibling_${ARM}_seed${SEED}.log" 2>&1
  echo "[seed2] probe ${ARM} seed${SEED} done"
  MUJOCO_GL=egl .venv/bin/python scripts/eval_cqn_as_bigym_checkpoint.py \
    --run-dir "${RUN_DIR}" \
    --snapshot "${RUN_DIR}/snapshots/101000_snapshot.pkl" \
    --gpu-id "${GPU}" \
    --num-eval-episodes 50 \
    --eval-seed-start 800 \
    --output "${BASE}/sealed50/${ARM}_s${SEED}_final.json" \
    > "${BASE}/sealed50/${ARM}_s${SEED}_final.log" 2>&1
  echo "[seed2] eval ${ARM} seed${SEED} done"
}

run_full cqn_as_pixel_bigym_stage159_explore_only exponly "${F_BASE}" 2 &
P1=$!
run_full cqn_as_pixel_bigym_stage159_decay_only decayonly "${F_BASE}" 4 &
P2=$!
run_full cqn_as_pixel_bigym_stage160_lowdim_mask ldmask "${M_BASE}" 5 &
P3=$!
wait "${P1}" "${P2}" "${P3}"
echo "[seed2] all complete"
