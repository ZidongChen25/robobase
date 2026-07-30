#!/usr/bin/env bash
# Stage-159 (all-GPU variant): 2x2 factorial arms in full parallel.
# GPU0: exponly s1, GPU2: exponly s2, GPU3: decayonly s1, GPU4: decayonly s2.
set -euo pipefail
cd "$(dirname "$0")/.."

STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_stage159_factorial"
OUT="${BASE}/sealed50"
mkdir -p "${OUT}"
cp "$0" "${BASE}/stage159_allgpu_controller.${STAMP}.sh"

run_full () {
  local LAUNCH="$1" ARM="$2" SEED="$3" GPU="$4"
  local RUN_DIR="${BASE}/move_plate_${ARM}_seed${SEED}_gpu${GPU}_${STAMP}"
  echo "[stage159] train ${ARM} seed${SEED} on GPU${GPU}"
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
    .venv/bin/python train.py \
    launch="${LAUNCH}" \
    env=bigym/move_plate \
    seed="${SEED}" \
    save_csv=true \
    wandb.name="cqn_as_stage159_${ARM}_seed${SEED}_move_plate" \
    hydra.run.dir="${RUN_DIR}" \
    > "${RUN_DIR}.launch.log" 2>&1
  echo "[stage159] done train ${ARM} seed${SEED}"
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
  echo "[stage159] probe ${ARM} seed${SEED} done"
  MUJOCO_GL=egl .venv/bin/python scripts/eval_cqn_as_bigym_checkpoint.py \
    --run-dir "${RUN_DIR}" \
    --snapshot "${RUN_DIR}/snapshots/101000_snapshot.pkl" \
    --gpu-id "${GPU}" \
    --num-eval-episodes 50 \
    --eval-seed-start 800 \
    --output "${OUT}/${ARM}_s${SEED}_final.json" \
    > "${OUT}/${ARM}_s${SEED}_final.log" 2>&1
  echo "[stage159] eval ${ARM} seed${SEED} done"
}

run_full cqn_as_pixel_bigym_stage159_explore_only exponly 1 0 &
P1=$!
run_full cqn_as_pixel_bigym_stage159_explore_only exponly 2 2 &
P2=$!
run_full cqn_as_pixel_bigym_stage159_decay_only decayonly 1 3 &
P3=$!
run_full cqn_as_pixel_bigym_stage159_decay_only decayonly 2 4 &
P4=$!
wait "${P1}" "${P2}" "${P3}" "${P4}"
echo "[stage159] all complete"
