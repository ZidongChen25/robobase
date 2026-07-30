#!/usr/bin/env bash
# Stage-158b: complete the seed-matched comparison (cqn-flow.md 39.4).
# Train exp100k seeds 3,4 (same seed values as official paper runs), probe
# each, then run 50-ep@800 evals of the official 100k checkpoints (4 seeds)
# and of the new seeds' final snapshots for the matched table.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_A="${1:-1}"
GPU_B="${2:-5}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_stage158_explore_100k"
OUT="${BASE}/sealed50"
mkdir -p "${OUT}"
cp "$0" "${BASE}/stage158b_controller.${STAMP}.sh"

probe () {
  local RUN_DIR="$1" TAG="$2" GPU="$3"
  MUJOCO_GL=egl .venv/bin/python scripts/analyze_cqn_branch_counterfactual.py \
    --run-dir "${RUN_DIR}" \
    --snapshot "${RUN_DIR}/snapshots/101000_snapshot.pkl" \
    --output "${BASE}/sibling_exp100k_${TAG}.json" \
    --gpu-id "${GPU}" \
    --eval-seeds 700,701,702,703,704,705,706,707,708,709,710,711 \
    --anchor-steps 30,75,120 \
    --intervention-mode sibling_horizon \
    --intervention-horizon 4 \
    --force-level 0 \
    --dimension-selection round_robin \
    --bootstrap-replicates 10000 \
    > "${BASE}/sibling_exp100k_${TAG}.log" 2>&1
  echo "[158b] probe ${TAG} done"
}

eval50 () {
  local RUN_DIR="$1" SNAP="$2" TAG="$3" GPU="$4"
  [ -f "${OUT}/${TAG}.json" ] && { echo "[158b] skip ${TAG}"; return; }
  MUJOCO_GL=egl .venv/bin/python scripts/eval_cqn_as_bigym_checkpoint.py \
    --run-dir "${RUN_DIR}" \
    --snapshot "${SNAP}" \
    --gpu-id "${GPU}" \
    --num-eval-episodes 50 \
    --eval-seed-start 800 \
    --output "${OUT}/${TAG}.json" \
    > "${OUT}/${TAG}.log" 2>&1
  echo "[158b] eval ${TAG} done"
}

train_and_finish () {
  local SEED="$1" GPU="$2"
  local RUN_DIR="${BASE}/move_plate_exp100k_seed${SEED}_gpu${GPU}_${STAMP}"
  echo "[158b] train seed${SEED} on GPU${GPU}"
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
    .venv/bin/python train.py \
    launch=cqn_as_pixel_bigym_stage158_explore_100k \
    env=bigym/move_plate \
    seed="${SEED}" \
    save_csv=true \
    wandb.name="cqn_as_stage158_exp100k_seed${SEED}_move_plate" \
    hydra.run.dir="${RUN_DIR}" \
    > "${RUN_DIR}.launch.log" 2>&1
  echo "[158b] done train seed${SEED}"
  probe "${RUN_DIR}" "seed${SEED}" "${GPU}"
  eval50 "${RUN_DIR}" "${RUN_DIR}/snapshots/101000_snapshot.pkl" "exp100k_s${SEED}_final" "${GPU}"
}

official_evals () {
  local GPU="$1"; shift
  for SEED in "$@"; do
    local RUN_DIR="exp_local/pixel_cqn_as/move_plate_paper_seed${SEED}_100k_nw0_20260721"
    eval50 "${RUN_DIR}" "${RUN_DIR}/snapshots/101000_snapshot.pkl" "official_s${SEED}_final" "${GPU}"
  done
}

worker_a () { train_and_finish 3 "${GPU_A}"; official_evals "${GPU_A}" 1 2; }
worker_b () { train_and_finish 4 "${GPU_B}"; official_evals "${GPU_B}" 3 4; }

worker_a &
PID_A=$!
worker_b &
PID_B=$!
wait "${PID_A}" "${PID_B}"
echo "[158b] all complete"
for J in "${OUT}"/*_final.json; do
  echo -n "$(basename "$J" .json): "
  .venv/bin/python -c "import json; print(json.load(open('$J')).get('success_percent'))"
done
