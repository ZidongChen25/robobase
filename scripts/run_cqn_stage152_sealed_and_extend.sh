#!/usr/bin/env bash
# Stage-152 phase 2 (cqn-flow.md 34.1):
#  (1) sealed 50-episode evals at fresh eval-seed-start 800, dual protocol
#      (primary = rule-resolved nearest-to-validation-best, secondary =
#      final snapshot), for ccff_full seeds 1-3 AND matched clean seeds 1-3
#      (re-sealed on 800 because seed-set effects are proven ~0.12).
#  (2) demo-10 inconclusive-resolution extension: seeds 4-6 for both
#      vanilla_d10 and ccff_d10 arms.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_A="${1:-1}"
GPU_B="${2:-5}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_stage152_coarse_flow"
OUT="${BASE}/sealed"
C149=exp_local/cqn_stage149_confirmation
mkdir -p "${OUT}"
cp "$0" "${BASE}/stage152b_controller.${STAMP}.sh"

declare -A RUNS=(
  [clean_s1]="${C149}/move_plate_clean_seed1_gpu1_20260726063318"
  [clean_s2]="${C149}/move_plate_clean_seed2_gpu1_20260726062149"
  [clean_s3]="${C149}/move_plate_clean_seed3_gpu5_20260726062149"
  [ccff_full_s1]="$(ls -d ${BASE}/move_plate_ccff_full_seed1_*/ | head -1)"
  [ccff_full_s2]="$(ls -d ${BASE}/move_plate_ccff_full_seed2_*/ | head -1)"
  [ccff_full_s3]="$(ls -d ${BASE}/move_plate_ccff_full_seed3_*/ | head -1)"
)

eval_one () {
  local LABEL="$1" RUN="$2" SNAP="$3" PROTO="$4" GPU="$5"
  local OUT_JSON="${OUT}/${LABEL}_${PROTO}.json"
  [ -f "${OUT_JSON}" ] && { echo "[152-sealed] skip ${LABEL} ${PROTO} (exists)"; return; }
  MUJOCO_GL=egl .venv/bin/python scripts/eval_cqn_as_bigym_checkpoint.py \
    --run-dir "${RUN}" \
    --snapshot "${SNAP}" \
    --gpu-id "${GPU}" \
    --num-eval-episodes 50 \
    --eval-seed-start 800 \
    --output "${OUT_JSON}" \
    > "${OUT}/${LABEL}_${PROTO}.log" 2>&1
  echo "[152-sealed] ${LABEL} ${PROTO} done"
}

sealed_worker () {
  local GPU="$1"; shift
  for LABEL in "$@"; do
    local RUN="${RUNS[$LABEL]}"
    read -r PRIMARY FINAL < <(.venv/bin/python scripts/resolve_cqn_validation_best_snapshot.py "${RUN}")
    eval_one "${LABEL}" "${RUN}" "${PRIMARY}" primary "${GPU}"
    if [ "${PRIMARY}" != "${FINAL}" ]; then
      eval_one "${LABEL}" "${RUN}" "${FINAL}" final "${GPU}"
    else
      cp "${OUT}/${LABEL}_primary.json" "${OUT}/${LABEL}_final.json" 2>/dev/null || true
      echo "[152-sealed] ${LABEL} final == primary"
    fi
  done
}

run_arm () {
  local LAUNCH="$1" ARM="$2" SEED="$3" GPU="$4"
  local RUN_DIR="${BASE}/move_plate_${ARM}_seed${SEED}_gpu${GPU}_${STAMP}"
  echo "[152-ext] ${ARM} seed${SEED} on GPU${GPU}"
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
    .venv/bin/python train.py \
    launch="${LAUNCH}" \
    env=bigym/move_plate \
    seed="${SEED}" \
    save_csv=true \
    demos=10 \
    env.expected_successful_demos=null \
    wandb.name="cqn_as_stage152_${ARM}_seed${SEED}_move_plate" \
    hydra.run.dir="${RUN_DIR}" \
    > "${RUN_DIR}.launch.log" 2>&1
  echo "[152-ext] done ${ARM} seed${SEED}"
}

worker_a () {
  sealed_worker "${GPU_A}" ccff_full_s1 ccff_full_s2 clean_s1
  for SEED in 4 5 6; do
    run_arm cqn_as_pixel_bigym_value_fidelity_gate vanilla_d10 "${SEED}" "${GPU_A}"
  done
}

worker_b () {
  sealed_worker "${GPU_B}" ccff_full_s3 clean_s2 clean_s3
  for SEED in 4 5 6; do
    run_arm cqn_as_pixel_bigym_stage152_coarse_flow_gate ccff_d10 "${SEED}" "${GPU_B}"
  done
}

worker_a &
PID_A=$!
worker_b &
PID_B=$!
wait "${PID_A}" "${PID_B}"
echo "[152-sealed] sealed+extension complete"
for J in "${OUT}"/*_primary.json "${OUT}"/*_final.json; do
  [ -f "$J" ] || continue
  echo -n "$(basename "$J" .json): "
  .venv/bin/python -c "import json,sys; d=json.load(open('$J')); print(d.get('success_percent'))"
done
