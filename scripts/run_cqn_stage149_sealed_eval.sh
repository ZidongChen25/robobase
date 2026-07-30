#!/usr/bin/env bash
# Stage-149 phase 2: sealed 50-episode evaluations, eval-seed-start 600.
# For every arm: primary = rule-resolved nearest-to-validation-best
# snapshot; secondary = final snapshot (selection-free).
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_A="${1:-1}"
GPU_B="${2:-5}"
CLEAN1="${3:?usage: ... GPU_A GPU_B CLEAN_SEED1_RUN_DIR}"
OUT="exp_local/cqn_stage149_confirmation/sealed"
mkdir -p "${OUT}"

C149=exp_local/cqn_stage149_confirmation
F146=exp_local/cqn_stage146_flow_rerank

declare -A RUNS=(
  [clean_s1]="${CLEAN1}"
  [clean_s2]="$(ls -d ${C149}/move_plate_clean_seed2_*/ | head -1)"
  [clean_s3]="$(ls -d ${C149}/move_plate_clean_seed3_*/ | head -1)"
  [b20k_s1]="$(ls -d ${F146}/move_plate_b20k_seed1_*/ | head -1)"
  [b20k_s2]="$(ls -d ${F146}/move_plate_b20k_seed2_*/ | head -1)"
  [b20k_s3]="$(ls -d ${F146}/move_plate_b20k_seed3_*/ | head -1)"
  [m16_s1]="$(ls -d ${F146}/move_plate_m16_seed1_*/ | head -1)"
  [m16_s2]="$(ls -d ${F146}/move_plate_m16_seed2_*/ | head -1)"
  [m16_s3]="$(ls -d ${F146}/move_plate_m16_seed3_*/ | head -1)"
)

eval_one () {
  local LABEL="$1" RUN="$2" SNAP="$3" PROTO="$4" GPU="$5"
  local OUT_JSON="${OUT}/${LABEL}_${PROTO}.json"
  [ -f "${OUT_JSON}" ] && { echo "[sealed] skip ${LABEL} ${PROTO} (exists)"; return; }
  MUJOCO_GL=egl .venv/bin/python scripts/eval_cqn_as_bigym_checkpoint.py \
    --run-dir "${RUN}" \
    --snapshot "${SNAP}" \
    --gpu-id "${GPU}" \
    --num-eval-episodes 50 \
    --eval-seed-start 600 \
    --output "${OUT_JSON}" \
    > "${OUT}/${LABEL}_${PROTO}.log" 2>&1
  echo "[sealed] ${LABEL} ${PROTO} done"
}

ARMS=(clean_s1 clean_s2 clean_s3 b20k_s1 b20k_s2 b20k_s3 m16_s1 m16_s2 m16_s3)

gpu_worker () {
  local GPU="$1"; shift
  for LABEL in "$@"; do
    local RUN="${RUNS[$LABEL]}"
    read -r PRIMARY FINAL < <(.venv/bin/python scripts/resolve_cqn_validation_best_snapshot.py "${RUN}")
    eval_one "${LABEL}" "${RUN}" "${PRIMARY}" primary "${GPU}"
    if [ "${PRIMARY}" != "${FINAL}" ]; then
      eval_one "${LABEL}" "${RUN}" "${FINAL}" final "${GPU}"
    else
      cp "${OUT}/${LABEL}_primary.json" "${OUT}/${LABEL}_final.json" 2>/dev/null || true
      echo "[sealed] ${LABEL} final == primary"
    fi
  done
}

gpu_worker "${GPU_A}" clean_s1 clean_s3 b20k_s2 m16_s1 m16_s3 &
PID_A=$!
gpu_worker "${GPU_B}" clean_s2 b20k_s1 b20k_s3 m16_s2 &
PID_B=$!
wait "${PID_A}" "${PID_B}"
echo "[sealed] all evaluations complete"
