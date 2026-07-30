#!/usr/bin/env bash
# Stage-154/155 sealed batch (cqn-flow.md 37.1): tdoff s1-3 + pureflow s1-3,
# 50-ep at eval-seed-start 800, dual protocol, for the three-way sealed
# decomposition against CCFF (78.0/80.0) and clean (68.0/63.3).
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_A="${1:-1}"
GPU_B="${2:-5}"
OUT="exp_local/cqn_stage155_pure_flow/sealed"
mkdir -p "${OUT}"

T154=exp_local/cqn_stage154_ccff_tdoff
T155=exp_local/cqn_stage155_pure_flow

declare -A RUNS=(
  [tdoff_s1]="$(ls -d ${T154}/move_plate_tdoff_seed1_*/ | head -1)"
  [tdoff_s2]="$(ls -d ${T154}/move_plate_tdoff_seed2_*/ | head -1)"
  [tdoff_s3]="$(ls -d ${T154}/move_plate_tdoff_seed3_*/ | head -1)"
  [pureflow_s1]="$(ls -d ${T155}/move_plate_pureflow_seed1_*/ | head -1)"
  [pureflow_s2]="$(ls -d ${T155}/move_plate_pureflow_seed2_*/ | head -1)"
  [pureflow_s3]="$(ls -d ${T155}/move_plate_pureflow_seed3_*/ | head -1)"
)

eval_one () {
  local LABEL="$1" RUN="$2" SNAP="$3" PROTO="$4" GPU="$5"
  local OUT_JSON="${OUT}/${LABEL}_${PROTO}.json"
  [ -f "${OUT_JSON}" ] && { echo "[155-sealed] skip ${LABEL} ${PROTO} (exists)"; return; }
  MUJOCO_GL=egl .venv/bin/python scripts/eval_cqn_as_bigym_checkpoint.py \
    --run-dir "${RUN}" \
    --snapshot "${SNAP}" \
    --gpu-id "${GPU}" \
    --num-eval-episodes 50 \
    --eval-seed-start 800 \
    --output "${OUT_JSON}" \
    > "${OUT}/${LABEL}_${PROTO}.log" 2>&1
  echo "[155-sealed] ${LABEL} ${PROTO} done"
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
      echo "[155-sealed] ${LABEL} final == primary"
    fi
  done
}

sealed_worker "${GPU_A}" tdoff_s1 tdoff_s3 pureflow_s2 &
PID_A=$!
sealed_worker "${GPU_B}" tdoff_s2 pureflow_s1 pureflow_s3 &
PID_B=$!
wait "${PID_A}" "${PID_B}"
echo "[155-sealed] batch complete"
for J in "${OUT}"/*.json; do
  echo -n "$(basename "$J" .json): "
  .venv/bin/python -c "import json; print(json.load(open('$J')).get('success_percent'))"
done
