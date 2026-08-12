#!/usr/bin/env bash
# R-line wave-3 D1 "cfaug": pre-populate each run's replay dir with a
# 120-episode subset of the generated counterfactual failures (COPIES, never
# links — artifact cleanup unlinks in place), then train 2 seeds on one card.
set -uo pipefail
cd "$(dirname "$0")/.."

STAMP="${1:-$(date +%Y%m%d%H%M%S)}"
GPU_UUID="${2:-GPU-03f1431f-36c0-b258-6ca1-05007175e3eb}"   # GPU3
EGL_ID="${3:-3}"
SRC=exp_local/cqn_rline/counterfactual_episodes_v1
OUT=exp_local/cqn_trunc_arms/cfaug_move_plate
N_INJECT=120

mkdir -p "${OUT}"
for SEED in 1 2; do
  RUN_DIR="${OUT}/seed${SEED}_${STAMP}"
  mkdir -p "${RUN_DIR}/replay"
  ls "${SRC}"/*.npz | head -n "${N_INJECT}" | while read -r f; do
    cp "$f" "${RUN_DIR}/replay/"
  done
  echo "[cfaug] seed${SEED}: injected $(ls ${RUN_DIR}/replay | wc -l) episodes"
done

(
  bash scripts/run_cqn_trunc_arm.sh cfaug "${GPU_UUID}" "${EGL_ID}" 1 "${STAMP}" &
  sleep 150
  bash scripts/run_cqn_trunc_arm.sh cfaug "${GPU_UUID}" "${EGL_ID}" 2 "${STAMP}" &
  wait
) > "${OUT}/rline_cfaug_${STAMP}.log" 2>&1
echo "[cfaug] complete $(date +%H:%M:%S)"
