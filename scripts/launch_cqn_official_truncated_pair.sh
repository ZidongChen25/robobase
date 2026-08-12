#!/usr/bin/env bash
# Launch seeds 1 and 2 of the truncated official baseline co-resident on one
# card (0.45 XLA slice each, 120 s stagger to dodge the CUDA init race).
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-1}"
STAMP="$(date +%Y%m%d%H%M%S)"
LOG="exp_local/cqn_official_truncated/pair_gpu${GPU}_${STAMP}.log"
mkdir -p exp_local/cqn_official_truncated

{
  echo "[pair] stamp ${STAMP} gpu ${GPU} start $(date +%H:%M:%S)"
  bash scripts/run_cqn_official_truncated.sh "${GPU}" 1 "${STAMP}" &
  P1=$!
  sleep 120
  bash scripts/run_cqn_official_truncated.sh "${GPU}" 2 "${STAMP}" &
  P2=$!
  wait "${P1}" "${P2}"
  echo "[pair] both arms exited $(date +%H:%M:%S)"
} > "${LOG}" 2>&1
