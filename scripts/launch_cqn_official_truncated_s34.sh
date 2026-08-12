#!/usr/bin/env bash
# Truncated official baseline, seeds 3 and 4, co-resident on one card.
# Robustness check for the +8.4pp paired result measured on seed 1.
set -euo pipefail
cd "$(dirname "$0")/.."
GPU="${1:-3}"
STAMP="$(date +%Y%m%d%H%M%S)"
LOG="exp_local/cqn_official_truncated/pair_s34_gpu${GPU}_${STAMP}.log"
mkdir -p exp_local/cqn_official_truncated
{
  echo "[pair34] stamp ${STAMP} gpu ${GPU} start $(date +%H:%M:%S)"
  bash scripts/run_cqn_official_truncated.sh "${GPU}" 3 "${STAMP}" & P1=$!
  sleep 120
  bash scripts/run_cqn_official_truncated.sh "${GPU}" 4 "${STAMP}" & P2=$!
  wait "${P1}" "${P2}"
  echo "[pair34] both exited $(date +%H:%M:%S)"
} > "${LOG}" 2>&1
