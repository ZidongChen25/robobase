#!/usr/bin/env bash
# R-line wave 1b + 2 (cqn-rline.md): nstep3 seeds 3/4 on GPU5, tokensplit
# seeds 1/2 on GPU2. EGL placements probe-verified 2026-08-09.
set -uo pipefail
cd "$(dirname "$0")/.."

STAMP="${1:-$(date +%Y%m%d%H%M%S)}"
U2=GPU-80b9cc0d-df5c-be12-e848-042d37578544   # physical GPU2 -> EGL 2
U5=GPU-2f044e6a-9150-0e30-7d97-009bdd425b11   # physical GPU5 -> EGL 1
mkdir -p exp_local/cqn_trunc_arms

(
  bash scripts/run_cqn_trunc_arm.sh tokensplit "${U2}" 2 1 "${STAMP}" &
  sleep 120
  bash scripts/run_cqn_trunc_arm.sh tokensplit "${U2}" 2 2 "${STAMP}" &
  wait
) > "exp_local/cqn_trunc_arms/rline_wave2_tokensplit_${STAMP}.log" 2>&1 &
TS_PID=$!

(
  bash scripts/run_cqn_trunc_arm.sh nstep3 "${U5}" 1 3 "${STAMP}" &
  sleep 120
  bash scripts/run_cqn_trunc_arm.sh nstep3 "${U5}" 1 4 "${STAMP}" &
  wait
) > "exp_local/cqn_trunc_arms/rline_wave1b_nstep3_${STAMP}.log" 2>&1 &
N34_PID=$!

echo "tokensplit controller pid ${TS_PID}, nstep3-s34 controller pid ${N34_PID}, stamp ${STAMP}"
wait
echo "[rline wave1b+2] all arms complete $(date +%H:%M:%S)"
