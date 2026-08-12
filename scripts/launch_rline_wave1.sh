#!/usr/bin/env bash
# R-line wave 1 (cqn-rline.md): arm alpha "rfloor" (floor 0.1 + constant
# bc_lambda 0.0125) seeds 1/2 on GPU3, arm gamma "nstep3" (replay.nstep=3)
# seeds 1/2 on GPU4. Two runs per card, 120 s stagger, UUID pins, EGL ids
# verified by render probe on 2026-08-09.
set -uo pipefail
cd "$(dirname "$0")/.."

STAMP="${1:-$(date +%Y%m%d%H%M%S)}"
U3=GPU-03f1431f-36c0-b258-6ca1-05007175e3eb   # physical GPU3 -> EGL 3
U4=GPU-d224bb2a-2407-1fae-e610-410b2f2bdd08   # physical GPU4 -> EGL 0
mkdir -p exp_local/cqn_trunc_arms

(
  bash scripts/run_cqn_trunc_arm.sh rfloor "${U3}" 3 1 "${STAMP}" &
  sleep 120
  bash scripts/run_cqn_trunc_arm.sh rfloor "${U3}" 3 2 "${STAMP}" &
  wait
) > "exp_local/cqn_trunc_arms/rline_wave1_rfloor_${STAMP}.log" 2>&1 &
RFLOOR_PID=$!

(
  bash scripts/run_cqn_trunc_arm.sh nstep3 "${U4}" 0 1 "${STAMP}" &
  sleep 120
  bash scripts/run_cqn_trunc_arm.sh nstep3 "${U4}" 0 2 "${STAMP}" &
  wait
) > "exp_local/cqn_trunc_arms/rline_wave1_nstep3_${STAMP}.log" 2>&1 &
NSTEP3_PID=$!

echo "rfloor controller pid ${RFLOOR_PID}, nstep3 controller pid ${NSTEP3_PID}, stamp ${STAMP}"
wait
echo "[rline wave1] all arms complete $(date +%H:%M:%S)"
