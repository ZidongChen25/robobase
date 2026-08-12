#!/usr/bin/env bash
# Tonight's wave: combined and QC, 4 seeds each, on 4 cards (2 runs per card,
# 120 s stagger). Compute and EGL are pinned to the SAME physical card using
# the 2026-08-03 measured EGL map: EGL 0->GPU4, 1->GPU5, 2->GPU2, 3->GPU3,
# 4->GPU0, 5->GPU1.
set -euo pipefail
cd "$(dirname "$0")/.."

STAMP="$(date +%Y%m%d%H%M%S)"
LOG="exp_local/cqn_trunc_arms/wave_${STAMP}.log"
mkdir -p exp_local/cqn_trunc_arms

U0=GPU-79eb6469-87b1-9dd3-ca2d-da93a916e919   # GPU0, EGL 4
U1=GPU-ce804993-c33e-3d10-5676-5bae093a7d96   # GPU1, EGL 5
U2=GPU-80b9cc0d-df5c-be12-e848-042d37578544   # GPU2, EGL 2
U4=GPU-d224bb2a-2407-1fae-e610-410b2f2bdd08   # GPU4, EGL 0

# card -> (arm, seed) x2 ; arms split across cards so a card failure does not
# take out a whole arm.
{
  echo "[wave] ${STAMP} start $(date +%H:%M:%S)"
  bash scripts/run_cqn_trunc_arm.sh combined "$U0" 4 1 "$STAMP" & sleep 120
  bash scripts/run_cqn_trunc_arm.sh qc       "$U0" 4 1 "$STAMP" & sleep 120
  bash scripts/run_cqn_trunc_arm.sh combined "$U1" 5 2 "$STAMP" & sleep 120
  bash scripts/run_cqn_trunc_arm.sh qc       "$U1" 5 2 "$STAMP" & sleep 120
  bash scripts/run_cqn_trunc_arm.sh combined "$U2" 2 3 "$STAMP" & sleep 120
  bash scripts/run_cqn_trunc_arm.sh qc       "$U2" 2 3 "$STAMP" & sleep 120
  bash scripts/run_cqn_trunc_arm.sh combined "$U4" 0 4 "$STAMP" & sleep 120
  bash scripts/run_cqn_trunc_arm.sh qc       "$U4" 0 4 "$STAMP" &
  wait
  echo "[wave] all exited $(date +%H:%M:%S)"
} > "${LOG}" 2>&1
