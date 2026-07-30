#!/usr/bin/env bash
# Stage-141 causal gate: same-state branch probe on all four 10.5k
# checkpoints (2 seeds x control/treatment).
#
# Pre-registered protocol (cqn-flow.md section 22.3):
#   intervention: structured_horizon, level 0 (cell 0.4), H=4
#   readout: deepest level (2), round_robin dimension (Q-independent)
#   fresh eval seeds 300-307, anchors 30/75/120, fixed 10.5k checkpoint
#   pass: treatment pairwise-sign state-bootstrap CI lower bound > 50%
#         AND treatment > matched control arm.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_A="${1:-1}"
GPU_B="${2:-5}"
STAMP="${3:?usage: run_cqn_stage141_causal_gate.sh GPU_A GPU_B RUN_STAMP}"
BASE="exp_local/cqn_stage141_cv_rct"
OUT="${BASE}/causal_gate"
mkdir -p "${OUT}"

probe () {
  local RUN_DIR="$1" LABEL="$2" GPU="$3"
  # Primary: Stage-VII-comparable structured_horizon with deepest-level
  # readout (q_span dimension selection; the tool restricts round_robin to
  # sibling_horizon).  The q_span protocol is equally critic-favorable for
  # control and treatment, so the matched contrast stays fair.
  .venv/bin/python scripts/analyze_cqn_branch_counterfactual.py \
    --run-dir "${RUN_DIR}" \
    --snapshot "${RUN_DIR}/snapshots/10500_snapshot.pkl" \
    --output "${OUT}/${LABEL}_branch_L0_scoreL2.json" \
    --gpu-id "${GPU}" \
    --eval-seeds 300,301,302,303,304,305,306,307 \
    --anchor-steps 30,75,120 \
    --intervention-mode structured_horizon \
    --intervention-horizon 4 \
    --score-level 2 \
    --bootstrap-replicates 10000 \
    > "${OUT}/${LABEL}_branch_L0_scoreL2.log" 2>&1
  echo "[gate] ${LABEL} primary done"
  # Secondary robustness: Q-independent round_robin over level-0 sibling
  # bins, the anti-cheat protocol from cqn-flow.md 21.101.
  .venv/bin/python scripts/analyze_cqn_branch_counterfactual.py \
    --run-dir "${RUN_DIR}" \
    --snapshot "${RUN_DIR}/snapshots/10500_snapshot.pkl" \
    --output "${OUT}/${LABEL}_sibling_L0_rr.json" \
    --gpu-id "${GPU}" \
    --eval-seeds 300,301,302,303,304,305,306,307 \
    --anchor-steps 30,75,120 \
    --intervention-mode sibling_horizon \
    --intervention-horizon 4 \
    --force-level 0 \
    --dimension-selection round_robin \
    --bootstrap-replicates 10000 \
    > "${OUT}/${LABEL}_sibling_L0_rr.log" 2>&1
  echo "[gate] ${LABEL} secondary done"
}

gpu_chain () {
  local GPU="$1"; shift
  while (($#)); do
    local SEED="$1" W="$2"; shift 2
    local TAG="seed${SEED}_w${W}"
    probe "${BASE}/move_plate_cv_rct_${TAG}_gpu${GPU}_${STAMP}" "${TAG}" "${GPU}"
  done
}

gpu_chain "${GPU_A}" 1 0p0 1 0p1 &
PID_A=$!
gpu_chain "${GPU_B}" 2 0p0 2 0p1 &
PID_B=$!
wait "${PID_A}" "${PID_B}"
echo "[gate] all probes complete"
