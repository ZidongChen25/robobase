#!/usr/bin/env bash
# Stage-157: verify 10.5k-era conclusions on the 100k paper checkpoints
# (cqn-flow.md 39).  Per seed: (1) value-fidelity calibration audit,
# (2) sibling counterfactual probe (Stage-143/151 protocol, eval seeds
# 700-711).
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_A="${1:-1}"
GPU_B="${2:-5}"
OUT="exp_local/cqn_stage157_100k_verify"
mkdir -p "${OUT}"

verify_seed () {
  local SEED="$1" GPU="$2"
  local RUN_DIR="exp_local/pixel_cqn_as/move_plate_paper_seed${SEED}_100k_nw0_20260721"
  local SNAP="${RUN_DIR}/snapshots/101000_snapshot.pkl"
  echo "[stage157] fidelity seed${SEED} on GPU${GPU}"
  MUJOCO_GL=egl .venv/bin/python scripts/analyze_cqn_value_fidelity.py \
    --run-dir "${RUN_DIR}" \
    --snapshot "${SNAP}" \
    --output "${OUT}/fidelity_seed${SEED}.json" \
    --gpu-id "${GPU}" \
    --offline-episode-count 51 \
    > "${OUT}/fidelity_seed${SEED}.log" 2>&1
  echo "[stage157] fidelity seed${SEED} done"
  echo "[stage157] probe seed${SEED} on GPU${GPU}"
  MUJOCO_GL=egl .venv/bin/python scripts/analyze_cqn_branch_counterfactual.py \
    --run-dir "${RUN_DIR}" \
    --snapshot "${SNAP}" \
    --output "${OUT}/sibling_seed${SEED}.json" \
    --gpu-id "${GPU}" \
    --eval-seeds 700,701,702,703,704,705,706,707,708,709,710,711 \
    --anchor-steps 30,75,120 \
    --intervention-mode sibling_horizon \
    --intervention-horizon 4 \
    --force-level 0 \
    --dimension-selection round_robin \
    --bootstrap-replicates 10000 \
    > "${OUT}/sibling_seed${SEED}.log" 2>&1
  echo "[stage157] probe seed${SEED} done"
}

worker_a () { verify_seed 1 "${GPU_A}"; verify_seed 2 "${GPU_A}"; }
worker_b () { verify_seed 3 "${GPU_B}"; verify_seed 4 "${GPU_B}"; }

worker_a &
PID_A=$!
worker_b &
PID_B=$!
wait "${PID_A}" "${PID_B}"
echo "[stage157] all verifications complete"
