#!/usr/bin/env bash
# Stage-144: 50k-frame CV-RCT scale test, then frozen sibling-protocol gate.
#
# Phase 1: seeds 4 and 5, control (w=0.0) and treatment (w=0.1); a seed's
#          two arms share one GPU so the matched pair sees identical
#          hardware.  Fixed 50.5k endpoint, no checkpoint selection.
# Phase 2: probe all four checkpoints with the frozen sibling protocol on
#          never-used eval seeds 500-511.
# Phase 3: crossed-bootstrap verdict (same summarizer as Stage-143).
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_A="${1:-1}"
GPU_B="${2:-5}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_stage144_cv_rct_50k"
OUT="${BASE}/gate"
mkdir -p "${OUT}"
cp "$0" "${BASE}/stage144_controller.${STAMP}.sh"

run_seed () {
  local SEED="$1" GPU="$2"
  for W in 0.0 0.1; do
    local TAG="seed${SEED}_w${W/0./0p}"
    local RUN_DIR="${BASE}/move_plate_cv_rct50k_${TAG}_gpu${GPU}_${STAMP}"
    echo "[stage144] train ${TAG} on GPU${GPU}"
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
      .venv/bin/python train.py \
      launch=cqn_as_pixel_bigym_stage144_cv_rct_50k_gate \
      env=bigym/move_plate \
      seed="${SEED}" \
      method.cv_rct_weight="${W}" \
      wandb.name="cqn_as_stage144_cv_rct50k_${TAG}_move_plate" \
      hydra.run.dir="${RUN_DIR}" \
      > "${RUN_DIR}.launch.log" 2>&1
    echo "[stage144] train ${TAG} done"
  done
}

run_seed 4 "${GPU_A}" &
PID_A=$!
run_seed 5 "${GPU_B}" &
PID_B=$!
wait "${PID_A}" "${PID_B}"

probe () {
  local RUN_DIR="$1" LABEL="$2" GPU="$3"
  .venv/bin/python scripts/analyze_cqn_branch_counterfactual.py \
    --run-dir "${RUN_DIR}" \
    --snapshot "${RUN_DIR}/snapshots/50500_snapshot.pkl" \
    --output "${OUT}/${LABEL}_sibling_L0_rr.json" \
    --gpu-id "${GPU}" \
    --eval-seeds 500,501,502,503,504,505,506,507,508,509,510,511 \
    --anchor-steps 30,75,120 \
    --intervention-mode sibling_horizon \
    --intervention-horizon 4 \
    --force-level 0 \
    --dimension-selection round_robin \
    --bootstrap-replicates 10000 \
    > "${OUT}/${LABEL}_sibling_L0_rr.log" 2>&1
  echo "[stage144] probe ${LABEL} done"
}

probe_chain () {
  local GPU="$1" SEED="$2"
  for W in 0p0 0p1; do
    probe "${BASE}/move_plate_cv_rct50k_seed${SEED}_w${W}_gpu${GPU}_${STAMP}" \
      "seed${SEED}_w${W}" "${GPU}"
  done
}

probe_chain "${GPU_A}" 4 &
PID_A=$!
probe_chain "${GPU_B}" 5 &
PID_B=$!
wait "${PID_A}" "${PID_B}"

.venv/bin/python scripts/summarize_cqn_stage143_gate.py \
  --gate-dir "${OUT}" \
  --seeds 4,5 \
  --output "${OUT}/stage144_gate_summary.json"
echo "[stage144] complete"
