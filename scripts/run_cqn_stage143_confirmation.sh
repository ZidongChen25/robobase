#!/usr/bin/env bash
# Stage-143: sibling-protocol confirmation (cqn-flow.md 23.4).
#
# Phase 1: train seed-3 control/treatment pair (one arm per GPU).
# Phase 2: probe all six checkpoints (seeds 1-3 x control/treatment) with
#          the pre-registered primary protocol on FRESH eval seeds 400-411
#          (the 300-307 set generated the hypothesis and is not reused).
# Phase 3: crossed-bootstrap verdict.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_A="${1:-1}"
GPU_B="${2:-5}"
TRAIN_STAMP="${3:?usage: run_cqn_stage143_confirmation.sh GPU_A GPU_B STAGE141_STAMP}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_stage141_cv_rct"
OUT="${BASE}/stage143_gate"
mkdir -p "${OUT}"
cp "$0" "${BASE}/stage143_controller.${STAMP}.sh"

train_arm () {
  local W="$1" GPU="$2"
  local TAG="seed3_w${W/0./0p}"
  local RUN_DIR="${BASE}/move_plate_cv_rct_${TAG}_gpu${GPU}_${STAMP}"
  echo "[stage143] train ${TAG} on GPU${GPU}"
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
    .venv/bin/python train.py \
    launch=cqn_as_pixel_bigym_stage141_cv_rct_gate \
    env=bigym/move_plate \
    seed=3 \
    method.cv_rct_weight="${W}" \
    wandb.name="cqn_as_stage143_cv_rct_${TAG}_move_plate" \
    hydra.run.dir="${RUN_DIR}" \
    > "${RUN_DIR}.launch.log" 2>&1
  echo "[stage143] train ${TAG} done"
}

probe () {
  local RUN_DIR="$1" LABEL="$2" GPU="$3"
  .venv/bin/python scripts/analyze_cqn_branch_counterfactual.py \
    --run-dir "${RUN_DIR}" \
    --snapshot "${RUN_DIR}/snapshots/10500_snapshot.pkl" \
    --output "${OUT}/${LABEL}_sibling_L0_rr.json" \
    --gpu-id "${GPU}" \
    --eval-seeds 400,401,402,403,404,405,406,407,408,409,410,411 \
    --anchor-steps 30,75,120 \
    --intervention-mode sibling_horizon \
    --intervention-horizon 4 \
    --force-level 0 \
    --dimension-selection round_robin \
    --bootstrap-replicates 10000 \
    > "${OUT}/${LABEL}_sibling_L0_rr.log" 2>&1
  echo "[stage143] probe ${LABEL} done"
}

# Phase 1: seed-3 arms in parallel.
train_arm 0.0 "${GPU_A}" &
PID_A=$!
train_arm 0.1 "${GPU_B}" &
PID_B=$!
wait "${PID_A}" "${PID_B}"

# Phase 2: six probes, three per GPU.
run_dir_for () {
  local SEED="$1" W="$2"
  if [ "${SEED}" = "3" ]; then
    local G="${GPU_A}"; [ "${W}" = "0p1" ] && G="${GPU_B}"
    echo "${BASE}/move_plate_cv_rct_seed3_w${W}_gpu${G}_${STAMP}"
  else
    local G="${GPU_A}"; [ "${SEED}" = "2" ] && G="${GPU_B}"
    echo "${BASE}/move_plate_cv_rct_seed${SEED}_w${W}_gpu${G}_${TRAIN_STAMP}"
  fi
}

probe_chain () {
  local GPU="$1"; shift
  while (($#)); do
    local SEED="$1" W="$2"; shift 2
    probe "$(run_dir_for "${SEED}" "${W}")" "seed${SEED}_w${W}" "${GPU}"
  done
}

probe_chain "${GPU_A}" 1 0p0 1 0p1 3 0p0 &
PID_A=$!
probe_chain "${GPU_B}" 2 0p0 2 0p1 3 0p1 &
PID_B=$!
wait "${PID_A}" "${PID_B}"

# Phase 3: verdict.
.venv/bin/python scripts/summarize_cqn_stage143_gate.py \
  --gate-dir "${OUT}" \
  --seeds 1,2,3 \
  --output "${OUT}/stage143_gate_summary.json"
echo "[stage143] complete"
