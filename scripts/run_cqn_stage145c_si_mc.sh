#!/usr/bin/env bash
# Stage-145c (CQN-value line v3): two single-variable arms vs the mc_only
# control (62.7%): (i) success-filtered self-imitation (official CQN-AS
# relabeling, binary filter instead of soft AWR weights); (ii) stronger MC
# anchor weight 1.0 (stage4 seed1 historical best 72%).  3 seeds each.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_A="${1:-1}"
GPU_B="${2:-5}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_stage145_awr"
mkdir -p "${BASE}"
cp "$0" "${BASE}/stage145c_controller.${STAMP}.sh"

run_arm () {
  local SEED="$1" GPU="$2" LABEL="$3"; shift 3
  local RUN_DIR="${BASE}/move_plate_${LABEL}_seed${SEED}_gpu${GPU}_${STAMP}"
  echo "[stage145c] start ${LABEL} seed${SEED} on GPU${GPU}"
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
    .venv/bin/python train.py \
    launch=cqn_as_pixel_bigym_stage145_awr_gate \
    env=bigym/move_plate \
    seed="${SEED}" \
    method.awr_beta=null \
    "$@" \
    wandb.name="cqn_as_stage145c_${LABEL}_seed${SEED}_move_plate" \
    hydra.run.dir="${RUN_DIR}" \
    > "${RUN_DIR}.launch.log" 2>&1
  echo "[stage145c] done ${LABEL} seed${SEED}"
}

(
  run_arm 1 "${GPU_A}" si use_self_imitation=true
  run_arm 3 "${GPU_A}" si use_self_imitation=true
  run_arm 3 "${GPU_A}" mc1p0 method.mc_return_weight=1.0
) &
PID_A=$!
(
  run_arm 2 "${GPU_B}" si use_self_imitation=true
  run_arm 1 "${GPU_B}" mc1p0 method.mc_return_weight=1.0
  run_arm 2 "${GPU_B}" mc1p0 method.mc_return_weight=1.0
) &
PID_B=$!
wait "${PID_A}" "${PID_B}"
echo "[stage145c] all arms complete"
