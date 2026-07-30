#!/usr/bin/env bash
# Level-0 coherent structured-exploration viability pilot (A-2 pre-check).
#
# Question: at level-0 cell width (0.4) with horizon-4 coherent interventions,
# which start probability keeps BC-policy success in a usable band (~20-50%)
# while producing enough per-dimension assignments for the causal RCT design?
#
# Three dose arms on one GPU, sequential.  Active-fraction targets via
# H*p/(1-p+H*p) with H=4: p_start 0.060 -> 20.3%, 0.027 -> 10.0%,
# 0.013 -> 5.0%.  This is a viability pilot: eval success and realized
# assignment statistics only; no causal claims and no checkpoint selection.
set -euo pipefail

cd "$(dirname "$0")/.."

GPU="${1:-1}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_zoom_coverage"
mkdir -p "${BASE}"

# Immutable snapshot of this controller script for auditability.
cp "$0" "${BASE}/pilot_level0_pscan_controller.${STAMP}.sh"

for P in 0.060 0.027 0.013; do
  TAG="p${P/0./}"
  RUN_DIR="${BASE}/pilot_level0_coherent_${TAG}_seed1_gpu${GPU}_${STAMP}"
  echo "[pilot] start ${TAG} -> ${RUN_DIR}"
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
    .venv/bin/python train.py \
    launch=cqn_as_pixel_bigym_coherent_exploration_gate \
    env=bigym/move_plate \
    method.mc_return_weight=0.1 \
    method.structured_exploration_level=0 \
    method.structured_exploration_prob="${P}" \
    num_train_frames=5500 \
    save_csv=true \
    wandb.name="cqn_as_pilot_level0_coherent_${TAG}_move_plate_seed1" \
    hydra.run.dir="${RUN_DIR}" \
    > "${RUN_DIR}.launch.log" 2>&1
  echo "[pilot] done ${TAG}"
done
echo "[pilot] all arms complete"
