#!/usr/bin/env bash
# Ablation: official CQN-AS + demo truncation + use_self_imitation=FALSE.
#
# Fills the missing cell of a 2x2 (all seed 1, sealed 200-ep @ seeds 800-999,
# fixed raw-101k endpoint, no checkpoint selection):
#
#                 self_imitation=true      self_imitation=false
#   untruncated   67.5  (newinfra s1)      44.0  (cqn_stage170_no_si)
#   truncated     running (trunc_s1)       <-- THIS RUN
#
# Hypothesis under test (user's): part of self-imitation's +23.5pp on the
# untruncated recipe comes from relabelled ONLINE successes carrying
# correctly-scaled returns (RTG <= 1.0, because the live env terminates on
# first success) and thereby diluting the 60 saturated demo episodes whose
# discounted RTG is 3.6-24.5 and clips to the C51 top atom. If so, once the
# demos are truncated there is no saturation left to dilute, and switching
# self-imitation off should cost much less than 23.5pp.
#
# Pinned by UUID (numeric CUDA ids do NOT match nvidia-smi order on this box)
# with an explicitly chosen EGL device, so neither compute nor render lands
# on another user's card.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_UUID="${1:-GPU-d224bb2a-2407-1fae-e610-410b2f2bdd08}"   # physical GPU4
EGL_ID="${2:-4}"                                            # EGL 4 -> physical GPU0 (idle)
SEED="${3:-1}"
STAMP="$(date +%Y%m%d%H%M%S)"
OUT="exp_local/cqn_trunc_nosi"
RUN_DIR="${OUT}/move_plate_trunc_nosi_seed${SEED}_${STAMP}"
mkdir -p "${OUT}"
echo "${RUN_DIR}" > "${OUT}/seed${SEED}_latest.txt"

echo "[nosi] train seed${SEED} on ${GPU_UUID} egl=${EGL_ID} ($(date +%H:%M:%S))"
if ! CUDA_VISIBLE_DEVICES="${GPU_UUID}" MUJOCO_EGL_DEVICE_ID="${EGL_ID}" \
  MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python train_fast.py \
  launch=cqn_as_pixel_bigym_demo_driven \
  env=bigym/move_plate \
  env.truncate_demo_at_success=true \
  use_self_imitation=false \
  seed="${SEED}" \
  xla_mem_fraction=0.45 \
  eval_every_steps=1000000 \
  num_eval_episodes=0 \
  num_eval_envs=0 \
  log_eval_video=false \
  save_snapshot=true \
  snapshot_every_n=5000 \
  save_csv=true \
  wandb.use=false \
  hydra.run.dir="${RUN_DIR}" \
  > "${RUN_DIR}.launch.log" 2>&1; then
  touch "${RUN_DIR}/failed"; echo "[nosi] TRAIN FAILED"; exit 1
fi
touch "${RUN_DIR}/train_complete"
echo "[nosi] train done ($(date +%H:%M:%S))"

# Validation curve (100 episodes: seeds 400-449 then 450-499), then the
# sealed fixed-endpoint read on the final snapshot only.
for BAND in 400 450; do
  CUDA_VISIBLE_DEVICES="${GPU_UUID}" MUJOCO_EGL_DEVICE_ID="${EGL_ID}" \
    MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${RUN_DIR}" --num-eval-episodes 50 --eval-seed-start "${BAND}" \
    --num-eval-envs 25 --csv-name "val50_seeds${BAND}.csv" \
    > "${RUN_DIR}/val50_seeds${BAND}.log" 2>&1 || touch "${RUN_DIR}/val_failed_${BAND}"
done
echo "[nosi] validation curves done ($(date +%H:%M:%S))"

CUDA_VISIBLE_DEVICES="${GPU_UUID}" MUJOCO_EGL_DEVICE_ID="${EGL_ID}" \
  MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
  --run-dir "${RUN_DIR}" --num-eval-episodes 200 --eval-seed-start 800 \
  --num-eval-envs 25 --csv-name ep200_seeds800.csv \
  --skip-steps "$(ls ${RUN_DIR}/snapshots/ | grep -oE '^[0-9]+' | sort -n | head -n -1 | tr '\n' ',' | sed 's/,$//')" \
  > "${RUN_DIR}/ep200.log" 2>&1 || touch "${RUN_DIR}/sealed_failed"
touch "${RUN_DIR}/complete"
echo "[nosi] sealed eval done ($(date +%H:%M:%S))"
tail -2 "${RUN_DIR}/ep200_seeds800.csv" 2>/dev/null || true
