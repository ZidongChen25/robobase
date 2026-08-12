#!/usr/bin/env bash
# Official CQN-AS MovePlate baseline on the new replay pipeline, WITH the
# demo post-success-tail correction (env.truncate_demo_at_success=true).
#
# Why: live BiGym terminates on the first successful step and pays reward
# once, but recorded demos keep a ~24-step post-success tail. Measured on
# move_plate: 96.0% of demo transitions have discounted RTG above the C51
# v_max=2.0 and clip to the same top atom. Truncation puts demo returns in
# [0.044, 1.0] and makes the demo replay the same MDP the agent acts in.
#
# This run re-measures the OFFICIAL baseline under that correction so the
# no-BC comparison has a valid reference. Everything else is stock official
# CQN-AS (demo_driven launch, BC/FOSD/margin on, no offline phase, official
# ensemble execution, 101k online).
#
# References for comparison (both UNTRUNCATED):
#   official 4-seed sealed 200-ep@800 fixed endpoint = 64.625%
#   new-infra seed-1 sealed 200-ep@800 fixed endpoint = 67.5% (cqn-flow.md 51.1)
#
# Usage: bash scripts/run_cqn_official_truncated.sh [GPU] [SEED]
# Async protocol: no in-loop eval, no wandb, no video. Evaluation is chained
# after training exits: 50-ep seeds 400-449 curve over all snapshots, then a
# sealed 200-ep seeds 800-999 read on the FINAL snapshot only (fixed
# endpoint, no checkpoint selection).
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-1}"
SEED="${2:-1}"
STAMP="${3:-$(date +%Y%m%d%H%M%S)}"
OUT="exp_local/cqn_official_truncated"
RUN_DIR="${OUT}/move_plate_trunc_seed${SEED}_gpu${GPU}_${STAMP}"
mkdir -p "${OUT}"
echo "${RUN_DIR}" > "${OUT}/seed${SEED}_latest.txt"

echo "[trunc] train official+truncation seed${SEED} on GPU${GPU} ($(date +%H:%M:%S))"
if ! MUJOCO_GL=egl .venv/bin/python train_fast.py \
  launch=cqn_as_pixel_bigym_demo_driven \
  env=bigym/move_plate \
  env.truncate_demo_at_success=true \
  seed="${SEED}" \
  gpu_id="${GPU}" \
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
  touch "${RUN_DIR}/failed"
  echo "[trunc] TRAIN FAILED seed${SEED} ($(date +%H:%M:%S))"
  exit 1
fi
touch "${RUN_DIR}/train_complete"
echo "[trunc] train done ($(date +%H:%M:%S))"

# Validation curve: every snapshot, 50 episodes, seeds 400-449.
XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
  .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
  --run-dir "${RUN_DIR}" --gpu-id "${GPU}" --num-eval-episodes 50 \
  --eval-seed-start 400 --num-eval-envs 25 --csv-name val50_seeds400.csv \
  > "${RUN_DIR}/val50.log" 2>&1 || touch "${RUN_DIR}/val_failed"
echo "[trunc] validation curve done ($(date +%H:%M:%S))"

# Sealed read: FINAL snapshot only, 200 episodes, seeds 800-999. No
# checkpoint selection -- matches how 64.625% and 67.5% were measured.
XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
  .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
  --run-dir "${RUN_DIR}" --gpu-id "${GPU}" --num-eval-episodes 200 \
  --eval-seed-start 800 --num-eval-envs 25 --csv-name ep200_seeds800.csv \
  --skip-steps "$(ls ${RUN_DIR}/snapshots/ | grep -oE '^[0-9]+' | sort -n | head -n -1 | tr '\n' ',' | sed 's/,$//')" \
  > "${RUN_DIR}/ep200.log" 2>&1 || touch "${RUN_DIR}/sealed_failed"
touch "${RUN_DIR}/complete"
echo "[trunc] sealed eval done ($(date +%H:%M:%S))"
tail -2 "${RUN_DIR}/ep200_seeds800.csv" 2>/dev/null || true
