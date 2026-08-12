#!/usr/bin/env bash
# cqn-flow.md 51: official CQN-AS MovePlate reproduction on the new replay
# pipeline (vectorized assembly + device merge). Full 101k train on one GPU
# (async protocol: no in-loop eval), then sealed 200-ep@800 on the final
# snapshot, official execution (no replan override). Criteria: task success
# in the official seed band (4-seed mean 64.6) + wall-clock ~2.6-2.7h.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-2}"
SEED="${2:-1}"
STAMP="$(date +%Y%m%d%H%M%S)"
OUT="exp_local/cqn_official_repro_newinfra"
RUN_DIR="${OUT}/move_plate_official_seed${SEED}_gpu${GPU}_${STAMP}"
mkdir -p "${OUT}"

echo "[repro] train official seed${SEED} on GPU${GPU} ($(date +%H:%M:%S))"
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
  .venv/bin/python train_fast.py \
  launch=cqn_as_pixel_bigym_demo_driven \
  env=bigym/move_plate \
  seed="${SEED}" \
  eval_every_steps=1000000 \
  num_eval_episodes=0 \
  num_eval_envs=0 \
  log_eval_video=false \
  save_snapshot=true \
  snapshot_every_n=5000 \
  save_csv=true \
  wandb.use=false \
  hydra.run.dir="${RUN_DIR}" \
  > "${RUN_DIR}.launch.log" 2>&1
echo "[repro] train done ($(date +%H:%M:%S))"

XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
  .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
  --run-dir "${RUN_DIR}" --gpu-id "${GPU}" --num-eval-episodes 200 \
  --eval-seed-start 800 --num-eval-envs 25 --csv-name ep200_seeds800.csv \
  --skip-steps "$(ls ${RUN_DIR}/snapshots/ | grep -oE '^[0-9]+' | sort -n | head -n -1 | tr '\n' ',' | sed 's/,$//')" \
  > "${RUN_DIR}/ep200.log" 2>&1
echo "[repro] sealed eval done ($(date +%H:%M:%S))"
tail -2 "${RUN_DIR}/ep200_seeds800.csv" 2>/dev/null || cat "${RUN_DIR}/ep200.log" | tail -3
awk -F, 'END{printf "[repro] train wall-clock: %.2f h for %d steps\n", $39/3600, $32}' "${RUN_DIR}/train.csv"
