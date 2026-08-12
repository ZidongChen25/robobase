#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
bash scripts/run_cqn_trunc_arm.sh noens GPU-ce804993-c33e-3d10-5676-5bae093a7d96 5 1 20260804162305 & sleep 120
bash scripts/run_cqn_trunc_arm.sh noens GPU-ce804993-c33e-3d10-5676-5bae093a7d96 5 2 20260804162305 &
wait
for s in 1 2; do
  D=$(cat exp_local/cqn_trunc_arms/noens/seed${s}_latest.txt)
  SK=$(ls "$D/snapshots" | grep -oE '^[0-9]+' | sort -n -u | grep -v '^100000$' | tr '\n' ',' | sed 's/,$//')
  CUDA_VISIBLE_DEVICES=GPU-ce804993-c33e-3d10-5676-5bae093a7d96 MUJOCO_EGL_DEVICE_ID=5 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py --run-dir "$D" \
    --num-eval-episodes 200 --eval-seed-start 800 --num-eval-envs 25 \
    --csv-name ep200_ensembleON.csv --skip-steps "$SK" --replan-interval 1 \
    > "$D/ep200_ensembleON.log" 2>&1
  echo "[noens] seed$s ensemble-ON eval done"
done
