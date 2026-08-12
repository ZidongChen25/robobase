#!/usr/bin/env bash
# Wait for the hard-task trainings to exit (frees the 2-card training quota),
# then train the noens arm (2 seeds) on GPU2 once the gain scan clears it.
set -uo pipefail
cd "$(dirname "$0")/.."
until [ -f "$(cat exp_local/cqn_trunc_arms/official_move_two_plates/seed1_latest.txt)/train_complete" ] \
   && [ -f "$(cat exp_local/cqn_trunc_arms/official_move_two_plates/seed2_latest.txt)/train_complete" ] \
   && [ -f "$(cat exp_local/cqn_trunc_arms/official_sandwich_remove/seed1_latest.txt)/train_complete" ] \
   && [ -f "$(cat exp_local/cqn_trunc_arms/official_sandwich_remove/seed2_latest.txt)/train_complete" ]; do sleep 60; done
echo "[noens] training quota free $(date +%H:%M:%S)"
U2=GPU-80b9cc0d-df5c-be12-e848-042d37578544
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2)" -lt 2000 ]; do sleep 60; done
echo "[noens] GPU2 clear $(date +%H:%M:%S)"
STAMP=$(date +%Y%m%d%H%M%S)
bash scripts/run_cqn_trunc_arm.sh noens "$U2" 2 1 "$STAMP" & sleep 120
bash scripts/run_cqn_trunc_arm.sh noens "$U2" 2 2 "$STAMP" &
wait
# The chained eval above ran with the ensemble OFF (as trained). Add the
# ensemble-ON sealed read -- the cell the user asked for.
for s in 1 2; do
  D=$(cat exp_local/cqn_trunc_arms/noens/seed${s}_latest.txt)
  SK=$(ls "$D/snapshots" | grep -oE '^[0-9]+' | sort -n -u | grep -v '^100000$' | tr '\n' ',' | sed 's/,$//')
  CUDA_VISIBLE_DEVICES="$U2" MUJOCO_EGL_DEVICE_ID=2 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py --run-dir "$D" \
    --num-eval-episodes 200 --eval-seed-start 800 --num-eval-envs 25 \
    --csv-name ep200_ensembleON.csv --skip-steps "$SK" --replan-interval 1 \
    > "$D/ep200_ensembleON.log" 2>&1
  echo "[noens] seed$s ensemble-ON eval done $(date +%H:%M:%S)"
done
echo "[noens] all complete $(date +%H:%M:%S)"
