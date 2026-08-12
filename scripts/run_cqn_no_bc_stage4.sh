#!/usr/bin/env bash
# One-GPU Stage-4 pair: top-1 vs top-2 unseen-Q reward floor.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-0}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_no_bc/stage4_seed1_gpu${GPU}_${STAMP}"
TOP1="${BASE}/mc_top1_floor"
TOP2="${BASE}/mc_top2_floor"
mkdir -p "${BASE}"
printf "%s\n" "${BASHPID}" > "${BASE}/controller.pid"
printf "%s\n" "${BASE}" > exp_local/cqn_no_bc/stage4_latest.txt

train_one () {
  local launch="$1"
  local run_dir="$2"
  XLA_PYTHON_CLIENT_PREALLOCATE=true \
    MUJOCO_GL=egl \
    .venv/bin/python train_fast.py \
    launch="${launch}" \
    env=bigym/move_plate \
    seed=1 \
    gpu_id="${GPU}" \
    xla_mem_fraction=0.45 \
    wandb.use=false \
    hydra.run.dir="${run_dir}" \
    > "${run_dir}.launch.log" 2>&1
}

train_one cqn_as_pixel_bigym_nobc_mc_top1_floor_gate "${TOP1}" &
TOP1_PID=$!
printf "%s\n" "${TOP1_PID}" > "${BASE}/mc_top1_floor.pid"
sleep 120
train_one cqn_as_pixel_bigym_nobc_mc_top2_floor_gate "${TOP2}" &
TOP2_PID=$!
printf "%s\n" "${TOP2_PID}" > "${BASE}/mc_top2_floor.pid"

status=0
wait "${TOP1_PID}" || status=$?
wait "${TOP2_PID}" || status=$?
if [[ "${status}" -ne 0 ]]; then
  printf "%s\n" "${status}" > "${BASE}/train_failed"
  exit "${status}"
fi
touch "${BASE}/training_complete"

evaluate_one () {
  local run_dir="$1"
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MUJOCO_GL=egl \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${run_dir}" \
    --gpu-id "${GPU}" \
    --num-eval-episodes 50 \
    --eval-seed-start 400 \
    --num-eval-envs 25 \
    --csv-name val50_seeds400.csv \
    --skip-steps 10500 \
    > "${run_dir}/val50_seeds400.log" 2>&1
}

evaluate_one "${TOP1}"
evaluate_one "${TOP2}"
touch "${BASE}/validation_complete"

.venv/bin/python scripts/summarize_cqn_no_bc_stage4.py \
  --top1-run "${TOP1}" \
  --top2-run "${TOP2}" \
  --output "${BASE}/stage4_summary.json" \
  > "${BASE}/stage4_summary.log" 2>&1
touch "${BASE}/complete"
