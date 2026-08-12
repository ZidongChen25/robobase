#!/usr/bin/env bash
# One-GPU Stage-7 pair: max expected-Q floor vs dense categorical Q targets.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-0}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_no_bc/stage7_seed1_gpu${GPU}_${STAMP}"
MAX_FLOOR="${BASE}/mc_max_floor"
DENSE_RETURN="${BASE}/mc_dense_return"
mkdir -p "${BASE}"
printf "%s\n" "${BASHPID}" > "${BASE}/controller.pid"
printf "%s\n" "${BASE}" > exp_local/cqn_no_bc/stage7_latest.txt

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

train_one cqn_as_pixel_bigym_nobc_mc_stage7_max_floor_gate \
  "${MAX_FLOOR}" &
MAX_FLOOR_PID=$!
printf "%s\n" "${MAX_FLOOR_PID}" > "${BASE}/mc_max_floor.pid"
sleep 120
train_one cqn_as_pixel_bigym_nobc_mc_stage7_dense_return_gate \
  "${DENSE_RETURN}" &
DENSE_RETURN_PID=$!
printf "%s\n" "${DENSE_RETURN_PID}" > "${BASE}/mc_dense_return.pid"

status=0
wait "${MAX_FLOOR_PID}" || status=$?
wait "${DENSE_RETURN_PID}" || status=$?
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

evaluate_one "${MAX_FLOOR}"
evaluate_one "${DENSE_RETURN}"
touch "${BASE}/validation_complete"

.venv/bin/python scripts/summarize_cqn_no_bc_stage7.py \
  --max-floor-run "${MAX_FLOOR}" \
  --dense-return-run "${DENSE_RETURN}" \
  --output "${BASE}/stage7_summary.json" \
  > "${BASE}/stage7_summary.log" 2>&1
touch "${BASE}/complete"
