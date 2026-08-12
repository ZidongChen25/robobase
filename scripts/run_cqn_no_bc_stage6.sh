#!/usr/bin/env bash
# One-GPU Stage-6 pair: parallel vs autoregressive action-dimension Q critic.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-0}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_no_bc/stage6_seed1_gpu${GPU}_${STAMP}"
PARALLEL="${BASE}/mc_parallel_dims"
AUTOREGRESSIVE="${BASE}/mc_autoregressive_dims"
mkdir -p "${BASE}"
printf "%s\n" "${BASHPID}" > "${BASE}/controller.pid"
printf "%s\n" "${BASE}" > exp_local/cqn_no_bc/stage6_latest.txt

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

train_one cqn_as_pixel_bigym_nobc_mc_parallel_dims_gate "${PARALLEL}" &
PARALLEL_PID=$!
printf "%s\n" "${PARALLEL_PID}" > "${BASE}/mc_parallel_dims.pid"
sleep 120
train_one cqn_as_pixel_bigym_nobc_mc_autoregressive_dims_gate \
  "${AUTOREGRESSIVE}" &
AUTOREGRESSIVE_PID=$!
printf "%s\n" "${AUTOREGRESSIVE_PID}" \
  > "${BASE}/mc_autoregressive_dims.pid"

status=0
wait "${PARALLEL_PID}" || status=$?
wait "${AUTOREGRESSIVE_PID}" || status=$?
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

evaluate_one "${PARALLEL}"
evaluate_one "${AUTOREGRESSIVE}"
touch "${BASE}/validation_complete"

.venv/bin/python scripts/summarize_cqn_no_bc_stage6.py \
  --parallel-run "${PARALLEL}" \
  --autoregressive-run "${AUTOREGRESSIVE}" \
  --output "${BASE}/stage6_summary.json" \
  > "${BASE}/stage6_summary.log" 2>&1
touch "${BASE}/complete"
