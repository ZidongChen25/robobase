#!/usr/bin/env bash
# One-GPU Stage-12 pair: K=8 execution with one-step vs eight-step backup.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-0}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_no_bc/stage12_chunk_horizon_gpu${GPU}_${STAMP}"
CONTROL="${BASE}/k8_nstep1"
TREATMENT="${BASE}/k8_nstep8"
mkdir -p "${BASE}"
printf "%s\n" "${BASHPID}" > "${BASE}/controller.pid"
printf "%s\n" "${BASE}" > exp_local/cqn_no_bc/stage12_latest.txt

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

train_one cqn_as_pixel_bigym_nobc_stage12_k8_nstep1_control \
  "${CONTROL}" &
CONTROL_PID=$!
printf "%s\n" "${CONTROL_PID}" > "${BASE}/k8_nstep1.pid"
sleep 120
train_one cqn_as_pixel_bigym_nobc_stage12_k8_nstep8_gate \
  "${TREATMENT}" &
TREATMENT_PID=$!
printf "%s\n" "${TREATMENT_PID}" > "${BASE}/k8_nstep8.pid"

status=0
wait "${CONTROL_PID}" || status=$?
wait "${TREATMENT_PID}" || status=$?
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

evaluate_one "${CONTROL}"
evaluate_one "${TREATMENT}"
touch "${BASE}/validation_complete"

.venv/bin/python scripts/summarize_cqn_no_bc_stage12.py \
  --control-run "${CONTROL}" \
  --treatment-run "${TREATMENT}" \
  --output "${BASE}/stage12_summary.json" \
  > "${BASE}/stage12_summary.log" 2>&1
touch "${BASE}/complete"
