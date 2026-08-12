#!/usr/bin/env bash
# One-GPU Stage-18 replication: dense expected-Q seeds 1 and 2.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-0}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_no_bc/stage18_expected_q_gpu${GPU}_${STAMP}"
SEED1="${BASE}/expected_q_seed1"
SEED2="${BASE}/expected_q_seed2"
BASELINE1="exp_local/cqn_no_bc/stage10_seed1_gpu0_20260731041356/discounted_dense"
BASELINE2="exp_local/cqn_no_bc/stage8_dense_multiseed_gpu0_20260731031717/dense_seed2"
mkdir -p "${BASE}"
printf "%s\n" "${BASHPID}" > "${BASE}/controller.pid"
printf "%s\n" "${BASE}" > exp_local/cqn_no_bc/stage18_latest.txt
printf "%s\n" "${BASELINE1}" > "${BASE}/baseline_seed1.txt"
printf "%s\n" "${BASELINE2}" > "${BASE}/baseline_seed2.txt"

train_one () {
  local seed="$1"
  local run_dir="$2"
  XLA_PYTHON_CLIENT_PREALLOCATE=true \
    MUJOCO_GL=egl \
    .venv/bin/python train_fast.py \
    launch=cqn_as_pixel_bigym_nobc_stage18_expected_q_gate \
    env=bigym/move_plate \
    seed="${seed}" \
    gpu_id="${GPU}" \
    xla_mem_fraction=0.45 \
    wandb.use=false \
    hydra.run.dir="${run_dir}" \
    > "${run_dir}.launch.log" 2>&1
}

train_one 1 "${SEED1}" &
SEED1_PID=$!
printf "%s\n" "${SEED1_PID}" > "${BASE}/expected_q_seed1.pid"
sleep 120
train_one 2 "${SEED2}" &
SEED2_PID=$!
printf "%s\n" "${SEED2_PID}" > "${BASE}/expected_q_seed2.pid"

status=0
wait "${SEED1_PID}" || status=$?
wait "${SEED2_PID}" || status=$?
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

evaluate_one "${SEED1}"
evaluate_one "${SEED2}"
touch "${BASE}/validation_complete"

.venv/bin/python scripts/summarize_cqn_no_bc_stage18.py \
  --baseline-seed1 "${BASELINE1}" \
  --baseline-seed2 "${BASELINE2}" \
  --treatment-seed1 "${SEED1}" \
  --treatment-seed2 "${SEED2}" \
  --output "${BASE}/stage18_summary.json" \
  > "${BASE}/stage18_summary.log" 2>&1
touch "${BASE}/complete"
