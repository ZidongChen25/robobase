#!/usr/bin/env bash
# One-GPU Stage-8 replication: dense-return training seeds 2 and 3.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-0}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_no_bc/stage8_dense_multiseed_gpu${GPU}_${STAMP}"
SEED2="${BASE}/dense_seed2"
SEED3="${BASE}/dense_seed3"
STAGE7="$(cat exp_local/cqn_no_bc/stage7_latest.txt)"
SEED1="${STAGE7}/mc_dense_return"
mkdir -p "${BASE}"
printf "%s\n" "${BASHPID}" > "${BASE}/controller.pid"
printf "%s\n" "${BASE}" > exp_local/cqn_no_bc/stage8_latest.txt
printf "%s\n" "${SEED1}" > "${BASE}/seed1_run.txt"

train_one () {
  local seed="$1"
  local run_dir="$2"
  XLA_PYTHON_CLIENT_PREALLOCATE=true \
    MUJOCO_GL=egl \
    .venv/bin/python train_fast.py \
    launch=cqn_as_pixel_bigym_nobc_mc_stage7_dense_return_gate \
    env=bigym/move_plate \
    seed="${seed}" \
    gpu_id="${GPU}" \
    xla_mem_fraction=0.45 \
    wandb.use=false \
    hydra.run.dir="${run_dir}" \
    > "${run_dir}.launch.log" 2>&1
}

train_one 2 "${SEED2}" &
SEED2_PID=$!
printf "%s\n" "${SEED2_PID}" > "${BASE}/dense_seed2.pid"
sleep 120
train_one 3 "${SEED3}" &
SEED3_PID=$!
printf "%s\n" "${SEED3_PID}" > "${BASE}/dense_seed3.pid"

status=0
wait "${SEED2_PID}" || status=$?
wait "${SEED3_PID}" || status=$?
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

evaluate_one "${SEED2}"
evaluate_one "${SEED3}"
touch "${BASE}/validation_complete"

.venv/bin/python scripts/summarize_cqn_no_bc_stage8.py \
  --seed1-run "${SEED1}" \
  --seed2-run "${SEED2}" \
  --seed3-run "${SEED3}" \
  --baseline-summary \
  exp_local/cqn_value_fidelity_stage22/clean_multiseed_summary.json \
  --output "${BASE}/stage8_summary.json" \
  > "${BASE}/stage8_summary.log" 2>&1
touch "${BASE}/complete"
