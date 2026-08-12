#!/usr/bin/env bash
# Stage 27: matched 10k-to-20k continuation of the late-rising Stage-19 arm.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-1}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_no_bc/stage27_reward_scale20k_gpu${GPU}_${STAMP}"
STAGE19="exp_local/cqn_no_bc/stage19_reward_scale_gpu0_20260731085818"
TREATMENT1="${STAGE19}/reward_scale_seed1"
TREATMENT2="${STAGE19}/reward_scale_seed2"
BASELINE1="exp_local/cqn_no_bc/stage10_seed1_gpu0_20260731041356/discounted_dense"
BASELINE2="exp_local/cqn_no_bc/stage8_dense_multiseed_gpu0_20260731031717/dense_seed2"

mkdir -p "${BASE}/config_10k"
printf "%s\n" "${BASHPID}" > "${BASE}/controller.pid"
printf "%s\n" "${BASE}" > exp_local/cqn_no_bc/stage27_latest.txt
printf "%s\n" "${TREATMENT1}" > "${BASE}/treatment_seed1.txt"
printf "%s\n" "${TREATMENT2}" > "${BASE}/treatment_seed2.txt"
printf "%s\n" "${BASELINE1}" > "${BASE}/baseline_seed1.txt"
printf "%s\n" "${BASELINE2}" > "${BASE}/baseline_seed2.txt"
cp "${TREATMENT1}/.hydra/config.yaml" "${BASE}/config_10k/seed1.yaml"
cp "${TREATMENT2}/.hydra/config.yaml" "${BASE}/config_10k/seed2.yaml"

train_one () {
  local seed="$1"
  local run_dir="$2"
  local log="$3"
  XLA_PYTHON_CLIENT_PREALLOCATE=true \
    MUJOCO_GL=egl \
    .venv/bin/python train_fast.py \
    launch=cqn_as_pixel_bigym_nobc_stage19_reward_scale_gate \
    env=bigym/move_plate \
    seed="${seed}" \
    num_train_frames=20000 \
    gpu_id="${GPU}" \
    xla_mem_fraction=0.45 \
    wandb.use=false \
    hydra.run.dir="${run_dir}" \
    > "${log}" 2>&1
}

train_one 1 "${TREATMENT1}" "${BASE}/seed1_resume20k.log" &
SEED1_PID=$!
printf "%s\n" "${SEED1_PID}" > "${BASE}/seed1.pid"
sleep 120
train_one 2 "${TREATMENT2}" "${BASE}/seed2_resume20k.log" &
SEED2_PID=$!
printf "%s\n" "${SEED2_PID}" > "${BASE}/seed2.pid"

status=0
wait "${SEED1_PID}" || status=$?
wait "${SEED2_PID}" || status=$?
if [[ "${status}" -ne 0 ]]; then
  printf "%s\n" "${status}" > "${BASE}/training_failed"
  exit "${status}"
fi
touch "${BASE}/training_complete"

evaluate_one () {
  local run_dir="$1"
  local log="$2"
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MUJOCO_GL=egl \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${run_dir}" \
    --gpu-id "${GPU}" \
    --num-eval-episodes 50 \
    --eval-seed-start 400 \
    --num-eval-envs 25 \
    --csv-name val50_ext20k_seeds400.csv \
    --skip-steps 2500,5000,7500,10000,10500 \
    > "${log}" 2>&1
}

evaluate_one "${TREATMENT1}" "${BASE}/seed1_val_ext20k.log" &
EVAL1_PID=$!
evaluate_one "${TREATMENT2}" "${BASE}/seed2_val_ext20k.log" &
EVAL2_PID=$!
status=0
wait "${EVAL1_PID}" || status=$?
wait "${EVAL2_PID}" || status=$?
if [[ "${status}" -ne 0 ]]; then
  printf "%s\n" "${status}" > "${BASE}/validation_failed"
  exit "${status}"
fi
touch "${BASE}/validation_complete"

.venv/bin/python -m scripts.summarize_cqn_no_bc_stage27 \
  --baseline-seed1 "${BASELINE1}" \
  --baseline-seed2 "${BASELINE2}" \
  --treatment-seed1 "${TREATMENT1}" \
  --treatment-seed2 "${TREATMENT2}" \
  --output "${BASE}/stage27_summary.json" \
  > "${BASE}/stage27_summary.log" 2>&1
touch "${BASE}/complete"
