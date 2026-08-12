#!/usr/bin/env bash
# Stage 32: clipped twin direct-C51 under the offline->online protocol.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-1}"
STAGE31_BASE="${2:-$(tr -d '\n' < exp_local/cqn_no_bc/stage31_latest.txt)}"
if [[ ! -f "${STAGE31_BASE}/stage31_summary.json" ]]; then
  echo "Stage-31 summary is required: ${STAGE31_BASE}/stage31_summary.json" >&2
  exit 2
fi

STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_no_bc/stage32_offline_pessimistic_twin_gpu${GPU}_${STAMP}"
LAUNCH="cqn_as_pixel_bigym_nobc_stage32_offline_pessimistic_twin_gate"
OFFLINE_UPDATES=10000
ONLINE_FRAMES=20000
GLOBAL_LIMIT=$((OFFLINE_UPDATES + ONLINE_FRAMES))

mkdir -p "${BASE}/phase_configs"
printf "%s\n" "${BASHPID}" > "${BASE}/controller.pid"
printf "%s\n" "${BASE}" > exp_local/cqn_no_bc/stage32_latest.txt
printf "%s\n" "${STAGE31_BASE}" > "${BASE}/stage31_baseline.txt"

train () {
  local seed="$1"
  local run_dir="$2"
  local pretrain_steps="$3"
  local train_frames="$4"
  local demo_batch_size="$5"
  local demo_only="$6"
  local force_probability="$7"
  local log="$8"
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MUJOCO_GL=egl \
    .venv/bin/python train_fast.py \
    launch="${LAUNCH}" \
    env=bigym/move_plate \
    seed="${seed}" \
    num_pretrain_steps="${pretrain_steps}" \
    num_train_frames="${train_frames}" \
    demo_batch_size="${demo_batch_size}" \
    replay.demo_only_updates="${demo_only}" \
    method.demo_behavior_force_probability="${force_probability}" \
    gpu_id="${GPU}" \
    xla_mem_fraction=0.35 \
    wandb.use=false \
    hydra.run.dir="${run_dir}" \
    > "${log}" 2>&1
}

run_seed () {
  local seed="$1"
  local run_dir="${BASE}/offline_pessimistic_twin_seed${seed}"
  local status=0
  train "${seed}" "${run_dir}" "${OFFLINE_UPDATES}" \
    "${OFFLINE_UPDATES}" 32 true 1.0 \
    "${BASE}/offline_seed${seed}.log" || status=$?
  if [[ "${status}" -ne 0 ]]; then
    printf "%s\n" "${status}" > "${BASE}/offline_seed${seed}_failed"
    return "${status}"
  fi
  cp "${run_dir}/.hydra/config.yaml" \
    "${BASE}/phase_configs/offline_seed${seed}.yaml"
  touch "${BASE}/offline_seed${seed}_complete"

  train "${seed}" "${run_dir}" "${OFFLINE_UPDATES}" \
    "${GLOBAL_LIMIT}" 16 false 0.0 \
    "${BASE}/online_seed${seed}.log" || status=$?
  if [[ "${status}" -ne 0 ]]; then
    printf "%s\n" "${status}" > "${BASE}/training_seed${seed}_failed"
    return "${status}"
  fi
  cp "${run_dir}/.hydra/config.yaml" \
    "${BASE}/phase_configs/online_seed${seed}.yaml"
  touch "${BASE}/training_seed${seed}_complete"
}

run_seed 1 &
SEED1_PID=$!
printf "%s\n" "${SEED1_PID}" > "${BASE}/seed1.pid"
sleep 120
run_seed 2 &
SEED2_PID=$!
printf "%s\n" "${SEED2_PID}" > "${BASE}/seed2.pid"
STATUS=0
wait "${SEED1_PID}" || STATUS=$?
SEED2_STATUS=0
wait "${SEED2_PID}" || SEED2_STATUS=$?
if [[ "${STATUS}" -eq 0 ]]; then
  STATUS="${SEED2_STATUS}"
fi
if [[ "${STATUS}" -ne 0 ]]; then
  printf "%s\n" "${STATUS}" > "${BASE}/training_failed"
  exit "${STATUS}"
fi
touch "${BASE}/training_complete"

evaluate () {
  local seed="$1"
  local run_dir="${BASE}/offline_pessimistic_twin_seed${seed}"
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MUJOCO_GL=egl \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${run_dir}" \
    --gpu-id "${GPU}" \
    --num-eval-episodes 50 \
    --eval-seed-start 400 \
    --num-eval-envs 25 \
    --only-steps "10000,12500,15000,17500,20000,22500,25000,27500,30000" \
    --csv-name val50_seeds400_raw_steps.csv \
    > "${BASE}/seed${seed}_val50.log" 2>&1
}

evaluate 1 &
EVAL1_PID=$!
printf "%s\n" "${EVAL1_PID}" > "${BASE}/seed1_val50.pid"
evaluate 2 &
EVAL2_PID=$!
printf "%s\n" "${EVAL2_PID}" > "${BASE}/seed2_val50.pid"
STATUS=0
wait "${EVAL1_PID}" || STATUS=$?
EVAL2_STATUS=0
wait "${EVAL2_PID}" || EVAL2_STATUS=$?
if [[ "${STATUS}" -eq 0 ]]; then
  STATUS="${EVAL2_STATUS}"
fi
if [[ "${STATUS}" -ne 0 ]]; then
  printf "%s\n" "${STATUS}" > "${BASE}/validation_failed"
  exit "${STATUS}"
fi
touch "${BASE}/validation_complete"

.venv/bin/python -m scripts.summarize_cqn_no_bc_stage32 \
  --stage31-summary "${STAGE31_BASE}/stage31_summary.json" \
  --twin-seed1 "${BASE}/offline_pessimistic_twin_seed1" \
  --twin-seed2 "${BASE}/offline_pessimistic_twin_seed2" \
  --output "${BASE}/stage32_summary.json" \
  > "${BASE}/stage32_summary.log" 2>&1
touch "${BASE}/complete"
