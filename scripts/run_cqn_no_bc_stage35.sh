#!/usr/bin/env bash
# Stage 35: matched 1-step versus normalized 1-step + 4-step twin-C51.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_SEED1="${1:-3}"
GPU_SEED2="${2:-5}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_no_bc/stage35_one_plus_four_fullscale_${STAMP}"
CONTROL_LAUNCH="cqn_as_pixel_bigym_nobc_stage35_one_step_control"
TREATMENT_LAUNCH="cqn_as_pixel_bigym_nobc_stage35_one_plus_four_gate"
OFFLINE_UPDATES=10000
ONLINE_FRAMES=101000
GLOBAL_LIMIT=$((OFFLINE_UPDATES + ONLINE_FRAMES))

for seed in 1 2; do
  for arm in control treatment; do
    mkdir -p "${BASE}/seed${seed}/${arm}/phase_configs"
  done
done
printf "%s\n" "${BASHPID}" > "${BASE}/controller.pid"
printf "%s\n" "${BASE}" > exp_local/cqn_no_bc/stage35_latest.txt
printf "%s\n" "${GPU_SEED1}" > "${BASE}/seed1_gpu.txt"
printf "%s\n" "${GPU_SEED2}" > "${BASE}/seed2_gpu.txt"

train () {
  local gpu="$1"
  local seed="$2"
  local launch="$3"
  local run_dir="$4"
  local pretrain_steps="$5"
  local train_frames="$6"
  local demo_batch_size="$7"
  local demo_only="$8"
  local force_probability="$9"
  local log="${10}"
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MUJOCO_GL=egl \
    .venv/bin/python train_fast.py \
    launch="${launch}" \
    env=bigym/move_plate \
    seed="${seed}" \
    num_pretrain_steps="${pretrain_steps}" \
    num_train_frames="${train_frames}" \
    demo_batch_size="${demo_batch_size}" \
    replay.demo_only_updates="${demo_only}" \
    method.demo_behavior_force_probability="${force_probability}" \
    snapshot_every_n=10000 \
    gpu_id="${gpu}" \
    xla_mem_fraction=0.35 \
    wandb.use=false \
    hydra.run.dir="${run_dir}" \
    > "${log}" 2>&1
}

run_arm () {
  local gpu="$1"
  local seed="$2"
  local arm="$3"
  local launch="$4"
  local arm_dir="${BASE}/seed${seed}/${arm}"
  local run_dir="${arm_dir}/offline_twin_seed${seed}"
  local status=0

  train "${gpu}" "${seed}" "${launch}" "${run_dir}" \
    "${OFFLINE_UPDATES}" "${OFFLINE_UPDATES}" 32 true 1.0 \
    "${arm_dir}/offline.log" || status=$?
  if [[ "${status}" -ne 0 ]]; then
    printf "%s\n" "${status}" > "${arm_dir}/offline_failed"
    return "${status}"
  fi
  cp "${run_dir}/.hydra/config.yaml" \
    "${arm_dir}/phase_configs/offline_seed${seed}.yaml"
  touch "${arm_dir}/offline_complete"

  train "${gpu}" "${seed}" "${launch}" "${run_dir}" \
    "${OFFLINE_UPDATES}" "${GLOBAL_LIMIT}" 16 false 0.0 \
    "${arm_dir}/online.log" || status=$?
  if [[ "${status}" -ne 0 ]]; then
    printf "%s\n" "${status}" > "${arm_dir}/training_failed"
    return "${status}"
  fi
  cp "${run_dir}/.hydra/config.yaml" \
    "${arm_dir}/phase_configs/online_seed${seed}.yaml"
  touch "${arm_dir}/training_complete"
}

run_card () {
  local gpu="$1"
  local seed="$2"
  run_arm "${gpu}" "${seed}" control "${CONTROL_LAUNCH}" &
  local control_pid=$!
  printf "%s\n" "${control_pid}" > "${BASE}/seed${seed}/control.pid"
  sleep 120
  run_arm "${gpu}" "${seed}" treatment "${TREATMENT_LAUNCH}" &
  local treatment_pid=$!
  printf "%s\n" "${treatment_pid}" > "${BASE}/seed${seed}/treatment.pid"
  local status=0
  wait "${control_pid}" || status=$?
  local treatment_status=0
  wait "${treatment_pid}" || treatment_status=$?
  if [[ "${status}" -eq 0 ]]; then
    status="${treatment_status}"
  fi
  return "${status}"
}

run_card "${GPU_SEED1}" 1 &
SEED1_CARD_PID=$!
printf "%s\n" "${SEED1_CARD_PID}" > "${BASE}/seed1_card.pid"
run_card "${GPU_SEED2}" 2 &
SEED2_CARD_PID=$!
printf "%s\n" "${SEED2_CARD_PID}" > "${BASE}/seed2_card.pid"
STATUS=0
wait "${SEED1_CARD_PID}" || STATUS=$?
SEED2_STATUS=0
wait "${SEED2_CARD_PID}" || SEED2_STATUS=$?
if [[ "${STATUS}" -eq 0 ]]; then STATUS="${SEED2_STATUS}"; fi
if [[ "${STATUS}" -ne 0 ]]; then
  printf "%s\n" "${STATUS}" > "${BASE}/training_failed"
  exit "${STATUS}"
fi
touch "${BASE}/training_complete"

evaluate () {
  local gpu="$1"
  local seed="$2"
  local arm="$3"
  local arm_dir="${BASE}/seed${seed}/${arm}"
  local run_dir="${arm_dir}/offline_twin_seed${seed}"
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MUJOCO_GL=egl \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${run_dir}" \
    --gpu-id "${gpu}" \
    --num-eval-episodes 50 \
    --eval-seed-start 400 \
    --num-eval-envs 25 \
    --only-steps "10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,110000,111000" \
    --csv-name val50_seeds400_full_raw_steps.csv \
    > "${arm_dir}/val50.log" 2>&1
}

evaluate "${GPU_SEED1}" 1 control &
EVAL_1C=$!
evaluate "${GPU_SEED1}" 1 treatment &
EVAL_1T=$!
evaluate "${GPU_SEED2}" 2 control &
EVAL_2C=$!
evaluate "${GPU_SEED2}" 2 treatment &
EVAL_2T=$!
printf "%s\n" "${EVAL_1C}" > "${BASE}/seed1/control_val50.pid"
printf "%s\n" "${EVAL_1T}" > "${BASE}/seed1/treatment_val50.pid"
printf "%s\n" "${EVAL_2C}" > "${BASE}/seed2/control_val50.pid"
printf "%s\n" "${EVAL_2T}" > "${BASE}/seed2/treatment_val50.pid"
STATUS=0
for pid in "${EVAL_1C}" "${EVAL_1T}" "${EVAL_2C}" "${EVAL_2T}"; do
  eval_status=0
  wait "${pid}" || eval_status=$?
  if [[ "${STATUS}" -eq 0 ]]; then STATUS="${eval_status}"; fi
done
if [[ "${STATUS}" -ne 0 ]]; then
  printf "%s\n" "${STATUS}" > "${BASE}/validation_failed"
  exit "${STATUS}"
fi
touch "${BASE}/validation_complete"

.venv/bin/python -m scripts.summarize_cqn_no_bc_stage35 \
  --base "${BASE}" \
  --output "${BASE}/stage35_summary.json" \
  > "${BASE}/stage35_summary.log" 2>&1
touch "${BASE}/complete"
