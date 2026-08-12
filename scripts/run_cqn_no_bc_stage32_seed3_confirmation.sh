#!/usr/bin/env bash
# Pre-registered seed-3 matched confirmation for Stage 32.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-5}"
PRIMARY_STAGE32="${2:-$(tr -d '\n' < exp_local/cqn_no_bc/stage32_latest.txt)}"
if [[ ! -d "${PRIMARY_STAGE32}" ]]; then
  echo "Primary Stage-32 directory is required: ${PRIMARY_STAGE32}" >&2
  exit 2
fi

STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_no_bc/stage32_seed3_matched_gpu${GPU}_${STAMP}"
OFFLINE_UPDATES=10000
ONLINE_FRAMES=20000
GLOBAL_LIMIT=$((OFFLINE_UPDATES + ONLINE_FRAMES))

mkdir -p "${BASE}/direct/phase_configs" "${BASE}/twin/phase_configs"
printf "%s\n" "${BASHPID}" > "${BASE}/controller.pid"
printf "%s\n" "${BASE}" > exp_local/cqn_no_bc/stage32_seed3_latest.txt
printf "%s\n" "${PRIMARY_STAGE32}" > "${BASE}/primary_stage32.txt"

train () {
  local launch="$1"
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
    launch="${launch}" \
    env=bigym/move_plate \
    seed=3 \
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

run_arm () {
  local label="$1"
  local launch="$2"
  local arm_base="${BASE}/${label}"
  local run_dir="${arm_base}/offline_${label}_seed3"
  local status=0

  train "${launch}" "${run_dir}" "${OFFLINE_UPDATES}" \
    "${OFFLINE_UPDATES}" 32 true 1.0 \
    "${BASE}/${label}_offline.log" || status=$?
  if [[ "${status}" -ne 0 ]]; then
    printf "%s\n" "${status}" > "${BASE}/${label}_offline_failed"
    return "${status}"
  fi
  cp "${run_dir}/.hydra/config.yaml" \
    "${arm_base}/phase_configs/offline_seed3.yaml"
  touch "${BASE}/${label}_offline_complete"

  train "${launch}" "${run_dir}" "${OFFLINE_UPDATES}" \
    "${GLOBAL_LIMIT}" 16 false 0.0 \
    "${BASE}/${label}_online.log" || status=$?
  if [[ "${status}" -ne 0 ]]; then
    printf "%s\n" "${status}" > "${BASE}/${label}_training_failed"
    return "${status}"
  fi
  cp "${run_dir}/.hydra/config.yaml" \
    "${arm_base}/phase_configs/online_seed3.yaml"
  touch "${BASE}/${label}_training_complete"
}

# Same training seed and data protocol; only one critic versus clipped twins.
run_arm direct cqn_as_pixel_bigym_nobc_stage31_offline_direct_head_gate &
DIRECT_PID=$!
printf "%s\n" "${DIRECT_PID}" > "${BASE}/direct.pid"
sleep 120
run_arm twin cqn_as_pixel_bigym_nobc_stage32_offline_pessimistic_twin_gate &
TWIN_PID=$!
printf "%s\n" "${TWIN_PID}" > "${BASE}/twin.pid"

STATUS=0
wait "${DIRECT_PID}" || STATUS=$?
TWIN_STATUS=0
wait "${TWIN_PID}" || TWIN_STATUS=$?
if [[ "${STATUS}" -eq 0 ]]; then
  STATUS="${TWIN_STATUS}"
fi
if [[ "${STATUS}" -ne 0 ]]; then
  printf "%s\n" "${STATUS}" > "${BASE}/training_failed"
  exit "${STATUS}"
fi
touch "${BASE}/training_complete"

# Freeze the primary seed-1/2 decision before opening confirmation rollouts.
while [[ ! -f "${PRIMARY_STAGE32}/stage32_summary.json" ]]; do
  if [[ -f "${PRIMARY_STAGE32}/training_failed" \
     || -f "${PRIMARY_STAGE32}/validation_failed" ]]; then
    echo "Primary Stage 32 failed before producing its decision." >&2
    touch "${BASE}/primary_stage32_failed"
    exit 3
  fi
  sleep 30
done
cp "${PRIMARY_STAGE32}/stage32_summary.json" \
  "${BASE}/primary_stage32_frozen.json"
touch "${BASE}/primary_decision_frozen"

evaluate () {
  local label="$1"
  local run_dir="${BASE}/${label}/offline_${label}_seed3"
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
    > "${BASE}/${label}_val50.log" 2>&1
}

evaluate direct &
DIRECT_EVAL_PID=$!
printf "%s\n" "${DIRECT_EVAL_PID}" > "${BASE}/direct_val50.pid"
evaluate twin &
TWIN_EVAL_PID=$!
printf "%s\n" "${TWIN_EVAL_PID}" > "${BASE}/twin_val50.pid"
STATUS=0
wait "${DIRECT_EVAL_PID}" || STATUS=$?
TWIN_EVAL_STATUS=0
wait "${TWIN_EVAL_PID}" || TWIN_EVAL_STATUS=$?
if [[ "${STATUS}" -eq 0 ]]; then
  STATUS="${TWIN_EVAL_STATUS}"
fi
if [[ "${STATUS}" -ne 0 ]]; then
  printf "%s\n" "${STATUS}" > "${BASE}/validation_failed"
  exit "${STATUS}"
fi
touch "${BASE}/validation_complete"

.venv/bin/python -m scripts.summarize_cqn_no_bc_stage32_seed3 \
  --primary-summary "${BASE}/primary_stage32_frozen.json" \
  --direct "${BASE}/direct/offline_direct_seed3" \
  --twin "${BASE}/twin/offline_twin_seed3" \
  --output "${BASE}/stage32_seed3_summary.json" \
  > "${BASE}/stage32_seed3_summary.log" 2>&1
touch "${BASE}/complete"
