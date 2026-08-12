#!/usr/bin/env bash
# Stage-33 scale sentinel: matched deterministic-vs-episodic twin at 101k online.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-5}"
PRIMARY_BASE="${2:-$(tr -d '\n' < exp_local/cqn_no_bc/stage33_latest.txt)}"
if [[ ! -f "${PRIMARY_BASE}/stage32_baseline.txt" ]]; then
  echo "Stage-33 primary run is required: ${PRIMARY_BASE}" >&2
  exit 2
fi

STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_no_bc/stage33_scale_sentinel_seed4_gpu${GPU}_${STAMP}"
BASELINE_LAUNCH="cqn_as_pixel_bigym_nobc_stage32_offline_pessimistic_twin_gate"
TREATMENT_LAUNCH="cqn_as_pixel_bigym_nobc_stage33_episodic_twin_explore_gate"
OFFLINE_UPDATES=10000
ONLINE_FRAMES=101000
GLOBAL_LIMIT=$((OFFLINE_UPDATES + ONLINE_FRAMES))
SEED=4

mkdir -p "${BASE}/baseline/phase_configs" "${BASE}/treatment/phase_configs"
printf "%s\n" "${BASHPID}" > "${BASE}/controller.pid"
printf "%s\n" "${BASE}" > exp_local/cqn_no_bc/stage33_scale_sentinel_latest.txt
printf "%s\n" "${PRIMARY_BASE}" > "${BASE}/stage33_primary.txt"

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
    seed="${SEED}" \
    num_pretrain_steps="${pretrain_steps}" \
    num_train_frames="${train_frames}" \
    demo_batch_size="${demo_batch_size}" \
    replay.demo_only_updates="${demo_only}" \
    method.demo_behavior_force_probability="${force_probability}" \
    snapshot_every_n=10000 \
    gpu_id="${GPU}" \
    xla_mem_fraction=0.35 \
    wandb.use=false \
    hydra.run.dir="${run_dir}" \
    > "${log}" 2>&1
}

run_arm () {
  local label="$1"
  local launch="$2"
  local run_dir="${BASE}/${label}/offline_twin_seed${SEED}"
  local phase_dir="${BASE}/${label}/phase_configs"
  local status=0
  train "${launch}" "${run_dir}" "${OFFLINE_UPDATES}" \
    "${OFFLINE_UPDATES}" 32 true 1.0 \
    "${BASE}/${label}_offline.log" || status=$?
  if [[ "${status}" -ne 0 ]]; then
    printf "%s\n" "${status}" > "${BASE}/${label}_offline_failed"
    return "${status}"
  fi
  cp "${run_dir}/.hydra/config.yaml" "${phase_dir}/offline_seed${SEED}.yaml"
  touch "${BASE}/${label}_offline_complete"

  train "${launch}" "${run_dir}" "${OFFLINE_UPDATES}" \
    "${GLOBAL_LIMIT}" 16 false 0.0 \
    "${BASE}/${label}_online.log" || status=$?
  if [[ "${status}" -ne 0 ]]; then
    printf "%s\n" "${status}" > "${BASE}/${label}_training_failed"
    return "${status}"
  fi
  cp "${run_dir}/.hydra/config.yaml" "${phase_dir}/online_seed${SEED}.yaml"
  touch "${BASE}/${label}_training_complete"
}

# This pair is the experiment: same seed/data/objective and full online budget,
# with only the episode-persistent behavior-head switch in the treatment.
run_arm baseline "${BASELINE_LAUNCH}" &
BASELINE_PID=$!
printf "%s\n" "${BASELINE_PID}" > "${BASE}/baseline.pid"
sleep 120
run_arm treatment "${TREATMENT_LAUNCH}" &
TREATMENT_PID=$!
printf "%s\n" "${TREATMENT_PID}" > "${BASE}/treatment.pid"
STATUS=0
wait "${BASELINE_PID}" || STATUS=$?
TREATMENT_STATUS=0
wait "${TREATMENT_PID}" || TREATMENT_STATUS=$?
if [[ "${STATUS}" -eq 0 ]]; then
  STATUS="${TREATMENT_STATUS}"
fi
if [[ "${STATUS}" -ne 0 ]]; then
  printf "%s\n" "${STATUS}" > "${BASE}/training_failed"
  exit "${STATUS}"
fi
touch "${BASE}/training_complete"

# Freeze the seed-1/2 primary decision before this sentinel is evaluated. The
# sentinel can expose a late-onset scale effect but cannot rewrite that gate.
while [[ ! -f "${PRIMARY_BASE}/stage33_summary.json" ]]; do
  if [[ -f "${PRIMARY_BASE}/training_failed" || -f "${PRIMARY_BASE}/validation_failed" ]]; then
    echo "Stage-33 primary failed before summary" >&2
    exit 3
  fi
  sleep 60
done
cp "${PRIMARY_BASE}/stage33_summary.json" "${BASE}/primary_stage33_frozen.json"
touch "${BASE}/primary_decision_frozen"

evaluate () {
  local label="$1"
  local run_dir="${BASE}/${label}/offline_twin_seed${SEED}"
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MUJOCO_GL=egl \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${run_dir}" \
    --gpu-id "${GPU}" \
    --num-eval-episodes 50 \
    --eval-seed-start 400 \
    --num-eval-envs 25 \
    --only-steps "10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,110000,111000" \
    --csv-name val50_seeds400_scale_raw_steps.csv \
    > "${BASE}/${label}_val50.log" 2>&1
}

evaluate baseline &
BASELINE_EVAL_PID=$!
printf "%s\n" "${BASELINE_EVAL_PID}" > "${BASE}/baseline_val50.pid"
evaluate treatment &
TREATMENT_EVAL_PID=$!
printf "%s\n" "${TREATMENT_EVAL_PID}" > "${BASE}/treatment_val50.pid"
STATUS=0
wait "${BASELINE_EVAL_PID}" || STATUS=$?
TREATMENT_EVAL_STATUS=0
wait "${TREATMENT_EVAL_PID}" || TREATMENT_EVAL_STATUS=$?
if [[ "${STATUS}" -eq 0 ]]; then
  STATUS="${TREATMENT_EVAL_STATUS}"
fi
if [[ "${STATUS}" -ne 0 ]]; then
  printf "%s\n" "${STATUS}" > "${BASE}/validation_failed"
  exit "${STATUS}"
fi
touch "${BASE}/validation_complete"

.venv/bin/python -m scripts.summarize_cqn_no_bc_stage33_scale_sentinel \
  --primary-summary "${BASE}/primary_stage33_frozen.json" \
  --baseline "${BASE}/baseline/offline_twin_seed${SEED}" \
  --treatment "${BASE}/treatment/offline_twin_seed${SEED}" \
  --output "${BASE}/stage33_scale_sentinel_summary.json" \
  > "${BASE}/stage33_scale_sentinel_summary.log" 2>&1
touch "${BASE}/complete"
