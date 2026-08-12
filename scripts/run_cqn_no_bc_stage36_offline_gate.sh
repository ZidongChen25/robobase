#!/usr/bin/env bash
# Stage 36: baseline-matched batch-256 offline-to-online progressive gate.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-4}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_no_bc/stage36_offline_gate_gpu${GPU}_${STAMP}"
CONTROL_LAUNCH="cqn_as_pixel_bigym_stage36_offline_bc256_control"
TREATMENT_LAUNCH="cqn_as_pixel_bigym_stage36_offline_nobc_candidate256_gate"
CONTROL_DIR="${BASE}/control/offline_then_online_seed1"
TREATMENT_DIR="${BASE}/treatment/offline_then_online_seed1"
OFFLINE_UPDATES=10000
INITIAL_ONLINE_STEPS=10000
INITIAL_GLOBAL_LIMIT=$((OFFLINE_UPDATES + INITIAL_ONLINE_STEPS))

mkdir -p "${BASE}/control/phase_configs" "${BASE}/treatment/phase_configs"
printf '%s\n' "${BASHPID}" > "${BASE}/controller.pid"
printf '%s\n' "${BASE}" > exp_local/cqn_no_bc/stage36_offline_gate_latest.txt
printf '%s\n' "${GPU}" > "${BASE}/gpu.txt"

train_phase () {
  local arm="$1"
  local launch="$2"
  local run_dir="$3"
  local demo_only="$4"
  local force_probability="$5"
  local train_frames="$6"
  local phase="$7"
  local log="${BASE}/${arm}/${phase}.log"
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MUJOCO_GL=egl \
    .venv/bin/python train_fast.py \
    launch="${launch}" env=bigym/move_plate seed=1 \
    batch_size=256 demo_batch_size=256 \
    num_pretrain_steps="${OFFLINE_UPDATES}" \
    num_train_frames="${train_frames}" \
    replay.demo_only_updates="${demo_only}" \
    method.demo_behavior_force_probability="${force_probability}" \
    eval_every_steps=1000000 num_eval_episodes=0 num_eval_envs=0 \
    snapshot_every_n=2500 gpu_id="${GPU}" xla_mem_fraction=0.45 \
    wandb.use=false hydra.run.dir="${run_dir}" > "${log}" 2>&1
}

run_training_pair () {
  local phase="$1"
  local demo_only="$2"
  local train_frames="$3"
  local control_force="$4"
  local treatment_force="$5"
  local status=0
  train_phase control "${CONTROL_LAUNCH}" "${CONTROL_DIR}" \
    "${demo_only}" "${control_force}" "${train_frames}" "${phase}" &
  local control_pid=$!
  printf '%s\n' "${control_pid}" > "${BASE}/control/${phase}.pid"
  sleep 120
  train_phase treatment "${TREATMENT_LAUNCH}" "${TREATMENT_DIR}" \
    "${demo_only}" "${treatment_force}" "${train_frames}" "${phase}" &
  local treatment_pid=$!
  printf '%s\n' "${treatment_pid}" > "${BASE}/treatment/${phase}.pid"
  wait "${control_pid}" || status=$?
  local treatment_status=0
  wait "${treatment_pid}" || treatment_status=$?
  if [[ "${status}" -eq 0 ]]; then status="${treatment_status}"; fi
  return "${status}"
}

status=0
run_training_pair offline true "${OFFLINE_UPDATES}" 0.0 1.0 || status=$?
if [[ "${status}" -ne 0 ]]; then
  printf '%s\n' "${status}" > "${BASE}/offline_failed"
  exit "${status}"
fi
test -s "${CONTROL_DIR}/snapshots/10000_snapshot.pkl"
test -s "${TREATMENT_DIR}/snapshots/10000_snapshot.pkl"
cp "${CONTROL_DIR}/.hydra/config.yaml" \
  "${BASE}/control/phase_configs/offline.yaml"
cp "${TREATMENT_DIR}/.hydra/config.yaml" \
  "${BASE}/treatment/phase_configs/offline.yaml"
touch "${BASE}/offline_complete"

status=0
run_training_pair online false "${INITIAL_GLOBAL_LIMIT}" 0.0 0.0 || status=$?
if [[ "${status}" -ne 0 ]]; then
  printf '%s\n' "${status}" > "${BASE}/training_failed"
  exit "${status}"
fi
test -s "${CONTROL_DIR}/snapshots/20000_snapshot.pkl"
test -s "${TREATMENT_DIR}/snapshots/20000_snapshot.pkl"
cp "${CONTROL_DIR}/.hydra/config.yaml" \
  "${BASE}/control/phase_configs/online.yaml"
cp "${TREATMENT_DIR}/.hydra/config.yaml" \
  "${BASE}/treatment/phase_configs/online.yaml"
touch "${BASE}/initial_training_complete"

evaluate () {
  local arm="$1"
  local run_dir="$2"
  local episodes="$3"
  local envs="$4"
  local csv_name="$5"
  local log="$6"
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MUJOCO_GL=egl \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${run_dir}" --gpu-id "${GPU}" \
    --num-eval-episodes "${episodes}" --eval-seed-start 400 \
    --num-eval-envs "${envs}" \
    --only-steps "10000,12500,15000,17500,20000" \
    --csv-name "${csv_name}" > "${BASE}/${arm}/${log}" 2>&1
}

# Futility screen: treatment only.  No control eval or larger budget is spent
# if every point is exactly zero.
evaluate treatment "${TREATMENT_DIR}" 20 20 \
  val20_seeds400_coarse.csv coarse_val20.log
.venv/bin/python -m scripts.summarize_cqn_no_bc_stage36_offline_gate \
  --base "${BASE}" --output "${BASE}/stage36_coarse_summary.json" \
  > "${BASE}/stage36_coarse_summary.log" 2>&1
touch "${BASE}/coarse_validation_complete"

if ! rg -q '"coarse_nonzero": true' "${BASE}/stage36_coarse_summary.json"; then
  touch "${BASE}/early_futility_stop"
  touch "${BASE}/complete"
  exit 0
fi

# A non-zero coarse point earns the matched selection curve, but never an
# automatic training extension.
evaluate control "${CONTROL_DIR}" 50 25 \
  val50_seeds400_selection.csv selection_val50.log &
control_eval_pid=$!
printf '%s\n' "${control_eval_pid}" > "${BASE}/control/selection_val50.pid"
evaluate treatment "${TREATMENT_DIR}" 50 25 \
  val50_seeds400_selection.csv selection_val50.log &
treatment_eval_pid=$!
printf '%s\n' "${treatment_eval_pid}" > "${BASE}/treatment/selection_val50.pid"
status=0
wait "${control_eval_pid}" || status=$?
treatment_status=0
wait "${treatment_eval_pid}" || treatment_status=$?
if [[ "${status}" -eq 0 ]]; then status="${treatment_status}"; fi
if [[ "${status}" -ne 0 ]]; then
  printf '%s\n' "${status}" > "${BASE}/validation_failed"
  exit "${status}"
fi

.venv/bin/python -m scripts.summarize_cqn_no_bc_stage36_offline_gate \
  --base "${BASE}" --output "${BASE}/stage36_summary.json" \
  > "${BASE}/stage36_summary.log" 2>&1
touch "${BASE}/matched_validation_complete"
touch "${BASE}/complete"
