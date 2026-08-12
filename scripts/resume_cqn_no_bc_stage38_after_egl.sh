#!/usr/bin/env bash
# Resume Stage 38 from its completed 10k offline checkpoints after a transient
# EGL initialization failure.  This never reruns offline updates or increases
# the registered raw-20k development budget.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="${1:-$(tr -d '\n' < exp_local/cqn_no_bc/stage38_latest.txt)}"
GPU="${2:-$(tr -d '\n' < "${BASE}/gpu.txt")}"
BASELINE_BASE="$(tr -d '\n' < "${BASE}/baseline_stage36.txt")"
LAUNCH="cqn_as_pixel_bigym_stage38_offline_nobc_dense256_gate"
OFFLINE_UPDATES=10000
INITIAL_GLOBAL_LIMIT=20000
EGL_RETRY_SECONDS="${STAGE38_EGL_RETRY_SECONDS:-60}"

run_dir () {
  printf '%s/dense_seed%s/offline_then_online' "${BASE}" "$1"
}

for seed in 1 2; do
  test -s "$(run_dir "${seed}")/snapshots/10000_snapshot.pkl"
done
test -e "${BASE}/offline_complete"
if [[ -e "${BASE}/complete" ]]; then
  printf 'Stage 38 is already complete: %s\n' "${BASE}"
  exit 0
fi

printf '%s\n' "${BASHPID}" > "${BASE}/online_recovery_controller.pid"
touch "${BASE}/online_recovery_started"

probe_egl () {
  timeout 30s env \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
    .venv/bin/python -c \
    "import mujoco; m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><geom size=\"0.1\"/></body></worldbody></mujoco>'); r=mujoco.Renderer(m,84,84); r.render(); r.close()" \
    > "${BASE}/egl_probe_latest.log" 2>&1
}

wait_for_egl () {
  local label="$1"
  local attempt=0
  while ! probe_egl; do
    attempt=$((attempt + 1))
    printf '%s label=%s attempt=%s status=unavailable\n' \
      "$(date --iso-8601=seconds)" "${label}" "${attempt}" \
      >> "${BASE}/egl_probe_history.log"
    touch "${BASE}/egl_waiting"
    sleep "${EGL_RETRY_SECONDS}"
  done
  printf '%s label=%s attempt=%s status=available\n' \
    "$(date --iso-8601=seconds)" "${label}" "$((attempt + 1))" \
    >> "${BASE}/egl_probe_history.log"
  touch "${BASE}/egl_recovered"
}

train_online_with_retry () {
  local seed="$1"
  local attempt=0
  local dir
  dir="$(run_dir "${seed}")"
  while true; do
    attempt=$((attempt + 1))
    local log="${BASE}/dense_seed${seed}/online_recovery_attempt${attempt}.log"
    local status=0
    if XLA_PYTHON_CLIENT_PREALLOCATE=false \
      MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
      CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
      .venv/bin/python train_fast.py \
      launch="${LAUNCH}" env=bigym/move_plate seed="${seed}" \
      batch_size=256 demo_batch_size=256 \
      num_pretrain_steps="${OFFLINE_UPDATES}" \
      num_train_frames="${INITIAL_GLOBAL_LIMIT}" \
      replay.demo_only_updates=false \
      method.demo_behavior_force_probability=0.0 \
      eval_every_steps=1000000 num_eval_episodes=0 num_eval_envs=0 \
      snapshot_every_n=2500 gpu_id="${GPU}" xla_mem_fraction=0.45 \
      wandb.use=false hydra.run.dir="${dir}" > "${log}" 2>&1; then
      return 0
    else
      status=$?
    fi
    if rg -q "Cannot initialize a EGL device display" "${log}"; then
      printf '%s seed=%s attempt=%s status=egl_retry\n' \
        "$(date --iso-8601=seconds)" "${seed}" "${attempt}" \
        >> "${BASE}/online_recovery_history.log"
      wait_for_egl "online_seed${seed}"
      continue
    fi
    printf '%s seed=%s attempt=%s status=fatal exit=%s\n' \
      "$(date --iso-8601=seconds)" "${seed}" "${attempt}" "${status}" \
      >> "${BASE}/online_recovery_history.log"
    return "${status}"
  done
}

run_online_pair () {
  local status=0
  train_online_with_retry 1 &
  local seed1_pid=$!
  printf '%s\n' "${seed1_pid}" > "${BASE}/dense_seed1/online_recovery.pid"
  sleep 120
  train_online_with_retry 2 &
  local seed2_pid=$!
  printf '%s\n' "${seed2_pid}" > "${BASE}/dense_seed2/online_recovery.pid"
  wait "${seed1_pid}" || status=$?
  local seed2_status=0
  wait "${seed2_pid}" || seed2_status=$?
  if [[ "${status}" -eq 0 ]]; then status="${seed2_status}"; fi
  return "${status}"
}

wait_for_egl initial_online_recovery
status=0
run_online_pair || status=$?
if [[ "${status}" -ne 0 ]]; then
  printf '%s\n' "${status}" > "${BASE}/online_recovery_failed"
  exit "${status}"
fi
for seed in 1 2; do
  dir="$(run_dir "${seed}")"
  test -s "${dir}/snapshots/20000_snapshot.pkl"
  cp "${dir}/.hydra/config.yaml" \
    "${BASE}/dense_seed${seed}/phase_configs/online_recovered.yaml"
done
touch "${BASE}/initial_training_complete"
touch "${BASE}/online_recovery_training_complete"

evaluate_with_retry () {
  local dir="$1"
  local episodes="$2"
  local envs="$3"
  local csv_name="$4"
  local log_base="$5"
  local attempt=0
  while true; do
    attempt=$((attempt + 1))
    local log="${log_base%.log}_attempt${attempt}.log"
    local status=0
    if XLA_PYTHON_CLIENT_PREALLOCATE=false \
      MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
      CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
      .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
      --run-dir "${dir}" --gpu-id "${GPU}" \
      --num-eval-episodes "${episodes}" --eval-seed-start 400 \
      --num-eval-envs "${envs}" \
      --only-steps "10000,12500,15000,17500,20000" \
      --csv-name "${csv_name}" > "${log}" 2>&1; then
      return 0
    else
      status=$?
    fi
    if rg -q "Cannot initialize a EGL device display" "${log}"; then
      wait_for_egl "evaluation"
      continue
    fi
    return "${status}"
  done
}

# Release the training contexts before opening the fixed evaluation wave.
sleep 60
wait_for_egl coarse_validation
evaluate_with_retry "$(run_dir 1)" 20 20 \
  val20_seeds400_coarse.csv "${BASE}/dense_seed1/coarse_val20_recovery.log" &
eval1=$!
evaluate_with_retry "$(run_dir 2)" 20 20 \
  val20_seeds400_coarse.csv "${BASE}/dense_seed2/coarse_val20_recovery.log" &
eval2=$!
status=0
wait "${eval1}" || status=$?
eval2_status=0; wait "${eval2}" || eval2_status=$?
if [[ "${status}" -eq 0 ]]; then status="${eval2_status}"; fi
if [[ "${status}" -ne 0 ]]; then
  printf '%s\n' "${status}" > "${BASE}/coarse_validation_failed"
  exit "${status}"
fi

.venv/bin/python -m scripts.summarize_cqn_no_bc_stage38 \
  --base "${BASE}" --baseline-base "${BASELINE_BASE}" \
  --output "${BASE}/stage38_coarse_summary.json" \
  > "${BASE}/stage38_coarse_summary.log" 2>&1
touch "${BASE}/coarse_validation_complete"
if ! rg -q '"coarse_qualification_pass": true' \
    "${BASE}/stage38_coarse_summary.json"; then
  touch "${BASE}/early_futility_stop"
  touch "${BASE}/complete"
  exit 0
fi

wait_for_egl candidate_selection
evaluate_with_retry "$(run_dir 1)" 50 25 \
  val50_seeds400_selection.csv "${BASE}/dense_seed1/selection_val50_recovery.log" &
eval1=$!
evaluate_with_retry "$(run_dir 2)" 50 25 \
  val50_seeds400_selection.csv "${BASE}/dense_seed2/selection_val50_recovery.log" &
eval2=$!
status=0
wait "${eval1}" || status=$?
eval2_status=0; wait "${eval2}" || eval2_status=$?
if [[ "${status}" -eq 0 ]]; then status="${eval2_status}"; fi
if [[ "${status}" -ne 0 ]]; then
  printf '%s\n' "${status}" > "${BASE}/candidate_validation_failed"
  exit "${status}"
fi

BASELINE_NOBC="${BASELINE_BASE}/treatment/offline_then_online_seed1"
BASELINE_BC="${BASELINE_BASE}/control/offline_then_online_seed1"
for dir in "${BASELINE_NOBC}" "${BASELINE_BC}"; do
  test -s "${dir}/snapshots/20000_snapshot.pkl"
done
wait_for_egl baseline_selection
evaluate_with_retry "${BASELINE_NOBC}" 50 25 \
  val50_seeds400_stage38.csv "${BASE}/stage36_nobc_selection_val50_recovery.log" &
eval1=$!
evaluate_with_retry "${BASELINE_BC}" 50 25 \
  val50_seeds400_stage38.csv "${BASE}/stage36_bc_selection_val50_recovery.log" &
eval2=$!
status=0
wait "${eval1}" || status=$?
eval2_status=0; wait "${eval2}" || eval2_status=$?
if [[ "${status}" -eq 0 ]]; then status="${eval2_status}"; fi
if [[ "${status}" -ne 0 ]]; then
  printf '%s\n' "${status}" > "${BASE}/baseline_validation_failed"
  exit "${status}"
fi

.venv/bin/python -m scripts.summarize_cqn_no_bc_stage38 \
  --base "${BASE}" --baseline-base "${BASELINE_BASE}" \
  --output "${BASE}/stage38_summary.json" \
  > "${BASE}/stage38_summary.log" 2>&1
touch "${BASE}/matched_validation_complete"
touch "${BASE}/complete"
