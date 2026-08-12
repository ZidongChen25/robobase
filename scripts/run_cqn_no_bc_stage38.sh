#!/usr/bin/env bash
# Stage 38: two-seed baseline-matched offline dense reward-Q progressive gate.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-5}"
BASELINE_BASE="${2:-$(tr -d '\n' < exp_local/cqn_no_bc/stage36_offline_gate_latest.txt)}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_no_bc/stage38_offline_dense_b256_gpu${GPU}_${STAMP}"
LAUNCH="cqn_as_pixel_bigym_stage38_offline_nobc_dense256_gate"
OFFLINE_UPDATES=10000
INITIAL_ONLINE_STEPS=10000
INITIAL_GLOBAL_LIMIT=$((OFFLINE_UPDATES + INITIAL_ONLINE_STEPS))

for seed in 1 2; do
  mkdir -p "${BASE}/dense_seed${seed}/phase_configs"
done
printf '%s\n' "${BASHPID}" > "${BASE}/controller.pid"
printf '%s\n' "${BASE}" > exp_local/cqn_no_bc/stage38_latest.txt
printf '%s\n' "${GPU}" > "${BASE}/gpu.txt"
printf '%s\n' "${BASELINE_BASE}" > "${BASE}/baseline_stage36.txt"

run_dir () {
  printf '%s/dense_seed%s/offline_then_online' "${BASE}" "$1"
}

train_phase () {
  local seed="$1"
  local demo_only="$2"
  local force_probability="$3"
  local train_frames="$4"
  local phase="$5"
  local dir
  dir="$(run_dir "${seed}")"
  XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
    .venv/bin/python train_fast.py \
    launch="${LAUNCH}" env=bigym/move_plate seed="${seed}" \
    batch_size=256 demo_batch_size=256 \
    num_pretrain_steps="${OFFLINE_UPDATES}" num_train_frames="${train_frames}" \
    replay.demo_only_updates="${demo_only}" \
    method.demo_behavior_force_probability="${force_probability}" \
    eval_every_steps=1000000 num_eval_episodes=0 num_eval_envs=0 \
    snapshot_every_n=2500 gpu_id="${GPU}" xla_mem_fraction=0.45 \
    wandb.use=false hydra.run.dir="${dir}" \
    > "${BASE}/dense_seed${seed}/${phase}.log" 2>&1
}

run_pair () {
  local phase="$1"
  local demo_only="$2"
  local force_probability="$3"
  local train_frames="$4"
  local status=0
  train_phase 1 "${demo_only}" "${force_probability}" "${train_frames}" \
    "${phase}" &
  local seed1_pid=$!
  printf '%s\n' "${seed1_pid}" > "${BASE}/dense_seed1/${phase}.pid"
  sleep 120
  train_phase 2 "${demo_only}" "${force_probability}" "${train_frames}" \
    "${phase}" &
  local seed2_pid=$!
  printf '%s\n' "${seed2_pid}" > "${BASE}/dense_seed2/${phase}.pid"
  wait "${seed1_pid}" || status=$?
  local seed2_status=0
  wait "${seed2_pid}" || seed2_status=$?
  if [[ "${status}" -eq 0 ]]; then status="${seed2_status}"; fi
  return "${status}"
}

status=0
run_pair offline true 1.0 "${OFFLINE_UPDATES}" || status=$?
if [[ "${status}" -ne 0 ]]; then
  printf '%s\n' "${status}" > "${BASE}/offline_failed"
  exit "${status}"
fi
for seed in 1 2; do
  dir="$(run_dir "${seed}")"
  test -s "${dir}/snapshots/10000_snapshot.pkl"
  cp "${dir}/.hydra/config.yaml" \
    "${BASE}/dense_seed${seed}/phase_configs/offline.yaml"
done
touch "${BASE}/offline_complete"

status=0
run_pair online false 0.0 "${INITIAL_GLOBAL_LIMIT}" || status=$?
if [[ "${status}" -ne 0 ]]; then
  printf '%s\n' "${status}" > "${BASE}/training_failed"
  exit "${status}"
fi
for seed in 1 2; do
  dir="$(run_dir "${seed}")"
  test -s "${dir}/snapshots/20000_snapshot.pkl"
  cp "${dir}/.hydra/config.yaml" \
    "${BASE}/dense_seed${seed}/phase_configs/online.yaml"
done
touch "${BASE}/initial_training_complete"

evaluate () {
  local dir="$1"
  local episodes="$2"
  local envs="$3"
  local csv_name="$4"
  local log="$5"
  XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${dir}" --gpu-id "${GPU}" \
    --num-eval-episodes "${episodes}" --eval-seed-start 400 \
    --num-eval-envs "${envs}" \
    --only-steps "10000,12500,15000,17500,20000" \
    --csv-name "${csv_name}" > "${log}" 2>&1
}

# Both training seeds must show a non-zero curve and at least one must reach
# 20% late before any 50-episode selection sweep is allowed.
evaluate "$(run_dir 1)" 20 20 val20_seeds400_coarse.csv \
  "${BASE}/dense_seed1/coarse_val20.log" &
eval1=$!
evaluate "$(run_dir 2)" 20 20 val20_seeds400_coarse.csv \
  "${BASE}/dense_seed2/coarse_val20.log" &
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

# Wave 1: the two dense treatments.  Training has fully released the card.
evaluate "$(run_dir 1)" 50 25 val50_seeds400_selection.csv \
  "${BASE}/dense_seed1/selection_val50.log" &
eval1=$!
evaluate "$(run_dir 2)" 50 25 val50_seeds400_selection.csv \
  "${BASE}/dense_seed2/selection_val50.log" &
eval2=$!
status=0
wait "${eval1}" || status=$?
eval2_status=0; wait "${eval2}" || eval2_status=$?
if [[ "${status}" -eq 0 ]]; then status="${eval2_status}"; fi
if [[ "${status}" -ne 0 ]]; then
  printf '%s\n' "${status}" > "${BASE}/candidate_validation_failed"
  exit "${status}"
fi

# Wave 2: the already-trained exact Stage-36 No-BC control and BC reference.
BASELINE_NOBC="${BASELINE_BASE}/treatment/offline_then_online_seed1"
BASELINE_BC="${BASELINE_BASE}/control/offline_then_online_seed1"
for dir in "${BASELINE_NOBC}" "${BASELINE_BC}"; do
  test -s "${dir}/snapshots/20000_snapshot.pkl"
done
evaluate "${BASELINE_NOBC}" 50 25 val50_seeds400_stage38.csv \
  "${BASE}/stage36_nobc_selection_val50.log" &
eval1=$!
evaluate "${BASELINE_BC}" 50 25 val50_seeds400_stage38.csv \
  "${BASE}/stage36_bc_selection_val50.log" &
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
