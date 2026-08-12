#!/usr/bin/env bash
# Stage 26: exact demo trajectory backup to 10.5k, candidate-max to 20k.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-1}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_no_bc/stage26_demo_trajectory20k_gpu${GPU}_${STAMP}"
SEED1="${BASE}/trajectory_seed1"
SEED2="${BASE}/trajectory_seed2"
ORDINARY1="exp_local/cqn_no_bc/stage10_seed1_gpu0_20260731041356/discounted_dense"
ORDINARY2="exp_local/cqn_no_bc/stage8_dense_multiseed_gpu0_20260731031717/dense_seed2"
STAGE22="$(cat exp_local/cqn_no_bc/stage22_latest.txt)"
CANDIDATE1="${STAGE22}/demo_candidate_seed1"
CANDIDATE2="${STAGE22}/demo_candidate_seed2"

mkdir -p "${BASE}/phase_a_configs"
printf "%s\n" "${BASHPID}" > "${BASE}/controller.pid"
printf "%s\n" "${BASE}" > exp_local/cqn_no_bc/stage26_latest.txt
printf "%s\n" "${ORDINARY1}" > "${BASE}/ordinary_seed1.txt"
printf "%s\n" "${ORDINARY2}" > "${BASE}/ordinary_seed2.txt"
printf "%s\n" "${CANDIDATE1}" > "${BASE}/candidate_seed1.txt"
printf "%s\n" "${CANDIDATE2}" > "${BASE}/candidate_seed2.txt"

train_phase () {
  local seed="$1"
  local run_dir="$2"
  local frames="$3"
  local force_probability="$4"
  local log="$5"
  XLA_PYTHON_CLIENT_PREALLOCATE=true \
    MUJOCO_GL=egl \
    .venv/bin/python train_fast.py \
    launch=cqn_as_pixel_bigym_nobc_stage26_demo_trajectory_gate \
    env=bigym/move_plate \
    seed="${seed}" \
    num_train_frames="${frames}" \
    method.demo_behavior_force_probability="${force_probability}" \
    gpu_id="${GPU}" \
    xla_mem_fraction=0.45 \
    wandb.use=false \
    hydra.run.dir="${run_dir}" \
    > "${log}" 2>&1
}

train_pair () {
  local frames="$1"
  local force_probability="$2"
  local phase="$3"
  train_phase 1 "${SEED1}" "${frames}" "${force_probability}" \
    "${BASE}/seed1_${phase}.log" &
  local seed1_pid=$!
  printf "%s\n" "${seed1_pid}" > "${BASE}/seed1_${phase}.pid"
  sleep 120
  train_phase 2 "${SEED2}" "${frames}" "${force_probability}" \
    "${BASE}/seed2_${phase}.log" &
  local seed2_pid=$!
  printf "%s\n" "${seed2_pid}" > "${BASE}/seed2_${phase}.pid"

  local status=0
  wait "${seed1_pid}" || status=$?
  wait "${seed2_pid}" || status=$?
  if [[ "${status}" -ne 0 ]]; then
    printf "%s\n" "${status}" > "${BASE}/${phase}_failed"
    exit "${status}"
  fi
}

train_pair 10500 1.0 phase_a
cp "${SEED1}/.hydra/config.yaml" "${BASE}/phase_a_configs/seed1.yaml"
cp "${SEED2}/.hydra/config.yaml" "${BASE}/phase_a_configs/seed2.yaml"
touch "${BASE}/phase_a_complete"

train_pair 20000 0.0 phase_b
touch "${BASE}/training_complete"

evaluate_full () {
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
    --csv-name val50_seeds400.csv \
    --skip-steps 10500 \
    > "${log}" 2>&1
}

evaluate_full "${SEED1}" "${BASE}/seed1_val50.log"
evaluate_full "${SEED2}" "${BASE}/seed2_val50.log"
touch "${BASE}/validation_complete"

.venv/bin/python -m scripts.summarize_cqn_no_bc_stage26 \
  --ordinary-seed1 "${ORDINARY1}" \
  --ordinary-seed2 "${ORDINARY2}" \
  --candidate-seed1 "${CANDIDATE1}" \
  --candidate-seed2 "${CANDIDATE2}" \
  --trajectory-seed1 "${SEED1}" \
  --trajectory-seed2 "${SEED2}" \
  --output "${BASE}/stage26_summary.json" \
  > "${BASE}/stage26_summary.log" 2>&1
touch "${BASE}/complete"
