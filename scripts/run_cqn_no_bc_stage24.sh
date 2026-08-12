#!/usr/bin/env bash
# One-GPU Stage 24: exact candidate-backup seeds 1 and 3 to 20k.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-1}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_no_bc/stage24_candidate_extend20k_gpu${GPU}_${STAMP}"
STAGE22="$(cat exp_local/cqn_no_bc/stage22_latest.txt)"
STAGE23="$(cat exp_local/cqn_no_bc/stage23_latest.txt)"
TREATMENT1="${STAGE22}/demo_candidate_seed1"
TREATMENT3="${STAGE23}/demo_candidate_seed3"
BASELINE1="exp_local/cqn_no_bc/stage10_seed1_gpu0_20260731041356/discounted_dense"
BASELINE3="exp_local/cqn_no_bc/stage8_dense_multiseed_gpu0_20260731031717/dense_seed3"

mkdir -p "${BASE}/configs_10k"
printf "%s\n" "${BASHPID}" > "${BASE}/controller.pid"
printf "%s\n" "${BASE}" > exp_local/cqn_no_bc/stage24_latest.txt
printf "%s\n" "${TREATMENT1}" > "${BASE}/treatment_seed1.txt"
printf "%s\n" "${TREATMENT3}" > "${BASE}/treatment_seed3.txt"
printf "%s\n" "${BASELINE1}" > "${BASE}/baseline_seed1.txt"
printf "%s\n" "${BASELINE3}" > "${BASE}/baseline_seed3.txt"
cp "${TREATMENT1}/.hydra/config.yaml" "${BASE}/configs_10k/seed1.yaml"
cp "${TREATMENT3}/.hydra/config.yaml" "${BASE}/configs_10k/seed3.yaml"

extend_one () {
  local seed="$1"
  local run_dir="$2"
  local log="$3"
  XLA_PYTHON_CLIENT_PREALLOCATE=true \
    MUJOCO_GL=egl \
    .venv/bin/python train_fast.py \
    launch=cqn_as_pixel_bigym_nobc_stage22_demo_candidate_gate \
    env=bigym/move_plate \
    seed="${seed}" \
    num_train_frames=20000 \
    gpu_id="${GPU}" \
    xla_mem_fraction=0.45 \
    wandb.use=false \
    hydra.run.dir="${run_dir}" \
    > "${log}" 2>&1
}

extend_one 1 "${TREATMENT1}" "${BASE}/seed1_resume20k.log" &
SEED1_PID=$!
printf "%s\n" "${SEED1_PID}" > "${BASE}/demo_candidate_seed1_ext.pid"
sleep 120
extend_one 3 "${TREATMENT3}" "${BASE}/seed3_resume20k.log" &
SEED3_PID=$!
printf "%s\n" "${SEED3_PID}" > "${BASE}/demo_candidate_seed3_ext.pid"

status=0
wait "${SEED1_PID}" || status=$?
wait "${SEED3_PID}" || status=$?
if [[ "${status}" -ne 0 ]]; then
  printf "%s\n" "${status}" > "${BASE}/train_failed"
  exit "${status}"
fi
touch "${BASE}/training_complete"

evaluate_extension () {
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

evaluate_extension "${TREATMENT1}" "${BASE}/seed1_val_ext20k.log"
evaluate_extension "${TREATMENT3}" "${BASE}/seed3_val_ext20k.log"
touch "${BASE}/validation_complete"

.venv/bin/python -m scripts.summarize_cqn_no_bc_stage24 \
  --baseline-seed1 "${BASELINE1}" \
  --baseline-seed3 "${BASELINE3}" \
  --treatment-seed1 "${TREATMENT1}" \
  --treatment-seed3 "${TREATMENT3}" \
  --output "${BASE}/stage24_summary.json" \
  > "${BASE}/stage24_summary.log" 2>&1
touch "${BASE}/complete"
