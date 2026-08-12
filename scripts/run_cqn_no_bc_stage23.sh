#!/usr/bin/env bash
# One-GPU Stage 23: seed-3 replication plus matched seed-2 20k extension.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-0}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_no_bc/stage23_candidate_repl_scale_gpu${GPU}_${STAMP}"
STAGE22="$(cat exp_local/cqn_no_bc/stage22_latest.txt)"
TREATMENT1="${STAGE22}/demo_candidate_seed1"
TREATMENT2="${STAGE22}/demo_candidate_seed2"
TREATMENT3="${BASE}/demo_candidate_seed3"
BASELINE1="exp_local/cqn_no_bc/stage10_seed1_gpu0_20260731041356/discounted_dense"
BASELINE2="exp_local/cqn_no_bc/stage8_dense_multiseed_gpu0_20260731031717/dense_seed2"
BASELINE3="exp_local/cqn_no_bc/stage8_dense_multiseed_gpu0_20260731031717/dense_seed3"

mkdir -p "${BASE}/configs_10k"
printf "%s\n" "${BASHPID}" > "${BASE}/controller.pid"
printf "%s\n" "${BASE}" > exp_local/cqn_no_bc/stage23_latest.txt
printf "%s\n" "${TREATMENT1}" > "${BASE}/treatment_seed1.txt"
printf "%s\n" "${TREATMENT2}" > "${BASE}/treatment_seed2.txt"
printf "%s\n" "${TREATMENT3}" > "${BASE}/treatment_seed3.txt"
printf "%s\n" "${BASELINE1}" > "${BASE}/baseline_seed1.txt"
printf "%s\n" "${BASELINE2}" > "${BASE}/baseline_seed2.txt"
printf "%s\n" "${BASELINE3}" > "${BASE}/baseline_seed3.txt"
cp "${TREATMENT2}/.hydra/config.yaml" "${BASE}/configs_10k/seed2.yaml"

train_seed3 () {
  XLA_PYTHON_CLIENT_PREALLOCATE=true \
    MUJOCO_GL=egl \
    .venv/bin/python train_fast.py \
    launch=cqn_as_pixel_bigym_nobc_stage22_demo_candidate_gate \
    env=bigym/move_plate \
    seed=3 \
    gpu_id="${GPU}" \
    xla_mem_fraction=0.45 \
    wandb.use=false \
    hydra.run.dir="${TREATMENT3}" \
    > "${BASE}/seed3_train.log" 2>&1
}

extend_seed2 () {
  XLA_PYTHON_CLIENT_PREALLOCATE=true \
    MUJOCO_GL=egl \
    .venv/bin/python train_fast.py \
    launch=cqn_as_pixel_bigym_nobc_stage22_demo_candidate_gate \
    env=bigym/move_plate \
    seed=2 \
    num_train_frames=20000 \
    gpu_id="${GPU}" \
    xla_mem_fraction=0.45 \
    wandb.use=false \
    hydra.run.dir="${TREATMENT2}" \
    > "${BASE}/seed2_resume20k.log" 2>&1
}

train_seed3 &
SEED3_PID=$!
printf "%s\n" "${SEED3_PID}" > "${BASE}/demo_candidate_seed3.pid"
sleep 120
extend_seed2 &
SEED2_PID=$!
printf "%s\n" "${SEED2_PID}" > "${BASE}/demo_candidate_seed2_ext.pid"

status=0
wait "${SEED3_PID}" || status=$?
wait "${SEED2_PID}" || status=$?
if [[ "${status}" -ne 0 ]]; then
  printf "%s\n" "${status}" > "${BASE}/train_failed"
  exit "${status}"
fi
touch "${BASE}/training_complete"

evaluate_seed3 () {
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MUJOCO_GL=egl \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${TREATMENT3}" \
    --gpu-id "${GPU}" \
    --num-eval-episodes 50 \
    --eval-seed-start 400 \
    --num-eval-envs 25 \
    --csv-name val50_seeds400.csv \
    --skip-steps 10500 \
    > "${BASE}/seed3_val50.log" 2>&1
}

evaluate_seed2_extension () {
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MUJOCO_GL=egl \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${TREATMENT2}" \
    --gpu-id "${GPU}" \
    --num-eval-episodes 50 \
    --eval-seed-start 400 \
    --num-eval-envs 25 \
    --csv-name val50_ext20k_seeds400.csv \
    --skip-steps 2500,5000,7500,10000,10500,20500 \
    > "${BASE}/seed2_val_ext20k.log" 2>&1
}

evaluate_seed3
evaluate_seed2_extension
touch "${BASE}/validation_complete"

.venv/bin/python scripts/summarize_cqn_no_bc_stage23.py \
  --baseline-seed1 "${BASELINE1}" \
  --baseline-seed2 "${BASELINE2}" \
  --baseline-seed3 "${BASELINE3}" \
  --treatment-seed1 "${TREATMENT1}" \
  --treatment-seed2 "${TREATMENT2}" \
  --treatment-seed3 "${TREATMENT3}" \
  --output "${BASE}/stage23_summary.json" \
  > "${BASE}/stage23_summary.log" 2>&1
touch "${BASE}/complete"
