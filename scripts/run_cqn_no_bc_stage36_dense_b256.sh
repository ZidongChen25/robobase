#!/usr/bin/env bash
# Stage 36: batch-scaled replication of the retained dense No-BC baseline.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-4}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_no_bc/stage36_dense_b256_gpu${GPU}_${STAMP}"
mkdir -p "${BASE}"
printf "%s\n" "${BASHPID}" > "${BASE}/controller.pid"
printf "%s\n" "${BASE}" > exp_local/cqn_no_bc/stage36_latest.txt

train_one () {
  local seed="$1"
  local run_dir="${BASE}/dense_b256_seed${seed}"
  mkdir -p "${run_dir}"
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MUJOCO_GL=egl \
    .venv/bin/python train_fast.py \
    launch=cqn_as_pixel_bigym_nobc_stage10_discounted_dense_control \
    env=bigym/move_plate \
    seed="${seed}" \
    num_pretrain_steps=0 \
    num_train_frames=20000 \
    batch_size=256 \
    demo_batch_size=256 \
    snapshot_every_n=2500 \
    gpu_id="${GPU}" \
    xla_mem_fraction=0.45 \
    wandb.use=false \
    hydra.run.dir="${run_dir}" \
    > "${BASE}/seed${seed}_train.log" 2>&1
  touch "${BASE}/seed${seed}_training_complete"
}

evaluate_one () {
  local seed="$1"
  local run_dir="${BASE}/dense_b256_seed${seed}"
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MUJOCO_GL=egl \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${run_dir}" \
    --gpu-id "${GPU}" \
    --num-eval-episodes 50 \
    --eval-seed-start 400 \
    --num-eval-envs 25 \
    --only-steps "2500,5000,7500,10000,12500,15000,17500,20000" \
    --csv-name val50_seeds400_steps.csv \
    > "${BASE}/seed${seed}_val50.log" 2>&1
  touch "${BASE}/seed${seed}_validation_complete"
}

# The first wave follows the measured two-runs/card protocol.
train_one 1 &
PID1=$!
printf "%s\n" "${PID1}" > "${BASE}/seed1.pid"
sleep 120
train_one 2 &
PID2=$!
printf "%s\n" "${PID2}" > "${BASE}/seed2.pid"
status=0
wait "${PID1}" || status=$?
seed2_status=0
wait "${PID2}" || seed2_status=$?
if [[ "${status}" -eq 0 ]]; then status="${seed2_status}"; fi
if [[ "${status}" -ne 0 ]]; then
  printf "%s\n" "${status}" > "${BASE}/training_failed"
  exit "${status}"
fi

# Evals do not overlap training. Two 50-episode sweeps may share the card.
evaluate_one 1 &
EVAL1=$!
evaluate_one 2 &
EVAL2=$!
status=0
wait "${EVAL1}" || status=$?
seed2_status=0
wait "${EVAL2}" || seed2_status=$?
if [[ "${status}" -eq 0 ]]; then status="${seed2_status}"; fi
if [[ "${status}" -ne 0 ]]; then
  printf "%s\n" "${status}" > "${BASE}/validation_failed"
  exit "${status}"
fi

# Seed 3 starts only after the first wave and its evals release the GPU.
train_one 3
evaluate_one 3
touch "${BASE}/validation_complete"

.venv/bin/python -m scripts.summarize_cqn_no_bc_stage36 \
  --base "${BASE}" \
  --output "${BASE}/stage36_summary.json" \
  > "${BASE}/stage36_summary.log" 2>&1
touch "${BASE}/complete"
