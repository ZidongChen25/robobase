#!/usr/bin/env bash
# One-GPU Stage-9 exact resume: dense-return seeds 2 and 3 to 20k.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-0}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_no_bc/stage9_dense_extend20k_gpu${GPU}_${STAMP}"
STAGE8="$(cat exp_local/cqn_no_bc/stage8_latest.txt)"
SEED2="${STAGE8}/dense_seed2"
SEED3="${STAGE8}/dense_seed3"
mkdir -p "${BASE}/configs_10k"
printf "%s\n" "${BASHPID}" > "${BASE}/controller.pid"
printf "%s\n" "${BASE}" > exp_local/cqn_no_bc/stage9_latest.txt
printf "%s\n" "${SEED2}" > "${BASE}/seed2_run.txt"
printf "%s\n" "${SEED3}" > "${BASE}/seed3_run.txt"
cp "${SEED2}/.hydra/config.yaml" "${BASE}/configs_10k/seed2.yaml"
cp "${SEED3}/.hydra/config.yaml" "${BASE}/configs_10k/seed3.yaml"

resume_one () {
  local seed="$1"
  local run_dir="$2"
  local log="$3"
  XLA_PYTHON_CLIENT_PREALLOCATE=true \
    MUJOCO_GL=egl \
    .venv/bin/python train_fast.py \
    launch=cqn_as_pixel_bigym_nobc_mc_stage7_dense_return_gate \
    env=bigym/move_plate \
    seed="${seed}" \
    num_train_frames=20000 \
    gpu_id="${GPU}" \
    xla_mem_fraction=0.45 \
    wandb.use=false \
    hydra.run.dir="${run_dir}" \
    >> "${log}" 2>&1
}

resume_one 2 "${SEED2}" "${BASE}/seed2_resume20k.log" &
SEED2_PID=$!
printf "%s\n" "${SEED2_PID}" > "${BASE}/dense_seed2.pid"
sleep 120
resume_one 3 "${SEED3}" "${BASE}/seed3_resume20k.log" &
SEED3_PID=$!
printf "%s\n" "${SEED3_PID}" > "${BASE}/dense_seed3.pid"

status=0
wait "${SEED2_PID}" || status=$?
wait "${SEED3_PID}" || status=$?
if [[ "${status}" -ne 0 ]]; then
  printf "%s\n" "${status}" > "${BASE}/train_failed"
  exit "${status}"
fi
touch "${BASE}/training_complete"

evaluate_new () {
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
    --skip-steps 2500,5000,7500,10000,10500,20500 \
    > "${log}" 2>&1
}

evaluate_new "${SEED2}" "${BASE}/seed2_val_ext20k.log"
evaluate_new "${SEED3}" "${BASE}/seed3_val_ext20k.log"
touch "${BASE}/validation_complete"

.venv/bin/python scripts/summarize_cqn_no_bc_stage9.py \
  --seed2-run "${SEED2}" \
  --seed3-run "${SEED3}" \
  --output "${BASE}/stage9_summary.json" \
  > "${BASE}/stage9_summary.log" 2>&1
touch "${BASE}/complete"
