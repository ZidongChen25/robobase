#!/usr/bin/env bash
# Stage 28: fresh reward-scale seed-3 replication at the matched 20k budget.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-1}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_no_bc/stage28_reward_scale_seed3_gpu${GPU}_${STAMP}"
TREATMENT3="${BASE}/reward_scale_seed3"
BASELINE3="exp_local/cqn_no_bc/stage8_dense_multiseed_gpu0_20260731031717/dense_seed3"
STAGE27="$(cat exp_local/cqn_no_bc/stage27_latest.txt)"

mkdir -p "${BASE}"
printf "%s\n" "${BASHPID}" > "${BASE}/controller.pid"
printf "%s\n" "${BASE}" > exp_local/cqn_no_bc/stage28_latest.txt
printf "%s\n" "${STAGE27}" > "${BASE}/stage27_source.txt"
printf "%s\n" "${BASELINE3}" > "${BASE}/baseline_seed3.txt"
printf "%s\n" "${TREATMENT3}" > "${BASE}/treatment_seed3.txt"

XLA_PYTHON_CLIENT_PREALLOCATE=false \
  MUJOCO_GL=egl \
  .venv/bin/python train_fast.py \
  launch=cqn_as_pixel_bigym_nobc_stage19_reward_scale_gate \
  env=bigym/move_plate \
  seed=3 \
  num_train_frames=20000 \
  gpu_id="${GPU}" \
  xla_mem_fraction=0.35 \
  wandb.use=false \
  hydra.run.dir="${TREATMENT3}" \
  > "${BASE}/seed3_train20k.log" 2>&1
touch "${BASE}/training_complete"

while [[ ! -f "${STAGE27}/complete" ]]; do
  sleep 30
done
touch "${BASE}/stage27_complete_seen"

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
touch "${BASE}/validation_complete"

.venv/bin/python -m scripts.summarize_cqn_no_bc_stage28 \
  --stage27-summary "${STAGE27}/stage27_summary.json" \
  --baseline-seed3 "${BASELINE3}" \
  --treatment-seed3 "${TREATMENT3}" \
  --output "${BASE}/stage28_summary.json" \
  > "${BASE}/stage28_summary.log" 2>&1
touch "${BASE}/complete"
