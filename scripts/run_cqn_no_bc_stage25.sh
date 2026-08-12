#!/usr/bin/env bash
# Stage 25: matched ordinary no-BC seed-1 extension and three-seed gate.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-1}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_no_bc/stage25_seed1_control20k_gpu${GPU}_${STAMP}"
STAGE22="$(cat exp_local/cqn_no_bc/stage22_latest.txt)"
STAGE23="$(cat exp_local/cqn_no_bc/stage23_latest.txt)"
STAGE24="$(cat exp_local/cqn_no_bc/stage24_latest.txt)"
TREATMENT1="${STAGE22}/demo_candidate_seed1"
TREATMENT2="${STAGE22}/demo_candidate_seed2"
TREATMENT3="${STAGE23}/demo_candidate_seed3"
BASELINE1="exp_local/cqn_no_bc/stage10_seed1_gpu0_20260731041356/discounted_dense"
BASELINE2="exp_local/cqn_no_bc/stage8_dense_multiseed_gpu0_20260731031717/dense_seed2"
BASELINE3="exp_local/cqn_no_bc/stage8_dense_multiseed_gpu0_20260731031717/dense_seed3"

mkdir -p "${BASE}/config_10k"
printf "%s\n" "${BASHPID}" > "${BASE}/controller.pid"
printf "%s\n" "${BASE}" > exp_local/cqn_no_bc/stage25_latest.txt
printf "%s\n" "${STAGE24}" > "${BASE}/stage24_source.txt"
printf "%s\n" "${TREATMENT1}" > "${BASE}/treatment_seed1.txt"
printf "%s\n" "${TREATMENT2}" > "${BASE}/treatment_seed2.txt"
printf "%s\n" "${TREATMENT3}" > "${BASE}/treatment_seed3.txt"
printf "%s\n" "${BASELINE1}" > "${BASE}/baseline_seed1.txt"
printf "%s\n" "${BASELINE2}" > "${BASE}/baseline_seed2.txt"
printf "%s\n" "${BASELINE3}" > "${BASE}/baseline_seed3.txt"
cp "${BASELINE1}/.hydra/config.yaml" "${BASE}/config_10k/seed1.yaml"

XLA_PYTHON_CLIENT_PREALLOCATE=true \
  MUJOCO_GL=egl \
  .venv/bin/python train_fast.py \
  launch=cqn_as_pixel_bigym_nobc_stage10_discounted_dense_control \
  env=bigym/move_plate \
  seed=1 \
  num_train_frames=20000 \
  gpu_id="${GPU}" \
  xla_mem_fraction=0.45 \
  wandb.use=false \
  hydra.run.dir="${BASELINE1}" \
  > "${BASE}/seed1_control_resume20k.log" 2>&1
touch "${BASE}/training_complete"

XLA_PYTHON_CLIENT_PREALLOCATE=false \
  MUJOCO_GL=egl \
  .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
  --run-dir "${BASELINE1}" \
  --gpu-id "${GPU}" \
  --num-eval-episodes 50 \
  --eval-seed-start 400 \
  --num-eval-envs 25 \
  --csv-name val50_ext20k_seeds400.csv \
  --skip-steps 2500,5000,7500,10000,10500 \
  > "${BASE}/seed1_control_val_ext20k.log" 2>&1
touch "${BASE}/validation_complete"

.venv/bin/python -m scripts.summarize_cqn_no_bc_stage25 \
  --baseline-seed1 "${BASELINE1}" \
  --baseline-seed2 "${BASELINE2}" \
  --baseline-seed3 "${BASELINE3}" \
  --treatment-seed1 "${TREATMENT1}" \
  --treatment-seed2 "${TREATMENT2}" \
  --treatment-seed3 "${TREATMENT3}" \
  --output "${BASE}/stage25_summary.json" \
  > "${BASE}/stage25_summary.log" 2>&1
touch "${BASE}/complete"
