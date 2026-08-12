#!/usr/bin/env bash
# Recover the externally terminated Stage-35 seed-2 treatment, then validate.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-3}"
BASE="$(tr -d '\n' < exp_local/cqn_no_bc/stage35_latest.txt)"
RUN_DIR="${BASE}/seed2/treatment/offline_twin_seed2"
printf "%s\n" "${BASHPID}" > "${BASE}/recovery_controller.pid"
printf "%s\n" "${GPU}" > "${BASE}/recovery_gpu.txt"

XLA_PYTHON_CLIENT_PREALLOCATE=false \
  MUJOCO_GL=egl \
  .venv/bin/python train_fast.py \
  launch=cqn_as_pixel_bigym_nobc_stage35_one_plus_four_gate \
  env=bigym/move_plate \
  seed=2 \
  num_pretrain_steps=10000 \
  num_train_frames=111000 \
  demo_batch_size=16 \
  replay.demo_only_updates=false \
  method.demo_behavior_force_probability=0.0 \
  snapshot_every_n=10000 \
  gpu_id="${GPU}" \
  xla_mem_fraction=0.35 \
  wandb.use=false \
  hydra.run.dir="${RUN_DIR}" \
  > "${BASE}/seed2/treatment/recovery_gpu${GPU}.log" 2>&1
touch "${BASE}/seed2/treatment/recovery_training_complete"

evaluate_one () {
  local seed="$1"
  local arm="$2"
  local run_dir="${BASE}/seed${seed}/${arm}/offline_twin_seed${seed}"
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MUJOCO_GL=egl \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${run_dir}" \
    --gpu-id "${GPU}" \
    --num-eval-episodes 50 \
    --eval-seed-start 400 \
    --num-eval-envs 25 \
    --only-steps "10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,110000,111000" \
    --csv-name val50_seeds400_full_raw_steps.csv \
    > "${BASE}/seed${seed}/${arm}/recovery_val50_gpu${GPU}.log" 2>&1
}

# Two evals may share the otherwise idle card; no eval overlaps training.
for seed in 1 2; do
  evaluate_one "${seed}" control &
  eval_control=$!
  evaluate_one "${seed}" treatment &
  eval_treatment=$!
  status=0
  wait "${eval_control}" || status=$?
  treatment_status=0
  wait "${eval_treatment}" || treatment_status=$?
  if [[ "${status}" -eq 0 ]]; then status="${treatment_status}"; fi
  if [[ "${status}" -ne 0 ]]; then
    printf "%s\n" "${status}" > "${BASE}/recovery_validation_failed"
    exit "${status}"
  fi
done
touch "${BASE}/recovery_validation_complete"

.venv/bin/python -m scripts.summarize_cqn_no_bc_stage35 \
  --base "${BASE}" \
  --output "${BASE}/stage35_summary.json" \
  > "${BASE}/stage35_summary.log" 2>&1
touch "${BASE}/recovery_complete"
