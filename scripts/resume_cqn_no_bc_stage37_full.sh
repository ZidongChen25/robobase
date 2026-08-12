#!/usr/bin/env bash
# Recover Stage 37 after an interrupted promoted full run.  Training resumes
# from the latest durable snapshot; evaluation waits for an otherwise idle GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN_GPU="${1:-4}"
PREFERRED_EVAL_GPU="${2:-5}"
BASE="$(tr -d '\n' < exp_local/cqn_no_bc/stage37_latest.txt)"
RUN_BASE="$(tr -d '\n' < "${BASE}/run_base.txt")"
RUN1="${RUN_BASE}/dense_b256_seed1"

printf '%s\n' "${BASHPID}" > "${BASE}/recovery_controller.pid"
touch "${BASE}/recovery_from_32500"

XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
  .venv/bin/python train_fast.py \
  launch=cqn_as_pixel_bigym_nobc_stage10_discounted_dense_control \
  env=bigym/move_plate seed=1 \
  num_pretrain_steps=0 num_train_frames=101000 \
  batch_size=256 demo_batch_size=256 \
  snapshot_every_n=2500 gpu_id="${TRAIN_GPU}" xla_mem_fraction=0.45 \
  wandb.use=false hydra.run.dir="${RUN1}" \
  > "${BASE}/seed1_to101000_gpu${TRAIN_GPU}_resume32500.log" 2>&1 &
TRAIN_PID=$!
printf '%s\n' "${TRAIN_PID}" > "${BASE}/recovery_train.pid"
status=0
wait "${TRAIN_PID}" || status=$?
if [[ "${status}" -ne 0 ]]; then
  printf '%s\n' "${status}" > "${BASE}/full_recovery_training_failed"
  exit "${status}"
fi
touch "${BASE}/full_training_complete"

# Require three consecutive idle samples so a transient nvidia-smi failure
# cannot accidentally admit an evaluation alongside training.
choose_idle_gpu () {
  local order gpu apps free_mb sample ok
  order="${PREFERRED_EVAL_GPU} 0 1 2 3 4 5"
  for gpu in ${order}; do
    [[ "${gpu}" =~ ^[0-5]$ ]] || continue
    ok=1
    for sample in 1 2 3; do
      if ! apps="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid \
          --format=csv,noheader,nounits 2>/dev/null)"; then
        ok=0
        break
      fi
      if [[ -n "${apps//[[:space:]]/}" ]]; then
        ok=0
        break
      fi
      if ! free_mb="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free \
          --format=csv,noheader,nounits 2>/dev/null)"; then
        ok=0
        break
      fi
      free_mb="${free_mb//[[:space:]]/}"
      if [[ ! "${free_mb}" =~ ^[0-9]+$ || "${free_mb}" -lt 2048 ]]; then
        ok=0
        break
      fi
      sleep 10
    done
    if [[ "${ok}" -eq 1 ]]; then
      printf '%s\n' "${gpu}"
      return 0
    fi
  done
  return 1
}

touch "${BASE}/waiting_for_eval_gpu"
EVAL_GPU=""
until EVAL_GPU="$(choose_idle_gpu)"; do sleep 30; done
printf '%s\n' "${EVAL_GPU}" > "${BASE}/eval_gpu.txt"
touch "${BASE}/eval_gpu_released"

XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
  .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
  --run-dir "${RUN1}" --gpu-id "${EVAL_GPU}" \
  --num-eval-episodes 50 --eval-seed-start 400 --num-eval-envs 25 \
  --only-steps "20000,30000,40000,50000,60000,70000,80000,90000,100000,101000" \
  --csv-name val50_seeds400_full.csv \
  > "${BASE}/seed1_full_val50.log" 2>&1

XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
  .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
  --run-dir "${RUN1}" --gpu-id "${EVAL_GPU}" \
  --num-eval-episodes 200 --eval-seed-start 800 --num-eval-envs 25 \
  --only-steps "101000" --csv-name heldout200_seeds800_endpoint.csv \
  > "${BASE}/seed1_heldout200.log" 2>&1

.venv/bin/python -m scripts.summarize_cqn_no_bc_stage37 \
  --run-base "${RUN_BASE}" --mode full \
  --output "${BASE}/stage37_full_summary.json" \
  > "${BASE}/stage37_full_summary.log" 2>&1
touch "${BASE}/complete"
