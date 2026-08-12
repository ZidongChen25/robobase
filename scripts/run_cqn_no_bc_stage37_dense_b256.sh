#!/usr/bin/env bash
# Stage 37: recover the requested dense No-BC b256 replication and gate a full run.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_PRIMARY="${1:-4}"
GPU_SEED3="${2:-3}"
RUN_BASE="$(tr -d '\n' < exp_local/cqn_no_bc/stage36_latest.txt)"
BLOCKER="$(tr -d '\n' < exp_local/cqn_no_bc/stage36_offline_gate_latest.txt 2>/dev/null || true)"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_no_bc/stage37_dense_b256_${STAMP}"
mkdir -p "${BASE}"
printf '%s\n' "${BASHPID}" > "${BASE}/controller.pid"
printf '%s\n' "${BASE}" > exp_local/cqn_no_bc/stage37_latest.txt
printf '%s\n' "${RUN_BASE}" > "${BASE}/run_base.txt"
printf '%s\n' "${BLOCKER}" > "${BASE}/gpu4_blocker.txt"

train_one () {
  local seed="$1"
  local gpu="$2"
  local frames="$3"
  local run_dir="${RUN_BASE}/dense_b256_seed${seed}"
  mkdir -p "${run_dir}"
  XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
    .venv/bin/python train_fast.py \
    launch=cqn_as_pixel_bigym_nobc_stage10_discounted_dense_control \
    env=bigym/move_plate seed="${seed}" \
    num_pretrain_steps=0 num_train_frames="${frames}" \
    batch_size=256 demo_batch_size=256 \
    snapshot_every_n=2500 gpu_id="${gpu}" xla_mem_fraction=0.45 \
    wandb.use=false hydra.run.dir="${run_dir}" \
    > "${BASE}/seed${seed}_to${frames}_gpu${gpu}.log" 2>&1
}

evaluate_short () {
  local seed="$1"
  local run_dir="${RUN_BASE}/dense_b256_seed${seed}"
  XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${run_dir}" --gpu-id "${GPU_PRIMARY}" \
    --num-eval-episodes 50 --eval-seed-start 400 --num-eval-envs 25 \
    --only-steps "2500,5000,7500,10000,12500,15000,17500,20000" \
    --csv-name val50_seeds400_steps.csv \
    > "${BASE}/seed${seed}_short_val50.log" 2>&1
}

# Seed 3 may safely share GPU 3 with one already-compiled training process.
train_one 3 "${GPU_SEED3}" 20000 &
SEED3_PID=$!
printf '%s\n' "${SEED3_PID}" > "${BASE}/seed3.pid"

# GPU 4 is temporarily owned by the distinct Stage-36 offline gate.
if [[ -n "${BLOCKER}" ]]; then
  while [[ ! -e "${BLOCKER}/complete" && ! -e "${BLOCKER}/offline_failed" \
      && ! -e "${BLOCKER}/training_failed" && ! -e "${BLOCKER}/validation_failed" ]]; do
    sleep 20
  done
  while pgrep -f "${BLOCKER}" >/dev/null 2>&1; do sleep 10; done
fi
touch "${BASE}/gpu4_released"

# Recover the partial seed-1/2 runs in one staggered two-run wave.
train_one 1 "${GPU_PRIMARY}" 20000 &
SEED1_PID=$!
printf '%s\n' "${SEED1_PID}" > "${BASE}/seed1.pid"
sleep 120
train_one 2 "${GPU_PRIMARY}" 20000 &
SEED2_PID=$!
printf '%s\n' "${SEED2_PID}" > "${BASE}/seed2.pid"

status=0
for pid in "${SEED1_PID}" "${SEED2_PID}" "${SEED3_PID}"; do
  child_status=0
  wait "${pid}" || child_status=$?
  if [[ "${status}" -eq 0 ]]; then status="${child_status}"; fi
done
if [[ "${status}" -ne 0 ]]; then
  printf '%s\n' "${status}" > "${BASE}/training_failed"
  exit "${status}"
fi
touch "${BASE}/short_training_complete"

# All fixed evals run after training has released GPU 4.
evaluate_short 1 & E1=$!
evaluate_short 2 & E2=$!
status=0
wait "${E1}" || status=$?
e2_status=0; wait "${E2}" || e2_status=$?
if [[ "${status}" -eq 0 ]]; then status="${e2_status}"; fi
if [[ "${status}" -ne 0 ]]; then exit "${status}"; fi
evaluate_short 3

.venv/bin/python -m scripts.summarize_cqn_no_bc_stage37 \
  --run-base "${RUN_BASE}" --mode short \
  --output "${BASE}/stage37_short_summary.json" \
  > "${BASE}/stage37_short_summary.log" 2>&1
touch "${BASE}/short_validation_complete"

if ! rg -q '"promotion_pass": true' "${BASE}/stage37_short_summary.json"; then
  touch "${BASE}/complete"
  exit 0
fi
touch "${BASE}/full_run_promoted"

# The training seed is fixed in advance; it is not chosen from the short curves.
train_one 1 "${GPU_PRIMARY}" 101000
touch "${BASE}/full_training_complete"
# Evaluations must run on an otherwise idle GPU.  Training above may have
# shared the card with an unrelated, hard-sliced process, so releasing our own
# training allocation alone is not sufficient evidence that eval is isolated.
touch "${BASE}/waiting_for_eval_gpu"
while nvidia-smi -i "${GPU_PRIMARY}" \
    --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | rg -q '[0-9]'; do
  sleep 30
done
touch "${BASE}/eval_gpu_released"
RUN1="${RUN_BASE}/dense_b256_seed1"
XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
  .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
  --run-dir "${RUN1}" --gpu-id "${GPU_PRIMARY}" \
  --num-eval-episodes 50 --eval-seed-start 400 --num-eval-envs 25 \
  --only-steps "20000,30000,40000,50000,60000,70000,80000,90000,100000,101000" \
  --csv-name val50_seeds400_full.csv > "${BASE}/seed1_full_val50.log" 2>&1
XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
  .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
  --run-dir "${RUN1}" --gpu-id "${GPU_PRIMARY}" \
  --num-eval-episodes 200 --eval-seed-start 800 --num-eval-envs 25 \
  --only-steps "101000" --csv-name heldout200_seeds800_endpoint.csv \
  > "${BASE}/seed1_heldout200.log" 2>&1
.venv/bin/python -m scripts.summarize_cqn_no_bc_stage37 \
  --run-base "${RUN_BASE}" --mode full \
  --output "${BASE}/stage37_full_summary.json" \
  > "${BASE}/stage37_full_summary.log" 2>&1
touch "${BASE}/complete"
