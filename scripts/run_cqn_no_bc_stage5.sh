#!/usr/bin/env bash
# One-GPU Stage-5 pair: mixed replay vs demo-only offline Q-learning.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-0}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_no_bc/stage5_seed1_gpu${GPU}_${STAMP}"
MIXED="${BASE}/mc_mixed_replay"
DEMO_ONLY="${BASE}/mc_demo_only"
mkdir -p "${BASE}"
printf "%s\n" "${BASHPID}" > "${BASE}/controller.pid"
printf "%s\n" "${BASE}" > exp_local/cqn_no_bc/stage5_latest.txt

train_one () {
  local launch="$1"
  local run_dir="$2"
  XLA_PYTHON_CLIENT_PREALLOCATE=true \
    MUJOCO_GL=egl \
    .venv/bin/python train_fast.py \
    launch="${launch}" \
    env=bigym/move_plate \
    seed=1 \
    gpu_id="${GPU}" \
    xla_mem_fraction=0.45 \
    wandb.use=false \
    hydra.run.dir="${run_dir}" \
    > "${run_dir}.launch.log" 2>&1
}

train_one cqn_as_pixel_bigym_nobc_mc_mixed_replay_gate "${MIXED}" &
MIXED_PID=$!
printf "%s\n" "${MIXED_PID}" > "${BASE}/mc_mixed_replay.pid"
sleep 120
train_one cqn_as_pixel_bigym_nobc_mc_demo_only_gate "${DEMO_ONLY}" &
DEMO_PID=$!
printf "%s\n" "${DEMO_PID}" > "${BASE}/mc_demo_only.pid"

status=0
wait "${MIXED_PID}" || status=$?
wait "${DEMO_PID}" || status=$?
if [[ "${status}" -ne 0 ]]; then
  printf "%s\n" "${status}" > "${BASE}/train_failed"
  exit "${status}"
fi
touch "${BASE}/training_complete"

evaluate_one () {
  local run_dir="$1"
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
    > "${run_dir}/val50_seeds400.log" 2>&1
}

evaluate_one "${MIXED}"
evaluate_one "${DEMO_ONLY}"
touch "${BASE}/validation_complete"

.venv/bin/python scripts/summarize_cqn_no_bc_stage5.py \
  --mixed-run "${MIXED}" \
  --demo-only-run "${DEMO_ONLY}" \
  --output "${BASE}/stage5_summary.json" \
  > "${BASE}/stage5_summary.log" 2>&1
touch "${BASE}/complete"
