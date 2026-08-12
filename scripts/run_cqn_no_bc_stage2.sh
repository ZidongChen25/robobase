#!/usr/bin/env bash
# One-GPU Stage-2 factorial completion: MC-only vs MC+return-floor.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-0}"
STAGE1="${2:?usage: run_cqn_no_bc_stage2.sh GPU STAGE1_DIR}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_no_bc/stage2_seed1_gpu${GPU}_${STAMP}"
MC="${BASE}/mc_only"
MC_FLOOR="${BASE}/mc_floor"
mkdir -p "${BASE}"
printf "%s\n" "${BASHPID}" > "${BASE}/controller.pid"
printf "%s\n" "${BASE}" > exp_local/cqn_no_bc/stage2_latest.txt

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

train_one cqn_as_pixel_bigym_nobc_mc_gate "${MC}" &
MC_PID=$!
printf "%s\n" "${MC_PID}" > "${BASE}/mc.pid"
sleep 120
train_one cqn_as_pixel_bigym_nobc_mc_floor_gate "${MC_FLOOR}" &
MC_FLOOR_PID=$!
printf "%s\n" "${MC_FLOOR_PID}" > "${BASE}/mc_floor.pid"

status=0
wait "${MC_PID}" || status=$?
wait "${MC_FLOOR_PID}" || status=$?
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

evaluate_one "${MC}"
evaluate_one "${MC_FLOOR}"
touch "${BASE}/validation_complete"

.venv/bin/python scripts/summarize_cqn_no_bc_stage2.py \
  --td-run "${STAGE1}/control_td" \
  --floor-run "${STAGE1}/treatment_floor" \
  --mc-run "${MC}" \
  --mc-floor-run "${MC_FLOOR}" \
  --output "${BASE}/stage2_summary.json" \
  > "${BASE}/stage2_summary.log" 2>&1
touch "${BASE}/complete"
