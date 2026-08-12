#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_no_bc/stage39_positive_return_dense_gpu${GPU}_${STAMP}"
mkdir -p "${BASE}"
printf '%s\n' "${BASE}" > exp_local/cqn_no_bc/stage39_latest.txt
printf '%s\n' "$$" > "${BASE}/controller.pid"

train_seed () {
  local seed="$1"
  local run="${BASE}/seed${seed}"
  env XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
    .venv/bin/python train_fast.py \
    launch=cqn_as_pixel_bigym_nobc_stage39_positive_return_dense_gate \
    env=bigym/move_plate seed="${seed}" \
    num_pretrain_steps=0 num_train_frames=20000 \
    batch_size=256 demo_batch_size=256 \
    eval_every_steps=1000000 num_eval_episodes=0 num_eval_envs=0 \
    snapshot_every_n=2500 gpu_id="${GPU}" xla_mem_fraction=0.45 \
    wandb.use=false hydra.run.dir="${run}" \
    > "${BASE}/seed${seed}_train.log" 2>&1 &
  printf '%s\n' "$!" > "${BASE}/seed${seed}.pid"
}

train_seed 1
sleep 120
if ! kill -0 "$(cat "${BASE}/seed1.pid")" 2>/dev/null; then
  touch "${BASE}/training_failed"
  exit 1
fi
train_seed 2

status=0
wait "$(cat "${BASE}/seed1.pid")" || status=1
wait "$(cat "${BASE}/seed2.pid")" || status=1
if [[ "${status}" -ne 0 ]]; then
  touch "${BASE}/training_failed"
  exit 1
fi
touch "${BASE}/training_complete"

for seed in 1 2; do
  env XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
    .venv/bin/python \
    scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${BASE}/seed${seed}" --gpu-id "${GPU}" \
    --num-eval-episodes 50 --eval-seed-start 400 --num-eval-envs 25 \
    --only-steps 2500,5000,7500,10000,12500,15000,17500,20000 \
    --csv-name val50_seeds400_steps.csv \
    > "${BASE}/seed${seed}_val50.log" 2>&1
done
touch "${BASE}/validation_complete"

STAGE37="$(cat exp_local/cqn_no_bc/stage37_latest.txt)"
BASELINE_RUN_BASE="$(cat "${STAGE37}/run_base.txt")"
.venv/bin/python scripts/summarize_cqn_no_bc_stage39.py \
  --stage-dir "${BASE}" --baseline-run-base "${BASELINE_RUN_BASE}" \
  > "${BASE}/stage39_summary.log"
touch "${BASE}/complete"
