#!/usr/bin/env bash
# Agent line Stage A2b: bounded online extension of the A2 de-saturated
# runs from 20k to 30k global steps (10k more online env steps), then a
# 4-checkpoint 50-episode sweep. Preregistered in cqn-no-bc-claude.md.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-0}"
BASE="${2:-$(tr -d '\n' < exp_local/cqn_no_bc/agent_a2_latest.txt)}"
LAUNCH="cqn_as_pixel_bigym_nobc_agent_a1_descale_gate"
OFFLINE_UPDATES=10000
EXT_LIMIT=30000

test -f "${BASE}/complete"
printf '%s\n' "${BASHPID}" > "${BASE}/a2b_controller.pid"

run_dir () {
  printf '%s/descale_seed%s/online' "${BASE}" "$1"
}

extend_online () {
  local seed="$1"
  local dir
  dir="$(run_dir "${seed}")"
  test -s "${dir}/snapshots/20000_snapshot.pkl"
  XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
    .venv/bin/python train_fast.py \
    launch="${LAUNCH}" env=bigym/move_plate seed="${seed}" \
    batch_size=256 demo_batch_size=256 \
    num_pretrain_steps="${OFFLINE_UPDATES}" \
    num_train_frames="${EXT_LIMIT}" \
    replay.demo_only_updates=false \
    method.demo_behavior_force_probability=0.0 \
    eval_every_steps=1000000 num_eval_episodes=0 num_eval_envs=0 \
    snapshot_every_n=2500 gpu_id="${GPU}" xla_mem_fraction=0.45 \
    wandb.use=false hydra.run.dir="${dir}" \
    > "${BASE}/descale_seed${seed}/online_ext.log" 2>&1
}

status=0
extend_online 1 &
seed1_pid=$!
printf '%s\n' "${seed1_pid}" > "${BASE}/descale_seed1/online_ext.pid"
sleep 120
extend_online 2 &
seed2_pid=$!
printf '%s\n' "${seed2_pid}" > "${BASE}/descale_seed2/online_ext.pid"
wait "${seed1_pid}" || status=$?
seed2_status=0
wait "${seed2_pid}" || seed2_status=$?
if [[ "${status}" -eq 0 ]]; then status="${seed2_status}"; fi
if [[ "${status}" -ne 0 ]]; then
  printf '%s\n' "${status}" > "${BASE}/a2b_online_failed"
  exit "${status}"
fi
for seed in 1 2; do
  dir="$(run_dir "${seed}")"
  test -s "${dir}/snapshots/${EXT_LIMIT}_snapshot.pkl"
done
touch "${BASE}/a2b_online_complete"

evaluate () {
  local dir="$1"
  local log="$2"
  XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${dir}" --gpu-id "${GPU}" \
    --num-eval-episodes 50 --eval-seed-start 400 \
    --num-eval-envs 25 \
    --only-steps "22500,25000,27500,30000" \
    --csv-name val50_seeds400_online_ext.csv > "${log}" 2>&1
}

evaluate "$(run_dir 1)" "${BASE}/descale_seed1/a2b_val50.log" &
eval1=$!
evaluate "$(run_dir 2)" "${BASE}/descale_seed2/a2b_val50.log" &
eval2=$!
status=0
wait "${eval1}" || status=$?
eval2_status=0
wait "${eval2}" || eval2_status=$?
if [[ "${status}" -eq 0 ]]; then status="${eval2_status}"; fi
if [[ "${status}" -ne 0 ]]; then
  printf '%s\n' "${status}" > "${BASE}/a2b_validation_failed"
  exit "${status}"
fi
touch "${BASE}/a2b_validation_complete"
touch "${BASE}/a2b_complete"
