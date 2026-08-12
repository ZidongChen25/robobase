#!/usr/bin/env bash
# Agent line Stage A6: offline-budget scaling measurement. Branch seeds
# 3-6 at their exact raw-10k pre-online offline states, extend pure
# demo-only training to 30k updates, evaluate raw 15/20/25/30k.
# Measurement stage (no gate). Preregistered in cqn-no-bc-claude.md.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-0}"
A5_BASE="${2:-$(tr -d '\n' < exp_local/cqn_no_bc/agent_a5_latest.txt)}"
LAUNCH="cqn_as_pixel_bigym_nobc_agent_a1_descale_gate"
EXT_UPDATES=30000
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_no_bc/agent_a6_offline_scaling_gpu${GPU}_${STAMP}"

test -f "${A5_BASE}/complete"
mkdir -p "${BASE}"
printf '%s\n' "${BASHPID}" > "${BASE}/controller.pid"
printf '%s\n' "${BASE}" > exp_local/cqn_no_bc/agent_a6_latest.txt

run_dir () {
  printf '%s/descale_seed%s/offline_ext' "${BASE}" "$1"
}

for seed in 3 4 5 6; do
  mkdir -p "${BASE}/descale_seed${seed}"
  .venv/bin/python scripts/prepare_cqn_no_bc_stage40_branch.py \
    --source-run "${A5_BASE}/descale_seed${seed}/full" \
    --destination-run "$(run_dir "${seed}")" \
    --snapshot-step 10000 \
    > "${BASE}/descale_seed${seed}/branch.log" 2>&1
done
touch "${BASE}/branch_complete"

train_ext () {
  local seed="$1"
  local dir
  dir="$(run_dir "${seed}")"
  test -s "${dir}/snapshots/10000_snapshot.pkl"
  XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
    .venv/bin/python train_fast.py \
    launch="${LAUNCH}" env=bigym/move_plate seed="${seed}" \
    batch_size=256 demo_batch_size=256 \
    num_pretrain_steps="${EXT_UPDATES}" \
    num_train_frames="${EXT_UPDATES}" \
    replay.demo_only_updates=true \
    method.demo_behavior_force_probability=1.0 \
    eval_every_steps=1000000 num_eval_episodes=0 num_eval_envs=0 \
    snapshot_every_n=2500 gpu_id="${GPU}" xla_mem_fraction=0.45 \
    wandb.use=false hydra.run.dir="${dir}" \
    > "${BASE}/descale_seed${seed}/offline_ext.log" 2>&1
}

evaluate () {
  local dir="$1"
  local log="$2"
  XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${dir}" --gpu-id "${GPU}" \
    --num-eval-episodes 50 --eval-seed-start 400 \
    --num-eval-envs 25 \
    --only-steps "15000,20000,25000,30000" \
    --csv-name val50_seeds400_offline_ext.csv > "${log}" 2>&1
}

run_pair () {
  local s_a="$1"
  local s_b="$2"
  local status=0
  train_ext "${s_a}" &
  local pid_a=$!
  sleep 120
  train_ext "${s_b}" &
  local pid_b=$!
  wait "${pid_a}" || status=$?
  local status_b=0
  wait "${pid_b}" || status_b=$?
  if [[ "${status}" -eq 0 ]]; then status="${status_b}"; fi
  if [[ "${status}" -ne 0 ]]; then
    printf '%s\n' "${status}" > "${BASE}/pair_${s_a}_${s_b}_train_failed"
    return "${status}"
  fi
  evaluate "$(run_dir "${s_a}")" "${BASE}/descale_seed${s_a}/val50.log" &
  local eval_a=$!
  evaluate "$(run_dir "${s_b}")" "${BASE}/descale_seed${s_b}/val50.log" &
  local eval_b=$!
  wait "${eval_a}" || status=$?
  local eval_b_status=0
  wait "${eval_b}" || eval_b_status=$?
  if [[ "${status}" -eq 0 ]]; then status="${eval_b_status}"; fi
  if [[ "${status}" -ne 0 ]]; then
    printf '%s\n' "${status}" > "${BASE}/pair_${s_a}_${s_b}_val_failed"
    return "${status}"
  fi
  touch "${BASE}/pair_${s_a}_${s_b}_complete"
  return 0
}

run_pair 3 4
run_pair 5 6
touch "${BASE}/complete"
