#!/usr/bin/env bash
# Agent line Stage A3: positive-only-dense online continuation of the de-saturated offline
# checkpoints, from BRANCHED copies of the exact raw-10k states so the
# clock matches Stage 38 (10k offline + 10k online). Same launch config
# (dense + q_reward_scale=0.07). Preregistered in cqn-no-bc-claude.md.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-0}"
A1_BASE="${2:-$(tr -d '\n' < exp_local/cqn_no_bc/agent_a1_latest.txt)}"
LAUNCH="cqn_as_pixel_bigym_nobc_agent_a1_descale_gate"
OFFLINE_UPDATES=10000
GLOBAL_LIMIT=20000
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_no_bc/agent_a3_posonly_online_gpu${GPU}_${STAMP}"

test -f "${A1_BASE}/a1b_complete"
mkdir -p "${BASE}"
printf '%s\n' "${BASHPID}" > "${BASE}/controller.pid"
printf '%s\n' "${BASE}" > exp_local/cqn_no_bc/agent_a3_latest.txt
printf '%s\n' "${A1_BASE}" > "${BASE}/a1_source.txt"

run_dir () {
  printf '%s/descale_seed%s/online' "${BASE}" "$1"
}

for seed in 1 2; do
  mkdir -p "${BASE}/descale_seed${seed}"
  .venv/bin/python scripts/prepare_cqn_no_bc_stage40_branch.py \
    --source-run "${A1_BASE}/descale_seed${seed}/offline" \
    --destination-run "$(run_dir "${seed}")" \
    --snapshot-step "${OFFLINE_UPDATES}" \
    > "${BASE}/descale_seed${seed}/branch.log" 2>&1
done
touch "${BASE}/branch_complete"

train_online () {
  local seed="$1"
  local dir
  dir="$(run_dir "${seed}")"
  test -s "${dir}/snapshots/${OFFLINE_UPDATES}_snapshot.pkl"
  XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
    .venv/bin/python train_fast.py \
    launch="${LAUNCH}" env=bigym/move_plate seed="${seed}" \
    batch_size=256 demo_batch_size=256 \
    num_pretrain_steps="${OFFLINE_UPDATES}" \
    num_train_frames="${GLOBAL_LIMIT}" \
    replay.demo_only_updates=false \
    method.demo_behavior_force_probability=0.0 \
    method.dense_return_positive_only=true \
    eval_every_steps=1000000 num_eval_episodes=0 num_eval_envs=0 \
    snapshot_every_n=2500 gpu_id="${GPU}" xla_mem_fraction=0.45 \
    wandb.use=false hydra.run.dir="${dir}" \
    > "${BASE}/descale_seed${seed}/online.log" 2>&1
}

status=0
train_online 1 &
seed1_pid=$!
printf '%s\n' "${seed1_pid}" > "${BASE}/descale_seed1/online.pid"
sleep 120
train_online 2 &
seed2_pid=$!
printf '%s\n' "${seed2_pid}" > "${BASE}/descale_seed2/online.pid"
wait "${seed1_pid}" || status=$?
seed2_status=0
wait "${seed2_pid}" || seed2_status=$?
if [[ "${status}" -eq 0 ]]; then status="${seed2_status}"; fi
if [[ "${status}" -ne 0 ]]; then
  printf '%s\n' "${status}" > "${BASE}/online_failed"
  exit "${status}"
fi
for seed in 1 2; do
  dir="$(run_dir "${seed}")"
  test -s "${dir}/snapshots/${GLOBAL_LIMIT}_snapshot.pkl"
  cp "${dir}/.hydra/config.yaml" \
    "${BASE}/descale_seed${seed}/online_config.yaml"
done
touch "${BASE}/online_complete"

evaluate () {
  local dir="$1"
  local log="$2"
  XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${dir}" --gpu-id "${GPU}" \
    --num-eval-episodes 50 --eval-seed-start 400 \
    --num-eval-envs 25 \
    --only-steps "12500,15000,17500,20000" \
    --csv-name val50_seeds400_online.csv > "${log}" 2>&1
}

evaluate "$(run_dir 1)" "${BASE}/descale_seed1/a2_val50.log" &
eval1=$!
evaluate "$(run_dir 2)" "${BASE}/descale_seed2/a2_val50.log" &
eval2=$!
status=0
wait "${eval1}" || status=$?
eval2_status=0
wait "${eval2}" || eval2_status=$?
if [[ "${status}" -eq 0 ]]; then status="${eval2_status}"; fi
if [[ "${status}" -ne 0 ]]; then
  printf '%s\n' "${status}" > "${BASE}/validation_failed"
  exit "${status}"
fi
touch "${BASE}/validation_complete"
touch "${BASE}/complete"
