#!/usr/bin/env bash
# Agent line Stage A1: offline-only de-saturation screen (q_reward_scale=0.07)
# on the exact Stage-38 recipe. Two seeds on one card, then 3-checkpoint
# 50-episode evaluation. No online phase, no 101k, held-out stays sealed.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-0}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_no_bc/agent_a10_saucepan_nobc_gpu${GPU}_${STAMP}"
LAUNCH="cqn_as_pixel_bigym_nobc_agent_a1_descale_gate"
OFFLINE_UPDATES=20000

for seed in 3 4; do
  mkdir -p "${BASE}/descale_seed${seed}"
done
printf '%s\n' "${BASHPID}" > "${BASE}/controller.pid"
printf '%s\n' "${BASE}" > exp_local/cqn_no_bc/agent_a10_latest.txt
printf '%s\n' "${GPU}" > "${BASE}/gpu.txt"

run_dir () {
  printf '%s/descale_seed%s/offline' "${BASE}" "$1"
}

train_offline () {
  local seed="$1"
  local dir
  dir="$(run_dir "${seed}")"
  XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
    .venv/bin/python train_fast.py \
    launch="${LAUNCH}" env=bigym/saucepan_to_hob demos=36 env.expected_successful_demos=null seed="${seed}" \
    batch_size=256 demo_batch_size=256 \
    num_pretrain_steps="${OFFLINE_UPDATES}" \
    num_train_frames="${OFFLINE_UPDATES}" \
    replay.demo_only_updates=true \
    method.demo_behavior_force_probability=1.0 \
    eval_every_steps=1000000 num_eval_episodes=0 num_eval_envs=0 \
    snapshot_every_n=2500 gpu_id="${GPU}" xla_mem_fraction=0.45 \
    wandb.use=false hydra.run.dir="${dir}" \
    > "${BASE}/descale_seed${seed}/offline.log" 2>&1
}

status=0
train_offline 3 &
seed1_pid=$!
printf '%s\n' "${seed1_pid}" > "${BASE}/descale_seed3/offline.pid"
sleep 120
train_offline 4 &
seed2_pid=$!
printf '%s\n' "${seed2_pid}" > "${BASE}/descale_seed4/offline.pid"
wait "${seed1_pid}" || status=$?
seed2_status=0
wait "${seed2_pid}" || seed2_status=$?
if [[ "${status}" -eq 0 ]]; then status="${seed2_status}"; fi
if [[ "${status}" -ne 0 ]]; then
  printf '%s\n' "${status}" > "${BASE}/offline_failed"
  exit "${status}"
fi
for seed in 3 4; do
  dir="$(run_dir "${seed}")"
  test -s "${dir}/snapshots/10000_snapshot.pkl"
  cp "${dir}/.hydra/config.yaml" \
    "${BASE}/descale_seed${seed}/offline_config.yaml"
done
touch "${BASE}/offline_complete"

evaluate () {
  local dir="$1"
  local log="$2"
  XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${dir}" --gpu-id "${GPU}" \
    --num-eval-episodes 50 --eval-seed-start 400 \
    --num-eval-envs 25 \
    --only-steps "10000,12500,15000,17500,20000" \
    --csv-name val50_seeds400_offline.csv > "${log}" 2>&1
}

evaluate "$(run_dir 3)" "${BASE}/descale_seed3/val50.log" &
eval1=$!
evaluate "$(run_dir 4)" "${BASE}/descale_seed4/val50.log" &
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
