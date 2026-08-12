#!/usr/bin/env bash
# Agent line Stage A5: statistical replication of the locked de-scaled
# dense recipe. Two sequential two-run pairs (seeds 3+4, then 5+6), each
# running offline 10k -> online 10k -> fixed 5-checkpoint evaluation.
# No mechanism variation. Preregistered in cqn-no-bc-claude.md.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-0}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_no_bc/agent_a5_replication_gpu${GPU}_${STAMP}"
LAUNCH="cqn_as_pixel_bigym_nobc_agent_a1_descale_gate"
OFFLINE_UPDATES=10000
GLOBAL_LIMIT=20000

mkdir -p "${BASE}"
printf '%s\n' "${BASHPID}" > "${BASE}/controller.pid"
printf '%s\n' "${BASE}" > exp_local/cqn_no_bc/agent_a5_latest.txt
printf '%s\n' "${GPU}" > "${BASE}/gpu.txt"

run_dir () {
  printf '%s/descale_seed%s/full' "${BASE}" "$1"
}

train_phase () {
  local seed="$1"
  local demo_only="$2"
  local force="$3"
  local frames="$4"
  local phase="$5"
  local dir
  dir="$(run_dir "${seed}")"
  mkdir -p "${BASE}/descale_seed${seed}"
  XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
    .venv/bin/python train_fast.py \
    launch="${LAUNCH}" env=bigym/move_plate seed="${seed}" \
    batch_size=256 demo_batch_size=256 \
    num_pretrain_steps="${OFFLINE_UPDATES}" \
    num_train_frames="${frames}" \
    replay.demo_only_updates="${demo_only}" \
    method.demo_behavior_force_probability="${force}" \
    eval_every_steps=1000000 num_eval_episodes=0 num_eval_envs=0 \
    snapshot_every_n=2500 gpu_id="${GPU}" xla_mem_fraction=0.45 \
    wandb.use=false hydra.run.dir="${dir}" \
    > "${BASE}/descale_seed${seed}/${phase}.log" 2>&1
}

run_pair_phase () {
  local s_a="$1"
  local s_b="$2"
  local demo_only="$3"
  local force="$4"
  local frames="$5"
  local phase="$6"
  local status=0
  train_phase "${s_a}" "${demo_only}" "${force}" "${frames}" "${phase}" &
  local pid_a=$!
  sleep 120
  train_phase "${s_b}" "${demo_only}" "${force}" "${frames}" "${phase}" &
  local pid_b=$!
  wait "${pid_a}" || status=$?
  local status_b=0
  wait "${pid_b}" || status_b=$?
  if [[ "${status}" -eq 0 ]]; then status="${status_b}"; fi
  return "${status}"
}

evaluate () {
  local dir="$1"
  local log="$2"
  XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${dir}" --gpu-id "${GPU}" \
    --num-eval-episodes 50 --eval-seed-start 400 \
    --num-eval-envs 25 \
    --only-steps "10000,12500,15000,17500,20000" \
    --csv-name val50_seeds400_full.csv > "${log}" 2>&1
}

run_pair () {
  local s_a="$1"
  local s_b="$2"
  local status=0
  run_pair_phase "${s_a}" "${s_b}" true 1.0 "${OFFLINE_UPDATES}" offline \
    || status=$?
  if [[ "${status}" -ne 0 ]]; then
    printf '%s\n' "${status}" > "${BASE}/pair_${s_a}_${s_b}_offline_failed"
    return "${status}"
  fi
  for seed in "${s_a}" "${s_b}"; do
    test -s "$(run_dir "${seed}")/snapshots/10000_snapshot.pkl"
  done
  run_pair_phase "${s_a}" "${s_b}" false 0.0 "${GLOBAL_LIMIT}" online \
    || status=$?
  if [[ "${status}" -ne 0 ]]; then
    printf '%s\n' "${status}" > "${BASE}/pair_${s_a}_${s_b}_online_failed"
    return "${status}"
  fi
  for seed in "${s_a}" "${s_b}"; do
    test -s "$(run_dir "${seed}")/snapshots/20000_snapshot.pkl"
    cp "$(run_dir "${seed}")/.hydra/config.yaml" \
      "${BASE}/descale_seed${seed}/online_config.yaml"
  done
  evaluate "$(run_dir "${s_a}")" "${BASE}/descale_seed${s_a}/val50.log" &
  local eval_a=$!
  evaluate "$(run_dir "${s_b}")" "${BASE}/descale_seed${s_b}/val50.log" &
  local eval_b=$!
  wait "${eval_a}" || status=$?
  local eval_b_status=0
  wait "${eval_b}" || eval_b_status=$?
  if [[ "${status}" -eq 0 ]]; then status="${eval_b_status}"; fi
  if [[ "${status}" -ne 0 ]]; then
    printf '%s\n' "${status}" > "${BASE}/pair_${s_a}_${s_b}_validation_failed"
    return "${status}"
  fi
  touch "${BASE}/pair_${s_a}_${s_b}_complete"
  return 0
}

run_pair 3 4
run_pair 5 6
touch "${BASE}/complete"
