#!/usr/bin/env bash
# Stage 30: matched online-only versus reward-only offline->online CQN-AS.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-1}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_no_bc/stage30_offline_then_online_gpu${GPU}_${STAMP}"
LAUNCH="cqn_as_pixel_bigym_nobc_stage30_offline_then_online_gate"
OFFLINE_UPDATES=10000
ONLINE_FRAMES=20000
# RoboBase currently includes pretrain updates in global_env_steps. The
# treatment therefore stops at 10k offline + 20k real environment steps.
TREATMENT_GLOBAL_LIMIT=$((OFFLINE_UPDATES + ONLINE_FRAMES))

mkdir -p "${BASE}/phase_configs"
printf "%s\n" "${BASHPID}" > "${BASE}/controller.pid"
printf "%s\n" "${BASE}" > exp_local/cqn_no_bc/stage30_latest.txt

train () {
  local seed="$1"
  local run_dir="$2"
  local pretrain_steps="$3"
  local train_frames="$4"
  local demo_batch_size="$5"
  local demo_only="$6"
  local force_probability="$7"
  local log="$8"
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MUJOCO_GL=egl \
    .venv/bin/python train_fast.py \
    launch="${LAUNCH}" \
    env=bigym/move_plate \
    seed="${seed}" \
    num_pretrain_steps="${pretrain_steps}" \
    num_train_frames="${train_frames}" \
    demo_batch_size="${demo_batch_size}" \
    replay.demo_only_updates="${demo_only}" \
    method.demo_behavior_force_probability="${force_probability}" \
    gpu_id="${GPU}" \
    xla_mem_fraction=0.35 \
    wandb.use=false \
    hydra.run.dir="${run_dir}" \
    > "${log}" 2>&1
}

run_seed_pair () {
  local seed="$1"
  local control="${BASE}/online_only_seed${seed}"
  local treatment="${BASE}/offline_then_online_seed${seed}"

  # Matched online-from-random control: canonical C51/MC candidate backup,
  # zero offline updates, and exactly 20k environment interactions.
  train "${seed}" "${control}" 0 "${ONLINE_FRAMES}" 16 false 0.0 \
    "${BASE}/control_seed${seed}.log" &
  local control_pid=$!
  printf "%s\n" "${control_pid}" > "${BASE}/control_seed${seed}.pid"

  # Stagger the two XLA compilations while retaining two runs on one card.
  sleep 120

  # Offline phase: all optimizer samples come from protected expert replay.
  # Exact action_tp1 is forced only as the reward Bellman continuation.
  train "${seed}" "${treatment}" "${OFFLINE_UPDATES}" \
    "${OFFLINE_UPDATES}" 32 true 1.0 \
    "${BASE}/offline_seed${seed}.log" &
  local offline_pid=$!
  printf "%s\n" "${offline_pid}" > "${BASE}/offline_seed${seed}.pid"

  local status=0
  wait "${offline_pid}" || status=$?
  if [[ "${status}" -ne 0 ]]; then
    printf "%s\n" "${status}" > "${BASE}/offline_seed${seed}_failed"
    wait "${control_pid}" || true
    exit "${status}"
  fi
  cp "${treatment}/.hydra/config.yaml" \
    "${BASE}/phase_configs/offline_seed${seed}.yaml"
  touch "${BASE}/offline_seed${seed}_complete"

  # Resume the exact critic/optimizer/replay state. Since _pretrain_step is
  # already 10k, pretraining is skipped; only the online phase runs. The
  # protected demo half remains, but target selection is ordinary candidate
  # max rather than forced expert continuation.
  train "${seed}" "${treatment}" "${OFFLINE_UPDATES}" \
    "${TREATMENT_GLOBAL_LIMIT}" 16 false 0.0 \
    "${BASE}/online_seed${seed}.log" &
  local treatment_pid=$!
  printf "%s\n" "${treatment_pid}" > "${BASE}/online_seed${seed}.pid"

  wait "${control_pid}" || status=$?
  wait "${treatment_pid}" || status=$?
  if [[ "${status}" -ne 0 ]]; then
    printf "%s\n" "${status}" > "${BASE}/training_seed${seed}_failed"
    exit "${status}"
  fi
  cp "${control}/.hydra/config.yaml" \
    "${BASE}/phase_configs/control_seed${seed}.yaml"
  cp "${treatment}/.hydra/config.yaml" \
    "${BASE}/phase_configs/online_seed${seed}.yaml"
  touch "${BASE}/training_seed${seed}_complete"
}

run_seed_pair 1
run_seed_pair 2
touch "${BASE}/training_complete"

evaluate () {
  local run_dir="$1"
  local only_steps="$2"
  local log="$3"
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MUJOCO_GL=egl \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${run_dir}" \
    --gpu-id "${GPU}" \
    --num-eval-episodes 50 \
    --eval-seed-start 400 \
    --num-eval-envs 25 \
    --only-steps "${only_steps}" \
    --csv-name val50_seeds400_raw_steps.csv \
    > "${log}" 2>&1
}

CONTROL_STEPS="2500,5000,7500,10000,12500,15000,17500,20000"
TREATMENT_STEPS="10000,12500,15000,17500,20000,22500,25000,27500,30000"

evaluate_seed_pair () {
  local seed="$1"
  evaluate "${BASE}/online_only_seed${seed}" "${CONTROL_STEPS}" \
    "${BASE}/control_seed${seed}_val50.log" &
  local control_eval_pid=$!
  printf "%s\n" "${control_eval_pid}" \
    > "${BASE}/control_seed${seed}_val50.pid"
  evaluate "${BASE}/offline_then_online_seed${seed}" "${TREATMENT_STEPS}" \
    "${BASE}/treatment_seed${seed}_val50.log" &
  local treatment_eval_pid=$!
  printf "%s\n" "${treatment_eval_pid}" \
    > "${BASE}/treatment_seed${seed}_val50.pid"

  local status=0
  wait "${control_eval_pid}" || status=$?
  wait "${treatment_eval_pid}" || status=$?
  if [[ "${status}" -ne 0 ]]; then
    printf "%s\n" "${status}" > "${BASE}/validation_seed${seed}_failed"
    exit "${status}"
  fi
  touch "${BASE}/validation_seed${seed}_complete"
}

evaluate_seed_pair 1
evaluate_seed_pair 2
touch "${BASE}/validation_complete"

.venv/bin/python -m scripts.summarize_cqn_no_bc_stage30 \
  --control-seed1 "${BASE}/online_only_seed1" \
  --control-seed2 "${BASE}/online_only_seed2" \
  --treatment-seed1 "${BASE}/offline_then_online_seed1" \
  --treatment-seed2 "${BASE}/offline_then_online_seed2" \
  --output "${BASE}/stage30_summary.json" \
  > "${BASE}/stage30_summary.log" 2>&1
touch "${BASE}/complete"
