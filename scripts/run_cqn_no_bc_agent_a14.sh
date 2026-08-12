#!/usr/bin/env bash
# Agent line Stage A9: paired satisficing-floor online arms.
# Branch the six existing raw-10k de-scaled offline states (A1 seeds 1-2,
# A5 seeds 3-6), run the online phase with dense_return_label_smoothing
# =0.05 (sole resolved diff vs the paired plain-dense controls), evaluate
# raw 12.5/15/17.5/20k. Three sequential pairs. Preregistered in
# cqn-no-bc-claude.md.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-5}"
A1_BASE="$(tr -d '\n' < exp_local/cqn_no_bc/agent_a1_latest.txt)"
A5_BASE="$(tr -d '\n' < exp_local/cqn_no_bc/agent_a5_latest.txt)"
LAUNCH="cqn_as_pixel_bigym_nobc_agent_a14_explore_gate"
OFFLINE_UPDATES=10000
GLOBAL_LIMIT=30000
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_no_bc/agent_a14_explore_gpu${GPU}_${STAMP}"

mkdir -p "${BASE}"
printf '%s\n' "${BASHPID}" > "${BASE}/controller.pid"
printf '%s\n' "${BASE}" > exp_local/cqn_no_bc/agent_a14_gpu${GPU}_latest.txt

source_dir () {
  local seed="$1"
  if [[ "${seed}" -le 2 ]]; then
    printf '%s/descale_seed%s/offline' "${A1_BASE}" "${seed}"
  else
    printf '%s/descale_seed%s/full' "${A5_BASE}" "${seed}"
  fi
}

run_dir () {
  printf '%s/descale_seed%s/online' "${BASE}" "$1"
}

for seed in 1 2 3 4 5 6; do
  mkdir -p "${BASE}/descale_seed${seed}"
  .venv/bin/python scripts/prepare_cqn_no_bc_stage40_branch.py \
    --source-run "$(source_dir "${seed}")" \
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
  local egl_env=()
  if [[ -f "${BASE}/egl_index_override" ]]; then
    egl_env=(MUJOCO_EGL_DEVICE_ID="$(cat "${BASE}/egl_index_override")")
  fi
  env XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl "${egl_env[@]}" \
    .venv/bin/python train_fast.py \
    launch="${LAUNCH}" env=bigym/move_plate seed="${seed}" \
    batch_size=256 demo_batch_size=256 \
    num_pretrain_steps="${OFFLINE_UPDATES}" \
    num_train_frames="${GLOBAL_LIMIT}" \
    replay.demo_only_updates=false \
    method.demo_behavior_force_probability=0.0 \
    eval_every_steps=1000000 num_eval_episodes=0 num_eval_envs=0 \
    snapshot_every_n=2500 gpu_id="${GPU}" xla_mem_fraction=0.45 \
    wandb.use=false hydra.run.dir="${dir}" \
    > "${BASE}/descale_seed${seed}/online.log" 2>&1
}

egl_smoke () {
  # One trivial worker probing env creation on this card; on the known
  # EGL-index ambiguity (boot-VGA enumeration), retry once with index 4.
  local dir="${BASE}/egl_smoke"
  rm -rf "${dir}"
  local egl_env=()
  if [[ -f "${BASE}/egl_index_override" ]]; then
    egl_env=(MUJOCO_EGL_DEVICE_ID="$(cat "${BASE}/egl_index_override")")
  fi
  env XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl "${egl_env[@]}" \
    timeout 600 .venv/bin/python train_fast.py \
    launch="${LAUNCH}" env=bigym/move_plate seed=1 \
    batch_size=16 demo_batch_size=16 demos=2 \
    env.expected_successful_demos=null \
    num_pretrain_steps=2 num_train_frames=2 \
    replay.demo_only_updates=true \
    method.demo_behavior_force_probability=1.0 \
    eval_every_steps=1000000 num_eval_episodes=0 num_eval_envs=0 \
    save_snapshot=false gpu_id="${GPU}" xla_mem_fraction=0.2 \
    wandb.use=false hydra.run.dir="${dir}" \
    > "${BASE}/egl_smoke.log" 2>&1
}

if ! egl_smoke; then
  if grep -qi 'egl' "${BASE}/egl_smoke.log"; then
    printf '4\n' > "${BASE}/egl_index_override"
    if ! egl_smoke; then
      touch "${BASE}/egl_blocked"
      exit 1
    fi
  else
    touch "${BASE}/smoke_failed"
    exit 1
  fi
fi
touch "${BASE}/smoke_ok"

evaluate () {
  local dir="$1"
  local log="$2"
  local egl_env=()
  if [[ -f "${BASE}/egl_index_override" ]]; then
    egl_env=(MUJOCO_EGL_DEVICE_ID="$(cat "${BASE}/egl_index_override")")
  fi
  env XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl "${egl_env[@]}" \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${dir}" --gpu-id "${GPU}" \
    --num-eval-episodes 50 --eval-seed-start 400 \
    --num-eval-envs 25 \
    --only-steps "12500,15000,17500,20000,25000,30000" \
    --csv-name val50_seeds400_online.csv > "${log}" 2>&1
}

run_pair () {
  local s_a="$1"
  local s_b="$2"
  local status=0
  train_online "${s_a}" &
  local pid_a=$!
  sleep 120
  train_online "${s_b}" &
  local pid_b=$!
  wait "${pid_a}" || status=$?
  local status_b=0
  wait "${pid_b}" || status_b=$?
  if [[ "${status}" -eq 0 ]]; then status="${status_b}"; fi
  if [[ "${status}" -ne 0 ]]; then
    printf '%s\n' "${status}" > "${BASE}/pair_${s_a}_${s_b}_train_failed"
    return "${status}"
  fi
  for seed in "${s_a}" "${s_b}"; do
    test -s "$(run_dir "${seed}")/snapshots/${GLOBAL_LIMIT}_snapshot.pkl"
    cp "$(run_dir "${seed}")/.hydra/config.yaml" \
      "${BASE}/descale_seed${seed}/online_config.yaml"
    .venv/bin/python - "$(run_dir "${seed}")/train.csv" <<'PYCHK'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
fired = float(rows[-1].get("bin_explore_fired_total", 0) or 0)
applied = float(rows[-1].get("bin_explore_applied_total", 0) or 0)
assert fired > 0 and applied > 0, f"explore never activated: fired={fired} applied={applied}"
print(f"explore wiring ok: fired={fired:.0f} applied={applied:.0f}")
PYCHK
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
    printf '%s\n' "${status}" > "${BASE}/pair_${s_a}_${s_b}_val_failed"
    return "${status}"
  fi
  touch "${BASE}/pair_${s_a}_${s_b}_complete"
  return 0
}

shift || true
while [[ "$#" -ge 2 ]]; do
  run_pair "$1" "$2"
  shift 2
done
touch "${BASE}/complete"
